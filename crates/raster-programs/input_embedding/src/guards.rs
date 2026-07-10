//! Structural and trace-shape guards for `input.embedding`.

#![cfg(test)]

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use raster::core::trace::{FnInput, FnInputValue, TraceEvent};
use raster::internal;
use raster::materialize_auth_result;

use crate::routine::*;
use crate::test_trace::{capture_trace_events, event_record};
use crate::types::{
    EmbeddingSource, GemmaInputEmbeddingMetadata, GemmaInputEmbeddingTable, InputEmbeddingConfig,
    InputEmbeddingLoopDrivers, InputEmbeddingOutput, InputEmbeddingPromptTokenIds,
    PromptTokenSource,
};

const TYPES_SRC: &str = include_str!("types.rs");
const PROGRAM_SOURCES: &[(&str, &str)] = &[
    ("types.rs", TYPES_SRC),
    ("prompt_tokens.rs", include_str!("prompt_tokens.rs")),
    ("embedding.rs", include_str!("embedding.rs")),
    ("routine.rs", include_str!("routine.rs")),
];

const RECUR_STATE_TYPES: &[&str] = &["InputEmbeddingCopyState"];

fn struct_fields(src: &str, struct_name: &str) -> Vec<(String, String)> {
    let header = alloc::format!("pub struct {struct_name} {{");
    let start = src
        .find(&header)
        .unwrap_or_else(|| panic!("struct {struct_name} not found"));
    let body = &src[start + header.len()..];
    let end = body.find("\n}").expect("struct body should close");
    body[..end]
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let declaration = line.strip_prefix("pub ")?;
            let (name, ty) = declaration.split_once(':')?;
            Some((
                name.trim().to_string(),
                ty.trim().trim_end_matches(',').to_string(),
            ))
        })
        .collect()
}

fn is_collection_type(ty: &str) -> bool {
    [
        "Vec<",
        "GemmaInputEmbeddingTable",
        "InputEmbeddingPromptTokenIds",
    ]
    .iter()
    .any(|marker| ty.contains(marker))
}

fn recur_state_idents() -> Vec<String> {
    let mut idents = Vec::new();
    for (_, src) in PROGRAM_SOURCES {
        for line in src.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            for marker in ["RecurSequenceState<", "RecurState<"] {
                let mut rest = trimmed;
                while let Some(at) = rest.find(marker) {
                    rest = &rest[at + marker.len()..];
                    let ident: String = rest
                        .chars()
                        .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
                        .collect();
                    if !ident.is_empty() {
                        idents.push(ident);
                    }
                }
            }
        }
    }
    idents.sort();
    idents.dedup();
    idents
}

#[test]
fn recur_state_types_are_exactly_the_known_set() {
    let mut expected: Vec<String> = RECUR_STATE_TYPES.iter().map(ToString::to_string).collect();
    expected.sort();
    assert_eq!(recur_state_idents(), expected);
}

#[test]
fn recur_state_types_carry_no_collections() {
    for state_type in RECUR_STATE_TYPES {
        let collections: Vec<_> = struct_fields(TYPES_SRC, state_type)
            .into_iter()
            .filter(|(_, ty)| is_collection_type(ty))
            .collect();
        assert!(
            collections.is_empty(),
            "recur state {state_type} must not accumulate collections, found {collections:?}"
        );
    }
}

#[test]
fn tiles_do_not_accept_large_eager_collections() {
    let forbidden = [
        "pub fn init_input_embedding_counts(\n    prompt_token_ids: Vec<u32>",
        "pub fn embed_one_token_chunk(\n    prompt_token_ids: Vec<u32>",
        "pub fn init_input_embedding_counts(\n    embedding_rows: Vec<",
        "pub fn embed_one_token_chunk(\n    embedding_rows: Vec<",
        "RecurState<InputEmbeddingPromptTokenIds>",
        "RecurState<GemmaInputEmbeddingTable>",
        "build_input_embedding_budgets",
        "validate_input_embedding_loop_drivers",
        "while ",
    ];
    for (name, src) in PROGRAM_SOURCES {
        for needle in forbidden {
            assert!(
                !src.contains(needle),
                "{name} must not use eager input-embedding parameter `{needle}`"
            );
        }
    }
}

const TOKEN_A: u32 = 0x0abc_def0;
const TOKEN_B: u32 = 0x0123_4567;
const ROW_A: i32 = 0x0102_0304;
const ROW_B: i32 = 0x0506_0708;
/// Hex-packed leaf forms of the row sentinels (the shape rows take in the
/// staged table since the one-leaf-per-row schema).
const PACKED_ROW_AB: &[u8] = b"0102030405060708";
const PACKED_ROW_BA: &[u8] = b"0506070801020304";
const TOKEN_DRIVER_SENTINEL: &[u32] = &[0, 1, 2, 3];

fn contains_i32_marker(bytes: &[u8], marker: i32) -> bool {
    bytes
        .windows(core::mem::size_of::<i32>())
        .any(|window| window == marker.to_le_bytes())
}

fn contains_u32_marker(bytes: &[u8], marker: u32) -> bool {
    bytes
        .windows(core::mem::size_of::<u32>())
        .any(|window| window == marker.to_le_bytes())
}

fn contains_u32_sequence(bytes: &[u8], markers: &[u32]) -> bool {
    let marker_bytes = markers
        .iter()
        .flat_map(|marker| marker.to_le_bytes())
        .collect::<Vec<_>>();
    bytes
        .windows(marker_bytes.len())
        .any(|window| window == marker_bytes.as_slice())
}

fn contains_byte_sequence(bytes: &[u8], marker: &[u8]) -> bool {
    bytes.windows(marker.len()).any(|window| window == marker)
}

fn row_payload_marker(bytes: &[u8]) -> bool {
    // Rows can leak either decoded (i32 LE bits) or in their hex-packed
    // staged leaf form; guard against both representations.
    [ROW_A, ROW_B]
        .iter()
        .any(|marker| contains_i32_marker(bytes, *marker))
        || [PACKED_ROW_AB, PACKED_ROW_BA]
            .iter()
            .any(|marker| contains_byte_sequence(bytes, marker))
}

fn inline_payload_marker(input: &FnInput) -> bool {
    input.values.iter().any(|value| match value {
        FnInputValue::Inline(bytes) => {
            row_payload_marker(bytes)
                || [TOKEN_A, TOKEN_B]
                    .iter()
                    .any(|marker| contains_u32_marker(bytes, *marker))
        }
        _ => false,
    })
}

fn input_data_payload_marker(input: &FnInput) -> bool {
    row_payload_marker(&input.data)
        || [TOKEN_A, TOKEN_B]
            .iter()
            .any(|marker| contains_u32_marker(&input.data, *marker))
}

fn run_input_embedding() -> core::result::Result<InputEmbeddingOutput, String> {
    let _guard = raster::__private::SequenceScopeGuard::enter("input_embedding_guard_tests");
    materialize_auth_result::<InputEmbeddingOutput, _>(__raster_sequence_auth_embed_input_tokens(
        internal!(
            PromptTokenSource,
            raster::store_internal_value(&PromptTokenSource::internal(
                raster::store_internal_value(&InputEmbeddingPromptTokenIds {
                    token_count: 4,
                    token_ids: vec![0, 1, 0, 1],
                    token_ids_sha256: "token-sha".to_string(),
                })
                .expect("store prompt ids")
            ))
            .expect("store prompt source")
        ),
        internal!(
            EmbeddingSource,
            raster::store_internal_value(&EmbeddingSource::internal(
                raster::store_internal_value(&GemmaInputEmbeddingTable {
                    metadata: GemmaInputEmbeddingMetadata {
                        source_id: "embedding-fixture".to_string(),
                        vocab_size: 2,
                        hidden_size: 2,
                        scale_bits: 0,
                    },
                    rows: vec![
                        crate::types::pack_embedding_row_hex(&[ROW_A, ROW_B]),
                        crate::types::pack_embedding_row_hex(&[ROW_B, ROW_A]),
                    ],
                })
                .expect("store embedding")
            ))
            .expect("store embedding source")
        ),
        internal!(
            InputEmbeddingLoopDrivers,
            raster::store_internal_value(&InputEmbeddingLoopDrivers {
                token_ordinals: TOKEN_DRIVER_SENTINEL.to_vec(),
            })
            .expect("store drivers")
        ),
        internal!(
            InputEmbeddingConfig,
            raster::store_internal_value(&InputEmbeddingConfig {
                tokens_per_tile: 1,
                prompt_token_ids_sha256: "token-sha".to_string(),
                prompt_token_ids_root: "prompt-root".to_string(),
                embedding_source_root: "embedding-root".to_string(),
            })
            .expect("store config")
        ),
    ))
}

#[test]
fn collections_cross_the_abi_only_as_authenticated_reads() {
    let (outcome, events) = capture_trace_events(run_input_embedding);
    let output = outcome.expect("sentinel input embedding should run");
    assert_eq!(output.activation_rows.len(), 4);

    for event in &events {
        if let TraceEvent::RecurTileIterationExec(record) = event {
            assert_eq!(
                record.fn_name, "embed_one_token_chunk",
                "only scalar/ref input-embedding recur tiles may appear"
            );
            let input = record.input.as_ref().expect("recur tile input");
            assert!(
                input.data.len() < 768,
                "recur-tile input must stay scalar/ref sized, got {} bytes",
                input.data.len()
            );
            assert!(
                !input_data_payload_marker(input),
                "input embedding recur-tile input must not inline token ids or rows"
            );
        }
    }

    for event in &events {
        let Some(record) = event_record(event) else {
            continue;
        };
        if let Some(output) = record.output.as_ref() {
            assert!(
                !contains_u32_sequence(&output.data, TOKEN_DRIVER_SENTINEL),
                "tile '{}' output carried staged token ordinal driver bytes",
                record.fn_name
            );
        }
        let Some(input) = record.input.as_ref() else {
            continue;
        };
        assert!(
            !contains_u32_sequence(&input.data, TOKEN_DRIVER_SENTINEL),
            "input embedding loop driver rode in FnInput.data for '{}'",
            record.fn_name
        );
        if inline_payload_marker(input) {
            panic!(
                "input embedding collections rode inline into '{}'; collections must cross as bindings or draft ops",
                record.fn_name
            );
        }
    }
}

#[test]
fn recur_trace_scales_with_chunks_not_tokens() {
    let (outcome, events) = capture_trace_events(run_input_embedding);
    outcome.expect("chunked input embedding should run");

    let mut iterations = 0usize;
    for event in &events {
        if let TraceEvent::RecurTileIterationExec(record) = event {
            assert_eq!(record.fn_name, "embed_one_token_chunk");
            iterations += 1;
        }
    }
    assert_eq!(
        iterations, 4,
        "4 tokens at 1 per tile must embed in 4 recur iterations"
    );
}
