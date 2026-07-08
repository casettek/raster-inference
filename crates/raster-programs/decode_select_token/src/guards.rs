//! Structural and trace-shape guards for `decode.select_token`.

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
    DecodeSelectConfig, DecodeSelectLogitSource, DecodeSelectLogits, DecodeSelectLoopDrivers,
    DecodeSelectOutput, DecodeSelectTokenIds, DecodeSelectTokenSource,
};

const TYPES_SRC: &str = include_str!("types.rs");
const PROGRAM_SOURCES: &[(&str, &str)] = &[
    ("types.rs", TYPES_SRC),
    ("logits.rs", include_str!("logits.rs")),
    ("token_ids.rs", include_str!("token_ids.rs")),
    ("routine.rs", include_str!("routine.rs")),
];

const RECUR_STATE_TYPES: &[&str] = &["DecodeSelectArgmaxState", "DecodeSelectCopyState"];

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
    ["Vec<", "DecodeSelectLogits", "DecodeSelectTokenIds"]
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
fn phase_tiles_do_not_accept_large_eager_collections() {
    let forbidden = [
        "publish_logits",
        "publish_token_ids",
        "logits: Vec<i32>",
        "state: Vec<",
        "RecurState<DecodeSelectTokenIds>",
        "build_decode_select_budgets",
        "validate_decode_select_loop_drivers",
        "ordinals_from(",
        "while ",
    ];
    for (name, src) in PROGRAM_SOURCES {
        for needle in forbidden {
            assert!(
                !src.contains(needle),
                "{name} must not use eager large decode-select parameter `{needle}`"
            );
        }
    }
}

const LOGIT_A: i32 = 0x1234_567;
const LOGIT_B: i32 = 0x2345_678;
const LOGIT_C: i32 = 0x3456_789;
/// Chunk ordinals for 7 logits at one logit per tile. The 28-byte LE
/// encoding is the tracer sentinel for staged-driver leaks.
const LOGIT_DRIVER_SENTINEL: &[u32] = &[0, 1, 2, 3, 4, 5, 6];
const TOKEN_A: u32 = 0x0abc_def0;
const TOKEN_B: u32 = 0x0123_4567;

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

fn inline_decode_payload_marker(input: &FnInput) -> bool {
    input.values.iter().any(|value| match value {
        FnInputValue::Inline(bytes) => {
            [LOGIT_A, LOGIT_B, LOGIT_C]
                .iter()
                .any(|marker| contains_i32_marker(bytes, *marker))
                || [TOKEN_A, TOKEN_B]
                    .iter()
                    .any(|marker| contains_u32_marker(bytes, *marker))
        }
        _ => false,
    })
}

fn input_data_decode_payload_marker(input: &FnInput) -> bool {
    [LOGIT_A, LOGIT_B, LOGIT_C]
        .iter()
        .any(|marker| contains_i32_marker(&input.data, *marker))
        || [TOKEN_A, TOKEN_B]
            .iter()
            .any(|marker| contains_u32_marker(&input.data, *marker))
}

fn run_decode_select(
    logit_bits: Vec<i32>,
    full_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    loop_drivers: DecodeSelectLoopDrivers,
    config: DecodeSelectConfig,
) -> core::result::Result<DecodeSelectOutput, String> {
    let _guard = raster::__private::SequenceScopeGuard::enter("decode_select_guard_tests");
    materialize_auth_result::<DecodeSelectOutput, _>(__raster_sequence_auth_select_decode_token(
        internal!(
            DecodeSelectLogitSource,
            raster::store_internal_value(&DecodeSelectLogitSource::internal(
                raster::store_internal_value(&DecodeSelectLogits {
                    row_count: logit_bits.len() as u32,
                    width: 1,
                    bits: logit_bits,
                })
                .expect("store logits")
            ))
            .expect("store logit source")
        ),
        internal!(
            DecodeSelectTokenSource,
            raster::store_internal_value(&DecodeSelectTokenSource::internal(
                raster::store_internal_value(&DecodeSelectTokenIds {
                    token_count: full_token_ids.len() as u32,
                    token_ids: full_token_ids,
                })
                .expect("store full tokens")
            ))
            .expect("store full token source")
        ),
        internal!(
            DecodeSelectTokenSource,
            raster::store_internal_value(&DecodeSelectTokenSource::internal(
                raster::store_internal_value(&DecodeSelectTokenIds {
                    token_count: generated_token_ids.len() as u32,
                    token_ids: generated_token_ids,
                })
                .expect("store generated tokens")
            ))
            .expect("store generated token source")
        ),
        internal!(
            DecodeSelectLoopDrivers,
            raster::store_internal_value(&loop_drivers).expect("store loop drivers")
        ),
        internal!(
            DecodeSelectConfig,
            raster::store_internal_value(&config).expect("store config")
        ),
    ))
}

fn select_with_sentinels() -> core::result::Result<DecodeSelectOutput, String> {
    run_decode_select(
        vec![LOGIT_A, LOGIT_C, LOGIT_B, 1, 2, 3, 4],
        vec![TOKEN_A],
        vec![TOKEN_B],
        DecodeSelectLoopDrivers {
            logit_ordinals: LOGIT_DRIVER_SENTINEL.to_vec(),
            full_token_ordinals: vec![0],
            generated_token_ordinals: vec![0],
        },
        DecodeSelectConfig {
            logits_per_tile: 1,
            token_ids_per_tile: 1,
        },
    )
}

#[test]
fn decode_collections_cross_the_abi_only_as_authenticated_reads() {
    let (outcome, events) = capture_trace_events(select_with_sentinels);
    let output = outcome.expect("sentinel select should run");
    assert_eq!(output.next_token, 1);
    assert_eq!(output.full_token_ids, vec![TOKEN_A, 1]);
    assert_eq!(output.generated_token_ids, vec![TOKEN_B, 1]);

    for event in &events {
        if let TraceEvent::TileExec(record) = event {
            assert_ne!(
                record.fn_name, "build_decode_select_budgets",
                "decode.select_token must not tile-produce recur ordinal budgets"
            );
            assert_ne!(
                record.fn_name, "validate_decode_select_loop_drivers",
                "decode.select_token must not tile-validate the whole recur driver"
            );
            if let Some(output) = record.output.as_ref() {
                assert!(
                    !contains_u32_sequence(&output.data, LOGIT_DRIVER_SENTINEL),
                    "tile '{}' output carried staged logit ordinal driver bytes",
                    record.fn_name
                );
            }
        }
    }

    for event in &events {
        if let TraceEvent::RecurTileIterationExec(record) = event {
            assert!(
                matches!(
                    record.fn_name.as_str(),
                    "scan_one_logit_chunk" | "copy_one_token_chunk"
                ),
                "only scalar/ref decode-select recur tiles may appear"
            );
            let input = record.input.as_ref().expect("recur tile input");
            assert!(
                input.data.len() < 512,
                "recur-tile input must stay scalar/ref sized, got {} bytes",
                input.data.len()
            );
            assert!(
                !input_data_decode_payload_marker(input),
                "scan recur-tile input must not inline logits or token ids"
            );
        }
    }

    for event in &events {
        let Some(record) = event_record(event) else {
            continue;
        };
        if let Some(output) = record.output.as_ref() {
            assert!(
                !contains_u32_sequence(&output.data, LOGIT_DRIVER_SENTINEL),
                "tile '{}' output carried staged logit ordinal driver bytes",
                record.fn_name
            );
        }
        let Some(input) = record.input.as_ref() else {
            continue;
        };
        assert!(
            !contains_u32_sequence(&input.data, LOGIT_DRIVER_SENTINEL),
            "decode loop driver rode in FnInput.data for '{}' ({event:?})",
            record.fn_name
        );
        if !inline_decode_payload_marker(input) {
            if input_data_decode_payload_marker(input) {
                assert!(
                    record.fn_name == "append_selected_token",
                    "decode collections rode in FnInput.data for '{}' ({event:?})",
                    record.fn_name
                );
            }
            continue;
        }
        panic!(
            "decode collections rode inline into '{}' ({event:?}); collections must cross as bindings or draft ops",
            record.fn_name
        );
    }
}

/// Recur iteration counts must scale with chunk count, not element count —
/// one iteration per `logits_per_tile` logits and per `token_ids_per_tile`
/// token ids. A per-element loop over a real vocab would put one trace
/// record per logit and blow the routine's trace budget.
#[test]
fn recur_trace_scales_with_chunks_not_elements() {
    let (outcome, events) = capture_trace_events(|| {
        run_decode_select(
            vec![1, 9, 2, 4, 3, 8, 7],
            vec![10, 11, 12],
            vec![20],
            DecodeSelectLoopDrivers {
                logit_ordinals: vec![0, 1, 2],
                full_token_ordinals: vec![0, 1],
                generated_token_ordinals: vec![0],
            },
            DecodeSelectConfig {
                logits_per_tile: 3,
                token_ids_per_tile: 2,
            },
        )
    });
    let output = outcome.expect("chunked select should run");
    assert_eq!(output.next_token, 1);
    assert_eq!(output.full_token_ids, vec![10, 11, 12, 1]);
    assert_eq!(output.generated_token_ids, vec![20, 1]);

    let mut scan_iterations = 0usize;
    let mut copy_iterations = 0usize;
    for event in &events {
        if let TraceEvent::RecurTileIterationExec(record) = event {
            match record.fn_name.as_str() {
                "scan_one_logit_chunk" => scan_iterations += 1,
                "copy_one_token_chunk" => copy_iterations += 1,
                other => panic!("unexpected decode-select recur tile '{other}'"),
            }
        }
    }
    assert_eq!(
        scan_iterations, 3,
        "7 logits at 3 per tile must scan in 3 recur iterations"
    );
    assert_eq!(
        copy_iterations, 3,
        "3+1 token ids at 2 per tile must copy in 2+1 recur iterations"
    );
}
