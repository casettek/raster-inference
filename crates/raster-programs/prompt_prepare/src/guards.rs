//! Structural guards for the storage-resident invariant.
//!
//! Three legs:
//!
//! 1. **Source-scan guard**: fails compilation-adjacent (test time) on any
//!    regression that adds an accumulating collection to a recur state
//!    type or a pieces-carrying field to a per-chunk/per-item decision
//!    struct. Prompt-derived collections must stay behind storage roots,
//!    selection bindings, and drafts.
//! 2. **Trace-shape assertion**: a native routine run with sentinel pieces
//!    must show prompt-derived collections only as draft ops, bindings,
//!    and the staged external — never inline in recur-sequence state,
//!    recur-tile iteration records (none may exist), or context args.
//! 3. **Functional matrix**: the cases not already covered by the
//!    module-level tests (`routine.rs` covers empty prompt, single piece /
//!    zero-round fallback, byte fallback, and the missing-piece error;
//!    `token_ids.rs` covers duplicate-conflict and out-of-range matches).
//!
//! Test-only module: never program surface.

#![cfg(test)]

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use raster::core::trace::{FnInput, FnInputValue, TraceEvent};
use raster::materialize_auth_result;
use raster::prelude::*;
use raster_program_gemma_externals::types::{
    GemmaBpeMerge, GemmaDecoderMetadata, GemmaTokenIdEntry, GemmaTokenizer, GemmaTokenizerMetadata,
};

use crate::routine::*;
use crate::test_trace::capture_trace_events;
use crate::types::{BpePieces, PromptTokenization, TokenizerTables};

// --- 1. Source-scan guard -------------------------------------------------

const TYPES_SRC: &str = include_str!("types.rs");

/// Program sources that may name recur state types (the probe module is
/// excluded: its tiles are test-only evidence, not program surface; P4's
/// collection-carrying recur-tile state exists exactly to characterize the
/// pinned rev).
const PROGRAM_SOURCES: &[(&str, &str)] = &[
    ("types.rs", TYPES_SRC),
    ("routine.rs", include_str!("routine.rs")),
    ("budgets.rs", include_str!("budgets.rs")),
    ("bpe_round.rs", include_str!("bpe_round.rs")),
    ("bpe_scan.rs", include_str!("bpe_scan.rs")),
    ("bpe_apply.rs", include_str!("bpe_apply.rs")),
    ("token_ids.rs", include_str!("token_ids.rs")),
];

/// The program's recur state types. All are scalar or handle state; none may
/// carry prompt-derived collections.
const RECUR_STATE_TYPES: &[&str] = &[
    "GemmaBpeLoopState",
    "GemmaBpeScanState",
    "GemmaBpeApplyCursor",
];

/// Scalar decision/context structs threaded to per-chunk or per-item
/// tiles: never a pieces field, never a collection.
const PER_ITEM_CONTEXT_TYPES: &[&str] = &["GemmaBpeApplyDecision", "GemmaBpeScanCandidate"];

/// Extracts `(field_name, field_type)` pairs of one struct in `src`.
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

/// A collection for the guard's purposes: a growable container or one of
/// the crate's collection-carrying selectable roots.
fn is_collection_type(ty: &str) -> bool {
    ["Vec<", "Map<", "BpePieces", "TokenIdMatches"]
        .iter()
        .any(|marker| ty.contains(marker))
}

/// Inner identifiers of every non-comment `RecurState<…>` /
/// `RecurSequenceState<…>` occurrence across the program sources.
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
    assert_eq!(
        recur_state_idents(),
        expected,
        "a new recur state type must be registered here and satisfy the \
         no-accumulating-collection guard"
    );
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
            "recur state {state_type} must not accumulate collections, \
             found {collections:?}"
        );
    }
}

#[test]
fn per_item_context_structs_carry_no_pieces_or_collections() {
    for context_type in PER_ITEM_CONTEXT_TYPES {
        for (name, ty) in struct_fields(TYPES_SRC, context_type) {
            assert_ne!(
                name, "pieces",
                "{context_type} must not carry a pieces field into a loop"
            );
            assert!(
                !is_collection_type(&ty),
                "{context_type}.{name}: {ty} must not carry a collection \
                 into a per-chunk/per-item tile"
            );
        }
    }
}

#[test]
fn phase_tiles_do_not_accept_large_eager_collections() {
    let forbidden = [
        "chunk: Vec<GemmaBpeMerge>",
        "chunk: Vec<GemmaTokenIdEntry>",
        "input: RecurSequenceInput<Vec<GemmaBpeMerge>>",
        "input: RecurSequenceInput<Vec<GemmaTokenIdEntry>>",
        "input: RecurSequenceInput<String>",
        "pairs: GemmaBpeAdjacentPairs",
        "final_pieces: BpePieces",
    ];
    for (name, src) in PROGRAM_SOURCES {
        for needle in forbidden {
            assert!(
                !src.contains(needle),
                "{name} must not use eager large prompt/tokenizer parameter `{needle}`"
            );
        }
    }
}

// --- 2. Trace-shape assertion ----------------------------------------------

/// Long unique sentinels so a byte scan over trace payloads cannot false
/// positive. The merged token is deliberately unrelated text: a candidate's
/// merged token is a permitted scalar decision leg and may ride inline in
/// the scan state, so the scan targets only the piece sentinels.
const LEFT_PIECE: &str = "SENTINEL_LEFT_PIECE_0123456789";
const RIGHT_PIECE: &str = "SENTINEL_RIGHT_PIECE_0123456789";
const THIRD_PIECE: &str = "SENTINEL_THIRD_PIECE_0123456789";
const MERGED_TOKEN: &str = "SENTINEL_MERGED_TOKEN_9876543210";

fn contains_marker(bytes: &[u8], marker: &str) -> bool {
    bytes
        .windows(marker.len())
        .any(|window| window == marker.as_bytes())
}

fn inline_piece_marker(input: &FnInput) -> bool {
    input.values.iter().any(|value| match value {
        FnInputValue::Inline(bytes) => [LEFT_PIECE, RIGHT_PIECE, THIRD_PIECE]
            .iter()
            .any(|marker| contains_marker(bytes, marker)),
        _ => false,
    })
}

fn input_data_piece_marker(input: &FnInput) -> bool {
    [LEFT_PIECE, RIGHT_PIECE, THIRD_PIECE]
        .iter()
        .any(|marker| contains_marker(&input.data, marker))
}

fn event_record(event: &TraceEvent) -> &raster::prelude::FnCallRecord {
    match event {
        TraceEvent::SequenceStart(record)
        | TraceEvent::SequenceEnd(record)
        | TraceEvent::RecurSequenceStart(record)
        | TraceEvent::RecurSequenceEnd(record)
        | TraceEvent::TileExec(record)
        | TraceEvent::RecurTileIterationExec(record)
        | TraceEvent::RecurTileExec(record)
        | TraceEvent::RecurSequenceExec(record) => record,
    }
}

#[test]
fn prompt_collections_cross_the_abi_only_as_authenticated_reads() {
    let (outcome, events) = capture_trace_events(|| {
        tokenize_with(
            vec![LEFT_PIECE, RIGHT_PIECE, THIRD_PIECE],
            vec![
                (LEFT_PIECE, 1),
                (RIGHT_PIECE, 2),
                (THIRD_PIECE, 3),
                (MERGED_TOKEN, 42),
            ],
            vec![(0, LEFT_PIECE, RIGHT_PIECE, MERGED_TOKEN, 42)],
            1,
        )
    });
    let tokenization = outcome.expect("sentinel prompt should tokenize");
    assert_eq!(tokenization.token_ids, vec![42, 3]);

    for event in &events {
        if let TraceEvent::RecurTileIterationExec(record) = event {
            assert_eq!(
                record.fn_name, "scan_one_merge_chunk",
                "only the scalar/ref scan recur tile may appear"
            );
            let input = record.input.as_ref().expect("recur tile input");
            assert!(
                input.data.len() < 512,
                "scan recur-tile input must stay scalar/ref sized, got {} bytes",
                input.data.len()
            );
            assert!(
                !input_data_piece_marker(input),
                "scan recur-tile input must not inline prompt pieces"
            );
        }
    }

    for event in &events {
        let record = event_record(event);
        let Some(input) = record.input.as_ref() else {
            continue;
        };
        if !inline_piece_marker(input) {
            if input_data_piece_marker(input) {
                assert_eq!(
                    record.fn_name, "publish_initial_pieces",
                    "prompt-derived pieces rode in FnInput.data for '{}' ({event:?}); \
                     only the initial publish tile may materialize the staged root",
                    record.fn_name
                );
            }
            continue;
        }
        panic!(
            "prompt-derived pieces rode inline into '{}' ({event:?}); \
             collections must cross as storage bindings or draft ops",
            record.fn_name
        );
    }
}

// --- 3. Functional matrix ---------------------------------------------------

fn chunk<T>(entries: Vec<T>, width: usize) -> Vec<Vec<T>> {
    let mut chunks: Vec<Vec<T>> = Vec::new();
    for entry in entries {
        match chunks.last_mut() {
            Some(chunk) if chunk.len() < width => chunk.push(entry),
            _ => chunks.push(vec![entry]),
        }
    }
    chunks
}

/// Drives the routine natively with an arbitrary fixture: sorted vocab
/// chunks, priority-ordered merge chunks, staged pieces behind the
/// selectable root as internal bindings (the committed-external shape).
fn tokenize_with(
    pieces: Vec<&str>,
    vocab: Vec<(&str, u32)>,
    merges: Vec<(u32, &str, &str, &str, u32)>,
    table_width: usize,
) -> core::result::Result<PromptTokenization, String> {
    let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_guard_tests");
    let mut vocab = vocab;
    vocab.sort_by(|left, right| left.0.cmp(right.0));
    let vocab_chunks = chunk(
        vocab
            .into_iter()
            .map(|(token, id)| GemmaTokenIdEntry {
                token: token.to_string(),
                id,
            })
            .collect(),
        table_width,
    );
    let merge_chunks = chunk(
        merges
            .into_iter()
            .map(
                |(merge_index, left, right, merged, token_id)| GemmaBpeMerge {
                    merge_index,
                    left: left.to_string(),
                    right: right.to_string(),
                    merged_token: merged.to_string(),
                    has_token_id: true,
                    token_id,
                },
            )
            .collect(),
        table_width,
    );

    let staged_pieces = raster::store_internal_value(&BpePieces {
        pieces: pieces.into_iter().map(ToString::to_string).collect(),
    })
    .expect("store staged pieces");
    let tokenizer = GemmaTokenizer {
        metadata: GemmaTokenizerMetadata {
            space_replacement: "\u{2581}".to_string(),
            split_delimiter: " ".to_string(),
            split_behavior: "merged_with_previous".to_string(),
            invert: false,
            unk_token: "<unk>".to_string(),
            fuse_unk: false,
            byte_fallback: true,
            ignore_merges: false,
        },
        decoder: GemmaDecoderMetadata {
            space_replacement: "\u{2581}".to_string(),
            byte_fallback: true,
            fuse_decoder: false,
        },
        token_lookup_chunks: vocab_chunks,
        tokens_by_id: Vec::new(),
        special_tokens: Vec::new(),
        merge_chunks,
    };
    let tokenizer = raster::store_internal_value(&tokenizer).expect("store tokenizer");
    materialize_auth_result::<PromptTokenization, _>(__raster_sequence_auth_tokenize_prompt_pieces(
        internal!(BpePieces, staged_pieces),
        TokenizerTables::internal(tokenizer),
    ))
}

#[test]
fn multi_piece_prompt_without_merges_keeps_every_piece() {
    let tokenization = tokenize_with(
        vec!["a", "c", "a"],
        vec![("a", 1), ("c", 7)],
        vec![(0, "a", "b", "ab", 3)],
        4,
    )
    .expect("tokenize");
    assert_eq!(tokenization.token_ids, vec![1, 7, 1]);
    assert_eq!(tokenization.token_count, 3);
}

#[test]
fn repeated_pieces_merge_at_every_occurrence() {
    let tokenization = tokenize_with(
        vec!["a", "b", "a", "b"],
        vec![("a", 1), ("b", 2), ("ab", 3)],
        vec![(0, "a", "b", "ab", 3)],
        4,
    )
    .expect("tokenize");
    assert_eq!(tokenization.token_ids, vec![3, 3]);
}

#[test]
fn several_rounds_cascade_to_the_fixed_point() {
    // Round 1: (a,b)@0 → [ab, a, b]; round 2: (a,b)@1 → [ab, ab];
    // round 3: (ab,ab)@0 → [abab]; round 4 converges (no-op).
    let tokenization = tokenize_with(
        vec!["a", "b", "a", "b"],
        vec![("a", 1), ("b", 2), ("ab", 3), ("abab", 9)],
        vec![(0, "a", "b", "ab", 3), (1, "ab", "ab", "abab", 9)],
        4,
    )
    .expect("tokenize");
    assert_eq!(tokenization.token_ids, vec![9]);
    assert_eq!(tokenization.token_count, 1);
}

#[test]
fn merge_rule_at_a_chunk_boundary_still_wins() {
    // Width 1 puts the matching rule alone in the second merge chunk; the
    // first chunk's rule matches nothing.
    let tokenization = tokenize_with(
        vec!["a", "b"],
        vec![("a", 1), ("b", 2), ("ab", 3)],
        vec![(0, "x", "y", "xy", 8), (1, "a", "b", "ab", 3)],
        1,
    )
    .expect("tokenize");
    assert_eq!(tokenization.token_ids, vec![3]);
}

#[test]
fn vocab_matches_at_chunk_boundaries_resolve() {
    // Width 2 over 4 sorted entries [a, b, c, d]: "b" is the last entry of
    // the first chunk and "c" the first entry of the second.
    let tokenization = tokenize_with(
        vec!["b", "c"],
        vec![("a", 1), ("b", 2), ("c", 3), ("d", 4)],
        vec![],
        2,
    )
    .expect("tokenize");
    assert_eq!(tokenization.token_ids, vec![2, 3]);
}
