//! Step 0 probes for the storage refactor: verify the two load-bearing
//! mechanics before rewriting the phases (plan
//! `prompt.prepare storage refactor`).
//!
//! 1. `Vec<Vec<GemmaBpeMerge>>` works as a `Selectable` struct field
//!    (`select!` of the whole chunk list, one chunk, and one nested entry).
//! 2. A `#[tile(kind = recur)]` over a chunked list (`RecurInput<Vec<T>>`)
//!    with `RecurControl::Break` mid-list behaves as expected (the flat
//!    case is WS1 probe P1).
//!
//! Extended for the trace-slimming ref-based refactor (plan
//! `trace-slimming ref-based refactor`, Step 0):
//!
//! 3. A `#[sequence(kind = recur)]` over a chunked list
//!    (`RecurSequenceInput<Vec<T>>`) whose body passes **both** the opaque
//!    item handle and the threaded state into one plain tile, the tile's
//!    `AuthRef` return re-entering the state.
//! 4. That recur sequence nested **inside** another recur-sequence
//!    iteration (the shape of the BPE round loop around the scan).
//!
//! Test-only module: these tiles are evidence, not program surface.

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster::Selectable;
use raster_program_gemma_externals::types::GemmaBpeMerge;
use serde::{Deserialize, Serialize};

/// Chunked-table shape mirroring the revised `GemmaTokenizer` fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbeTable {
    pub chunks: Vec<Vec<GemmaBpeMerge>>,
}

/// Materialized outcome of the nested selections.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbeSelection {
    pub chunk_len: u32,
    pub entry_merge_index: u32,
    pub entry_merged: String,
}

/// Loop state of the chunked scan probe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeScanState {
    pub visited_chunks: u32,
    pub found: bool,
    pub found_merge_index: u32,
}

#[tile]
pub fn summarize_selection(chunk: Vec<GemmaBpeMerge>, entry: GemmaBpeMerge) -> ProbeSelection {
    ProbeSelection {
        chunk_len: chunk.len() as u32,
        entry_merge_index: entry.merge_index,
        entry_merged: entry.merged_token,
    }
}

/// Mechanic 1: nested-list selection through a `Selectable` field — the
/// whole chunk list, a single chunk (`[1]`), and a nested entry (`[1][0]`).
#[sequence]
pub fn probe_select_chunks(table: ProbeTable) -> ProbeSelection {
    let second_chunk = select!(Vec<GemmaBpeMerge>, table.clone().chunks[1]);
    let first_of_second = select!(GemmaBpeMerge, table.chunks[1][0]);
    call!(summarize_selection, second_chunk, first_of_second)
}

/// Mechanic 2: one chunk per recur iteration; `Break` as soon as a rule
/// matching `needle_left` is found (mid-list for the fixture).
#[tile(kind = recur)]
pub fn probe_scan_chunk(
    input: RecurInput<Vec<GemmaBpeMerge>>,
    state: RecurState<ProbeScanState>,
    needle_left: String,
) -> RecurControl<RecurState<ProbeScanState>> {
    let mut state = state;
    state.visited_chunks += 1;
    for merge in input.value() {
        if merge.left == needle_left {
            state.found = true;
            state.found_merge_index = merge.merge_index;
            return RecurControl::Break(state);
        }
    }
    RecurControl::Continue(state)
}

/// Drives the chunked recur over a `select!`-ed `Vec<Vec<GemmaBpeMerge>>`
/// field — the exact shape the refactored phases use.
#[sequence]
pub fn probe_chunked_recur(table: ProbeTable, needle_left: String) -> ProbeScanState {
    let chunks = select!(Vec<Vec<GemmaBpeMerge>>, table.chunks);
    call_recur!(
        tile = probe_scan_chunk,
        input = chunks,
        state = ProbeScanState {
            visited_chunks: 0,
            found: false,
            found_merge_index: 0,
        },
        args = (needle_left,)
    )
}

/// Nested-fixture shape for mechanic 4: the chunked table plus a bounded
/// round list, both selected out of one committed source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbeNestedFixture {
    pub chunks: Vec<Vec<GemmaBpeMerge>>,
    pub rounds: Vec<u32>,
}

/// Loop state threaded through the mechanic-4 outer round sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeRoundState {
    pub rounds_run: u32,
    pub total_visited_chunks: u32,
    pub found_rounds: u32,
    pub found_merge_index: u32,
}

/// Mechanic 3's per-chunk tile: the chunk arrives as a selection binding,
/// the tiny scan state inline; already-found scans no-op (recur sequences
/// cannot break early — gap G1).
#[tile]
pub fn probe_scan_one_chunk(
    state: ProbeScanState,
    chunk: Vec<GemmaBpeMerge>,
    needle_left: String,
) -> ProbeScanState {
    let mut state = state;
    if state.found {
        return state;
    }
    state.visited_chunks += 1;
    for merge in chunk {
        if merge.left == needle_left {
            state.found = true;
            state.found_merge_index = merge.merge_index;
            break;
        }
    }
    state
}

/// Mechanic 3: a recur *sequence* over the chunked list whose body passes
/// both the opaque item handle and the threaded state into one plain tile;
/// the tile's `AuthRef` return re-enters the state.
#[sequence(kind = recur)]
pub fn probe_scan_chunks_seq(
    input: RecurSequenceInput<Vec<GemmaBpeMerge>>,
    state: RecurSequenceState<ProbeScanState>,
    needle_left: String,
) -> RecurSequenceState<ProbeScanState> {
    call!(probe_scan_one_chunk, state, input, needle_left)
}

/// Drives mechanic 3 directly over a `select!`-ed chunk list.
#[sequence]
pub fn probe_chunked_recur_seq(table: ProbeTable, needle_left: String) -> ProbeScanState {
    let chunks = select!(Vec<Vec<GemmaBpeMerge>>, table.chunks);
    call_recur_seq!(
        sequence = probe_scan_chunks_seq,
        input = chunks,
        state = ProbeScanState {
            visited_chunks: 0,
            found: false,
            found_merge_index: 0,
        },
        args = (needle_left,)
    )
}

/// Folds one round's scan outcome into the outer round state.
#[tile]
pub fn probe_fold_round(state: ProbeRoundState, scan: ProbeScanState) -> ProbeRoundState {
    ProbeRoundState {
        rounds_run: state.rounds_run + 1,
        total_visited_chunks: state.total_visited_chunks + scan.visited_chunks,
        found_rounds: state.found_rounds + u32::from(scan.found),
        found_merge_index: if scan.found {
            scan.found_merge_index
        } else {
            state.found_merge_index
        },
    }
}

/// Mechanic 4: the mechanic-3 recur sequence nested inside another
/// recur-sequence iteration (the round-loop-around-scan shape).
#[sequence(kind = recur)]
pub fn probe_round_seq(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<ProbeRoundState>,
    chunks: Vec<Vec<GemmaBpeMerge>>,
    needle_left: String,
) -> RecurSequenceState<ProbeRoundState> {
    let _round_ordinal = &input;
    let scan = call_recur_seq!(
        sequence = probe_scan_chunks_seq,
        input = chunks,
        state = ProbeScanState {
            visited_chunks: 0,
            found: false,
            found_merge_index: 0,
        },
        args = (needle_left,)
    );
    call!(probe_fold_round, state, scan)
}

/// Drives mechanic 4: outer recur sequence over the round list, inner
/// recur sequence over the chunked table.
#[sequence]
pub fn probe_nested_recur_seq(
    fixture: ProbeNestedFixture,
    needle_left: String,
) -> ProbeRoundState {
    let chunks = select!(Vec<Vec<GemmaBpeMerge>>, fixture.clone().chunks);
    let rounds = select!(Vec<u32>, fixture.rounds);
    call_recur_seq!(
        sequence = probe_round_seq,
        input = rounds,
        state = ProbeRoundState {
            rounds_run: 0,
            total_visited_chunks: 0,
            found_rounds: 0,
            found_merge_index: 0,
        },
        args = (chunks, needle_left)
    )
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_chunk_probes");
        run()
    }

    fn merge(merge_index: u32, left: &str, right: &str) -> GemmaBpeMerge {
        GemmaBpeMerge {
            merge_index,
            left: left.to_string(),
            right: right.to_string(),
            merged_token: alloc::format!("{left}{right}"),
            has_token_id: true,
            token_id: merge_index,
        }
    }

    fn fixture() -> ProbeTable {
        ProbeTable {
            chunks: vec![
                vec![merge(0, "a", "b"), merge(1, "b", "a")],
                vec![merge(2, "c", "d"), merge(3, "d", "c"), merge(4, "e", "f")],
                vec![merge(5, "g", "h")],
            ],
        }
    }

    #[test]
    fn nested_chunk_lists_select_through_the_schema() {
        let selection = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeSelection, _>(
                __raster_sequence_auth_probe_select_chunks(internal!(ProbeTable, stored)),
            )
        });
        assert_eq!(
            selection,
            ProbeSelection {
                chunk_len: 3,
                entry_merge_index: 2,
                entry_merged: "cd".to_string(),
            }
        );
    }

    #[test]
    fn chunked_recur_input_breaks_mid_list() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur(
                    internal!(ProbeTable, stored),
                    "d".to_string(),
                ),
            )
        });
        assert!(state.found, "needle rule lives in the second chunk");
        assert_eq!(state.found_merge_index, 3);
        assert_eq!(
            state.visited_chunks, 2,
            "Break must stop the scan before the third chunk"
        );
    }

    #[test]
    fn chunked_recur_input_visits_every_chunk_without_a_match() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur(
                    internal!(ProbeTable, stored),
                    "z".to_string(),
                ),
            )
        });
        assert!(!state.found);
        assert_eq!(state.visited_chunks, 3);
    }

    #[test]
    fn recur_sequence_passes_chunk_handle_and_state_into_one_tile() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur_seq(
                    internal!(ProbeTable, stored),
                    "d".to_string(),
                ),
            )
        });
        assert!(state.found, "needle rule lives in the second chunk");
        assert_eq!(state.found_merge_index, 3);
        assert_eq!(
            state.visited_chunks, 2,
            "post-match chunks must no-op (recur sequences cannot break)"
        );
    }

    #[test]
    fn recur_sequence_without_a_match_scans_every_chunk() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur_seq(
                    internal!(ProbeTable, stored),
                    "z".to_string(),
                ),
            )
        });
        assert!(!state.found);
        assert_eq!(state.visited_chunks, 3);
    }

    #[test]
    fn recur_sequence_nests_inside_a_recur_sequence_iteration() {
        let nested_fixture = ProbeNestedFixture {
            chunks: fixture().chunks,
            rounds: vec![0, 1],
        };
        let state = in_scope(|| {
            let stored =
                raster::store_internal_value(&nested_fixture).expect("store nested fixture");
            raster::materialize_auth_return::<ProbeRoundState, _>(
                __raster_sequence_auth_probe_nested_recur_seq(
                    internal!(ProbeNestedFixture, stored),
                    "d".to_string(),
                ),
            )
        });
        assert_eq!(state.rounds_run, 2, "every round ordinal must execute");
        assert_eq!(state.found_rounds, 2, "each round's inner scan must find the rule");
        assert_eq!(state.found_merge_index, 3);
        assert_eq!(
            state.total_visited_chunks, 4,
            "each round scans two chunks before the match no-ops the rest"
        );
    }
}
