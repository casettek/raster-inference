//! BPE merge-round scan phase (sim `tiles.rs`: `init_bpe_merge_scan`,
//! `scan_bpe_merge_candidates`, `finalize_bpe_merge_scan`).
//!
//! The sim's `auth_read(tokenizer, GemmaBpeMergeRequest)` becomes an in-tile
//! binary search over the tokenizer external's `merge_lookup` (sorted by
//! `(left, right)`) — port-plan deviation D7, the tokenizer-PoC idiom. The
//! merged token is captured from the lookup entry at scan time (D7b), so the
//! sim's separate `GemmaBpeMergedTokenRequest` read disappears.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaBpeMergeLookupEntry;

use crate::types::{
    BpeConfig, GemmaBpeLoopState, GemmaBpeMergeDecision, GemmaBpeRoundContext,
    GemmaBpeScanCandidate, GemmaBpeScanState,
};

/// Binary search of the sorted `(left, right)` merge table. Plain helper —
/// only reachable from tile bodies.
pub(crate) fn find_merge<'a>(
    merge_lookup: &'a [GemmaBpeMergeLookupEntry],
    left: &str,
    right: &str,
) -> Option<&'a GemmaBpeMergeLookupEntry> {
    merge_lookup
        .binary_search_by(|entry| {
            entry
                .left
                .as_str()
                .cmp(left)
                .then_with(|| entry.right.as_str().cmp(right))
        })
        .ok()
        .map(|idx| &merge_lookup[idx])
}

/// Opens one BPE merge round: resolves the round's working pieces (the
/// staged `initial_pieces` on the first executed round — port-plan
/// constraint A2 — and the loop state's pieces afterwards) and flags no-op
/// rounds (loop complete or errored; recur sequences cannot break early —
/// gap G1).
#[tile]
pub fn init_bpe_merge_scan(
    state: GemmaBpeLoopState,
    initial_pieces: Vec<String>,
) -> GemmaBpeRoundContext {
    let pieces = if state.initialized {
        state.pieces
    } else {
        initial_pieces
    };
    let pair_count = pieces.len().saturating_sub(1) as u32;
    GemmaBpeRoundContext {
        skip: state.complete || state.error.is_some(),
        round: state.round,
        pieces,
        pair_count,
        error: state.error,
    }
}

/// Chunked pair scan: each iteration inspects up to `bpe_pairs_per_tile`
/// adjacent pairs and keeps the candidate with the lowest merge priority
/// (`merge_index` — the sim's `rank`; ties keep the earlier candidate, sim
/// `tiles.rs:320-329`). Bounded input list + `RecurControl::Break` per gap
/// G1; infallible per port-plan constraint A1.
#[tile(kind = recur)]
pub fn scan_bpe_merge_candidates(
    input: RecurInput<u32>,
    state: RecurState<GemmaBpeScanState>,
    round: GemmaBpeRoundContext,
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
    config: BpeConfig,
) -> RecurControl<RecurState<GemmaBpeScanState>> {
    let _chunk_ordinal = input.value();
    let mut state = state;
    if round.skip || state.done {
        state.done = true;
        return RecurControl::Break(state);
    }

    let end_pair_idx = state
        .next_pair_idx
        .saturating_add(config.bpe_pairs_per_tile)
        .min(round.pair_count);
    for pair_idx in state.next_pair_idx..end_pair_idx {
        let left = &round.pieces[pair_idx as usize];
        let right = &round.pieces[pair_idx as usize + 1];
        if let Some(entry) = find_merge(&merge_lookup, left, right) {
            let better = match &state.best {
                Some(best) => entry.candidate.merge_index < best.merge_index,
                None => true,
            };
            if better {
                state.best = Some(GemmaBpeScanCandidate {
                    pair_idx,
                    merge_index: entry.candidate.merge_index,
                    merged: entry.candidate.merged_token.clone(),
                });
            }
        }
    }

    state.next_pair_idx = end_pair_idx;
    if state.next_pair_idx >= round.pair_count {
        state.done = true;
        RecurControl::Break(state)
    } else {
        RecurControl::Continue(state)
    }
}

/// Closes the scan phase: completion check (sim `tiles.rs:345-351`, error
/// deferred per A1) and decision assembly. `selection: None` means the
/// round converged (no merge candidate remains).
#[tile]
pub fn finalize_bpe_merge_scan(
    scan: GemmaBpeScanState,
    round: GemmaBpeRoundContext,
) -> GemmaBpeMergeDecision {
    let mut error = round.error;
    if error.is_none() && !round.skip && scan.next_pair_idx != round.pair_count {
        error = Some(format!(
            "BPE merge scan finalized at pair {}, expected {}",
            scan.next_pair_idx, round.pair_count
        ));
    }
    let selection = if round.skip || error.is_some() {
        None
    } else {
        scan.best
    };
    GemmaBpeMergeDecision {
        skip: round.skip,
        round: round.round,
        pieces: round.pieces,
        selection,
        error,
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;
    use raster_program_gemma_externals::types::GemmaBpeMergeCandidate;

    use super::*;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_scan_tests");
        run()
    }

    fn lookup_entry(left: &str, right: &str, merge_index: u32) -> GemmaBpeMergeLookupEntry {
        GemmaBpeMergeLookupEntry {
            left: left.to_string(),
            right: right.to_string(),
            candidate: GemmaBpeMergeCandidate {
                merge_index,
                merged_token: format!("{left}{right}"),
                has_token_id: true,
                token_id: 0,
            },
        }
    }

    fn sorted_lookup(mut entries: Vec<GemmaBpeMergeLookupEntry>) -> Vec<GemmaBpeMergeLookupEntry> {
        entries.sort_by(|a, b| a.left.cmp(&b.left).then_with(|| a.right.cmp(&b.right)));
        entries
    }

    fn config(pairs: u32) -> BpeConfig {
        BpeConfig {
            bpe_pairs_per_tile: pairs,
            bpe_pieces_per_tile: 64,
        }
    }

    fn run_scan(
        round: &GemmaBpeRoundContext,
        merge_lookup: &[GemmaBpeMergeLookupEntry],
        config: &BpeConfig,
    ) -> GemmaBpeScanState {
        let mut state = GemmaBpeScanState::initial();
        for chunk in 0u32..16 {
            let control = scan_bpe_merge_candidates(
                RecurInput::new(chunk, chunk as u64, 16),
                RecurState::new(state),
                round.clone(),
                merge_lookup.to_vec(),
                config.clone(),
            );
            match control {
                RecurControl::Continue(next) => state = next.into_inner(),
                RecurControl::Break(done) => {
                    state = done.into_inner();
                    break;
                }
            }
        }
        state
    }

    #[test]
    fn first_round_initializes_from_staged_pieces() {
        let round = in_scope(|| {
            init_bpe_merge_scan(
                GemmaBpeLoopState::initial(),
                vec!["a".to_string(), "b".to_string()],
            )
        });
        assert!(!round.skip);
        assert_eq!(round.pieces, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(round.pair_count, 1);
    }

    #[test]
    fn completed_loop_rounds_are_skipped() {
        let mut state = GemmaBpeLoopState::initial();
        state.initialized = true;
        state.complete = true;
        state.pieces = vec!["ab".to_string()];
        let round = in_scope(|| init_bpe_merge_scan(state, vec![]));
        assert!(round.skip);
        assert_eq!(round.pieces, vec!["ab".to_string()]);
    }

    #[test]
    fn scan_keeps_the_lowest_merge_index_and_earlier_candidate_on_ties() {
        let round = in_scope(|| {
            init_bpe_merge_scan(
                GemmaBpeLoopState::initial(),
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "a".to_string(),
                    "b".to_string(),
                ],
            )
        });
        let lookup = sorted_lookup(vec![lookup_entry("a", "b", 0), lookup_entry("b", "a", 1)]);
        // Chunk width 1 exercises multi-iteration scanning.
        let state = in_scope(|| run_scan(&round, &lookup, &config(1)));
        let best = state.best.expect("candidate should be found");
        assert_eq!(best.pair_idx, 0, "ties keep the earlier candidate");
        assert_eq!(best.merge_index, 0);
        assert_eq!(best.merged, "ab");
        assert!(state.done);
    }

    #[test]
    fn scan_without_candidates_converges() {
        let round = in_scope(|| {
            init_bpe_merge_scan(
                GemmaBpeLoopState::initial(),
                vec!["x".to_string(), "y".to_string()],
            )
        });
        let lookup = sorted_lookup(vec![lookup_entry("a", "b", 0)]);
        let state = in_scope(|| run_scan(&round, &lookup, &config(8)));
        assert!(state.best.is_none());
        let decision = in_scope(|| finalize_bpe_merge_scan(state, round));
        assert!(decision.selection.is_none());
        assert!(decision.error.is_none());
    }

    #[test]
    fn incomplete_scan_defers_the_sim_error() {
        let round = in_scope(|| {
            init_bpe_merge_scan(
                GemmaBpeLoopState::initial(),
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
            )
        });
        let stalled = GemmaBpeScanState {
            next_pair_idx: 1,
            best: None,
            done: false,
        };
        let decision = in_scope(|| finalize_bpe_merge_scan(stalled, round));
        assert_eq!(
            decision.error.as_deref(),
            Some("BPE merge scan finalized at pair 1, expected 2")
        );
        assert!(decision.selection.is_none());
    }
}
