//! BPE merge-round scan phase (sim `tiles.rs`: `init_bpe_merge_scan`,
//! `scan_bpe_merge_candidates`, `finalize_bpe_merge_scan`).
//!
//! Trace-slimming shape (port-plan deviation D13, supersedes D10's
//! recur-tile inputs): the chunk loop is a `#[sequence(kind = recur)]` over
//! the model-scoped merge table (`merge_chunks`, pre-chunked in the
//! tokenizer external, priority order preserved). Each iteration passes the
//! opaque chunk handle into one plain tile, so the chunk crosses the tile
//! ABI as an external-selection binding (~100B commitment + selector in the
//! trace) and materializes only at tile execution. The scan visits rules in
//! priority order; the first rule with an adjacent-pair occurrence in the
//! round's pieces wins (lowest `merge_index` globally, leftmost pair).
//! Recur sequences cannot break early (gap G1), so post-winner and skipped
//! chunks no-op. Semantically equivalent to the sim's min-rank/earliest-pair
//! selection over pairs (proven by the equivalence tests below).
//!
//! The prompt-scoped round context (pieces, flags) rides `args`.

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaBpeMerge;

use crate::types::{
    GemmaBpeLoopState, GemmaBpeMergeDecision, GemmaBpeRoundContext, GemmaBpeScanCandidate,
    GemmaBpeScanState,
};

/// Leftmost adjacent-pair occurrence of `(left, right)` in `pieces`. Plain
/// helper — only reachable from tile bodies.
pub(crate) fn find_leftmost_pair(pieces: &[String], left: &str, right: &str) -> Option<u32> {
    pieces
        .windows(2)
        .position(|pair| pair[0] == left && pair[1] == right)
        .map(|pair_idx| pair_idx as u32)
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

/// One merge-table chunk per execution: checks its rules, in order, for an
/// adjacent-pair occurrence in the round's pieces. The first hit is the
/// global winner — chunks preserve priority order — with the leftmost pair
/// occurrence; it sets `best` + `done` and every later chunk no-ops (as do
/// all chunks of a skipped round). Infallible per port-plan constraint A1.
#[tile]
pub fn scan_one_merge_chunk(
    state: GemmaBpeScanState,
    chunk: Vec<GemmaBpeMerge>,
    round: GemmaBpeRoundContext,
) -> GemmaBpeScanState {
    let mut state = state;
    if round.skip || state.done {
        return state;
    }

    state.chunks_scanned += 1;
    for rule in chunk {
        if let Some(pair_idx) = find_leftmost_pair(&round.pieces, &rule.left, &rule.right) {
            state.best = Some(GemmaBpeScanCandidate {
                pair_idx,
                merge_index: rule.merge_index,
                merged: rule.merged_token,
            });
            state.done = true;
            return state;
        }
    }
    state
}

/// Priority-order scan loop: a recur sequence over the chunked merge table.
/// The chunk reaches `scan_one_merge_chunk` as an external-selection
/// binding; only the tiny scan state and the prompt-scoped round context
/// ride inline.
#[sequence(kind = recur)]
pub fn scan_merge_chunks(
    input: RecurSequenceInput<Vec<GemmaBpeMerge>>,
    state: RecurSequenceState<GemmaBpeScanState>,
    round: GemmaBpeRoundContext,
) -> RecurSequenceState<GemmaBpeScanState> {
    call!(scan_one_merge_chunk, state, input, round)
}

/// Closes the scan phase: decision assembly. `selection: None` means the
/// round converged (no merge rule matches any adjacent pair — the scan
/// exhausted the table). Deferred round errors ride through (A1).
#[tile]
pub fn finalize_bpe_merge_scan(
    scan: GemmaBpeScanState,
    round: GemmaBpeRoundContext,
) -> GemmaBpeMergeDecision {
    let selection = if round.skip || round.error.is_some() {
        None
    } else {
        scan.best
    };
    GemmaBpeMergeDecision {
        skip: round.skip,
        round: round.round,
        pieces: round.pieces,
        selection,
        error: round.error,
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_scan_tests");
        run()
    }

    fn merge_rule(merge_index: u32, left: &str, right: &str) -> GemmaBpeMerge {
        GemmaBpeMerge {
            merge_index,
            left: left.to_string(),
            right: right.to_string(),
            merged_token: format!("{left}{right}"),
            has_token_id: true,
            token_id: merge_index,
        }
    }

    /// Priority-ordered rules split into `width`-wide chunks (the encoder's
    /// `chunk_table` shape).
    fn chunked(rules: Vec<GemmaBpeMerge>, width: usize) -> Vec<Vec<GemmaBpeMerge>> {
        let mut chunks: Vec<Vec<GemmaBpeMerge>> = Vec::new();
        for rule in rules {
            match chunks.last_mut() {
                Some(chunk) if chunk.len() < width => chunk.push(rule),
                _ => chunks.push(vec![rule]),
            }
        }
        chunks
    }

    /// Drives the per-chunk tile over every chunk — the recur sequence's
    /// full pass (no early break).
    fn run_scan(
        round: &GemmaBpeRoundContext,
        merge_chunks: &[Vec<GemmaBpeMerge>],
    ) -> GemmaBpeScanState {
        let mut state = GemmaBpeScanState::initial();
        for chunk in merge_chunks {
            state = scan_one_merge_chunk(state, chunk.clone(), round.clone());
        }
        state
    }

    /// The sim's selection rule, as specified by `tiles.rs:320-329`: lowest
    /// rank (merge_index) wins; ties keep the earlier pair occurrence.
    fn sim_reference_selection(
        pieces: &[String],
        rules: &[GemmaBpeMerge],
    ) -> Option<GemmaBpeScanCandidate> {
        let mut best: Option<GemmaBpeScanCandidate> = None;
        for pair_idx in 0..pieces.len().saturating_sub(1) {
            let left = &pieces[pair_idx];
            let right = &pieces[pair_idx + 1];
            let Some(rule) = rules
                .iter()
                .find(|rule| &rule.left == left && &rule.right == right)
            else {
                continue;
            };
            let better = match &best {
                Some(candidate) => rule.merge_index < candidate.merge_index,
                None => true,
            };
            if better {
                best = Some(GemmaBpeScanCandidate {
                    pair_idx: pair_idx as u32,
                    merge_index: rule.merge_index,
                    merged: rule.merged_token.clone(),
                });
            }
        }
        best
    }

    fn round_for(pieces: Vec<&str>) -> GemmaBpeRoundContext {
        in_scope(|| {
            init_bpe_merge_scan(
                GemmaBpeLoopState::initial(),
                pieces
                    .into_iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
            )
        })
    }

    #[test]
    fn first_round_initializes_from_staged_pieces() {
        let round = round_for(vec!["a", "b"]);
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
    fn skipped_rounds_no_op_every_chunk() {
        let mut state = GemmaBpeLoopState::initial();
        state.initialized = true;
        state.complete = true;
        state.pieces = vec!["ab".to_string()];
        let round = in_scope(|| init_bpe_merge_scan(state, vec![]));
        let chunks = chunked(vec![merge_rule(0, "a", "b"), merge_rule(1, "b", "a")], 1);
        let scan = in_scope(|| run_scan(&round, &chunks));
        assert_eq!(scan.chunks_scanned, 0, "skip must not scan any rules");
        assert!(scan.best.is_none());
        assert!(!scan.done);
    }

    #[test]
    fn priority_order_wins_over_pair_position() {
        // Rule 0 ("b","a") matches at pair 1; rule 1 ("a","b") matches at
        // pair 0. Priority (merge_index 0) must win even though its pair
        // occurs later in the pieces.
        let round = round_for(vec!["a", "b", "a"]);
        let rules = vec![merge_rule(0, "b", "a"), merge_rule(1, "a", "b")];
        let chunks = chunked(rules.clone(), 1);
        let scan = in_scope(|| run_scan(&round, &chunks));
        let best = scan.best.clone().expect("candidate should be found");
        assert_eq!(best.merge_index, 0);
        assert_eq!(best.pair_idx, 1);
        assert_eq!(best.merged, "ba");
        assert!(scan.done, "the winning chunk must set done");
        assert_eq!(
            scan.chunks_scanned, 1,
            "chunks after the winner must no-op"
        );
        assert_eq!(
            Some(best),
            sim_reference_selection(&round.pieces, &rules),
            "priority scan must match the sim's min-rank selection"
        );
    }

    #[test]
    fn repeated_pairs_keep_the_leftmost_occurrence() {
        // Sim tie case: the same rule matches at pairs 0 and 2; the earlier
        // occurrence wins.
        let round = round_for(vec!["a", "b", "a", "b"]);
        let rules = vec![merge_rule(0, "a", "b"), merge_rule(1, "b", "a")];
        let chunks = chunked(rules.clone(), 2);
        let scan = in_scope(|| run_scan(&round, &chunks));
        let best = scan.best.expect("candidate should be found");
        assert_eq!(best.pair_idx, 0, "ties keep the earlier candidate");
        assert_eq!(best.merge_index, 0);
        assert_eq!(best.merged, "ab");
        assert_eq!(
            Some(best),
            sim_reference_selection(&round.pieces, &rules),
            "leftmost-occurrence rule must match the sim's tie handling"
        );
    }

    #[test]
    fn chunk_width_does_not_change_the_selection() {
        let pieces = vec!["a", "b", "a", "c", "a", "b"];
        let rules = vec![
            merge_rule(0, "c", "a"),
            merge_rule(1, "a", "b"),
            merge_rule(2, "b", "a"),
            merge_rule(3, "a", "c"),
        ];
        let reference = {
            let round = round_for(pieces.clone());
            sim_reference_selection(&round.pieces, &rules)
        };
        for width in 1..=4 {
            let round = round_for(pieces.clone());
            let chunks = chunked(rules.clone(), width);
            let scan = in_scope(|| run_scan(&round, &chunks));
            assert_eq!(
                scan.best, reference,
                "chunk width {width} must not change the winner"
            );
        }
    }

    #[test]
    fn scan_without_candidates_converges_after_a_full_pass() {
        let round = round_for(vec!["x", "y"]);
        let chunks = chunked(vec![merge_rule(0, "a", "b"), merge_rule(1, "b", "a")], 1);
        let scan = in_scope(|| run_scan(&round, &chunks));
        assert!(scan.best.is_none());
        assert!(!scan.done, "no winner: done stays clear");
        assert_eq!(
            scan.chunks_scanned, 2,
            "a converged round pays one full table pass"
        );
        let decision = in_scope(|| finalize_bpe_merge_scan(scan, round));
        assert!(decision.selection.is_none());
        assert!(decision.error.is_none());
    }

    #[test]
    fn deferred_round_errors_suppress_the_selection() {
        let mut round = round_for(vec!["a", "b"]);
        round.error = Some("BPE merge apply finalized with 0 pieces, expected 1".to_string());
        let mut scan = GemmaBpeScanState::initial();
        scan.best = Some(GemmaBpeScanCandidate {
            pair_idx: 0,
            merge_index: 0,
            merged: "ab".to_string(),
        });
        scan.done = true;
        let decision = in_scope(|| finalize_bpe_merge_scan(scan, round));
        assert!(decision.selection.is_none());
        assert_eq!(
            decision.error.as_deref(),
            Some("BPE merge apply finalized with 0 pieces, expected 1")
        );
    }
}
