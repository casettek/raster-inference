//! BPE merge-round scan phase (sim `tiles.rs`: `init_bpe_merge_scan`,
//! `scan_bpe_merge_candidates`, `finalize_bpe_merge_scan`).
//!
//! Storage-resident shape (deviations D15/D17, tightening D13): the round
//! opens through `open_round`, whose output carries the round scalars and
//! the round's pieces behind the selectable `BpePieces` root — pieces reach
//! everything downstream only via authenticated reads. `build_pairs`
//! publishes the round's adjacent-pair list as its own internal ref; the
//! per-chunk scan tile takes it as a selection-bound arg and materializes
//! the full prompt-scoped pair set at execution (rule-2 permitted read; gap
//! G6 computed-key selection is the named unlock for the sim's keyed-lookup
//! orientation).
//!
//! The scan keeps the D11 orientation: a recur sequence over the
//! model-scoped `merge_chunks` (priority order preserved); the first rule
//! with an adjacent-pair occurrence wins (lowest `merge_index` globally,
//! leftmost pair). Recur sequences cannot break early (gap G1), so
//! post-winner and skipped chunks no-op, and a converged round pays one
//! full table pass.

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaBpeMerge;

use crate::types::{
    BpePieces, GemmaBpeAdjacentPair, GemmaBpeAdjacentPairs, GemmaBpeApplyDecision,
    GemmaBpeLoopState, GemmaBpeOpenedRound, GemmaBpeScanCandidate, GemmaBpeScanState,
};

/// Leftmost occurrence of `(left, right)` in the round's adjacent-pair
/// list. Plain helper — only reachable from tile bodies.
pub(crate) fn find_leftmost_pair(
    pairs: &[GemmaBpeAdjacentPair],
    left: &str,
    right: &str,
) -> Option<u32> {
    pairs
        .iter()
        .position(|pair| pair.left == left && pair.right == right)
        .map(|pair_idx| pair_idx as u32)
}

/// Opens one BPE merge round: resolves the round's working pieces (the
/// staged `BpePieces` external on the first executed round — port-plan
/// constraint A2 — and the loop state's pieces afterwards), flags no-op
/// rounds (loop complete or errored; recur sequences cannot break early —
/// gap G1), and republishes the pieces behind the selectable root so the
/// rest of the round consumes them only through authenticated reads. The
/// deferred error flattens to `(has_error, error)` scalars (no `Option` in
/// selectable shapes — G3); the message text rides through untouched (H4).
#[tile]
pub fn open_round(state: GemmaBpeLoopState, staged: BpePieces) -> GemmaBpeOpenedRound {
    let pieces = if state.initialized {
        state.pieces
    } else {
        staged.pieces
    };
    let piece_count = pieces.len() as u32;
    let (has_error, error) = match state.error {
        Some(message) => (true, message),
        None => (false, String::new()),
    };
    GemmaBpeOpenedRound {
        skip: state.complete || has_error,
        round: state.round,
        piece_count,
        has_error,
        error,
        pieces: BpePieces { pieces },
    }
}

/// Derives the round's adjacent-pair list as its own internal ref,
/// consumed by the scan via selection. A separate tile (not folded into
/// `open_round`): it keeps the opened round free of derived data and each
/// tile at one job — simplicity and verifiability over the one extra
/// per-round execution (invariant rule 7).
#[tile]
pub fn build_pairs(pieces: BpePieces) -> GemmaBpeAdjacentPairs {
    GemmaBpeAdjacentPairs {
        pairs: pieces
            .pieces
            .windows(2)
            .map(|window| GemmaBpeAdjacentPair {
                left: window[0].clone(),
                right: window[1].clone(),
            })
            .collect(),
    }
}

/// One merge-table chunk per execution: checks its rules, in order, for an
/// occurrence in the round's adjacent pairs. The first hit is the global
/// winner — chunks preserve priority order — with the leftmost pair
/// occurrence; it sets `best` + `done` and every later chunk no-ops (as do
/// all chunks of a skipped round). Infallible per port-plan constraint A1.
#[tile]
pub fn scan_one_merge_chunk(
    state: GemmaBpeScanState,
    chunk: Vec<GemmaBpeMerge>,
    skip: bool,
    pairs: GemmaBpeAdjacentPairs,
) -> GemmaBpeScanState {
    let mut state = state;
    if skip || state.done {
        return state;
    }

    state.chunks_scanned += 1;
    for rule in chunk {
        if let Some(pair_idx) = find_leftmost_pair(&pairs.pairs, &rule.left, &rule.right) {
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
/// binding and the round's pairs as an internal-selection binding; only the
/// tiny scan state and the `skip` scalar ride inline.
#[sequence(kind = recur)]
pub fn scan_merge_chunks(
    input: RecurSequenceInput<Vec<GemmaBpeMerge>>,
    state: RecurSequenceState<GemmaBpeScanState>,
    skip: bool,
    pairs: GemmaBpeAdjacentPairs,
) -> RecurSequenceState<GemmaBpeScanState> {
    call!(scan_one_merge_chunk, state, input, skip, pairs)
}

/// Closes the scan phase into the apply decision, scalars only (D15).
/// `selection: None` (the scan exhausted the table) and skipped rounds
/// collapse into `skip`; `finalize_round` carries the incoming pieces
/// forward for both.
#[tile]
pub fn finalize_bpe_merge_scan(scan: GemmaBpeScanState, skip: bool) -> GemmaBpeApplyDecision {
    let selection = if skip { None } else { scan.best };
    match selection {
        Some(candidate) => GemmaBpeApplyDecision {
            skip: false,
            merge_piece_idx: candidate.pair_idx,
            merged: candidate.merged,
        },
        None => GemmaBpeApplyDecision {
            skip: true,
            merge_piece_idx: 0,
            merged: String::new(),
        },
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
        opened: &GemmaBpeOpenedRound,
        merge_chunks: &[Vec<GemmaBpeMerge>],
    ) -> GemmaBpeScanState {
        in_scope(|| {
            let pairs = build_pairs(opened.pieces.clone());
            let mut state = GemmaBpeScanState::initial();
            for chunk in merge_chunks {
                state = scan_one_merge_chunk(state, chunk.clone(), opened.skip, pairs.clone());
            }
            state
        })
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

    fn staged(pieces: Vec<&str>) -> BpePieces {
        BpePieces {
            pieces: pieces
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        }
    }

    fn round_for(pieces: Vec<&str>) -> GemmaBpeOpenedRound {
        in_scope(|| open_round(GemmaBpeLoopState::initial(), staged(pieces)))
    }

    #[test]
    fn first_round_initializes_from_staged_pieces() {
        let round = round_for(vec!["a", "b"]);
        assert!(!round.skip);
        assert!(!round.has_error);
        assert_eq!(round.pieces.pieces, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(round.piece_count, 2);
    }

    #[test]
    fn completed_loop_rounds_are_skipped() {
        let mut state = GemmaBpeLoopState::initial();
        state.initialized = true;
        state.complete = true;
        state.piece_count = 1;
        state.pieces = vec!["ab".to_string()];
        let round = in_scope(|| open_round(state, staged(vec![])));
        assert!(round.skip);
        assert_eq!(round.pieces.pieces, vec!["ab".to_string()]);
    }

    #[test]
    fn deferred_errors_flatten_to_scalars_with_the_message_intact() {
        let mut state = GemmaBpeLoopState::initial();
        state.initialized = true;
        state.piece_count = 2;
        state.pieces = vec!["a".to_string(), "b".to_string()];
        state.error = Some("BPE merge apply finalized with 0 pieces, expected 1".to_string());
        let round = in_scope(|| open_round(state, staged(vec![])));
        assert!(round.skip, "errored rounds must no-op");
        assert!(round.has_error);
        assert_eq!(
            round.error,
            "BPE merge apply finalized with 0 pieces, expected 1"
        );
    }

    #[test]
    fn pairs_are_the_adjacent_windows_of_the_round_pieces() {
        let pairs = in_scope(|| build_pairs(staged(vec!["a", "b", "a"])));
        assert_eq!(
            pairs.pairs,
            vec![
                GemmaBpeAdjacentPair {
                    left: "a".to_string(),
                    right: "b".to_string(),
                },
                GemmaBpeAdjacentPair {
                    left: "b".to_string(),
                    right: "a".to_string(),
                },
            ]
        );
        assert!(in_scope(|| build_pairs(staged(vec!["a"]))).pairs.is_empty());
        assert!(in_scope(|| build_pairs(staged(vec![]))).pairs.is_empty());
    }

    #[test]
    fn skipped_rounds_no_op_every_chunk() {
        let mut state = GemmaBpeLoopState::initial();
        state.initialized = true;
        state.complete = true;
        state.piece_count = 1;
        state.pieces = vec!["ab".to_string()];
        let round = in_scope(|| open_round(state, staged(vec![])));
        let chunks = chunked(vec![merge_rule(0, "a", "b"), merge_rule(1, "b", "a")], 1);
        let scan = run_scan(&round, &chunks);
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
        let scan = run_scan(&round, &chunks);
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
            sim_reference_selection(&round.pieces.pieces, &rules),
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
        let scan = run_scan(&round, &chunks);
        let best = scan.best.expect("candidate should be found");
        assert_eq!(best.pair_idx, 0, "ties keep the earlier candidate");
        assert_eq!(best.merge_index, 0);
        assert_eq!(best.merged, "ab");
        assert_eq!(
            Some(best),
            sim_reference_selection(&round.pieces.pieces, &rules),
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
            sim_reference_selection(&round.pieces.pieces, &rules)
        };
        for width in 1..=4 {
            let round = round_for(pieces.clone());
            let chunks = chunked(rules.clone(), width);
            let scan = run_scan(&round, &chunks);
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
        let scan = run_scan(&round, &chunks);
        assert!(scan.best.is_none());
        assert!(!scan.done, "no winner: done stays clear");
        assert_eq!(
            scan.chunks_scanned, 2,
            "a converged round pays one full table pass"
        );
        let decision = in_scope(|| finalize_bpe_merge_scan(scan, round.skip));
        assert!(decision.skip, "no selection collapses into the skip flag");
        assert!(decision.merged.is_empty());
    }

    #[test]
    fn skipped_rounds_suppress_the_selection() {
        // A stale candidate must not survive a skipped round (the deferred
        // error and completion cases both arrive as `skip`).
        let mut scan = GemmaBpeScanState::initial();
        scan.best = Some(GemmaBpeScanCandidate {
            pair_idx: 0,
            merge_index: 0,
            merged: "ab".to_string(),
        });
        scan.done = true;
        let decision = in_scope(|| finalize_bpe_merge_scan(scan, true));
        assert!(decision.skip);
        assert!(decision.merged.is_empty());
    }

    #[test]
    fn winning_candidates_become_the_apply_decision() {
        let mut scan = GemmaBpeScanState::initial();
        scan.best = Some(GemmaBpeScanCandidate {
            pair_idx: 2,
            merge_index: 7,
            merged: "ab".to_string(),
        });
        scan.done = true;
        let decision = in_scope(|| finalize_bpe_merge_scan(scan, false));
        assert!(!decision.skip);
        assert_eq!(decision.merge_piece_idx, 2);
        assert_eq!(decision.merged, "ab");
    }
}
