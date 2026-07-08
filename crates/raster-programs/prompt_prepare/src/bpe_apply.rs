//! BPE merge-round apply phase (sim `tiles.rs`: `init_bpe_merge_iteration`,
//! `apply_bpe_merge_chunk_or_complete`, `finalize_bpe_merge_iteration`).
//!
//! Storage-resident shape (deviation D15, supersedes the apply legs of
//! D6/D9): the next round's pieces accumulate in a fresh
//! `RecurOutput<BpePieces>` draft — the real-raster form of the sim's
//! `bpe-pieces-{N+1}` builder — through a recur *sequence* over the round's
//! pieces (piece ordinals for the opened round; not a recur tile — probe P4
//! / gap G7). Each iteration runs one plain tile threading the cursor-only
//! state and the draft; the tile reads the current piece from the round's
//! internal ref only when it needs to copy that piece. Skipped rounds and
//! post-merge-completion iterations no-op (no `Break` — gap G1).
//!
//! `finalize_round` closes the round: count check preserved, deferred
//! errors preserved (A1), and a round with no selection carries the
//! incoming pieces forward unchanged — never the empty draft.

use alloc::format;
use raster::prelude::*;
use raster::InternalRef;

use crate::types::{
    read_bpe_piece, BpePieces, BpePiecesDraftExt, GemmaBpeApplyCursor, GemmaBpeApplyDecision,
    GemmaBpeLoopState, GemmaBpeOpenedRound,
};

/// One piece per execution: pushes the piece (or the merged token at the
/// merge point) into the next round's draft and advances the cursor.
/// `emitted == merge_piece_idx` fires exactly once — before the merge every
/// piece is pushed, so `emitted` tracks the input index until the winning
/// pair's left piece; `skip_next` swallows the consumed right-hand piece.
/// Skip rounds no-op every iteration. Infallible per port-plan constraint
/// A1.
#[tile]
pub fn apply_one_piece(
    state: GemmaBpeApplyCursor,
    piece_idx: u32,
    output: Draft<BpePieces>,
    decision: GemmaBpeApplyDecision,
    opened: GemmaBpeOpenedRound,
) -> (RecurState<GemmaBpeApplyCursor>, RecurOutput<BpePieces>) {
    let mut state = state;
    let mut output = output;
    if decision.skip {
        return (RecurState::new(state), output);
    }
    if state.skip_next {
        state.skip_next = false;
        return (RecurState::new(state), output);
    }
    if state.emitted == decision.merge_piece_idx {
        output.pieces().push(decision.merged);
        state.skip_next = true;
    } else {
        let piece = read_bpe_piece(&opened.pieces_ref, piece_idx);
        output.pieces().push(piece);
    }
    state.emitted += 1;
    (RecurState::new(state), output)
}

/// Per-piece apply loop: a state+output recur sequence over the round's
/// pieces. The draft is created fresh per round (`output = new!(…)` at the
/// call site) and finalizes into internal storage at loop end; only draft
/// ops and the tiny cursor cross the ABI per iteration.
#[sequence(kind = recur)]
pub fn apply_round_pieces(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<GemmaBpeApplyCursor>,
    output: RecurSequenceOutput<BpePieces>,
    decision: GemmaBpeApplyDecision,
    opened: GemmaBpeOpenedRound,
) -> (
    RecurSequenceState<GemmaBpeApplyCursor>,
    RecurSequenceOutput<BpePieces>,
) {
    let (cursor, output) = call!(apply_one_piece, state, input, output, decision, opened);
    let cursor: RecurSequenceState<GemmaBpeApplyCursor> = cursor.into_inner().into();
    let output: RecurSequenceOutput<BpePieces> = output.into();
    (cursor, output)
}

/// Closes one merge round (sim `tiles.rs:519-560`): reconstructs the
/// deferred error (A1), keeps the sim's range and completion checks with
/// their exact messages (H4), and produces the next loop state — `complete`
/// when the round had nothing to apply (incoming pieces carried forward
/// unchanged, never the empty draft), otherwise one piece fewer and the
/// round counter advanced. `applied` arrives through an authenticated read
/// of the finalized draft; the incoming pieces carry forward by handle on
/// skip/error paths.
#[tile]
pub fn finalize_round(
    decision: GemmaBpeApplyDecision,
    opened: GemmaBpeOpenedRound,
    prior_state: GemmaBpeLoopState,
    applied_ref: InternalRef,
) -> GemmaBpeLoopState {
    let error = opened.has_error.then_some(opened.error);
    if decision.skip {
        return GemmaBpeLoopState {
            complete: true,
            round: opened.round,
            piece_count: opened.piece_count,
            pieces_ref: prior_state.pieces_ref,
            error,
        };
    }

    if decision.merge_piece_idx + 1 >= opened.piece_count {
        return GemmaBpeLoopState {
            complete: true,
            round: opened.round,
            piece_count: opened.piece_count,
            pieces_ref: prior_state.pieces_ref,
            error: Some(format!(
                "BPE merge index {} is out of range for {} pieces",
                decision.merge_piece_idx, opened.piece_count
            )),
        };
    }

    let expected_piece_count = opened.piece_count.saturating_sub(1);
    let applied = raster::resolve_internal_value::<BpePieces>(applied_ref.clone())
        .unwrap_or_else(|error| panic!("Failed to resolve applied BPE pieces: {error}"))
        .value;
    let applied_count = applied.pieces.len() as u32;
    if applied_count != expected_piece_count {
        return GemmaBpeLoopState {
            complete: true,
            round: opened.round,
            piece_count: opened.piece_count,
            pieces_ref: prior_state.pieces_ref,
            error: Some(format!(
                "BPE merge apply finalized with {applied_count} pieces, \
                 expected {expected_piece_count}"
            )),
        };
    }

    GemmaBpeLoopState {
        complete: false,
        round: opened.round + 1,
        piece_count: expected_piece_count,
        pieces_ref: applied_ref,
        error: None,
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::bpe_scan::open_round;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_apply_tests");
        run()
    }

    fn pieces(items: Vec<&str>) -> BpePieces {
        BpePieces {
            pieces: items
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        }
    }

    fn state_for(pieces: &BpePieces) -> GemmaBpeLoopState {
        let pieces_ref = raster::store_internal_value(pieces).expect("store BPE pieces");
        let mut state = GemmaBpeLoopState::initial(pieces_ref);
        state.piece_count = pieces.pieces.len() as u32;
        state
    }

    fn opened(items: Vec<&str>) -> GemmaBpeOpenedRound {
        in_scope(|| open_round(state_for(&pieces(items))))
    }

    fn resolved_loop_pieces(state: &GemmaBpeLoopState) -> Vec<String> {
        raster::resolve_internal_value::<BpePieces>(state.pieces_ref.clone())
            .expect("resolve loop pieces")
            .value
            .pieces
    }

    fn finalize_for_test(
        applied: BpePieces,
        decision: GemmaBpeApplyDecision,
        opened: GemmaBpeOpenedRound,
    ) -> GemmaBpeLoopState {
        let prior_state = GemmaBpeLoopState {
            complete: false,
            round: opened.round,
            piece_count: opened.piece_count,
            pieces_ref: opened.pieces_ref.clone(),
            error: None,
        };
        finalize_for_test_with_prior(applied, decision, opened, prior_state)
    }

    fn finalize_for_test_with_prior(
        applied: BpePieces,
        decision: GemmaBpeApplyDecision,
        opened: GemmaBpeOpenedRound,
        prior_state: GemmaBpeLoopState,
    ) -> GemmaBpeLoopState {
        let applied_ref = raster::store_internal_value(&applied).expect("store applied pieces");
        finalize_round(decision, opened, prior_state, applied_ref)
    }

    fn merge_decision(merge_piece_idx: u32, merged: &str) -> GemmaBpeApplyDecision {
        GemmaBpeApplyDecision {
            skip: false,
            merge_piece_idx,
            merged: merged.to_string(),
        }
    }

    fn skip_decision() -> GemmaBpeApplyDecision {
        GemmaBpeApplyDecision {
            skip: true,
            merge_piece_idx: 0,
            merged: String::new(),
        }
    }

    /// Drives the per-piece tile over every input piece — the recur
    /// sequence's full pass — accumulating pushes in a plain vector in
    /// place of the draft (the draft mechanics themselves are probe P2 and
    /// routine-level evidence).
    fn run_apply(round: &GemmaBpeOpenedRound, decision: &GemmaBpeApplyDecision) -> BpePieces {
        in_scope(|| {
            let mut cursor = GemmaBpeApplyCursor::initial();
            let mut draft = raster::new_draft::<BpePieces>();
            for piece_idx in 0..round.piece_count {
                let (next, returned) =
                    apply_one_piece(cursor, piece_idx, draft, decision.clone(), round.clone());
                cursor = next.into_inner();
                draft = returned;
            }
            raster::materialize_auth_return::<BpePieces, _>(raster::finalize(draft))
        })
    }

    #[test]
    fn apply_rebuilds_pieces_around_the_merge_point() {
        in_scope(|| {
            let round = opened(vec!["x", "a", "b", "y"]);
            let decision = merge_decision(1, "ab");
            let applied = run_apply(&round, &decision);
            assert_eq!(
                applied.pieces,
                vec!["x".to_string(), "ab".to_string(), "y".to_string()]
            );
            let next = finalize_for_test(applied, decision, round);
            assert!(!next.complete);
            assert_eq!(next.round, 1);
            assert_eq!(next.piece_count, 3);
            assert_eq!(
                resolved_loop_pieces(&next),
                vec!["x".to_string(), "ab".to_string(), "y".to_string()]
            );
        });
    }

    #[test]
    fn merge_at_the_first_pair_swallows_the_right_piece() {
        in_scope(|| {
            let round = opened(vec!["a", "b", "c"]);
            let decision = merge_decision(0, "ab");
            let applied = run_apply(&round, &decision);
            assert_eq!(applied.pieces, vec!["ab".to_string(), "c".to_string()]);
        });
    }

    #[test]
    fn merge_at_the_last_pair_consumes_the_tail() {
        in_scope(|| {
            let round = opened(vec!["a", "b", "c"]);
            let decision = merge_decision(1, "bc");
            let applied = run_apply(&round, &decision);
            assert_eq!(applied.pieces, vec!["a".to_string(), "bc".to_string()]);
        });
    }

    #[test]
    fn skip_rounds_push_nothing() {
        let round = opened(vec!["a", "b"]);
        let applied = run_apply(&round, &skip_decision());
        assert!(applied.pieces.is_empty(), "skip iterations must no-op");
    }

    #[test]
    fn converged_round_completes_the_loop_with_pieces_unchanged() {
        // No selection: the incoming pieces carry forward unchanged — the
        // empty draft never becomes the current pieces.
        in_scope(|| {
            let round = opened(vec!["ab"]);
            let applied = run_apply(&round, &skip_decision());
            assert!(applied.pieces.is_empty());
            let next = finalize_for_test(applied, skip_decision(), round);
            assert!(next.complete);
            assert!(next.error.is_none());
            assert_eq!(resolved_loop_pieces(&next), vec!["ab".to_string()]);
            assert_eq!(next.piece_count, 1);
        });
    }

    #[test]
    fn skipped_rounds_preserve_the_deferred_error() {
        let next = in_scope(|| {
            let prior_state = state_for(&pieces(vec!["a"]));
            let round = GemmaBpeOpenedRound {
                skip: true,
                round: 0,
                piece_count: 1,
                has_error: true,
                error: "BPE merge apply finalized with 0 pieces, expected 1".to_string(),
                pieces_ref: prior_state.pieces_ref.clone(),
            };
            finalize_for_test_with_prior(
                BpePieces { pieces: vec![] },
                skip_decision(),
                round,
                prior_state,
            )
        });
        assert!(next.complete);
        assert_eq!(
            next.error.as_deref(),
            Some("BPE merge apply finalized with 0 pieces, expected 1")
        );
        assert_eq!(resolved_loop_pieces(&next), vec!["a".to_string()]);
    }

    #[test]
    fn out_of_range_merge_defers_the_sim_error() {
        in_scope(|| {
            let round = opened(vec!["a", "b"]);
            // A structurally impossible decision (pairs stop at piece_count−2):
            // the guard stays as a defensive sim-parity check.
            let decision = merge_decision(1, "ab");
            let next = finalize_for_test(BpePieces { pieces: vec![] }, decision, round);
            assert!(next.complete);
            assert_eq!(
                next.error.as_deref(),
                Some("BPE merge index 1 is out of range for 2 pieces")
            );
            assert_eq!(
                resolved_loop_pieces(&next),
                vec!["a".to_string(), "b".to_string()],
                "the incoming pieces carry forward on the error path"
            );
        });
    }

    #[test]
    fn incomplete_apply_defers_the_sim_error() {
        let round = opened(vec!["a", "b", "c"]);
        let decision = merge_decision(0, "ab");
        let stalled = BpePieces {
            pieces: vec!["ab".to_string()],
        };
        let next = in_scope(|| finalize_for_test(stalled, decision, round));
        assert!(next.complete);
        assert_eq!(
            next.error.as_deref(),
            Some("BPE merge apply finalized with 1 pieces, expected 2")
        );
    }
}
