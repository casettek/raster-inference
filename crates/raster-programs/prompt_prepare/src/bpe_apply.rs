//! BPE merge-round apply phase (sim `tiles.rs`: `init_bpe_merge_iteration`,
//! `apply_bpe_merge_chunk_or_complete`, `finalize_bpe_merge_iteration`).
//!
//! The next round's pieces accumulate in the loop-carried apply state (the
//! sim built them into a `bpe-pieces-{N+1}` store artifact — port-plan
//! deviation D6). The sim's `Complete`/`Applying` enum becomes the
//! `complete` flag on the iteration context (catalog C28: branching stays
//! inside tiles).

use alloc::format;
use raster::prelude::*;

use crate::types::{
    BpeConfig, GemmaBpeApplyState, GemmaBpeIterationContext, GemmaBpeLoopState,
    GemmaBpeMergeDecision,
};

/// Opens the apply phase (sim `tiles.rs:386-433`): a round with no
/// selection, a skipped round, or a deferred error applies nothing; the
/// sim's merge-range guard defers through the error field (port-plan
/// constraint A1).
#[tile]
pub fn init_bpe_merge_iteration(decision: GemmaBpeMergeDecision) -> GemmaBpeIterationContext {
    let mut error = decision.error;
    let selection = if decision.skip || error.is_some() {
        None
    } else {
        decision.selection
    };

    let Some(selection) = selection else {
        return GemmaBpeIterationContext {
            complete: true,
            round: decision.round,
            pieces: decision.pieces,
            merge_piece_idx: 0,
            merged: Default::default(),
            error,
        };
    };

    let piece_count = decision.pieces.len() as u32;
    if selection.pair_idx + 1 >= piece_count {
        error = Some(format!(
            "BPE merge index {} is out of range for {} pieces",
            selection.pair_idx, piece_count
        ));
        return GemmaBpeIterationContext {
            complete: true,
            round: decision.round,
            pieces: decision.pieces,
            merge_piece_idx: 0,
            merged: Default::default(),
            error,
        };
    }

    GemmaBpeIterationContext {
        complete: false,
        round: decision.round,
        pieces: decision.pieces,
        merge_piece_idx: selection.pair_idx,
        merged: selection.merged,
        error: None,
    }
}

/// Chunked pieces rebuild (sim `tiles.rs:435-516`): each iteration emits up
/// to `bpe_pieces_per_tile` output pieces; the merge point consumes two
/// input pieces and emits the merged token. Bounded input list +
/// `RecurControl::Break` per gap G1.
#[tile(kind = recur)]
pub fn apply_bpe_merge_chunk_or_complete(
    input: RecurInput<u32>,
    state: RecurState<GemmaBpeApplyState>,
    iteration: GemmaBpeIterationContext,
    config: BpeConfig,
) -> RecurControl<RecurState<GemmaBpeApplyState>> {
    let _chunk_ordinal = input.value();
    let mut state = state;
    if iteration.complete || state.done {
        state.done = true;
        return RecurControl::Break(state);
    }

    let max_output_cursor = iteration.pieces.len().saturating_sub(1) as u32;
    let output_limit = state
        .output_cursor
        .saturating_add(config.bpe_pieces_per_tile)
        .min(max_output_cursor);
    while state.output_cursor < output_limit {
        if state.input_cursor == iteration.merge_piece_idx {
            state.output.push(iteration.merged.clone());
            state.input_cursor += 2;
        } else {
            let piece = iteration.pieces[state.input_cursor as usize].clone();
            state.output.push(piece);
            state.input_cursor += 1;
        }
        state.output_cursor += 1;
    }

    if state.output_cursor >= max_output_cursor {
        state.done = true;
        RecurControl::Break(state)
    } else {
        RecurControl::Continue(state)
    }
}

/// Closes one merge round (sim `tiles.rs:519-560`): completion check
/// (error deferred per A1) and the next loop state — `complete` when the
/// round had nothing to apply, otherwise one piece fewer and the round
/// counter advanced.
#[tile]
pub fn finalize_bpe_merge_iteration(
    apply: GemmaBpeApplyState,
    iteration: GemmaBpeIterationContext,
) -> GemmaBpeLoopState {
    if iteration.complete {
        return GemmaBpeLoopState {
            initialized: true,
            complete: true,
            round: iteration.round,
            pieces: iteration.pieces,
            error: iteration.error,
        };
    }

    let expected_piece_count = iteration.pieces.len().saturating_sub(1) as u32;
    if apply.output_cursor != expected_piece_count {
        return GemmaBpeLoopState {
            initialized: true,
            complete: true,
            round: iteration.round,
            pieces: iteration.pieces,
            error: Some(format!(
                "BPE merge apply finalized with {} pieces, expected {expected_piece_count}",
                apply.output_cursor
            )),
        };
    }

    GemmaBpeLoopState {
        initialized: true,
        complete: false,
        round: iteration.round + 1,
        pieces: apply.output,
        error: None,
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::types::GemmaBpeScanCandidate;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_apply_tests");
        run()
    }

    fn decision(pieces: Vec<&str>, selection: Option<(u32, &str)>) -> GemmaBpeMergeDecision {
        GemmaBpeMergeDecision {
            skip: false,
            round: 0,
            pieces: pieces.into_iter().map(String::from).collect(),
            selection: selection.map(|(pair_idx, merged)| GemmaBpeScanCandidate {
                pair_idx,
                merge_index: 0,
                merged: merged.to_string(),
            }),
            error: None,
        }
    }

    fn config(pieces_per_tile: u32) -> BpeConfig {
        BpeConfig {
            bpe_pairs_per_tile: 64,
            bpe_pieces_per_tile: pieces_per_tile,
        }
    }

    fn run_apply(iteration: &GemmaBpeIterationContext, config: &BpeConfig) -> GemmaBpeApplyState {
        let mut state = GemmaBpeApplyState::initial();
        for chunk in 0u32..16 {
            let control = apply_bpe_merge_chunk_or_complete(
                RecurInput::new(chunk, chunk as u64, 16),
                RecurState::new(state),
                iteration.clone(),
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
    fn apply_rebuilds_pieces_around_the_merge_point() {
        let iteration = in_scope(|| {
            init_bpe_merge_iteration(decision(vec!["x", "a", "b", "y"], Some((1, "ab"))))
        });
        assert!(!iteration.complete);
        // Chunk width 1 exercises multi-iteration application.
        let state = in_scope(|| run_apply(&iteration, &config(1)));
        assert_eq!(
            state.output,
            vec!["x".to_string(), "ab".to_string(), "y".to_string()]
        );
        let next = in_scope(|| finalize_bpe_merge_iteration(state, iteration));
        assert!(!next.complete);
        assert_eq!(next.round, 1);
        assert_eq!(
            next.pieces,
            vec!["x".to_string(), "ab".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn converged_round_completes_the_loop_with_pieces_unchanged() {
        let iteration = in_scope(|| init_bpe_merge_iteration(decision(vec!["ab"], None)));
        assert!(iteration.complete);
        let state = in_scope(|| run_apply(&iteration, &config(8)));
        assert!(state.output.is_empty());
        let next = in_scope(|| finalize_bpe_merge_iteration(state, iteration));
        assert!(next.complete);
        assert!(next.error.is_none());
        assert_eq!(next.pieces, vec!["ab".to_string()]);
    }

    #[test]
    fn out_of_range_merge_defers_the_sim_error() {
        let iteration =
            in_scope(|| init_bpe_merge_iteration(decision(vec!["a", "b"], Some((1, "ab")))));
        assert!(iteration.complete);
        assert_eq!(
            iteration.error.as_deref(),
            Some("BPE merge index 1 is out of range for 2 pieces")
        );
        let next =
            in_scope(|| finalize_bpe_merge_iteration(GemmaBpeApplyState::initial(), iteration));
        assert!(next.complete);
        assert!(next.error.is_some());
    }

    #[test]
    fn incomplete_apply_defers_the_sim_error() {
        let iteration =
            in_scope(|| init_bpe_merge_iteration(decision(vec!["a", "b", "c"], Some((0, "ab")))));
        let stalled = GemmaBpeApplyState {
            output: vec!["ab".to_string()],
            input_cursor: 2,
            output_cursor: 1,
            done: false,
        };
        let next = in_scope(|| finalize_bpe_merge_iteration(stalled, iteration));
        assert!(next.complete);
        assert_eq!(
            next.error.as_deref(),
            Some("BPE merge apply finalized with 1 pieces, expected 2")
        );
    }
}
