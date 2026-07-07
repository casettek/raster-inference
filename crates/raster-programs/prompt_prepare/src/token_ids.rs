//! Token-id finalization (sim `tiles.rs`: `finalize_bpe_tokenize_prompt` +
//! `init_token_id_finalization`, `finalize_next_token_ids`,
//! `finalize_tokenize_prompt`).
//!
//! Trace-slimming shape (port-plan deviation D14, supersedes D12's nested
//! per-piece loop): one inverted pass — a recur *sequence* over the
//! model-scoped `token_lookup_chunks` (sorted by token, pre-chunked in the
//! tokenizer external). Per chunk, one plain tile binary-searches every
//! still-unresolved final piece against the chunk; each chunk crosses the
//! tile ABI as an external-selection binding and materializes only at tile
//! execution. Once every piece resolves, the remaining chunks no-op (recur
//! sequences cannot break early — gap G1).
//!
//! `finalize_tokenize_prompt` surfaces the first vocab miss with the sim's
//! exact message, preserving the H4/A1 error contract.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaTokenIdEntry;

use crate::types::{GemmaBpeLoopState, GemmaTokenIdContext, GemmaTokenResolutionState,
    PromptTokenization};

/// Opens token-id finalization from the finished BPE loop state. The
/// zero-round case (a single-piece or empty prompt exhausts the round
/// budget without executing) leaves the loop state uninitialized; the
/// staged `initial_pieces` are the final pieces then.
#[tile]
pub fn init_token_id_finalization(
    loop_state: GemmaBpeLoopState,
    initial_pieces: Vec<String>,
) -> GemmaTokenIdContext {
    let pieces = if loop_state.initialized {
        loop_state.pieces
    } else {
        initial_pieces
    };
    let piece_count = pieces.len() as u32;
    GemmaTokenIdContext {
        pieces,
        piece_count,
        error: loop_state.error,
    }
}

/// One sorted vocab chunk per execution: binary-searches every still-
/// unresolved piece against the chunk. The resolution state sizes itself
/// from the context on the first executed iteration (A2: the loop seeds
/// from a literal); once every piece resolves, later chunks no-op.
/// Infallible per port-plan constraint A1: misses stay unresolved slots,
/// surfaced by `finalize_tokenize_prompt`.
#[tile]
pub fn resolve_pieces_in_vocab_chunk(
    state: GemmaTokenResolutionState,
    chunk: Vec<GemmaTokenIdEntry>,
    ctx: GemmaTokenIdContext,
) -> GemmaTokenResolutionState {
    let mut state = state;
    if !state.initialized {
        state.initialized = true;
        state.resolved = alloc::vec![None; ctx.pieces.len()];
    }
    if state.resolved.iter().all(|slot| slot.is_some()) {
        return state;
    }
    for (slot, piece) in state.resolved.iter_mut().zip(&ctx.pieces) {
        if slot.is_some() {
            continue;
        }
        if let Ok(idx) =
            chunk.binary_search_by(|entry| entry.token.as_str().cmp(piece.as_str()))
        {
            *slot = Some(chunk[idx].id);
        }
    }
    state
}

/// The inverted vocab pass: a recur sequence over the chunked vocab table.
/// The chunk reaches `resolve_pieces_in_vocab_chunk` as an external-
/// selection binding; only the prompt-scoped resolution state and piece
/// context ride inline.
#[sequence(kind = recur)]
pub fn resolve_vocab_chunk(
    input: RecurSequenceInput<Vec<GemmaTokenIdEntry>>,
    state: RecurSequenceState<GemmaTokenResolutionState>,
    ctx: GemmaTokenIdContext,
) -> RecurSequenceState<GemmaTokenResolutionState> {
    call!(resolve_pieces_in_vocab_chunk, state, input, ctx)
}

/// Closes tokenization (sim `tiles.rs:196-223`): surfaces every deferred
/// error as the terminal outcome (catalog C23 — a committed, fault-provable
/// `Err`) and checks completion. The first vocab miss carries the sim's
/// exact message (H4/A1).
#[tile]
pub fn finalize_tokenize_prompt(
    resolution: GemmaTokenResolutionState,
    ctx: GemmaTokenIdContext,
) -> Result<PromptTokenization> {
    if let Some(error) = ctx.error {
        return Err(error);
    }
    if let Some(missing_idx) = resolution
        .resolved
        .iter()
        .position(|slot| slot.is_none())
    {
        let piece = &ctx.pieces[missing_idx];
        return Err(format!(
            "Gemma tokenizer piece {piece:?} is missing from vocab"
        ));
    }
    if resolution.resolved.len() as u32 != ctx.piece_count {
        return Err(format!(
            "token-id finalization stopped at piece {}, expected {}",
            resolution.resolved.len(),
            ctx.piece_count
        ));
    }
    let token_ids: Vec<u32> = resolution.resolved.into_iter().flatten().collect();
    let token_count = token_ids.len() as u32;
    Ok(PromptTokenization {
        token_ids,
        token_count,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_token_tests");
        run()
    }

    /// Sorted vocab entries split into `width`-wide chunks (the encoder's
    /// `chunk_table` shape).
    fn chunked_lookup(
        mut entries: Vec<(&str, u32)>,
        width: usize,
    ) -> Vec<Vec<GemmaTokenIdEntry>> {
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut chunks: Vec<Vec<GemmaTokenIdEntry>> = Vec::new();
        for (token, id) in entries {
            let entry = GemmaTokenIdEntry {
                token: token.to_string(),
                id,
            };
            match chunks.last_mut() {
                Some(chunk) if chunk.len() < width => chunk.push(entry),
                _ => chunks.push(vec![entry]),
            }
        }
        chunks
    }

    /// Drives the per-chunk tile over every chunk — the recur sequence's
    /// full single pass (no early break).
    fn run_resolution(
        ctx: &GemmaTokenIdContext,
        chunks: &[Vec<GemmaTokenIdEntry>],
    ) -> GemmaTokenResolutionState {
        let mut state = GemmaTokenResolutionState::initial();
        for chunk in chunks {
            state = resolve_pieces_in_vocab_chunk(state, chunk.clone(), ctx.clone());
        }
        state
    }

    fn ctx_for(pieces: Vec<&str>) -> GemmaTokenIdContext {
        in_scope(|| {
            init_token_id_finalization(
                GemmaBpeLoopState::initial(),
                pieces
                    .into_iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
            )
        })
    }

    #[test]
    fn single_pass_resolves_ids_across_chunks() {
        let ctx = ctx_for(vec!["a", "ab", "b", "c"]);
        let chunks = chunked_lookup(vec![("a", 1), ("ab", 3), ("b", 2), ("c", 7)], 2);
        let state = in_scope(|| run_resolution(&ctx, &chunks));
        assert_eq!(
            state.resolved,
            vec![Some(1), Some(3), Some(2), Some(7)],
            "every piece must resolve regardless of which chunk holds it"
        );
    }

    #[test]
    fn repeated_pieces_each_get_a_slot() {
        let ctx = ctx_for(vec!["a", "b", "a"]);
        let chunks = chunked_lookup(vec![("a", 1), ("b", 2)], 1);
        let state = in_scope(|| run_resolution(&ctx, &chunks));
        assert_eq!(state.resolved, vec![Some(1), Some(2), Some(1)]);
    }

    #[test]
    fn missing_pieces_stay_unresolved() {
        let ctx = ctx_for(vec!["a", "aa", "z"]);
        let chunks = chunked_lookup(vec![("a", 1), ("ab", 3), ("b", 2), ("c", 7)], 2);
        let state = in_scope(|| run_resolution(&ctx, &chunks));
        assert_eq!(state.resolved, vec![Some(1), None, None]);
    }

    #[test]
    fn chunk_width_does_not_change_resolution() {
        let entries = vec![("a", 1), ("ab", 3), ("b", 2), ("c", 7), ("d", 9)];
        let ctx = ctx_for(vec!["ab", "d", "x"]);
        for width in 1..=5 {
            let chunks = chunked_lookup(entries.clone(), width);
            let state = in_scope(|| run_resolution(&ctx, &chunks));
            assert_eq!(
                state.resolved,
                vec![Some(3), Some(9), None],
                "width {width}"
            );
        }
    }

    #[test]
    fn uninitialized_loop_state_falls_back_to_staged_pieces() {
        let ctx = ctx_for(vec!["ab"]);
        assert_eq!(ctx.pieces, vec!["ab".to_string()]);
        assert_eq!(ctx.piece_count, 1);
    }

    #[test]
    fn resolved_pieces_finalize_to_token_ids() {
        let ctx = in_scope(|| {
            init_token_id_finalization(
                GemmaBpeLoopState {
                    initialized: true,
                    complete: true,
                    round: 1,
                    piece_count: 2,
                    pieces: vec!["a".to_string(), "ab".to_string()],
                    error: None,
                },
                vec![],
            )
        });
        let chunks = chunked_lookup(vec![("a", 1), ("ab", 3)], 1);
        let resolution = in_scope(|| run_resolution(&ctx, &chunks));
        let tokenization =
            in_scope(|| finalize_tokenize_prompt(resolution, ctx)).expect("ids");
        assert_eq!(tokenization.token_ids, vec![1, 3]);
        assert_eq!(tokenization.token_count, 2);
    }

    #[test]
    fn missing_vocab_piece_is_a_terminal_error() {
        let ctx = ctx_for(vec!["z"]);
        let chunks = chunked_lookup(vec![("a", 1)], 1);
        let resolution = in_scope(|| run_resolution(&ctx, &chunks));
        let error =
            in_scope(|| finalize_tokenize_prompt(resolution, ctx)).expect_err("missing piece");
        assert_eq!(error, "Gemma tokenizer piece \"z\" is missing from vocab");
    }

    #[test]
    fn the_first_missing_piece_names_the_error() {
        let ctx = ctx_for(vec!["a", "y", "z"]);
        let chunks = chunked_lookup(vec![("a", 1)], 1);
        let resolution = in_scope(|| run_resolution(&ctx, &chunks));
        assert_eq!(resolution.resolved, vec![Some(1), None, None]);
        let error =
            in_scope(|| finalize_tokenize_prompt(resolution, ctx)).expect_err("missing piece");
        assert_eq!(error, "Gemma tokenizer piece \"y\" is missing from vocab");
    }

    #[test]
    fn deferred_loop_errors_surface_first() {
        let ctx = in_scope(|| {
            init_token_id_finalization(
                GemmaBpeLoopState {
                    initialized: true,
                    complete: true,
                    round: 0,
                    piece_count: 1,
                    pieces: vec!["a".to_string()],
                    error: Some("BPE merge apply finalized with 0 pieces, expected 1".to_string()),
                },
                vec![],
            )
        });
        assert!(ctx.error.is_some());
        let resolution = GemmaTokenResolutionState::initial();
        let error =
            in_scope(|| finalize_tokenize_prompt(resolution, ctx)).expect_err("deferred");
        assert_eq!(error, "BPE merge apply finalized with 0 pieces, expected 1");
    }

    #[test]
    fn incomplete_finalization_reports_the_sim_error() {
        // A stalled resolution (fewer slots than pieces, none of them
        // missing) reports the sim's completion-check message.
        let ctx = ctx_for(vec!["a", "b"]);
        let stalled = GemmaTokenResolutionState {
            initialized: true,
            resolved: vec![Some(1)],
        };
        let error = in_scope(|| finalize_tokenize_prompt(stalled, ctx)).expect_err("stalled");
        assert_eq!(
            error,
            "token-id finalization stopped at piece 1, expected 2"
        );
    }

    #[test]
    fn empty_pieces_tokenize_to_nothing() {
        let ctx = ctx_for(vec![]);
        let resolution = in_scope(|| run_resolution(&ctx, &chunked_lookup(vec![("a", 1)], 1)));
        let tokenization =
            in_scope(|| finalize_tokenize_prompt(resolution, ctx)).expect("empty");
        assert!(tokenization.token_ids.is_empty());
        assert_eq!(tokenization.token_count, 0);
    }
}
