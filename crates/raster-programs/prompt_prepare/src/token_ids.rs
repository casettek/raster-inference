//! Token-id finalization (sim `tiles.rs`: `finalize_bpe_tokenize_prompt` +
//! `init_token_id_finalization`, `finalize_next_token_ids`,
//! `finalize_tokenize_prompt`).
//!
//! The sim's `auth_read(tokenizer, GemmaTokenIdRequest)` becomes an in-tile
//! binary search over the tokenizer external's `token_lookup` (sorted by
//! `token`) — port-plan deviation D7. Ids accumulate in the loop-carried
//! state (the sim appended to the `prompt-token-ids` builder — D6); the
//! trivial `finalize_bpe_tokenize_prompt` cast is folded into the init tile
//! (D6a).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaTokenIdEntry;

use crate::types::{
    BpeConfig, GemmaBpeLoopState, GemmaTokenIdContext, GemmaTokenIdState, PromptTokenization,
};

/// Binary search of the sorted token table. Plain helper — only reachable
/// from tile bodies.
pub(crate) fn find_token_id(token_lookup: &[GemmaTokenIdEntry], token: &str) -> Option<u32> {
    token_lookup
        .binary_search_by(|entry| entry.token.as_str().cmp(token))
        .ok()
        .map(|idx| token_lookup[idx].id)
}

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

/// Chunked vocab lookup (sim `tiles.rs:149-194`): each iteration resolves
/// up to `bpe_pieces_per_tile` pieces to token ids. A missing piece defers
/// the sim's vocab error through the state (port-plan constraint A1) and
/// breaks.
#[tile(kind = recur)]
pub fn finalize_next_token_ids(
    input: RecurInput<u32>,
    state: RecurState<GemmaTokenIdState>,
    ctx: GemmaTokenIdContext,
    token_lookup: Vec<GemmaTokenIdEntry>,
    config: BpeConfig,
) -> RecurControl<RecurState<GemmaTokenIdState>> {
    let _chunk_ordinal = input.value();
    let mut state = state;
    if ctx.error.is_some() || state.error.is_some() {
        return RecurControl::Break(state);
    }
    if state.next_piece_idx >= ctx.piece_count {
        return RecurControl::Break(state);
    }

    let end_piece_idx = state
        .next_piece_idx
        .saturating_add(config.bpe_pieces_per_tile)
        .min(ctx.piece_count);
    for piece_idx in state.next_piece_idx..end_piece_idx {
        let piece = &ctx.pieces[piece_idx as usize];
        let Some(token_id) = find_token_id(&token_lookup, piece) else {
            state.error = Some(format!(
                "Gemma tokenizer piece {piece:?} is missing from vocab"
            ));
            state.next_piece_idx = piece_idx;
            return RecurControl::Break(state);
        };
        state.token_ids.push(token_id);
    }
    state.next_piece_idx = end_piece_idx;

    if state.next_piece_idx >= ctx.piece_count {
        RecurControl::Break(state)
    } else {
        RecurControl::Continue(state)
    }
}

/// Closes tokenization (sim `tiles.rs:196-223`): surfaces every deferred
/// error as the terminal outcome (catalog C23 — a committed, fault-provable
/// `Err`) and checks completion.
#[tile]
pub fn finalize_tokenize_prompt(
    state: GemmaTokenIdState,
    ctx: GemmaTokenIdContext,
) -> Result<PromptTokenization> {
    if let Some(error) = ctx.error {
        return Err(error);
    }
    if let Some(error) = state.error {
        return Err(error);
    }
    if state.next_piece_idx != ctx.piece_count {
        return Err(format!(
            "token-id finalization stopped at piece {}, expected {}",
            state.next_piece_idx, ctx.piece_count
        ));
    }
    let token_count = state.token_ids.len() as u32;
    Ok(PromptTokenization {
        token_ids: state.token_ids,
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

    fn sorted_lookup(mut entries: Vec<(&str, u32)>) -> Vec<GemmaTokenIdEntry> {
        entries.sort_by(|a, b| a.0.cmp(b.0));
        entries
            .into_iter()
            .map(|(token, id)| GemmaTokenIdEntry {
                token: token.to_string(),
                id,
            })
            .collect()
    }

    fn config(pieces_per_tile: u32) -> BpeConfig {
        BpeConfig {
            bpe_pairs_per_tile: 64,
            bpe_pieces_per_tile: pieces_per_tile,
        }
    }

    fn run_lookup(
        ctx: &GemmaTokenIdContext,
        token_lookup: &[GemmaTokenIdEntry],
        config: &BpeConfig,
    ) -> GemmaTokenIdState {
        let mut state = GemmaTokenIdState::initial();
        for chunk in 0u32..16 {
            let control = finalize_next_token_ids(
                RecurInput::new(chunk, chunk as u64, 16),
                RecurState::new(state),
                ctx.clone(),
                token_lookup.to_vec(),
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
    fn token_ids_resolve_in_chunks() {
        let ctx = in_scope(|| {
            init_token_id_finalization(
                GemmaBpeLoopState {
                    initialized: true,
                    complete: true,
                    round: 1,
                    pieces: vec!["a".to_string(), "ab".to_string()],
                    error: None,
                },
                vec![],
            )
        });
        let lookup = sorted_lookup(vec![("a", 1), ("ab", 3)]);
        let state = in_scope(|| run_lookup(&ctx, &lookup, &config(1)));
        let tokenization = in_scope(|| finalize_tokenize_prompt(state, ctx)).expect("ids");
        assert_eq!(tokenization.token_ids, vec![1, 3]);
        assert_eq!(tokenization.token_count, 2);
    }

    #[test]
    fn uninitialized_loop_state_falls_back_to_staged_pieces() {
        let ctx = in_scope(|| {
            init_token_id_finalization(GemmaBpeLoopState::initial(), vec!["ab".to_string()])
        });
        assert_eq!(ctx.pieces, vec!["ab".to_string()]);
        assert_eq!(ctx.piece_count, 1);
    }

    #[test]
    fn missing_vocab_piece_is_a_terminal_error() {
        let ctx = in_scope(|| {
            init_token_id_finalization(GemmaBpeLoopState::initial(), vec!["z".to_string()])
        });
        let lookup = sorted_lookup(vec![("a", 1)]);
        let state = in_scope(|| run_lookup(&ctx, &lookup, &config(8)));
        let error = in_scope(|| finalize_tokenize_prompt(state, ctx)).expect_err("missing piece");
        assert_eq!(error, "Gemma tokenizer piece \"z\" is missing from vocab");
    }

    #[test]
    fn deferred_loop_errors_surface_first() {
        let ctx = in_scope(|| {
            init_token_id_finalization(
                GemmaBpeLoopState {
                    initialized: true,
                    complete: true,
                    round: 0,
                    pieces: vec!["a".to_string()],
                    error: Some("BPE merge scan finalized at pair 0, expected 1".to_string()),
                },
                vec![],
            )
        });
        let state = in_scope(|| run_lookup(&ctx, &sorted_lookup(vec![("a", 1)]), &config(8)));
        assert!(state.token_ids.is_empty(), "errored context must not scan");
        let error = in_scope(|| finalize_tokenize_prompt(state, ctx)).expect_err("deferred");
        assert_eq!(error, "BPE merge scan finalized at pair 0, expected 1");
    }

    #[test]
    fn incomplete_finalization_reports_the_sim_error() {
        let ctx = in_scope(|| {
            init_token_id_finalization(
                GemmaBpeLoopState::initial(),
                vec!["a".to_string(), "b".to_string()],
            )
        });
        let stalled = GemmaTokenIdState {
            token_ids: vec![1],
            next_piece_idx: 1,
            error: None,
        };
        let error = in_scope(|| finalize_tokenize_prompt(stalled, ctx)).expect_err("stalled");
        assert_eq!(
            error,
            "token-id finalization stopped at piece 1, expected 2"
        );
    }

    #[test]
    fn empty_pieces_tokenize_to_nothing() {
        let ctx = in_scope(|| init_token_id_finalization(GemmaBpeLoopState::initial(), vec![]));
        let state = in_scope(|| run_lookup(&ctx, &sorted_lookup(vec![]), &config(8)));
        let tokenization = in_scope(|| finalize_tokenize_prompt(state, ctx)).expect("empty");
        assert!(tokenization.token_ids.is_empty());
        assert_eq!(tokenization.token_count, 0);
    }
}
