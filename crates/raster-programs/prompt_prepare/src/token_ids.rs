//! Token-id finalization (sim `tiles.rs`: `finalize_bpe_tokenize_prompt` +
//! `init_token_id_finalization`, `finalize_next_token_ids`,
//! `finalize_tokenize_prompt`).
//!
//! Storage-read shape (deviation D16, supersedes D14's threaded slot vector
//! while keeping its single-pass orientation): one recur sequence over
//! token-lookup chunk ordinals, append-only — no threaded resolution state.
//! Per chunk, one plain tile reads the selected vocab chunk and final pieces
//! from raster storage, then appends a `TokenIdMatch` for each hit into the
//! `RecurOutput<TokenIdMatches>` draft. Chunks that resolve nothing push
//! nothing (no `Break` — gap G1).
//!
//! `init_token_id_finalization` is the first fallible plain tile after the
//! BPE loop: it surfaces the deferred loop error (A1) and returns the final
//! pieces ref/count. The terminal `finalize_tokenize_prompt` materializes
//! the matches and the pieces once, at the program boundary (invariant rule
//! 6), orders by `piece_idx`, and surfaces the first vocab miss with the
//! sim's exact message (H4/A1).

use alloc::format;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaTokenIdEntry;

use crate::types::{
    field_index_selector, read_tokenizer_selection, resolve_bpe_pieces_ref, FinalPieces,
    GemmaBpeLoopState, PromptTokenization, TokenIdMatch, TokenIdMatches, TokenIdMatchesDraftExt,
    TokenizerTables,
};

/// Opens token-id finalization from the finished BPE loop state: surfaces
/// the deferred loop error as the terminal `Err` (A1 — the first fallible
/// plain tile after the loop) and republishes the final pieces from the
/// loop state's storage handle behind the selectable root.
#[tile]
pub fn init_token_id_finalization(loop_state: GemmaBpeLoopState) -> Result<FinalPieces> {
    if let Some(error) = loop_state.error {
        return Err(error);
    }
    Ok(FinalPieces {
        pieces_ref: loop_state.pieces_ref,
        piece_count: loop_state.piece_count,
    })
}

/// One sorted vocab chunk per execution: binary-searches every final piece
/// against the chunk and appends a `TokenIdMatch` per hit. Append-only —
/// no threaded state, a full pass over every chunk (inefficiency accepted;
/// invariant rule 7). Infallible per port-plan constraint A1: misses stay
/// unmatched, surfaced by `finalize_tokenize_prompt`.
#[tile]
pub fn resolve_pieces_in_vocab_chunk(
    chunk_idx: u32,
    final_pieces: FinalPieces,
    tokenizer: TokenizerTables,
    output: Draft<TokenIdMatches>,
) -> Draft<TokenIdMatches> {
    let mut output = output;
    let chunk = read_tokenizer_selection::<Vec<GemmaTokenIdEntry>>(
        &tokenizer,
        field_index_selector("token_lookup_chunks", chunk_idx),
        "token lookup chunk",
    );
    let final_pieces = resolve_bpe_pieces_ref(final_pieces.pieces_ref, "vocab chunk resolution");
    for (piece_idx, piece) in final_pieces.pieces.iter().enumerate() {
        if let Ok(idx) = chunk.binary_search_by(|entry| entry.token.as_str().cmp(piece.as_str())) {
            output.matches().push(TokenIdMatch {
                piece_idx: piece_idx as u32,
                token_id: chunk[idx].id,
            });
        }
    }
    output
}

/// The inverted vocab pass: an output-only recur sequence over the chunked
/// vocab table. The chunk reaches `resolve_pieces_in_vocab_chunk` as an
/// external-selection binding, the final pieces as an internal-selection
/// binding; only the draft replay handle rides the iteration record.
#[sequence(kind = recur)]
pub fn resolve_vocab_chunks(
    input: RecurSequenceInput<u32>,
    output: RecurSequenceOutput<TokenIdMatches>,
    final_pieces: FinalPieces,
    tokenizer: TokenizerTables,
) -> RecurSequenceOutput<TokenIdMatches> {
    call!(
        resolve_pieces_in_vocab_chunk,
        input,
        final_pieces,
        tokenizer,
        output
    )
}

/// Closes tokenization (sim `tiles.rs:196-223`): materializes the finalized
/// matches and the final pieces at the program boundary (rule 6 — the only
/// place ordered token ids materialize for the host), orders by
/// `piece_idx`, errors on conflicting duplicate matches (defensive — the
/// sorted vocab resolves each piece in exactly one chunk; deterministic
/// message per H4) and on the first unresolved piece with the sim's exact
/// missing-vocab message (H4/A1).
#[tile]
pub fn finalize_tokenize_prompt(
    matches: TokenIdMatches,
    final_pieces: FinalPieces,
) -> Result<PromptTokenization> {
    let final_pieces = resolve_bpe_pieces_ref(final_pieces.pieces_ref, "tokenize finalization");
    let piece_count = final_pieces.pieces.len() as u32;
    let mut resolved: Vec<Option<u32>> = alloc::vec![None; final_pieces.pieces.len()];
    for token_match in matches.matches {
        if token_match.piece_idx >= piece_count {
            return Err(format!(
                "token-id finalization matched out-of-range piece {} of {}",
                token_match.piece_idx, piece_count
            ));
        }
        let slot = &mut resolved[token_match.piece_idx as usize];
        if slot.is_some() {
            return Err(format!(
                "token-id finalization found conflicting ids for piece {}",
                token_match.piece_idx
            ));
        }
        *slot = Some(token_match.token_id);
    }
    if let Some(missing_idx) = resolved.iter().position(|slot| slot.is_none()) {
        let piece = &final_pieces.pieces[missing_idx];
        return Err(format!(
            "Gemma tokenizer piece {piece:?} is missing from vocab"
        ));
    }
    let token_ids: Vec<u32> = resolved.into_iter().flatten().collect();
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
    use alloc::vec::Vec;
    use raster_program_gemma_externals::types::{
        GemmaDecoderMetadata, GemmaTokenizer, GemmaTokenizerMetadata,
    };

    use super::*;
    use crate::types::BpePieces;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_token_tests");
        run()
    }

    /// Sorted vocab entries split into `width`-wide chunks (the encoder's
    /// `chunk_table` shape).
    fn chunked_lookup(mut entries: Vec<(&str, u32)>, width: usize) -> Vec<Vec<GemmaTokenIdEntry>> {
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

    fn pieces(items: Vec<&str>) -> BpePieces {
        BpePieces {
            pieces: items.into_iter().map(ToString::to_string).collect(),
        }
    }

    fn loop_state_for(pieces: BpePieces) -> GemmaBpeLoopState {
        let pieces_ref = raster::store_internal_value(&pieces).expect("store BPE pieces");
        let mut state = GemmaBpeLoopState::initial(pieces_ref);
        state.piece_count = pieces.pieces.len() as u32;
        state
    }

    fn final_handle(pieces: &BpePieces) -> FinalPieces {
        let pieces_ref = raster::store_internal_value(pieces).expect("store final pieces");
        FinalPieces {
            pieces_ref,
            piece_count: pieces.pieces.len() as u32,
        }
    }

    fn resolved_final_pieces(final_pieces: &FinalPieces) -> BpePieces {
        raster::resolve_internal_value::<BpePieces>(final_pieces.pieces_ref.clone())
            .expect("resolve final pieces")
            .value
    }

    fn tokenizer_with_vocab_chunks(chunks: Vec<Vec<GemmaTokenIdEntry>>) -> TokenizerTables {
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
            token_lookup_chunks: chunks,
            tokens_by_id: Vec::new(),
            special_tokens: Vec::new(),
            merge_chunks: Vec::new(),
        };
        TokenizerTables::internal(
            raster::store_internal_value(&tokenizer).expect("store tokenizer"),
        )
    }

    fn matched(entries: Vec<(u32, u32)>) -> TokenIdMatches {
        TokenIdMatches {
            matches: entries
                .into_iter()
                .map(|(piece_idx, token_id)| TokenIdMatch {
                    piece_idx,
                    token_id,
                })
                .collect(),
        }
    }

    /// Drives the per-chunk tile over every chunk — the recur sequence's
    /// full single pass — accumulating appends in a real draft and
    /// finalizing it (the driver-level mechanics are probe P2 and
    /// routine-level evidence).
    fn run_resolution(
        final_pieces: &BpePieces,
        chunks: &[Vec<GemmaTokenIdEntry>],
    ) -> TokenIdMatches {
        in_scope(|| {
            let final_pieces = final_handle(final_pieces);
            let tokenizer = tokenizer_with_vocab_chunks(chunks.to_vec());
            let mut draft = raster::new_draft::<TokenIdMatches>();
            for chunk_idx in 0..chunks.len() as u32 {
                draft = resolve_pieces_in_vocab_chunk(
                    chunk_idx,
                    final_pieces.clone(),
                    tokenizer.clone(),
                    draft,
                );
            }
            raster::materialize_auth_return::<TokenIdMatches, _>(raster::finalize(draft))
        })
    }

    #[test]
    fn single_pass_resolves_ids_across_chunks() {
        let final_pieces = pieces(vec!["a", "ab", "b", "c"]);
        let chunks = chunked_lookup(vec![("a", 1), ("ab", 3), ("b", 2), ("c", 7)], 2);
        let matches = run_resolution(&final_pieces, &chunks);
        let tokenization =
            in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
                .expect("ids");
        assert_eq!(
            tokenization.token_ids,
            vec![1, 3, 2, 7],
            "every piece must resolve regardless of which chunk holds it"
        );
    }

    #[test]
    fn repeated_pieces_each_get_a_match() {
        let final_pieces = pieces(vec!["a", "b", "a"]);
        let chunks = chunked_lookup(vec![("a", 1), ("b", 2)], 1);
        let matches = run_resolution(&final_pieces, &chunks);
        let tokenization =
            in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
                .expect("ids");
        assert_eq!(tokenization.token_ids, vec![1, 2, 1]);
    }

    #[test]
    fn matches_append_in_chunk_order_and_reorder_by_piece_idx() {
        // "d" (piece 1) lives in a later chunk than "a" (piece 0) but the
        // finalizer orders by piece_idx, not append order.
        let final_pieces = pieces(vec!["d", "a"]);
        let chunks = chunked_lookup(vec![("a", 1), ("d", 9)], 1);
        let matches = run_resolution(&final_pieces, &chunks);
        assert_eq!(
            matches.matches,
            vec![
                TokenIdMatch {
                    piece_idx: 1,
                    token_id: 1,
                },
                TokenIdMatch {
                    piece_idx: 0,
                    token_id: 9,
                },
            ],
            "appends follow chunk order"
        );
        let tokenization =
            in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
                .expect("ids");
        assert_eq!(tokenization.token_ids, vec![9, 1]);
    }

    #[test]
    fn chunk_width_does_not_change_resolution() {
        let entries = vec![("a", 1), ("ab", 3), ("b", 2), ("c", 7), ("d", 9)];
        let final_pieces = pieces(vec!["ab", "d"]);
        for width in 1..=5 {
            let chunks = chunked_lookup(entries.clone(), width);
            let matches = run_resolution(&final_pieces, &chunks);
            let tokenization =
                in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
                    .expect("ids");
            assert_eq!(tokenization.token_ids, vec![3, 9], "width {width}");
        }
    }

    #[test]
    fn zero_round_loop_state_supplies_initial_pieces_by_handle() {
        let final_pieces =
            in_scope(|| init_token_id_finalization(loop_state_for(pieces(vec!["ab"]))))
                .expect("fallback");
        assert_eq!(
            resolved_final_pieces(&final_pieces).pieces,
            vec!["ab".to_string()]
        );
    }

    #[test]
    fn loop_state_supplies_the_final_pieces_by_handle() {
        let final_pieces = in_scope(|| {
            let mut state = loop_state_for(pieces(vec!["a", "ab"]));
            state.complete = true;
            state.round = 1;
            init_token_id_finalization(state)
        })
        .expect("final pieces");
        assert_eq!(
            resolved_final_pieces(&final_pieces).pieces,
            vec!["a".to_string(), "ab".to_string()]
        );
    }

    #[test]
    fn deferred_loop_errors_surface_at_the_first_fallible_tile() {
        let error = in_scope(|| {
            let mut state = loop_state_for(pieces(vec!["a"]));
            state.complete = true;
            state.error = Some("BPE merge apply finalized with 0 pieces, expected 1".to_string());
            init_token_id_finalization(state)
        })
        .expect_err("deferred");
        assert_eq!(error, "BPE merge apply finalized with 0 pieces, expected 1");
    }

    #[test]
    fn missing_vocab_piece_is_a_terminal_error() {
        let final_pieces = pieces(vec!["z"]);
        let chunks = chunked_lookup(vec![("a", 1)], 1);
        let matches = run_resolution(&final_pieces, &chunks);
        let error = in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
            .expect_err("missing piece");
        assert_eq!(error, "Gemma tokenizer piece \"z\" is missing from vocab");
    }

    #[test]
    fn the_first_missing_piece_names_the_error() {
        let final_pieces = pieces(vec!["a", "y", "z"]);
        let chunks = chunked_lookup(vec![("a", 1)], 1);
        let matches = run_resolution(&final_pieces, &chunks);
        assert_eq!(
            matches.matches,
            vec![TokenIdMatch {
                piece_idx: 0,
                token_id: 1,
            }]
        );
        let error = in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
            .expect_err("missing piece");
        assert_eq!(error, "Gemma tokenizer piece \"y\" is missing from vocab");
    }

    #[test]
    fn conflicting_duplicate_matches_are_a_terminal_error() {
        let final_pieces = pieces(vec!["a"]);
        let matches = matched(vec![(0, 1), (0, 2)]);
        let error = in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
            .expect_err("duplicate");
        assert_eq!(
            error,
            "token-id finalization found conflicting ids for piece 0"
        );
    }

    #[test]
    fn out_of_range_matches_are_a_terminal_error() {
        let final_pieces = pieces(vec!["a"]);
        let matches = matched(vec![(0, 1), (3, 2)]);
        let error = in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
            .expect_err("out of range");
        assert_eq!(
            error,
            "token-id finalization matched out-of-range piece 3 of 1"
        );
    }

    #[test]
    fn empty_pieces_tokenize_to_nothing() {
        let final_pieces = pieces(vec![]);
        let matches = run_resolution(&final_pieces, &chunked_lookup(vec![("a", 1)], 1));
        let tokenization =
            in_scope(|| finalize_tokenize_prompt(matches, final_handle(&final_pieces)))
                .expect("empty");
        assert!(tokenization.token_ids.is_empty());
        assert_eq!(tokenization.token_count, 0);
    }
}
