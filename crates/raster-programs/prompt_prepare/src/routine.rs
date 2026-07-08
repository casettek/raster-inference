//! The routine sequence (sim `#[sequence] main`, `tiles.rs:20-65`, minus
//! its host-side legs — see the port plan's tile map).
//!
//! Parameterized so program-crate tests can drive it natively through the
//! generated `__raster_sequence_auth_*` wrapper (catalog C25); the zero-arg
//! program `main` binds the committed externals and delegates here.
//!
//! Data placement (trace-slimming refactor, D13/D14): the chunked
//! model-scoped tables (`token_lookup_chunks`, `merge_chunks`) arrive as
//! `AuthRef`s and flow untouched into recur-*sequence* input positions —
//! each chunk crosses the tile ABI as a selection binding and materializes
//! only at tile execution. Native tests stage them as internal values
//! (recur input lists need a selectable list source, not an inline value).
//!
//! Not ported from the sim: the tokenizer root-equality guard (D1 — the
//! runtime's manifest commitment check enforces source identity), the
//! roots-collecting `finalize_raster_prompt_preparation` (D8 — checkpoint
//! payloads are host-side), and the test-only `tokenize_bpe_state` entry
//! (D5 — this sequence is the natively drivable entry).

use alloc::vec::Vec;
use raster::prelude::*;

// Glob imports: `call!`/`call_recur!`/`call_recur_seq!` resolve hidden
// per-tile marker types and generated drivers, so the defining modules must
// be in scope.
use crate::bpe_round::*;
use crate::budgets::*;
use crate::token_ids::*;
use crate::types::{
    BpePieces, GemmaBpeLoopState, PromptTokenization, TokenIdMatches, TokenizerTables,
};

/// Staged pieces + chunked tokenizer tables → prompt token ids.
///
/// The staged pieces arrive as the selectable `BpePieces` root and stay an
/// `AuthRef` throughout: the count is a one-shot authenticated read
/// (`count_pieces`), and the loops receive the pieces as selection-bound
/// args materialized only at tile execution.
#[sequence]
pub fn tokenize_prompt_pieces(
    initial_pieces: BpePieces,
    tokenizer: TokenizerTables,
) -> Result<PromptTokenization> {
    let current_pieces = call!(publish_initial_pieces, initial_pieces);
    let count = call!(count_pieces, current_pieces.reference().clone());
    let piece_count = select!(u32, count.piece_count);
    let budgets = call!(build_chunk_budgets, piece_count, tokenizer.clone());
    let rounds = select!(Vec<u32>, budgets.clone().rounds);
    let merge_chunk_ordinals = select!(Vec<u32>, budgets.clone().merge_chunk_ordinals);
    let token_lookup_chunk_ordinals = select!(Vec<u32>, budgets.token_lookup_chunk_ordinals);

    let bpe_state = call_recur_seq!(
        sequence = merge_bpe_round,
        input = rounds,
        state = GemmaBpeLoopState::initial(current_pieces.reference().clone()),
        args = (tokenizer.clone(), merge_chunk_ordinals)
    );

    let final_pieces = call!(init_token_id_finalization, bpe_state)?;
    let matches = call_recur_seq!(
        sequence = resolve_vocab_chunks,
        input = token_lookup_chunk_ordinals,
        output = new!(TokenIdMatches),
        args = (final_pieces.clone(), tokenizer)
    );
    let tokenization = call!(finalize_tokenize_prompt, matches, final_pieces)?;
    Ok(tokenization)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;
    use raster::materialize_auth_result;
    use raster_program_gemma_externals::types::{
        GemmaBpeMerge, GemmaDecoderMetadata, GemmaTokenIdEntry, GemmaTokenizer,
        GemmaTokenizerMetadata,
    };

    use super::*;

    /// The sim test fixture (`raster/tests.rs::test_tokenizer_spec`) in the
    /// chunked committed-external shape: sorted vocab chunks,
    /// priority-ordered merge chunks.
    fn token_lookup_chunks(width: usize) -> Vec<Vec<GemmaTokenIdEntry>> {
        let mut entries = vec![
            ("<unk>", 0u32),
            ("a", 1),
            ("b", 2),
            ("ab", 3),
            ("aba", 12),
            ("\u{2581}", 4),
            ("<bos>", 5),
            ("<0xC3>", 10),
            ("<0xA9>", 11),
        ];
        entries.sort_by(|left, right| left.0.cmp(right.0));
        chunk(
            entries
                .into_iter()
                .map(|(token, id)| GemmaTokenIdEntry {
                    token: token.to_string(),
                    id,
                })
                .collect(),
            width,
        )
    }

    fn merge_chunks(width: usize) -> Vec<Vec<GemmaBpeMerge>> {
        let rules = vec![("a", "b", 0u32, "ab", 3u32), ("ab", "a", 1, "aba", 12)]
            .into_iter()
            .map(
                |(left, right, merge_index, merged, token_id)| GemmaBpeMerge {
                    merge_index,
                    left: left.to_string(),
                    right: right.to_string(),
                    merged_token: merged.to_string(),
                    has_token_id: true,
                    token_id,
                },
            )
            .collect::<Vec<_>>();
        chunk(rules, width)
    }

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

    fn tokenizer(width: usize) -> GemmaTokenizer {
        GemmaTokenizer {
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
            token_lookup_chunks: token_lookup_chunks(width),
            tokens_by_id: Vec::new(),
            special_tokens: Vec::new(),
            merge_chunks: merge_chunks(width),
        }
    }

    /// Drives the routine natively. The staged pieces and chunked tables
    /// are stored as internal values first: recur input lists and `select!`
    /// roots must be selectable external/internal sources, exactly like the
    /// bindings `main` makes from the committed externals.
    fn tokenize_chunked(
        pieces: Vec<&str>,
        table_width: usize,
    ) -> core::result::Result<PromptTokenization, String> {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_routine_tests");
        let staged_pieces = raster::store_internal_value(&BpePieces {
            pieces: pieces
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        })
        .expect("store staged pieces");
        let tokenizer =
            raster::store_internal_value(&tokenizer(table_width)).expect("store tokenizer");
        materialize_auth_result::<PromptTokenization, _>(
            __raster_sequence_auth_tokenize_prompt_pieces(
                internal!(BpePieces, staged_pieces),
                TokenizerTables::internal(tokenizer),
            ),
        )
    }

    fn tokenize(pieces: Vec<&str>) -> core::result::Result<PromptTokenization, String> {
        tokenize_chunked(pieces, 4)
    }

    #[test]
    fn recursive_bpe_merges_apply() {
        // Sim `tokenize_prompt_applies_recursive_bpe_merges`: "ab" → [3].
        let tokenization = tokenize(vec!["a", "b"]).expect("tokenize");
        assert_eq!(tokenization.token_ids, vec![3]);
        assert_eq!(tokenization.token_count, 1);
    }

    #[test]
    fn chunk_sizes_do_not_change_results() {
        // Sim `tokenize_prompt_chunk_sizes_do_not_change_results`:
        // "aba" → [12] under both tiny and large encode-time table chunk
        // widths (the only remaining width — per-tile apply widths died
        // with the storage-resident refactor).
        let tiny = tokenize_chunked(vec!["a", "b", "a"], 1).expect("tiny chunks");
        let large = tokenize_chunked(vec!["a", "b", "a"], 16).expect("large chunks");
        assert_eq!(tiny.token_ids, large.token_ids);
        assert_eq!(tiny.token_ids, vec![12]);
    }

    #[test]
    fn byte_fallback_pieces_resolve_to_ids() {
        // Sim `tokenize_prompt_uses_byte_fallback_for_unknown_chars`: the
        // host derives "é" into byte-fallback pieces; the program resolves
        // them to [10, 11].
        let tokenization = tokenize(vec!["<0xC3>", "<0xA9>"]).expect("tokenize");
        assert_eq!(tokenization.token_ids, vec![10, 11]);
    }

    #[test]
    fn single_piece_needs_no_merge_rounds() {
        let tokenization = tokenize(vec!["ab"]).expect("tokenize");
        assert_eq!(tokenization.token_ids, vec![3]);
    }

    #[test]
    fn empty_pieces_tokenize_to_nothing() {
        let tokenization = tokenize(vec![]).expect("tokenize");
        assert!(tokenization.token_ids.is_empty());
        assert_eq!(tokenization.token_count, 0);
    }

    #[test]
    fn missing_vocab_piece_is_a_committed_terminal_error() {
        let error = tokenize(vec!["z"]).expect_err("missing piece");
        assert_eq!(error, "Gemma tokenizer piece \"z\" is missing from vocab");
    }
}
