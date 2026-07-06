//! The routine sequence (sim `#[sequence] main`, `tiles.rs:20-65`, minus
//! its host-side legs — see the port plan's tile map).
//!
//! Parameterized so program-crate tests can drive it natively through the
//! generated `__raster_sequence_auth_*` wrapper (catalog C25); the zero-arg
//! program `main` binds the committed externals and delegates here.
//!
//! Not ported from the sim: the tokenizer root-equality guard (D1 — the
//! runtime's manifest commitment check enforces source identity), the
//! roots-collecting `finalize_raster_prompt_preparation` (D8 — checkpoint
//! payloads are host-side), and the test-only `tokenize_bpe_state` entry
//! (D5 — this sequence is the natively drivable entry).

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::{GemmaBpeMergeLookupEntry, GemmaTokenIdEntry};

// Glob imports: `call!`/`call_recur!`/`call_recur_seq!` resolve hidden
// per-tile marker types and generated drivers, so the defining modules must
// be in scope.
use crate::bpe_round::*;
use crate::budgets::*;
use crate::token_ids::*;
use crate::types::{BpeConfig, GemmaBpeLoopState, GemmaTokenIdState, PromptTokenization};

/// Staged pieces + tokenizer lookup tables → prompt token ids.
#[sequence]
pub fn tokenize_prompt_pieces(
    initial_pieces: Vec<String>,
    config: BpeConfig,
    token_lookup: Vec<GemmaTokenIdEntry>,
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
) -> Result<PromptTokenization> {
    let budgets = call!(build_chunk_budgets, initial_pieces.clone(), config.clone())?;
    let rounds = select!(Vec<u32>, budgets.clone().rounds);
    let scan_chunks = select!(Vec<u32>, budgets.clone().scan_chunks);
    let apply_chunks = select!(Vec<u32>, budgets.clone().apply_chunks);
    let token_chunks = select!(Vec<u32>, budgets.token_chunks);

    let bpe_state = call_recur_seq!(
        sequence = merge_bpe_round,
        input = rounds,
        state = GemmaBpeLoopState::initial(),
        args = (
            initial_pieces.clone(),
            config.clone(),
            merge_lookup,
            scan_chunks,
            apply_chunks,
        )
    );

    let token_ctx = call!(init_token_id_finalization, bpe_state, initial_pieces);
    let ids_state = call_recur!(
        tile = finalize_next_token_ids,
        input = token_chunks,
        state = GemmaTokenIdState::initial(),
        args = (token_ctx.clone(), token_lookup, config)
    );
    let tokenization = call!(finalize_tokenize_prompt, ids_state, token_ctx)?;
    Ok(tokenization)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;
    use raster::materialize_auth_result;
    use raster_program_gemma_externals::types::GemmaBpeMergeCandidate;

    use super::*;

    /// The sim test fixture (`raster/tests.rs::test_tokenizer_spec`) in the
    /// committed-external shape: sorted token lookup, priority-ordered
    /// merges in a sorted `(left, right)` lookup.
    fn token_lookup() -> Vec<GemmaTokenIdEntry> {
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
        ]
        .into_iter()
        .map(|(token, id)| GemmaTokenIdEntry {
            token: token.to_string(),
            id,
        })
        .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.token.cmp(&right.token));
        entries
    }

    fn merge_lookup() -> Vec<GemmaBpeMergeLookupEntry> {
        let mut entries = vec![("a", "b", 0u32, "ab", 3u32), ("ab", "a", 1, "aba", 12)]
            .into_iter()
            .map(
                |(left, right, merge_index, merged, token_id)| GemmaBpeMergeLookupEntry {
                    left: left.to_string(),
                    right: right.to_string(),
                    candidate: GemmaBpeMergeCandidate {
                        merge_index,
                        merged_token: merged.to_string(),
                        has_token_id: true,
                        token_id,
                    },
                },
            )
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| a.left.cmp(&b.left).then_with(|| a.right.cmp(&b.right)));
        entries
    }

    fn config(pairs: u32, pieces: u32) -> BpeConfig {
        BpeConfig {
            bpe_pairs_per_tile: pairs,
            bpe_pieces_per_tile: pieces,
        }
    }

    fn tokenize(
        pieces: Vec<&str>,
        config: BpeConfig,
    ) -> core::result::Result<PromptTokenization, String> {
        materialize_auth_result::<PromptTokenization, _>(
            __raster_sequence_auth_tokenize_prompt_pieces(
                pieces
                    .into_iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                config,
                token_lookup(),
                merge_lookup(),
            ),
        )
    }

    #[test]
    fn recursive_bpe_merges_apply() {
        // Sim `tokenize_prompt_applies_recursive_bpe_merges`: "ab" → [3].
        let tokenization = tokenize(vec!["a", "b"], config(1, 1)).expect("tokenize");
        assert_eq!(tokenization.token_ids, vec![3]);
        assert_eq!(tokenization.token_count, 1);
    }

    #[test]
    fn chunk_sizes_do_not_change_results() {
        // Sim `tokenize_prompt_chunk_sizes_do_not_change_results`:
        // "aba" → [12] under both tiny and large chunk widths.
        let tiny = tokenize(vec!["a", "b", "a"], config(1, 1)).expect("tiny chunks");
        let large = tokenize(vec!["a", "b", "a"], config(8, 8)).expect("large chunks");
        assert_eq!(tiny.token_ids, large.token_ids);
        assert_eq!(tiny.token_ids, vec![12]);
    }

    #[test]
    fn byte_fallback_pieces_resolve_to_ids() {
        // Sim `tokenize_prompt_uses_byte_fallback_for_unknown_chars`: the
        // host derives "é" into byte-fallback pieces; the program resolves
        // them to [10, 11].
        let tokenization = tokenize(vec!["<0xC3>", "<0xA9>"], config(2, 2)).expect("tokenize");
        assert_eq!(tokenization.token_ids, vec![10, 11]);
    }

    #[test]
    fn single_piece_needs_no_merge_rounds() {
        let tokenization = tokenize(vec!["ab"], config(64, 64)).expect("tokenize");
        assert_eq!(tokenization.token_ids, vec![3]);
    }

    #[test]
    fn empty_pieces_tokenize_to_nothing() {
        let tokenization = tokenize(vec![], config(64, 64)).expect("tokenize");
        assert!(tokenization.token_ids.is_empty());
        assert_eq!(tokenization.token_count, 0);
    }

    #[test]
    fn missing_vocab_piece_is_a_committed_terminal_error() {
        let error = tokenize(vec!["z"], config(64, 64)).expect_err("missing piece");
        assert_eq!(error, "Gemma tokenizer piece \"z\" is missing from vocab");
    }

    #[test]
    fn zero_chunk_width_is_a_committed_terminal_error() {
        let error = tokenize(vec!["a", "b"], config(0, 1)).expect_err("zero width");
        assert_eq!(
            error,
            "raster tokenizer BPE pairs per tile must be greater than zero"
        );
    }
}
