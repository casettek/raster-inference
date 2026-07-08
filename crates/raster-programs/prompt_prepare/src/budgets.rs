//! Chunk-budget derivation (port-plan deviation D9, gap G1) and the
//! one-shot staged-pieces count.
//!
//! Real recur loops are list-driven; this tile derives the bounded round
//! list from the staged pieces count. The sim's zero-width guards died
//! with the per-tile chunk widths (storage-resident refactor: the apply
//! loop recurs over the round's own pieces, so no width configuration
//! remains).

use alloc::vec::Vec;
use raster::prelude::*;
use raster::InternalRef;

use crate::types::{
    field_selector, resolve_bpe_pieces_ref, BpePieces, ChunkBudgets, GemmaBpeOpenedRound,
    PieceCount, PieceOrdinals, TokenizerTables,
};
use raster_program_gemma_externals::types::{GemmaBpeMerge, GemmaTokenIdEntry};

/// One-shot authenticated read of the staged `BpePieces` external: derives
/// the piece count in-program instead of staging a separate count (an
/// unchecked staged count is an integrity hole; a checked one is
/// redundant).
#[tile]
pub fn count_pieces(pieces_ref: InternalRef) -> PieceCount {
    let pieces = resolve_bpe_pieces_ref(pieces_ref, "piece count");
    PieceCount {
        piece_count: pieces.pieces.len() as u32,
    }
}

/// Publishes the staged pieces as a tile output so the BPE loop can thread
/// only the resulting internal-storage reference.
#[tile]
pub fn publish_initial_pieces(pieces: BpePieces) -> BpePieces {
    pieces
}

/// Bounds: rounds = `piece_count − 1` (each merge removes one piece).
/// Shorter real iterations no-op after convergence (recur sequences cannot
/// break early — gap G1); an empty list finalizes cleanly with the initial
/// state (WS1 probe P1). The scan, apply, and token-id phases need no
/// derived budgets (storage-resident refactor): the chunked model tables
/// and the round's pieces list are their own bounded recur inputs.
#[tile]
pub fn build_chunk_budgets(piece_count: u32, tokenizer: TokenizerTables) -> ChunkBudgets {
    let merge_chunks = crate::types::read_tokenizer_selection::<Vec<Vec<GemmaBpeMerge>>>(
        &tokenizer,
        field_selector("merge_chunks"),
        "merge_chunks length",
    );
    let token_lookup_chunks = crate::types::read_tokenizer_selection::<Vec<Vec<GemmaTokenIdEntry>>>(
        &tokenizer,
        field_selector("token_lookup_chunks"),
        "token_lookup_chunks length",
    );
    ChunkBudgets {
        rounds: ordinals(piece_count.saturating_sub(1)),
        apply_piece_ordinals: ordinals(piece_count),
        merge_chunk_ordinals: ordinals(merge_chunks.len() as u32),
        token_lookup_chunk_ordinals: ordinals(token_lookup_chunks.len() as u32),
    }
}

pub(crate) fn ordinals(count: u32) -> Vec<u32> {
    (0..count).collect()
}

#[tile]
pub fn build_piece_ordinals(piece_count: u32) -> PieceOrdinals {
    PieceOrdinals {
        ordinals: ordinals(piece_count),
    }
}

#[tile]
pub fn build_round_piece_ordinals(opened: GemmaBpeOpenedRound) -> PieceOrdinals {
    PieceOrdinals {
        ordinals: ordinals(opened.piece_count),
    }
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

    /// Tile fns require an active sequence scope even when driven natively.
    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_budget_tests");
        run()
    }

    fn tokenizer(merge_chunks: usize, vocab_chunks: usize) -> TokenizerTables {
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
            token_lookup_chunks: vec![Vec::new(); vocab_chunks],
            tokens_by_id: Vec::new(),
            special_tokens: Vec::new(),
            merge_chunks: vec![Vec::new(); merge_chunks],
        };
        TokenizerTables::internal(
            raster::store_internal_value(&tokenizer).expect("store tokenizer"),
        )
    }

    #[test]
    fn count_reads_the_staged_pieces() {
        let count = in_scope(|| {
            let pieces = BpePieces {
                pieces: vec!["a".to_string(), "b".to_string()],
            };
            let reference = raster::store_internal_value(&pieces).expect("store pieces");
            count_pieces(reference)
        });
        assert_eq!(count.piece_count, 2);
    }

    #[test]
    fn publish_initial_pieces_returns_the_staged_root_value() {
        let pieces = in_scope(|| {
            publish_initial_pieces(BpePieces {
                pieces: vec!["a".to_string(), "b".to_string()],
            })
        });
        assert_eq!(pieces.pieces, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn rounds_cover_one_merge_per_removed_piece() {
        let budgets = in_scope(|| build_chunk_budgets(5, tokenizer(2, 3)));
        assert_eq!(budgets.rounds, vec![0, 1, 2, 3]);
        assert_eq!(budgets.apply_piece_ordinals, vec![0, 1, 2, 3, 4]);
        assert_eq!(budgets.merge_chunk_ordinals, vec![0, 1]);
        assert_eq!(budgets.token_lookup_chunk_ordinals, vec![0, 1, 2]);
    }

    #[test]
    fn single_piece_needs_no_rounds() {
        let budgets = in_scope(|| build_chunk_budgets(1, tokenizer(0, 0)));
        assert!(budgets.rounds.is_empty());
    }

    #[test]
    fn empty_pieces_need_no_iterations() {
        let budgets = in_scope(|| build_chunk_budgets(0, tokenizer(0, 0)));
        assert!(budgets.rounds.is_empty());
        assert!(budgets.apply_piece_ordinals.is_empty());
    }
}
