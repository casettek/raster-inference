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

use crate::types::{BpePieces, ChunkBudgets, PieceCount};

/// One-shot authenticated read of the staged `BpePieces` external: derives
/// the piece count in-program instead of staging a separate count (an
/// unchecked staged count is an integrity hole; a checked one is
/// redundant).
#[tile]
pub fn count_pieces(pieces: BpePieces) -> PieceCount {
    PieceCount {
        piece_count: pieces.pieces.len() as u32,
    }
}

/// Bounds: rounds = `piece_count − 1` (each merge removes one piece).
/// Shorter real iterations no-op after convergence (recur sequences cannot
/// break early — gap G1); an empty list finalizes cleanly with the initial
/// state (WS1 probe P1). The scan, apply, and token-id phases need no
/// derived budgets (storage-resident refactor): the chunked model tables
/// and the round's pieces list are their own bounded recur inputs.
#[tile]
pub fn build_chunk_budgets(piece_count: u32) -> ChunkBudgets {
    ChunkBudgets {
        rounds: ordinals(piece_count.saturating_sub(1)),
    }
}

fn ordinals(count: u32) -> Vec<u32> {
    (0..count).collect()
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;

    /// Tile fns require an active sequence scope even when driven natively.
    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_budget_tests");
        run()
    }

    #[test]
    fn count_reads_the_staged_pieces() {
        let count = in_scope(|| {
            count_pieces(BpePieces {
                pieces: vec!["a".to_string(), "b".to_string()],
            })
        });
        assert_eq!(count.piece_count, 2);
    }

    #[test]
    fn rounds_cover_one_merge_per_removed_piece() {
        let budgets = in_scope(|| build_chunk_budgets(5));
        assert_eq!(budgets.rounds, vec![0, 1, 2, 3]);
    }

    #[test]
    fn single_piece_needs_no_rounds() {
        let budgets = in_scope(|| build_chunk_budgets(1));
        assert!(budgets.rounds.is_empty());
    }

    #[test]
    fn empty_pieces_need_no_iterations() {
        let budgets = in_scope(|| build_chunk_budgets(0));
        assert!(budgets.rounds.is_empty());
    }
}
