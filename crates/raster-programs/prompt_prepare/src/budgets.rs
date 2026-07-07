//! Chunk-budget derivation (port-plan deviation D9, gap G1) and the
//! one-shot staged-pieces count.
//!
//! Real recur loops are list-driven; this tile derives every bounded
//! iteration list the program needs from the staged inputs, and hoists the
//! sim's per-tile zero-width guards (`ensure_tokenizer_controls`,
//! `utils.rs:376`) into one fallible place. Message text matches the sim
//! exactly (deterministic, committed on the terminal path — catalog C23).

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;

use crate::types::{BpeConfig, BpePieces, ChunkBudgets, PieceCount};

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
/// and the round's pieces list are their own bounded recur inputs. Both
/// zero-width guards stay — the staged config is validated in one place
/// with the sim's exact messages.
#[tile]
pub fn build_chunk_budgets(piece_count: u32, config: BpeConfig) -> Result<ChunkBudgets> {
    if config.bpe_pairs_per_tile == 0 {
        return Err(String::from(
            "raster tokenizer BPE pairs per tile must be greater than zero",
        ));
    }
    if config.bpe_pieces_per_tile == 0 {
        return Err(String::from(
            "raster tokenizer BPE pieces per tile must be greater than zero",
        ));
    }

    let max_pairs = piece_count.saturating_sub(1);

    Ok(ChunkBudgets {
        rounds: ordinals(max_pairs),
        apply_chunks: ordinals(div_ceil(max_pairs, config.bpe_pieces_per_tile)),
    })
}

fn ordinals(count: u32) -> Vec<u32> {
    (0..count).collect()
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    debug_assert!(divisor > 0);
    value.div_ceil(divisor)
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

    fn config(pairs: u32, pieces: u32) -> BpeConfig {
        BpeConfig {
            bpe_pairs_per_tile: pairs,
            bpe_pieces_per_tile: pieces,
        }
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
    fn budgets_cover_the_worst_round() {
        let budgets = in_scope(|| build_chunk_budgets(5, config(2, 3))).expect("budgets");
        assert_eq!(budgets.rounds, vec![0, 1, 2, 3]);
        assert_eq!(budgets.apply_chunks, vec![0, 1]); // ceil(4 / 3)
    }

    #[test]
    fn single_piece_needs_no_rounds() {
        let budgets = in_scope(|| build_chunk_budgets(1, config(64, 64))).expect("budgets");
        assert!(budgets.rounds.is_empty());
        assert!(budgets.apply_chunks.is_empty());
    }

    #[test]
    fn empty_pieces_need_no_iterations() {
        let budgets = in_scope(|| build_chunk_budgets(0, config(64, 64))).expect("budgets");
        assert!(budgets.rounds.is_empty());
        assert!(budgets.apply_chunks.is_empty());
    }

    #[test]
    fn zero_chunk_widths_are_terminal_errors() {
        let error = in_scope(|| build_chunk_budgets(2, config(0, 1))).expect_err("zero pairs");
        assert_eq!(
            error,
            "raster tokenizer BPE pairs per tile must be greater than zero"
        );
        let error = in_scope(|| build_chunk_budgets(2, config(1, 0))).expect_err("zero pieces");
        assert_eq!(
            error,
            "raster tokenizer BPE pieces per tile must be greater than zero"
        );
    }
}
