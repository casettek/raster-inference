//! Chunk-budget derivation (port-plan deviation D9, gap G1).
//!
//! Real recur loops are list-driven; this tile derives every bounded
//! iteration list the program needs from the staged inputs, and hoists the
//! sim's per-tile zero-width guards (`ensure_tokenizer_controls`,
//! `utils.rs:376`) into one fallible place. Message text matches the sim
//! exactly (deterministic, committed on the terminal path — catalog C23).

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;

use crate::types::{BpeConfig, ChunkBudgets};

/// Bounds: rounds = `piece_count − 1` (each merge removes one piece); scan
/// and apply chunks cover the worst (first) round; token chunks cover the
/// initial count (final count ≤ initial). Shorter real iterations stop via
/// `RecurControl::Break`; an empty list finalizes cleanly with the initial
/// state (WS1 probe P1).
#[tile]
pub fn build_chunk_budgets(initial_pieces: Vec<String>, config: BpeConfig) -> Result<ChunkBudgets> {
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

    let piece_count = initial_pieces.len() as u32;
    let max_pairs = piece_count.saturating_sub(1);

    Ok(ChunkBudgets {
        rounds: ordinals(max_pairs),
        scan_chunks: ordinals(div_ceil(max_pairs, config.bpe_pairs_per_tile)),
        apply_chunks: ordinals(div_ceil(max_pairs, config.bpe_pieces_per_tile)),
        token_chunks: ordinals(div_ceil(piece_count, config.bpe_pieces_per_tile)),
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

    fn pieces(count: usize) -> Vec<String> {
        (0..count).map(|idx| idx.to_string()).collect()
    }

    fn config(pairs: u32, pieces: u32) -> BpeConfig {
        BpeConfig {
            bpe_pairs_per_tile: pairs,
            bpe_pieces_per_tile: pieces,
        }
    }

    #[test]
    fn budgets_cover_the_worst_round() {
        let budgets = in_scope(|| build_chunk_budgets(pieces(5), config(2, 3))).expect("budgets");
        assert_eq!(budgets.rounds, vec![0, 1, 2, 3]);
        assert_eq!(budgets.scan_chunks, vec![0, 1]); // ceil(4 / 2)
        assert_eq!(budgets.apply_chunks, vec![0, 1]); // ceil(4 / 3)
        assert_eq!(budgets.token_chunks, vec![0, 1]); // ceil(5 / 3)
    }

    #[test]
    fn single_piece_needs_no_rounds() {
        let budgets = in_scope(|| build_chunk_budgets(pieces(1), config(64, 64))).expect("budgets");
        assert!(budgets.rounds.is_empty());
        assert!(budgets.scan_chunks.is_empty());
        assert!(budgets.apply_chunks.is_empty());
        assert_eq!(budgets.token_chunks, vec![0]);
    }

    #[test]
    fn empty_pieces_need_no_iterations() {
        let budgets = in_scope(|| build_chunk_budgets(pieces(0), config(64, 64))).expect("budgets");
        assert!(budgets.rounds.is_empty());
        assert!(budgets.token_chunks.is_empty());
    }

    #[test]
    fn zero_chunk_widths_are_terminal_errors() {
        let error =
            in_scope(|| build_chunk_budgets(pieces(2), config(0, 1))).expect_err("zero pairs");
        assert_eq!(
            error,
            "raster tokenizer BPE pairs per tile must be greater than zero"
        );
        let error =
            in_scope(|| build_chunk_budgets(pieces(2), config(1, 0))).expect_err("zero pieces");
        assert_eq!(
            error,
            "raster tokenizer BPE pieces per tile must be greater than zero"
        );
    }
}
