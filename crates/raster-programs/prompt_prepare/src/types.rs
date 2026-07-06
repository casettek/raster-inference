//! Cross-tile types for the `prompt.prepare` program.
//!
//! Postcard-safe owned equivalents of the sim types
//! (`src/routines/prompt_prepare/raster/types.rs` in the main crate), per
//! the port plan's types table: fixed-width integers (catalog C12), roots
//! legs deleted (C15), tuples replaced by named structs (G3), in-loop guard
//! errors deferred through `error` fields because recur tiles are
//! infallible at the pinned rev (port-plan constraint A1).
//!
//! Tokenizer-side schema types (`GemmaTokenIdEntry`,
//! `GemmaBpeMergeLookupEntry`, …) come from the shared
//! `raster-program-gemma-externals` crate.

use alloc::string::String;
use alloc::vec::Vec;
use raster::Selectable;
use serde::{Deserialize, Serialize};

/// Staged tokenizer chunk-width configuration (catalog C27: widths are
/// staged by the host from `RasterSizingControls`, one obvious place for
/// WS8 retuning).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct BpeConfig {
    pub bpe_pairs_per_tile: u32,
    pub bpe_pieces_per_tile: u32,
}

/// Bounded iteration lists for every recur loop in the program (gap G1:
/// real recur is list-driven; every sim until-done loop has a derivable
/// bound at loop start). Derived in-program by `build_chunk_budgets`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ChunkBudgets {
    /// BPE merge rounds: `initial_piece_count − 1` ordinals (each merge
    /// removes one piece).
    pub rounds: Vec<u32>,
    /// Pair-scan chunks per round: `ceil((initial_piece_count − 1) /
    /// bpe_pairs_per_tile)` ordinals — an upper bound for every round.
    pub scan_chunks: Vec<u32>,
    /// Merge-apply chunks per round: `ceil((initial_piece_count − 1) /
    /// bpe_pieces_per_tile)` ordinals.
    pub apply_chunks: Vec<u32>,
    /// Token-id finalization chunks: `ceil(initial_piece_count /
    /// bpe_pieces_per_tile)` ordinals (final count ≤ initial).
    pub token_chunks: Vec<u32>,
}

/// Loop-carried state of the outer BPE merge-round sequence (sim
/// `GemmaBpeTokenizeSequenceState` with the roots leg deleted and the
/// working pieces carried inline — port-plan deviation D6).
///
/// Real recur loops seed from a plain literal (port-plan constraint A2), so
/// the state starts uninitialized and the first executed round populates
/// `pieces` from the staged `initial_pieces` input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeLoopState {
    pub initialized: bool,
    /// No merge candidate remained (or an error was recorded); remaining
    /// rounds no-op (G1: recur sequences cannot break early).
    pub complete: bool,
    pub round: u32,
    pub pieces: Vec<String>,
    /// Deferred in-loop guard error (A1); surfaced as the terminal `Err` by
    /// the first fallible plain tile after the loop.
    pub error: Option<String>,
}

impl GemmaBpeLoopState {
    pub fn initial() -> Self {
        Self {
            initialized: false,
            complete: false,
            round: 0,
            pieces: Vec::new(),
            error: None,
        }
    }
}

/// Read-only context of one BPE merge round, produced by
/// `init_bpe_merge_scan` and threaded to the round's recur tiles through
/// `args = (…)` (A2: loop state seeds must be literals, heavy context rides
/// the materialized-once args).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeRoundContext {
    /// The round is a no-op (loop already complete or errored).
    pub skip: bool,
    pub round: u32,
    pub pieces: Vec<String>,
    pub pair_count: u32,
    pub error: Option<String>,
}

/// Best merge candidate found by the pair scan. `merge_index` is the merge
/// priority (the external's `merges` are ordered by it — the sim's `rank`);
/// the merged token is captured at scan time from the `merge_lookup` entry
/// (port-plan deviation D7b).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanCandidate {
    pub pair_idx: u32,
    pub merge_index: u32,
    pub merged: String,
}

/// Loop-carried cursor of the chunked pair scan (sim `GemmaBpeScanState`
/// minus the roots/pieces legs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanState {
    pub next_pair_idx: u32,
    pub best: Option<GemmaBpeScanCandidate>,
    pub done: bool,
}

impl GemmaBpeScanState {
    pub fn initial() -> Self {
        Self {
            next_pair_idx: 0,
            best: None,
            done: false,
        }
    }
}

/// Outcome of one round's scan phase (sim `GemmaBpeMergeDecision` with the
/// selection candidate folded in).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeMergeDecision {
    pub skip: bool,
    pub round: u32,
    pub pieces: Vec<String>,
    pub selection: Option<GemmaBpeScanCandidate>,
    pub error: Option<String>,
}

/// Read-only context of one round's apply phase (sim
/// `GemmaBpeMergeIterationState::Applying` payload; the `Complete` variant
/// becomes the `complete` flag — catalog C28 branching stays in tiles).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeIterationContext {
    /// Nothing to apply this round (no selection, skip, or error).
    pub complete: bool,
    pub round: u32,
    pub pieces: Vec<String>,
    pub merge_piece_idx: u32,
    pub merged: String,
    pub error: Option<String>,
}

/// Loop-carried state of the chunked merge apply: the next round's pieces
/// accumulate in `output` (sim built them into a `bpe-pieces-{N+1}` store
/// artifact — deviation D6).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyState {
    pub output: Vec<String>,
    pub input_cursor: u32,
    pub output_cursor: u32,
    pub done: bool,
}

impl GemmaBpeApplyState {
    pub fn initial() -> Self {
        Self {
            output: Vec::new(),
            input_cursor: 0,
            output_cursor: 0,
            done: false,
        }
    }
}

/// Read-only context of the token-id finalization loop, produced by
/// `init_token_id_finalization` from the finished BPE loop state (sim
/// `GemmaBpeOutput` + builder start, with the builder dissolved — D6/D6a).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdContext {
    pub pieces: Vec<String>,
    pub piece_count: u32,
    pub error: Option<String>,
}

/// Loop-carried state of the chunked token-id finalization: ids accumulate
/// in state (sim appended to the `prompt-token-ids` builder — D6).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdState {
    pub token_ids: Vec<u32>,
    pub next_piece_idx: u32,
    pub error: Option<String>,
}

impl GemmaTokenIdState {
    pub fn initial() -> Self {
        Self {
            token_ids: Vec::new(),
            next_piece_idx: 0,
            error: None,
        }
    }
}

/// The program's output value (sim `RasterTokenizationResult` in value form:
/// the ids themselves, not a store root). Written to `output.bin` as
/// `postcard(Result<PromptTokenization, String>)`; the host decodes it with
/// a field-order-matching mirror struct (WS2 §9.6 postcard layout contract).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct PromptTokenization {
    pub token_ids: Vec<u32>,
    pub token_count: u32,
}
