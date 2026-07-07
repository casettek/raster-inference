//! Cross-tile types for the `prompt.prepare` program.
//!
//! Postcard-safe owned equivalents of the sim types
//! (`src/routines/prompt_prepare/raster/types.rs` in the main crate), per
//! the port plan's types table: fixed-width integers (catalog C12), roots
//! legs deleted (C15), tuples replaced by named structs (G3), in-loop guard
//! errors deferred through `error` fields because recur tiles are
//! infallible at the pinned rev (port-plan constraint A1).
//!
//! Tokenizer-side schema types (`GemmaTokenIdEntry`, `GemmaBpeMerge`, …)
//! come from the shared `raster-program-gemma-externals` crate.
//!
//! Data-placement rules (the storage-refactor design contract, tightened
//! by the trace-slimming refactor D13/D14):
//! - **Model-scoped tables** (vocab, merges) are pre-chunked committed
//!   externals consumed *only* as recur-sequence input lists — each chunk
//!   crosses the tile ABI as an external-selection binding and
//!   materializes only at tile execution; never in `args`, never in loop
//!   state, never inline in the trace.
//! - **Prompt-scoped data** (pieces, pairs, token ids) rides loop state or
//!   small args — bounded by the prompt, not the model.
//! - **Cursors/flags/candidates** are tiny loop-state structs.

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

/// Prompt-scoped pieces behind a selectable root (the storage-resident
/// refactor's `BpePieces`): the staged `initial_pieces` external, every
/// round's `open_round` output, and every round's finalized
/// `RecurOutput<BpePieces>` draft all carry pieces in this shape, so tiles
/// consume them only through authenticated reads — `select!` projections,
/// input-handle selections, and selection-bound args. Postcard encodes a
/// single-field struct identically to the bare `Vec<String>`, so staging
/// keeps its byte layout (WS2 §9.6).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct BpePieces {
    pub pieces: Vec<String>,
}

/// One resolved vocab match of the append-only token-id pass: the piece's
/// position in the final pieces and its resolved id. Appended by
/// `resolve_pieces_in_vocab_chunk` into the `RecurOutput<TokenIdMatches>`
/// draft; ordered by `piece_idx` in the terminal finalizer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct TokenIdMatch {
    pub piece_idx: u32,
    pub token_id: u32,
}

/// Draft-accumulated matches of the token-id phase (supersedes the
/// threaded `GemmaTokenResolutionState` slot vector): append-only, no
/// threaded resolution state; the ordered ids materialize once, in
/// `finalize_tokenize_prompt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct TokenIdMatches {
    pub matches: Vec<TokenIdMatch>,
}

/// Output of the one-shot `count_pieces` tile: the staged pieces count,
/// derived in-program through an authenticated read of the `initial_pieces`
/// external (a staged count would be either unchecked — an integrity hole —
/// or redundant).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct PieceCount {
    pub piece_count: u32,
}

/// Bounded iteration lists for every recur loop in the program (gap G1:
/// real recur is list-driven; every sim until-done loop has a derivable
/// bound at loop start). Derived in-program by `build_chunk_budgets`.
/// The scan and token-id phases need no derived budgets anymore: the
/// chunked model tables are their own bounded recur-sequence input lists
/// (the scan loops over `merge_chunks`, the inverted vocab pass over
/// `token_lookup_chunks` — D14).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ChunkBudgets {
    /// BPE merge rounds: `initial_piece_count − 1` ordinals (each merge
    /// removes one piece).
    pub rounds: Vec<u32>,
    /// Merge-apply chunks per round: `ceil((initial_piece_count − 1) /
    /// bpe_pieces_per_tile)` ordinals.
    pub apply_chunks: Vec<u32>,
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

/// Winning merge candidate of the priority-order scan. `merge_index` is
/// the merge priority (the external's `merge_chunks` are ordered by it —
/// the sim's `rank`); the merged token is captured at scan time from the
/// winning `GemmaBpeMerge` rule (port-plan deviation D11).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanCandidate {
    pub pair_idx: u32,
    pub merge_index: u32,
    pub merged: String,
}

/// Loop-carried state of the priority-order merge scan: one merge-table
/// chunk per iteration; the first rule with an adjacent-pair occurrence
/// wins (lowest `merge_index` globally — chunks preserve priority order)
/// and sets `done`, after which the remaining chunks no-op (recur
/// sequences cannot break early — gap G1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanState {
    /// Chunks visited (diagnostic; a converged round pays one full pass).
    pub chunks_scanned: u32,
    pub best: Option<GemmaBpeScanCandidate>,
    pub done: bool,
}

impl GemmaBpeScanState {
    pub fn initial() -> Self {
        Self {
            chunks_scanned: 0,
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

/// Read-only context of the token-id resolution loop, produced by
/// `init_token_id_finalization` from the finished BPE loop state (sim
/// `GemmaBpeOutput` + builder start, with the builder dissolved — D6/D6a).
/// Prompt-scoped: rides the vocab pass as a small `args` context (nothing
/// selects out of it anymore — D14).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdContext {
    pub pieces: Vec<String>,
    pub piece_count: u32,
    /// Deferred BPE-loop error (A1); surfaced as the terminal `Err` by
    /// `finalize_tokenize_prompt`.
    pub error: Option<String>,
}

/// Loop-carried state of the inverted vocab pass (D14): one slot per final
/// piece, filled as the single pass over `token_lookup_chunks` encounters
/// each piece's sorted position. Prompt-scoped; seeds from a literal (A2)
/// and sizes itself from the context on the first executed iteration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaTokenResolutionState {
    pub initialized: bool,
    pub resolved: Vec<Option<u32>>,
}

impl GemmaTokenResolutionState {
    pub fn initial() -> Self {
        Self {
            initialized: false,
            resolved: Vec::new(),
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
