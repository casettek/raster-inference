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
//! - **Prompt-scoped data** (pieces, pairs, token ids) lives behind
//!   storage-backed selectable roots or drafts; loop state carries only
//!   scalar metadata and storage handles.
//! - **Cursors/flags/candidates** are tiny loop-state structs.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use raster::{
    ExternalRef, ExternalSelection, InternalRef, Selectable, SelectorPath, SelectorSegment,
};
use raster_program_gemma_externals::types::GemmaTokenizer;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

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

/// Compact tokenizer-table source descriptor. Program `main` uses the
/// committed external name; native tests use an internal `GemmaTokenizer`
/// reference so they can drive the same in-tile storage-read path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenizerSource {
    External(ExternalRef),
    Internal(InternalRef),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerTables {
    pub source: TokenizerSource,
}

impl TokenizerTables {
    pub fn external(name: &str) -> Self {
        Self {
            source: TokenizerSource::External(ExternalRef::new(name)),
        }
    }

    pub fn internal(reference: InternalRef) -> Self {
        Self {
            source: TokenizerSource::Internal(reference),
        }
    }
}

/// Bounded iteration lists for every recur loop in the program (gap G1:
/// real recur is list-driven; every sim until-done loop has a derivable
/// bound at loop start). Derived in-program by `build_chunk_budgets`.
/// The storage-read refactor uses scalar ordinals for model chunks and
/// pieces; tiles use refs/selectors to read the actual payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ChunkBudgets {
    /// BPE merge rounds: `initial_piece_count − 1` ordinals (each merge
    /// removes one piece).
    pub rounds: Vec<u32>,
    pub apply_piece_ordinals: Vec<u32>,
    pub merge_chunk_ordinals: Vec<u32>,
    pub token_lookup_chunk_ordinals: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct PieceOrdinals {
    pub ordinals: Vec<u32>,
}

/// Loop-carried state of the outer BPE merge-round sequence: scalar BPE
/// metadata plus the current pieces' internal-storage handle. The state
/// itself still traces inline (pinned-rev `RecurSequenceState` behavior),
/// but prompt pieces no longer ride that inline state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeLoopState {
    /// No merge candidate remained (or an error was recorded); remaining
    /// rounds no-op (G1: recur sequences cannot break early).
    pub complete: bool,
    pub round: u32,
    pub piece_count: u32,
    /// Internal-storage reference for the current round's `BpePieces`.
    pub pieces_ref: InternalRef,
    /// Deferred in-loop guard error (A1); surfaced as the terminal `Err` by
    /// the first fallible plain tile after the loop.
    pub error: Option<String>,
}

impl GemmaBpeLoopState {
    pub fn initial(pieces_ref: InternalRef) -> Self {
        Self {
            complete: false,
            round: 0,
            piece_count: 0,
            pieces_ref,
            error: None,
        }
    }
}

pub(crate) fn resolve_bpe_pieces_ref(pieces_ref: InternalRef, context: &str) -> BpePieces {
    raster::resolve_internal_value::<BpePieces>(pieces_ref)
        .unwrap_or_else(|error| panic!("Failed to resolve BPE pieces for {context}: {error}"))
        .value
}

/// One opened BPE merge round: the round scalars plus the round's working
/// pieces behind the selectable `BpePieces` root. Produced once per round
/// by `open_round`; everything downstream consumes it through authenticated
/// reads — `select!` projections (the scan's `skip`, the apply loop's
/// pieces list) and selection-bound args (`build_pairs`, `finalize_round`).
///
/// The deferred round error is flattened to `(has_error, error)` scalars
/// because `Option` has no `Selectable` schema at the pinned rev (catalog
/// G3: no enum payloads in selectable shapes); `GemmaBpeLoopState` keeps
/// the `Option<String>` convention (A1) and `finalize_round` reconstructs
/// it, message text untouched (H4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeOpenedRound {
    /// The round is a no-op (loop already complete or errored).
    pub skip: bool,
    pub round: u32,
    pub piece_count: u32,
    pub has_error: bool,
    /// Deferred error message when `has_error`; empty otherwise.
    pub error: String,
    pub pieces_ref: InternalRef,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPieces {
    pub pieces_ref: InternalRef,
    pub piece_count: u32,
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

/// Outcome of one round's scan phase, scalars only (successor of the
/// pieces-carrying `GemmaBpeMergeDecision` — deviation D15): `skip` covers
/// skipped rounds, deferred errors, and no-selection convergence alike; the
/// apply loop no-ops on it and `finalize_round` carries the incoming pieces
/// forward.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyDecision {
    /// Nothing to apply this round (skipped round or no selection).
    pub skip: bool,
    /// Left index of the winning adjacent pair.
    pub merge_piece_idx: u32,
    /// The winning rule's merged token (one bounded token — a scalar-sized
    /// decision leg, not a prompt-derived collection).
    pub merged: String,
}

/// Cursor-only state of the per-piece apply loop (deviation D15): the next
/// round's pieces accumulate in the `RecurOutput<BpePieces>` draft, never
/// in this state. `emitted` doubles as the merge-point detector — before
/// the merge every piece is pushed, so `emitted == merge_piece_idx` fires
/// exactly at the winning pair's left piece; `skip_next` swallows the
/// consumed right-hand piece.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyCursor {
    pub skip_next: bool,
    pub emitted: u32,
}

impl GemmaBpeApplyCursor {
    pub fn initial() -> Self {
        Self {
            skip_next: false,
            emitted: 0,
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

pub(crate) fn field_selector(field: &str) -> SelectorPath {
    SelectorPath::new(alloc::vec![SelectorSegment::Field(field.to_string())])
}

pub(crate) fn field_index_selector(field: &str, index: u32) -> SelectorPath {
    SelectorPath::new(alloc::vec![
        SelectorSegment::Field(field.to_string()),
        SelectorSegment::Index(index as u64),
    ])
}

pub(crate) fn piece_selector(index: u32) -> SelectorPath {
    field_index_selector("pieces", index)
}

pub(crate) fn read_tokenizer_selection<T>(
    tokenizer: &TokenizerTables,
    selector: SelectorPath,
    context: &str,
) -> T
where
    T: DeserializeOwned + Serialize,
{
    match &tokenizer.source {
        TokenizerSource::External(reference) => {
            raster::resolve_typed_external_value::<GemmaTokenizer, T>(ExternalSelection {
                reference: ExternalRef::with_selector(reference.name.clone(), selector),
            })
            .unwrap_or_else(|error| panic!("Failed to resolve tokenizer {context}: {error}"))
            .value
        }
        TokenizerSource::Internal(reference) => {
            raster::select_stored_internal_value::<T>(reference, &selector)
                .unwrap_or_else(|error| panic!("Failed to resolve tokenizer {context}: {error}"))
                .value
        }
    }
}

pub(crate) fn read_bpe_piece(pieces_ref: &InternalRef, index: u32) -> String {
    raster::select_stored_internal_value::<String>(pieces_ref, &piece_selector(index))
        .unwrap_or_else(|error| panic!("Failed to resolve BPE piece {index}: {error}"))
        .value
}
