use crate::routines::output_finalize::raster::auth_source::{
    OutputPendingBytesRef, OutputTextRef, OutputTokenIdsCommitmentState, OutputUtf8ValidationState,
};
use crate::shared::api::output::OutputDecodeStopReason;
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterRoutineOutput, RasterTokenIdSequenceRef,
};

// Types and constants used by the raster sequences and tiles.

pub const DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE: usize = 16;

pub(in super::super) const INVALID_UTF8_REPLACEMENT: &str = "�";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterOutputFinalizeInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub generated_token_ids_ref: RasterTokenIdSequenceRef,
    pub tokenizer_source_root: String,
    pub output_text_source_name: String,
    pub pending_bytes_source_prefix: String,
    pub byte_flush_bytes_per_tile: usize,
    pub stop_reason: OutputDecodeStopReason,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterOutputFinalizeRefs {
    pub generated_token_ids_ref: RasterTokenIdSequenceRef,
    pub generated_text_ref: OutputTextRef,
    pub generated_token_ids_sha256: String,
    pub generated_token_count: usize,
    pub stop_reason: OutputDecodeStopReason,
}

pub type RasterOutputFinalizeOutput = RasterRoutineOutput<RasterOutputFinalizeRefs>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterOutputDetokenizeState {
    pub(in super::super) artifact_store_roots: RasterArtifactStoreRoots,
    pub(in super::super) token_ids_ref: RasterTokenIdSequenceRef,
    pub(in super::super) next_token_idx: usize,
    pub(in super::super) token_count: usize,
    pub(in super::super) tokenizer_source_root: String,
    pub(in super::super) text_builder_source_name: String,
    pub(in super::super) pending_bytes_source_prefix: String,
    pub(in super::super) pending_bytes_builder_source_name: Option<String>,
    pub(in super::super) pending_bytes_written: usize,
    pub(in super::super) pending_segment_idx: usize,
    pub(in super::super) text_chunk_count: usize,
    pub(in super::super) text_byte_len: usize,
    pub(in super::super) text_char_count: usize,
    pub(in super::super) token_commitment: OutputTokenIdsCommitmentState,
    pub(in super::super) replacement_pattern: String,
    pub(in super::super) replacement_content: String,
    pub(in super::super) byte_fallback: bool,
    pub(in super::super) phase: RasterOutputDetokenizePhase,
    pub(in super::super) byte_flush_bytes_per_tile: usize,
    pub(in super::super) stop_reason: OutputDecodeStopReason,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) enum RasterOutputDetokenizePhase {
    ReadNextToken,
    ValidatePendingBytes {
        continuation: RasterOutputPendingFlushContinuation,
        pending_ref: OutputPendingBytesRef,
        next_byte_idx: usize,
        validation_state: OutputUtf8ValidationState,
    },
    FlushPendingBytes {
        continuation: RasterOutputPendingFlushContinuation,
        pending_ref: OutputPendingBytesRef,
        next_byte_idx: usize,
        valid_utf8: bool,
    },
    Complete,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) enum RasterOutputPendingFlushContinuation {
    ReadNextToken,
    ReplayCurrentToken,
    Complete,
}
