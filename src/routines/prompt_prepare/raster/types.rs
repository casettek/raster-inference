use crate::shared::api::input::RasterPromptPreparationState;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::gemma::tokenizer::GemmaBpeState;

// Types and constants used by the raster sequences and tiles.

pub const DEFAULT_BPE_PAIRS_PER_TILE: usize = 64;

pub const DEFAULT_BPE_PIECES_PER_TILE: usize = 64;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TokenizePromptInput {
    pub rendered_prompt: String,
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone)]
pub struct RasterPromptPreparationResult {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub state: RasterPromptPreparationState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPromptPreparedInputs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub input_roots: RasterPromptInputRoots,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPromptInputRoots {
    pub tokenizer_source_root: String,
    pub bpe_state: GemmaBpeState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterTokenizationResult {
    pub token_ids_root: String,
    pub token_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeTokenizeSequenceState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub bpe_state: GemmaBpeState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum GemmaBpeMergeIterationState {
    Complete(GemmaBpeTokenizeSequenceState),
    Applying(GemmaBpeApplyTileState),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeMergeDecision {
    pub sequence_state: GemmaBpeTokenizeSequenceState,
    pub selection: Option<GemmaBpeMergeSelection>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeMergeSelection {
    pub piece_idx: usize,
    pub merge_index: usize,
    pub merged: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanCandidate {
    pub piece_idx: usize,
    pub rank: usize,
    pub merge_index: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanState {
    pub piece_count: usize,
    pub iteration: u64,
    pub next_pair_idx: usize,
    pub best_candidate: Option<GemmaBpeScanCandidate>,
    pub bpe_pairs_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanTileState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub bpe_state: GemmaBpeState,
    pub scan_state: GemmaBpeScanState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyState {
    pub input_piece_count: usize,
    pub merge_piece_idx: usize,
    pub merged: String,
    pub input_cursor: usize,
    pub output_cursor: usize,
    pub add_special_tokens: bool,
    pub iteration: u64,
    pub bpe_pairs_per_tile: usize,
    pub bpe_pieces_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyTileState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub apply_state: GemmaBpeApplyState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdFinalizeState {
    pub piece_count: usize,
    pub pieces_iteration: u64,
    pub next_piece_idx: usize,
    pub pieces_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdFinalizeTileState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub token_id_state: GemmaTokenIdFinalizeState,
}
