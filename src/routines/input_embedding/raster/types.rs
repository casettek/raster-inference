use crate::shared::artifacts::raster_artifact_store::{
    RasterActivationSequenceArtifactRef, RasterRoutineOutput,
};

// Types and constants used by the raster sequences and tiles.

pub(in super::super) const EMBEDDED_PROMPT_ARTIFACT_NAME: &str = "input.embedding.embedded_prompt";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterInputEmbeddingInputRoots {
    pub prompt_token_ids_root: String,
    pub prompt_token_count: usize,
    pub embedding_source_root: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterInputEmbeddingRefs {
    pub source_id: String,
    pub embedding_source_root: String,
    pub prompt_token_ids_root: String,
    pub prompt_token_count: usize,
    pub embedded_prompt_activations_ref: RasterActivationSequenceArtifactRef,
}

pub type RasterInputEmbeddingOutput = RasterRoutineOutput<RasterInputEmbeddingRefs>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct InputEmbeddingRasterState {
    pub(in super::super) source_id: String,
    pub(in super::super) embedding_source_root: String,
    pub(in super::super) prompt_token_ids_root: String,
    pub(in super::super) prompt_token_count: usize,
    pub(in super::super) hidden_size: usize,
    pub(in super::super) next_token_idx: usize,
}

impl InputEmbeddingRasterState {
    pub(in super::super) fn is_complete(&self) -> bool {
        self.next_token_idx >= self.prompt_token_count
    }
}
