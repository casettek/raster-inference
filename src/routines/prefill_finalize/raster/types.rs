use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterRoutineOutput,
};
use crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef;

// Types and constants used by the raster sequences and tiles.

pub const NORMALIZED_FINAL_POSITION_ARTIFACT_NAME: &str =
    "prefill.finalize.normalized_final_position";

pub const PREFILL_LOGITS_ARTIFACT_NAME: &str = "prefill.finalize.logits";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillFinalizeInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub prompt_token_count: usize,
    pub finalize_source_root: String,
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<crate::routines::prefill_range::raster::PrefillLayerCacheSlot>,
    pub projection_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillFinalizeRefs {
    pub source_id: String,
    pub finalize_source_root: String,
    pub prompt_token_count: usize,
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<crate::routines::prefill_range::raster::PrefillLayerCacheSlot>,
    pub normalized_final_position_ref: RasterActivationSequenceRef,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
}

pub type RasterPrefillFinalizeOutput = RasterRoutineOutput<RasterPrefillFinalizeRefs>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillFinalizeRasterState {
    pub(in super::super) artifact_store_roots: RasterArtifactStoreRoots,
    pub(in super::super) source_id: String,
    pub(in super::super) finalize_source_root: String,
    pub(in super::super) prompt_token_count: usize,
    pub(in super::super) final_hidden_states_ref: RasterActivationSequenceRef,
    pub(in super::super) normalized_final_position_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) next_logit_idx: usize,
    pub(in super::super) logit_count: usize,
    pub(in super::super) hidden_width: usize,
    pub(in super::super) softcap_bits: Option<i32>,
    pub(in super::super) projection_rows_per_tile: usize,
}

impl PrefillFinalizeRasterState {
    pub(in super::super) fn is_complete(&self) -> bool {
        self.next_logit_idx >= self.logit_count
    }
}
