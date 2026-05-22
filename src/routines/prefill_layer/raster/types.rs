use crate::shared::raster_contracts::prefill_layer::GemmaPrefillLayerMetadata;
use crate::shared::tensors::raster_tensor_artifacts::{
    RasterActivationSequenceRef, RasterKvCacheRef,
};

// Types and constants used by the raster sequences and tiles.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerRasterState {
    pub(in super::super) current_activations_ref: RasterActivationSequenceRef,
    pub(in super::super) next_layer_idx: usize,
    pub(in super::super) layer_count: usize,
    pub(in super::super) layer_caches: Vec<PrefillLayerCacheSlot>,
    pub(in super::super) per_layer_inputs: Vec<Option<RasterActivationSequenceRef>>,
    pub(in super::super) completed_layer_output_sha256s: Vec<String>,
    pub(in super::super) completed_layer_output_det_sha256s: Vec<Option<String>>,
    pub(in super::super) projection_rows_per_tile: usize,
    pub(in super::super) attention_kv_rows_per_tile: usize,
    pub(in super::super) sequence_rows_per_tile: usize,
    pub(in super::super) head_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum PrefillLayerCacheSlot {
    Empty { num_kv_heads: usize },
    Ref(RasterKvCacheRef),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerOutputRefs {
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<PrefillLayerCacheSlot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerContext {
    pub(in super::super) layer_idx: usize,
    pub(in super::super) layer: GemmaPrefillLayerMetadata,
    pub(in super::super) donor_cache: Option<PrefillLayerCacheSlot>,
    pub(in super::super) per_layer_input: Option<RasterActivationSequenceRef>,
}
