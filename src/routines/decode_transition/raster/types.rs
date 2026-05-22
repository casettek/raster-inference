use crate::decode_transition::raster::auth_source::{
    GemmaDecodeLayerMatrixKind, GemmaDecodeLayerMetadata,
};
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterSelectedTokenRef,
};
use crate::shared::model::transformer::{TransformerDecodeState, TransformerDecodeStepResult};
use crate::shared::raster_kernels::transformer::RasterActivationRow;
use crate::shared::tensors::raster_tensor_artifacts::{
    RasterActivationSequenceRef, RasterAttentionHeadsRef, RasterKvCacheBuilderRef,
    RasterKvCacheRef, RasterProjectionOutputBuilderRef, RasterTensorBuilderRef,
};
use crate::RasterSizingControls;

// Types and constants used by the raster sequences and tiles.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeTransitionRasterState {
    pub(in super::super) artifact_store_roots: RasterArtifactStoreRoots,
    pub(in super::super) decode_input_ref: RasterActivationSequenceRef,
    pub(in super::super) current_activation_ref: RasterActivationSequenceRef,
    pub(in super::super) next_token: u32,
    pub(in super::super) position: usize,
    pub(in super::super) token_count: usize,
    pub(in super::super) next_layer_idx: usize,
    pub(in super::super) layer_count: usize,
    pub(in super::super) original_layer_caches: Vec<DecodeLayerCacheSlot>,
    pub(in super::super) updated_layer_caches: Vec<DecodeLayerCacheSlot>,
    pub(in super::super) completed_layer_output_sha256s: Vec<String>,
    pub(in super::super) completed_layer_output_det_sha256s: Vec<Option<String>>,
    pub(in super::super) projection_rows_per_tile: usize,
    pub(in super::super) attention_kv_rows_per_tile: usize,
    pub(in super::super) output_source_prefix: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterDecodeTransitionInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub transformer_decode_state: TransformerDecodeState,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub decode_transition_source_root: String,
    pub output_source_prefix: String,
    pub raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeTransitionInputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub position: usize,
    pub token_count: usize,
    pub layer_caches: Vec<DecodeLayerCacheSlot>,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub decode_transition_source_root: String,
    pub output_source_prefix: String,
    pub raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterDecodeTransitionOutputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub transition_result: TransformerDecodeStepResult,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeTransitionOutputStateRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub final_hidden_state_ref: RasterActivationSequenceRef,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
    pub layer_caches: Vec<DecodeLayerCacheSlot>,
    pub position: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum DecodeLayerCacheSlot {
    Empty { num_kv_heads: usize },
    Ref(RasterKvCacheRef),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeLogitsRasterState {
    pub(in super::super) normalized_final_position: RasterActivationRow,
    pub(in super::super) next_logit_idx: usize,
    pub(in super::super) logit_count: usize,
    pub(in super::super) output_builder_ref: RasterProjectionOutputBuilderRef,
    pub(in super::super) softcap_bits: Option<i32>,
    pub(in super::super) projection_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum DecodeProjectionKind {
    LayerMatrix {
        layer_idx: usize,
        matrix: GemmaDecodeLayerMatrixKind,
    },
    PleModel {
        layer_idx: usize,
    },
    FinalLogits,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeRowProjectionState {
    pub(in super::super) input: RasterActivationRow,
    pub(in super::super) projection_kind: DecodeProjectionKind,
    pub(in super::super) next_projection_row_idx: usize,
    pub(in super::super) projection_rows: usize,
    pub(in super::super) input_width: usize,
    pub(in super::super) output_builder_ref: RasterProjectionOutputBuilderRef,
    pub(in super::super) rows_per_tile: usize,
    pub(in super::super) softcap_bits: Option<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeRowProjectionArtifactState {
    pub(in super::super) input_ref: RasterActivationSequenceRef,
    pub(in super::super) projection_kind: DecodeProjectionKind,
    pub(in super::super) output_source_name: String,
    pub(in super::super) current_row_bits: Vec<i32>,
    pub(in super::super) next_projection_row_idx: usize,
    pub(in super::super) projection_rows: usize,
    pub(in super::super) input_width: usize,
    pub(in super::super) rows_per_tile: usize,
    pub(in super::super) softcap_bits: Option<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeAttentionState {
    pub(in super::super) query_ref: RasterAttentionHeadsRef,
    pub(in super::super) cache_ref: RasterKvCacheRef,
    pub(in super::super) output_builder_ref: RasterTensorBuilderRef,
    pub(in super::super) phase: DecodeAttentionPhase,
    pub(in super::super) attention_id_prefix: String,
    pub(in super::super) next_query_head_idx: usize,
    pub(in super::super) query_head_count: usize,
    pub(in super::super) kv_head_count: usize,
    pub(in super::super) kv_groups: usize,
    pub(in super::super) key_start: usize,
    pub(in super::super) row_count: usize,
    pub(in super::super) head_dim: usize,
    pub(in super::super) kv_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeAttentionArtifactState {
    pub(in super::super) query_ref: RasterAttentionHeadsRef,
    pub(in super::super) cache_ref: RasterKvCacheRef,
    pub(in super::super) output_source_name: String,
    pub(in super::super) phase: DecodeAttentionArtifactPhase,
    pub(in super::super) attention_id_prefix: String,
    pub(in super::super) next_query_head_idx: usize,
    pub(in super::super) query_head_count: usize,
    pub(in super::super) kv_head_count: usize,
    pub(in super::super) kv_groups: usize,
    pub(in super::super) key_start: usize,
    pub(in super::super) row_count: usize,
    pub(in super::super) head_dim: usize,
    pub(in super::super) kv_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) struct DecodeLayerContext {
    pub(in super::super) layer_idx: usize,
    pub(in super::super) layer: GemmaDecodeLayerMetadata,
    pub(in super::super) cache_slot: DecodeLayerCacheSlot,
    pub(in super::super) donor_cache_slot: Option<DecodeLayerCacheSlot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeKvCacheAppendState {
    pub(in super::super) old_cache_ref: Option<RasterKvCacheRef>,
    pub(in super::super) key_ref: RasterAttentionHeadsRef,
    pub(in super::super) value_ref: RasterAttentionHeadsRef,
    pub(in super::super) output_builder_ref: RasterKvCacheBuilderRef,
    pub(in super::super) retained_old_start: usize,
    pub(in super::super) retained_old_len: usize,
    pub(in super::super) next_head_idx: usize,
    pub(in super::super) next_old_offset: usize,
    pub(in super::super) head_count: usize,
    pub(in super::super) rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeKvCacheAppendArtifactState {
    pub(in super::super) old_cache_ref: Option<RasterKvCacheRef>,
    pub(in super::super) key_ref: RasterAttentionHeadsRef,
    pub(in super::super) value_ref: RasterAttentionHeadsRef,
    pub(in super::super) keys_source_name: String,
    pub(in super::super) values_source_name: String,
    pub(in super::super) retained_old_start: usize,
    pub(in super::super) retained_old_len: usize,
    pub(in super::super) next_head_idx: usize,
    pub(in super::super) next_old_offset: usize,
    pub(in super::super) head_count: usize,
    pub(in super::super) current_len: usize,
    pub(in super::super) head_dim: usize,
    pub(in super::super) rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) enum DecodeAttentionPhase {
    CollectScores {
        score_builder_ref: RasterTensorBuilderRef,
        next_kv_offset: usize,
    },
    FindSoftmaxMax {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: Option<usize>,
        max_logit_bits: i32,
    },
    SumSoftmaxExp {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
    },
    BuildRawSoftmaxWeights {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        raw_weight_builder_ref: RasterTensorBuilderRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
        summed_weight_bits: i32,
    },
    CorrectSoftmaxResidual {
        raw_weight_ref:
            crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        final_weight_builder_ref: RasterTensorBuilderRef,
        next_weight_row_idx: usize,
        max_index: usize,
        residual_bits: i32,
    },
    ApplyValues {
        weight_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_kv_offset: usize,
        weighted_sum_acc_bits: Vec<i64>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) enum DecodeAttentionArtifactPhase {
    CollectScores {
        score_source_name: String,
        next_kv_offset: usize,
    },
    FindSoftmaxMax {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: Option<usize>,
        max_logit_bits: i32,
    },
    SumSoftmaxExp {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
    },
    BuildRawSoftmaxWeights {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        raw_weight_source_name: String,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
        summed_weight_bits: i32,
    },
    CorrectSoftmaxResidual {
        raw_weight_ref:
            crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        final_weight_source_name: String,
        next_weight_row_idx: usize,
        max_index: usize,
        residual_bits: i32,
    },
    ApplyValues {
        weight_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_kv_offset: usize,
        weighted_sum_acc_bits: Vec<i64>,
    },
}
