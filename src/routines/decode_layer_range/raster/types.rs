use crate::decode_layer_range::raster::auth_source::{
    GemmaDecodeLayerMatrixKind, GemmaDecodeLayerMetadata, GemmaDecodePleScalars,
};
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterSelectedTokenRef,
};
use crate::shared::model::transformer::{TransformerDecodeState, TransformerDecodeStepResult};
use crate::shared::numerics::det_num::Wgt;
use crate::shared::raster_kernels::transformer::RasterActivationRow;
use crate::shared::tensors::raster_tensor_artifacts::{
    RasterActivationSequenceRef, RasterAttentionHeadsRef, RasterKvCacheBuilderRef,
    RasterKvCacheRef, RasterProjectionOutputBuilderRef, RasterTensorBuilderRef,
};
use crate::RasterSizingControls;

// Types and constants used by the raster sequences and tiles.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeLayerRangeRasterState {
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

pub type RasterDecodeLayerRangeState = DecodeLayerRangeRasterState;

impl DecodeLayerRangeRasterState {
    pub(crate) fn is_complete(&self) -> bool {
        self.next_layer_idx >= self.layer_count
    }

    pub(crate) fn next_layer_idx(&self) -> usize {
        self.next_layer_idx
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn token_count(&self) -> usize {
        self.token_count
    }

    pub(crate) fn next_token(&self) -> u32 {
        self.next_token
    }

    pub(crate) fn current_activation_ref(&self) -> &RasterActivationSequenceRef {
        &self.current_activation_ref
    }

    pub(crate) fn decode_input_ref(&self) -> &RasterActivationSequenceRef {
        &self.decode_input_ref
    }

    pub(crate) fn artifact_store_roots(&self) -> &RasterArtifactStoreRoots {
        &self.artifact_store_roots
    }

    pub(crate) fn effective_layer_caches(&self) -> Vec<DecodeLayerCacheSlot> {
        let mut caches = self.updated_layer_caches.clone();
        caches.extend(
            self.original_layer_caches
                .iter()
                .skip(self.updated_layer_caches.len())
                .cloned(),
        );
        caches
    }

    pub(crate) fn updated_layer_caches(&self) -> &[DecodeLayerCacheSlot] {
        &self.updated_layer_caches
    }

    pub(crate) fn completed_layer_output_sha256s(&self) -> &[String] {
        &self.completed_layer_output_sha256s
    }

    pub(crate) fn completed_layer_output_det_sha256s(&self) -> &[Option<String>] {
        &self.completed_layer_output_det_sha256s
    }

    pub(crate) fn from_parts(
        artifact_store_roots: RasterArtifactStoreRoots,
        decode_input_ref: RasterActivationSequenceRef,
        current_activation_ref: RasterActivationSequenceRef,
        next_token: u32,
        position: usize,
        token_count: usize,
        next_layer_idx: usize,
        layer_count: usize,
        original_layer_caches: Vec<DecodeLayerCacheSlot>,
        updated_layer_caches: Vec<DecodeLayerCacheSlot>,
        completed_layer_output_sha256s: Vec<String>,
        completed_layer_output_det_sha256s: Vec<Option<String>>,
        projection_rows_per_tile: usize,
        attention_kv_rows_per_tile: usize,
        output_source_prefix: String,
    ) -> Self {
        Self {
            artifact_store_roots,
            decode_input_ref,
            current_activation_ref,
            next_token,
            position,
            token_count,
            next_layer_idx,
            layer_count,
            original_layer_caches,
            updated_layer_caches,
            completed_layer_output_sha256s,
            completed_layer_output_det_sha256s,
            projection_rows_per_tile,
            attention_kv_rows_per_tile,
            output_source_prefix,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterDecodeLayerRangeInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub transformer_decode_state: TransformerDecodeState,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub decode_layer_range_source_root: String,
    pub output_source_prefix: String,
    pub raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeLayerRangeInputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub position: usize,
    pub token_count: usize,
    pub layer_caches: Vec<DecodeLayerCacheSlot>,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub decode_layer_range_source_root: String,
    pub output_source_prefix: String,
    pub raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterDecodeLayerRangeOutputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub transition_result: TransformerDecodeStepResult,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeTransitionFinalizeOutputRefs {
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

pub(in super::super) enum DecodeLayerWork {
    Complete(DecodeLayerRangeRasterState),
    Active(DecodeActiveLayerWork),
}

pub(in super::super) struct DecodeActiveLayerWork {
    pub(in super::super) decode_state: DecodeLayerRangeRasterState,
    pub(in super::super) layer_idx: usize,
    pub(in super::super) layer: GemmaDecodeLayerMetadata,
    pub(in super::super) cache_slot: DecodeLayerCacheSlot,
    pub(in super::super) donor_cache_slot: Option<DecodeLayerCacheSlot>,
    pub(in super::super) per_layer_input_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) attention_normed_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) q_projected_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) k_projected_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) v_projected_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) q_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) k_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) v_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) attention_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) attention_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) attention_output_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) xs_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) mlp_normed_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) mlp_gate_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) mlp_up_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) mlp_hidden_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) mlp_out_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) ple_gate_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) ple_projected_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) layer_output_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) updated_cache: Option<DecodeLayerCacheSlot>,
    pub(in super::super) output_prefix: String,
}

pub(in super::super) enum DecodePleInputWork {
    Skip(DecodeLayerWork),
    Active {
        work: DecodeActiveLayerWork,
        scalars: GemmaDecodePleScalars,
        norm_weights: Vec<Wgt>,
        embedded_ref: RasterActivationSequenceRef,
        projected_ref: Option<RasterActivationSequenceRef>,
        output_prefix: String,
    },
}

pub(in super::super) enum DecodeProjectionContinuation {
    PleInput(DecodePleInputWork),
    FinalLogits(DecodeTransitionFinalWork),
    LayerAttentionQuery(DecodeLayerWork),
    LayerAttentionKey(DecodeLayerWork),
    LayerAttentionValue(DecodeLayerWork),
    LayerAttentionOutput(DecodeLayerWork),
    LayerMlpGate(DecodeLayerWork),
    LayerMlpUp(DecodeLayerWork),
    LayerMlpDown(DecodeLayerWork),
    LayerPleGate(DecodeLayerWork),
    LayerPleOutput(DecodeLayerWork),
}

pub(in super::super) enum DecodeProjectionWork {
    Skip {
        continuation: DecodeProjectionContinuation,
    },
    Active {
        continuation: DecodeProjectionContinuation,
        state: DecodeRowProjectionArtifactState,
    },
}

pub(in super::super) enum DecodeKvCacheAppendWork {
    Skip(DecodeLayerWork),
    Active {
        work: DecodeLayerWork,
        state: DecodeKvCacheAppendArtifactState,
    },
}

pub(in super::super) enum DecodeAttentionWork {
    Skip(DecodeLayerWork),
    Active {
        work: DecodeLayerWork,
        state: DecodeAttentionArtifactState,
    },
}

pub(in super::super) struct DecodeTransitionFinalWork {
    pub(in super::super) artifact_store_roots: RasterArtifactStoreRoots,
    pub(in super::super) final_hidden_state_ref: RasterActivationSequenceRef,
    pub(in super::super) normalized_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) logits_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) output_source_prefix: String,
    pub(in super::super) projection_rows_per_tile: usize,
    pub(in super::super) layer_caches: Vec<DecodeLayerCacheSlot>,
    pub(in super::super) position: usize,
    pub(in super::super) token_count: usize,
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
