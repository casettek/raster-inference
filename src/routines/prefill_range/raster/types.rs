use crate::shared::raster_contracts::prefill_layer::{
    GemmaPrefillLayerMatrixKind, GemmaPrefillLayerMetadata, GemmaPrefillLayerScalars,
};
use crate::shared::raster_kernels::transformer::{
    RasterAttentionArtifactRowState, RasterCombineHeadsArtifactState, RasterHeadUnaryArtifactState,
    RasterKvCacheBuildArtifactState, RasterReshapeHeadsArtifactState,
    RasterSequenceBinaryArtifactState, RasterSequenceProjectionArtifactState,
    RasterSequenceUnaryArtifactState,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    RasterActivationSequenceRef, RasterAttentionHeadsRef, RasterKvCacheRef,
};

// Types and constants used by the raster sequences and tiles.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerRasterState {
    pub(crate) current_activations_ref: RasterActivationSequenceRef,
    pub(crate) next_layer_idx: usize,
    pub(crate) layer_count: usize,
    pub(crate) layer_caches: Vec<PrefillLayerCacheSlot>,
    pub(crate) per_layer_inputs: Vec<Option<RasterActivationSequenceRef>>,
    pub(crate) completed_layer_output_sha256s: Vec<String>,
    pub(crate) completed_layer_output_det_sha256s: Vec<Option<String>>,
    pub(crate) projection_rows_per_tile: usize,
    pub(crate) attention_kv_rows_per_tile: usize,
    pub(crate) sequence_rows_per_tile: usize,
    pub(crate) head_rows_per_tile: usize,
    pub(crate) prefill_token_range_width: usize,
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
pub(crate) enum PrefillLayerStep {
    Complete {
        layer_state: PrefillLayerRasterState,
    },
    Compute {
        layer_state: PrefillLayerRasterState,
        context: PrefillLayerContext,
    },
    Computed {
        layer_state: PrefillLayerRasterState,
        layer_idx: usize,
        layer_output_ref: RasterActivationSequenceRef,
        layer_cache: PrefillLayerCacheSlot,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerContext {
    pub(in super::super) layer_idx: usize,
    pub(in super::super) layer: GemmaPrefillLayerMetadata,
    pub(in super::super) donor_cache: Option<PrefillLayerCacheSlot>,
    pub(in super::super) per_layer_input: Option<RasterActivationSequenceRef>,
}

pub(in super::super) enum PrefillLayerArtifactWork {
    Passthrough(PrefillLayerStep),
    Active(Box<PrefillLayerActiveWork>),
}

pub(in super::super) struct PrefillLayerActiveWork {
    pub(in super::super) layer_state: PrefillLayerRasterState,
    pub(in super::super) layer_idx: usize,
    pub(in super::super) layer: GemmaPrefillLayerMetadata,
    pub(in super::super) donor_cache: Option<PrefillLayerCacheSlot>,
    pub(in super::super) per_layer_input_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) projection_rows_per_tile: usize,
    pub(in super::super) attention_kv_rows_per_tile: usize,
    pub(in super::super) sequence_rows_per_tile: usize,
    pub(in super::super) head_rows_per_tile: usize,
    pub(in super::super) input_ref: RasterActivationSequenceRef,
    pub(in super::super) scalars: GemmaPrefillLayerScalars,
    pub(in super::super) normed_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) q_projected_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) k_projected_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) v_projected_ref: Option<RasterActivationSequenceRef>,
    pub(in super::super) q_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) k_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) v_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) layer_cache: Option<PrefillLayerCacheSlot>,
    pub(in super::super) donor_cache_ref: Option<RasterKvCacheRef>,
    pub(in super::super) attention_heads_ref: Option<RasterAttentionHeadsRef>,
    pub(in super::super) attention_sequence_ref: Option<RasterActivationSequenceRef>,
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
}

pub(in super::super) enum PrefillLayerSequenceUnaryWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterSequenceUnaryArtifactState,
    },
}

pub(in super::super) enum PrefillLayerSequenceProjectionWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterSequenceProjectionArtifactState,
        layer_idx: usize,
        matrix: GemmaPrefillLayerMatrixKind,
    },
}

pub(in super::super) enum PrefillLayerReshapeHeadsWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterReshapeHeadsArtifactState,
    },
}

pub(in super::super) enum PrefillLayerHeadUnaryWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterHeadUnaryArtifactState,
    },
}

pub(in super::super) enum PrefillLayerKvCacheWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterKvCacheBuildArtifactState,
    },
}

pub(in super::super) enum PrefillLayerAttentionWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterAttentionArtifactRowState,
    },
}

pub(in super::super) enum PrefillLayerCombineHeadsWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterCombineHeadsArtifactState,
    },
}

pub(in super::super) enum PrefillLayerSequenceBinaryWork {
    Skip(PrefillLayerArtifactWork),
    Active {
        work: PrefillLayerArtifactWork,
        state: RasterSequenceBinaryArtifactState,
    },
}
