use crate::shared::artifacts::raster_artifact_store::{
    RasterActivationSequenceArtifactRef, RasterRoutineOutput,
};
use crate::RasterSizingControls;

// Types and constants used by the raster sequences and tiles.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleRasterState {
    pub(in super::super) source_id: String,
    pub(in super::super) token_ids_source_name: String,
    pub(in super::super) token_count: usize,
    pub(in super::super) input_activations_ref: Option<RasterActivationSequenceArtifactRef>,
    pub(in super::super) next_layer_idx: usize,
    pub(in super::super) layer_count: usize,
    pub(in super::super) per_layer_inputs: Vec<Option<RasterActivationSequenceArtifactRef>>,
    pub(in super::super) has_ple_global: bool,
    pub(in super::super) raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillPleInputRoots {
    pub(in super::super) source_id: String,
    pub(in super::super) token_ids_source_name: String,
    pub(in super::super) token_count: usize,
    pub(in super::super) input_activations_ref: Option<RasterActivationSequenceArtifactRef>,
    pub(in super::super) layer_count: usize,
    pub(in super::super) has_ple_global: bool,
    pub(in super::super) raster_sizing: RasterSizingControls,
}

pub type RasterPrefillPleOutput = RasterRoutineOutput<Option<String>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) enum PrefillPleLayerStep {
    Complete {
        ple_state: PrefillPleRasterState,
    },
    Skip {
        ple_state: PrefillPleRasterState,
        layer_idx: usize,
    },
    Compute(PrefillPleLayerComputeState),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) struct PrefillPleLayerComputeState {
    pub(in super::super) ple_state: PrefillPleRasterState,
    pub(in super::super) layer_idx: usize,
    pub(in super::super) input_activations_ref: RasterActivationSequenceArtifactRef,
    pub(in super::super) ple_width: usize,
    pub(in super::super) projection_rows: usize,
    pub(in super::super) embedding_scale_bits: Option<i32>,
    pub(in super::super) projection_scalar_bits: Option<i32>,
    pub(in super::super) input_scale_bits: Option<i32>,
    pub(in super::super) rms_norm_eps_bits: Option<i64>,
    pub(in super::super) norm_weight_bits: Option<Vec<i32>>,
    pub(in super::super) embedded_ref: Option<RasterActivationSequenceArtifactRef>,
    pub(in super::super) projected_ref: Option<RasterActivationSequenceArtifactRef>,
    pub(in super::super) combined_ref: Option<RasterActivationSequenceArtifactRef>,
    pub(in super::super) layer_input_ref: Option<RasterActivationSequenceArtifactRef>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleTokenEmbeddingState {
    pub(in super::super) token_ids_source_name: String,
    pub(in super::super) layer_idx: usize,
    pub(in super::super) scale_bits: i32,
    pub(in super::super) next_token_idx: usize,
    pub(in super::super) token_count: usize,
    pub(in super::super) row_width: usize,
    pub(in super::super) output_source_name: String,
}

impl PrefillPleTokenEmbeddingState {
    pub(in super::super) fn is_complete(&self) -> bool {
        self.next_token_idx >= self.token_count
    }
}

pub(in super::super) type PrefillPleLayerTokenEmbeddingStep =
    PrefillPleLayerSubstep<PrefillPleTokenEmbeddingState>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleSequenceProjectionState {
    pub(in super::super) input_ref: RasterActivationSequenceArtifactRef,
    pub(in super::super) output_source_name: String,
    pub(in super::super) current_row_bits: Vec<i32>,
    pub(in super::super) next_token_idx: usize,
    pub(in super::super) next_projection_row_idx: usize,
    pub(in super::super) token_count: usize,
    pub(in super::super) input_width: usize,
    pub(in super::super) projection_rows: usize,
    pub(in super::super) rows_per_tile: usize,
}

impl PrefillPleSequenceProjectionState {
    pub(in super::super) fn is_complete(&self) -> bool {
        self.next_token_idx >= self.token_count
    }
}

pub(in super::super) type PrefillPleLayerProjectionStep =
    PrefillPleLayerSubstep<PrefillPleSequenceProjectionState>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) enum PrefillPleSequenceUnaryOp {
    RmsNorm {
        norm_weight_bits: Vec<i32>,
        eps_bits: i64,
    },
    Scale {
        scalar_bits: i32,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleSequenceUnaryState {
    pub(in super::super) input_ref: RasterActivationSequenceArtifactRef,
    pub(in super::super) output_source_name: String,
    pub(in super::super) op: PrefillPleSequenceUnaryOp,
    pub(in super::super) next_row_idx: usize,
    pub(in super::super) row_count: usize,
    pub(in super::super) width: usize,
    pub(in super::super) rows_per_tile: usize,
}

impl PrefillPleSequenceUnaryState {
    pub(in super::super) fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

pub(in super::super) type PrefillPleLayerUnaryStep =
    PrefillPleLayerSubstep<PrefillPleSequenceUnaryState>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleSequenceBinaryState {
    pub(in super::super) lhs_ref: RasterActivationSequenceArtifactRef,
    pub(in super::super) rhs_ref: RasterActivationSequenceArtifactRef,
    pub(in super::super) output_source_name: String,
    pub(in super::super) next_row_idx: usize,
    pub(in super::super) row_count: usize,
    pub(in super::super) width: usize,
    pub(in super::super) rows_per_tile: usize,
}

impl PrefillPleSequenceBinaryState {
    pub(in super::super) fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

pub(in super::super) type PrefillPleLayerBinaryStep =
    PrefillPleLayerSubstep<PrefillPleSequenceBinaryState>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(in super::super) enum PrefillPleLayerSubstep<T> {
    Noop(PrefillPleLayerStep),
    Compute {
        compute_state: PrefillPleLayerComputeState,
        operation_state: T,
    },
}
