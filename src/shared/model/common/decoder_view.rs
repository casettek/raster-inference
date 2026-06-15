use std::sync::Arc;

use crate::shared::model::common::architecture::{ArchitectureSpec, AttentionKind, RopeSpec};
use crate::shared::model::transformer::{DetNumMatrix, DetNumTensorSliceSource, MatrixF32};
use crate::shared::numerics::det_num::{Acc, Act, Wgt};

#[derive(Debug, Clone)]
pub struct DecoderModelView<'a> {
    pub spec: ArchitectureSpec,
    pub embeddings: EmbeddingView<'a>,
    pub layers: Vec<DecoderLayerView<'a>>,
    pub ple: Option<PleGlobalView<'a>>,
    pub final_norm: NormView<'a>,
    pub lm_head: ProjectionView<'a>,
    pub rms_norm_eps: Option<Acc>,
    pub has_final_logit_softcapping: bool,
    pub final_logit_softcapping: Option<Act>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecoderLayerView<'a> {
    pub layer_idx: usize,
    pub hidden_size: usize,
    pub input_norm: NormView<'a>,
    pub attention: AttentionView<'a>,
    pub post_attention_norm: NormView<'a>,
    pub pre_feedforward_norm: NormView<'a>,
    pub post_feedforward_norm: NormView<'a>,
    pub ffn: FfnKind<'a>,
    pub ple: Option<PleLayerView<'a>>,
    pub rms_norm_eps: Option<Acc>,
    pub rope_base: Option<Acc>,
    pub has_layer_scalar: bool,
    pub layer_scalar: Option<Act>,
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionView<'a> {
    pub kind: AttentionKind,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub cache_sliding_window: Option<usize>,
    pub rope: RopeSpec,
    pub kv_shared_layer_index: Option<usize>,
    pub attention_k_eq_v: bool,
    pub q_proj: WeightMatrixView<'a>,
    pub k_proj: WeightMatrixView<'a>,
    pub v_proj: Option<WeightMatrixView<'a>>,
    pub o_proj: WeightMatrixView<'a>,
    pub q_norm: NormView<'a>,
    pub k_norm: NormView<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum FfnKind<'a> {
    Dense(DenseFfnView<'a>),
    Moe,
}

#[derive(Debug, Clone, Copy)]
pub struct DenseFfnView<'a> {
    pub gate_proj: WeightMatrixView<'a>,
    pub up_proj: WeightMatrixView<'a>,
    pub down_proj: WeightMatrixView<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct PleLayerView<'a> {
    pub input_gate: WeightMatrixView<'a>,
    pub layer_projection: WeightMatrixView<'a>,
    pub post_input_norm: NormView<'a>,
}

#[derive(Debug, Clone)]
pub struct PleGlobalView<'a> {
    pub token_embeddings: Vec<WeightMatrixView<'a>>,
    pub model_projections: Vec<WeightMatrixView<'a>>,
    pub projection_norm: NormView<'a>,
    pub embedding_scale: Option<Act>,
    pub projection_scalar: Option<Act>,
    pub input_scale: Option<Act>,
    pub rms_norm_eps: Option<Acc>,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingView<'a> {
    pub weights: Option<WeightMatrixView<'a>>,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectionView<'a> {
    pub kind: ProjectionKind,
    pub weights: Option<WeightMatrixView<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    UntiedLmHead,
    TiedEmbedding,
}

#[derive(Debug, Clone, Copy)]
pub struct NormView<'a> {
    pub weights: &'a [f32],
    pub det_weights: Option<&'a [Wgt]>,
}

#[derive(Debug, Clone, Copy)]
pub enum WeightMatrixView<'a> {
    Materialized(&'a MatrixF32),
    DetNumSlice(&'a DetNumTensorSliceSource),
    DetNumMatrix(&'a Arc<DetNumMatrix>),
    ShapeOnly(MatrixShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    pub rows: usize,
    pub cols: usize,
}

impl WeightMatrixView<'_> {
    pub fn shape(&self) -> MatrixShape {
        match self {
            Self::Materialized(matrix) => MatrixShape {
                rows: matrix.rows,
                cols: matrix.cols,
            },
            Self::DetNumSlice(source) => MatrixShape {
                rows: source.row_count,
                cols: source.col_count,
            },
            Self::DetNumMatrix(matrix) => MatrixShape {
                rows: matrix.rows,
                cols: matrix.cols,
            },
            Self::ShapeOnly(shape) => *shape,
        }
    }

    pub fn det_num_slice(&self) -> Option<&DetNumTensorSliceSource> {
        match self {
            Self::DetNumSlice(source) => Some(source),
            _ => None,
        }
    }

    pub fn det_num_matrix(&self) -> Option<&Arc<DetNumMatrix>> {
        match self {
            Self::DetNumMatrix(matrix) => Some(matrix),
            _ => None,
        }
    }
}
