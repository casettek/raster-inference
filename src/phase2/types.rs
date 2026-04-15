use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

fn default_embedding_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixF32 {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingTable {
    pub rows: Vec<Vec<f32>>,
    #[serde(default = "default_embedding_scale")]
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActivationSequence {
    #[serde(skip_serializing, default)]
    pub activations: Vec<Vec<f32>>,
    pub activations_sha256: String,
}

pub type EmbeddedTokenSequence = ActivationSequence;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Phase2State {
    pub token_embeddings: ActivationSequence,
    pub final_hidden_states: ActivationSequence,
    pub prefill_logits: PrefillLogits,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4PleGlobalWeights {
    pub token_embeddings: Vec<MatrixF32>,
    pub model_projections: Vec<MatrixF32>,
    pub projection_norm_weight: Vec<f32>,
    pub embedding_scale: f32,
    pub projection_scalar: f32,
    pub input_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4PleLayerWeights {
    pub input_gate: MatrixF32,
    pub layer_projection: MatrixF32,
    pub post_input_norm_weight: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4AttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4LayerWeights {
    pub attention_kind: Gemma4AttentionKind,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_base: f32,
    pub partial_rotary_dim: usize,
    pub attention_k_eq_v: bool,
    pub q_proj: MatrixF32,
    pub k_proj: MatrixF32,
    pub v_proj: Option<MatrixF32>,
    pub o_proj: MatrixF32,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub pre_feedforward_layernorm_weight: Vec<f32>,
    pub post_feedforward_layernorm_weight: Vec<f32>,
    pub gate_proj: MatrixF32,
    pub up_proj: MatrixF32,
    pub down_proj: MatrixF32,
    pub ple: Option<Gemma4PleLayerWeights>,
    pub layer_scalar: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4PrefillPleInputs {
    pub per_layer_inputs: Vec<Option<Vec<Vec<f32>>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Gemma4LogitsProjection {
    UntiedLmHead(MatrixF32),
    TiedEmbedding(MatrixF32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrefillLogits {
    #[serde(skip_serializing, default)]
    pub logits: Vec<f32>,
    pub final_logits_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GemmaEmbeddingTensorSource {
    Single {
        weights_path: PathBuf,
        tensor_name: String,
        hidden_size: usize,
        scale: f32,
    },
    Indexed {
        root_dir: PathBuf,
        weight_map: HashMap<String, String>,
        tensor_name: String,
        hidden_size: usize,
        scale: f32,
    },
}

impl GemmaEmbeddingTensorSource {
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Single { hidden_size, .. } | Self::Indexed { hidden_size, .. } => *hidden_size,
        }
    }

    pub fn scale(&self) -> f32 {
        match self {
            Self::Single { scale, .. } | Self::Indexed { scale, .. } => *scale,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Phase2Model {
    pub embedding_table: Option<EmbeddingTable>,
    pub embedding_source: Option<GemmaEmbeddingTensorSource>,
    pub layers: Vec<Gemma4LayerWeights>,
    pub ple_global: Option<Gemma4PleGlobalWeights>,
    pub final_norm_weight: Vec<f32>,
    pub logits_projection: Gemma4LogitsProjection,
    pub final_logit_softcapping: Option<f32>,
    pub rms_norm_eps: f32,
}
