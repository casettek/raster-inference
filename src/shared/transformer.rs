use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use memmap2::Mmap;
use safetensors::Dtype;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetNumMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<i32>,
}

#[derive(Debug, Clone)]
pub enum Gemma4LayerMatrixSource {
    Materialized(Arc<MatrixF32>),
    Lazy {
        source: GemmaTensorSliceSource,
        cache: Arc<Mutex<Option<Arc<MatrixF32>>>>,
    },
    DetNumLazy {
        source: DetNumTensorSliceSource,
        cache: Arc<Mutex<Option<Arc<MatrixF32>>>>,
        det_cache: Arc<Mutex<Option<Arc<DetNumMatrix>>>>,
    },
}

impl Gemma4LayerMatrixSource {
    pub fn from_source(source: GemmaTensorSliceSource) -> Self {
        Self::Lazy {
            source,
            cache: Arc::new(Mutex::new(None)),
        }
    }

    pub fn from_det_num_source(source: DetNumTensorSliceSource) -> Self {
        Self::DetNumLazy {
            source,
            cache: Arc::new(Mutex::new(None)),
            det_cache: Arc::new(Mutex::new(None)),
        }
    }
}

impl From<MatrixF32> for Gemma4LayerMatrixSource {
    fn from(value: MatrixF32) -> Self {
        Self::Materialized(Arc::new(value))
    }
}

impl PartialEq for Gemma4LayerMatrixSource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Materialized(lhs), Self::Materialized(rhs)) => lhs.as_ref() == rhs.as_ref(),
            (Self::Lazy { source: lhs, .. }, Self::Lazy { source: rhs, .. }) => lhs == rhs,
            (Self::DetNumLazy { source: lhs, .. }, Self::DetNumLazy { source: rhs, .. }) => {
                lhs == rhs
            }
            _ => false,
        }
    }
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
pub struct TransformerStateTransitionState {
    pub activation_states: Vec<ActivationSequence>,
    #[serde(skip_serializing, default)]
    pub prefill_logits: PrefillLogits,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LayerKvCache {
    pub keys: Vec<VecDeque<Vec<f32>>>,
    pub values: Vec<VecDeque<Vec<f32>>>,
}

impl LayerKvCache {
    pub fn new(num_kv_heads: usize) -> Self {
        Self {
            keys: vec![VecDeque::new(); num_kv_heads],
            values: vec![VecDeque::new(); num_kv_heads],
        }
    }

    pub fn current_len(&self) -> usize {
        self.keys.first().map(VecDeque::len).unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TransformerDecodeState {
    pub layer_caches: Vec<LayerKvCache>,
    pub position: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerPrefillResult {
    pub transformer_state: TransformerStateTransitionState,
    pub transformer_decode_state: TransformerDecodeState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerDecodeStepResult {
    pub transformer_decode_state: TransformerDecodeState,
    pub activation_state: ActivationSequence,
    pub prefill_logits: PrefillLogits,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GemmaTensorSliceSource {
    pub weights_path: PathBuf,
    pub dtype: Dtype,
    pub total_rows: usize,
    pub total_cols: usize,
    pub data_offset: usize,
    pub row_offset: usize,
    pub row_count: usize,
    pub col_offset: usize,
    pub col_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetNumTensorSliceSource {
    pub weights_path: PathBuf,
    pub total_rows: usize,
    pub total_cols: usize,
    pub data_offset: usize,
    pub row_offset: usize,
    pub row_count: usize,
    pub col_offset: usize,
    pub col_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Gemma4PleMatrixSource {
    Materialized(MatrixF32),
    Lazy(GemmaTensorSliceSource),
    DetNumLazy(DetNumTensorSliceSource),
}

#[derive(Debug, Clone)]
pub struct Gemma4PleGlobalWeights {
    pub(crate) token_embeddings: Vec<Gemma4PleMatrixSource>,
    pub(crate) model_projections: Vec<Gemma4PleMatrixSource>,
    pub projection_norm_weight: Vec<f32>,
    pub embedding_scale: f32,
    pub projection_scalar: f32,
    pub input_scale: f32,
    pub(crate) mmap_cache: Arc<Mutex<HashMap<PathBuf, Arc<Mmap>>>>,
    pub(crate) token_row_cache: Arc<Mutex<HashMap<(usize, usize), Vec<f32>>>>,
    pub(crate) model_projection_cache: Arc<Mutex<HashMap<usize, MatrixF32>>>,
    pub(crate) model_projection_det_cache: Arc<Mutex<HashMap<usize, Arc<DetNumMatrix>>>>,
}

impl Gemma4PleGlobalWeights {
    pub fn from_materialized(
        token_embeddings: Vec<MatrixF32>,
        model_projections: Vec<MatrixF32>,
        projection_norm_weight: Vec<f32>,
        embedding_scale: f32,
        projection_scalar: f32,
        input_scale: f32,
    ) -> Self {
        Self::new(
            token_embeddings
                .into_iter()
                .map(Gemma4PleMatrixSource::Materialized)
                .collect(),
            model_projections
                .into_iter()
                .map(Gemma4PleMatrixSource::Materialized)
                .collect(),
            projection_norm_weight,
            embedding_scale,
            projection_scalar,
            input_scale,
        )
    }

    pub(crate) fn from_sources(
        token_embeddings: Vec<GemmaTensorSliceSource>,
        model_projections: Vec<GemmaTensorSliceSource>,
        projection_norm_weight: Vec<f32>,
        embedding_scale: f32,
        projection_scalar: f32,
        input_scale: f32,
    ) -> Self {
        Self::new(
            token_embeddings
                .into_iter()
                .map(Gemma4PleMatrixSource::Lazy)
                .collect(),
            model_projections
                .into_iter()
                .map(Gemma4PleMatrixSource::Lazy)
                .collect(),
            projection_norm_weight,
            embedding_scale,
            projection_scalar,
            input_scale,
        )
    }

    pub(crate) fn from_det_num_sources(
        token_embeddings: Vec<DetNumTensorSliceSource>,
        model_projections: Vec<DetNumTensorSliceSource>,
        projection_norm_weight: Vec<f32>,
        embedding_scale: f32,
        projection_scalar: f32,
        input_scale: f32,
    ) -> Self {
        Self::new(
            token_embeddings
                .into_iter()
                .map(Gemma4PleMatrixSource::DetNumLazy)
                .collect(),
            model_projections
                .into_iter()
                .map(Gemma4PleMatrixSource::DetNumLazy)
                .collect(),
            projection_norm_weight,
            embedding_scale,
            projection_scalar,
            input_scale,
        )
    }

    fn new(
        token_embeddings: Vec<Gemma4PleMatrixSource>,
        model_projections: Vec<Gemma4PleMatrixSource>,
        projection_norm_weight: Vec<f32>,
        embedding_scale: f32,
        projection_scalar: f32,
        input_scale: f32,
    ) -> Self {
        Self {
            token_embeddings,
            model_projections,
            projection_norm_weight,
            embedding_scale,
            projection_scalar,
            input_scale,
            mmap_cache: Arc::new(Mutex::new(HashMap::new())),
            token_row_cache: Arc::new(Mutex::new(HashMap::new())),
            model_projection_cache: Arc::new(Mutex::new(HashMap::new())),
            model_projection_det_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn token_embedding_layer_count(&self) -> usize {
        self.token_embeddings.len()
    }

    pub fn model_projection_layer_count(&self) -> usize {
        self.model_projections.len()
    }
}

impl PartialEq for Gemma4PleGlobalWeights {
    fn eq(&self, other: &Self) -> bool {
        self.token_embeddings == other.token_embeddings
            && self.model_projections == other.model_projections
            && self.projection_norm_weight == other.projection_norm_weight
            && self.embedding_scale == other.embedding_scale
            && self.projection_scalar == other.projection_scalar
            && self.input_scale == other.input_scale
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4PleLayerWeights {
    pub input_gate: Gemma4LayerMatrixSource,
    pub layer_projection: Gemma4LayerMatrixSource,
    pub post_input_norm_weight: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGemma4PleLayerWeights {
    pub input_gate: Arc<MatrixF32>,
    pub layer_projection: Arc<MatrixF32>,
    pub input_gate_det: Option<Arc<DetNumMatrix>>,
    pub layer_projection_det: Option<Arc<DetNumMatrix>>,
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
    pub cache_sliding_window: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_base: f32,
    pub partial_rotary_dim: usize,
    pub rope_freq_base_dim: usize,
    pub kv_shared_layer_index: Option<usize>,
    pub attention_k_eq_v: bool,
    pub q_proj: Gemma4LayerMatrixSource,
    pub k_proj: Gemma4LayerMatrixSource,
    pub v_proj: Option<Gemma4LayerMatrixSource>,
    pub o_proj: Gemma4LayerMatrixSource,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub pre_feedforward_layernorm_weight: Vec<f32>,
    pub post_feedforward_layernorm_weight: Vec<f32>,
    pub gate_proj: Gemma4LayerMatrixSource,
    pub up_proj: Gemma4LayerMatrixSource,
    pub down_proj: Gemma4LayerMatrixSource,
    pub ple: Option<Gemma4PleLayerWeights>,
    pub layer_scalar: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGemma4LayerWeights {
    pub attention_kind: Gemma4AttentionKind,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub cache_sliding_window: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_base: f32,
    pub partial_rotary_dim: usize,
    pub rope_freq_base_dim: usize,
    pub kv_shared_layer_index: Option<usize>,
    pub attention_k_eq_v: bool,
    pub q_proj: Arc<MatrixF32>,
    pub k_proj: Arc<MatrixF32>,
    pub v_proj: Option<Arc<MatrixF32>>,
    pub o_proj: Arc<MatrixF32>,
    pub q_proj_det: Option<Arc<DetNumMatrix>>,
    pub k_proj_det: Option<Arc<DetNumMatrix>>,
    pub v_proj_det: Option<Arc<DetNumMatrix>>,
    pub o_proj_det: Option<Arc<DetNumMatrix>>,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub pre_feedforward_layernorm_weight: Vec<f32>,
    pub post_feedforward_layernorm_weight: Vec<f32>,
    pub gate_proj: Arc<MatrixF32>,
    pub up_proj: Arc<MatrixF32>,
    pub down_proj: Arc<MatrixF32>,
    pub gate_proj_det: Option<Arc<DetNumMatrix>>,
    pub up_proj_det: Option<Arc<DetNumMatrix>>,
    pub down_proj_det: Option<Arc<DetNumMatrix>>,
    pub ple: Option<ResolvedGemma4PleLayerWeights>,
    pub layer_scalar: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4PrefillPleInputs {
    pub per_layer_inputs: Vec<Option<Vec<Vec<f32>>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Gemma4LogitsProjection {
    UntiedLmHead {
        weight: MatrixF32,
        det_weight: Option<Arc<DetNumMatrix>>,
    },
    TiedEmbedding(MatrixF32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PrefillLogits {
    #[serde(skip_serializing, default)]
    pub logits: Vec<f32>,
    pub final_logits_sha256: String,
}

#[derive(Debug, Clone)]
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
    Deterministic {
        source: DetNumTensorSliceSource,
        scale: f32,
        det_cache: Arc<Mutex<Option<Arc<DetNumMatrix>>>>,
    },
}

impl GemmaEmbeddingTensorSource {
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Single { hidden_size, .. } | Self::Indexed { hidden_size, .. } => *hidden_size,
            Self::Deterministic { source, .. } => source.col_count,
        }
    }

    pub fn scale(&self) -> f32 {
        match self {
            Self::Single { scale, .. }
            | Self::Indexed { scale, .. }
            | Self::Deterministic { scale, .. } => *scale,
        }
    }
}

impl PartialEq for GemmaEmbeddingTensorSource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Single {
                    weights_path: lhs_weights_path,
                    tensor_name: lhs_tensor_name,
                    hidden_size: lhs_hidden_size,
                    scale: lhs_scale,
                },
                Self::Single {
                    weights_path: rhs_weights_path,
                    tensor_name: rhs_tensor_name,
                    hidden_size: rhs_hidden_size,
                    scale: rhs_scale,
                },
            ) => {
                lhs_weights_path == rhs_weights_path
                    && lhs_tensor_name == rhs_tensor_name
                    && lhs_hidden_size == rhs_hidden_size
                    && lhs_scale == rhs_scale
            }
            (
                Self::Indexed {
                    root_dir: lhs_root_dir,
                    weight_map: lhs_weight_map,
                    tensor_name: lhs_tensor_name,
                    hidden_size: lhs_hidden_size,
                    scale: lhs_scale,
                },
                Self::Indexed {
                    root_dir: rhs_root_dir,
                    weight_map: rhs_weight_map,
                    tensor_name: rhs_tensor_name,
                    hidden_size: rhs_hidden_size,
                    scale: rhs_scale,
                },
            ) => {
                lhs_root_dir == rhs_root_dir
                    && lhs_weight_map == rhs_weight_map
                    && lhs_tensor_name == rhs_tensor_name
                    && lhs_hidden_size == rhs_hidden_size
                    && lhs_scale == rhs_scale
            }
            (
                Self::Deterministic {
                    source: lhs_source,
                    scale: lhs_scale,
                    ..
                },
                Self::Deterministic {
                    source: rhs_source,
                    scale: rhs_scale,
                    ..
                },
            ) => lhs_source == rhs_source && lhs_scale == rhs_scale,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TransformerModel {
    pub embedding_table: Option<EmbeddingTable>,
    pub embedding_source: Option<GemmaEmbeddingTensorSource>,
    pub layers: Vec<Gemma4LayerWeights>,
    pub ple_global: Option<Gemma4PleGlobalWeights>,
    pub final_norm_weight: Vec<f32>,
    pub logits_projection: Gemma4LogitsProjection,
    pub final_logit_softcapping: Option<f32>,
    pub rms_norm_eps: f32,
}
