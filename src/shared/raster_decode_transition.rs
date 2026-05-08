use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use crate::shared::artifact_io::AuthRead;
use crate::shared::det_num::{scale_act, Acc, Act, Wgt};
use crate::shared::raster_transformer_kernels::det_num_tensor_slice_row_wgts;
use crate::shared::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
    Gemma4PleMatrixSource, Gemma4TransformerModel, GemmaEmbeddingTensorSource,
};

#[derive(Debug, Clone)]
pub struct AuthenticatedGemmaDecodeTransitionSource {
    identifier: String,
    metadata: GemmaDecodeTransitionMetadata,
    embedding: DetNumTensorSliceSource,
    embedding_scale: Act,
    layers: Vec<GemmaDecodeLayerMetadata>,
    backing_layers: Vec<GemmaDecodeLayerBacking>,
    ple: Option<GemmaDecodePleBacking>,
    final_norm_weights: Vec<Wgt>,
    final_scalars: GemmaDecodeFinalScalars,
    projection: GemmaDecodeProjectionBacking,
}

#[derive(Debug, Clone)]
struct GemmaDecodeLayerBacking {
    matrices: GemmaDecodeLayerMatrices,
    norms: GemmaDecodeLayerNorms,
    scalars: GemmaDecodeLayerScalars,
}

#[derive(Debug, Clone)]
struct GemmaDecodeLayerMatrices {
    q_proj: Gemma4LayerMatrixSource,
    k_proj: Gemma4LayerMatrixSource,
    v_proj: Option<Gemma4LayerMatrixSource>,
    o_proj: Gemma4LayerMatrixSource,
    gate_proj: Gemma4LayerMatrixSource,
    up_proj: Gemma4LayerMatrixSource,
    down_proj: Gemma4LayerMatrixSource,
    ple: Option<GemmaDecodePleLayerMatrices>,
}

#[derive(Debug, Clone)]
struct GemmaDecodePleLayerMatrices {
    input_gate: Gemma4LayerMatrixSource,
    layer_projection: Gemma4LayerMatrixSource,
}

#[derive(Debug, Clone)]
struct GemmaDecodeLayerNorms {
    q_norm: Vec<Wgt>,
    k_norm: Vec<Wgt>,
    input_layernorm: Vec<Wgt>,
    post_attention_layernorm: Vec<Wgt>,
    pre_feedforward_layernorm: Vec<Wgt>,
    post_feedforward_layernorm: Vec<Wgt>,
    ple_post_input_norm: Option<Vec<Wgt>>,
}

#[derive(Debug, Clone)]
struct GemmaDecodePleBacking {
    token_embeddings: Vec<Gemma4PleMatrixSource>,
    model_projections: Vec<Gemma4PleMatrixSource>,
    projection_norm_weights: Vec<Wgt>,
    scalars: GemmaDecodePleScalars,
}

#[derive(Debug, Clone)]
enum GemmaDecodeProjectionBacking {
    Matrix(Arc<DetNumMatrix>),
    TensorSlice(DetNumTensorSliceSource),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaDecodeTransitionMetadata {
    pub source_id: String,
    pub layer_count: usize,
    pub embedding_vocab_size: usize,
    pub embedding_width: usize,
    pub final_norm_width: usize,
    pub projection_rows: usize,
    pub projection_cols: usize,
    pub projection_kind: GemmaDecodeProjectionKind,
    pub has_ple_global: bool,
    pub has_final_logit_softcapping: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaDecodeLayerMetadata {
    pub layer_idx: usize,
    pub attention_kind: GemmaDecodeAttentionKind,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub cache_sliding_window: Option<usize>,
    pub partial_rotary_dim: usize,
    pub rope_freq_base_dim: usize,
    pub kv_shared_layer_index: Option<usize>,
    pub attention_k_eq_v: bool,
    pub has_v_proj: bool,
    pub has_ple: bool,
    pub has_layer_scalar: bool,
    pub q_proj_shape: GemmaDecodeMatrixShape,
    pub k_proj_shape: GemmaDecodeMatrixShape,
    pub v_proj_shape: Option<GemmaDecodeMatrixShape>,
    pub o_proj_shape: GemmaDecodeMatrixShape,
    pub gate_proj_shape: GemmaDecodeMatrixShape,
    pub up_proj_shape: GemmaDecodeMatrixShape,
    pub down_proj_shape: GemmaDecodeMatrixShape,
    pub ple_input_gate_shape: Option<GemmaDecodeMatrixShape>,
    pub ple_layer_projection_shape: Option<GemmaDecodeMatrixShape>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum GemmaDecodeAttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum GemmaDecodeProjectionKind {
    UntiedLmHead,
    TiedEmbedding,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaDecodeMatrixShape {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeLayerScalars {
    pub rms_norm_eps: Acc,
    pub rope_base: Option<Acc>,
    pub layer_scalar: Option<Act>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodePleScalars {
    pub embedding_scale: Act,
    pub projection_scalar: Act,
    pub input_scale: Act,
    pub rms_norm_eps: Acc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeFinalScalars {
    pub rms_norm_eps: Acc,
    pub final_logit_softcapping: Option<Act>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeTransitionMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeEmbeddingRowRequest {
    pub token_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeLayerMetadataRequest {
    pub layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeLayerScalarsRequest {
    pub layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeLayerMatrixRowRequest {
    pub layer_idx: usize,
    pub matrix: GemmaDecodeLayerMatrixKind,
    pub row_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeLayerNormWeightsRequest {
    pub layer_idx: usize,
    pub norm: GemmaDecodeLayerNormKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodePleTokenEmbeddingRowRequest {
    pub layer_idx: usize,
    pub token_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodePleModelProjectionRowRequest {
    pub layer_idx: usize,
    pub row_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodePleProjectionNormWeightsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodePleScalarsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeFinalNormWeightsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeFinalScalarsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeProjectionRowRequest {
    pub row_idx: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum GemmaDecodeLayerMatrixKind {
    Query,
    Key,
    Value,
    Output,
    Gate,
    Up,
    Down,
    PleInputGate,
    PleLayerProjection,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum GemmaDecodeLayerNormKind {
    Query,
    Key,
    InputLayer,
    PostAttention,
    PreFeedForward,
    PostFeedForward,
    PlePostInput,
}

impl AuthenticatedGemmaDecodeTransitionSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        if model.provenance != Gemma4ModelProvenance::DetNumWgt {
            bail!(
                "deterministic raster decode transition source requires a model loaded from a .detwgt artifact"
            );
        }

        let (embedding, embedding_scale) = canonical_embedding(model)?;
        let mut layers = Vec::with_capacity(model.layers.len());
        let mut backing_layers = Vec::with_capacity(model.layers.len());
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let (metadata, backing) = build_layer(layer_idx, layer)?;
            layers.push(metadata);
            backing_layers.push(backing);
        }
        let ple = model
            .ple_global
            .as_ref()
            .map(|ple| build_ple(ple, model.rms_norm_eps_det))
            .transpose()?;
        if ple.is_none() && model.layers.iter().any(|layer| layer.ple.is_some()) {
            bail!("Gemma decode transition source has PLE layers but no global PLE weights");
        }
        if let Some(ple) = ple.as_ref() {
            validate_ple_shapes(&layers, ple)?;
        }

        let final_norm_weights = canonical_final_norm_weights(&model.final_norm_weight_det)?;
        let final_logit_softcapping = if model.final_logit_softcapping.is_some() {
            Some(model.final_logit_softcapping_det.ok_or_else(|| {
                anyhow!(
                    "deterministic raster decode transition requires canonical final logit softcap"
                )
            })?)
        } else {
            None
        };
        let final_scalars = GemmaDecodeFinalScalars {
            rms_norm_eps: model.rms_norm_eps_det.ok_or_else(|| {
                anyhow!("deterministic raster decode transition requires canonical RMSNorm epsilon")
            })?,
            final_logit_softcapping,
        };
        let (projection_kind, projection_rows, projection_cols, projection) =
            canonical_projection(model)?;
        if projection_cols != final_norm_weights.len() {
            bail!(
                "Gemma decode projection width {projection_cols} does not match final norm width {}",
                final_norm_weights.len()
            );
        }

        Ok(Self {
            metadata: GemmaDecodeTransitionMetadata {
                source_id: identifier.clone(),
                layer_count: layers.len(),
                embedding_vocab_size: embedding.row_count,
                embedding_width: embedding.col_count,
                final_norm_width: final_norm_weights.len(),
                projection_rows,
                projection_cols,
                projection_kind,
                has_ple_global: ple.is_some(),
                has_final_logit_softcapping: final_scalars.final_logit_softcapping.is_some(),
            },
            identifier,
            embedding,
            embedding_scale,
            layers,
            backing_layers,
            ple,
            final_norm_weights,
            final_scalars,
            projection,
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    fn layer_metadata(&self, layer_idx: usize) -> Result<GemmaDecodeLayerMetadata> {
        self.layers.get(layer_idx).cloned().ok_or_else(|| {
            anyhow!(
                "Gemma decode layer index {layer_idx} is out of range for {} layers",
                self.layers.len()
            )
        })
    }

    fn backing_layer(&self, layer_idx: usize) -> Result<&GemmaDecodeLayerBacking> {
        self.backing_layers.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "Gemma decode layer backing {layer_idx} is out of range for {} layers",
                self.backing_layers.len()
            )
        })
    }

    fn ple(&self) -> Result<&GemmaDecodePleBacking> {
        self.ple
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma decode transition source has no global PLE weights"))
    }
}

impl AuthRead<GemmaDecodeTransitionMetadataRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = GemmaDecodeTransitionMetadata;

    fn auth_read(&self, _request: GemmaDecodeTransitionMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata.clone())
    }
}

impl AuthRead<GemmaDecodeEmbeddingRowRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaDecodeEmbeddingRowRequest) -> Result<Self::Output> {
        let row = det_num_tensor_slice_row_wgts(
            &self.embedding,
            usize::try_from(request.token_id).expect("u32 should fit into usize"),
            "decode embedding",
        )?;
        Ok(row
            .into_iter()
            .map(|value| scale_act(Act::from_bits(value.to_bits()), self.embedding_scale))
            .collect())
    }
}

impl AuthRead<GemmaDecodeLayerMetadataRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = GemmaDecodeLayerMetadata;

    fn auth_read(&self, request: GemmaDecodeLayerMetadataRequest) -> Result<Self::Output> {
        self.layer_metadata(request.layer_idx)
    }
}

impl AuthRead<GemmaDecodeLayerScalarsRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = GemmaDecodeLayerScalars;

    fn auth_read(&self, request: GemmaDecodeLayerScalarsRequest) -> Result<Self::Output> {
        Ok(self.backing_layer(request.layer_idx)?.scalars)
    }
}

impl AuthRead<GemmaDecodeLayerMatrixRowRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeLayerMatrixRowRequest) -> Result<Self::Output> {
        let layer = self.backing_layer(request.layer_idx)?;
        let matrix = layer.matrix_source(request.matrix, request.layer_idx)?;
        matrix_row_wgts(matrix, request.row_idx, request.matrix.label())
    }
}

impl AuthRead<GemmaDecodeLayerNormWeightsRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeLayerNormWeightsRequest) -> Result<Self::Output> {
        let layer = self.backing_layer(request.layer_idx)?;
        layer.norm_weights(request.norm, request.layer_idx)
    }
}

impl AuthRead<GemmaDecodePleTokenEmbeddingRowRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaDecodePleTokenEmbeddingRowRequest) -> Result<Self::Output> {
        let ple = self.ple()?;
        let matrix = ple.token_embeddings.get(request.layer_idx).ok_or_else(|| {
            anyhow!(
                "Gemma decode PLE token embedding layer {} is out of range",
                request.layer_idx
            )
        })?;
        Ok(ple_token_row(matrix, request.token_id)?)
    }
}

impl AuthRead<GemmaDecodePleModelProjectionRowRequest>
    for AuthenticatedGemmaDecodeTransitionSource
{
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodePleModelProjectionRowRequest) -> Result<Self::Output> {
        let ple = self.ple()?;
        let matrix = ple
            .model_projections
            .get(request.layer_idx)
            .ok_or_else(|| {
                anyhow!(
                    "Gemma decode PLE model projection layer {} is out of range",
                    request.layer_idx
                )
            })?;
        ple_projection_row(matrix, request.row_idx)
    }
}

impl AuthRead<GemmaDecodePleProjectionNormWeightsRequest>
    for AuthenticatedGemmaDecodeTransitionSource
{
    type Output = Vec<Wgt>;

    fn auth_read(
        &self,
        _request: GemmaDecodePleProjectionNormWeightsRequest,
    ) -> Result<Self::Output> {
        Ok(self.ple()?.projection_norm_weights.clone())
    }
}

impl AuthRead<GemmaDecodePleScalarsRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = GemmaDecodePleScalars;

    fn auth_read(&self, _request: GemmaDecodePleScalarsRequest) -> Result<Self::Output> {
        Ok(self.ple()?.scalars)
    }
}

impl AuthRead<GemmaDecodeFinalNormWeightsRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, _request: GemmaDecodeFinalNormWeightsRequest) -> Result<Self::Output> {
        Ok(self.final_norm_weights.clone())
    }
}

impl AuthRead<GemmaDecodeFinalScalarsRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = GemmaDecodeFinalScalars;

    fn auth_read(&self, _request: GemmaDecodeFinalScalarsRequest) -> Result<Self::Output> {
        Ok(self.final_scalars)
    }
}

impl AuthRead<GemmaDecodeProjectionRowRequest> for AuthenticatedGemmaDecodeTransitionSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeProjectionRowRequest) -> Result<Self::Output> {
        match &self.projection {
            GemmaDecodeProjectionBacking::Matrix(matrix) => {
                matrix_row_wgts_from_det_matrix(matrix, request.row_idx, "decode lm_head")
            }
            GemmaDecodeProjectionBacking::TensorSlice(source) => {
                det_num_tensor_slice_row_wgts(source, request.row_idx, "decode tied embedding")
            }
        }
    }
}

impl GemmaDecodeLayerBacking {
    fn matrix_source(
        &self,
        kind: GemmaDecodeLayerMatrixKind,
        layer_idx: usize,
    ) -> Result<&Gemma4LayerMatrixSource> {
        match kind {
            GemmaDecodeLayerMatrixKind::Query => Ok(&self.matrices.q_proj),
            GemmaDecodeLayerMatrixKind::Key => Ok(&self.matrices.k_proj),
            GemmaDecodeLayerMatrixKind::Value => self
                .matrices
                .v_proj
                .as_ref()
                .ok_or_else(|| anyhow!("Gemma decode layer {layer_idx} has no v_proj matrix")),
            GemmaDecodeLayerMatrixKind::Output => Ok(&self.matrices.o_proj),
            GemmaDecodeLayerMatrixKind::Gate => Ok(&self.matrices.gate_proj),
            GemmaDecodeLayerMatrixKind::Up => Ok(&self.matrices.up_proj),
            GemmaDecodeLayerMatrixKind::Down => Ok(&self.matrices.down_proj),
            GemmaDecodeLayerMatrixKind::PleInputGate => self
                .matrices
                .ple
                .as_ref()
                .map(|ple| &ple.input_gate)
                .ok_or_else(|| {
                    anyhow!("Gemma decode layer {layer_idx} has no PLE input gate matrix")
                }),
            GemmaDecodeLayerMatrixKind::PleLayerProjection => self
                .matrices
                .ple
                .as_ref()
                .map(|ple| &ple.layer_projection)
                .ok_or_else(|| {
                    anyhow!("Gemma decode layer {layer_idx} has no PLE layer projection matrix")
                }),
        }
    }

    fn norm_weights(&self, kind: GemmaDecodeLayerNormKind, layer_idx: usize) -> Result<Vec<Wgt>> {
        let weights = match kind {
            GemmaDecodeLayerNormKind::Query => &self.norms.q_norm,
            GemmaDecodeLayerNormKind::Key => &self.norms.k_norm,
            GemmaDecodeLayerNormKind::InputLayer => &self.norms.input_layernorm,
            GemmaDecodeLayerNormKind::PostAttention => &self.norms.post_attention_layernorm,
            GemmaDecodeLayerNormKind::PreFeedForward => &self.norms.pre_feedforward_layernorm,
            GemmaDecodeLayerNormKind::PostFeedForward => &self.norms.post_feedforward_layernorm,
            GemmaDecodeLayerNormKind::PlePostInput => {
                return self.norms.ple_post_input_norm.clone().ok_or_else(|| {
                    anyhow!("Gemma decode layer {layer_idx} has no PLE post-input norm weights")
                });
            }
        };
        Ok(weights.clone())
    }
}

impl From<Gemma4AttentionKind> for GemmaDecodeAttentionKind {
    fn from(value: Gemma4AttentionKind) -> Self {
        match value {
            Gemma4AttentionKind::Sliding => Self::Sliding,
            Gemma4AttentionKind::Full => Self::Full,
        }
    }
}

impl GemmaDecodeLayerMatrixKind {
    fn label(self) -> &'static str {
        match self {
            Self::Query => "decode q_proj",
            Self::Key => "decode k_proj",
            Self::Value => "decode v_proj",
            Self::Output => "decode o_proj",
            Self::Gate => "decode gate_proj",
            Self::Up => "decode up_proj",
            Self::Down => "decode down_proj",
            Self::PleInputGate => "decode PLE input_gate",
            Self::PleLayerProjection => "decode PLE layer_projection",
        }
    }
}

fn validate_identifier(identifier: String) -> Result<String> {
    if identifier.is_empty() {
        bail!("Gemma decode transition source identifier must not be empty");
    }
    Ok(identifier)
}

fn canonical_embedding(model: &Gemma4TransformerModel) -> Result<(DetNumTensorSliceSource, Act)> {
    let source = match model.embedding_source.as_ref() {
        Some(GemmaEmbeddingTensorSource::Deterministic { source, scale, .. }) => {
            (source.clone(), Act::from_num(*scale))
        }
        Some(_) | None => {
            bail!("deterministic raster decode transition requires a .detwgt embedding source")
        }
    };
    if source.0.row_count == 0 || source.0.col_count == 0 {
        bail!("Gemma decode embedding source must have non-zero shape");
    }
    if source.0.row_offset != 0
        || source.0.row_count != source.0.total_rows
        || source.0.col_offset != 0
        || source.0.col_count != source.0.total_cols
    {
        bail!("Gemma decode embedding source must reference the full embedding matrix");
    }
    Ok(source)
}

fn build_layer(
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
) -> Result<(GemmaDecodeLayerMetadata, GemmaDecodeLayerBacking)> {
    if layer.v_proj.is_none() && !layer.attention_k_eq_v {
        bail!("Gemma decode layer {layer_idx} is missing v_proj without attention_k_eq_v enabled");
    }

    let q_proj_shape = canonical_matrix_shape(layer_idx, "q_proj", &layer.q_proj)?;
    let k_proj_shape = canonical_matrix_shape(layer_idx, "k_proj", &layer.k_proj)?;
    let v_proj_shape = layer
        .v_proj
        .as_ref()
        .map(|source| canonical_matrix_shape(layer_idx, "v_proj", source))
        .transpose()?;
    let o_proj_shape = canonical_matrix_shape(layer_idx, "o_proj", &layer.o_proj)?;
    let gate_proj_shape = canonical_matrix_shape(layer_idx, "gate_proj", &layer.gate_proj)?;
    let up_proj_shape = canonical_matrix_shape(layer_idx, "up_proj", &layer.up_proj)?;
    let down_proj_shape = canonical_matrix_shape(layer_idx, "down_proj", &layer.down_proj)?;

    let q_norm = canonical_norm_weights(layer_idx, "q_norm_weight", &layer.q_norm_weight_det)?;
    let k_norm = canonical_norm_weights(layer_idx, "k_norm_weight", &layer.k_norm_weight_det)?;
    let input_layernorm = canonical_norm_weights(
        layer_idx,
        "input_layernorm_weight",
        &layer.input_layernorm_weight_det,
    )?;
    let post_attention_layernorm = canonical_norm_weights(
        layer_idx,
        "post_attention_layernorm_weight",
        &layer.post_attention_layernorm_weight_det,
    )?;
    let pre_feedforward_layernorm = canonical_norm_weights(
        layer_idx,
        "pre_feedforward_layernorm_weight",
        &layer.pre_feedforward_layernorm_weight_det,
    )?;
    let post_feedforward_layernorm = canonical_norm_weights(
        layer_idx,
        "post_feedforward_layernorm_weight",
        &layer.post_feedforward_layernorm_weight_det,
    )?;

    let (ple_matrices, ple_input_gate_shape, ple_layer_projection_shape, ple_post_input_norm) =
        if let Some(ple) = &layer.ple {
            let input_gate_shape =
                canonical_matrix_shape(layer_idx, "PLE input_gate", &ple.input_gate)?;
            let layer_projection_shape =
                canonical_matrix_shape(layer_idx, "PLE layer_projection", &ple.layer_projection)?;
            let post_input_norm = canonical_norm_weights(
                layer_idx,
                "PLE post_input_norm_weight",
                &ple.post_input_norm_weight_det,
            )?;
            (
                Some(GemmaDecodePleLayerMatrices {
                    input_gate: ple.input_gate.clone(),
                    layer_projection: ple.layer_projection.clone(),
                }),
                Some(input_gate_shape),
                Some(layer_projection_shape),
                Some(post_input_norm),
            )
        } else {
            (None, None, None, None)
        };

    let scalars = canonical_layer_scalars(layer_idx, layer)?;
    let metadata = GemmaDecodeLayerMetadata {
        layer_idx,
        attention_kind: layer.attention_kind.into(),
        hidden_size: layer.hidden_size,
        num_heads: layer.num_heads,
        num_kv_heads: layer.num_kv_heads,
        head_dim: layer.head_dim,
        sliding_window: layer.sliding_window,
        cache_sliding_window: layer.cache_sliding_window,
        partial_rotary_dim: layer.partial_rotary_dim,
        rope_freq_base_dim: layer.rope_freq_base_dim,
        kv_shared_layer_index: layer.kv_shared_layer_index,
        attention_k_eq_v: layer.attention_k_eq_v,
        has_v_proj: layer.v_proj.is_some(),
        has_ple: layer.ple.is_some(),
        has_layer_scalar: scalars.layer_scalar.is_some(),
        q_proj_shape,
        k_proj_shape,
        v_proj_shape,
        o_proj_shape,
        gate_proj_shape,
        up_proj_shape,
        down_proj_shape,
        ple_input_gate_shape,
        ple_layer_projection_shape,
    };
    let backing = GemmaDecodeLayerBacking {
        matrices: GemmaDecodeLayerMatrices {
            q_proj: layer.q_proj.clone(),
            k_proj: layer.k_proj.clone(),
            v_proj: layer.v_proj.clone(),
            o_proj: layer.o_proj.clone(),
            gate_proj: layer.gate_proj.clone(),
            up_proj: layer.up_proj.clone(),
            down_proj: layer.down_proj.clone(),
            ple: ple_matrices,
        },
        norms: GemmaDecodeLayerNorms {
            q_norm,
            k_norm,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            ple_post_input_norm,
        },
        scalars,
    };

    Ok((metadata, backing))
}

fn build_ple(
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps_det: Option<Acc>,
) -> Result<GemmaDecodePleBacking> {
    ensure_ple_backing_is_canonical(ple_global)?;
    let projection_norm_weights = canonical_norm_weights(
        0,
        "PLE projection_norm_weight",
        &ple_global.projection_norm_weight_det,
    )?;
    Ok(GemmaDecodePleBacking {
        token_embeddings: ple_global.token_embeddings.clone(),
        model_projections: ple_global.model_projections.clone(),
        projection_norm_weights,
        scalars: GemmaDecodePleScalars {
            embedding_scale: ple_global.embedding_scale_det.ok_or_else(|| {
                anyhow!("deterministic raster decode PLE source requires canonical embedding scale")
            })?,
            projection_scalar: ple_global.projection_scalar_det.ok_or_else(|| {
                anyhow!(
                    "deterministic raster decode PLE source requires canonical projection scalar"
                )
            })?,
            input_scale: ple_global.input_scale_det.ok_or_else(|| {
                anyhow!("deterministic raster decode PLE source requires canonical input scale")
            })?,
            rms_norm_eps: rms_norm_eps_det.ok_or_else(|| {
                anyhow!("deterministic raster decode PLE source requires canonical RMSNorm epsilon")
            })?,
        },
    })
}

fn ensure_ple_backing_is_canonical(ple_global: &Gemma4PleGlobalWeights) -> Result<()> {
    for (layer_idx, source) in ple_global.token_embeddings.iter().enumerate() {
        if !matches!(source, Gemma4PleMatrixSource::DetNumLazy(_)) {
            bail!("deterministic raster decode PLE token embedding layer {layer_idx} requires .detwgt backing");
        }
    }
    for (layer_idx, source) in ple_global.model_projections.iter().enumerate() {
        if !matches!(source, Gemma4PleMatrixSource::DetNumLazy(_)) {
            bail!("deterministic raster decode PLE model projection layer {layer_idx} requires .detwgt backing");
        }
    }
    Ok(())
}

fn validate_ple_shapes(
    layers: &[GemmaDecodeLayerMetadata],
    ple: &GemmaDecodePleBacking,
) -> Result<()> {
    for layer in layers.iter().filter(|layer| layer.has_ple) {
        let ple_width = layer
            .ple_input_gate_shape
            .ok_or_else(|| {
                anyhow!(
                    "Gemma decode PLE layer {} is missing input gate shape",
                    layer.layer_idx
                )
            })?
            .rows;
        let token_embedding_shape =
            ple_matrix_shape(ple.token_embeddings.get(layer.layer_idx).ok_or_else(|| {
                anyhow!(
                    "Gemma decode PLE token embedding layer {} is missing",
                    layer.layer_idx
                )
            })?)?;
        if token_embedding_shape.cols != ple_width {
            bail!(
                "Gemma decode PLE token embedding width {} does not match layer {} PLE width {}",
                token_embedding_shape.cols,
                layer.layer_idx,
                ple_width
            );
        }

        let model_projection_shape =
            ple_matrix_shape(ple.model_projections.get(layer.layer_idx).ok_or_else(|| {
                anyhow!(
                    "Gemma decode PLE model projection layer {} is missing",
                    layer.layer_idx
                )
            })?)?;
        if model_projection_shape.rows != ple_width
            || model_projection_shape.cols != layer.hidden_size
        {
            bail!(
                "Gemma decode PLE model projection shape {}x{} does not match layer {} expected {}x{}",
                model_projection_shape.rows,
                model_projection_shape.cols,
                layer.layer_idx,
                ple_width,
                layer.hidden_size
            );
        }
        if ple.projection_norm_weights.len() != ple_width {
            bail!(
                "Gemma decode PLE projection norm width {} does not match layer {} PLE width {}",
                ple.projection_norm_weights.len(),
                layer.layer_idx,
                ple_width
            );
        }
    }
    Ok(())
}

fn ple_matrix_shape(source: &Gemma4PleMatrixSource) -> Result<GemmaDecodeMatrixShape> {
    let Gemma4PleMatrixSource::DetNumLazy(source) = source else {
        bail!("deterministic raster decode PLE matrix shape requires .detwgt backing");
    };
    Ok(GemmaDecodeMatrixShape {
        rows: source.row_count,
        cols: source.col_count,
    })
}

fn canonical_projection(
    model: &Gemma4TransformerModel,
) -> Result<(
    GemmaDecodeProjectionKind,
    usize,
    usize,
    GemmaDecodeProjectionBacking,
)> {
    match &model.logits_projection {
        Gemma4LogitsProjection::UntiedLmHead {
            det_weight: Some(det_weight),
            ..
        } => {
            validate_det_matrix_shape(det_weight, "decode lm_head")?;
            Ok((
                GemmaDecodeProjectionKind::UntiedLmHead,
                det_weight.rows,
                det_weight.cols,
                GemmaDecodeProjectionBacking::Matrix(det_weight.clone()),
            ))
        }
        Gemma4LogitsProjection::UntiedLmHead {
            det_weight: None, ..
        } => bail!("deterministic raster decode transition requires canonical lm_head det_weight"),
        Gemma4LogitsProjection::TiedEmbedding(_) => {
            let source = match model.embedding_source.as_ref() {
                Some(GemmaEmbeddingTensorSource::Deterministic { source, .. }) => source,
                Some(_) | None => bail!(
                    "deterministic raster decode tied embedding logits require a .detwgt embedding source"
                ),
            };
            if source.row_count == 0 || source.col_count == 0 {
                bail!("Gemma decode tied embedding source must have non-zero shape");
            }
            Ok((
                GemmaDecodeProjectionKind::TiedEmbedding,
                source.row_count,
                source.col_count,
                GemmaDecodeProjectionBacking::TensorSlice(source.clone()),
            ))
        }
    }
}

fn canonical_matrix_shape(
    layer_idx: usize,
    label: &str,
    source: &Gemma4LayerMatrixSource,
) -> Result<GemmaDecodeMatrixShape> {
    let Gemma4LayerMatrixSource::DetNumLazy { source, .. } = source else {
        bail!(
            "deterministic raster decode layer source requires .detwgt {label} source at layer {layer_idx}"
        );
    };
    if source.row_count == 0 || source.col_count == 0 {
        bail!("Gemma decode layer {layer_idx} {label} matrix must have non-zero shape");
    }
    Ok(GemmaDecodeMatrixShape {
        rows: source.row_count,
        cols: source.col_count,
    })
}

fn canonical_norm_weights(
    layer_idx: usize,
    label: &str,
    weights: &Option<Vec<Wgt>>,
) -> Result<Vec<Wgt>> {
    let weights = weights.clone().ok_or_else(|| {
        anyhow!("deterministic raster decode layer {layer_idx} requires canonical {label}")
    })?;
    if weights.is_empty() {
        bail!("Gemma decode layer {layer_idx} {label} must not be empty");
    }
    Ok(weights)
}

fn canonical_final_norm_weights(weights: &Option<Vec<Wgt>>) -> Result<Vec<Wgt>> {
    let weights = weights.clone().ok_or_else(|| {
        anyhow!("deterministic raster decode transition requires canonical final norm weights")
    })?;
    if weights.is_empty() {
        bail!("Gemma decode final norm weights must not be empty");
    }
    Ok(weights)
}

fn canonical_layer_scalars(
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
) -> Result<GemmaDecodeLayerScalars> {
    let rms_norm_eps = layer.rms_norm_eps_det.ok_or_else(|| {
        anyhow!("deterministic raster decode layer {layer_idx} requires canonical RMSNorm epsilon")
    })?;
    let rope_base = if layer.partial_rotary_dim == 0 {
        layer.rope_base_det
    } else {
        Some(layer.rope_base_det.ok_or_else(|| {
            anyhow!("deterministic raster decode layer {layer_idx} requires canonical RoPE base")
        })?)
    };
    let layer_scalar = if layer.layer_scalar.is_some() {
        Some(layer.layer_scalar_det.ok_or_else(|| {
            anyhow!("deterministic raster decode layer {layer_idx} requires canonical layer scalar")
        })?)
    } else {
        None
    };

    Ok(GemmaDecodeLayerScalars {
        rms_norm_eps,
        rope_base,
        layer_scalar,
    })
}

fn matrix_row_wgts(
    source: &Gemma4LayerMatrixSource,
    row_idx: usize,
    label: &str,
) -> Result<Vec<Wgt>> {
    let Gemma4LayerMatrixSource::DetNumLazy { source, .. } = source else {
        bail!("deterministic raster decode source requires .detwgt {label} source");
    };
    det_num_tensor_slice_row_wgts(source, row_idx, label)
}

fn matrix_row_wgts_from_det_matrix(
    matrix: &DetNumMatrix,
    row_idx: usize,
    label: &str,
) -> Result<Vec<Wgt>> {
    if matrix.rows == 0 || matrix.cols == 0 {
        bail!("Gemma {label} matrix must have non-zero shape");
    }
    if row_idx >= matrix.rows {
        bail!(
            "Gemma {label} row {row_idx} is out of range for {} rows",
            matrix.rows
        );
    }
    let start = row_idx
        .checked_mul(matrix.cols)
        .ok_or_else(|| anyhow!("Gemma {label} row offset overflowed"))?;
    let end = start
        .checked_add(matrix.cols)
        .ok_or_else(|| anyhow!("Gemma {label} row offset overflowed"))?;
    Ok(matrix.values[start..end]
        .iter()
        .copied()
        .map(Wgt::from_bits)
        .collect())
}

fn validate_det_matrix_shape(matrix: &DetNumMatrix, label: &str) -> Result<()> {
    if matrix.rows == 0 || matrix.cols == 0 {
        bail!("Gemma {label} matrix must have non-zero shape");
    }
    if matrix.values.len() != matrix.rows * matrix.cols {
        bail!(
            "Gemma {label} matrix values length {} does not match shape {}x{}",
            matrix.values.len(),
            matrix.rows,
            matrix.cols
        );
    }
    Ok(())
}

fn ple_token_row(source: &Gemma4PleMatrixSource, token_id: u32) -> Result<Vec<Act>> {
    let Gemma4PleMatrixSource::DetNumLazy(source) = source else {
        bail!("deterministic raster decode PLE token row requires .detwgt backing");
    };
    let row = det_num_tensor_slice_row_wgts(
        source,
        usize::try_from(token_id).expect("u32 should fit into usize"),
        "decode PLE token embedding",
    )?;
    Ok(row
        .into_iter()
        .map(|value| Act::from_bits(value.to_bits()))
        .collect())
}

fn ple_projection_row(source: &Gemma4PleMatrixSource, row_idx: usize) -> Result<Vec<Wgt>> {
    let Gemma4PleMatrixSource::DetNumLazy(source) = source else {
        bail!("deterministic raster decode PLE projection row requires .detwgt backing");
    };
    det_num_tensor_slice_row_wgts(source, row_idx, "decode PLE model projection")
}
