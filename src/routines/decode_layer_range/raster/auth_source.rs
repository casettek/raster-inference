use std::{cell::RefCell, marker::PhantomData, sync::Arc};

use anyhow::{anyhow, bail, Result};

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::external_artifacts::{
    decode_i32_vec_response, decode_postcard_response, postcard_external_source_entry,
    postcard_i32_vec_external_source_entry, postcard_request_key, register_external_source,
    CommittedExternalRequest, CommittedExternalSource, ExternalSourceEntry, ExternalSourceId,
    ExternalSourceRef,
};
#[cfg(feature = "unchecked-raster-integrity")]
use crate::shared::artifacts::integrity_mode::raster_integrity_is_unchecked;
use crate::shared::model::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
    Gemma4PleMatrixSource, Gemma4TransformerModel, GemmaEmbeddingTensorSource,
};
use crate::shared::numerics::det_num::{scale_act, Acc, Act, Wgt};
use crate::shared::raster_kernels::transformer::det_num_tensor_slice_row_wgts;

#[derive(Debug, Clone)]
pub struct AuthenticatedGemmaDecodeLayerRangeSource {
    identifier: String,
    metadata: GemmaDecodeLayerRangeMetadata,
    embedding: DetNumTensorSliceSource,
    embedding_scale: Act,
    layers: Vec<GemmaDecodeLayerMetadata>,
    backing_layers: Vec<GemmaDecodeLayerBacking>,
    ple: Option<GemmaDecodePleBacking>,
    final_norm_weights: Vec<Wgt>,
    final_scalars: GemmaDecodeFinalScalars,
    projection: GemmaDecodeProjectionBacking,
    committed_source: RefCell<Option<CommittedExternalSource>>,
}

#[derive(Debug)]
pub enum RasterDecodeLayerRangeSource<'a> {
    Committed {
        source: CommittedExternalSource,
        _marker: PhantomData<&'a AuthenticatedGemmaDecodeLayerRangeSource>,
    },
    #[cfg(feature = "unchecked-raster-integrity")]
    DirectUnchecked {
        source: &'a AuthenticatedGemmaDecodeLayerRangeSource,
        root: String,
    },
}

impl<'a> RasterDecodeLayerRangeSource<'a> {
    pub fn for_current_integrity_mode(
        source: &'a AuthenticatedGemmaDecodeLayerRangeSource,
    ) -> Result<Self> {
        #[cfg(feature = "unchecked-raster-integrity")]
        if raster_integrity_is_unchecked() {
            return Ok(Self::DirectUnchecked {
                root: format!(
                    "raster-unchecked-test:direct-decode-layer-range:{}",
                    source.identifier()
                ),
                source,
            });
        }

        Ok(Self::Committed {
            source: source.committed_source()?,
            _marker: PhantomData,
        })
    }

    pub fn root(&self) -> &str {
        match self {
            Self::Committed { source, .. } => source.root(),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { root, .. } => root,
        }
    }
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
pub struct GemmaDecodeLayerRangeMetadata {
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

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct GemmaDecodeLayerScalarsPayload {
    rms_norm_eps_bits: i64,
    rope_base_bits: Option<i64>,
    layer_scalar_bits: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodePleScalars {
    pub embedding_scale: Act,
    pub projection_scalar: Act,
    pub input_scale: Act,
    pub rms_norm_eps: Acc,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct GemmaDecodePleScalarsPayload {
    embedding_scale_bits: i32,
    projection_scalar_bits: i32,
    input_scale_bits: i32,
    rms_norm_eps_bits: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeFinalScalars {
    pub rms_norm_eps: Acc,
    pub final_logit_softcapping: Option<Act>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct GemmaDecodeFinalScalarsPayload {
    rms_norm_eps_bits: i64,
    final_logit_softcapping_bits: Option<i32>,
}

const GEMMA_DECODE_LAYER_RANGE_SOURCE_KIND: &str = "gemma_decode_layer_range";
const GEMMA_DECODE_LAYER_RANGE_SOURCE_DOMAIN: &str =
    "raster-external-source-gemma-decode-layer-range-merkle-v1";
const DECODE_LAYER_RANGE_METADATA_REQUEST: &str = "gemma_decode_layer_range.metadata";
const DECODE_EMBEDDING_ROW_REQUEST: &str = "gemma_decode_layer_range.embedding_row";
const DECODE_LAYER_METADATA_REQUEST: &str = "gemma_decode_layer_range.layer_metadata";
const DECODE_LAYER_SCALARS_REQUEST: &str = "gemma_decode_layer_range.layer_scalars";
const DECODE_LAYER_MATRIX_ROW_REQUEST: &str = "gemma_decode_layer_range.layer_matrix_row";
const DECODE_LAYER_NORM_WEIGHTS_REQUEST: &str = "gemma_decode_layer_range.layer_norm_weights";
const DECODE_PLE_TOKEN_EMBEDDING_ROW_REQUEST: &str =
    "gemma_decode_layer_range.ple_token_embedding_row";
const DECODE_PLE_MODEL_PROJECTION_ROW_REQUEST: &str =
    "gemma_decode_layer_range.ple_model_projection_row";
const DECODE_PLE_PROJECTION_NORM_WEIGHTS_REQUEST: &str =
    "gemma_decode_layer_range.ple_projection_norm_weights";
const DECODE_PLE_SCALARS_REQUEST: &str = "gemma_decode_layer_range.ple_scalars";
const DECODE_FINAL_NORM_WEIGHTS_REQUEST: &str = "gemma_decode_layer_range.final_norm_weights";
const DECODE_FINAL_SCALARS_REQUEST: &str = "gemma_decode_layer_range.final_scalars";
const DECODE_PROJECTION_ROW_REQUEST: &str = "gemma_decode_layer_range.projection_row";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecodeLayerRangeMetadataRequest;

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

impl AuthenticatedGemmaDecodeLayerRangeSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        if model.provenance != Gemma4ModelProvenance::DetNumWgt {
            bail!(
                "deterministic raster decode layer range source requires a model loaded from a .detwgt artifact"
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
            bail!("Gemma decode layer range source has PLE layers but no global PLE weights");
        }
        if let Some(ple) = ple.as_ref() {
            validate_ple_shapes(&layers, ple)?;
        }

        let final_norm_weights = canonical_final_norm_weights(&model.final_norm_weight_det)?;
        let final_logit_softcapping = if model.final_logit_softcapping.is_some() {
            Some(model.final_logit_softcapping_det.ok_or_else(|| {
                anyhow!(
                    "deterministic raster decode layer range requires canonical final logit softcap"
                )
            })?)
        } else {
            None
        };
        let final_scalars = GemmaDecodeFinalScalars {
            rms_norm_eps: model.rms_norm_eps_det.ok_or_else(|| {
                anyhow!(
                    "deterministic raster decode layer range requires canonical RMSNorm epsilon"
                )
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
            metadata: GemmaDecodeLayerRangeMetadata {
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
            committed_source: RefCell::new(None),
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn committed_source_ref(&self) -> Result<ExternalSourceRef> {
        Ok(self.committed_source()?.source_ref().clone())
    }

    pub fn committed_source(&self) -> Result<CommittedExternalSource> {
        if let Some(source) = self.committed_source.borrow().clone() {
            return Ok(source);
        }
        let source_ref = register_external_source(
            ExternalSourceId::new(format!("decode-layer-range:{}", self.identifier))?,
            GEMMA_DECODE_LAYER_RANGE_SOURCE_KIND,
            GEMMA_DECODE_LAYER_RANGE_SOURCE_DOMAIN,
            self.committed_source_entries()?,
        )?;
        let source = CommittedExternalSource::new(source_ref);
        *self.committed_source.borrow_mut() = Some(source.clone());
        Ok(source)
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
            .ok_or_else(|| anyhow!("Gemma decode layer range source has no global PLE weights"))
    }

    fn committed_source_entries(&self) -> Result<Vec<ExternalSourceEntry>> {
        let mut entries = Vec::new();
        entries.push(postcard_external_source_entry(
            GemmaDecodeLayerRangeMetadataRequest.request_key()?,
            &self.metadata,
        )?);
        for token_id in 0..self.metadata.embedding_vocab_size {
            let request = GemmaDecodeEmbeddingRowRequest {
                token_id: u32::try_from(token_id)
                    .map_err(|_| anyhow!("decode embedding token id {token_id} exceeds u32"))?,
            };
            let row_bits = self
                .native_embedding_row(request)?
                .into_iter()
                .map(|value| value.to_bits());
            entries.push(postcard_i32_vec_external_source_entry(
                request.request_key()?,
                row_bits,
            )?);
        }
        for layer in &self.layers {
            entries.push(postcard_external_source_entry(
                GemmaDecodeLayerMetadataRequest {
                    layer_idx: layer.layer_idx,
                }
                .request_key()?,
                layer,
            )?);
            entries.push(postcard_external_source_entry(
                GemmaDecodeLayerScalarsRequest {
                    layer_idx: layer.layer_idx,
                }
                .request_key()?,
                &self
                    .native_layer_scalars(GemmaDecodeLayerScalarsRequest {
                        layer_idx: layer.layer_idx,
                    })?
                    .payload(),
            )?);
            for norm in decode_layer_norm_requests(layer) {
                let request = GemmaDecodeLayerNormWeightsRequest {
                    layer_idx: layer.layer_idx,
                    norm,
                };
                let weights = self
                    .native_layer_norm_weights(request)?
                    .into_iter()
                    .map(|value| value.to_bits());
                entries.push(postcard_i32_vec_external_source_entry(
                    request.request_key()?,
                    weights,
                )?);
            }
            for (matrix, rows) in decode_layer_matrix_requests(layer) {
                for row_idx in 0..rows {
                    let request = GemmaDecodeLayerMatrixRowRequest {
                        layer_idx: layer.layer_idx,
                        matrix,
                        row_idx,
                    };
                    let row_bits = self
                        .native_layer_matrix_row(request)?
                        .into_iter()
                        .map(|value| value.to_bits());
                    entries.push(postcard_i32_vec_external_source_entry(
                        request.request_key()?,
                        row_bits,
                    )?);
                }
            }
        }
        if self.ple.is_some() {
            entries.push(postcard_i32_vec_external_source_entry(
                GemmaDecodePleProjectionNormWeightsRequest.request_key()?,
                self.native_ple_projection_norm_weights()?
                    .into_iter()
                    .map(|value| value.to_bits()),
            )?);
            entries.push(postcard_external_source_entry(
                GemmaDecodePleScalarsRequest.request_key()?,
                &self.native_ple_scalars()?.payload(),
            )?);
            for layer in self.layers.iter().filter(|layer| layer.has_ple) {
                let token_rows = self
                    .ple()?
                    .token_embeddings
                    .get(layer.layer_idx)
                    .map(ple_matrix_shape)
                    .transpose()?
                    .ok_or_else(|| {
                        anyhow!(
                            "Gemma decode PLE token embedding layer {} is missing",
                            layer.layer_idx
                        )
                    })?
                    .rows;
                for token_id in 0..token_rows {
                    let request = GemmaDecodePleTokenEmbeddingRowRequest {
                        layer_idx: layer.layer_idx,
                        token_id: u32::try_from(token_id).map_err(|_| {
                            anyhow!(
                                "decode PLE token id {token_id} exceeds u32 at layer {}",
                                layer.layer_idx
                            )
                        })?,
                    };
                    let row_bits = self
                        .native_ple_token_embedding_row(request)?
                        .into_iter()
                        .map(|value| value.to_bits());
                    entries.push(postcard_i32_vec_external_source_entry(
                        request.request_key()?,
                        row_bits,
                    )?);
                }
                let projection_rows = self
                    .ple()?
                    .model_projections
                    .get(layer.layer_idx)
                    .map(ple_matrix_shape)
                    .transpose()?
                    .ok_or_else(|| {
                        anyhow!(
                            "Gemma decode PLE model projection layer {} is missing",
                            layer.layer_idx
                        )
                    })?
                    .rows;
                for row_idx in 0..projection_rows {
                    let request = GemmaDecodePleModelProjectionRowRequest {
                        layer_idx: layer.layer_idx,
                        row_idx,
                    };
                    let row_bits = self
                        .native_ple_model_projection_row(request)?
                        .into_iter()
                        .map(|value| value.to_bits());
                    entries.push(postcard_i32_vec_external_source_entry(
                        request.request_key()?,
                        row_bits,
                    )?);
                }
            }
        }
        entries.push(postcard_i32_vec_external_source_entry(
            GemmaDecodeFinalNormWeightsRequest.request_key()?,
            self.native_final_norm_weights()?
                .into_iter()
                .map(|value| value.to_bits()),
        )?);
        entries.push(postcard_external_source_entry(
            GemmaDecodeFinalScalarsRequest.request_key()?,
            &self.native_final_scalars().payload(),
        )?);
        for row_idx in 0..self.metadata.projection_rows {
            let request = GemmaDecodeProjectionRowRequest { row_idx };
            let row_bits = self
                .native_projection_row(request)?
                .into_iter()
                .map(|value| value.to_bits());
            entries.push(postcard_i32_vec_external_source_entry(
                request.request_key()?,
                row_bits,
            )?);
        }
        Ok(entries)
    }

    fn native_embedding_row(&self, request: GemmaDecodeEmbeddingRowRequest) -> Result<Vec<Act>> {
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

    fn native_layer_scalars(
        &self,
        request: GemmaDecodeLayerScalarsRequest,
    ) -> Result<GemmaDecodeLayerScalars> {
        Ok(self.backing_layer(request.layer_idx)?.scalars)
    }

    fn native_layer_matrix_row(
        &self,
        request: GemmaDecodeLayerMatrixRowRequest,
    ) -> Result<Vec<Wgt>> {
        let layer = self.backing_layer(request.layer_idx)?;
        let matrix = layer.matrix_source(request.matrix, request.layer_idx)?;
        matrix_row_wgts(matrix, request.row_idx, request.matrix.label())
    }

    fn native_layer_norm_weights(
        &self,
        request: GemmaDecodeLayerNormWeightsRequest,
    ) -> Result<Vec<Wgt>> {
        let layer = self.backing_layer(request.layer_idx)?;
        layer.norm_weights(request.norm, request.layer_idx)
    }

    fn native_ple_token_embedding_row(
        &self,
        request: GemmaDecodePleTokenEmbeddingRowRequest,
    ) -> Result<Vec<Act>> {
        let ple = self.ple()?;
        let matrix = ple.token_embeddings.get(request.layer_idx).ok_or_else(|| {
            anyhow!(
                "Gemma decode PLE token embedding layer {} is out of range",
                request.layer_idx
            )
        })?;
        ple_token_row(matrix, request.token_id)
    }

    fn native_ple_model_projection_row(
        &self,
        request: GemmaDecodePleModelProjectionRowRequest,
    ) -> Result<Vec<Wgt>> {
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

    fn native_ple_projection_norm_weights(&self) -> Result<Vec<Wgt>> {
        Ok(self.ple()?.projection_norm_weights.clone())
    }

    fn native_ple_scalars(&self) -> Result<GemmaDecodePleScalars> {
        Ok(self.ple()?.scalars)
    }

    fn native_final_norm_weights(&self) -> Result<Vec<Wgt>> {
        Ok(self.final_norm_weights.clone())
    }

    fn native_final_scalars(&self) -> GemmaDecodeFinalScalars {
        self.final_scalars
    }

    fn native_projection_row(&self, request: GemmaDecodeProjectionRowRequest) -> Result<Vec<Wgt>> {
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

impl AuthRead<GemmaDecodeLayerRangeMetadataRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = GemmaDecodeLayerRangeMetadata;

    fn auth_read(&self, request: GemmaDecodeLayerRangeMetadataRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeLayerRangeMetadataRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = GemmaDecodeLayerRangeMetadata;

    fn auth_read(&self, request: GemmaDecodeLayerRangeMetadataRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => Ok(source.metadata.clone()),
        }
    }
}

impl AuthRead<GemmaDecodeEmbeddingRowRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaDecodeEmbeddingRowRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeEmbeddingRowRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaDecodeEmbeddingRowRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.native_embedding_row(request),
        }
    }
}

impl AuthRead<GemmaDecodeLayerMetadataRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = GemmaDecodeLayerMetadata;

    fn auth_read(&self, request: GemmaDecodeLayerMetadataRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeLayerMetadataRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = GemmaDecodeLayerMetadata;

    fn auth_read(&self, request: GemmaDecodeLayerMetadataRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source
                .layers
                .get(request.layer_idx)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "Gemma decode layer index {} is out of range for {} layers",
                        request.layer_idx,
                        source.layers.len()
                    )
                }),
        }
    }
}

impl AuthRead<GemmaDecodeLayerScalarsRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = GemmaDecodeLayerScalars;

    fn auth_read(&self, request: GemmaDecodeLayerScalarsRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeLayerScalarsRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = GemmaDecodeLayerScalars;

    fn auth_read(&self, request: GemmaDecodeLayerScalarsRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.native_layer_scalars(request),
        }
    }
}

impl AuthRead<GemmaDecodeLayerMatrixRowRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeLayerMatrixRowRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeLayerMatrixRowRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeLayerMatrixRowRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.native_layer_matrix_row(request),
        }
    }
}

impl AuthRead<GemmaDecodeLayerNormWeightsRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeLayerNormWeightsRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeLayerNormWeightsRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeLayerNormWeightsRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.native_layer_norm_weights(request),
        }
    }
}

impl AuthRead<GemmaDecodePleTokenEmbeddingRowRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaDecodePleTokenEmbeddingRowRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodePleTokenEmbeddingRowRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaDecodePleTokenEmbeddingRowRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.native_ple_token_embedding_row(request),
        }
    }
}

impl AuthRead<GemmaDecodePleModelProjectionRowRequest>
    for AuthenticatedGemmaDecodeLayerRangeSource
{
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodePleModelProjectionRowRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodePleModelProjectionRowRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodePleModelProjectionRowRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.native_ple_model_projection_row(request),
        }
    }
}

impl AuthRead<GemmaDecodePleProjectionNormWeightsRequest>
    for AuthenticatedGemmaDecodeLayerRangeSource
{
    type Output = Vec<Wgt>;

    fn auth_read(
        &self,
        request: GemmaDecodePleProjectionNormWeightsRequest,
    ) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodePleProjectionNormWeightsRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(
        &self,
        request: GemmaDecodePleProjectionNormWeightsRequest,
    ) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => {
                let _ = request;
                source.native_ple_projection_norm_weights()
            }
        }
    }
}

impl AuthRead<GemmaDecodePleScalarsRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = GemmaDecodePleScalars;

    fn auth_read(&self, request: GemmaDecodePleScalarsRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodePleScalarsRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = GemmaDecodePleScalars;

    fn auth_read(&self, request: GemmaDecodePleScalarsRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => {
                let _ = request;
                source.native_ple_scalars()
            }
        }
    }
}

impl AuthRead<GemmaDecodeFinalNormWeightsRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeFinalNormWeightsRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeFinalNormWeightsRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeFinalNormWeightsRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => {
                let _ = request;
                source.native_final_norm_weights()
            }
        }
    }
}

impl AuthRead<GemmaDecodeFinalScalarsRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = GemmaDecodeFinalScalars;

    fn auth_read(&self, request: GemmaDecodeFinalScalarsRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeFinalScalarsRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = GemmaDecodeFinalScalars;

    fn auth_read(&self, request: GemmaDecodeFinalScalarsRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => {
                let _ = request;
                Ok(source.native_final_scalars())
            }
        }
    }
}

impl AuthRead<GemmaDecodeProjectionRowRequest> for AuthenticatedGemmaDecodeLayerRangeSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeProjectionRowRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaDecodeProjectionRowRequest> for RasterDecodeLayerRangeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaDecodeProjectionRowRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.native_projection_row(request),
        }
    }
}

impl CommittedExternalRequest for GemmaDecodeLayerRangeMetadataRequest {
    type Output = GemmaDecodeLayerRangeMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_LAYER_RANGE_METADATA_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl CommittedExternalRequest for GemmaDecodeEmbeddingRowRequest {
    type Output = Vec<Act>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_EMBEDDING_ROW_REQUEST, &self.token_id)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Act::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaDecodeLayerMetadataRequest {
    type Output = GemmaDecodeLayerMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_LAYER_METADATA_REQUEST, &self.layer_idx)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl CommittedExternalRequest for GemmaDecodeLayerScalarsRequest {
    type Output = GemmaDecodeLayerScalars;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_LAYER_SCALARS_REQUEST, &self.layer_idx)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(GemmaDecodeLayerScalars::from_payload(
            decode_postcard_response(response_payload)?,
        ))
    }
}

impl CommittedExternalRequest for GemmaDecodeLayerMatrixRowRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(
            DECODE_LAYER_MATRIX_ROW_REQUEST,
            &(self.layer_idx, self.matrix, self.row_idx),
        )
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaDecodeLayerNormWeightsRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(
            DECODE_LAYER_NORM_WEIGHTS_REQUEST,
            &(self.layer_idx, self.norm),
        )
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaDecodePleTokenEmbeddingRowRequest {
    type Output = Vec<Act>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(
            DECODE_PLE_TOKEN_EMBEDDING_ROW_REQUEST,
            &(self.layer_idx, self.token_id),
        )
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Act::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaDecodePleModelProjectionRowRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(
            DECODE_PLE_MODEL_PROJECTION_ROW_REQUEST,
            &(self.layer_idx, self.row_idx),
        )
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaDecodePleProjectionNormWeightsRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_PLE_PROJECTION_NORM_WEIGHTS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaDecodePleScalarsRequest {
    type Output = GemmaDecodePleScalars;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_PLE_SCALARS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(GemmaDecodePleScalars::from_payload(
            decode_postcard_response(response_payload)?,
        ))
    }
}

impl CommittedExternalRequest for GemmaDecodeFinalNormWeightsRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_FINAL_NORM_WEIGHTS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaDecodeFinalScalarsRequest {
    type Output = GemmaDecodeFinalScalars;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_FINAL_SCALARS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(GemmaDecodeFinalScalars::from_payload(
            decode_postcard_response(response_payload)?,
        ))
    }
}

impl CommittedExternalRequest for GemmaDecodeProjectionRowRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(DECODE_PROJECTION_ROW_REQUEST, &self.row_idx)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
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

fn decode_layer_matrix_requests(
    layer: &GemmaDecodeLayerMetadata,
) -> Vec<(GemmaDecodeLayerMatrixKind, usize)> {
    let mut matrices = vec![
        (GemmaDecodeLayerMatrixKind::Query, layer.q_proj_shape.rows),
        (GemmaDecodeLayerMatrixKind::Key, layer.k_proj_shape.rows),
        (GemmaDecodeLayerMatrixKind::Output, layer.o_proj_shape.rows),
        (GemmaDecodeLayerMatrixKind::Gate, layer.gate_proj_shape.rows),
        (GemmaDecodeLayerMatrixKind::Up, layer.up_proj_shape.rows),
        (GemmaDecodeLayerMatrixKind::Down, layer.down_proj_shape.rows),
    ];
    if let Some(shape) = layer.v_proj_shape {
        matrices.push((GemmaDecodeLayerMatrixKind::Value, shape.rows));
    }
    if let Some(shape) = layer.ple_input_gate_shape {
        matrices.push((GemmaDecodeLayerMatrixKind::PleInputGate, shape.rows));
    }
    if let Some(shape) = layer.ple_layer_projection_shape {
        matrices.push((GemmaDecodeLayerMatrixKind::PleLayerProjection, shape.rows));
    }
    matrices
}

fn decode_layer_norm_requests(layer: &GemmaDecodeLayerMetadata) -> Vec<GemmaDecodeLayerNormKind> {
    let mut norms = vec![
        GemmaDecodeLayerNormKind::Query,
        GemmaDecodeLayerNormKind::Key,
        GemmaDecodeLayerNormKind::InputLayer,
        GemmaDecodeLayerNormKind::PostAttention,
        GemmaDecodeLayerNormKind::PreFeedForward,
        GemmaDecodeLayerNormKind::PostFeedForward,
    ];
    if layer.has_ple {
        norms.push(GemmaDecodeLayerNormKind::PlePostInput);
    }
    norms
}

impl GemmaDecodeLayerScalars {
    fn payload(self) -> GemmaDecodeLayerScalarsPayload {
        GemmaDecodeLayerScalarsPayload {
            rms_norm_eps_bits: self.rms_norm_eps.to_bits(),
            rope_base_bits: self.rope_base.map(|value| value.to_bits()),
            layer_scalar_bits: self.layer_scalar.map(|value| value.to_bits()),
        }
    }

    fn from_payload(payload: GemmaDecodeLayerScalarsPayload) -> Self {
        Self {
            rms_norm_eps: Acc::from_bits(payload.rms_norm_eps_bits),
            rope_base: payload.rope_base_bits.map(Acc::from_bits),
            layer_scalar: payload.layer_scalar_bits.map(Act::from_bits),
        }
    }
}

impl GemmaDecodePleScalars {
    fn payload(self) -> GemmaDecodePleScalarsPayload {
        GemmaDecodePleScalarsPayload {
            embedding_scale_bits: self.embedding_scale.to_bits(),
            projection_scalar_bits: self.projection_scalar.to_bits(),
            input_scale_bits: self.input_scale.to_bits(),
            rms_norm_eps_bits: self.rms_norm_eps.to_bits(),
        }
    }

    fn from_payload(payload: GemmaDecodePleScalarsPayload) -> Self {
        Self {
            embedding_scale: Act::from_bits(payload.embedding_scale_bits),
            projection_scalar: Act::from_bits(payload.projection_scalar_bits),
            input_scale: Act::from_bits(payload.input_scale_bits),
            rms_norm_eps: Acc::from_bits(payload.rms_norm_eps_bits),
        }
    }
}

impl GemmaDecodeFinalScalars {
    fn payload(self) -> GemmaDecodeFinalScalarsPayload {
        GemmaDecodeFinalScalarsPayload {
            rms_norm_eps_bits: self.rms_norm_eps.to_bits(),
            final_logit_softcapping_bits: self.final_logit_softcapping.map(|value| value.to_bits()),
        }
    }

    fn from_payload(payload: GemmaDecodeFinalScalarsPayload) -> Self {
        Self {
            rms_norm_eps: Acc::from_bits(payload.rms_norm_eps_bits),
            final_logit_softcapping: payload.final_logit_softcapping_bits.map(Act::from_bits),
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
        bail!("Gemma decode layer range source identifier must not be empty");
    }
    Ok(identifier)
}

fn canonical_embedding(model: &Gemma4TransformerModel) -> Result<(DetNumTensorSliceSource, Act)> {
    let source = match model.embedding_source.as_ref() {
        Some(GemmaEmbeddingTensorSource::Deterministic { source, scale, .. }) => {
            (source.clone(), Act::from_num(*scale))
        }
        Some(_) | None => {
            bail!("deterministic raster decode layer range requires a .detwgt embedding source")
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
        } => bail!("deterministic raster decode layer range requires canonical lm_head det_weight"),
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
        anyhow!("deterministic raster decode layer range requires canonical final norm weights")
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
    let row = matrix
        .values
        .get_widened(start, end)
        .ok_or_else(|| anyhow!("Gemma {label} row range is out of bounds"))?;
    Ok(row.into_iter().map(Wgt::from_bits).collect())
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
