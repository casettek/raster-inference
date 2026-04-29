use std::fs::File;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;

use crate::raster_authoring::AuthRead;
use crate::shared::det_num::{Acc, Act, Wgt};
use crate::shared::transformer::{
    DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource, Gemma4LayerWeights,
    Gemma4ModelProvenance, Gemma4TransformerModel,
};

#[derive(Debug, Clone)]
pub struct AuthenticatedGemmaPrefillLayerSource {
    identifier: String,
    layers: Vec<GemmaPrefillLayerMetadata>,
    backing_layers: Vec<GemmaPrefillLayerBacking>,
}

#[derive(Debug, Clone)]
struct GemmaPrefillLayerBacking {
    matrices: GemmaPrefillLayerMatrices,
    norms: GemmaPrefillLayerNorms,
    scalars: GemmaPrefillLayerScalars,
}

#[derive(Debug, Clone)]
struct GemmaPrefillLayerMatrices {
    q_proj: Gemma4LayerMatrixSource,
    k_proj: Gemma4LayerMatrixSource,
    v_proj: Option<Gemma4LayerMatrixSource>,
    o_proj: Gemma4LayerMatrixSource,
    gate_proj: Gemma4LayerMatrixSource,
    up_proj: Gemma4LayerMatrixSource,
    down_proj: Gemma4LayerMatrixSource,
    ple: Option<GemmaPrefillPleLayerMatrices>,
}

#[derive(Debug, Clone)]
struct GemmaPrefillPleLayerMatrices {
    input_gate: Gemma4LayerMatrixSource,
    layer_projection: Gemma4LayerMatrixSource,
}

#[derive(Debug, Clone)]
struct GemmaPrefillLayerNorms {
    q_norm: Vec<Wgt>,
    k_norm: Vec<Wgt>,
    input_layernorm: Vec<Wgt>,
    post_attention_layernorm: Vec<Wgt>,
    pre_feedforward_layernorm: Vec<Wgt>,
    post_feedforward_layernorm: Vec<Wgt>,
    ple_post_input_norm: Option<Vec<Wgt>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPrefillLayerSourceMetadata {
    pub source_id: String,
    pub layer_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPrefillLayerMetadata {
    pub layer_idx: usize,
    pub attention_kind: GemmaPrefillAttentionKind,
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
    pub q_proj_shape: GemmaPrefillMatrixShape,
    pub k_proj_shape: GemmaPrefillMatrixShape,
    pub v_proj_shape: Option<GemmaPrefillMatrixShape>,
    pub o_proj_shape: GemmaPrefillMatrixShape,
    pub gate_proj_shape: GemmaPrefillMatrixShape,
    pub up_proj_shape: GemmaPrefillMatrixShape,
    pub down_proj_shape: GemmaPrefillMatrixShape,
    pub ple_input_gate_shape: Option<GemmaPrefillMatrixShape>,
    pub ple_layer_projection_shape: Option<GemmaPrefillMatrixShape>,
    pub q_norm_width: usize,
    pub k_norm_width: usize,
    pub input_layernorm_width: usize,
    pub post_attention_layernorm_width: usize,
    pub pre_feedforward_layernorm_width: usize,
    pub post_feedforward_layernorm_width: usize,
    pub ple_post_input_norm_width: Option<usize>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum GemmaPrefillAttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPrefillMatrixShape {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillLayerScalars {
    pub rms_norm_eps: Acc,
    pub rope_base: Option<Acc>,
    pub layer_scalar: Option<Act>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillLayerSourceMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillLayerMetadataRequest {
    pub layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillLayerScalarsRequest {
    pub layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillLayerMatrixRowRequest {
    pub layer_idx: usize,
    pub matrix: GemmaPrefillLayerMatrixKind,
    pub row_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillLayerNormWeightsRequest {
    pub layer_idx: usize,
    pub norm: GemmaPrefillLayerNormKind,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum GemmaPrefillLayerMatrixKind {
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

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum GemmaPrefillLayerNormKind {
    Query,
    Key,
    InputLayer,
    PostAttention,
    PreFeedForward,
    PostFeedForward,
    PlePostInput,
}

impl AuthenticatedGemmaPrefillLayerSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        if model.provenance != Gemma4ModelProvenance::DetNumWgt {
            bail!(
                "deterministic raster prefill layer source requires a model loaded from a .detwgt artifact"
            );
        }

        let mut layers = Vec::with_capacity(model.layers.len());
        let mut backing_layers = Vec::with_capacity(model.layers.len());
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let (metadata, backing) = build_layer(layer_idx, layer)?;
            layers.push(metadata);
            backing_layers.push(backing);
        }

        Ok(Self {
            identifier,
            layers,
            backing_layers,
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    fn source_metadata(&self) -> GemmaPrefillLayerSourceMetadata {
        GemmaPrefillLayerSourceMetadata {
            source_id: self.identifier.clone(),
            layer_count: self.layers.len(),
        }
    }

    fn layer_metadata(&self, layer_idx: usize) -> Result<GemmaPrefillLayerMetadata> {
        self.layers.get(layer_idx).cloned().ok_or_else(|| {
            anyhow!(
                "Gemma prefill layer index {layer_idx} is out of range for {} layers",
                self.layers.len()
            )
        })
    }

    fn backing_layer(&self, layer_idx: usize) -> Result<&GemmaPrefillLayerBacking> {
        self.backing_layers.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "Gemma prefill layer backing {layer_idx} is out of range for {} layers",
                self.backing_layers.len()
            )
        })
    }
}

impl AuthRead<GemmaPrefillLayerSourceMetadataRequest> for AuthenticatedGemmaPrefillLayerSource {
    type Output = GemmaPrefillLayerSourceMetadata;

    fn auth_read(&self, _request: GemmaPrefillLayerSourceMetadataRequest) -> Result<Self::Output> {
        Ok(self.source_metadata())
    }
}

impl AuthRead<GemmaPrefillLayerMetadataRequest> for AuthenticatedGemmaPrefillLayerSource {
    type Output = GemmaPrefillLayerMetadata;

    fn auth_read(&self, request: GemmaPrefillLayerMetadataRequest) -> Result<Self::Output> {
        self.layer_metadata(request.layer_idx)
    }
}

impl AuthRead<GemmaPrefillLayerScalarsRequest> for AuthenticatedGemmaPrefillLayerSource {
    type Output = GemmaPrefillLayerScalars;

    fn auth_read(&self, request: GemmaPrefillLayerScalarsRequest) -> Result<Self::Output> {
        Ok(self.backing_layer(request.layer_idx)?.scalars)
    }
}

impl AuthRead<GemmaPrefillLayerMatrixRowRequest> for AuthenticatedGemmaPrefillLayerSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillLayerMatrixRowRequest) -> Result<Self::Output> {
        let layer = self.backing_layer(request.layer_idx)?;
        let matrix = layer.matrix_source(request.matrix, request.layer_idx)?;
        matrix_row_wgts(matrix, request.row_idx, request.matrix.label())
    }
}

impl AuthRead<GemmaPrefillLayerNormWeightsRequest> for AuthenticatedGemmaPrefillLayerSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillLayerNormWeightsRequest) -> Result<Self::Output> {
        let layer = self.backing_layer(request.layer_idx)?;
        layer.norm_weights(request.norm, request.layer_idx)
    }
}

impl GemmaPrefillLayerBacking {
    fn matrix_source(
        &self,
        kind: GemmaPrefillLayerMatrixKind,
        layer_idx: usize,
    ) -> Result<&Gemma4LayerMatrixSource> {
        match kind {
            GemmaPrefillLayerMatrixKind::Query => Ok(&self.matrices.q_proj),
            GemmaPrefillLayerMatrixKind::Key => Ok(&self.matrices.k_proj),
            GemmaPrefillLayerMatrixKind::Value => self
                .matrices
                .v_proj
                .as_ref()
                .ok_or_else(|| anyhow!("Gemma prefill layer {layer_idx} has no v_proj matrix")),
            GemmaPrefillLayerMatrixKind::Output => Ok(&self.matrices.o_proj),
            GemmaPrefillLayerMatrixKind::Gate => Ok(&self.matrices.gate_proj),
            GemmaPrefillLayerMatrixKind::Up => Ok(&self.matrices.up_proj),
            GemmaPrefillLayerMatrixKind::Down => Ok(&self.matrices.down_proj),
            GemmaPrefillLayerMatrixKind::PleInputGate => self
                .matrices
                .ple
                .as_ref()
                .map(|ple| &ple.input_gate)
                .ok_or_else(|| {
                    anyhow!("Gemma prefill layer {layer_idx} has no PLE input gate matrix")
                }),
            GemmaPrefillLayerMatrixKind::PleLayerProjection => self
                .matrices
                .ple
                .as_ref()
                .map(|ple| &ple.layer_projection)
                .ok_or_else(|| {
                    anyhow!("Gemma prefill layer {layer_idx} has no PLE layer projection matrix")
                }),
        }
    }

    fn norm_weights(&self, kind: GemmaPrefillLayerNormKind, layer_idx: usize) -> Result<Vec<Wgt>> {
        let weights = match kind {
            GemmaPrefillLayerNormKind::Query => &self.norms.q_norm,
            GemmaPrefillLayerNormKind::Key => &self.norms.k_norm,
            GemmaPrefillLayerNormKind::InputLayer => &self.norms.input_layernorm,
            GemmaPrefillLayerNormKind::PostAttention => &self.norms.post_attention_layernorm,
            GemmaPrefillLayerNormKind::PreFeedForward => &self.norms.pre_feedforward_layernorm,
            GemmaPrefillLayerNormKind::PostFeedForward => &self.norms.post_feedforward_layernorm,
            GemmaPrefillLayerNormKind::PlePostInput => {
                return self.norms.ple_post_input_norm.clone().ok_or_else(|| {
                    anyhow!("Gemma prefill layer {layer_idx} has no PLE post-input norm weights")
                });
            }
        };
        Ok(weights.clone())
    }
}

impl From<Gemma4AttentionKind> for GemmaPrefillAttentionKind {
    fn from(value: Gemma4AttentionKind) -> Self {
        match value {
            Gemma4AttentionKind::Sliding => Self::Sliding,
            Gemma4AttentionKind::Full => Self::Full,
        }
    }
}

impl GemmaPrefillLayerMatrixKind {
    fn label(self) -> &'static str {
        match self {
            Self::Query => "q_proj",
            Self::Key => "k_proj",
            Self::Value => "v_proj",
            Self::Output => "o_proj",
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
            Self::PleInputGate => "PLE input_gate",
            Self::PleLayerProjection => "PLE layer_projection",
        }
    }
}

fn validate_identifier(identifier: String) -> Result<String> {
    if identifier.is_empty() {
        bail!("Gemma prefill layer source identifier must not be empty");
    }
    Ok(identifier)
}

fn build_layer(
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
) -> Result<(GemmaPrefillLayerMetadata, GemmaPrefillLayerBacking)> {
    if layer.v_proj.is_none() && !layer.attention_k_eq_v {
        bail!("Gemma prefill layer {layer_idx} is missing v_proj without attention_k_eq_v enabled");
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
                Some(GemmaPrefillPleLayerMatrices {
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
    let metadata = GemmaPrefillLayerMetadata {
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
        q_norm_width: q_norm.len(),
        k_norm_width: k_norm.len(),
        input_layernorm_width: input_layernorm.len(),
        post_attention_layernorm_width: post_attention_layernorm.len(),
        pre_feedforward_layernorm_width: pre_feedforward_layernorm.len(),
        post_feedforward_layernorm_width: post_feedforward_layernorm.len(),
        ple_post_input_norm_width: ple_post_input_norm.as_ref().map(Vec::len),
    };
    let backing = GemmaPrefillLayerBacking {
        matrices: GemmaPrefillLayerMatrices {
            q_proj: layer.q_proj.clone(),
            k_proj: layer.k_proj.clone(),
            v_proj: layer.v_proj.clone(),
            o_proj: layer.o_proj.clone(),
            gate_proj: layer.gate_proj.clone(),
            up_proj: layer.up_proj.clone(),
            down_proj: layer.down_proj.clone(),
            ple: ple_matrices,
        },
        norms: GemmaPrefillLayerNorms {
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

fn canonical_matrix_shape(
    layer_idx: usize,
    label: &str,
    source: &Gemma4LayerMatrixSource,
) -> Result<GemmaPrefillMatrixShape> {
    let Gemma4LayerMatrixSource::DetNumLazy { source, .. } = source else {
        bail!(
            "deterministic raster prefill layer source requires .detwgt {label} source at layer {layer_idx}"
        );
    };
    if source.row_count == 0 || source.col_count == 0 {
        bail!("Gemma prefill layer {layer_idx} {label} matrix must have non-zero shape");
    }
    Ok(GemmaPrefillMatrixShape {
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
        anyhow!("deterministic raster prefill layer {layer_idx} requires canonical {label}")
    })?;
    if weights.is_empty() {
        bail!("Gemma prefill layer {layer_idx} {label} must not be empty");
    }
    Ok(weights)
}

fn canonical_layer_scalars(
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
) -> Result<GemmaPrefillLayerScalars> {
    let rms_norm_eps = layer.rms_norm_eps_det.ok_or_else(|| {
        anyhow!("deterministic raster prefill layer {layer_idx} requires canonical RMSNorm epsilon")
    })?;
    let rope_base = if layer.partial_rotary_dim == 0 {
        layer.rope_base_det
    } else {
        Some(layer.rope_base_det.ok_or_else(|| {
            anyhow!("deterministic raster prefill layer {layer_idx} requires canonical RoPE base")
        })?)
    };
    let layer_scalar = if layer.layer_scalar.is_some() {
        Some(layer.layer_scalar_det.ok_or_else(|| {
            anyhow!(
                "deterministic raster prefill layer {layer_idx} requires canonical layer scalar"
            )
        })?)
    } else {
        None
    };

    Ok(GemmaPrefillLayerScalars {
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
        bail!("deterministic raster prefill layer source requires .detwgt {label} source");
    };
    det_num_matrix_row_wgts(source, row_idx, label)
}

fn det_num_matrix_row_wgts(
    source: &DetNumTensorSliceSource,
    row_idx: usize,
    label: &str,
) -> Result<Vec<Wgt>> {
    if row_idx >= source.row_count {
        bail!(
            "Gemma prefill {label} row {row_idx} is out of range for {} rows",
            source.row_count
        );
    }

    let file = File::open(&source.weights_path).with_context(|| {
        format!(
            "failed to open deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
        format!(
            "failed to mmap deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let row_bytes = source
        .total_cols
        .checked_mul(4)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let global_row_idx = source.row_offset + row_idx;
    let start = source
        .data_offset
        .checked_add(global_row_idx * row_bytes)
        .and_then(|offset| offset.checked_add(source.col_offset * 4))
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let end = start
        .checked_add(source.col_count * 4)
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let encoded_row = mmap
        .get(start..end)
        .ok_or_else(|| anyhow!("matrix slice byte range is out of bounds"))?;
    let mut row = Vec::with_capacity(source.col_count);
    for encoded_value in encoded_row.chunks_exact(4) {
        row.push(Wgt::from_bits(i32::from_le_bytes(
            encoded_value
                .try_into()
                .expect("i32 byte width should match"),
        )));
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedGemmaPrefillLayerSource, GemmaPrefillAttentionKind,
        GemmaPrefillLayerMatrixKind, GemmaPrefillLayerMatrixRowRequest,
        GemmaPrefillLayerMetadataRequest, GemmaPrefillLayerNormKind,
        GemmaPrefillLayerNormWeightsRequest, GemmaPrefillLayerScalarsRequest,
        GemmaPrefillLayerSourceMetadataRequest,
    };
    use crate::shared::det_num::{Acc, Act, Wgt};
    use crate::shared::transformer::{
        DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource, Gemma4LayerWeights,
        Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleLayerWeights,
        Gemma4TransformerModel, MatrixF32,
    };
    use anyhow::{Context, Result};
    use std::path::{Path, PathBuf};

    #[test]
    fn canonical_model_reads_source_and_layer_metadata() {
        let (_path, model) = canonical_model(true, false);
        let source =
            AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer-fixture", &model)
                .expect("source should build");

        let metadata = crate::auth_read!(&source, GemmaPrefillLayerSourceMetadataRequest)
            .expect("source metadata should read");
        assert_eq!(source.identifier(), "prefill-layer-fixture");
        assert_eq!(metadata.source_id, "prefill-layer-fixture");
        assert_eq!(metadata.layer_count, 1);

        let layer = crate::auth_read!(&source, GemmaPrefillLayerMetadataRequest { layer_idx: 0 })
            .expect("layer metadata should read");
        assert_eq!(layer.attention_kind, GemmaPrefillAttentionKind::Full);
        assert_eq!(layer.hidden_size, 2);
        assert_eq!(layer.num_heads, 1);
        assert_eq!(layer.num_kv_heads, 1);
        assert!(layer.has_v_proj);
        assert!(layer.has_ple);
        assert!(layer.has_layer_scalar);
        assert_eq!(layer.q_proj_shape.rows, 2);
        assert_eq!(layer.q_proj_shape.cols, 2);
        assert_eq!(layer.ple_input_gate_shape.expect("PLE input gate").rows, 2);
        assert_eq!(layer.ple_post_input_norm_width, Some(2));
    }

    #[test]
    fn canonical_model_reads_matrix_rows() {
        let (_path, model) = canonical_model(true, false);
        let source =
            AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer-fixture", &model)
                .expect("source should build");

        let row = crate::auth_read!(
            &source,
            GemmaPrefillLayerMatrixRowRequest {
                layer_idx: 0,
                matrix: GemmaPrefillLayerMatrixKind::Query,
                row_idx: 1,
            },
        )
        .expect("matrix row should read");

        assert_eq!(row, vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)]);
    }

    #[test]
    fn canonical_model_reads_norm_weights_and_scalars() {
        let (_path, model) = canonical_model(true, false);
        let source =
            AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer-fixture", &model)
                .expect("source should build");

        let q_norm = crate::auth_read!(
            &source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: 0,
                norm: GemmaPrefillLayerNormKind::Query,
            },
        )
        .expect("q norm should read");
        assert_eq!(q_norm, vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]);

        let ple_norm = crate::auth_read!(
            &source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: 0,
                norm: GemmaPrefillLayerNormKind::PlePostInput,
            },
        )
        .expect("PLE norm should read");
        assert_eq!(ple_norm, vec![Wgt::from_num(0.75), Wgt::from_num(1.25)]);

        let scalars = crate::auth_read!(&source, GemmaPrefillLayerScalarsRequest { layer_idx: 0 },)
            .expect("scalars should read");
        assert_eq!(scalars.rms_norm_eps, Acc::from_num(0.001));
        assert_eq!(scalars.rope_base, None);
        assert_eq!(scalars.layer_scalar, Some(Act::from_num(0.5)));
    }

    #[test]
    fn source_without_ple_reports_absence_and_rejects_ple_reads() {
        let (_path, model) = canonical_model(false, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("no-ple", &model)
            .expect("source should build");

        let layer = crate::auth_read!(&source, GemmaPrefillLayerMetadataRequest { layer_idx: 0 })
            .expect("layer metadata should read");
        assert!(!layer.has_ple);
        assert_eq!(layer.ple_input_gate_shape, None);
        assert_eq!(layer.ple_post_input_norm_width, None);

        let error = crate::auth_read!(
            &source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: 0,
                norm: GemmaPrefillLayerNormKind::PlePostInput,
            },
        )
        .expect_err("PLE norm read should fail");
        assert!(error.to_string().contains("no PLE post-input norm weights"));

        let error = crate::auth_read!(
            &source,
            GemmaPrefillLayerMatrixRowRequest {
                layer_idx: 0,
                matrix: GemmaPrefillLayerMatrixKind::PleInputGate,
                row_idx: 0,
            },
        )
        .expect_err("PLE matrix read should fail");
        assert!(error.to_string().contains("no PLE input gate matrix"));
    }

    #[test]
    fn source_allows_missing_v_proj_when_k_equals_v_is_enabled() {
        let (_path, model) = canonical_model(false, true);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("k-eq-v", &model)
            .expect("source should build");

        let layer = crate::auth_read!(&source, GemmaPrefillLayerMetadataRequest { layer_idx: 0 })
            .expect("layer metadata should read");
        assert!(!layer.has_v_proj);
        assert!(layer.attention_k_eq_v);

        let error = crate::auth_read!(
            &source,
            GemmaPrefillLayerMatrixRowRequest {
                layer_idx: 0,
                matrix: GemmaPrefillLayerMatrixKind::Value,
                row_idx: 0,
            },
        )
        .expect_err("missing v_proj row should fail");
        assert!(error.to_string().contains("has no v_proj matrix"));
    }

    #[test]
    fn invalid_layer_index_fails_with_clear_layer_count() {
        let (_path, model) = canonical_model(false, false);
        let source =
            AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer-fixture", &model)
                .expect("source should build");

        let error = crate::auth_read!(&source, GemmaPrefillLayerMetadataRequest { layer_idx: 1 })
            .expect_err("invalid layer should fail");

        assert!(error.to_string().contains("out of range for 1 layers"));
    }

    #[test]
    fn invalid_matrix_row_fails_with_clear_row_count() {
        let (_path, model) = canonical_model(false, false);
        let source =
            AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer-fixture", &model)
                .expect("source should build");

        let error = crate::auth_read!(
            &source,
            GemmaPrefillLayerMatrixRowRequest {
                layer_idx: 0,
                matrix: GemmaPrefillLayerMatrixKind::Query,
                row_idx: 2,
            },
        )
        .expect_err("invalid row should fail");

        assert!(error
            .to_string()
            .contains("row 2 is out of range for 2 rows"));
    }

    #[test]
    fn construction_rejects_fp32_model_provenance() {
        let (_path, mut model) = canonical_model(false, false);
        model.provenance = Gemma4ModelProvenance::Fp32;

        let error = AuthenticatedGemmaPrefillLayerSource::from_model("fp32", &model)
            .expect_err("fp32 model should fail");

        assert!(error.to_string().contains(".detwgt artifact"));
    }

    #[test]
    fn construction_rejects_non_canonical_layer_matrix() {
        let (_path, mut model) = canonical_model(false, false);
        model.layers[0].q_proj = Gemma4LayerMatrixSource::from(MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![1.0, 0.0, 0.0, 1.0],
        });

        let error = AuthenticatedGemmaPrefillLayerSource::from_model("non-canonical", &model)
            .expect_err("non-canonical matrix should fail");

        assert!(error.to_string().contains(".detwgt q_proj source"));
    }

    #[test]
    fn construction_rejects_missing_v_proj_without_k_equals_v() {
        let (_path, mut model) = canonical_model(false, false);
        model.layers[0].v_proj = None;
        model.layers[0].attention_k_eq_v = false;

        let error = AuthenticatedGemmaPrefillLayerSource::from_model("missing-v", &model)
            .expect_err("missing v_proj should fail");

        assert!(error
            .to_string()
            .contains("missing v_proj without attention_k_eq_v enabled"));
    }

    fn canonical_model(has_ple: bool, attention_k_eq_v: bool) -> (PathBuf, Gemma4TransformerModel) {
        let matrices = vec![
            matrix(vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
                vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)],
            ]),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
        ];
        let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
        let mut sources = sources.into_iter();
        let q_proj = det_matrix(sources.next().expect("q source"));
        let k_proj = det_matrix(sources.next().expect("k source"));
        let v_proj = (!attention_k_eq_v).then(|| det_matrix(sources.next().expect("v source")));
        if attention_k_eq_v {
            let _skipped_v_source = sources.next().expect("v source");
        }
        let o_proj = det_matrix(sources.next().expect("o source"));
        let gate_proj = det_matrix(sources.next().expect("gate source"));
        let up_proj = det_matrix(sources.next().expect("up source"));
        let down_proj = det_matrix(sources.next().expect("down source"));
        let ple_input_gate = det_matrix(sources.next().expect("PLE input source"));
        let ple_layer_projection = det_matrix(sources.next().expect("PLE projection source"));

        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Full,
            hidden_size: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: None,
            cache_sliding_window: None,
            rms_norm_eps: 0.001,
            rms_norm_eps_det: Some(Acc::from_num(0.001)),
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm_weight: vec![1.0, 0.5],
            q_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]),
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(1.0)]),
            input_layernorm_weight: vec![1.0, 1.0],
            input_layernorm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(1.0)]),
            post_attention_layernorm_weight: vec![1.0, 1.0],
            post_attention_layernorm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(1.0)]),
            pre_feedforward_layernorm_weight: vec![1.0, 1.0],
            pre_feedforward_layernorm_weight_det: Some(vec![
                Wgt::from_num(1.0),
                Wgt::from_num(1.0),
            ]),
            post_feedforward_layernorm_weight: vec![1.0, 1.0],
            post_feedforward_layernorm_weight_det: Some(vec![
                Wgt::from_num(1.0),
                Wgt::from_num(1.0),
            ]),
            gate_proj,
            up_proj,
            down_proj,
            ple: has_ple.then(|| Gemma4PleLayerWeights {
                input_gate: ple_input_gate,
                layer_projection: ple_layer_projection,
                post_input_norm_weight: vec![0.75, 1.25],
                post_input_norm_weight_det: Some(vec![Wgt::from_num(0.75), Wgt::from_num(1.25)]),
            }),
            layer_scalar: Some(0.5),
            layer_scalar_det: Some(Act::from_num(0.5)),
        };

        (
            path,
            Gemma4TransformerModel {
                provenance: Gemma4ModelProvenance::DetNumWgt,
                embedding_table: None,
                embedding_source: None,
                layers: vec![layer],
                ple_global: None,
                final_norm_weight: vec![1.0, 1.0],
                final_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(1.0)]),
                logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                    weight: MatrixF32 {
                        rows: 1,
                        cols: 2,
                        values: vec![0.0, 0.0],
                    },
                    det_weight: None,
                },
                final_logit_softcapping: None,
                final_logit_softcapping_det: None,
                rms_norm_eps: 0.001,
                rms_norm_eps_det: Some(Acc::from_num(0.001)),
            },
        )
    }

    fn identity_matrix() -> Vec<Vec<Wgt>> {
        matrix(vec![
            vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
            vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
        ])
    }

    fn matrix(rows: Vec<Vec<Wgt>>) -> Vec<Vec<Wgt>> {
        rows
    }

    fn det_matrix(source: DetNumTensorSliceSource) -> Gemma4LayerMatrixSource {
        Gemma4LayerMatrixSource::from_det_num_source(source)
    }

    fn write_det_matrices(
        matrices: Vec<Vec<Vec<Wgt>>>,
    ) -> Result<(PathBuf, Vec<DetNumTensorSliceSource>)> {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-prefill-layer-{}-{}-{}.detwgt",
            std::process::id(),
            unique_suffix,
            crate::trace::sha256_hex(&format!("{:?}", matrices))
        ));
        let mut bytes = Vec::new();
        let mut sources = Vec::new();

        for rows in matrices {
            let data_offset = bytes.len();
            for row in &rows {
                for value in row {
                    bytes.extend(value.to_bits().to_le_bytes());
                }
            }
            sources.push(det_source(&path, rows.len(), rows[0].len(), data_offset));
        }

        std::fs::write(&path, bytes)
            .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
        Ok((path, sources))
    }

    fn det_source(
        path: &Path,
        rows: usize,
        cols: usize,
        data_offset: usize,
    ) -> DetNumTensorSliceSource {
        DetNumTensorSliceSource {
            weights_path: path.to_path_buf(),
            total_rows: rows,
            total_cols: cols,
            data_offset,
            row_offset: 0,
            row_count: rows,
            col_offset: 0,
            col_count: cols,
        }
    }
}
