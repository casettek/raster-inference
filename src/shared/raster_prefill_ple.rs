use anyhow::{anyhow, bail, Result};

use crate::raster_authoring::AuthRead;
use crate::shared::det_num::{Acc, Act, Wgt};
use crate::shared::raster_row_store::RasterActivationSequenceRef;
use crate::shared::raster_transformer_kernels::det_num_matrix_row_wgts;
use crate::shared::transformer::{
    Gemma4ModelProvenance, Gemma4PleGlobalWeights, Gemma4PleMatrixSource, Gemma4TransformerModel,
};

#[derive(Debug, Clone)]
pub struct AuthenticatedGemmaPleSource {
    identifier: String,
    layers: Vec<GemmaPleLayerMetadata>,
    projection_norm_weights: Option<Vec<Wgt>>,
    scalars: Option<GemmaPleScalars>,
    backing: GemmaPleBacking,
}

#[derive(Debug, Clone)]
enum GemmaPleBacking {
    None,
    Owned {
        token_embeddings: Vec<Vec<Vec<Act>>>,
        model_projections: Vec<Vec<Vec<Wgt>>>,
    },
    Model(Gemma4PleGlobalWeights),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPleLayerConfig {
    pub has_ple: bool,
    pub hidden_width: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPleMetadata {
    pub source_id: String,
    pub has_ple_global: bool,
    pub layer_count: usize,
    pub token_embedding_layer_count: usize,
    pub model_projection_layer_count: usize,
    pub projection_norm_width: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillPleInputRefs {
    source_id: String,
    layer_count: usize,
    token_count: usize,
    per_layer_inputs: Vec<Option<RasterActivationSequenceRef>>,
}

impl RasterPrefillPleInputRefs {
    pub fn new(
        source_id: impl Into<String>,
        layer_count: usize,
        token_count: usize,
        per_layer_inputs: Vec<Option<RasterActivationSequenceRef>>,
    ) -> Result<Self> {
        let source_id = validate_identifier(source_id.into())?;
        if layer_count == 0 {
            bail!("raster PLE input refs require at least one layer");
        }
        if token_count == 0 {
            bail!("raster PLE input refs require at least one token");
        }
        if per_layer_inputs.len() != layer_count {
            bail!(
                "raster PLE input refs received {} layers, expected {layer_count}",
                per_layer_inputs.len()
            );
        }
        Ok(Self {
            source_id,
            layer_count,
            token_count,
            per_layer_inputs,
        })
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn per_layer_inputs(&self) -> &[Option<RasterActivationSequenceRef>] {
        &self.per_layer_inputs
    }

    pub fn clone_layer_ref(&self, layer_idx: usize) -> Result<Option<RasterActivationSequenceRef>> {
        self.per_layer_inputs
            .get(layer_idx)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "raster PLE input ref layer {layer_idx} is out of range for {} layers",
                    self.layer_count
                )
            })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPleLayerMetadata {
    pub layer_idx: usize,
    pub has_ple: bool,
    pub hidden_width: usize,
    pub token_embedding_vocab_size: Option<usize>,
    pub ple_width: Option<usize>,
    pub model_projection_rows: Option<usize>,
    pub model_projection_cols: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleScalars {
    pub embedding_scale: Act,
    pub projection_scalar: Act,
    pub input_scale: Act,
    pub rms_norm_eps: Acc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleLayerMetadataRequest {
    pub layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleTokenEmbeddingRowRequest {
    pub layer_idx: usize,
    pub token_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleModelProjectionRowRequest {
    pub layer_idx: usize,
    pub row_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleProjectionNormWeightsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleScalarsRequest;

impl AuthenticatedGemmaPleSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        let layers = model
            .layers
            .iter()
            .map(|layer| GemmaPleLayerConfig {
                has_ple: layer.ple.is_some(),
                hidden_width: layer.hidden_size,
            })
            .collect();
        Self::from_ple_global(
            identifier,
            model.provenance,
            layers,
            model.ple_global.clone(),
            model.rms_norm_eps_det,
        )
    }

    pub fn from_ple_global(
        identifier: impl Into<String>,
        provenance: Gemma4ModelProvenance,
        layer_configs: Vec<GemmaPleLayerConfig>,
        ple_global: Option<Gemma4PleGlobalWeights>,
        rms_norm_eps_det: Option<Acc>,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        let has_ple_layer = layer_configs.iter().any(|layer| layer.has_ple);
        let Some(ple_global) = ple_global else {
            if has_ple_layer {
                bail!("Gemma PLE source has PLE layers but no global PLE weights");
            }
            let layers = layer_configs
                .into_iter()
                .enumerate()
                .map(|(layer_idx, config)| GemmaPleLayerMetadata {
                    layer_idx,
                    has_ple: config.has_ple,
                    hidden_width: config.hidden_width,
                    token_embedding_vocab_size: None,
                    ple_width: None,
                    model_projection_rows: None,
                    model_projection_cols: None,
                })
                .collect();
            return Ok(Self {
                identifier,
                layers,
                projection_norm_weights: None,
                scalars: None,
                backing: GemmaPleBacking::None,
            });
        };

        if provenance != Gemma4ModelProvenance::DetNumWgt {
            bail!(
                "deterministic raster PLE source requires a model loaded from a .detwgt artifact"
            );
        }

        ensure_model_backing_is_canonical(&ple_global)?;
        let projection_norm_weights = canonical_projection_norm_weights(&ple_global)?;
        let scalars = canonical_scalars(&ple_global, rms_norm_eps_det)?;
        let layers = build_layer_metadata(
            layer_configs,
            &ple_global.token_embeddings,
            &ple_global.model_projections,
            projection_norm_weights.len(),
        )?;

        Ok(Self {
            identifier,
            layers,
            projection_norm_weights: Some(projection_norm_weights),
            scalars: Some(scalars),
            backing: GemmaPleBacking::Model(ple_global),
        })
    }

    pub fn from_canonical_parts(
        identifier: impl Into<String>,
        layer_configs: Vec<GemmaPleLayerConfig>,
        token_embeddings: Vec<Vec<Vec<Act>>>,
        model_projections: Vec<Vec<Vec<Wgt>>>,
        projection_norm_weights: Vec<Wgt>,
        scalars: GemmaPleScalars,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        let layers = build_owned_layer_metadata(
            layer_configs,
            &token_embeddings,
            &model_projections,
            projection_norm_weights.len(),
        )?;

        Ok(Self {
            identifier,
            layers,
            projection_norm_weights: Some(projection_norm_weights),
            scalars: Some(scalars),
            backing: GemmaPleBacking::Owned {
                token_embeddings,
                model_projections,
            },
        })
    }

    pub fn no_ple(
        identifier: impl Into<String>,
        layer_configs: Vec<GemmaPleLayerConfig>,
    ) -> Result<Self> {
        Self::from_ple_global(
            identifier,
            Gemma4ModelProvenance::DetNumWgt,
            layer_configs,
            None,
            None,
        )
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    fn metadata(&self) -> GemmaPleMetadata {
        let has_ple_global = !matches!(self.backing, GemmaPleBacking::None);
        GemmaPleMetadata {
            source_id: self.identifier.clone(),
            has_ple_global,
            layer_count: self.layers.len(),
            token_embedding_layer_count: if has_ple_global { self.layers.len() } else { 0 },
            model_projection_layer_count: if has_ple_global { self.layers.len() } else { 0 },
            projection_norm_width: self.projection_norm_weights.as_ref().map(Vec::len),
        }
    }

    fn layer_metadata(&self, layer_idx: usize) -> Result<GemmaPleLayerMetadata> {
        self.layers.get(layer_idx).cloned().ok_or_else(|| {
            anyhow!(
                "Gemma PLE layer index {layer_idx} is out of range for {} layers",
                self.layers.len()
            )
        })
    }

    fn require_ple_global(&self) -> Result<()> {
        if matches!(self.backing, GemmaPleBacking::None) {
            bail!("Gemma PLE source has no global PLE weights");
        }
        Ok(())
    }
}

impl AuthRead<GemmaPleMetadataRequest> for AuthenticatedGemmaPleSource {
    type Output = GemmaPleMetadata;

    fn auth_read(&self, _request: GemmaPleMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata())
    }
}

impl AuthRead<GemmaPleLayerMetadataRequest> for AuthenticatedGemmaPleSource {
    type Output = GemmaPleLayerMetadata;

    fn auth_read(&self, request: GemmaPleLayerMetadataRequest) -> Result<Self::Output> {
        self.layer_metadata(request.layer_idx)
    }
}

impl AuthRead<GemmaPleTokenEmbeddingRowRequest> for AuthenticatedGemmaPleSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaPleTokenEmbeddingRowRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        let row_idx = usize::try_from(request.token_id).expect("u32 should fit into usize");
        match &self.backing {
            GemmaPleBacking::None => unreachable!("require_ple_global checked this case"),
            GemmaPleBacking::Owned {
                token_embeddings, ..
            } => token_embeddings
                .get(request.layer_idx)
                .ok_or_else(|| {
                    anyhow!(
                        "transformer PLE token embedding slice count mismatch at layer {}",
                        request.layer_idx
                    )
                })?
                .get(row_idx)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "Gemma PLE token id {} is out of range for layer {}",
                        request.token_id,
                        request.layer_idx
                    )
                }),
            GemmaPleBacking::Model(ple_global) => {
                let row = crate::io::load_ple_token_embedding_row_internal(
                    ple_global,
                    request.layer_idx,
                    request.token_id,
                )?;
                row.det_values().map(<[Act]>::to_vec).ok_or_else(|| {
                    anyhow!("deterministic raster PLE token row requires canonical Act values")
                })
            }
        }
    }
}

impl AuthRead<GemmaPleModelProjectionRowRequest> for AuthenticatedGemmaPleSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPleModelProjectionRowRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        match &self.backing {
            GemmaPleBacking::None => unreachable!("require_ple_global checked this case"),
            GemmaPleBacking::Owned {
                model_projections, ..
            } => model_projections
                .get(request.layer_idx)
                .ok_or_else(|| {
                    anyhow!(
                        "transformer PLE model projection slice count mismatch at layer {}",
                        request.layer_idx
                    )
                })?
                .get(request.row_idx)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "Gemma PLE model projection row {} is out of range for layer {}",
                        request.row_idx,
                        request.layer_idx
                    )
                }),
            GemmaPleBacking::Model(ple_global) => {
                let matrix = crate::io::materialize_det_num_ple_model_projection(
                    ple_global,
                    request.layer_idx,
                )?
                .ok_or_else(|| {
                    anyhow!(
                        "deterministic raster PLE model projection layer {} requires canonical matrix",
                        request.layer_idx
                    )
                })?;
                det_num_matrix_row_wgts(&matrix, request.row_idx, "PLE model projection")
            }
        }
    }
}

impl AuthRead<GemmaPleProjectionNormWeightsRequest> for AuthenticatedGemmaPleSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, _request: GemmaPleProjectionNormWeightsRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        self.projection_norm_weights.clone().ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical norm weights")
        })
    }
}

impl AuthRead<GemmaPleScalarsRequest> for AuthenticatedGemmaPleSource {
    type Output = GemmaPleScalars;

    fn auth_read(&self, _request: GemmaPleScalarsRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        self.scalars
            .ok_or_else(|| anyhow!("deterministic raster PLE source requires canonical scalars"))
    }
}

fn validate_identifier(identifier: String) -> Result<String> {
    if identifier.is_empty() {
        bail!("Gemma PLE source identifier must not be empty");
    }
    Ok(identifier)
}

fn ensure_model_backing_is_canonical(ple_global: &Gemma4PleGlobalWeights) -> Result<()> {
    for (layer_idx, source) in ple_global.token_embeddings.iter().enumerate() {
        if !matches!(source, Gemma4PleMatrixSource::DetNumLazy(_)) {
            bail!(
                "deterministic raster PLE source requires .detwgt token embedding source at layer {layer_idx}"
            );
        }
    }
    for (layer_idx, source) in ple_global.model_projections.iter().enumerate() {
        if !matches!(source, Gemma4PleMatrixSource::DetNumLazy(_)) {
            bail!(
                "deterministic raster PLE source requires .detwgt model projection source at layer {layer_idx}"
            );
        }
    }
    Ok(())
}

fn canonical_projection_norm_weights(ple_global: &Gemma4PleGlobalWeights) -> Result<Vec<Wgt>> {
    ple_global
        .projection_norm_weight_det
        .clone()
        .ok_or_else(|| anyhow!("deterministic raster PLE source requires canonical norm weights"))
}

fn canonical_scalars(
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps_det: Option<Acc>,
) -> Result<GemmaPleScalars> {
    Ok(GemmaPleScalars {
        embedding_scale: ple_global.embedding_scale_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical embedding scale")
        })?,
        projection_scalar: ple_global.projection_scalar_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical projection scalar")
        })?,
        input_scale: ple_global.input_scale_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical input scale")
        })?,
        rms_norm_eps: rms_norm_eps_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical RMSNorm epsilon")
        })?,
    })
}

fn build_layer_metadata(
    layer_configs: Vec<GemmaPleLayerConfig>,
    token_embeddings: &[Gemma4PleMatrixSource],
    model_projections: &[Gemma4PleMatrixSource],
    projection_norm_width: usize,
) -> Result<Vec<GemmaPleLayerMetadata>> {
    if token_embeddings.len() != layer_configs.len() {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            token_embeddings.len(),
            layer_configs.len()
        );
    }
    if model_projections.len() != layer_configs.len() {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            model_projections.len(),
            layer_configs.len()
        );
    }

    layer_configs
        .into_iter()
        .enumerate()
        .map(|(layer_idx, config)| {
            let token_shape = source_shape(&token_embeddings[layer_idx]);
            let projection_shape = source_shape(&model_projections[layer_idx]);
            validate_layer_shape(
                layer_idx,
                config,
                token_shape,
                projection_shape,
                projection_norm_width,
            )
        })
        .collect()
}

fn build_owned_layer_metadata(
    layer_configs: Vec<GemmaPleLayerConfig>,
    token_embeddings: &[Vec<Vec<Act>>],
    model_projections: &[Vec<Vec<Wgt>>],
    projection_norm_width: usize,
) -> Result<Vec<GemmaPleLayerMetadata>> {
    if token_embeddings.len() != layer_configs.len() {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            token_embeddings.len(),
            layer_configs.len()
        );
    }
    if model_projections.len() != layer_configs.len() {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            model_projections.len(),
            layer_configs.len()
        );
    }

    layer_configs
        .into_iter()
        .enumerate()
        .map(|(layer_idx, config)| {
            let token_shape = owned_act_matrix_shape(
                token_embeddings
                    .get(layer_idx)
                    .expect("layer count was validated"),
                "PLE token embedding",
                layer_idx,
            )?;
            let projection_shape = owned_wgt_matrix_shape(
                model_projections
                    .get(layer_idx)
                    .expect("layer count was validated"),
                "PLE model projection",
                layer_idx,
            )?;
            validate_layer_shape(
                layer_idx,
                config,
                token_shape,
                projection_shape,
                projection_norm_width,
            )
        })
        .collect()
}

fn validate_layer_shape(
    layer_idx: usize,
    config: GemmaPleLayerConfig,
    token_shape: (usize, usize),
    projection_shape: (usize, usize),
    projection_norm_width: usize,
) -> Result<GemmaPleLayerMetadata> {
    let (token_embedding_vocab_size, ple_width) = token_shape;
    let (model_projection_rows, model_projection_cols) = projection_shape;

    if config.has_ple {
        if ple_width != model_projection_rows {
            bail!(
                "Gemma PLE width mismatch at layer {layer_idx}: token embedding width {ple_width} vs projection rows {model_projection_rows}"
            );
        }
        if model_projection_cols != config.hidden_width {
            bail!(
                "Gemma PLE projection hidden width mismatch at layer {layer_idx}: projection cols {model_projection_cols} vs hidden width {}",
                config.hidden_width
            );
        }
        if projection_norm_width != ple_width {
            bail!(
                "Gemma PLE projection norm width mismatch at layer {layer_idx}: norm width {projection_norm_width} vs PLE width {ple_width}"
            );
        }
    }

    Ok(GemmaPleLayerMetadata {
        layer_idx,
        has_ple: config.has_ple,
        hidden_width: config.hidden_width,
        token_embedding_vocab_size: Some(token_embedding_vocab_size),
        ple_width: Some(ple_width),
        model_projection_rows: Some(model_projection_rows),
        model_projection_cols: Some(model_projection_cols),
    })
}

fn source_shape(source: &Gemma4PleMatrixSource) -> (usize, usize) {
    match source {
        Gemma4PleMatrixSource::Materialized(matrix) => (matrix.rows, matrix.cols),
        Gemma4PleMatrixSource::Lazy(source) => (source.row_count, source.col_count),
        Gemma4PleMatrixSource::DetNumLazy(source) => (source.row_count, source.col_count),
    }
}

fn owned_act_matrix_shape(
    matrix: &[Vec<Act>],
    label: &str,
    layer_idx: usize,
) -> Result<(usize, usize)> {
    let width = rectangular_width(matrix, label, layer_idx)?;
    Ok((matrix.len(), width))
}

fn owned_wgt_matrix_shape(
    matrix: &[Vec<Wgt>],
    label: &str,
    layer_idx: usize,
) -> Result<(usize, usize)> {
    let width = rectangular_width(matrix, label, layer_idx)?;
    Ok((matrix.len(), width))
}

fn rectangular_width<T>(matrix: &[Vec<T>], label: &str, layer_idx: usize) -> Result<usize> {
    let Some(first_row) = matrix.first() else {
        bail!("Gemma {label} matrix at layer {layer_idx} must have at least one row");
    };
    let width = first_row.len();
    if width == 0 {
        bail!("Gemma {label} matrix at layer {layer_idx} must have at least one column");
    }
    if matrix.iter().any(|row| row.len() != width) {
        bail!("Gemma {label} matrix at layer {layer_idx} is not rectangular");
    }
    Ok(width)
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedGemmaPleSource, GemmaPleLayerConfig, GemmaPleLayerMetadataRequest,
        GemmaPleMetadataRequest, GemmaPleModelProjectionRowRequest,
        GemmaPleProjectionNormWeightsRequest, GemmaPleScalars, GemmaPleScalarsRequest,
        GemmaPleTokenEmbeddingRowRequest,
    };
    use crate::shared::det_num::{Acc, Act, Wgt};
    use crate::shared::transformer::{
        DetNumTensorSliceSource, Gemma4ModelProvenance, Gemma4PleGlobalWeights, MatrixF32,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn canonical_source_reads_metadata_and_layer_shape() {
        let source = canonical_source();

        let metadata =
            crate::auth_read!(&source, GemmaPleMetadataRequest).expect("metadata should read");
        assert_eq!(source.identifier(), "ple-fixture");
        assert_eq!(metadata.source_id, "ple-fixture");
        assert!(metadata.has_ple_global);
        assert_eq!(metadata.layer_count, 1);
        assert_eq!(metadata.token_embedding_layer_count, 1);
        assert_eq!(metadata.model_projection_layer_count, 1);
        assert_eq!(metadata.projection_norm_width, Some(2));

        let layer = crate::auth_read!(&source, GemmaPleLayerMetadataRequest { layer_idx: 0 })
            .expect("layer metadata should read");
        assert!(layer.has_ple);
        assert_eq!(layer.hidden_width, 3);
        assert_eq!(layer.token_embedding_vocab_size, Some(2));
        assert_eq!(layer.ple_width, Some(2));
        assert_eq!(layer.model_projection_rows, Some(2));
        assert_eq!(layer.model_projection_cols, Some(3));
    }

    #[test]
    fn canonical_source_reads_token_embedding_rows() {
        let source = canonical_source();

        let row = crate::auth_read!(
            &source,
            GemmaPleTokenEmbeddingRowRequest {
                layer_idx: 0,
                token_id: 1,
            },
        )
        .expect("token embedding row should read");

        assert_eq!(row, vec![Act::from_num(0.25), Act::from_num(-0.5)]);
    }

    #[test]
    fn canonical_source_reads_model_projection_rows() {
        let source = canonical_source();

        let row = crate::auth_read!(
            &source,
            GemmaPleModelProjectionRowRequest {
                layer_idx: 0,
                row_idx: 1,
            },
        )
        .expect("model projection row should read");

        assert_eq!(
            row,
            vec![
                Wgt::from_num(0.5),
                Wgt::from_num(0.25),
                Wgt::from_num(-0.25)
            ]
        );
    }

    #[test]
    fn model_backing_reads_projection_row_from_det_source() {
        let (path, token_source, projection_source) = write_det_sources(
            vec![
                vec![Act::from_num(1.0), Act::from_num(0.0)],
                vec![Act::from_num(0.0), Act::from_num(1.0)],
            ],
            vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0), Wgt::from_num(-1.0)],
                vec![
                    Wgt::from_num(0.5),
                    Wgt::from_num(0.25),
                    Wgt::from_num(-0.25),
                ],
            ],
        );
        let ple_global = Gemma4PleGlobalWeights::from_det_num_sources_with_canonical(
            vec![token_source],
            vec![projection_source],
            vec![1.0, 1.0],
            vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
            1.0,
            Act::from_num(1.0),
            1.0,
            Act::from_num(1.0),
            1.0,
            Act::from_num(1.0),
        );
        let source = AuthenticatedGemmaPleSource::from_ple_global(
            "det-source-ple",
            Gemma4ModelProvenance::DetNumWgt,
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            Some(ple_global),
            Some(Acc::from_num(0.001)),
        )
        .expect("det source should build");

        let row = crate::auth_read!(
            &source,
            GemmaPleModelProjectionRowRequest {
                layer_idx: 0,
                row_idx: 1,
            }
        )
        .expect("projection row should read");

        assert_eq!(
            row,
            vec![
                Wgt::from_num(0.5),
                Wgt::from_num(0.25),
                Wgt::from_num(-0.25)
            ]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn canonical_source_reads_norm_weights_and_scalars() {
        let source = canonical_source();

        let norm = crate::auth_read!(&source, GemmaPleProjectionNormWeightsRequest)
            .expect("norm weights should read");
        assert_eq!(norm, vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]);

        let scalars =
            crate::auth_read!(&source, GemmaPleScalarsRequest).expect("scalars should read");
        assert_eq!(scalars.embedding_scale, Act::from_num(2.0));
        assert_eq!(scalars.projection_scalar, Act::from_num(0.5));
        assert_eq!(scalars.input_scale, Act::from_num(0.25));
        assert_eq!(scalars.rms_norm_eps, Acc::from_num(0.001));
    }

    #[test]
    fn source_with_no_ple_reports_no_ple_state() {
        let source = AuthenticatedGemmaPleSource::no_ple(
            "no-ple",
            vec![GemmaPleLayerConfig {
                has_ple: false,
                hidden_width: 3,
            }],
        )
        .expect("no-PLE source should build");

        let metadata =
            crate::auth_read!(&source, GemmaPleMetadataRequest).expect("metadata should read");
        assert!(!metadata.has_ple_global);
        assert_eq!(metadata.token_embedding_layer_count, 0);
        assert_eq!(metadata.model_projection_layer_count, 0);
        assert_eq!(metadata.projection_norm_width, None);

        let layer = crate::auth_read!(&source, GemmaPleLayerMetadataRequest { layer_idx: 0 })
            .expect("layer metadata should read");
        assert!(!layer.has_ple);
        assert_eq!(layer.ple_width, None);

        let error = crate::auth_read!(&source, GemmaPleProjectionNormWeightsRequest)
            .expect_err("PLE reads should fail without globals");
        assert!(error.to_string().contains("no global PLE weights"));
    }

    #[test]
    fn invalid_layer_index_fails_with_clear_layer_count() {
        let source = canonical_source();

        let error = crate::auth_read!(&source, GemmaPleLayerMetadataRequest { layer_idx: 1 })
            .expect_err("invalid layer should fail");

        assert!(error.to_string().contains("out of range for 1 layers"));
    }

    #[test]
    fn construction_rejects_ple_layers_without_global_weights() {
        let error = AuthenticatedGemmaPleSource::no_ple(
            "bad-no-ple",
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
        )
        .expect_err("PLE layer without globals should fail");

        assert!(error
            .to_string()
            .contains("PLE layers but no global PLE weights"));
    }

    #[test]
    fn construction_rejects_fp32_ple_backing() {
        let ple_global = Gemma4PleGlobalWeights::from_materialized(
            vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            }],
            vec![MatrixF32 {
                rows: 2,
                cols: 3,
                values: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );

        let error = AuthenticatedGemmaPleSource::from_ple_global(
            "fp32-ple",
            Gemma4ModelProvenance::Fp32,
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            Some(ple_global),
            Some(Acc::from_num(0.001)),
        )
        .expect_err("fp32 PLE backing should fail");

        assert!(error.to_string().contains(".detwgt artifact"));
    }

    #[test]
    fn construction_rejects_non_canonical_ple_weights() {
        let ple_global = Gemma4PleGlobalWeights::from_materialized(
            vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            }],
            vec![MatrixF32 {
                rows: 2,
                cols: 3,
                values: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );

        let error = AuthenticatedGemmaPleSource::from_ple_global(
            "non-canonical-ple",
            Gemma4ModelProvenance::DetNumWgt,
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            Some(ple_global),
            Some(Acc::from_num(0.001)),
        )
        .expect_err("non-canonical PLE backing should fail");

        assert!(error.to_string().contains(".detwgt token embedding source"));
    }

    #[test]
    fn construction_rejects_layer_count_mismatch() {
        let error = AuthenticatedGemmaPleSource::from_canonical_parts(
            "bad-shape",
            vec![
                GemmaPleLayerConfig {
                    has_ple: true,
                    hidden_width: 3,
                },
                GemmaPleLayerConfig {
                    has_ple: false,
                    hidden_width: 3,
                },
            ],
            vec![vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]],
            vec![vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0), Wgt::from_num(0.0)],
                vec![Wgt::from_num(0.0), Wgt::from_num(1.0), Wgt::from_num(0.0)],
            ]],
            vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
            test_scalars(),
        )
        .expect_err("layer count mismatch should fail");

        assert!(error
            .to_string()
            .contains("token embedding slice count mismatch: 1 vs 2"));
    }

    fn canonical_source() -> AuthenticatedGemmaPleSource {
        AuthenticatedGemmaPleSource::from_canonical_parts(
            "ple-fixture",
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            vec![vec![
                vec![Act::from_num(1.0), Act::from_num(0.5)],
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
            ]],
            vec![vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0), Wgt::from_num(-1.0)],
                vec![
                    Wgt::from_num(0.5),
                    Wgt::from_num(0.25),
                    Wgt::from_num(-0.25),
                ],
            ]],
            vec![Wgt::from_num(1.0), Wgt::from_num(0.5)],
            test_scalars(),
        )
        .expect("canonical source should build")
    }

    fn test_scalars() -> GemmaPleScalars {
        GemmaPleScalars {
            embedding_scale: Act::from_num(2.0),
            projection_scalar: Act::from_num(0.5),
            input_scale: Act::from_num(0.25),
            rms_norm_eps: Acc::from_num(0.001),
        }
    }

    fn write_det_sources(
        token_embeddings: Vec<Vec<Act>>,
        model_projection: Vec<Vec<Wgt>>,
    ) -> (PathBuf, DetNumTensorSliceSource, DetNumTensorSliceSource) {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-prefill-ple-source-{}-{unique_suffix}.detwgt",
            std::process::id()
        ));
        let mut bytes = Vec::new();
        let token_offset = bytes.len();
        for row in &token_embeddings {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        let projection_offset = bytes.len();
        for row in &model_projection {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        std::fs::write(&path, bytes).expect("fixture weights should write");

        (
            path.clone(),
            det_source(
                &path,
                token_embeddings.len(),
                token_embeddings[0].len(),
                token_offset,
            ),
            det_source(
                &path,
                model_projection.len(),
                model_projection[0].len(),
                projection_offset,
            ),
        )
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
