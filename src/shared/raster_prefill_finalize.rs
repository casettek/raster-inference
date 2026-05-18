use std::{cell::RefCell, collections::HashMap, sync::Arc};

use anyhow::{anyhow, bail, Result};
use serde::Serialize;

use crate::shared::artifact_io::AuthRead;
use crate::shared::det_num::{Acc, Act, Wgt};
use crate::shared::external_artifacts::{
    register_external_source_leaves, CommittedExternalSource, ExternalSourceId, ExternalSourceRef,
};
use crate::shared::raster_transformer_kernels::det_num_matrix_row_wgts;
use crate::shared::transformer::{
    DetNumMatrix, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource,
};

const GEMMA_PREFILL_FINALIZE_SOURCE_KIND: &str = "gemma_prefill_finalize";
const GEMMA_PREFILL_FINALIZE_SOURCE_DOMAIN: &str =
    "raster-external-source-gemma-prefill-finalize-merkle-v1";
const GEMMA_PREFILL_FINALIZE_SOURCE_CHUNK_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedGemmaPrefillFinalizeSource {
    identifier: String,
    metadata: GemmaPrefillFinalizeMetadata,
    final_norm_weights: Vec<Wgt>,
    scalars: GemmaPrefillFinalizeScalars,
    projection: GemmaPrefillFinalizeProjectionBacking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GemmaPrefillFinalizeProjectionBacking {
    Matrix(Arc<DetNumMatrix>),
}

#[derive(Serialize)]
struct GemmaPrefillFinalizeSourcePayload<'a> {
    identifier: &'a str,
    metadata: &'a GemmaPrefillFinalizeMetadata,
    final_norm_weight_bits: Vec<i32>,
    rms_norm_eps_bits: i64,
    final_logit_softcapping_bits: Option<i32>,
    projection: GemmaPrefillFinalizeSourcePayloadProjection,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum GemmaPrefillFinalizeSourcePayloadProjection {
    Matrix {
        rows: usize,
        cols: usize,
        values: Vec<i32>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPrefillFinalizeMetadata {
    pub source_id: String,
    pub hidden_width: usize,
    pub projection_rows: usize,
    pub projection_cols: usize,
    pub projection_kind: GemmaPrefillFinalizeProjectionKind,
    pub has_final_logit_softcapping: bool,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum GemmaPrefillFinalizeProjectionKind {
    UntiedLmHead,
    TiedEmbedding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillFinalizeScalars {
    pub rms_norm_eps: Acc,
    pub final_logit_softcapping: Option<Act>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillFinalizeMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillFinalizeNormWeightsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillFinalizeScalarsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPrefillFinalizeProjectionRowRequest {
    pub row_idx: usize,
}

impl AuthenticatedGemmaPrefillFinalizeSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        if model.provenance != Gemma4ModelProvenance::DetNumWgt {
            bail!(
                "deterministic raster prefill finalize source requires a model loaded from a .detwgt artifact"
            );
        }

        let final_norm_weights = canonical_final_norm_weights(&model.final_norm_weight_det)?;
        let rms_norm_eps = model.rms_norm_eps_det.ok_or_else(|| {
            anyhow!("deterministic raster prefill finalize requires canonical RMSNorm epsilon")
        })?;
        let final_logit_softcapping = if model.final_logit_softcapping.is_some() {
            Some(model.final_logit_softcapping_det.ok_or_else(|| {
                anyhow!(
                    "deterministic raster prefill finalize requires canonical final logit softcap"
                )
            })?)
        } else {
            None
        };

        let (projection_kind, projection_rows, projection_cols, projection) =
            canonical_projection_backing(model)?;
        let hidden_width = final_norm_weights.len();
        if projection_cols != hidden_width {
            bail!(
                "Gemma prefill finalize projection width {projection_cols} does not match final norm width {hidden_width}"
            );
        }

        Ok(Self {
            metadata: GemmaPrefillFinalizeMetadata {
                source_id: identifier.clone(),
                hidden_width,
                projection_rows,
                projection_cols,
                projection_kind,
                has_final_logit_softcapping: final_logit_softcapping.is_some(),
            },
            identifier,
            final_norm_weights,
            scalars: GemmaPrefillFinalizeScalars {
                rms_norm_eps,
                final_logit_softcapping,
            },
            projection,
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn committed_source_ref(&self) -> Result<ExternalSourceRef> {
        let source_ref = register_external_source_leaves(
            ExternalSourceId::new(format!("prefill-finalize:{}", self.identifier))?,
            GEMMA_PREFILL_FINALIZE_SOURCE_KIND,
            GEMMA_PREFILL_FINALIZE_SOURCE_DOMAIN,
            source_payload_chunks(&self.source_payload()),
        )?;
        register_native_committed_prefill_finalize(source_ref.root(), self)?;
        Ok(source_ref)
    }

    pub fn committed_source(&self) -> Result<CommittedExternalSource> {
        Ok(CommittedExternalSource::new(self.committed_source_ref()?))
    }

    fn source_payload(&self) -> Vec<u8> {
        let projection = match &self.projection {
            GemmaPrefillFinalizeProjectionBacking::Matrix(matrix) => {
                GemmaPrefillFinalizeSourcePayloadProjection::Matrix {
                    rows: matrix.rows,
                    cols: matrix.cols,
                    values: matrix.values.clone(),
                }
            }
        };
        serde_json::to_vec(&GemmaPrefillFinalizeSourcePayload {
            identifier: &self.identifier,
            metadata: &self.metadata,
            final_norm_weight_bits: self
                .final_norm_weights
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            rms_norm_eps_bits: self.scalars.rms_norm_eps.to_bits(),
            final_logit_softcapping_bits: self
                .scalars
                .final_logit_softcapping
                .map(|value| value.to_bits()),
            projection,
        })
        .expect("canonical prefill finalize source payload should serialize")
    }
}

impl AuthRead<GemmaPrefillFinalizeMetadataRequest> for AuthenticatedGemmaPrefillFinalizeSource {
    type Output = GemmaPrefillFinalizeMetadata;

    fn auth_read(&self, _request: GemmaPrefillFinalizeMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata.clone())
    }
}

impl AuthRead<GemmaPrefillFinalizeNormWeightsRequest> for AuthenticatedGemmaPrefillFinalizeSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, _request: GemmaPrefillFinalizeNormWeightsRequest) -> Result<Self::Output> {
        Ok(self.final_norm_weights.clone())
    }
}

impl AuthRead<GemmaPrefillFinalizeScalarsRequest> for AuthenticatedGemmaPrefillFinalizeSource {
    type Output = GemmaPrefillFinalizeScalars;

    fn auth_read(&self, _request: GemmaPrefillFinalizeScalarsRequest) -> Result<Self::Output> {
        Ok(self.scalars)
    }
}

impl AuthRead<GemmaPrefillFinalizeProjectionRowRequest>
    for AuthenticatedGemmaPrefillFinalizeSource
{
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeProjectionRowRequest) -> Result<Self::Output> {
        match &self.projection {
            GemmaPrefillFinalizeProjectionBacking::Matrix(matrix) => {
                matrix_row_wgts(matrix, request.row_idx, "lm_head")
            }
        }
    }
}

impl AuthRead<GemmaPrefillFinalizeMetadataRequest> for CommittedExternalSource {
    type Output = GemmaPrefillFinalizeMetadata;

    fn auth_read(&self, request: GemmaPrefillFinalizeMetadataRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self.root(), |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaPrefillFinalizeNormWeightsRequest> for CommittedExternalSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeNormWeightsRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self.root(), |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaPrefillFinalizeScalarsRequest> for CommittedExternalSource {
    type Output = GemmaPrefillFinalizeScalars;

    fn auth_read(&self, request: GemmaPrefillFinalizeScalarsRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self.root(), |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaPrefillFinalizeProjectionRowRequest> for CommittedExternalSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeProjectionRowRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self.root(), |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaPrefillFinalizeMetadataRequest> for str {
    type Output = GemmaPrefillFinalizeMetadata;

    fn auth_read(&self, request: GemmaPrefillFinalizeMetadataRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self, |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaPrefillFinalizeNormWeightsRequest> for str {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeNormWeightsRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self, |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaPrefillFinalizeScalarsRequest> for str {
    type Output = GemmaPrefillFinalizeScalars;

    fn auth_read(&self, request: GemmaPrefillFinalizeScalarsRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self, |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaPrefillFinalizeProjectionRowRequest> for str {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeProjectionRowRequest) -> Result<Self::Output> {
        with_native_committed_prefill_finalize(self, |source| source.auth_read(request))
    }
}

thread_local! {
    static NATIVE_COMMITTED_PREFILL_FINALIZE_SOURCES: RefCell<HashMap<String, AuthenticatedGemmaPrefillFinalizeSource>> =
        RefCell::new(HashMap::new());
}

fn register_native_committed_prefill_finalize(
    root: &str,
    source: &AuthenticatedGemmaPrefillFinalizeSource,
) -> Result<()> {
    NATIVE_COMMITTED_PREFILL_FINALIZE_SOURCES.with(|sources_ref| {
        let mut sources = sources_ref.borrow_mut();
        match sources.get(root) {
            Some(existing) if existing != source => {
                bail!("committed prefill finalize root {root} is already registered with different data")
            }
            Some(_) => Ok(()),
            None => {
                sources.insert(root.to_string(), source.clone());
                Ok(())
            }
        }
    })
}

fn with_native_committed_prefill_finalize<T>(
    root: &str,
    f: impl FnOnce(&AuthenticatedGemmaPrefillFinalizeSource) -> Result<T>,
) -> Result<T> {
    NATIVE_COMMITTED_PREFILL_FINALIZE_SOURCES.with(|sources_ref| {
        let sources = sources_ref.borrow();
        let source = sources.get(root).ok_or_else(|| {
            anyhow!("committed prefill finalize root {root} is not registered natively")
        })?;
        f(source)
    })
}

fn validate_identifier(identifier: String) -> Result<String> {
    if identifier.is_empty() {
        bail!("Gemma prefill finalize source identifier must not be empty");
    }
    Ok(identifier)
}

fn canonical_final_norm_weights(weights: &Option<Vec<Wgt>>) -> Result<Vec<Wgt>> {
    let weights = weights.clone().ok_or_else(|| {
        anyhow!("deterministic raster prefill finalize requires canonical final norm weights")
    })?;
    if weights.is_empty() {
        bail!("Gemma prefill finalize final norm weights must not be empty");
    }
    Ok(weights)
}

fn canonical_projection_backing(
    model: &Gemma4TransformerModel,
) -> Result<(
    GemmaPrefillFinalizeProjectionKind,
    usize,
    usize,
    GemmaPrefillFinalizeProjectionBacking,
)> {
    match &model.logits_projection {
        Gemma4LogitsProjection::UntiedLmHead {
            det_weight: Some(det_weight),
            ..
        } => {
            validate_det_matrix_shape(det_weight, "lm_head")?;
            Ok((
                GemmaPrefillFinalizeProjectionKind::UntiedLmHead,
                det_weight.rows,
                det_weight.cols,
                GemmaPrefillFinalizeProjectionBacking::Matrix(det_weight.clone()),
            ))
        }
        Gemma4LogitsProjection::UntiedLmHead {
            det_weight: None, ..
        } => bail!("deterministic raster prefill finalize requires canonical lm_head det_weight"),
        Gemma4LogitsProjection::TiedEmbedding(_) => {
            let matrix = match model.embedding_source.as_ref() {
                Some(GemmaEmbeddingTensorSource::Deterministic { .. }) => {
                    crate::io::materialize_det_num_embedding_matrix(
                        model
                            .embedding_source
                            .as_ref()
                            .expect("embedding source should exist"),
                    )?
                    .ok_or_else(|| {
                        anyhow!("deterministic raster tied embedding logits require canonical embedding matrix")
                    })?
                }
                Some(_) | None => {
                    bail!(
                        "deterministic raster tied embedding logits require a .detwgt embedding source"
                    )
                }
            };
            validate_det_matrix_shape(&matrix, "tied embedding")?;
            Ok((
                GemmaPrefillFinalizeProjectionKind::TiedEmbedding,
                matrix.rows,
                matrix.cols,
                GemmaPrefillFinalizeProjectionBacking::Matrix(matrix),
            ))
        }
    }
}

fn validate_det_matrix_shape(matrix: &DetNumMatrix, label: &str) -> Result<()> {
    if matrix.rows == 0 || matrix.cols == 0 {
        bail!("Gemma prefill finalize {label} matrix must have non-zero shape");
    }
    let expected_len = matrix
        .rows
        .checked_mul(matrix.cols)
        .ok_or_else(|| anyhow!("Gemma prefill finalize {label} matrix shape overflowed"))?;
    if matrix.values.len() != expected_len {
        bail!(
            "Gemma prefill finalize {label} matrix has {} values, expected {expected_len}",
            matrix.values.len()
        );
    }
    Ok(())
}

fn matrix_row_wgts(matrix: &DetNumMatrix, row_idx: usize, label: &str) -> Result<Vec<Wgt>> {
    det_num_matrix_row_wgts(matrix, row_idx, label)
}

fn source_payload_chunks(payload: &[u8]) -> Vec<Vec<u8>> {
    payload
        .chunks(GEMMA_PREFILL_FINALIZE_SOURCE_CHUNK_BYTES)
        .map(|chunk| chunk.to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedGemmaPrefillFinalizeSource, GemmaPrefillFinalizeMetadataRequest,
        GemmaPrefillFinalizeNormWeightsRequest, GemmaPrefillFinalizeProjectionKind,
        GemmaPrefillFinalizeProjectionRowRequest, GemmaPrefillFinalizeScalarsRequest,
    };
    use crate::shared::det_num::{Acc, Wgt};
    use crate::shared::transformer::{
        DetNumMatrix, DetNumTensorSliceSource, Gemma4LogitsProjection, Gemma4ModelProvenance,
        Gemma4TransformerModel, GemmaEmbeddingTensorSource, MatrixF32,
    };
    use anyhow::{Context, Result};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    #[test]
    fn untied_source_reads_metadata_scalars_norm_and_projection_rows() {
        let model = untied_model();
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");

        let metadata = crate::auth_read!(&source, GemmaPrefillFinalizeMetadataRequest)
            .expect("metadata should read");
        assert_eq!(source.identifier(), "finalize");
        assert_eq!(metadata.source_id, "finalize");
        assert_eq!(metadata.hidden_width, 2);
        assert_eq!(metadata.projection_rows, 2);
        assert_eq!(metadata.projection_cols, 2);
        assert_eq!(
            metadata.projection_kind,
            GemmaPrefillFinalizeProjectionKind::UntiedLmHead
        );
        assert!(!metadata.has_final_logit_softcapping);

        let norm = crate::auth_read!(&source, GemmaPrefillFinalizeNormWeightsRequest)
            .expect("norm should read");
        assert_eq!(norm, vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]);

        let scalars = crate::auth_read!(&source, GemmaPrefillFinalizeScalarsRequest)
            .expect("scalars should read");
        assert_eq!(scalars.rms_norm_eps, Acc::from_num(0.001));
        assert_eq!(scalars.final_logit_softcapping, None);

        let row = crate::auth_read!(
            &source,
            GemmaPrefillFinalizeProjectionRowRequest { row_idx: 1 },
        )
        .expect("projection row should read");
        assert_eq!(row, vec![Wgt::from_num(0.0), Wgt::from_num(1.0)]);
    }

    #[test]
    fn committed_source_root_reads_metadata_scalars_norm_and_projection_rows() {
        let model = untied_model();
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("committed", &model)
            .expect("source should build");
        let committed = source.committed_source().expect("source should commit");
        let root = committed.root().to_string();

        let direct_metadata = crate::auth_read!(&source, GemmaPrefillFinalizeMetadataRequest)
            .expect("direct metadata should read");
        let committed_metadata =
            crate::auth_read!(root.as_str(), GemmaPrefillFinalizeMetadataRequest)
                .expect("committed metadata should read");
        assert_eq!(committed_metadata, direct_metadata);
        assert_eq!(
            crate::auth_read!(&committed, GemmaPrefillFinalizeNormWeightsRequest)
                .expect("committed norm should read"),
            vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]
        );
        assert_eq!(
            crate::auth_read!(
                root.as_str(),
                GemmaPrefillFinalizeProjectionRowRequest { row_idx: 1 },
            )
            .expect("committed projection row should read"),
            vec![Wgt::from_num(0.0), Wgt::from_num(1.0)]
        );
        assert!(crate::auth_read!(
            "missing-prefill-finalize-root",
            GemmaPrefillFinalizeScalarsRequest
        )
        .is_err());
    }

    #[test]
    fn committed_source_rejects_same_id_with_different_data() {
        let source =
            AuthenticatedGemmaPrefillFinalizeSource::from_model("conflict", &untied_model())
                .expect("source should build");
        source
            .committed_source_ref()
            .expect("first source should commit");
        let mut model = untied_model();
        model.final_norm_weight_det = Some(vec![Wgt::from_num(2.0), Wgt::from_num(0.5)]);
        let conflicting = AuthenticatedGemmaPrefillFinalizeSource::from_model("conflict", &model)
            .expect("conflicting source should build");

        assert!(conflicting.committed_source_ref().is_err());
    }

    #[test]
    fn tied_source_reads_embedding_rows_as_projection_rows() {
        let (_path, model) = tied_model().expect("tied model fixture should build");
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("tied", &model)
            .expect("source should build");

        let metadata = crate::auth_read!(&source, GemmaPrefillFinalizeMetadataRequest)
            .expect("metadata should read");
        assert_eq!(
            metadata.projection_kind,
            GemmaPrefillFinalizeProjectionKind::TiedEmbedding
        );
        assert_eq!(metadata.projection_rows, 2);

        let row = crate::auth_read!(
            &source,
            GemmaPrefillFinalizeProjectionRowRequest { row_idx: 0 },
        )
        .expect("projection row should read");
        assert_eq!(row, vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)]);
    }

    #[test]
    fn tied_source_honors_nonzero_embedding_data_offset() {
        let (path, source) = write_prefixed_det_matrix(
            vec![0xaa; 13],
            vec![
                vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)],
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            ],
        )
        .expect("prefixed tied fixture should build");
        let embedding_source = GemmaEmbeddingTensorSource::Deterministic {
            source,
            scale: 1.0,
            det_cache: Arc::new(Mutex::new(None)),
        };
        let model = base_model(
            Gemma4LogitsProjection::TiedEmbedding(matrix_f32(2, 2)),
            Some(embedding_source),
        );
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("offset", &model)
            .expect("source should build");

        let row = crate::auth_read!(
            &source,
            GemmaPrefillFinalizeProjectionRowRequest { row_idx: 1 },
        )
        .expect("projection row should read");

        assert!(path.exists());
        assert_eq!(row, vec![Wgt::from_num(0.5), Wgt::from_num(1.0)]);
    }

    #[test]
    fn construction_rejects_fp32_model_provenance() {
        let mut model = untied_model();
        model.provenance = Gemma4ModelProvenance::Fp32;

        let error = AuthenticatedGemmaPrefillFinalizeSource::from_model("fp32", &model)
            .expect_err("fp32 model should fail");

        assert!(error.to_string().contains(".detwgt artifact"));
    }

    #[test]
    fn construction_rejects_missing_softcap_scalar() {
        let mut model = untied_model();
        model.final_logit_softcapping = Some(1.0);
        model.final_logit_softcapping_det = None;

        let error = AuthenticatedGemmaPrefillFinalizeSource::from_model("softcap", &model)
            .expect_err("missing softcap scalar should fail");

        assert!(error.to_string().contains("canonical final logit softcap"));
    }

    #[test]
    fn construction_rejects_missing_projection_backing() {
        let mut model = untied_model();
        model.logits_projection = Gemma4LogitsProjection::UntiedLmHead {
            weight: matrix_f32(2, 2),
            det_weight: None,
        };

        let error = AuthenticatedGemmaPrefillFinalizeSource::from_model("projection", &model)
            .expect_err("missing projection should fail");

        assert!(error.to_string().contains("canonical lm_head det_weight"));
    }

    fn untied_model() -> Gemma4TransformerModel {
        base_model(
            Gemma4LogitsProjection::UntiedLmHead {
                weight: matrix_f32(2, 2),
                det_weight: Some(Arc::new(DetNumMatrix {
                    rows: 2,
                    cols: 2,
                    values: vec![
                        Wgt::from_num(1.0).to_bits(),
                        Wgt::from_num(0.0).to_bits(),
                        Wgt::from_num(0.0).to_bits(),
                        Wgt::from_num(1.0).to_bits(),
                    ],
                })),
            },
            None,
        )
    }

    fn tied_model() -> Result<(PathBuf, Gemma4TransformerModel)> {
        let (path, source) = write_det_matrix(vec![
            vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)],
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
        ])?;
        let embedding_source = GemmaEmbeddingTensorSource::Deterministic {
            source,
            scale: 1.0,
            det_cache: Arc::new(Mutex::new(None)),
        };
        Ok((
            path,
            base_model(
                Gemma4LogitsProjection::TiedEmbedding(matrix_f32(2, 2)),
                Some(embedding_source),
            ),
        ))
    }

    fn base_model(
        logits_projection: Gemma4LogitsProjection,
        embedding_source: Option<GemmaEmbeddingTensorSource>,
    ) -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::DetNumWgt,
            embedding_table: None,
            embedding_source,
            layers: vec![],
            ple_global: None,
            final_norm_weight: vec![1.0, 0.5],
            final_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]),
            logits_projection,
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 0.001,
            rms_norm_eps_det: Some(Acc::from_num(0.001)),
        }
    }

    fn matrix_f32(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }

    fn write_det_matrix(rows: Vec<Vec<Wgt>>) -> Result<(PathBuf, DetNumTensorSliceSource)> {
        write_prefixed_det_matrix(Vec::new(), rows)
    }

    fn write_prefixed_det_matrix(
        prefix: Vec<u8>,
        rows: Vec<Vec<Wgt>>,
    ) -> Result<(PathBuf, DetNumTensorSliceSource)> {
        let path = std::env::temp_dir().join(format!(
            "raster-prefill-finalize-source-{}-{}.detwgt",
            std::process::id(),
            crate::trace::sha256_hex(&format!("{:?}{:?}", prefix, rows))
        ));
        let mut bytes = prefix;
        let data_offset = bytes.len();
        for row in &rows {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        std::fs::write(&path, bytes)
            .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
        let source = det_source(&path, rows.len(), rows[0].len(), data_offset);
        Ok((path, source))
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
