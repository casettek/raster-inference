use std::{marker::PhantomData, sync::Arc};

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
    DetNumMatrix, Gemma4LogitsProjection, Gemma4TransformerModel, GemmaEmbeddingTensorSource,
};
use crate::shared::numerics::det_num::{Acc, Act, Wgt};
use crate::shared::raster_kernels::transformer::det_num_matrix_row_wgts;
use anyhow::{anyhow, bail, Result};

const GEMMA_PREFILL_FINALIZE_SOURCE_KIND: &str = "gemma_prefill_finalize";
const GEMMA_PREFILL_FINALIZE_SOURCE_DOMAIN: &str =
    "raster-external-source-gemma-prefill-finalize-merkle-v1";
const PREFILL_FINALIZE_METADATA_REQUEST: &str = "gemma_prefill_finalize.metadata";
const PREFILL_FINALIZE_NORM_WEIGHTS_REQUEST: &str = "gemma_prefill_finalize.norm_weights";
const PREFILL_FINALIZE_SCALARS_REQUEST: &str = "gemma_prefill_finalize.scalars";
const PREFILL_FINALIZE_PROJECTION_ROW_REQUEST: &str = "gemma_prefill_finalize.projection_row";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedDecoderPrefillFinalizeSource {
    identifier: String,
    metadata: GemmaPrefillFinalizeMetadata,
    final_norm_weights: Vec<Wgt>,
    scalars: GemmaPrefillFinalizeScalars,
    projection: GemmaPrefillFinalizeProjectionBacking,
}

#[derive(Debug)]
pub enum RasterPrefillFinalizeSource<'a> {
    Committed {
        source: CommittedExternalSource,
        _marker: PhantomData<&'a AuthenticatedDecoderPrefillFinalizeSource>,
    },
    #[cfg(feature = "unchecked-raster-integrity")]
    DirectUnchecked {
        source: &'a AuthenticatedDecoderPrefillFinalizeSource,
        root: String,
    },
}

impl<'a> RasterPrefillFinalizeSource<'a> {
    pub fn for_current_integrity_mode(
        source: &'a AuthenticatedDecoderPrefillFinalizeSource,
    ) -> Result<Self> {
        #[cfg(feature = "unchecked-raster-integrity")]
        if raster_integrity_is_unchecked() {
            return Ok(Self::DirectUnchecked {
                root: format!(
                    "raster-unchecked-test:direct-prefill-finalize:{}",
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

    pub fn from_committed_root(root: &str) -> Result<Self> {
        Ok(Self::Committed {
            source: CommittedExternalSource::from_root(root)?,
            _marker: PhantomData,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GemmaPrefillFinalizeProjectionBacking {
    Matrix(Arc<DetNumMatrix>),
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

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct GemmaPrefillFinalizeScalarsPayload {
    rms_norm_eps_bits: i64,
    final_logit_softcapping_bits: Option<i32>,
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

impl AuthenticatedDecoderPrefillFinalizeSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
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
        register_external_source(
            ExternalSourceId::new(format!("prefill-finalize:{}", self.identifier))?,
            GEMMA_PREFILL_FINALIZE_SOURCE_KIND,
            GEMMA_PREFILL_FINALIZE_SOURCE_DOMAIN,
            self.committed_source_entries()?,
        )
    }

    pub fn committed_source(&self) -> Result<CommittedExternalSource> {
        Ok(CommittedExternalSource::new(self.committed_source_ref()?))
    }

    fn committed_source_entries(&self) -> Result<Vec<ExternalSourceEntry>> {
        let mut entries = Vec::with_capacity(self.metadata.projection_rows + 3);
        entries.push(postcard_external_source_entry(
            GemmaPrefillFinalizeMetadataRequest.request_key()?,
            &self.metadata,
        )?);
        let norm_bits = self.final_norm_weights.iter().map(|value| value.to_bits());
        entries.push(postcard_i32_vec_external_source_entry(
            GemmaPrefillFinalizeNormWeightsRequest.request_key()?,
            norm_bits,
        )?);
        entries.push(postcard_external_source_entry(
            GemmaPrefillFinalizeScalarsRequest.request_key()?,
            &self.scalars.payload(),
        )?);
        for row_idx in 0..self.metadata.projection_rows {
            let request = GemmaPrefillFinalizeProjectionRowRequest { row_idx };
            let row_bits = self
                .auth_read(request)?
                .into_iter()
                .map(|value| value.to_bits());
            entries.push(postcard_i32_vec_external_source_entry(
                request.request_key()?,
                row_bits,
            )?);
        }
        Ok(entries)
    }
}

impl AuthRead<GemmaPrefillFinalizeMetadataRequest> for AuthenticatedDecoderPrefillFinalizeSource {
    type Output = GemmaPrefillFinalizeMetadata;

    fn auth_read(&self, _request: GemmaPrefillFinalizeMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata.clone())
    }
}

impl AuthRead<GemmaPrefillFinalizeMetadataRequest> for RasterPrefillFinalizeSource<'_> {
    type Output = GemmaPrefillFinalizeMetadata;

    fn auth_read(&self, request: GemmaPrefillFinalizeMetadataRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.auth_read(request),
        }
    }
}

impl AuthRead<GemmaPrefillFinalizeNormWeightsRequest>
    for AuthenticatedDecoderPrefillFinalizeSource
{
    type Output = Vec<Wgt>;

    fn auth_read(&self, _request: GemmaPrefillFinalizeNormWeightsRequest) -> Result<Self::Output> {
        Ok(self.final_norm_weights.clone())
    }
}

impl AuthRead<GemmaPrefillFinalizeNormWeightsRequest> for RasterPrefillFinalizeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeNormWeightsRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.auth_read(request),
        }
    }
}

impl AuthRead<GemmaPrefillFinalizeScalarsRequest> for AuthenticatedDecoderPrefillFinalizeSource {
    type Output = GemmaPrefillFinalizeScalars;

    fn auth_read(&self, _request: GemmaPrefillFinalizeScalarsRequest) -> Result<Self::Output> {
        Ok(self.scalars)
    }
}

impl AuthRead<GemmaPrefillFinalizeScalarsRequest> for RasterPrefillFinalizeSource<'_> {
    type Output = GemmaPrefillFinalizeScalars;

    fn auth_read(&self, request: GemmaPrefillFinalizeScalarsRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.auth_read(request),
        }
    }
}

impl AuthRead<GemmaPrefillFinalizeProjectionRowRequest>
    for AuthenticatedDecoderPrefillFinalizeSource
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

impl AuthRead<GemmaPrefillFinalizeProjectionRowRequest> for RasterPrefillFinalizeSource<'_> {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeProjectionRowRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.auth_read(request),
        }
    }
}

impl CommittedExternalRequest for GemmaPrefillFinalizeMetadataRequest {
    type Output = GemmaPrefillFinalizeMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PREFILL_FINALIZE_METADATA_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl CommittedExternalRequest for GemmaPrefillFinalizeNormWeightsRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PREFILL_FINALIZE_NORM_WEIGHTS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaPrefillFinalizeScalarsRequest {
    type Output = GemmaPrefillFinalizeScalars;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PREFILL_FINALIZE_SCALARS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(GemmaPrefillFinalizeScalars::from_payload(
            decode_postcard_response(response_payload)?,
        ))
    }
}

impl CommittedExternalRequest for GemmaPrefillFinalizeProjectionRowRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PREFILL_FINALIZE_PROJECTION_ROW_REQUEST, &self.row_idx)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl AuthRead<GemmaPrefillFinalizeMetadataRequest> for str {
    type Output = GemmaPrefillFinalizeMetadata;

    fn auth_read(&self, request: GemmaPrefillFinalizeMetadataRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl AuthRead<GemmaPrefillFinalizeNormWeightsRequest> for str {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeNormWeightsRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl AuthRead<GemmaPrefillFinalizeScalarsRequest> for str {
    type Output = GemmaPrefillFinalizeScalars;

    fn auth_read(&self, request: GemmaPrefillFinalizeScalarsRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl AuthRead<GemmaPrefillFinalizeProjectionRowRequest> for str {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPrefillFinalizeProjectionRowRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl GemmaPrefillFinalizeScalars {
    fn payload(self) -> GemmaPrefillFinalizeScalarsPayload {
        GemmaPrefillFinalizeScalarsPayload {
            rms_norm_eps_bits: self.rms_norm_eps.to_bits(),
            final_logit_softcapping_bits: self.final_logit_softcapping.map(|value| value.to_bits()),
        }
    }

    fn from_payload(payload: GemmaPrefillFinalizeScalarsPayload) -> Self {
        Self {
            rms_norm_eps: Acc::from_bits(payload.rms_norm_eps_bits),
            final_logit_softcapping: payload.final_logit_softcapping_bits.map(Act::from_bits),
        }
    }
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

#[cfg(test)]
mod tests;
