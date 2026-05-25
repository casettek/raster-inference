use anyhow::{anyhow, bail, Result};

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::external_artifacts::{
    decode_i32_vec_response, decode_postcard_response, postcard_external_source_entry,
    postcard_i32_vec_external_source_entry, postcard_request_key, register_external_source,
    CommittedExternalRequest, CommittedExternalSource, ExternalSourceEntry, ExternalSourceId,
    ExternalSourceRef,
};
use crate::shared::model::transformer::{
    DetNumTensorSliceSource, Gemma4ModelProvenance, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource,
};
use crate::shared::numerics::det_num::{scale_act, Act};
use crate::shared::raster_kernels::transformer::det_num_tensor_slice_row_wgts;

const GEMMA_INPUT_EMBEDDING_SOURCE_KIND: &str = "gemma_input_embedding";
const GEMMA_INPUT_EMBEDDING_SOURCE_DOMAIN: &str =
    "raster-external-source-gemma-input-embedding-merkle-v1";
const INPUT_EMBEDDING_METADATA_REQUEST: &str = "gemma_input_embedding.metadata";
const INPUT_EMBEDDING_ROW_REQUEST: &str = "gemma_input_embedding.row";

#[derive(Debug, Clone, PartialEq)]
pub struct AuthenticatedGemmaInputEmbeddingSource {
    identifier: String,
    vocab_size: usize,
    hidden_size: usize,
    scale: Act,
    backing: GemmaInputEmbeddingBacking,
}

#[derive(Debug, Clone, PartialEq)]
enum GemmaInputEmbeddingBacking {
    Owned(Vec<Vec<Act>>),
    Model(DetNumTensorSliceSource),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaInputEmbeddingMetadata {
    pub source_id: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub scale_bits: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaInputEmbeddingMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaInputEmbeddingRowRequest {
    pub token_id: u32,
}

impl AuthenticatedGemmaInputEmbeddingSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        if model.provenance != Gemma4ModelProvenance::DetNumWgt {
            bail!("deterministic raster input embedding requires a model loaded from a .detwgt artifact");
        }

        let (source, scale) = match model.embedding_source.as_ref() {
            Some(GemmaEmbeddingTensorSource::Deterministic { source, scale, .. }) => {
                (source.clone(), Act::from_num(*scale))
            }
            Some(_) | None => {
                bail!("deterministic raster input embedding requires a .detwgt embedding source")
            }
        };
        validate_full_embedding_source(&source)?;

        Ok(Self {
            identifier: validate_identifier(identifier.into())?,
            vocab_size: source.row_count,
            hidden_size: source.col_count,
            scale,
            backing: GemmaInputEmbeddingBacking::Model(source),
        })
    }

    pub fn from_canonical_rows(
        identifier: impl Into<String>,
        rows: Vec<Vec<Act>>,
        scale: Act,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        if rows.is_empty() {
            bail!("Gemma input embedding source requires at least one row");
        }
        let hidden_size = rows[0].len();
        if hidden_size == 0 {
            bail!("Gemma input embedding source rows must have non-zero width");
        }
        if let Some((row_idx, row)) = rows
            .iter()
            .enumerate()
            .find(|(_, row)| row.len() != hidden_size)
        {
            bail!(
                "Gemma input embedding row {row_idx} has width {}, expected {hidden_size}",
                row.len()
            );
        }

        Ok(Self {
            identifier,
            vocab_size: rows.len(),
            hidden_size,
            scale,
            backing: GemmaInputEmbeddingBacking::Owned(rows),
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn metadata(&self) -> GemmaInputEmbeddingMetadata {
        GemmaInputEmbeddingMetadata {
            source_id: self.identifier.clone(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            scale_bits: self.scale.to_bits(),
        }
    }

    pub fn committed_source_ref(&self) -> Result<ExternalSourceRef> {
        let source_ref = register_external_source(
            ExternalSourceId::new(self.identifier.clone())?,
            GEMMA_INPUT_EMBEDDING_SOURCE_KIND,
            GEMMA_INPUT_EMBEDDING_SOURCE_DOMAIN,
            self.committed_source_entries()?,
        )?;
        Ok(source_ref)
    }

    pub fn committed_source(&self) -> Result<CommittedExternalSource> {
        Ok(CommittedExternalSource::new(self.committed_source_ref()?))
    }

    fn committed_source_entries(&self) -> Result<Vec<ExternalSourceEntry>> {
        let mut entries = Vec::with_capacity(self.vocab_size + 1);
        entries.push(postcard_external_source_entry(
            GemmaInputEmbeddingMetadataRequest.request_key()?,
            &self.metadata(),
        )?);
        for token_id in 0..self.vocab_size {
            let request = GemmaInputEmbeddingRowRequest {
                token_id: u32::try_from(token_id)
                    .map_err(|_| anyhow!("input embedding token id {token_id} exceeds u32"))?,
            };
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

impl AuthRead<GemmaInputEmbeddingMetadataRequest> for AuthenticatedGemmaInputEmbeddingSource {
    type Output = GemmaInputEmbeddingMetadata;

    fn auth_read(&self, _request: GemmaInputEmbeddingMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata())
    }
}

impl AuthRead<GemmaInputEmbeddingRowRequest> for AuthenticatedGemmaInputEmbeddingSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaInputEmbeddingRowRequest) -> Result<Self::Output> {
        let row_idx = usize::try_from(request.token_id).expect("u32 should fit into usize");
        let row = match &self.backing {
            GemmaInputEmbeddingBacking::Owned(rows) => {
                rows.get(row_idx).cloned().ok_or_else(|| {
                    anyhow!(
                        "Gemma input embedding token id {} is out of range for {} rows",
                        request.token_id,
                        rows.len()
                    )
                })?
            }
            GemmaInputEmbeddingBacking::Model(source) => {
                det_num_tensor_slice_row_wgts(source, row_idx, "input embedding")?
                    .into_iter()
                    .map(|value| Act::from_bits(value.to_bits()))
                    .collect()
            }
        };

        if row.len() != self.hidden_size {
            bail!(
                "Gemma input embedding row {} has width {}, expected {}",
                request.token_id,
                row.len(),
                self.hidden_size
            );
        }

        Ok(row
            .into_iter()
            .map(|value| scale_act(value, self.scale))
            .collect())
    }
}

impl CommittedExternalRequest for GemmaInputEmbeddingMetadataRequest {
    type Output = GemmaInputEmbeddingMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(INPUT_EMBEDDING_METADATA_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl CommittedExternalRequest for GemmaInputEmbeddingRowRequest {
    type Output = Vec<Act>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(INPUT_EMBEDDING_ROW_REQUEST, &self.token_id)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Act::from_bits)
            .collect())
    }
}

impl AuthRead<GemmaInputEmbeddingMetadataRequest> for str {
    type Output = GemmaInputEmbeddingMetadata;

    fn auth_read(&self, request: GemmaInputEmbeddingMetadataRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl AuthRead<GemmaInputEmbeddingRowRequest> for str {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaInputEmbeddingRowRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

fn validate_full_embedding_source(source: &DetNumTensorSliceSource) -> Result<()> {
    if source.row_count == 0 || source.col_count == 0 {
        bail!("Gemma input embedding source must have non-zero shape");
    }
    if source.row_offset != 0
        || source.row_count != source.total_rows
        || source.col_offset != 0
        || source.col_count != source.total_cols
    {
        bail!("Gemma input embedding source must reference the full embedding matrix");
    }
    Ok(())
}

fn validate_identifier(identifier: String) -> Result<String> {
    if identifier.is_empty() {
        bail!("Gemma input embedding source identifier must not be empty");
    }
    Ok(identifier)
}
