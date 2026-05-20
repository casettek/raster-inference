use std::{cell::RefCell, collections::HashMap};

use anyhow::{anyhow, bail, Result};
use serde::Serialize;

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::external_artifacts::{
    register_external_source_leaves, CommittedExternalSource, ExternalSourceId, ExternalSourceRef,
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
const GEMMA_INPUT_EMBEDDING_SOURCE_CHUNK_BYTES: usize = 1 << 20;

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

#[derive(Serialize)]
struct GemmaInputEmbeddingSourcePayload<'a> {
    identifier: &'a str,
    vocab_size: usize,
    hidden_size: usize,
    scale_bits: i32,
    backing: GemmaInputEmbeddingSourcePayloadBacking,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum GemmaInputEmbeddingSourcePayloadBacking {
    Owned {
        rows: Vec<Vec<i32>>,
    },
    Model {
        weights_path: String,
        total_rows: usize,
        total_cols: usize,
        data_offset: usize,
        row_offset: usize,
        row_count: usize,
        col_offset: usize,
        col_count: usize,
    },
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
        let source_ref = register_external_source_leaves(
            ExternalSourceId::new(self.identifier.clone())?,
            GEMMA_INPUT_EMBEDDING_SOURCE_KIND,
            GEMMA_INPUT_EMBEDDING_SOURCE_DOMAIN,
            source_payload_chunks(&self.source_payload()),
        )?;
        register_native_committed_input_embedding(source_ref.root(), self)?;
        Ok(source_ref)
    }

    pub fn committed_source(&self) -> Result<CommittedExternalSource> {
        Ok(CommittedExternalSource::new(self.committed_source_ref()?))
    }

    fn source_payload(&self) -> Vec<u8> {
        let backing = match &self.backing {
            GemmaInputEmbeddingBacking::Owned(rows) => {
                GemmaInputEmbeddingSourcePayloadBacking::Owned {
                    rows: rows
                        .iter()
                        .map(|row| row.iter().map(|value| value.to_bits()).collect())
                        .collect(),
                }
            }
            GemmaInputEmbeddingBacking::Model(source) => {
                GemmaInputEmbeddingSourcePayloadBacking::Model {
                    weights_path: source.weights_path.to_string_lossy().into_owned(),
                    total_rows: source.total_rows,
                    total_cols: source.total_cols,
                    data_offset: source.data_offset,
                    row_offset: source.row_offset,
                    row_count: source.row_count,
                    col_offset: source.col_offset,
                    col_count: source.col_count,
                }
            }
        };
        serde_json::to_vec(&GemmaInputEmbeddingSourcePayload {
            identifier: &self.identifier,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            scale_bits: self.scale.to_bits(),
            backing,
        })
        .expect("canonical input embedding source payload should serialize")
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

impl AuthRead<GemmaInputEmbeddingMetadataRequest> for CommittedExternalSource {
    type Output = GemmaInputEmbeddingMetadata;

    fn auth_read(&self, request: GemmaInputEmbeddingMetadataRequest) -> Result<Self::Output> {
        with_native_committed_input_embedding(self.root(), |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaInputEmbeddingRowRequest> for CommittedExternalSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaInputEmbeddingRowRequest) -> Result<Self::Output> {
        with_native_committed_input_embedding(self.root(), |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaInputEmbeddingMetadataRequest> for str {
    type Output = GemmaInputEmbeddingMetadata;

    fn auth_read(&self, request: GemmaInputEmbeddingMetadataRequest) -> Result<Self::Output> {
        with_native_committed_input_embedding(self, |source| source.auth_read(request))
    }
}

impl AuthRead<GemmaInputEmbeddingRowRequest> for str {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaInputEmbeddingRowRequest) -> Result<Self::Output> {
        with_native_committed_input_embedding(self, |source| source.auth_read(request))
    }
}

thread_local! {
    static NATIVE_COMMITTED_INPUT_EMBEDDINGS: RefCell<HashMap<String, AuthenticatedGemmaInputEmbeddingSource>> =
        RefCell::new(HashMap::new());
}

fn register_native_committed_input_embedding(
    root: &str,
    source: &AuthenticatedGemmaInputEmbeddingSource,
) -> Result<()> {
    NATIVE_COMMITTED_INPUT_EMBEDDINGS.with(|sources_ref| {
        let mut sources = sources_ref.borrow_mut();
        match sources.get(root) {
            Some(existing) if existing != source => {
                bail!("committed input embedding root {root} is already registered with different data")
            }
            Some(_) => Ok(()),
            None => {
                sources.insert(root.to_string(), source.clone());
                Ok(())
            }
        }
    })
}

fn with_native_committed_input_embedding<T>(
    root: &str,
    f: impl FnOnce(&AuthenticatedGemmaInputEmbeddingSource) -> Result<T>,
) -> Result<T> {
    NATIVE_COMMITTED_INPUT_EMBEDDINGS.with(|sources_ref| {
        let sources = sources_ref.borrow();
        let source = sources.get(root).ok_or_else(|| {
            anyhow!("committed input embedding root {root} is not registered natively")
        })?;
        f(source)
    })
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

fn source_payload_chunks(payload: &[u8]) -> Vec<Vec<u8>> {
    payload
        .chunks(GEMMA_INPUT_EMBEDDING_SOURCE_CHUNK_BYTES)
        .map(|chunk| chunk.to_vec())
        .collect()
}
