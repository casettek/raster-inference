use std::fs::File;
use std::marker::PhantomData;

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::external_artifacts::{
    decode_i32_vec_response, decode_postcard_response, postcard_external_source_entry,
    postcard_i32_vec_external_source_entry, postcard_request_key, register_external_source,
    CommittedExternalRequest, CommittedExternalSource, ExternalSourceEntry, ExternalSourceId,
    ExternalSourceRef,
};
#[cfg(feature = "unchecked-raster-integrity")]
use crate::shared::artifacts::integrity_mode::raster_integrity_is_unchecked;
use crate::shared::model::common::{DecoderModelView, ModelFamily, WeightMatrixView};
use crate::shared::model::transformer::{DetNumTensorSliceSource, Gemma4TransformerModel};
use crate::shared::numerics::det_num::{scale_act, Act};
use crate::shared::raster_kernels::transformer::det_num_tensor_slice_row_wgts;

const GEMMA_INPUT_EMBEDDING_SOURCE_KIND: &str = "gemma_input_embedding";
const GEMMA_INPUT_EMBEDDING_SOURCE_DOMAIN: &str =
    "raster-external-source-gemma-input-embedding-merkle-v1";
const INPUT_EMBEDDING_METADATA_REQUEST: &str = "gemma_input_embedding.metadata";
const INPUT_EMBEDDING_ROW_REQUEST: &str = "gemma_input_embedding.row";

/// Domain prefix for [`AuthenticatedDecoderEmbeddingSource::cache_fingerprint`].
/// Versioned with the fingerprint construction: any change to the hashed
/// fields must bump this string (and the external cache kind).
const INPUT_EMBEDDING_FINGERPRINT_DOMAIN: &str =
    "raster-inference-gemma-input-embedding-cache-fingerprint-v2";

#[derive(Debug, Clone, PartialEq)]
pub struct AuthenticatedDecoderEmbeddingSource {
    identifier: String,
    vocab_size: usize,
    hidden_size: usize,
    scale: Act,
    backing: GemmaInputEmbeddingBacking,
}

pub enum RasterInputEmbeddingSource<'a> {
    Committed {
        source: CommittedExternalSource,
        _marker: PhantomData<&'a AuthenticatedDecoderEmbeddingSource>,
    },
    #[cfg(feature = "unchecked-raster-integrity")]
    DirectUnchecked {
        source: &'a AuthenticatedDecoderEmbeddingSource,
        root: String,
    },
}

impl<'a> RasterInputEmbeddingSource<'a> {
    pub fn for_current_integrity_mode(
        source: &'a AuthenticatedDecoderEmbeddingSource,
    ) -> Result<Self> {
        #[cfg(feature = "unchecked-raster-integrity")]
        if raster_integrity_is_unchecked() {
            return Ok(Self::DirectUnchecked {
                root: format!(
                    "raster-unchecked-test:direct-input-embedding:{}",
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

impl AuthenticatedDecoderEmbeddingSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        Self::from_decoder_view(identifier, &model.decoder_view())
    }

    pub fn from_decoder_view(
        identifier: impl Into<String>,
        view: &DecoderModelView<'_>,
    ) -> Result<Self> {
        ensure_gemma_view(view)?;
        let source = match view.embeddings.weights {
            Some(WeightMatrixView::DetNumSlice(source)) => source.clone(),
            Some(_) | None => {
                bail!("deterministic raster input embedding requires a .detwgt embedding source")
            }
        };
        let scale = Act::from_num(view.embeddings.scale);
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

    /// Cheap model-identity fingerprint addressing this table's entry in
    /// the pre-encoded externals directory (cache kind
    /// `gemma-input-embedding-v2`).
    ///
    /// For model-backed sources this hashes the raw deterministic tensor
    /// byte region (mmap slice — no row decoding, no scaling) together with
    /// the source identity: domain prefix, identifier, shape, scale bits,
    /// and storage element width. Content-addressed, so moving or renaming
    /// the weights file does not change the fingerprint. Owned-rows sources
    /// (test fixtures) hash the row bits directly — those tables are tiny.
    pub(crate) fn cache_fingerprint(&self) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(INPUT_EMBEDDING_FINGERPRINT_DOMAIN.as_bytes());
        hasher.update((self.identifier.len() as u64).to_le_bytes());
        hasher.update(self.identifier.as_bytes());
        hasher.update((self.vocab_size as u64).to_le_bytes());
        hasher.update((self.hidden_size as u64).to_le_bytes());
        hasher.update(self.scale.to_bits().to_le_bytes());
        match &self.backing {
            GemmaInputEmbeddingBacking::Owned(rows) => {
                hasher.update(b"owned");
                for row in rows {
                    for value in row {
                        hasher.update(value.to_bits().to_le_bytes());
                    }
                }
            }
            GemmaInputEmbeddingBacking::Model(source) => {
                hasher.update(b"model");
                hasher.update(source.element_width.tag().to_le_bytes());
                // The source is validated to reference the full embedding
                // matrix, so the region is one contiguous byte range.
                let elem_bytes = source.element_width.byte_width();
                let byte_len = source
                    .total_rows
                    .checked_mul(source.total_cols)
                    .and_then(|elements| elements.checked_mul(elem_bytes))
                    .ok_or_else(|| anyhow!("embedding tensor byte size overflowed"))?;
                let end = source
                    .data_offset
                    .checked_add(byte_len)
                    .ok_or_else(|| anyhow!("embedding tensor byte range overflowed"))?;
                let file = File::open(&source.weights_path).with_context(|| {
                    format!(
                        "failed to open deterministic artifact {}",
                        source.weights_path.display()
                    )
                })?;
                let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| {
                    format!(
                        "failed to mmap deterministic artifact {}",
                        source.weights_path.display()
                    )
                })?;
                let region = mmap.get(source.data_offset..end).ok_or_else(|| {
                    anyhow!(
                        "embedding tensor byte range {}..{end} is out of bounds for {}",
                        source.data_offset,
                        source.weights_path.display()
                    )
                })?;
                hasher.update(region);
            }
        }
        Ok(format!("{:x}", hasher.finalize()))
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

fn ensure_gemma_view(view: &DecoderModelView<'_>) -> Result<()> {
    if view.spec.family != ModelFamily::Gemma {
        bail!("Gemma input embedding source requires a Gemma decoder view");
    }
    Ok(())
}

impl AuthRead<GemmaInputEmbeddingMetadataRequest> for AuthenticatedDecoderEmbeddingSource {
    type Output = GemmaInputEmbeddingMetadata;

    fn auth_read(&self, _request: GemmaInputEmbeddingMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata())
    }
}

impl AuthRead<GemmaInputEmbeddingMetadataRequest> for RasterInputEmbeddingSource<'_> {
    type Output = GemmaInputEmbeddingMetadata;

    fn auth_read(&self, request: GemmaInputEmbeddingMetadataRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.auth_read(request),
        }
    }
}

impl AuthRead<GemmaInputEmbeddingRowRequest> for AuthenticatedDecoderEmbeddingSource {
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

impl AuthRead<GemmaInputEmbeddingRowRequest> for RasterInputEmbeddingSource<'_> {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaInputEmbeddingRowRequest) -> Result<Self::Output> {
        match self {
            Self::Committed { source, .. } => source.auth_read(request),
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::DirectUnchecked { source, .. } => source.auth_read(request),
        }
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::shared::numerics::det_num::DetWgtElementWidth;

    fn owned_source(identifier: &str, rows: Vec<Vec<Act>>) -> AuthenticatedDecoderEmbeddingSource {
        AuthenticatedDecoderEmbeddingSource::from_canonical_rows(
            identifier,
            rows,
            Act::from_bits(1),
        )
        .expect("owned source should build")
    }

    fn act_rows(seed: i32) -> Vec<Vec<Act>> {
        vec![
            vec![Act::from_bits(seed), Act::from_bits(seed + 1)],
            vec![Act::from_bits(seed + 2), Act::from_bits(seed + 3)],
        ]
    }

    #[test]
    fn owned_rows_fingerprint_is_stable_and_distinct_per_table() {
        let source = owned_source("fixture", act_rows(10));
        let first = source.cache_fingerprint().expect("fingerprint");
        let second = source.cache_fingerprint().expect("fingerprint");
        assert_eq!(first, second, "fingerprint must be stable across calls");

        let other_table = owned_source("fixture", act_rows(11));
        assert_ne!(
            first,
            other_table.cache_fingerprint().expect("fingerprint"),
            "different row bits must fingerprint differently"
        );

        let other_id = owned_source("fixture-b", act_rows(10));
        assert_ne!(
            first,
            other_id.cache_fingerprint().expect("fingerprint"),
            "different identifiers must fingerprint differently"
        );
    }

    fn model_source(weights_path: PathBuf) -> AuthenticatedDecoderEmbeddingSource {
        AuthenticatedDecoderEmbeddingSource {
            identifier: "model-fixture".to_string(),
            vocab_size: 2,
            hidden_size: 3,
            scale: Act::from_bits(7),
            backing: GemmaInputEmbeddingBacking::Model(DetNumTensorSliceSource {
                weights_path,
                total_rows: 2,
                total_cols: 3,
                data_offset: 4,
                element_width: DetWgtElementWidth::I16,
                row_offset: 0,
                row_count: 2,
                col_offset: 0,
                col_count: 3,
            }),
        }
    }

    #[test]
    fn model_fingerprint_tracks_the_tensor_byte_region() {
        let dir = std::env::temp_dir().join(format!(
            "raster-embedding-fingerprint-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        // 4-byte header + 2x3 i16 payload.
        let mut bytes = vec![0xAAu8; 4];
        bytes.extend_from_slice(&[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
        let path = dir.join("model.detwgt");
        std::fs::write(&path, &bytes).expect("write fixture weights");

        let source = model_source(path.clone());
        let first = source.cache_fingerprint().expect("fingerprint");
        assert_eq!(
            first,
            source.cache_fingerprint().expect("fingerprint"),
            "same tensor bytes must fingerprint identically across calls"
        );

        // Moving the file must not change the fingerprint (content-addressed).
        let moved = dir.join("renamed.detwgt");
        std::fs::rename(&path, &moved).expect("rename fixture weights");
        assert_eq!(
            first,
            model_source(moved.clone()).cache_fingerprint().expect("fingerprint"),
            "renaming the weights file must not change the fingerprint"
        );

        // Flipping one byte inside the tensor region must change it.
        let mut tampered = bytes.clone();
        tampered[5] ^= 0xff;
        std::fs::write(&moved, &tampered).expect("write tampered weights");
        assert_ne!(
            first,
            model_source(moved).cache_fingerprint().expect("fingerprint"),
            "a byte flip in the tensor region must change the fingerprint"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
