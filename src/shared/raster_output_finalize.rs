use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use sha2::{Digest, Sha256};

use crate::shared::artifact_io::AuthRead;

#[derive(Debug, Clone)]
pub struct AuthenticatedOutputTokenIdsSource {
    identifier: String,
    metadata: OutputTokenIdsMetadata,
    token_ids: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputTokenIdsMetadata {
    pub source_id: String,
    pub token_count: usize,
    pub det_token_ids_sha256: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputTokenIdsRef {
    source_id: String,
    token_count: usize,
    det_token_ids_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputTokenIdsMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputTokenIdRequest {
    pub token_idx: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputTextRef {
    output_id: String,
    chunk_count: usize,
    byte_len: usize,
    char_count: usize,
    det_text_sha256: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputTextBuilderRef {
    output_id: String,
    chunks_written: usize,
    byte_len: usize,
    char_count: usize,
    running_commitment: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PendingByteBuilderRef {
    output_id: String,
    bytes_written: usize,
    running_commitment: String,
}

#[derive(Debug, Default, Clone)]
pub struct AuthenticatedOutputFinalizeStore {
    texts: HashMap<String, StoredText>,
    text_builders: HashMap<String, TextBuilderState>,
    pending_byte_builders: HashMap<String, PendingByteBuilderState>,
}

#[derive(Debug, Clone)]
struct StoredText {
    text: String,
    chunk_count: usize,
}

#[derive(Debug, Clone)]
struct TextBuilderState {
    chunks: Vec<String>,
}

#[derive(Debug, Clone)]
struct PendingByteBuilderState {
    bytes: Vec<u8>,
}

impl AuthenticatedOutputTokenIdsSource {
    pub fn from_token_ids(identifier: impl Into<String>, token_ids: &[u32]) -> Result<Self> {
        let identifier = validate_identifier(identifier.into(), "output token ids source")?;
        let det_token_ids_sha256 = build_output_token_ids_commitment(token_ids)?;
        Ok(Self {
            metadata: OutputTokenIdsMetadata {
                source_id: identifier.clone(),
                token_count: token_ids.len(),
                det_token_ids_sha256,
            },
            identifier,
            token_ids: token_ids.to_vec(),
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn token_ids_ref(&self) -> OutputTokenIdsRef {
        OutputTokenIdsRef::from_metadata(&self.metadata)
    }

    pub fn materialize_token_ids(&self, token_ids_ref: &OutputTokenIdsRef) -> Result<Vec<u32>> {
        token_ids_ref.validate_against(&self.metadata)?;
        Ok(self.token_ids.clone())
    }
}

impl AuthRead<OutputTokenIdsMetadataRequest> for AuthenticatedOutputTokenIdsSource {
    type Output = OutputTokenIdsMetadata;

    fn auth_read(&self, _request: OutputTokenIdsMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata.clone())
    }
}

impl AuthRead<OutputTokenIdRequest> for AuthenticatedOutputTokenIdsSource {
    type Output = u32;

    fn auth_read(&self, request: OutputTokenIdRequest) -> Result<Self::Output> {
        self.token_ids
            .get(request.token_idx)
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "raster output token id {} is out of range for {} tokens",
                    request.token_idx,
                    self.token_ids.len()
                )
            })
    }
}

impl OutputTokenIdsMetadata {
    pub fn token_ids_ref(&self) -> OutputTokenIdsRef {
        OutputTokenIdsRef::from_metadata(self)
    }
}

impl OutputTokenIdsRef {
    fn from_metadata(metadata: &OutputTokenIdsMetadata) -> Self {
        Self {
            source_id: metadata.source_id.clone(),
            token_count: metadata.token_count,
            det_token_ids_sha256: metadata.det_token_ids_sha256.clone(),
        }
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn det_token_ids_sha256(&self) -> &str {
        &self.det_token_ids_sha256
    }

    pub fn validate_against(&self, metadata: &OutputTokenIdsMetadata) -> Result<()> {
        if self.source_id != metadata.source_id
            || self.token_count != metadata.token_count
            || self.det_token_ids_sha256 != metadata.det_token_ids_sha256
        {
            bail!("raster output token ids ref metadata mismatch");
        }
        Ok(())
    }
}

impl OutputTextRef {
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn char_count(&self) -> usize {
        self.char_count
    }

    pub fn det_text_sha256(&self) -> &str {
        &self.det_text_sha256
    }
}

impl OutputTextBuilderRef {
    pub fn chunks_written(&self) -> usize {
        self.chunks_written
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn char_count(&self) -> usize {
        self.char_count
    }

    pub fn running_commitment(&self) -> &str {
        &self.running_commitment
    }
}

impl PendingByteBuilderRef {
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }

    pub fn running_commitment(&self) -> &str {
        &self.running_commitment
    }
}

impl AuthenticatedOutputFinalizeStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start_text_builder(
        &mut self,
        output_id: impl Into<String>,
    ) -> Result<OutputTextBuilderRef> {
        let output_id = validate_identifier(output_id.into(), "output text builder")?;
        if self.texts.contains_key(&output_id) || self.text_builders.contains_key(&output_id) {
            bail!("raster output text id {output_id} is already registered");
        }

        let state = TextBuilderState { chunks: Vec::new() };
        let builder_ref = text_builder_ref(&output_id, &state);
        self.text_builders.insert(output_id, state);
        Ok(builder_ref)
    }

    pub fn append_text_chunk(
        &mut self,
        builder_ref: &mut OutputTextBuilderRef,
        chunk: &str,
    ) -> Result<()> {
        let state = self.text_builder_mut(builder_ref)?;
        state.chunks.push(chunk.to_string());
        update_text_builder_ref(builder_ref, state);
        Ok(())
    }

    pub fn finalize_text_builder(
        &mut self,
        builder_ref: OutputTextBuilderRef,
    ) -> Result<OutputTextRef> {
        let state = self
            .text_builders
            .remove(&builder_ref.output_id)
            .ok_or_else(|| {
                anyhow!(
                    "raster output text builder {} is not registered",
                    builder_ref.output_id
                )
            })?;
        let expected = text_builder_ref(&builder_ref.output_id, &state);
        if expected != builder_ref {
            bail!("raster output text builder metadata mismatch");
        }

        let text = state.chunks.concat();
        let text_ref = OutputTextRef {
            output_id: builder_ref.output_id.clone(),
            chunk_count: state.chunks.len(),
            byte_len: text.len(),
            char_count: text.chars().count(),
            det_text_sha256: build_text_commitment(&text),
        };
        self.texts.insert(
            builder_ref.output_id,
            StoredText {
                text,
                chunk_count: state.chunks.len(),
            },
        );
        Ok(text_ref)
    }

    pub fn materialize_text(&self, text_ref: &OutputTextRef) -> Result<String> {
        let stored = self.texts.get(&text_ref.output_id).ok_or_else(|| {
            anyhow!(
                "raster output text ref {} is not registered",
                text_ref.output_id
            )
        })?;
        let expected = OutputTextRef {
            output_id: text_ref.output_id.clone(),
            chunk_count: stored.chunk_count,
            byte_len: stored.text.len(),
            char_count: stored.text.chars().count(),
            det_text_sha256: build_text_commitment(&stored.text),
        };
        if expected != *text_ref {
            bail!("raster output text ref metadata mismatch");
        }
        Ok(stored.text.clone())
    }

    pub fn start_pending_byte_builder(
        &mut self,
        output_id: impl Into<String>,
    ) -> Result<PendingByteBuilderRef> {
        let output_id = validate_identifier(output_id.into(), "pending byte builder")?;
        if self.pending_byte_builders.contains_key(&output_id) {
            bail!("raster pending byte id {output_id} is already registered");
        }

        let state = PendingByteBuilderState { bytes: Vec::new() };
        let builder_ref = pending_byte_builder_ref(&output_id, &state);
        self.pending_byte_builders.insert(output_id, state);
        Ok(builder_ref)
    }

    pub fn append_pending_byte(
        &mut self,
        builder_ref: &mut PendingByteBuilderRef,
        byte: u8,
    ) -> Result<()> {
        let state = self.pending_byte_builder_mut(builder_ref)?;
        state.bytes.push(byte);
        update_pending_byte_builder_ref(builder_ref, state);
        Ok(())
    }

    pub fn pending_bytes_are_valid_utf8(
        &self,
        builder_ref: &PendingByteBuilderRef,
    ) -> Result<bool> {
        let state = self.pending_byte_builder(builder_ref)?;
        Ok(std::str::from_utf8(&state.bytes).is_ok())
    }

    pub fn append_pending_utf8_chunk_to_text(
        &mut self,
        byte_builder_ref: &PendingByteBuilderRef,
        text_builder_ref: &mut OutputTextBuilderRef,
        start_byte_idx: usize,
        max_bytes: usize,
    ) -> Result<usize> {
        if max_bytes == 0 {
            bail!("raster output UTF-8 flush chunk requires non-zero max bytes");
        }
        let bytes = self.pending_byte_builder(byte_builder_ref)?.bytes.clone();
        if start_byte_idx >= bytes.len() {
            bail!("raster output UTF-8 flush start is out of range");
        }
        let mut end = start_byte_idx.saturating_add(max_bytes).min(bytes.len());
        while end > start_byte_idx && std::str::from_utf8(&bytes[start_byte_idx..end]).is_err() {
            end -= 1;
        }
        if end == start_byte_idx {
            bail!("raster output UTF-8 flush could not find a valid chunk boundary");
        }

        let chunk = std::str::from_utf8(&bytes[start_byte_idx..end])?;
        self.append_text_chunk(text_builder_ref, chunk)?;
        Ok(end - start_byte_idx)
    }

    pub fn clear_pending_bytes(&mut self, builder_ref: &mut PendingByteBuilderRef) -> Result<()> {
        let state = self.pending_byte_builder_mut(builder_ref)?;
        state.bytes.clear();
        update_pending_byte_builder_ref(builder_ref, state);
        Ok(())
    }

    fn text_builder_mut(
        &mut self,
        builder_ref: &OutputTextBuilderRef,
    ) -> Result<&mut TextBuilderState> {
        let state = self
            .text_builders
            .get_mut(&builder_ref.output_id)
            .ok_or_else(|| {
                anyhow!(
                    "raster output text builder {} is not registered",
                    builder_ref.output_id
                )
            })?;
        if text_builder_ref(&builder_ref.output_id, state) != *builder_ref {
            bail!("raster output text builder metadata mismatch");
        }
        Ok(state)
    }

    fn pending_byte_builder(
        &self,
        builder_ref: &PendingByteBuilderRef,
    ) -> Result<&PendingByteBuilderState> {
        let state = self
            .pending_byte_builders
            .get(&builder_ref.output_id)
            .ok_or_else(|| {
                anyhow!(
                    "raster pending byte builder {} is not registered",
                    builder_ref.output_id
                )
            })?;
        if pending_byte_builder_ref(&builder_ref.output_id, state) != *builder_ref {
            bail!("raster pending byte builder metadata mismatch");
        }
        Ok(state)
    }

    fn pending_byte_builder_mut(
        &mut self,
        builder_ref: &PendingByteBuilderRef,
    ) -> Result<&mut PendingByteBuilderState> {
        let state = self
            .pending_byte_builders
            .get_mut(&builder_ref.output_id)
            .ok_or_else(|| {
                anyhow!(
                    "raster pending byte builder {} is not registered",
                    builder_ref.output_id
                )
            })?;
        if pending_byte_builder_ref(&builder_ref.output_id, state) != *builder_ref {
            bail!("raster pending byte builder metadata mismatch");
        }
        Ok(state)
    }
}

pub fn build_output_token_ids_commitment(token_ids: &[u32]) -> Result<String> {
    let payload =
        serde_json::to_vec(token_ids).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}

fn validate_identifier(identifier: String, label: &str) -> Result<String> {
    if identifier.is_empty() {
        bail!("raster {label} identifier must not be empty");
    }
    Ok(identifier)
}

fn text_builder_ref(output_id: &str, state: &TextBuilderState) -> OutputTextBuilderRef {
    let text = state.chunks.concat();
    OutputTextBuilderRef {
        output_id: output_id.to_string(),
        chunks_written: state.chunks.len(),
        byte_len: text.len(),
        char_count: text.chars().count(),
        running_commitment: build_text_builder_commitment(&state.chunks),
    }
}

fn update_text_builder_ref(builder_ref: &mut OutputTextBuilderRef, state: &TextBuilderState) {
    *builder_ref = text_builder_ref(&builder_ref.output_id, state);
}

fn pending_byte_builder_ref(
    output_id: &str,
    state: &PendingByteBuilderState,
) -> PendingByteBuilderRef {
    PendingByteBuilderRef {
        output_id: output_id.to_string(),
        bytes_written: state.bytes.len(),
        running_commitment: build_pending_byte_builder_commitment(&state.bytes),
    }
}

fn update_pending_byte_builder_ref(
    builder_ref: &mut PendingByteBuilderRef,
    state: &PendingByteBuilderState,
) {
    *builder_ref = pending_byte_builder_ref(&builder_ref.output_id, state);
}

fn build_text_commitment(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-output-text-v1");
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
    hex_digest(hasher.finalize())
}

fn build_text_builder_commitment(chunks: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-output-text-builder-v1");
    hasher.update((chunks.len() as u64).to_le_bytes());
    for chunk in chunks {
        hasher.update((chunk.len() as u64).to_le_bytes());
        hasher.update(chunk.as_bytes());
    }
    hex_digest(hasher.finalize())
}

fn build_pending_byte_builder_commitment(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-output-pending-bytes-builder-v1");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hex_digest(hasher.finalize())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedOutputFinalizeStore, AuthenticatedOutputTokenIdsSource, OutputTextBuilderRef,
        OutputTokenIdRequest, OutputTokenIdsMetadataRequest,
    };
    use crate::shared::artifact_io::AuthRead;

    #[test]
    fn token_source_reads_by_index_with_metadata() {
        let source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[4, 7, 9])
            .expect("source should build");

        let metadata = source
            .auth_read(OutputTokenIdsMetadataRequest)
            .expect("metadata should read");
        assert_eq!(metadata.source_id, "generated");
        assert_eq!(metadata.token_count, 3);
        assert_eq!(
            source
                .auth_read(OutputTokenIdRequest { token_idx: 1 })
                .expect("token should read"),
            7
        );
    }

    #[test]
    fn token_source_allows_empty_metadata_and_rejects_reads() {
        let source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[])
            .expect("empty source should build");

        let metadata = source
            .auth_read(OutputTokenIdsMetadataRequest)
            .expect("metadata should read");
        assert_eq!(metadata.token_count, 0);
        assert!(source
            .auth_read(OutputTokenIdRequest { token_idx: 0 })
            .expect_err("empty source read should fail")
            .to_string()
            .contains("out of range"));
    }

    #[test]
    fn token_source_materialization_rejects_tampered_refs() {
        let source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[4])
            .expect("source should build");
        let mut token_ids_ref = source.token_ids_ref();
        token_ids_ref.det_token_ids_sha256 = "tampered".to_string();

        assert!(source.materialize_token_ids(&token_ids_ref).is_err());
    }

    #[test]
    fn text_builder_materializes_text_and_commitment() {
        let mut store = AuthenticatedOutputFinalizeStore::new();
        let mut builder = store
            .start_text_builder("output")
            .expect("builder should start");

        store
            .append_text_chunk(&mut builder, " ab")
            .expect("first chunk should append");
        store
            .append_text_chunk(&mut builder, "c")
            .expect("second chunk should append");
        let text_ref = store
            .finalize_text_builder(builder)
            .expect("builder should finalize");

        assert_eq!(
            store
                .materialize_text(&text_ref)
                .expect("text should materialize"),
            " abc"
        );
        assert_eq!(text_ref.chunk_count(), 2);
        assert_eq!(text_ref.byte_len(), 4);
    }

    #[test]
    fn text_builder_materialization_rejects_tampered_refs() {
        let mut store = AuthenticatedOutputFinalizeStore::new();
        let builder = store
            .start_text_builder("output")
            .expect("builder should start");
        let mut text_ref = store
            .finalize_text_builder(builder)
            .expect("builder should finalize");
        text_ref.det_text_sha256 = "tampered".to_string();

        assert!(store.materialize_text(&text_ref).is_err());
    }

    #[test]
    fn text_builder_materialization_rejects_tampered_chunk_counts() {
        let mut store = AuthenticatedOutputFinalizeStore::new();
        let mut builder = store
            .start_text_builder("output")
            .expect("builder should start");
        store
            .append_text_chunk(&mut builder, "text")
            .expect("chunk should append");
        let mut text_ref = store
            .finalize_text_builder(builder)
            .expect("builder should finalize");
        text_ref.chunk_count += 1;

        assert!(store.materialize_text(&text_ref).is_err());
    }

    #[test]
    fn pending_byte_builder_tracks_bytes_without_exposing_them_in_ref() {
        let mut store = AuthenticatedOutputFinalizeStore::new();
        let mut builder = store
            .start_pending_byte_builder("pending")
            .expect("builder should start");

        store
            .append_pending_byte(&mut builder, 0xC3)
            .expect("first byte should append");
        store
            .append_pending_byte(&mut builder, 0xA9)
            .expect("second byte should append");

        let serialized = serde_json::to_string(&builder).expect("builder should serialize");
        assert_eq!(builder.bytes_written(), 2);
        assert!(!serialized.contains("195"));
        assert!(!serialized.contains("169"));
        assert!(!serialized.contains("bytes:"));
    }

    #[test]
    fn builder_refs_fail_closed_on_metadata_mismatch() {
        let mut store = AuthenticatedOutputFinalizeStore::new();
        let mut builder = store
            .start_text_builder("output")
            .expect("builder should start");
        store
            .append_text_chunk(&mut builder, "text")
            .expect("chunk should append");

        let tampered = OutputTextBuilderRef {
            running_commitment: "tampered".to_string(),
            ..builder
        };
        assert!(store.finalize_text_builder(tampered).is_err());
    }
}
