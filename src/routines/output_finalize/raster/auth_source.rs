use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use sha2::{Digest, Sha256};

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactMetadata, RasterArtifactRef, RasterArtifactStoreRoots,
};

pub const OUTPUT_TEXT_CHUNK_ARTIFACT_KIND: &str = "output_text_chunks";
pub const OUTPUT_PENDING_BYTE_ARTIFACT_KIND: &str = "output_pending_bytes";
pub const OUTPUT_TEXT_CHUNK_ARTIFACT_DOMAIN: &str = "raster-output-text-chunks-merkle-v1";
pub const OUTPUT_PENDING_BYTE_ARTIFACT_DOMAIN: &str = "raster-output-pending-bytes-merkle-v1";

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    artifact_ref: Option<RasterArtifactRef>,
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
    pub fn from_artifact(
        artifact_ref: RasterArtifactRef,
        byte_len: usize,
        char_count: usize,
        det_text_sha256: String,
    ) -> Result<Self> {
        if artifact_ref.metadata().kind() != OUTPUT_TEXT_CHUNK_ARTIFACT_KIND {
            bail!(
                "raster output text artifact kind mismatch: {}",
                artifact_ref.metadata().kind()
            );
        }
        let chunk_count = artifact_ref.metadata().leaf_count();
        Ok(Self {
            output_id: artifact_ref.id().source_name().to_string(),
            chunk_count,
            byte_len,
            char_count,
            det_text_sha256,
            artifact_ref: Some(artifact_ref),
        })
    }

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

    pub fn artifact_ref(&self) -> Result<&RasterArtifactRef> {
        self.artifact_ref
            .as_ref()
            .ok_or_else(|| anyhow!("raster output text ref is not artifact-backed"))
    }

    pub fn root(&self) -> Option<&str> {
        self.artifact_ref.as_ref().map(RasterArtifactRef::root)
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
            artifact_ref: None,
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
            artifact_ref: None,
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputTokenIdsCommitmentState {
    h: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
    finalized: bool,
}

impl OutputTokenIdsCommitmentState {
    pub fn new() -> Self {
        let mut state = Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: Vec::new(),
            total_len: 0,
            finalized: false,
        };
        state.update(b"[");
        state
    }

    pub fn update_token(&mut self, token_id: u32, token_idx: usize) -> Result<()> {
        if self.finalized {
            bail!("raster output token commitment is already finalized");
        }
        if token_idx > 0 {
            self.update(b",");
        }
        self.update(token_id.to_string().as_bytes());
        Ok(())
    }

    pub fn finish(mut self) -> String {
        if !self.finalized {
            self.update(b"]");
            self.finalize_padding();
            self.finalized = true;
        }
        self.h
            .iter()
            .flat_map(|word| word.to_be_bytes())
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn update(&mut self, mut input: &[u8]) {
        self.total_len = self
            .total_len
            .checked_add(input.len() as u64)
            .expect("output token commitment input length should fit in u64");
        if !self.buffer.is_empty() {
            let remaining = 64 - self.buffer.len();
            let take = remaining.min(input.len());
            self.buffer.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.buffer.len() == 64 {
                let block: [u8; 64] = self
                    .buffer
                    .as_slice()
                    .try_into()
                    .expect("buffer length checked");
                self.compress(&block);
                self.buffer.clear();
            }
        }
        while input.len() >= 64 {
            let block: [u8; 64] = input[..64].try_into().expect("block length checked");
            self.compress(&block);
            input = &input[64..];
        }
        if !input.is_empty() {
            self.buffer.extend_from_slice(input);
        }
    }

    fn finalize_padding(&mut self) {
        let bit_len = self
            .total_len
            .checked_mul(8)
            .expect("output token commitment bit length should fit in u64");
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());
        let blocks = std::mem::take(&mut self.buffer);
        for block in blocks.chunks_exact(64) {
            let block: [u8; 64] = block.try_into().expect("padding block length checked");
            self.compress(&block);
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (idx, chunk) in block.chunks_exact(4).take(16).enumerate() {
            w[idx] = u32::from_be_bytes(chunk.try_into().expect("word length checked"));
        }
        for idx in 16..64 {
            let s0 =
                w[idx - 15].rotate_right(7) ^ w[idx - 15].rotate_right(18) ^ (w[idx - 15] >> 3);
            let s1 = w[idx - 2].rotate_right(17) ^ w[idx - 2].rotate_right(19) ^ (w[idx - 2] >> 10);
            w[idx] = w[idx - 16]
                .wrapping_add(s0)
                .wrapping_add(w[idx - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for idx in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[idx])
                .wrapping_add(w[idx]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
    }
}

impl Default for OutputTokenIdsCommitmentState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputUtf8ValidationState {
    needed: u8,
    min_next: u8,
    max_next: u8,
}

impl OutputUtf8ValidationState {
    pub fn new() -> Self {
        Self {
            needed: 0,
            min_next: 0x80,
            max_next: 0xbf,
        }
    }

    pub fn push(&mut self, byte: u8) -> bool {
        if self.needed == 0 {
            match byte {
                0x00..=0x7f => true,
                0xc2..=0xdf => {
                    self.expect_continuations(1, 0x80, 0xbf);
                    true
                }
                0xe0 => {
                    self.expect_continuations(2, 0xa0, 0xbf);
                    true
                }
                0xe1..=0xec | 0xee..=0xef => {
                    self.expect_continuations(2, 0x80, 0xbf);
                    true
                }
                0xed => {
                    self.expect_continuations(2, 0x80, 0x9f);
                    true
                }
                0xf0 => {
                    self.expect_continuations(3, 0x90, 0xbf);
                    true
                }
                0xf1..=0xf3 => {
                    self.expect_continuations(3, 0x80, 0xbf);
                    true
                }
                0xf4 => {
                    self.expect_continuations(3, 0x80, 0x8f);
                    true
                }
                _ => false,
            }
        } else if (self.min_next..=self.max_next).contains(&byte) {
            self.needed -= 1;
            self.min_next = 0x80;
            self.max_next = 0xbf;
            true
        } else {
            false
        }
    }

    pub fn is_complete(&self) -> bool {
        self.needed == 0
    }

    fn expect_continuations(&mut self, needed: u8, min_next: u8, max_next: u8) {
        self.needed = needed;
        self.min_next = min_next;
        self.max_next = max_next;
    }
}

impl Default for OutputUtf8ValidationState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputPendingBytesRef {
    artifact_ref: RasterArtifactRef,
}

impl OutputPendingBytesRef {
    pub fn new(artifact_ref: RasterArtifactRef) -> Result<Self> {
        if artifact_ref.metadata().kind() != OUTPUT_PENDING_BYTE_ARTIFACT_KIND {
            bail!(
                "raster output pending-byte artifact kind mismatch: {}",
                artifact_ref.metadata().kind()
            );
        }
        Ok(Self { artifact_ref })
    }

    pub fn artifact_ref(&self) -> &RasterArtifactRef {
        &self.artifact_ref
    }

    pub fn byte_count(&self) -> usize {
        self.artifact_ref.metadata().leaf_count()
    }
}

pub fn output_text_metadata() -> Result<RasterArtifactMetadata> {
    RasterArtifactMetadata::open(
        OUTPUT_TEXT_CHUNK_ARTIFACT_KIND,
        OUTPUT_TEXT_CHUNK_ARTIFACT_DOMAIN,
        Vec::new(),
    )
}

pub fn output_pending_bytes_metadata() -> Result<RasterArtifactMetadata> {
    RasterArtifactMetadata::open(
        OUTPUT_PENDING_BYTE_ARTIFACT_KIND,
        OUTPUT_PENDING_BYTE_ARTIFACT_DOMAIN,
        Vec::new(),
    )
}

pub fn text_chunk_leaf(chunk: &str) -> Vec<u8> {
    postcard_leaf(&chunk, "output text chunk")
}

pub fn decode_text_chunk_leaf(payload: &[u8]) -> Result<String> {
    postcard::from_bytes(payload)
        .map_err(|error| anyhow!("failed to deserialize raster output text chunk leaf: {error}"))
}

pub fn pending_byte_leaf(byte: u8) -> Vec<u8> {
    postcard_leaf(&byte, "output pending byte")
}

pub fn decode_pending_byte_leaf(payload: &[u8]) -> Result<u8> {
    postcard::from_bytes(payload)
        .map_err(|error| anyhow!("failed to deserialize raster output pending-byte leaf: {error}"))
}

fn postcard_leaf<T: serde::Serialize + ?Sized>(value: &T, label: &str) -> Vec<u8> {
    postcard::to_allocvec(value)
        .unwrap_or_else(|error| panic!("failed to serialize raster {label} leaf: {error}"))
}

pub fn build_output_text_commitment(text: &str) -> String {
    build_text_commitment(text)
}

pub fn materialize_text_from_roots(
    roots: &RasterArtifactStoreRoots,
    text_ref: &OutputTextRef,
) -> Result<String> {
    let artifact_ref = text_ref.artifact_ref()?;
    let entry = roots.artifact_entry_for_source_name(artifact_ref.id().source_name())?;
    if entry.root() != artifact_ref.root() {
        bail!(
            "raster output text artifact root mismatch for {}: snapshot has {}, ref has {}",
            artifact_ref.id().source_name(),
            entry.root(),
            artifact_ref.root()
        );
    }

    let mut text = String::new();
    for chunk_idx in 0..text_ref.chunk_count() {
        let read =
            crate::shared::artifacts::artifact_io::ArtifactIo::read_verified_leaf_from_roots(
                roots,
                artifact_ref,
                chunk_idx,
            )?;
        text.push_str(&decode_text_chunk_leaf(read.bytes())?);
    }
    let text_commitment = build_text_commitment(&text);
    let artifact_commitment = artifact_ref.root();
    if text.len() != text_ref.byte_len()
        || text.chars().count() != text_ref.char_count()
        || (text_commitment != text_ref.det_text_sha256()
            && artifact_commitment != text_ref.det_text_sha256())
    {
        bail!("raster output text ref metadata mismatch");
    }
    Ok(text)
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
mod tests;
