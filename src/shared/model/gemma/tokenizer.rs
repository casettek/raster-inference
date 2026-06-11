use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use serde::Serialize;

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::external_artifacts::{
    decode_postcard_response, postcard_external_source_entry, postcard_request_key,
    register_external_source, CommittedExternalRequest, CommittedExternalSource,
    ExternalSourceEntry, ExternalSourceId, ExternalSourceRef,
};
#[cfg(feature = "unchecked-raster-integrity")]
use crate::shared::artifacts::integrity_mode::raster_integrity_is_unchecked;
use crate::shared::artifacts::raster_artifact_store::RasterBpePieceSequenceRef;

const GEMMA_TOKENIZER_SOURCE_KIND: &str = "gemma_tokenizer";
const GEMMA_TOKENIZER_SOURCE_DOMAIN: &str = "raster-external-source-gemma-tokenizer-merkle-v1";
const TOKENIZER_METADATA_REQUEST: &str = "gemma_tokenizer.metadata";
const TOKENIZER_DECODER_METADATA_REQUEST: &str = "gemma_tokenizer.decoder_metadata";
const TOKENIZER_TOKEN_ID_REQUEST: &str = "gemma_tokenizer.token_id";
const TOKENIZER_TOKEN_BY_ID_REQUEST: &str = "gemma_tokenizer.token_by_id";
const TOKENIZER_BPE_MERGE_REQUEST: &str = "gemma_tokenizer.bpe_merge";
const TOKENIZER_BPE_MERGED_TOKEN_REQUEST: &str = "gemma_tokenizer.bpe_merged_token";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaVocabEntry {
    pub token: String,
    pub id: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeMerge {
    pub left: String,
    pub right: String,
    pub merged: String,
    pub rank: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaAddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmaTokenizerSpec {
    pub tokenizer_sha256: String,
    pub vocab: Vec<GemmaVocabEntry>,
    pub merges: Vec<GemmaBpeMerge>,
    pub added_tokens: Vec<GemmaAddedToken>,
    pub unk_token: String,
    pub unk_token_id: u32,
    pub byte_fallback: bool,
    pub space_replacement: String,
    pub split_pattern: String,
    vocab_by_token: HashMap<String, u32>,
    token_by_id: HashMap<u32, String>,
    special_token_ids: HashSet<u32>,
    merge_by_pair: HashMap<String, HashMap<String, usize>>,
    special_tokens_by_length: Vec<GemmaAddedToken>,
    decoder_metadata: Option<GemmaDecoderMetadata>,
    source_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedGemmaTokenizer {
    identifier: String,
    spec: GemmaTokenizerSpec,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaTokenizerMetadata {
    pub tokenizer_sha256: String,
    pub unk_token: String,
    pub unk_token_id: u32,
    pub byte_fallback: bool,
    pub space_replacement: String,
    pub split_pattern: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaDecoderMetadata {
    pub tokenizer_sha256: String,
    pub replacement_pattern: String,
    pub replacement_content: String,
    pub byte_fallback: bool,
    pub fuse: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaDecodedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeMergeCandidate {
    pub merge_index: usize,
    pub rank: usize,
}

#[derive(Serialize)]
struct GemmaTokenizerSourcePayload<'a> {
    tokenizer_sha256: &'a str,
    vocab: &'a [GemmaVocabEntry],
    merges: &'a [GemmaBpeMerge],
    added_tokens: &'a [GemmaAddedToken],
    unk_token: &'a str,
    byte_fallback: bool,
    space_replacement: &'a str,
    split_pattern: &'a str,
    decoder_metadata: &'a Option<GemmaDecoderMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaTokenizerMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaDecoderMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaTokenIdRequest<'a> {
    pub token: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaTokenByIdRequest {
    pub token_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaSpecialTokenAtRequest<'a> {
    pub input: &'a str,
    pub byte_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaBpeMergeRequest<'a> {
    pub left: &'a str,
    pub right: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaBpeMergedTokenRequest {
    pub merge_index: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaNormalizedText {
    pub text: String,
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPreTokenizedText {
    pub segments: Vec<String>,
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeState {
    pub piece_count: usize,
    pub add_special_tokens: bool,
    pub iteration: u64,
    pub bpe_pairs_per_tile: usize,
    pub bpe_pieces_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeOutput {
    pub piece_count: usize,
    pub iteration: u64,
    pub add_special_tokens: bool,
    pub bpe_pieces_per_tile: usize,
}

impl GemmaTokenizerSpec {
    pub fn new(
        tokenizer_sha256: String,
        vocab: Vec<GemmaVocabEntry>,
        merges: Vec<GemmaBpeMerge>,
        added_tokens: Vec<GemmaAddedToken>,
        unk_token: String,
        byte_fallback: bool,
        space_replacement: String,
        split_pattern: String,
    ) -> Result<Self> {
        let decoder_metadata = Some(GemmaDecoderMetadata {
            tokenizer_sha256: tokenizer_sha256.clone(),
            replacement_pattern: space_replacement.clone(),
            replacement_content: " ".to_string(),
            byte_fallback,
            fuse: true,
        });
        Self::new_with_decoder_metadata(
            tokenizer_sha256,
            vocab,
            merges,
            added_tokens,
            unk_token,
            byte_fallback,
            space_replacement,
            split_pattern,
            decoder_metadata,
            None,
        )
    }

    pub(crate) fn new_with_decoder_metadata(
        tokenizer_sha256: String,
        mut vocab: Vec<GemmaVocabEntry>,
        merges: Vec<GemmaBpeMerge>,
        added_tokens: Vec<GemmaAddedToken>,
        unk_token: String,
        byte_fallback: bool,
        space_replacement: String,
        split_pattern: String,
        decoder_metadata: Option<GemmaDecoderMetadata>,
        source_payload: Option<Vec<u8>>,
    ) -> Result<Self> {
        if vocab.is_empty() {
            bail!("Gemma tokenizer spec requires a non-empty vocab");
        }

        vocab.sort_by(|left, right| left.id.cmp(&right.id).then(left.token.cmp(&right.token)));

        let mut seen_tokens = HashSet::new();
        let mut seen_ids = HashSet::new();
        let mut vocab_by_token = HashMap::with_capacity(vocab.len());
        let mut token_by_id = HashMap::with_capacity(vocab.len() + added_tokens.len());
        for entry in &vocab {
            if !seen_tokens.insert(entry.token.clone()) {
                bail!(
                    "Gemma tokenizer spec has duplicate vocab token {}",
                    entry.token
                );
            }
            if !seen_ids.insert(entry.id) {
                bail!("Gemma tokenizer spec has duplicate vocab id {}", entry.id);
            }
            vocab_by_token.insert(entry.token.clone(), entry.id);
            token_by_id.insert(entry.id, entry.token.clone());
        }

        let Some(unk_token_id) = vocab_by_token.get(&unk_token).copied() else {
            bail!("Gemma tokenizer spec unk token {unk_token} is missing from vocab");
        };

        let mut special_token_ids = HashSet::new();
        for token in &added_tokens {
            if token.special {
                special_token_ids.insert(token.id);
            }
            match token_by_id.get(&token.id) {
                Some(existing) if existing != &token.content => {
                    bail!(
                        "Gemma tokenizer added token id {} content {} conflicts with vocab token {}",
                        token.id,
                        token.content,
                        existing
                    );
                }
                Some(_) => {}
                None => {
                    token_by_id.insert(token.id, token.content.clone());
                }
            }
        }

        let mut seen_merges = HashSet::new();
        let mut merge_by_pair = HashMap::<String, HashMap<String, usize>>::new();
        for (merge_idx, merge) in merges.iter().enumerate() {
            if merge.left.is_empty() || merge.right.is_empty() || merge.merged.is_empty() {
                bail!("Gemma tokenizer spec has an empty BPE merge component");
            }
            if !seen_merges.insert((merge.left.clone(), merge.right.clone())) {
                bail!(
                    "Gemma tokenizer spec has duplicate BPE merge pair ({}, {})",
                    merge.left,
                    merge.right
                );
            }
            merge_by_pair
                .entry(merge.left.clone())
                .or_default()
                .insert(merge.right.clone(), merge_idx);
        }

        let mut special_tokens_by_length = added_tokens
            .iter()
            .filter(|token| token.special)
            .cloned()
            .collect::<Vec<_>>();
        special_tokens_by_length.sort_by(|left, right| {
            right
                .content
                .len()
                .cmp(&left.content.len())
                .then(left.content.cmp(&right.content))
        });

        let source_payload = source_payload.unwrap_or_else(|| {
            canonical_tokenizer_source_payload(GemmaTokenizerSourcePayload {
                tokenizer_sha256: &tokenizer_sha256,
                vocab: &vocab,
                merges: &merges,
                added_tokens: &added_tokens,
                unk_token: &unk_token,
                byte_fallback,
                space_replacement: &space_replacement,
                split_pattern: &split_pattern,
                decoder_metadata: &decoder_metadata,
            })
        });

        Ok(Self {
            tokenizer_sha256,
            vocab,
            merges,
            added_tokens,
            unk_token,
            unk_token_id,
            byte_fallback,
            space_replacement,
            split_pattern,
            vocab_by_token,
            token_by_id,
            special_token_ids,
            merge_by_pair,
            special_tokens_by_length,
            decoder_metadata,
            source_payload,
        })
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab_by_token.get(token).copied()
    }

    pub fn token_by_id(&self, token_id: u32) -> Option<GemmaDecodedToken> {
        self.token_by_id
            .get(&token_id)
            .map(|content| GemmaDecodedToken {
                id: token_id,
                content: content.clone(),
                special: self.special_token_ids.contains(&token_id),
            })
    }

    pub fn longest_special_token_at(
        &self,
        input: &str,
        byte_idx: usize,
    ) -> Option<&GemmaAddedToken> {
        self.special_tokens_by_length
            .iter()
            .find(|token| input[byte_idx..].starts_with(&token.content))
    }

    pub fn bpe_merge(&self, left: &str, right: &str) -> Option<&GemmaBpeMerge> {
        self.bpe_merge_index(left, right)
            .and_then(|merge_idx| self.bpe_merge_by_index(merge_idx))
    }

    pub fn bpe_merge_index(&self, left: &str, right: &str) -> Option<usize> {
        self.merge_by_pair
            .get(left)
            .and_then(|rights| rights.get(right))
            .copied()
    }

    pub fn bpe_merge_by_index(&self, merge_index: usize) -> Option<&GemmaBpeMerge> {
        self.merges.get(merge_index)
    }

    pub fn byte_fallback_token(byte: u8) -> String {
        format!("<0x{byte:02X}>")
    }
}

impl AuthenticatedGemmaTokenizer {
    pub fn new(spec: GemmaTokenizerSpec) -> Self {
        Self {
            identifier: spec.tokenizer_sha256.clone(),
            spec,
        }
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn committed_source_ref(&self) -> Result<ExternalSourceRef> {
        register_external_source(
            ExternalSourceId::new(self.identifier.clone())?,
            GEMMA_TOKENIZER_SOURCE_KIND,
            GEMMA_TOKENIZER_SOURCE_DOMAIN,
            self.committed_source_entries()?,
        )
    }

    pub fn committed_source(&self) -> Result<CommittedExternalSource> {
        Ok(CommittedExternalSource::new(self.committed_source_ref()?))
    }

    pub fn raster_source_root_for_current_integrity_mode(&self) -> Result<String> {
        #[cfg(feature = "unchecked-raster-integrity")]
        if raster_integrity_is_unchecked() {
            return Ok(format!(
                "raster-unchecked-test:direct-tokenizer:{}",
                self.identifier()
            ));
        }

        Ok(self.committed_source_ref()?.root().to_string())
    }

    fn metadata(&self) -> GemmaTokenizerMetadata {
        GemmaTokenizerMetadata {
            tokenizer_sha256: self.spec.tokenizer_sha256.clone(),
            unk_token: self.spec.unk_token.clone(),
            unk_token_id: self.spec.unk_token_id,
            byte_fallback: self.spec.byte_fallback,
            space_replacement: self.spec.space_replacement.clone(),
            split_pattern: self.spec.split_pattern.clone(),
        }
    }

    fn committed_source_entries(&self) -> Result<Vec<ExternalSourceEntry>> {
        let mut entries = Vec::new();
        entries.push(postcard_external_source_entry(
            GemmaTokenizerMetadataRequest.request_key()?,
            &self.metadata(),
        )?);
        if let Some(decoder_metadata) = self.spec.decoder_metadata.as_ref() {
            entries.push(postcard_external_source_entry(
                GemmaDecoderMetadataRequest.request_key()?,
                decoder_metadata,
            )?);
        }
        for entry in &self.spec.vocab {
            entries.push(postcard_external_source_entry(
                GemmaTokenIdRequest {
                    token: &entry.token,
                }
                .request_key()?,
                &entry.id,
            )?);
        }
        for token_id in self.spec.token_by_id.keys().copied() {
            if let Some(decoded) = self.spec.token_by_id(token_id) {
                entries.push(postcard_external_source_entry(
                    GemmaTokenByIdRequest { token_id }.request_key()?,
                    &decoded,
                )?);
            }
        }
        for (merge_index, merge) in self.spec.merges.iter().enumerate() {
            entries.push(postcard_external_source_entry(
                GemmaBpeMergeRequest {
                    left: &merge.left,
                    right: &merge.right,
                }
                .request_key()?,
                &GemmaBpeMergeCandidate {
                    merge_index,
                    rank: merge.rank,
                },
            )?);
            entries.push(postcard_external_source_entry(
                GemmaBpeMergedTokenRequest { merge_index }.request_key()?,
                &merge.merged,
            )?);
        }
        Ok(entries)
    }
}

impl From<GemmaTokenizerSpec> for AuthenticatedGemmaTokenizer {
    fn from(spec: GemmaTokenizerSpec) -> Self {
        Self::new(spec)
    }
}

impl AuthRead<GemmaTokenizerMetadataRequest> for AuthenticatedGemmaTokenizer {
    type Output = GemmaTokenizerMetadata;

    fn auth_read(&self, _request: GemmaTokenizerMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata())
    }
}

impl AuthRead<GemmaDecoderMetadataRequest> for AuthenticatedGemmaTokenizer {
    type Output = GemmaDecoderMetadata;

    fn auth_read(&self, _request: GemmaDecoderMetadataRequest) -> Result<Self::Output> {
        let Some(metadata) = self.spec.decoder_metadata.clone() else {
            bail!("Gemma tokenizer spec is missing supported decoder metadata");
        };
        Ok(metadata)
    }
}

impl<'a> AuthRead<GemmaTokenIdRequest<'a>> for AuthenticatedGemmaTokenizer {
    type Output = Option<u32>;

    fn auth_read(&self, request: GemmaTokenIdRequest<'a>) -> Result<Self::Output> {
        Ok(self.spec.token_id(request.token))
    }
}

impl AuthRead<GemmaTokenByIdRequest> for AuthenticatedGemmaTokenizer {
    type Output = Option<GemmaDecodedToken>;

    fn auth_read(&self, request: GemmaTokenByIdRequest) -> Result<Self::Output> {
        Ok(self.spec.token_by_id(request.token_id))
    }
}

impl<'a> AuthRead<GemmaSpecialTokenAtRequest<'a>> for AuthenticatedGemmaTokenizer {
    type Output = Option<GemmaAddedToken>;

    fn auth_read(&self, request: GemmaSpecialTokenAtRequest<'a>) -> Result<Self::Output> {
        if !request.input.is_char_boundary(request.byte_idx) {
            bail!(
                "Gemma tokenizer special-token lookup byte index {} is not a char boundary",
                request.byte_idx
            );
        }

        Ok(self
            .spec
            .longest_special_token_at(request.input, request.byte_idx)
            .cloned())
    }
}

impl<'a> AuthRead<GemmaBpeMergeRequest<'a>> for AuthenticatedGemmaTokenizer {
    type Output = Option<GemmaBpeMergeCandidate>;

    fn auth_read(&self, request: GemmaBpeMergeRequest<'a>) -> Result<Self::Output> {
        Ok(self
            .spec
            .bpe_merge_index(request.left, request.right)
            .map(|merge_index| GemmaBpeMergeCandidate {
                merge_index,
                rank: self.spec.merges[merge_index].rank,
            }))
    }
}

impl AuthRead<GemmaBpeMergedTokenRequest> for AuthenticatedGemmaTokenizer {
    type Output = Option<String>;

    fn auth_read(&self, request: GemmaBpeMergedTokenRequest) -> Result<Self::Output> {
        Ok(self
            .spec
            .bpe_merge_by_index(request.merge_index)
            .map(|merge| merge.merged.clone()))
    }
}

impl CommittedExternalRequest for GemmaTokenizerMetadataRequest {
    type Output = GemmaTokenizerMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(TOKENIZER_METADATA_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl CommittedExternalRequest for GemmaDecoderMetadataRequest {
    type Output = GemmaDecoderMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(TOKENIZER_DECODER_METADATA_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl<'a> CommittedExternalRequest for GemmaTokenIdRequest<'a> {
    type Output = Option<u32>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(TOKENIZER_TOKEN_ID_REQUEST, self.token)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response::<u32>(response_payload).map(Some)
    }

    fn decode_missing_response(&self) -> Result<Self::Output> {
        Ok(None)
    }
}

impl CommittedExternalRequest for GemmaTokenByIdRequest {
    type Output = Option<GemmaDecodedToken>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(TOKENIZER_TOKEN_BY_ID_REQUEST, &self.token_id)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response::<GemmaDecodedToken>(response_payload).map(Some)
    }

    fn decode_missing_response(&self) -> Result<Self::Output> {
        Ok(None)
    }
}

impl<'a> CommittedExternalRequest for GemmaBpeMergeRequest<'a> {
    type Output = Option<GemmaBpeMergeCandidate>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(TOKENIZER_BPE_MERGE_REQUEST, &(self.left, self.right))
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response::<GemmaBpeMergeCandidate>(response_payload).map(Some)
    }

    fn decode_missing_response(&self) -> Result<Self::Output> {
        Ok(None)
    }
}

impl CommittedExternalRequest for GemmaBpeMergedTokenRequest {
    type Output = Option<String>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(TOKENIZER_BPE_MERGED_TOKEN_REQUEST, &self.merge_index)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response::<String>(response_payload).map(Some)
    }

    fn decode_missing_response(&self) -> Result<Self::Output> {
        Ok(None)
    }
}

impl AuthRead<GemmaTokenizerMetadataRequest> for str {
    type Output = GemmaTokenizerMetadata;

    fn auth_read(&self, request: GemmaTokenizerMetadataRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl AuthRead<GemmaDecoderMetadataRequest> for str {
    type Output = GemmaDecoderMetadata;

    fn auth_read(&self, request: GemmaDecoderMetadataRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl<'a> AuthRead<GemmaTokenIdRequest<'a>> for str {
    type Output = Option<u32>;

    fn auth_read(&self, request: GemmaTokenIdRequest<'a>) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl AuthRead<GemmaTokenByIdRequest> for str {
    type Output = Option<GemmaDecodedToken>;

    fn auth_read(&self, request: GemmaTokenByIdRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl<'a> AuthRead<GemmaBpeMergeRequest<'a>> for str {
    type Output = Option<GemmaBpeMergeCandidate>;

    fn auth_read(&self, request: GemmaBpeMergeRequest<'a>) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

impl AuthRead<GemmaBpeMergedTokenRequest> for str {
    type Output = Option<String>;

    fn auth_read(&self, request: GemmaBpeMergedTokenRequest) -> Result<Self::Output> {
        CommittedExternalSource::from_root(self)?.auth_read(request)
    }
}

fn canonical_tokenizer_source_payload(payload: GemmaTokenizerSourcePayload<'_>) -> Vec<u8> {
    serde_json::to_vec(&payload).expect("canonical tokenizer payload should serialize")
}

impl GemmaBpeState {
    pub fn new(
        pieces_ref: RasterBpePieceSequenceRef,
        add_special_tokens: bool,
        bpe_pairs_per_tile: usize,
        bpe_pieces_per_tile: usize,
    ) -> Self {
        let piece_count = pieces_ref.piece_count();
        Self {
            piece_count,
            add_special_tokens,
            iteration: 0,
            bpe_pairs_per_tile,
            bpe_pieces_per_tile,
        }
    }

    pub fn into_output(self) -> GemmaBpeOutput {
        GemmaBpeOutput {
            piece_count: self.piece_count,
            iteration: self.iteration,
            add_special_tokens: self.add_special_tokens,
            bpe_pieces_per_tile: self.bpe_pieces_per_tile,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeMergeRequest,
        GemmaBpeMergedTokenRequest, GemmaDecoderMetadataRequest, GemmaSpecialTokenAtRequest,
        GemmaTokenByIdRequest, GemmaTokenIdRequest, GemmaTokenizerMetadataRequest,
        GemmaTokenizerSpec, GemmaVocabEntry,
    };

    #[test]
    fn spec_stores_vocab_and_merges_with_stable_lookup() {
        let spec = GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![
                GemmaVocabEntry {
                    token: "<unk>".to_string(),
                    id: 0,
                },
                GemmaVocabEntry {
                    token: "a".to_string(),
                    id: 1,
                },
                GemmaVocabEntry {
                    token: "b".to_string(),
                    id: 2,
                },
                GemmaVocabEntry {
                    token: "ab".to_string(),
                    id: 3,
                },
            ],
            vec![GemmaBpeMerge {
                left: "a".to_string(),
                right: "b".to_string(),
                merged: "ab".to_string(),
                rank: 0,
            }],
            vec![GemmaAddedToken {
                id: 4,
                content: "<bos>".to_string(),
                special: true,
            }],
            "<unk>".to_string(),
            true,
            "▁".to_string(),
            " ".to_string(),
        )
        .expect("spec should build");

        assert_eq!(spec.token_id("ab"), Some(3));
        assert_eq!(
            spec.bpe_merge("a", "b").expect("merge should exist").merged,
            "ab"
        );
        assert_eq!(spec.unk_token_id, 0);
        assert_eq!(
            spec.longest_special_token_at("<bos>hello", 0)
                .expect("special token")
                .id,
            4
        );
    }

    #[test]
    fn spec_rejects_duplicate_vocab_ids() {
        let error = GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![
                GemmaVocabEntry {
                    token: "<unk>".to_string(),
                    id: 0,
                },
                GemmaVocabEntry {
                    token: "a".to_string(),
                    id: 0,
                },
            ],
            Vec::new(),
            Vec::new(),
            "<unk>".to_string(),
            true,
            "▁".to_string(),
            " ".to_string(),
        )
        .expect_err("duplicate ids should fail");

        assert!(error.to_string().contains("duplicate vocab id"));
    }

    #[test]
    fn spec_rejects_missing_unk_token() {
        let error = GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![GemmaVocabEntry {
                token: "a".to_string(),
                id: 1,
            }],
            Vec::new(),
            Vec::new(),
            "<unk>".to_string(),
            true,
            "▁".to_string(),
            " ".to_string(),
        )
        .expect_err("missing unk token should fail");

        assert!(error.to_string().contains("missing from vocab"));
    }

    #[test]
    fn spec_rejects_added_token_id_that_conflicts_with_vocab_content() {
        let error = GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![
                GemmaVocabEntry {
                    token: "<unk>".to_string(),
                    id: 0,
                },
                GemmaVocabEntry {
                    token: "a".to_string(),
                    id: 1,
                },
            ],
            Vec::new(),
            vec![GemmaAddedToken {
                id: 1,
                content: "<bos>".to_string(),
                special: true,
            }],
            "<unk>".to_string(),
            true,
            "▁".to_string(),
            " ".to_string(),
        )
        .expect_err("conflicting added token id should fail");

        assert!(error.to_string().contains("conflicts with vocab token"));
    }

    #[test]
    fn authenticated_tokenizer_reads_metadata() {
        let source = AuthenticatedGemmaTokenizer::new(test_spec());
        let metadata = crate::auth_read!(&source, GemmaTokenizerMetadataRequest)
            .expect("metadata should read");

        assert_eq!(source.identifier(), "digest");
        assert_eq!(metadata.tokenizer_sha256, "digest");
        assert_eq!(metadata.unk_token, "<unk>");
        assert_eq!(metadata.unk_token_id, 0);
        assert!(metadata.byte_fallback);
        assert_eq!(metadata.space_replacement, "▁");
        assert_eq!(metadata.split_pattern, " ");
    }

    #[test]
    fn authenticated_tokenizer_reads_token_ids() {
        let source = AuthenticatedGemmaTokenizer::new(test_spec());

        assert_eq!(
            crate::auth_read!(&source, GemmaTokenIdRequest { token: "ab" })
                .expect("token id should read"),
            Some(3)
        );
        assert_eq!(
            crate::auth_read!(&source, GemmaTokenIdRequest { token: "missing" })
                .expect("token id should read"),
            None
        );
    }

    #[test]
    fn authenticated_tokenizer_reads_decoder_metadata() {
        let source = AuthenticatedGemmaTokenizer::new(test_spec());
        let metadata = crate::auth_read!(&source, GemmaDecoderMetadataRequest)
            .expect("decoder metadata should read");

        assert_eq!(metadata.tokenizer_sha256, "digest");
        assert_eq!(metadata.replacement_pattern, "▁");
        assert_eq!(metadata.replacement_content, " ");
        assert!(metadata.byte_fallback);
        assert!(metadata.fuse);
    }

    #[test]
    fn authenticated_tokenizer_reads_tokens_by_id() {
        let source = AuthenticatedGemmaTokenizer::new(test_spec());

        let token = crate::auth_read!(&source, GemmaTokenByIdRequest { token_id: 3 })
            .expect("token by id should read")
            .expect("token should exist");
        assert_eq!(token.content, "ab");
        assert!(!token.special);

        let special = crate::auth_read!(&source, GemmaTokenByIdRequest { token_id: 4 })
            .expect("special token by id should read")
            .expect("special token should exist");
        assert_eq!(special.content, "<bos>");
        assert!(special.special);

        assert_eq!(
            crate::auth_read!(&source, GemmaTokenByIdRequest { token_id: 99 })
                .expect("missing token read should succeed"),
            None
        );
    }

    #[test]
    fn authenticated_tokenizer_reads_merges_by_pair() {
        let source = AuthenticatedGemmaTokenizer::new(test_spec());
        let merge = crate::auth_read!(
            &source,
            GemmaBpeMergeRequest {
                left: "a",
                right: "b",
            },
        )
        .expect("merge should read")
        .expect("merge should exist");

        assert_eq!(merge.rank, 0);
        assert_eq!(
            crate::auth_read!(
                &source,
                GemmaBpeMergedTokenRequest {
                    merge_index: merge.merge_index,
                },
            )
            .expect("merged token should read"),
            Some("ab".to_string())
        );
    }

    #[test]
    fn committed_tokenizer_reads_match_raster_source() {
        let source = AuthenticatedGemmaTokenizer::new(test_spec());
        let committed = source
            .committed_source()
            .expect("tokenizer source should commit");
        let source_ref = source
            .committed_source_ref()
            .expect("tokenizer source ref should be idempotent");

        assert_eq!(committed.root(), source_ref.root());
        assert_eq!(
            crate::auth_read!(&committed, GemmaTokenIdRequest { token: "ab" })
                .expect("token id should read"),
            Some(3)
        );
        assert_eq!(
            crate::auth_read!(
                &committed,
                GemmaBpeMergeRequest {
                    left: "a",
                    right: "b",
                },
            )
            .expect("merge should read")
            .expect("merge should exist")
            .rank,
            0
        );
        assert_eq!(
            crate::auth_read!(
                &committed,
                GemmaBpeMergeRequest {
                    left: "missing",
                    right: "pair",
                },
            )
            .expect("missing merge should read"),
            None
        );
    }

    #[test]
    fn authenticated_tokenizer_reads_longest_special_token() {
        let source = AuthenticatedGemmaTokenizer::new(
            GemmaTokenizerSpec::new(
                "digest".to_string(),
                vec![
                    GemmaVocabEntry {
                        token: "<unk>".to_string(),
                        id: 0,
                    },
                    GemmaVocabEntry {
                        token: "<bos>".to_string(),
                        id: 2,
                    },
                    GemmaVocabEntry {
                        token: "<bos>extra".to_string(),
                        id: 3,
                    },
                ],
                Vec::new(),
                vec![
                    GemmaAddedToken {
                        id: 2,
                        content: "<bos>".to_string(),
                        special: true,
                    },
                    GemmaAddedToken {
                        id: 3,
                        content: "<bos>extra".to_string(),
                        special: true,
                    },
                ],
                "<unk>".to_string(),
                true,
                "▁".to_string(),
                " ".to_string(),
            )
            .expect("spec should build"),
        );

        let token = crate::auth_read!(
            &source,
            GemmaSpecialTokenAtRequest {
                input: "<bos>extra text",
                byte_idx: 0,
            },
        )
        .expect("special token should read")
        .expect("special token should exist");

        assert_eq!(token.content, "<bos>extra");
    }

    #[test]
    fn authenticated_tokenizer_rejects_special_token_lookup_inside_char() {
        let source = AuthenticatedGemmaTokenizer::new(test_spec());
        let error = crate::auth_read!(
            &source,
            GemmaSpecialTokenAtRequest {
                input: "é",
                byte_idx: 1,
            },
        )
        .expect_err("lookup inside a multibyte char should fail");

        assert!(error.to_string().contains("not a char boundary"));
    }

    fn test_spec() -> GemmaTokenizerSpec {
        GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![
                GemmaVocabEntry {
                    token: "<unk>".to_string(),
                    id: 0,
                },
                GemmaVocabEntry {
                    token: "a".to_string(),
                    id: 1,
                },
                GemmaVocabEntry {
                    token: "b".to_string(),
                    id: 2,
                },
                GemmaVocabEntry {
                    token: "ab".to_string(),
                    id: 3,
                },
            ],
            vec![GemmaBpeMerge {
                left: "a".to_string(),
                right: "b".to_string(),
                merged: "ab".to_string(),
                rank: 0,
            }],
            vec![GemmaAddedToken {
                id: 4,
                content: "<bos>".to_string(),
                special: true,
            }],
            "<unk>".to_string(),
            true,
            "▁".to_string(),
            " ".to_string(),
        )
        .expect("spec should build")
    }
}
