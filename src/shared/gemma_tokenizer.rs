use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};

use crate::raster_authoring::AuthRead;

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
    merge_by_pair: HashMap<String, HashMap<String, usize>>,
    special_tokens_by_length: Vec<GemmaAddedToken>,
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
pub struct GemmaBpeMergeCandidate {
    pub merge_index: usize,
    pub rank: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaTokenizerMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaTokenIdRequest<'a> {
    pub token: &'a str,
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
    pub pieces: Vec<String>,
    pub add_special_tokens: bool,
    pub iteration: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeOutput {
    pub pieces: Vec<String>,
    pub add_special_tokens: bool,
}

impl GemmaTokenizerSpec {
    pub fn new(
        tokenizer_sha256: String,
        mut vocab: Vec<GemmaVocabEntry>,
        merges: Vec<GemmaBpeMerge>,
        added_tokens: Vec<GemmaAddedToken>,
        unk_token: String,
        byte_fallback: bool,
        space_replacement: String,
        split_pattern: String,
    ) -> Result<Self> {
        if vocab.is_empty() {
            bail!("Gemma tokenizer spec requires a non-empty vocab");
        }

        vocab.sort_by(|left, right| left.id.cmp(&right.id).then(left.token.cmp(&right.token)));

        let mut seen_tokens = HashSet::new();
        let mut seen_ids = HashSet::new();
        let mut vocab_by_token = HashMap::with_capacity(vocab.len());
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
        }

        let Some(unk_token_id) = vocab_by_token.get(&unk_token).copied() else {
            bail!("Gemma tokenizer spec unk token {unk_token} is missing from vocab");
        };

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
            merge_by_pair,
            special_tokens_by_length,
        })
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab_by_token.get(token).copied()
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
}

impl From<GemmaTokenizerSpec> for AuthenticatedGemmaTokenizer {
    fn from(spec: GemmaTokenizerSpec) -> Self {
        Self::new(spec)
    }
}

impl AuthRead<GemmaTokenizerMetadataRequest> for AuthenticatedGemmaTokenizer {
    type Output = GemmaTokenizerMetadata;

    fn auth_read(&self, _request: GemmaTokenizerMetadataRequest) -> Result<Self::Output> {
        Ok(GemmaTokenizerMetadata {
            tokenizer_sha256: self.spec.tokenizer_sha256.clone(),
            unk_token: self.spec.unk_token.clone(),
            unk_token_id: self.spec.unk_token_id,
            byte_fallback: self.spec.byte_fallback,
            space_replacement: self.spec.space_replacement.clone(),
            split_pattern: self.spec.split_pattern.clone(),
        })
    }
}

impl<'a> AuthRead<GemmaTokenIdRequest<'a>> for AuthenticatedGemmaTokenizer {
    type Output = Option<u32>;

    fn auth_read(&self, request: GemmaTokenIdRequest<'a>) -> Result<Self::Output> {
        Ok(self.spec.token_id(request.token))
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

impl GemmaBpeState {
    pub fn new(pieces: Vec<String>, add_special_tokens: bool) -> Self {
        Self {
            pieces,
            add_special_tokens,
            iteration: 0,
        }
    }

    pub fn into_output(self) -> GemmaBpeOutput {
        GemmaBpeOutput {
            pieces: self.pieces,
            add_special_tokens: self.add_special_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeMergeRequest,
        GemmaBpeMergedTokenRequest, GemmaSpecialTokenAtRequest, GemmaTokenIdRequest,
        GemmaTokenizerMetadataRequest, GemmaTokenizerSpec, GemmaVocabEntry,
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
