use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};

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
    special_tokens_by_length: Vec<GemmaAddedToken>,
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
    pub merge_rules: Vec<GemmaBpeMerge>,
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
        for merge in &merges {
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

    pub fn byte_fallback_token(byte: u8) -> String {
        format!("<0x{byte:02X}>")
    }
}

impl GemmaBpeState {
    pub fn new(
        pieces: Vec<String>,
        merge_rules: Vec<GemmaBpeMerge>,
        add_special_tokens: bool,
    ) -> Self {
        Self {
            pieces,
            merge_rules,
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
    use super::{GemmaAddedToken, GemmaBpeMerge, GemmaTokenizerSpec, GemmaVocabEntry};

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
}
