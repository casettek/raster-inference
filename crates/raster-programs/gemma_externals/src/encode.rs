//! Offline encoder: HuggingFace `tokenizer.json` → raster-encoded
//! `GemmaTokenizer` committed external in a content-addressed cache.
//!
//! Loader provenance: the raster-tokenizer PoC's `encode_tokenizer` bin —
//! same raw-JSON shape validation, same derived-table construction (sorted
//! lookups, dense id table, longest-match special-token order), so an
//! encoding of the same `tokenizer.json` is byte-identical to the PoC's.
//!
//! Cache convention (WS2): entries live at
//! `<cache_root>/gemma-tokenizer/<sha256(tokenizer.json)>/` containing
//! `tokenizer.rastered`, `tokenizer.rindex`, and `root_commitment.txt`.
//! Re-encoding an already cached source is a no-op that re-reports the
//! stored commitment; encoding is deterministic (asserted by the main-crate
//! tokenizer-external test).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::types::{
    GemmaAddedToken, GemmaBpeMerge, GemmaBpeMergeCandidate, GemmaBpeMergeLookupEntry,
    GemmaDecodedToken, GemmaDecoderMetadata, GemmaTokenIdEntry, GemmaTokenizer,
    GemmaTokenizerMetadata,
};

/// One encoded cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedTokenizerExternal {
    pub data_path: PathBuf,
    pub index_path: PathBuf,
    /// Raster index root commitment — the `input_manifest.json` value.
    pub root_commitment: String,
    /// SHA-256 of the source `tokenizer.json` bytes (the cache address).
    pub source_sha256: String,
}

/// Encodes `tokenizer_json_path` into the content-addressed cache under
/// `cache_root`, reusing an existing entry when present.
pub fn encode_tokenizer_to_cache(
    tokenizer_json_path: &Path,
    cache_root: &Path,
) -> Result<EncodedTokenizerExternal> {
    let source_bytes = fs::read(tokenizer_json_path)
        .with_context(|| format!("failed to read {}", tokenizer_json_path.display()))?;
    let source_sha256 = sha256_hex(&source_bytes);

    let entry_dir = cache_root.join("gemma-tokenizer").join(&source_sha256);
    let data_path = entry_dir.join("tokenizer.rastered");
    let index_path = entry_dir.join("tokenizer.rindex");
    let commitment_path = entry_dir.join("root_commitment.txt");

    if data_path.is_file() && index_path.is_file() && commitment_path.is_file() {
        let root_commitment = fs::read_to_string(&commitment_path)
            .with_context(|| format!("failed to read {}", commitment_path.display()))?
            .trim()
            .to_string();
        return Ok(EncodedTokenizerExternal {
            data_path,
            index_path,
            root_commitment,
            source_sha256,
        });
    }

    let raw: RawTokenizer = serde_json::from_slice(&source_bytes)
        .context("failed to parse tokenizer.json into the raw tokenizer model")?;
    let tokenizer = build_tokenizer(raw)?;

    fs::create_dir_all(&entry_dir)
        .with_context(|| format!("failed to create cache entry {}", entry_dir.display()))?;
    let root_commitment = raster::write_raster_files(&tokenizer, &data_path, &index_path)
        .map_err(|error| anyhow!("failed to raster-encode tokenizer: {error}"))?;
    fs::write(&commitment_path, &root_commitment)
        .with_context(|| format!("failed to write {}", commitment_path.display()))?;

    Ok(EncodedTokenizerExternal {
        data_path,
        index_path,
        root_commitment,
        source_sha256,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn build_tokenizer(raw: RawTokenizer) -> Result<GemmaTokenizer> {
    validate_raw_tokenizer(&raw)?;

    let metadata = extract_tokenizer_metadata(&raw);
    let decoder = extract_decoder_metadata(&raw)?;
    let RawTokenizer {
        added_tokens,
        model,
        ..
    } = raw;

    let special_tokens = build_special_tokens(added_tokens);
    let token_lookup = build_token_lookup(&model.vocab);
    let tokens_by_id = build_tokens_by_id(&model.vocab, &special_tokens)?;
    let merges = build_merges(&model.vocab, model.merges);
    let merge_lookup = build_merge_lookup(&merges);

    Ok(GemmaTokenizer {
        metadata,
        decoder,
        token_lookup,
        tokens_by_id,
        special_tokens,
        merges,
        merge_lookup,
    })
}

fn validate_raw_tokenizer(raw: &RawTokenizer) -> Result<()> {
    if raw.normalizer.normalizer_type != "Replace" {
        bail!(
            "unsupported normalizer type '{}'",
            raw.normalizer.normalizer_type
        );
    }
    if raw.pre_tokenizer.pre_tokenizer_type != "Split" {
        bail!(
            "unsupported pre_tokenizer type '{}'",
            raw.pre_tokenizer.pre_tokenizer_type
        );
    }
    if raw.decoder.decoder_type != "Sequence" {
        bail!("unsupported decoder type '{}'", raw.decoder.decoder_type);
    }
    if raw.model.model_type != "BPE" {
        bail!("unsupported model type '{}'", raw.model.model_type);
    }
    if raw.normalizer.pattern.string != " " {
        bail!(
            "unsupported normalizer pattern '{}'",
            raw.normalizer.pattern.string
        );
    }
    Ok(())
}

fn extract_tokenizer_metadata(raw: &RawTokenizer) -> GemmaTokenizerMetadata {
    GemmaTokenizerMetadata {
        space_replacement: raw.normalizer.content.clone(),
        split_delimiter: raw.pre_tokenizer.pattern.string.clone(),
        split_behavior: raw.pre_tokenizer.behavior.clone(),
        invert: raw.pre_tokenizer.invert,
        unk_token: raw.model.unk_token.clone(),
        fuse_unk: raw.model.fuse_unk,
        byte_fallback: raw.model.byte_fallback,
        ignore_merges: raw.model.ignore_merges,
    }
}

fn extract_decoder_metadata(raw: &RawTokenizer) -> Result<GemmaDecoderMetadata> {
    let replace_decoder = raw
        .decoder
        .decoders
        .iter()
        .find_map(|decoder| match decoder {
            RawDecoderStep::Replace { pattern, .. } => Some(pattern.string.clone()),
            _ => None,
        })
        .ok_or_else(|| anyhow!("decoder sequence is missing a Replace decoder"))?;
    let byte_fallback = raw
        .decoder
        .decoders
        .iter()
        .any(|decoder| matches!(decoder, RawDecoderStep::ByteFallback));
    let fuse_decoder = raw
        .decoder
        .decoders
        .iter()
        .any(|decoder| matches!(decoder, RawDecoderStep::Fuse));

    Ok(GemmaDecoderMetadata {
        space_replacement: replace_decoder,
        byte_fallback,
        fuse_decoder,
    })
}

fn build_special_tokens(added_tokens: Vec<RawAddedToken>) -> Vec<GemmaAddedToken> {
    let mut special_tokens: Vec<GemmaAddedToken> = added_tokens
        .into_iter()
        .filter(|token| token.special)
        .map(Into::into)
        .collect();
    special_tokens.sort_by(|left, right| {
        right
            .content
            .len()
            .cmp(&left.content.len())
            .then_with(|| left.content.cmp(&right.content))
    });
    special_tokens
}

fn build_token_lookup(vocab: &BTreeMap<String, u32>) -> Vec<GemmaTokenIdEntry> {
    let mut token_lookup: Vec<GemmaTokenIdEntry> = vocab
        .iter()
        .map(|(token, id)| GemmaTokenIdEntry {
            token: token.clone(),
            id: *id,
        })
        .collect();
    token_lookup.sort_by(|left, right| left.token.cmp(&right.token));
    token_lookup
}

fn build_tokens_by_id(
    vocab: &BTreeMap<String, u32>,
    special_tokens: &[GemmaAddedToken],
) -> Result<Vec<GemmaDecodedToken>> {
    let max_id = vocab
        .values()
        .copied()
        .max()
        .ok_or_else(|| anyhow!("tokenizer vocab is empty"))? as usize;
    let mut tokens_by_id = vec![None; max_id + 1];

    for (token, id) in vocab {
        let special = special_tokens.iter().any(|added| added.id == *id);
        tokens_by_id[*id as usize] = Some(GemmaDecodedToken {
            id: *id,
            token: token.clone(),
            special,
        });
    }

    tokens_by_id
        .into_iter()
        .enumerate()
        .map(|(idx, token)| {
            token.ok_or_else(|| anyhow!("tokenizer vocab is missing token id {idx}"))
        })
        .collect()
}

fn build_merges(
    vocab: &BTreeMap<String, u32>,
    raw_merges: Vec<(String, String)>,
) -> Vec<GemmaBpeMerge> {
    raw_merges
        .into_iter()
        .enumerate()
        .map(|(merge_index, merge)| {
            let merged_token = format!("{}{}", merge.0, merge.1);
            let token_id = vocab.get(&merged_token).copied();
            GemmaBpeMerge {
                merge_index: merge_index as u32,
                left: merge.0,
                right: merge.1,
                merged_token,
                has_token_id: token_id.is_some(),
                token_id: token_id.unwrap_or_default(),
            }
        })
        .collect()
}

fn build_merge_lookup(merges: &[GemmaBpeMerge]) -> Vec<GemmaBpeMergeLookupEntry> {
    let mut merge_lookup: Vec<GemmaBpeMergeLookupEntry> = merges
        .iter()
        .map(|merge| GemmaBpeMergeLookupEntry {
            left: merge.left.clone(),
            right: merge.right.clone(),
            candidate: GemmaBpeMergeCandidate {
                merge_index: merge.merge_index,
                merged_token: merge.merged_token.clone(),
                has_token_id: merge.has_token_id,
                token_id: merge.token_id,
            },
        })
        .collect();
    merge_lookup.sort_by(|left, right| {
        left.left
            .cmp(&right.left)
            .then_with(|| left.right.cmp(&right.right))
    });
    merge_lookup
}

#[derive(Debug, Deserialize)]
struct RawTokenizer {
    added_tokens: Vec<RawAddedToken>,
    normalizer: RawNormalizer,
    pre_tokenizer: RawPreTokenizer,
    decoder: RawDecoder,
    model: RawBpeModel,
}

#[derive(Debug, Deserialize)]
struct RawAddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

impl From<RawAddedToken> for GemmaAddedToken {
    fn from(value: RawAddedToken) -> Self {
        Self {
            id: value.id,
            content: value.content,
            single_word: value.single_word,
            lstrip: value.lstrip,
            rstrip: value.rstrip,
            normalized: value.normalized,
            special: value.special,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawNormalizer {
    #[serde(rename = "type")]
    normalizer_type: String,
    pattern: RawStringPattern,
    content: String,
}

#[derive(Debug, Deserialize)]
struct RawPreTokenizer {
    #[serde(rename = "type")]
    pre_tokenizer_type: String,
    pattern: RawStringPattern,
    behavior: String,
    invert: bool,
}

#[derive(Debug, Deserialize)]
struct RawDecoder {
    #[serde(rename = "type")]
    decoder_type: String,
    decoders: Vec<RawDecoderStep>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RawDecoderStep {
    Replace {
        pattern: RawStringPattern,
        #[allow(dead_code)]
        content: String,
    },
    ByteFallback,
    Fuse,
}

#[derive(Debug, Deserialize)]
struct RawStringPattern {
    #[serde(rename = "String")]
    string: String,
}

#[derive(Debug, Deserialize)]
struct RawBpeModel {
    #[serde(rename = "type")]
    model_type: String,
    unk_token: String,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: BTreeMap<String, u32>,
    merges: Vec<(String, String)>,
}
