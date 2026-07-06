//! The Gemma tokenizer committed-external schema.
//!
//! Mirrors the raster-tokenizer PoC's `GemmaTokenizer` shape — the verified
//! authoring idiom for exactly this data (WS1 catalog C13, evidence: the
//! PoC's end-to-end run). Sorted lookup vectors replace the sim path's
//! hashed request keys: point lookups become binary search over a selected
//! sub-list or index-computing tiles plus `select!` by index.
//!
//! Guest-visible integers are fixed-width (`u32`) per catalog C12.

use alloc::string::String;
use alloc::vec::Vec;
use raster::Selectable;
use serde::{Deserialize, Serialize};

/// Normalizer/pre-tokenizer/BPE-model metadata (one `select!` for all
/// scalar configuration).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaTokenizerMetadata {
    pub space_replacement: String,
    pub split_delimiter: String,
    pub split_behavior: String,
    pub invert: bool,
    pub unk_token: String,
    pub fuse_unk: bool,
    pub byte_fallback: bool,
    pub ignore_merges: bool,
}

/// Decoder-sequence metadata (detokenization; `output.finalize`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaDecoderMetadata {
    pub space_replacement: String,
    pub byte_fallback: bool,
    pub fuse_decoder: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaAddedToken {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

/// `tokens_by_id[id]` entry: id-indexed decode table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaDecodedToken {
    pub id: u32,
    pub token: String,
    pub special: bool,
}

/// `token_lookup` entry; the vector is sorted by `token` for binary search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaTokenIdEntry {
    pub token: String,
    pub id: u32,
}

/// `merges` entry, ordered by merge priority (`merge_index`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaBpeMerge {
    pub merge_index: u32,
    pub left: String,
    pub right: String,
    pub merged_token: String,
    pub has_token_id: bool,
    pub token_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaBpeMergeCandidate {
    pub merge_index: u32,
    pub merged_token: String,
    pub has_token_id: bool,
    pub token_id: u32,
}

/// `merge_lookup` entry; the vector is sorted by `(left, right)` for binary
/// search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaBpeMergeLookupEntry {
    pub left: String,
    pub right: String,
    pub candidate: GemmaBpeMergeCandidate,
}

/// Root schema of the `tokenizer` committed external.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaTokenizer {
    pub metadata: GemmaTokenizerMetadata,
    pub decoder: GemmaDecoderMetadata,
    /// Sorted by `token` (binary search for token → id).
    pub token_lookup: Vec<GemmaTokenIdEntry>,
    /// Indexed by token id (dense; every id present).
    pub tokens_by_id: Vec<GemmaDecodedToken>,
    /// Sorted by content length (desc), then content (asc) — longest-match
    /// special-token scanning order.
    pub special_tokens: Vec<GemmaAddedToken>,
    /// Ordered by merge priority.
    pub merges: Vec<GemmaBpeMerge>,
    /// Sorted by `(left, right)` (binary search for pair → candidate).
    pub merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
}
