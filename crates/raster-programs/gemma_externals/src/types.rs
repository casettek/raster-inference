//! The Gemma tokenizer committed-external schema.
//!
//! Derived from the raster-tokenizer PoC's `GemmaTokenizer` shape (WS1
//! catalog C13), revised for the prompt.prepare storage refactor: the
//! model-scoped tables (vocab, merges) are **pre-chunked**
//! (`Vec<Vec<Entry>>`, encode-time width `TOKENIZER_CHUNK_WIDTH` in
//! `encode.rs`) so programs consume them only as recur input lists — one
//! bite-sized chunk per tile execution, never as materialized tile
//! arguments or loop state.
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

/// `token_lookup_chunks` entry; the flattened chunk list is sorted by
/// `token`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaTokenIdEntry {
    pub token: String,
    pub id: u32,
}

/// `merge_chunks` entry; the flattened chunk list is ordered by merge
/// priority (`merge_index`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaBpeMerge {
    pub merge_index: u32,
    pub left: String,
    pub right: String,
    pub merged_token: String,
    pub has_token_id: bool,
    pub token_id: u32,
}

/// Root schema of the `tokenizer` committed external.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaTokenizer {
    pub metadata: GemmaTokenizerMetadata,
    pub decoder: GemmaDecoderMetadata,
    /// Sorted by `token`, then chunked (`TOKENIZER_CHUNK_WIDTH` entries per
    /// chunk) — consumed as a recur input list, one chunk per tile.
    pub token_lookup_chunks: Vec<Vec<GemmaTokenIdEntry>>,
    /// Indexed by token id (dense; every id present).
    pub tokens_by_id: Vec<GemmaDecodedToken>,
    /// Sorted by content length (desc), then content (asc) — longest-match
    /// special-token scanning order.
    pub special_tokens: Vec<GemmaAddedToken>,
    /// Ordered by merge priority, then chunked (`TOKENIZER_CHUNK_WIDTH`
    /// entries per chunk) — consumed as a recur input list for the
    /// priority-order scan.
    pub merge_chunks: Vec<Vec<GemmaBpeMerge>>,
}
