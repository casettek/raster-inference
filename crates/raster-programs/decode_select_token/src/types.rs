//! Cross-tile types for the `decode.select_token` program.
//!
//! Staged logits and token-id vectors stay behind external/internal source
//! descriptors. Loop state stays scalar/source sized; token outputs accumulate
//! in `Draft<DecodeSelectTokenDraft>`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use raster::{
    ExternalRef, ExternalSelection, InternalRef, Selectable, SelectorPath, SelectorSegment,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct DecodeSelectLogits {
    /// Shape of the canonical logits. Accepted forms are Nx1 and 1xN.
    pub row_count: u32,
    pub width: u32,
    /// Row-major canonical Act bits.
    pub bits: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct DecodeSelectTokenIds {
    pub token_count: u32,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeSelectEncodedInputs {
    pub logits: DecodeSelectLogits,
    pub full_token_ids: DecodeSelectTokenIds,
    pub generated_token_ids: DecodeSelectTokenIds,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct DecodeSelectTokenDraft {
    pub token_ids: Vec<u32>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecodeSelectSource {
    External(ExternalRef),
    Internal(InternalRef),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeSelectLogitSource {
    pub source: DecodeSelectSource,
}

impl DecodeSelectLogitSource {
    pub fn external(name: &str) -> Self {
        Self {
            source: DecodeSelectSource::External(ExternalRef::new(name)),
        }
    }

    pub fn internal(reference: InternalRef) -> Self {
        Self {
            source: DecodeSelectSource::Internal(reference),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeSelectTokenSource {
    pub source: DecodeSelectSource,
}

impl DecodeSelectTokenSource {
    pub fn external(name: &str) -> Self {
        Self {
            source: DecodeSelectSource::External(ExternalRef::new(name)),
        }
    }

    pub fn internal(reference: InternalRef) -> Self {
        Self {
            source: DecodeSelectSource::Internal(reference),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct DecodeSelectConfig {
    pub logits_per_tile: u32,
    pub token_ids_per_tile: u32,
}

/// Chunk-ordinal recur drivers. Each ordinal names one chunk of
/// `logits_per_tile` logits (or `token_ids_per_tile` token ids); tiles read
/// the chunk's elements from storage inside the tile body, so driver length
/// scales with `count / per_tile`, not with the element count.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct DecodeSelectLoopDrivers {
    pub logit_ordinals: Vec<u32>,
    pub full_token_ordinals: Vec<u32>,
    pub generated_token_ordinals: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct DecodeSelectCounts {
    pub full_token_count: u32,
    pub generated_token_count: u32,
    pub logit_count: u32,
    pub logits_per_tile: u32,
    pub token_ids_per_tile: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeSelectArgmaxState {
    pub initialized: bool,
    pub next_token_idx: u32,
    pub best_token_id: u32,
    pub best_logit_bits: i32,
    pub has_error: bool,
    pub error: String,
}

impl DecodeSelectArgmaxState {
    pub fn initial() -> Self {
        Self {
            initialized: false,
            next_token_idx: 0,
            best_token_id: 0,
            best_logit_bits: 0,
            has_error: false,
            error: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeSelectCopyState {
    pub next_token_idx: u32,
}

impl DecodeSelectCopyState {
    pub fn initial() -> Self {
        Self { next_token_idx: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeSelectSelectedState {
    pub next_token: u32,
    pub logit_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct DecodeSelectOutput {
    pub next_token: u32,
    pub full_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub logit_count: u32,
}

pub(crate) fn field_index_selector(field: &str, index: u32) -> SelectorPath {
    SelectorPath::new(vec![
        SelectorSegment::Field(field.to_string()),
        SelectorSegment::Index(index as u64),
    ])
}

pub(crate) fn field_selector(field: &str) -> SelectorPath {
    SelectorPath::new(vec![SelectorSegment::Field(field.to_string())])
}

pub(crate) fn read_logit_selection<T>(
    source: &DecodeSelectLogitSource,
    selector: SelectorPath,
    context: &str,
) -> T
where
    T: DeserializeOwned + Serialize,
{
    read_source_selection::<DecodeSelectLogits, T>(&source.source, selector, context)
}

pub(crate) fn read_token_selection<T>(
    source: &DecodeSelectTokenSource,
    selector: SelectorPath,
    context: &str,
) -> T
where
    T: DeserializeOwned + Serialize,
{
    read_source_selection::<DecodeSelectTokenIds, T>(&source.source, selector, context)
}

fn read_source_selection<Root, T>(
    source: &DecodeSelectSource,
    selector: SelectorPath,
    context: &str,
) -> T
where
    Root: DeserializeOwned + Serialize + Selectable,
    T: DeserializeOwned + Serialize,
{
    match source {
        DecodeSelectSource::External(reference) => {
            raster::resolve_typed_external_value::<Root, T>(ExternalSelection {
                reference: ExternalRef::with_selector(reference.name.clone(), selector),
            })
            .unwrap_or_else(|error| panic!("Failed to resolve decode select {context}: {error}"))
            .value
        }
        DecodeSelectSource::Internal(reference) => {
            raster::select_stored_internal_value::<T>(reference, &selector)
                .unwrap_or_else(|error| {
                    panic!("Failed to resolve decode select {context}: {error}")
                })
                .value
        }
    }
}

pub(crate) fn read_logit_bits(source: &DecodeSelectLogitSource, token_idx: u32) -> i32 {
    read_logit_selection::<i32>(
        source,
        field_index_selector("bits", token_idx),
        "logit bits",
    )
}

pub(crate) fn read_token_id(source: &DecodeSelectTokenSource, token_idx: u32) -> u32 {
    read_token_selection::<u32>(
        source,
        field_index_selector("token_ids", token_idx),
        "token id",
    )
}

/// Number of recur-loop chunks needed to cover `item_count` items at
/// `per_tile` items per chunk. Zero-sized chunks produce an empty loop; the
/// init tile rejects them before any recur loop runs.
pub(crate) fn chunk_count(item_count: u32, per_tile: u32) -> u32 {
    if per_tile == 0 {
        return 0;
    }
    item_count.div_ceil(per_tile)
}

pub(crate) fn shape_logit_count(row_count: u32, width: u32) -> core::result::Result<u32, String> {
    let _expected_len = (row_count as usize)
        .checked_mul(width as usize)
        .ok_or_else(|| "raster decode select logits shape size overflowed".to_string())?;
    match (row_count, width) {
        (0, _) | (_, 0) => {
            Err("raster decode select token requires at least one canonical logit".to_string())
        }
        (rows, 1) => Ok(rows),
        (1, cols) => Ok(cols),
        (rows, width) => Err(format!(
            "raster decode select logits shape {rows}x{width} must be Nx1 or 1xN"
        )),
    }
}
