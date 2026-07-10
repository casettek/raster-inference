//! Cross-tile types for the `input.embedding` program.
//!
//! Staged prompt token ids and embedding rows stay behind external/internal
//! source descriptors. Recur state is scalar-sized; activation rows accumulate
//! in a draft and are materialized only as the routine's final output.

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
pub struct InputEmbeddingPromptTokenIds {
    pub token_count: u32,
    pub token_ids: Vec<u32>,
    pub token_ids_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaInputEmbeddingMetadata {
    pub source_id: String,
    pub vocab_size: u32,
    pub hidden_size: u32,
    /// Informational scale bits from the host authenticated source. Rows are
    /// staged as already-scaled canonical Act bits.
    pub scale_bits: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct GemmaInputEmbeddingTable {
    pub metadata: GemmaInputEmbeddingMetadata,
    /// Hex-packed embedding rows: one selectable `String` leaf per row, 8
    /// lowercase hex chars per canonical Act bit pattern (see
    /// [`pack_embedding_row_hex`]). Packing each row into a single leaf
    /// keeps the raster index O(vocab) nodes; a `Vec<Vec<i32>>` shape puts
    /// every value in its own index node, which is unencodable at real
    /// model scale (537M nodes for Gemma E4B).
    pub rows: Vec<String>,
}

/// Packs one embedding row's canonical bit patterns into the schema's
/// hex-string leaf form: 8 lowercase hex chars per value, concatenated.
/// The host adapter's staging mirror must produce byte-identical strings.
pub fn pack_embedding_row_hex(values: &[i32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(values.len() * 8);
    for value in values {
        let bits = *value as u32;
        for shift in (0..8).rev() {
            out.push(HEX[((bits >> (shift * 4)) & 0xf) as usize]);
        }
    }
    String::from_utf8(out).expect("hex packing is always ascii")
}

/// Decodes a hex-packed embedding row back into canonical bit patterns.
/// Errors are committed tile outcomes (malformed staged data), not panics.
pub fn unpack_embedding_row_hex(packed: &str) -> core::result::Result<Vec<i32>, String> {
    let bytes = packed.as_bytes();
    if bytes.len() % 8 != 0 {
        return Err(format!(
            "packed embedding row length {} is not a multiple of 8 hex chars",
            bytes.len()
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut bits: u32 = 0;
        for &ch in chunk {
            let nibble = match ch {
                b'0'..=b'9' => ch - b'0',
                b'a'..=b'f' => ch - b'a' + 10,
                _ => {
                    return Err(format!(
                        "packed embedding row contains non-hex byte 0x{ch:02x}"
                    ))
                }
            };
            bits = (bits << 4) | u32::from(nibble);
        }
        values.push(bits as i32);
    }
    Ok(values)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputEmbeddingEncodedInputs {
    pub prompt_token_ids: Option<InputEmbeddingPromptTokenIds>,
    pub embedding: Option<GemmaInputEmbeddingTable>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InputEmbeddingSource {
    External(ExternalRef),
    Internal(InternalRef),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptTokenSource {
    pub source: InputEmbeddingSource,
}

impl PromptTokenSource {
    pub fn external(name: &str) -> Self {
        Self {
            source: InputEmbeddingSource::External(ExternalRef::new(name)),
        }
    }

    pub fn internal(reference: InternalRef) -> Self {
        Self {
            source: InputEmbeddingSource::Internal(reference),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingSource {
    pub source: InputEmbeddingSource,
}

impl EmbeddingSource {
    pub fn external(name: &str) -> Self {
        Self {
            source: InputEmbeddingSource::External(ExternalRef::new(name)),
        }
    }

    pub fn internal(reference: InternalRef) -> Self {
        Self {
            source: InputEmbeddingSource::Internal(reference),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct InputEmbeddingConfig {
    pub tokens_per_tile: u32,
    pub prompt_token_ids_sha256: String,
    pub prompt_token_ids_root: String,
    pub embedding_source_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct InputEmbeddingLoopDrivers {
    pub token_ordinals: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct InputEmbeddingCounts {
    pub prompt_token_count: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub tokens_per_tile: u32,
    pub source_id: String,
    pub prompt_token_ids_sha256: String,
    pub prompt_token_ids_root: String,
    pub embedding_source_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputEmbeddingCopyState {
    pub next_token_idx: u32,
}

impl InputEmbeddingCopyState {
    pub fn initial() -> Self {
        Self { next_token_idx: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct InputEmbeddingActivationDraft {
    pub rows: Vec<Vec<i32>>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct InputEmbeddingOutput {
    pub source_id: String,
    pub prompt_token_ids_sha256: String,
    pub prompt_token_ids_root: String,
    pub embedding_source_root: String,
    pub prompt_token_count: u32,
    pub hidden_size: u32,
    pub activation_rows: Vec<Vec<i32>>,
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

pub(crate) fn read_prompt_selection<T>(
    source: &PromptTokenSource,
    selector: SelectorPath,
    context: &str,
) -> T
where
    T: DeserializeOwned + Serialize,
{
    read_source_selection::<InputEmbeddingPromptTokenIds, T>(&source.source, selector, context)
}

pub(crate) fn read_embedding_selection<T>(
    source: &EmbeddingSource,
    selector: SelectorPath,
    context: &str,
) -> T
where
    T: DeserializeOwned + Serialize,
{
    read_source_selection::<GemmaInputEmbeddingTable, T>(&source.source, selector, context)
}

fn read_source_selection<Root, T>(
    source: &InputEmbeddingSource,
    selector: SelectorPath,
    context: &str,
) -> T
where
    Root: DeserializeOwned + Serialize + Selectable,
    T: DeserializeOwned + Serialize,
{
    match source {
        InputEmbeddingSource::External(reference) => {
            raster::resolve_typed_external_value::<Root, T>(ExternalSelection {
                reference: ExternalRef::with_selector(reference.name.clone(), selector),
            })
            .unwrap_or_else(|error| panic!("Failed to resolve input embedding {context}: {error}"))
            .value
        }
        InputEmbeddingSource::Internal(reference) => {
            raster::select_stored_internal_value::<T>(reference, &selector)
                .unwrap_or_else(|error| {
                    panic!("Failed to resolve input embedding {context}: {error}")
                })
                .value
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod pack_tests {
    use super::{pack_embedding_row_hex, unpack_embedding_row_hex};
    use alloc::vec;

    #[test]
    fn hex_rows_round_trip_including_negative_bits() {
        let rows = vec![
            vec![0, 1, -1, i32::MIN, i32::MAX, 0x0102_0304],
            vec![-0x0506_0708],
        ];
        for row in rows {
            let packed = pack_embedding_row_hex(&row);
            assert_eq!(packed.len(), row.len() * 8);
            assert!(packed.bytes().all(|ch| ch.is_ascii_hexdigit()
                && !ch.is_ascii_uppercase()));
            assert_eq!(unpack_embedding_row_hex(&packed).expect("round trip"), row);
        }
    }

    #[test]
    fn packed_form_is_stable() {
        assert_eq!(
            pack_embedding_row_hex(&[0x0102_0304, -1]),
            "01020304ffffffff"
        );
    }

    #[test]
    fn unpack_rejects_malformed_rows() {
        let error = unpack_embedding_row_hex("0102030").expect_err("bad length");
        assert!(error.contains("multiple of 8"));
        let error = unpack_embedding_row_hex("0102030Z").expect_err("bad digit");
        assert!(error.contains("non-hex"));
        let error = unpack_embedding_row_hex("0102030F").expect_err("uppercase digit");
        assert!(error.contains("non-hex"));
    }
}

pub(crate) fn chunk_count(item_count: u32, per_tile: u32) -> u32 {
    if per_tile == 0 {
        return 0;
    }
    item_count.div_ceil(per_tile)
}

pub(crate) fn validate_loop_driver(
    label: &str,
    input_index: u64,
    input_len: u64,
    chunk_idx: u32,
    item_count: u32,
    per_tile: u32,
) -> core::result::Result<(), String> {
    let expected_len = chunk_count(item_count, per_tile) as u64;
    let expected_idx = input_index as u32;
    if input_len != expected_len || chunk_idx != expected_idx {
        return Err(format!(
            "raster input embedding {label} loop driver ordinal {input_index} was {chunk_idx} in list len {input_len}, expected {expected_idx} in list len {expected_len}"
        ));
    }
    Ok(())
}
