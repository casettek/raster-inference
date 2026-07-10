//! Embedding committed storage reads for `input.embedding`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::types::{
    field_index_selector, field_selector, read_embedding_selection, unpack_embedding_row_hex,
    EmbeddingSource, GemmaInputEmbeddingMetadata,
};

pub(crate) fn read_embedding_metadata(source: &EmbeddingSource) -> GemmaInputEmbeddingMetadata {
    read_embedding_selection::<GemmaInputEmbeddingMetadata>(
        source,
        field_selector("metadata"),
        "embedding metadata",
    )
}

/// Reads one hex-packed row leaf and decodes it. A malformed packed row is
/// a committed `Err` outcome for the calling tile, not a resolution panic.
pub(crate) fn read_embedding_row(
    source: &EmbeddingSource,
    token_id: u32,
) -> core::result::Result<Vec<i32>, String> {
    let packed = read_embedding_selection::<String>(
        source,
        field_index_selector("rows", token_id),
        "embedding row",
    );
    unpack_embedding_row_hex(&packed)
}
