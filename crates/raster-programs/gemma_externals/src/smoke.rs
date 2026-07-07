//! Smoke tile for the tokenizer committed external: consumes values
//! selected through the schema, proving the encoded artifact resolves and
//! selects end-to-end (driven by the main-crate tokenizer-external test).

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use serde::{Deserialize, Serialize};

use crate::types::{GemmaAddedToken, GemmaTokenIdEntry};

/// Materialized by the smoke program's output; the main-crate test decodes
/// it with a field-order-matching mirror struct (postcard layout contract).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct TokenizerSmoke {
    pub unk_token: String,
    pub first_token: String,
    pub first_token_id: u32,
    pub special_token_count: u32,
}

#[tile]
pub fn summarize_tokenizer(
    unk_token: String,
    first_entry: GemmaTokenIdEntry,
    special_tokens: Vec<GemmaAddedToken>,
) -> Result<TokenizerSmoke> {
    if first_entry.token.is_empty() {
        return Err(String::from(
            "summarize_tokenizer: token_lookup_chunks[0][0] has an empty token",
        ));
    }
    Ok(TokenizerSmoke {
        unk_token,
        first_token: first_entry.token,
        first_token_id: first_entry.id,
        special_token_count: special_tokens.len() as u32,
    })
}
