//! Tokenizer-external smoke program (WS2).
//!
//! Committed input: `tokenizer` — raster-encoded `GemmaTokenizer`
//! (mmap), pre-encoded by this crate's `encode` bin. Selects through the
//! schema (scalar field, list index, whole sub-list), summarizes in a tile,
//! and returns the result via the output-file convention.

use raster::prelude::*;
use raster::println;

use raster_program_gemma_externals::smoke::*;
use raster_program_gemma_externals::types::{GemmaAddedToken, GemmaTokenIdEntry, GemmaTokenizer};

#[sequence]
fn tokenizer_smoke() -> Result<TokenizerSmoke> {
    let unk_token = select!(
        String,
        external!(GemmaTokenizer, "tokenizer").metadata.unk_token
    );
    let first_entry = select!(
        GemmaTokenIdEntry,
        external!(GemmaTokenizer, "tokenizer").token_lookup[0]
    );
    let special_tokens = select!(
        Vec<GemmaAddedToken>,
        external!(GemmaTokenizer, "tokenizer").special_tokens
    );
    let smoke = call!(summarize_tokenizer, unk_token, first_entry, special_tokens)?;
    Ok(smoke)
}

#[sequence]
fn main() {
    let outcome = materialize_auth_result::<TokenizerSmoke, _>(call_seq!(tokenizer_smoke));
    raster_program_support::write_program_output(&outcome);
    match &outcome {
        Ok(smoke) => println!("tokenizer smoke ok: {smoke:?}"),
        Err(message) => println!("tokenizer smoke terminal err: {message}"),
    }
}
