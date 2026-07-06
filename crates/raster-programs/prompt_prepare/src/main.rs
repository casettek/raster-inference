//! `prompt.prepare` program host (WS3).
//!
//! Committed inputs (staged by the routine's `run_raster_core` host adapter
//! through `StagedInputs` — see the port plan's I/O inventory):
//! - `tokenizer`      — raster-encoded `GemmaTokenizer` external, mmap
//! - `initial_pieces` — postcard `Vec<String>`, the host-derived initial
//!   BPE pieces (the sim's pre-staged `bpe-pieces-0`)
//! - `bpe_config`     — postcard `BpeConfig` chunk widths
//!
//! The materialized outcome goes back to the host through the WS2
//! output-file convention (`raster-program-support`,
//! `RASTER_CORE_OUTPUT_PATH`); a terminal `Err` is a committed result, not
//! a broken run.

use raster::prelude::*;
use raster::println;
use raster_program_gemma_externals::types::{
    GemmaBpeMergeLookupEntry, GemmaTokenIdEntry, GemmaTokenizer,
};
use raster_program_prompt_prepare::routine::*;
use raster_program_prompt_prepare::types::{BpeConfig, PromptTokenization};

#[sequence]
fn main() {
    let tokenizer = external!(GemmaTokenizer, "tokenizer");
    let token_lookup = select!(Vec<GemmaTokenIdEntry>, tokenizer.clone().token_lookup);
    let merge_lookup = select!(Vec<GemmaBpeMergeLookupEntry>, tokenizer.merge_lookup);
    let initial_pieces = select!(Vec<String>, external!(Vec<String>, "initial_pieces"));
    let config = select!(BpeConfig, external!(BpeConfig, "bpe_config"));

    let outcome = materialize_auth_result::<PromptTokenization, _>(call_seq!(
        tokenize_prompt_pieces,
        initial_pieces,
        config,
        token_lookup,
        merge_lookup
    ));
    raster_program_support::write_program_output(&outcome);
    match &outcome {
        Ok(tokenization) => println!(
            "prompt.prepare ok: {} prompt token ids",
            tokenization.token_count
        ),
        Err(message) => println!("prompt.prepare terminal err: {message}"),
    }
}
