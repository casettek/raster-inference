//! `prompt.prepare` program host (WS3).
//!
//! Committed inputs (staged by the routine's `run_raster_core` host adapter
//! through `StagedInputs` — see the port plan's I/O inventory):
//! - `tokenizer`      — raster-encoded `GemmaTokenizer` external, mmap
//! - `initial_pieces` — postcard `BpePieces`, the host-derived initial
//!   BPE pieces behind a selectable root (the sim's pre-staged
//!   `bpe-pieces-0`); byte-identical to the bare `Vec<String>`
//! - `bpe_config`     — postcard `BpeConfig` chunk widths
//!
//! The model-scoped tables are selected as chunked lists
//! (`token_lookup_chunks`, `merge_chunks`) and consumed only as recur
//! input lists — one chunk per tile execution (storage-refactor data
//! placement).
//!
//! The materialized outcome goes back to the host through the WS2
//! output-file convention (`raster-program-support`,
//! `RASTER_CORE_OUTPUT_PATH`); a terminal `Err` is a committed result, not
//! a broken run.

use raster::prelude::*;
use raster::println;
use raster_program_gemma_externals::types::{GemmaBpeMerge, GemmaTokenIdEntry, GemmaTokenizer};
use raster_program_prompt_prepare::routine::*;
use raster_program_prompt_prepare::types::{BpeConfig, BpePieces, PromptTokenization};

#[sequence]
fn main() {
    let tokenizer = external!(GemmaTokenizer, "tokenizer");
    let token_lookup_chunks = select!(
        Vec<Vec<GemmaTokenIdEntry>>,
        tokenizer.clone().token_lookup_chunks
    );
    let merge_chunks = select!(Vec<Vec<GemmaBpeMerge>>, tokenizer.merge_chunks);
    let initial_pieces = external!(BpePieces, "initial_pieces");
    let config = select!(BpeConfig, external!(BpeConfig, "bpe_config"));

    let outcome = materialize_auth_result::<PromptTokenization, _>(call_seq!(
        tokenize_prompt_pieces,
        initial_pieces,
        config,
        token_lookup_chunks,
        merge_chunks
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
