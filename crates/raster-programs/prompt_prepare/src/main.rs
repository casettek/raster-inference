//! `prompt.prepare` program host (WS3).
//!
//! Committed inputs (staged by the routine's `run_raster_core` host adapter
//! through `StagedInputs` — see the port plan's I/O inventory):
//! - `tokenizer`      — raster-encoded `GemmaTokenizer` external, mmap
//! - `initial_pieces` — postcard `BpePieces`, the host-derived initial
//!   BPE pieces behind a selectable root (the sim's pre-staged
//!   `bpe-pieces-0`); byte-identical to the bare `Vec<String>`
//!
//! The model-scoped tokenizer tables stay behind the committed `tokenizer`
//! external. The program threads small source descriptors and chunk
//! ordinals; tiles resolve the selected chunks from raster storage inside
//! tile execution.
//!
//! The materialized outcome goes back to the host through the WS2
//! output-file convention (`raster-program-support`,
//! `RASTER_CORE_OUTPUT_PATH`); a terminal `Err` is a committed result, not
//! a broken run.

use raster::prelude::*;
use raster::println;
use raster_program_gemma_externals::types::GemmaTokenizer;
use raster_program_prompt_prepare::routine::*;
use raster_program_prompt_prepare::types::{BpePieces, PromptTokenization, TokenizerTables};

#[sequence]
fn main() {
    let _tokenizer = external!(GemmaTokenizer, "tokenizer");
    let tokenizer_tables = TokenizerTables::external("tokenizer");
    let initial_pieces = external!(BpePieces, "initial_pieces");

    let outcome = materialize_auth_result::<PromptTokenization, _>(call_seq!(
        tokenize_prompt_pieces,
        initial_pieces,
        tokenizer_tables
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
