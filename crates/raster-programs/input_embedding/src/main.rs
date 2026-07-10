//! `input.embedding` program host.
//!
//! Committed inputs are staged by the routine's `run_raster_core` host
//! adapter:
//! - `prompt_token_ids` — raster-encoded token ids reconstructed from the
//!   upstream committed `PromptPreparationState`
//! - `embedding` — raster-encoded Gemma input embedding table
//! - `loop_drivers` — postcard chunk-ordinal drivers
//! - `config` — postcard roots, commitments, and chunk sizing
//!
//! The materialized outcome goes back to the host through the WS2 output-file
//! convention. A terminal `Err` is a committed result.

use raster::prelude::*;
use raster::println;
use raster_program_input_embedding::routine::*;
use raster_program_input_embedding::types::{
    EmbeddingSource, GemmaInputEmbeddingTable, InputEmbeddingConfig, InputEmbeddingLoopDrivers,
    InputEmbeddingOutput, InputEmbeddingPromptTokenIds, PromptTokenSource,
};

#[sequence]
fn main() {
    let _prompt_token_ids = external!(InputEmbeddingPromptTokenIds, "prompt_token_ids");
    let _embedding = external!(GemmaInputEmbeddingTable, "embedding");
    let loop_drivers = external!(InputEmbeddingLoopDrivers, "loop_drivers");
    let config = external!(InputEmbeddingConfig, "config");
    let prompt_token_ids = PromptTokenSource::external("prompt_token_ids");
    let embedding = EmbeddingSource::external("embedding");

    let outcome = materialize_auth_result::<InputEmbeddingOutput, _>(call_seq!(
        embed_input_tokens,
        prompt_token_ids,
        embedding,
        loop_drivers,
        config
    ));
    raster_program_support::write_program_output(&outcome);
    match &outcome {
        Ok(output) => println!(
            "input.embedding ok: {} activation rows width {}",
            output.prompt_token_count, output.hidden_size
        ),
        Err(message) => println!("input.embedding terminal err: {message}"),
    }
}
