//! `decode.select_token` program host.
//!
//! Committed inputs are staged by the host adapter:
//! - `logits` — raster-encoded canonical deterministic logit bits plus shape
//! - `full_token_ids` — raster-encoded token ids before this decode step
//! - `generated_token_ids` — raster-encoded generated ids before this step
//! - `loop_drivers` — postcard chunk-ordinal drivers for explicit recur loops
//! - `config` — postcard real-raster chunk sizing
//!
//! The materialized outcome goes back to the host through the WS2 output-file
//! convention. A terminal `Err` is a committed result.

use raster::prelude::*;
use raster::println;
use raster_program_decode_select_token::routine::*;
use raster_program_decode_select_token::types::{
    DecodeSelectConfig, DecodeSelectLogitSource, DecodeSelectLogits, DecodeSelectLoopDrivers,
    DecodeSelectOutput, DecodeSelectTokenIds, DecodeSelectTokenSource,
};

#[sequence]
fn main() {
    let _logits = external!(DecodeSelectLogits, "logits");
    let _full_token_ids = external!(DecodeSelectTokenIds, "full_token_ids");
    let _generated_token_ids = external!(DecodeSelectTokenIds, "generated_token_ids");
    let loop_drivers = external!(DecodeSelectLoopDrivers, "loop_drivers");
    let config = external!(DecodeSelectConfig, "config");
    let logits = DecodeSelectLogitSource::external("logits");
    let full_token_ids = DecodeSelectTokenSource::external("full_token_ids");
    let generated_token_ids = DecodeSelectTokenSource::external("generated_token_ids");

    let outcome = materialize_auth_result::<DecodeSelectOutput, _>(call_seq!(
        select_decode_token,
        logits,
        full_token_ids,
        generated_token_ids,
        loop_drivers,
        config
    ));
    raster_program_support::write_program_output(&outcome);
    match &outcome {
        Ok(output) => println!("decode.select_token ok: selected {}", output.next_token),
        Err(message) => println!("decode.select_token terminal err: {message}"),
    }
}
