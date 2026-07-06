//! WS2 round-trip fixture program.
//!
//! Committed inputs (staged by the main-crate test through `StagedInputs`):
//! - `values`  — postcard `Vec<u64>`, read
//! - `divisor` — postcard `u64`, read
//! - `config`  — raster-encoded `FixtureConfig`, mmap (pre-encoded by this
//!   crate's `encode` bin)
//!
//! Computes `sum(values) * config.scale / divisor` through tiles, then hands
//! the materialized outcome to the host via the output-file convention
//! (`raster-program-support`, `RASTER_CORE_OUTPUT_PATH`). A zero divisor
//! makes the outcome a committed terminal `Err`.

use raster::prelude::*;
use raster::println;

use raster_program_roundtrip::tiles::*;
use raster_program_roundtrip::types::FixtureConfig;

#[sequence]
fn compute_weighted_quotient() -> Result<u64> {
    let values = select!(Vec<u64>, external!(Vec<u64>, "values"));
    let scale = select!(u64, external!(FixtureConfig, "config").scale);
    let sum = call!(weighted_sum, values, scale);
    let divisor = select!(u64, external!(u64, "divisor"));
    let quotient = call!(checked_div, sum, divisor)?;
    Ok(quotient)
}

#[sequence]
fn main() {
    let outcome = materialize_auth_result::<u64, _>(call_seq!(compute_weighted_quotient));
    raster_program_support::write_program_output(&outcome);
    match &outcome {
        Ok(value) => println!("roundtrip ok: {value}"),
        Err(message) => println!("roundtrip terminal err: {message}"),
    }
}
