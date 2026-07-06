//! WS2 round-trip fixture tiles.

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;

/// Deterministic aggregation over the postcard-staged value list, weighted
/// by the raster-encoded config's scale.
#[tile]
pub fn weighted_sum(values: Vec<u64>, scale: u64) -> u64 {
    values.iter().sum::<u64>() * scale
}

/// Fallible tile: the round-trip test's terminal-outcome leg stages a zero
/// divisor so this Err becomes the program's committed outcome.
#[tile]
pub fn checked_div(numerator: u64, divisor: u64) -> Result<u64> {
    if divisor == 0 {
        return Err(String::from("checked_div: divisor is zero"));
    }
    Ok(numerator / divisor)
}
