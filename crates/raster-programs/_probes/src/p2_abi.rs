//! P2 — multi-arg tile ABI and the user error contract (catalog C1/C12/C23/C28).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;

use crate::types::ProbeMode;

/// Two-argument tile (postcard tuple ABI).
#[tile]
pub fn add_pair(a: u64, b: u64) -> u64 {
    a + b
}

/// Three-argument fallible tile: owned String + Vec + scalar.
#[tile]
pub fn describe_triple(label: String, values: Vec<u32>, scale: u32) -> Result<String> {
    if values.is_empty() {
        return Err(String::from("describe_triple: values must not be empty"));
    }
    let sum: u32 = values.iter().sum();
    Ok(format!("{label}:{}", sum * scale))
}

/// Fallible tile exercising both Ok and Err terminal outcomes.
#[tile]
pub fn checked_div(numerator: u64, divisor: u64) -> Result<u64> {
    if divisor == 0 {
        return Err(String::from("checked_div: divisor is zero"));
    }
    Ok(numerator / divisor)
}

/// Enum-carrying ABI (serde enum through postcard).
#[tile]
pub fn mode_name(mode: ProbeMode) -> String {
    match mode {
        ProbeMode::Fast => String::from("fast"),
        ProbeMode::Careful { retries } => format!("careful:{retries}"),
    }
}
