//! P5 — sequence composition tiles (catalog C3/C5/C6).

use raster::prelude::*;

#[tile]
pub fn double(x: u64) -> u64 {
    x * 2
}

#[tile]
pub fn inc(x: u64) -> u64 {
    x + 1
}
