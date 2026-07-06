//! P3 — external input binding + selection tiles (catalog C13/C14/C22).

use alloc::format;
use alloc::string::String;
use raster::prelude::*;

/// Consumes values selected out of the committed `p3_config` external.
#[tile]
pub fn combine_config(label: String, threshold: u32, scale: u32) -> Result<String> {
    if scale == 0 {
        return Err(String::from("combine_config: scale must be non-zero"));
    }
    Ok(format!("{label}:{}", threshold * scale))
}
