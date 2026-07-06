//! WS0 placeholder tile, kept while the port lands so the existing `main`
//! binary keeps compiling; removed with the real routine sequence (port
//! plan commit 3).

use alloc::string::String;
use raster::prelude::*;

/// WS0 placeholder tile; replaced by the real prompt.prepare tiles in WS3.
#[tile]
pub fn placeholder_echo_prompt(prompt: String) -> Result<String> {
    if prompt.is_empty() {
        return Err(String::from("prompt must not be empty"));
    }
    Ok(prompt)
}
