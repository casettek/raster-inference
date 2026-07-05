//! WS0 placeholder library for the `prompt.prepare` program crate.
//!
//! Tiles authored here follow the real-raster constraints (no_std + alloc,
//! free functions, serde-compatible types, `raster::exec::Result<T>` String
//! errors). The placeholder tile below exists only to prove the build; the
//! real `prompt.prepare` tile set is WS3 scope, re-expressed from the sim
//! path (`src/routines/prompt_prepare/raster/`), which remains the logical
//! specification.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

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
