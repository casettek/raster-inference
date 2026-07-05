//! WS0 placeholder host for the `prompt.prepare` program crate.
//!
//! Mirrors the tokenizer-PoC host shape: a `#[sequence] fn main()` that binds
//! committed external inputs and dispatches into the library tiles. Never
//! invoked by any raster-inference binary or test in WS0; execution wiring is
//! WS2/WS3 scope.

use raster::prelude::*;
use raster_program_prompt_prepare::*;

#[sequence]
fn main() {
    let prompt = select!(String, external!(String, "prompt"));
    let _prepared = call!(placeholder_echo_prompt, prompt);
}
