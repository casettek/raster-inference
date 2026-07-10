//! Real-raster program for the `input.embedding` routine.
//!
//! The sim path (`src/routines/input_embedding/raster/` in the main crate) is
//! the logical specification. This crate re-expresses the routine with
//! storage-read real-raster constraints: prompt token ids and embedding rows
//! are committed raster inputs, loop state carries only scalars and source
//! descriptors, and activation rows materialize only at the host boundary.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod embedding;
#[cfg(all(test, feature = "std"))]
pub mod guards;
pub mod prompt_tokens;
pub mod routine;
#[cfg(all(test, feature = "std"))]
pub mod test_trace;
pub mod types;
