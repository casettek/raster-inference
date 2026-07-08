//! Real-raster program for the `decode.select_token` routine.
//!
//! The sim path (`src/routines/decode_select_token/raster/` in the main
//! crate) is the logical specification. This crate re-expresses that routine
//! with storage-read real-raster constraints: staged vectors are published
//! once, recur state carries refs and scalar cursors, and append-only token
//! outputs are built with drafts.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[cfg(all(test, feature = "std"))]
pub mod guards;
pub mod logits;
pub mod routine;
#[cfg(all(test, feature = "std"))]
pub mod test_trace;
pub mod token_ids;
pub mod types;
