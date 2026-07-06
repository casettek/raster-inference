//! WS2 round-trip fixture tiles and types.
//!
//! Layout constraint (WS1 catalog C33): the CFS builder parses only
//! top-level functions per `.rs` file and skips `lib.rs`-style module roots,
//! so tiles live in their own file and this root only declares modules.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod tiles;
pub mod types;
