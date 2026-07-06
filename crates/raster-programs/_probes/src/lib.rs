//! WS1 probe tiles.
//!
//! Each module holds the tiles for one catalog probe (P1–P5); the sequences
//! that drive them live in `src/main.rs` (std host program). Types shared with
//! the staging bin live in `types`.
//!
//! Everything here follows the real-raster authoring constraints the catalog
//! records: no_std + alloc, free functions, owned serde types across the tile
//! ABI, `raster::exec::Result<T>` (String) for user-terminal errors.
//!
//! Layout constraint (catalog C33): the CFS builder parses only *top-level*
//! functions per `.rs` file and skips `mod.rs` files entirely
//! (`raster-compiler/src/ast.rs`), so every module here is its own file and
//! no tile/sequence lives inside an inline `mod` block.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod p1_recur;
pub mod p2_abi;
pub mod p3_external;
pub mod p4_draft;
pub mod p5_sequences;
pub mod types;
