//! Real-raster program for the `prompt.prepare` routine (WS3).
//!
//! Ported from the sim specification (`src/routines/prompt_prepare/raster/`
//! in the main crate) per the port plan
//! (`src/routines/prompt_prepare/raster_core/PORT_PLAN.md`). Tiles follow
//! the real-raster constraints: no_std + alloc, free functions,
//! serde-compatible owned types, `raster::exec::Result<T>` String errors.
//!
//! Layout constraint (WS1 catalog C33): tiles and sequences live one module
//! per file; this root only declares modules.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod budgets;
pub mod placeholder;
pub mod types;
