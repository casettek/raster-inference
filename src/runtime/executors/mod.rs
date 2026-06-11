//! Executor seam: per-routine native vs raster execution beneath the
//! phase-sequencing skeleton (`runtime::sequence`).
//!
//! The two things the protocol compares — native and raster execution of the
//! same routine — live in sibling modules behind this seam so neither path is
//! constructed by interleaved flag plumbing:
//!
//! - [`native`]: the native deterministic/fp32 executor, including the
//!   selective raster detour hooks (the detour is the native executor
//!   swapping in exactly one raster routine occurrence, governed by
//!   `RasterDetourController`).
//! - [`raster`]: the full root-backed raster tile executor.

pub(crate) mod native;
