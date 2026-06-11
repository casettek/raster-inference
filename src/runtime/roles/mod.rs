//! Protocol role entry points.
//!
//! The protocol defines two roles with distinct flows, both built on the
//! phase-sequencing skeleton (`runtime::sequence`) and the executor seam:
//!
//! - [`claimer`]: runs native deterministic inference end-to-end with
//!   checkpoint commitment on, producing the trace artifact for on-chain
//!   commitment.
//! - [`challenger`]: replays a claimed inference natively, locates the first
//!   divergent committed checkpoint, and re-executes the spanning routine
//!   occurrence at raster (tile) level, producing the dispute's raster
//!   detour trace.
//!
//! Both roles use the process-global trace collector and artifact stores;
//! runs must not be interleaved across threads (the same constraint as the
//! legacy entry points).

pub mod challenger;
pub mod claimer;
