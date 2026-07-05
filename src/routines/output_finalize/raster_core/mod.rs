//! Raster-core (real toolchain) host adapter for this routine.
//!
//! WS0 stub: this module gains exactly one public `run_raster_core`
//! entrypoint when the routine's WS3 migration lands (tiles authored in the
//! routine's program crate under `crates/raster-programs/`, staged inputs
//! committed via the WS2 conventions, outputs and commit artifacts ingested
//! back into native state). Until then it must expose no public functions —
//! the guard test in `src/routines/mod.rs` enforces the contract per routine
//! as migration lands.
