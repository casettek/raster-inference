//! Public DSL surface for raster tile and sequence implementations.
//!
//! Rust requires procedural attribute macros such as `#[tile]` and `#[sequence]`
//! to live in a separate `proc-macro` crate. The companion `dsl-macros` crate
//! provides those attributes; this module is the runtime-facing DSL API used by
//! raster routines.

mod macros;
mod runtime;

#[cfg(test)]
mod tests;

pub use crate::shared::artifacts::artifact_io::{auth_read, AuthRead};
pub use dsl_macros::{sequence, tile};
#[doc(hidden)]
pub use runtime::record_tile_invocation;
pub use runtime::{external, start_tile_invocation_counting, stop_tile_invocation_counting};
pub use runtime::{External, ExternalRef};

pub mod prelude {
    pub use crate::{
        auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, dsl::sequence, dsl::tile,
        dsl::AuthRead, dsl::External, dsl::ExternalRef, external,
    };
}
