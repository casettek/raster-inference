//! Host-boundary output helper for raster-core program crates (WS2).
//!
//! `cargo raster run` offers no channel for a program's result values: the
//! CLI captures only `[output]` stdout lines, and the trace/commit artifacts
//! are raster-internal formats the raster-inference host must not decode
//! (charter invariant 6: no `raster` dependency in the main crate). The WS2
//! convention closes that gap: the host sets [`OUTPUT_PATH_ENV`] on the
//! `cargo raster run` subprocess (inherited by the guest binary), and the
//! program's `#[sequence] fn main()` ends by materializing its result and
//! calling [`write_program_output`], which writes `postcard(Result<T,
//! String>)` to that path.
//!
//! A terminal `Err` is written like any other outcome — it is a committed,
//! fault-provable result, and the host classifies it separately from broken
//! runs (see `RasterCoreError` in the main crate).
//!
//! Everything here is host-side std code; the no_std (guest) surface of this
//! crate is intentionally empty.

#![cfg_attr(not(feature = "std"), no_std)]

/// Environment variable naming the output file. Must match
/// `runtime::raster_core::OUTPUT_PATH_ENV` in the raster-inference crate
/// (asserted by the WS2 round-trip test).
pub const OUTPUT_PATH_ENV: &str = "RASTER_CORE_OUTPUT_PATH";

/// Writes the program's materialized outcome to the host-provided output
/// path as `postcard(Result<T, String>)`.
///
/// Call once, at the end of `#[sequence] fn main()`, after materializing the
/// program's result. When [`OUTPUT_PATH_ENV`] is unset (e.g. a manual
/// `cargo raster run` outside the raster-inference host), the outcome is not
/// written anywhere and this is a no-op.
///
/// Panics on serialization or IO failure: at that point the program's work
/// (and its trace) is already complete, and a loud infrastructure failure is
/// the correct surface — the host treats a missing output file as an
/// infrastructure error, never a committed outcome.
#[cfg(feature = "std")]
pub fn write_program_output<T: serde::Serialize>(outcome: &Result<T, std::string::String>) {
    let Some(path) = std::env::var_os(OUTPUT_PATH_ENV) else {
        return;
    };
    let bytes = postcard::to_allocvec(outcome)
        .unwrap_or_else(|error| panic!("failed to postcard-encode program output: {error}"));
    std::fs::write(&path, bytes).unwrap_or_else(|error| {
        panic!(
            "failed to write program output to {}: {error}",
            std::path::Path::new(&path).display()
        )
    });
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Single test (not split) because it mutates the process-wide env var.
    #[test]
    fn round_trips_outcomes_and_ignores_a_missing_env_var() {
        std::env::remove_var(OUTPUT_PATH_ENV);
        write_program_output(&Ok(1u8)); // no-op, must not panic

        let dir = std::env::temp_dir().join(format!(
            "raster-program-support-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("test dir");

        for (name, outcome) in [
            ("ok.bin", Ok(1234u64)),
            ("err.bin", Err("terminal outcome".to_string())),
        ] {
            let path = dir.join(name);
            std::env::set_var(OUTPUT_PATH_ENV, &path);
            write_program_output(&outcome);
            let bytes = std::fs::read(&path).expect("output file");
            let decoded: Result<u64, String> =
                postcard::from_bytes(&bytes).expect("postcard Result payload");
            assert_eq!(decoded, outcome);
        }

        std::env::remove_var(OUTPUT_PATH_ENV);
        std::fs::remove_dir_all(&dir).ok();
    }
}
