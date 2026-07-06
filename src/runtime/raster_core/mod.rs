//! Host-side scaffolding for raster-core (real toolchain) detour execution.
//!
//! WS0 scope: run-directory conventions and the `cargo raster run` subprocess
//! wrapper only — no routine executes on this path yet. The execution shape
//! (subprocess-first, per the migration ADR) is:
//!
//! 1. The detour stages committed inputs into a fresh run directory
//!    (`input.json` + `input_manifest.json`, tokenizer-PoC idiom: logical
//!    input name → file binding, and logical input name → commitment).
//! 2. The host invokes `cargo raster run --backend native --input ...
//!    --input-manifest ... --commit ...` inside the routine's program crate
//!    (`crates/raster-programs/<routine>/`).
//! 3. Outputs and the commit artifact are ingested back into native state
//!    (WS2 scope).
//!
//! Nothing in this module touches committed checkpoint payloads; raster-core
//! details (run directories, commit artifacts, backend labels) never enter
//! checkpoint schemas.
//!
//! Known upstream gap (recorded in the migration ADR): `cargo raster run`
//! prints a failed guest process's exit status but still exits `0`, so the
//! wrapper's exit-status check is not sufficient to detect guest failure.
//! Ingestion (WS2+) must validate the produced artifacts instead of trusting
//! the CLI exit code.

pub mod ingest;
pub mod staging;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

use crate::runtime::checkpoints::RoutineId;

/// File name for the private input bindings inside a run directory.
pub const INPUT_FILE_NAME: &str = "input.json";
/// File name for the public input commitments inside a run directory.
pub const INPUT_MANIFEST_FILE_NAME: &str = "input_manifest.json";
/// File name for the trace commitment artifact produced by `--commit`.
pub const COMMIT_FILE_NAME: &str = "commit.bin";
/// File name for the program's materialized output value (WS2 convention:
/// the program writes `postcard(Result<T, String>)` here — see
/// [`OUTPUT_PATH_ENV`]).
pub const OUTPUT_FILE_NAME: &str = "output.bin";
/// Environment variable through which the host hands the guest program its
/// output-file path. `cargo raster run` has no result channel of its own
/// (values live only in the raster-formatted trace, which the main crate
/// cannot decode without a `raster` dependency), so the program's
/// `#[sequence] fn main()` materializes its result and writes it to this
/// path via the `raster-program-support` helper.
pub const OUTPUT_PATH_ENV: &str = "RASTER_CORE_OUTPUT_PATH";

/// A fresh, uniquely named run directory for one raster-core routine
/// invocation, holding the staged inputs and the commit artifact.
///
/// The directory is *not* deleted on drop: run directories are debugging
/// evidence for parity failures, and callers that want cleanup do it
/// explicitly once ingestion has succeeded.
#[derive(Debug)]
pub struct RasterCoreRunDir {
    root: PathBuf,
}

impl RasterCoreRunDir {
    /// Creates a unique run directory for the given routine occurrence under
    /// the system temp dir.
    pub fn create(routine_id: RoutineId, occurrence: usize) -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "raster-core-{}-{}-{}-{}",
            routine_id.as_str().replace('.', "_"),
            occurrence,
            std::process::id(),
            nonce,
        ));
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed to create raster-core run dir {}", root.display()))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn input_path(&self) -> PathBuf {
        self.root.join(INPUT_FILE_NAME)
    }

    pub fn input_manifest_path(&self) -> PathBuf {
        self.root.join(INPUT_MANIFEST_FILE_NAME)
    }

    pub fn commit_path(&self) -> PathBuf {
        self.root.join(COMMIT_FILE_NAME)
    }

    pub fn output_path(&self) -> PathBuf {
        self.root.join(OUTPUT_FILE_NAME)
    }
}

/// Output of one `cargo raster run` invocation.
///
/// `run_artifacts_dir` and `trace_path` are parsed from the CLI's stdout
/// banner (`Run artifacts dir: …` / `Trace path: …`) — the only mechanism
/// the CLI offers for discovering which `target/raster/runs/<run-id>/`
/// directory an invocation created. `None` when the banner was absent
/// (e.g. the CLI failed before creating run artifacts).
///
/// `cli_success` records the CLI's exit status for debugging only — it is
/// meaningless for run-outcome decisions in *both* directions: the CLI
/// exits `0` when the guest program fails (ADR gap G4), and it can exit
/// non-zero for failures that ingestion must still classify (e.g. the CLI
/// panics building a trace commitment after a guest integrity rejection
/// leaves the trace shorter than the verification window). Ingestion
/// validates the produced artifacts instead.
#[derive(Debug)]
pub struct CargoRasterRunOutput {
    pub stdout: String,
    pub stderr: String,
    pub cli_success: bool,
    pub run_artifacts_dir: Option<PathBuf>,
    pub trace_path: Option<PathBuf>,
}

/// Extracts the path following `prefix` on any stdout line.
fn parse_stdout_path(stdout: &str, prefix: &str) -> Option<PathBuf> {
    stdout.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix(prefix)
            .map(|rest| PathBuf::from(rest.trim()))
    })
}

/// Subprocess wrapper for the real toolchain's `cargo raster run`.
///
/// Invokes the `cargo-raster` binary directly (with the `raster` subcommand
/// prefix cargo would normally supply) so execution does not depend on cargo
/// subcommand resolution.
#[derive(Debug)]
pub struct CargoRasterRunner {
    program: OsString,
}

impl Default for CargoRasterRunner {
    fn default() -> Self {
        Self {
            program: OsString::from("cargo-raster"),
        }
    }
}

impl CargoRasterRunner {
    /// Overrides the `cargo-raster` program path (tests, hermetic setups).
    pub fn with_program(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
        }
    }

    /// Runs `cargo raster run --backend native` for the program crate at
    /// `program_crate_dir` against the staged inputs in `run_dir`, writing
    /// the trace commitment to the run directory's commit path.
    ///
    /// The trace is always requested as ndjson (`--trace-format json`) so
    /// ingestion can validate run structure with plain `serde_json` — the
    /// binary trace format is raster-internal postcard the main crate must
    /// not depend on. The guest program inherits [`OUTPUT_PATH_ENV`] pointing
    /// at the run directory's output path.
    pub fn run(
        &self,
        program_crate_dir: &Path,
        run_dir: &RasterCoreRunDir,
    ) -> Result<CargoRasterRunOutput> {
        let output = Command::new(&self.program)
            .arg("raster")
            .arg("run")
            .arg("--backend")
            .arg("native")
            .arg("--input")
            .arg(run_dir.input_path())
            .arg("--input-manifest")
            .arg(run_dir.input_manifest_path())
            .arg("--commit")
            .arg(run_dir.commit_path())
            .arg("--trace-format")
            .arg("json")
            .env(OUTPUT_PATH_ENV, run_dir.output_path())
            .current_dir(program_crate_dir)
            .output()
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    anyhow::anyhow!(
                        "cargo-raster was not found on PATH; install the raster toolchain CLI \
                         (cargo install --path <raster checkout>/crates/raster-cli) to run \
                         raster-core detours"
                    )
                } else {
                    anyhow::Error::new(error).context(format!(
                        "failed to launch cargo-raster for program crate {}",
                        program_crate_dir.display()
                    ))
                }
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        // A non-zero CLI exit is *not* an error here: run-outcome
        // classification belongs to ingestion, which validates artifacts
        // (see `CargoRasterRunOutput::cli_success`).
        // Banner paths are printed relative to the CLI's working directory
        // (the program crate); resolve them so callers can read them from
        // any cwd.
        let resolve = |path: PathBuf| {
            if path.is_absolute() {
                path
            } else {
                program_crate_dir.join(path)
            }
        };
        let run_artifacts_dir = parse_stdout_path(&stdout, "Run artifacts dir:").map(resolve);
        let trace_path = parse_stdout_path(&stdout, "Trace path:").map(resolve);
        Ok(CargoRasterRunOutput {
            stdout,
            stderr,
            cli_success: output.status.success(),
            run_artifacts_dir,
            trace_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CargoRasterRunner, RasterCoreRunDir};
    use crate::runtime::checkpoints::RoutineId;

    #[test]
    fn run_dir_uses_committed_input_conventions() {
        let run_dir = RasterCoreRunDir::create(RoutineId::PromptPrepare, 2)
            .expect("run dir should be created");
        assert!(run_dir.root().is_dir());
        assert!(run_dir.input_path().ends_with("input.json"));
        assert!(run_dir
            .input_manifest_path()
            .ends_with("input_manifest.json"));
        assert!(run_dir.commit_path().ends_with("commit.bin"));
        assert!(run_dir.output_path().ends_with("output.bin"));

        let second = RasterCoreRunDir::create(RoutineId::PromptPrepare, 2)
            .expect("second run dir should be created");
        assert_ne!(
            run_dir.root(),
            second.root(),
            "run directories must be unique per invocation"
        );

        std::fs::remove_dir_all(run_dir.root()).ok();
        std::fs::remove_dir_all(second.root()).ok();
    }

    #[test]
    fn run_banner_paths_are_parsed_from_stdout() {
        let stdout = "Raster Run\n  Project: fixture\n  Run ID: 001-pid1-000001\n  \
                      Run artifacts dir: /tmp/target/raster/runs/001\n  \
                      Trace path: /tmp/target/raster/runs/001/trace.ndjson\n";
        assert_eq!(
            super::parse_stdout_path(stdout, "Run artifacts dir:"),
            Some(std::path::PathBuf::from("/tmp/target/raster/runs/001"))
        );
        assert_eq!(
            super::parse_stdout_path(stdout, "Trace path:"),
            Some(std::path::PathBuf::from(
                "/tmp/target/raster/runs/001/trace.ndjson"
            ))
        );
        assert_eq!(super::parse_stdout_path(stdout, "Commit path:"), None);
    }

    #[test]
    fn missing_cargo_raster_binary_errors_clearly() {
        let run_dir = RasterCoreRunDir::create(RoutineId::PrefillRange, 1)
            .expect("run dir should be created");
        let runner = CargoRasterRunner::with_program("cargo-raster-definitely-not-installed");
        let error = runner
            .run(run_dir.root(), &run_dir)
            .expect_err("missing cargo-raster should fail");
        assert!(
            error
                .to_string()
                .contains("cargo-raster was not found on PATH"),
            "unexpected error: {error:#}"
        );
        std::fs::remove_dir_all(run_dir.root()).ok();
    }
}
