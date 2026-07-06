//! Output and commit-artifact ingestion for raster-core runs (WS2).
//!
//! `cargo raster run` exits `0` even when the guest program fails (migration
//! ADR, named gap G4), so ingestion never trusts exit codes: it validates the
//! artifacts a successful run must have produced — the program's output file,
//! the trace commitment, and a structurally complete trace — and classifies
//! failures into the three H4 error classes (WS1 catalog row C23) that must
//! never collapse into each other:
//!
//! - [`RasterCoreError::Infrastructure`] — the run itself broke (missing
//!   toolchain, staging IO, missing or undecodable artifacts). No committed
//!   outcome exists; retrying or fixing the environment is appropriate.
//! - [`RasterCoreError::Terminal`] — the program ran to completion and
//!   committed an `Err(String)` outcome. This is a fault-provable result, not
//!   a broken run: replaying the trace reproduces it.
//! - [`RasterCoreError::Verification`] — committed-input integrity was
//!   rejected by the runtime at resolve time, or the produced artifacts are
//!   mutually inconsistent (an output value without the trace structure that
//!   should have produced it).

use std::fmt;
use std::path::PathBuf;

use serde::de::DeserializeOwned;

use super::staging::{sha256_hex, StagedInputCommitments};
use super::{CargoRasterRunOutput, RasterCoreRunDir};

/// Stderr signature of the runtime's committed-input integrity rejection
/// (`raster-runtime/src/external_storage.rs::verify_input_commitment`).
const INTEGRITY_REJECTION_SIGNATURE: &str = "failed integrity check";

/// Error contract for one raster-core run (H4 made concrete).
#[derive(Debug)]
pub enum RasterCoreError {
    /// Broken run: no committed outcome exists.
    Infrastructure(anyhow::Error),
    /// Committed, fault-provable `Err` outcome produced by tile logic.
    Terminal { message: String },
    /// Committed-input integrity rejection or artifact inconsistency.
    Verification { detail: String },
}

impl fmt::Display for RasterCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infrastructure(error) => {
                write!(f, "raster-core infrastructure failure: {error:#}")
            }
            Self::Terminal { message } => {
                write!(f, "raster-core terminal outcome: {message}")
            }
            Self::Verification { detail } => {
                write!(f, "raster-core verification failure: {detail}")
            }
        }
    }
}

impl std::error::Error for RasterCoreError {}

/// Everything a host adapter gets back from a validated raster-core run.
#[derive(Debug)]
pub struct RasterCoreRunResult<T> {
    /// The program's materialized output value.
    pub value: T,
    /// Captured public commitments of every staged input (WS3 exit
    /// criterion: staged inputs carry captured commitments visible in the
    /// run result).
    pub input_commitments: StagedInputCommitments,
    /// SHA-256 of the commit artifact (`commit.bin`), captured as an opaque
    /// fingerprint — decoding the trace commitment is WS7 scope.
    pub commit_artifact_sha256: String,
    pub commit_path: PathBuf,
    /// The ndjson trace file inside the CLI's run-artifacts directory.
    pub trace_path: PathBuf,
    /// The staging run directory (kept on failure as debugging evidence).
    pub run_dir_root: PathBuf,
}

/// Validates a finished run's artifacts and materializes the program's
/// output value.
///
/// `T` is the program's output type; the program writes
/// `postcard(Result<T, String>)` to the run directory's output path via the
/// `raster-program-support` helper.
pub fn ingest<T: DeserializeOwned>(
    run_dir: &RasterCoreRunDir,
    run_output: &CargoRasterRunOutput,
    input_commitments: StagedInputCommitments,
) -> Result<RasterCoreRunResult<T>, RasterCoreError> {
    // 1. The program's output file. Its absence means the guest never reached
    //    the end of `main` — distinguish integrity rejection (verification)
    //    from everything else (infrastructure) by the runtime's signature.
    let output_path = run_dir.output_path();
    let output_bytes = match std::fs::read(&output_path) {
        Ok(bytes) => bytes,
        Err(io_error) => {
            let guest_failure = format!(
                "stdout:\n{}\nstderr:\n{}",
                run_output.stdout.trim_end(),
                run_output.stderr.trim_end()
            );
            if run_output.stderr.contains(INTEGRITY_REJECTION_SIGNATURE)
                || run_output.stdout.contains(INTEGRITY_REJECTION_SIGNATURE)
            {
                return Err(RasterCoreError::Verification {
                    detail: format!(
                        "committed input rejected by the runtime integrity check; \
                         no program output was produced\n{guest_failure}"
                    ),
                });
            }
            return Err(RasterCoreError::Infrastructure(anyhow::anyhow!(
                "program output file {} is missing or unreadable ({io_error}); \
                 the guest did not complete\n{guest_failure}",
                output_path.display()
            )));
        }
    };
    let outcome: Result<T, String> = postcard::from_bytes(&output_bytes).map_err(|error| {
        RasterCoreError::Infrastructure(anyhow::anyhow!(
            "program output file {} is not a postcard Result payload: {error}",
            output_path.display()
        ))
    })?;

    // 2. The commit artifact: required evidence for any committed outcome
    //    (Ok or terminal Err), captured as an opaque fingerprint.
    let commit_path = run_dir.commit_path();
    let commit_bytes = std::fs::read(&commit_path).map_err(|error| {
        RasterCoreError::Infrastructure(anyhow::anyhow!(
            "commit artifact {} is missing or unreadable ({error})",
            commit_path.display()
        ))
    })?;
    if commit_bytes.is_empty() {
        return Err(RasterCoreError::Infrastructure(anyhow::anyhow!(
            "commit artifact {} is empty",
            commit_path.display()
        )));
    }

    // 3. Trace structure: an output value must be backed by a trace whose
    //    `main` sequence completed; anything else is an inconsistency.
    let trace_path = run_output.trace_path.clone().ok_or_else(|| {
        RasterCoreError::Infrastructure(anyhow::anyhow!(
            "cargo raster run stdout carried no 'Trace path:' banner; \
             cannot validate the run's trace"
        ))
    })?;
    let trace = std::fs::read_to_string(&trace_path).map_err(|error| {
        RasterCoreError::Infrastructure(anyhow::anyhow!(
            "trace file {} is missing or unreadable ({error})",
            trace_path.display()
        ))
    })?;
    if !trace_contains_main_sequence_end(&trace) {
        return Err(RasterCoreError::Verification {
            detail: format!(
                "program produced an output value but the trace at {} has no \
                 SequenceEnd for 'main'; artifacts are inconsistent",
                trace_path.display()
            ),
        });
    }

    let value = outcome.map_err(|message| RasterCoreError::Terminal { message })?;
    Ok(RasterCoreRunResult {
        value,
        input_commitments,
        commit_artifact_sha256: sha256_hex(&commit_bytes),
        commit_path,
        trace_path,
        run_dir_root: run_dir.root().to_path_buf(),
    })
}

/// Scans an ndjson trace for the `main` sequence's completion event
/// (externally tagged serde enum: `{"SequenceEnd":{"fn_name":"main",…}}`).
fn trace_contains_main_sequence_end(trace_ndjson: &str) -> bool {
    trace_ndjson.lines().any(|line| {
        serde_json::from_str::<serde_json::Value>(line)
            .ok()
            .and_then(|event| {
                event
                    .get("SequenceEnd")
                    .and_then(|record| record.get("fn_name"))
                    .map(|name| name == "main")
            })
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::checkpoints::RoutineId;

    const MAIN_SEQUENCE_END: &str = r#"{"SequenceEnd":{"fn_name":"main","input":null,"output":null,"draft_transition_witness":null}}"#;

    struct Fixture {
        run_dir: RasterCoreRunDir,
        trace_path: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let run_dir = RasterCoreRunDir::create(RoutineId::PromptPrepare, 1).expect("run dir");
            let trace_path = run_dir.root().join("trace.ndjson");
            Self {
                run_dir,
                trace_path,
            }
        }

        fn write_output(&self, outcome: &Result<u64, String>) {
            std::fs::write(
                self.run_dir.output_path(),
                postcard::to_allocvec(outcome).expect("encode outcome"),
            )
            .expect("write output.bin");
        }

        fn write_commit(&self, bytes: &[u8]) {
            std::fs::write(self.run_dir.commit_path(), bytes).expect("write commit.bin");
        }

        fn write_trace(&self, lines: &[&str]) {
            std::fs::write(&self.trace_path, lines.join("\n")).expect("write trace");
        }

        fn run_output(&self, stdout: &str, stderr: &str) -> CargoRasterRunOutput {
            CargoRasterRunOutput {
                stdout: stdout.to_string(),
                stderr: stderr.to_string(),
                run_artifacts_dir: Some(self.run_dir.root().to_path_buf()),
                trace_path: Some(self.trace_path.clone()),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(self.run_dir.root()).ok();
        }
    }

    #[test]
    fn complete_run_ingests_value_commitments_and_commit_hash() {
        let fixture = Fixture::new();
        fixture.write_output(&Ok(42));
        fixture.write_commit(b"commitment-bytes");
        fixture.write_trace(&[
            r#"{"SequenceStart":{"fn_name":"main","input":null,"output":null,"draft_transition_witness":null}}"#,
            MAIN_SEQUENCE_END,
        ]);

        let result: RasterCoreRunResult<u64> = ingest(
            &fixture.run_dir,
            &fixture.run_output("ok", ""),
            StagedInputCommitments::new(),
        )
        .expect("ingest should succeed");
        assert_eq!(result.value, 42);
        assert_eq!(
            result.commit_artifact_sha256,
            sha256_hex(b"commitment-bytes")
        );
        assert_eq!(result.trace_path, fixture.trace_path);
        assert_eq!(result.run_dir_root, fixture.run_dir.root());
    }

    #[test]
    fn terminal_err_outcome_is_classified_terminal() {
        let fixture = Fixture::new();
        fixture.write_output(&Err("divisor is zero".to_string()));
        fixture.write_commit(b"commitment-bytes");
        fixture.write_trace(&[MAIN_SEQUENCE_END]);

        let error = ingest::<u64>(
            &fixture.run_dir,
            &fixture.run_output("", ""),
            StagedInputCommitments::new(),
        )
        .expect_err("terminal outcome expected");
        match error {
            RasterCoreError::Terminal { message } => assert_eq!(message, "divisor is zero"),
            other => panic!("expected Terminal, got {other}"),
        }
    }

    #[test]
    fn integrity_rejection_is_classified_verification() {
        let fixture = Fixture::new();
        // No output file: the guest panicked at resolve time.
        let error = ingest::<u64>(
            &fixture.run_dir,
            &fixture.run_output(
                "",
                "External input 'p3_config' failed integrity check. Expected SHA256 aa, got bb",
            ),
            StagedInputCommitments::new(),
        )
        .expect_err("verification failure expected");
        match error {
            RasterCoreError::Verification { detail } => {
                assert!(detail.contains("integrity check"), "detail: {detail}")
            }
            other => panic!("expected Verification, got {other}"),
        }
    }

    #[test]
    fn missing_output_without_integrity_signature_is_infrastructure() {
        let fixture = Fixture::new();
        let error = ingest::<u64>(
            &fixture.run_dir,
            &fixture.run_output("", "thread 'main' panicked at src/main.rs:1:1"),
            StagedInputCommitments::new(),
        )
        .expect_err("infrastructure failure expected");
        assert!(
            matches!(error, RasterCoreError::Infrastructure(_)),
            "{error}"
        );
    }

    #[test]
    fn missing_or_empty_commit_artifact_is_infrastructure() {
        let fixture = Fixture::new();
        fixture.write_output(&Ok(1));
        fixture.write_trace(&[MAIN_SEQUENCE_END]);
        let error = ingest::<u64>(
            &fixture.run_dir,
            &fixture.run_output("", ""),
            StagedInputCommitments::new(),
        )
        .expect_err("missing commit artifact must fail");
        assert!(
            matches!(error, RasterCoreError::Infrastructure(_)),
            "{error}"
        );

        fixture.write_commit(b"");
        let error = ingest::<u64>(
            &fixture.run_dir,
            &fixture.run_output("", ""),
            StagedInputCommitments::new(),
        )
        .expect_err("empty commit artifact must fail");
        assert!(
            matches!(error, RasterCoreError::Infrastructure(_)),
            "{error}"
        );
    }

    #[test]
    fn output_without_main_sequence_end_is_verification() {
        let fixture = Fixture::new();
        fixture.write_output(&Ok(1));
        fixture.write_commit(b"commitment-bytes");
        fixture.write_trace(&[
            r#"{"SequenceStart":{"fn_name":"main","input":null,"output":null,"draft_transition_witness":null}}"#,
            r#"{"TileExec":{"fn_name":"some_tile","input":null,"output":null,"draft_transition_witness":null}}"#,
        ]);

        let error = ingest::<u64>(
            &fixture.run_dir,
            &fixture.run_output("", ""),
            StagedInputCommitments::new(),
        )
        .expect_err("inconsistent artifacts must fail");
        match error {
            RasterCoreError::Verification { detail } => {
                assert!(detail.contains("SequenceEnd"), "detail: {detail}")
            }
            other => panic!("expected Verification, got {other}"),
        }
    }
}
