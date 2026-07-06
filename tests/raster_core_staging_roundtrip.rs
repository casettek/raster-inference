//! WS2 staging round trip: stage → `cargo raster run` → ingest, end to end
//! against the `raster-program-roundtrip` fixture crate, with one leg per
//! error class of the H4 contract (see `runtime::raster_core::ingest`).
//!
//! Requires the `cargo-raster` CLI (built from the pinned `raster` checkout)
//! on PATH — the same prerequisite as the WS1 probe runs. Without it the
//! test skips loudly; CI sets `REQUIRE_CARGO_RASTER=1` so the skip can never
//! happen silently there. Re-run recipe: docs/plans/ws2-staging.md.

use std::path::{Path, PathBuf};
use std::process::Command;

use raster_inference::runtime::checkpoints::RoutineId;
use raster_inference::runtime::raster_core::ingest::{ingest, RasterCoreError};
use raster_inference::runtime::raster_core::staging::StagedInputs;
use raster_inference::runtime::raster_core::{CargoRasterRunner, RasterCoreRunDir};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn program_crate_dir() -> PathBuf {
    workspace_root().join("crates/raster-programs/_roundtrip")
}

/// The fixture program is not a routine; run directories just need a name.
const FIXTURE_ROUTINE: RoutineId = RoutineId::PromptPrepare;

fn cargo_raster_available() -> bool {
    Command::new("cargo-raster")
        .arg("--version")
        .output()
        .is_ok()
}

/// Returns `false` (skip) when `cargo-raster` is unavailable and the run is
/// not required to have it; panics when `REQUIRE_CARGO_RASTER=1`.
fn require_or_skip() -> bool {
    if cargo_raster_available() {
        return true;
    }
    if std::env::var_os("REQUIRE_CARGO_RASTER").is_some_and(|v| v == "1") {
        panic!(
            "REQUIRE_CARGO_RASTER=1 but cargo-raster is not on PATH; install it from the \
             pinned raster checkout (cargo install --path ../raster/crates/raster-cli)"
        );
    }
    eprintln!(
        "SKIPPED: raster_core_staging_roundtrip requires the cargo-raster CLI on PATH \
         (cargo install --path ../raster/crates/raster-cli from the pinned checkout). \
         CI runs this test with REQUIRE_CARGO_RASTER=1."
    );
    false
}

/// Runs the fixture's encoder bin, returning the raster index root
/// commitment it prints (`root_commitment: <hex>`).
fn encode_fixture_config(out_dir: &Path, scale: u64) -> String {
    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "-p",
            "raster-program-roundtrip",
            "--features",
            "encode",
            "--bin",
            "encode",
            "--",
        ])
        .arg(out_dir)
        .arg(scale.to_string())
        .current_dir(workspace_root())
        .output()
        .expect("encoder bin should launch");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "encoder bin failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("root_commitment: "))
        .unwrap_or_else(|| panic!("encoder printed no root_commitment line\nstdout:\n{stdout}"))
        .trim()
        .to_string()
}

struct StagedRun {
    run_dir: RasterCoreRunDir,
    commitments: raster_inference::runtime::raster_core::staging::StagedInputCommitments,
}

/// Stages the fixture's three committed inputs (values, divisor, config).
fn stage_fixture_inputs(cache_dir: &Path, config_commitment: &str, divisor: u64) -> StagedRun {
    let run_dir = RasterCoreRunDir::create(FIXTURE_ROUTINE, 1).expect("run dir");
    let mut staged = StagedInputs::new();
    staged
        .add_postcard("values", &vec![1u64, 2, 3, 4])
        .expect("stage values");
    staged
        .add_postcard("divisor", &divisor)
        .expect("stage divisor");
    staged
        .add_raster_encoded(
            "config",
            &cache_dir.join("config.rastered"),
            &cache_dir.join("config.rindex"),
            config_commitment,
        )
        .expect("stage config");
    let commitments = staged.write(&run_dir).expect("write staged inputs");
    StagedRun {
        run_dir,
        commitments,
    }
}

/// One test fn drives all cargo-raster legs sequentially: concurrent
/// `cargo raster run` invocations would contend on the workspace build lock.
#[test]
fn staged_inputs_run_and_ingest_across_all_error_classes() {
    if !require_or_skip() {
        return;
    }
    let cache_dir =
        std::env::temp_dir().join(format!("ws2-roundtrip-cache-{}", std::process::id()));
    std::fs::create_dir_all(&cache_dir).expect("cache dir");
    let runner = CargoRasterRunner::default();
    let program_dir = program_crate_dir();

    // The host and program halves of the output-file convention must agree.
    assert_eq!(
        raster_inference::runtime::raster_core::OUTPUT_PATH_ENV,
        raster_program_support::OUTPUT_PATH_ENV,
    );

    // Encoder determinism: same value, same root commitment (WS2 encoder
    // rule; the cached files are reused by every leg below).
    let commitment = encode_fixture_config(&cache_dir, 3);
    let second_dir = cache_dir.join("second");
    let commitment_again = encode_fixture_config(&second_dir, 3);
    assert_eq!(
        commitment, commitment_again,
        "raster encoding must be deterministic"
    );

    // Leg 1 — happy path: sum(1..=4) * 3 / 2 = 15.
    let staged = stage_fixture_inputs(&cache_dir, &commitment, 2);
    let run_output = runner
        .run(&program_dir, &staged.run_dir)
        .expect("cargo raster run should complete");
    let result = ingest::<u64>(&staged.run_dir, &run_output, staged.commitments.clone())
        .unwrap_or_else(|error| {
            panic!(
                "happy-path ingest failed: {error}\nstdout:\n{}\nstderr:\n{}",
                run_output.stdout, run_output.stderr
            )
        });
    assert_eq!(result.value, 15);
    assert_eq!(result.input_commitments.len(), 3);
    assert_eq!(
        result.input_commitments["config"].commitment, commitment,
        "raster-encoded input commitment must be captured in the run result"
    );
    assert_eq!(result.input_commitments["values"].encoding, "postcard");
    assert!(!result.commit_artifact_sha256.is_empty());
    assert!(result.trace_path.is_file(), "trace file must exist");
    std::fs::remove_dir_all(&result.run_dir_root).ok();

    // Leg 2 — terminal outcome: zero divisor commits Err through the tile.
    let staged = stage_fixture_inputs(&cache_dir, &commitment, 0);
    let run_output = runner
        .run(&program_dir, &staged.run_dir)
        .expect("cargo raster run should complete (guest outcome is terminal, not broken)");
    let error = ingest::<u64>(&staged.run_dir, &run_output, staged.commitments.clone())
        .expect_err("zero divisor must classify as an error");
    match &error {
        RasterCoreError::Terminal { message } => {
            assert!(
                message.contains("divisor is zero"),
                "unexpected terminal message: {message}"
            );
        }
        other => panic!(
            "expected Terminal, got {other}\nstdout:\n{}\nstderr:\n{}",
            run_output.stdout, run_output.stderr
        ),
    }
    std::fs::remove_dir_all(staged.run_dir.root()).ok();

    // Leg 3 — verification: tamper a committed input after the manifest is
    // written; the runtime must reject it at resolve time.
    let staged = stage_fixture_inputs(&cache_dir, &commitment, 2);
    let values_path = staged.run_dir.root().join("values.bin");
    let mut bytes = std::fs::read(&values_path).expect("staged values payload");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&values_path, bytes).expect("tamper staged values payload");
    // The CLI's exit status is meaningless here in both directions (ADR gap
    // G4 and the trace-commitment panic on short traces); classification is
    // ingestion's job.
    let run_output = runner
        .run(&program_dir, &staged.run_dir)
        .expect("runner launch should succeed regardless of guest outcome");
    let error = ingest::<u64>(&staged.run_dir, &run_output, staged.commitments.clone())
        .expect_err("tampered input must classify as an error");
    match &error {
        RasterCoreError::Verification { detail } => {
            assert!(
                detail.contains("integrity"),
                "unexpected verification detail: {detail}"
            );
        }
        other => panic!(
            "expected Verification, got {other}\nstdout:\n{}\nstderr:\n{}",
            run_output.stdout, run_output.stderr
        ),
    }
    std::fs::remove_dir_all(staged.run_dir.root()).ok();

    std::fs::remove_dir_all(&cache_dir).ok();
}

/// Infrastructure classification needs no toolchain: a missing runner binary
/// is a broken run with no committed outcome.
#[test]
fn missing_toolchain_classifies_as_infrastructure() {
    let run_dir = RasterCoreRunDir::create(FIXTURE_ROUTINE, 1).expect("run dir");
    StagedInputs::new().write(&run_dir).expect("empty staging");
    let runner = CargoRasterRunner::with_program("cargo-raster-definitely-not-installed");
    let error = RasterCoreError::Infrastructure(
        runner
            .run(&program_crate_dir(), &run_dir)
            .expect_err("missing cargo-raster must fail"),
    );
    assert!(
        matches!(error, RasterCoreError::Infrastructure(_)),
        "host adapters map runner launch failures into the Infrastructure class"
    );
    assert!(error.to_string().contains("cargo-raster was not found"));
    std::fs::remove_dir_all(run_dir.root()).ok();
}
