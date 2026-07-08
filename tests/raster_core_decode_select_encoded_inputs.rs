//! `decode.select_token` run-local raster externals: encode → stage → run →
//! ingest, plus committed-input tamper legs.

use std::path::{Path, PathBuf};
use std::process::Command;

use raster_inference::runtime::checkpoints::RoutineId;
use raster_inference::runtime::raster_core::ingest::{ingest, RasterCoreError};
use raster_inference::runtime::raster_core::staging::StagedInputs;
use raster_inference::runtime::raster_core::{CargoRasterRunner, RasterCoreRunDir};

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct DecodeSelectLogitsMirror {
    row_count: u32,
    width: u32,
    bits: Vec<i32>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct DecodeSelectTokenIdsMirror {
    token_count: u32,
    token_ids: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct DecodeSelectEncodedInputsMirror {
    logits: DecodeSelectLogitsMirror,
    full_token_ids: DecodeSelectTokenIdsMirror,
    generated_token_ids: DecodeSelectTokenIdsMirror,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct DecodeSelectConfigMirror {
    logits_per_tile: u32,
    token_ids_per_tile: u32,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct DecodeSelectLoopDriversMirror {
    logit_ordinals: Vec<u32>,
    full_token_ordinals: Vec<u32>,
    generated_token_ordinals: Vec<u32>,
}

#[derive(Debug, serde::Deserialize, PartialEq, Eq)]
struct DecodeSelectOutputMirror {
    next_token: u32,
    full_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    logit_count: u32,
}

#[derive(Debug)]
struct EncodedInput {
    data_path: PathBuf,
    index_path: PathBuf,
    root_commitment: String,
}

#[derive(Debug)]
struct EncodedInputs {
    logits: EncodedInput,
    full_token_ids: EncodedInput,
    generated_token_ids: EncodedInput,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn program_crate_dir() -> PathBuf {
    workspace_root().join("crates/raster-programs/decode_select_token")
}

fn cargo_raster_available() -> bool {
    Command::new("cargo-raster")
        .arg("--version")
        .output()
        .is_ok()
}

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
        "SKIPPED: raster_core_decode_select_encoded_inputs requires the cargo-raster CLI on PATH. \
         CI runs this test with REQUIRE_CARGO_RASTER=1."
    );
    false
}

fn fixture_inputs() -> DecodeSelectEncodedInputsMirror {
    DecodeSelectEncodedInputsMirror {
        logits: DecodeSelectLogitsMirror {
            row_count: 4,
            width: 1,
            bits: vec![1, 4, 9, 3],
        },
        full_token_ids: DecodeSelectTokenIdsMirror {
            token_count: 1,
            token_ids: vec![7],
        },
        generated_token_ids: DecodeSelectTokenIdsMirror {
            token_count: 0,
            token_ids: vec![],
        },
    }
}

fn encode_inputs(temp_root: &Path, inputs: &DecodeSelectEncodedInputsMirror) -> EncodedInputs {
    let source_path = temp_root.join("decode_select_source.json");
    let out_dir = temp_root.join("encoded");
    std::fs::create_dir_all(temp_root).expect("temp root");
    std::fs::write(
        &source_path,
        serde_json::to_vec(inputs).expect("serialize source"),
    )
    .expect("write source");
    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "-p",
            "raster-program-decode-select-token",
            "--features",
            "encode",
            "--bin",
            "encode",
            "--",
        ])
        .arg(&source_path)
        .arg(&out_dir)
        .current_dir(workspace_root())
        .output()
        .expect("encoder bin should launch");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "encoder bin failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    EncodedInputs {
        logits: parse_encoded_input("logits", &stdout),
        full_token_ids: parse_encoded_input("full_token_ids", &stdout),
        generated_token_ids: parse_encoded_input("generated_token_ids", &stdout),
    }
}

fn parse_encoded_input(name: &str, stdout: &str) -> EncodedInput {
    let field = |suffix: &str| {
        let prefix = format!("{name}_{suffix}: ");
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("encoder printed no '{prefix}' line\nstdout:\n{stdout}"))
            .trim()
            .to_string()
    };
    EncodedInput {
        data_path: PathBuf::from(field("data_path")),
        index_path: PathBuf::from(field("index_path")),
        root_commitment: field("root_commitment"),
    }
}

fn stage_inputs(
    encoded: &EncodedInputs,
    logits_commitment: &str,
    full_token_index_path: &Path,
) -> RasterCoreRunDir {
    let run_dir = RasterCoreRunDir::create(RoutineId::SelectOutputToken, 1).expect("run dir");
    let mut staged = StagedInputs::new();
    staged
        .add_raster_encoded(
            "logits",
            &encoded.logits.data_path,
            &encoded.logits.index_path,
            logits_commitment,
        )
        .expect("stage logits");
    staged
        .add_raster_encoded(
            "full_token_ids",
            &encoded.full_token_ids.data_path,
            full_token_index_path,
            &encoded.full_token_ids.root_commitment,
        )
        .expect("stage full tokens");
    staged
        .add_raster_encoded(
            "generated_token_ids",
            &encoded.generated_token_ids.data_path,
            &encoded.generated_token_ids.index_path,
            &encoded.generated_token_ids.root_commitment,
        )
        .expect("stage generated tokens");
    staged
        .add_postcard(
            "loop_drivers",
            &DecodeSelectLoopDriversMirror {
                logit_ordinals: vec![0, 1],
                full_token_ordinals: vec![0],
                generated_token_ordinals: vec![],
            },
        )
        .expect("stage drivers");
    staged
        .add_postcard(
            "config",
            &DecodeSelectConfigMirror {
                logits_per_tile: 2,
                token_ids_per_tile: 2,
            },
        )
        .expect("stage config");
    staged.write(&run_dir).expect("write staged inputs");
    run_dir
}

#[test]
fn encoded_decode_select_inputs_run_and_classify_tampering() {
    if !require_or_skip() {
        return;
    }
    let temp_root = std::env::temp_dir().join(format!(
        "decode-select-encoded-inputs-{}",
        std::process::id()
    ));
    let inputs = fixture_inputs();
    let encoded = encode_inputs(&temp_root, &inputs);
    let encoded_again = encode_inputs(&temp_root.join("again"), &inputs);
    assert_eq!(
        encoded.logits.root_commitment, encoded_again.logits.root_commitment,
        "raster encoding of identical logits must be deterministic"
    );

    let runner = CargoRasterRunner::default();
    let program_dir = program_crate_dir();

    let run_dir = stage_inputs(
        &encoded,
        &encoded.logits.root_commitment,
        &encoded.full_token_ids.index_path,
    );
    let run_output = runner
        .run(&program_dir, &run_dir)
        .expect("cargo raster run should complete");
    let result = ingest::<DecodeSelectOutputMirror>(&run_dir, &run_output, Default::default())
        .unwrap_or_else(|error| {
            panic!(
                "encoded decode-select ingest failed: {error}\nstdout:\n{}\nstderr:\n{}",
                run_output.stdout, run_output.stderr
            )
        });
    assert_eq!(
        result.value,
        DecodeSelectOutputMirror {
            next_token: 2,
            full_token_ids: vec![7, 2],
            generated_token_ids: vec![2],
            logit_count: 4,
        }
    );
    std::fs::remove_dir_all(run_dir.root()).ok();

    let mut bad_commitment = encoded.logits.root_commitment.clone();
    let last_char = bad_commitment.pop().expect("commitment is non-empty");
    bad_commitment.push(if last_char == '0' { '1' } else { '0' });
    let run_dir = stage_inputs(
        &encoded,
        &bad_commitment,
        &encoded.full_token_ids.index_path,
    );
    let run_output = runner
        .run(&program_dir, &run_dir)
        .expect("runner launch should succeed regardless of guest outcome");
    let error = ingest::<DecodeSelectOutputMirror>(&run_dir, &run_output, Default::default())
        .expect_err("commitment mismatch must classify as an error");
    assert!(
        matches!(error, RasterCoreError::Verification { .. }),
        "expected Verification for commitment mismatch, got {error}"
    );
    std::fs::remove_dir_all(run_dir.root()).ok();

    let tampered_dir = temp_root.join("tampered");
    std::fs::create_dir_all(&tampered_dir).expect("tampered dir");
    let tampered_index = tampered_dir.join("full_token_ids.rindex");
    let mut index_bytes =
        std::fs::read(&encoded.full_token_ids.index_path).expect("read full token index");
    let last = index_bytes.len() - 1;
    index_bytes[last] ^= 0xff;
    std::fs::write(&tampered_index, index_bytes).expect("write tampered index");
    let run_dir = stage_inputs(&encoded, &encoded.logits.root_commitment, &tampered_index);
    let run_output = runner
        .run(&program_dir, &run_dir)
        .expect("runner launch should succeed regardless of guest outcome");
    let error = ingest::<DecodeSelectOutputMirror>(&run_dir, &run_output, Default::default())
        .expect_err("corrupt rindex must classify as an error");
    assert!(
        matches!(error, RasterCoreError::Infrastructure(_)),
        "expected Infrastructure for corrupt rindex, got {error}"
    );
    std::fs::remove_dir_all(run_dir.root()).ok();
    std::fs::remove_dir_all(&temp_root).ok();
}
