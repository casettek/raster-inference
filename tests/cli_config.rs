//! CLI integration test: execution tuning supplied via `--config` takes
//! effect through the real binary.
//!
//! Ported from the sizing-sensitive
//! `run_inference_prefill_finalize_detour_uses_projection_tile_sizing` test
//! in `inference_sequence.rs`: the same `prefill.finalize` detour with
//! single-row vs multi-row projection chunks, driven through `detour
//! --config` instead of `InferenceControls`. Smaller projection chunks must
//! invoke more raster tiles while the decoded output stays identical.

use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use serde_json::Value;

fn run_detour_with_config(config_text: &str, tag: &str) -> Value {
    let temp_dir = env::temp_dir().join(format!("raster-cli-config-{}-{tag}", std::process::id()));
    fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");
    let config_path = temp_dir.join("tuning.toml");
    fs::write(&config_path, config_text).expect("config file should be writable");
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/tiny-gemma-dev");

    let output = Command::new(env!("CARGO_BIN_EXE_raster-inference"))
        .arg("detour")
        .arg("--model")
        .arg(&model_dir)
        .args(["--at", "prefill.finalize", "--max-new-tokens", "2"])
        .arg("--config")
        .arg(&config_path)
        .arg("--trace-dir")
        .arg(temp_dir.join("traces"))
        .args(["hello", "world"])
        .output()
        .expect("CLI binary should run");
    assert!(
        output.status.success(),
        "detour run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = serde_json::from_slice(&output.stdout).expect("stdout should be a JSON document");
    fs::remove_dir_all(&temp_dir).ok();
    result
}

#[test]
fn config_tile_sizing_takes_effect() {
    let single_row_chunks = run_detour_with_config(
        "[tile_sizing]\nprojection_rows_per_tile = 1\n",
        "single-row",
    );
    let multi_row_chunks = run_detour_with_config(
        "[tile_sizing]\nprojection_rows_per_tile = 2\n",
        "multi-row",
    );

    let tile_invocations = |result: &Value| {
        result["state"]["raster_tile_invocations"]
            .as_u64()
            .expect("detour state should report raster tile invocations")
    };
    assert!(
        tile_invocations(&single_row_chunks) > tile_invocations(&multi_row_chunks),
        "smaller projection chunks should invoke more raster tiles \
         (single-row {} vs multi-row {})",
        tile_invocations(&single_row_chunks),
        tile_invocations(&multi_row_chunks)
    );
    assert_eq!(
        single_row_chunks["state"]["output_decode"], multi_row_chunks["state"]["output_decode"],
        "tile sizing must not change the decoded output"
    );
}
