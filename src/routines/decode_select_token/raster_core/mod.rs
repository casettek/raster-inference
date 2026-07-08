//! Raster-core (real toolchain) host adapter for `decode.select_token`.
//!
//! Stage → run → ingest → materialize, per the WS2 conventions. The program
//! crate receives raster-encoded hot inputs (canonical deterministic logits
//! and current token histories) plus postcard control inputs (chunk ordinals
//! and chunk sizing). The selected token and updated histories come from the
//! ingested program output only.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::runtime::checkpoints::RoutineId;
use crate::runtime::raster_core::ingest::{ingest, RasterCoreError};
use crate::runtime::raster_core::staging::StagedInputs;
use crate::runtime::raster_core::{CargoRasterRunner, RasterCoreRunDir, COMMIT_FILE_NAME};
use crate::shared::api::output::DecodeState;
use crate::RasterSizingControls;

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedDecodeSelectLogits {
    row_count: u32,
    width: u32,
    bits: Vec<i32>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedDecodeSelectTokenIds {
    token_count: u32,
    token_ids: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedDecodeSelectEncodedInputs {
    logits: StagedDecodeSelectLogits,
    full_token_ids: StagedDecodeSelectTokenIds,
    generated_token_ids: StagedDecodeSelectTokenIds,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedDecodeSelectConfig {
    logits_per_tile: u32,
    token_ids_per_tile: u32,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedDecodeSelectLoopDrivers {
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

/// Runs one selected `decode.select_token` occurrence on the real raster
/// toolchain and mutates the native decode state from the committed output.
pub fn run_raster_core(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
    raster_sizing: RasterSizingControls,
) -> Result<u32> {
    if super::native::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens)
        .is_some()
    {
        bail!("selective raster decode.select_token detour reached stop condition unexpectedly");
    }

    let occurrence = decode_state.generated_token_ids.len() + 1;
    let internal_logits = decode_state.clone_internal_logits();
    let det_logits = internal_logits.det_values().with_context(|| {
        "raster-core decode.select_token detour requires canonical deterministic logits"
    })?;
    let logit_count = usize_to_u32(det_logits.len(), "decode select logit count")?;
    let logits = StagedDecodeSelectLogits {
        row_count: logit_count,
        width: 1,
        bits: det_logits.iter().map(|value| value.to_bits()).collect(),
    };
    let full_token_ids = StagedDecodeSelectTokenIds {
        token_count: usize_to_u32(
            decode_state.full_token_ids.len(),
            "decode select full token count",
        )?,
        token_ids: decode_state.full_token_ids.clone(),
    };
    let generated_token_ids = StagedDecodeSelectTokenIds {
        token_count: usize_to_u32(
            decode_state.generated_token_ids.len(),
            "decode select generated token count",
        )?,
        token_ids: decode_state.generated_token_ids.clone(),
    };
    let chunk_width = usize_to_u32(
        raster_sizing.sequence_rows_per_tile,
        "decode select chunk width",
    )?;
    if chunk_width == 0 {
        bail!("raster-core decode.select_token requires a nonzero sequence_rows_per_tile");
    }
    let config = StagedDecodeSelectConfig {
        logits_per_tile: chunk_width,
        token_ids_per_tile: chunk_width,
    };
    // One recur ordinal per chunk of `chunk_width` elements: the program
    // reads chunk elements from committed storage inside the tile body, so
    // driver size and trace size scale with chunk count, not element count.
    let loop_drivers = StagedDecodeSelectLoopDrivers {
        logit_ordinals: chunk_ordinals(logit_count, chunk_width),
        full_token_ordinals: chunk_ordinals(full_token_ids.token_count, chunk_width),
        generated_token_ordinals: chunk_ordinals(generated_token_ids.token_count, chunk_width),
    };

    let run_dir = RasterCoreRunDir::create(RoutineId::SelectOutputToken, occurrence)?;
    let encoded_inputs =
        encode_decode_select_inputs(&run_dir, &logits, &full_token_ids, &generated_token_ids)
            .map_err(RasterCoreError::Infrastructure)
            .map_err(anyhow::Error::new)?;
    let mut staged = StagedInputs::new();
    staged.add_raster_encoded(
        "logits",
        &encoded_inputs.logits.data_path,
        &encoded_inputs.logits.index_path,
        &encoded_inputs.logits.root_commitment,
    )?;
    staged.add_raster_encoded(
        "full_token_ids",
        &encoded_inputs.full_token_ids.data_path,
        &encoded_inputs.full_token_ids.index_path,
        &encoded_inputs.full_token_ids.root_commitment,
    )?;
    staged.add_raster_encoded(
        "generated_token_ids",
        &encoded_inputs.generated_token_ids.data_path,
        &encoded_inputs.generated_token_ids.index_path,
        &encoded_inputs.generated_token_ids.root_commitment,
    )?;
    staged.add_postcard("loop_drivers", &loop_drivers)?;
    staged.add_postcard("config", &config)?;
    let input_commitments = staged
        .write(&run_dir)
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;

    let run_output = CargoRasterRunner::default()
        .run(&program_crate_dir(), &run_dir)
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;
    let run_result = ingest::<DecodeSelectOutputMirror>(&run_dir, &run_output, input_commitments)
        .map_err(anyhow::Error::new)?;

    let output = run_result.value;
    validate_output(&output, decode_state, logit_count, &run_result.run_dir_root)?;

    decode_state.full_token_ids = output.full_token_ids;
    decode_state.generated_token_ids = output.generated_token_ids;
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &super::decode_select_checkpoint_state(decode_state, output.next_token, max_new_tokens)?,
    );

    if let Some(artifacts_dir) = run_result.trace_path.parent() {
        std::fs::copy(
            &run_result.commit_path,
            artifacts_dir.join(COMMIT_FILE_NAME),
        )
        .ok();
    }
    std::fs::remove_dir_all(run_dir.root()).ok();

    Ok(output.next_token)
}

fn validate_output(
    output: &DecodeSelectOutputMirror,
    decode_state: &DecodeState,
    expected_logit_count: u32,
    run_dir_root: &std::path::Path,
) -> Result<()> {
    if output.logit_count != expected_logit_count {
        bail!(
            "raster-core decode.select_token reported {} logits, expected {} (run dir kept at {})",
            output.logit_count,
            expected_logit_count,
            run_dir_root.display()
        );
    }
    if output.full_token_ids.len() != decode_state.full_token_ids.len() + 1 {
        bail!(
            "raster-core decode.select_token full-token output has {} ids, expected {} (run dir kept at {})",
            output.full_token_ids.len(),
            decode_state.full_token_ids.len() + 1,
            run_dir_root.display()
        );
    }
    if output.generated_token_ids.len() != decode_state.generated_token_ids.len() + 1 {
        bail!(
            "raster-core decode.select_token generated-token output has {} ids, expected {} (run dir kept at {})",
            output.generated_token_ids.len(),
            decode_state.generated_token_ids.len() + 1,
            run_dir_root.display()
        );
    }
    if !output
        .full_token_ids
        .starts_with(&decode_state.full_token_ids)
        || output.full_token_ids.last().copied() != Some(output.next_token)
    {
        bail!(
            "raster-core decode.select_token full-token output did not append the selected token (run dir kept at {})",
            run_dir_root.display()
        );
    }
    if !output
        .generated_token_ids
        .starts_with(&decode_state.generated_token_ids)
        || output.generated_token_ids.last().copied() != Some(output.next_token)
    {
        bail!(
            "raster-core decode.select_token generated-token output did not append the selected token (run dir kept at {})",
            run_dir_root.display()
        );
    }
    Ok(())
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{label} exceeds u32"))
}

fn chunk_ordinals(item_count: u32, per_tile: u32) -> Vec<u32> {
    (0..item_count.div_ceil(per_tile)).collect()
}

#[derive(Debug)]
struct EncodedDecodeSelectInputs {
    logits: EncodedDecodeSelectInput,
    full_token_ids: EncodedDecodeSelectInput,
    generated_token_ids: EncodedDecodeSelectInput,
}

#[derive(Debug)]
struct EncodedDecodeSelectInput {
    data_path: PathBuf,
    index_path: PathBuf,
    root_commitment: String,
}

fn encode_decode_select_inputs(
    run_dir: &RasterCoreRunDir,
    logits: &StagedDecodeSelectLogits,
    full_token_ids: &StagedDecodeSelectTokenIds,
    generated_token_ids: &StagedDecodeSelectTokenIds,
) -> Result<EncodedDecodeSelectInputs> {
    let source = StagedDecodeSelectEncodedInputs {
        logits: logits.clone(),
        full_token_ids: full_token_ids.clone(),
        generated_token_ids: generated_token_ids.clone(),
    };
    let source_path = run_dir.root().join("decode_select_encoded_inputs.json");
    let encoded_dir = run_dir.root().join("encoded_inputs");
    let source_bytes = serde_json::to_vec(&source)
        .context("failed to serialize decode.select_token encoded input source")?;
    std::fs::write(&source_path, source_bytes)
        .with_context(|| format!("failed to write {}", source_path.display()))?;

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
        .arg(&encoded_dir)
        .current_dir(workspace_root())
        .output()
        .context("failed to launch the decode_select_token input encoder")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        bail!(
            "decode_select_token input encoder failed\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(EncodedDecodeSelectInputs {
        logits: parse_encoded_input("logits", &stdout)?,
        full_token_ids: parse_encoded_input("full_token_ids", &stdout)?,
        generated_token_ids: parse_encoded_input("generated_token_ids", &stdout)?,
    })
}

fn parse_encoded_input(name: &str, stdout: &str) -> Result<EncodedDecodeSelectInput> {
    let field = |suffix: &str| -> Result<String> {
        let prefix = format!("{name}_{suffix}: ");
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .map(|rest| rest.trim().to_string())
            .with_context(|| format!("decode_select_token encoder printed no '{prefix}' line"))
    };
    Ok(EncodedDecodeSelectInput {
        data_path: PathBuf::from(field("data_path")?),
        index_path: PathBuf::from(field("index_path")?),
        root_commitment: field("root_commitment")?,
    })
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn program_crate_dir() -> PathBuf {
    workspace_root().join("crates/raster-programs/decode_select_token")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_output_rejects_inconsistent_lengths() {
        let decode_state = DecodeState::new(
            vec![7],
            vec![0.0],
            crate::shared::model::transformer::TransformerDecodeState::default(),
        );
        let output = DecodeSelectOutputMirror {
            next_token: 1,
            full_token_ids: vec![7],
            generated_token_ids: vec![1],
            logit_count: 1,
        };

        let error = validate_output(&output, &decode_state, 1, std::path::Path::new("/tmp/run"))
            .expect_err("bad full length should fail");
        assert!(error.to_string().contains("full-token output"));
    }

    #[test]
    fn encoder_output_parser_requires_all_fields() {
        let stdout = "\
logits_data_path: /tmp/logits.rastered
logits_index_path: /tmp/logits.rindex
logits_root_commitment: abc123
";
        let parsed = parse_encoded_input("logits", stdout).expect("parse logits");
        assert_eq!(parsed.data_path, PathBuf::from("/tmp/logits.rastered"));
        assert_eq!(parsed.index_path, PathBuf::from("/tmp/logits.rindex"));
        assert_eq!(parsed.root_commitment, "abc123");

        let error = parse_encoded_input("full_token_ids", stdout).expect_err("missing fields");
        assert!(
            error.to_string().contains("full_token_ids_data_path"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn hot_inputs_stage_as_raster_encoded_externals() {
        let run_dir = RasterCoreRunDir::create(RoutineId::SelectOutputToken, 999).expect("run dir");
        let logits = StagedDecodeSelectLogits {
            row_count: 3,
            width: 1,
            bits: vec![1, 9, 3],
        };
        let full_token_ids = StagedDecodeSelectTokenIds {
            token_count: 1,
            token_ids: vec![7],
        };
        let generated_token_ids = StagedDecodeSelectTokenIds {
            token_count: 0,
            token_ids: vec![],
        };
        let encoded =
            encode_decode_select_inputs(&run_dir, &logits, &full_token_ids, &generated_token_ids)
                .expect("encode inputs");

        let mut staged = StagedInputs::new();
        staged
            .add_raster_encoded(
                "logits",
                &encoded.logits.data_path,
                &encoded.logits.index_path,
                &encoded.logits.root_commitment,
            )
            .expect("stage logits");
        staged
            .add_raster_encoded(
                "full_token_ids",
                &encoded.full_token_ids.data_path,
                &encoded.full_token_ids.index_path,
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
                &StagedDecodeSelectLoopDrivers {
                    logit_ordinals: vec![0],
                    full_token_ordinals: vec![0],
                    generated_token_ordinals: vec![],
                },
            )
            .expect("stage drivers");
        staged
            .add_postcard(
                "config",
                &StagedDecodeSelectConfig {
                    logits_per_tile: 3,
                    token_ids_per_tile: 3,
                },
            )
            .expect("stage config");
        let commitments = staged.write(&run_dir).expect("write staged inputs");

        assert_eq!(commitments["logits"].encoding, "raster");
        assert_eq!(commitments["full_token_ids"].encoding, "raster");
        assert_eq!(commitments["generated_token_ids"].encoding, "raster");
        assert_eq!(commitments["loop_drivers"].encoding, "postcard");
        assert_eq!(commitments["config"].encoding, "postcard");

        let input: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_dir.input_path()).expect("input.json"))
                .expect("parse input.json");
        for name in ["logits", "full_token_ids", "generated_token_ids"] {
            assert_eq!(input[name]["load_preference"], "mmap");
            assert!(
                input[name]["index_path"].as_str().is_some(),
                "{name} should carry a raster index path"
            );
        }
        for name in ["loop_drivers", "config"] {
            assert_eq!(input[name]["load_preference"], "read");
            assert!(
                input[name]["index_path"].is_null(),
                "{name} should remain a postcard control input"
            );
        }
        std::fs::remove_dir_all(run_dir.root()).ok();
    }
}
