//! Raster-core (real toolchain) host adapter for `input.embedding`.
//!
//! Stage → run → ingest → materialize, per the WS2 conventions. This adapter
//! is intentionally a bridge from upstream committed native state to the
//! storage shape a sequential raster run would have: prompt token ids are
//! verified against `PromptPreparationState.prompt_token_ids_sha256`, then
//! staged as a run-local raster-encoded input before the program executes.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::routines::input_embedding::raster::auth_source::GemmaInputEmbeddingRowRequest;
use crate::runtime::checkpoints::RoutineId;
use crate::runtime::raster_core::externals::{
    lookup_external_entry, missing_external_error, resolve_run_externals_dir, EncodedExternal,
};
use crate::runtime::raster_core::ingest::{ingest, RasterCoreError};
use crate::runtime::raster_core::staging::StagedInputs;
use crate::runtime::raster_core::{CargoRasterRunner, RasterCoreRunDir, COMMIT_FILE_NAME};
use crate::shared::api::input::PromptPreparationState;
use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::model::runtime::LoadedModel;
use crate::shared::model::transformer::{ActivationSequence, InternalActivationSequence};
use crate::shared::numerics::det_num::Act;
use crate::shared::numerics::transformer_kernels::build_det_activation_commitment;
use crate::trace::routine_scope;
use crate::RasterSizingControls;

use super::raster::auth_source::AuthenticatedDecoderEmbeddingSource;

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedInputEmbeddingPromptTokenIds {
    token_count: u32,
    token_ids: Vec<u32>,
    token_ids_sha256: String,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedGemmaInputEmbeddingMetadata {
    source_id: String,
    vocab_size: u32,
    hidden_size: u32,
    scale_bits: i32,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedGemmaInputEmbeddingTable {
    metadata: StagedGemmaInputEmbeddingMetadata,
    /// Hex-packed rows: one `String` leaf per row (mirror of the program
    /// crate's `pack_embedding_row_hex` — 8 lowercase hex chars per
    /// canonical Act bit pattern). One leaf per row keeps the raster index
    /// O(vocab) nodes; per-value leaves are unencodable at real model scale.
    rows: Vec<String>,
}

/// Byte-identical mirror of the program crate's `pack_embedding_row_hex`
/// (the main crate must not depend on program crates that carry the real
/// `raster` dependency; parity is enforced end to end by the tiny-gemma
/// detour trace-identity tests).
fn pack_embedding_row_hex(values: &[i32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(values.len() * 8);
    for value in values {
        let bits = *value as u32;
        for shift in (0..8).rev() {
            out.push(HEX[((bits >> (shift * 4)) & 0xf) as usize]);
        }
    }
    String::from_utf8(out).expect("hex packing is always ascii")
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedInputEmbeddingEncodedInputs {
    prompt_token_ids: Option<StagedInputEmbeddingPromptTokenIds>,
    embedding: Option<StagedGemmaInputEmbeddingTable>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedInputEmbeddingLoopDrivers {
    token_ordinals: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedInputEmbeddingConfig {
    tokens_per_tile: u32,
    prompt_token_ids_sha256: String,
    prompt_token_ids_root: String,
    embedding_source_root: String,
}

#[derive(Debug, serde::Deserialize, PartialEq, Eq)]
struct InputEmbeddingOutputMirror {
    source_id: String,
    prompt_token_ids_sha256: String,
    prompt_token_ids_root: String,
    embedding_source_root: String,
    prompt_token_count: u32,
    hidden_size: u32,
    activation_rows: Vec<Vec<i32>>,
}

/// Runs `input.embedding` on the real raster toolchain and materializes the
/// native-form activation sequence consumed by downstream native routines and
/// checkpoint formatting.
pub fn run_raster_core(
    prompt_preparation: &PromptPreparationState,
    model: &LoadedModel,
    raster_sizing: RasterSizingControls,
) -> Result<ActivationSequence> {
    let _routine = routine_scope(RoutineId::InputEmbedding, "mode=raster_core");

    // Model externals are strict lookup-only at run time: the embedding
    // entry must have been pre-encoded via `encode-externals` into the
    // resolved directory. Fail before staging when the directory is unset
    // or the entry is missing.
    let externals_dir = resolve_run_externals_dir(RoutineId::InputEmbedding)
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;
    let prompt_token_ids = stageable_prompt_token_ids(prompt_preparation)?;
    let embedding_source = model.input_embedding_source()?;
    let embedding_external = lookup_embedding_external(&embedding_source, &externals_dir)
        .and_then(|entry| {
            entry.ok_or_else(|| {
                missing_external_error(
                    RoutineId::InputEmbedding,
                    EMBEDDING_CACHE_KIND,
                    &externals_dir,
                )
            })
        })
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;

    let tokens_per_tile = usize_to_u32(
        raster_sizing.sequence_rows_per_tile,
        "input embedding tokens per tile",
    )?;
    if tokens_per_tile == 0 {
        bail!("raster-core input.embedding requires a nonzero sequence_rows_per_tile");
    }

    let run_dir = RasterCoreRunDir::create(RoutineId::InputEmbedding, 1)?;
    let prompt_external = encode_prompt_token_ids(&run_dir, &prompt_token_ids)
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;
    let loop_drivers = StagedInputEmbeddingLoopDrivers {
        token_ordinals: chunk_ordinals(prompt_token_ids.token_count, tokens_per_tile),
    };
    let config = StagedInputEmbeddingConfig {
        tokens_per_tile,
        prompt_token_ids_sha256: prompt_token_ids.token_ids_sha256.clone(),
        prompt_token_ids_root: prompt_external.root_commitment.clone(),
        embedding_source_root: embedding_external.root_commitment.clone(),
    };

    let mut staged = StagedInputs::new();
    staged.add_raster_encoded(
        "prompt_token_ids",
        &prompt_external.data_path,
        &prompt_external.index_path,
        &prompt_external.root_commitment,
    )?;
    staged.add_raster_encoded(
        "embedding",
        &embedding_external.data_path,
        &embedding_external.index_path,
        &embedding_external.root_commitment,
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
    let run_result = ingest::<InputEmbeddingOutputMirror>(&run_dir, &run_output, input_commitments)
        .map_err(anyhow::Error::new)?;

    let output = run_result.value;
    validate_output(
        &output,
        &prompt_token_ids,
        &config,
        &run_result.run_dir_root,
    )?;
    let token_embeddings = materialize_output(output)?;

    if let Some(artifacts_dir) = run_result.trace_path.parent() {
        std::fs::copy(
            &run_result.commit_path,
            artifacts_dir.join(COMMIT_FILE_NAME),
        )
        .ok();
    }
    std::fs::remove_dir_all(run_dir.root()).ok();

    Ok(token_embeddings)
}

fn stageable_prompt_token_ids(
    prompt_preparation: &PromptPreparationState,
) -> Result<StagedInputEmbeddingPromptTokenIds> {
    let actual = crate::routines::prompt_prepare::native::build_prompt_commitment(
        &prompt_preparation.prompt_token_ids,
    )?;
    if actual != prompt_preparation.prompt_token_ids_sha256 {
        bail!(
            "raster-core input.embedding prompt token ids hash {actual} does not match upstream commitment {}",
            prompt_preparation.prompt_token_ids_sha256
        );
    }
    Ok(StagedInputEmbeddingPromptTokenIds {
        token_count: usize_to_u32(
            prompt_preparation.prompt_token_ids.len(),
            "input embedding prompt token count",
        )?,
        token_ids: prompt_preparation.prompt_token_ids.clone(),
        token_ids_sha256: prompt_preparation.prompt_token_ids_sha256.clone(),
    })
}

#[derive(Debug)]
struct EncodedInputEmbeddingInput {
    data_path: PathBuf,
    index_path: PathBuf,
    root_commitment: String,
}

fn encode_prompt_token_ids(
    run_dir: &RasterCoreRunDir,
    prompt_token_ids: &StagedInputEmbeddingPromptTokenIds,
) -> Result<EncodedInputEmbeddingInput> {
    let encoded_dir = run_dir.root().join("encoded_inputs");
    std::fs::create_dir_all(&encoded_dir)
        .with_context(|| format!("failed to create {}", encoded_dir.display()))?;
    let source_path = encoded_dir.join("prompt_token_ids_source.json");
    let source = StagedInputEmbeddingEncodedInputs {
        prompt_token_ids: Some(prompt_token_ids.clone()),
        embedding: None,
    };
    let source_bytes = serde_json::to_vec_pretty(&source)?;
    std::fs::write(&source_path, source_bytes)
        .with_context(|| format!("failed to write {}", source_path.display()))?;
    let stdout = run_encoder(&source_path, &encoded_dir)?;
    parse_encoded_input("prompt_token_ids", &stdout)
}

/// Input-embedding cache kind, versioned with the staged schema and key:
/// `-v2` moved the key to the cheap tensor-region fingerprint
/// ([`AuthenticatedDecoderEmbeddingSource::cache_fingerprint`]); `-v3`
/// packs each row into a single hex-string leaf so the encoded index stays
/// O(vocab) nodes. Old-kind entries are never confused with new ones.
pub(crate) const EMBEDDING_CACHE_KIND: &str = "gemma-input-embedding-v3";

/// Pure filesystem lookup of the pre-encoded embedding external — the only
/// resolution the run path performs. Never materializes the table: the
/// entry is addressed by the tensor-region fingerprint.
pub(crate) fn lookup_embedding_external(
    source: &AuthenticatedDecoderEmbeddingSource,
    externals_dir: &Path,
) -> Result<Option<EncodedExternal>> {
    let fingerprint = source.cache_fingerprint()?;
    lookup_external_entry(externals_dir, EMBEDDING_CACHE_KIND, &fingerprint, "embedding")
}

/// Encodes the embedding committed external into `externals_dir` (warm-up
/// path only), reusing an existing entry when present. The table is
/// materialized and encoded into a temp sibling directory, then atomically
/// renamed into the fingerprint-addressed entry — concurrent warm-ups
/// sharing a persistent directory never observe a partial entry. Returns
/// the entry plus whether it was reused.
pub(crate) fn encode_embedding_external(
    source: &AuthenticatedDecoderEmbeddingSource,
    externals_dir: &Path,
) -> Result<(EncodedExternal, bool)> {
    if let Some(entry) = lookup_embedding_external(source, externals_dir)? {
        return Ok((entry, true));
    }

    let fingerprint = source.cache_fingerprint()?;
    let kind_dir = externals_dir.join(EMBEDDING_CACHE_KIND);
    let entry_dir = kind_dir.join(&fingerprint);
    let temp_dir = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NONCE: AtomicU64 = AtomicU64::new(0);
        kind_dir.join(format!(
            ".encode-{fingerprint}-{}-{}",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed),
        ))
    };
    std::fs::create_dir_all(&temp_dir)
        .with_context(|| format!("failed to create {}", temp_dir.display()))?;

    let embedding = staged_embedding_table(source)?;
    let encode_source = StagedInputEmbeddingEncodedInputs {
        prompt_token_ids: None,
        embedding: Some(embedding),
    };
    // Stream compact JSON straight to disk: for a real model this table is
    // gigabytes, so buffering the serialized form in memory (or pretty-
    // printing it) is prohibitive. The JSON is encode scratch — the cache
    // key is the tensor fingerprint and the commitment comes from the
    // raster encoder — so the byte shape of this file is irrelevant.
    let source_path = temp_dir.join("embedding_source.json");
    let source_file = std::fs::File::create(&source_path)
        .with_context(|| format!("failed to create {}", source_path.display()))?;
    let mut writer = std::io::BufWriter::new(source_file);
    serde_json::to_writer(&mut writer, &encode_source)
        .with_context(|| format!("failed to write {}", source_path.display()))?;
    std::io::Write::flush(&mut writer)
        .with_context(|| format!("failed to flush {}", source_path.display()))?;
    drop(writer);
    drop(encode_source);
    let encoded = match run_encoder(&source_path, &temp_dir)
        .and_then(|stdout| parse_encoded_input("embedding", &stdout))
    {
        Ok(encoded) => encoded,
        Err(error) => {
            // The scratch JSON is gigabytes for a real model; never leave
            // it behind on a failed encode (the error carries the encoder's
            // stdout/stderr as the debugging evidence).
            std::fs::remove_dir_all(&temp_dir).ok();
            return Err(error);
        }
    };
    // The JSON source is encode scratch — drop it so the persistent entry
    // carries only the raster pair + commitment. root_commitment.txt is
    // written last: a lookup only hits once all three files exist.
    std::fs::remove_file(&source_path).ok();
    std::fs::write(temp_dir.join("root_commitment.txt"), &encoded.root_commitment)
        .with_context(|| format!("failed to write {}", temp_dir.display()))?;

    if let Err(error) = std::fs::rename(&temp_dir, &entry_dir) {
        // A concurrent warm-up may have renamed its entry first; that
        // entry has the same fingerprint-addressed content, so reuse it.
        std::fs::remove_dir_all(&temp_dir).ok();
        if !entry_dir.is_dir() {
            return Err(anyhow::Error::new(error).context(format!(
                "failed to move encoded embedding entry into {}",
                entry_dir.display()
            )));
        }
    }

    let entry = lookup_embedding_external(source, externals_dir)?.ok_or_else(|| {
        anyhow::anyhow!(
            "embedding encoder reported success but no cache entry appeared at {}",
            entry_dir.display()
        )
    })?;
    Ok((entry, false))
}

fn staged_embedding_table(
    source: &AuthenticatedDecoderEmbeddingSource,
) -> Result<StagedGemmaInputEmbeddingTable> {
    let metadata = source.metadata();
    let vocab_size = usize_to_u32(metadata.vocab_size, "input embedding vocab size")?;
    let hidden_size = usize_to_u32(metadata.hidden_size, "input embedding hidden size")?;
    let mut rows = Vec::with_capacity(metadata.vocab_size);
    for token_id in 0..vocab_size {
        let row_bits = source
            .auth_read(GemmaInputEmbeddingRowRequest { token_id })?
            .into_iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        rows.push(pack_embedding_row_hex(&row_bits));
    }
    Ok(StagedGemmaInputEmbeddingTable {
        metadata: StagedGemmaInputEmbeddingMetadata {
            source_id: metadata.source_id,
            vocab_size,
            hidden_size,
            scale_bits: metadata.scale_bits,
        },
        rows,
    })
}

fn run_encoder(source_path: &Path, out_dir: &Path) -> Result<String> {
    // --release: encoding a real model's table is compute-bound (multi-GB
    // JSON parse + Merkle commitments over the full vocab); a debug-profile
    // encoder turns the one-time warm-up into hours.
    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "--release",
            "-p",
            "raster-program-input-embedding",
            "--features",
            "encode",
            "--bin",
            "encode",
            "--",
        ])
        .arg(source_path)
        .arg(out_dir)
        .current_dir(workspace_root())
        .output()
        .with_context(|| "failed to launch input_embedding encoder")?;
    if !output.status.success() {
        bail!(
            "input_embedding encoder failed (status {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn parse_encoded_input(name: &str, stdout: &str) -> Result<EncodedInputEmbeddingInput> {
    let field = |suffix: &str| -> Result<String> {
        let prefix = format!("{name}_{suffix}: ");
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .map(|rest| rest.trim().to_string())
            .with_context(|| format!("input_embedding encoder printed no '{prefix}' line"))
    };
    Ok(EncodedInputEmbeddingInput {
        data_path: PathBuf::from(field("data_path")?),
        index_path: PathBuf::from(field("index_path")?),
        root_commitment: field("root_commitment")?,
    })
}

fn validate_output(
    output: &InputEmbeddingOutputMirror,
    prompt_token_ids: &StagedInputEmbeddingPromptTokenIds,
    config: &StagedInputEmbeddingConfig,
    run_dir_root: &Path,
) -> Result<()> {
    if output.prompt_token_count != prompt_token_ids.token_count {
        bail!(
            "raster-core input.embedding reported {} rows, expected {} (run dir kept at {})",
            output.prompt_token_count,
            prompt_token_ids.token_count,
            run_dir_root.display()
        );
    }
    if output.prompt_token_ids_sha256 != prompt_token_ids.token_ids_sha256 {
        bail!(
            "raster-core input.embedding reported prompt-token commitment {}, expected {} (run dir kept at {})",
            output.prompt_token_ids_sha256,
            prompt_token_ids.token_ids_sha256,
            run_dir_root.display()
        );
    }
    if output.prompt_token_ids_root != config.prompt_token_ids_root {
        bail!(
            "raster-core input.embedding reported prompt-token root {}, expected {} (run dir kept at {})",
            output.prompt_token_ids_root,
            config.prompt_token_ids_root,
            run_dir_root.display()
        );
    }
    if output.embedding_source_root != config.embedding_source_root {
        bail!(
            "raster-core input.embedding reported embedding root {}, expected {} (run dir kept at {})",
            output.embedding_source_root,
            config.embedding_source_root,
            run_dir_root.display()
        );
    }
    if output.activation_rows.len() != output.prompt_token_count as usize {
        bail!(
            "raster-core input.embedding output carries {} activation rows, expected {} (run dir kept at {})",
            output.activation_rows.len(),
            output.prompt_token_count,
            run_dir_root.display()
        );
    }
    for (row_idx, row) in output.activation_rows.iter().enumerate() {
        if row.len() != output.hidden_size as usize {
            bail!(
                "raster-core input.embedding row {row_idx} has width {}, expected {} (run dir kept at {})",
                row.len(),
                output.hidden_size,
                run_dir_root.display()
            );
        }
    }
    Ok(())
}

fn materialize_output(output: InputEmbeddingOutputMirror) -> Result<ActivationSequence> {
    let det_rows = output
        .activation_rows
        .into_iter()
        .map(|row| row.into_iter().map(Act::from_bits).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let det_activations_sha256 = build_det_activation_commitment(&det_rows);
    let internal = InternalActivationSequence::from_det_values(det_rows);
    Ok(ActivationSequence::from_det_internal(
        internal,
        Some(det_activations_sha256),
    ))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn program_crate_dir() -> PathBuf {
    workspace_root().join("crates/raster-programs/input_embedding")
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{label} exceeds u32"))
}

fn chunk_ordinals(item_count: u32, per_tile: u32) -> Vec<u32> {
    (0..item_count.div_ceil(per_tile)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_encoded_input_reads_encoder_stdout() {
        let stdout = "\
prompt_token_ids_data_path: /tmp/prompt_token_ids.rastered
prompt_token_ids_index_path: /tmp/prompt_token_ids.rindex
prompt_token_ids_root_commitment: abc123
";
        let parsed = parse_encoded_input("prompt_token_ids", stdout).expect("parse");
        assert_eq!(
            parsed.data_path,
            PathBuf::from("/tmp/prompt_token_ids.rastered")
        );
        assert_eq!(
            parsed.index_path,
            PathBuf::from("/tmp/prompt_token_ids.rindex")
        );
        assert_eq!(parsed.root_commitment, "abc123");
    }

    #[test]
    fn packed_rows_mirror_the_program_schema_form() {
        // Byte-for-byte contract with the program crate's
        // `pack_embedding_row_hex` (asserted there by `packed_form_is_stable`).
        assert_eq!(
            pack_embedding_row_hex(&[0x0102_0304, -1]),
            "01020304ffffffff"
        );
        assert_eq!(pack_embedding_row_hex(&[]), "");
        assert_eq!(pack_embedding_row_hex(&[i32::MIN]), "80000000");
    }
}
