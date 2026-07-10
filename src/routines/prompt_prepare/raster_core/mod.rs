//! Raster-core (real toolchain) host adapter for `prompt.prepare` (WS3).
//!
//! Stage → run → ingest → materialize, per the WS2 conventions
//! (`docs/plans/ws2-staging.md`) and this routine's port plan
//! (`PORT_PLAN.md` in this directory):
//!
//! 1. The sim's pre-DSL host derivation runs unchanged (decode → messages →
//!    render → normalize → split → initial BPE pieces) and its outputs are
//!    staged as committed inputs: the raster-encoded Gemma tokenizer
//!    external (content-addressed cache, `gemma_externals` encoder) plus
//!    the postcard `initial_pieces` (the selectable `BpePieces` root).
//! 2. `cargo raster run` executes the routine's program crate against the
//!    committed inputs, producing `output.bin` and `commit.bin`.
//! 3. Ingestion validates the artifacts (never the CLI exit code) and
//!    classifies failures into the three H4 error classes; the prompt token
//!    ids come from the program's committed output only.
//!
//! The checkpoint payload stays host-side and backend-invariant: the native
//! executor commits it from the `PromptPreparationState` this adapter
//! returns, through exactly the same formatter as a full-native run.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::runtime::checkpoints::RoutineId;
use crate::runtime::raster_core::externals::{
    lookup_external_entry, missing_external_error, resolve_run_externals_dir, EncodedExternal,
};
use crate::runtime::raster_core::ingest::{ingest, RasterCoreError};
use crate::runtime::raster_core::staging::StagedInputs;
use crate::runtime::raster_core::{CargoRasterRunner, RasterCoreRunDir, COMMIT_FILE_NAME};
use crate::shared::api::input::{InferenceRequest, ModelSpec, PromptPreparationState};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::model::gemma::tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaTokenizerMetadataRequest,
};
use crate::trace::routine_scope;

use super::native::{
    build_gemma4_messages, build_prompt_commitment, decode_prompt_bytes, render_prompt,
};
use super::raster::utils::{
    init_tokenize_prompt, initial_bpe_pieces, normalize_tokenize_prompt, split_tokenize_prompt,
};

/// Field-order mirror of the program crate's staged `BpePieces` (postcard
/// layout contract, WS2 §9.6: the main crate must not depend on program
/// crates that carry the real `raster` dependency). A single-field postcard
/// struct encodes identically to the bare `Vec<String>` it wraps
/// (`staged_pieces_keep_the_bare_vec_byte_layout`), so the selectable-root
/// shape costs nothing at the staging boundary.
#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
struct StagedBpePieces {
    pieces: Vec<String>,
}

/// Field-order mirror of the program crate's `PromptTokenization` output.
#[derive(Debug, serde::Deserialize, PartialEq, Eq)]
struct PromptTokenizationMirror {
    token_ids: Vec<u32>,
    token_count: u32,
}

/// Runs `prompt.prepare` on the real raster toolchain and materializes the
/// native-form state the rest of the run (and the checkpoint formatter)
/// consumes. The prompt token ids are never recomputed natively — they come
/// from the ingested program output.
pub fn run_raster_core(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<PromptPreparationState> {
    let _routine = routine_scope(RoutineId::PromptPrepare, "mode=raster_core");

    // Pre-DSL host derivation — identical to the sim path's
    // `prepare_raster_prompt_input_roots` (spec-faithful boundary, catalog
    // C18): these values are inputs of the verifiable execution, committed
    // via the input manifest.
    let prompt_text = decode_prompt_bytes(&request.prompt_bytes, request.text_decoding_policy)?;
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request.add_generation_prompt)?;
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    let input = init_tokenize_prompt(&rendered_prompt, request.add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer)?;
    let pre_tokenized = split_tokenize_prompt(normalized, tokenizer)?;
    let metadata = ArtifactIo::auth_read(tokenizer, GemmaTokenizerMetadataRequest)?;
    let mut initial_pieces: Vec<String> = Vec::new();
    for segment in &pre_tokenized.segments {
        initial_pieces.extend(initial_bpe_pieces(segment, tokenizer, &metadata)?);
    }

    // Committed-input staging (WS2 §2/§3). Model externals are strict
    // lookup-only at run time: the tokenizer entry must have been
    // pre-encoded via `encode-externals` into the resolved directory.
    let externals_dir = resolve_run_externals_dir(RoutineId::PromptPrepare)
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;
    let tokenizer_external = lookup_tokenizer_external(&model.tokenizer_path, &externals_dir)
        .and_then(|entry| {
            entry.ok_or_else(|| {
                missing_external_error(
                    RoutineId::PromptPrepare,
                    TOKENIZER_CACHE_KIND,
                    &externals_dir,
                )
            })
        })
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;
    let run_dir = RasterCoreRunDir::create(RoutineId::PromptPrepare, 1)?;
    let mut staged = StagedInputs::new();
    staged.add_raster_encoded(
        "tokenizer",
        &tokenizer_external.data_path,
        &tokenizer_external.index_path,
        &tokenizer_external.root_commitment,
    )?;
    staged.add_postcard(
        "initial_pieces",
        &StagedBpePieces {
            pieces: initial_pieces,
        },
    )?;
    let input_commitments = staged
        .write(&run_dir)
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;

    // Execute + ingest (run directory kept on failure as debugging
    // evidence; the CLI exit code is never consulted — WS2 §6).
    let run_output = CargoRasterRunner::default()
        .run(&program_crate_dir(), &run_dir)
        .map_err(RasterCoreError::Infrastructure)
        .map_err(anyhow::Error::new)?;
    let run_result = ingest::<PromptTokenizationMirror>(&run_dir, &run_output, input_commitments)
        .map_err(anyhow::Error::new)?;

    let tokenization = run_result.value;
    if tokenization.token_count as usize != tokenization.token_ids.len() {
        bail!(
            "raster-core prompt tokenization is inconsistent: output reports {} tokens but \
             carries {} ids (run dir kept at {})",
            tokenization.token_count,
            tokenization.token_ids.len(),
            run_result.run_dir_root.display()
        );
    }

    let prompt_token_ids = tokenization.token_ids;
    let prompt_token_ids_sha256 = build_prompt_commitment(&prompt_token_ids)?;
    // Preserve the trace commitment next to the CLI's persistent run
    // artifacts (`target/raster/runs/<run-id>/`, alongside `trace.ndjson`)
    // before the temp staging dir is cleaned up.
    if let Some(artifacts_dir) = run_result.trace_path.parent() {
        std::fs::copy(
            &run_result.commit_path,
            artifacts_dir.join(COMMIT_FILE_NAME),
        )
        .ok();
    }
    std::fs::remove_dir_all(run_dir.root()).ok();

    Ok(PromptPreparationState {
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}

/// Gemma-tokenizer cache kind (WS2 cache convention:
/// `<externals_dir>/gemma-tokenizer-v2/<sha256(tokenizer.json)>/`; the
/// cache kind is versioned with the schema — `-v2` is the chunked shape —
/// and must match the `gemma_externals` encoder's `CACHE_KIND`).
pub(crate) const TOKENIZER_CACHE_KIND: &str = "gemma-tokenizer-v2";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn program_crate_dir() -> PathBuf {
    workspace_root().join("crates/raster-programs/prompt_prepare")
}

/// Pure filesystem lookup of the pre-encoded tokenizer external — the only
/// resolution the run path performs (strict lookup-only contract; no lazy
/// encode).
pub(crate) fn lookup_tokenizer_external(
    tokenizer_json: &Path,
    externals_dir: &Path,
) -> Result<Option<EncodedExternal>> {
    let source_bytes = std::fs::read(tokenizer_json)
        .with_context(|| format!("failed to read {}", tokenizer_json.display()))?;
    let source_sha256 = format!("{:x}", Sha256::digest(&source_bytes));
    lookup_external_entry(externals_dir, TOKENIZER_CACHE_KIND, &source_sha256, "tokenizer")
}

/// Encodes the tokenizer committed external into `externals_dir` (warm-up
/// path only), reusing an existing entry when present. Encoding stays out
/// of the main crate's dependency graph (charter invariant 6): the
/// `gemma_externals` encoder runs as a `cargo run` subprocess and manages
/// the content-addressed entry layout itself. Returns the entry plus
/// whether it was reused.
pub(crate) fn encode_tokenizer_external(
    tokenizer_json: &Path,
    externals_dir: &Path,
) -> Result<(EncodedExternal, bool)> {
    if let Some(entry) = lookup_tokenizer_external(tokenizer_json, externals_dir)? {
        return Ok((entry, true));
    }

    // --release: a real model's tokenizer carries a ~262k-entry vocab; the
    // one-time encode is compute-bound and painfully slow in debug.
    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "--release",
            "-p",
            "raster-program-gemma-externals",
            "--features",
            "encode",
            "--bin",
            "encode",
            "--",
        ])
        .arg(tokenizer_json)
        .arg(externals_dir)
        .current_dir(workspace_root())
        .output()
        .context("failed to launch the gemma_externals tokenizer encoder")?;
    if !output.status.success() {
        bail!(
            "gemma_externals tokenizer encoder failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let entry = lookup_tokenizer_external(tokenizer_json, externals_dir)?.ok_or_else(|| {
        anyhow::anyhow!(
            "tokenizer encoder reported success but no cache entry appeared under {}",
            externals_dir.join(TOKENIZER_CACHE_KIND).display()
        )
    })?;
    Ok((entry, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_pieces_keep_the_bare_vec_byte_layout() {
        // WS2 §9.6 layout contract: wrapping the staged pieces in the
        // selectable `BpePieces` root must not change the postcard bytes
        // (postcard structs are unframed field sequences).
        let pieces = vec!["a".to_string(), "▁b".to_string(), String::new()];
        let wrapped = postcard::to_allocvec(&StagedBpePieces {
            pieces: pieces.clone(),
        })
        .expect("encode wrapped");
        let bare = postcard::to_allocvec(&pieces).expect("encode bare");
        assert_eq!(wrapped, bare);
    }

    #[test]
    fn missing_tokenizer_json_is_reported_with_its_path() {
        let error = lookup_tokenizer_external(
            Path::new("/nonexistent/tokenizer.json"),
            Path::new("/nonexistent/externals"),
        )
        .expect_err("missing source must fail");
        assert!(error.to_string().contains("/nonexistent/tokenizer.json"));
    }

    #[test]
    fn lookup_misses_in_an_empty_externals_dir() {
        let dir = std::env::temp_dir().join(format!(
            "raster-tokenizer-lookup-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let tokenizer_json = dir.join("tokenizer.json");
        std::fs::write(&tokenizer_json, b"{}").expect("tokenizer fixture");
        let entry = lookup_tokenizer_external(&tokenizer_json, &dir)
            .expect("lookup should not error against an empty dir");
        assert!(entry.is_none(), "empty externals dir must be a miss");
        std::fs::remove_dir_all(&dir).ok();
    }
}
