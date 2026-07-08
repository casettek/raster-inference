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

    // Committed-input staging (WS2 §2/§3).
    let tokenizer_external = encode_tokenizer_external_cached(&model.tokenizer_path)
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

/// One encoded Gemma-tokenizer cache entry (WS2 cache convention:
/// `<cache_root>/gemma-tokenizer-v2/<sha256(tokenizer.json)>/`; the cache
/// kind is versioned with the schema — `-v2` is the chunked shape — and
/// must match the `gemma_externals` encoder's `CACHE_KIND`).
#[derive(Debug)]
struct EncodedTokenizerExternal {
    data_path: PathBuf,
    index_path: PathBuf,
    root_commitment: String,
}

/// Cache root for raster-encoded externals. Overridable for hermetic tests
/// via `RASTER_CORE_EXTERNAL_CACHE`.
fn external_cache_root() -> PathBuf {
    std::env::var_os("RASTER_CORE_EXTERNAL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("raster-inference-gemma-external-cache"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn program_crate_dir() -> PathBuf {
    workspace_root().join("crates/raster-programs/prompt_prepare")
}

/// Resolves the tokenizer committed external from the content-addressed
/// cache, invoking the `gemma_externals` offline encoder on a miss.
/// Encoding stays out of the main crate's dependency graph (charter
/// invariant 6): the encoder runs as a `cargo run` subprocess, exactly like
/// the WS2 tokenizer-external test drives it.
fn encode_tokenizer_external_cached(tokenizer_json: &Path) -> Result<EncodedTokenizerExternal> {
    let source_bytes = std::fs::read(tokenizer_json)
        .with_context(|| format!("failed to read {}", tokenizer_json.display()))?;
    let source_sha256 = format!("{:x}", Sha256::digest(&source_bytes));

    let cache_root = external_cache_root();
    let entry_dir = cache_root.join("gemma-tokenizer-v2").join(&source_sha256);
    let data_path = entry_dir.join("tokenizer.rastered");
    let index_path = entry_dir.join("tokenizer.rindex");
    let commitment_path = entry_dir.join("root_commitment.txt");
    if data_path.is_file() && index_path.is_file() && commitment_path.is_file() {
        let root_commitment = std::fs::read_to_string(&commitment_path)
            .with_context(|| format!("failed to read {}", commitment_path.display()))?
            .trim()
            .to_string();
        return Ok(EncodedTokenizerExternal {
            data_path,
            index_path,
            root_commitment,
        });
    }

    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "-p",
            "raster-program-gemma-externals",
            "--features",
            "encode",
            "--bin",
            "encode",
            "--",
        ])
        .arg(tokenizer_json)
        .arg(&cache_root)
        .current_dir(workspace_root())
        .output()
        .context("failed to launch the gemma_externals tokenizer encoder")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        bail!(
            "gemma_externals tokenizer encoder failed\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let field = |prefix: &str| -> Result<String> {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .map(|rest| rest.trim().to_string())
            .with_context(|| format!("tokenizer encoder printed no '{prefix}' line"))
    };
    Ok(EncodedTokenizerExternal {
        data_path: PathBuf::from(field("data_path: ")?),
        index_path: PathBuf::from(field("index_path: ")?),
        root_commitment: field("root_commitment: ")?,
    })
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
        let error = encode_tokenizer_external_cached(Path::new("/nonexistent/tokenizer.json"))
            .expect_err("missing source must fail");
        assert!(error.to_string().contains("/nonexistent/tokenizer.json"));
    }
}
