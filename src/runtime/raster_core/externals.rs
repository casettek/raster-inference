//! Pre-encoded model externals directory: one-time explicit encode, strict
//! lookup-only consumption at run time.
//!
//! Model-scoped raster externals (the Gemma tokenizer, the input-embedding
//! table) are encoded exactly once into a persistent, user-chosen directory
//! via [`warm_model_externals`] (CLI: `encode-externals`). Raster-core run
//! adapters never encode model externals: they resolve the directory
//! strictly ([`resolve_run_externals_dir`]) and require a cache hit,
//! failing with an actionable error naming the missing kind, the resolved
//! directory, and the exact `encode-externals` command otherwise
//! ([`missing_external_error`]).
//!
//! The directory keeps the content-addressed cache layout
//! (`<root>/<kind>/<key>/<stem>.rastered` + `<stem>.rindex` +
//! `root_commitment.txt`); the pre-encode output directory *is* the
//! external cache root. Run-local encodes (per-request commitments like
//! prompt token ids) are not model externals and stay untouched.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::routines::{input_embedding, prompt_prepare};
use crate::runtime::checkpoints::RoutineId;
use crate::shared::model::runtime::LoadedModel;

/// Environment variable naming the pre-encoded externals directory. The
/// CLI's `--externals-dir` flag sets it (same pattern as `--trace-dir` →
/// `RASTER_TRACE_DIR`).
pub const EXTERNAL_CACHE_ENV: &str = "RASTER_CORE_EXTERNAL_CACHE";

/// One resolved cache entry: the raster-encoded pair plus its index root
/// commitment (the `input_manifest.json` value).
#[derive(Debug, Clone)]
pub(crate) struct EncodedExternal {
    pub data_path: PathBuf,
    pub index_path: PathBuf,
    pub root_commitment: String,
}

/// One pre-encoded external reported by [`warm_model_externals`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct WarmedExternal {
    /// Cache kind segment, e.g. `gemma-tokenizer-v2`.
    pub kind: String,
    /// The content-addressed entry directory under the externals dir.
    pub entry_dir: PathBuf,
    /// Raster index root commitment of the encoded entry.
    pub root_commitment: String,
    /// `true` when the entry already existed and was not re-encoded.
    pub reused: bool,
}

/// Result of one [`warm_model_externals`] invocation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WarmedExternals {
    pub externals_dir: PathBuf,
    pub externals: Vec<WarmedExternal>,
}

/// Encodes every model-scoped external the currently migrated raster-core
/// routines consume (Gemma tokenizer, input-embedding table) into
/// `externals_dir`, reusing entries that are already present. This is the
/// single public warm-up entry point — per-routine encode helpers stay
/// `pub(crate)` so the routine hosts keep their single-public-entrypoint
/// contract.
pub fn warm_model_externals(
    model: &LoadedModel,
    externals_dir: &Path,
) -> Result<WarmedExternals> {
    std::fs::create_dir_all(externals_dir).with_context(|| {
        format!(
            "failed to create externals directory {}",
            externals_dir.display()
        )
    })?;

    let mut externals = Vec::new();

    let (tokenizer, tokenizer_reused) = prompt_prepare::raster_core::encode_tokenizer_external(
        &model.model_spec().tokenizer_path,
        externals_dir,
    )?;
    externals.push(warmed_entry(
        prompt_prepare::raster_core::TOKENIZER_CACHE_KIND,
        &tokenizer,
        tokenizer_reused,
    )?);

    let embedding_source = model.input_embedding_source()?;
    let (embedding, embedding_reused) = input_embedding::raster_core::encode_embedding_external(
        &embedding_source,
        externals_dir,
    )?;
    externals.push(warmed_entry(
        input_embedding::raster_core::EMBEDDING_CACHE_KIND,
        &embedding,
        embedding_reused,
    )?);

    Ok(WarmedExternals {
        externals_dir: externals_dir.to_path_buf(),
        externals,
    })
}

fn warmed_entry(kind: &str, entry: &EncodedExternal, reused: bool) -> Result<WarmedExternal> {
    let entry_dir = entry
        .data_path
        .parent()
        .ok_or_else(|| {
            anyhow!(
                "encoded external {} has no parent entry directory",
                entry.data_path.display()
            )
        })?
        .to_path_buf();
    Ok(WarmedExternal {
        kind: kind.to_string(),
        entry_dir,
        root_commitment: entry.root_commitment.clone(),
        reused,
    })
}

/// Strict externals-directory resolution for raster-core run adapters:
/// [`EXTERNAL_CACHE_ENV`] must be set (the CLI's `--externals-dir` flag
/// sets it). There is no temp-dir fallback on the run path — a raster-core
/// run without a pre-encoded directory fails up front with the remediation
/// command.
pub(crate) fn resolve_run_externals_dir(routine: RoutineId) -> Result<PathBuf> {
    std::env::var_os(EXTERNAL_CACHE_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow!(
                "raster-core {} requires a pre-encoded model externals directory: pass \
                 --externals-dir <DIR> (or set {EXTERNAL_CACHE_ENV}) pointing at a directory \
                 prepared with: raster-inference encode-externals --model <model-dir> \
                 --externals-dir <DIR>",
                routine.as_str()
            )
        })
}

/// The actionable missing-entry error for a strict lookup miss: names the
/// external kind, the resolved directory, and the exact command to run.
pub(crate) fn missing_external_error(
    routine: RoutineId,
    kind: &str,
    externals_dir: &Path,
) -> anyhow::Error {
    anyhow!(
        "{} requires the pre-encoded '{kind}' external; not found under {}. Run: \
         raster-inference encode-externals --model <model-dir> --externals-dir {}",
        routine.as_str(),
        externals_dir.display(),
        externals_dir.display()
    )
}

/// Pure filesystem lookup of one content-addressed cache entry
/// (`<externals_dir>/<kind>/<key>/<stem>.rastered` etc). Returns `None`
/// when any of the three entry files is absent.
pub(crate) fn lookup_external_entry(
    externals_dir: &Path,
    kind: &str,
    key: &str,
    stem: &str,
) -> Result<Option<EncodedExternal>> {
    let entry_dir = externals_dir.join(kind).join(key);
    let data_path = entry_dir.join(format!("{stem}.rastered"));
    let index_path = entry_dir.join(format!("{stem}.rindex"));
    let commitment_path = entry_dir.join("root_commitment.txt");
    if !(data_path.is_file() && index_path.is_file() && commitment_path.is_file()) {
        return Ok(None);
    }
    let root_commitment = std::fs::read_to_string(&commitment_path)
        .with_context(|| format!("failed to read {}", commitment_path.display()))?
        .trim()
        .to_string();
    Ok(Some(EncodedExternal {
        data_path,
        index_path,
        root_commitment,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_misses_on_absent_and_partial_entries() {
        let root = std::env::temp_dir().join(format!(
            "raster-externals-lookup-test-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&root).ok();

        assert!(lookup_external_entry(&root, "kind-v1", "key", "stem")
            .expect("lookup should not error")
            .is_none());

        let entry_dir = root.join("kind-v1").join("key");
        std::fs::create_dir_all(&entry_dir).expect("entry dir");
        std::fs::write(entry_dir.join("stem.rastered"), b"data").expect("data");
        std::fs::write(entry_dir.join("stem.rindex"), b"index").expect("index");
        assert!(
            lookup_external_entry(&root, "kind-v1", "key", "stem")
                .expect("lookup should not error")
                .is_none(),
            "an entry without root_commitment.txt must be a miss"
        );

        std::fs::write(entry_dir.join("root_commitment.txt"), "abc123\n").expect("commitment");
        let hit = lookup_external_entry(&root, "kind-v1", "key", "stem")
            .expect("lookup should not error")
            .expect("complete entry should hit");
        assert_eq!(hit.root_commitment, "abc123");
        assert_eq!(hit.data_path, entry_dir.join("stem.rastered"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_external_error_names_kind_dir_and_command() {
        let error = missing_external_error(
            RoutineId::InputEmbedding,
            "gemma-input-embedding-v2",
            Path::new("/some/dir"),
        );
        let message = error.to_string();
        assert!(message.contains("input.embedding"));
        assert!(message.contains("gemma-input-embedding-v2"));
        assert!(message.contains("/some/dir"));
        assert!(message.contains("encode-externals"));
    }
}
