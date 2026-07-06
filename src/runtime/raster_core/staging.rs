//! Committed-input staging for raster-core runs (WS2).
//!
//! Materializes native state into the committed input set the real toolchain
//! consumes: `input.json` (private file bindings) + `input_manifest.json`
//! (public commitments), per the tokenizer-PoC idiom the WS1 catalog verified
//! (probe P3). Two encodings are supported:
//!
//! - **postcard** — request-scoped values serialized by the host into the run
//!   directory; commitment = SHA-256 of the raw file bytes. The main crate
//!   stages these directly (postcard + sha2 are existing dependencies; no
//!   `raster` dependency is needed, preserving charter invariant 6).
//! - **raster** — model-scoped data pre-encoded offline by a program crate's
//!   encoder bin (`.rastered` + `.rindex`, mmap load preference); commitment =
//!   the raster index root, computed by the encoder and passed in here. The
//!   staged entry references the cached files by absolute path.
//!
//! Every staged input's commitment is captured and returned from
//! [`StagedInputs::write`] so host adapters can surface them in run results
//! (WS3 exit criterion: staged inputs carry captured commitments).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::RasterCoreRunDir;

/// One staged input's captured public commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedInputCommitment {
    /// `postcard` or `raster` — mirrors the manifest `encoding` field.
    pub encoding: &'static str,
    /// Hex commitment as written to `input_manifest.json`.
    pub commitment: String,
}

/// Map from logical input name to its captured commitment.
pub type StagedInputCommitments = BTreeMap<String, StagedInputCommitment>;

#[derive(Debug)]
enum StagedEntry {
    Postcard {
        bytes: Vec<u8>,
    },
    RasterEncoded {
        path: PathBuf,
        index_path: PathBuf,
        root_commitment: String,
    },
}

/// Builder for one run's committed input set.
///
/// Accumulates entries in memory; [`StagedInputs::write`] materializes the
/// payload files and both JSON documents into a [`RasterCoreRunDir`]. Payload
/// files are written before the manifest so a partially staged directory can
/// never carry a manifest that commits to absent bytes.
#[derive(Debug, Default)]
pub struct StagedInputs {
    entries: BTreeMap<String, StagedEntry>,
}

impl StagedInputs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stages a postcard-encoded value under `name` (`<name>.bin` in the run
    /// directory, `load_preference: read`).
    pub fn add_postcard<T: Serialize>(&mut self, name: &str, value: &T) -> Result<&mut Self> {
        let bytes = postcard::to_allocvec(value)
            .with_context(|| format!("failed to postcard-encode staged input '{name}'"))?;
        self.insert(name, StagedEntry::Postcard { bytes })?;
        Ok(self)
    }

    /// Stages a reference to pre-encoded raster files (`load_preference:
    /// mmap`). `root_commitment` is the raster index root the encoder
    /// reported; the runtime re-derives and verifies it at resolve time.
    ///
    /// Paths are canonicalized to absolute form: the run directory is not
    /// where the cached files live, and `input.json` paths resolve relative
    /// to the document's directory.
    pub fn add_raster_encoded(
        &mut self,
        name: &str,
        path: &Path,
        index_path: &Path,
        root_commitment: &str,
    ) -> Result<&mut Self> {
        let path = path.canonicalize().with_context(|| {
            format!(
                "raster-encoded input '{name}' payload does not exist: {}",
                path.display()
            )
        })?;
        let index_path = index_path.canonicalize().with_context(|| {
            format!(
                "raster-encoded input '{name}' index does not exist: {}",
                index_path.display()
            )
        })?;
        self.insert(
            name,
            StagedEntry::RasterEncoded {
                path,
                index_path,
                root_commitment: root_commitment.to_string(),
            },
        )?;
        Ok(self)
    }

    /// Writes the payload files, `input.json`, and `input_manifest.json` into
    /// the run directory; returns the captured commitment map.
    pub fn write(&self, run_dir: &RasterCoreRunDir) -> Result<StagedInputCommitments> {
        let mut input = serde_json::Map::new();
        let mut manifest = serde_json::Map::new();
        let mut commitments = StagedInputCommitments::new();

        for (name, entry) in &self.entries {
            let (input_entry, manifest_entry, captured) = match entry {
                StagedEntry::Postcard { bytes } => {
                    let file_name = format!("{name}.bin");
                    let file_path = run_dir.root().join(&file_name);
                    std::fs::write(&file_path, bytes).with_context(|| {
                        format!(
                            "failed to write staged input '{name}' to {}",
                            file_path.display()
                        )
                    })?;
                    let commitment = sha256_hex(bytes);
                    (
                        serde_json::json!({
                            "path": file_name,
                            "load_preference": "read",
                        }),
                        serde_json::json!({
                            "type": "sha256",
                            "encoding": "postcard",
                            "commitment": commitment,
                        }),
                        StagedInputCommitment {
                            encoding: "postcard",
                            commitment,
                        },
                    )
                }
                StagedEntry::RasterEncoded {
                    path,
                    index_path,
                    root_commitment,
                } => (
                    serde_json::json!({
                        "path": path,
                        "index_path": index_path,
                        "load_preference": "mmap",
                    }),
                    serde_json::json!({
                        "type": "sha256",
                        "encoding": "raster",
                        "commitment": root_commitment,
                    }),
                    StagedInputCommitment {
                        encoding: "raster",
                        commitment: root_commitment.clone(),
                    },
                ),
            };
            input.insert(name.clone(), input_entry);
            manifest.insert(name.clone(), manifest_entry);
            commitments.insert(name.clone(), captured);
        }

        write_json(&run_dir.input_path(), &serde_json::Value::Object(input))?;
        write_json(
            &run_dir.input_manifest_path(),
            &serde_json::Value::Object(manifest),
        )?;
        Ok(commitments)
    }

    fn insert(&mut self, name: &str, entry: StagedEntry) -> Result<()> {
        validate_input_name(name)?;
        if self.entries.contains_key(name) {
            bail!("staged input '{name}' declared twice");
        }
        self.entries.insert(name.to_string(), entry);
        Ok(())
    }
}

/// Logical input names double as payload file stems and manifest keys; keep
/// them to a shell-safe, collision-free alphabet.
fn validate_input_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if !valid {
        bail!("staged input name '{name}' must be non-empty [a-z0-9_]+");
    }
    Ok(())
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    std::fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::checkpoints::RoutineId;

    fn temp_run_dir() -> RasterCoreRunDir {
        RasterCoreRunDir::create(RoutineId::PromptPrepare, 1).expect("run dir")
    }

    #[test]
    fn postcard_inputs_stage_with_sha256_commitments() {
        let run_dir = temp_run_dir();
        let mut staged = StagedInputs::new();
        staged
            .add_postcard("prompt", &"hello".to_string())
            .expect("stage prompt");
        staged
            .add_postcard("chunk_widths", &vec![4u64, 8, 16])
            .expect("stage widths");
        let commitments = staged.write(&run_dir).expect("write staged inputs");

        let prompt_bytes = std::fs::read(run_dir.root().join("prompt.bin")).expect("prompt file");
        assert_eq!(
            postcard::from_bytes::<String>(&prompt_bytes).expect("round-trip"),
            "hello"
        );
        assert_eq!(
            commitments["prompt"].commitment,
            sha256_hex(&prompt_bytes),
            "captured commitment must equal the payload file hash"
        );
        assert_eq!(commitments["prompt"].encoding, "postcard");

        let input: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_dir.input_path()).expect("input.json"))
                .expect("parse input.json");
        assert_eq!(input["prompt"]["path"], "prompt.bin");
        assert_eq!(input["prompt"]["load_preference"], "read");
        assert_eq!(input["chunk_widths"]["path"], "chunk_widths.bin");

        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run_dir.input_manifest_path()).expect("input_manifest.json"),
        )
        .expect("parse input_manifest.json");
        assert_eq!(manifest["prompt"]["type"], "sha256");
        assert_eq!(manifest["prompt"]["encoding"], "postcard");
        assert_eq!(
            manifest["prompt"]["commitment"],
            serde_json::Value::String(commitments["prompt"].commitment.clone())
        );

        std::fs::remove_dir_all(run_dir.root()).ok();
    }

    #[test]
    fn raster_encoded_inputs_reference_cached_files_by_absolute_path() {
        let run_dir = temp_run_dir();
        let cache_dir = run_dir.root().join("fake-cache");
        std::fs::create_dir_all(&cache_dir).expect("cache dir");
        let payload = cache_dir.join("tokenizer.rastered");
        let index = cache_dir.join("tokenizer.rindex");
        std::fs::write(&payload, b"payload").expect("payload");
        std::fs::write(&index, b"index").expect("index");

        let mut staged = StagedInputs::new();
        staged
            .add_raster_encoded("tokenizer", &payload, &index, "abc123")
            .expect("stage tokenizer");
        let commitments = staged.write(&run_dir).expect("write staged inputs");
        assert_eq!(commitments["tokenizer"].encoding, "raster");
        assert_eq!(commitments["tokenizer"].commitment, "abc123");

        let input: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_dir.input_path()).expect("input.json"))
                .expect("parse input.json");
        let staged_path = input["tokenizer"]["path"].as_str().expect("path string");
        let staged_index = input["tokenizer"]["index_path"]
            .as_str()
            .expect("index path string");
        assert!(Path::new(staged_path).is_absolute());
        assert!(Path::new(staged_index).is_absolute());
        assert_eq!(input["tokenizer"]["load_preference"], "mmap");

        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run_dir.input_manifest_path()).expect("input_manifest.json"),
        )
        .expect("parse input_manifest.json");
        assert_eq!(manifest["tokenizer"]["encoding"], "raster");
        assert_eq!(manifest["tokenizer"]["commitment"], "abc123");

        std::fs::remove_dir_all(run_dir.root()).ok();
    }

    #[test]
    fn missing_raster_files_and_bad_names_are_rejected() {
        let run_dir = temp_run_dir();
        let mut staged = StagedInputs::new();
        let missing = run_dir.root().join("nope.rastered");
        let error = staged
            .add_raster_encoded("tokenizer", &missing, &missing, "abc")
            .expect_err("missing files must fail at staging time");
        assert!(error.to_string().contains("does not exist"));

        for bad in ["", "Tokenizer", "with-dash", "a b"] {
            assert!(
                staged.add_postcard(bad, &1u8).is_err(),
                "name '{bad}' should be rejected"
            );
        }

        staged.add_postcard("dup", &1u8).expect("first dup ok");
        let error = staged
            .add_postcard("dup", &2u8)
            .expect_err("duplicate names must fail");
        assert!(error.to_string().contains("declared twice"));

        std::fs::remove_dir_all(run_dir.root()).ok();
    }
}
