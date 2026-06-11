//! Protocol-facing audit types for the challenger role.
//!
//! These types describe the outcome of replaying a claimed inference and
//! comparing committed checkpoint traces. They are model-agnostic by design:
//! nothing here may reference a model family.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::runtime::checkpoints::RasterDetourSpec;

/// One committed checkpoint parsed from a serialized trace artifact, with its
/// 1-based occurrence among entries sharing the same checkpoint id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimedTraceEntry {
    pub entry_index: usize,
    pub checkpoint_id: String,
    pub occurrence: usize,
    /// SHA-256 hex commitment over the checkpoint payload.
    pub commitment: String,
}

/// A parsed claimed checkpoint trace — the exact serialized artifact written
/// by a `--commit-checkpoints` run (a JSON array of single-key
/// `{checkpoint_id: sha256-hex}` objects in commit order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedTrace {
    pub entries: Vec<ClaimedTraceEntry>,
}

impl ClaimedTrace {
    /// Parses the raw bytes of a serialized checkpoint trace artifact.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let value: Value =
            serde_json::from_slice(bytes).context("claimed trace artifact is not valid JSON")?;
        Self::from_value(&value)
    }

    /// Parses an in-memory checkpoint payload (the same JSON array shape as
    /// the serialized artifact).
    pub fn from_value(value: &Value) -> Result<Self> {
        let array = value
            .as_array()
            .context("claimed trace artifact should be a JSON array of checkpoint entries")?;
        let mut occurrences: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let entries = array
            .iter()
            .enumerate()
            .map(|(entry_index, entry)| {
                let object = entry.as_object().with_context(|| {
                    format!("claimed trace entry {entry_index} should be an object")
                })?;
                if object.len() != 1 {
                    anyhow::bail!(
                        "claimed trace entry {entry_index} should contain exactly one commitment"
                    );
                }
                let (checkpoint_id, commitment) = object
                    .iter()
                    .next()
                    .expect("single-entry object should have one key");
                let commitment = commitment.as_str().with_context(|| {
                    format!("claimed trace entry {entry_index} commitment should be a string")
                })?;
                let occurrence = occurrences
                    .entry(checkpoint_id.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
                Ok(ClaimedTraceEntry {
                    entry_index,
                    checkpoint_id: checkpoint_id.clone(),
                    occurrence: *occurrence,
                    commitment: commitment.to_string(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { entries })
    }
}

/// The first divergence between a claimed trace and the challenger's honest
/// native replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointDivergence {
    /// Index into the committed checkpoint sequence (commit order).
    pub entry_index: usize,
    /// Checkpoint id at the divergent entry, taken from the replayed
    /// (honest) sequence when available, otherwise from the claimed trace's
    /// unmatched extra entry.
    pub checkpoint_id: String,
    /// 1-based occurrence of `checkpoint_id` at the divergent entry.
    pub occurrence: usize,
    /// Set when the claimed trace committed a different checkpoint id at
    /// this entry (id-sequence divergence rather than value divergence).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_checkpoint_id: Option<String>,
    /// Claimed commitment at the entry; `None` when the claimed trace ends
    /// before this entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_commitment: Option<String>,
    /// Replayed commitment at the entry; `None` when the honest replay ends
    /// before this entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replayed_commitment: Option<String>,
}

/// Raster detour trace produced for a dispute: the spanning routine
/// occurrence re-executed at tile level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetourArtifact {
    /// The detoured `(routine, occurrence)` — same semantics as
    /// `--raster-at`.
    pub spec: RasterDetourSpec,
    /// Path of the serialized raster detour trace artifact.
    pub trace_path: PathBuf,
}

/// Outcome of `challenger::audit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditOutcome {
    /// Every committed checkpoint in the claimed trace matches the honest
    /// native replay (same ids, same order, same commitments).
    NoDivergence,
    /// The claimed trace diverges from the honest replay; `divergence`
    /// identifies the exact first divergent checkpoint.
    Diverged {
        divergence: CheckpointDivergence,
        /// Raster detour trace for the spanning routine occurrence. `None`
        /// when the divergence is structural (id-sequence or length
        /// mismatch) or the divergent routine does not support detours
        /// (e.g. `prompt.prepare`).
        detour: Option<DetourArtifact>,
    },
}
