//! Challenger role: replay, locate the first divergence, detour to raster.
//!
//! # v1 semantics: re-run, not resume
//!
//! The challenger does not resume from intermediate inference state. Both
//! the honest replay and the raster detour re-run the request from the start
//! with controls set (full-native with commitment; then `--raster-at`
//! equivalent detour controls). This is acceptable for v1 — checkpoint
//! commitments are deterministic, so a re-run reproduces the identical
//! committed sequence — and keeps the detour trace byte-compatible with what
//! `--raster-at` produces today. True mid-state resumption is deferred.

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::runtime::checkpoints::{RasterDetourSpec, RoutineId};
use crate::runtime::inference::{InferenceControls, InferenceRunOutcome};
use crate::runtime::roles::{detour, ExecutionTuning};
use crate::runtime::{sequence, trace};
use crate::shared::api::audit::{
    AuditOutcome, CheckpointDivergence, ClaimedTrace, ClaimedTraceEntry,
};
use crate::shared::api::input::{InferenceExecutionMode, InferenceRequest, ModelSpec};
use crate::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::Gemma4TransformerModel;

/// Routines whose selective raster detour is implemented on the native path.
/// Divergences at other routines (`prompt.prepare`,
/// `prefill.range_finalize`) are reported without a detour artifact.
const DETOURABLE_ROUTINES: [RoutineId; 8] = [
    RoutineId::InputEmbedding,
    RoutineId::PrefillPrepareAux,
    RoutineId::PrefillRange,
    RoutineId::PrefillFinalize,
    RoutineId::SelectOutputToken,
    RoutineId::DecodeLayerRange,
    RoutineId::DecodeTransitionFinalize,
    RoutineId::FinalizeOutput,
];

/// Audits a claimed inference trace.
///
/// 1. Replays the request natively (claimer policy) and collects the honest
///    committed checkpoint sequence.
/// 2. Compares it positionally against the claimed trace — the exact
///    serialized artifact a `--commit-checkpoints` run writes.
/// 3. On the first divergence, re-executes from the start with a
///    single-detour policy at the divergent routine occurrence (reusing
///    `RasterDetourController` semantics, so the detour trace matches the
///    equivalent `--raster-at` output) and returns the divergence report
///    plus the raster detour trace handle.
///
/// Auditing requires deterministic execution: commitments are only
/// reproducible in deterministic mode, and the raster detour itself requires
/// it.
pub fn audit(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    raster_tokenizer_source: AuthenticatedGemmaTokenizer,
    claimed_trace: &[u8],
    tuning: &ExecutionTuning,
) -> Result<AuditOutcome> {
    if request.execution_mode != InferenceExecutionMode::Deterministic {
        anyhow::bail!("challenger audit requires deterministic execution");
    }
    let claimed = ClaimedTrace::from_json_bytes(claimed_trace)?;

    let mut replay_controls = InferenceControls {
        commit_checkpoints: true,
        raster_tokenizer_source: Some(raster_tokenizer_source.clone()),
        ..Default::default()
    };
    tuning.apply(&mut replay_controls);
    run_to_completion(
        request,
        model,
        tokenizer,
        transformer_model,
        &replay_controls,
        "challenger replay",
    )?;
    let replayed_payload = trace::completed_checkpoint_payload()
        .context("challenger replay did not produce a committed checkpoint payload")?;
    let replayed = ClaimedTrace::from_value(&replayed_payload)?;

    let Some(divergence) = locate_first_divergence(&claimed, &replayed) else {
        return Ok(AuditOutcome::NoDivergence);
    };

    let detour = detour_spec_for(&divergence)
        .map(|spec| {
            let outcome = detour::run(
                request,
                model,
                tokenizer,
                transformer_model,
                raster_tokenizer_source.clone(),
                spec,
                tuning,
            )
            .context("challenger raster detour failed")?;
            Ok::<_, anyhow::Error>(outcome.artifact)
        })
        .transpose()?;

    Ok(AuditOutcome::Diverged { divergence, detour })
}

fn run_to_completion(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    controls: &InferenceControls,
    description: &str,
) -> Result<()> {
    match sequence::run(request, model, tokenizer, transformer_model, controls)? {
        InferenceRunOutcome::Completed(_) => Ok(()),
        InferenceRunOutcome::Paused(paused) => anyhow::bail!(
            "{description} paused unexpectedly at checkpoint {}",
            paused.terminal_checkpoint_id
        ),
        InferenceRunOutcome::RasterPromptPrepared(state) => anyhow::bail!(
            "{description} stopped unexpectedly at routine boundary {}",
            state.terminal_checkpoint_id
        ),
    }
}

/// Finds the first divergence between the claimed and replayed committed
/// checkpoint sequences: a commitment mismatch, an id-sequence mismatch, or
/// a length mismatch. Returns `None` when the traces agree entirely.
fn locate_first_divergence(
    claimed: &ClaimedTrace,
    replayed: &ClaimedTrace,
) -> Option<CheckpointDivergence> {
    let common_len = claimed.entries.len().min(replayed.entries.len());
    for idx in 0..common_len {
        let claimed_entry = &claimed.entries[idx];
        let replayed_entry = &replayed.entries[idx];
        if claimed_entry.checkpoint_id != replayed_entry.checkpoint_id {
            return Some(CheckpointDivergence {
                entry_index: idx,
                checkpoint_id: replayed_entry.checkpoint_id.clone(),
                occurrence: replayed_entry.occurrence,
                claimed_checkpoint_id: Some(claimed_entry.checkpoint_id.clone()),
                claimed_commitment: Some(claimed_entry.commitment.clone()),
                replayed_commitment: Some(replayed_entry.commitment.clone()),
            });
        }
        if claimed_entry.commitment != replayed_entry.commitment {
            return Some(CheckpointDivergence {
                entry_index: idx,
                checkpoint_id: replayed_entry.checkpoint_id.clone(),
                occurrence: replayed_entry.occurrence,
                claimed_checkpoint_id: None,
                claimed_commitment: Some(claimed_entry.commitment.clone()),
                replayed_commitment: Some(replayed_entry.commitment.clone()),
            });
        }
    }
    if claimed.entries.len() != replayed.entries.len() {
        let entry: &ClaimedTraceEntry = if replayed.entries.len() > common_len {
            &replayed.entries[common_len]
        } else {
            &claimed.entries[common_len]
        };
        let replay_has_extra = replayed.entries.len() > common_len;
        return Some(CheckpointDivergence {
            entry_index: common_len,
            checkpoint_id: entry.checkpoint_id.clone(),
            occurrence: entry.occurrence,
            claimed_checkpoint_id: None,
            claimed_commitment: (!replay_has_extra).then(|| entry.commitment.clone()),
            replayed_commitment: replay_has_extra.then(|| entry.commitment.clone()),
        });
    }
    None
}

/// Maps a divergence to the `--raster-at` spec for its spanning routine
/// occurrence. Returns `None` for structural divergences (id-sequence or
/// length mismatch) and for routines without an implemented detour.
///
/// Committed checkpoint ids coincide with `RoutineId` names, and each
/// detourable routine commits exactly one checkpoint per invocation, so the
/// checkpoint occurrence equals the routine occurrence the detour controller
/// counts (validated by the tampered-fixture audit tests).
fn detour_spec_for(divergence: &CheckpointDivergence) -> Option<RasterDetourSpec> {
    if divergence.claimed_checkpoint_id.is_some()
        || divergence.claimed_commitment.is_none()
        || divergence.replayed_commitment.is_none()
    {
        return None;
    }
    let routine_id: RoutineId = divergence.checkpoint_id.parse().ok()?;
    if !DETOURABLE_ROUTINES.contains(&routine_id) {
        return None;
    }
    RasterDetourSpec::parse(&format!(
        "{}:{}",
        divergence.checkpoint_id, divergence.occurrence
    ))
    .ok()
}
