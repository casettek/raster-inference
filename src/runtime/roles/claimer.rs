//! Claimer role: native deterministic inference with checkpoint commitment.

use std::path::PathBuf;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::runtime::inference::{
    InferenceControls, InferenceRunOutcome, InferenceState, PausedInferenceState,
};
use crate::runtime::roles::ExecutionTuning;
use crate::runtime::{sequence, trace};
use crate::shared::api::input::{InferenceRequest, ModelSpec};
use crate::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::Gemma4TransformerModel;

/// Result of a claimer run: the final inference state plus the serialized
/// checkpoint trace artifact the claimer commits on-chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimerOutcome {
    pub state: InferenceState,
    /// Path of the serialized checkpoint trace artifact (written under
    /// `RASTER_TRACE_DIR`, default `raster-traces/`). Its bytes are the
    /// protocol object.
    pub trace_path: PathBuf,
}

/// Optional claimer behavior: a debug terminal checkpoint (pause the run
/// after the named checkpoint) and execution tuning. The default reproduces
/// the plain protocol run.
#[derive(Debug, Clone, Default)]
pub struct ClaimerOptions {
    /// `checkpoint-id[:occurrence]` to pause after, e.g. `prefill.finalize`
    /// or `prefill.range_finalize:2`.
    pub terminal_checkpoint: Option<String>,
    pub tuning: ExecutionTuning,
}

/// Outcome of [`run`]: completed with a trace artifact, or paused at the
/// requested terminal checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum ClaimerRunOutcome {
    Completed(ClaimerOutcome),
    Paused(PausedInferenceState),
}

/// Runs one inference request as the claimer: full-native policy, checkpoint
/// commitment on. Produces a trace artifact byte-identical to the legacy
/// native `--commit-checkpoints` path for the same request.
///
/// The protocol's claimer flow uses deterministic execution
/// (`request.execution_mode == Deterministic`); the authenticated tokenizer
/// source anchors the deterministic `prompt.prepare` checkpoint, exactly as
/// the CLI's deterministic path does.
///
/// A `Paused` outcome is only produced when `options.terminal_checkpoint`
/// requests it; an unexpected pause is an error.
pub fn run(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    raster_tokenizer_source: AuthenticatedGemmaTokenizer,
    options: &ClaimerOptions,
) -> Result<ClaimerRunOutcome> {
    let mut controls = InferenceControls {
        commit_checkpoints: true,
        terminal_checkpoint: options.terminal_checkpoint.clone(),
        raster_tokenizer_source: Some(raster_tokenizer_source),
        ..Default::default()
    };
    options.tuning.apply(&mut controls);
    match sequence::run(request, model, tokenizer, transformer_model, &controls)? {
        InferenceRunOutcome::Completed(state) => {
            let trace_path = trace::completed_trace_path()
                .context("claimer run did not produce a serialized trace artifact")?;
            Ok(ClaimerRunOutcome::Completed(ClaimerOutcome {
                state,
                trace_path,
            }))
        }
        InferenceRunOutcome::Paused(paused) => {
            if options.terminal_checkpoint.is_none() {
                anyhow::bail!(
                    "claimer run paused unexpectedly at checkpoint {}",
                    paused.terminal_checkpoint_id
                );
            }
            Ok(ClaimerRunOutcome::Paused(paused))
        }
        InferenceRunOutcome::RasterPromptPrepared(state) => anyhow::bail!(
            "claimer run stopped unexpectedly at routine boundary {}",
            state.terminal_checkpoint_id
        ),
    }
}
