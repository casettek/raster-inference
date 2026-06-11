//! Claimer role: native deterministic inference with checkpoint commitment.

use std::path::PathBuf;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::runtime::inference::{InferenceControls, InferenceRunOutcome, InferenceState};
use crate::runtime::{sequence, trace};
use crate::shared::api::input::{InferenceRequest, ModelSpec};
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
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

/// Runs one inference request as the claimer: full-native policy, checkpoint
/// commitment on, no terminal checkpoint. Produces a trace artifact
/// byte-identical to the legacy native `--commit-checkpoints` path for the
/// same request.
///
/// The protocol's claimer flow uses deterministic execution
/// (`request.execution_mode == Deterministic`); the authenticated tokenizer
/// source anchors the deterministic `prompt.prepare` checkpoint, exactly as
/// the CLI's deterministic path does.
pub fn run(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    raster_tokenizer_source: AuthenticatedGemmaTokenizer,
) -> Result<ClaimerOutcome> {
    let controls = InferenceControls {
        commit_checkpoints: true,
        raster_tokenizer_source: Some(raster_tokenizer_source),
        ..Default::default()
    };
    match sequence::run(request, model, tokenizer, transformer_model, &controls)? {
        InferenceRunOutcome::Completed(state) => {
            let trace_path = trace::completed_trace_path()
                .context("claimer run did not produce a serialized trace artifact")?;
            Ok(ClaimerOutcome { state, trace_path })
        }
        InferenceRunOutcome::Paused(paused) => anyhow::bail!(
            "claimer run paused unexpectedly at checkpoint {}",
            paused.terminal_checkpoint_id
        ),
        InferenceRunOutcome::RasterPromptPrepared(state) => anyhow::bail!(
            "claimer run stopped unexpectedly at routine boundary {}",
            state.terminal_checkpoint_id
        ),
    }
}
