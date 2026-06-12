//! Single-routine raster detour: native deterministic execution with exactly
//! one selected raster routine occurrence swapped in.
//!
//! This is the `--raster-at` execution policy as a role entry point. It is
//! used directly by a challenger who has already identified the divergent
//! routine occurrence, and internally by [`challenger::audit`] after it
//! locates a divergence. Both paths share this implementation, so the detour
//! trace stays byte-compatible with the legacy `--raster-at` output.
//!
//! [`challenger::audit`]: crate::runtime::roles::challenger::audit

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::runtime::checkpoints::RasterDetourSpec;
use crate::runtime::inference::{InferenceControls, InferenceRunOutcome, InferenceState};
use crate::runtime::roles::ExecutionTuning;
use crate::runtime::{sequence, trace};
use crate::shared::api::audit::DetourArtifact;
use crate::shared::api::input::{InferenceExecutionMode, InferenceRequest, ModelSpec};
use crate::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::Gemma4TransformerModel;

/// Result of a detour run: the final inference state plus the raster detour
/// trace artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct DetourOutcome {
    pub state: InferenceState,
    pub artifact: DetourArtifact,
}

/// Re-runs one inference request with a single-detour policy: native
/// deterministic execution everywhere except the selected routine
/// occurrence, which executes at raster (tile) level. Checkpoint commitment
/// is on; the serialized trace is the dispute's detour artifact.
pub fn run(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    raster_tokenizer_source: AuthenticatedGemmaTokenizer,
    spec: RasterDetourSpec,
    tuning: &ExecutionTuning,
) -> Result<DetourOutcome> {
    if request.execution_mode != InferenceExecutionMode::Deterministic {
        anyhow::bail!("raster detour requires deterministic execution");
    }
    let mut controls = InferenceControls {
        commit_checkpoints: true,
        raster_detour: Some(spec),
        raster_tokenizer_source: Some(raster_tokenizer_source),
        ..Default::default()
    };
    tuning.apply(&mut controls);
    let state = match sequence::run(request, model, tokenizer, transformer_model, &controls)? {
        InferenceRunOutcome::Completed(state) => state,
        InferenceRunOutcome::Paused(paused) => anyhow::bail!(
            "raster detour paused unexpectedly at checkpoint {}",
            paused.terminal_checkpoint_id
        ),
        InferenceRunOutcome::RasterPromptPrepared(state) => anyhow::bail!(
            "raster detour stopped unexpectedly at routine boundary {}",
            state.terminal_checkpoint_id
        ),
    };
    let trace_path = trace::completed_trace_path()
        .context("raster detour did not produce a trace artifact")?;
    Ok(DetourOutcome {
        state,
        artifact: DetourArtifact { spec, trace_path },
    })
}
