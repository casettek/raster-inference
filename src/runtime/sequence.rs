//! Phase-sequencing skeleton for one inference run.
//!
//! This module is the single place that knows the canonical phase/routine
//! order of the protocol:
//!
//! ```text
//! prompt.prepare
//!   → input.embedding
//!   → prefill.prepare_aux
//!   → prefill.range(s) → prefill.range_finalize (per layer)
//!   → prefill.finalize
//!   → [decode loop: decode.select_token → decode.layer_range(s)
//!      → decode.transition_finalize]
//!   → output.finalize
//! ```
//!
//! It owns terminal-checkpoint stop logic, outcome assembly, and the trace
//! lifecycle (`start`/`finish`/`abort`), and dispatches each phase to the
//! native or raster executor through [`ExecutionPolicy`] — it is itself
//! mode-agnostic. Trace scopes are established in exactly the same nesting
//! order as the pre-split monolith so committed trace artifacts are
//! byte-identical (enforced by `tests/golden_traces.rs`).

use std::ops::ControlFlow;

use anyhow::Result;
use serde_json::json;

use crate::routines::input_embedding;
use crate::runtime::checkpoints::{PhaseId, RoutineId};
use crate::runtime::executors::{native, raster, ExecutionPolicy, StepMode};
use crate::runtime::inference::{
    InferenceControls, InferenceRunOutcome, InferenceState, InputEmbeddingState,
    PausedInferenceState,
};
use crate::runtime::trace;
use crate::shared::api::input::InferenceRequest;
use crate::shared::artifacts::integrity_mode::current_raster_integrity_mode;
use crate::shared::model::runtime::LoadedModel;

/// Runs one inference request through the canonical phase sequence under the
/// given controls. This is the engine behind both the legacy
/// `run_inference_with_controls` entry point and the role APIs.
pub fn run(
    request: &InferenceRequest,
    model: &LoadedModel,
    controls: &InferenceControls,
) -> Result<InferenceRunOutcome> {
    let terminal_checkpoint = controls.terminal_checkpoint_spec()?;
    trace::with_terminal_checkpoint(terminal_checkpoint.clone(), || {
        trace::with_checkpointing_enabled(controls.commit_checkpoints, || {
            if controls.raster && controls.raster_detour.is_some() {
                anyhow::bail!("--raster and selective raster detour cannot be used together");
            }
            let mut policy = ExecutionPolicy::from_controls(controls);
            let count_raster_tiles = policy.is_full_raster() || policy.is_detour_active();
            let raster_sizing_controls = if policy.is_full_raster()
                || policy.is_detour_active()
                || controls.prefill_token_range_width.is_some()
                || controls.decode_layer_range_width.is_some()
            {
                Some(controls.raster_sizing_controls()?)
            } else {
                None
            };
            trace::start_inference_trace(&json!({
                "model_id": model.model_spec().model_id,
                "execution_mode": "deterministic",
                "det_num_spec_version": crate::shared::numerics::det_num::DET_NUM_SPEC_VERSION,
                "model_provenance": "DetNumWgt",
                "prompt_bytes_sha256": trace::sha256_hex(&request.prompt_bytes),
                "max_new_tokens": request.sampling.max_new_tokens,
                "transformer_layer_count": model.transformer_layer_count(),
                "terminal_checkpoint": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.checkpoint_id()),
                "terminal_checkpoint_occurrence": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.occurrence()),
                "commit_checkpoints": controls.commit_checkpoints,
                "tile_dsl_mode": if policy.is_full_raster() { "raster" } else { "native" },
                "raster_detour": policy.selected_detour_spec().map(|spec| spec.to_string()),
                "raster_integrity_mode": current_raster_integrity_mode().label(),
                "raster_sizing_controls": raster_sizing_controls,
            }));
            if count_raster_tiles {
                crate::dsl::start_tile_invocation_counting();
            }
            let selected_raster_detour_routine =
                policy.selected_detour_spec().map(|spec| spec.routine_id());

            let result = (|| {
                // --- prompt.prepare ---
                let (
                    prompt_preparation,
                    raster_prompt_preparation_for_embedding,
                    raster_prompt_preparation_roots_for_embedding,
                ) = if policy.is_full_raster() {
                    match raster::run_prompt_prepare(
                        request,
                        model,
                        controls,
                        raster_sizing_controls.as_ref(),
                    )? {
                        ControlFlow::Break(outcome) => return Ok(outcome),
                        ControlFlow::Continue(prepared) => (
                            prepared.prompt_preparation,
                            Some(prepared.raster_state),
                            Some(prepared.raster_roots),
                        ),
                    }
                } else {
                    let raster_core_detour = match policy.mode_for(RoutineId::PromptPrepare) {
                        // No sim detour call site exists for prompt.prepare
                        // (pre-WS3 behavior preserved: a selected sim spec
                        // fails as unimplemented at this decision point).
                        StepMode::Raster => {
                            return Err(policy.selected_detour_unimplemented_error())
                        }
                        StepMode::RasterCore => true,
                        StepMode::Native => false,
                    };
                    match native::run_prompt_prepare(request, model, controls, raster_core_detour)?
                    {
                        ControlFlow::Break(outcome) => return Ok(outcome),
                        ControlFlow::Continue(prepared) => (
                            prepared.prompt_preparation,
                            prepared.raster_checkpoint_state,
                            prepared.raster_checkpoint_roots,
                        ),
                    }
                };

                // --- input.embedding ---
                let (token_embeddings, raster_input_embedding_refs) = if policy.is_full_raster() {
                    let (token_embeddings, output) = raster::run_input_embedding_full(
                        model,
                        raster_prompt_preparation_for_embedding.as_ref(),
                        raster_prompt_preparation_roots_for_embedding.as_ref(),
                    )?;
                    (token_embeddings, Some(output))
                } else {
                    match policy.mode_for(RoutineId::InputEmbedding) {
                        StepMode::Raster => {
                            let (token_embeddings, output) = raster::run_input_embedding_detour(
                                model,
                                raster_prompt_preparation_for_embedding.as_ref(),
                                raster_prompt_preparation_roots_for_embedding.as_ref(),
                            )?;
                            (token_embeddings, Some(output))
                        }
                        StepMode::RasterCore => {
                            return Err(policy.selected_detour_unimplemented_error());
                        }
                        StepMode::Native => native::run_input_embedding(
                            &prompt_preparation.prompt_token_ids,
                            model,
                            raster_prompt_preparation_for_embedding.as_ref(),
                        )?,
                    }
                };
                let input_embedding = InputEmbeddingState {
                    prompt_preparation: prompt_preparation.clone(),
                    embedded_prompt_activations_sha256: token_embeddings.activations_sha256.clone(),
                    det_embedded_prompt_activations_sha256: token_embeddings
                        .det_activations_sha256
                        .clone(),
                };
                let input_embedding_raster_refs_for_checkpoint = if policy.is_full_raster()
                    || selected_raster_detour_routine == Some(RoutineId::InputEmbedding)
                {
                    raster_input_embedding_refs
                        .as_ref()
                        .map(|output| &output.refs)
                } else {
                    None
                };
                input_embedding::trace_input_embedding_checkpoint(
                    &prompt_preparation.prompt_token_ids,
                    &token_embeddings,
                    input_embedding_raster_refs_for_checkpoint,
                );
                if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                    return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                        terminal_checkpoint_id,
                        input_embedding,
                        transformer_state_transition: None,
                        output_decode: None,
                        raster_tile_invocations: None,
                    }));
                }

                // --- prefill: prepare_aux → range(s) → finalize ---
                let mut raster_decode_state_for_output = None;
                let prefill = if policy.is_full_raster() {
                    match raster::run_prefill(
                        model,
                        controls,
                        raster_sizing_controls,
                        policy.detour_controller_mut(),
                        prompt_preparation.prompt_token_ids.len(),
                        raster_prompt_preparation_for_embedding.as_ref(),
                        raster_input_embedding_refs.as_ref(),
                    )? {
                        ControlFlow::Break(terminal_checkpoint_id) => {
                            return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                                terminal_checkpoint_id,
                                input_embedding,
                                transformer_state_transition: None,
                                output_decode: None,
                                raster_tile_invocations: None,
                            }));
                        }
                        ControlFlow::Continue(raster_prefill) => {
                            raster_decode_state_for_output = Some(raster_prefill.decode_loop_state);
                            raster_prefill.prefill
                        }
                    }
                } else {
                    match native::run_prefill(
                        model,
                        controls,
                        raster_sizing_controls,
                        policy.detour_controller_mut(),
                        &prompt_preparation.prompt_token_ids,
                        &token_embeddings,
                        raster_input_embedding_refs.as_ref(),
                    )? {
                        ControlFlow::Break(terminal_checkpoint_id) => {
                            return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                                terminal_checkpoint_id,
                                input_embedding,
                                transformer_state_transition: None,
                                output_decode: None,
                                raster_tile_invocations: None,
                            }));
                        }
                        ControlFlow::Continue(prefill) => prefill,
                    }
                };
                let mut transformer_state_transition = prefill.transformer_state.clone();
                if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                    trace::phase_paused(PhaseId::TransformerStateTransition);
                    return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                        terminal_checkpoint_id,
                        input_embedding,
                        transformer_state_transition: Some(transformer_state_transition),
                        output_decode: None,
                        raster_tile_invocations: None,
                    }));
                }

                // --- output decode loop → output.finalize ---
                let output_decode = if let Some(raster_decode_state) =
                    raster_decode_state_for_output
                {
                    raster::run_output_decode(
                        raster_decode_state,
                        &request.sampling,
                        model,
                        raster_sizing_controls.expect("raster sizing controls should be validated"),
                    )?
                } else {
                    native::run_output_decode(
                        &prompt_preparation.prompt_token_ids,
                        &prefill,
                        &request.sampling,
                        model,
                        Some(policy.detour_controller_mut()),
                        raster_sizing_controls,
                    )?
                };
                transformer_state_transition
                    .activation_states
                    .extend(output_decode.decode_transition_states.iter().cloned());
                if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                    trace::phase_paused(PhaseId::OutputDecode);
                    return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                        terminal_checkpoint_id,
                        input_embedding,
                        transformer_state_transition: Some(transformer_state_transition),
                        output_decode: Some(output_decode),
                        raster_tile_invocations: None,
                    }));
                }
                policy.ensure_matched_if_active()?;
                Ok(InferenceRunOutcome::Completed(InferenceState {
                    input_embedding,
                    transformer_state_transition,
                    output_decode,
                    raster_tile_invocations: None,
                }))
            })();

            finish_run(result, count_raster_tiles)
        })
    })
}

/// Post-run bookkeeping shared by every outcome: attaches the raster tile
/// invocation total to the outcome and finishes (or aborts) the inference
/// trace, writing the serialized checkpoint artifact.
fn finish_run(
    result: Result<InferenceRunOutcome>,
    count_raster_tiles: bool,
) -> Result<InferenceRunOutcome> {
    let raster_tile_invocations = count_raster_tiles
        .then(crate::dsl::stop_tile_invocation_counting)
        .flatten();
    if let Some(total) = raster_tile_invocations {
        trace::raster_tile_invocations_finished(total);
    }

    let mut result = result;
    if let Some(total) = raster_tile_invocations {
        match &mut result {
            Ok(InferenceRunOutcome::Completed(state)) => {
                state.raster_tile_invocations = Some(total);
            }
            Ok(InferenceRunOutcome::Paused(state)) => {
                state.raster_tile_invocations = Some(total);
            }
            Ok(InferenceRunOutcome::RasterPromptPrepared(state)) => {
                state.raster_tile_invocations = Some(total);
            }
            Err(_) => {}
        }
    }

    match &result {
        Ok(InferenceRunOutcome::Completed(state)) => trace::finish_inference_trace(&json!({
            "input_embedding_prompt_token_ids_sha256": state.input_embedding.prompt_preparation.prompt_token_ids_sha256,
            "output_decode_generated_token_ids_sha256": state.output_decode.generated_token_ids_sha256,
            "output_decode_generated_text_sha256": trace::sha256_hex(&state.output_decode.generated_text),
            "generated_token_count": state.output_decode.generated_token_count,
            "raster_tile_invocations": state.raster_tile_invocations,
        })),
        Ok(InferenceRunOutcome::Paused(state)) => trace::finish_inference_trace(&json!({
            "terminal_checkpoint_id": state.terminal_checkpoint_id,
            "input_embedding_prompt_token_ids_sha256": state.input_embedding.prompt_preparation.prompt_token_ids_sha256,
            "raster_tile_invocations": state.raster_tile_invocations,
        })),
        Ok(InferenceRunOutcome::RasterPromptPrepared(state)) => {
            trace::finish_inference_trace(&json!({
                "terminal_checkpoint_id": state.terminal_checkpoint_id,
                "prompt_token_ids_root": state.prompt_preparation.prompt_token_ids_root,
                "prompt_token_count": state.prompt_preparation.prompt_token_count,
                "raster_tile_invocations": state.raster_tile_invocations,
            }))
        }
        Err(error) => trace::abort_inference_trace(error),
    }

    result
}

pub(crate) fn reached_terminal_checkpoint_id(controls: &InferenceControls) -> Option<String> {
    controls.terminal_checkpoint.as_ref()?;
    trace::reached_terminal_checkpoint_id()
}
