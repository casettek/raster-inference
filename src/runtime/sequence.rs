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
//! It owns terminal-checkpoint stop logic, checkpoint emission points that
//! are not inside routines, outcome assembly, and the trace lifecycle
//! (`start`/`finish`/`abort`). Trace scopes are established in exactly the
//! same nesting order as the pre-split monolith so committed trace artifacts
//! are byte-identical (enforced by `tests/golden_traces.rs`).

use anyhow::{Context, Result};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::input_embedding::raster::auth_source::AuthenticatedGemmaInputEmbeddingSource;
use crate::runtime::checkpoints::{PhaseId, RasterDetourController, RoutineId};
use crate::runtime::inference::{
    InferenceControls, InferenceRunOutcome, InferenceState, InputEmbeddingState,
    PausedInferenceState, RasterPromptPreparedState,
};
use crate::runtime::{pipeline, trace};
use crate::shared::api::input::{
    InferenceExecutionMode, InferenceRequest, ModelSpec, PromptPreparationState,
    RasterPromptPreparationState,
};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::integrity_mode::current_raster_integrity_mode;
use crate::shared::artifacts::raster_artifact_store::RasterTokenIdSequenceRef;
use crate::shared::model::transformer::Gemma4TransformerModel;
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource;
use crate::shared::raster_contracts::prefill_ple::AuthenticatedGemmaPleSource;
use crate::{
    input_embedding, prefill_finalize, prefill_prepare_aux, prefill_range, prompt_prepare,
};

/// Runs one inference request through the canonical phase sequence under the
/// given controls. This is the engine behind both the legacy
/// `run_inference_with_controls` entry point and the role APIs.
pub fn run(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    controls: &InferenceControls,
) -> Result<InferenceRunOutcome> {
    let terminal_checkpoint = controls.terminal_checkpoint_spec()?;
    trace::with_terminal_checkpoint(terminal_checkpoint.clone(), || {
        trace::with_checkpointing_enabled(controls.commit_checkpoints, || {
            if controls.raster && request.execution_mode != InferenceExecutionMode::Deterministic {
                anyhow::bail!("raster tile inference requires deterministic execution");
            }
            if controls.raster && controls.raster_detour.is_some() {
                anyhow::bail!("--raster and selective raster detour cannot be used together");
            }
            if controls.raster_detour.is_some()
                && request.execution_mode != InferenceExecutionMode::Deterministic
            {
                anyhow::bail!("selective raster detour requires deterministic execution");
            }
            let mut raster_detour_controller = RasterDetourController::new(controls.raster_detour);
            let use_raster_prefill = controls.raster;
            let use_raster_decode = controls.raster;
            let count_raster_tiles = use_raster_decode || raster_detour_controller.is_active();
            let raster_sizing_controls = if controls.raster
                || raster_detour_controller.is_active()
                || controls.prefill_token_range_width.is_some()
                || controls.decode_layer_range_width.is_some()
            {
                Some(controls.raster_sizing_controls()?)
            } else {
                None
            };
            transformer_model.validate_execution_mode(request.execution_mode)?;
            trace::start_inference_trace(&json!({
                "model_id": model.model_id,
                "execution_mode": request.execution_mode,
                "det_num_spec_version": crate::shared::numerics::det_num::DET_NUM_SPEC_VERSION,
                "model_provenance": format!("{:?}", transformer_model.provenance),
                "prompt_bytes_sha256": trace::sha256_hex(&request.prompt_bytes),
                "max_new_tokens": request.sampling.max_new_tokens,
                "transformer_layer_count": transformer_model.layers.len(),
                "terminal_checkpoint": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.checkpoint_id()),
                "terminal_checkpoint_occurrence": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.occurrence()),
                "commit_checkpoints": controls.commit_checkpoints,
                "tile_dsl_mode": if use_raster_prefill { "raster" } else { "native" },
                "raster_detour": raster_detour_controller.selected_spec().map(|spec| spec.to_string()),
                "raster_integrity_mode": current_raster_integrity_mode().label(),
                "raster_sizing_controls": raster_sizing_controls,
            }));
            if count_raster_tiles {
                crate::dsl::start_tile_invocation_counting();
            }
            let selected_raster_detour_routine = raster_detour_controller
                .selected_spec()
                .map(|spec| spec.routine_id());
            let deterministic_prompt_checkpoint = |prompt_preparation: &PromptPreparationState| {
                let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                    "deterministic CPU prompt.prepare checkpoint requires an authenticated Gemma tokenizer",
                )?;
                prompt_prepare::format_native_prompt_as_raster_checkpoint_for_trace(
                    request,
                    model,
                    tokenizer_source,
                    prompt_preparation,
                )
            };

            let result = (|| {
                let mut raster_prompt_preparation_for_embedding = None;
                let mut raster_prompt_preparation_roots_for_embedding = None;
                raster_detour_controller
                    .reject_if_selected_unsupported(RoutineId::PromptPrepare)?;
                let prompt_preparation = if use_raster_prefill {
                    let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                        "raster tile inference requires an authenticated Gemma tokenizer",
                    )?;
                    let raster_sizing_controls = raster_sizing_controls
                        .as_ref()
                        .expect("raster sizing controls should be validated");
                    let raster_prompt_preparation = prompt_prepare::run_raster(
                        request,
                        model,
                        tokenizer_source,
                        *raster_sizing_controls,
                    )?;
                    trace::trace_checkpoint(
                        "prompt.prepare",
                        &json!({
                            "prompt_bytes_root": raster_prompt_preparation.state.prompt_bytes_root.clone(),
                            "prompt_text_root": raster_prompt_preparation.state.prompt_text_root.clone(),
                            "rendered_prompt_root": raster_prompt_preparation.state.rendered_prompt_root.clone(),
                            "normalized_prompt_root": raster_prompt_preparation.state.normalized_prompt_root.clone(),
                            "prompt_token_count": raster_prompt_preparation.state.prompt_token_count,
                            "prompt_token_ids_root": raster_prompt_preparation.state.prompt_token_ids_root.clone(),
                            "sampling": request.sampling.clone(),
                        }),
                    );
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        return Ok(InferenceRunOutcome::RasterPromptPrepared(
                            RasterPromptPreparedState {
                                terminal_checkpoint_id,
                                prompt_preparation: raster_prompt_preparation.state,
                                sampling: request.sampling.clone(),
                                raster_tile_invocations: None,
                            },
                        ));
                    }
                    raster_prompt_preparation_roots_for_embedding =
                        Some(raster_prompt_preparation.artifact_store_roots.clone());
                    raster_prompt_preparation_for_embedding =
                        Some(raster_prompt_preparation.state.clone());
                    prompt_preparation_from_raster_prompt(
                        request,
                        &raster_prompt_preparation.state,
                    )?
                } else {
                    let prompt_preparation = prompt_prepare::run(request, model, tokenizer)?;
                    if request.execution_mode == InferenceExecutionMode::Deterministic
                        && controls.raster_tokenizer_source.is_some()
                    {
                        let prompt_checkpoint =
                            deterministic_prompt_checkpoint(&prompt_preparation)?;
                        trace::trace_checkpoint(
                            "prompt.prepare",
                            &json!({
                                "prompt_text": prompt_preparation.prompt_text.clone(),
                                "prompt_token_ids": prompt_preparation.prompt_token_ids.clone(),
                                "prompt_token_ids_sha256": prompt_preparation.prompt_token_ids_sha256.clone(),
                                "sampling": request.sampling.clone(),
                            }),
                        );
                        if let Some(terminal_checkpoint_id) =
                            reached_terminal_checkpoint_id(controls)
                        {
                            return Ok(InferenceRunOutcome::RasterPromptPrepared(
                                RasterPromptPreparedState {
                                    terminal_checkpoint_id,
                                    prompt_preparation: prompt_checkpoint,
                                    sampling: request.sampling.clone(),
                                    raster_tile_invocations: None,
                                },
                            ));
                        }
                        raster_prompt_preparation_roots_for_embedding =
                            Some(ArtifactIo::export_store_roots());
                        raster_prompt_preparation_for_embedding = Some(prompt_checkpoint);
                    }
                    prompt_preparation
                };
                let detour_input_embedding =
                    raster_detour_controller.should_detour(RoutineId::InputEmbedding);
                let (token_embeddings, raster_input_embedding_refs) = if use_raster_prefill {
                    let raster_prompt_preparation = raster_prompt_preparation_for_embedding
                        .as_ref()
                        .expect("raster prompt preparation should exist for raster prefill");
                    let raster_prompt_preparation_roots =
                        raster_prompt_preparation_roots_for_embedding
                            .clone()
                            .expect(
                                "raster prompt preparation roots should exist for raster prefill",
                            );
                    let embedding_source = AuthenticatedGemmaInputEmbeddingSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let input_embedding_output = input_embedding::run_raster(
                        raster_prompt_preparation_roots,
                        raster_prompt_preparation,
                        &embedding_source,
                    )?;
                    let token_embeddings =
                        input_embedding::materialize_input_embedding_refs_for_trace(
                            &input_embedding_output.refs,
                        )?;
                    (token_embeddings, Some(input_embedding_output))
                } else if detour_input_embedding {
                    let raster_prompt_preparation = raster_prompt_preparation_for_embedding
                        .as_ref()
                        .context(
                            "selective raster input.embedding detour requires an authenticated Gemma tokenizer",
                        )?;
                    let raster_prompt_preparation_roots =
                        raster_prompt_preparation_roots_for_embedding.clone().context(
                            "selective raster input.embedding detour requires prompt artifact roots",
                        )?;
                    let embedding_source = AuthenticatedGemmaInputEmbeddingSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let input_embedding_output = input_embedding::run_raster(
                        raster_prompt_preparation_roots,
                        raster_prompt_preparation,
                        &embedding_source,
                    )?;
                    let token_embeddings =
                        input_embedding::materialize_input_embedding_refs_for_trace(
                            &input_embedding_output.refs,
                        )?;
                    (token_embeddings, Some(input_embedding_output))
                } else {
                    let token_embeddings = input_embedding::run(
                        &prompt_preparation.prompt_token_ids,
                        transformer_model,
                        request.execution_mode,
                    )?;
                    let input_embedding_output = raster_prompt_preparation_for_embedding
                        .as_ref()
                        .map(|raster_prompt_preparation| {
                            let embedding_source =
                                AuthenticatedGemmaInputEmbeddingSource::from_model(
                                    model.model_id.clone(),
                                    transformer_model,
                                )?;
                            let embedding_source_ref = embedding_source.committed_source_ref()?;
                            input_embedding::format_native_input_embedding_as_raster_checkpoint_for_trace(
                                model.model_id.clone(),
                                embedding_source_ref.root().to_string(),
                                raster_prompt_preparation,
                                &token_embeddings,
                            )
                        })
                        .transpose()?;
                    (token_embeddings, input_embedding_output)
                };
                let input_embedding = InputEmbeddingState {
                    prompt_preparation: prompt_preparation.clone(),
                    embedded_prompt_activations_sha256: token_embeddings.activations_sha256.clone(),
                    det_embedded_prompt_activations_sha256: token_embeddings
                        .det_activations_sha256
                        .clone(),
                };
                if request.execution_mode != InferenceExecutionMode::Deterministic {
                    trace::trace_checkpoint(
                        "prompt.prepare",
                        &json!({
                            "prompt_text": prompt_preparation.prompt_text.clone(),
                            "prompt_token_ids": prompt_preparation.prompt_token_ids.clone(),
                            "prompt_token_ids_sha256": prompt_preparation.prompt_token_ids_sha256.clone(),
                            "embedded_prompt_activations": token_embeddings.activations.clone(),
                            "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
                            "det_embedded_prompt_activations_sha256": token_embeddings.det_activations_sha256.clone(),
                            "sampling": request.sampling.clone(),
                        }),
                    );
                }
                let input_embedding_raster_refs_for_checkpoint = if use_raster_prefill
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
                let mut raster_decode_state_for_output = None;
                let prefill = if use_raster_prefill {
                    let raster_sizing =
                        raster_sizing_controls.expect("raster sizing controls should be validated");
                    raster_detour_controller
                        .reject_if_selected_unsupported(RoutineId::PrefillPrepareAux)?;
                    let ple_source = AuthenticatedGemmaPleSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let input_embedding_output = raster_input_embedding_refs
                        .as_ref()
                        .expect("raster prefill requires raster input embedding refs");
                    let ple_output = prefill_prepare_aux::run_raster(
                        input_embedding_output.artifact_store_roots.clone(),
                        &input_embedding_output.refs,
                        &ple_source,
                        raster_sizing,
                    )?;
                    let (layer_roots, ple_input_manifest_root) = ple_output.into_parts();
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    let layer_source =
                    crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    raster_detour_controller
                        .reject_if_selected_unsupported(RoutineId::PrefillRangeFinalize)?;
                    let (layer_roots, layer_refs) = prefill_range::run_raster(
                        layer_roots,
                        &input_embedding_output.refs,
                        &layer_source,
                        ple_input_manifest_root.as_deref(),
                        raster_sizing,
                    )?;
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    let finalize_source =
                    crate::prefill_finalize::raster::auth_source::AuthenticatedGemmaPrefillFinalizeSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    raster_detour_controller
                        .reject_if_selected_unsupported(RoutineId::PrefillFinalize)?;
                    let prefill_output = prefill_finalize::run_raster(
                        layer_roots,
                        prompt_preparation.prompt_token_ids.len(),
                        &finalize_source,
                        layer_refs.final_hidden_states_ref,
                        layer_refs.layer_caches,
                        raster_sizing.projection_rows_per_tile,
                    )?;
                    let full_token_ids_ref =
                        RasterTokenIdSequenceRef::new(ArtifactIo::artifact_ref_for_root(
                            &raster_prompt_preparation_for_embedding
                                .as_ref()
                                .expect("raster prompt preparation should exist for raster prefill")
                                .prompt_token_ids_root,
                        )?)?;
                    raster_decode_state_for_output = Some(RasterDecodeLoopState::new(
                        prefill_output.artifact_store_roots.clone(),
                        Some(full_token_ids_ref),
                        prompt_preparation.prompt_token_ids.len(),
                        None,
                        0,
                        prefill_output.refs.logits_ref.clone(),
                        prefill_output.refs.logit_count,
                        prefill_output
                            .refs
                            .layer_caches
                            .iter()
                            .cloned()
                            .map(Into::into)
                            .collect(),
                        prompt_preparation.prompt_token_ids.len(),
                        prompt_preparation.prompt_token_ids.len(),
                        Some(prefill_output.refs.final_hidden_states_ref.clone()),
                    )?);
                    prefill_finalize::materialize_raster_output_refs_for_api(&prefill_output)?
                } else {
                    let detour_prefill_prepare_aux =
                        raster_detour_controller.should_detour(RoutineId::PrefillPrepareAux);
                    let ple_inputs = if detour_prefill_prepare_aux {
                        let input_embedding_output = raster_input_embedding_refs.as_ref().context(
                            "selective raster prefill.prepare_aux detour requires input embedding raster refs",
                        )?;
                        let raster_sizing = raster_sizing_controls
                            .expect("raster sizing controls should be validated");
                        let ple_source = AuthenticatedGemmaPleSource::from_model(
                            model.model_id.clone(),
                            transformer_model,
                        )?;
                        let ple_output = prefill_prepare_aux::run_raster(
                            input_embedding_output.artifact_store_roots.clone(),
                            &input_embedding_output.refs,
                            &ple_source,
                            raster_sizing,
                        )?;
                        let (artifact_store_roots, ple_input_manifest_root) =
                            ple_output.into_parts();
                        let ple_input_refs =
                            prefill_prepare_aux::prefill_ple_input_refs_from_manifest(
                                artifact_store_roots,
                                ple_input_manifest_root.as_deref(),
                            )?;
                        prefill_prepare_aux::materialize_prefill_ple_inputs(
                            ple_input_refs.as_ref(),
                        )?
                    } else {
                        prefill_prepare_aux::run(
                            &prompt_preparation.prompt_token_ids,
                            transformer_model,
                            &token_embeddings,
                            request.execution_mode,
                        )?
                    };
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    let prefill_layer_source = if raster_detour_controller
                        .selected_spec()
                        .is_some_and(|spec| spec.routine_id() == RoutineId::PrefillRange)
                    {
                        Some(AuthenticatedGemmaPrefillLayerSource::from_model(
                            model.model_id.clone(),
                            transformer_model,
                        )?)
                    } else {
                        None
                    };
                    let prefill_layer_raster_detour =
                        prefill_layer_source.as_ref().map(|layer_source| {
                            prefill_range::PrefillLayerRasterDetour {
                                layer_source,
                                raster_sizing: raster_sizing_controls
                                    .expect("raster sizing controls should be validated"),
                            }
                        });
                    let (final_hidden_states, layer_caches) =
                        prefill_range::run_with_mode_internal_with_detour(
                            token_embeddings.clone_internal(),
                            transformer_model,
                            ple_inputs.as_ref(),
                            request.execution_mode,
                            Some(&mut raster_detour_controller),
                            prefill_layer_raster_detour,
                            raster_sizing_controls
                                .map(|controls| controls.prefill_token_range_width)
                                .unwrap_or_else(|| {
                                    controls
                                        .prefill_token_range_width()
                                        .expect("prefill token range width should validate")
                                }),
                        )?;
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    if raster_detour_controller.should_detour(RoutineId::PrefillFinalize) {
                        let raster_sizing = raster_sizing_controls
                            .expect("raster sizing controls should be validated");
                        prefill_finalize::run_selected_raster_detour_from_native_boundary(
                            model.model_id.clone(),
                            transformer_model,
                            prompt_preparation.prompt_token_ids.len(),
                            final_hidden_states,
                            layer_caches,
                            raster_sizing.projection_rows_per_tile,
                        )?
                    } else {
                        prefill_finalize::run(
                            &prompt_preparation.prompt_token_ids,
                            transformer_model,
                            final_hidden_states,
                            layer_caches,
                            request.execution_mode,
                        )?
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
                let output_decode = if let Some(raster_decode_state) =
                    raster_decode_state_for_output
                {
                    let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                        "raster tile inference requires an authenticated Gemma tokenizer",
                    )?;
                    pipeline::run_output_decode_with_raster_state(
                        raster_decode_state,
                        &request.sampling,
                        tokenizer_source,
                        transformer_model,
                        raster_sizing_controls.expect("raster sizing controls should be validated"),
                    )?
                } else {
                    pipeline::run_output_decode_with_mode_and_detour(
                        &prompt_preparation.prompt_token_ids,
                        &prefill,
                        &request.sampling,
                        tokenizer,
                        controls.raster_tokenizer_source.as_ref(),
                        transformer_model,
                        request.execution_mode,
                        Some(&mut raster_detour_controller),
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
                raster_detour_controller.ensure_matched_if_active()?;
                Ok(InferenceRunOutcome::Completed(InferenceState {
                    input_embedding,
                    transformer_state_transition,
                    output_decode,
                    raster_tile_invocations: None,
                }))
            })();

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
                Ok(InferenceRunOutcome::Completed(state)) => {
                    trace::finish_inference_trace(&json!({
                        "input_embedding_prompt_token_ids_sha256": state.input_embedding.prompt_preparation.prompt_token_ids_sha256,
                        "output_decode_generated_token_ids_sha256": state.output_decode.generated_token_ids_sha256,
                        "output_decode_generated_text_sha256": trace::sha256_hex(&state.output_decode.generated_text),
                        "generated_token_count": state.output_decode.generated_token_count,
                        "raster_tile_invocations": state.raster_tile_invocations,
                    }))
                }
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
        })
    })
}

fn reached_terminal_checkpoint_id(controls: &InferenceControls) -> Option<String> {
    controls.terminal_checkpoint.as_ref()?;
    trace::reached_terminal_checkpoint_id()
}

fn prompt_preparation_from_raster_prompt(
    request: &InferenceRequest,
    raster_prompt: &RasterPromptPreparationState,
) -> Result<PromptPreparationState> {
    let prompt_text = prompt_prepare::native::decode_prompt_bytes(
        &request.prompt_bytes,
        request.text_decoding_policy,
    )?;
    let prompt_token_ids = materialize_raster_prompt_token_ids(
        &raster_prompt.prompt_token_ids_root,
        raster_prompt.prompt_token_count,
    )?;
    let prompt_token_ids_sha256 =
        prompt_prepare::native::build_prompt_commitment(&prompt_token_ids)?;

    Ok(PromptPreparationState {
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}

fn materialize_raster_prompt_token_ids(
    token_ids_root: &str,
    token_count: usize,
) -> Result<Vec<u32>> {
    let token_ids_ref =
        RasterTokenIdSequenceRef::new(ArtifactIo::artifact_ref_for_root(token_ids_root)?)?;
    if token_ids_ref.token_count() != token_count {
        anyhow::bail!(
            "raster prompt token count mismatch: root has {}, checkpoint expected {}",
            token_ids_ref.token_count(),
            token_count
        );
    }

    (0..token_count)
        .map(|token_idx| {
            ArtifactIo::read_authenticated_leaf(token_ids_ref.artifact_ref(), token_idx)?
                .deserialize()
        })
        .collect()
}
