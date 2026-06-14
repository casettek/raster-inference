//! Native deterministic/fp32 executor.
//!
//! Per-phase entry points called by the phase-sequencing skeleton
//! (`runtime::sequence`) when a routine executes natively. The selective
//! raster detour hooks live here too: a detour run is the native executor
//! swapping in exactly one raster routine occurrence, decided by
//! `RasterDetourController::should_detour` at the same call sites as the
//! pre-split monolith (occurrence counting is order-sensitive).
//!
//! Phase functions return `ControlFlow::Break` when the configured terminal
//! checkpoint was reached inside the phase; the skeleton assembles the
//! corresponding paused outcome.

use std::ops::ControlFlow;

use anyhow::{Context, Result};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::routines::{
    input_embedding, prefill_finalize, prefill_prepare_aux, prefill_range, prompt_prepare,
};
use crate::runtime::checkpoints::{PhaseId, RasterDetourController, RoutineId};
use crate::runtime::inference::{
    InferenceControls, InferenceRunOutcome, RasterPromptPreparedState,
};
use crate::runtime::sequence::reached_terminal_checkpoint_id;
use crate::runtime::trace;
use crate::shared::api::input::{
    InferenceRequest, PromptPreparationState, RasterPromptPreparationState, SamplingConfig,
};
use crate::shared::api::output::OutputDecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::runtime::LoadedModel;
use crate::shared::model::transformer::{ActivationSequence, TransformerPrefillResult};
use crate::trace::{trace_event, trace_scope};
use crate::RasterSizingControls;

/// Result of the native prompt-prepare phase when the run continues.
pub(crate) struct NativePromptPrepared {
    pub prompt_preparation: PromptPreparationState,
    /// Artifact store roots captured when a deterministic CPU
    /// `prompt.prepare` checkpoint was emitted (deterministic mode with an
    /// authenticated tokenizer source); consumed by raster detours of later
    /// routines.
    pub raster_checkpoint_roots: Option<RasterArtifactStoreRoots>,
    /// Raster-form prompt preparation state matching
    /// `raster_checkpoint_roots`.
    pub raster_checkpoint_state: Option<RasterPromptPreparationState>,
}

/// Native `prompt.prepare`, including the deterministic CPU checkpoint
/// emission used to anchor raster detours of later routines.
pub(crate) fn run_prompt_prepare(
    request: &InferenceRequest,
    model: &LoadedModel,
    controls: &InferenceControls,
) -> Result<ControlFlow<InferenceRunOutcome, NativePromptPrepared>> {
    let model_spec = model.model_spec();
    let tokenizer = model.tokenizer();
    let prompt_preparation = prompt_prepare::run(request, model_spec, tokenizer)?;
    let mut raster_checkpoint_roots = None;
    let mut raster_checkpoint_state = None;
    if controls.raster_tokenizer_enabled {
        let tokenizer_source = model.raster_tokenizer().context(
            "deterministic CPU prompt.prepare checkpoint requires raster tokenizer capability",
        )?;
        let prompt_checkpoint =
            prompt_prepare::format_native_prompt_as_raster_checkpoint_for_trace(
                request,
                model_spec,
                tokenizer_source,
                &prompt_preparation,
            )?;
        trace::trace_checkpoint(
            "prompt.prepare",
            &json!({
                "prompt_text": prompt_preparation.prompt_text.clone(),
                "prompt_token_ids": prompt_preparation.prompt_token_ids.clone(),
                "prompt_token_ids_sha256": prompt_preparation.prompt_token_ids_sha256.clone(),
                "sampling": request.sampling.clone(),
            }),
        );
        if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
            return Ok(ControlFlow::Break(
                InferenceRunOutcome::RasterPromptPrepared(RasterPromptPreparedState {
                    terminal_checkpoint_id,
                    prompt_preparation: prompt_checkpoint,
                    sampling: request.sampling.clone(),
                    raster_tile_invocations: None,
                }),
            ));
        }
        raster_checkpoint_roots = Some(ArtifactIo::export_store_roots());
        raster_checkpoint_state = Some(prompt_checkpoint);
    }
    Ok(ControlFlow::Continue(NativePromptPrepared {
        prompt_preparation,
        raster_checkpoint_roots,
        raster_checkpoint_state,
    }))
}

/// Native `input.embedding`. When a raster-form prompt checkpoint exists
/// (deterministic mode with tokenizer source), the equivalent raster
/// checkpoint payload is formatted for the trace alongside the native
/// activations.
pub(crate) fn run_input_embedding(
    prompt_token_ids: &[u32],
    model: &LoadedModel,
    raster_prompt_preparation: Option<&RasterPromptPreparationState>,
) -> Result<(
    ActivationSequence,
    Option<input_embedding::raster::RasterInputEmbeddingOutput>,
)> {
    let token_embeddings = input_embedding::run(prompt_token_ids, model.transformer_model())?;
    let input_embedding_output = raster_prompt_preparation
        .map(|raster_prompt_preparation| {
            let embedding_source = model.input_embedding_source()?;
            let embedding_source_ref = embedding_source.committed_source_ref()?;
            input_embedding::format_native_input_embedding_as_raster_checkpoint_for_trace(
                model.model_spec().model_id.clone(),
                embedding_source_ref.root().to_string(),
                raster_prompt_preparation,
                &token_embeddings,
            )
        })
        .transpose()?;
    Ok((token_embeddings, input_embedding_output))
}

/// Native prefill (`prefill.prepare_aux` → `prefill.range(s)` →
/// `prefill.finalize`) with selective raster detour hooks for
/// `prefill.prepare_aux`, `prefill.range`, and `prefill.finalize`.
///
/// Returns `Break(terminal_checkpoint_id)` when the configured terminal
/// checkpoint was reached inside the phase (after emitting `phase_paused`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_prefill(
    model: &LoadedModel,
    controls: &InferenceControls,
    raster_sizing_controls: Option<RasterSizingControls>,
    raster_detour_controller: &mut RasterDetourController,
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    raster_input_embedding_refs: Option<&input_embedding::raster::RasterInputEmbeddingOutput>,
) -> Result<ControlFlow<String, TransformerPrefillResult>> {
    let detour_prefill_prepare_aux =
        raster_detour_controller.should_detour(RoutineId::PrefillPrepareAux);
    let ple_inputs = if detour_prefill_prepare_aux {
        let input_embedding_output = raster_input_embedding_refs.context(
            "selective raster prefill.prepare_aux detour requires input embedding raster refs",
        )?;
        let raster_sizing =
            raster_sizing_controls.expect("raster sizing controls should be validated");
        let ple_source = model.prefill_ple_source()?;
        let ple_output = prefill_prepare_aux::run_raster(
            input_embedding_output.artifact_store_roots.clone(),
            &input_embedding_output.refs,
            &ple_source,
            raster_sizing,
        )?;
        let (artifact_store_roots, ple_input_manifest_root) = ple_output.into_parts();
        let ple_input_refs = prefill_prepare_aux::prefill_ple_input_refs_from_manifest(
            artifact_store_roots,
            ple_input_manifest_root.as_deref(),
        )?;
        prefill_prepare_aux::materialize_prefill_ple_inputs(ple_input_refs.as_ref())?
    } else {
        prefill_prepare_aux::run(
            prompt_token_ids,
            model.transformer_model(),
            token_embeddings,
        )?
    };
    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
        trace::phase_paused(PhaseId::TransformerStateTransition);
        return Ok(ControlFlow::Break(terminal_checkpoint_id));
    }
    let prefill_layer_source = if raster_detour_controller
        .selected_spec()
        .is_some_and(|spec| spec.routine_id() == RoutineId::PrefillRange)
    {
        Some(model.prefill_layer_source()?)
    } else {
        None
    };
    let prefill_layer_raster_detour =
        prefill_layer_source
            .as_ref()
            .map(|layer_source| prefill_range::PrefillLayerRasterDetour {
                layer_source,
                raster_sizing: raster_sizing_controls
                    .expect("raster sizing controls should be validated"),
            });
    let (final_hidden_states, layer_caches) = prefill_range::run_internal_with_detour(
        token_embeddings.clone_internal(),
        model.transformer_model(),
        ple_inputs.as_ref(),
        Some(raster_detour_controller),
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
        return Ok(ControlFlow::Break(terminal_checkpoint_id));
    }
    let prefill = if raster_detour_controller.should_detour(RoutineId::PrefillFinalize) {
        let raster_sizing =
            raster_sizing_controls.expect("raster sizing controls should be validated");
        let finalize_source = model.prefill_finalize_source()?;
        prefill_finalize::run_selected_raster_detour_from_native_boundary(
            &finalize_source,
            prompt_token_ids.len(),
            final_hidden_states,
            layer_caches,
            raster_sizing.projection_rows_per_tile,
        )?
    } else {
        prefill_finalize::run(
            prompt_token_ids,
            model.transformer_model(),
            final_hidden_states,
            layer_caches,
        )?
    };
    Ok(ControlFlow::Continue(prefill))
}

/// Native output-decode loop (`decode.select_token` → `decode.layer_range(s)`
/// → `decode.transition_finalize`, then `output.finalize`), with selective
/// raster detour hooks for each of those routines. Returns early with the
/// partial state when the configured terminal checkpoint is reached
/// mid-decode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_output_decode(
    prompt_token_ids: &[u32],
    initial_transformer_state: &TransformerPrefillResult,
    sampling: &SamplingConfig,
    model: &LoadedModel,
    mut detour_controller: Option<&mut RasterDetourController>,
    raster_sizing: Option<RasterSizingControls>,
) -> Result<OutputDecodeState> {
    let _trace = trace_scope("decode.run");
    let max_new_tokens = crate::runtime::pipeline::validate_sampling_config(sampling)?;
    let tokenizer = model.tokenizer();
    let mut decode_transition_states = Vec::new();
    let mut decode_state = crate::shared::api::output::DecodeState::new(
        prompt_token_ids.to_vec(),
        initial_transformer_state
            .transformer_state
            .prefill_logits
            .logits
            .clone(),
        initial_transformer_state.transformer_decode_state.clone(),
    );
    decode_state.set_internal_logits(
        initial_transformer_state
            .transformer_state
            .prefill_logits
            .clone_internal(),
    );

    loop {
        if crate::routines::decode_select_token::native::check_stop_condition(
            decode_state.generated_token_ids.len(),
            max_new_tokens,
        )
        .is_some()
        {
            let detour_finalize_output = detour_controller
                .as_deref_mut()
                .is_some_and(|controller| controller.should_detour(RoutineId::FinalizeOutput));
            trace_event("output.detokenize");
            let mut output_decode_state = if detour_finalize_output {
                let raster_sizing = raster_sizing.context(
                    "selective raster output.finalize detour requires raster sizing controls",
                )?;
                let raster_tokenizer = model.raster_tokenizer().context(
                    "selective raster output.finalize detour requires raster tokenizer capability",
                )?;
                crate::routines::output_finalize::run_selected_raster_detour_from_native_boundary(
                    decode_state,
                    raster_tokenizer,
                    raster_sizing,
                )?
            } else {
                crate::routines::output_finalize::run(decode_state, tokenizer)?
            };
            output_decode_state.decode_transition_states = decode_transition_states;
            return Ok(output_decode_state);
        }

        let detour_select_token = detour_controller
            .as_deref_mut()
            .is_some_and(|controller| controller.should_detour(RoutineId::SelectOutputToken));
        trace_event("decode.select_token");
        let next_token = if detour_select_token {
            let raster_sizing = raster_sizing.context(
                "selective raster decode.select_token detour requires raster sizing controls",
            )?;
            crate::routines::decode_select_token::run_selected_raster_detour_from_native_boundary(
                &mut decode_state,
                max_new_tokens,
                raster_sizing,
            )?
        } else {
            crate::routines::decode_select_token::run(&mut decode_state, max_new_tokens)?
                .expect("stop condition should have returned earlier")
        };

        trace_event("decode.step");
        let decode_layer_range_width = raster_sizing
            .map(|sizing| sizing.decode_layer_range_width)
            .unwrap_or(crate::InferenceControls::DEFAULT_DECODE_LAYER_RANGE_WIDTH);
        let decode_state_before_transition = decode_state.clone();
        let transformer_decode_state = std::mem::take(&mut decode_state.transformer_decode_state);
        let mut range_state =
            crate::routines::decode_layer_range::native::deterministic_tiles::init_state(
                transformer_decode_state,
                next_token,
                model.transformer_model(),
            )?;
        while !range_state.is_complete() {
            let detour_decode_layer_range = detour_controller
                .as_deref_mut()
                .is_some_and(|controller| controller.should_detour(RoutineId::DecodeLayerRange));
            if detour_decode_layer_range {
                let raster_sizing = raster_sizing.context(
                    "selective raster decode.layer_range detour requires raster sizing controls",
                )?;
                let source = model.decode_layer_range_source(format!(
                    "decode.layer_range.detour.position_{}.layer_{}",
                    range_state.position, range_state.next_layer_idx
                ))?;
                range_state =
                    crate::routines::decode_layer_range::run_selected_raster_detour_from_native_boundary(
                        range_state,
                        &source,
                        raster_sizing,
                    )?;
            } else {
                let (next_range_state, _reached_terminal) =
                    crate::routines::decode_layer_range::native::deterministic_tiles::run_range(
                        range_state,
                        model.transformer_model(),
                        decode_layer_range_width,
                    )?;
                range_state = next_range_state;
            }
        }
        let detour_decode_transition_finalize =
            detour_controller.as_deref_mut().is_some_and(|controller| {
                controller.should_detour(RoutineId::DecodeTransitionFinalize)
            });
        let decode_transition = if detour_decode_transition_finalize {
            let raster_sizing = raster_sizing.context(
                "selective raster decode.transition_finalize detour requires raster sizing controls",
            )?;
            let source = model.decode_transition_source(format!(
                "decode.transition_finalize.detour.position_{}",
                range_state.position
            ))?;
            crate::routines::decode_transition_finalize::run_selected_raster_detour_from_native_boundary(
                &decode_state_before_transition,
                range_state,
                &source,
                raster_sizing,
            )?
        } else {
            crate::routines::decode_transition_finalize::native::run(
                range_state,
                model.transformer_model(),
            )?
        };
        decode_transition_states.push(decode_transition.activation_state.clone());
        decode_state.set_internal_logits(decode_transition.prefill_logits.clone_internal());
        decode_state.transformer_decode_state = decode_transition.transformer_decode_state;
        crate::routines::decode_transition_finalize::trace_checkpoint(&decode_state)?;
        if crate::trace::reached_terminal_checkpoint_id().is_some() {
            let mut output_decode_state =
                build_current_output_decode_state(&decode_state, tokenizer)?;
            output_decode_state.decode_transition_states = decode_transition_states;
            return Ok(output_decode_state);
        }
    }
}

fn build_current_output_decode_state(
    decode_state: &crate::shared::api::output::DecodeState,
    tokenizer: &Tokenizer,
) -> Result<OutputDecodeState> {
    let generated_token_ids = decode_state.generated_token_ids.clone();
    let generated_text = crate::routines::output_finalize::native::detokenize_output_tokens(
        tokenizer,
        &generated_token_ids,
    )?;
    let generated_token_ids_sha256 =
        crate::routines::output_finalize::native::build_output_decode_commitment(
            &generated_token_ids,
        )?;

    Ok(OutputDecodeState {
        generated_token_count: generated_token_ids.len(),
        generated_token_ids,
        generated_token_ids_sha256,
        generated_text,
        stop_reason: crate::shared::api::output::OutputDecodeStopReason::MaxNewTokens,
        decode_transition_states: Vec::new(),
    })
}
