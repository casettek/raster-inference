//! Full root-backed raster tile executor.
//!
//! Per-phase entry points called by the phase-sequencing skeleton
//! (`runtime::sequence`) when the full-raster policy is active, plus the
//! raster entry points used when the native executor detours a single
//! routine occurrence (`run_input_embedding_detour`).
//!
//! Phase functions return `ControlFlow::Break` when the configured terminal
//! checkpoint was reached inside the phase; the skeleton assembles the
//! corresponding paused outcome.

use std::ops::ControlFlow;

use anyhow::{Context, Result};
use serde_json::json;

use crate::routines::input_embedding::raster::auth_source::AuthenticatedGemmaInputEmbeddingSource;
use crate::runtime::checkpoints::{PhaseId, RasterDetourController, RoutineId};
use crate::runtime::inference::{
    InferenceControls, InferenceRunOutcome, RasterPromptPreparedState,
};
use crate::runtime::sequence::reached_terminal_checkpoint_id;
use crate::runtime::trace;
use crate::shared::api::input::{
    InferenceRequest, ModelSpec, PromptPreparationState, RasterPromptPreparationState,
    SamplingConfig,
};
use crate::shared::api::output::OutputDecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4TransformerModel, InternalActivationSequence,
    TransformerPrefillResult,
};
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::raster_contracts::prefill_ple::AuthenticatedGemmaPleSource;
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
};
use crate::trace::{trace_event, trace_scope};
use crate::RasterSizingControls;
use crate::routines::{
    input_embedding, prefill_finalize, prefill_prepare_aux, prefill_range, prompt_prepare,
};

/// Result of the raster prompt-prepare phase when the run continues.
pub(crate) struct RasterPromptPrepared {
    /// Native-form prompt preparation materialized from the raster artifacts
    /// (the canonical state carried through the rest of the run).
    pub prompt_preparation: PromptPreparationState,
    /// Raster-form prompt preparation state, consumed by raster input
    /// embedding.
    pub raster_state: RasterPromptPreparationState,
    /// Artifact store roots committed by the raster prompt preparation.
    pub raster_roots: RasterArtifactStoreRoots,
}

/// Raster `prompt.prepare`: tokenizes through the authenticated raster
/// tokenizer and commits artifact-root-form checkpoint payloads.
pub(crate) fn run_prompt_prepare(
    request: &InferenceRequest,
    model: &ModelSpec,
    controls: &InferenceControls,
    raster_sizing_controls: Option<&RasterSizingControls>,
) -> Result<ControlFlow<InferenceRunOutcome, RasterPromptPrepared>> {
    let tokenizer_source = controls
        .raster_tokenizer_source
        .as_ref()
        .context("raster tile inference requires an authenticated Gemma tokenizer")?;
    let raster_sizing_controls =
        raster_sizing_controls.expect("raster sizing controls should be validated");
    let raster_prompt_preparation =
        prompt_prepare::run_raster(request, model, tokenizer_source, *raster_sizing_controls)?;
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
        return Ok(ControlFlow::Break(
            InferenceRunOutcome::RasterPromptPrepared(RasterPromptPreparedState {
                terminal_checkpoint_id,
                prompt_preparation: raster_prompt_preparation.state,
                sampling: request.sampling.clone(),
                raster_tile_invocations: None,
            }),
        ));
    }
    let raster_roots = raster_prompt_preparation.artifact_store_roots.clone();
    let raster_state = raster_prompt_preparation.state.clone();
    let prompt_preparation =
        prompt_preparation_from_raster_prompt(request, &raster_prompt_preparation.state)?;
    Ok(ControlFlow::Continue(RasterPromptPrepared {
        prompt_preparation,
        raster_state,
        raster_roots,
    }))
}

/// Raster `input.embedding` on the full-raster path.
pub(crate) fn run_input_embedding_full(
    model: &ModelSpec,
    transformer_model: &Gemma4TransformerModel,
    raster_prompt_preparation: Option<&RasterPromptPreparationState>,
    raster_prompt_preparation_roots: Option<&RasterArtifactStoreRoots>,
) -> Result<(
    ActivationSequence,
    input_embedding::raster::RasterInputEmbeddingOutput,
)> {
    let raster_prompt_preparation = raster_prompt_preparation
        .expect("raster prompt preparation should exist for raster prefill");
    let raster_prompt_preparation_roots = raster_prompt_preparation_roots
        .cloned()
        .expect("raster prompt preparation roots should exist for raster prefill");
    run_input_embedding_from_prompt(
        model,
        transformer_model,
        raster_prompt_preparation,
        raster_prompt_preparation_roots,
    )
}

/// Raster `input.embedding` as a selective detour from the native path.
pub(crate) fn run_input_embedding_detour(
    model: &ModelSpec,
    transformer_model: &Gemma4TransformerModel,
    raster_prompt_preparation: Option<&RasterPromptPreparationState>,
    raster_prompt_preparation_roots: Option<&RasterArtifactStoreRoots>,
) -> Result<(
    ActivationSequence,
    input_embedding::raster::RasterInputEmbeddingOutput,
)> {
    let raster_prompt_preparation = raster_prompt_preparation.context(
        "selective raster input.embedding detour requires an authenticated Gemma tokenizer",
    )?;
    let raster_prompt_preparation_roots = raster_prompt_preparation_roots
        .cloned()
        .context("selective raster input.embedding detour requires prompt artifact roots")?;
    run_input_embedding_from_prompt(
        model,
        transformer_model,
        raster_prompt_preparation,
        raster_prompt_preparation_roots,
    )
}

fn run_input_embedding_from_prompt(
    model: &ModelSpec,
    transformer_model: &Gemma4TransformerModel,
    raster_prompt_preparation: &RasterPromptPreparationState,
    raster_prompt_preparation_roots: RasterArtifactStoreRoots,
) -> Result<(
    ActivationSequence,
    input_embedding::raster::RasterInputEmbeddingOutput,
)> {
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
        input_embedding::materialize_input_embedding_refs_for_trace(&input_embedding_output.refs)?;
    Ok((token_embeddings, input_embedding_output))
}

/// Result of the raster prefill phase when the run continues.
pub(crate) struct RasterPrefill {
    pub prefill: TransformerPrefillResult,
    /// Seed state for the raster output-decode loop.
    pub decode_loop_state: RasterDecodeLoopState,
}

/// Raster prefill (`prefill.prepare_aux` → `prefill.range(s)` →
/// `prefill.finalize`) on the full-raster path.
///
/// Returns `Break(terminal_checkpoint_id)` when the configured terminal
/// checkpoint was reached inside the phase (after emitting `phase_paused`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_prefill(
    model: &ModelSpec,
    transformer_model: &Gemma4TransformerModel,
    controls: &InferenceControls,
    raster_sizing_controls: Option<RasterSizingControls>,
    raster_detour_controller: &mut RasterDetourController,
    prompt_token_count: usize,
    raster_prompt_preparation: Option<&RasterPromptPreparationState>,
    raster_input_embedding_refs: Option<&input_embedding::raster::RasterInputEmbeddingOutput>,
) -> Result<ControlFlow<String, RasterPrefill>> {
    let raster_sizing = raster_sizing_controls.expect("raster sizing controls should be validated");
    raster_detour_controller.reject_if_selected_unsupported(RoutineId::PrefillPrepareAux)?;
    let ple_source =
        AuthenticatedGemmaPleSource::from_model(model.model_id.clone(), transformer_model)?;
    let input_embedding_output =
        raster_input_embedding_refs.expect("raster prefill requires raster input embedding refs");
    let ple_output = prefill_prepare_aux::run_raster(
        input_embedding_output.artifact_store_roots.clone(),
        &input_embedding_output.refs,
        &ple_source,
        raster_sizing,
    )?;
    let (layer_roots, ple_input_manifest_root) = ple_output.into_parts();
    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
        trace::phase_paused(PhaseId::TransformerStateTransition);
        return Ok(ControlFlow::Break(terminal_checkpoint_id));
    }
    let layer_source =
        crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource::from_model(
            model.model_id.clone(),
            transformer_model,
        )?;
    raster_detour_controller.reject_if_selected_unsupported(RoutineId::PrefillRangeFinalize)?;
    let (layer_roots, layer_refs) = prefill_range::run_raster(
        layer_roots,
        &input_embedding_output.refs,
        &layer_source,
        ple_input_manifest_root.as_deref(),
        raster_sizing,
    )?;
    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
        trace::phase_paused(PhaseId::TransformerStateTransition);
        return Ok(ControlFlow::Break(terminal_checkpoint_id));
    }
    let finalize_source =
        crate::routines::prefill_finalize::raster::auth_source::AuthenticatedGemmaPrefillFinalizeSource::from_model(
            model.model_id.clone(),
            transformer_model,
        )?;
    raster_detour_controller.reject_if_selected_unsupported(RoutineId::PrefillFinalize)?;
    let prefill_output = prefill_finalize::run_raster(
        layer_roots,
        prompt_token_count,
        &finalize_source,
        layer_refs.final_hidden_states_ref,
        layer_refs.layer_caches,
        raster_sizing.projection_rows_per_tile,
    )?;
    let full_token_ids_ref = RasterTokenIdSequenceRef::new(ArtifactIo::artifact_ref_for_root(
        &raster_prompt_preparation
            .expect("raster prompt preparation should exist for raster prefill")
            .prompt_token_ids_root,
    )?)?;
    let decode_loop_state = RasterDecodeLoopState::new(
        prefill_output.artifact_store_roots.clone(),
        Some(full_token_ids_ref),
        prompt_token_count,
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
        prompt_token_count,
        prompt_token_count,
        Some(prefill_output.refs.final_hidden_states_ref.clone()),
    )?;
    let prefill = prefill_finalize::materialize_raster_output_refs_for_api(&prefill_output)?;
    Ok(ControlFlow::Continue(RasterPrefill {
        prefill,
        decode_loop_state,
    }))
}

/// Raster output-decode loop (`decode.select_token` → `decode.layer_range(s)`
/// → `decode.transition_finalize`, then `output.finalize`). Returns early
/// with the partial state when the configured terminal checkpoint is reached
/// mid-decode.
pub(crate) fn run_output_decode(
    initial_decode_state: RasterDecodeLoopState,
    sampling: &SamplingConfig,
    raster_tokenizer: &AuthenticatedGemmaTokenizer,
    transformer_model: &Gemma4TransformerModel,
    raster_sizing: RasterSizingControls,
) -> Result<OutputDecodeState> {
    let _trace = trace_scope("decode.run");
    let max_new_tokens = crate::runtime::pipeline::validate_sampling_config(sampling)?;
    let mut decode_transition_state_refs = Vec::new();
    let mut decode_state = initial_decode_state;

    loop {
        if crate::routines::decode_select_token::raster::check_stop_condition(
            decode_state.generated_token_count,
            max_new_tokens,
        )
        .is_some()
        {
            trace_event("output.detokenize");
            let output_refs = crate::routines::output_finalize::run_raster(
                decode_state.clone(),
                raster_tokenizer,
                raster_sizing.output_byte_flush_bytes_per_tile,
            )?;
            let mut output_decode_state =
                crate::routines::output_finalize::materialize_output_decode_state_for_api(
                    decode_state,
                    output_refs,
                )?;
            output_decode_state.decode_transition_states = decode_transition_state_refs
                .into_iter()
                .map(|(roots, activation_ref)| {
                    materialize_activation_sequence_from_ref(&roots, &activation_ref)
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(output_decode_state);
        }

        trace_event("decode.select_token");
        let (selected_state, select_output) = crate::routines::decode_select_token::run_raster_with_sizing(
            decode_state,
            max_new_tokens,
            raster_sizing,
        )?;
        let select_output = select_output.expect("stop condition should have returned earlier");
        crate::routines::decode_select_token::trace_raster_checkpoint_from_state(
            &selected_state,
            select_output.next_token,
            max_new_tokens,
        )?;

        trace_event("decode.step");
        let source =
            crate::routines::decode_layer_range::raster::auth_source::AuthenticatedGemmaDecodeLayerRangeSource::from_model(
                format!("decode.layer_range.position_{}", selected_state.position),
                transformer_model,
            )?;
        let selected_state_for_finalize = selected_state.clone();
        let mut range_state = crate::routines::decode_layer_range::init_raster_state_from_decode_loop(
            selected_state,
            select_output.selected_token_ref,
            &source,
            raster_sizing,
        )?;
        while !range_state.is_complete() {
            range_state = crate::routines::decode_layer_range::run_raster(
                range_state,
                &source,
                raster_sizing.decode_layer_range_width,
            )?;
        }
        decode_state = crate::routines::decode_transition_finalize::run_raster(
            selected_state_for_finalize,
            range_state,
            &source,
        )?;
        if let Some(activation_ref) = decode_state.activation_state_ref.clone() {
            decode_transition_state_refs
                .push((decode_state.artifact_store_roots.clone(), activation_ref));
        }
        crate::routines::decode_transition_finalize::finalize_raster_state_for_trace(&decode_state)?;
        if crate::trace::reached_terminal_checkpoint_id().is_some() {
            let materialized_transition_states = decode_transition_state_refs
                .into_iter()
                .map(|(roots, activation_ref)| {
                    materialize_activation_sequence_from_ref(&roots, &activation_ref)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut output_decode_state = build_current_output_decode_state_from_raster_state(
                &decode_state,
                raster_tokenizer,
                raster_sizing,
            )?;
            output_decode_state.decode_transition_states = materialized_transition_states;
            return Ok(output_decode_state);
        }
    }
}

fn materialize_activation_sequence_from_ref(
    roots: &RasterArtifactStoreRoots,
    activation_ref: &RasterActivationSequenceRef,
) -> Result<ActivationSequence> {
    let (row_count, _) = activation_ref.tensor_ref().shape().sequence_metadata()?;
    let rows = (0..row_count)
        .map(|row_idx| {
            read_sequence_row_from_roots(
                roots,
                RasterSequenceRowRequest {
                    tensor_ref: activation_ref.clone(),
                    row_idx,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let det_rows = rows.iter().map(|row| row.acts()).collect::<Vec<_>>();
    Ok(ActivationSequence::from_det_internal(
        InternalActivationSequence::from_det_values_only(det_rows.clone()),
        Some(
            crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
                &det_rows,
            ),
        ),
    ))
}

fn build_current_output_decode_state_from_raster_state(
    decode_state: &RasterDecodeLoopState,
    raster_tokenizer: &AuthenticatedGemmaTokenizer,
    raster_sizing: RasterSizingControls,
) -> Result<OutputDecodeState> {
    let generated_token_ids = match decode_state.generated_token_ids_ref.as_ref() {
        Some(generated_token_ids_ref) => {
            crate::routines::output_finalize::raster::materialize_token_ids_from_roots(
                &decode_state.artifact_store_roots,
                generated_token_ids_ref,
            )?
        }
        None => Vec::new(),
    };
    crate::routines::output_finalize::raster::run_with_byte_flush_bytes_per_tile(
        &generated_token_ids,
        raster_tokenizer,
        raster_sizing.output_byte_flush_bytes_per_tile,
    )
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
