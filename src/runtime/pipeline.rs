//! Composed inference bundle helpers built from the routines.
//!
//! This module is **not** on the production inference path: the
//! phase-sequencing skeleton (`runtime::sequence`) calls the executors in
//! `runtime::executors` directly. The bundles here (`run_prefill_pass*`,
//! `decode_step*`, `run_output_decode*`, `run_transformer_state_transition*`)
//! exist for tests, benches, and golden capture that exercise prefill/decode
//! spans without full run orchestration, plus `validate_sampling_config`,
//! which the executors share.

use anyhow::Result;

use crate::shared::api::input::{PromptPreparationState, SamplingConfig};
use crate::shared::api::output::OutputDecodeState;
use crate::shared::model::runtime::LoadedModel;
use crate::shared::model::transformer::{
    ActivationSequence, TransformerDecodeState, TransformerDecodeStepResult,
    TransformerPrefillResult, TransformerStateTransitionState,
};
use crate::trace::{trace_event, trace_scope};

pub fn validate_sampling_config(sampling: &SamplingConfig) -> Result<usize> {
    const DEFAULT_TEMPERATURE: f32 = 1.0;

    if let Some(temperature) = sampling.temperature {
        if (temperature - DEFAULT_TEMPERATURE).abs() > f32::EPSILON {
            anyhow::bail!(
                "output decode only supports deterministic greedy decode; expected temperature {DEFAULT_TEMPERATURE}, got {temperature}"
            );
        }
    }
    if let Some(top_k) = sampling.top_k {
        anyhow::bail!("output decode does not support top_k yet, got {top_k}");
    }
    if let Some(top_p) = sampling.top_p {
        anyhow::bail!("output decode does not support top_p yet, got {top_p}");
    }

    Ok(sampling.max_new_tokens.unwrap_or(0))
}

pub fn run_prefill_pass(
    prompt_preparation_state: &PromptPreparationState,
    model: &LoadedModel,
    token_embeddings: &ActivationSequence,
) -> Result<TransformerPrefillResult> {
    run_prefill_pass_for_token_ids(
        &prompt_preparation_state.prompt_token_ids,
        model,
        token_embeddings,
    )
}

fn run_prefill_pass_for_token_ids(
    prompt_token_ids: &[u32],
    model: &LoadedModel,
    token_embeddings: &ActivationSequence,
) -> Result<TransformerPrefillResult> {
    let _trace = trace_scope("prefill.run");
    trace_event(format!(
        "prefill.summary tokens={} layers={}",
        prompt_token_ids.len(),
        model.transformer_layer_count()
    ));
    let ple_inputs = crate::routines::prefill_prepare_aux::run(
        prompt_token_ids,
        model.transformer_model(),
        token_embeddings,
    )?;
    trace_event("prefill.layer_stack");
    let (final_hidden_states, layer_caches) = crate::routines::prefill_range::run_internal(
        token_embeddings.clone_internal(),
        model.transformer_model(),
        ple_inputs.as_ref(),
    )?;
    crate::routines::prefill_finalize::run(
        prompt_token_ids,
        model.transformer_model(),
        final_hidden_states,
        layer_caches,
    )
}

fn embed_token_ids(token_ids: &[u32], model: &LoadedModel) -> Result<ActivationSequence> {
    trace_event("prefill.embed_tokens");
    model.embed_token_ids(token_ids)
}

pub fn run_transformer_state_transition_for_token_ids(
    token_ids: &[u32],
    model: &LoadedModel,
) -> Result<TransformerStateTransitionState> {
    let _trace = trace_scope("prefill.from_token_ids");
    let token_embeddings = embed_token_ids(token_ids, model)?;
    Ok(run_prefill_pass_for_token_ids(token_ids, model, &token_embeddings)?.transformer_state)
}

pub fn run_transformer_state_transition(
    prompt_preparation_state: &PromptPreparationState,
    model: &LoadedModel,
) -> Result<TransformerStateTransitionState> {
    let _trace = trace_scope("prefill.from_input_embedding");
    run_transformer_state_transition_for_token_ids(
        &prompt_preparation_state.prompt_token_ids,
        model,
    )
}

pub fn decode_step(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &LoadedModel,
) -> Result<TransformerDecodeStepResult> {
    let _trace = trace_scope("decode.step");
    trace_event(format!(
        "decode.summary token={} position={} layers={}",
        next_token,
        transformer_decode_state.position,
        model.transformer_layer_count()
    ));
    trace_event("decode.layer_stack");
    let mut range_state =
        crate::routines::decode_layer_range::native::deterministic_tiles::init_state(
            transformer_decode_state,
            next_token,
            model.transformer_model(),
        )?;
    while !range_state.is_complete() {
        let (next_range_state, _) =
            crate::routines::decode_layer_range::native::deterministic_tiles::run_range(
                range_state,
                model.transformer_model(),
                crate::InferenceControls::DEFAULT_DECODE_LAYER_RANGE_WIDTH,
            )?;
        range_state = next_range_state;
    }
    trace_event("decode.project_to_logits");
    crate::routines::decode_transition_finalize::native::run(range_state, model.transformer_model())
}

pub fn run_output_decode(
    prompt_token_ids: &[u32],
    initial_transformer_state: &TransformerPrefillResult,
    sampling: &SamplingConfig,
    model: &LoadedModel,
) -> Result<OutputDecodeState> {
    crate::runtime::executors::native::run_output_decode(
        prompt_token_ids,
        initial_transformer_state,
        sampling,
        model,
        None,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::validate_sampling_config;
    use crate::SamplingConfig;

    #[test]
    fn validate_sampling_config_rejects_non_default_temperature() {
        let error = validate_sampling_config(&SamplingConfig {
            max_new_tokens: Some(4),
            temperature: Some(0.7),
            top_k: None,
            top_p: None,
        })
        .expect_err("non-default temperature should fail");

        assert!(error.to_string().contains("temperature"));
    }

    #[test]
    fn validate_sampling_config_rejects_top_k_and_top_p() {
        let top_k_error = validate_sampling_config(&SamplingConfig {
            max_new_tokens: Some(4),
            temperature: Some(1.0),
            top_k: Some(5),
            top_p: None,
        })
        .expect_err("top_k should fail");
        assert!(top_k_error.to_string().contains("top_k"));

        let top_p_error = validate_sampling_config(&SamplingConfig {
            max_new_tokens: Some(4),
            temperature: Some(1.0),
            top_k: None,
            top_p: Some(0.9),
        })
        .expect_err("top_p should fail");
        assert!(top_p_error.to_string().contains("top_p"));
    }

    #[test]
    fn validate_sampling_config_returns_max_new_tokens() {
        let max = validate_sampling_config(&SamplingConfig {
            max_new_tokens: Some(7),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        })
        .expect("default sampling should validate");
        assert_eq!(max, 7);
    }
}
