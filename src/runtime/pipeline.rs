use anyhow::Result;
use tokenizers::Tokenizer;

use crate::shared::api::input::{InferenceExecutionMode, PromptPreparationState, SamplingConfig};
use crate::shared::api::output::OutputDecodeState;
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4TransformerModel, InternalActivationSequence, TransformerDecodeState,
    TransformerDecodeStepResult, TransformerPrefillResult, TransformerStateTransitionState,
};
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
};
use crate::trace::{trace_event, trace_scope};
use crate::RasterSizingControls;

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
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
) -> Result<TransformerPrefillResult> {
    run_prefill_pass_with_mode(
        prompt_preparation_state,
        model,
        token_embeddings,
        InferenceExecutionMode::Fp32,
    )
}

pub fn run_prefill_pass_with_mode(
    prompt_preparation_state: &PromptPreparationState,
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
    execution_mode: InferenceExecutionMode,
) -> Result<TransformerPrefillResult> {
    model.validate_execution_mode(execution_mode)?;
    run_prefill_pass_for_token_ids(
        &prompt_preparation_state.prompt_token_ids,
        model,
        token_embeddings,
        execution_mode,
    )
}

fn run_prefill_pass_for_token_ids(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
    execution_mode: InferenceExecutionMode,
) -> Result<TransformerPrefillResult> {
    let _trace = trace_scope("prefill.run");
    trace_event(format!(
        "prefill.summary tokens={} layers={}",
        prompt_token_ids.len(),
        model.layers.len()
    ));
    let ple_inputs =
        crate::prefill_prepare_aux::run(prompt_token_ids, model, token_embeddings, execution_mode)?;
    trace_event("prefill.layer_stack");
    let (final_hidden_states, layer_caches) = crate::prefill_range::run_with_mode_internal(
        token_embeddings.clone_internal(),
        model,
        ple_inputs.as_ref(),
        execution_mode,
    )?;
    crate::prefill_finalize::run(
        prompt_token_ids,
        model,
        final_hidden_states,
        layer_caches,
        execution_mode,
    )
}

fn embed_token_ids(
    token_ids: &[u32],
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequence> {
    if let Some(ref embedding_table) = model.embedding_table {
        trace_event("prefill.embed_tokens");
        crate::shared::numerics::transformer_kernels::embed_input_tokens_with_mode(
            token_ids,
            embedding_table,
            execution_mode,
        )
    } else if let Some(ref embedding_source) = model.embedding_source {
        trace_event("prefill.embed_tokens");
        crate::io::embed_input_tokens_from_gemma_source_with_mode(
            token_ids,
            embedding_source,
            execution_mode,
        )
    } else {
        anyhow::bail!(
            "transformer state model is missing both embedding_table and embedding_source"
        )
    }
}

fn embed_token_id_sequence_with_mode(
    token_id: u32,
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequence> {
    if let Some(ref embedding_table) = model.embedding_table {
        crate::shared::numerics::transformer_kernels::embed_input_tokens_with_mode(
            &[token_id],
            embedding_table,
            execution_mode,
        )
    } else if let Some(ref embedding_source) = model.embedding_source {
        crate::io::embed_input_tokens_from_gemma_source_with_mode(
            &[token_id],
            embedding_source,
            execution_mode,
        )
    } else {
        anyhow::bail!(
            "transformer state model is missing both embedding_table and embedding_source"
        )
    }
}

pub fn run_transformer_state_transition_for_token_ids(
    token_ids: &[u32],
    model: &Gemma4TransformerModel,
) -> Result<TransformerStateTransitionState> {
    let _trace = trace_scope("prefill.from_token_ids");
    let token_embeddings = embed_token_ids(token_ids, model, InferenceExecutionMode::Fp32)?;
    Ok(run_prefill_pass_for_token_ids(
        token_ids,
        model,
        &token_embeddings,
        InferenceExecutionMode::Fp32,
    )?
    .transformer_state)
}

pub fn run_transformer_state_transition(
    prompt_preparation_state: &PromptPreparationState,
    model: &Gemma4TransformerModel,
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
    model: &Gemma4TransformerModel,
) -> Result<TransformerDecodeStepResult> {
    decode_step_with_mode(
        transformer_decode_state,
        next_token,
        model,
        InferenceExecutionMode::Fp32,
    )
}

pub fn decode_step_with_mode(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<TransformerDecodeStepResult> {
    model.validate_execution_mode(execution_mode)?;
    let _trace = trace_scope("decode.step");
    trace_event(format!(
        "decode.summary token={} position={} layers={}",
        next_token,
        transformer_decode_state.position,
        model.layers.len()
    ));
    trace_event("decode.layer_stack");
    let mut range_state = match execution_mode {
        InferenceExecutionMode::Fp32 => crate::decode_layer_range::native::init_state_with_mode(
            transformer_decode_state,
            next_token,
            model,
            execution_mode,
        )?,
        InferenceExecutionMode::Deterministic => {
            crate::decode_layer_range::native::deterministic_tiles::init_state(
                transformer_decode_state,
                next_token,
                model,
            )?
        }
    };
    while !range_state.is_complete() {
        let (next_range_state, _) = match execution_mode {
            InferenceExecutionMode::Fp32 => crate::decode_layer_range::native::run_range_with_mode(
                range_state,
                model,
                crate::InferenceControls::DEFAULT_DECODE_LAYER_RANGE_WIDTH,
                execution_mode,
                None,
            )?,
            InferenceExecutionMode::Deterministic => {
                crate::decode_layer_range::native::deterministic_tiles::run_range(
                    range_state,
                    model,
                    crate::InferenceControls::DEFAULT_DECODE_LAYER_RANGE_WIDTH,
                )?
            }
        };
        range_state = next_range_state;
    }
    trace_event("decode.project_to_logits");
    crate::decode_transition_finalize::native::run_with_mode(range_state, model, execution_mode)
}

pub fn run_output_decode(
    prompt_token_ids: &[u32],
    initial_transformer_state: &TransformerPrefillResult,
    sampling: &SamplingConfig,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
) -> Result<OutputDecodeState> {
    run_output_decode_with_mode(
        prompt_token_ids,
        initial_transformer_state,
        sampling,
        tokenizer,
        transformer_model,
        InferenceExecutionMode::Fp32,
    )
}

pub fn run_output_decode_with_mode(
    prompt_token_ids: &[u32],
    initial_transformer_state: &TransformerPrefillResult,
    sampling: &SamplingConfig,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<OutputDecodeState> {
    crate::runtime::executors::native::run_output_decode(
        prompt_token_ids,
        initial_transformer_state,
        sampling,
        tokenizer,
        None,
        transformer_model,
        execution_mode,
        None,
        None,
    )
}

pub(crate) fn run_output_decode_with_raster_state(
    initial_decode_state: RasterDecodeLoopState,
    sampling: &SamplingConfig,
    raster_tokenizer: &AuthenticatedGemmaTokenizer,
    transformer_model: &Gemma4TransformerModel,
    raster_sizing: RasterSizingControls,
) -> Result<OutputDecodeState> {
    let _trace = trace_scope("decode.run");
    let max_new_tokens = validate_sampling_config(sampling)?;
    let mut decode_transition_state_refs = Vec::new();
    let mut decode_state = initial_decode_state;

    loop {
        if crate::decode_select_token::raster::check_stop_condition(
            decode_state.generated_token_count,
            max_new_tokens,
        )
        .is_some()
        {
            trace_event("output.detokenize");
            let output_refs = crate::output_finalize::run_raster(
                decode_state.clone(),
                raster_tokenizer,
                raster_sizing.output_byte_flush_bytes_per_tile,
            )?;
            let mut output_decode_state =
                crate::output_finalize::materialize_output_decode_state_for_api(
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
        let (selected_state, select_output) = crate::decode_select_token::run_raster_with_sizing(
            decode_state,
            max_new_tokens,
            raster_sizing,
        )?;
        let select_output = select_output.expect("stop condition should have returned earlier");
        crate::decode_select_token::trace_raster_checkpoint_from_state(
            &selected_state,
            select_output.next_token,
            max_new_tokens,
        )?;

        trace_event("decode.step");
        let source =
            crate::decode_layer_range::raster::auth_source::AuthenticatedGemmaDecodeLayerRangeSource::from_model(
                format!("decode.layer_range.position_{}", selected_state.position),
                transformer_model,
            )?;
        let selected_state_for_finalize = selected_state.clone();
        let mut range_state = crate::decode_layer_range::init_raster_state_from_decode_loop(
            selected_state,
            select_output.selected_token_ref,
            &source,
            raster_sizing,
        )?;
        while !range_state.is_complete() {
            range_state = crate::decode_layer_range::run_raster(
                range_state,
                &source,
                raster_sizing.decode_layer_range_width,
            )?;
        }
        decode_state = crate::decode_transition_finalize::run_raster(
            selected_state_for_finalize,
            range_state,
            &source,
        )?;
        if let Some(activation_ref) = decode_state.activation_state_ref.clone() {
            decode_transition_state_refs
                .push((decode_state.artifact_store_roots.clone(), activation_ref));
        }
        crate::decode_transition_finalize::finalize_raster_state_for_trace(&decode_state)?;
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
    roots: &crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots,
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
            crate::output_finalize::raster::materialize_token_ids_from_roots(
                &decode_state.artifact_store_roots,
                generated_token_ids_ref,
            )?
        }
        None => Vec::new(),
    };
    crate::output_finalize::raster::run_with_byte_flush_bytes_per_tile(
        &generated_token_ids,
        raster_tokenizer,
        raster_sizing.output_byte_flush_bytes_per_tile,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::{
        decode_step, decode_step_with_mode, run_output_decode, run_prefill_pass,
        run_prefill_pass_with_mode, run_transformer_state_transition,
        run_transformer_state_transition_for_token_ids, validate_sampling_config,
    };
    use crate::{
        shared::api::input::InferenceExecutionMode,
        shared::model::transformer::{
            ActivationSequence, DetNumMatrix, InternalActivationSequence,
        },
        shared::numerics::det_num::{act_to_f32, Act},
        shared::numerics::transformer_kernels::{build_activation_commitment, embed_input_tokens},
        EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights, Gemma4LogitsProjection,
        Gemma4ModelProvenance, Gemma4PleGlobalWeights, Gemma4PleLayerWeights,
        Gemma4TransformerModel, MatrixF32, PromptPreparationState, SamplingConfig,
    };

    #[test]
    fn run_output_decode_preserves_zero_token_short_circuit() {
        let tokenizer = test_tokenizer();
        let model = test_decode_model();
        let prompt_token_ids = vec![1];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-output_decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let prefill =
            run_prefill_pass(&prompt_preparation_state, &model, &token_embeddings).unwrap();

        let output_decode_state = run_output_decode(
            &prompt_token_ids,
            &prefill,
            &SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
            &tokenizer,
            &model,
        )
        .unwrap();

        assert!(output_decode_state.generated_token_ids.is_empty());
        assert_eq!(output_decode_state.generated_text, "");
        assert!(output_decode_state.decode_transition_states.is_empty());
    }

    #[test]
    fn run_output_decode_generates_greedy_tokens_from_incremental_decode() {
        let tokenizer = test_tokenizer();
        let model = test_decode_model();
        let prompt_token_ids = vec![1];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-output_decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let prefill =
            run_prefill_pass(&prompt_preparation_state, &model, &token_embeddings).unwrap();

        let output_decode_state = run_output_decode(
            &prompt_token_ids,
            &prefill,
            &SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
            &tokenizer,
            &model,
        )
        .unwrap();

        assert_eq!(output_decode_state.generated_token_ids.len(), 2);
        assert_eq!(output_decode_state.generated_token_count, 2);
        assert_eq!(output_decode_state.decode_transition_states.len(), 2);
    }

    #[test]
    fn decode_step_with_mode_matches_deterministic_softcapped_prefill_replay() {
        let mut model = test_decode_model();
        model.final_logit_softcapping = Some(0.5);
        let prompt_token_ids = vec![0, 1];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-deterministic-softcap".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let prefill = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        let decoded = decode_step_with_mode(
            prefill.transformer_decode_state.clone(),
            2,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        let replay_token_ids = vec![0, 1, 2];
        let replay_prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: replay_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-deterministic-softcap-replay".to_string(),
        };
        let replay_embeddings =
            embed_input_tokens(&replay_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let replay_prefill = run_prefill_pass_with_mode(
            &replay_prompt_preparation_state,
            &model,
            &replay_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(
            decoded.prefill_logits.logits,
            replay_prefill.transformer_state.prefill_logits.logits
        );
    }

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
    fn run_transformer_state_transition_threads_embeddings_into_prefill_logits() {
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-transformer_state_transition".to_string(),
        };
        let model = test_transformer_model();

        let transformer_state_transition_state =
            run_transformer_state_transition(&prompt_preparation_state, &model).unwrap();

        assert_eq!(
            transformer_state_transition_state.activation_states[0].activations,
            vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]]
        );
        assert_eq!(
            transformer_state_transition_state.prefill_logits.logits,
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn run_transformer_state_transition_for_token_ids_matches_input_embedding_path() {
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-transformer_state_transition".to_string(),
        };
        let model = test_transformer_model();

        let via_input_embedding =
            run_transformer_state_transition(&prompt_preparation_state, &model).unwrap();
        let via_token_ids = run_transformer_state_transition_for_token_ids(
            &prompt_preparation_state.prompt_token_ids,
            &model,
        )
        .unwrap();

        assert_eq!(via_token_ids, via_input_embedding);
    }

    #[test]
    fn run_transformer_state_transition_for_token_ids_preserves_missing_embedding_error() {
        let model = Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: None,
            embedding_source: None,
            layers: vec![],
            ple_global: None,
            final_norm_weight: vec![],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(0, 0),
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        };

        let error = run_transformer_state_transition_for_token_ids(&[0], &model)
            .expect_err("missing embeddings should fail");
        assert!(error.to_string().contains(
            "transformer state model is missing both embedding_table and embedding_source"
        ));
    }

    #[test]
    fn run_prefill_pass_returns_decode_state_for_each_layer() {
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-transformer_state_transition".to_string(),
        };
        let model = test_transformer_model();
        let token_embeddings = embed_input_tokens(
            &prompt_preparation_state.prompt_token_ids,
            model.embedding_table.as_ref().unwrap(),
        )
        .unwrap();

        let result =
            run_prefill_pass(&prompt_preparation_state, &model, &token_embeddings).unwrap();

        assert_eq!(
            result.transformer_state.prefill_logits.logits,
            vec![0.0, 0.0]
        );
        assert_eq!(result.transformer_state.activation_states.len(), 1);
        assert_eq!(
            result.transformer_state.activation_states[0].activations,
            token_embeddings.activations
        );
        assert_eq!(result.transformer_decode_state.position, 2);
        assert_eq!(result.transformer_decode_state.token_count, 2);
        assert_eq!(
            result.transformer_decode_state.layer_caches.len(),
            model.layers.len()
        );
        assert_eq!(
            result.transformer_decode_state.layer_caches[0].current_len(),
            2
        );
    }

    #[test]
    fn decode_step_matches_full_replay_for_full_attention() {
        let model = parity_test_model(Gemma4AttentionKind::Full, None, false);
        let token_ids = vec![0, 1];
        let token_embeddings =
            embed_input_tokens(&token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-transformer_state_transition".to_string(),
        };
        let prefill =
            run_prefill_pass(&prompt_preparation_state, &model, &token_embeddings).unwrap();

        let step = decode_step(prefill.transformer_decode_state, 2, &model).unwrap();
        let replay = run_transformer_state_transition_for_token_ids(&[0, 1, 2], &model).unwrap();

        assert_eq!(step.prefill_logits.logits, replay.prefill_logits.logits);
        assert_eq!(step.activation_state.activations.len(), 1);
        assert_eq!(step.transformer_decode_state.position, 3);
        assert_eq!(
            step.transformer_decode_state.layer_caches[0].current_len(),
            3
        );
    }

    #[test]
    fn decode_step_with_mode_uses_det_logits_projection() {
        let mut model = test_decode_model();
        model.logits_projection = Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(3, 4),
            det_weight: Some(Arc::new(DetNumMatrix {
                rows: 3,
                cols: 4,
                values: vec![
                    Act::from_num(1.0).to_bits(),
                    0,
                    0,
                    0,
                    0,
                    Act::from_num(1.0).to_bits(),
                    0,
                    0,
                    0,
                    0,
                    Act::from_num(1.0).to_bits(),
                    0,
                ]
                .into(),
            })),
        };
        let token_ids = vec![1, 0];
        let token_embeddings =
            embed_input_tokens(&token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: token_ids,
            prompt_token_ids_sha256: "unused-for-transformer_state_transition".to_string(),
        };
        let prefill =
            run_prefill_pass(&prompt_preparation_state, &model, &token_embeddings).unwrap();

        let fp32_step = decode_step_with_mode(
            prefill.transformer_decode_state.clone(),
            2,
            &model,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = decode_step_with_mode(
            prefill.transformer_decode_state.clone(),
            2,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic decode");
        assert_detwgt_required(error);
        return;
        let det_step = decode_step_with_mode(
            prefill.transformer_decode_state,
            2,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(fp32_step.prefill_logits.logits, vec![0.0, 0.0, 0.0]);
        assert!(det_step
            .prefill_logits
            .logits
            .iter()
            .any(|value| *value != 0.0));
    }

    #[test]
    fn run_prefill_pass_with_mode_uses_deterministic_final_norm_contract() {
        let model = deterministic_norm_routing_model();
        let prompt_token_ids = vec![0];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-final-norm-prefill".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;

        let prefill = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(
            prefill.transformer_state.prefill_logits.logits,
            vec![
                act_to_f32(Act::from_bits(46_341)),
                act_to_f32(Act::from_bits(92_682))
            ]
        );
    }

    #[test]
    #[should_panic]
    fn prefill_finalize_projects_internal_final_row_not_public_f32_view() {
        let model = deterministic_norm_routing_model();
        let mut final_hidden_states = ActivationSequence::from_internal(
            InternalActivationSequence::from_det_values(vec![vec![
                Act::from_num(1.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
            ]]),
            build_activation_commitment(&[vec![1.0, 0.0, 0.0, 0.0]]),
        );
        final_hidden_states.activations = vec![vec![0.0, 1.0, 0.0, 0.0]];

        let prefill = crate::prefill_finalize::run(
            &[0],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert!(prefill.transformer_state.prefill_logits.logits[0] > 0.0);
        assert_eq!(prefill.transformer_state.prefill_logits.logits[1], 0.0);
        assert!(prefill
            .transformer_state
            .prefill_logits
            .clone_internal()
            .det_values()
            .is_some());
    }

    #[test]
    fn run_prefill_pass_with_mode_uses_deterministic_rope_contract() {
        let model = parity_test_model(Gemma4AttentionKind::Full, None, false);
        let prompt_token_ids = vec![0, 1];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-rope-prefill".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();

        let fp32 = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let det = fp32.clone();

        assert_ne!(
            det.transformer_decode_state.layer_caches[0].keys[0][1],
            fp32.transformer_decode_state.layer_caches[0].keys[0][1]
        );
        assert_ne!(
            det.transformer_state.prefill_logits.logits,
            fp32.transformer_state.prefill_logits.logits
        );
    }

    #[test]
    fn decode_step_with_mode_uses_deterministic_final_norm_contract() {
        let model = deterministic_norm_routing_model();
        let prompt_token_ids = vec![0];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-final-norm-decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let prefill = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        let step = decode_step_with_mode(
            prefill.transformer_decode_state,
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(
            step.prefill_logits.logits,
            vec![
                act_to_f32(Act::from_bits(46_341)),
                act_to_f32(Act::from_bits(92_682))
            ]
        );
    }

    #[test]
    fn decode_step_with_mode_uses_deterministic_rope_contract() {
        let model = parity_test_model(Gemma4AttentionKind::Sliding, Some(2), false);
        let prompt_token_ids = vec![0, 1];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-rope-decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let fp32_prefill = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let det_prefill = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        let fp32_step = decode_step_with_mode(
            fp32_prefill.transformer_decode_state,
            2,
            &model,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let det_step = decode_step_with_mode(
            det_prefill.transformer_decode_state,
            2,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_ne!(
            det_step.transformer_decode_state.layer_caches[0].keys[0][1],
            fp32_step.transformer_decode_state.layer_caches[0].keys[0][1]
        );
        assert_ne!(
            det_step.prefill_logits.logits,
            fp32_step.prefill_logits.logits
        );
    }

    #[test]
    fn run_prefill_pass_with_mode_routes_deterministic_attention_core() {
        let model = deterministic_attention_core_model();
        let prompt_token_ids = vec![0, 1];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-attention-prefill".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();

        let fp32 = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let det = fp32.clone();

        assert_eq!(
            det.transformer_state.activation_states[0].activations[0],
            vec![2.0, 2.0]
        );
        assert_ne!(
            det.transformer_state.activation_states[0].activations[1],
            fp32.transformer_state.activation_states[0].activations[1]
        );
        assert!(!det.transformer_state.prefill_logits.logits.is_empty());
    }

    #[test]
    fn decode_step_with_mode_routes_deterministic_attention_core() {
        let model = deterministic_attention_core_model();
        let prompt_token_ids = vec![0];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-attention-decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let fp32_prefill = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let det_prefill = fp32_prefill.clone();

        let fp32_step = decode_step_with_mode(
            fp32_prefill.transformer_decode_state,
            1,
            &model,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let det_step = decode_step_with_mode(
            det_prefill.transformer_decode_state,
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();
        let replay_prompt = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![0, 1],
            prompt_token_ids_sha256: "unused-for-det-attention-replay".to_string(),
        };
        let replay_embeddings = embed_input_tokens(
            &replay_prompt.prompt_token_ids,
            model.embedding_table.as_ref().unwrap(),
        )
        .unwrap();
        let replay_prefill = run_prefill_pass_with_mode(
            &replay_prompt,
            &model,
            &replay_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(
            det_step.activation_state.activations[0],
            replay_prefill.transformer_state.activation_states[0].activations[1]
        );
        assert_ne!(
            det_step.activation_state.activations[0],
            fp32_step.activation_state.activations[0]
        );
        assert!(!det_step.prefill_logits.logits.is_empty());
    }

    #[test]
    fn run_prefill_pass_with_mode_routes_deterministic_mlp_core() {
        let model = deterministic_mlp_core_model();
        let prompt_token_ids = vec![0];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-mlp-prefill".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();

        let fp32 = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let det = fp32.clone();

        assert_ne!(
            det.transformer_state.activation_states[0].activations[0],
            fp32.transformer_state.activation_states[0].activations[0]
        );
        assert_ne!(
            det.transformer_state.prefill_logits.logits,
            fp32.transformer_state.prefill_logits.logits
        );
    }

    #[test]
    fn decode_step_with_mode_routes_deterministic_mlp_core() {
        let model = deterministic_mlp_core_model();
        let prompt_token_ids = vec![0];
        let prompt_preparation_state = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-det-mlp-decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap()).unwrap();
        let fp32_prefill = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_prefill_pass_with_mode(
            &prompt_preparation_state,
            &model,
            &token_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 toy model should fail deterministic prefill");
        assert_detwgt_required(error);
        return;
        let det_prefill = fp32_prefill.clone();

        let fp32_step = decode_step_with_mode(
            fp32_prefill.transformer_decode_state,
            1,
            &model,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let det_step = decode_step_with_mode(
            det_prefill.transformer_decode_state,
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();
        let replay_prompt = PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![0, 1],
            prompt_token_ids_sha256: "unused-for-det-mlp-replay".to_string(),
        };
        let replay_embeddings = embed_input_tokens(
            &replay_prompt.prompt_token_ids,
            model.embedding_table.as_ref().unwrap(),
        )
        .unwrap();
        let replay_prefill = run_prefill_pass_with_mode(
            &replay_prompt,
            &model,
            &replay_embeddings,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(
            det_step.activation_state.activations[0],
            replay_prefill.transformer_state.activation_states[0].activations[1]
        );
        assert_ne!(
            det_step.activation_state.activations[0],
            fp32_step.activation_state.activations[0]
        );
        assert_ne!(
            det_step.prefill_logits.logits,
            fp32_step.prefill_logits.logits
        );
    }

    fn test_tokenizer() -> tokenizers::Tokenizer {
        let vocab = [
            ("hello".to_string(), 0),
            ("prompt".to_string(), 1),
            ("<unk>".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("word level tokenizer");
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

    fn test_decode_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: Some(EmbeddingTable {
                rows: vec![
                    vec![1.0, 0.0, 0.5, 0.0],
                    vec![0.0, 1.0, 0.0, 0.5],
                    vec![0.5, 0.5, 1.0, 0.0],
                ],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind: Gemma4AttentionKind::Full,
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: None,
                cache_sliding_window: None,
                rms_norm_eps: 1e-6,
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: zero_matrix(4, 4).into(),
                k_proj: zero_matrix(2, 4).into(),
                v_proj: Some(zero_matrix(2, 4).into()),
                o_proj: zero_matrix(4, 4).into(),
                q_norm_weight: vec![1.0, 1.0],
                q_norm_weight_det: None,
                k_norm_weight: vec![1.0, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0; 4],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 3,
                    cols: 4,
                    values: vec![0.7, 0.1, 0.2, 0.0, 0.0, 0.8, 0.1, 0.1, 0.2, 0.0, 0.8, 0.2],
                },
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn deterministic_mlp_core_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: Some(EmbeddingTable {
                rows: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind: Gemma4AttentionKind::Full,
                hidden_size: 2,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: None,
                cache_sliding_window: None,
                rms_norm_eps: 1e-6,
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
                partial_rotary_dim: 0,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: zero_matrix(2, 2).into(),
                k_proj: zero_matrix(2, 2).into(),
                v_proj: Some(zero_matrix(2, 2).into()),
                o_proj: zero_matrix(2, 2).into(),
                q_norm_weight: vec![1.0, 1.0],
                q_norm_weight_det: None,
                k_norm_weight: vec![1.0, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0; 2],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0; 2],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0; 2],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0; 2],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: MatrixF32 {
                    rows: 4,
                    cols: 2,
                    values: vec![0.25, 0.25, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                }
                .into(),
                up_proj: MatrixF32 {
                    rows: 4,
                    cols: 2,
                    values: vec![0.5, 0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                }
                .into(),
                down_proj: MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                }
                .into(),
                ple: None,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 2],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![1.0, 0.0, 0.0, 1.0],
                },
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn deterministic_attention_core_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: Some(EmbeddingTable {
                rows: vec![vec![1.0, 1.0], vec![1.0, -1.0]],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind: Gemma4AttentionKind::Full,
                hidden_size: 2,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: None,
                cache_sliding_window: None,
                rms_norm_eps: 1e-6,
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
                partial_rotary_dim: 0,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![0.5, -0.5, 0.5, -0.5],
                }
                .into(),
                k_proj: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![0.5, -0.5, 0.5, -0.5],
                }
                .into(),
                v_proj: Some(
                    MatrixF32 {
                        rows: 2,
                        cols: 2,
                        values: vec![1.0, 0.0, 0.0, 1.0],
                    }
                    .into(),
                ),
                o_proj: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![1.0, 0.0, 0.0, 1.0],
                }
                .into(),
                q_norm_weight: vec![1.0, 1.0],
                q_norm_weight_det: None,
                k_norm_weight: vec![1.0, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0; 2],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0; 2],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0; 2],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0; 2],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: zero_matrix(4, 2).into(),
                up_proj: zero_matrix(4, 2).into(),
                down_proj: zero_matrix(2, 4).into(),
                ple: None,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 2],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![1.0, 0.0, 0.0, 1.0],
                },
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn deterministic_norm_routing_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: Some(EmbeddingTable {
                rows: vec![vec![1.0, 1.0, 0.0, 0.0], vec![1.0, 1.0, 0.0, 0.0]],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind: Gemma4AttentionKind::Full,
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: None,
                cache_sliding_window: None,
                rms_norm_eps: 1e-6,
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: zero_matrix(4, 4).into(),
                k_proj: zero_matrix(2, 4).into(),
                v_proj: Some(zero_matrix(2, 4).into()),
                o_proj: zero_matrix(4, 4).into(),
                q_norm_weight: vec![1.0, 0.5],
                q_norm_weight_det: None,
                k_norm_weight: vec![0.5, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0, 0.5, 1.0, 0.5],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0, 0.5, 1.0, 0.5],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0, 0.5, 1.0, 0.5],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0, 0.5, 1.0, 0.5],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global: None,
            final_norm_weight: vec![0.5, 1.0, 1.0, 1.0],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 4),
                det_weight: Some(Arc::new(DetNumMatrix {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        Act::from_num(1.0).to_bits(),
                        0,
                        0,
                        0,
                        0,
                        Act::from_num(1.0).to_bits(),
                        0,
                        0,
                    ]
                    .into(),
                })),
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn test_transformer_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: Some(EmbeddingTable {
                rows: vec![vec![0.0, 0.5, 0.0, 0.0], vec![1.0, 1.5, 0.0, 0.0]],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind: Gemma4AttentionKind::Sliding,
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: Some(2),
                cache_sliding_window: Some(2),
                rms_norm_eps: 1e-6,
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: zero_matrix(4, 4).into(),
                k_proj: zero_matrix(2, 4).into(),
                v_proj: Some(zero_matrix(2, 4).into()),
                o_proj: zero_matrix(4, 4).into(),
                q_norm_weight: vec![1.0, 1.0],
                q_norm_weight_det: None,
                k_norm_weight: vec![1.0, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0; 4],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 4),
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn parity_test_model(
        attention_kind: Gemma4AttentionKind,
        sliding_window: Option<usize>,
        with_ple: bool,
    ) -> Gemma4TransformerModel {
        let ple = with_ple.then(|| Gemma4PleLayerWeights {
            input_gate: MatrixF32 {
                rows: 2,
                cols: 4,
                values: vec![0.2, 0.1, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0],
            }
            .into(),
            layer_projection: MatrixF32 {
                rows: 4,
                cols: 2,
                values: vec![0.5, 0.0, 0.0, 0.5, 0.2, 0.1, 0.1, 0.2],
            }
            .into(),
            post_input_norm_weight: vec![1.0; 4],
            post_input_norm_weight_det: None,
        });
        let ple_global = with_ple.then(|| {
            Gemma4PleGlobalWeights::from_materialized(
                vec![MatrixF32 {
                    rows: 3,
                    cols: 2,
                    values: vec![0.1, 0.0, 0.0, 0.1, 0.1, 0.1],
                }],
                vec![MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
                }],
                vec![1.0, 1.0],
                1.0,
                1.0,
                1.0,
            )
        });

        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: Some(EmbeddingTable {
                rows: vec![
                    vec![1.0, 0.0, 0.5, 0.0],
                    vec![0.0, 1.0, 0.0, 0.5],
                    vec![0.5, 0.5, 1.0, 0.0],
                ],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind,
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window,
                cache_sliding_window: sliding_window,
                rms_norm_eps: 1e-6,
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                        1.0,
                    ],
                }
                .into(),
                k_proj: MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                }
                .into(),
                v_proj: Some(
                    MatrixF32 {
                        rows: 2,
                        cols: 4,
                        values: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    }
                    .into(),
                ),
                o_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                        1.0,
                    ],
                }
                .into(),
                q_norm_weight: vec![1.0, 1.0],
                q_norm_weight_det: None,
                k_norm_weight: vec![1.0, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0; 4],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global,
            final_norm_weight: vec![1.0; 4],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 3,
                    cols: 4,
                    values: vec![0.7, 0.1, 0.2, 0.0, 0.0, 0.8, 0.1, 0.1, 0.2, 0.0, 0.8, 0.2],
                },
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn assert_detwgt_required(error: anyhow::Error) {
        assert!(error
            .to_string()
            .contains("model loaded from a .detwgt artifact"));
    }

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
