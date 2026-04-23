use anyhow::Result;
use tokenizers::Tokenizer;

use crate::shared::input::{InferenceExecutionMode, PromptPreparationState, SamplingConfig};
use crate::shared::output::OutputDecodeState;
use crate::shared::transformer::{
    ActivationSequence, Gemma4TransformerModel, TransformerDecodeState,
    TransformerDecodeStepResult, TransformerPrefillResult, TransformerStateTransitionState,
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
    let ple_inputs = crate::prefill_prepare_aux::run(prompt_token_ids, model, token_embeddings)?;
    trace_event("prefill.layer_stack");
    let (final_hidden_states, layer_caches) = crate::prefill_layer::run_with_mode(
        &token_embeddings.activations,
        model,
        ple_inputs.as_ref(),
        execution_mode,
    )?;
    crate::prefill_finalize::run(prompt_token_ids, model, final_hidden_states, layer_caches)
}

fn embed_token_ids(
    token_ids: &[u32],
    model: &Gemma4TransformerModel,
) -> Result<ActivationSequence> {
    if let Some(ref embedding_table) = model.embedding_table {
        trace_event("prefill.embed_tokens");
        crate::shared::transformer_kernels::embed_input_tokens(token_ids, embedding_table)
    } else if let Some(ref embedding_source) = model.embedding_source {
        trace_event("prefill.embed_tokens");
        crate::io::embed_input_tokens_from_gemma_source(token_ids, embedding_source)
    } else {
        anyhow::bail!(
            "transformer state model is missing both embedding_table and embedding_source"
        )
    }
}

fn embed_token_id(token_id: u32, model: &Gemma4TransformerModel) -> Result<Vec<f32>> {
    if let Some(ref embedding_table) = model.embedding_table {
        crate::shared::transformer_kernels::embed_input_token(token_id, embedding_table)
    } else if let Some(ref embedding_source) = model.embedding_source {
        let embedded =
            crate::io::embed_input_tokens_from_gemma_source(&[token_id], embedding_source)?;
        embedded
            .activations
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("transformer embedding returned no activation rows"))
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
    let token_embeddings = embed_token_ids(token_ids, model)?;
    Ok(
        run_prefill_pass_for_token_ids(
            token_ids,
            model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )?
        .transformer_state,
    )
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
    let _trace = trace_scope("decode.step");
    let TransformerDecodeState {
        layer_caches,
        position,
        token_count,
    } = transformer_decode_state;
    trace_event(format!(
        "decode.summary token={} position={} layers={}",
        next_token,
        position,
        model.layers.len()
    ));
    let embedded_token = embed_token_id(next_token, model)?;
    trace_event("decode.layer_stack");
    let final_hidden_state = match execution_mode {
        InferenceExecutionMode::Fp32 => crate::decode_transition::tiles::run_text_layers_decode_step(
            &embedded_token,
            next_token,
            model,
            layer_caches,
            position,
        )?,
        InferenceExecutionMode::Deterministic => {
            crate::decode_transition::deterministic_tiles::run_text_layers_decode_step(
                &embedded_token,
                next_token,
                model,
                layer_caches,
                position,
            )?
        }
    };
    trace_event("decode.project_to_logits");
    let prefill_logits = crate::shared::transformer_kernels::project_decode_hidden_to_logits(
        &final_hidden_state.activation_state.activations[0],
        &model.final_norm_weight,
        model.rms_norm_eps,
        &model.logits_projection,
        model.final_logit_softcapping,
    )?;

    Ok(TransformerDecodeStepResult {
        transformer_decode_state: TransformerDecodeState {
            layer_caches: final_hidden_state.layer_caches,
            position: position + 1,
            token_count: token_count + 1,
        },
        activation_state: final_hidden_state.activation_state,
        prefill_logits,
    })
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
    let _trace = trace_scope("decode.run");
    let max_new_tokens = validate_sampling_config(sampling)?;
    let mut decode_transition_states = Vec::new();
    let mut decode_state = crate::shared::output::DecodeState::new(
        prompt_token_ids.to_vec(),
        initial_transformer_state
            .transformer_state
            .prefill_logits
            .logits
            .clone(),
        initial_transformer_state.transformer_decode_state.clone(),
    );

    loop {
        if crate::decode_select_token::tiles::check_stop_condition(
            decode_state.generated_token_ids.len(),
            max_new_tokens,
        )
        .is_some()
        {
            trace_event("output.detokenize");
            let mut output_decode_state = crate::output_finalize::run(decode_state, tokenizer)?;
            output_decode_state.decode_transition_states = decode_transition_states;
            return Ok(output_decode_state);
        }

        trace_event("decode.select_token");
        let next_token = crate::decode_select_token::run(&mut decode_state, max_new_tokens)?
            .expect("stop condition should have returned earlier");

        trace_event("decode.step");
        let transformer_decode_state = std::mem::take(&mut decode_state.transformer_decode_state);
        let decode_transition = crate::decode_transition::run_with_mode(
            transformer_decode_state,
            next_token,
            transformer_model,
            execution_mode,
        )?;
        decode_transition_states.push(decode_transition.activation_state.clone());
        decode_state.current_logits = decode_transition.prefill_logits.logits;
        decode_state.transformer_decode_state = decode_transition.transformer_decode_state;
        crate::decode_transition::finalize(&decode_state)?;
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::{
        decode_step, run_output_decode, run_prefill_pass, run_transformer_state_transition,
        run_transformer_state_transition_for_token_ids, validate_sampling_config,
    };
    use crate::{
        shared::transformer_kernels::embed_input_tokens, EmbeddingTable, Gemma4AttentionKind,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4PleGlobalWeights, Gemma4PleLayerWeights,
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
            embedding_table: None,
            embedding_source: None,
            layers: vec![],
            ple_global: None,
            final_norm_weight: vec![],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(zero_matrix(0, 0)),
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
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
                rope_base: 10_000.0,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: zero_matrix(4, 4).into(),
                k_proj: zero_matrix(2, 4).into(),
                v_proj: Some(zero_matrix(2, 4).into()),
                o_proj: zero_matrix(4, 4).into(),
                q_norm_weight: vec![1.0, 1.0],
                k_norm_weight: vec![1.0, 1.0],
                input_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight: vec![1.0; 4],
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(MatrixF32 {
                rows: 3,
                cols: 4,
                values: vec![0.7, 0.1, 0.2, 0.0, 0.0, 0.8, 0.1, 0.1, 0.2, 0.0, 0.8, 0.2],
            }),
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
        }
    }

    fn test_transformer_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
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
                rope_base: 10_000.0,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: zero_matrix(4, 4).into(),
                k_proj: zero_matrix(2, 4).into(),
                v_proj: Some(zero_matrix(2, 4).into()),
                o_proj: zero_matrix(4, 4).into(),
                q_norm_weight: vec![1.0, 1.0],
                k_norm_weight: vec![1.0, 1.0],
                input_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight: vec![1.0; 4],
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(zero_matrix(2, 4)),
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
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
                rope_base: 10_000.0,
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
                k_norm_weight: vec![1.0, 1.0],
                input_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight: vec![1.0; 4],
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple,
                layer_scalar: None,
            }],
            ple_global,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(MatrixF32 {
                rows: 3,
                cols: 4,
                values: vec![0.7, 0.1, 0.2, 0.0, 0.0, 0.8, 0.1, 0.1, 0.2, 0.0, 0.8, 0.2],
            }),
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
        }
    }

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
