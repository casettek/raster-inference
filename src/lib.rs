use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;

pub mod checkpoints;
pub mod decode_select_token;
pub mod decode_transition;
pub mod io;
pub mod output_finalize;
mod pipeline;
pub mod prefill_finalize;
pub mod prefill_layer;
pub mod prefill_prepare_aux;
pub mod prompt_prepare;
pub mod shared;
pub mod trace;

pub use checkpoints::{classify_checkpoint, CheckpointTaxonomy, PhaseId, RoutineId};
pub use decode_select_token::run as run_decode_select_token;
pub use decode_transition::tiles::run_text_layers_decode_step;
pub use decode_transition::{finalize as finalize_decode_transition, run as run_decode_transition};
pub use io::{
    load_chat_template, load_embedding_table_from_gemma_model_path, load_embedding_table_from_path,
    load_tokenizer_from_path, load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path,
};
pub use output_finalize::run as run_output_finalize;
pub use pipeline::{
    decode_step, decode_step_with_mode, run_output_decode, run_output_decode_with_mode,
    run_prefill_pass, run_prefill_pass_with_mode, run_transformer_state_transition,
    run_transformer_state_transition_for_token_ids, validate_sampling_config,
};
pub use prefill_finalize::run as run_prefill_finalize;
pub use prefill_layer::run as run_prefill_layer;
pub use prefill_layer::run_with_mode as run_prefill_layer_with_mode;
pub use prefill_layer::tiles::{run_text_layers_prefill, run_text_layers_prefill_with_cache};
pub use prefill_prepare_aux::run as run_prefill_prepare_aux;
pub use prompt_prepare::run as run_prompt_prepare;
pub use shared::input::{
    Gemma4Prompt, InferenceExecutionMode, InferenceRequest, MessageRole, ModelSpec,
    PromptPreparationState, SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use shared::output::{DecodeState, OutputDecodeState, OutputDecodeStopReason};
pub use shared::transformer::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4PleGlobalWeights, Gemma4PleLayerWeights,
    Gemma4PrefillPleInputs, Gemma4TransformerModel, GemmaEmbeddingTensorSource, LayerKvCache,
    MatrixF32, PrefillLogits, TransformerDecodeState, TransformerDecodeStepResult,
    TransformerPrefillResult, TransformerStateTransitionState,
};
pub use shared::transformer_kernels::{
    append_kv_cache, apply_final_logit_softcapping, apply_final_norm, compute_decode_ple_input,
    compute_prefill_ple_inputs, embed_input_token, embed_input_tokens, extract_prefill_logits,
    project_decode_hidden_to_logits, project_to_logits, run_gemma4_layer, run_gemma4_layer_decode,
    select_final_position,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputEmbeddingState {
    #[serde(flatten)]
    pub prompt_preparation: PromptPreparationState,
    pub embedded_prompt_activations_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceState {
    pub input_embedding: InputEmbeddingState,
    pub transformer_state_transition: TransformerStateTransitionState,
    pub output_decode: OutputDecodeState,
}

pub fn run_inference(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
) -> Result<InferenceState> {
    trace::start_inference_trace(&json!({
        "model_id": model.model_id,
        "execution_mode": request.execution_mode,
        "prompt_bytes_sha256": trace::sha256_hex(&request.prompt_bytes),
        "max_new_tokens": request.sampling.max_new_tokens,
        "transformer_layer_count": transformer_model.layers.len(),
    }));

    let result = (|| {
        let prompt_preparation = run_prompt_prepare(request, model, tokenizer)?;
        let token_embeddings =
            if let Some(embedding_table) = transformer_model.embedding_table.as_ref() {
                embed_input_tokens(&prompt_preparation.prompt_token_ids, embedding_table)?
            } else if let Some(embedding_source) = transformer_model.embedding_source.as_ref() {
                io::embed_input_tokens_from_gemma_source(
                    &prompt_preparation.prompt_token_ids,
                    embedding_source,
                )?
            } else {
                anyhow::bail!(
                    "transformer state model is missing both embedding_table and embedding_source"
                )
            };
        let input_embedding = InputEmbeddingState {
            prompt_preparation: prompt_preparation.clone(),
            embedded_prompt_activations_sha256: token_embeddings.activations_sha256.clone(),
        };
        trace::trace_checkpoint(
            "prompt.prepare",
            &json!({
                "prompt_text": prompt_preparation.prompt_text.clone(),
                "prompt_token_ids": prompt_preparation.prompt_token_ids.clone(),
                "prompt_token_ids_sha256": prompt_preparation.prompt_token_ids_sha256.clone(),
                "embedded_prompt_activations": token_embeddings.activations.clone(),
                "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
                "sampling": request.sampling.clone(),
            }),
        );
        let ple_inputs = run_prefill_prepare_aux(
            &prompt_preparation.prompt_token_ids,
            transformer_model,
            &token_embeddings,
        )?;
        let (final_hidden_states, layer_caches) = run_prefill_layer_with_mode(
            &token_embeddings.activations,
            transformer_model,
            ple_inputs.as_ref(),
            request.execution_mode,
        )?;
        let prefill = run_prefill_finalize(
            &prompt_preparation.prompt_token_ids,
            transformer_model,
            final_hidden_states,
            layer_caches,
        )?;
        let mut transformer_state_transition = prefill.transformer_state.clone();
        let output_decode = run_output_decode_with_mode(
            &prompt_preparation.prompt_token_ids,
            &prefill,
            &request.sampling,
            tokenizer,
            transformer_model,
            request.execution_mode,
        )?;
        transformer_state_transition
            .activation_states
            .extend(output_decode.decode_transition_states.iter().cloned());

        Ok(InferenceState {
            input_embedding,
            transformer_state_transition,
            output_decode,
        })
    })();

    match &result {
        Ok(state) => trace::finish_inference_trace(&json!({
            "input_embedding_prompt_token_ids_sha256": state.input_embedding.prompt_preparation.prompt_token_ids_sha256,
            "output_decode_generated_token_ids_sha256": state.output_decode.generated_token_ids_sha256,
            "output_decode_generated_text_sha256": trace::sha256_hex(&state.output_decode.generated_text),
            "generated_token_count": state.output_decode.generated_token_count,
        })),
        Err(error) => trace::abort_inference_trace(error),
    }

    result
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::{
        embed_input_tokens, finalize_decode_transition, run_decode_select_token,
        run_decode_transition, run_inference, run_output_finalize, run_prefill_finalize,
        run_prefill_layer, run_prefill_prepare_aux, run_prompt_prepare, DecodeState,
        EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights, Gemma4LogitsProjection,
        Gemma4TransformerModel, InferenceExecutionMode, InferenceRequest, MatrixF32, ModelSpec,
        OutputDecodeStopReason, SamplingConfig, TextDecodingPolicy,
    };

    #[test]
    fn run_inference_generates_greedy_text_for_max_new_tokens() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect("inference should succeed");

        assert_eq!(
            inference_state
                .input_embedding
                .prompt_preparation
                .prompt_token_ids,
            vec![1]
        );
        assert_eq!(
            inference_state.output_decode.generated_token_ids,
            vec![0, 0]
        );
        assert_eq!(inference_state.output_decode.generated_text, "hello hello");
        assert_eq!(inference_state.output_decode.generated_token_count, 2);
        assert_eq!(
            inference_state.output_decode.stop_reason,
            OutputDecodeStopReason::MaxNewTokens
        );
    }

    #[test]
    fn run_inference_returns_empty_generation_when_max_new_tokens_is_zero() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect("inference should succeed");

        assert!(inference_state.output_decode.generated_token_ids.is_empty());
        assert_eq!(inference_state.output_decode.generated_text, "");
        assert_eq!(inference_state.output_decode.generated_token_count, 0);
    }

    #[test]
    fn run_inference_rejects_non_default_sampling_before_decode_loop() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: Some(5),
                top_p: None,
            },
        };

        let error = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect_err("top_k should fail");
        assert!(error.to_string().contains("top_k"));
    }

    #[test]
    fn inference_state_serializes_with_protocol_phase_keys() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect("inference should succeed");
        let serialized = serde_json::to_value(&inference_state).expect("serialize inference state");
        let object = serialized
            .as_object()
            .expect("serialized inference state should be an object");

        assert!(object.contains_key("input_embedding"));
        assert!(object.contains_key("transformer_state_transition"));
        assert!(object.contains_key("output_decode"));
    }

    #[test]
    fn inference_state_deserializes_protocol_taxonomy_keys() {
        let serialized_shape = json!({
            "input_embedding": {
                "prompt_text": "prompt",
                "prompt_token_ids": [1],
                "prompt_token_ids_sha256": "prompt-digest",
                "embedded_prompt_activations_sha256": "embed-digest"
            },
            "transformer_state_transition": {
                "activation_states": [
                    {
                        "activations_sha256": "hidden-digest"
                    }
                ],
                "prefill_logits": {
                    "final_logits_sha256": "logits-digest"
                }
            },
            "output_decode": {
                "generated_token_ids": [0],
                "generated_token_ids_sha256": "generated-digest",
                "generated_text": "hello"
            }
        });

        let inference_state: super::InferenceState =
            serde_json::from_value(serialized_shape).expect("serialized shape should deserialize");

        assert_eq!(
            inference_state
                .input_embedding
                .prompt_preparation
                .prompt_token_ids,
            vec![1]
        );
        assert_eq!(
            inference_state
                .input_embedding
                .embedded_prompt_activations_sha256,
            "embed-digest"
        );
        assert_eq!(
            inference_state
                .transformer_state_transition
                .prefill_logits
                .final_logits_sha256,
            "logits-digest"
        );
        assert_eq!(inference_state.output_decode.generated_text, "hello");
    }

    #[test]
    fn routine_exports_support_manual_inference_orchestration() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let prompt_preparation =
            run_prompt_prepare(&request, &model, &tokenizer).expect("prompt prepare");
        let token_embeddings = embed_input_tokens(
            &prompt_preparation.prompt_token_ids,
            transformer_model
                .embedding_table
                .as_ref()
                .expect("embedding table"),
        )
        .expect("embed tokens");
        let ple_inputs = run_prefill_prepare_aux(
            &prompt_preparation.prompt_token_ids,
            &transformer_model,
            &token_embeddings,
        )
        .expect("prefill prepare aux");
        let (final_hidden_states, layer_caches) = run_prefill_layer(
            &token_embeddings.activations,
            &transformer_model,
            ple_inputs.as_ref(),
        )
        .expect("prefill layer");
        let prefill = run_prefill_finalize(
            &prompt_preparation.prompt_token_ids,
            &transformer_model,
            final_hidden_states,
            layer_caches,
        )
        .expect("prefill finalize");

        let mut decode_state = DecodeState::new(
            prompt_preparation.prompt_token_ids.clone(),
            prefill.transformer_state.prefill_logits.logits.clone(),
            prefill.transformer_decode_state.clone(),
        );
        let next_token =
            run_decode_select_token(&mut decode_state, 1).expect("decode select token");
        let next_token = next_token.expect("should select a token");
        let decode_transition = run_decode_transition(
            std::mem::take(&mut decode_state.transformer_decode_state),
            next_token,
            &transformer_model,
        )
        .expect("decode transition");
        decode_state.current_logits = decode_transition.prefill_logits.logits;
        decode_state.transformer_decode_state = decode_transition.transformer_decode_state;
        finalize_decode_transition(&decode_state).expect("decode finalize trace");

        let output = run_output_finalize(decode_state, &tokenizer).expect("output finalize");
        assert_eq!(output.generated_token_ids, vec![0]);
        assert_eq!(output.generated_text, "hello");
        assert_eq!(output.stop_reason, OutputDecodeStopReason::MaxNewTokens);
    }

    fn test_model_spec() -> ModelSpec {
        ModelSpec {
            model_id: "gemma-4-test".to_string(),
            tokenizer_path: "tokenizer.json".into(),
            chat_template: "{{ messages[0].content }}".to_string(),
            bos_token: None,
            eos_token: None,
            unk_token: Some("<unk>".to_string()),
        }
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

    fn test_transformer_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            embedding_table: Some(EmbeddingTable {
                rows: vec![
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0],
                ],
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
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(zero_matrix(3, 4)),
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
