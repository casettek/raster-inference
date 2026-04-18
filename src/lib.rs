use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;

pub mod io;
pub mod phase1;
pub mod phase2;
pub mod phase3;
pub mod trace;

pub use io::{
    load_chat_template, load_embedding_table_from_gemma_model_path,
    load_embedding_table_from_path, load_phase2_model_from_gemma_model_path,
    load_tokenizer_from_path,
};
pub use phase1::{
    run_phase1, Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, Phase1State,
    SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use phase2::{
    append_kv_cache, apply_final_logit_softcapping, apply_final_norm, compute_decode_ple_input,
    compute_prefill_ple_inputs, decode_step, embed_input_token, embed_input_tokens,
    extract_prefill_logits, project_decode_hidden_to_logits, project_to_logits, run_gemma4_layer,
    run_gemma4_layer_decode, run_phase2, run_phase2_for_token_ids, run_prefill_pass,
    run_text_layers_decode_step, run_text_layers_prefill, run_text_layers_prefill_with_cache,
    select_final_position, ActivationSequence, EmbeddedTokenSequence, EmbeddingTable,
    Gemma4AttentionKind, Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4Phase2Model,
    Gemma4PleGlobalWeights, Gemma4PleLayerWeights, Gemma4PrefillPleInputs,
    GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, Phase2DecodeState,
    Phase2DecodeStepResult, Phase2PrefillResult, Phase2State, PrefillLogits,
};
pub use phase3::{
    append_token, build_phase3_commitment, check_stop_condition, detokenize_output_tokens,
    run_phase3, select_next_token, validate_sampling_config, DecodeState, Phase3State,
    Phase3StopReason,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceState {
    pub phase1: Phase1State,
    pub phase2: Phase2State,
    pub phase3: Phase3State,
}

pub fn run_inference(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    phase2_model: &Gemma4Phase2Model,
) -> Result<InferenceState> {
    trace::start_inference_trace(&json!({
        "model_id": model.model_id,
        "prompt_bytes_sha256": trace::sha256_hex(&request.prompt_bytes),
        "max_new_tokens": request.sampling.max_new_tokens,
        "phase2_layer_count": phase2_model.layers.len(),
    }));

    let result = (|| {
        let phase1 = run_phase1(request, model, tokenizer)?;
        let token_embeddings = if let Some(embedding_table) = phase2_model.embedding_table.as_ref() {
            embed_input_tokens(&phase1.prompt_token_ids, embedding_table)?
        } else if let Some(embedding_source) = phase2_model.embedding_source.as_ref() {
            io::embed_input_tokens_from_gemma_source(&phase1.prompt_token_ids, embedding_source)?
        } else {
            anyhow::bail!("phase 2 model is missing both embedding_table and embedding_source")
        };
        trace::trace_checkpoint("prompt.prepare", &json!({
            "prompt_text": phase1.prompt_text.clone(),
            "prompt_token_ids": phase1.prompt_token_ids.clone(),
            "prompt_token_ids_sha256": phase1.prompt_token_ids_sha256.clone(),
            "embedded_prompt_activations": token_embeddings.activations.clone(),
            "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
            "sampling": request.sampling.clone(),
        }));
        let prefill = run_prefill_pass(&phase1, phase2_model, &token_embeddings)?;
        let mut phase2 = prefill.phase2_state.clone();
        let phase3 = run_phase3(
            &phase1.prompt_token_ids,
            &prefill,
            &request.sampling,
            tokenizer,
            phase2_model,
        )?;
        phase2
            .activation_states
            .extend(phase3.phase2_activation_states.iter().cloned());

        Ok(InferenceState {
            phase1,
            phase2,
            phase3,
        })
    })();

    match &result {
        Ok(state) => trace::finish_inference_trace(&json!({
            "phase1_prompt_token_ids_sha256": state.phase1.prompt_token_ids_sha256,
            "phase3_generated_token_ids_sha256": state.phase3.generated_token_ids_sha256,
            "phase3_generated_text_sha256": trace::sha256_hex(&state.phase3.generated_text),
            "generated_token_count": state.phase3.generated_token_count,
        })),
        Err(error) => trace::abort_inference_trace(error),
    }

    result
}

#[cfg(test)]
mod tests {
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::{
        run_inference, EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights,
        Gemma4LogitsProjection, Gemma4Phase2Model, InferenceRequest, ModelSpec, SamplingConfig,
        TextDecodingPolicy,
    };

    #[test]
    fn run_inference_generates_greedy_text_for_max_new_tokens() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let phase2_model = test_phase2_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state =
            run_inference(&request, &model, &tokenizer, &phase2_model).expect("inference should succeed");

        assert_eq!(inference_state.phase3.generated_token_ids, vec![0, 0]);
        assert_eq!(inference_state.phase3.generated_text, "hello hello");
        assert_eq!(inference_state.phase3.generated_token_count, 2);
        assert_eq!(
            inference_state.phase3.stop_reason,
            crate::phase3::Phase3StopReason::MaxNewTokens
        );
    }

    #[test]
    fn run_inference_returns_empty_generation_when_max_new_tokens_is_zero() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let phase2_model = test_phase2_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state =
            run_inference(&request, &model, &tokenizer, &phase2_model).expect("inference should succeed");

        assert!(inference_state.phase3.generated_token_ids.is_empty());
        assert_eq!(inference_state.phase3.generated_text, "");
        assert_eq!(inference_state.phase3.generated_token_count, 0);
    }

    #[test]
    fn run_inference_rejects_non_default_sampling_before_decode_loop() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let phase2_model = test_phase2_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: Some(5),
                top_p: None,
            },
        };

        let error =
            run_inference(&request, &model, &tokenizer, &phase2_model).expect_err("top_k should fail");
        assert!(error.to_string().contains("top_k"));
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

    fn test_phase2_model() -> Gemma4Phase2Model {
        Gemma4Phase2Model {
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

    fn zero_matrix(rows: usize, cols: usize) -> crate::phase2::MatrixF32 {
        crate::phase2::MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
