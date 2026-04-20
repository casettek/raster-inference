use anyhow::Result;
use tokenizers::Tokenizer;

use crate::trace::{trace_event, trace_scope};

pub mod tiles;
pub mod types;

pub use tiles::{
    build_output_decode_commitment, check_stop_condition, validate_sampling_config,
};
pub use types::{DecodeState, OutputDecodeState};

pub fn run_output_decode(
    prompt_token_ids: &[u32],
    initial_transformer_state: &crate::transformer_state_transition::TransformerPrefillResult,
    sampling: &crate::shared::input::SamplingConfig,
    tokenizer: &Tokenizer,
    transformer_model: &crate::transformer_state_transition::Gemma4TransformerModel,
) -> Result<OutputDecodeState> {
    let _trace = trace_scope("decode.run");
    let max_new_tokens = validate_sampling_config(sampling)?;
    let mut decode_transition_states = Vec::new();
    let mut decode_state = DecodeState::new(
        prompt_token_ids.to_vec(),
        initial_transformer_state
            .transformer_state
            .prefill_logits
            .logits
            .clone(),
        initial_transformer_state.transformer_decode_state.clone(),
    );

    loop {
        if check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens).is_some() {
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
        let decode_transition =
            crate::decode_transition::run(transformer_decode_state, next_token, transformer_model)?;
        decode_transition_states.push(decode_transition.activation_state.clone());
        decode_state.current_logits = decode_transition.prefill_logits.logits;
        decode_state.transformer_decode_state = decode_transition.transformer_decode_state;
        crate::decode_transition::finalize(&decode_state)?;
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::run_output_decode;
    use crate::SamplingConfig;
    use crate::transformer_state_transition::{
        embed_input_tokens, run_prefill_pass, EmbeddingTable, Gemma4AttentionKind,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4TransformerModel, MatrixF32,
    };

    #[test]
    fn run_output_decode_preserves_zero_token_short_circuit() {
        let tokenizer = test_tokenizer();
        let model = test_transformer_model();
        let prompt_token_ids = vec![1];
        let prompt_preparation_state = crate::input_embedding::PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-output_decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let prefill = run_prefill_pass(&prompt_preparation_state, &model, &token_embeddings)
            .expect("prefill");

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
        .expect("output_decode should succeed");

        assert!(output_decode_state.generated_token_ids.is_empty());
        assert_eq!(output_decode_state.generated_text, "");
        assert!(output_decode_state.decode_transition_states.is_empty());
    }

    #[test]
    fn run_output_decode_generates_greedy_tokens_from_incremental_decode() {
        let tokenizer = test_tokenizer();
        let model = test_transformer_model();
        let prompt_token_ids = vec![1];
        let prompt_preparation_state = crate::input_embedding::PromptPreparationState {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-output_decode".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let prefill = run_prefill_pass(&prompt_preparation_state, &model, &token_embeddings)
            .expect("prefill");

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
        .expect("output_decode should succeed");

        assert_eq!(output_decode_state.generated_token_ids.len(), 2);
        assert_eq!(output_decode_state.generated_token_count, 2);
        assert_eq!(output_decode_state.decode_transition_states.len(), 2);
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

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
