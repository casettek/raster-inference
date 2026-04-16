use anyhow::Result;
use tokenizers::Tokenizer;

use crate::trace::{trace_event, trace_scope};

pub mod tiles;
pub mod types;

pub use tiles::{
    append_token, build_phase3_commitment, check_stop_condition, detokenize_output_tokens,
    select_next_token, validate_sampling_config,
};
pub use types::{DecodeState, Phase3State, Phase3StopReason};

pub fn run_phase3(
    prompt_token_ids: &[u32],
    initial_phase2_state: &crate::phase2::Phase2PrefillResult,
    sampling: &crate::phase1::SamplingConfig,
    tokenizer: &Tokenizer,
    phase2_model: &crate::phase2::Gemma4Phase2Model,
) -> Result<Phase3State> {
    let _trace = trace_scope("phase3.run_phase3");
    let max_new_tokens = validate_sampling_config(sampling)?;
    let mut phase2_activation_states = Vec::new();
    let mut decode_state = DecodeState::new(
        prompt_token_ids.to_vec(),
        initial_phase2_state.phase2_state.prefill_logits.logits.clone(),
        initial_phase2_state.decode_state.clone(),
    );

    loop {
        trace_event("phase3.check_stop_condition");
        if let Some(stop_reason) =
            check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens)
        {
            trace_event("phase3.detokenize_output_tokens");
            let generated_token_count = decode_state.generated_token_ids.len();
            let generated_text =
                detokenize_output_tokens(tokenizer, &decode_state.generated_token_ids)?;
            let generated_token_ids_sha256 =
                build_phase3_commitment(&decode_state.generated_token_ids)?;

            return Ok(Phase3State {
                generated_token_ids: decode_state.generated_token_ids,
                generated_token_ids_sha256,
                generated_text,
                generated_token_count,
                stop_reason,
                phase2_activation_states,
            });
        }

        trace_event("phase3.select_next_token");
        let next_token = select_next_token(&decode_state.current_logits)?;

        trace_event("phase3.append_token");
        decode_state.full_token_ids = append_token(&decode_state.full_token_ids, next_token);
        decode_state.generated_token_ids =
            append_token(&decode_state.generated_token_ids, next_token);

        trace_event("phase3.decode_step");
        let phase2_state =
            crate::phase2::decode_step(&decode_state.phase2_decode_state, next_token, phase2_model)?;
        phase2_activation_states.push(phase2_state.activation_state.clone());
        decode_state.current_logits = phase2_state.prefill_logits.logits;
        decode_state.phase2_decode_state = phase2_state.decode_state;
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::run_phase3;
    use crate::phase1::SamplingConfig;
    use crate::phase2::{
        embed_input_tokens, run_prefill_pass, EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights,
        Gemma4LogitsProjection, Gemma4Phase2Model, MatrixF32,
    };

    #[test]
    fn run_phase3_preserves_zero_token_short_circuit() {
        let tokenizer = test_tokenizer();
        let model = test_phase2_model();
        let prompt_token_ids = vec![1];
        let phase1_state = crate::phase1::Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-phase3".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let prefill = run_prefill_pass(&phase1_state, &model, &token_embeddings).expect("prefill");

        let phase3_state = run_phase3(
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
        .expect("phase3 should succeed");

        assert!(phase3_state.generated_token_ids.is_empty());
        assert_eq!(phase3_state.generated_text, "");
        assert!(phase3_state.phase2_activation_states.is_empty());
    }

    #[test]
    fn run_phase3_generates_greedy_tokens_from_incremental_decode() {
        let tokenizer = test_tokenizer();
        let model = test_phase2_model();
        let prompt_token_ids = vec![1];
        let phase1_state = crate::phase1::Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-phase3".to_string(),
        };
        let token_embeddings =
            embed_input_tokens(&prompt_token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let prefill = run_prefill_pass(&phase1_state, &model, &token_embeddings).expect("prefill");

        let phase3_state = run_phase3(
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
        .expect("phase3 should succeed");

        assert_eq!(phase3_state.generated_token_ids.len(), 2);
        assert_eq!(phase3_state.generated_token_count, 2);
        assert_eq!(phase3_state.phase2_activation_states.len(), 2);
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
                rms_norm_eps: 1e-6,
                rope_base: 10_000.0,
                partial_rotary_dim: 2,
                attention_k_eq_v: false,
                q_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                },
                k_proj: MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                    ],
                },
                v_proj: Some(MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                }),
                o_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                },
                q_norm_weight: vec![1.0, 1.0],
                k_norm_weight: vec![1.0, 1.0],
                input_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight: vec![1.0; 4],
                gate_proj: zero_matrix(8, 4),
                up_proj: zero_matrix(8, 4),
                down_proj: zero_matrix(4, 8),
                ple: None,
                layer_scalar: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(MatrixF32 {
                rows: 3,
                cols: 4,
                values: vec![
                    0.7, 0.1, 0.2, 0.0,
                    0.0, 0.8, 0.1, 0.1,
                    0.2, 0.0, 0.8, 0.2,
                ],
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
