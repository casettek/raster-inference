use anyhow::Result;

pub mod tiles;
pub mod types;

pub use tiles::{embed_input_tokens, run_first_gemma4_layer};
pub use types::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4Layer0Weights,
    Gemma4Phase2Model, Gemma4PleLayerWeights, GemmaEmbeddingTensorSource, MatrixF32, Phase2State,
};

pub fn run_phase2(
    phase1_state: &crate::phase1::Phase1State,
    model: &Gemma4Phase2Model,
) -> Result<Phase2State> {
    let token_embeddings = if let Some(ref embedding_table) = model.embedding_table {
        embed_input_tokens(&phase1_state.prompt_token_ids, embedding_table)?
    } else if let Some(ref embedding_source) = model.embedding_source {
        crate::io::embed_input_tokens_from_gemma_source(&phase1_state.prompt_token_ids, embedding_source)?
    } else {
        anyhow::bail!("phase 2 model is missing both embedding_table and embedding_source")
    };
    let layer0_output = run_first_gemma4_layer(
        &phase1_state.prompt_token_ids,
        &token_embeddings.activations,
        &model.layer0,
    )?;

    Ok(Phase2State {
        token_embeddings,
        layer0_output,
    })
}

#[cfg(test)]
mod tests {
    use super::{run_phase2, EmbeddingTable, Gemma4Layer0Weights, Gemma4Phase2Model, MatrixF32};
    use crate::phase1::Phase1State;

    #[test]
    fn run_phase2_threads_embeddings_into_layer0() {
        let phase1_state = Phase1State {
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let model = Gemma4Phase2Model {
            embedding_table: Some(EmbeddingTable {
                rows: vec![vec![0.0, 0.5, 0.0, 0.0], vec![1.0, 1.5, 0.0, 0.0]],
                scale: 1.0,
            }),
            embedding_source: None,
            layer0: Gemma4Layer0Weights {
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: 2,
                rms_norm_eps: 1e-6,
                q_proj: zero_matrix(4, 4),
                k_proj: zero_matrix(2, 4),
                v_proj: zero_matrix(2, 4),
                o_proj: zero_matrix(4, 4),
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
            },
        };

        let phase2_state = run_phase2(&phase1_state, &model).expect("phase 2 should succeed");

        assert_eq!(
            phase2_state.token_embeddings.activations,
            vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]]
        );
        assert_eq!(
            phase2_state.layer0_output.activations,
            phase2_state.token_embeddings.activations
        );
    }

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
