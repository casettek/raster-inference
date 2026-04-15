use anyhow::Result;
use crate::trace::{trace_event, trace_scope};

pub mod tiles;
pub mod types;

pub use tiles::{
    apply_final_logit_softcapping, apply_final_norm, compute_prefill_ple_inputs,
    embed_input_tokens, extract_prefill_logits, project_to_logits, run_gemma4_layer,
    run_text_layers_prefill, select_final_position,
};
pub use types::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4Phase2Model, Gemma4PleGlobalWeights,
    Gemma4PleLayerWeights, Gemma4PrefillPleInputs, GemmaEmbeddingTensorSource, MatrixF32,
    Phase2State, PrefillLogits,
};

pub fn run_prefill_pass(
    phase1_state: &crate::phase1::Phase1State,
    model: &Gemma4Phase2Model,
    token_embeddings: &ActivationSequence,
) -> Result<Phase2State> {
    let _trace = trace_scope("phase2.run_prefill_pass");
    let ple_inputs = model
        .ple_global
        .as_ref()
        .map(|ple_global| {
            trace_event("phase2.compute_prefill_ple_inputs");
            compute_prefill_ple_inputs(
                &phase1_state.prompt_token_ids,
                &token_embeddings.activations,
                &model.layers,
                ple_global,
                model.rms_norm_eps,
            )
        })
        .transpose()?;
    trace_event("phase2.run_text_layers_prefill");
    let final_hidden_states =
        run_text_layers_prefill(&token_embeddings.activations, model, ple_inputs.as_ref())?;
    trace_event("phase2.apply_final_norm");
    let normalized_hidden_states = apply_final_norm(
        &final_hidden_states.activations,
        &model.final_norm_weight,
        model.rms_norm_eps,
    )?;
    trace_event("phase2.select_final_position");
    let final_position = select_final_position(&normalized_hidden_states.activations)?;
    trace_event("phase2.project_to_logits");
    let mut logits = project_to_logits(&final_position, &model.logits_projection)?;
    if let Some(softcap) = model.final_logit_softcapping {
        trace_event("phase2.apply_final_logit_softcapping");
        logits = apply_final_logit_softcapping(&logits, softcap);
    }
    trace_event("phase2.extract_prefill_logits");
    let prefill_logits = extract_prefill_logits(&logits);

    Ok(Phase2State {
        token_embeddings: token_embeddings.clone(),
        final_hidden_states,
        prefill_logits,
    })
}

pub fn run_phase2(
    phase1_state: &crate::phase1::Phase1State,
    model: &Gemma4Phase2Model,
) -> Result<Phase2State> {
    let _trace = trace_scope("phase2.run_phase2");
    let token_embeddings = if let Some(ref embedding_table) = model.embedding_table {
        trace_event("phase2.embed_input_tokens");
        embed_input_tokens(&phase1_state.prompt_token_ids, embedding_table)?
    } else if let Some(ref embedding_source) = model.embedding_source {
        trace_event("phase2.embed_input_tokens_from_gemma_source");
        crate::io::embed_input_tokens_from_gemma_source(&phase1_state.prompt_token_ids, embedding_source)?
    } else {
        anyhow::bail!("phase 2 model is missing both embedding_table and embedding_source")
    };
    run_prefill_pass(phase1_state, model, &token_embeddings)
}

#[cfg(test)]
mod tests {
    use super::{
        run_phase2, EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights,
        Gemma4LogitsProjection, Gemma4Phase2Model, MatrixF32,
    };
    use crate::phase1::Phase1State;

    #[test]
    fn run_phase2_threads_embeddings_into_prefill_logits() {
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
            layers: vec![Gemma4LayerWeights {
                attention_kind: Gemma4AttentionKind::Sliding,
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: Some(2),
                rms_norm_eps: 1e-6,
                rope_base: 10_000.0,
                partial_rotary_dim: 2,
                attention_k_eq_v: false,
                q_proj: zero_matrix(4, 4),
                k_proj: zero_matrix(2, 4),
                v_proj: Some(zero_matrix(2, 4)),
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
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(zero_matrix(2, 4)),
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
        };

        let phase2_state = run_phase2(&phase1_state, &model).expect("phase 2 should succeed");

        assert_eq!(
            phase2_state.token_embeddings.activations,
            vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]]
        );
        assert_eq!(
            phase2_state.final_hidden_states.activations,
            phase2_state.token_embeddings.activations
        );
        assert_eq!(phase2_state.prefill_logits.logits, vec![0.0, 0.0]);
    }

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
