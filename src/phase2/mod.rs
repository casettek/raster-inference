use anyhow::Result;
use serde_json::json;
use crate::trace::{trace_event, trace_scope};

pub mod tiles;
pub mod types;

pub use tiles::{
    append_kv_cache, apply_final_logit_softcapping, apply_final_norm, compute_decode_ple_input,
    compute_prefill_ple_inputs, embed_input_token, embed_input_tokens, extract_prefill_logits,
    project_decode_hidden_to_logits, project_to_logits, run_gemma4_layer,
    run_gemma4_layer_decode, run_text_layers_decode_step, run_text_layers_prefill,
    run_text_layers_prefill_with_cache, select_final_position,
};
pub use types::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4Phase2Model, Gemma4PleGlobalWeights,
    Gemma4LayerMatrixSource, Gemma4PleLayerWeights, Gemma4PrefillPleInputs,
    GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, Phase2DecodeState,
    Phase2DecodeStepResult, Phase2PrefillResult, Phase2State, PrefillLogits,
    ResolvedGemma4LayerWeights, ResolvedGemma4PleLayerWeights,
};

pub fn run_prefill_pass(
    phase1_state: &crate::phase1::Phase1State,
    model: &Gemma4Phase2Model,
    token_embeddings: &ActivationSequence,
) -> Result<Phase2PrefillResult> {
    run_prefill_pass_for_token_ids(&phase1_state.prompt_token_ids, model, token_embeddings)
}

fn run_prefill_pass_for_token_ids(
    prompt_token_ids: &[u32],
    model: &Gemma4Phase2Model,
    token_embeddings: &ActivationSequence,
) -> Result<Phase2PrefillResult> {
    let _trace = trace_scope("phase2.run_prefill_pass");
    trace_event(format!(
        "phase2.prefill_summary tokens={} layers={}",
        prompt_token_ids.len(),
        model.layers.len()
    ));
    let ple_inputs = model
        .ple_global
        .as_ref()
        .map(|ple_global| {
            // trace_event("phase2.compute_prefill_ple_inputs");
            compute_prefill_ple_inputs(
                prompt_token_ids,
                &token_embeddings.activations,
                &model.layers,
                ple_global,
                model.rms_norm_eps,
            )
        })
        .transpose()?;
    crate::trace::trace_checkpoint("phase2a_prep", &json!({
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": crate::trace::sha256_hex(&prompt_token_ids),
        "embedded_prompt_activations": token_embeddings.activations.clone(),
        "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
        "per_layer_prefill_inputs": ple_inputs.as_ref().map(|inputs| inputs.per_layer_inputs.clone()),
        "per_layer_prefill_input_sha256s": ple_inputs.as_ref().map(|inputs| {
            inputs
                .per_layer_inputs
                .iter()
                .map(|input| input.as_ref().map(crate::trace::sha256_hex))
                .collect::<Vec<_>>()
        }),
    }));
    trace_event("phase2.run_text_layers_prefill");
    let (final_hidden_states, layer_caches) =
        run_text_layers_prefill_with_cache(&token_embeddings.activations, model, ple_inputs.as_ref())?;
    trace_event("phase2.apply_final_norm");
    let normalized_hidden_states = apply_final_norm(
        &final_hidden_states.activations,
        &model.final_norm_weight,
        model.rms_norm_eps,
    )?;
    // trace_event("phase2.select_final_position");
    let final_position = select_final_position(&normalized_hidden_states.activations)?;
    trace_event("phase2.project_to_logits");
    let mut logits = project_to_logits(&final_position, &model.logits_projection)?;
    if let Some(softcap) = model.final_logit_softcapping {
        // trace_event("phase2.apply_final_logit_softcapping");
        logits = apply_final_logit_softcapping(&logits, softcap);
    }
    // trace_event("phase2.extract_prefill_logits");
    let prefill_logits = extract_prefill_logits(&logits);
    crate::trace::trace_checkpoint("phase2a_out", &json!({
        "final_hidden_states": final_hidden_states.activations.clone(),
        "final_hidden_states_sha256": final_hidden_states.activations_sha256.clone(),
        "prefill_logits": prefill_logits.logits.clone(),
        "prefill_logits_sha256": prefill_logits.final_logits_sha256.clone(),
        "decode_position": prompt_token_ids.len(),
        "decode_token_count": prompt_token_ids.len(),
        "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
    }));

    let phase2_state = Phase2State {
        activation_states: vec![final_hidden_states],
        prefill_logits,
    };

    Ok(Phase2PrefillResult {
        decode_state: Phase2DecodeState {
            layer_caches,
            position: prompt_token_ids.len(),
            token_count: prompt_token_ids.len(),
        },
        phase2_state,
    })
}

fn embed_token_ids(
    token_ids: &[u32],
    model: &Gemma4Phase2Model,
) -> Result<ActivationSequence> {
    if let Some(ref embedding_table) = model.embedding_table {
        trace_event("phase2.embed_input_tokens");
        embed_input_tokens(token_ids, embedding_table)
    } else if let Some(ref embedding_source) = model.embedding_source {
        trace_event("phase2.embed_input_tokens");
        crate::io::embed_input_tokens_from_gemma_source(token_ids, embedding_source)
    } else {
        anyhow::bail!("phase 2 model is missing both embedding_table and embedding_source")
    }
}

fn embed_token_id(token_id: u32, model: &Gemma4Phase2Model) -> Result<Vec<f32>> {
    if let Some(ref embedding_table) = model.embedding_table {
        // trace_event("phase2.embed_input_token");
        embed_input_token(token_id, embedding_table)
    } else if let Some(ref embedding_source) = model.embedding_source {
        // trace_event("phase2.embed_input_token");
        let embedded = crate::io::embed_input_tokens_from_gemma_source(&[token_id], embedding_source)?;
        embedded
            .activations
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("phase 2 embedding returned no activation rows"))
    } else {
        anyhow::bail!("phase 2 model is missing both embedding_table and embedding_source")
    }
}

pub fn run_phase2_for_token_ids(
    token_ids: &[u32],
    model: &Gemma4Phase2Model,
) -> Result<Phase2State> {
    let _trace = trace_scope("phase2.run_phase2_for_token_ids");
    let token_embeddings = embed_token_ids(token_ids, model)?;
    Ok(run_prefill_pass_for_token_ids(token_ids, model, &token_embeddings)?.phase2_state)
}

pub fn run_phase2(
    phase1_state: &crate::phase1::Phase1State,
    model: &Gemma4Phase2Model,
) -> Result<Phase2State> {
    let _trace = trace_scope("phase2.run_phase2");
    run_phase2_for_token_ids(&phase1_state.prompt_token_ids, model)
}

pub fn decode_step(
    decode_state: Phase2DecodeState,
    next_token: u32,
    model: &Gemma4Phase2Model,
) -> Result<Phase2DecodeStepResult> {
    let _trace = trace_scope("phase2.decode_step");
    let Phase2DecodeState {
        layer_caches,
        position,
        token_count,
    } = decode_state;
    trace_event(format!(
        "phase2.decode_summary token={} position={} layers={}",
        next_token,
        position,
        model.layers.len()
    ));
    let embedded_token = embed_token_id(next_token, model)?;
    trace_event("phase2.run_text_layers_decode_step");
    let final_hidden_state = run_text_layers_decode_step(
        &embedded_token,
        next_token,
        model,
        layer_caches,
        position,
    )?;
    trace_event("phase2.project_decode_hidden_to_logits");
    let prefill_logits = project_decode_hidden_to_logits(
        &final_hidden_state.activation_state.activations[0],
        &model.final_norm_weight,
        model.rms_norm_eps,
        &model.logits_projection,
        model.final_logit_softcapping,
    )?;

    Ok(Phase2DecodeStepResult {
        decode_state: Phase2DecodeState {
            layer_caches: final_hidden_state.layer_caches,
            position: position + 1,
            token_count: token_count + 1,
        },
        activation_state: final_hidden_state.activation_state,
        prefill_logits,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        decode_step, run_phase2, run_phase2_for_token_ids, run_prefill_pass, EmbeddingTable,
        Gemma4AttentionKind, Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4Phase2Model,
        Gemma4PleGlobalWeights, Gemma4PleLayerWeights, MatrixF32,
    };
    use crate::phase1::Phase1State;

    #[test]
    fn run_phase2_threads_embeddings_into_prefill_logits() {
        let phase1_state = Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let model = test_phase2_model();

        let phase2_state = run_phase2(&phase1_state, &model).expect("phase 2 should succeed");

        assert_eq!(
            phase2_state.activation_states[0].activations,
            vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]]
        );
        assert_eq!(phase2_state.prefill_logits.logits, vec![0.0, 0.0]);
    }

    #[test]
    fn run_phase2_for_token_ids_matches_run_phase2_for_same_tokens() {
        let phase1_state = Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let model = test_phase2_model();

        let via_phase1 = run_phase2(&phase1_state, &model).expect("phase 2 should succeed");
        let via_token_ids =
            run_phase2_for_token_ids(&phase1_state.prompt_token_ids, &model).expect("phase 2 replay");

        assert_eq!(via_token_ids, via_phase1);
    }

    #[test]
    fn run_phase2_for_token_ids_preserves_missing_embedding_error() {
        let model = Gemma4Phase2Model {
            embedding_table: None,
            embedding_source: None,
            layers: vec![],
            ple_global: None,
            final_norm_weight: vec![],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(zero_matrix(0, 0)),
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
        };

        let error = run_phase2_for_token_ids(&[0], &model).expect_err("missing embeddings should fail");
        assert!(error
            .to_string()
            .contains("phase 2 model is missing both embedding_table and embedding_source"));
    }

    #[test]
    fn run_prefill_pass_returns_decode_state_for_each_layer() {
        let phase1_state = Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let model = test_phase2_model();
        let token_embeddings =
            crate::phase2::embed_input_tokens(&phase1_state.prompt_token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");

        let result = run_prefill_pass(&phase1_state, &model, &token_embeddings).expect("prefill result");

        assert_eq!(result.phase2_state.prefill_logits.logits, vec![0.0, 0.0]);
        assert_eq!(result.phase2_state.activation_states.len(), 1);
        assert_eq!(result.phase2_state.activation_states[0].activations, token_embeddings.activations);
        assert_eq!(result.decode_state.position, 2);
        assert_eq!(result.decode_state.token_count, 2);
        assert_eq!(result.decode_state.layer_caches.len(), model.layers.len());
        assert_eq!(result.decode_state.layer_caches[0].current_len(), 2);
    }

    #[test]
    fn decode_step_matches_full_replay_for_full_attention() {
        let model = parity_test_model(Gemma4AttentionKind::Full, None, false);
        let token_ids = vec![0, 1];
        let token_embeddings =
            crate::phase2::embed_input_tokens(&token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let phase1_state = Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let prefill = run_prefill_pass(&phase1_state, &model, &token_embeddings).expect("prefill result");

        let step = decode_step(prefill.decode_state, 2, &model).expect("decode step");
        let replay = run_phase2_for_token_ids(&[0, 1, 2], &model).expect("replay phase2");

        assert_eq!(step.prefill_logits.logits, replay.prefill_logits.logits);
        assert_eq!(step.activation_state.activations.len(), 1);
        assert_eq!(step.decode_state.position, 3);
        assert_eq!(step.decode_state.layer_caches[0].current_len(), 3);
    }

    #[test]
    fn decode_step_matches_full_replay_for_ple_enabled_layers() {
        let model = parity_test_model(Gemma4AttentionKind::Full, None, true);
        let token_ids = vec![0, 1];
        let token_embeddings =
            crate::phase2::embed_input_tokens(&token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let phase1_state = Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let prefill = run_prefill_pass(&phase1_state, &model, &token_embeddings).expect("prefill result");

        let step = decode_step(prefill.decode_state, 2, &model).expect("decode step");
        let replay = run_phase2_for_token_ids(&[0, 1, 2], &model).expect("replay phase2");

        assert_eq!(step.prefill_logits.logits, replay.prefill_logits.logits);
    }

    #[test]
    fn decode_step_preserves_sliding_window_cache_length() {
        let model = parity_test_model(Gemma4AttentionKind::Sliding, Some(2), false);
        let token_ids = vec![0, 1, 2];
        let token_embeddings =
            crate::phase2::embed_input_tokens(&token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let phase1_state = Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let prefill = run_prefill_pass(&phase1_state, &model, &token_embeddings).expect("prefill result");

        assert_eq!(prefill.decode_state.layer_caches[0].current_len(), 2);

        let step = decode_step(prefill.decode_state, 0, &model).expect("decode step");
        let replay = run_phase2_for_token_ids(&[0, 1, 2, 0], &model).expect("replay phase2");

        assert_eq!(step.prefill_logits.logits, replay.prefill_logits.logits);
        assert_eq!(step.decode_state.layer_caches[0].current_len(), 2);
    }

    #[test]
    fn decode_step_matches_full_replay_for_kv_shared_sliding_layers() {
        let model = shared_kv_test_model();
        let token_ids = vec![0, 1, 2];
        let token_embeddings =
            crate::phase2::embed_input_tokens(&token_ids, model.embedding_table.as_ref().unwrap())
                .expect("embedding should succeed");
        let phase1_state = Phase1State {
            prompt_text: "prompt".to_string(),
            prompt_token_ids: token_ids.clone(),
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let prefill = run_prefill_pass(&phase1_state, &model, &token_embeddings).expect("prefill result");

        assert_eq!(prefill.decode_state.layer_caches[0].current_len(), 3);
        assert_eq!(prefill.decode_state.layer_caches[1].current_len(), 0);

        let step = decode_step(prefill.decode_state, 0, &model).expect("decode step");
        let replay = run_phase2_for_token_ids(&[0, 1, 2, 0], &model).expect("replay phase2");

        assert_eq!(step.prefill_logits.logits, replay.prefill_logits.logits);
        assert_eq!(step.decode_state.layer_caches[0].current_len(), 4);
        assert_eq!(step.decode_state.layer_caches[1].current_len(), 0);
    }

    fn test_phase2_model() -> Gemma4Phase2Model {
        Gemma4Phase2Model {
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
    ) -> Gemma4Phase2Model {
        let ple = with_ple.then(|| Gemma4PleLayerWeights {
            input_gate: MatrixF32 {
                rows: 2,
                cols: 4,
                values: vec![
                    0.2, 0.1, 0.0, 0.0,
                    0.0, 0.3, 0.1, 0.0,
                ],
            }
            .into(),
            layer_projection: MatrixF32 {
                rows: 4,
                cols: 2,
                values: vec![
                    0.5, 0.0,
                    0.0, 0.5,
                    0.2, 0.1,
                    0.1, 0.2,
                ],
            }
            .into(),
            post_input_norm_weight: vec![1.0; 4],
        });
        let ple_global = with_ple.then(|| {
            Gemma4PleGlobalWeights::from_materialized(
                vec![MatrixF32 {
                    rows: 3,
                    cols: 2,
                    values: vec![
                        0.1, 0.0,
                        0.0, 0.1,
                        0.1, 0.1,
                    ],
                }],
                vec![MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        0.4, 0.0, 0.0, 0.0,
                        0.0, 0.4, 0.0, 0.0,
                    ],
                }],
                vec![1.0, 1.0],
                1.0,
                1.0,
                1.0,
            )
        });

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
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                }
                .into(),
                k_proj: MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                    ],
                }
                .into(),
                v_proj: Some(MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                }
                .into()),
                o_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
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

    fn shared_kv_test_model() -> Gemma4Phase2Model {
        let donor_layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
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
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ],
            }
            .into(),
            k_proj: MatrixF32 {
                rows: 2,
                cols: 4,
                values: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                ],
            }
            .into(),
            v_proj: Some(MatrixF32 {
                rows: 2,
                cols: 4,
                values: vec![
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ],
            }
            .into()),
            o_proj: MatrixF32 {
                rows: 4,
                cols: 4,
                values: vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
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
        };
        let shared_layer = Gemma4LayerWeights {
            kv_shared_layer_index: Some(0),
            ..donor_layer.clone()
        };

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
            layers: vec![donor_layer, shared_layer],
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
