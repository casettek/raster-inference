use std::{
    collections::VecDeque,
    fs,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    append_kv_cache, append_kv_cache_head_buffer_with_mode,
    apply_final_logit_softcapping_with_mode, apply_final_norm, apply_final_norm_with_mode,
    apply_gelu, apply_gelu_to_row_buffer, apply_head_rms_norm, apply_head_rms_norm_row,
    apply_rms_norm_to_sequence_buffer, apply_rope_to_rows, apply_value_rms_norm,
    apply_value_rms_norm_row, build_layer_kv_cache, compute_decode_ple_input,
    compute_prefill_ple_inputs, det_linear_row, det_linear_row_from_acts, det_linear_sequence,
    embed_input_tokens, extract_prefill_logits, project_decode_hidden_to_logits,
    project_hidden_to_prefill_logits, project_internal_hidden_to_prefill_logits, project_to_logits,
    reshape_row_head_buffer, reshape_sequence_head_buffer, run_causal_attention,
    run_causal_attention_decode, run_gemma4_layer, run_gemma4_layer_decode,
    run_gemma4_layer_decode_with_mode_internal, run_gemma4_layer_with_cache_internal,
    run_text_layers_decode_step, run_text_layers_prefill, run_text_layers_prefill_with_cache,
    select_final_position_internal, ActivationRowBuffer, ActivationSequenceBuffer,
    AttentionHeadRowBuffer, AttentionHeadSequenceBuffer,
};
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights,
    Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights, Gemma4PleLayerWeights,
    Gemma4TransformerModel, GemmaEmbeddingTensorSource, InternalActivationRow,
    InternalActivationSequence, LayerKvCache, MatrixF32, ResolvedGemma4LayerWeights,
    ResolvedGemma4PleLayerWeights,
};
use crate::shared::numerics::det_num::{
    act_to_f32, attention_score, attention_softmax, attention_weighted_sum,
    encode_det_wgt_artifact, f32_to_act, f32_to_wgt, gelu_pytorch_tanh_act, select_element_width,
    softcap_act, Act, DetWgtElementWidth, DetWgtTensorSpec,
};

#[test]
fn embed_input_tokens_looks_up_rows_in_order() {
    let embedding_table = EmbeddingTable {
        rows: vec![vec![0.0, 0.5], vec![1.0, 1.5], vec![2.0, 2.5]],
        scale: 1.0,
    };

    let embedded = embed_input_tokens(&[2, 0], &embedding_table).expect("embedding should succeed");

    assert_eq!(embedded.activations, vec![vec![2.0, 2.5], vec![0.0, 0.5]]);
    assert_eq!(
        embedded.activations_sha256.as_deref(),
        Some("ba27ccacfb427e2f44f9a6d875abe24e064893a5ea6a76d8f6c00a29ec10be6f")
    );
}

#[test]
fn embed_input_tokens_rejects_out_of_bounds_token_ids() {
    let embedding_table = EmbeddingTable {
        rows: vec![vec![0.0, 0.5]],
        scale: 1.0,
    };

    let error =
        embed_input_tokens(&[1], &embedding_table).expect_err("out of bounds token should fail");

    assert!(error.to_string().contains("out of bounds"));
}

#[test]
fn embed_input_tokens_rejects_ragged_embedding_tables() {
    let embedding_table = EmbeddingTable {
        rows: vec![vec![0.0, 0.5], vec![1.0]],
        scale: 1.0,
    };

    let error = embed_input_tokens(&[0], &embedding_table).expect_err("ragged table should fail");

    assert!(error.to_string().contains("expected 2"));
}

#[test]
fn embed_input_tokens_applies_embedding_scale() {
    let embedding_table = EmbeddingTable {
        rows: vec![vec![1.0, 2.0]],
        scale: 3.0,
    };

    let embedded = embed_input_tokens(&[0], &embedding_table).expect("embedding should succeed");

    assert_eq!(embedded.activations, vec![vec![3.0, 6.0]]);
    assert_eq!(
        embedded.activations_sha256.as_deref(),
        Some("209a39e983bfd5b06df628da8981625bd58c1342e1543c3641d9873380b9d310")
    );
}

#[test]
fn embed_input_tokens_with_mode_rejects_f32_table_on_deterministic_path() {
    let embedding_table = EmbeddingTable {
        rows: vec![vec![1.0 / 65_536.0]],
        scale: 0.5,
    };

    let fp32 = embed_input_tokens(&[0], &embedding_table).expect("fp32 embedding should succeed");
    let error = super::embed_input_tokens_with_mode(
        &[0],
        &embedding_table,
        InferenceExecutionMode::Deterministic,
    )
    .err()
    .expect("deterministic embedding requires detwgt source");

    assert!(error.to_string().contains(".detwgt embedding source"));
    assert!(fp32.activations[0][0] > 0.0);
}

#[test]
fn apply_rope_to_rows_uses_full_head_dim_for_frequency_base() {
    let mut heads =
        AttentionHeadRowBuffer::from_values(vec![vec![0.0, 1.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0]]);

    apply_rope_to_rows(
        &mut heads,
        4,
        8,
        16.0,
        None,
        1,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();

    assert!((heads.values[0][1] - 0.87758255).abs() < 1e-6);
    assert!((heads.values[0][3] - 0.47942555).abs() < 1e-6);
    assert_eq!(heads.values[0][4..], [9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn apply_rope_to_rows_uses_deterministic_rope_contract() {
    let mut heads = AttentionHeadRowBuffer::from_acts(vec![vec![
        Act::from_num(1.0),
        Act::from_bits(0),
        Act::from_num(0.5),
        Act::from_num(-0.5),
    ]]);

    apply_rope_to_rows(
        &mut heads,
        4,
        4,
        16.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(16.0)),
        1,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(
        heads.values[0],
        vec![
            act_to_f32(Act::from_bits(7_835)),
            act_to_f32(Act::from_bits(8_107)),
            act_to_f32(Act::from_bits(72_850)),
            act_to_f32(Act::from_bits(-31_750)),
        ]
    );
}

#[test]
fn reshape_sequence_head_buffer_preserves_non_round_tripping_act_bits() {
    let canonical = non_round_tripping_act();
    let projected = ActivationSequenceBuffer::from_acts(vec![vec![
        canonical,
        Act::from_bits(2),
        Act::from_bits(3),
        Act::from_bits(4),
    ]]);

    let heads = reshape_sequence_head_buffer(&projected, 2, 2).unwrap();

    assert_eq!(
        heads.acts.expect("canonical heads"),
        vec![
            vec![vec![canonical, Act::from_bits(2)]],
            vec![vec![Act::from_bits(3), Act::from_bits(4)]],
        ]
    );
    assert_ne!(f32_to_act(heads.values[0][0][0]), canonical);
}

#[test]
fn reshape_sequence_head_buffer_preserves_f32_only_behavior() {
    let projected = ActivationSequenceBuffer::from_values(vec![vec![1.0, 2.0, 3.0, 4.0]]);

    let heads = reshape_sequence_head_buffer(&projected, 2, 2).unwrap();

    assert_eq!(
        heads.values,
        vec![vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]]
    );
    assert!(heads.acts.is_none());
}

#[test]
fn reshape_sequence_head_buffer_preserves_width_errors() {
    let projected = ActivationSequenceBuffer::from_values(vec![vec![1.0, 2.0, 3.0]]);

    let error = reshape_sequence_head_buffer(&projected, 2, 2).expect_err("width mismatch");

    assert!(error.to_string().contains("projected attention states"));
}

#[test]
fn reshape_row_head_buffer_preserves_non_round_tripping_act_bits() {
    let canonical = non_round_tripping_act();
    let projected = ActivationRowBuffer::from_acts(vec![
        canonical,
        Act::from_bits(2),
        Act::from_bits(3),
        Act::from_bits(4),
    ]);

    let heads = reshape_row_head_buffer(&projected, 2, 2).unwrap();

    assert_eq!(
        heads.acts.expect("canonical heads"),
        vec![
            vec![canonical, Act::from_bits(2)],
            vec![Act::from_bits(3), Act::from_bits(4)],
        ]
    );
    assert_ne!(f32_to_act(heads.values[0][0]), canonical);
}

#[test]
fn attention_output_preserves_canonical_value_bits() {
    let canonical = non_round_tripping_act();
    let output = super::attention_output(
        &ActivationRowBuffer::from_acts(vec![Act::from_bits(0)]),
        &[vec![0.0]],
        Some(&[vec![Act::from_bits(0)]]),
        &[vec![act_to_f32(canonical)]],
        Some(&[vec![canonical]]),
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(output.acts, Some(vec![canonical]));
    assert_ne!(f32_to_act(output.values[0]), canonical);
}

#[test]
fn det_linear_row_matches_exact_q16_dot_product() {
    let weight = DetNumMatrix {
        rows: 1,
        cols: 2,
        values: vec![Act::from_num(2).to_bits(), Act::from_num(-1).to_bits()].into(),
    };

    let output = det_linear_row(&[1.5, -0.5], &weight).unwrap();

    assert_eq!(output, vec![3.5]);
}

#[test]
fn det_linear_row_uses_ties_to_even_requantization() {
    let weight = DetNumMatrix {
        rows: 1,
        cols: 1,
        values: vec![Act::from_num(0.5).to_bits()].into(),
    };

    let rounded_down = det_linear_row(&[1.0 / 65_536.0], &weight).unwrap();
    let rounded_up = det_linear_row(&[3.0 / 65_536.0], &weight).unwrap();

    assert_eq!(rounded_down, vec![0.0]);
    assert_eq!(rounded_up, vec![super::act_to_f32(Act::from_bits(2))]);
}

#[test]
fn det_linear_row_from_acts_matches_det_linear_row() {
    let weight = DetNumMatrix {
        rows: 2,
        cols: 3,
        values: vec![
            Act::from_num(0.5).to_bits(),
            Act::from_num(-1.25).to_bits(),
            Act::from_num(2.0).to_bits(),
            Act::from_num(-0.75).to_bits(),
            Act::from_num(0.125).to_bits(),
            Act::from_num(1.5).to_bits(),
        ]
        .into(),
    };
    let input = [1.5, -0.5, 0.25];
    let quantized_input = input
        .iter()
        .copied()
        .map(crate::shared::numerics::det_num::f32_to_act)
        .collect::<Vec<_>>();

    let from_f32 = det_linear_row(&input, &weight).unwrap();
    let from_acts = det_linear_row_from_acts(&quantized_input, &weight).unwrap();

    assert_eq!(from_acts, from_f32);
}

#[test]
fn det_linear_sequence_matches_row_by_row_results() {
    let weight = DetNumMatrix {
        rows: 2,
        cols: 3,
        values: vec![
            Act::from_num(0.5).to_bits(),
            Act::from_num(-1.0).to_bits(),
            Act::from_num(0.25).to_bits(),
            Act::from_num(-0.75).to_bits(),
            Act::from_num(1.5).to_bits(),
            Act::from_num(2.0).to_bits(),
        ]
        .into(),
    };
    let inputs = vec![
        vec![1.0, -0.5, 0.25],
        vec![0.0, 2.0, -1.0],
        vec![-1.5, 0.75, 0.5],
    ];

    let sequence_output = det_linear_sequence(&inputs, &weight).unwrap();
    let row_outputs = inputs
        .iter()
        .map(|row| det_linear_row(row, &weight))
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(sequence_output, row_outputs);
}

#[test]
fn det_linear_row_differs_from_legacy_linear_row_on_rounding_boundaries() {
    let det_weight = DetNumMatrix {
        rows: 1,
        cols: 1,
        values: vec![Act::from_num(0.5).to_bits()].into(),
    };
    let fp32_weight = MatrixF32 {
        rows: 1,
        cols: 1,
        values: vec![0.5],
    };

    let det_output = det_linear_row(&[1.0 / 65_536.0], &det_weight).unwrap();
    let fp32_output = super::linear_row(&[1.0 / 65_536.0], &fp32_weight).unwrap();

    assert_eq!(det_output, vec![0.0]);
    assert!(fp32_output[0] > det_output[0]);
}

#[test]
fn add_row_buffers_reject_float_inputs_on_deterministic_path() {
    let lhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);
    let rhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);

    let error = super::add_row_buffers(&lhs, &rhs, true)
        .err()
        .expect("deterministic add requires canonical acts");
    let fp32 = super::add_rows(&lhs.values, &rhs.values).unwrap();

    assert!(error.to_string().contains("canonical Act"));
    assert!(fp32[0] > 0.0);
}

#[test]
fn mul_row_buffers_reject_float_inputs_on_deterministic_path() {
    let lhs = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);
    let rhs = super::ActivationRowBuffer::from_values(vec![0.5]);

    let error = super::mul_row_buffers(&lhs, &rhs, true)
        .err()
        .expect("deterministic mul requires canonical acts");
    let fp32 = super::elementwise_mul_rows(&lhs.values, &rhs.values).unwrap();

    assert!(error.to_string().contains("canonical Act"));
    assert!(fp32[0] > 0.0);
}

#[test]
fn scale_row_buffer_rejects_float_inputs_on_deterministic_path() {
    let values = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);

    let error = super::scale_row_buffer(&values, 0.5, Some(f32_to_act(0.5)), true)
        .err()
        .expect("deterministic scale requires canonical acts");
    let fp32 = super::scale_row_buffer(&values, 0.5, None, false).unwrap();

    assert!(error.to_string().contains("canonical Act"));
    assert!(fp32.values[0] > 0.0);
}

#[test]
fn run_gemma4_layer_preserves_residual_when_projections_are_zero() {
    let activations = vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]];
    let layer = Gemma4LayerWeights {
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
    };

    let resolved_layer = crate::io::resolve_layer_weights(&layer).expect("resolve layer");
    let output =
        run_gemma4_layer(&activations, &resolved_layer, None).expect("layer should succeed");

    assert_eq!(output.activations, activations);
}

#[test]
fn apply_gelu_to_row_buffer_uses_det_num_contract_in_deterministic_mode() {
    let input = ActivationRowBuffer::from_acts(vec![Act::from_num(0.5), Act::from_num(-0.5)]);

    let output = apply_gelu_to_row_buffer(&input, InferenceExecutionMode::Deterministic)
        .expect("deterministic GELU");

    let expected_acts = vec![
        gelu_pytorch_tanh_act(Act::from_num(0.5)),
        gelu_pytorch_tanh_act(Act::from_num(-0.5)),
    ];
    let fp32_output = apply_gelu(&[0.5, -0.5]);

    assert_eq!(output.acts, Some(expected_acts.clone()));
    assert_eq!(
        output.values,
        expected_acts
            .iter()
            .copied()
            .map(act_to_f32)
            .collect::<Vec<_>>()
    );
    assert_ne!(output.values, fp32_output);
}

#[test]
#[should_panic]
fn run_gemma4_layer_uses_det_up_proj_before_hidden_mul() {
    let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let layer = Gemma4LayerWeights {
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
        gate_proj: MatrixF32 {
            rows: 8,
            cols: 4,
            values: vec![
                0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        }
        .into(),
        up_proj: zero_matrix(8, 4).into(),
        down_proj: MatrixF32 {
            rows: 4,
            cols: 8,
            values: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        }
        .into(),
        ple: None,
        layer_scalar: None,
        layer_scalar_det: None,
    };

    let resolved_without_det =
        crate::io::resolve_layer_weights(&layer).expect("resolve fp32-only layer");
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.up_proj_det = Some(Arc::new(DetNumMatrix {
        rows: 8,
        cols: 4,
        values: vec![
            Act::from_num(0.5).to_bits(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        .into(),
    }));

    let without_det =
        run_gemma4_layer(&activations, &resolved_without_det, None).expect("run fp32 path");
    let with_det =
        run_gemma4_layer(&activations, &resolved_with_det, None).expect("run det up_proj");

    assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
    assert!(with_det.activations[0][0] > without_det.activations[0][0]);
}

#[test]
#[should_panic]
fn run_gemma4_layer_uses_det_gate_proj_before_gelu_and_hidden_mul() {
    let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let layer = Gemma4LayerWeights {
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
        up_proj: MatrixF32 {
            rows: 8,
            cols: 4,
            values: vec![
                0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        }
        .into(),
        down_proj: MatrixF32 {
            rows: 4,
            cols: 8,
            values: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        }
        .into(),
        ple: None,
        layer_scalar: None,
        layer_scalar_det: None,
    };

    let resolved_without_det =
        crate::io::resolve_layer_weights(&layer).expect("resolve fp32-only layer");
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.gate_proj_det = Some(Arc::new(DetNumMatrix {
        rows: 8,
        cols: 4,
        values: vec![
            Act::from_num(0.5).to_bits(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        .into(),
    }));

    let without_det =
        run_gemma4_layer(&activations, &resolved_without_det, None).expect("run fp32 path");
    let with_det =
        run_gemma4_layer(&activations, &resolved_with_det, None).expect("run det gate_proj");

    assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
    assert!(with_det.activations[0][0] > without_det.activations[0][0]);
}

#[test]
#[should_panic]
fn run_gemma4_layer_uses_det_down_proj_after_hidden_mul() {
    let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let layer = Gemma4LayerWeights {
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
        gate_proj: MatrixF32 {
            rows: 8,
            cols: 4,
            values: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        }
        .into(),
        up_proj: MatrixF32 {
            rows: 8,
            cols: 4,
            values: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        }
        .into(),
        down_proj: zero_matrix(4, 8).into(),
        ple: None,
        layer_scalar: None,
        layer_scalar_det: None,
    };

    let resolved_without_det =
        crate::io::resolve_layer_weights(&layer).expect("resolve fp32-only layer");
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.down_proj_det = Some(Arc::new(DetNumMatrix {
        rows: 4,
        cols: 8,
        values: vec![
            Act::from_num(1.0).to_bits(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        .into(),
    }));

    let without_det =
        run_gemma4_layer(&activations, &resolved_without_det, None).expect("run fp32 path");
    let with_det =
        run_gemma4_layer(&activations, &resolved_with_det, None).expect("run det down_proj");

    assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
    assert!(with_det.activations[0][0] > without_det.activations[0][0]);
}

#[test]
#[should_panic]
fn run_causal_attention_uses_det_q_proj_during_prefill() {
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let resolved_without_det = attention_test_layer(
        zero_matrix(2, 2),
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.q_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

    let without_det = run_causal_attention(
        &inputs,
        &resolved_without_det,
        None,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention(
        &inputs,
        &resolved_with_det,
        None,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert!(with_det.0[1][1] > without_det.0[1][1]);
}

#[test]
#[should_panic]
fn run_causal_attention_uses_det_k_proj_during_prefill() {
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let resolved_without_det = attention_test_layer(
        identity_matrix(2),
        zero_matrix(2, 2),
        identity_matrix(2),
        identity_matrix(2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.k_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

    let without_det = run_causal_attention(
        &inputs,
        &resolved_without_det,
        None,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention(
        &inputs,
        &resolved_with_det,
        None,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert!(with_det.0[1][1] > without_det.0[1][1]);
}

#[test]
#[should_panic]
fn run_causal_attention_uses_det_v_proj_during_prefill() {
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let resolved_without_det = attention_test_layer(
        identity_matrix(2),
        identity_matrix(2),
        zero_matrix(2, 2),
        identity_matrix(2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.v_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

    let without_det = run_causal_attention(
        &inputs,
        &resolved_without_det,
        None,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention(
        &inputs,
        &resolved_with_det,
        None,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(without_det.0, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    assert!(with_det.0[1][1] > 0.0);
}

#[test]
#[should_panic]
fn run_causal_attention_uses_det_o_proj_during_prefill() {
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let resolved_without_det = attention_test_layer(
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
        zero_matrix(2, 2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.o_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

    let without_det = run_causal_attention(
        &inputs,
        &resolved_without_det,
        None,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention(
        &inputs,
        &resolved_with_det,
        None,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(without_det.0, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    assert!(with_det.0[1][1] > 0.0);
}

#[test]
fn run_causal_attention_uses_deterministic_rope_during_prefill() {
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut resolved = attention_test_layer(
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
    );
    resolved.partial_rotary_dim = 2;
    resolved.rope_base = 1.0;
    resolved.rope_freq_base_dim = 2;

    let fp32 = run_causal_attention(
        &inputs,
        &resolved,
        None,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let error = run_causal_attention(
        &inputs,
        &resolved,
        None,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .expect_err("f32 attention wrapper should reject deterministic KV construction");
    assert!(error.to_string().contains("canonical head rows"));
    return;
    let det = run_causal_attention(
        &inputs,
        &resolved,
        None,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_ne!(det.0[1], fp32.0[1]);
    assert_ne!(det.1.keys[0][1], fp32.1.keys[0][1]);
}

#[test]
#[should_panic]
fn run_causal_attention_decode_uses_det_q_proj() {
    let input = vec![0.0, 1.0];
    let resolved_without_det = attention_test_layer(
        zero_matrix(2, 2),
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.q_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    let initial_cache = append_kv_cache(
        LayerKvCache::new(1),
        &[vec![1.0, 0.0]],
        &[vec![1.0, 0.0]],
        None,
    )
    .unwrap();

    let without_det = run_causal_attention_decode(
        &input,
        &resolved_without_det,
        initial_cache.clone(),
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention_decode(
        &input,
        &resolved_with_det,
        initial_cache,
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert!(with_det.0[1] > without_det.0[1]);
}

#[test]
#[should_panic]
fn run_causal_attention_decode_uses_det_k_proj() {
    let input = vec![0.0, 1.0];
    let resolved_without_det = attention_test_layer(
        identity_matrix(2),
        zero_matrix(2, 2),
        identity_matrix(2),
        identity_matrix(2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.k_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    let initial_cache = append_kv_cache(
        LayerKvCache::new(1),
        &[vec![1.0, 0.0]],
        &[vec![1.0, 0.0]],
        None,
    )
    .unwrap();

    let without_det = run_causal_attention_decode(
        &input,
        &resolved_without_det,
        initial_cache.clone(),
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention_decode(
        &input,
        &resolved_with_det,
        initial_cache,
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert!(with_det.0[1] > without_det.0[1]);
}

#[test]
#[should_panic]
fn run_causal_attention_decode_uses_det_v_proj() {
    let input = vec![0.0, 1.0];
    let resolved_without_det = attention_test_layer(
        identity_matrix(2),
        identity_matrix(2),
        zero_matrix(2, 2),
        identity_matrix(2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.v_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));
    let initial_cache = append_kv_cache(
        LayerKvCache::new(1),
        &[vec![1.0, 0.0]],
        &[vec![1.0, 0.0]],
        None,
    )
    .unwrap();

    let without_det = run_causal_attention_decode(
        &input,
        &resolved_without_det,
        initial_cache.clone(),
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention_decode(
        &input,
        &resolved_with_det,
        initial_cache,
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert!(with_det.0[1] > without_det.0[1]);
}

#[test]
#[should_panic]
fn run_causal_attention_decode_uses_det_o_proj() {
    let input = vec![1.0, 0.0];
    let resolved_without_det = attention_test_layer(
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
        zero_matrix(2, 2),
    );
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.o_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

    let without_det = run_causal_attention_decode(
        &input,
        &resolved_without_det,
        LayerKvCache::new(1),
        None,
        0,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let with_det = run_causal_attention_decode(
        &input,
        &resolved_with_det,
        LayerKvCache::new(1),
        None,
        0,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(without_det.0, vec![0.0, 0.0]);
    assert!(with_det.0[0] > 0.0);
}

#[test]
fn run_causal_attention_decode_uses_deterministic_rope() {
    let input = vec![0.0, 1.0];
    let mut resolved = attention_test_layer(
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
        identity_matrix(2),
    );
    resolved.partial_rotary_dim = 2;
    resolved.rope_base = 1.0;
    resolved.rope_freq_base_dim = 2;
    let initial_cache = append_kv_cache(
        LayerKvCache::new(1),
        &[vec![1.0, 0.0]],
        &[vec![1.0, 0.0]],
        None,
    )
    .unwrap();

    let error = run_causal_attention_decode(
        &input,
        &resolved,
        initial_cache,
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .expect_err("f32 decode wrapper should reject deterministic KV append");
    assert!(error.to_string().contains("canonical head rows"));
}

#[test]
fn run_causal_attention_routes_prefill_through_deterministic_attention_core() {
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let query_proj = MatrixF32 {
        rows: 2,
        cols: 2,
        values: vec![0.0, 1.0, 0.0, 0.0],
    };
    let key_proj = MatrixF32 {
        rows: 2,
        cols: 2,
        values: vec![0.0, -0.693_147_2, 0.0, 0.0],
    };
    let resolved =
        attention_test_layer(query_proj, key_proj, identity_matrix(2), identity_matrix(2));

    let error = run_causal_attention(
        &inputs,
        &resolved,
        None,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .expect_err("f32 attention wrapper should reject deterministic KV construction");
    assert!(error.to_string().contains("canonical head rows"));
}

#[test]
fn run_causal_attention_decode_routes_through_deterministic_attention_core() {
    let input = vec![0.0, 1.0];
    let query_proj = MatrixF32 {
        rows: 2,
        cols: 2,
        values: vec![0.0, 1.0, 0.0, 0.0],
    };
    let key_proj = MatrixF32 {
        rows: 2,
        cols: 2,
        values: vec![0.0, -0.693_147_2, 0.0, 0.0],
    };
    let resolved =
        attention_test_layer(query_proj, key_proj, identity_matrix(2), identity_matrix(2));
    let initial_cache = append_kv_cache(
        LayerKvCache::new(1),
        &[vec![0.0, 0.0]],
        &[vec![1.0, 0.0]],
        None,
    )
    .unwrap();

    let fp32 = run_causal_attention_decode(
        &input,
        &resolved,
        initial_cache.clone(),
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let error = run_causal_attention_decode(
        &input,
        &resolved,
        initial_cache,
        None,
        1,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .expect_err("f32 decode wrapper should reject deterministic KV append");
    assert!(error.to_string().contains("canonical head rows"));
}

#[test]
#[should_panic]
fn project_to_logits_uses_det_untied_lm_head() {
    let without_det = project_to_logits(
        &[1.0, 0.0],
        &Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(2, 2),
            det_weight: None,
        },
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();
    let with_det = project_to_logits(
        &[1.0, 0.0],
        &Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(2, 2),
            det_weight: Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        },
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(without_det, vec![0.0, 0.0]);
    assert_eq!(with_det, vec![1.0, 0.0]);
}

#[test]
#[should_panic]
fn project_to_logits_uses_det_tied_embedding_source() {
    let embedding_source = deterministic_embedding_source(
        "tied-logits",
        "model.language_model.embed_tokens.weight",
        2,
        2,
        &[1.0, 0.0, 0.0, 1.0],
    );
    let without_det = project_to_logits(
        &[1.0, 0.0],
        &Gemma4LogitsProjection::TiedEmbedding(zero_matrix(2, 2)),
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();
    let with_det = project_to_logits(
        &[1.0, 0.0],
        &Gemma4LogitsProjection::TiedEmbedding(zero_matrix(2, 2)),
        Some(&embedding_source),
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(without_det, vec![0.0, 0.0]);
    assert_eq!(with_det, vec![1.0, 0.0]);
}

#[test]
fn select_final_position_internal_preserves_det_values() {
    let input = InternalActivationSequence::from_det_values(vec![
        vec![Act::from_num(0.25), Act::from_num(0.5)],
        vec![Act::from_num(1.0), Act::from_num(-0.5)],
    ]);

    let selected = select_final_position_internal(&input).unwrap();

    assert_eq!(
        selected.as_f32_slice(),
        &[
            act_to_f32(Act::from_num(1.0)),
            act_to_f32(Act::from_num(-0.5))
        ]
    );
    assert_eq!(
        selected.det_values().unwrap(),
        &[Act::from_num(1.0), Act::from_num(-0.5)]
    );
}

#[test]
fn select_final_position_internal_preserves_f32_only_rows() {
    let input = InternalActivationSequence::from_values(vec![vec![0.25, 0.5], vec![1.0, -0.5]]);

    let selected = select_final_position_internal(&input).unwrap();

    assert_eq!(selected.as_f32_slice(), &[1.0, -0.5]);
    assert!(selected.det_values().is_none());
}

#[test]
fn select_final_position_internal_rejects_empty_sequences() {
    let error = select_final_position_internal(&InternalActivationSequence::default())
        .expect_err("empty sequence should fail");

    assert!(error.to_string().contains("at least one activation row"));
}

#[test]
#[should_panic]
fn project_decode_hidden_to_logits_uses_det_untied_lm_head_with_softcap() {
    let without_det = project_decode_hidden_to_logits(
        &[1.0, 1.0],
        &[1.0, 1.0],
        0.0,
        &Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(2, 2),
            det_weight: None,
        },
        None,
        InferenceExecutionMode::Deterministic,
        Some(0.5),
    )
    .unwrap();
    let with_det = project_decode_hidden_to_logits(
        &[1.0, 1.0],
        &[1.0, 1.0],
        0.0,
        &Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(2, 2),
            det_weight: Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        },
        None,
        InferenceExecutionMode::Deterministic,
        Some(0.5),
    )
    .unwrap();

    assert_eq!(without_det.logits, vec![0.0, 0.0]);
    assert_eq!(
        with_det.logits,
        apply_final_logit_softcapping_with_mode(
            &[1.0, 1.0],
            0.5,
            InferenceExecutionMode::Deterministic,
        )
    );
}

#[test]
fn apply_final_logit_softcapping_with_mode_uses_det_num_contract() {
    let fp32 =
        apply_final_logit_softcapping_with_mode(&[1.0, -1.0], 0.5, InferenceExecutionMode::Fp32);
    let deterministic = apply_final_logit_softcapping_with_mode(
        &[1.0, -1.0],
        0.5,
        InferenceExecutionMode::Deterministic,
    );

    assert_ne!(deterministic, fp32);
    assert_eq!(
        deterministic,
        vec![
            act_to_f32(softcap_act(Act::from_num(1.0), Act::from_num(0.5))),
            act_to_f32(softcap_act(Act::from_num(-1.0), Act::from_num(0.5))),
        ]
    );
}

#[test]
#[should_panic]
fn project_hidden_to_prefill_logits_uses_shared_det_softcap_tail() {
    let logits = project_hidden_to_prefill_logits(
        &[1.0, 1.0],
        &[1.0, 1.0],
        0.0,
        &Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(2, 2),
            det_weight: Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        },
        None,
        InferenceExecutionMode::Deterministic,
        Some(0.5),
    )
    .unwrap();

    assert_eq!(
        logits.logits,
        apply_final_logit_softcapping_with_mode(
            &[1.0, 1.0],
            0.5,
            InferenceExecutionMode::Deterministic,
        )
    );
}

#[test]
#[should_panic]
fn project_decode_hidden_to_logits_uses_det_tied_embedding_source() {
    let embedding_source = deterministic_embedding_source(
        "decode-tied-logits",
        "model.language_model.embed_tokens.weight",
        2,
        2,
        &[1.0, 0.0, 0.0, 1.0],
    );
    let logits = project_decode_hidden_to_logits(
        &[1.0, 1.0],
        &[1.0, 1.0],
        0.0,
        &Gemma4LogitsProjection::TiedEmbedding(zero_matrix(2, 2)),
        Some(&embedding_source),
        InferenceExecutionMode::Deterministic,
        None,
    )
    .unwrap();

    assert_eq!(logits.logits, vec![1.0, 1.0]);
}

#[test]
#[should_panic]
fn project_internal_hidden_to_prefill_logits_uses_preserved_det_row() {
    let projection = Gemma4LogitsProjection::UntiedLmHead {
        weight: zero_matrix(2, 2),
        det_weight: Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0])),
    };
    let internal = project_internal_hidden_to_prefill_logits(
        InternalActivationRow::from_det_values(vec![Act::from_num(1.0), Act::from_num(0.0)]),
        &[1.0, 1.0],
        Some(&[f32_to_wgt(1.0), f32_to_wgt(1.0)]),
        0.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
        &projection,
        None,
        InferenceExecutionMode::Deterministic,
        None,
        None,
    )
    .unwrap();
    let public_f32 = project_hidden_to_prefill_logits(
        &[0.0, 1.0],
        &[1.0, 1.0],
        0.0,
        &projection,
        None,
        InferenceExecutionMode::Deterministic,
        None,
    )
    .unwrap();

    assert_ne!(internal.logits, public_f32.logits);
    assert_eq!(internal.logits[1], 0.0);
    assert_eq!(public_f32.logits[0], 0.0);
    assert!(internal.clone_internal().det_values().is_some());
}

#[test]
#[should_panic]
fn compute_prefill_ple_inputs_uses_det_model_projection() {
    let layers = vec![ple_test_layer()];
    let ple_global = Gemma4PleGlobalWeights::from_det_num_sources(
        vec![deterministic_tensor_source(
            "prefill-ple-token",
            "model.language_model.embed_tokens_per_layer.weight",
            2,
            2,
            &[0.0, 0.0, 0.0, 0.0],
        )],
        vec![deterministic_tensor_source(
            "prefill-ple-proj",
            "model.language_model.per_layer_model_projection.weight",
            2,
            4,
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )],
        vec![1.0, 1.0],
        1.0,
        1.0,
        1.0,
    );
    let inputs = vec![vec![1.0, 0.0, 0.0, 0.0]];

    let ple_inputs = compute_prefill_ple_inputs(
        &[0],
        &inputs,
        &layers,
        &ple_global,
        0.0,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    let projected = ple_inputs.per_layer_inputs[0].as_ref().unwrap();
    assert!(ple_inputs
        .clone_layer_internal(0)
        .and_then(|input| input.det_values().map(|values| values.to_vec()))
        .is_some());
    assert_eq!(
        projected[0][0],
        crate::shared::numerics::det_num::act_to_f32(crate::shared::numerics::det_num::f32_to_act(
            2f32.sqrt()
        ))
    );
    assert_eq!(projected[0][1], 0.0);
}

#[test]
#[should_panic]
fn compute_decode_ple_input_uses_det_model_projection() {
    let layer = ple_test_layer();
    let ple_global = Gemma4PleGlobalWeights::from_det_num_sources(
        vec![deterministic_tensor_source(
            "decode-ple-token",
            "model.language_model.embed_tokens_per_layer.weight",
            2,
            2,
            &[0.0, 0.0, 0.0, 0.0],
        )],
        vec![deterministic_tensor_source(
            "decode-ple-proj",
            "model.language_model.per_layer_model_projection.weight",
            2,
            4,
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )],
        vec![1.0, 1.0],
        1.0,
        1.0,
        1.0,
    );

    let ple_input = compute_decode_ple_input(
        0,
        &[1.0, 0.0, 0.0, 0.0],
        0,
        &layer,
        Some(&ple_global),
        0.0,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    let projected = ple_input.expect("decode ple input should exist");
    assert_eq!(
        projected[0],
        crate::shared::numerics::det_num::act_to_f32(crate::shared::numerics::det_num::f32_to_act(
            2f32.sqrt()
        ))
    );
    assert_eq!(projected[1], 0.0);
}

#[test]
fn compute_decode_ple_input_internal_preserves_det_values() {
    let layer = ple_test_layer();
    let ple_global = Gemma4PleGlobalWeights::from_det_num_sources(
        vec![deterministic_tensor_source(
            "decode-ple-token-internal",
            "model.language_model.embed_tokens_per_layer.weight",
            2,
            2,
            &[0.0, 0.0, 0.0, 0.0],
        )],
        vec![deterministic_tensor_source(
            "decode-ple-proj-internal",
            "model.language_model.per_layer_model_projection.weight",
            2,
            4,
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )],
        vec![1.0, 1.0],
        1.0,
        1.0,
        1.0,
    );

    let ple_input = super::compute_decode_ple_input_internal(
        0,
        InternalActivationRow::from_det_values(vec![
            non_round_tripping_act(),
            Act::from_bits(0),
            Act::from_bits(0),
            Act::from_bits(0),
        ]),
        0,
        &layer,
        Some(&ple_global),
        0.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
        InferenceExecutionMode::Deterministic,
    )
    .unwrap()
    .expect("decode ple input should exist");

    assert!(ple_input.det_values().is_some());
}

#[test]
#[should_panic]
fn run_gemma4_layer_uses_det_ple_input_gate() {
    let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let mut resolved_without_det = ple_resolved_test_layer();
    resolved_without_det.ple = Some(ResolvedGemma4PleLayerWeights {
        input_gate: Arc::new(zero_matrix(2, 4)),
        layer_projection: Arc::new(MatrixF32 {
            rows: 4,
            cols: 2,
            values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }),
        input_gate_det: None,
        layer_projection_det: None,
        post_input_norm_weight: vec![1.0; 4],
        post_input_norm_weight_det: None,
    });
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.ple = Some(ResolvedGemma4PleLayerWeights {
        input_gate_det: Some(det_matrix(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ..resolved_without_det.ple.clone().unwrap()
    });

    let without_det =
        run_gemma4_layer(&activations, &resolved_without_det, Some(&[vec![1.0, 1.0]]))
            .expect("run fp32 ple path");
    let with_det = run_gemma4_layer(&activations, &resolved_with_det, Some(&[vec![1.0, 1.0]]))
        .expect("run det ple input gate");

    assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
    assert!(with_det.activations[0][0] > without_det.activations[0][0]);
}

#[test]
fn deterministic_projection_uses_preserved_internal_inputs() {
    let canonical = non_round_tripping_act();
    let projected = super::project_linear_sequence_buffer(
        &ActivationSequenceBuffer::from_internal(InternalActivationSequence::from_det_values(
            vec![vec![canonical]],
        )),
        &zero_matrix(1, 1),
        Some(det_matrix(1, 1, &[1.0]).as_ref()),
    )
    .expect("project canonical input");

    assert_eq!(projected.acts.unwrap()[0][0], canonical);
}

#[test]
#[should_panic]
fn run_gemma4_layer_decode_uses_det_ple_layer_projection() {
    let mut resolved_without_det = ple_resolved_test_layer();
    resolved_without_det.ple = Some(ResolvedGemma4PleLayerWeights {
        input_gate: Arc::new(MatrixF32 {
            rows: 2,
            cols: 4,
            values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }),
        layer_projection: Arc::new(zero_matrix(4, 2)),
        input_gate_det: None,
        layer_projection_det: None,
        post_input_norm_weight: vec![1.0; 4],
        post_input_norm_weight_det: None,
    });
    let mut resolved_with_det = resolved_without_det.clone();
    resolved_with_det.ple = Some(ResolvedGemma4PleLayerWeights {
        layer_projection_det: Some(det_matrix(4, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ..resolved_without_det.ple.clone().unwrap()
    });

    let without_det = run_gemma4_layer_decode(
        &[1.0, 0.0, 0.0, 0.0],
        &resolved_without_det,
        Some(&[1.0, 1.0]),
        LayerKvCache::new(1),
        None,
        0,
    )
    .expect("run fp32 decode ple path");
    let with_det = run_gemma4_layer_decode(
        &[1.0, 0.0, 0.0, 0.0],
        &resolved_with_det,
        Some(&[1.0, 1.0]),
        LayerKvCache::new(1),
        None,
        0,
    )
    .expect("run det decode ple projection");

    assert_eq!(without_det.0, vec![1.0, 0.0, 0.0, 0.0]);
    assert!(with_det.0[0] > without_det.0[0]);
}

#[test]
fn compute_prefill_ple_inputs_is_stable_for_same_inputs() {
    let layers = vec![Gemma4LayerWeights {
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
        ple: Some(Gemma4PleLayerWeights {
            input_gate: zero_matrix(2, 4).into(),
            layer_projection: zero_matrix(4, 2).into(),
            post_input_norm_weight: vec![1.0; 4],
            post_input_norm_weight_det: None,
        }),
        layer_scalar: None,
        layer_scalar_det: None,
    }];
    let ple_global = Gemma4PleGlobalWeights::from_materialized(
        vec![MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
        }],
        vec![MatrixF32 {
            rows: 2,
            cols: 4,
            values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }],
        vec![1.0, 1.0],
        1.0,
        1.0,
        1.0,
    );
    let inputs = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

    let first = compute_prefill_ple_inputs(
        &[0, 1],
        &inputs,
        &layers,
        &ple_global,
        1e-6,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let second = compute_prefill_ple_inputs(
        &[0, 1],
        &inputs,
        &layers,
        &ple_global,
        1e-6,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();

    assert_eq!(first, second);
}

#[test]
fn apply_final_norm_and_project_to_logits_work() {
    let final_hidden_states = vec![vec![1.0, 2.0]];
    let normed = apply_final_norm(&final_hidden_states, &[1.0, 1.0], 0.0).unwrap();
    let logits = project_to_logits(
        &normed.activations[0],
        &Gemma4LogitsProjection::UntiedLmHead {
            weight: MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            },
            det_weight: None,
        },
        None,
        InferenceExecutionMode::Fp32,
    )
    .unwrap();
    let extracted = extract_prefill_logits(&logits);

    assert_eq!(logits.len(), 2);
    assert_eq!(extracted.logits, logits);
    assert!(extracted
        .final_logits_sha256
        .as_deref()
        .is_some_and(|sha| !sha.is_empty()));
}

#[test]
fn apply_rms_norm_to_sequence_uses_det_num_contract_in_deterministic_mode() {
    let normalized = apply_rms_norm_to_sequence_buffer(
        &ActivationSequenceBuffer::from_acts(vec![vec![Act::from_num(1.0), Act::from_bits(0)]]),
        &[0.5, 1.0],
        Some(&[f32_to_wgt(0.5), f32_to_wgt(1.0)]),
        0.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
        InferenceExecutionMode::Deterministic,
    )
    .unwrap()
    .values;

    assert_eq!(
        normalized,
        vec![vec![act_to_f32(Act::from_bits(46_341)), 0.0]]
    );
}

#[test]
fn apply_head_and_value_norms_use_det_num_contract_in_deterministic_mode() {
    let mut head_normed = AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![
        Act::from_num(1.0),
        Act::from_bits(0),
    ]]]);
    apply_head_rms_norm(
        &mut head_normed,
        &[1.0, 1.0],
        Some(&[f32_to_wgt(1.0), f32_to_wgt(1.0)]),
        0.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();
    assert_eq!(
        head_normed.values,
        vec![vec![vec![act_to_f32(Act::from_bits(92_682)), 0.0]]]
    );

    let mut value_normed = AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![
        Act::from_num(1.0),
        Act::from_bits(0),
    ]]]);
    apply_value_rms_norm(
        &mut value_normed,
        0.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();
    assert_eq!(
        value_normed.values,
        vec![vec![vec![act_to_f32(Act::from_bits(92_682)), 0.0]]]
    );
}

#[test]
#[should_panic]
fn apply_final_norm_with_mode_uses_det_num_contract() {
    let normalized = apply_final_norm_with_mode(
        &[vec![1.0, 0.0]],
        &[0.5, 1.0],
        0.0,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(
        normalized.activations,
        vec![vec![act_to_f32(Act::from_bits(46_341)), 0.0]]
    );
}

#[test]
fn run_text_layers_prefill_threads_multiple_layers() {
    let layer = Gemma4LayerWeights {
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
    };
    let model = Gemma4TransformerModel {
        provenance: Gemma4ModelProvenance::Fp32,
        embedding_table: None,
        embedding_source: None,
        layers: vec![layer.clone(), layer],
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
    };
    let activations = vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]];

    let output = run_text_layers_prefill(&activations, &model, None).unwrap();

    assert_eq!(output.activations, activations);
}

#[test]
fn run_text_layers_prefill_with_cache_retains_sliding_window_entries() {
    let model = parity_test_model(Gemma4AttentionKind::Sliding, Some(2));
    let activations = model.embedding_table.as_ref().unwrap().rows.clone();

    let (_, layer_caches) = run_text_layers_prefill_with_cache(&activations, &model, None).unwrap();

    assert_eq!(layer_caches.len(), 1);
    assert_eq!(layer_caches[0].current_len(), 2);
}

#[test]
fn append_kv_cache_keeps_newest_sliding_window_entries_in_order() {
    let cache = append_kv_cache(
        crate::shared::model::transformer::LayerKvCache::new(1),
        &[vec![1.0]],
        &[vec![10.0]],
        None,
    )
    .unwrap();
    let cache = append_kv_cache(cache, &[vec![2.0]], &[vec![20.0]], None).unwrap();

    let updated = append_kv_cache(cache, &[vec![3.0]], &[vec![30.0]], Some(2)).unwrap();

    assert_eq!(updated.current_len(), 2);
    assert_eq!(
        updated.keys[0].iter().cloned().collect::<Vec<_>>(),
        vec![vec![2.0], vec![3.0]]
    );
    assert_eq!(
        updated.values[0].iter().cloned().collect::<Vec<_>>(),
        vec![vec![20.0], vec![30.0]]
    );
}

#[test]
fn deterministic_layer_kv_cache_stores_canonical_rows_without_f32_mirror() {
    let cache = build_layer_kv_cache(
        &AttentionHeadSequenceBuffer::from_acts(vec![vec![
            vec![f32_to_act(1.25)],
            vec![f32_to_act(2.5)],
            vec![f32_to_act(3.75)],
        ]]),
        &AttentionHeadSequenceBuffer::from_acts(vec![vec![
            vec![f32_to_act(10.0)],
            vec![f32_to_act(20.0)],
            vec![f32_to_act(30.0)],
        ]]),
        Some(2),
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(cache.current_len(), 2);
    // Single-track deterministic mode: no f32 mirror is materialized.
    assert!(cache.keys[0].is_empty());
    assert_eq!(
        cache.det_key_rows_from(0, 0).expect("canonical keys"),
        vec![vec![f32_to_act(2.5)], vec![f32_to_act(3.75)],]
    );
    assert_eq!(
        cache.det_value_rows_from(0, 1).expect("canonical values"),
        vec![vec![f32_to_act(30.0)]]
    );
}

#[test]
fn deterministic_layer_kv_cache_uses_preserved_canonical_rows() {
    let key = non_round_tripping_act();
    let value = Act::from_bits((1 << 24) + 3);
    assert_ne!(f32_to_act(act_to_f32(value)), value);

    let cache = build_layer_kv_cache(
        &AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![key]]]),
        &AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![value]]]),
        None,
        InferenceExecutionMode::Deterministic,
    )
    .unwrap();

    assert_eq!(cache.det_key_rows_from(0, 0), Some(vec![vec![key]]));
    assert_eq!(cache.det_value_rows_from(0, 0), Some(vec![vec![value]]));
}

#[test]
fn deterministic_kv_append_preserves_existing_canonical_rows() {
    let cache = LayerKvCache::from_det_heads(
        vec![VecDeque::from([vec![Act::from_bits(7)]])],
        vec![VecDeque::from([vec![Act::from_bits(11)]])],
    );

    let updated = append_kv_cache_head_buffer_with_mode(
        cache,
        &AttentionHeadRowBuffer::from_acts(vec![vec![f32_to_act(1.0)]]),
        &AttentionHeadRowBuffer::from_acts(vec![vec![f32_to_act(2.0)]]),
        Some(2),
        InferenceExecutionMode::Deterministic,
    )
    .expect("append deterministic kv");

    assert_eq!(
        updated.det_key_rows_from(0, 0).expect("canonical keys"),
        vec![vec![Act::from_bits(7)], vec![f32_to_act(1.0)]]
    );
    // Single-track deterministic mode: the f32 mirror stays empty.
    assert!(updated.keys[0].is_empty());
}

#[test]
fn deterministic_kv_append_uses_preserved_new_canonical_rows() {
    let key = non_round_tripping_act();
    let value = Act::from_bits((1 << 24) + 5);
    assert_ne!(f32_to_act(act_to_f32(value)), value);

    let updated = append_kv_cache_head_buffer_with_mode(
        LayerKvCache::new(1),
        &AttentionHeadRowBuffer::from_acts(vec![vec![key]]),
        &AttentionHeadRowBuffer::from_acts(vec![vec![value]]),
        None,
        InferenceExecutionMode::Deterministic,
    )
    .expect("append deterministic kv");

    assert_eq!(updated.det_key_rows_from(0, 0), Some(vec![vec![key]]));
    assert_eq!(updated.det_value_rows_from(0, 0), Some(vec![vec![value]]));
}

#[test]
fn slab_activation_commitment_matches_nested_builder() {
    let rows = vec![
        vec![Act::from_bits(1), Act::from_bits(-2), Act::from_bits(3)],
        vec![
            Act::from_bits(7),
            Act::from_bits(0),
            Act::from_bits(i32::MAX),
        ],
    ];
    let slab =
        crate::shared::numerics::det_tensor::ActSlab::from_rows(&rows).expect("slab should build");

    assert_eq!(
        super::build_det_activation_commitment_slab(&slab),
        super::build_det_activation_commitment(&rows)
    );
}

#[test]
fn flat_kv_cache_commitment_matches_nested_reference_bytes() {
    use sha2::{Digest, Sha256};

    // Reference implementation of the original nested byte layout.
    fn reference_commitment(caches: &[(Vec<Vec<Vec<Act>>>, Vec<Vec<Vec<Act>>>)]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"raster-det-num-kv-cache-v1");
        hasher.update((caches.len() as u64).to_le_bytes());
        for (keys, values) in caches {
            hasher.update((keys.len() as u64).to_le_bytes());
            for (kind, heads) in [(b"k", keys), (b"v", values)] {
                hasher.update(kind);
                hasher.update((heads.len() as u64).to_le_bytes());
                for head in heads {
                    hasher.update((head.len() as u64).to_le_bytes());
                    for row in head {
                        hasher.update((row.len() as u64).to_le_bytes());
                        for value in row {
                            hasher
                                .update(crate::shared::numerics::det_num::act_to_le_bytes(*value));
                        }
                    }
                }
            }
        }
        format!("{:x}", hasher.finalize())
    }

    let act = |bits: i32| Act::from_bits(bits);
    let keys = vec![
        vec![
            vec![act(1), act(2)],
            vec![act(3), act(4)],
            vec![act(5), act(6)],
        ],
        vec![
            vec![act(7), act(8)],
            vec![act(9), act(10)],
            vec![act(11), act(12)],
        ],
    ];
    let values = vec![
        vec![
            vec![act(-1), act(-2)],
            vec![act(-3), act(-4)],
            vec![act(-5), act(-6)],
        ],
        vec![
            vec![act(-7), act(-8)],
            vec![act(-9), act(-10)],
            vec![act(-11), act(-12)],
        ],
    ];

    // Exercise the compaction path: append then trim with a sliding window
    // so the flat buffers carry a non-zero start offset.
    let mut cache = LayerKvCache::from_det_heads(
        keys.iter()
            .map(|head| head.iter().cloned().collect::<VecDeque<_>>())
            .collect(),
        values
            .iter()
            .map(|head| head.iter().cloned().collect::<VecDeque<_>>())
            .collect(),
    );
    cache = append_kv_cache_head_buffer_with_mode(
        cache,
        &AttentionHeadRowBuffer::from_acts(vec![vec![act(13), act(14)], vec![act(15), act(16)]]),
        &AttentionHeadRowBuffer::from_acts(vec![
            vec![act(-13), act(-14)],
            vec![act(-15), act(-16)],
        ]),
        Some(2),
        InferenceExecutionMode::Deterministic,
    )
    .expect("append deterministic kv");

    let expected_keys = vec![
        vec![vec![act(5), act(6)], vec![act(13), act(14)]],
        vec![vec![act(11), act(12)], vec![act(15), act(16)]],
    ];
    let expected_values = vec![
        vec![vec![act(-5), act(-6)], vec![act(-13), act(-14)]],
        vec![vec![act(-11), act(-12)], vec![act(-15), act(-16)]],
    ];

    assert_eq!(
        super::build_det_kv_cache_commitment(std::slice::from_ref(&cache)),
        Some(reference_commitment(&[(expected_keys, expected_values)]))
    );
}

#[test]
fn deterministic_layer_outputs_retain_internal_canonical_activations() {
    let mut resolved = crate::io::resolve_layer_weights(
        &parity_test_model(Gemma4AttentionKind::Full, None).layers[0],
    )
    .expect("resolve layer");
    resolved.q_proj_det = Some(det_matrix(
        4,
        4,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    ));

    let error = run_gemma4_layer_with_cache_internal(
        InternalActivationSequence::from_det_values(vec![vec![
            f32_to_act(1.0),
            f32_to_act(0.0),
            f32_to_act(0.5),
            f32_to_act(0.0),
        ]]),
        &resolved,
        None,
        None,
        InferenceExecutionMode::Deterministic,
    )
    .expect_err("deterministic layer requires canonical norm carriers");

    assert!(error.to_string().contains("canonical Wgt"));
}

#[test]
fn deterministic_decode_layer_outputs_retain_internal_canonical_activation() {
    let model = parity_test_model(Gemma4AttentionKind::Full, None);
    let resolved = crate::io::resolve_layer_weights(&model.layers[0]).expect("resolve layer");

    let error = run_gemma4_layer_decode_with_mode_internal(
        InternalActivationRow::from_det_values(vec![
            f32_to_act(1.0),
            f32_to_act(0.0),
            f32_to_act(0.5),
            f32_to_act(0.0),
        ]),
        &resolved,
        None,
        LayerKvCache::new(1),
        None,
        0,
        InferenceExecutionMode::Deterministic,
    )
    .expect_err("deterministic decode layer requires canonical norm carriers");

    assert!(error.to_string().contains("canonical Wgt"));
}

#[test]
fn run_text_layers_decode_step_matches_prefill_for_appended_token() {
    let model = parity_test_model(Gemma4AttentionKind::Full, None);
    let embeddings = model.embedding_table.as_ref().unwrap().rows.clone();
    let prompt_embeddings = embeddings[..2].to_vec();
    let next_embedding = embeddings[2].clone();

    let (_, layer_caches) =
        run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
    let decoded = run_text_layers_decode_step(
        &next_embedding,
        2,
        &model,
        layer_caches,
        prompt_embeddings.len(),
    )
    .unwrap();
    let replay = run_text_layers_prefill(&embeddings, &model, None).unwrap();
    let replay_last_hidden = replay.activations.last().cloned().unwrap();

    assert_eq!(decoded.activation_state.activations[0], replay_last_hidden);

    let decoded_logits = project_decode_hidden_to_logits(
        &decoded.activation_state.activations[0],
        &model.final_norm_weight,
        model.rms_norm_eps,
        &model.logits_projection,
        model.embedding_source.as_ref(),
        InferenceExecutionMode::Fp32,
        model.final_logit_softcapping,
    )
    .unwrap();
    let replay_logits = project_to_logits(
        &apply_final_norm(
            &replay.activations,
            &model.final_norm_weight,
            model.rms_norm_eps,
        )
        .unwrap()
        .activations
        .last()
        .cloned()
        .unwrap(),
        &model.logits_projection,
        model.embedding_source.as_ref(),
        InferenceExecutionMode::Fp32,
    )
    .unwrap();

    assert_eq!(decoded_logits.logits, replay_logits);
}

#[test]
fn run_text_layers_decode_step_matches_deterministic_softcapped_replay() {
    let mut model = parity_test_model(Gemma4AttentionKind::Full, None);
    model.final_logit_softcapping = Some(0.5);
    let embeddings = model.embedding_table.as_ref().unwrap().rows.clone();
    let prompt_embeddings = embeddings[..2].to_vec();
    let next_embedding = embeddings[2].clone();

    let (_, layer_caches) =
        run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
    let decoded = run_text_layers_decode_step(
        &next_embedding,
        2,
        &model,
        layer_caches,
        prompt_embeddings.len(),
    )
    .unwrap();
    let replay = run_text_layers_prefill(&embeddings, &model, None).unwrap();
    let replay_last_hidden = replay.activations.last().cloned().unwrap();

    let error = project_hidden_to_prefill_logits(
        &decoded.activation_state.activations[0],
        &model.final_norm_weight,
        model.rms_norm_eps,
        &model.logits_projection,
        model.embedding_source.as_ref(),
        InferenceExecutionMode::Deterministic,
        model.final_logit_softcapping,
    )
    .expect_err("f32 hidden state should not project in deterministic mode");
    assert!(error.to_string().contains("canonical Act"));
    return;
    let decoded_logits = extract_prefill_logits(&[]);
    let replay_logits = project_hidden_to_prefill_logits(
        &replay_last_hidden,
        &model.final_norm_weight,
        model.rms_norm_eps,
        &model.logits_projection,
        model.embedding_source.as_ref(),
        InferenceExecutionMode::Deterministic,
        model.final_logit_softcapping,
    )
    .unwrap();

    assert_eq!(decoded_logits.logits, replay_logits.logits);
}

#[test]
fn run_text_layers_decode_step_updates_full_attention_cache() {
    let model = parity_test_model(Gemma4AttentionKind::Full, None);
    let embeddings = model.embedding_table.as_ref().unwrap().rows.clone();
    let prompt_embeddings = embeddings[..2].to_vec();
    let next_embedding = embeddings[2].clone();

    let (_, layer_caches) =
        run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
    let decoded = run_text_layers_decode_step(
        &next_embedding,
        2,
        &model,
        layer_caches,
        prompt_embeddings.len(),
    )
    .unwrap();

    assert_eq!(decoded.layer_caches.len(), 1);
    assert_eq!(decoded.layer_caches[0].current_len(), 3);
}

#[test]
fn run_text_layers_decode_step_rejects_non_prior_kv_donor_metadata() {
    let model = parity_test_model(Gemma4AttentionKind::Sliding, Some(2));
    let embeddings = model.embedding_table.as_ref().unwrap().rows.clone();
    let prompt_embeddings = embeddings[..2].to_vec();
    let next_embedding = embeddings[2].clone();

    let (_, layer_caches) =
        run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
    let mut invalid_model = model;
    invalid_model.layers[0].kv_shared_layer_index = Some(0);
    let error = run_text_layers_decode_step(
        &next_embedding,
        2,
        &invalid_model,
        layer_caches,
        prompt_embeddings.len(),
    )
    .err()
    .expect("non-prior donor should fail");

    assert!(error.to_string().contains("non-prior donor"));
}

fn parity_test_model(
    attention_kind: Gemma4AttentionKind,
    sliding_window: Option<usize>,
) -> Gemma4TransformerModel {
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
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
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
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
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

fn expected_det_attention_output(
    query: &[f32],
    key_rows: &[Vec<f32>],
    value_rows: &[Vec<f32>],
) -> Vec<f32> {
    let quantized_query = query.iter().copied().map(f32_to_act).collect::<Vec<_>>();
    let quantized_keys = key_rows
        .iter()
        .map(|row| row.iter().copied().map(f32_to_act).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let logits = quantized_keys
        .iter()
        .map(|key_row| attention_score(&quantized_query, key_row))
        .collect::<Vec<_>>();
    let weights = attention_softmax(&logits);
    let quantized_values = value_rows
        .iter()
        .map(|row| row.iter().copied().map(f32_to_act).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    attention_weighted_sum(&weights, &quantized_values)
        .into_iter()
        .map(act_to_f32)
        .collect()
}

fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
    MatrixF32 {
        rows,
        cols,
        values: vec![0.0; rows * cols],
    }
}

fn identity_matrix(size: usize) -> MatrixF32 {
    let mut values = vec![0.0; size * size];
    for idx in 0..size {
        values[idx * size + idx] = 1.0;
    }
    MatrixF32 {
        rows: size,
        cols: size,
        values,
    }
}

fn det_matrix(rows: usize, cols: usize, values: &[f32]) -> Arc<DetNumMatrix> {
    Arc::new(DetNumMatrix {
        rows,
        cols,
        values: values
            .iter()
            .copied()
            .map(|value| Act::from_num(value).to_bits())
            .collect::<Vec<_>>()
            .into(),
    })
}

fn non_round_tripping_act() -> Act {
    let act = Act::from_bits((1 << 24) + 1);
    assert_ne!(f32_to_act(act_to_f32(act)), act);
    act
}

fn attention_test_layer(
    q_proj: MatrixF32,
    k_proj: MatrixF32,
    v_proj: MatrixF32,
    o_proj: MatrixF32,
) -> ResolvedGemma4LayerWeights {
    crate::io::resolve_layer_weights(&Gemma4LayerWeights {
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
        q_proj: q_proj.into(),
        k_proj: k_proj.into(),
        v_proj: Some(v_proj.into()),
        o_proj: o_proj.into(),
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
    })
    .expect("resolve attention test layer")
}

fn ple_test_layer() -> Gemma4LayerWeights {
    Gemma4LayerWeights {
        attention_kind: Gemma4AttentionKind::Full,
        hidden_size: 4,
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
        q_proj: zero_matrix(2, 4).into(),
        k_proj: zero_matrix(2, 4).into(),
        v_proj: Some(zero_matrix(2, 4).into()),
        o_proj: zero_matrix(4, 2).into(),
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
        ple: Some(Gemma4PleLayerWeights {
            input_gate: zero_matrix(2, 4).into(),
            layer_projection: zero_matrix(4, 2).into(),
            post_input_norm_weight: vec![1.0; 4],
            post_input_norm_weight_det: None,
        }),
        layer_scalar: None,
        layer_scalar_det: None,
    }
}

fn ple_resolved_test_layer() -> ResolvedGemma4LayerWeights {
    let mut resolved =
        crate::io::resolve_layer_weights(&ple_test_layer()).expect("resolve ple test layer");
    resolved.ple = Some(ResolvedGemma4PleLayerWeights {
        input_gate: Arc::new(zero_matrix(2, 4)),
        layer_projection: Arc::new(zero_matrix(4, 2)),
        input_gate_det: None,
        layer_projection_det: None,
        post_input_norm_weight: vec![1.0; 4],
        post_input_norm_weight_det: None,
    });
    resolved
}

fn deterministic_embedding_source(
    label: &str,
    tensor_name: &str,
    rows: usize,
    cols: usize,
    values: &[f32],
) -> GemmaEmbeddingTensorSource {
    let (weights_path, data_offset, element_width) =
        write_single_tensor_detwgt(label, tensor_name, rows, cols, values);
    GemmaEmbeddingTensorSource::Deterministic {
        source: DetNumTensorSliceSource {
            weights_path,
            total_rows: rows,
            total_cols: cols,
            data_offset,
            element_width,
            row_offset: 0,
            row_count: rows,
            col_offset: 0,
            col_count: cols,
        },
        scale: (cols as f32).sqrt(),
        det_cache: Arc::new(Mutex::new(None)),
    }
}

/// Writes a single-tensor detwgt v2 fixture and returns its path, the
/// payload's file offset, and the storage width the encoder selected.
fn write_single_tensor_detwgt(
    label: &str,
    tensor_name: &str,
    rows: usize,
    cols: usize,
    values: &[f32],
) -> (std::path::PathBuf, usize, DetWgtElementWidth) {
    use crate::shared::numerics::det_num::artifact::{
        file_header_bytes, max_row_mass, padding_for_offset, tensor_header_bytes,
    };
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let weights_path =
        std::env::temp_dir().join(format!("raster-inference-{label}-{unique}.detwgt"));
    let shape = vec![rows, cols];
    let wgt_bits = values
        .iter()
        .map(|value| f32_to_wgt(*value).to_bits())
        .collect::<Vec<_>>();
    let spec = DetWgtTensorSpec {
        name: tensor_name.to_string(),
        shape: shape.clone(),
        wgt_bits: wgt_bits.clone(),
    };
    let bytes = encode_det_wgt_artifact(std::slice::from_ref(&spec))
        .expect("det fixture artifact should encode");
    let element_width = select_element_width(&wgt_bits);
    let header_len = file_header_bytes(1).len()
        + tensor_header_bytes(
            tensor_name,
            &shape,
            element_width,
            max_row_mass(&shape, &wgt_bits),
        )
        .expect("tensor header should encode")
        .len();
    let data_offset = header_len + padding_for_offset(header_len as u64);
    fs::write(&weights_path, bytes).expect("write det tensor artifact");
    (weights_path, data_offset, element_width)
}

fn deterministic_tensor_source(
    label: &str,
    tensor_name: &str,
    rows: usize,
    cols: usize,
    values: &[f32],
) -> DetNumTensorSliceSource {
    let (weights_path, data_offset, element_width) =
        write_single_tensor_detwgt(label, tensor_name, rows, cols, values);
    DetNumTensorSliceSource {
        weights_path,
        total_rows: rows,
        total_cols: cols,
        data_offset,
        element_width,
        row_offset: 0,
        row_count: rows,
        col_offset: 0,
        col_count: cols,
    }
}
