use std::{
    collections::VecDeque,
    fs,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    append_kv_cache_head_buffer, apply_gelu_to_row_buffer, apply_head_rms_norm,
    apply_rms_norm_to_sequence_buffer, apply_rope_to_rows, apply_value_rms_norm,
    build_layer_kv_cache, det_linear_row, det_linear_row_from_acts, det_linear_sequence,
    reshape_row_head_buffer, reshape_sequence_head_buffer, run_gemma4_layer_decode_internal,
    run_gemma4_layer_with_cache_internal, select_final_position_internal, ActivationRowBuffer,
    ActivationSequenceBuffer, AttentionHeadRowBuffer, AttentionHeadSequenceBuffer,
};
use crate::shared::model::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerWeights,
    Gemma4LogitsProjection, Gemma4PleGlobalWeights, Gemma4PleLayerWeights, Gemma4TransformerModel,
    InternalActivationRow, InternalActivationSequence, LayerKvCache, MatrixF32,
};
use crate::shared::numerics::det_num::{
    act_to_f32, encode_det_wgt_artifact, f32_to_act, f32_to_wgt, gelu_pytorch_tanh_act,
    select_element_width, Act, DetWgtElementWidth, DetWgtTensorSpec,
};

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
fn apply_rope_to_rows_rejects_float_only_inputs() {
    let mut heads =
        AttentionHeadRowBuffer::from_values(vec![vec![0.0, 1.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0]]);

    let error = apply_rope_to_rows(
        &mut heads,
        4,
        8,
        16.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(16.0)),
        1,
    )
    .err()
    .expect("deterministic RoPE requires canonical head rows");

    assert!(error.to_string().contains("canonical head rows"));
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
    )
    .unwrap();

    assert_eq!(output.acts, Some(vec![canonical]));
    assert_ne!(f32_to_act(output.values[0]), canonical);
}

#[test]
fn attention_output_rejects_missing_canonical_key_rows() {
    let error = super::attention_output(
        &ActivationRowBuffer::from_acts(vec![Act::from_bits(0)]),
        &[vec![0.0]],
        None,
        &[vec![1.0]],
        Some(&[vec![Act::from_bits(1)]]),
    )
    .err()
    .expect("deterministic attention requires canonical key cache rows");

    assert!(error.to_string().contains("canonical key cache rows"));
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
fn add_row_buffers_reject_float_inputs() {
    let lhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);
    let rhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);

    let error = super::add_row_buffers(&lhs, &rhs)
        .err()
        .expect("deterministic add requires canonical acts");

    assert!(error.to_string().contains("canonical Act"));
}

#[test]
fn mul_row_buffers_reject_float_inputs() {
    let lhs = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);
    let rhs = super::ActivationRowBuffer::from_values(vec![0.5]);

    let error = super::mul_row_buffers(&lhs, &rhs)
        .err()
        .expect("deterministic mul requires canonical acts");

    assert!(error.to_string().contains("canonical Act"));
}

#[test]
fn scale_row_buffer_rejects_float_inputs() {
    let values = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);

    let error = super::scale_row_buffer(&values, Some(f32_to_act(0.5)))
        .err()
        .expect("deterministic scale requires canonical acts");

    assert!(error.to_string().contains("canonical Act"));
}

#[test]
fn scale_row_buffer_requires_canonical_scalar() {
    let values = super::ActivationRowBuffer::from_acts(vec![Act::from_num(1.0)]);

    let error = super::scale_row_buffer(&values, None)
        .err()
        .expect("deterministic scale requires canonical scalar");

    assert!(error.to_string().contains("canonical Act scalar"));
}

#[test]
fn apply_gelu_to_row_buffer_uses_det_num_contract() {
    let input = ActivationRowBuffer::from_acts(vec![Act::from_num(0.5), Act::from_num(-0.5)]);

    let output = apply_gelu_to_row_buffer(&input).expect("deterministic GELU");

    let expected_acts = vec![
        gelu_pytorch_tanh_act(Act::from_num(0.5)),
        gelu_pytorch_tanh_act(Act::from_num(-0.5)),
    ];

    assert_eq!(output.acts, Some(expected_acts.clone()));
    assert_eq!(
        output.values,
        expected_acts
            .iter()
            .copied()
            .map(act_to_f32)
            .collect::<Vec<_>>()
    );
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
    )
    .unwrap()
    .expect("decode ple input should exist");

    assert!(ple_input.det_values().is_some());
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
fn project_linear_sequence_buffer_requires_canonical_det_weight() {
    let error = super::project_linear_sequence_buffer(
        &ActivationSequenceBuffer::from_acts(vec![vec![Act::from_num(1.0)]]),
        &zero_matrix(1, 1),
        None,
    )
    .err()
    .expect("deterministic projection requires canonical det_weight");

    assert!(error.to_string().contains("canonical det_weight"));
}

#[test]
fn apply_rms_norm_to_sequence_uses_det_num_contract() {
    let normalized = apply_rms_norm_to_sequence_buffer(
        &ActivationSequenceBuffer::from_acts(vec![vec![Act::from_num(1.0), Act::from_bits(0)]]),
        &[0.5, 1.0],
        Some(&[f32_to_wgt(0.5), f32_to_wgt(1.0)]),
        0.0,
        Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
    )
    .unwrap()
    .values;

    assert_eq!(
        normalized,
        vec![vec![act_to_f32(Act::from_bits(46_341)), 0.0]]
    );
}

#[test]
fn apply_head_and_value_norms_use_det_num_contract() {
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
    )
    .unwrap();
    assert_eq!(
        value_normed.values,
        vec![vec![vec![act_to_f32(Act::from_bits(92_682)), 0.0]]]
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

    let updated = append_kv_cache_head_buffer(
        cache,
        &AttentionHeadRowBuffer::from_acts(vec![vec![f32_to_act(1.0)]]),
        &AttentionHeadRowBuffer::from_acts(vec![vec![f32_to_act(2.0)]]),
        Some(2),
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

    let updated = append_kv_cache_head_buffer(
        LayerKvCache::new(1),
        &AttentionHeadRowBuffer::from_acts(vec![vec![key]]),
        &AttentionHeadRowBuffer::from_acts(vec![vec![value]]),
        None,
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
    cache = append_kv_cache_head_buffer(
        cache,
        &AttentionHeadRowBuffer::from_acts(vec![vec![act(13), act(14)], vec![act(15), act(16)]]),
        &AttentionHeadRowBuffer::from_acts(vec![
            vec![act(-13), act(-14)],
            vec![act(-15), act(-16)],
        ]),
        Some(2),
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
    )
    .expect_err("deterministic layer requires canonical norm carriers");

    assert!(error.to_string().contains("canonical Wgt"));
}

#[test]
fn deterministic_decode_layer_outputs_retain_internal_canonical_activation() {
    let model = parity_test_model(Gemma4AttentionKind::Full, None);
    let resolved = crate::io::resolve_layer_weights(&model.layers[0]).expect("resolve layer");

    let error = run_gemma4_layer_decode_internal(
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
    )
    .expect_err("deterministic decode layer requires canonical norm carriers");

    assert!(error.to_string().contains("canonical Wgt"));
}

fn parity_test_model(
    attention_kind: Gemma4AttentionKind,
    sliding_window: Option<usize>,
) -> Gemma4TransformerModel {
    Gemma4TransformerModel {
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

fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
    MatrixF32 {
        rows,
        cols,
        values: vec![0.0; rows * cols],
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
