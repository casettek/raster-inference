use std::{mem::size_of, panic};

use super::{
    acc_add_sat, acc_to_le_bytes, act_to_f32, act_to_le_bytes, add_sat, argmax_first,
    attention_score, attention_softmax, attention_softmax_exp_term, attention_softmax_raw_weight,
    attention_softmax_residual, attention_weighted_sum, clip_act, div_acc_by_u32, div_act,
    f32_to_acc, f32_to_act, f32_to_wgt, gelu_pytorch_tanh_act, mac, mac_bits, mul_sat, mul_wide,
    requantize, rms_norm, rms_norm_scale, rope_rotate_pairs, rshift_round_ties_even, scale_act,
    softcap_act, sub_sat, tanh_act, types::ACC_FRACTIONAL_BITS, types::ACT_FRACTIONAL_BITS,
    types::REQUANTIZE_SHIFT, value_rms_norm, wgt_to_le_bytes, Acc, Act, Wgt,
};

#[test]
fn fixed_type_mapping_matches_specified_bit_layout() {
    assert_eq!(size_of::<Act>(), 4);
    assert_eq!(size_of::<Wgt>(), 4);
    assert_eq!(size_of::<Acc>(), 8);

    assert_eq!(ACT_FRACTIONAL_BITS, 16);
    assert_eq!(ACC_FRACTIONAL_BITS, 32);
    assert_eq!(REQUANTIZE_SHIFT, 16);

    assert_eq!(Act::from_num(1).to_bits(), 1_i32 << ACT_FRACTIONAL_BITS);
    assert_eq!(Wgt::from_num(1).to_bits(), 1_i32 << ACT_FRACTIONAL_BITS);
    assert_eq!(Acc::from_num(1).to_bits(), 1_i64 << ACC_FRACTIONAL_BITS);
}

#[test]
fn mul_wide_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        a_bits: i32,
        b_bits: i32,
        expected_bits: i64,
    }

    let cases = [
        Case {
            name: "zero",
            a_bits: 0,
            b_bits: 123_456,
            expected_bits: 0,
        },
        Case {
            name: "positive_fractional",
            a_bits: 98_304,                // 1.5 in Q16
            b_bits: 131_072,               // 2.0 in Q16
            expected_bits: 12_884_901_888, // 3.0 in Q32
        },
        Case {
            name: "mixed_sign",
            a_bits: 98_304,
            b_bits: -131_072,
            expected_bits: -12_884_901_888,
        },
        Case {
            name: "full_width_headroom",
            a_bits: i32::MAX,
            b_bits: i32::MAX,
            expected_bits: i64::from(i32::MAX) * i64::from(i32::MAX),
        },
    ];

    for case in cases {
        assert_eq!(
            mul_wide(Act::from_bits(case.a_bits), Wgt::from_bits(case.b_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn mac_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        acc_bits: i64,
        a_bits: i32,
        b_bits: i32,
        expected_bits: i64,
    }

    let cases = [
        Case {
            name: "exact_accumulation",
            acc_bits: 10,
            a_bits: 3,
            b_bits: 4,
            expected_bits: 22,
        },
        Case {
            name: "positive_saturation",
            acc_bits: i64::MAX - 3,
            a_bits: 2,
            b_bits: 2,
            expected_bits: i64::MAX,
        },
        Case {
            name: "negative_saturation",
            acc_bits: i64::MIN + 3,
            a_bits: -2,
            b_bits: 2,
            expected_bits: i64::MIN,
        },
    ];

    for case in cases {
        assert_eq!(
            mac(
                Acc::from_bits(case.acc_bits),
                Act::from_bits(case.a_bits),
                Wgt::from_bits(case.b_bits),
            )
            .to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn mac_bits_matches_mac_for_representative_vectors() {
    struct Case {
        name: &'static str,
        acc_bits: i64,
        a_bits: i32,
        b_bits: i32,
    }

    let cases = [
        Case {
            name: "exact_accumulation",
            acc_bits: 10,
            a_bits: 3,
            b_bits: 4,
        },
        Case {
            name: "negative_product",
            acc_bits: 25,
            a_bits: -7,
            b_bits: 9,
        },
        Case {
            name: "full_width_product",
            acc_bits: 123,
            a_bits: i32::MAX,
            b_bits: i32::MAX,
        },
        Case {
            name: "positive_saturation",
            acc_bits: i64::MAX - 3,
            a_bits: 2,
            b_bits: 2,
        },
        Case {
            name: "negative_saturation",
            acc_bits: i64::MIN + 3,
            a_bits: -2,
            b_bits: 2,
        },
    ];

    for case in cases {
        assert_eq!(
            mac_bits(case.acc_bits, case.a_bits, case.b_bits),
            mac(
                Acc::from_bits(case.acc_bits),
                Act::from_bits(case.a_bits),
                Wgt::from_bits(case.b_bits),
            )
            .to_bits(),
            "{}",
            case.name
        );
    }
}

#[test]
fn add_sat_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        a_bits: i32,
        b_bits: i32,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "exact_sum",
            a_bits: 7,
            b_bits: 9,
            expected_bits: 16,
        },
        Case {
            name: "positive_edge_without_saturation",
            a_bits: i32::MAX - 1,
            b_bits: 1,
            expected_bits: i32::MAX,
        },
        Case {
            name: "positive_saturation",
            a_bits: i32::MAX,
            b_bits: 1,
            expected_bits: i32::MAX,
        },
        Case {
            name: "negative_saturation",
            a_bits: i32::MIN,
            b_bits: -1,
            expected_bits: i32::MIN,
        },
    ];

    for case in cases {
        assert_eq!(
            add_sat(Act::from_bits(case.a_bits), Act::from_bits(case.b_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn mul_sat_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        a_bits: i32,
        b_bits: i32,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "exact_product",
            a_bits: Act::from_num(1.5).to_bits(),
            b_bits: Act::from_num(2.0).to_bits(),
            expected_bits: Act::from_num(3.0).to_bits(),
        },
        Case {
            name: "ties_to_even_down",
            a_bits: 1,
            b_bits: 0x8000,
            expected_bits: 0,
        },
        Case {
            name: "ties_to_even_up",
            a_bits: 3,
            b_bits: 0x8000,
            expected_bits: 2,
        },
        Case {
            name: "positive_saturation",
            a_bits: i32::MAX,
            b_bits: i32::MAX,
            expected_bits: i32::MAX,
        },
        Case {
            name: "negative_saturation",
            a_bits: i32::MIN,
            b_bits: i32::MAX,
            expected_bits: i32::MIN,
        },
    ];

    for case in cases {
        assert_eq!(
            mul_sat(Act::from_bits(case.a_bits), Act::from_bits(case.b_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn scale_act_matches_mul_sat_contract() {
    let scale = Act::from_num(0.5);
    let value = Act::from_num(3.0);

    assert_eq!(scale_act(value, scale), mul_sat(value, scale));
    assert_eq!(act_to_f32(scale_act(value, scale)), 1.5);
}

#[test]
fn div_act_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        numerator_bits: i32,
        denominator_bits: i32,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "exact_division",
            numerator_bits: Act::from_num(3.0).to_bits(),
            denominator_bits: Act::from_num(2.0).to_bits(),
            expected_bits: Act::from_num(1.5).to_bits(),
        },
        Case {
            name: "half_tie_stays_even",
            numerator_bits: 1,
            denominator_bits: Act::from_num(2.0).to_bits(),
            expected_bits: 0,
        },
        Case {
            name: "negative_rounding",
            numerator_bits: -3,
            denominator_bits: Act::from_num(2.0).to_bits(),
            expected_bits: -2,
        },
    ];

    for case in cases {
        assert_eq!(
            div_act(
                Act::from_bits(case.numerator_bits),
                Act::from_bits(case.denominator_bits),
            )
            .to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn div_acc_by_u32_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        value_bits: i64,
        divisor: u32,
        expected_bits: i64,
    }

    let cases = [
        Case {
            name: "exact_division",
            value_bits: 8,
            divisor: 4,
            expected_bits: 2,
        },
        Case {
            name: "positive_half_tie_rounds_to_even",
            value_bits: 3,
            divisor: 2,
            expected_bits: 2,
        },
        Case {
            name: "positive_above_half_rounds_up",
            value_bits: 7,
            divisor: 4,
            expected_bits: 2,
        },
        Case {
            name: "negative_half_tie_rounds_to_even",
            value_bits: -3,
            divisor: 2,
            expected_bits: -2,
        },
        Case {
            name: "negative_below_half_rounds_toward_zero",
            value_bits: -5,
            divisor: 4,
            expected_bits: -1,
        },
    ];

    for case in cases {
        assert_eq!(
            div_acc_by_u32(Acc::from_bits(case.value_bits), case.divisor).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn rms_norm_scale_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        input_bits: &'static [i32],
        eps_bits: i64,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "unit_vector",
            input_bits: &[65_536],
            eps_bits: 0,
            expected_bits: 65_536,
        },
        Case {
            name: "half_energy",
            input_bits: &[65_536, 0],
            eps_bits: 0,
            expected_bits: 92_682,
        },
        Case {
            name: "zero_row_uses_epsilon",
            input_bits: &[0],
            eps_bits: 1_i64 << ACC_FRACTIONAL_BITS,
            expected_bits: 65_536,
        },
    ];

    for case in cases {
        let input = case
            .input_bits
            .iter()
            .copied()
            .map(Act::from_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            rms_norm_scale(&input, Acc::from_bits(case.eps_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn rms_norm_matches_golden_vectors() {
    let normalized = rms_norm(
        &[Act::from_bits(65_536), Act::from_bits(0)],
        &[Wgt::from_bits(32_768), Wgt::from_bits(65_536)],
        Acc::from_bits(0),
    );
    assert_eq!(
        normalized
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![46_341, 0]
    );

    let signed = rms_norm(
        &[Act::from_bits(65_536), Act::from_bits(-65_536)],
        &[Wgt::from_bits(65_536), Wgt::from_bits(65_536)],
        Acc::from_bits(0),
    );
    assert_eq!(
        signed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![65_536, -65_536]
    );
}

#[test]
fn value_rms_norm_matches_golden_vectors() {
    let normalized = value_rms_norm(
        &[Act::from_bits(65_536), Act::from_bits(0)],
        Acc::from_bits(0),
    );
    assert_eq!(
        normalized
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![92_682, 0]
    );

    let zero_row = value_rms_norm(
        &[Act::from_bits(0)],
        Acc::from_bits(1_i64 << ACC_FRACTIONAL_BITS),
    );
    assert_eq!(
        zero_row
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0]
    );
}

#[test]
fn attention_score_matches_golden_vectors() {
    let score = attention_score(
        &[Act::from_num(1.0), Act::from_num(-0.5)],
        &[Act::from_num(0.5), Act::from_num(0.25)],
    );
    assert_eq!(score.to_bits(), Act::from_num(0.375).to_bits());

    let zero_score = attention_score(&[Act::from_num(0.0)], &[Act::from_num(4.0)]);
    assert_eq!(zero_score.to_bits(), 0);
}

#[test]
fn attention_softmax_matches_contract_vectors() {
    let equal = attention_softmax(&[Act::from_num(0.0), Act::from_num(0.0)]);
    assert_eq!(
        equal
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![32_768, 32_768]
    );

    let ln2_split = attention_softmax(&[Act::from_num(0.0), Act::from_bits(-45_426)]);
    assert_eq!(
        ln2_split
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![43_695, 21_841]
    );

    let stable_tie =
        attention_softmax(&[Act::from_num(0.0), Act::from_num(0.0), Act::from_num(0.0)]);
    assert_eq!(
        stable_tie
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![21_846, 21_845, 21_845]
    );
}

#[test]
fn attention_softmax_helpers_reconstruct_contract_vectors() {
    for logits in [
        vec![Act::from_num(0.0), Act::from_num(0.0)],
        vec![Act::from_num(0.0), Act::from_bits(-45_426)],
        vec![Act::from_num(0.0), Act::from_num(0.0), Act::from_num(0.0)],
        vec![Act::from_num(-1.0), Act::from_num(2.0), Act::from_num(2.0)],
    ] {
        let max_index = argmax_first(&logits);
        let max_logit = logits[max_index];
        let exp_terms = logits
            .iter()
            .map(|logit| attention_softmax_exp_term(*logit, max_logit))
            .collect::<Vec<_>>();
        let sum_exp = exp_terms
            .iter()
            .copied()
            .fold(Acc::from_bits(0), acc_add_sat);
        let mut weights = exp_terms
            .iter()
            .map(|term| attention_softmax_raw_weight(*term, sum_exp))
            .collect::<Vec<_>>();
        let summed_weights = weights.iter().copied().fold(Act::from_bits(0), add_sat);
        let residual = attention_softmax_residual(summed_weights);
        weights[max_index] = add_sat(weights[max_index], residual);

        assert_eq!(
            weights
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            attention_softmax(&logits)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn attention_weighted_sum_matches_golden_vectors() {
    let mixed = attention_weighted_sum(
        &[Act::from_num(0.5), Act::from_num(0.5)],
        &[
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ],
    );
    assert_eq!(
        mixed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![32_768, 32_768]
    );

    let saturated = attention_weighted_sum(
        &[Act::from_num(1.0), Act::from_num(1.0)],
        &[
            vec![Act::from_bits(i32::MAX)],
            vec![Act::from_bits(i32::MAX)],
        ],
    );
    assert_eq!(saturated, vec![Act::from_bits(i32::MAX)]);
}

#[test]
fn tanh_act_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        input_bits: i32,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "zero",
            input_bits: 0,
            expected_bits: 0,
        },
        Case {
            name: "near_zero_preserves_sign",
            input_bits: 1,
            expected_bits: 1,
        },
        Case {
            name: "positive_midrange",
            input_bits: Act::from_num(0.5).to_bits(),
            expected_bits: 30_527,
        },
        Case {
            name: "negative_midrange",
            input_bits: Act::from_num(-0.5).to_bits(),
            expected_bits: -30_527,
        },
        Case {
            name: "saturates_positive_at_three",
            input_bits: Act::from_num(3.0).to_bits(),
            expected_bits: Act::from_num(1.0).to_bits(),
        },
        Case {
            name: "saturates_negative_at_three",
            input_bits: Act::from_num(-3.0).to_bits(),
            expected_bits: Act::from_num(-1.0).to_bits(),
        },
    ];

    for case in cases {
        assert_eq!(
            tanh_act(Act::from_bits(case.input_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn softcap_act_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        input_bits: i32,
        softcap_bits: i32,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "zero_input_stays_zero",
            input_bits: 0,
            softcap_bits: Act::from_num(0.5).to_bits(),
            expected_bits: 0,
        },
        Case {
            name: "unit_softcap_matches_tanh",
            input_bits: Act::from_num(0.5).to_bits(),
            softcap_bits: Act::from_num(1.0).to_bits(),
            expected_bits: 30_527,
        },
        Case {
            name: "negative_input_preserves_sign",
            input_bits: Act::from_num(-0.5).to_bits(),
            softcap_bits: Act::from_num(1.0).to_bits(),
            expected_bits: -30_527,
        },
        Case {
            name: "saturates_at_softcap_boundary",
            input_bits: Act::from_num(3.0).to_bits(),
            softcap_bits: Act::from_num(0.5).to_bits(),
            expected_bits: Act::from_num(0.5).to_bits(),
        },
    ];

    for case in cases {
        assert_eq!(
            softcap_act(
                Act::from_bits(case.input_bits),
                Act::from_bits(case.softcap_bits)
            )
            .to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
#[should_panic(expected = "softcap_act requires a strictly positive softcap")]
fn softcap_act_rejects_non_positive_softcap() {
    let _ = softcap_act(Act::from_num(1.0), Act::from_bits(0));
}

#[test]
fn gelu_pytorch_tanh_act_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        input_bits: i32,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "zero",
            input_bits: 0,
            expected_bits: 0,
        },
        Case {
            name: "half_step_rounds_to_zero",
            input_bits: 1,
            expected_bits: 0,
        },
        Case {
            name: "positive_midrange",
            input_bits: Act::from_num(0.5).to_bits(),
            expected_bits: 22_691,
        },
        Case {
            name: "negative_midrange",
            input_bits: Act::from_num(-0.5).to_bits(),
            expected_bits: -10_077,
        },
        Case {
            name: "large_positive",
            input_bits: Act::from_num(2.0).to_bits(),
            expected_bits: 129_512,
        },
        Case {
            name: "large_negative",
            input_bits: Act::from_num(-2.0).to_bits(),
            expected_bits: -1_560,
        },
        Case {
            name: "tie_sensitive_small_positive",
            input_bits: 3,
            expected_bits: 2,
        },
    ];

    for case in cases {
        assert_eq!(
            gelu_pytorch_tanh_act(Act::from_bits(case.input_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn rope_rotate_pairs_matches_golden_vectors() {
    let rotated = rope_rotate_pairs(
        &[
            Act::from_num(1.0),
            Act::from_num(0.0),
            Act::from_num(0.5),
            Act::from_num(-0.5),
        ],
        4,
        4,
        Acc::from_num(16.0),
        1,
    );

    assert_eq!(
        rotated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![7_835, 8_107, 72_850, -31_750]
    );
}

#[test]
fn rope_rotate_pairs_preserves_zero_position_and_tail() {
    let input = vec![
        Act::from_num(1.0),
        Act::from_num(0.25),
        Act::from_num(0.5),
        Act::from_num(-0.25),
        Act::from_num(3.0),
    ];

    let rotated = rope_rotate_pairs(&input, 4, 4, Acc::from_num(16.0), 0);
    assert_eq!(rotated, input);
}

#[test]
fn rope_rotate_pairs_handles_large_base_without_overflow() {
    let input = vec![
        Act::from_num(1.0),
        Act::from_num(0.0),
        Act::from_num(0.5),
        Act::from_num(-0.5),
        Act::from_num(3.0),
    ];

    let rotated = rope_rotate_pairs(&input, 4, 4, Acc::from_num(1_000_000.0), 4_096);

    assert_eq!(rotated.len(), input.len());
    assert_eq!(rotated[4], input[4]);
    assert_ne!(rotated[1], input[1]);
}

#[test]
#[should_panic(expected = "rope_rotate_pairs requires an even rotary_dim")]
fn rope_rotate_pairs_rejects_odd_rotary_dim() {
    let _ = rope_rotate_pairs(
        &[Act::from_num(1.0), Act::from_num(0.0), Act::from_num(0.5)],
        3,
        4,
        Acc::from_num(16.0),
        1,
    );
}

#[test]
fn sub_sat_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        a_bits: i32,
        b_bits: i32,
        expected_bits: i32,
    }

    let cases = [
        Case {
            name: "exact_difference",
            a_bits: 20,
            b_bits: 3,
            expected_bits: 17,
        },
        Case {
            name: "negative_edge_without_saturation",
            a_bits: i32::MIN + 1,
            b_bits: 1,
            expected_bits: i32::MIN,
        },
        Case {
            name: "negative_saturation",
            a_bits: i32::MIN,
            b_bits: 1,
            expected_bits: i32::MIN,
        },
        Case {
            name: "positive_saturation",
            a_bits: i32::MAX,
            b_bits: -1,
            expected_bits: i32::MAX,
        },
    ];

    for case in cases {
        assert_eq!(
            sub_sat(Act::from_bits(case.a_bits), Act::from_bits(case.b_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn acc_add_sat_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        a_bits: i64,
        b_bits: i64,
        expected_bits: i64,
    }

    let cases = [
        Case {
            name: "exact_sum",
            a_bits: 7,
            b_bits: 9,
            expected_bits: 16,
        },
        Case {
            name: "positive_saturation",
            a_bits: i64::MAX,
            b_bits: 1,
            expected_bits: i64::MAX,
        },
        Case {
            name: "negative_saturation",
            a_bits: i64::MIN,
            b_bits: -1,
            expected_bits: i64::MIN,
        },
    ];

    for case in cases {
        assert_eq!(
            acc_add_sat(Acc::from_bits(case.a_bits), Acc::from_bits(case.b_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn requantize_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        acc_bits: i64,
        expected_bits: i32,
    }

    let max_exact = i64::from(i32::MAX) << REQUANTIZE_SHIFT;
    let min_exact = i64::from(i32::MIN) << REQUANTIZE_SHIFT;

    let cases = [
        Case {
            name: "exact_zero",
            acc_bits: 0,
            expected_bits: 0,
        },
        Case {
            name: "below_half_rounds_down",
            acc_bits: (2_i64 << REQUANTIZE_SHIFT) + 0x7fff,
            expected_bits: 2,
        },
        Case {
            name: "half_tie_stays_even",
            acc_bits: (2_i64 << REQUANTIZE_SHIFT) + 0x8000,
            expected_bits: 2,
        },
        Case {
            name: "half_tie_rounds_up_to_even",
            acc_bits: (3_i64 << REQUANTIZE_SHIFT) + 0x8000,
            expected_bits: 4,
        },
        Case {
            name: "above_half_rounds_up",
            acc_bits: (2_i64 << REQUANTIZE_SHIFT) + 0x8001,
            expected_bits: 3,
        },
        Case {
            name: "negative_below_half_rounds_toward_zero",
            acc_bits: -((2_i64 << REQUANTIZE_SHIFT) + 0x7fff),
            expected_bits: -2,
        },
        Case {
            name: "negative_half_tie_stays_even",
            acc_bits: -((2_i64 << REQUANTIZE_SHIFT) + 0x8000),
            expected_bits: -2,
        },
        Case {
            name: "negative_half_tie_rounds_to_even",
            acc_bits: -((3_i64 << REQUANTIZE_SHIFT) + 0x8000),
            expected_bits: -4,
        },
        Case {
            name: "negative_half_to_zero",
            acc_bits: -0x8000,
            expected_bits: 0,
        },
        Case {
            name: "max_exact_boundary",
            acc_bits: max_exact,
            expected_bits: i32::MAX,
        },
        Case {
            name: "min_exact_boundary",
            acc_bits: min_exact,
            expected_bits: i32::MIN,
        },
        Case {
            name: "just_above_max_saturates",
            acc_bits: (i64::from(i32::MAX) + 1) << REQUANTIZE_SHIFT,
            expected_bits: i32::MAX,
        },
        Case {
            name: "just_below_min_saturates",
            acc_bits: (i64::from(i32::MIN) - 1) << REQUANTIZE_SHIFT,
            expected_bits: i32::MIN,
        },
        Case {
            name: "global_max_saturates",
            acc_bits: i64::MAX,
            expected_bits: i32::MAX,
        },
        Case {
            name: "global_min_saturates",
            acc_bits: i64::MIN,
            expected_bits: i32::MIN,
        },
    ];

    for case in cases {
        assert_eq!(
            requantize(Acc::from_bits(case.acc_bits)).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn clip_act_matches_requantize_golden_vectors() {
    let cases = [
        0,
        (2_i64 << REQUANTIZE_SHIFT) + 0x7fff,
        (3_i64 << REQUANTIZE_SHIFT) + 0x8000,
        -((3_i64 << REQUANTIZE_SHIFT) + 0x8000),
        i64::MAX,
        i64::MIN,
    ];

    for acc_bits in cases {
        let x = Acc::from_bits(acc_bits);
        assert_eq!(clip_act(x), requantize(x), "{acc_bits}");
    }
}

#[test]
fn rshift_round_ties_even_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        bits: i64,
        shift: u32,
        expected_bits: i64,
    }

    let cases = [
        Case {
            name: "zero_shift",
            bits: 42,
            shift: 0,
            expected_bits: 42,
        },
        Case {
            name: "positive_exact",
            bits: 8,
            shift: 2,
            expected_bits: 2,
        },
        Case {
            name: "positive_below_half",
            bits: 9,
            shift: 2,
            expected_bits: 2,
        },
        Case {
            name: "positive_half_stays_even",
            bits: 10,
            shift: 2,
            expected_bits: 2,
        },
        Case {
            name: "positive_half_rounds_up_to_even",
            bits: 14,
            shift: 2,
            expected_bits: 4,
        },
        Case {
            name: "positive_above_half",
            bits: 11,
            shift: 2,
            expected_bits: 3,
        },
        Case {
            name: "negative_exact",
            bits: -8,
            shift: 2,
            expected_bits: -2,
        },
        Case {
            name: "negative_below_half",
            bits: -9,
            shift: 2,
            expected_bits: -2,
        },
        Case {
            name: "negative_half_stays_even",
            bits: -10,
            shift: 2,
            expected_bits: -2,
        },
        Case {
            name: "negative_half_rounds_to_even",
            bits: -14,
            shift: 2,
            expected_bits: -4,
        },
        Case {
            name: "negative_half_to_zero",
            bits: -1,
            shift: 1,
            expected_bits: 0,
        },
        Case {
            name: "max_supported_shift_small_value",
            bits: 1,
            shift: 63,
            expected_bits: 0,
        },
        Case {
            name: "max_supported_shift_min_value",
            bits: i64::MIN,
            shift: 63,
            expected_bits: -1,
        },
    ];

    for case in cases {
        assert_eq!(
            rshift_round_ties_even(Acc::from_bits(case.bits), case.shift).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn argmax_first_matches_contract_cases() {
    let larger_value = [Act::from_bits(4), Act::from_bits(9), Act::from_bits(3)];
    let equal_maxima = [
        Act::from_bits(4),
        Act::from_bits(9),
        Act::from_bits(9),
        Act::from_bits(3),
    ];
    let all_equal = [Act::from_bits(7), Act::from_bits(7), Act::from_bits(7)];
    let single = [Act::from_bits(-123)];

    assert_eq!(argmax_first(&larger_value), 1);
    assert_eq!(argmax_first(&equal_maxima), 1);
    assert_eq!(argmax_first(&all_equal), 0);
    assert_eq!(argmax_first(&single), 0);
}

#[test]
fn serialization_helpers_match_golden_vectors() {
    struct ActCase {
        name: &'static str,
        bits: i32,
        expected: [u8; 4],
    }

    struct AccCase {
        name: &'static str,
        bits: i64,
        expected: [u8; 8],
    }

    struct WgtCase {
        name: &'static str,
        bits: i32,
        expected: [u8; 4],
    }

    let act_cases = [
        ActCase {
            name: "zero",
            bits: 0,
            expected: [0x00, 0x00, 0x00, 0x00],
        },
        ActCase {
            name: "positive",
            bits: 0x1234_5678,
            expected: [0x78, 0x56, 0x34, 0x12],
        },
        ActCase {
            name: "negative",
            bits: -2,
            expected: [0xfe, 0xff, 0xff, 0xff],
        },
    ];

    let acc_cases = [
        AccCase {
            name: "zero",
            bits: 0,
            expected: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        },
        AccCase {
            name: "positive",
            bits: 0x0102_0304_0506_0708,
            expected: [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
        },
        AccCase {
            name: "negative",
            bits: -2,
            expected: [0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        },
    ];

    let wgt_cases = [
        WgtCase {
            name: "zero",
            bits: 0,
            expected: [0x00, 0x00, 0x00, 0x00],
        },
        WgtCase {
            name: "positive",
            bits: 0x0102_0304,
            expected: [0x04, 0x03, 0x02, 0x01],
        },
        WgtCase {
            name: "negative",
            bits: -2,
            expected: [0xfe, 0xff, 0xff, 0xff],
        },
    ];

    for case in act_cases {
        assert_eq!(
            act_to_le_bytes(Act::from_bits(case.bits)),
            case.expected,
            "{}",
            case.name
        );
    }

    for case in acc_cases {
        assert_eq!(
            acc_to_le_bytes(Acc::from_bits(case.bits)),
            case.expected,
            "{}",
            case.name
        );
    }

    for case in wgt_cases {
        assert_eq!(
            wgt_to_le_bytes(Wgt::from_bits(case.bits)),
            case.expected,
            "{}",
            case.name
        );
    }
}

#[test]
fn f32_to_wgt_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        value: f32,
        expected_bits: i32,
    }

    let q16_step = 1.0 / 65_536.0;
    let cases = [
        Case {
            name: "zero",
            value: 0.0,
            expected_bits: 0,
        },
        Case {
            name: "exact_value",
            value: 1.5,
            expected_bits: 98_304,
        },
        Case {
            name: "half_step_to_zero",
            value: 0.5 * q16_step,
            expected_bits: 0,
        },
        Case {
            name: "positive_half_tie_stays_even",
            value: 2.5 * q16_step,
            expected_bits: 2,
        },
        Case {
            name: "positive_half_tie_up_to_even",
            value: 3.5 * q16_step,
            expected_bits: 4,
        },
        Case {
            name: "negative_half_tie_to_even",
            value: -3.5 * q16_step,
            expected_bits: -4,
        },
        Case {
            name: "largest_subnormal_rounds_to_zero",
            value: f32::from_bits(0x007f_ffff),
            expected_bits: 0,
        },
        Case {
            name: "positive_saturation",
            value: 40_000.0,
            expected_bits: i32::MAX,
        },
        Case {
            name: "negative_saturation",
            value: -40_000.0,
            expected_bits: i32::MIN,
        },
    ];

    for case in cases {
        assert_eq!(
            f32_to_wgt(case.value).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn f32_to_acc_matches_golden_vectors() {
    struct Case {
        name: &'static str,
        value: f32,
        expected_bits: i64,
    }

    let q32_step = 1.0 / 4_294_967_296.0;
    let cases = [
        Case {
            name: "zero",
            value: 0.0,
            expected_bits: 0,
        },
        Case {
            name: "exact_value",
            value: 1.5,
            expected_bits: 6_442_450_944,
        },
        Case {
            name: "half_step_to_zero",
            value: 0.5 * q32_step,
            expected_bits: 0,
        },
        Case {
            name: "positive_half_tie_stays_even",
            value: 2.5 * q32_step,
            expected_bits: 2,
        },
        Case {
            name: "positive_half_tie_up_to_even",
            value: 3.5 * q32_step,
            expected_bits: 4,
        },
        Case {
            name: "tiny_epsilon_survives_q32",
            value: 0.000_001,
            expected_bits: 4_295,
        },
        Case {
            name: "positive_saturation",
            value: 3_000_000_000.0,
            expected_bits: i64::MAX,
        },
        Case {
            name: "negative_saturation",
            value: -3_000_000_000.0,
            expected_bits: i64::MIN,
        },
    ];

    for case in cases {
        assert_eq!(
            f32_to_acc(case.value).to_bits(),
            case.expected_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn f32_to_act_matches_f32_to_wgt_for_q16_16_conversion() {
    let samples = [
        0.0,
        1.5,
        -2.25,
        0.5 * (1.0 / 65_536.0),
        3.5 * (1.0 / 65_536.0),
        40_000.0,
        -40_000.0,
    ];

    for sample in samples {
        assert_eq!(f32_to_act(sample).to_bits(), f32_to_wgt(sample).to_bits());
    }
}

#[test]
fn f32_to_wgt_panics_on_non_finite_values() {
    let nan = panic::catch_unwind(|| f32_to_wgt(f32::NAN));
    assert!(nan.is_err());
    let inf = panic::catch_unwind(|| f32_to_wgt(f32::INFINITY));
    assert!(inf.is_err());
}

#[test]
fn f32_to_act_panics_on_non_finite_values() {
    let nan = panic::catch_unwind(|| f32_to_act(f32::NAN));
    assert!(nan.is_err());
    let inf = panic::catch_unwind(|| f32_to_act(f32::INFINITY));
    assert!(inf.is_err());
}

#[test]
#[should_panic(expected = "argmax_first requires a non-empty slice")]
fn argmax_first_panics_on_empty_slice() {
    let xs: [Act; 0] = [];
    let _ = argmax_first(&xs);
}

#[test]
fn rshift_round_ties_even_panics_when_shift_is_out_of_range() {
    let panic = panic::catch_unwind(|| rshift_round_ties_even(Acc::from_bits(1), 64));
    assert!(panic.is_err());
}
