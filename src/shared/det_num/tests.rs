use std::{mem::size_of, panic};

use super::{
    acc_add_sat, acc_to_le_bytes, act_to_le_bytes, add_sat, argmax_first, clip_act, mac, mul_wide,
    requantize, rshift_round_ties_even, sub_sat, types::ACC_FRACTIONAL_BITS,
    types::ACT_FRACTIONAL_BITS, types::REQUANTIZE_SHIFT, Acc, Act, Wgt,
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
