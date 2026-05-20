use super::types::{Acc, Act, Wgt, ACC_FRACTIONAL_BITS, ACT_FRACTIONAL_BITS};

/// Converts a finite FP32 value into the canonical `det_num` v0 Q16.16 activation.
pub fn f32_to_act(x: f32) -> Act {
    assert!(x.is_finite(), "f32_to_act requires a finite source value");
    Act::from_bits(f32_to_q16_16_bits(x))
}

/// Converts a finite FP32 value into the canonical `det_num` v0 Q32.32 accumulator.
pub fn f32_to_acc(x: f32) -> Acc {
    assert!(x.is_finite(), "f32_to_acc requires a finite source value");
    Acc::from_bits(f32_to_q32_32_bits(x))
}

/// Converts a canonical Q16.16 activation back into FP32 for host-side APIs.
pub fn act_to_f32(x: Act) -> f32 {
    x.to_bits() as f32 / (1_u32 << ACT_FRACTIONAL_BITS) as f32
}

/// Converts a finite FP32 value into the canonical `det_num` v0 Q16.16 weight.
///
/// This implements `sat_i32(round_ties_even(x * 65536.0))` exactly from the
/// source float's IEEE-754 payload so the result is deterministic and does not
/// depend on host floating-point rounding behavior.
pub fn f32_to_wgt(x: f32) -> Wgt {
    assert!(x.is_finite(), "f32_to_wgt requires a finite source value");
    Wgt::from_bits(f32_to_q16_16_bits(x))
}

fn f32_to_q16_16_bits(x: f32) -> i32 {
    let bits = f32_to_fixed_bits(x, ACT_FRACTIONAL_BITS, i32::MAX as u128, 1_u128 << 31);
    if bits.is_negative() {
        if bits <= i128::from(i32::MIN) {
            i32::MIN
        } else {
            bits as i32
        }
    } else {
        bits.min(i128::from(i32::MAX)) as i32
    }
}

fn f32_to_q32_32_bits(x: f32) -> i64 {
    let bits = f32_to_fixed_bits(x, ACC_FRACTIONAL_BITS, i64::MAX as u128, 1_u128 << 63);
    if bits.is_negative() {
        if bits <= i128::from(i64::MIN) {
            i64::MIN
        } else {
            bits as i64
        }
    } else {
        bits.min(i128::from(i64::MAX)) as i64
    }
}

fn f32_to_fixed_bits(
    x: f32,
    fractional_bits: u32,
    max_positive_magnitude: u128,
    max_negative_magnitude: u128,
) -> i128 {
    let bits = x.to_bits();
    let is_negative = (bits >> 31) != 0;
    let exponent_bits = ((bits >> 23) & 0xff) as i32;
    let fraction_bits = bits & 0x7f_ff_ff;

    if exponent_bits == 0 && fraction_bits == 0 {
        return 0;
    }

    let (significand, exponent) = if exponent_bits == 0 {
        (u128::from(fraction_bits), -149)
    } else {
        (
            u128::from((1_u32 << 23) | fraction_bits),
            exponent_bits - 127 - 23,
        )
    };
    let scaled_exponent = exponent + fractional_bits as i32;
    let max_magnitude = if is_negative {
        max_negative_magnitude
    } else {
        max_positive_magnitude
    };

    let magnitude = if scaled_exponent >= 0 {
        saturating_shift_left(significand, scaled_exponent as u32, max_magnitude)
    } else {
        round_ties_even_div_pow2(significand, (-scaled_exponent) as u32).min(max_magnitude)
    };

    if is_negative {
        if magnitude >= max_negative_magnitude {
            -(max_negative_magnitude as i128)
        } else {
            -(magnitude as i128)
        }
    } else {
        magnitude as i128
    }
}

fn saturating_shift_left(value: u128, shift: u32, max_magnitude: u128) -> u128 {
    if value == 0 {
        return 0;
    }
    if shift >= 128 {
        return max_magnitude;
    }

    let threshold = max_magnitude >> shift;
    if value > threshold {
        max_magnitude
    } else {
        value << shift
    }
}

fn round_ties_even_div_pow2(value: u128, shift: u32) -> u128 {
    if shift == 0 {
        return value;
    }
    if shift >= 128 {
        return 0;
    }

    let quotient = value >> shift;
    let remainder_mask = (1_u128 << shift) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1_u128 << (shift - 1);

    if remainder > halfway || (remainder == halfway && (quotient & 1) == 1) {
        quotient + 1
    } else {
        quotient
    }
}
