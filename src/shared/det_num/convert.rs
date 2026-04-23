use super::types::{Act, Wgt, ACT_FRACTIONAL_BITS};

/// Converts a finite FP32 value into the canonical `det_num` v0 Q16.16 activation.
pub fn f32_to_act(x: f32) -> Act {
    assert!(x.is_finite(), "f32_to_act requires a finite source value");
    Act::from_bits(f32_to_q16_16_bits(x))
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
    let bits = x.to_bits();
    let is_negative = (bits >> 31) != 0;
    let exponent_bits = ((bits >> 23) & 0xff) as i32;
    let fraction_bits = bits & 0x7f_ff_ff;

    if exponent_bits == 0 && fraction_bits == 0 {
        return 0;
    }

    let (significand, exponent) = if exponent_bits == 0 {
        (u64::from(fraction_bits), -149)
    } else {
        (
            u64::from((1_u32 << 23) | fraction_bits),
            exponent_bits - 127 - 23,
        )
    };
    let scaled_exponent = exponent + ACT_FRACTIONAL_BITS as i32;
    let max_magnitude = if is_negative {
        1_u64 << 31
    } else {
        i32::MAX as u64
    };

    let magnitude = if scaled_exponent >= 0 {
        saturating_shift_left(significand, scaled_exponent as u32, max_magnitude)
    } else {
        round_ties_even_div_pow2(significand, (-scaled_exponent) as u32).min(max_magnitude)
    };

    if is_negative {
        if magnitude >= (1_u64 << 31) {
            i32::MIN
        } else {
            -(magnitude as i64) as i32
        }
    } else {
        magnitude as i32
    }
}

fn saturating_shift_left(value: u64, shift: u32, max_magnitude: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    if shift >= 64 {
        return max_magnitude;
    }

    let threshold = max_magnitude >> shift;
    if value > threshold {
        max_magnitude
    } else {
        value << shift
    }
}

fn round_ties_even_div_pow2(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    if shift >= 64 {
        return 0;
    }

    let quotient = value >> shift;
    let remainder_mask = (1_u64 << shift) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1_u64 << (shift - 1);

    if remainder > halfway || (remainder == halfway && (quotient & 1) == 1) {
        quotient + 1
    } else {
        quotient
    }
}
