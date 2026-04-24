use super::types::{Acc, Act, Wgt, REQUANTIZE_SHIFT};

/// Multiplies an activation by a weight in widened accumulator precision.
pub fn mul_wide(a: Act, b: Wgt) -> Acc {
    let product_bits = i64::from(a.to_bits()) * i64::from(b.to_bits());
    Acc::from_bits(product_bits)
}

/// Computes saturating MAC directly on raw fixed-point bit patterns.
pub fn mac_bits(acc_bits: i64, act_bits: i32, wgt_bits: i32) -> i64 {
    acc_bits.saturating_add(i64::from(act_bits) * i64::from(wgt_bits))
}

/// Computes a canonical saturating multiply-accumulate in accumulator precision.
pub fn mac(acc: Acc, a: Act, b: Wgt) -> Acc {
    Acc::from_bits(mac_bits(acc.to_bits(), a.to_bits(), b.to_bits()))
}

/// Adds activation values with saturating overflow semantics.
pub fn add_sat(a: Act, b: Act) -> Act {
    Act::from_bits(a.to_bits().saturating_add(b.to_bits()))
}

/// Multiplies activation values under the canonical requantize-and-saturate contract.
pub fn mul_sat(a: Act, b: Act) -> Act {
    requantize(Acc::from_bits(i64::from(a.to_bits()) * i64::from(b.to_bits())))
}

/// Applies a Q16.16 scalar to an activation under the canonical multiply contract.
pub fn scale_act(value: Act, scalar: Act) -> Act {
    mul_sat(value, scalar)
}

/// Divides an accumulator by an unsigned integer with ties-to-even rounding.
pub fn div_acc_by_u32(value: Acc, divisor: u32) -> Acc {
    assert!(divisor != 0, "div_acc_by_u32 requires a non-zero divisor");

    let dividend = i128::from(value.to_bits());
    let divisor = i128::from(divisor);
    let quotient = dividend / divisor;
    let remainder = dividend % divisor;

    if remainder == 0 {
        return Acc::from_bits(quotient as i64);
    }

    let twice_abs_remainder = remainder.abs() * 2;
    let rounded = if twice_abs_remainder < divisor {
        quotient
    } else if twice_abs_remainder > divisor {
        quotient + if dividend.is_negative() { -1 } else { 1 }
    } else if (quotient & 1) == 0 {
        quotient
    } else {
        quotient + if dividend.is_negative() { -1 } else { 1 }
    };

    Acc::from_bits(clamp_i128_to_i64(rounded))
}

/// Computes deterministic weighted RMSNorm over Q16.16 activations.
pub fn rms_norm(input: &[Act], weight: &[Wgt], eps: Acc) -> Vec<Act> {
    assert!(!input.is_empty(), "rms_norm requires a non-empty input slice");
    assert_eq!(
        input.len(),
        weight.len(),
        "rms_norm requires input and weight slices to have matching widths"
    );

    let scale = rms_norm_scale(input, eps);
    input
        .iter()
        .zip(weight)
        .map(|(value, norm_weight)| {
            let scaled = mul_sat(*value, scale);
            mul_sat(scaled, Act::from_bits(norm_weight.to_bits()))
        })
        .collect()
}

/// Computes deterministic weightless RMS normalization over Q16.16 activations.
pub fn value_rms_norm(input: &[Act], eps: Acc) -> Vec<Act> {
    assert!(
        !input.is_empty(),
        "value_rms_norm requires a non-empty input slice"
    );

    let scale = rms_norm_scale(input, eps);
    input
        .iter()
        .map(|value| mul_sat(*value, scale))
        .collect()
}

/// Returns the canonical deterministic inverse-RMS scale for a row of activations.
pub fn rms_norm_scale(input: &[Act], eps: Acc) -> Act {
    assert!(
        !input.is_empty(),
        "rms_norm_scale requires a non-empty input slice"
    );
    assert!(
        eps.to_bits() >= 0,
        "rms_norm_scale requires a non-negative epsilon"
    );

    let sum_squares = input.iter().fold(Acc::from_bits(0), |acc, value| {
        acc_add_sat(acc, mul_wide(*value, Wgt::from_bits(value.to_bits())))
    });
    let mean_square = div_acc_by_u32(
        sum_squares,
        u32::try_from(input.len()).expect("row width should fit in u32"),
    );
    let adjusted = acc_add_sat(mean_square, eps);
    inv_sqrt_acc(adjusted)
}

/// Subtracts activation values with saturating overflow semantics.
pub fn sub_sat(a: Act, b: Act) -> Act {
    Act::from_bits(a.to_bits().saturating_sub(b.to_bits()))
}

/// Adds accumulator values with saturating overflow semantics.
pub fn acc_add_sat(a: Acc, b: Acc) -> Acc {
    Acc::from_bits(a.to_bits().saturating_add(b.to_bits()))
}

/// Canonically right-shifts with round-to-nearest, ties-to-even semantics.
///
/// Panics if `shift` is greater than or equal to the accumulator bit width.
pub fn rshift_round_ties_even(x: Acc, shift: u32) -> Acc {
    Acc::from_bits(rshift_round_ties_even_bits(x.to_bits(), shift))
}

/// Explicitly narrows an accumulator into an activation with ties-to-even rounding.
pub fn requantize(x: Acc) -> Act {
    let rounded_bits = rshift_round_ties_even_bits(x.to_bits(), REQUANTIZE_SHIFT);
    narrow_act_sat(rounded_bits)
}

/// Saturating activation clip for v0, routed through the canonical narrowing path.
///
/// v0 does not define distinct clipping semantics beyond canonical narrowing, so
/// `clip_act` and `requantize` intentionally share the same behavior.
pub fn clip_act(x: Acc) -> Act {
    // The v0 spec requires this helper but does not distinguish it from requantization,
    // so the smallest contract-preserving implementation shares the same canonical path.
    requantize(x)
}

/// Returns the index of the first maximum encoded activation value.
///
/// Panics if `xs` is empty.
pub fn argmax_first(xs: &[Act]) -> usize {
    assert!(!xs.is_empty(), "argmax_first requires a non-empty slice");

    let mut best_index = 0usize;
    let mut best_bits = xs[0].to_bits();

    for (index, value) in xs.iter().enumerate().skip(1) {
        let bits = value.to_bits();
        if bits > best_bits {
            best_index = index;
            best_bits = bits;
        }
    }

    best_index
}

fn narrow_act_sat(bits: i64) -> Act {
    let clipped = bits.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    Act::from_bits(clipped)
}

fn clamp_i128_to_i64(value: i128) -> i64 {
    value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn inv_sqrt_acc(x: Acc) -> Act {
    let x_bits = x.to_bits();
    assert!(x_bits >= 0, "inv_sqrt_acc requires a non-negative input");
    if x_bits == 0 {
        return Act::from_bits(i32::MAX);
    }

    let target = 1_u128 << 64;
    let divisor = x_bits as u128;
    let max_bits = i32::MAX as u32;

    if inv_sqrt_product(max_bits, divisor) <= target {
        return Act::from_bits(i32::MAX);
    }

    let mut low = 0_u32;
    let mut high = max_bits;
    while low < high {
        let mid = low + ((high - low + 1) / 2);
        if inv_sqrt_product(mid, divisor) <= target {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    let lower_bits = low;
    let upper_bits = lower_bits.saturating_add(1).min(max_bits);
    let lower_product = inv_sqrt_product(lower_bits, divisor);
    let upper_product = inv_sqrt_product(upper_bits, divisor);
    let lower_error = target - lower_product;
    let upper_error = upper_product.saturating_sub(target);

    let rounded_bits = if upper_bits == lower_bits || lower_error < upper_error {
        lower_bits
    } else if upper_error < lower_error {
        upper_bits
    } else if (lower_bits & 1) == 0 {
        lower_bits
    } else {
        upper_bits
    };

    Act::from_bits(rounded_bits as i32)
}

fn inv_sqrt_product(candidate_bits: u32, divisor: u128) -> u128 {
    let candidate = u128::from(candidate_bits);
    candidate
        .saturating_mul(candidate)
        .saturating_mul(divisor)
}

fn rshift_round_ties_even_bits(x: i64, shift: u32) -> i64 {
    if shift == 0 {
        return x;
    }

    assert!(shift < i64::BITS, "shift must be less than 64 bits");

    let divisor = 1_i128 << shift;
    let x = i128::from(x);
    let quotient = x / divisor;
    let remainder = x % divisor;

    if remainder == 0 {
        return quotient as i64;
    }

    let twice_abs_remainder = remainder.abs() * 2;
    let step = if x.is_negative() { -1 } else { 1 };

    let rounded = if twice_abs_remainder < divisor {
        quotient
    } else if twice_abs_remainder > divisor {
        quotient + step
    } else if (quotient & 1) == 0 {
        quotient
    } else {
        quotient + step
    };

    rounded as i64
}
