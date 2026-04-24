use super::types::{Acc, Act, Wgt, ACC_FRACTIONAL_BITS, ACT_FRACTIONAL_BITS, REQUANTIZE_SHIFT};

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

/// Divides one activation-scaled value by another with ties-to-even rounding.
pub fn div_act(numerator: Act, denominator: Act) -> Act {
    assert!(denominator.to_bits() != 0, "div_act requires a non-zero denominator");

    let dividend = i128::from(numerator.to_bits()) << ACT_FRACTIONAL_BITS;
    let divisor = i128::from(denominator.to_bits());
    Act::from_bits(clamp_i128_to_i32(round_ties_even_division(dividend, divisor)))
}

/// Divides one accumulator-scaled value by another with ties-to-even rounding.
fn div_acc(numerator: Acc, denominator: Acc) -> Acc {
    assert!(denominator.to_bits() != 0, "div_acc requires a non-zero denominator");

    let dividend = i128::from(numerator.to_bits()) << ACC_FRACTIONAL_BITS;
    let divisor = i128::from(denominator.to_bits());
    Acc::from_bits(clamp_i128_to_i64(round_ties_even_division(dividend, divisor)))
}

/// Divides an accumulator by an unsigned integer with ties-to-even rounding.
pub fn div_acc_by_u32(value: Acc, divisor: u32) -> Acc {
    assert!(divisor != 0, "div_acc_by_u32 requires a non-zero divisor");

    Acc::from_bits(clamp_i128_to_i64(round_ties_even_division(
        i128::from(value.to_bits()),
        i128::from(divisor),
    )))
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

/// Rotates the RoPE prefix of a row under the deterministic fixed-point contract.
pub fn rope_rotate_pairs(
    input: &[Act],
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Acc,
    position: usize,
) -> Vec<Act> {
    if rotary_dim == 0 {
        return input.to_vec();
    }

    assert!(
        rotary_dim.is_multiple_of(2),
        "rope_rotate_pairs requires an even rotary_dim"
    );
    assert!(
        rotary_dim <= input.len(),
        "rope_rotate_pairs requires rotary_dim to fit within the input width"
    );
    assert!(
        freq_base_dim >= 2 && freq_base_dim.is_multiple_of(2),
        "rope_rotate_pairs requires an even freq_base_dim of at least 2"
    );
    assert!(
        base.to_bits() > 0,
        "rope_rotate_pairs requires a positive base"
    );

    if position == 0 {
        return input.to_vec();
    }

    let half_dim = rotary_dim / 2;
    let mut output = input.to_vec();
    let mut inv_frequency = one_acc();
    let frequency_step = rope_frequency_step(base, freq_base_dim);

    for dim_idx in 0..half_dim {
        let angle = mul_usize_by_acc(position, inv_frequency);
        let (cos, sin) = sin_cos_acc(angle);
        let lhs = input[dim_idx];
        let rhs = input[dim_idx + half_dim];
        output[dim_idx] = sub_sat(mul_sat(lhs, cos), mul_sat(rhs, sin));
        output[dim_idx + half_dim] = add_sat(mul_sat(rhs, cos), mul_sat(lhs, sin));
        inv_frequency = acc_mul(inv_frequency, frequency_step);
    }

    output
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

fn clamp_i128_to_i32(value: i128) -> i32 {
    value.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
}

fn clamp_i128_to_i64(value: i128) -> i64 {
    value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn one_acc() -> Acc {
    Acc::from_bits(1_i64 << ACC_FRACTIONAL_BITS)
}

fn rope_frequency_step(base: Acc, freq_base_dim: usize) -> Acc {
    let root_degree = u32::try_from(freq_base_dim / 2).expect("freq_base_dim should fit in u32");
    let denominator = nth_root_acc(base, root_degree);
    div_acc(one_acc(), denominator)
}

fn nth_root_acc(value: Acc, degree: u32) -> Acc {
    assert!(degree > 0, "nth_root_acc requires a positive degree");
    assert!(value.to_bits() > 0, "nth_root_acc requires a positive input");

    if degree == 1 || value == one_acc() {
        return value;
    }

    let one_bits = one_acc().to_bits();
    let mut low_bits = 0_i64;
    let mut high_bits = value.to_bits().max(one_bits);

    while low_bits < high_bits {
        let mid_bits = low_bits + ((high_bits - low_bits + 1) / 2);
        let mid = Acc::from_bits(mid_bits);
        if pow_acc_u32(mid, degree).to_bits() <= value.to_bits() {
            low_bits = mid_bits;
        } else {
            high_bits = mid_bits - 1;
        }
    }

    let lower_bits = low_bits;
    let upper_bits = lower_bits.saturating_add(1);
    let lower_value = pow_acc_u32(Acc::from_bits(lower_bits), degree).to_bits();
    let upper_value = pow_acc_u32(Acc::from_bits(upper_bits), degree).to_bits();
    let lower_error = (i128::from(value.to_bits()) - i128::from(lower_value)).abs();
    let upper_error = (i128::from(upper_value) - i128::from(value.to_bits())).abs();
    let rounded_bits = if upper_error < lower_error {
        upper_bits
    } else if lower_error < upper_error {
        lower_bits
    } else if (lower_bits & 1) == 0 {
        lower_bits
    } else {
        upper_bits
    };

    Acc::from_bits(rounded_bits)
}

fn pow_acc_u32(base: Acc, exponent: u32) -> Acc {
    let mut result = one_acc();
    for _ in 0..exponent {
        result = acc_mul(result, base);
    }
    result
}

fn mul_usize_by_acc(multiplier: usize, value: Acc) -> Acc {
    let bits = i128::try_from(multiplier).expect("multiplier should fit in i128")
        * i128::from(value.to_bits());
    Acc::from_bits(clamp_i128_to_i64(bits))
}

fn sin_cos_acc(angle: Acc) -> (Act, Act) {
    const HALF_PI_BITS: i64 = 6_746_518_852;
    const PI_BITS: i64 = 13_493_037_705;
    const THREE_HALF_PI_BITS: i64 = 20_239_556_557;
    const TWO_PI_BITS: i64 = 26_986_075_409;

    let reduced = angle.to_bits().rem_euclid(TWO_PI_BITS);
    let (quadrant_angle, sin_sign, cos_sign) = if reduced <= HALF_PI_BITS {
        (Acc::from_bits(reduced), 1_i32, 1_i32)
    } else if reduced <= PI_BITS {
        (Acc::from_bits(PI_BITS - reduced), 1_i32, -1_i32)
    } else if reduced <= THREE_HALF_PI_BITS {
        (Acc::from_bits(reduced - PI_BITS), -1_i32, -1_i32)
    } else {
        (Acc::from_bits(TWO_PI_BITS - reduced), -1_i32, 1_i32)
    };

    let mut sin = requantize(sin_polynomial(quadrant_angle));
    let mut cos = requantize(cos_polynomial(quadrant_angle));
    if sin_sign < 0 {
        sin = Act::from_bits(sin.to_bits().saturating_neg());
    }
    if cos_sign < 0 {
        cos = Act::from_bits(cos.to_bits().saturating_neg());
    }
    (cos, sin)
}

fn sin_polynomial(x: Acc) -> Acc {
    let x2 = acc_mul(x, x);
    let x3 = acc_mul(x2, x);
    let x5 = acc_mul(x3, x2);
    let x7 = acc_mul(x5, x2);

    acc_add_sat(
        acc_add_sat(
            x,
            Acc::from_bits(div_acc_by_u32(x3, 6).to_bits().saturating_neg()),
        ),
        acc_add_sat(
            div_acc_by_u32(x5, 120),
            Acc::from_bits(div_acc_by_u32(x7, 5_040).to_bits().saturating_neg()),
        ),
    )
}

fn cos_polynomial(x: Acc) -> Acc {
    let x2 = acc_mul(x, x);
    let x4 = acc_mul(x2, x2);
    let x6 = acc_mul(x4, x2);

    acc_add_sat(
        acc_add_sat(
            one_acc(),
            Acc::from_bits(div_acc_by_u32(x2, 2).to_bits().saturating_neg()),
        ),
        acc_add_sat(
            div_acc_by_u32(x4, 24),
            Acc::from_bits(div_acc_by_u32(x6, 720).to_bits().saturating_neg()),
        ),
    )
}

fn acc_mul(lhs: Acc, rhs: Acc) -> Acc {
    let product = i128::from(lhs.to_bits()) * i128::from(rhs.to_bits());
    let rounded = round_ties_even_division(product, 1_i128 << ACC_FRACTIONAL_BITS);
    Acc::from_bits(clamp_i128_to_i64(rounded))
}

fn round_ties_even_division(dividend: i128, divisor: i128) -> i128 {
    assert!(divisor != 0, "round_ties_even_division requires a non-zero divisor");

    let quotient = dividend / divisor;
    let remainder = dividend % divisor;
    if remainder == 0 {
        return quotient;
    }

    let twice_abs_remainder = remainder.abs() * 2;
    let abs_divisor = divisor.abs();
    let step = if (dividend < 0) ^ (divisor < 0) { -1 } else { 1 };

    if twice_abs_remainder < abs_divisor {
        quotient
    } else if twice_abs_remainder > abs_divisor {
        quotient + step
    } else if (quotient & 1) == 0 {
        quotient
    } else {
        quotient + step
    }
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
