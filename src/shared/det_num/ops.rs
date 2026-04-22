use super::types::{Acc, Act, Wgt, REQUANTIZE_SHIFT};

/// Multiplies an activation by a weight in widened accumulator precision.
pub fn mul_wide(a: Act, b: Wgt) -> Acc {
    let product_bits = i64::from(a.to_bits()) * i64::from(b.to_bits());
    Acc::from_bits(product_bits)
}

/// Computes a canonical saturating multiply-accumulate in accumulator precision.
pub fn mac(acc: Acc, a: Act, b: Wgt) -> Acc {
    acc_add_sat(acc, mul_wide(a, b))
}

/// Adds activation values with saturating overflow semantics.
pub fn add_sat(a: Act, b: Act) -> Act {
    Act::from_bits(a.to_bits().saturating_add(b.to_bits()))
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
