//! Native-only SIMD MAC kernels for the deterministic inference path.
//!
//! Under DET_NUM_SPEC v1 the MAC reduction is a wrapping i64 sum of exact
//! i32×i32→i64 products, which is associative and commutative; any lane
//! arrangement, accumulator split, or horizontal-reduction tree built from
//! wrapping i64 adds therefore produces bit-identical results to the serial
//! scalar fold. Every kernel here computes the same multiset of exact product
//! terms as the canonical loops in `det_num` and combines them exclusively
//! with wrapping i64 addition.
//!
//! This module is pure schedule: it never touches `requantize` or any
//! saturating operation. Callers narrow the final accumulator exactly as the
//! scalar reference does.
//!
//! Guest-profile modules (`det_num`, `raster_kernels`, `routines/*/raster`)
//! must never reach this module; the `simd_lint` test denies `std::arch`
//! there.

/// Kernel backend selected once per process.
///
/// `RASTER_DET_KERNEL_BACKEND=scalar` forces the canonical scalar loops (the
/// standing oracle for differential debugging and guest-parity runs);
/// `=neon` requests NEON explicitly; unset/`auto` detects at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

fn detect_backend() -> KernelBackend {
    let requested = std::env::var("RASTER_DET_KERNEL_BACKEND")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_else(|_| String::from("auto"));
    match requested.as_str() {
        "scalar" => KernelBackend::Scalar,
        "neon" => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return KernelBackend::Neon;
                }
            }
            eprintln!(
                "RASTER_DET_KERNEL_BACKEND=neon requested but NEON is unavailable; \
                 falling back to scalar (results are identical)"
            );
            KernelBackend::Scalar
        }
        _ => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return KernelBackend::Neon;
                }
            }
            KernelBackend::Scalar
        }
    }
}

pub fn kernel_backend() -> KernelBackend {
    static BACKEND: std::sync::OnceLock<KernelBackend> = std::sync::OnceLock::new();
    *BACKEND.get_or_init(detect_backend)
}

// ---------------------------------------------------------------------------
// Scalar reference kernels (canonical serial fold; the oracle)
// ---------------------------------------------------------------------------

fn dot_i32_scalar(acts: &[i32], wgts: &[i32]) -> i64 {
    debug_assert_eq!(acts.len(), wgts.len());
    let mut acc = 0_i64;
    for (act, wgt) in acts.iter().zip(wgts) {
        acc = acc.wrapping_add(i64::from(*act) * i64::from(*wgt));
    }
    acc
}

fn dot_i32_i16_scalar(acts: &[i32], wgts: &[i16]) -> i64 {
    debug_assert_eq!(acts.len(), wgts.len());
    let mut acc = 0_i64;
    for (act, wgt) in acts.iter().zip(wgts) {
        acc = acc.wrapping_add(i64::from(*act) * i64::from(*wgt));
    }
    acc
}

fn axpy_acc_scalar(acc: &mut [i64], row: &[i32], weight: i32) {
    debug_assert_eq!(acc.len(), row.len());
    for (acc_value, row_value) in acc.iter_mut().zip(row) {
        *acc_value = acc_value.wrapping_add(i64::from(*row_value) * i64::from(weight));
    }
}

// ---------------------------------------------------------------------------
// NEON kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Wrapping i64 dot product of i32 slices: eight independent
    /// `int64x2_t` lane accumulators (16 elements/iteration via `vmlal_s32`,
    /// the exact widening i32×i32→i64 MAC) so the 3-cycle MAC latency is
    /// hidden, combined and horizontally reduced with wrapping i64 adds; the
    /// tail uses the scalar wrapping MAC.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot_i32(acts: &[i32], wgts: &[i32]) -> i64 {
        debug_assert_eq!(acts.len(), wgts.len());
        let len = acts.len();
        let act_ptr = acts.as_ptr();
        let wgt_ptr = wgts.as_ptr();
        let mut acc0 = vdupq_n_s64(0);
        let mut acc1 = vdupq_n_s64(0);
        let mut acc2 = vdupq_n_s64(0);
        let mut acc3 = vdupq_n_s64(0);
        let mut acc4 = vdupq_n_s64(0);
        let mut acc5 = vdupq_n_s64(0);
        let mut acc6 = vdupq_n_s64(0);
        let mut acc7 = vdupq_n_s64(0);
        let chunks = len / 16;
        for chunk_idx in 0..chunks {
            let base = chunk_idx * 16;
            let act0 = vld1q_s32(act_ptr.add(base));
            let act1 = vld1q_s32(act_ptr.add(base + 4));
            let act2 = vld1q_s32(act_ptr.add(base + 8));
            let act3 = vld1q_s32(act_ptr.add(base + 12));
            let wgt0 = vld1q_s32(wgt_ptr.add(base));
            let wgt1 = vld1q_s32(wgt_ptr.add(base + 4));
            let wgt2 = vld1q_s32(wgt_ptr.add(base + 8));
            let wgt3 = vld1q_s32(wgt_ptr.add(base + 12));
            acc0 = vmlal_s32(acc0, vget_low_s32(act0), vget_low_s32(wgt0));
            acc1 = vmlal_high_s32(acc1, act0, wgt0);
            acc2 = vmlal_s32(acc2, vget_low_s32(act1), vget_low_s32(wgt1));
            acc3 = vmlal_high_s32(acc3, act1, wgt1);
            acc4 = vmlal_s32(acc4, vget_low_s32(act2), vget_low_s32(wgt2));
            acc5 = vmlal_high_s32(acc5, act2, wgt2);
            acc6 = vmlal_s32(acc6, vget_low_s32(act3), vget_low_s32(wgt3));
            acc7 = vmlal_high_s32(acc7, act3, wgt3);
        }
        // NEON integer adds are modular (two's-complement wrapping).
        let acc = vaddq_s64(
            vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3)),
            vaddq_s64(vaddq_s64(acc4, acc5), vaddq_s64(acc6, acc7)),
        );
        let mut total =
            vgetq_lane_s64::<0>(acc).wrapping_add(vgetq_lane_s64::<1>(acc));
        for idx in chunks * 16..len {
            total = total.wrapping_add(
                i64::from(*acts.get_unchecked(idx)) * i64::from(*wgts.get_unchecked(idx)),
            );
        }
        total
    }

    /// i16-weight twin of [`dot_i32`]: weights are sign-extended to i32 on
    /// load (`vmovl_s16`), then run through the identical exact widening MAC
    /// path — storage width never changes a product bit.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot_i32_i16(acts: &[i32], wgts: &[i16]) -> i64 {
        debug_assert_eq!(acts.len(), wgts.len());
        let len = acts.len();
        let act_ptr = acts.as_ptr();
        let wgt_ptr = wgts.as_ptr();
        let mut acc0 = vdupq_n_s64(0);
        let mut acc1 = vdupq_n_s64(0);
        let mut acc2 = vdupq_n_s64(0);
        let mut acc3 = vdupq_n_s64(0);
        let mut acc4 = vdupq_n_s64(0);
        let mut acc5 = vdupq_n_s64(0);
        let mut acc6 = vdupq_n_s64(0);
        let mut acc7 = vdupq_n_s64(0);
        let chunks = len / 16;
        for chunk_idx in 0..chunks {
            let base = chunk_idx * 16;
            let act0 = vld1q_s32(act_ptr.add(base));
            let act1 = vld1q_s32(act_ptr.add(base + 4));
            let act2 = vld1q_s32(act_ptr.add(base + 8));
            let act3 = vld1q_s32(act_ptr.add(base + 12));
            let wgt_narrow0 = vld1q_s16(wgt_ptr.add(base));
            let wgt_narrow1 = vld1q_s16(wgt_ptr.add(base + 8));
            let wgt0 = vmovl_s16(vget_low_s16(wgt_narrow0));
            let wgt1 = vmovl_high_s16(wgt_narrow0);
            let wgt2 = vmovl_s16(vget_low_s16(wgt_narrow1));
            let wgt3 = vmovl_high_s16(wgt_narrow1);
            acc0 = vmlal_s32(acc0, vget_low_s32(act0), vget_low_s32(wgt0));
            acc1 = vmlal_high_s32(acc1, act0, wgt0);
            acc2 = vmlal_s32(acc2, vget_low_s32(act1), vget_low_s32(wgt1));
            acc3 = vmlal_high_s32(acc3, act1, wgt1);
            acc4 = vmlal_s32(acc4, vget_low_s32(act2), vget_low_s32(wgt2));
            acc5 = vmlal_high_s32(acc5, act2, wgt2);
            acc6 = vmlal_s32(acc6, vget_low_s32(act3), vget_low_s32(wgt3));
            acc7 = vmlal_high_s32(acc7, act3, wgt3);
        }
        let acc = vaddq_s64(
            vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3)),
            vaddq_s64(vaddq_s64(acc4, acc5), vaddq_s64(acc6, acc7)),
        );
        let mut total =
            vgetq_lane_s64::<0>(acc).wrapping_add(vgetq_lane_s64::<1>(acc));
        for idx in chunks * 16..len {
            total = total.wrapping_add(
                i64::from(*acts.get_unchecked(idx)) * i64::from(*wgts.get_unchecked(idx)),
            );
        }
        total
    }

    /// `acc[d] += row[d] * weight` with wrapping i64 accumulation: the
    /// row-major inner step of the attention weighted sum. Per output
    /// element this contributes the same exact product as the canonical
    /// column-major loop; wrapping adds make the row/column order swap
    /// bit-identical.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn axpy_acc(acc: &mut [i64], row: &[i32], weight: i32) {
        debug_assert_eq!(acc.len(), row.len());
        let len = acc.len();
        let acc_ptr = acc.as_mut_ptr();
        let row_ptr = row.as_ptr();
        let weight_pair = vdup_n_s32(weight);
        let chunks = len / 4;
        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;
            let values = vld1q_s32(row_ptr.add(base));
            let acc_lo = vld1q_s64(acc_ptr.add(base));
            let acc_hi = vld1q_s64(acc_ptr.add(base + 2));
            let acc_lo = vmlal_s32(acc_lo, vget_low_s32(values), weight_pair);
            let acc_hi = vmlal_s32(acc_hi, vget_high_s32(values), weight_pair);
            vst1q_s64(acc_ptr.add(base), acc_lo);
            vst1q_s64(acc_ptr.add(base + 2), acc_hi);
        }
        for idx in chunks * 4..len {
            let acc_value = acc.get_unchecked_mut(idx);
            *acc_value = acc_value
                .wrapping_add(i64::from(*row.get_unchecked(idx)) * i64::from(weight));
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatching entry points
// ---------------------------------------------------------------------------

/// Wrapping i64 dot product of exact i32×i32→i64 products (GEMV row dot and
/// attention score core). Bit-identical across backends.
pub fn dot_act_i32(acts: &[i32], wgts: &[i32]) -> i64 {
    match kernel_backend() {
        KernelBackend::Scalar => dot_i32_scalar(acts, wgts),
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON presence was verified during backend selection.
        KernelBackend::Neon => unsafe { neon::dot_i32(acts, wgts) },
    }
}

/// i16-weight twin of [`dot_act_i32`]; weights are sign-extended before the
/// exact widening multiply, so products are identical to i32 storage.
pub fn dot_act_i16(acts: &[i32], wgts: &[i16]) -> i64 {
    match kernel_backend() {
        KernelBackend::Scalar => dot_i32_i16_scalar(acts, wgts),
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON presence was verified during backend selection.
        KernelBackend::Neon => unsafe { neon::dot_i32_i16(acts, wgts) },
    }
}

/// `acc[d] = acc[d] (+wrap) row[d] * weight` over a full row (attention
/// weighted-sum inner step). Bit-identical across backends.
pub fn axpy_acc_i64(acc: &mut [i64], row: &[i32], weight: i32) {
    match kernel_backend() {
        KernelBackend::Scalar => axpy_acc_scalar(acc, row, weight),
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON presence was verified during backend selection.
        KernelBackend::Neon => unsafe { neon::axpy_acc(acc, row, weight) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* generator for fuzz inputs.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut state = self.0;
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            self.0 = state;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Adversarial i32 patterns: extremes, saturation-adjacent activations,
    /// i16 boundary values, and sign-bit edge cases.
    const ADVERSARIAL_I32: &[i32] = &[
        0,
        1,
        -1,
        i32::MAX,
        i32::MIN,
        i32::MAX - 1,
        i32::MIN + 1,
        0x7FFF,  // i16::MAX
        -0x8000, // i16::MIN
        0x8000,  // first value outside i16
        -0x8001,
        1 << 16,
        -(1 << 16),
        0x5555_5555,
        -0x5555_5556,
    ];

    const ADVERSARIAL_I16: &[i16] = &[0, 1, -1, i16::MAX, i16::MIN, i16::MAX - 1, i16::MIN + 1];

    /// Mixes full-range random values with adversarial picks.
    fn fuzz_i32(rng: &mut Rng, len: usize) -> Vec<i32> {
        (0..len)
            .map(|_| {
                let roll = rng.next();
                if roll % 4 == 0 {
                    ADVERSARIAL_I32[(roll >> 8) as usize % ADVERSARIAL_I32.len()]
                } else {
                    (roll >> 16) as i32
                }
            })
            .collect()
    }

    fn fuzz_i16(rng: &mut Rng, len: usize) -> Vec<i16> {
        (0..len)
            .map(|_| {
                let roll = rng.next();
                if roll % 4 == 0 {
                    ADVERSARIAL_I16[(roll >> 8) as usize % ADVERSARIAL_I16.len()]
                } else {
                    (roll >> 16) as i16
                }
            })
            .collect()
    }

    /// Every tail length around the 8-wide (dot) and 4-wide (axpy) kernel
    /// strides, plus longer GEMV-realistic lengths.
    fn fuzz_lengths() -> impl Iterator<Item = usize> {
        (0..=40).chain([63, 64, 65, 127, 128, 129, 255, 256, 1000])
    }

    #[test]
    fn scalar_dot_matches_mac_bits_fold() {
        use crate::shared::numerics::det_num::mac_bits;
        let mut rng = Rng(1);
        let acts = fuzz_i32(&mut rng, 133);
        let wgts = fuzz_i32(&mut rng, 133);
        let reference = acts
            .iter()
            .zip(&wgts)
            .fold(0_i64, |acc, (act, wgt)| mac_bits(acc, *act, *wgt));
        assert_eq!(dot_i32_scalar(&acts, &wgts), reference);
    }

    #[test]
    fn scalar_i16_dot_matches_widened_i32_dot() {
        let mut rng = Rng(2);
        for len in fuzz_lengths() {
            let acts = fuzz_i32(&mut rng, len);
            let wgts = fuzz_i16(&mut rng, len);
            let widened = wgts.iter().map(|wgt| i32::from(*wgt)).collect::<Vec<_>>();
            assert_eq!(
                dot_i32_i16_scalar(&acts, &wgts),
                dot_i32_scalar(&acts, &widened),
                "i16 storage diverged from widened i32 at len {len}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_dot_matches_scalar_fuzz() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for seed in 1..=8_u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for len in fuzz_lengths() {
                let acts = fuzz_i32(&mut rng, len);
                let wgts = fuzz_i32(&mut rng, len);
                let reference = dot_i32_scalar(&acts, &wgts);
                let simd = unsafe { neon::dot_i32(&acts, &wgts) };
                assert_eq!(simd, reference, "NEON dot diverged at len {len} seed {seed}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i16_dot_matches_scalar_fuzz() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for seed in 1..=8_u64 {
            let mut rng = Rng(seed.wrapping_mul(0xD134_2543_DE82_EF95));
            for len in fuzz_lengths() {
                let acts = fuzz_i32(&mut rng, len);
                let wgts = fuzz_i16(&mut rng, len);
                let reference = dot_i32_i16_scalar(&acts, &wgts);
                let simd = unsafe { neon::dot_i32_i16(&acts, &wgts) };
                assert_eq!(
                    simd, reference,
                    "NEON i16 dot diverged at len {len} seed {seed}"
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_axpy_matches_scalar_fuzz() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for seed in 1..=8_u64 {
            let mut rng = Rng(seed.wrapping_mul(0xA076_1D64_78BD_642F));
            for len in fuzz_lengths() {
                let row = fuzz_i32(&mut rng, len);
                let weight =
                    ADVERSARIAL_I32[(rng.next() as usize) % ADVERSARIAL_I32.len()];
                let mut reference = fuzz_i32(&mut rng, len)
                    .into_iter()
                    .map(i64::from)
                    .map(|value| value.wrapping_mul(0x0123_4567_89AB_CDEF))
                    .collect::<Vec<_>>();
                let mut simd = reference.clone();
                axpy_acc_scalar(&mut reference, &row, weight);
                unsafe { neon::axpy_acc(&mut simd, &row, weight) };
                assert_eq!(
                    simd, reference,
                    "NEON axpy diverged at len {len} seed {seed}"
                );
            }
        }
    }

    /// Wraparound-engineered reduction: terms chosen so partial sums overflow
    /// i64 in both directions; lane splits must still agree exactly under
    /// wrapping arithmetic.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_dot_matches_scalar_under_engineered_i64_wraparound() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        // Each product is i32::MIN * i32::MIN = 2^62; five of them wrap i64.
        for len in [5_usize, 8, 9, 16, 17, 33] {
            let acts = vec![i32::MIN; len];
            let wgts = vec![i32::MIN; len];
            let reference = dot_i32_scalar(&acts, &wgts);
            let simd = unsafe { neon::dot_i32(&acts, &wgts) };
            assert_eq!(simd, reference, "wraparound dot diverged at len {len}");
        }
    }
}
