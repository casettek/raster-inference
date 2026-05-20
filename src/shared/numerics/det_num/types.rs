use fixed::types::{I16F16, I32F32};

/// Canonical activation scalar type for `det_num` v0.
///
/// The spec names this layout `I32F16`: a signed 32-bit fixed-point value
/// with 16 fractional bits. In the `fixed` crate, the equivalent concrete
/// type is `I16F16`, where the prefix counts integer bits rather than total bits.
pub type Act = I16F16;

/// Canonical weight scalar type for `det_num` v0.
///
/// This matches the spec's "signed 32-bit fixed-point with 16 fractional bits"
/// contract, represented by `fixed::types::I16F16`.
pub type Wgt = I16F16;

/// Canonical accumulator scalar type for `det_num` v0.
///
/// The spec names this layout `I64F32`: a signed 64-bit fixed-point value
/// with 32 fractional bits. In the `fixed` crate, the equivalent concrete
/// type is `I32F32`.
pub type Acc = I32F32;

pub(crate) const ACT_FRACTIONAL_BITS: u32 = 16;
pub(crate) const ACC_FRACTIONAL_BITS: u32 = 32;
pub(crate) const REQUANTIZE_SHIFT: u32 = ACC_FRACTIONAL_BITS - ACT_FRACTIONAL_BITS;
