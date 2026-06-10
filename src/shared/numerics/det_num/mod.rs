mod convert;
mod ops;
mod serialize;
mod types;

pub const DET_WGT_ARTIFACT_MAGIC: &[u8; 8] = b"DNWGTV0\0";
pub const DET_WGT_ARTIFACT_FORMAT_VERSION: u32 = 1;
pub const DET_NUM_SPEC_VERSION: u32 = 1;

/// Normative conversion-time overflow bound on weight rows (spec v1):
/// `sum_i |wgt_bits[r][i]| < 2^31` for every output row `r`, which guarantees
/// `sum_i |wgt_bits[r][i]| * A_MAX < 2^62` for any representable activation.
pub const DET_WGT_ROW_MASS_LIMIT: u64 = 1 << 31;

pub use self::convert::{
    act_to_f32, enter_det_single_track_region, f32_to_acc, f32_to_act, f32_to_wgt,
    DetSingleTrackRegionGuard,
};
pub use self::ops::{
    acc_add_sat, acc_combine, add_sat, argmax_first, attention_score, attention_softmax,
    attention_softmax_exp_term, attention_softmax_into, attention_softmax_raw_weight,
    attention_softmax_residual, attention_weighted_sum, attention_weighted_sum_flat_into, clip_act,
    div_acc_by_u32, div_act, gelu_pytorch_tanh_act, mac, mac_bits, mul_sat, mul_wide, requantize,
    rms_norm, rms_norm_in_place, rms_norm_scale, rope_rotate_pairs, rope_rotate_pairs_in_place,
    rshift_round_ties_even, scale_act, softcap_act, sub_sat, tanh_act, value_rms_norm,
    value_rms_norm_in_place,
};
pub use self::serialize::{acc_to_le_bytes, act_to_le_bytes, wgt_to_le_bytes};
pub use self::types::{Acc, Act, Wgt};

#[cfg(test)]
mod tests;
