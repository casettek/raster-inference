mod convert;
mod ops;
mod serialize;
mod types;

pub const DET_WGT_ARTIFACT_MAGIC: &[u8; 8] = b"DNWGTV0\0";
pub const DET_WGT_ARTIFACT_FORMAT_VERSION: u32 = 0;
pub const DET_NUM_SPEC_VERSION: u32 = 0;

pub use self::convert::{act_to_f32, f32_to_acc, f32_to_act, f32_to_wgt};
pub use self::ops::{
    acc_add_sat, add_sat, argmax_first, clip_act, div_acc_by_u32, div_act, mac, mac_bits,
    mul_sat, mul_wide, requantize, rms_norm, rms_norm_scale, rope_rotate_pairs,
    rshift_round_ties_even, scale_act, sub_sat, value_rms_norm,
};
pub use self::serialize::{acc_to_le_bytes, act_to_le_bytes, wgt_to_le_bytes};
pub use self::types::{Acc, Act, Wgt};

#[cfg(test)]
mod tests;
