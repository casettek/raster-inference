mod convert;
mod ops;
mod serialize;
mod types;

pub const DET_WGT_ARTIFACT_MAGIC: &[u8; 8] = b"DNWGTV0\0";
pub const DET_WGT_ARTIFACT_FORMAT_VERSION: u32 = 0;
pub const DET_NUM_SPEC_VERSION: u32 = 0;

pub use self::convert::{f32_to_act, f32_to_wgt};
pub use self::ops::{
    acc_add_sat, add_sat, argmax_first, clip_act, mac, mac_bits, mul_wide, requantize,
    rshift_round_ties_even, sub_sat,
};
pub use self::serialize::{acc_to_le_bytes, act_to_le_bytes, wgt_to_le_bytes};
pub use self::types::{Acc, Act, Wgt};

#[cfg(test)]
mod tests;
