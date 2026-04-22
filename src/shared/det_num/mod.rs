mod ops;
mod serialize;
mod types;

pub use self::ops::{
    acc_add_sat, add_sat, argmax_first, clip_act, mac, mul_wide, requantize,
    rshift_round_ties_even, sub_sat,
};
pub use self::serialize::{acc_to_le_bytes, act_to_le_bytes};
pub use self::types::{Acc, Act, Wgt};

#[cfg(test)]
mod tests;
