use super::types::{Acc, Act, Wgt};

/// Serializes an activation using canonical little-endian fixed-width encoding.
pub fn act_to_le_bytes(x: Act) -> [u8; 4] {
    x.to_bits().to_le_bytes()
}

/// Serializes a weight using canonical little-endian fixed-width encoding.
pub fn wgt_to_le_bytes(x: Wgt) -> [u8; 4] {
    x.to_bits().to_le_bytes()
}

/// Serializes an accumulator using canonical little-endian fixed-width encoding.
pub fn acc_to_le_bytes(x: Acc) -> [u8; 8] {
    x.to_bits().to_le_bytes()
}
