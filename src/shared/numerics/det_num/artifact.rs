//! detwgt v2 artifact encoding helpers.
//!
//! The byte layout is canonical contract surface, documented in
//! `docs/plans/DETWGT_V2_FORMAT.md`. Storage width is representation only:
//! the canonical value of every weight is its sign-extended integer, all
//! arithmetic occurs after widening to i32/i64, and products are exact and
//! identical regardless of storage width.
//!
//! The production converter (`tools/gemma-det-num-wgt-converter`) streams
//! through the low-level header/padding helpers; tests and dev tools use
//! [`encode_det_wgt_artifact`] to build whole artifacts in memory. Both paths
//! share the same header encoding so the format cannot drift.

use anyhow::{anyhow, bail, Result};

use super::{DET_NUM_SPEC_VERSION, DET_WGT_ARTIFACT_FORMAT_VERSION, DET_WGT_ARTIFACT_MAGIC};

/// Tensor payloads start at file offsets that are multiples of this, so
/// little-endian hosts can borrow zero-copy mmap views of every width.
pub const DET_WGT_PAYLOAD_ALIGNMENT: usize = 64;

/// Per-tensor storage width of a detwgt v2 payload.
///
/// Semantic invariant: a stored element's canonical value is its
/// sign-extended integer; `I16` storage is valid exactly when every value of
/// the tensor fits `i16`, and widening on read reproduces the identical i32
/// bit pattern that `I32` storage would carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetWgtElementWidth {
    I16,
    I32,
}

impl DetWgtElementWidth {
    pub fn byte_width(self) -> usize {
        match self {
            Self::I16 => 2,
            Self::I32 => 4,
        }
    }

    /// The on-disk `element_width` tag (bit count).
    pub fn tag(self) -> u32 {
        match self {
            Self::I16 => 16,
            Self::I32 => 32,
        }
    }

    pub fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            16 => Ok(Self::I16),
            32 => Ok(Self::I32),
            other => bail!("unsupported detwgt element width tag {other}; expected 16 or 32"),
        }
    }
}

/// Returns the storage width a tensor qualifies for: `I16` when every
/// canonical weight bit pattern fits `i16`, else `I32`.
pub fn select_element_width(wgt_bits: &[i32]) -> DetWgtElementWidth {
    if wgt_bits.iter().all(|bits| i16::try_from(*bits).is_ok()) {
        DetWgtElementWidth::I16
    } else {
        DetWgtElementWidth::I32
    }
}

/// Number of zero padding bytes required to advance `offset` to the next
/// payload alignment boundary.
pub fn padding_for_offset(offset: u64) -> usize {
    let alignment = DET_WGT_PAYLOAD_ALIGNMENT as u64;
    ((alignment - offset % alignment) % alignment) as usize
}

/// Encodes the detwgt v2 file header.
pub fn file_header_bytes(tensor_count: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(24);
    bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
    bytes.extend_from_slice(&tensor_count.to_le_bytes());
    bytes
}

/// Encodes one detwgt v2 per-tensor header (without payload padding).
pub fn tensor_header_bytes(
    name: &str,
    shape: &[usize],
    element_width: DetWgtElementWidth,
    max_row_mass: u64,
) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let name_len = u32::try_from(name_bytes.len()).map_err(|_| anyhow!("tensor name too long"))?;
    let rank = u32::try_from(shape.len()).map_err(|_| anyhow!("tensor rank too large"))?;
    let element_count = shape
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim as u64))
        .ok_or_else(|| anyhow!("tensor shape overflowed"))?;
    let payload_len = element_count
        .checked_mul(element_width.byte_width() as u64)
        .ok_or_else(|| anyhow!("tensor payload byte count overflowed"))?;

    let mut bytes = Vec::with_capacity(name_bytes.len() + 8 * shape.len() + 40);
    bytes.extend_from_slice(&name_len.to_le_bytes());
    bytes.extend_from_slice(name_bytes);
    bytes.extend_from_slice(&rank.to_le_bytes());
    for dim in shape {
        bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&element_count.to_le_bytes());
    bytes.extend_from_slice(&element_width.tag().to_le_bytes());
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(&max_row_mass.to_le_bytes());
    Ok(bytes)
}

/// Encodes one canonical weight bit pattern at the given storage width.
pub fn encode_wgt_bits(bits: i32, element_width: DetWgtElementWidth, out: &mut Vec<u8>) {
    match element_width {
        DetWgtElementWidth::I16 => {
            let narrow =
                i16::try_from(bits).expect("i16 storage width requires every value to fit i16");
            out.extend_from_slice(&narrow.to_le_bytes());
        }
        DetWgtElementWidth::I32 => out.extend_from_slice(&bits.to_le_bytes()),
    }
}

/// Computes the per-tensor max row mass over last-dimension rows
/// (`max_r sum_i |wgt_bits[r][i]|`), matching the converter's audit metric.
pub fn max_row_mass(shape: &[usize], wgt_bits: &[i32]) -> u64 {
    let row_len = shape.last().copied().unwrap_or(1).max(1);
    wgt_bits
        .chunks(row_len)
        .map(|row| {
            row.iter()
                .map(|bits| u64::from(bits.unsigned_abs()))
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0)
}

/// One tensor of an in-memory detwgt v2 artifact, carrying canonical
/// (widened) weight bit patterns.
#[derive(Debug, Clone)]
pub struct DetWgtTensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub wgt_bits: Vec<i32>,
}

/// Per-tensor storage width policy for [`encode_det_wgt_artifact_with_widths`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetWgtWidthPolicy {
    /// Store each tensor as i16 when every value fits, else i32.
    Auto,
    /// Store every tensor as i32 regardless of value range.
    ForceI32,
}

/// Builds a complete detwgt v2 artifact in memory (test/dev fixture path).
pub fn encode_det_wgt_artifact(tensors: &[DetWgtTensorSpec]) -> Result<Vec<u8>> {
    encode_det_wgt_artifact_with_widths(tensors, DetWgtWidthPolicy::Auto)
}

/// Builds a complete detwgt v2 artifact in memory with an explicit width
/// policy. `ForceI32` exists so conformance tests can compare the same
/// tensor stored both ways.
pub fn encode_det_wgt_artifact_with_widths(
    tensors: &[DetWgtTensorSpec],
    policy: DetWgtWidthPolicy,
) -> Result<Vec<u8>> {
    let mut bytes = file_header_bytes(tensors.len() as u64);
    for tensor in tensors {
        let element_count = tensor
            .shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| anyhow!("tensor shape overflowed"))?;
        if element_count != tensor.wgt_bits.len() {
            bail!(
                "tensor `{}` has {} values but shape product {element_count}",
                tensor.name,
                tensor.wgt_bits.len()
            );
        }
        let element_width = match policy {
            DetWgtWidthPolicy::Auto => select_element_width(&tensor.wgt_bits),
            DetWgtWidthPolicy::ForceI32 => DetWgtElementWidth::I32,
        };
        let row_mass = max_row_mass(&tensor.shape, &tensor.wgt_bits);
        bytes.extend_from_slice(&tensor_header_bytes(
            &tensor.name,
            &tensor.shape,
            element_width,
            row_mass,
        )?);
        bytes.resize(bytes.len() + padding_for_offset(bytes.len() as u64), 0);
        for bits in &tensor.wgt_bits {
            encode_wgt_bits(*bits, element_width, &mut bytes);
        }
    }
    Ok(bytes)
}

/// Decodes a little-endian payload row of `element_width` storage into
/// canonical (sign-extended) i32 weight bit patterns.
pub fn decode_wgt_bits_le(encoded: &[u8], element_width: DetWgtElementWidth) -> Result<Vec<i32>> {
    let byte_width = element_width.byte_width();
    if encoded.len() % byte_width != 0 {
        bail!(
            "encoded weight payload length {} is not a multiple of element width {byte_width}",
            encoded.len()
        );
    }
    let mut bits = Vec::with_capacity(encoded.len() / byte_width);
    match element_width {
        DetWgtElementWidth::I16 => {
            for chunk in encoded.chunks_exact(2) {
                bits.push(i32::from(i16::from_le_bytes(
                    chunk.try_into().expect("i16 byte width should match"),
                )));
            }
        }
        DetWgtElementWidth::I32 => {
            for chunk in encoded.chunks_exact(4) {
                bits.push(i32::from_le_bytes(
                    chunk.try_into().expect("i32 byte width should match"),
                ));
            }
        }
    }
    Ok(bits)
}
