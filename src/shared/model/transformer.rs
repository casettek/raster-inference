use std::{collections::VecDeque, path::PathBuf, sync::Arc};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::shared::numerics::det_num::{act_to_f32, Act, DetWgtElementWidth};
use crate::shared::numerics::det_tensor::DetKvCacheData;

/// Compatibility re-export for the pre-containment paths; new code should
/// import Gemma types from `shared::model::gemma::transformer`.
pub use super::gemma::transformer::*;

fn default_embedding_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixF32 {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

/// Borrowed view of a canonical weight payload at its storage width.
///
/// Storage width is representation only (detwgt v2): the canonical value of
/// every weight is its sign-extended integer, so an `I16` payload widened to
/// i32 is bit-identical to the same tensor stored `I32`.
#[derive(Debug, Clone, Copy)]
pub enum WgtPayload<'a> {
    I32(&'a [i32]),
    I16(&'a [i16]),
}

impl WgtPayload<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::I16(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Canonical (sign-extended) i32 bit pattern at `idx`.
    pub fn wgt_bits(&self, idx: usize) -> i32 {
        match self {
            Self::I32(values) => values[idx],
            Self::I16(values) => i32::from(values[idx]),
        }
    }

    /// Canonical widened values for `start..end`, or `None` when out of
    /// bounds.
    pub fn get_widened(&self, start: usize, end: usize) -> Option<Vec<i32>> {
        match self {
            Self::I32(values) => values.get(start..end).map(<[i32]>::to_vec),
            Self::I16(values) => values
                .get(start..end)
                .map(|narrow| narrow.iter().map(|value| i32::from(*value)).collect()),
        }
    }

    pub fn to_widened_vec(&self) -> Vec<i32> {
        self.get_widened(0, self.len())
            .expect("full-range widening should be in bounds")
    }
}

/// Storage for canonical weight payloads: either an owned copy or a borrowed
/// view into the mmapped `.detwgt` artifact (zero-copy weight loading), at
/// either storage width (detwgt v2).
///
/// The in-memory representation is not contract surface; every variant
/// exposes the same canonical i32 values via [`WgtPayload`].
#[derive(Clone)]
pub enum DetNumValues {
    Owned(Vec<i32>),
    OwnedI16(Vec<i16>),
    Mmap {
        map: Arc<Mmap>,
        /// Byte offset of the payload within the map; element-aligned by
        /// construction (loaders fall back to owned copies on misalignment).
        byte_offset: usize,
        /// Payload length in i32 elements.
        len: usize,
    },
    MmapI16 {
        map: Arc<Mmap>,
        /// Byte offset of the payload within the map; element-aligned by
        /// construction (loaders fall back to owned copies on misalignment).
        byte_offset: usize,
        /// Payload length in i16 elements.
        len: usize,
    },
}

impl DetNumValues {
    pub fn payload(&self) -> WgtPayload<'_> {
        match self {
            Self::Owned(values) => WgtPayload::I32(values),
            Self::OwnedI16(values) => WgtPayload::I16(values),
            Self::Mmap {
                map,
                byte_offset,
                len,
            } => {
                let bytes = &map[*byte_offset..*byte_offset + *len * 4];
                debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<i32>(), 0);
                // SAFETY: alignment and bounds are validated at load time; the
                // payload is encoded as little-endian i32 and this view is
                // only constructed on little-endian hosts.
                WgtPayload::I32(unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const i32, *len)
                })
            }
            Self::MmapI16 {
                map,
                byte_offset,
                len,
            } => {
                let bytes = &map[*byte_offset..*byte_offset + *len * 2];
                debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<i16>(), 0);
                // SAFETY: alignment and bounds are validated at load time; the
                // payload is encoded as little-endian i16 and this view is
                // only constructed on little-endian hosts.
                WgtPayload::I16(unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const i16, *len)
                })
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Owned(values) => values.len(),
            Self::OwnedI16(values) => values.len(),
            Self::Mmap { len, .. } | Self::MmapI16 { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Canonical (sign-extended) i32 bit pattern at `idx`.
    pub fn wgt_bits(&self, idx: usize) -> i32 {
        self.payload().wgt_bits(idx)
    }

    /// Canonical widened values for `start..end`, or `None` when out of
    /// bounds.
    pub fn get_widened(&self, start: usize, end: usize) -> Option<Vec<i32>> {
        self.payload().get_widened(start, end)
    }

    pub fn to_widened_vec(&self) -> Vec<i32> {
        self.payload().to_widened_vec()
    }

    pub fn is_mmap_backed(&self) -> bool {
        matches!(self, Self::Mmap { .. } | Self::MmapI16 { .. })
    }
}

impl From<Vec<i32>> for DetNumValues {
    fn from(values: Vec<i32>) -> Self {
        Self::Owned(values)
    }
}

impl std::fmt::Debug for DetNumValues {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DetNumValues")
            .field("mmap_backed", &self.is_mmap_backed())
            .field("values", &self.to_widened_vec())
            .finish()
    }
}

impl PartialEq for DetNumValues {
    fn eq(&self, other: &Self) -> bool {
        // Canonical value equality: storage width is representation only.
        self.to_widened_vec() == other.to_widened_vec()
    }
}

impl Eq for DetNumValues {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetNumMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: DetNumValues,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingTable {
    pub rows: Vec<Vec<f32>>,
    #[serde(default = "default_embedding_scale")]
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActivationSequence {
    #[serde(skip_serializing, default)]
    pub activations: Vec<Vec<f32>>,
    #[serde(skip, default)]
    pub(crate) internal: InternalActivationSequence,
    /// f32 compatibility commitment. Always `Some` in fp32 mode; `None` for
    /// deterministic-mode runs (spec v1 retires f32 compatibility commitments
    /// on the deterministic path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activations_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub det_activations_sha256: Option<String>,
}

impl ActivationSequence {
    pub(crate) fn from_internal(
        internal: InternalActivationSequence,
        activations_sha256: String,
    ) -> Self {
        Self {
            activations: internal.clone_f32(),
            internal,
            activations_sha256: Some(activations_sha256),
            det_activations_sha256: None,
        }
    }

    /// Deterministic-mode constructor: canonical commitment only, no f32 view
    /// commitment.
    pub(crate) fn from_det_internal(
        internal: InternalActivationSequence,
        det_activations_sha256: Option<String>,
    ) -> Self {
        Self {
            activations: internal.clone_f32(),
            internal,
            activations_sha256: None,
            det_activations_sha256,
        }
    }

    pub(crate) fn from_values(activations: Vec<Vec<f32>>, activations_sha256: String) -> Self {
        Self::from_internal(
            InternalActivationSequence::from_values(activations),
            activations_sha256,
        )
    }

    pub(crate) fn clone_internal(&self) -> InternalActivationSequence {
        if self.internal.as_f32_slice().is_empty()
            && self.internal.det_values().is_none()
            && !self.activations.is_empty()
        {
            // Older deserialized payloads only carry the public f32 view.
            return InternalActivationSequence::from_values(self.activations.clone());
        }
        self.internal.clone()
    }
}

pub type EmbeddedTokenSequence = ActivationSequence;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransformerStateTransitionState {
    pub activation_states: Vec<ActivationSequence>,
    #[serde(skip_serializing, default)]
    pub prefill_logits: PrefillLogits,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LayerKvCache {
    pub keys: Vec<VecDeque<Vec<f32>>>,
    pub values: Vec<VecDeque<Vec<f32>>>,
    pub(crate) det: Option<DetKvCacheData>,
}

impl LayerKvCache {
    pub fn new(num_kv_heads: usize) -> Self {
        Self {
            keys: vec![VecDeque::new(); num_kv_heads],
            values: vec![VecDeque::new(); num_kv_heads],
            det: None,
        }
    }

    pub(crate) fn from_f32_heads(
        keys: Vec<VecDeque<Vec<f32>>>,
        values: Vec<VecDeque<Vec<f32>>>,
    ) -> Self {
        Self {
            keys,
            values,
            det: None,
        }
    }

    pub(crate) fn from_det_heads(
        det_keys: Vec<VecDeque<Vec<Act>>>,
        det_values: Vec<VecDeque<Vec<Act>>>,
    ) -> Self {
        let nested_keys = det_keys
            .into_iter()
            .map(|head| head.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let nested_values = det_values
            .into_iter()
            .map(|head| head.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Self::from_det_data(DetKvCacheData::from_nested_rows(
            &nested_keys,
            &nested_values,
            0,
        ))
    }

    pub(crate) fn from_det_data(det: DetKvCacheData) -> Self {
        let num_kv_heads = det.num_heads();
        Self {
            keys: vec![VecDeque::new(); num_kv_heads],
            values: vec![VecDeque::new(); num_kv_heads],
            det: Some(det),
        }
    }

    pub fn current_len(&self) -> usize {
        match &self.det {
            Some(det) => det.len(),
            None => self.keys.first().map(VecDeque::len).unwrap_or(0),
        }
    }

    pub(crate) fn det_data(&self) -> Option<&DetKvCacheData> {
        self.det.as_ref()
    }

    fn clamped_window(det: &DetKvCacheData, start: usize, len: usize) -> (usize, usize) {
        let start = start.min(det.len());
        (start, len.min(det.len() - start))
    }

    pub(crate) fn det_key_rows_from(&self, head_idx: usize, start: usize) -> Option<Vec<Vec<Act>>> {
        self.det_key_rows_window(head_idx, start, usize::MAX)
    }

    pub(crate) fn det_key_rows_window(
        &self,
        head_idx: usize,
        start: usize,
        len: usize,
    ) -> Option<Vec<Vec<Act>>> {
        let det = self.det.as_ref()?;
        let (start, len) = Self::clamped_window(det, start, len);
        (head_idx < det.num_heads()).then(|| {
            det.key_window(head_idx, start, len)
                .chunks(det.head_dim().max(1))
                .map(<[Act]>::to_vec)
                .collect()
        })
    }

    pub(crate) fn det_value_rows_from(
        &self,
        head_idx: usize,
        start: usize,
    ) -> Option<Vec<Vec<Act>>> {
        self.det_value_rows_window(head_idx, start, usize::MAX)
    }

    pub(crate) fn det_value_rows_window(
        &self,
        head_idx: usize,
        start: usize,
        len: usize,
    ) -> Option<Vec<Vec<Act>>> {
        let det = self.det.as_ref()?;
        let (start, len) = Self::clamped_window(det, start, len);
        (head_idx < det.num_heads()).then(|| {
            det.value_window(head_idx, start, len)
                .chunks(det.head_dim().max(1))
                .map(<[Act]>::to_vec)
                .collect()
        })
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TransformerDecodeState {
    pub layer_caches: Vec<LayerKvCache>,
    pub position: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerPrefillResult {
    pub transformer_state: TransformerStateTransitionState,
    pub transformer_decode_state: TransformerDecodeState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerDecodeStepResult {
    pub transformer_decode_state: TransformerDecodeState,
    pub activation_state: ActivationSequence,
    pub prefill_logits: PrefillLogits,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetNumTensorSliceSource {
    pub weights_path: PathBuf,
    pub total_rows: usize,
    pub total_cols: usize,
    pub data_offset: usize,
    /// Storage width of the tensor payload (detwgt v2); values are widened
    /// to canonical i32 on read.
    pub element_width: DetWgtElementWidth,
    pub row_offset: usize,
    pub row_count: usize,
    pub col_offset: usize,
    pub col_count: usize,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct InternalActivationRow {
    values: Vec<f32>,
    det_values: Option<Vec<Act>>,
}

impl InternalActivationRow {
    pub(crate) fn from_values(values: Vec<f32>) -> Self {
        Self {
            values,
            det_values: None,
        }
    }

    pub(crate) fn from_det_values(det_values: Vec<Act>) -> Self {
        Self {
            values: det_values.iter().copied().map(act_to_f32).collect(),
            det_values: Some(det_values),
        }
    }

    /// Single-track deterministic constructor: no f32 mirror is materialized.
    pub(crate) fn from_det_values_only(det_values: Vec<Act>) -> Self {
        Self {
            values: Vec::new(),
            det_values: Some(det_values),
        }
    }

    pub(crate) fn as_f32_slice(&self) -> &[f32] {
        &self.values
    }

    pub(crate) fn clone_f32(&self) -> Vec<f32> {
        self.values.clone()
    }

    pub(crate) fn det_values(&self) -> Option<&[Act]> {
        self.det_values.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct InternalActivationSequence {
    values: Vec<Vec<f32>>,
    det_values: Option<Vec<Vec<Act>>>,
}

impl InternalActivationSequence {
    pub(crate) fn from_values(values: Vec<Vec<f32>>) -> Self {
        Self {
            values,
            det_values: None,
        }
    }

    pub(crate) fn from_det_values(det_values: Vec<Vec<Act>>) -> Self {
        Self {
            values: det_values
                .iter()
                .map(|row| row.iter().copied().map(act_to_f32).collect())
                .collect(),
            det_values: Some(det_values),
        }
    }

    /// Single-track deterministic constructor: no f32 mirror is materialized.
    pub(crate) fn from_det_values_only(det_values: Vec<Vec<Act>>) -> Self {
        Self {
            values: Vec::new(),
            det_values: Some(det_values),
        }
    }

    pub(crate) fn as_f32_slice(&self) -> &[Vec<f32>] {
        &self.values
    }

    pub(crate) fn clone_f32(&self) -> Vec<Vec<f32>> {
        self.values.clone()
    }

    pub(crate) fn det_values(&self) -> Option<&[Vec<Act>]> {
        self.det_values.as_deref()
    }

    pub(crate) fn last_row(&self) -> Option<InternalActivationRow> {
        if let Some(det_rows) = self.det_values.as_ref() {
            let det_values = det_rows.last()?.clone();
            let values = self.values.last().cloned().unwrap_or_default();
            return Some(InternalActivationRow {
                values,
                det_values: Some(det_values),
            });
        }
        let values = self.values.last()?.clone();
        Some(InternalActivationRow {
            values,
            det_values: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct InternalLogits {
    values: Vec<f32>,
    det_values: Option<Vec<Act>>,
}

impl InternalLogits {
    pub(crate) fn from_values(values: Vec<f32>) -> Self {
        Self {
            values,
            det_values: None,
        }
    }

    pub(crate) fn from_det_values(det_values: Vec<Act>) -> Self {
        Self {
            values: det_values.iter().copied().map(act_to_f32).collect(),
            det_values: Some(det_values),
        }
    }

    /// Single-track deterministic constructor: no f32 mirror is materialized.
    pub(crate) fn from_det_values_only(det_values: Vec<Act>) -> Self {
        Self {
            values: Vec::new(),
            det_values: Some(det_values),
        }
    }

    pub(crate) fn as_f32_slice(&self) -> &[f32] {
        &self.values
    }

    pub(crate) fn clone_f32(&self) -> Vec<f32> {
        self.values.clone()
    }

    pub(crate) fn det_values(&self) -> Option<&[Act]> {
        self.det_values.as_deref()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PrefillLogits {
    #[serde(skip_serializing, default)]
    pub logits: Vec<f32>,
    #[serde(skip, default)]
    pub(crate) internal: InternalLogits,
    /// f32 compatibility commitment. Always `Some` in fp32 mode; `None` for
    /// deterministic-mode runs (spec v1 retires f32 compatibility commitments
    /// on the deterministic path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logits_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub det_final_logits_sha256: Option<String>,
}

impl PrefillLogits {
    pub(crate) fn from_internal(internal: InternalLogits, final_logits_sha256: String) -> Self {
        Self {
            logits: internal.clone_f32(),
            internal,
            final_logits_sha256: Some(final_logits_sha256),
            det_final_logits_sha256: None,
        }
    }

    /// Deterministic-mode constructor: canonical commitment only, no f32 view
    /// commitment.
    pub(crate) fn from_det_internal(
        internal: InternalLogits,
        det_final_logits_sha256: Option<String>,
    ) -> Self {
        Self {
            logits: internal.clone_f32(),
            internal,
            final_logits_sha256: None,
            det_final_logits_sha256,
        }
    }

    pub(crate) fn clone_internal(&self) -> InternalLogits {
        if self.internal.as_f32_slice().is_empty()
            && self.internal.det_values().is_none()
            && !self.logits.is_empty()
        {
            // Older deserialized payloads only carry the public f32 view.
            return InternalLogits::from_values(self.logits.clone());
        }
        self.internal.clone()
    }
}
