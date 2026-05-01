use std::fs::File;

use anyhow::{anyhow, bail, Context, Result};

use crate::auth_read;
use crate::raster_authoring::AuthRead;
use crate::shared::det_num::{
    act_to_f32, add_sat, attention_score as det_attention_score,
    attention_softmax as det_attention_softmax,
    attention_weighted_sum as det_attention_weighted_sum, gelu_pytorch_tanh_act, mac_bits, mul_sat,
    requantize, rms_norm as det_rms_norm, rope_rotate_pairs, scale_act,
    value_rms_norm as det_value_rms_norm, Acc, Act, Wgt,
};
use crate::shared::raster_prefill_layer::{
    GemmaPrefillLayerMatrixKind, GemmaPrefillLayerMatrixRowRequest,
};
use crate::shared::raster_prefill_ple::GemmaPleModelProjectionRowRequest;
use crate::shared::transformer::DetNumTensorSliceSource;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterActivationRow {
    act_bits: Vec<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterActivationSequence {
    rows: Vec<RasterActivationRow>,
}

impl RasterActivationRow {
    pub fn from_acts(acts: Vec<Act>) -> Self {
        Self {
            act_bits: acts.into_iter().map(|value| value.to_bits()).collect(),
        }
    }

    pub fn from_act_bits(act_bits: Vec<i32>) -> Self {
        Self { act_bits }
    }

    pub fn acts(&self) -> Vec<Act> {
        self.act_bits.iter().copied().map(Act::from_bits).collect()
    }

    pub fn act_bits(&self) -> &[i32] {
        &self.act_bits
    }

    pub fn width(&self) -> usize {
        self.act_bits.len()
    }

    pub fn to_f32_values(&self) -> Vec<f32> {
        self.acts().into_iter().map(act_to_f32).collect()
    }
}

impl RasterActivationSequence {
    pub fn from_rows(rows: Vec<RasterActivationRow>) -> Self {
        Self { rows }
    }

    pub fn from_acts(rows: Vec<Vec<Act>>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(RasterActivationRow::from_acts)
                .collect(),
        }
    }

    pub fn from_act_bits(rows: Vec<Vec<i32>>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(RasterActivationRow::from_act_bits)
                .collect(),
        }
    }

    pub fn rows(&self) -> &[RasterActivationRow] {
        &self.rows
    }

    pub fn into_rows(self) -> Vec<RasterActivationRow> {
        self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn width(&self) -> Result<usize> {
        sequence_width(self)
    }

    pub fn to_f32_values(&self) -> Vec<Vec<f32>> {
        self.rows
            .iter()
            .map(RasterActivationRow::to_f32_values)
            .collect()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterAttentionHeadSequence {
    heads: Vec<Vec<RasterActivationRow>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterKvCache {
    keys: Vec<Vec<RasterActivationRow>>,
    values: Vec<Vec<RasterActivationRow>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterAttentionRowState {
    queries: RasterAttentionHeadSequence,
    keys: RasterAttentionHeadSequence,
    values: RasterAttentionHeadSequence,
    donor_cache: Option<RasterKvCache>,
    attention_window: Option<usize>,
    next_query_head_idx: usize,
    next_query_token_idx: usize,
    output_heads: Vec<Vec<RasterActivationRow>>,
    sequence_len: usize,
    kv_head_count: usize,
    kv_groups: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RasterSequenceUnaryOp {
    RmsNorm {
        norm_weight_bits: Vec<i32>,
        eps_bits: i64,
    },
    Gelu,
    Scale {
        scalar_bits: i32,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceUnaryState {
    input: RasterActivationSequence,
    op: RasterSequenceUnaryOp,
    next_row_idx: usize,
    output_rows: Vec<RasterActivationRow>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RasterSequenceBinaryOp {
    Add,
    Mul,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceBinaryState {
    lhs: RasterActivationSequence,
    rhs: RasterActivationSequence,
    op: RasterSequenceBinaryOp,
    next_row_idx: usize,
    output_rows: Vec<RasterActivationRow>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RasterHeadUnaryOp {
    RmsNorm {
        norm_weight_bits: Vec<i32>,
        eps_bits: i64,
    },
    ValueRmsNorm {
        eps_bits: i64,
    },
    Rope {
        rotary_dim: usize,
        freq_base_dim: usize,
        base_bits: Option<i64>,
        position_offset: usize,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterHeadUnaryState {
    heads: RasterAttentionHeadSequence,
    op: RasterHeadUnaryOp,
    next_head_idx: usize,
    next_token_idx: usize,
    output_heads: Vec<Vec<RasterActivationRow>>,
    sequence_len: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterReshapeHeadsState {
    input: RasterActivationSequence,
    num_heads: usize,
    head_dim: usize,
    next_row_idx: usize,
    output_heads: Vec<Vec<RasterActivationRow>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterCombineHeadsState {
    heads: RasterAttentionHeadSequence,
    next_token_idx: usize,
    output_rows: Vec<RasterActivationRow>,
    sequence_len: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterKvCacheBuildState {
    keys: RasterAttentionHeadSequence,
    values: RasterAttentionHeadSequence,
    retained_start: usize,
    next_head_idx: usize,
    next_token_idx: usize,
    output_keys: Vec<Vec<RasterActivationRow>>,
    output_values: Vec<Vec<RasterActivationRow>>,
    sequence_len: usize,
}

impl RasterAttentionHeadSequence {
    pub fn from_heads(heads: Vec<Vec<RasterActivationRow>>) -> Self {
        Self { heads }
    }

    pub fn from_acts(heads: Vec<Vec<Vec<Act>>>) -> Self {
        Self {
            heads: heads
                .into_iter()
                .map(|head| {
                    head.into_iter()
                        .map(RasterActivationRow::from_acts)
                        .collect()
                })
                .collect(),
        }
    }

    pub fn heads(&self) -> &[Vec<RasterActivationRow>] {
        &self.heads
    }

    pub fn into_heads(self) -> Vec<Vec<RasterActivationRow>> {
        self.heads
    }

    pub fn head_count(&self) -> usize {
        self.heads.len()
    }

    pub fn sequence_len(&self) -> Result<usize> {
        attention_sequence_len(self)
    }

    pub fn head_width(&self) -> Result<usize> {
        attention_head_width(self)
    }
}

impl RasterKvCache {
    pub fn empty(num_kv_heads: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_kv_heads],
            values: vec![Vec::new(); num_kv_heads],
        }
    }

    pub fn from_heads(
        keys: Vec<Vec<RasterActivationRow>>,
        values: Vec<Vec<RasterActivationRow>>,
    ) -> Result<Self> {
        if keys.len() != values.len() {
            bail!(
                "KV cache head count mismatch: {} vs {}",
                keys.len(),
                values.len()
            );
        }
        let cache = Self { keys, values };
        validate_kv_cache(&cache)?;
        Ok(cache)
    }

    pub fn keys(&self) -> &[Vec<RasterActivationRow>] {
        &self.keys
    }

    pub fn values(&self) -> &[Vec<RasterActivationRow>] {
        &self.values
    }

    pub fn head_count(&self) -> usize {
        self.keys.len()
    }

    pub fn current_len(&self) -> usize {
        self.keys.first().map(Vec::len).unwrap_or(0)
    }

    pub fn key_rows_window(
        &self,
        head_idx: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<RasterActivationRow>> {
        self.keys
            .get(head_idx)
            .ok_or_else(|| {
                anyhow!(
                    "KV cache key head {head_idx} is out of range for {} heads",
                    self.keys.len()
                )
            })
            .map(|head| head.iter().skip(start).take(len).cloned().collect())
    }

    pub fn value_rows_window(
        &self,
        head_idx: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<RasterActivationRow>> {
        self.values
            .get(head_idx)
            .ok_or_else(|| {
                anyhow!(
                    "KV cache value head {head_idx} is out of range for {} heads",
                    self.values.len()
                )
            })
            .map(|head| head.iter().skip(start).take(len).cloned().collect())
    }
}

impl RasterAttentionRowState {
    pub fn is_complete(&self) -> bool {
        self.next_query_head_idx >= self.queries.head_count()
    }

    pub fn next_query_head_idx(&self) -> usize {
        self.next_query_head_idx
    }

    pub fn next_query_token_idx(&self) -> usize {
        self.next_query_token_idx
    }
}

impl RasterSequenceUnaryState {
    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.input.len()
    }
}

impl RasterSequenceBinaryState {
    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.lhs.len()
    }
}

impl RasterHeadUnaryState {
    pub fn is_complete(&self) -> bool {
        self.next_head_idx >= self.heads.head_count()
    }
}

impl RasterReshapeHeadsState {
    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.input.len()
    }
}

impl RasterCombineHeadsState {
    pub fn is_complete(&self) -> bool {
        self.next_token_idx >= self.sequence_len
    }
}

impl RasterKvCacheBuildState {
    pub fn is_complete(&self) -> bool {
        self.next_head_idx >= self.keys.head_count()
    }
}

pub fn scale_sequence(
    input: &RasterActivationSequence,
    scalar: Option<Act>,
) -> Result<RasterActivationSequence> {
    let scalar = scalar
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    validate_non_empty_sequence(input, "deterministic sequence scaling")?;

    Ok(RasterActivationSequence::from_rows(
        input
            .rows()
            .iter()
            .map(|row| {
                RasterActivationRow::from_acts(
                    row.acts()
                        .into_iter()
                        .map(|value| scale_act(value, scalar))
                        .collect(),
                )
            })
            .collect(),
    ))
}

pub fn add_sequences(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
) -> Result<RasterActivationSequence> {
    let width = sequence_width(lhs)?;
    validate_sequence_width(rhs, width, "right sequence")?;
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    Ok(RasterActivationSequence::from_rows(
        lhs.rows()
            .iter()
            .zip(rhs.rows())
            .map(|(lhs_row, rhs_row)| {
                RasterActivationRow::from_acts(
                    lhs_row
                        .acts()
                        .into_iter()
                        .zip(rhs_row.acts())
                        .map(|(lhs_value, rhs_value)| add_sat(lhs_value, rhs_value))
                        .collect(),
                )
            })
            .collect(),
    ))
}

pub fn project_sequence(
    input: &RasterActivationSequence,
    projection_rows: &[Vec<Wgt>],
) -> Result<RasterActivationSequence> {
    validate_projection_rows(projection_rows)?;
    validate_sequence_width(
        input,
        projection_rows[0].len(),
        "deterministic linear input",
    )?;

    Ok(RasterActivationSequence::from_rows(
        input
            .rows()
            .iter()
            .map(|input_row| project_row(input_row, projection_rows))
            .collect::<Result<Vec<_>>>()?,
    ))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceProjectionState {
    input: RasterActivationSequence,
    next_row_idx: usize,
    projection_rows: usize,
    rows_per_tile: usize,
    output_act_bits: Vec<Vec<i32>>,
}

impl RasterSequenceProjectionState {
    pub fn next_row_idx(&self) -> usize {
        self.next_row_idx
    }

    pub fn projection_rows(&self) -> usize {
        self.projection_rows
    }

    pub fn rows_per_tile(&self) -> usize {
        self.rows_per_tile
    }

    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.projection_rows
    }
}

pub fn validate_projection_rows_per_tile(rows_per_tile: usize) -> Result<()> {
    if rows_per_tile == 0 {
        bail!("raster projection rows per tile must be greater than zero");
    }
    Ok(())
}

pub fn init_sequence_projection_state(
    input: &RasterActivationSequence,
    projection_rows: usize,
    rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }
    validate_projection_rows_per_tile(rows_per_tile)?;
    sequence_width(input)?;

    Ok(RasterSequenceProjectionState {
        input: input.clone(),
        next_row_idx: 0,
        projection_rows,
        rows_per_tile,
        output_act_bits: vec![Vec::with_capacity(projection_rows); input.len()],
    })
}

pub fn append_projection_row_to_state(
    state: &mut RasterSequenceProjectionState,
    projection_row: &[Wgt],
) -> Result<()> {
    if state.is_complete() {
        bail!(
            "raster projection already completed {} rows",
            state.projection_rows
        );
    }
    if state.output_act_bits.len() != state.input.len() {
        bail!(
            "raster projection state has {} output rows for {} input rows",
            state.output_act_bits.len(),
            state.input.len()
        );
    }

    for (token_idx, input_row) in state.input.rows().iter().enumerate() {
        let projected = project_row_with_weights(input_row, projection_row)?;
        state.output_act_bits[token_idx].push(projected.to_bits());
    }
    state.next_row_idx += 1;
    Ok(())
}

pub fn finalize_sequence_projection_state(
    state: RasterSequenceProjectionState,
) -> Result<RasterActivationSequence> {
    if state.next_row_idx != state.projection_rows {
        bail!(
            "raster projection completed {} rows, expected {}",
            state.next_row_idx,
            state.projection_rows
        );
    }
    if let Some((row_idx, row)) = state
        .output_act_bits
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != state.projection_rows)
    {
        bail!(
            "raster projection output row {row_idx} has width {}, expected {}",
            row.len(),
            state.projection_rows
        );
    }

    Ok(RasterActivationSequence::from_act_bits(
        state.output_act_bits,
    ))
}

pub fn project_sequence_with_source<S>(
    input: &RasterActivationSequence,
    source: &S,
    layer_idx: usize,
    projection_rows: usize,
) -> Result<RasterActivationSequence>
where
    S: AuthRead<GemmaPleModelProjectionRowRequest, Output = Vec<Wgt>>,
{
    project_sequence_with_source_chunked(input, source, layer_idx, projection_rows, projection_rows)
}

pub fn project_sequence_with_source_chunked<S>(
    input: &RasterActivationSequence,
    source: &S,
    layer_idx: usize,
    projection_rows: usize,
    rows_per_tile: usize,
) -> Result<RasterActivationSequence>
where
    S: AuthRead<GemmaPleModelProjectionRowRequest, Output = Vec<Wgt>>,
{
    let mut state = init_sequence_projection_state(input, projection_rows, rows_per_tile)?;
    while !state.is_complete() {
        let end = state
            .next_row_idx()
            .saturating_add(state.rows_per_tile())
            .min(state.projection_rows());
        while state.next_row_idx() < end {
            let row = auth_read!(
                source,
                GemmaPleModelProjectionRowRequest {
                    layer_idx,
                    row_idx: state.next_row_idx(),
                }
            )?;
            append_projection_row_to_state(&mut state, &row)?;
        }
    }
    finalize_sequence_projection_state(state)
}

pub(crate) fn det_num_tensor_slice_row_wgts(
    source: &DetNumTensorSliceSource,
    row_idx: usize,
    label: &str,
) -> Result<Vec<Wgt>> {
    if row_idx >= source.row_count {
        bail!(
            "Gemma prefill {label} row {row_idx} is out of range for {} rows",
            source.row_count
        );
    }

    let file = File::open(&source.weights_path).with_context(|| {
        format!(
            "failed to open deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| {
        format!(
            "failed to mmap deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let row_bytes = source
        .total_cols
        .checked_mul(4)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let global_row_idx = source.row_offset + row_idx;
    let start = source
        .data_offset
        .checked_add(global_row_idx * row_bytes)
        .and_then(|offset| offset.checked_add(source.col_offset * 4))
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let end = start
        .checked_add(source.col_count * 4)
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let encoded_row = mmap
        .get(start..end)
        .ok_or_else(|| anyhow!("matrix slice byte range is out of bounds"))?;
    let mut row = Vec::with_capacity(source.col_count);
    for encoded_value in encoded_row.chunks_exact(4) {
        row.push(Wgt::from_bits(i32::from_le_bytes(
            encoded_value
                .try_into()
                .expect("i32 byte width should match"),
        )));
    }
    Ok(row)
}

pub fn project_sequence_with_prefill_source<S>(
    input: &RasterActivationSequence,
    source: &S,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
    projection_rows: usize,
) -> Result<RasterActivationSequence>
where
    S: AuthRead<GemmaPrefillLayerMatrixRowRequest, Output = Vec<Wgt>>,
{
    project_sequence_with_prefill_source_chunked(
        input,
        source,
        layer_idx,
        matrix,
        projection_rows,
        projection_rows,
    )
}

pub fn project_sequence_with_prefill_source_chunked<S>(
    input: &RasterActivationSequence,
    source: &S,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
    projection_rows: usize,
    rows_per_tile: usize,
) -> Result<RasterActivationSequence>
where
    S: AuthRead<GemmaPrefillLayerMatrixRowRequest, Output = Vec<Wgt>>,
{
    let mut state = init_sequence_projection_state(input, projection_rows, rows_per_tile)?;
    while !state.is_complete() {
        let end = state
            .next_row_idx()
            .saturating_add(state.rows_per_tile())
            .min(state.projection_rows());
        while state.next_row_idx() < end {
            let row = auth_read!(
                source,
                GemmaPrefillLayerMatrixRowRequest {
                    layer_idx,
                    matrix,
                    row_idx: state.next_row_idx(),
                }
            )?;
            append_projection_row_to_state(&mut state, &row)?;
        }
    }
    finalize_sequence_projection_state(state)
}

pub fn rms_norm_sequence(
    input: &RasterActivationSequence,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
) -> Result<RasterActivationSequence> {
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical norm weights"))?;
    let eps = eps.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    validate_sequence_width(input, norm_weights.len(), "deterministic RMSNorm input")?;

    Ok(RasterActivationSequence::from_rows(
        input
            .rows()
            .iter()
            .map(|row| RasterActivationRow::from_acts(det_rms_norm(&row.acts(), norm_weights, eps)))
            .collect(),
    ))
}

pub fn mul_sequences(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
) -> Result<RasterActivationSequence> {
    let width = sequence_width(lhs)?;
    validate_sequence_width(rhs, width, "right sequence")?;
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    Ok(RasterActivationSequence::from_rows(
        lhs.rows()
            .iter()
            .zip(rhs.rows())
            .map(|(lhs_row, rhs_row)| {
                RasterActivationRow::from_acts(
                    lhs_row
                        .acts()
                        .into_iter()
                        .zip(rhs_row.acts())
                        .map(|(lhs_value, rhs_value)| mul_sat(lhs_value, rhs_value))
                        .collect(),
                )
            })
            .collect(),
    ))
}

pub fn gelu_sequence(input: &RasterActivationSequence) -> Result<RasterActivationSequence> {
    validate_non_empty_sequence(input, "deterministic GELU")?;
    Ok(RasterActivationSequence::from_rows(
        input
            .rows()
            .iter()
            .map(|row| {
                RasterActivationRow::from_acts(
                    row.acts().into_iter().map(gelu_pytorch_tanh_act).collect(),
                )
            })
            .collect(),
    ))
}

pub fn init_sequence_rms_norm_row_state(
    input: &RasterActivationSequence,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
) -> Result<RasterSequenceUnaryState> {
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical norm weights"))?;
    let eps = eps.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    validate_sequence_width(input, norm_weights.len(), "deterministic RMSNorm input")?;

    Ok(RasterSequenceUnaryState {
        input: input.clone(),
        op: RasterSequenceUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
        next_row_idx: 0,
        output_rows: Vec::with_capacity(input.len()),
    })
}

pub fn init_sequence_gelu_row_state(
    input: &RasterActivationSequence,
) -> Result<RasterSequenceUnaryState> {
    validate_non_empty_sequence(input, "deterministic GELU")?;

    Ok(RasterSequenceUnaryState {
        input: input.clone(),
        op: RasterSequenceUnaryOp::Gelu,
        next_row_idx: 0,
        output_rows: Vec::with_capacity(input.len()),
    })
}

pub fn init_sequence_scale_row_state(
    input: &RasterActivationSequence,
    scalar: Option<Act>,
) -> Result<RasterSequenceUnaryState> {
    let scalar = scalar
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    validate_non_empty_sequence(input, "deterministic sequence scaling")?;

    Ok(RasterSequenceUnaryState {
        input: input.clone(),
        op: RasterSequenceUnaryOp::Scale {
            scalar_bits: scalar.to_bits(),
        },
        next_row_idx: 0,
        output_rows: Vec::with_capacity(input.len()),
    })
}

pub fn compute_next_sequence_unary_row(
    mut state: RasterSequenceUnaryState,
) -> Result<(bool, RasterSequenceUnaryState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let row = state.input.rows().get(state.next_row_idx).ok_or_else(|| {
        anyhow!(
            "sequence unary row {} is out of range for {} rows",
            state.next_row_idx,
            state.input.len()
        )
    })?;
    let output_row = match &state.op {
        RasterSequenceUnaryOp::RmsNorm {
            norm_weight_bits,
            eps_bits,
        } => {
            let norm_weights = norm_weight_bits
                .iter()
                .copied()
                .map(Wgt::from_bits)
                .collect::<Vec<_>>();
            RasterActivationRow::from_acts(det_rms_norm(
                &row.acts(),
                &norm_weights,
                Acc::from_bits(*eps_bits),
            ))
        }
        RasterSequenceUnaryOp::Gelu => RasterActivationRow::from_acts(
            row.acts().into_iter().map(gelu_pytorch_tanh_act).collect(),
        ),
        RasterSequenceUnaryOp::Scale { scalar_bits } => RasterActivationRow::from_acts(
            row.acts()
                .into_iter()
                .map(|value| scale_act(value, Act::from_bits(*scalar_bits)))
                .collect(),
        ),
    };
    state.output_rows.push(output_row);
    state.next_row_idx += 1;
    Ok((false, state))
}

pub fn finalize_sequence_unary_row_state(
    state: RasterSequenceUnaryState,
) -> Result<RasterActivationSequence> {
    if !state.is_complete() {
        bail!(
            "sequence unary state completed {} rows, expected {}",
            state.next_row_idx,
            state.input.len()
        );
    }
    if state.output_rows.len() != state.input.len() {
        bail!(
            "sequence unary output has {} rows, expected {}",
            state.output_rows.len(),
            state.input.len()
        );
    }

    Ok(RasterActivationSequence::from_rows(state.output_rows))
}

pub fn init_sequence_add_row_state(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_binary_row_state(lhs, rhs, RasterSequenceBinaryOp::Add)
}

pub fn init_sequence_mul_row_state(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_binary_row_state(lhs, rhs, RasterSequenceBinaryOp::Mul)
}

fn init_sequence_binary_row_state(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    op: RasterSequenceBinaryOp,
) -> Result<RasterSequenceBinaryState> {
    let width = sequence_width(lhs)?;
    validate_sequence_width(rhs, width, "right sequence")?;
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    Ok(RasterSequenceBinaryState {
        lhs: lhs.clone(),
        rhs: rhs.clone(),
        op,
        next_row_idx: 0,
        output_rows: Vec::with_capacity(lhs.len()),
    })
}

pub fn compute_next_sequence_binary_row(
    mut state: RasterSequenceBinaryState,
) -> Result<(bool, RasterSequenceBinaryState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let lhs_row = state.lhs.rows().get(state.next_row_idx).ok_or_else(|| {
        anyhow!(
            "left sequence row {} is out of range for {} rows",
            state.next_row_idx,
            state.lhs.len()
        )
    })?;
    let rhs_row = state.rhs.rows().get(state.next_row_idx).ok_or_else(|| {
        anyhow!(
            "right sequence row {} is out of range for {} rows",
            state.next_row_idx,
            state.rhs.len()
        )
    })?;
    let output_row = match state.op {
        RasterSequenceBinaryOp::Add => RasterActivationRow::from_acts(
            lhs_row
                .acts()
                .into_iter()
                .zip(rhs_row.acts())
                .map(|(lhs_value, rhs_value)| add_sat(lhs_value, rhs_value))
                .collect(),
        ),
        RasterSequenceBinaryOp::Mul => RasterActivationRow::from_acts(
            lhs_row
                .acts()
                .into_iter()
                .zip(rhs_row.acts())
                .map(|(lhs_value, rhs_value)| mul_sat(lhs_value, rhs_value))
                .collect(),
        ),
    };
    state.output_rows.push(output_row);
    state.next_row_idx += 1;
    Ok((false, state))
}

pub fn finalize_sequence_binary_row_state(
    state: RasterSequenceBinaryState,
) -> Result<RasterActivationSequence> {
    if !state.is_complete() {
        bail!(
            "sequence binary state completed {} rows, expected {}",
            state.next_row_idx,
            state.lhs.len()
        );
    }
    if state.output_rows.len() != state.lhs.len() {
        bail!(
            "sequence binary output has {} rows, expected {}",
            state.output_rows.len(),
            state.lhs.len()
        );
    }

    Ok(RasterActivationSequence::from_rows(state.output_rows))
}

pub fn reshape_sequence_heads(
    input: &RasterActivationSequence,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterAttentionHeadSequence> {
    if num_heads == 0 {
        bail!("attention reshape requires at least one head");
    }
    if head_dim == 0 {
        bail!("attention reshape requires non-zero head dimension");
    }
    let expected_width = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("attention reshape width overflowed"))?;
    validate_sequence_width(input, expected_width, "attention reshape input")?;

    let mut heads = vec![Vec::with_capacity(input.len()); num_heads];
    for row in input.rows() {
        let acts = row.acts();
        for (head_idx, head) in heads.iter_mut().enumerate() {
            let start = head_idx * head_dim;
            head.push(RasterActivationRow::from_acts(
                acts[start..start + head_dim].to_vec(),
            ));
        }
    }
    Ok(RasterAttentionHeadSequence::from_heads(heads))
}

pub fn combine_attention_heads(
    heads: &RasterAttentionHeadSequence,
) -> Result<RasterActivationSequence> {
    let sequence_len = attention_sequence_len(heads)?;
    let _head_width = attention_head_width(heads)?;
    let mut rows = Vec::with_capacity(sequence_len);
    for token_idx in 0..sequence_len {
        let mut row = Vec::new();
        for head in heads.heads() {
            row.extend(head[token_idx].acts());
        }
        rows.push(RasterActivationRow::from_acts(row));
    }
    Ok(RasterActivationSequence::from_rows(rows))
}

pub fn value_rms_norm_heads(
    heads: &RasterAttentionHeadSequence,
    eps: Option<Acc>,
) -> Result<RasterAttentionHeadSequence> {
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    let _head_width = attention_head_width(heads)?;
    Ok(RasterAttentionHeadSequence::from_heads(
        heads
            .heads()
            .iter()
            .map(|head| {
                head.iter()
                    .map(|row| RasterActivationRow::from_acts(det_value_rms_norm(&row.acts(), eps)))
                    .collect()
            })
            .collect(),
    ))
}

pub fn rms_norm_heads(
    heads: &RasterAttentionHeadSequence,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
) -> Result<RasterAttentionHeadSequence> {
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical norm weights"))?;
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical Acc epsilon"))?;
    let head_width = attention_head_width(heads)?;
    if norm_weights.len() != head_width {
        bail!(
            "deterministic head RMSNorm weight width mismatch: {} vs {}",
            norm_weights.len(),
            head_width
        );
    }
    Ok(RasterAttentionHeadSequence::from_heads(
        heads
            .heads()
            .iter()
            .map(|head| {
                head.iter()
                    .map(|row| {
                        RasterActivationRow::from_acts(det_rms_norm(&row.acts(), norm_weights, eps))
                    })
                    .collect()
            })
            .collect(),
    ))
}

pub fn apply_rope_to_heads(
    heads: &RasterAttentionHeadSequence,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<Acc>,
    position_offset: usize,
) -> Result<RasterAttentionHeadSequence> {
    if rotary_dim == 0 {
        return Ok(heads.clone());
    }
    let base = base.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    let head_width = attention_head_width(heads)?;
    if rotary_dim > head_width {
        bail!("RoPE rotary_dim {rotary_dim} exceeds attention head width {head_width}");
    }
    Ok(RasterAttentionHeadSequence::from_heads(
        heads
            .heads()
            .iter()
            .map(|head| {
                head.iter()
                    .enumerate()
                    .map(|(position, row)| {
                        RasterActivationRow::from_acts(rope_rotate_pairs(
                            &row.acts(),
                            rotary_dim,
                            freq_base_dim,
                            base,
                            position_offset + position,
                        ))
                    })
                    .collect()
            })
            .collect(),
    ))
}

pub fn init_head_rms_norm_row_state(
    heads: &RasterAttentionHeadSequence,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
) -> Result<RasterHeadUnaryState> {
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical norm weights"))?;
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical Acc epsilon"))?;
    let head_width = attention_head_width(heads)?;
    if norm_weights.len() != head_width {
        bail!(
            "deterministic head RMSNorm weight width mismatch: {} vs {}",
            norm_weights.len(),
            head_width
        );
    }
    init_head_unary_row_state(
        heads,
        RasterHeadUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
    )
}

pub fn init_value_rms_norm_row_state(
    heads: &RasterAttentionHeadSequence,
    eps: Option<Acc>,
) -> Result<RasterHeadUnaryState> {
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    attention_head_width(heads)?;
    init_head_unary_row_state(
        heads,
        RasterHeadUnaryOp::ValueRmsNorm {
            eps_bits: eps.to_bits(),
        },
    )
}

pub fn init_rope_row_state(
    heads: &RasterAttentionHeadSequence,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<Acc>,
    position_offset: usize,
) -> Result<RasterHeadUnaryState> {
    if rotary_dim != 0 {
        let head_width = attention_head_width(heads)?;
        if rotary_dim > head_width {
            bail!("RoPE rotary_dim {rotary_dim} exceeds attention head width {head_width}");
        }
        base.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    } else {
        attention_sequence_len(heads)?;
    }
    init_head_unary_row_state(
        heads,
        RasterHeadUnaryOp::Rope {
            rotary_dim,
            freq_base_dim,
            base_bits: base.map(Acc::to_bits),
            position_offset,
        },
    )
}

fn init_head_unary_row_state(
    heads: &RasterAttentionHeadSequence,
    op: RasterHeadUnaryOp,
) -> Result<RasterHeadUnaryState> {
    let sequence_len = attention_sequence_len(heads)?;
    let output_heads = (0..heads.head_count())
        .map(|_| Vec::with_capacity(sequence_len))
        .collect();
    Ok(RasterHeadUnaryState {
        heads: heads.clone(),
        op,
        next_head_idx: 0,
        next_token_idx: 0,
        output_heads,
        sequence_len,
    })
}

pub fn compute_next_head_unary_row(
    mut state: RasterHeadUnaryState,
) -> Result<(bool, RasterHeadUnaryState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let head_idx = state.next_head_idx;
    let token_idx = state.next_token_idx;
    let head = state.heads.heads().get(head_idx).ok_or_else(|| {
        anyhow!(
            "head unary head {head_idx} is out of range for {} heads",
            state.heads.head_count()
        )
    })?;
    let row = head.get(token_idx).ok_or_else(|| {
        anyhow!(
            "head unary row {token_idx} is out of range for {} rows",
            head.len()
        )
    })?;
    let output_row = match &state.op {
        RasterHeadUnaryOp::RmsNorm {
            norm_weight_bits,
            eps_bits,
        } => {
            let norm_weights = norm_weight_bits
                .iter()
                .copied()
                .map(Wgt::from_bits)
                .collect::<Vec<_>>();
            RasterActivationRow::from_acts(det_rms_norm(
                &row.acts(),
                &norm_weights,
                Acc::from_bits(*eps_bits),
            ))
        }
        RasterHeadUnaryOp::ValueRmsNorm { eps_bits } => RasterActivationRow::from_acts(
            det_value_rms_norm(&row.acts(), Acc::from_bits(*eps_bits)),
        ),
        RasterHeadUnaryOp::Rope {
            rotary_dim,
            freq_base_dim,
            base_bits,
            position_offset,
        } => {
            if *rotary_dim == 0 {
                row.clone()
            } else {
                let base_bits = base_bits
                    .ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
                RasterActivationRow::from_acts(rope_rotate_pairs(
                    &row.acts(),
                    *rotary_dim,
                    *freq_base_dim,
                    Acc::from_bits(base_bits),
                    position_offset + token_idx,
                ))
            }
        }
    };
    let output_head_count = state.output_heads.len();
    let output_head = state.output_heads.get_mut(head_idx).ok_or_else(|| {
        anyhow!("head unary output head {head_idx} is out of range for {output_head_count} heads")
    })?;
    if output_head.len() != token_idx {
        bail!(
            "head unary output head {head_idx} has {} rows, expected {token_idx}",
            output_head.len()
        );
    }
    output_head.push(output_row);

    if token_idx + 1 < state.sequence_len {
        state.next_token_idx += 1;
    } else {
        state.next_head_idx += 1;
        state.next_token_idx = 0;
    }

    Ok((false, state))
}

pub fn finalize_head_unary_row_state(
    state: RasterHeadUnaryState,
) -> Result<RasterAttentionHeadSequence> {
    if !state.is_complete() {
        bail!(
            "head unary state finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.heads.head_count()
        );
    }
    if let Some((head_idx, head)) = state
        .output_heads
        .iter()
        .enumerate()
        .find(|(_, head)| head.len() != state.sequence_len)
    {
        bail!(
            "head unary output head {head_idx} has {} rows, expected {}",
            head.len(),
            state.sequence_len
        );
    }
    Ok(RasterAttentionHeadSequence::from_heads(state.output_heads))
}

pub fn attention_output_row(
    query: &RasterActivationRow,
    key_rows: &[RasterActivationRow],
    value_rows: &[RasterActivationRow],
) -> Result<RasterActivationRow> {
    if key_rows.is_empty() {
        bail!("deterministic attention requires at least one key row");
    }
    if key_rows.len() != value_rows.len() {
        bail!(
            "deterministic attention key/value row count mismatch: {} vs {}",
            key_rows.len(),
            value_rows.len()
        );
    }
    let query_acts = query.acts();
    validate_rows_width(key_rows, query.width(), "attention key rows")?;
    validate_rows_width(value_rows, query.width(), "attention value rows")?;
    let key_acts = key_rows
        .iter()
        .map(RasterActivationRow::acts)
        .collect::<Vec<_>>();
    let logits = key_acts
        .iter()
        .map(|key_row| det_attention_score(&query_acts, key_row))
        .collect::<Vec<_>>();
    let weights = det_attention_softmax(&logits);
    let value_acts = value_rows
        .iter()
        .map(RasterActivationRow::acts)
        .collect::<Vec<_>>();
    Ok(RasterActivationRow::from_acts(det_attention_weighted_sum(
        &weights,
        &value_acts,
    )))
}

pub fn causal_attention_heads(
    queries: &RasterAttentionHeadSequence,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    attention_window: Option<usize>,
) -> Result<RasterAttentionHeadSequence> {
    validate_matching_attention_heads(queries, keys, "query/key")?;
    validate_matching_attention_heads(queries, values, "query/value")?;
    let sequence_len = attention_sequence_len(queries)?;

    let heads = queries
        .heads()
        .iter()
        .zip(keys.heads())
        .zip(values.heads())
        .map(|((query_head, key_head), value_head)| {
            let mut output_rows = Vec::with_capacity(sequence_len);
            for query_idx in 0..sequence_len {
                let start = attention_window
                    .map(|window| query_idx.saturating_add(1).saturating_sub(window))
                    .unwrap_or(0);
                output_rows.push(attention_output_row(
                    &query_head[query_idx],
                    &key_head[start..=query_idx],
                    &value_head[start..=query_idx],
                )?);
            }
            Ok(output_rows)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RasterAttentionHeadSequence::from_heads(heads))
}

pub fn causal_attention_heads_with_cache(
    queries: &RasterAttentionHeadSequence,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    donor_cache: Option<&RasterKvCache>,
    attention_window: Option<usize>,
) -> Result<RasterAttentionHeadSequence> {
    let sequence_len = attention_sequence_len(queries)?;
    let query_width = attention_head_width(queries)?;
    let key_width = attention_head_width(keys)?;
    let value_width = attention_head_width(values)?;
    if query_width != key_width || query_width != value_width {
        bail!(
            "attention head width mismatch: query {query_width}, key {key_width}, value {value_width}"
        );
    }
    let kv_head_count = keys.head_count();
    if kv_head_count == 0 {
        bail!("attention requires at least one KV head");
    }
    if values.head_count() != kv_head_count {
        bail!(
            "attention key/value head count mismatch: {} vs {}",
            kv_head_count,
            values.head_count()
        );
    }
    if queries.head_count() % kv_head_count != 0 {
        bail!(
            "attention query head count {} must be divisible by KV head count {}",
            queries.head_count(),
            kv_head_count
        );
    }
    let key_sequence_len = attention_sequence_len(keys)?;
    let value_sequence_len = attention_sequence_len(values)?;
    if key_sequence_len != sequence_len || value_sequence_len != sequence_len {
        bail!(
            "attention sequence length mismatch: query {sequence_len}, key {key_sequence_len}, value {value_sequence_len}"
        );
    }
    if let Some(cache) = donor_cache {
        if cache.head_count() != kv_head_count {
            bail!(
                "attention donor cache head count mismatch: {} vs {}",
                cache.head_count(),
                kv_head_count
            );
        }
    }

    let kv_groups = queries.head_count() / kv_head_count;
    let heads = queries
        .heads()
        .iter()
        .enumerate()
        .map(|(query_head_idx, query_head)| {
            let kv_head_idx = query_head_idx / kv_groups;
            let key_head = &keys.heads()[kv_head_idx];
            let value_head = &values.heads()[kv_head_idx];
            let mut output_rows = Vec::with_capacity(sequence_len);
            for query_idx in 0..sequence_len {
                let start = attention_window
                    .map(|window| query_idx.saturating_add(1).saturating_sub(window))
                    .unwrap_or(0);
                let row_count = query_idx + 1 - start;
                let (key_rows, value_rows) = if let Some(cache) = donor_cache {
                    (
                        cache.key_rows_window(kv_head_idx, start, row_count)?,
                        cache.value_rows_window(kv_head_idx, start, row_count)?,
                    )
                } else {
                    (
                        key_head[start..=query_idx].to_vec(),
                        value_head[start..=query_idx].to_vec(),
                    )
                };
                output_rows.push(attention_output_row(
                    &query_head[query_idx],
                    &key_rows,
                    &value_rows,
                )?);
            }
            Ok(output_rows)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RasterAttentionHeadSequence::from_heads(heads))
}

pub fn init_attention_row_state(
    queries: &RasterAttentionHeadSequence,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    donor_cache: Option<&RasterKvCache>,
    attention_window: Option<usize>,
) -> Result<RasterAttentionRowState> {
    let sequence_len = attention_sequence_len(queries)?;
    let query_width = attention_head_width(queries)?;
    let key_width = attention_head_width(keys)?;
    let value_width = attention_head_width(values)?;
    if query_width != key_width || query_width != value_width {
        bail!(
            "attention head width mismatch: query {query_width}, key {key_width}, value {value_width}"
        );
    }
    let kv_head_count = keys.head_count();
    if kv_head_count == 0 {
        bail!("attention requires at least one KV head");
    }
    if values.head_count() != kv_head_count {
        bail!(
            "attention key/value head count mismatch: {} vs {}",
            kv_head_count,
            values.head_count()
        );
    }
    if queries.head_count() % kv_head_count != 0 {
        bail!(
            "attention query head count {} must be divisible by KV head count {}",
            queries.head_count(),
            kv_head_count
        );
    }
    let key_sequence_len = attention_sequence_len(keys)?;
    let value_sequence_len = attention_sequence_len(values)?;
    if key_sequence_len != sequence_len || value_sequence_len != sequence_len {
        bail!(
            "attention sequence length mismatch: query {sequence_len}, key {key_sequence_len}, value {value_sequence_len}"
        );
    }
    if let Some(cache) = donor_cache {
        if cache.head_count() != kv_head_count {
            bail!(
                "attention donor cache head count mismatch: {} vs {}",
                cache.head_count(),
                kv_head_count
            );
        }
    }

    let output_heads = (0..queries.head_count())
        .map(|_| Vec::with_capacity(sequence_len))
        .collect();

    Ok(RasterAttentionRowState {
        queries: queries.clone(),
        keys: keys.clone(),
        values: values.clone(),
        donor_cache: donor_cache.cloned(),
        attention_window,
        next_query_head_idx: 0,
        next_query_token_idx: 0,
        output_heads,
        sequence_len,
        kv_head_count,
        kv_groups: queries.head_count() / kv_head_count,
    })
}

pub fn compute_next_attention_row(
    mut state: RasterAttentionRowState,
) -> Result<(bool, RasterAttentionRowState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let query_head_idx = state.next_query_head_idx;
    let query_idx = state.next_query_token_idx;
    let kv_head_idx = query_head_idx / state.kv_groups;
    let start = state
        .attention_window
        .map(|window| query_idx.saturating_add(1).saturating_sub(window))
        .unwrap_or(0);
    let row_count = query_idx + 1 - start;
    let query_head = state.queries.heads().get(query_head_idx).ok_or_else(|| {
        anyhow!(
            "attention query head {query_head_idx} is out of range for {} heads",
            state.queries.head_count()
        )
    })?;
    let query = query_head.get(query_idx).ok_or_else(|| {
        anyhow!(
            "attention query row {query_idx} is out of range for {} rows",
            query_head.len()
        )
    })?;
    let (key_rows, value_rows) = if let Some(cache) = &state.donor_cache {
        (
            cache.key_rows_window(kv_head_idx, start, row_count)?,
            cache.value_rows_window(kv_head_idx, start, row_count)?,
        )
    } else {
        let key_head = state.keys.heads().get(kv_head_idx).ok_or_else(|| {
            anyhow!(
                "attention key head {kv_head_idx} is out of range for {} heads",
                state.kv_head_count
            )
        })?;
        let value_head = state.values.heads().get(kv_head_idx).ok_or_else(|| {
            anyhow!(
                "attention value head {kv_head_idx} is out of range for {} heads",
                state.kv_head_count
            )
        })?;
        (
            key_head[start..=query_idx].to_vec(),
            value_head[start..=query_idx].to_vec(),
        )
    };

    let output_row = attention_output_row(query, &key_rows, &value_rows)?;
    let output_head_count = state.output_heads.len();
    let output_head = state.output_heads.get_mut(query_head_idx).ok_or_else(|| {
        anyhow!(
            "attention output head {query_head_idx} is out of range for {} heads",
            output_head_count
        )
    })?;
    if output_head.len() != query_idx {
        bail!(
            "attention output head {query_head_idx} has {} rows, expected {query_idx}",
            output_head.len()
        );
    }
    output_head.push(output_row);

    if query_idx + 1 < state.sequence_len {
        state.next_query_token_idx += 1;
    } else {
        state.next_query_head_idx += 1;
        state.next_query_token_idx = 0;
    }

    Ok((false, state))
}

pub fn finalize_attention_row_state(
    state: RasterAttentionRowState,
) -> Result<RasterAttentionHeadSequence> {
    if !state.is_complete() {
        bail!(
            "attention row state finalized at head {} token {}, expected {} heads",
            state.next_query_head_idx,
            state.next_query_token_idx,
            state.queries.head_count()
        );
    }
    if let Some((head_idx, head)) = state
        .output_heads
        .iter()
        .enumerate()
        .find(|(_, head)| head.len() != state.sequence_len)
    {
        bail!(
            "attention output head {head_idx} has {} rows, expected {}",
            head.len(),
            state.sequence_len
        );
    }

    Ok(RasterAttentionHeadSequence::from_heads(state.output_heads))
}

pub fn build_raster_kv_cache(
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    sliding_window: Option<usize>,
) -> Result<RasterKvCache> {
    validate_matching_attention_heads(keys, values, "key/value")?;
    let sequence_len = attention_sequence_len(keys)?;
    let retained = sliding_window.map_or(0, |window| sequence_len.saturating_sub(window));
    RasterKvCache::from_heads(
        keys.heads()
            .iter()
            .map(|head| head[retained..].to_vec())
            .collect(),
        values
            .heads()
            .iter()
            .map(|head| head[retained..].to_vec())
            .collect(),
    )
}

pub fn init_reshape_heads_state(
    input: &RasterActivationSequence,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterReshapeHeadsState> {
    if num_heads == 0 {
        bail!("attention reshape requires at least one head");
    }
    if head_dim == 0 {
        bail!("attention reshape requires non-zero head dimension");
    }
    let expected_width = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("attention reshape width overflowed"))?;
    validate_sequence_width(input, expected_width, "attention reshape input")?;

    Ok(RasterReshapeHeadsState {
        input: input.clone(),
        num_heads,
        head_dim,
        next_row_idx: 0,
        output_heads: vec![Vec::with_capacity(input.len()); num_heads],
    })
}

pub fn compute_next_reshape_heads_row(
    mut state: RasterReshapeHeadsState,
) -> Result<(bool, RasterReshapeHeadsState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let row = state.input.rows().get(state.next_row_idx).ok_or_else(|| {
        anyhow!(
            "attention reshape row {} is out of range for {} rows",
            state.next_row_idx,
            state.input.len()
        )
    })?;
    let acts = row.acts();
    for head_idx in 0..state.num_heads {
        let start = head_idx * state.head_dim;
        let head = state.output_heads.get_mut(head_idx).ok_or_else(|| {
            anyhow!(
                "attention reshape output head {head_idx} is out of range for {} heads",
                state.num_heads
            )
        })?;
        head.push(RasterActivationRow::from_acts(
            acts[start..start + state.head_dim].to_vec(),
        ));
    }
    state.next_row_idx += 1;
    Ok((false, state))
}

pub fn finalize_reshape_heads_state(
    state: RasterReshapeHeadsState,
) -> Result<RasterAttentionHeadSequence> {
    if !state.is_complete() {
        bail!(
            "attention reshape completed {} rows, expected {}",
            state.next_row_idx,
            state.input.len()
        );
    }
    if let Some((head_idx, head)) = state
        .output_heads
        .iter()
        .enumerate()
        .find(|(_, head)| head.len() != state.input.len())
    {
        bail!(
            "attention reshape output head {head_idx} has {} rows, expected {}",
            head.len(),
            state.input.len()
        );
    }
    Ok(RasterAttentionHeadSequence::from_heads(state.output_heads))
}

pub fn init_combine_heads_state(
    heads: &RasterAttentionHeadSequence,
) -> Result<RasterCombineHeadsState> {
    let sequence_len = attention_sequence_len(heads)?;
    attention_head_width(heads)?;
    Ok(RasterCombineHeadsState {
        heads: heads.clone(),
        next_token_idx: 0,
        output_rows: Vec::with_capacity(sequence_len),
        sequence_len,
    })
}

pub fn compute_next_combine_heads_row(
    mut state: RasterCombineHeadsState,
) -> Result<(bool, RasterCombineHeadsState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let mut row = Vec::new();
    for head in state.heads.heads() {
        let head_row = head.get(state.next_token_idx).ok_or_else(|| {
            anyhow!(
                "attention combine row {} is out of range for {} rows",
                state.next_token_idx,
                head.len()
            )
        })?;
        row.extend(head_row.acts());
    }
    state.output_rows.push(RasterActivationRow::from_acts(row));
    state.next_token_idx += 1;
    Ok((false, state))
}

pub fn finalize_combine_heads_state(
    state: RasterCombineHeadsState,
) -> Result<RasterActivationSequence> {
    if !state.is_complete() {
        bail!(
            "attention combine completed {} rows, expected {}",
            state.next_token_idx,
            state.sequence_len
        );
    }
    if state.output_rows.len() != state.sequence_len {
        bail!(
            "attention combine output has {} rows, expected {}",
            state.output_rows.len(),
            state.sequence_len
        );
    }
    Ok(RasterActivationSequence::from_rows(state.output_rows))
}

pub fn init_kv_cache_build_state(
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    sliding_window: Option<usize>,
) -> Result<RasterKvCacheBuildState> {
    validate_matching_attention_heads(keys, values, "key/value")?;
    let sequence_len = attention_sequence_len(keys)?;
    let retained_start = sliding_window.map_or(0, |window| sequence_len.saturating_sub(window));
    Ok(RasterKvCacheBuildState {
        keys: keys.clone(),
        values: values.clone(),
        retained_start,
        next_head_idx: 0,
        next_token_idx: retained_start,
        output_keys: vec![Vec::new(); keys.head_count()],
        output_values: vec![Vec::new(); keys.head_count()],
        sequence_len,
    })
}

pub fn compute_next_kv_cache_row(
    mut state: RasterKvCacheBuildState,
) -> Result<(bool, RasterKvCacheBuildState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    if state.next_token_idx >= state.sequence_len {
        state.next_head_idx += 1;
        state.next_token_idx = state.retained_start;
        return Ok((false, state));
    }

    let key_head = state.keys.heads().get(state.next_head_idx).ok_or_else(|| {
        anyhow!(
            "KV cache build key head {} is out of range for {} heads",
            state.next_head_idx,
            state.keys.head_count()
        )
    })?;
    let value_head = state
        .values
        .heads()
        .get(state.next_head_idx)
        .ok_or_else(|| {
            anyhow!(
                "KV cache build value head {} is out of range for {} heads",
                state.next_head_idx,
                state.values.head_count()
            )
        })?;
    let key_row = key_head.get(state.next_token_idx).ok_or_else(|| {
        anyhow!(
            "KV cache build key row {} is out of range for {} rows",
            state.next_token_idx,
            key_head.len()
        )
    })?;
    let value_row = value_head.get(state.next_token_idx).ok_or_else(|| {
        anyhow!(
            "KV cache build value row {} is out of range for {} rows",
            state.next_token_idx,
            value_head.len()
        )
    })?;
    state.output_keys[state.next_head_idx].push(key_row.clone());
    state.output_values[state.next_head_idx].push(value_row.clone());
    state.next_token_idx += 1;
    Ok((false, state))
}

pub fn finalize_kv_cache_build_state(state: RasterKvCacheBuildState) -> Result<RasterKvCache> {
    if !state.is_complete() {
        bail!(
            "KV cache build finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.keys.head_count()
        );
    }
    let expected_len = state.sequence_len.saturating_sub(state.retained_start);
    if let Some((head_idx, head)) = state
        .output_keys
        .iter()
        .enumerate()
        .find(|(_, head)| head.len() != expected_len)
    {
        bail!(
            "KV cache build key head {head_idx} has {} rows, expected {expected_len}",
            head.len()
        );
    }
    if let Some((head_idx, head)) = state
        .output_values
        .iter()
        .enumerate()
        .find(|(_, head)| head.len() != expected_len)
    {
        bail!(
            "KV cache build value head {head_idx} has {} rows, expected {expected_len}",
            head.len()
        );
    }
    RasterKvCache::from_heads(state.output_keys, state.output_values)
}

fn project_row(
    input: &RasterActivationRow,
    projection_rows: &[Vec<Wgt>],
) -> Result<RasterActivationRow> {
    let width = projection_rows[0].len();
    if input.width() != width {
        bail!(
            "deterministic linear input width mismatch: {} vs {}",
            input.width(),
            width
        );
    }

    let mut output = Vec::with_capacity(projection_rows.len());
    for row in projection_rows {
        output.push(project_row_with_weights(input, row)?);
    }
    Ok(RasterActivationRow::from_acts(output))
}

pub(crate) fn project_row_with_weights(
    input: &RasterActivationRow,
    projection_row: &[Wgt],
) -> Result<Act> {
    if input.width() != projection_row.len() {
        bail!(
            "deterministic linear input width mismatch: {} vs {}",
            input.width(),
            projection_row.len()
        );
    }

    let input_acts = input.acts();
    let mut acc_bits = 0_i64;
    for (act, weight) in input_acts.iter().zip(projection_row) {
        acc_bits = mac_bits(acc_bits, act.to_bits(), weight.to_bits());
    }
    Ok(requantize(Acc::from_bits(acc_bits)))
}

fn validate_non_empty_sequence(input: &RasterActivationSequence, label: &str) -> Result<()> {
    if input.is_empty() {
        bail!("{label} requires at least one activation row");
    }
    Ok(())
}

fn sequence_width(input: &RasterActivationSequence) -> Result<usize> {
    validate_non_empty_sequence(input, "deterministic sequence operation")?;
    let width = input.rows()[0].width();
    if width == 0 {
        bail!("deterministic sequence operation requires non-empty activation rows");
    }
    validate_sequence_width(input, width, "activation sequence")?;
    Ok(width)
}

fn validate_sequence_width(
    input: &RasterActivationSequence,
    expected_width: usize,
    label: &str,
) -> Result<()> {
    validate_non_empty_sequence(input, label)?;
    if expected_width == 0 {
        bail!("{label} requires a non-zero width");
    }
    if let Some((row_idx, row)) = input
        .rows()
        .iter()
        .enumerate()
        .find(|(_, row)| row.width() != expected_width)
    {
        bail!(
            "{label} row {row_idx} has width {}, expected {expected_width}",
            row.width()
        );
    }
    Ok(())
}

fn attention_sequence_len(input: &RasterAttentionHeadSequence) -> Result<usize> {
    let Some(first_head) = input.heads().first() else {
        bail!("attention head sequence requires at least one head");
    };
    if first_head.is_empty() {
        bail!("attention head sequence requires at least one row per head");
    }
    let sequence_len = first_head.len();
    if let Some((head_idx, head)) = input
        .heads()
        .iter()
        .enumerate()
        .find(|(_, head)| head.len() != sequence_len)
    {
        bail!(
            "attention head {head_idx} has {} rows, expected {sequence_len}",
            head.len()
        );
    }
    Ok(sequence_len)
}

fn attention_head_width(input: &RasterAttentionHeadSequence) -> Result<usize> {
    let _sequence_len = attention_sequence_len(input)?;
    let width = input.heads()[0][0].width();
    if width == 0 {
        bail!("attention head rows must have non-zero width");
    }
    for (head_idx, head) in input.heads().iter().enumerate() {
        validate_rows_width(head, width, &format!("attention head {head_idx}"))?;
    }
    Ok(width)
}

fn validate_rows_width(
    rows: &[RasterActivationRow],
    expected_width: usize,
    label: &str,
) -> Result<()> {
    if expected_width == 0 {
        bail!("{label} requires a non-zero width");
    }
    if rows.is_empty() {
        bail!("{label} requires at least one row");
    }
    if let Some((row_idx, row)) = rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.width() != expected_width)
    {
        bail!(
            "{label} row {row_idx} has width {}, expected {expected_width}",
            row.width()
        );
    }
    Ok(())
}

fn validate_matching_attention_heads(
    lhs: &RasterAttentionHeadSequence,
    rhs: &RasterAttentionHeadSequence,
    label: &str,
) -> Result<()> {
    let lhs_len = attention_sequence_len(lhs)?;
    let rhs_len = attention_sequence_len(rhs)?;
    if lhs.head_count() != rhs.head_count() {
        bail!(
            "attention {label} head count mismatch: {} vs {}",
            lhs.head_count(),
            rhs.head_count()
        );
    }
    if lhs_len != rhs_len {
        bail!("attention {label} sequence length mismatch: {lhs_len} vs {rhs_len}");
    }
    let lhs_width = attention_head_width(lhs)?;
    let rhs_width = attention_head_width(rhs)?;
    if lhs_width != rhs_width {
        bail!("attention {label} head width mismatch: {lhs_width} vs {rhs_width}");
    }
    Ok(())
}

fn validate_kv_cache(cache: &RasterKvCache) -> Result<()> {
    if cache.keys.len() != cache.values.len() {
        bail!(
            "KV cache head count mismatch: {} vs {}",
            cache.keys.len(),
            cache.values.len()
        );
    }
    if cache.keys.is_empty() {
        return Ok(());
    }
    let expected_len = cache.keys[0].len();
    let expected_width = cache
        .keys
        .iter()
        .chain(cache.values.iter())
        .find_map(|head| head.first().map(RasterActivationRow::width))
        .unwrap_or(0);
    for (head_idx, (key_head, value_head)) in cache.keys.iter().zip(&cache.values).enumerate() {
        if key_head.len() != expected_len {
            bail!(
                "KV cache key head {head_idx} has {} rows, expected {expected_len}",
                key_head.len()
            );
        }
        if value_head.len() != expected_len {
            bail!(
                "KV cache value head {head_idx} has {} rows, expected {expected_len}",
                value_head.len()
            );
        }
        if expected_width != 0 {
            validate_rows_width(
                key_head,
                expected_width,
                &format!("KV cache key head {head_idx}"),
            )?;
            validate_rows_width(
                value_head,
                expected_width,
                &format!("KV cache value head {head_idx}"),
            )?;
        }
    }
    Ok(())
}

fn validate_projection_rows(projection_rows: &[Vec<Wgt>]) -> Result<()> {
    let Some(first_row) = projection_rows.first() else {
        bail!("deterministic linear projection requires at least one projection row");
    };
    let width = first_row.len();
    if width == 0 {
        bail!("deterministic linear projection rows must have non-zero width");
    }
    if let Some((row_idx, row)) = projection_rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != width)
    {
        bail!(
            "deterministic linear projection row {row_idx} has width {}, expected {width}",
            row.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        add_sequences, apply_rope_to_heads, attention_output_row, build_raster_kv_cache,
        causal_attention_heads, causal_attention_heads_with_cache, combine_attention_heads,
        compute_next_attention_row, compute_next_combine_heads_row, compute_next_head_unary_row,
        compute_next_kv_cache_row, compute_next_reshape_heads_row,
        compute_next_sequence_binary_row, compute_next_sequence_unary_row,
        finalize_attention_row_state, finalize_combine_heads_state, finalize_head_unary_row_state,
        finalize_kv_cache_build_state, finalize_reshape_heads_state,
        finalize_sequence_binary_row_state, finalize_sequence_unary_row_state, gelu_sequence,
        init_attention_row_state, init_combine_heads_state, init_head_rms_norm_row_state,
        init_kv_cache_build_state, init_reshape_heads_state, init_rope_row_state,
        init_sequence_add_row_state, init_sequence_gelu_row_state, init_sequence_mul_row_state,
        init_sequence_rms_norm_row_state, init_sequence_scale_row_state,
        init_value_rms_norm_row_state, mul_sequences, project_sequence,
        project_sequence_with_prefill_source, project_sequence_with_prefill_source_chunked,
        project_sequence_with_source, project_sequence_with_source_chunked, reshape_sequence_heads,
        rms_norm_heads, rms_norm_sequence, scale_sequence, value_rms_norm_heads,
        RasterActivationRow, RasterActivationSequence, RasterAttentionHeadSequence, RasterKvCache,
    };
    use crate::raster_authoring::AuthRead;
    use crate::shared::det_num::{
        add_sat, attention_score, attention_softmax, attention_weighted_sum, gelu_pytorch_tanh_act,
        mul_sat, rms_norm, rope_rotate_pairs, scale_act, value_rms_norm, Acc, Act, Wgt,
    };
    use crate::shared::raster_prefill_layer::{
        GemmaPrefillLayerMatrixKind, GemmaPrefillLayerMatrixRowRequest,
    };
    use crate::shared::raster_prefill_ple::GemmaPleModelProjectionRowRequest;
    use anyhow::{anyhow, Result};
    use std::collections::HashMap;

    #[test]
    fn scaling_sequence_uses_det_num_act_scalar_semantics() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.5), Act::from_num(-2.0)],
            vec![Act::from_num(0.25), Act::from_num(4.0)],
        ]);

        let scaled = scale_sequence(&input, Some(Act::from_num(0.5))).expect("scale should work");

        assert_eq!(
            scaled_bits(&scaled),
            vec![
                vec![
                    scale_act(Act::from_num(1.5), Act::from_num(0.5)).to_bits(),
                    scale_act(Act::from_num(-2.0), Act::from_num(0.5)).to_bits(),
                ],
                vec![
                    scale_act(Act::from_num(0.25), Act::from_num(0.5)).to_bits(),
                    scale_act(Act::from_num(4.0), Act::from_num(0.5)).to_bits(),
                ],
            ]
        );
    }

    #[test]
    fn adding_sequences_preserves_shape_and_saturates() {
        let lhs = RasterActivationSequence::from_act_bits(vec![
            vec![i32::MAX, Act::from_num(0.25).to_bits()],
            vec![Act::from_num(-0.5).to_bits(), Act::from_num(1.0).to_bits()],
        ]);
        let rhs = RasterActivationSequence::from_act_bits(vec![
            vec![1, Act::from_num(0.25).to_bits()],
            vec![Act::from_num(1.0).to_bits(), Act::from_num(-0.25).to_bits()],
        ]);

        let added = add_sequences(&lhs, &rhs).expect("add should work");

        assert_eq!(
            scaled_bits(&added),
            vec![
                vec![
                    i32::MAX,
                    add_sat(Act::from_num(0.25), Act::from_num(0.25)).to_bits(),
                ],
                vec![
                    add_sat(Act::from_num(-0.5), Act::from_num(1.0)).to_bits(),
                    add_sat(Act::from_num(1.0), Act::from_num(-0.25)).to_bits(),
                ],
            ]
        );
        assert_eq!(added.len(), 2);
        assert_eq!(added.width().expect("width"), 2);
    }

    #[test]
    fn projection_sequence_matches_hand_computed_det_num_fixture() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);
        let projection_rows = vec![
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            vec![Wgt::from_num(0.0), Wgt::from_num(2.0)],
        ];

        let projected = project_sequence(&input, &projection_rows).expect("project should work");

        assert_eq!(
            scaled_bits(&projected),
            vec![
                vec![
                    Act::from_num(1.0).to_bits(),
                    Act::from_num(-0.875).to_bits(),
                    Act::from_num(1.0).to_bits(),
                ],
                vec![
                    Act::from_num(1.5).to_bits(),
                    Act::from_num(1.5).to_bits(),
                    Act::from_num(4.0).to_bits(),
                ],
            ]
        );
    }

    #[test]
    fn projection_can_read_rows_from_authenticated_source() {
        let source = ProjectionSource::new(HashMap::from([
            ((0, 0), vec![Wgt::from_num(0.5), Wgt::from_num(1.0)]),
            ((0, 1), vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)]),
        ]));
        let input =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let projected =
            project_sequence_with_source(&input, &source, 0, 2).expect("project should work");

        assert_eq!(
            scaled_bits(&projected),
            vec![vec![
                Act::from_num(1.0).to_bits(),
                Act::from_num(-0.875).to_bits(),
            ]]
        );
    }

    #[test]
    fn projection_chunk_size_does_not_change_output() {
        let source = ProjectionSource::new(HashMap::from([
            ((0, 0), vec![Wgt::from_num(0.5), Wgt::from_num(1.0)]),
            ((0, 1), vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)]),
            ((0, 2), vec![Wgt::from_num(2.0), Wgt::from_num(-2.0)]),
        ]));
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);

        let one_row =
            project_sequence_with_source_chunked(&input, &source, 0, 3, 1).expect("project");
        let two_rows =
            project_sequence_with_source_chunked(&input, &source, 0, 3, 2).expect("project");
        let oversized =
            project_sequence_with_source_chunked(&input, &source, 0, 3, 10).expect("project");

        assert_eq!(scaled_bits(&one_row), scaled_bits(&two_rows));
        assert_eq!(scaled_bits(&one_row), scaled_bits(&oversized));
    }

    #[test]
    fn projection_rejects_zero_rows_per_tile() {
        let source = ProjectionSource::new(HashMap::new());
        let input =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let error = project_sequence_with_source_chunked(&input, &source, 0, 1, 0)
            .expect_err("zero rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn prefill_projection_can_read_rows_from_authenticated_source() {
        let source = PrefillProjectionSource::new(HashMap::from([
            (
                (0, GemmaPrefillLayerMatrixKind::Query, 0),
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            ),
            (
                (0, GemmaPrefillLayerMatrixKind::Query, 1),
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ),
        ]));
        let input =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let projected = project_sequence_with_prefill_source(
            &input,
            &source,
            0,
            GemmaPrefillLayerMatrixKind::Query,
            2,
        )
        .expect("project should work");

        assert_eq!(
            scaled_bits(&projected),
            vec![vec![
                Act::from_num(1.0).to_bits(),
                Act::from_num(-0.875).to_bits(),
            ]]
        );
    }

    #[test]
    fn prefill_projection_chunk_size_does_not_change_output() {
        let source = PrefillProjectionSource::new(HashMap::from([
            (
                (0, GemmaPrefillLayerMatrixKind::Query, 0),
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            ),
            (
                (0, GemmaPrefillLayerMatrixKind::Query, 1),
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ),
            (
                (0, GemmaPrefillLayerMatrixKind::Query, 2),
                vec![Wgt::from_num(2.0), Wgt::from_num(-2.0)],
            ),
        ]));
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);

        let one_row = project_sequence_with_prefill_source_chunked(
            &input,
            &source,
            0,
            GemmaPrefillLayerMatrixKind::Query,
            3,
            1,
        )
        .expect("project");
        let two_rows = project_sequence_with_prefill_source_chunked(
            &input,
            &source,
            0,
            GemmaPrefillLayerMatrixKind::Query,
            3,
            2,
        )
        .expect("project");

        assert_eq!(scaled_bits(&one_row), scaled_bits(&two_rows));
    }

    #[test]
    fn multiplying_sequences_uses_det_num_mul_contract() {
        let lhs = RasterActivationSequence::from_acts(vec![vec![
            Act::from_num(2.0),
            Act::from_num(-1.0),
        ]]);
        let rhs = RasterActivationSequence::from_acts(vec![vec![
            Act::from_num(0.25),
            Act::from_num(0.5),
        ]]);

        let multiplied = mul_sequences(&lhs, &rhs).expect("mul should work");

        assert_eq!(
            scaled_bits(&multiplied),
            vec![vec![
                mul_sat(Act::from_num(2.0), Act::from_num(0.25)).to_bits(),
                mul_sat(Act::from_num(-1.0), Act::from_num(0.5)).to_bits(),
            ]]
        );
    }

    #[test]
    fn gelu_sequence_uses_det_num_gelu_contract() {
        let input = RasterActivationSequence::from_acts(vec![vec![
            Act::from_num(-1.0),
            Act::from_num(0.0),
            Act::from_num(1.0),
        ]]);

        let gelu = gelu_sequence(&input).expect("gelu should work");

        assert_eq!(
            scaled_bits(&gelu),
            vec![vec![
                gelu_pytorch_tanh_act(Act::from_num(-1.0)).to_bits(),
                gelu_pytorch_tanh_act(Act::from_num(0.0)).to_bits(),
                gelu_pytorch_tanh_act(Act::from_num(1.0)).to_bits(),
            ]]
        );
    }

    #[test]
    fn chunked_sequence_unary_ops_match_full_helpers() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
            vec![Act::from_num(0.25), Act::from_num(0.75)],
        ]);
        let weights = vec![Wgt::from_num(1.0), Wgt::from_num(0.5)];
        let eps = Acc::from_num(0.001);

        let rms = run_sequence_unary_state(
            init_sequence_rms_norm_row_state(&input, Some(&weights), Some(eps)).expect("init"),
        )
        .expect("rms");
        assert_eq!(
            scaled_bits(&rms),
            scaled_bits(&rms_norm_sequence(&input, Some(&weights), Some(eps)).expect("full"))
        );

        let gelu = run_sequence_unary_state(init_sequence_gelu_row_state(&input).expect("init"))
            .expect("gelu");
        assert_eq!(
            scaled_bits(&gelu),
            scaled_bits(&gelu_sequence(&input).expect("full"))
        );

        let scale = run_sequence_unary_state(
            init_sequence_scale_row_state(&input, Some(Act::from_num(0.5))).expect("init"),
        )
        .expect("scale");
        assert_eq!(
            scaled_bits(&scale),
            scaled_bits(&scale_sequence(&input, Some(Act::from_num(0.5))).expect("full"))
        );
    }

    #[test]
    fn chunked_sequence_unary_ops_fail_closed() {
        let input = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);
        let empty = RasterActivationSequence::from_acts(Vec::new());
        let zero_width = RasterActivationSequence::from_act_bits(vec![Vec::new()]);

        let error = init_sequence_rms_norm_row_state(&input, None, Some(Acc::from_num(0.0)))
            .expect_err("missing weights should fail");
        assert!(error.to_string().contains("canonical norm weights"));

        let error = init_sequence_rms_norm_row_state(&input, Some(&[Wgt::from_num(1.0)]), None)
            .expect_err("missing epsilon should fail");
        assert!(error.to_string().contains("canonical Acc epsilon"));

        let error =
            init_sequence_rms_norm_row_state(&zero_width, Some(&[]), Some(Acc::from_num(0.0)))
                .expect_err("zero-width RMSNorm should fail");
        assert!(error.to_string().contains("non-zero width"));

        let error = init_sequence_gelu_row_state(&empty).expect_err("empty GELU should fail");
        assert!(error
            .to_string()
            .contains("requires at least one activation row"));

        let error =
            init_sequence_scale_row_state(&input, None).expect_err("missing scalar should fail");
        assert!(error.to_string().contains("canonical Act scalar"));
    }

    #[test]
    fn chunked_sequence_binary_ops_match_full_helpers() {
        let lhs = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
            vec![Act::from_num(0.25), Act::from_num(0.75)],
        ]);
        let rhs = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(0.5), Act::from_num(0.5)],
            vec![Act::from_num(-0.25), Act::from_num(1.0)],
        ]);

        let added =
            run_sequence_binary_state(init_sequence_add_row_state(&lhs, &rhs).expect("init add"))
                .expect("add");
        assert_eq!(
            scaled_bits(&added),
            scaled_bits(&add_sequences(&lhs, &rhs).expect("full"))
        );

        let multiplied =
            run_sequence_binary_state(init_sequence_mul_row_state(&lhs, &rhs).expect("init mul"))
                .expect("mul");
        assert_eq!(
            scaled_bits(&multiplied),
            scaled_bits(&mul_sequences(&lhs, &rhs).expect("full"))
        );
    }

    #[test]
    fn chunked_sequence_binary_ops_fail_closed() {
        let lhs = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);
        let wrong_width =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(2.0)]]);
        let wrong_len = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(2.0)],
        ]);

        let error = init_sequence_add_row_state(&lhs, &wrong_width)
            .expect_err("width mismatch should fail");
        assert!(error.to_string().contains("right sequence row 0 has width"));

        let error =
            init_sequence_mul_row_state(&lhs, &wrong_len).expect_err("length mismatch should fail");
        assert!(error.to_string().contains("sequence length mismatch"));
    }

    #[test]
    fn reshape_and_combine_attention_heads_round_trip() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![
                Act::from_num(1.0),
                Act::from_num(2.0),
                Act::from_num(3.0),
                Act::from_num(4.0),
            ],
            vec![
                Act::from_num(5.0),
                Act::from_num(6.0),
                Act::from_num(7.0),
                Act::from_num(8.0),
            ],
        ]);

        let heads = reshape_sequence_heads(&input, 2, 2).expect("reshape should work");
        assert_eq!(heads.head_count(), 2);
        assert_eq!(heads.sequence_len().expect("len"), 2);
        assert_eq!(heads.head_width().expect("width"), 2);

        let combined = combine_attention_heads(&heads).expect("combine should work");
        assert_eq!(scaled_bits(&combined), scaled_bits(&input));
    }

    #[test]
    fn value_rms_norm_heads_matches_det_num_fixture() {
        let heads = RasterAttentionHeadSequence::from_acts(vec![vec![vec![
            Act::from_bits(65_536),
            Act::from_bits(0),
        ]]]);

        let normalized =
            value_rms_norm_heads(&heads, Some(Acc::from_bits(0))).expect("norm should work");

        assert_eq!(
            head_bits(&normalized),
            vec![vec![value_rms_norm(
                &[Act::from_bits(65_536), Act::from_bits(0)],
                Acc::from_bits(0),
            )
            .into_iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()]]
        );
    }

    #[test]
    fn rms_norm_heads_matches_det_num_fixture() {
        let heads = RasterAttentionHeadSequence::from_acts(vec![vec![vec![
            Act::from_bits(65_536),
            Act::from_bits(0),
        ]]]);
        let weights = vec![Wgt::from_bits(32_768), Wgt::from_bits(65_536)];

        let normalized =
            rms_norm_heads(&heads, Some(&weights), Some(Acc::from_bits(0))).expect("norm");

        assert_eq!(
            head_bits(&normalized),
            vec![vec![rms_norm(
                &[Act::from_bits(65_536), Act::from_bits(0)],
                &weights,
                Acc::from_bits(0),
            )
            .into_iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()]]
        );
    }

    #[test]
    fn rope_heads_matches_det_num_fixture() {
        let heads = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(1.0), Act::from_num(0.0)],
        ]]);

        let rotated = apply_rope_to_heads(&heads, 2, 2, Some(Acc::from_num(10_000.0)), 0)
            .expect("rope should work");

        assert_eq!(
            head_bits(&rotated),
            vec![vec![
                rope_rotate_pairs(
                    &[Act::from_num(1.0), Act::from_num(0.0)],
                    2,
                    2,
                    Acc::from_num(10_000.0),
                    0,
                )
                .into_iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
                rope_rotate_pairs(
                    &[Act::from_num(1.0), Act::from_num(0.0)],
                    2,
                    2,
                    Acc::from_num(10_000.0),
                    1,
                )
                .into_iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            ]]
        );
    }

    #[test]
    fn chunked_head_unary_ops_match_full_helpers() {
        let heads = RasterAttentionHeadSequence::from_acts(vec![
            vec![
                vec![Act::from_num(1.0), Act::from_num(0.0)],
                vec![Act::from_num(0.0), Act::from_num(1.0)],
            ],
            vec![
                vec![Act::from_num(0.5), Act::from_num(-0.5)],
                vec![Act::from_num(0.25), Act::from_num(0.75)],
            ],
        ]);
        let weights = vec![Wgt::from_num(1.0), Wgt::from_num(0.5)];
        let eps = Acc::from_num(0.001);

        let head_rms = run_head_unary_state(
            init_head_rms_norm_row_state(&heads, Some(&weights), Some(eps)).expect("init"),
        )
        .expect("head rms");
        assert_eq!(
            head_bits(&head_rms),
            head_bits(&rms_norm_heads(&heads, Some(&weights), Some(eps)).expect("full"))
        );

        let value_rms =
            run_head_unary_state(init_value_rms_norm_row_state(&heads, Some(eps)).expect("init"))
                .expect("value rms");
        assert_eq!(
            head_bits(&value_rms),
            head_bits(&value_rms_norm_heads(&heads, Some(eps)).expect("full"))
        );

        let rope = run_head_unary_state(
            init_rope_row_state(&heads, 2, 2, Some(Acc::from_num(10_000.0)), 3).expect("init rope"),
        )
        .expect("rope");
        assert_eq!(
            head_bits(&rope),
            head_bits(
                &apply_rope_to_heads(&heads, 2, 2, Some(Acc::from_num(10_000.0)), 3).expect("full")
            )
        );
    }

    #[test]
    fn chunked_head_unary_ops_fail_closed() {
        let heads = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);

        let error = init_head_rms_norm_row_state(&heads, None, Some(Acc::from_num(0.0)))
            .expect_err("missing head weights should fail");
        assert!(error.to_string().contains("canonical norm weights"));

        let error = init_value_rms_norm_row_state(&heads, None)
            .expect_err("missing value norm epsilon should fail");
        assert!(error.to_string().contains("canonical Acc epsilon"));

        let error = init_rope_row_state(&heads, 2, 2, Some(Acc::from_num(10_000.0)), 0)
            .expect_err("rotary dim should fail");
        assert!(error.to_string().contains("exceeds attention head width"));

        let error =
            init_rope_row_state(&heads, 1, 2, None, 0).expect_err("missing rope base should fail");
        assert!(error.to_string().contains("canonical Acc base"));
    }

    #[test]
    fn attention_output_row_uses_det_num_attention_contract() {
        let query = RasterActivationRow::from_acts(vec![Act::from_num(1.0), Act::from_num(0.0)]);
        let keys = vec![
            RasterActivationRow::from_acts(vec![Act::from_num(1.0), Act::from_num(0.0)]),
            RasterActivationRow::from_acts(vec![Act::from_num(0.0), Act::from_num(1.0)]),
        ];
        let values = vec![
            RasterActivationRow::from_acts(vec![Act::from_num(2.0), Act::from_num(0.0)]),
            RasterActivationRow::from_acts(vec![Act::from_num(0.0), Act::from_num(4.0)]),
        ];

        let output = attention_output_row(&query, &keys, &values).expect("attention should work");

        let logits = keys
            .iter()
            .map(|key| attention_score(&query.acts(), &key.acts()))
            .collect::<Vec<_>>();
        let weights = attention_softmax(&logits);
        let value_rows = values
            .iter()
            .map(RasterActivationRow::acts)
            .collect::<Vec<_>>();
        assert_eq!(
            output.act_bits(),
            attention_weighted_sum(&weights, &value_rows)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn causal_attention_heads_respects_attention_window() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
            vec![Act::from_num(1.0), Act::from_num(0.0)],
        ]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(2.0)],
            vec![Act::from_num(3.0), Act::from_num(0.0)],
        ]]);

        let output = causal_attention_heads(&queries, &keys, &values, Some(2))
            .expect("attention should work");

        let query_head = &queries.heads()[0];
        let key_head = &keys.heads()[0];
        let value_head = &values.heads()[0];
        let expected_last =
            attention_output_row(&query_head[2], &key_head[1..=2], &value_head[1..=2])
                .expect("last attention output");
        assert_eq!(output.heads()[0][2], expected_last);
    }

    #[test]
    fn grouped_causal_attention_maps_query_heads_to_kv_heads() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
            vec![vec![Act::from_num(3.0)]],
            vec![vec![Act::from_num(4.0)]],
        ]);
        let keys = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
        ]);
        let values = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(5.0)]],
            vec![vec![Act::from_num(7.0)]],
        ]);

        let output = causal_attention_heads_with_cache(&queries, &keys, &values, None, None)
            .expect("grouped attention should work");

        assert_eq!(output.head_count(), 4);
        assert_eq!(
            output.heads()[0][0].act_bits(),
            &[Act::from_num(5.0).to_bits()]
        );
        assert_eq!(
            output.heads()[1][0].act_bits(),
            &[Act::from_num(5.0).to_bits()]
        );
        assert_eq!(
            output.heads()[2][0].act_bits(),
            &[Act::from_num(7.0).to_bits()]
        );
        assert_eq!(
            output.heads()[3][0].act_bits(),
            &[Act::from_num(7.0).to_bits()]
        );
    }

    #[test]
    fn donor_cache_attention_reads_prior_cache_rows() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let current_keys = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(9.0)],
            vec![Act::from_num(9.0)],
        ]]);
        let current_values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(9.0)],
            vec![Act::from_num(9.0)],
        ]]);
        let donor_cache = RasterKvCache::from_heads(
            vec![vec![
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
            ]],
            vec![vec![
                RasterActivationRow::from_acts(vec![Act::from_num(3.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(4.0)]),
            ]],
        )
        .expect("cache should build");

        let output = causal_attention_heads_with_cache(
            &queries,
            &current_keys,
            &current_values,
            Some(&donor_cache),
            Some(1),
        )
        .expect("donor attention should work");

        assert_eq!(
            output.heads()[0][0].act_bits(),
            &[Act::from_num(3.0).to_bits()]
        );
        assert_eq!(
            output.heads()[0][1].act_bits(),
            &[Act::from_num(4.0).to_bits()]
        );
    }

    #[test]
    fn attention_row_state_matches_full_attention() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(2.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(4.0)],
        ]]);

        let row_output =
            run_attention_row_state(&queries, &keys, &values, None, None).expect("row attention");
        let full_output = causal_attention_heads_with_cache(&queries, &keys, &values, None, None)
            .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn attention_row_state_matches_sliding_window_attention() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
            vec![Act::from_num(1.0), Act::from_num(0.0)],
        ]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(2.0)],
            vec![Act::from_num(3.0), Act::from_num(0.0)],
        ]]);

        let row_output = run_attention_row_state(&queries, &keys, &values, None, Some(2))
            .expect("row attention");
        let full_output =
            causal_attention_heads_with_cache(&queries, &keys, &values, None, Some(2))
                .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn attention_row_state_maps_grouped_query_heads_to_kv_heads() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
            vec![vec![Act::from_num(3.0)]],
            vec![vec![Act::from_num(4.0)]],
        ]);
        let keys = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
        ]);
        let values = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(5.0)]],
            vec![vec![Act::from_num(7.0)]],
        ]);

        let row_output =
            run_attention_row_state(&queries, &keys, &values, None, None).expect("row attention");
        let full_output = causal_attention_heads_with_cache(&queries, &keys, &values, None, None)
            .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn attention_row_state_matches_donor_cache_attention() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let current_keys = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(9.0)],
            vec![Act::from_num(9.0)],
        ]]);
        let current_values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(9.0)],
            vec![Act::from_num(9.0)],
        ]]);
        let donor_cache = RasterKvCache::from_heads(
            vec![vec![
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
            ]],
            vec![vec![
                RasterActivationRow::from_acts(vec![Act::from_num(3.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(4.0)]),
            ]],
        )
        .expect("cache should build");

        let row_output = run_attention_row_state(
            &queries,
            &current_keys,
            &current_values,
            Some(&donor_cache),
            Some(1),
        )
        .expect("row attention");
        let full_output = causal_attention_heads_with_cache(
            &queries,
            &current_keys,
            &current_values,
            Some(&donor_cache),
            Some(1),
        )
        .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn attention_row_state_fails_closed_for_invalid_shapes() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let mismatched_width_keys = RasterAttentionHeadSequence::from_acts(vec![vec![vec![
            Act::from_num(1.0),
            Act::from_num(2.0),
        ]]]);
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let error = init_attention_row_state(&queries, &mismatched_width_keys, &values, None, None)
            .expect_err("width mismatch should fail");
        assert!(error.to_string().contains("head width mismatch"));

        let mismatched_len_keys = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let error = init_attention_row_state(&queries, &mismatched_len_keys, &values, None, None)
            .expect_err("sequence length mismatch should fail");
        assert!(error.to_string().contains("sequence length mismatch"));

        let grouped_queries = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
            vec![vec![Act::from_num(3.0)]],
        ]);
        let grouped_keys = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
        ]);
        let grouped_values = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
        ]);
        let error =
            init_attention_row_state(&grouped_queries, &grouped_keys, &grouped_values, None, None)
                .expect_err("non-divisible grouped heads should fail");
        assert!(error.to_string().contains("must be divisible"));

        let donor_cache = RasterKvCache::from_heads(
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                1.0,
            )])]],
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                1.0,
            )])]],
        )
        .expect("cache should build");
        let error = init_attention_row_state(
            &grouped_keys,
            &grouped_keys,
            &grouped_values,
            Some(&donor_cache),
            None,
        )
        .expect_err("donor cache head mismatch should fail");
        assert!(error
            .to_string()
            .contains("donor cache head count mismatch"));
    }

    #[test]
    fn build_raster_kv_cache_retains_sliding_window_suffix() {
        let keys = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(2.0)],
            vec![Act::from_num(3.0)],
        ]]);
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(4.0)],
            vec![Act::from_num(5.0)],
            vec![Act::from_num(6.0)],
        ]]);

        let cache = build_raster_kv_cache(&keys, &values, Some(2)).expect("cache should build");

        assert_eq!(cache.head_count(), 1);
        assert_eq!(cache.current_len(), 2);
        assert_eq!(
            cache.keys()[0]
                .iter()
                .map(|row| row.act_bits().to_vec())
                .collect::<Vec<_>>(),
            vec![
                vec![Act::from_num(2.0).to_bits()],
                vec![Act::from_num(3.0).to_bits()],
            ]
        );
        assert_eq!(
            cache.values()[0]
                .iter()
                .map(|row| row.act_bits().to_vec())
                .collect::<Vec<_>>(),
            vec![
                vec![Act::from_num(5.0).to_bits()],
                vec![Act::from_num(6.0).to_bits()],
            ]
        );
    }

    #[test]
    fn chunked_layout_helpers_match_full_helpers() {
        let sequence = RasterActivationSequence::from_acts(vec![
            vec![
                Act::from_num(1.0),
                Act::from_num(2.0),
                Act::from_num(3.0),
                Act::from_num(4.0),
            ],
            vec![
                Act::from_num(5.0),
                Act::from_num(6.0),
                Act::from_num(7.0),
                Act::from_num(8.0),
            ],
        ]);

        let reshaped =
            run_reshape_heads_state(init_reshape_heads_state(&sequence, 2, 2).expect("init"))
                .expect("reshape");
        let full_reshaped = reshape_sequence_heads(&sequence, 2, 2).expect("full reshape");
        assert_eq!(head_bits(&reshaped), head_bits(&full_reshaped));

        let combined = run_combine_heads_state(init_combine_heads_state(&reshaped).expect("init"))
            .expect("combine");
        assert_eq!(
            scaled_bits(&combined),
            scaled_bits(&combine_attention_heads(&reshaped).expect("full combine"))
        );

        let cache = run_kv_cache_build_state(
            init_kv_cache_build_state(&reshaped, &full_reshaped, None).expect("init cache"),
        )
        .expect("cache");
        let full_cache =
            build_raster_kv_cache(&reshaped, &full_reshaped, None).expect("full cache");
        assert_eq!(cache, full_cache);

        let sliding_cache = run_kv_cache_build_state(
            init_kv_cache_build_state(&reshaped, &full_reshaped, Some(1)).expect("init cache"),
        )
        .expect("sliding cache");
        let full_sliding_cache =
            build_raster_kv_cache(&reshaped, &full_reshaped, Some(1)).expect("full cache");
        assert_eq!(sliding_cache, full_sliding_cache);
    }

    #[test]
    fn chunked_layout_helpers_fail_closed() {
        let sequence = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);
        let error = init_reshape_heads_state(&sequence, 0, 1).expect_err("zero heads should fail");
        assert!(error.to_string().contains("at least one head"));

        let error = init_reshape_heads_state(&sequence, 1, 0).expect_err("zero dim should fail");
        assert!(error.to_string().contains("non-zero head dimension"));

        let ragged_heads = RasterAttentionHeadSequence::from_heads(vec![
            vec![RasterActivationRow::from_acts(vec![Act::from_num(1.0)])],
            vec![
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
            ],
        ]);
        let error = init_combine_heads_state(&ragged_heads).expect_err("ragged heads should fail");
        assert!(error.to_string().contains("attention head 1 has 2 rows"));

        let keys = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let values = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
        ]);
        let error = init_kv_cache_build_state(&keys, &values, None)
            .expect_err("cache head mismatch should fail");
        assert!(error.to_string().contains("head count mismatch"));
    }

    #[test]
    fn rms_norm_sequence_matches_det_num_fixture() {
        let input = RasterActivationSequence::from_acts(vec![vec![
            Act::from_bits(65_536),
            Act::from_bits(0),
        ]]);
        let weights = vec![Wgt::from_bits(32_768), Wgt::from_bits(65_536)];

        let normalized =
            rms_norm_sequence(&input, Some(&weights), Some(Acc::from_bits(0))).expect("norm");

        assert_eq!(
            scaled_bits(&normalized),
            vec![rms_norm(
                &[Act::from_bits(65_536), Act::from_bits(0)],
                &weights,
                Acc::from_bits(0),
            )
            .into_iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()]
        );
    }

    #[test]
    fn empty_sequence_fails_clearly() {
        let input = RasterActivationSequence::from_acts(Vec::new());

        let error = scale_sequence(&input, Some(Act::from_num(1.0)))
            .expect_err("empty sequence should fail");

        assert!(error
            .to_string()
            .contains("requires at least one activation row"));
    }

    #[test]
    fn width_mismatch_fails_clearly() {
        let lhs = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);
        let rhs =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(2.0)]]);

        let error = add_sequences(&lhs, &rhs).expect_err("width mismatch should fail");

        assert!(error
            .to_string()
            .contains("right sequence row 0 has width 2"));
    }

    #[test]
    fn missing_canonical_inputs_fail_closed() {
        let input = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);

        let scale_error = scale_sequence(&input, None).expect_err("missing scalar should fail");
        assert!(scale_error
            .to_string()
            .contains("requires canonical Act scalar"));

        let norm_error =
            rms_norm_sequence(&input, None, Some(Acc::from_num(0.0))).expect_err("missing norm");
        assert!(norm_error
            .to_string()
            .contains("requires canonical norm weights"));

        let eps_error =
            rms_norm_sequence(&input, Some(&[Wgt::from_num(1.0)]), None).expect_err("missing eps");
        assert!(eps_error
            .to_string()
            .contains("requires canonical Acc epsilon"));
    }

    #[test]
    fn missing_projection_row_from_source_fails_closed() {
        let source = ProjectionSource::new(HashMap::from([(
            (0, 0),
            vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
        )]));
        let input =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let error = project_sequence_with_source(&input, &source, 0, 2)
            .expect_err("missing projection row should fail");

        assert!(error
            .to_string()
            .contains("missing projection row 1 for layer 0"));
    }

    fn scaled_bits(sequence: &RasterActivationSequence) -> Vec<Vec<i32>> {
        sequence
            .rows()
            .iter()
            .map(|row| row.act_bits().to_vec())
            .collect()
    }

    fn run_sequence_unary_state(
        mut state: super::RasterSequenceUnaryState,
    ) -> Result<RasterActivationSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_sequence_unary_row(state)?;
            state = next_state;
        }
        finalize_sequence_unary_row_state(state)
    }

    fn run_sequence_binary_state(
        mut state: super::RasterSequenceBinaryState,
    ) -> Result<RasterActivationSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_sequence_binary_row(state)?;
            state = next_state;
        }
        finalize_sequence_binary_row_state(state)
    }

    fn run_head_unary_state(
        mut state: super::RasterHeadUnaryState,
    ) -> Result<RasterAttentionHeadSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_head_unary_row(state)?;
            state = next_state;
        }
        finalize_head_unary_row_state(state)
    }

    fn run_reshape_heads_state(
        mut state: super::RasterReshapeHeadsState,
    ) -> Result<RasterAttentionHeadSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_reshape_heads_row(state)?;
            state = next_state;
        }
        finalize_reshape_heads_state(state)
    }

    fn run_combine_heads_state(
        mut state: super::RasterCombineHeadsState,
    ) -> Result<RasterActivationSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_combine_heads_row(state)?;
            state = next_state;
        }
        finalize_combine_heads_state(state)
    }

    fn run_kv_cache_build_state(
        mut state: super::RasterKvCacheBuildState,
    ) -> Result<RasterKvCache> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_kv_cache_row(state)?;
            state = next_state;
        }
        finalize_kv_cache_build_state(state)
    }

    fn run_attention_row_state(
        queries: &RasterAttentionHeadSequence,
        keys: &RasterAttentionHeadSequence,
        values: &RasterAttentionHeadSequence,
        donor_cache: Option<&RasterKvCache>,
        attention_window: Option<usize>,
    ) -> Result<RasterAttentionHeadSequence> {
        let mut state =
            init_attention_row_state(queries, keys, values, donor_cache, attention_window)?;
        while !state.is_complete() {
            let (_done, next_state) = compute_next_attention_row(state)?;
            state = next_state;
        }
        finalize_attention_row_state(state)
    }

    fn head_bits(sequence: &RasterAttentionHeadSequence) -> Vec<Vec<Vec<i32>>> {
        sequence
            .heads()
            .iter()
            .map(|head| head.iter().map(|row| row.act_bits().to_vec()).collect())
            .collect()
    }

    struct ProjectionSource {
        rows: HashMap<(usize, usize), Vec<Wgt>>,
    }

    impl ProjectionSource {
        fn new(rows: HashMap<(usize, usize), Vec<Wgt>>) -> Self {
            Self { rows }
        }
    }

    impl AuthRead<GemmaPleModelProjectionRowRequest> for ProjectionSource {
        type Output = Vec<Wgt>;

        fn auth_read(&self, request: GemmaPleModelProjectionRowRequest) -> Result<Self::Output> {
            self.rows
                .get(&(request.layer_idx, request.row_idx))
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "missing projection row {} for layer {}",
                        request.row_idx,
                        request.layer_idx
                    )
                })
        }
    }

    struct PrefillProjectionSource {
        rows: HashMap<(usize, GemmaPrefillLayerMatrixKind, usize), Vec<Wgt>>,
    }

    impl PrefillProjectionSource {
        fn new(rows: HashMap<(usize, GemmaPrefillLayerMatrixKind, usize), Vec<Wgt>>) -> Self {
            Self { rows }
        }
    }

    impl AuthRead<GemmaPrefillLayerMatrixRowRequest> for PrefillProjectionSource {
        type Output = Vec<Wgt>;

        fn auth_read(&self, request: GemmaPrefillLayerMatrixRowRequest) -> Result<Self::Output> {
            self.rows
                .get(&(request.layer_idx, request.matrix, request.row_idx))
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "missing {:?} projection row {} for layer {}",
                        request.matrix,
                        request.row_idx,
                        request.layer_idx
                    )
                })
        }
    }
}
