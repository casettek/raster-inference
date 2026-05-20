use std::fs::File;

use anyhow::{anyhow, bail, Context, Result};

use crate::auth_read;
use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::raster_artifact_store::{RasterArtifactId, RasterArtifactStoreRoots};
use crate::shared::model::transformer::{DetNumMatrix, DetNumTensorSliceSource};
use crate::shared::numerics::det_num::{
    acc_add_sat, act_to_f32, add_sat, attention_score as det_attention_score,
    attention_softmax as det_attention_softmax, attention_softmax_exp_term,
    attention_softmax_raw_weight, attention_softmax_residual,
    attention_weighted_sum as det_attention_weighted_sum, gelu_pytorch_tanh_act, mac_bits, mul_sat,
    requantize, rms_norm as det_rms_norm, rope_rotate_pairs, scale_act,
    value_rms_norm as det_value_rms_norm, Acc, Act, Wgt,
};
use crate::shared::raster_contracts::prefill_layer::{
    GemmaPrefillLayerMatrixKind, GemmaPrefillLayerMatrixRowRequest,
};
use crate::shared::raster_contracts::prefill_ple::GemmaPleModelProjectionRowRequest;
use crate::shared::tensors::raster_row_store::{
    append_head_row_by_source_name_with_roots, append_sequence_row_by_source_name_with_roots,
    finalize_heads_builder_by_source_name_with_roots,
    finalize_kv_cache_builders_by_source_name_with_roots,
    finalize_sequence_builder_by_source_name_with_roots, read_head_row_from_roots,
    read_kv_row_from_roots, read_sequence_row_from_roots, start_sequence_builder_with_roots,
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterAttentionHeadsRef,
    RasterHeadRowRequest, RasterKvCacheRef, RasterKvRowKind, RasterKvRowRequest,
    RasterProjectionOutputBuilderRef, RasterSequenceRowRequest, RasterTensorBuilderRef,
    RasterTensorId,
};

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
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    output_builder_ref: RasterTensorBuilderRef,
    attention_window: Option<usize>,
    phase: RasterAttentionRowPhase,
    attention_id_prefix: String,
    next_query_head_idx: usize,
    next_query_token_idx: usize,
    sequence_len: usize,
    query_head_count: usize,
    kv_head_count: usize,
    kv_groups: usize,
    kv_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterAttentionArtifactRowState {
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    output_source_name: String,
    attention_window: Option<usize>,
    phase: RasterAttentionArtifactRowPhase,
    attention_id_prefix: String,
    next_query_head_idx: usize,
    next_query_token_idx: usize,
    sequence_len: usize,
    query_head_count: usize,
    kv_head_count: usize,
    kv_groups: usize,
    kv_rows_per_tile: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum RasterAttentionRowPhase {
    CollectScores {
        score_builder_ref: RasterTensorBuilderRef,
        next_kv_token_idx: usize,
    },
    FindSoftmaxMax {
        score_ref: RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: Option<usize>,
        max_logit_bits: i32,
    },
    SumSoftmaxExp {
        score_ref: RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
    },
    BuildRawSoftmaxWeights {
        score_ref: RasterActivationSequenceRef,
        raw_weight_builder_ref: RasterTensorBuilderRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
        summed_weight_bits: i32,
    },
    CorrectSoftmaxResidual {
        raw_weight_ref: RasterActivationSequenceRef,
        final_weight_builder_ref: RasterTensorBuilderRef,
        next_weight_row_idx: usize,
        max_index: usize,
        residual_bits: i32,
    },
    ApplyValues {
        weight_ref: RasterActivationSequenceRef,
        next_kv_token_idx: usize,
        weighted_sum_acc_bits: Vec<i64>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum RasterAttentionArtifactRowPhase {
    CollectScores {
        score_source_name: String,
        next_kv_token_idx: usize,
    },
    FindSoftmaxMax {
        score_ref: RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: Option<usize>,
        max_logit_bits: i32,
    },
    SumSoftmaxExp {
        score_ref: RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
    },
    BuildRawSoftmaxWeights {
        score_ref: RasterActivationSequenceRef,
        raw_weight_source_name: String,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
        summed_weight_bits: i32,
    },
    CorrectSoftmaxResidual {
        raw_weight_ref: RasterActivationSequenceRef,
        final_weight_source_name: String,
        next_weight_row_idx: usize,
        max_index: usize,
        residual_bits: i32,
    },
    ApplyValues {
        weight_ref: RasterActivationSequenceRef,
        next_kv_token_idx: usize,
        weighted_sum_acc_bits: Vec<i64>,
    },
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
    input_ref: RasterActivationSequenceRef,
    output_builder_ref: RasterTensorBuilderRef,
    op: RasterSequenceUnaryOp,
    next_row_idx: usize,
    row_count: usize,
    width: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceUnaryArtifactState {
    input_ref: RasterActivationSequenceRef,
    output_source_name: String,
    op: RasterSequenceUnaryOp,
    next_row_idx: usize,
    row_count: usize,
    width: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RasterSequenceBinaryOp {
    Add,
    Mul,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceBinaryState {
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_builder_ref: RasterTensorBuilderRef,
    op: RasterSequenceBinaryOp,
    next_row_idx: usize,
    row_count: usize,
    width: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceBinaryArtifactState {
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_source_name: String,
    op: RasterSequenceBinaryOp,
    next_row_idx: usize,
    row_count: usize,
    width: usize,
    rows_per_tile: usize,
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
    heads_ref: RasterAttentionHeadsRef,
    output_builder_ref: RasterTensorBuilderRef,
    op: RasterHeadUnaryOp,
    next_head_idx: usize,
    next_token_idx: usize,
    head_count: usize,
    sequence_len: usize,
    head_dim: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterHeadUnaryArtifactState {
    heads_ref: RasterAttentionHeadsRef,
    output_source_name: String,
    op: RasterHeadUnaryOp,
    next_head_idx: usize,
    next_token_idx: usize,
    head_count: usize,
    sequence_len: usize,
    head_dim: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterReshapeHeadsState {
    input_ref: RasterActivationSequenceRef,
    output_builder_ref: RasterTensorBuilderRef,
    num_heads: usize,
    head_dim: usize,
    next_row_idx: usize,
    row_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterReshapeHeadsArtifactState {
    input_ref: RasterActivationSequenceRef,
    output_source_name: String,
    num_heads: usize,
    head_dim: usize,
    next_head_idx: usize,
    next_row_idx: usize,
    row_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterCombineHeadsState {
    heads_ref: RasterAttentionHeadsRef,
    output_builder_ref: RasterTensorBuilderRef,
    next_token_idx: usize,
    head_count: usize,
    sequence_len: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterCombineHeadsArtifactState {
    heads_ref: RasterAttentionHeadsRef,
    output_source_name: String,
    next_token_idx: usize,
    head_count: usize,
    sequence_len: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterKvCacheBuildState {
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    output_builder_ref: crate::shared::tensors::raster_row_store::RasterKvCacheBuilderRef,
    retained_start: usize,
    next_head_idx: usize,
    next_token_idx: usize,
    head_count: usize,
    sequence_len: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterKvCacheBuildArtifactState {
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_source_name: String,
    values_source_name: String,
    retained_start: usize,
    next_head_idx: usize,
    next_token_idx: usize,
    head_count: usize,
    current_len: usize,
    sequence_len: usize,
    head_dim: usize,
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
        self.next_query_head_idx >= self.query_head_count
    }

    pub fn next_query_head_idx(&self) -> usize {
        self.next_query_head_idx
    }

    pub fn next_query_token_idx(&self) -> usize {
        self.next_query_token_idx
    }
}

impl RasterAttentionArtifactRowState {
    pub fn is_complete(&self) -> bool {
        self.next_query_head_idx >= self.query_head_count
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
        self.next_row_idx >= self.row_count
    }
}

impl RasterSequenceUnaryArtifactState {
    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

impl RasterSequenceBinaryState {
    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

impl RasterSequenceBinaryArtifactState {
    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

impl RasterHeadUnaryState {
    pub fn is_complete(&self) -> bool {
        self.next_head_idx >= self.head_count
    }
}

impl RasterHeadUnaryArtifactState {
    pub fn is_complete(&self) -> bool {
        self.next_head_idx >= self.head_count
    }
}

impl RasterReshapeHeadsState {
    pub fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

impl RasterReshapeHeadsArtifactState {
    pub fn is_complete(&self) -> bool {
        self.next_head_idx >= self.num_heads
    }
}

impl RasterCombineHeadsState {
    pub fn is_complete(&self) -> bool {
        self.next_token_idx >= self.sequence_len
    }
}

impl RasterCombineHeadsArtifactState {
    pub fn is_complete(&self) -> bool {
        self.next_token_idx >= self.sequence_len
    }
}

impl RasterKvCacheBuildState {
    pub fn is_complete(&self) -> bool {
        self.next_head_idx >= self.head_count
    }
}

impl RasterKvCacheBuildArtifactState {
    pub fn is_complete(&self) -> bool {
        self.next_head_idx >= self.head_count
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
    input_ref: RasterActivationSequenceRef,
    output_builder_ref: RasterProjectionOutputBuilderRef,
    next_token_idx: usize,
    next_projection_row_idx: usize,
    token_count: usize,
    input_width: usize,
    projection_rows: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceProjectionArtifactState {
    input_ref: RasterActivationSequenceRef,
    output_source_name: String,
    current_row_bits: Vec<i32>,
    next_token_idx: usize,
    next_projection_row_idx: usize,
    token_count: usize,
    input_width: usize,
    projection_rows: usize,
    rows_per_tile: usize,
}

impl RasterSequenceProjectionState {
    pub fn next_token_idx(&self) -> usize {
        self.next_token_idx
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn next_projection_row_idx(&self) -> usize {
        self.next_projection_row_idx
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn projection_rows(&self) -> usize {
        self.projection_rows
    }

    pub fn rows_per_tile(&self) -> usize {
        self.rows_per_tile
    }

    pub fn is_complete(&self) -> bool {
        self.next_token_idx >= self.token_count
    }
}

impl RasterSequenceProjectionArtifactState {
    pub fn next_token_idx(&self) -> usize {
        self.next_token_idx
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn next_projection_row_idx(&self) -> usize {
        self.next_projection_row_idx
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn projection_rows(&self) -> usize {
        self.projection_rows
    }

    pub fn rows_per_tile(&self) -> usize {
        self.rows_per_tile
    }

    pub fn is_complete(&self) -> bool {
        self.next_token_idx >= self.token_count
    }
}

pub fn validate_projection_rows_per_tile(rows_per_tile: usize) -> Result<()> {
    if rows_per_tile == 0 {
        bail!("raster projection rows per tile must be greater than zero");
    }
    Ok(())
}

pub fn validate_attention_kv_rows_per_tile(rows_per_tile: usize) -> Result<()> {
    if rows_per_tile == 0 {
        bail!("raster attention KV rows per tile must be greater than zero");
    }
    Ok(())
}

pub fn validate_sequence_rows_per_tile(rows_per_tile: usize) -> Result<()> {
    if rows_per_tile == 0 {
        bail!("raster sequence rows per tile must be greater than zero");
    }
    Ok(())
}

pub fn validate_head_rows_per_tile(rows_per_tile: usize) -> Result<()> {
    if rows_per_tile == 0 {
        bail!("raster head rows per tile must be greater than zero");
    }
    Ok(())
}

pub fn init_sequence_projection_state(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    projection_rows: usize,
    rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }
    validate_projection_rows_per_tile(rows_per_tile)?;
    let input_width = sequence_width(input)?;
    let input_ref = store.insert_activation_sequence(
        RasterTensorId::new("sequence.projection.input")?,
        input.clone(),
    )?;
    let output_builder_ref = store.start_projection_output_builder(
        RasterTensorId::new("sequence.projection.output")?,
        input.len(),
        projection_rows,
    )?;

    Ok(RasterSequenceProjectionState {
        input_ref,
        output_builder_ref,
        next_token_idx: 0,
        next_projection_row_idx: 0,
        token_count: input.len(),
        input_width,
        projection_rows,
        rows_per_tile,
    })
}

pub fn init_sequence_projection_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    projection_rows: usize,
    rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }
    validate_projection_rows_per_tile(rows_per_tile)?;
    let (token_count, input_width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    let output_builder_ref =
        store.start_projection_output_builder(output_id, token_count, projection_rows)?;

    Ok(RasterSequenceProjectionState {
        input_ref,
        output_builder_ref,
        next_token_idx: 0,
        next_projection_row_idx: 0,
        token_count,
        input_width,
        projection_rows,
        rows_per_tile,
    })
}

pub fn append_projection_chunk_to_state(
    state: &mut RasterSequenceProjectionState,
    store: &mut AuthenticatedRasterTensorStore,
    projection_rows: &[Vec<Wgt>],
) -> Result<()> {
    if state.is_complete() {
        bail!(
            "raster projection already completed {} token rows",
            state.token_count
        );
    }
    if projection_rows.is_empty() {
        bail!("raster projection chunk requires at least one projection row");
    }
    if projection_rows.len() > state.rows_per_tile {
        bail!(
            "raster projection chunk has {} rows, exceeding rows_per_tile {}",
            projection_rows.len(),
            state.rows_per_tile
        );
    }
    if projection_rows.len()
        > state
            .projection_rows
            .saturating_sub(state.next_projection_row_idx)
    {
        bail!(
            "raster projection chunk overshoots projection rows: start {}, len {}, total {}",
            state.next_projection_row_idx,
            projection_rows.len(),
            state.projection_rows
        );
    }

    let input_row = auth_read!(
        store,
        RasterSequenceRowRequest {
            tensor_ref: state.input_ref.clone(),
            row_idx: state.next_token_idx,
        }
    )?;
    if input_row.width() != state.input_width {
        bail!(
            "raster projection input row {} has width {}, expected {}",
            state.next_token_idx,
            input_row.width(),
            state.input_width
        );
    }
    let output_bits = projection_rows
        .iter()
        .map(|projection_row| {
            project_row_with_weights(&input_row, projection_row)
                .map(|projected| projected.to_bits())
        })
        .collect::<Result<Vec<_>>>()?;
    store.append_projection_output_chunk(
        &mut state.output_builder_ref,
        state.next_token_idx,
        state.next_projection_row_idx,
        &output_bits,
    )?;
    state.next_token_idx = state.output_builder_ref.next_token_idx();
    state.next_projection_row_idx = state.output_builder_ref.next_projection_row_idx();
    Ok(())
}

pub fn finalize_sequence_projection_state(
    state: RasterSequenceProjectionState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    if state.next_token_idx != state.token_count || state.next_projection_row_idx != 0 {
        bail!(
            "raster projection completed token {}, projection row {}, expected {} complete token rows",
            state.next_token_idx,
            state.next_projection_row_idx,
            state.token_count
        );
    }

    let output_ref = store.finalize_projection_output_builder(state.output_builder_ref)?;
    store.materialize_sequence(&output_ref)
}

pub fn finalize_sequence_projection_state_ref(
    state: RasterSequenceProjectionState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    if state.next_token_idx != state.token_count || state.next_projection_row_idx != 0 {
        bail!(
            "raster projection completed token {}, projection row {}, expected {} complete token rows",
            state.next_token_idx,
            state.next_projection_row_idx,
            state.token_count
        );
    }

    store.finalize_projection_output_builder(state.output_builder_ref)
}

pub fn init_sequence_projection_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    projection_rows: usize,
    rows_per_tile: usize,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterSequenceProjectionArtifactState,
)> {
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }
    validate_projection_rows_per_tile(rows_per_tile)?;
    let (token_count, input_width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        token_count,
        projection_rows,
    )?;

    Ok((
        artifact_store_roots,
        RasterSequenceProjectionArtifactState {
            input_ref,
            output_source_name,
            current_row_bits: Vec::new(),
            next_token_idx: 0,
            next_projection_row_idx: 0,
            token_count,
            input_width,
            projection_rows,
            rows_per_tile,
        },
    ))
}

pub fn append_projection_chunk_to_artifact_state(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterSequenceProjectionArtifactState,
    projection_rows: &[Vec<Wgt>],
) -> Result<(
    RasterArtifactStoreRoots,
    RasterSequenceProjectionArtifactState,
)> {
    if state.is_complete() {
        bail!(
            "raster projection already completed {} token rows",
            state.token_count
        );
    }
    if projection_rows.is_empty() {
        bail!("raster projection chunk requires at least one projection row");
    }
    if projection_rows.len() > state.rows_per_tile {
        bail!(
            "raster projection chunk has {} rows, exceeding rows_per_tile {}",
            projection_rows.len(),
            state.rows_per_tile
        );
    }
    if projection_rows.len()
        > state
            .projection_rows
            .saturating_sub(state.next_projection_row_idx)
    {
        bail!(
            "raster projection chunk overshoots projection rows: start {}, len {}, total {}",
            state.next_projection_row_idx,
            projection_rows.len(),
            state.projection_rows
        );
    }

    let input_row = read_sequence_row_from_roots(
        &artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: state.input_ref.clone(),
            row_idx: state.next_token_idx,
        },
    )?;
    if input_row.width() != state.input_width {
        bail!(
            "raster projection input row {} has width {}, expected {}",
            state.next_token_idx,
            input_row.width(),
            state.input_width
        );
    }
    let output_bits = projection_rows
        .iter()
        .map(|projection_row| {
            project_row_with_weights(&input_row, projection_row)
                .map(|projected| projected.to_bits())
        })
        .collect::<Result<Vec<_>>>()?;
    state.current_row_bits.extend(output_bits);
    state.next_projection_row_idx = state.current_row_bits.len();
    if state.current_row_bits.len() == state.projection_rows {
        let row_bits = std::mem::take(&mut state.current_row_bits);
        artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &artifact_store_roots,
            &state.output_source_name,
            state.next_token_idx,
            RasterActivationRow::from_act_bits(row_bits),
        )?;
        state.next_token_idx += 1;
        state.next_projection_row_idx = 0;
    }
    Ok((artifact_store_roots, state))
}

pub fn finalize_sequence_projection_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceProjectionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if state.next_token_idx != state.token_count || state.next_projection_row_idx != 0 {
        bail!(
            "raster projection completed token {}, projection row {}, expected {} complete token rows",
            state.next_token_idx,
            state.next_projection_row_idx,
            state.token_count
        );
    }
    if !state.current_row_bits.is_empty() {
        bail!(
            "raster projection finalized with partial row width {}",
            state.current_row_bits.len()
        );
    }
    let output_id = RasterTensorId::new(state.output_source_name.clone())?;
    finalize_sequence_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        output_id,
    )
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
    let mut store = AuthenticatedRasterTensorStore::new();
    let mut state =
        init_sequence_projection_state(&mut store, input, projection_rows, rows_per_tile)?;
    while !state.is_complete() {
        let end = state
            .next_projection_row_idx()
            .saturating_add(state.rows_per_tile())
            .min(state.projection_rows());
        let mut rows = Vec::with_capacity(end - state.next_projection_row_idx());
        for row_idx in state.next_projection_row_idx()..end {
            rows.push(auth_read!(
                source,
                GemmaPleModelProjectionRowRequest { layer_idx, row_idx }
            )?);
        }
        append_projection_chunk_to_state(&mut state, &mut store, &rows)?;
    }
    finalize_sequence_projection_state(state, &mut store)
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

pub(crate) fn det_num_matrix_row_wgts(
    matrix: &DetNumMatrix,
    row_idx: usize,
    label: &str,
) -> Result<Vec<Wgt>> {
    if row_idx >= matrix.rows {
        bail!(
            "Gemma prefill {label} row {row_idx} is out of range for {} rows",
            matrix.rows
        );
    }
    let start = row_idx
        .checked_mul(matrix.cols)
        .ok_or_else(|| anyhow!("matrix row range overflowed"))?;
    let end = start
        .checked_add(matrix.cols)
        .ok_or_else(|| anyhow!("matrix row range overflowed"))?;
    let encoded_row = matrix
        .values
        .get(start..end)
        .ok_or_else(|| anyhow!("matrix row range is out of bounds"))?;
    Ok(encoded_row.iter().copied().map(Wgt::from_bits).collect())
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
    let mut store = AuthenticatedRasterTensorStore::new();
    let mut state =
        init_sequence_projection_state(&mut store, input, projection_rows, rows_per_tile)?;
    while !state.is_complete() {
        let end = state
            .next_projection_row_idx()
            .saturating_add(state.rows_per_tile())
            .min(state.projection_rows());
        let mut rows = Vec::with_capacity(end - state.next_projection_row_idx());
        for row_idx in state.next_projection_row_idx()..end {
            rows.push(auth_read!(
                source,
                GemmaPrefillLayerMatrixRowRequest {
                    layer_idx,
                    matrix,
                    row_idx,
                }
            )?);
        }
        append_projection_chunk_to_state(&mut state, &mut store, &rows)?;
    }
    finalize_sequence_projection_state(state, &mut store)
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
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical norm weights"))?;
    let eps = eps.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    validate_sequence_width(input, norm_weights.len(), "deterministic RMSNorm input")?;
    let input_ref = store
        .insert_activation_sequence(RasterTensorId::new("sequence.unary.input")?, input.clone())?;
    let output_builder_ref = store.start_sequence_builder(
        RasterTensorId::new("sequence.unary.output")?,
        input.len(),
        input.width()?,
    )?;

    Ok(RasterSequenceUnaryState {
        input_ref,
        output_builder_ref,
        op: RasterSequenceUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
        next_row_idx: 0,
        row_count: input.len(),
        width: input.width()?,
        rows_per_tile,
    })
}

pub fn init_sequence_rms_norm_row_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical norm weights"))?;
    let eps = eps.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    if norm_weights.len() != width {
        bail!(
            "deterministic RMSNorm input width mismatch: row width {}, norm width {}",
            width,
            norm_weights.len()
        );
    }
    let output_builder_ref = store.start_sequence_builder(output_id, row_count, width)?;
    Ok(RasterSequenceUnaryState {
        input_ref,
        output_builder_ref,
        op: RasterSequenceUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
        next_row_idx: 0,
        row_count,
        width,
        rows_per_tile,
    })
}

pub fn init_sequence_gelu_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    validate_non_empty_sequence(input, "deterministic GELU")?;
    let input_ref = store
        .insert_activation_sequence(RasterTensorId::new("sequence.unary.input")?, input.clone())?;
    let output_builder_ref = store.start_sequence_builder(
        RasterTensorId::new("sequence.unary.output")?,
        input.len(),
        input.width()?,
    )?;

    Ok(RasterSequenceUnaryState {
        input_ref,
        output_builder_ref,
        op: RasterSequenceUnaryOp::Gelu,
        next_row_idx: 0,
        row_count: input.len(),
        width: input.width()?,
        rows_per_tile,
    })
}

pub fn init_sequence_gelu_row_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    let output_builder_ref = store.start_sequence_builder(output_id, row_count, width)?;
    Ok(RasterSequenceUnaryState {
        input_ref,
        output_builder_ref,
        op: RasterSequenceUnaryOp::Gelu,
        next_row_idx: 0,
        row_count,
        width,
        rows_per_tile,
    })
}

pub fn init_sequence_scale_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    scalar: Option<Act>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let scalar = scalar
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    validate_non_empty_sequence(input, "deterministic sequence scaling")?;
    let input_ref = store
        .insert_activation_sequence(RasterTensorId::new("sequence.unary.input")?, input.clone())?;
    let output_builder_ref = store.start_sequence_builder(
        RasterTensorId::new("sequence.unary.output")?,
        input.len(),
        input.width()?,
    )?;

    Ok(RasterSequenceUnaryState {
        input_ref,
        output_builder_ref,
        op: RasterSequenceUnaryOp::Scale {
            scalar_bits: scalar.to_bits(),
        },
        next_row_idx: 0,
        row_count: input.len(),
        width: input.width()?,
        rows_per_tile,
    })
}

pub fn init_sequence_scale_row_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    scalar: Option<Act>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let scalar = scalar
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    let output_builder_ref = store.start_sequence_builder(output_id, row_count, width)?;
    Ok(RasterSequenceUnaryState {
        input_ref,
        output_builder_ref,
        op: RasterSequenceUnaryOp::Scale {
            scalar_bits: scalar.to_bits(),
        },
        next_row_idx: 0,
        row_count,
        width,
        rows_per_tile,
    })
}

pub fn compute_next_sequence_unary_row(
    mut state: RasterSequenceUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterSequenceUnaryState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let end = state
        .next_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.row_count);
    while state.next_row_idx < end {
        let row = auth_read!(
            store,
            RasterSequenceRowRequest {
                tensor_ref: state.input_ref.clone(),
                row_idx: state.next_row_idx,
            }
        )?;
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
        store.append_sequence_row(
            &mut state.output_builder_ref,
            state.next_row_idx,
            output_row,
        )?;
        state.next_row_idx += 1;
    }
    Ok((false, state))
}

pub fn finalize_sequence_unary_row_state(
    state: RasterSequenceUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    if !state.is_complete() {
        bail!(
            "sequence unary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }

    let output_ref = store.finalize_sequence_builder(state.output_builder_ref)?;
    store.materialize_sequence(&output_ref)
}

pub fn finalize_sequence_unary_row_state_ref(
    state: RasterSequenceUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    if !state.is_complete() {
        bail!(
            "sequence unary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }

    store.finalize_sequence_builder(state.output_builder_ref)
}

pub fn init_sequence_rms_norm_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceUnaryArtifactState)> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical norm weights"))?;
    let eps = eps.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    if norm_weights.len() != width {
        bail!(
            "deterministic RMSNorm input width mismatch: row width {}, norm width {}",
            width,
            norm_weights.len()
        );
    }
    init_sequence_unary_artifact_state(
        artifact_store_roots,
        input_ref,
        output_id,
        RasterSequenceUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
        row_count,
        width,
        rows_per_tile,
    )
}

pub fn init_sequence_gelu_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceUnaryArtifactState)> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    init_sequence_unary_artifact_state(
        artifact_store_roots,
        input_ref,
        output_id,
        RasterSequenceUnaryOp::Gelu,
        row_count,
        width,
        rows_per_tile,
    )
}

pub fn init_sequence_scale_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    scalar: Option<Act>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceUnaryArtifactState)> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let scalar = scalar
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    init_sequence_unary_artifact_state(
        artifact_store_roots,
        input_ref,
        output_id,
        RasterSequenceUnaryOp::Scale {
            scalar_bits: scalar.to_bits(),
        },
        row_count,
        width,
        rows_per_tile,
    )
}

fn init_sequence_unary_artifact_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    op: RasterSequenceUnaryOp,
    row_count: usize,
    width: usize,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceUnaryArtifactState)> {
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        row_count,
        width,
    )?;
    Ok((
        artifact_store_roots,
        RasterSequenceUnaryArtifactState {
            input_ref,
            output_source_name,
            op,
            next_row_idx: 0,
            row_count,
            width,
            rows_per_tile,
        },
    ))
}

pub fn compute_next_sequence_unary_artifact_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterSequenceUnaryArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceUnaryArtifactState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let end = state
        .next_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.row_count);
    while state.next_row_idx < end {
        let row = read_sequence_row_from_roots(
            &artifact_store_roots,
            RasterSequenceRowRequest {
                tensor_ref: state.input_ref.clone(),
                row_idx: state.next_row_idx,
            },
        )?;
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
        artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &artifact_store_roots,
            &state.output_source_name,
            state.next_row_idx,
            output_row,
        )?;
        state.next_row_idx += 1;
    }
    Ok((false, artifact_store_roots, state))
}

pub fn finalize_sequence_unary_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceUnaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if !state.is_complete() {
        bail!(
            "sequence unary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }
    let output_id = RasterTensorId::new(state.output_source_name.clone())?;
    finalize_sequence_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        output_id,
    )
}

pub fn init_sequence_add_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_binary_row_state(store, lhs, rhs, RasterSequenceBinaryOp::Add, rows_per_tile)
}

pub fn init_sequence_mul_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_binary_row_state(store, lhs, rhs, RasterSequenceBinaryOp::Mul, rows_per_tile)
}

pub fn init_sequence_add_row_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_binary_row_state_from_refs(
        store,
        lhs_ref,
        rhs_ref,
        output_id,
        RasterSequenceBinaryOp::Add,
        rows_per_tile,
    )
}

pub fn init_sequence_mul_row_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_binary_row_state_from_refs(
        store,
        lhs_ref,
        rhs_ref,
        output_id,
        RasterSequenceBinaryOp::Mul,
        rows_per_tile,
    )
}

fn init_sequence_binary_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    op: RasterSequenceBinaryOp,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let width = sequence_width(lhs)?;
    validate_sequence_width(rhs, width, "right sequence")?;
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }
    let lhs_ref = store
        .insert_activation_sequence(RasterTensorId::new("sequence.binary.lhs")?, lhs.clone())?;
    let rhs_ref = store
        .insert_activation_sequence(RasterTensorId::new("sequence.binary.rhs")?, rhs.clone())?;
    let output_builder_ref = store.start_sequence_builder(
        RasterTensorId::new("sequence.binary.output")?,
        lhs.len(),
        width,
    )?;

    Ok(RasterSequenceBinaryState {
        lhs_ref,
        rhs_ref,
        output_builder_ref,
        op,
        next_row_idx: 0,
        row_count: lhs.len(),
        width,
        rows_per_tile,
    })
}

fn init_sequence_binary_row_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    op: RasterSequenceBinaryOp,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let (lhs_rows, width) = lhs_ref.tensor_ref().shape().sequence_metadata()?;
    let (rhs_rows, rhs_width) = rhs_ref.tensor_ref().shape().sequence_metadata()?;
    if width != rhs_width {
        bail!("right sequence row 0 has width {rhs_width}, expected {width}");
    }
    if lhs_rows != rhs_rows {
        bail!("sequence length mismatch: {} vs {}", lhs_rows, rhs_rows);
    }
    let output_builder_ref = store.start_sequence_builder(output_id, lhs_rows, width)?;
    Ok(RasterSequenceBinaryState {
        lhs_ref,
        rhs_ref,
        output_builder_ref,
        op,
        next_row_idx: 0,
        row_count: lhs_rows,
        width,
        rows_per_tile,
    })
}

pub fn compute_next_sequence_binary_row(
    mut state: RasterSequenceBinaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterSequenceBinaryState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let end = state
        .next_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.row_count);
    while state.next_row_idx < end {
        let lhs_row = auth_read!(
            store,
            RasterSequenceRowRequest {
                tensor_ref: state.lhs_ref.clone(),
                row_idx: state.next_row_idx,
            }
        )?;
        let rhs_row = auth_read!(
            store,
            RasterSequenceRowRequest {
                tensor_ref: state.rhs_ref.clone(),
                row_idx: state.next_row_idx,
            }
        )?;
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
        store.append_sequence_row(
            &mut state.output_builder_ref,
            state.next_row_idx,
            output_row,
        )?;
        state.next_row_idx += 1;
    }
    Ok((false, state))
}

pub fn finalize_sequence_binary_row_state(
    state: RasterSequenceBinaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    if !state.is_complete() {
        bail!(
            "sequence binary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }

    let output_ref = store.finalize_sequence_builder(state.output_builder_ref)?;
    store.materialize_sequence(&output_ref)
}

pub fn finalize_sequence_binary_row_state_ref(
    state: RasterSequenceBinaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    if !state.is_complete() {
        bail!(
            "sequence binary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }

    store.finalize_sequence_builder(state.output_builder_ref)
}

pub fn init_sequence_add_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceBinaryArtifactState)> {
    init_sequence_binary_artifact_state_from_refs(
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        output_id,
        RasterSequenceBinaryOp::Add,
        rows_per_tile,
    )
}

pub fn init_sequence_mul_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceBinaryArtifactState)> {
    init_sequence_binary_artifact_state_from_refs(
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        output_id,
        RasterSequenceBinaryOp::Mul,
        rows_per_tile,
    )
}

fn init_sequence_binary_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    op: RasterSequenceBinaryOp,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceBinaryArtifactState)> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let (lhs_rows, lhs_width) = lhs_ref.tensor_ref().shape().sequence_metadata()?;
    let (rhs_rows, rhs_width) = rhs_ref.tensor_ref().shape().sequence_metadata()?;
    if lhs_rows != rhs_rows {
        bail!("sequence length mismatch: {lhs_rows} vs {rhs_rows}");
    }
    if lhs_width != rhs_width {
        bail!("right sequence row 0 has width {rhs_width}, expected {lhs_width}");
    }
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        lhs_rows,
        lhs_width,
    )?;
    Ok((
        artifact_store_roots,
        RasterSequenceBinaryArtifactState {
            lhs_ref,
            rhs_ref,
            output_source_name,
            op,
            next_row_idx: 0,
            row_count: lhs_rows,
            width: lhs_width,
            rows_per_tile,
        },
    ))
}

pub fn compute_next_sequence_binary_artifact_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterSequenceBinaryArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceBinaryArtifactState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let end = state
        .next_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.row_count);
    while state.next_row_idx < end {
        let lhs_row = read_sequence_row_from_roots(
            &artifact_store_roots,
            RasterSequenceRowRequest {
                tensor_ref: state.lhs_ref.clone(),
                row_idx: state.next_row_idx,
            },
        )?;
        let rhs_row = read_sequence_row_from_roots(
            &artifact_store_roots,
            RasterSequenceRowRequest {
                tensor_ref: state.rhs_ref.clone(),
                row_idx: state.next_row_idx,
            },
        )?;
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
        artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &artifact_store_roots,
            &state.output_source_name,
            state.next_row_idx,
            output_row,
        )?;
        state.next_row_idx += 1;
    }
    Ok((false, artifact_store_roots, state))
}

pub fn finalize_sequence_binary_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceBinaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if !state.is_complete() {
        bail!(
            "sequence binary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }
    let output_id = RasterTensorId::new(state.output_source_name.clone())?;
    finalize_sequence_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        output_id,
    )
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
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    validate_head_rows_per_tile(rows_per_tile)?;
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
        store,
        heads,
        RasterHeadUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
        rows_per_tile,
    )
}

pub fn init_value_rms_norm_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    validate_head_rows_per_tile(rows_per_tile)?;
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    attention_head_width(heads)?;
    init_head_unary_row_state(
        store,
        heads,
        RasterHeadUnaryOp::ValueRmsNorm {
            eps_bits: eps.to_bits(),
        },
        rows_per_tile,
    )
}

pub fn init_rope_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    validate_head_rows_per_tile(rows_per_tile)?;
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
        store,
        heads,
        RasterHeadUnaryOp::Rope {
            rotary_dim,
            freq_base_dim,
            base_bits: base.map(Acc::to_bits),
            position_offset,
        },
        rows_per_tile,
    )
}

pub fn init_head_rms_norm_row_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    validate_head_rows_per_tile(rows_per_tile)?;
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical norm weights"))?;
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical Acc epsilon"))?;
    let (_, _, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    if norm_weights.len() != head_dim {
        bail!(
            "deterministic head RMSNorm weight width mismatch: {} vs {}",
            norm_weights.len(),
            head_dim
        );
    }
    init_head_unary_row_state_from_ref(
        store,
        heads_ref,
        output_id,
        RasterHeadUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
        rows_per_tile,
    )
}

pub fn init_value_rms_norm_row_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    validate_head_rows_per_tile(rows_per_tile)?;
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    init_head_unary_row_state_from_ref(
        store,
        heads_ref,
        output_id,
        RasterHeadUnaryOp::ValueRmsNorm {
            eps_bits: eps.to_bits(),
        },
        rows_per_tile,
    )
}

pub fn init_rope_row_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    validate_head_rows_per_tile(rows_per_tile)?;
    let (_, _, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    if rotary_dim != 0 {
        if rotary_dim > head_dim {
            bail!("RoPE rotary_dim {rotary_dim} exceeds attention head width {head_dim}");
        }
        base.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    }
    init_head_unary_row_state_from_ref(
        store,
        heads_ref,
        output_id,
        RasterHeadUnaryOp::Rope {
            rotary_dim,
            freq_base_dim,
            base_bits: base.map(Acc::to_bits),
            position_offset,
        },
        rows_per_tile,
    )
}

fn init_head_unary_row_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
    op: RasterHeadUnaryOp,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    let sequence_len = attention_sequence_len(heads)?;
    let head_count = heads.head_count();
    let head_dim = attention_head_width(heads)?;
    let heads_ref =
        store.insert_attention_heads(RasterTensorId::new("head.unary.input")?, heads.clone())?;
    let output_builder_ref = store.start_heads_builder(
        RasterTensorId::new("head.unary.output")?,
        head_count,
        sequence_len,
        head_dim,
    )?;
    Ok(RasterHeadUnaryState {
        heads_ref,
        output_builder_ref,
        op,
        next_head_idx: 0,
        next_token_idx: 0,
        head_count,
        sequence_len,
        head_dim,
        rows_per_tile,
    })
}

fn init_head_unary_row_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    op: RasterHeadUnaryOp,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    let (head_count, sequence_len, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    let output_builder_ref =
        store.start_heads_builder(output_id, head_count, sequence_len, head_dim)?;
    Ok(RasterHeadUnaryState {
        heads_ref,
        output_builder_ref,
        op,
        next_head_idx: 0,
        next_token_idx: 0,
        head_count,
        sequence_len,
        head_dim,
        rows_per_tile,
    })
}

pub fn compute_next_head_unary_row(
    mut state: RasterHeadUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterHeadUnaryState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let mut rows_processed = 0;
    while rows_processed < state.rows_per_tile && !state.is_complete() {
        let head_idx = state.next_head_idx;
        let token_idx = state.next_token_idx;
        let row = auth_read!(
            store,
            RasterHeadRowRequest {
                tensor_ref: state.heads_ref.clone(),
                head_idx,
                token_idx,
            }
        )?;
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
        store.append_head_row(
            &mut state.output_builder_ref,
            head_idx,
            token_idx,
            output_row,
        )?;

        if token_idx + 1 < state.sequence_len {
            state.next_token_idx += 1;
        } else {
            state.next_head_idx += 1;
            state.next_token_idx = 0;
        }
        rows_processed += 1;
    }

    Ok((false, state))
}

pub fn finalize_head_unary_row_state(
    state: RasterHeadUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadSequence> {
    if !state.is_complete() {
        bail!(
            "head unary state finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.head_count
        );
    }
    let output_ref = store.finalize_heads_builder(state.output_builder_ref)?;
    store.materialize_heads(&output_ref)
}

pub fn finalize_head_unary_row_state_ref(
    state: RasterHeadUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadsRef> {
    if !state.is_complete() {
        bail!(
            "head unary state finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.head_count
        );
    }
    store.finalize_heads_builder(state.output_builder_ref)
}

pub fn init_head_rms_norm_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    validate_head_rows_per_tile(rows_per_tile)?;
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical norm weights"))?;
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic head RMSNorm requires canonical Acc epsilon"))?;
    let (_, _, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    if norm_weights.len() != head_dim {
        bail!(
            "deterministic head RMSNorm weight width mismatch: {} vs {}",
            norm_weights.len(),
            head_dim
        );
    }
    init_head_unary_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        output_id,
        RasterHeadUnaryOp::RmsNorm {
            norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
            eps_bits: eps.to_bits(),
        },
        rows_per_tile,
    )
}

pub fn init_value_rms_norm_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    eps: Option<Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    validate_head_rows_per_tile(rows_per_tile)?;
    let eps =
        eps.ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    init_head_unary_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        output_id,
        RasterHeadUnaryOp::ValueRmsNorm {
            eps_bits: eps.to_bits(),
        },
        rows_per_tile,
    )
}

pub fn init_rope_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    validate_head_rows_per_tile(rows_per_tile)?;
    let (_, _, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    if rotary_dim != 0 {
        if rotary_dim > head_dim {
            bail!("RoPE rotary_dim {rotary_dim} exceeds attention head width {head_dim}");
        }
        base.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    }
    init_head_unary_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        output_id,
        RasterHeadUnaryOp::Rope {
            rotary_dim,
            freq_base_dim,
            base_bits: base.map(Acc::to_bits),
            position_offset,
        },
        rows_per_tile,
    )
}

fn init_head_unary_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    op: RasterHeadUnaryOp,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    let (head_count, sequence_len, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        head_count * sequence_len,
        head_dim,
    )?;
    Ok((
        artifact_store_roots,
        RasterHeadUnaryArtifactState {
            heads_ref,
            output_source_name,
            op,
            next_head_idx: 0,
            next_token_idx: 0,
            head_count,
            sequence_len,
            head_dim,
            rows_per_tile,
        },
    ))
}

pub fn compute_next_head_unary_artifact_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterHeadUnaryArtifactState,
) -> Result<(bool, RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let mut rows_processed = 0;
    while rows_processed < state.rows_per_tile && !state.is_complete() {
        let head_idx = state.next_head_idx;
        let token_idx = state.next_token_idx;
        let row = read_head_row_from_roots(
            &artifact_store_roots,
            RasterHeadRowRequest {
                tensor_ref: state.heads_ref.clone(),
                head_idx,
                token_idx,
            },
        )?;
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
        artifact_store_roots = append_head_row_by_source_name_with_roots(
            &artifact_store_roots,
            &state.output_source_name,
            head_idx,
            token_idx,
            state.sequence_len,
            output_row,
        )?;

        if token_idx + 1 < state.sequence_len {
            state.next_token_idx += 1;
        } else {
            state.next_head_idx += 1;
            state.next_token_idx = 0;
        }
        rows_processed += 1;
    }

    Ok((false, artifact_store_roots, state))
}

pub fn finalize_head_unary_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterHeadUnaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    if !state.is_complete() {
        bail!(
            "head unary state finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.head_count
        );
    }
    let output_id = RasterTensorId::new(state.output_source_name.clone())?;
    finalize_heads_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        output_id,
        state.head_count,
        state.sequence_len,
        state.head_dim,
    )
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
    store: &mut AuthenticatedRasterTensorStore,
    queries: &RasterAttentionHeadSequence,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    donor_cache: Option<&RasterKvCache>,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<RasterAttentionRowState> {
    validate_attention_kv_rows_per_tile(kv_rows_per_tile)?;
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

    let query_ref =
        store.insert_attention_heads(RasterTensorId::new("attention.queries")?, queries.clone())?;
    let key_ref =
        store.insert_attention_heads(RasterTensorId::new("attention.keys")?, keys.clone())?;
    let value_ref =
        store.insert_attention_heads(RasterTensorId::new("attention.values")?, values.clone())?;
    let donor_cache_ref = donor_cache
        .map(|cache| {
            store.insert_kv_cache(
                RasterTensorId::new("attention.donor.keys")?,
                RasterTensorId::new("attention.donor.values")?,
                cache.clone(),
            )
        })
        .transpose()?;
    let output_builder_ref = store.start_heads_builder(
        RasterTensorId::new("attention.output")?,
        queries.head_count(),
        sequence_len,
        query_width,
    )?;
    let (initial_visible_start, initial_visible_rows) =
        attention_visible_range(0, attention_window);
    let phase = init_attention_score_phase(
        store,
        "attention",
        0,
        0,
        initial_visible_start,
        initial_visible_rows,
    )?;

    Ok(RasterAttentionRowState {
        query_ref,
        key_ref,
        value_ref,
        donor_cache_ref,
        output_builder_ref,
        attention_window,
        phase,
        attention_id_prefix: "attention".to_string(),
        next_query_head_idx: 0,
        next_query_token_idx: 0,
        sequence_len,
        query_head_count: queries.head_count(),
        kv_head_count,
        kv_groups: queries.head_count() / kv_head_count,
        kv_rows_per_tile,
    })
}

pub fn init_attention_row_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    output_id: RasterTensorId,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<RasterAttentionRowState> {
    validate_attention_kv_rows_per_tile(kv_rows_per_tile)?;
    let (query_head_count, sequence_len, query_width) =
        query_ref.tensor_ref().shape().heads_metadata()?;
    let (kv_head_count, key_sequence_len, key_width) =
        key_ref.tensor_ref().shape().heads_metadata()?;
    let (value_head_count, value_sequence_len, value_width) =
        value_ref.tensor_ref().shape().heads_metadata()?;
    if query_width != key_width || query_width != value_width {
        bail!(
            "attention head width mismatch: query {query_width}, key {key_width}, value {value_width}"
        );
    }
    if value_head_count != kv_head_count {
        bail!(
            "attention key/value head count mismatch: {} vs {}",
            kv_head_count,
            value_head_count
        );
    }
    if query_head_count % kv_head_count != 0 {
        bail!(
            "attention query head count {} must be divisible by KV head count {}",
            query_head_count,
            kv_head_count
        );
    }
    if key_sequence_len != sequence_len || value_sequence_len != sequence_len {
        bail!(
            "attention sequence length mismatch: query {sequence_len}, key {key_sequence_len}, value {value_sequence_len}"
        );
    }
    if let Some(cache_ref) = &donor_cache_ref {
        let (cache_head_count, _, cache_head_dim) = cache_ref.shape().kv_cache_metadata()?;
        if cache_head_count != kv_head_count {
            bail!(
                "attention donor cache head count mismatch: {} vs {}",
                cache_head_count,
                kv_head_count
            );
        }
        if cache_head_dim != query_width {
            bail!(
                "attention donor cache head width mismatch: {} vs {}",
                cache_head_dim,
                query_width
            );
        }
    }
    let attention_id_prefix = output_id.source_name().to_string();
    let output_builder_ref =
        store.start_heads_builder(output_id, query_head_count, sequence_len, query_width)?;
    let (initial_visible_start, initial_visible_rows) =
        attention_visible_range(0, attention_window);
    let phase = init_attention_score_phase(
        store,
        &attention_id_prefix,
        0,
        0,
        initial_visible_start,
        initial_visible_rows,
    )?;

    Ok(RasterAttentionRowState {
        query_ref,
        key_ref,
        value_ref,
        donor_cache_ref,
        output_builder_ref,
        attention_window,
        phase,
        attention_id_prefix,
        next_query_head_idx: 0,
        next_query_token_idx: 0,
        sequence_len,
        query_head_count,
        kv_head_count,
        kv_groups: query_head_count / kv_head_count,
        kv_rows_per_tile,
    })
}

pub fn init_attention_artifact_row_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    output_id: RasterTensorId,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionArtifactRowState)> {
    validate_attention_kv_rows_per_tile(kv_rows_per_tile)?;
    let (query_head_count, sequence_len, query_width) =
        query_ref.tensor_ref().shape().heads_metadata()?;
    let (kv_head_count, key_sequence_len, key_width) =
        key_ref.tensor_ref().shape().heads_metadata()?;
    let (value_head_count, value_sequence_len, value_width) =
        value_ref.tensor_ref().shape().heads_metadata()?;
    if query_width != key_width || query_width != value_width {
        bail!(
            "attention head width mismatch: query {query_width}, key {key_width}, value {value_width}"
        );
    }
    if value_head_count != kv_head_count {
        bail!(
            "attention key/value head count mismatch: {} vs {}",
            kv_head_count,
            value_head_count
        );
    }
    if query_head_count % kv_head_count != 0 {
        bail!(
            "attention query head count {} must be divisible by KV head count {}",
            query_head_count,
            kv_head_count
        );
    }
    if key_sequence_len != sequence_len || value_sequence_len != sequence_len {
        bail!(
            "attention sequence length mismatch: query {sequence_len}, key {key_sequence_len}, value {value_sequence_len}"
        );
    }
    if let Some(cache_ref) = &donor_cache_ref {
        let (cache_head_count, _, cache_head_dim) = cache_ref.shape().kv_cache_metadata()?;
        if cache_head_count != kv_head_count {
            bail!(
                "attention donor cache head count mismatch: {} vs {}",
                cache_head_count,
                kv_head_count
            );
        }
        if cache_head_dim != query_width {
            bail!(
                "attention donor cache head width mismatch: {} vs {}",
                cache_head_dim,
                query_width
            );
        }
    }
    let attention_id_prefix = output_id.source_name().to_string();
    let output_source_name = attention_id_prefix.clone();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        query_head_count * sequence_len,
        query_width,
    )?;
    let (initial_visible_start, initial_visible_rows) =
        attention_visible_range(0, attention_window);
    let (artifact_store_roots, phase) = init_attention_artifact_score_phase(
        artifact_store_roots,
        &attention_id_prefix,
        0,
        0,
        initial_visible_start,
        initial_visible_rows,
    )?;

    Ok((
        artifact_store_roots,
        RasterAttentionArtifactRowState {
            query_ref,
            key_ref,
            value_ref,
            donor_cache_ref,
            output_source_name,
            attention_window,
            phase,
            attention_id_prefix,
            next_query_head_idx: 0,
            next_query_token_idx: 0,
            sequence_len,
            query_head_count,
            kv_head_count,
            kv_groups: query_head_count / kv_head_count,
            kv_rows_per_tile,
            head_dim: query_width,
        },
    ))
}

fn attention_visible_range(query_idx: usize, attention_window: Option<usize>) -> (usize, usize) {
    let start = attention_window
        .map(|window| query_idx.saturating_add(1).saturating_sub(window))
        .unwrap_or(0);
    (start, query_idx + 1 - start)
}

fn init_attention_score_phase(
    store: &mut AuthenticatedRasterTensorStore,
    id_prefix: &str,
    query_head_idx: usize,
    query_idx: usize,
    visible_start: usize,
    visible_row_count: usize,
) -> Result<RasterAttentionRowPhase> {
    let score_builder_ref = store.start_sequence_builder(
        RasterTensorId::new(format!(
            "{id_prefix}.scores.head_{query_head_idx}.token_{query_idx}"
        ))?,
        visible_row_count,
        1,
    )?;
    Ok(RasterAttentionRowPhase::CollectScores {
        score_builder_ref,
        next_kv_token_idx: visible_start,
    })
}

fn init_attention_artifact_score_phase(
    artifact_store_roots: RasterArtifactStoreRoots,
    id_prefix: &str,
    query_head_idx: usize,
    query_idx: usize,
    visible_start: usize,
    visible_row_count: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionArtifactRowPhase)> {
    let score_source_name = format!("{id_prefix}.scores.head_{query_head_idx}.token_{query_idx}");
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&score_source_name)?,
        visible_row_count,
        1,
    )?;
    Ok((
        artifact_store_roots,
        RasterAttentionArtifactRowPhase::CollectScores {
            score_source_name,
            next_kv_token_idx: visible_start,
        },
    ))
}

pub fn compute_next_attention_row(
    mut state: RasterAttentionRowState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterAttentionRowState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let query_head_idx = state.next_query_head_idx;
    let query_idx = state.next_query_token_idx;
    let kv_head_idx = query_head_idx / state.kv_groups;
    let (start, row_count) = attention_visible_range(query_idx, state.attention_window);
    let query = auth_read!(
        store,
        RasterHeadRowRequest {
            tensor_ref: state.query_ref.clone(),
            head_idx: query_head_idx,
            token_idx: query_idx,
        }
    )?;

    match state.phase.clone() {
        RasterAttentionRowPhase::CollectScores {
            mut score_builder_ref,
            next_kv_token_idx,
        } => {
            let end = next_kv_token_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(start + row_count);
            for token_idx in next_kv_token_idx..end {
                let key_row = read_attention_key_row(&state, store, kv_head_idx, token_idx)?;
                if key_row.width() != query.width() {
                    bail!(
                        "attention key row ({kv_head_idx}, {token_idx}) has width {}, expected {}",
                        key_row.width(),
                        query.width()
                    );
                }
                let score = det_attention_score(&query.acts(), &key_row.acts());
                store.append_sequence_row(
                    &mut score_builder_ref,
                    token_idx - start,
                    RasterActivationRow::from_acts(vec![score]),
                )?;
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.scores head={} token={} kv_rows={}..{} of {}",
                query_head_idx,
                query_idx,
                next_kv_token_idx,
                end,
                start + row_count
            ));
            if end < start + row_count {
                state.phase = RasterAttentionRowPhase::CollectScores {
                    score_builder_ref,
                    next_kv_token_idx: end,
                };
                return Ok((false, state));
            }

            let score_ref = store.finalize_sequence_builder(score_builder_ref)?;
            state.phase = RasterAttentionRowPhase::FindSoftmaxMax {
                score_ref,
                next_score_row_idx: 0,
                max_index: None,
                max_logit_bits: 0,
            };
            Ok((false, state))
        }
        RasterAttentionRowPhase::FindSoftmaxMax {
            score_ref,
            next_score_row_idx,
            mut max_index,
            mut max_logit_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            for row_idx in next_score_row_idx..end {
                let score = read_attention_scalar_row(store, &score_ref, row_idx, "score")?;
                let score_bits = score.to_bits();
                if max_index.is_none() || score_bits > max_logit_bits {
                    max_index = Some(row_idx);
                    max_logit_bits = score_bits;
                }
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.max head={} token={} score_rows={}..{} of {}",
                query_head_idx, query_idx, next_score_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionRowPhase::FindSoftmaxMax {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                };
                return Ok((false, state));
            }

            let max_index = max_index
                .ok_or_else(|| anyhow!("attention softmax requires at least one score"))?;
            state.phase = RasterAttentionRowPhase::SumSoftmaxExp {
                score_ref,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits: 0,
            };
            Ok((false, state))
        }
        RasterAttentionRowPhase::SumSoftmaxExp {
            score_ref,
            next_score_row_idx,
            max_index,
            max_logit_bits,
            mut sum_exp_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            let max_logit = Act::from_bits(max_logit_bits);
            let mut sum_exp = Acc::from_bits(sum_exp_bits);
            for row_idx in next_score_row_idx..end {
                let score = read_attention_scalar_row(store, &score_ref, row_idx, "score")?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                sum_exp = acc_add_sat(sum_exp, exp_term);
            }
            sum_exp_bits = sum_exp.to_bits();
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.exp_sum head={} token={} score_rows={}..{} of {}",
                query_head_idx, query_idx, next_score_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionRowPhase::SumSoftmaxExp {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                };
                return Ok((false, state));
            }
            if sum_exp_bits == 0 {
                bail!("attention softmax exp sum is zero");
            }

            let raw_weight_builder_ref = store.start_sequence_builder(
                RasterTensorId::new(format!(
                    "{}.raw_weights.head_{query_head_idx}.token_{query_idx}",
                    state.attention_id_prefix
                ))?,
                row_count,
                1,
            )?;
            state.phase = RasterAttentionRowPhase::BuildRawSoftmaxWeights {
                score_ref,
                raw_weight_builder_ref,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits,
                summed_weight_bits: 0,
            };
            Ok((false, state))
        }
        RasterAttentionRowPhase::BuildRawSoftmaxWeights {
            score_ref,
            mut raw_weight_builder_ref,
            next_score_row_idx,
            max_index,
            max_logit_bits,
            sum_exp_bits,
            mut summed_weight_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            let max_logit = Act::from_bits(max_logit_bits);
            let sum_exp = Acc::from_bits(sum_exp_bits);
            let mut summed_weight = Act::from_bits(summed_weight_bits);
            for row_idx in next_score_row_idx..end {
                let score = read_attention_scalar_row(store, &score_ref, row_idx, "score")?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                let weight = attention_softmax_raw_weight(exp_term, sum_exp);
                store.append_sequence_row(
                    &mut raw_weight_builder_ref,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
                summed_weight = add_sat(summed_weight, weight);
            }
            summed_weight_bits = summed_weight.to_bits();
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.raw_weights head={} token={} score_rows={}..{} of {}",
                query_head_idx, query_idx, next_score_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionRowPhase::BuildRawSoftmaxWeights {
                    score_ref,
                    raw_weight_builder_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                    summed_weight_bits,
                };
                return Ok((false, state));
            }

            let raw_weight_ref = store.finalize_sequence_builder(raw_weight_builder_ref)?;
            let final_weight_builder_ref = store.start_sequence_builder(
                RasterTensorId::new(format!(
                    "{}.weights.head_{query_head_idx}.token_{query_idx}",
                    state.attention_id_prefix
                ))?,
                row_count,
                1,
            )?;
            let residual = attention_softmax_residual(Act::from_bits(summed_weight_bits));
            state.phase = RasterAttentionRowPhase::CorrectSoftmaxResidual {
                raw_weight_ref,
                final_weight_builder_ref,
                next_weight_row_idx: 0,
                max_index,
                residual_bits: residual.to_bits(),
            };
            Ok((false, state))
        }
        RasterAttentionRowPhase::CorrectSoftmaxResidual {
            raw_weight_ref,
            mut final_weight_builder_ref,
            next_weight_row_idx,
            max_index,
            residual_bits,
        } => {
            let end = next_weight_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            let residual = Act::from_bits(residual_bits);
            for row_idx in next_weight_row_idx..end {
                let mut weight =
                    read_attention_scalar_row(store, &raw_weight_ref, row_idx, "weight")?;
                if row_idx == max_index {
                    weight = add_sat(weight, residual);
                }
                store.append_sequence_row(
                    &mut final_weight_builder_ref,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.correct head={} token={} weight_rows={}..{} of {}",
                query_head_idx, query_idx, next_weight_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionRowPhase::CorrectSoftmaxResidual {
                    raw_weight_ref,
                    final_weight_builder_ref,
                    next_weight_row_idx: end,
                    max_index,
                    residual_bits,
                };
                return Ok((false, state));
            }

            let weight_ref = store.finalize_sequence_builder(final_weight_builder_ref)?;
            state.phase = RasterAttentionRowPhase::ApplyValues {
                weight_ref,
                next_kv_token_idx: start,
                weighted_sum_acc_bits: vec![0; query.width()],
            };
            Ok((false, state))
        }
        RasterAttentionRowPhase::ApplyValues {
            weight_ref,
            next_kv_token_idx,
            mut weighted_sum_acc_bits,
        } => {
            let end = next_kv_token_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(start + row_count);
            for token_idx in next_kv_token_idx..end {
                let weight =
                    read_attention_scalar_row(store, &weight_ref, token_idx - start, "weight")?;
                let value_row = read_attention_value_row(&state, store, kv_head_idx, token_idx)?;
                if value_row.width() != weighted_sum_acc_bits.len() {
                    bail!(
                        "attention value row ({kv_head_idx}, {token_idx}) has width {}, expected {}",
                        value_row.width(),
                        weighted_sum_acc_bits.len()
                    );
                }
                for (acc_bits, value) in weighted_sum_acc_bits.iter_mut().zip(value_row.acts()) {
                    *acc_bits = mac_bits(*acc_bits, value.to_bits(), weight.to_bits());
                }
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.values head={} token={} kv_rows={}..{} of {}",
                query_head_idx,
                query_idx,
                next_kv_token_idx,
                end,
                start + row_count
            ));
            if end < start + row_count {
                state.phase = RasterAttentionRowPhase::ApplyValues {
                    weight_ref,
                    next_kv_token_idx: end,
                    weighted_sum_acc_bits,
                };
                return Ok((false, state));
            }

            let output_row = RasterActivationRow::from_acts(
                weighted_sum_acc_bits
                    .into_iter()
                    .map(|bits| requantize(Acc::from_bits(bits)))
                    .collect(),
            );
            store.append_head_row(
                &mut state.output_builder_ref,
                query_head_idx,
                query_idx,
                output_row,
            )?;

            if query_idx + 1 < state.sequence_len {
                state.next_query_token_idx += 1;
            } else {
                state.next_query_head_idx += 1;
                state.next_query_token_idx = 0;
            }
            if !state.is_complete() {
                let (next_start, next_row_count) =
                    attention_visible_range(state.next_query_token_idx, state.attention_window);
                state.phase = init_attention_score_phase(
                    store,
                    &state.attention_id_prefix,
                    state.next_query_head_idx,
                    state.next_query_token_idx,
                    next_start,
                    next_row_count,
                )?;
            }
            Ok((false, state))
        }
    }
}

pub fn compute_next_attention_artifact_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterAttentionArtifactRowState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterAttentionArtifactRowState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let query_head_idx = state.next_query_head_idx;
    let query_idx = state.next_query_token_idx;
    let kv_head_idx = query_head_idx / state.kv_groups;
    let (start, row_count) = attention_visible_range(query_idx, state.attention_window);
    let query = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: state.query_ref.clone(),
            head_idx: query_head_idx,
            token_idx: query_idx,
        },
    )?;

    match state.phase.clone() {
        RasterAttentionArtifactRowPhase::CollectScores {
            score_source_name,
            next_kv_token_idx,
        } => {
            let end = next_kv_token_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(start + row_count);
            for token_idx in next_kv_token_idx..end {
                let key_row = read_attention_artifact_key_row(
                    &state,
                    &artifact_store_roots,
                    kv_head_idx,
                    token_idx,
                )?;
                if key_row.width() != query.width() {
                    bail!(
                        "attention key row ({kv_head_idx}, {token_idx}) has width {}, expected {}",
                        key_row.width(),
                        query.width()
                    );
                }
                let score = det_attention_score(&query.acts(), &key_row.acts());
                artifact_store_roots = append_sequence_row_by_source_name_with_roots(
                    &artifact_store_roots,
                    &score_source_name,
                    token_idx - start,
                    RasterActivationRow::from_acts(vec![score]),
                )?;
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.scores head={} token={} kv_rows={}..{} of {}",
                query_head_idx,
                query_idx,
                next_kv_token_idx,
                end,
                start + row_count
            ));
            if end < start + row_count {
                state.phase = RasterAttentionArtifactRowPhase::CollectScores {
                    score_source_name,
                    next_kv_token_idx: end,
                };
                return Ok((false, artifact_store_roots, state));
            }

            let score_id = RasterTensorId::new(score_source_name.clone())?;
            let (next_roots, score_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &score_source_name,
                score_id,
            )?;
            artifact_store_roots = next_roots;
            state.phase = RasterAttentionArtifactRowPhase::FindSoftmaxMax {
                score_ref,
                next_score_row_idx: 0,
                max_index: None,
                max_logit_bits: 0,
            };
            Ok((false, artifact_store_roots, state))
        }
        RasterAttentionArtifactRowPhase::FindSoftmaxMax {
            score_ref,
            next_score_row_idx,
            mut max_index,
            mut max_logit_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            for row_idx in next_score_row_idx..end {
                let score = read_attention_artifact_scalar_row(
                    &artifact_store_roots,
                    &score_ref,
                    row_idx,
                    "score",
                )?;
                let score_bits = score.to_bits();
                if max_index.is_none() || score_bits > max_logit_bits {
                    max_index = Some(row_idx);
                    max_logit_bits = score_bits;
                }
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.max head={} token={} score_rows={}..{} of {}",
                query_head_idx, query_idx, next_score_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionArtifactRowPhase::FindSoftmaxMax {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }

            let max_index = max_index
                .ok_or_else(|| anyhow!("attention softmax requires at least one score"))?;
            state.phase = RasterAttentionArtifactRowPhase::SumSoftmaxExp {
                score_ref,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits: 0,
            };
            Ok((false, artifact_store_roots, state))
        }
        RasterAttentionArtifactRowPhase::SumSoftmaxExp {
            score_ref,
            next_score_row_idx,
            max_index,
            max_logit_bits,
            mut sum_exp_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            let max_logit = Act::from_bits(max_logit_bits);
            let mut sum_exp = Acc::from_bits(sum_exp_bits);
            for row_idx in next_score_row_idx..end {
                let score = read_attention_artifact_scalar_row(
                    &artifact_store_roots,
                    &score_ref,
                    row_idx,
                    "score",
                )?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                sum_exp = acc_add_sat(sum_exp, exp_term);
            }
            sum_exp_bits = sum_exp.to_bits();
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.exp_sum head={} token={} score_rows={}..{} of {}",
                query_head_idx, query_idx, next_score_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionArtifactRowPhase::SumSoftmaxExp {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }
            if sum_exp_bits == 0 {
                bail!("attention softmax exp sum is zero");
            }

            let raw_weight_source_name = format!(
                "{}.raw_weights.head_{query_head_idx}.token_{query_idx}",
                state.attention_id_prefix
            );
            artifact_store_roots = start_sequence_builder_with_roots(
                &artifact_store_roots,
                RasterArtifactId::new(&raw_weight_source_name)?,
                row_count,
                1,
            )?;
            state.phase = RasterAttentionArtifactRowPhase::BuildRawSoftmaxWeights {
                score_ref,
                raw_weight_source_name,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits,
                summed_weight_bits: 0,
            };
            Ok((false, artifact_store_roots, state))
        }
        RasterAttentionArtifactRowPhase::BuildRawSoftmaxWeights {
            score_ref,
            raw_weight_source_name,
            next_score_row_idx,
            max_index,
            max_logit_bits,
            sum_exp_bits,
            mut summed_weight_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            let max_logit = Act::from_bits(max_logit_bits);
            let sum_exp = Acc::from_bits(sum_exp_bits);
            let mut summed_weight = Act::from_bits(summed_weight_bits);
            for row_idx in next_score_row_idx..end {
                let score = read_attention_artifact_scalar_row(
                    &artifact_store_roots,
                    &score_ref,
                    row_idx,
                    "score",
                )?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                let weight = attention_softmax_raw_weight(exp_term, sum_exp);
                artifact_store_roots = append_sequence_row_by_source_name_with_roots(
                    &artifact_store_roots,
                    &raw_weight_source_name,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
                summed_weight = add_sat(summed_weight, weight);
            }
            summed_weight_bits = summed_weight.to_bits();
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.raw_weights head={} token={} score_rows={}..{} of {}",
                query_head_idx, query_idx, next_score_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionArtifactRowPhase::BuildRawSoftmaxWeights {
                    score_ref,
                    raw_weight_source_name,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                    summed_weight_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }

            let raw_weight_id = RasterTensorId::new(raw_weight_source_name.clone())?;
            let (next_roots, raw_weight_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &raw_weight_source_name,
                raw_weight_id,
            )?;
            artifact_store_roots = next_roots;
            let final_weight_source_name = format!(
                "{}.weights.head_{query_head_idx}.token_{query_idx}",
                state.attention_id_prefix
            );
            artifact_store_roots = start_sequence_builder_with_roots(
                &artifact_store_roots,
                RasterArtifactId::new(&final_weight_source_name)?,
                row_count,
                1,
            )?;
            let residual = attention_softmax_residual(Act::from_bits(summed_weight_bits));
            state.phase = RasterAttentionArtifactRowPhase::CorrectSoftmaxResidual {
                raw_weight_ref,
                final_weight_source_name,
                next_weight_row_idx: 0,
                max_index,
                residual_bits: residual.to_bits(),
            };
            Ok((false, artifact_store_roots, state))
        }
        RasterAttentionArtifactRowPhase::CorrectSoftmaxResidual {
            raw_weight_ref,
            final_weight_source_name,
            next_weight_row_idx,
            max_index,
            residual_bits,
        } => {
            let end = next_weight_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(row_count);
            let residual = Act::from_bits(residual_bits);
            for row_idx in next_weight_row_idx..end {
                let mut weight = read_attention_artifact_scalar_row(
                    &artifact_store_roots,
                    &raw_weight_ref,
                    row_idx,
                    "weight",
                )?;
                if row_idx == max_index {
                    weight = add_sat(weight, residual);
                }
                artifact_store_roots = append_sequence_row_by_source_name_with_roots(
                    &artifact_store_roots,
                    &final_weight_source_name,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.softmax.correct head={} token={} weight_rows={}..{} of {}",
                query_head_idx, query_idx, next_weight_row_idx, end, row_count
            ));
            if end < row_count {
                state.phase = RasterAttentionArtifactRowPhase::CorrectSoftmaxResidual {
                    raw_weight_ref,
                    final_weight_source_name,
                    next_weight_row_idx: end,
                    max_index,
                    residual_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }

            let final_weight_id = RasterTensorId::new(final_weight_source_name.clone())?;
            let (next_roots, weight_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &final_weight_source_name,
                final_weight_id,
            )?;
            artifact_store_roots = next_roots;
            state.phase = RasterAttentionArtifactRowPhase::ApplyValues {
                weight_ref,
                next_kv_token_idx: start,
                weighted_sum_acc_bits: vec![0; query.width()],
            };
            Ok((false, artifact_store_roots, state))
        }
        RasterAttentionArtifactRowPhase::ApplyValues {
            weight_ref,
            next_kv_token_idx,
            mut weighted_sum_acc_bits,
        } => {
            let end = next_kv_token_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(start + row_count);
            for token_idx in next_kv_token_idx..end {
                let weight = read_attention_artifact_scalar_row(
                    &artifact_store_roots,
                    &weight_ref,
                    token_idx - start,
                    "weight",
                )?;
                let value_row = read_attention_artifact_value_row(
                    &state,
                    &artifact_store_roots,
                    kv_head_idx,
                    token_idx,
                )?;
                if value_row.width() != weighted_sum_acc_bits.len() {
                    bail!(
                        "attention value row ({kv_head_idx}, {token_idx}) has width {}, expected {}",
                        value_row.width(),
                        weighted_sum_acc_bits.len()
                    );
                }
                for (acc_bits, value) in weighted_sum_acc_bits.iter_mut().zip(value_row.acts()) {
                    *acc_bits = mac_bits(*acc_bits, value.to_bits(), weight.to_bits());
                }
            }
            crate::trace::trace_event(format!(
                "progress prefill.attention.values head={} token={} kv_rows={}..{} of {}",
                query_head_idx,
                query_idx,
                next_kv_token_idx,
                end,
                start + row_count
            ));
            if end < start + row_count {
                state.phase = RasterAttentionArtifactRowPhase::ApplyValues {
                    weight_ref,
                    next_kv_token_idx: end,
                    weighted_sum_acc_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }

            let output_row = RasterActivationRow::from_acts(
                weighted_sum_acc_bits
                    .into_iter()
                    .map(|bits| requantize(Acc::from_bits(bits)))
                    .collect(),
            );
            artifact_store_roots = append_head_row_by_source_name_with_roots(
                &artifact_store_roots,
                &state.output_source_name,
                query_head_idx,
                query_idx,
                state.sequence_len,
                output_row,
            )?;

            if query_idx + 1 < state.sequence_len {
                state.next_query_token_idx += 1;
            } else {
                state.next_query_head_idx += 1;
                state.next_query_token_idx = 0;
            }
            if !state.is_complete() {
                let (next_start, next_row_count) =
                    attention_visible_range(state.next_query_token_idx, state.attention_window);
                let (next_roots, phase) = init_attention_artifact_score_phase(
                    artifact_store_roots,
                    &state.attention_id_prefix,
                    state.next_query_head_idx,
                    state.next_query_token_idx,
                    next_start,
                    next_row_count,
                )?;
                artifact_store_roots = next_roots;
                state.phase = phase;
            }
            Ok((false, artifact_store_roots, state))
        }
    }
}

fn read_attention_key_row(
    state: &RasterAttentionRowState,
    store: &AuthenticatedRasterTensorStore,
    kv_head_idx: usize,
    token_idx: usize,
) -> Result<RasterActivationRow> {
    if let Some(cache_ref) = &state.donor_cache_ref {
        auth_read!(
            store,
            RasterKvRowRequest {
                cache_ref: cache_ref.clone(),
                row_kind: RasterKvRowKind::Key,
                head_idx: kv_head_idx,
                token_idx,
            }
        )
    } else {
        auth_read!(
            store,
            RasterHeadRowRequest {
                tensor_ref: state.key_ref.clone(),
                head_idx: kv_head_idx,
                token_idx,
            }
        )
    }
}

fn read_attention_artifact_key_row(
    state: &RasterAttentionArtifactRowState,
    roots: &RasterArtifactStoreRoots,
    kv_head_idx: usize,
    token_idx: usize,
) -> Result<RasterActivationRow> {
    if let Some(cache_ref) = &state.donor_cache_ref {
        read_kv_row_from_roots(
            roots,
            RasterKvRowRequest {
                cache_ref: cache_ref.clone(),
                row_kind: RasterKvRowKind::Key,
                head_idx: kv_head_idx,
                token_idx,
            },
        )
    } else {
        read_head_row_from_roots(
            roots,
            RasterHeadRowRequest {
                tensor_ref: state.key_ref.clone(),
                head_idx: kv_head_idx,
                token_idx,
            },
        )
    }
}

fn read_attention_value_row(
    state: &RasterAttentionRowState,
    store: &AuthenticatedRasterTensorStore,
    kv_head_idx: usize,
    token_idx: usize,
) -> Result<RasterActivationRow> {
    if let Some(cache_ref) = &state.donor_cache_ref {
        auth_read!(
            store,
            RasterKvRowRequest {
                cache_ref: cache_ref.clone(),
                row_kind: RasterKvRowKind::Value,
                head_idx: kv_head_idx,
                token_idx,
            }
        )
    } else {
        auth_read!(
            store,
            RasterHeadRowRequest {
                tensor_ref: state.value_ref.clone(),
                head_idx: kv_head_idx,
                token_idx,
            }
        )
    }
}

fn read_attention_artifact_value_row(
    state: &RasterAttentionArtifactRowState,
    roots: &RasterArtifactStoreRoots,
    kv_head_idx: usize,
    token_idx: usize,
) -> Result<RasterActivationRow> {
    if let Some(cache_ref) = &state.donor_cache_ref {
        read_kv_row_from_roots(
            roots,
            RasterKvRowRequest {
                cache_ref: cache_ref.clone(),
                row_kind: RasterKvRowKind::Value,
                head_idx: kv_head_idx,
                token_idx,
            },
        )
    } else {
        read_head_row_from_roots(
            roots,
            RasterHeadRowRequest {
                tensor_ref: state.value_ref.clone(),
                head_idx: kv_head_idx,
                token_idx,
            },
        )
    }
}

fn read_attention_scalar_row(
    store: &AuthenticatedRasterTensorStore,
    tensor_ref: &RasterActivationSequenceRef,
    row_idx: usize,
    label: &str,
) -> Result<Act> {
    let row = auth_read!(
        store,
        RasterSequenceRowRequest {
            tensor_ref: tensor_ref.clone(),
            row_idx,
        }
    )?;
    if row.width() != 1 {
        bail!(
            "attention {label} row {row_idx} has width {}, expected 1",
            row.width()
        );
    }
    Ok(row.acts()[0])
}

fn read_attention_artifact_scalar_row(
    roots: &RasterArtifactStoreRoots,
    tensor_ref: &RasterActivationSequenceRef,
    row_idx: usize,
    label: &str,
) -> Result<Act> {
    let row = read_sequence_row_from_roots(
        roots,
        RasterSequenceRowRequest {
            tensor_ref: tensor_ref.clone(),
            row_idx,
        },
    )?;
    if row.width() != 1 {
        bail!(
            "attention {label} row {row_idx} has width {}, expected 1",
            row.width()
        );
    }
    Ok(row.acts()[0])
}

pub fn finalize_attention_row_state(
    state: RasterAttentionRowState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadSequence> {
    if !state.is_complete() {
        bail!(
            "attention row state finalized at head {} token {}, expected {} heads",
            state.next_query_head_idx,
            state.next_query_token_idx,
            state.query_head_count
        );
    }

    let output_ref = store.finalize_heads_builder(state.output_builder_ref)?;
    store.materialize_heads(&output_ref)
}

pub fn finalize_attention_row_state_ref(
    state: RasterAttentionRowState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadsRef> {
    if !state.is_complete() {
        bail!(
            "attention row state finalized at head {} token {}, expected {} heads",
            state.next_query_head_idx,
            state.next_query_token_idx,
            state.query_head_count
        );
    }

    store.finalize_heads_builder(state.output_builder_ref)
}

pub fn finalize_attention_artifact_row_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterAttentionArtifactRowState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    if !state.is_complete() {
        bail!(
            "attention row state finalized at head {} token {}, expected {} heads",
            state.next_query_head_idx,
            state.next_query_token_idx,
            state.query_head_count
        );
    }

    let output_id = RasterTensorId::new(state.output_source_name.clone())?;
    finalize_heads_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        output_id,
        state.query_head_count,
        state.sequence_len,
        state.head_dim,
    )
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
    store: &mut AuthenticatedRasterTensorStore,
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
    let input_ref =
        store.insert_activation_sequence(RasterTensorId::new("reshape.input")?, input.clone())?;
    let output_builder_ref = store.start_heads_builder(
        RasterTensorId::new("reshape.output")?,
        num_heads,
        input.len(),
        head_dim,
    )?;

    Ok(RasterReshapeHeadsState {
        input_ref,
        output_builder_ref,
        num_heads,
        head_dim,
        next_row_idx: 0,
        row_count: input.len(),
    })
}

pub fn init_reshape_heads_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
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
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    if width != expected_width {
        bail!(
            "attention reshape input row width {width} does not match heads {num_heads} * head_dim {head_dim}"
        );
    }
    let output_builder_ref =
        store.start_heads_builder(output_id, num_heads, row_count, head_dim)?;
    Ok(RasterReshapeHeadsState {
        input_ref,
        output_builder_ref,
        num_heads,
        head_dim,
        next_row_idx: 0,
        row_count,
    })
}

pub fn compute_next_reshape_heads_row(
    mut state: RasterReshapeHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterReshapeHeadsState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let row = auth_read!(
        store,
        RasterSequenceRowRequest {
            tensor_ref: state.input_ref.clone(),
            row_idx: state.next_row_idx,
        }
    )?;
    let acts = row.acts();
    for head_idx in 0..state.num_heads {
        let start = head_idx * state.head_dim;
        store.append_head_row(
            &mut state.output_builder_ref,
            head_idx,
            state.next_row_idx,
            RasterActivationRow::from_acts(acts[start..start + state.head_dim].to_vec()),
        )?;
    }
    state.next_row_idx += 1;
    Ok((false, state))
}

pub fn finalize_reshape_heads_state(
    state: RasterReshapeHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadSequence> {
    if !state.is_complete() {
        bail!(
            "attention reshape completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }
    let output_ref = store.finalize_heads_builder(state.output_builder_ref)?;
    store.materialize_heads(&output_ref)
}

pub fn finalize_reshape_heads_state_ref(
    state: RasterReshapeHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadsRef> {
    if !state.is_complete() {
        bail!(
            "attention reshape completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }
    store.finalize_heads_builder(state.output_builder_ref)
}

pub fn init_reshape_heads_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    num_heads: usize,
    head_dim: usize,
) -> Result<(RasterArtifactStoreRoots, RasterReshapeHeadsArtifactState)> {
    if num_heads == 0 {
        bail!("attention reshape requires at least one head");
    }
    if head_dim == 0 {
        bail!("attention reshape requires non-zero head dimension");
    }
    let expected_width = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("attention reshape width overflowed"))?;
    let (row_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    if width != expected_width {
        bail!(
            "attention reshape input row width {width} does not match heads {num_heads} * head_dim {head_dim}"
        );
    }
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        num_heads * row_count,
        head_dim,
    )?;
    Ok((
        artifact_store_roots,
        RasterReshapeHeadsArtifactState {
            input_ref,
            output_source_name,
            num_heads,
            head_dim,
            next_head_idx: 0,
            next_row_idx: 0,
            row_count,
        },
    ))
}

pub fn compute_next_reshape_heads_artifact_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterReshapeHeadsArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterReshapeHeadsArtifactState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    if state.next_row_idx >= state.row_count {
        state.next_head_idx += 1;
        state.next_row_idx = 0;
        return Ok((false, artifact_store_roots, state));
    }

    let row = read_sequence_row_from_roots(
        &artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: state.input_ref.clone(),
            row_idx: state.next_row_idx,
        },
    )?;
    let acts = row.acts();
    let start = state.next_head_idx * state.head_dim;
    artifact_store_roots = append_head_row_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        state.next_head_idx,
        state.next_row_idx,
        state.row_count,
        RasterActivationRow::from_acts(acts[start..start + state.head_dim].to_vec()),
    )?;
    state.next_row_idx += 1;
    Ok((false, artifact_store_roots, state))
}

pub fn finalize_reshape_heads_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterReshapeHeadsArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    if !state.is_complete() {
        bail!(
            "attention reshape completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }
    let output_id = RasterTensorId::new(state.output_source_name.clone())?;
    finalize_heads_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        output_id,
        state.num_heads,
        state.row_count,
        state.head_dim,
    )
}

pub fn init_combine_heads_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
) -> Result<RasterCombineHeadsState> {
    let sequence_len = attention_sequence_len(heads)?;
    let head_dim = attention_head_width(heads)?;
    let head_count = heads.head_count();
    let heads_ref =
        store.insert_attention_heads(RasterTensorId::new("combine.input")?, heads.clone())?;
    let output_builder_ref = store.start_sequence_builder(
        RasterTensorId::new("combine.output")?,
        sequence_len,
        head_count * head_dim,
    )?;
    Ok(RasterCombineHeadsState {
        heads_ref,
        output_builder_ref,
        next_token_idx: 0,
        head_count,
        sequence_len,
        head_dim,
    })
}

pub fn init_combine_heads_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
) -> Result<RasterCombineHeadsState> {
    let (head_count, sequence_len, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    let output_builder_ref = store.start_sequence_builder(
        output_id,
        sequence_len,
        head_count
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow!("attention combine width overflowed"))?,
    )?;
    Ok(RasterCombineHeadsState {
        heads_ref,
        output_builder_ref,
        next_token_idx: 0,
        head_count,
        sequence_len,
        head_dim,
    })
}

pub fn compute_next_combine_heads_row(
    mut state: RasterCombineHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterCombineHeadsState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let mut row = Vec::new();
    for head_idx in 0..state.head_count {
        let head_row = auth_read!(
            store,
            RasterHeadRowRequest {
                tensor_ref: state.heads_ref.clone(),
                head_idx,
                token_idx: state.next_token_idx,
            }
        )?;
        row.extend(head_row.acts());
    }
    store.append_sequence_row(
        &mut state.output_builder_ref,
        state.next_token_idx,
        RasterActivationRow::from_acts(row),
    )?;
    state.next_token_idx += 1;
    Ok((false, state))
}

pub fn finalize_combine_heads_state(
    state: RasterCombineHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    if !state.is_complete() {
        bail!(
            "attention combine completed {} rows, expected {}",
            state.next_token_idx,
            state.sequence_len
        );
    }
    let output_ref = store.finalize_sequence_builder(state.output_builder_ref)?;
    store.materialize_sequence(&output_ref)
}

pub fn finalize_combine_heads_state_ref(
    state: RasterCombineHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    if !state.is_complete() {
        bail!(
            "attention combine completed {} rows, expected {}",
            state.next_token_idx,
            state.sequence_len
        );
    }
    store.finalize_sequence_builder(state.output_builder_ref)
}

pub fn init_combine_heads_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
) -> Result<(RasterArtifactStoreRoots, RasterCombineHeadsArtifactState)> {
    let (head_count, sequence_len, head_dim) = heads_ref.tensor_ref().shape().heads_metadata()?;
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        sequence_len,
        head_count
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow!("attention combine width overflowed"))?,
    )?;
    Ok((
        artifact_store_roots,
        RasterCombineHeadsArtifactState {
            heads_ref,
            output_source_name,
            next_token_idx: 0,
            head_count,
            sequence_len,
            head_dim,
        },
    ))
}

pub fn compute_next_combine_heads_artifact_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterCombineHeadsArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterCombineHeadsArtifactState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let mut row = Vec::new();
    for head_idx in 0..state.head_count {
        let head_row = read_head_row_from_roots(
            &artifact_store_roots,
            RasterHeadRowRequest {
                tensor_ref: state.heads_ref.clone(),
                head_idx,
                token_idx: state.next_token_idx,
            },
        )?;
        row.extend(head_row.acts());
    }
    artifact_store_roots = append_sequence_row_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        state.next_token_idx,
        RasterActivationRow::from_acts(row),
    )?;
    state.next_token_idx += 1;
    Ok((false, artifact_store_roots, state))
}

pub fn finalize_combine_heads_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterCombineHeadsArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if !state.is_complete() {
        bail!(
            "attention combine completed {} rows, expected {}",
            state.next_token_idx,
            state.sequence_len
        );
    }
    let output_id = RasterTensorId::new(state.output_source_name.clone())?;
    finalize_sequence_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        output_id,
    )
}

pub fn init_kv_cache_build_state(
    store: &mut AuthenticatedRasterTensorStore,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    sliding_window: Option<usize>,
) -> Result<RasterKvCacheBuildState> {
    validate_matching_attention_heads(keys, values, "key/value")?;
    let sequence_len = attention_sequence_len(keys)?;
    let head_count = keys.head_count();
    let head_dim = attention_head_width(keys)?;
    let retained_start = sliding_window.map_or(0, |window| sequence_len.saturating_sub(window));
    let key_ref =
        store.insert_attention_heads(RasterTensorId::new("kv.build.keys")?, keys.clone())?;
    let value_ref =
        store.insert_attention_heads(RasterTensorId::new("kv.build.values")?, values.clone())?;
    let output_builder_ref = store.start_kv_cache_builder(
        RasterTensorId::new("kv.build.output.keys")?,
        RasterTensorId::new("kv.build.output.values")?,
        head_count,
        sequence_len.saturating_sub(retained_start),
        head_dim,
    )?;
    Ok(RasterKvCacheBuildState {
        key_ref,
        value_ref,
        output_builder_ref,
        retained_start,
        next_head_idx: 0,
        next_token_idx: retained_start,
        head_count,
        sequence_len,
        head_dim,
    })
}

pub fn init_kv_cache_build_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    sliding_window: Option<usize>,
) -> Result<RasterKvCacheBuildState> {
    let (head_count, sequence_len, head_dim) = key_ref.tensor_ref().shape().heads_metadata()?;
    let (value_head_count, value_sequence_len, value_head_dim) =
        value_ref.tensor_ref().shape().heads_metadata()?;
    if head_count != value_head_count
        || sequence_len != value_sequence_len
        || head_dim != value_head_dim
    {
        bail!("key/value attention heads shape mismatch");
    }
    let retained_start = sliding_window.map_or(0, |window| sequence_len.saturating_sub(window));
    let output_builder_ref = store.start_kv_cache_builder(
        keys_id,
        values_id,
        head_count,
        sequence_len.saturating_sub(retained_start),
        head_dim,
    )?;
    Ok(RasterKvCacheBuildState {
        key_ref,
        value_ref,
        output_builder_ref,
        retained_start,
        next_head_idx: 0,
        next_token_idx: retained_start,
        head_count,
        sequence_len,
        head_dim,
    })
}

pub fn compute_next_kv_cache_row(
    mut state: RasterKvCacheBuildState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterKvCacheBuildState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    if state.next_token_idx >= state.sequence_len {
        state.next_head_idx += 1;
        state.next_token_idx = state.retained_start;
        return Ok((false, state));
    }

    let key_row = auth_read!(
        store,
        RasterHeadRowRequest {
            tensor_ref: state.key_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: state.next_token_idx,
        }
    )?;
    let value_row = auth_read!(
        store,
        RasterHeadRowRequest {
            tensor_ref: state.value_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: state.next_token_idx,
        }
    )?;
    let output_token_idx = state.next_token_idx - state.retained_start;
    store.append_kv_row(
        state.output_builder_ref.keys_mut(),
        RasterKvRowKind::Key,
        state.next_head_idx,
        output_token_idx,
        key_row,
    )?;
    store.append_kv_row(
        state.output_builder_ref.values_mut(),
        RasterKvRowKind::Value,
        state.next_head_idx,
        output_token_idx,
        value_row,
    )?;
    state.next_token_idx += 1;
    Ok((false, state))
}

pub fn finalize_kv_cache_build_state(
    state: RasterKvCacheBuildState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterKvCache> {
    if !state.is_complete() {
        bail!(
            "KV cache build finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.head_count
        );
    }
    let output_ref = store.finalize_kv_cache_builder(state.output_builder_ref)?;
    store.materialize_kv_cache(&output_ref)
}

pub fn finalize_kv_cache_build_state_ref(
    state: RasterKvCacheBuildState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterKvCacheRef> {
    if !state.is_complete() {
        bail!(
            "KV cache build finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.head_count
        );
    }
    store.finalize_kv_cache_builder(state.output_builder_ref)
}

pub fn init_kv_cache_build_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    sliding_window: Option<usize>,
) -> Result<(RasterArtifactStoreRoots, RasterKvCacheBuildArtifactState)> {
    let (head_count, sequence_len, head_dim) = key_ref.tensor_ref().shape().heads_metadata()?;
    let (value_head_count, value_sequence_len, value_head_dim) =
        value_ref.tensor_ref().shape().heads_metadata()?;
    if head_count != value_head_count
        || sequence_len != value_sequence_len
        || head_dim != value_head_dim
    {
        bail!("key/value attention heads shape mismatch");
    }
    let retained_start = sliding_window.map_or(0, |window| sequence_len.saturating_sub(window));
    let current_len = sequence_len.saturating_sub(retained_start);
    let keys_source_name = keys_id.source_name().to_string();
    let values_source_name = values_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&keys_source_name)?,
        head_count * current_len,
        head_dim,
    )?;
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&values_source_name)?,
        head_count * current_len,
        head_dim,
    )?;
    Ok((
        artifact_store_roots,
        RasterKvCacheBuildArtifactState {
            key_ref,
            value_ref,
            keys_source_name,
            values_source_name,
            retained_start,
            next_head_idx: 0,
            next_token_idx: retained_start,
            head_count,
            current_len,
            sequence_len,
            head_dim,
        },
    ))
}

pub fn compute_next_kv_cache_artifact_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: RasterKvCacheBuildArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterKvCacheBuildArtifactState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    if state.next_token_idx >= state.sequence_len {
        state.next_head_idx += 1;
        state.next_token_idx = state.retained_start;
        return Ok((false, artifact_store_roots, state));
    }

    let key_row = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: state.key_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: state.next_token_idx,
        },
    )?;
    let value_row = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: state.value_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: state.next_token_idx,
        },
    )?;
    let output_token_idx = state.next_token_idx - state.retained_start;
    artifact_store_roots = append_head_row_by_source_name_with_roots(
        &artifact_store_roots,
        &state.keys_source_name,
        state.next_head_idx,
        output_token_idx,
        state.current_len,
        key_row,
    )?;
    artifact_store_roots = append_head_row_by_source_name_with_roots(
        &artifact_store_roots,
        &state.values_source_name,
        state.next_head_idx,
        output_token_idx,
        state.current_len,
        value_row,
    )?;
    state.next_token_idx += 1;
    Ok((false, artifact_store_roots, state))
}

pub fn finalize_kv_cache_build_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterKvCacheBuildArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterKvCacheRef)> {
    if !state.is_complete() {
        bail!(
            "KV cache build finalized at head {} token {}, expected {} heads",
            state.next_head_idx,
            state.next_token_idx,
            state.head_count
        );
    }
    let keys_id = RasterTensorId::new(state.keys_source_name.clone())?;
    let values_id = RasterTensorId::new(state.values_source_name.clone())?;
    finalize_kv_cache_builders_by_source_name_with_roots(
        &artifact_store_roots,
        &state.keys_source_name,
        &state.values_source_name,
        keys_id,
        values_id,
        state.head_count,
        state.current_len,
        state.head_dim,
    )
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
        add_sequences, append_projection_chunk_to_state, apply_rope_to_heads, attention_output_row,
        build_raster_kv_cache, causal_attention_heads, causal_attention_heads_with_cache,
        combine_attention_heads, compute_next_attention_row, compute_next_combine_heads_row,
        compute_next_head_unary_row, compute_next_kv_cache_row, compute_next_reshape_heads_row,
        compute_next_sequence_binary_row, compute_next_sequence_unary_row,
        finalize_attention_row_state, finalize_attention_row_state_ref,
        finalize_combine_heads_state, finalize_head_unary_row_state, finalize_kv_cache_build_state,
        finalize_reshape_heads_state, finalize_sequence_binary_row_state,
        finalize_sequence_projection_state, finalize_sequence_unary_row_state, gelu_sequence,
        init_attention_row_state, init_combine_heads_state, init_head_rms_norm_row_state,
        init_kv_cache_build_state, init_reshape_heads_state, init_rope_row_state,
        init_sequence_add_row_state, init_sequence_gelu_row_state, init_sequence_mul_row_state,
        init_sequence_projection_state, init_sequence_rms_norm_row_state,
        init_sequence_scale_row_state, init_value_rms_norm_row_state, mul_sequences,
        project_sequence, project_sequence_with_prefill_source,
        project_sequence_with_prefill_source_chunked, project_sequence_with_source,
        project_sequence_with_source_chunked, reshape_sequence_heads, rms_norm_heads,
        rms_norm_sequence, scale_sequence, value_rms_norm_heads, RasterActivationRow,
        RasterActivationSequence, RasterAttentionHeadSequence, RasterKvCache,
    };
    use crate::shared::artifacts::artifact_io::AuthRead;
    use crate::shared::numerics::det_num::{
        add_sat, attention_score, attention_softmax, attention_weighted_sum, gelu_pytorch_tanh_act,
        mul_sat, rms_norm, rope_rotate_pairs, scale_act, value_rms_norm, Acc, Act, Wgt,
    };
    use crate::shared::raster_contracts::prefill_layer::{
        GemmaPrefillLayerMatrixKind, GemmaPrefillLayerMatrixRowRequest,
    };
    use crate::shared::raster_contracts::prefill_ple::GemmaPleModelProjectionRowRequest;
    use crate::shared::tensors::raster_row_store::AuthenticatedRasterTensorStore;
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
    fn projection_can_read_rows_from_raster_auth_source() {
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
    fn projection_state_serializes_refs_and_builder_not_materialized_rows() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let state =
            init_sequence_projection_state(&mut store, &input, 3, 2).expect("projection state");

        let encoded = serde_json::to_string(&state).expect("serialize projection state");

        assert!(encoded.contains("input_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(encoded.contains("next_token_idx"));
        assert!(encoded.contains("next_projection_row_idx"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_act_bits"));
    }

    #[test]
    fn projection_state_computes_one_token_row_by_projection_chunk() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);
        let projection_rows = vec![
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            vec![Wgt::from_num(2.0), Wgt::from_num(-2.0)],
        ];
        let mut store = AuthenticatedRasterTensorStore::new();
        let mut state =
            init_sequence_projection_state(&mut store, &input, projection_rows.len(), 2)
                .expect("projection state");

        append_projection_chunk_to_state(&mut state, &mut store, &projection_rows[..2])
            .expect("first chunk");
        assert_eq!(state.next_token_idx(), 0);
        assert_eq!(state.next_projection_row_idx(), 2);
        append_projection_chunk_to_state(&mut state, &mut store, &projection_rows[2..])
            .expect("finish first token");
        assert_eq!(state.next_token_idx(), 1);
        assert_eq!(state.next_projection_row_idx(), 0);
        append_projection_chunk_to_state(&mut state, &mut store, &projection_rows[..2])
            .expect("second token first chunk");
        append_projection_chunk_to_state(&mut state, &mut store, &projection_rows[2..])
            .expect("second token final chunk");
        assert!(state.is_complete());

        let chunked = finalize_sequence_projection_state(state, &mut store).expect("finalize");
        let full = project_sequence(&input, &projection_rows).expect("full projection");

        assert_eq!(scaled_bits(&chunked), scaled_bits(&full));
    }

    #[test]
    fn projection_state_fails_closed_for_invalid_init_and_chunks() {
        let input =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);
        let empty = RasterActivationSequence::from_acts(Vec::new());
        let zero_width = RasterActivationSequence::from_act_bits(vec![Vec::new()]);

        assert!(init_sequence_projection_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &input,
            0,
            1
        )
        .is_err());
        assert!(init_sequence_projection_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &input,
            1,
            0
        )
        .is_err());
        assert!(init_sequence_projection_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &empty,
            1,
            1
        )
        .is_err());
        assert!(init_sequence_projection_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &zero_width,
            1,
            1
        )
        .is_err());

        let mut store = AuthenticatedRasterTensorStore::new();
        let mut state = init_sequence_projection_state(&mut store, &input, 2, 1).expect("state");
        assert!(append_projection_chunk_to_state(&mut state, &mut store, &[]).is_err());
        assert!(append_projection_chunk_to_state(
            &mut state,
            &mut store,
            &[
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
                vec![Wgt::from_num(0.0), Wgt::from_num(1.0)]
            ]
        )
        .is_err());
        append_projection_chunk_to_state(
            &mut state,
            &mut store,
            &[vec![Wgt::from_num(1.0), Wgt::from_num(0.0)]],
        )
        .expect("first row");
        assert!(finalize_sequence_projection_state(state.clone(), &mut store).is_err());
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
    fn prefill_projection_can_read_rows_from_raster_auth_source() {
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

        let mut store = AuthenticatedRasterTensorStore::new();
        let rms = run_sequence_unary_state(
            init_sequence_rms_norm_row_state(&mut store, &input, Some(&weights), Some(eps), 2)
                .expect("init"),
            &mut store,
        )
        .expect("rms");
        assert_eq!(
            scaled_bits(&rms),
            scaled_bits(&rms_norm_sequence(&input, Some(&weights), Some(eps)).expect("full"))
        );

        let mut store = AuthenticatedRasterTensorStore::new();
        let gelu = run_sequence_unary_state(
            init_sequence_gelu_row_state(&mut store, &input, 2).expect("init"),
            &mut store,
        )
        .expect("gelu");
        assert_eq!(
            scaled_bits(&gelu),
            scaled_bits(&gelu_sequence(&input).expect("full"))
        );

        let mut store = AuthenticatedRasterTensorStore::new();
        let scale = run_sequence_unary_state(
            init_sequence_scale_row_state(&mut store, &input, Some(Act::from_num(0.5)), 2)
                .expect("init"),
            &mut store,
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

        let error = init_sequence_rms_norm_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &input,
            None,
            Some(Acc::from_num(0.0)),
            1,
        )
        .expect_err("missing weights should fail");
        assert!(error.to_string().contains("canonical norm weights"));

        let error = init_sequence_rms_norm_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &input,
            Some(&[Wgt::from_num(1.0)]),
            None,
            1,
        )
        .expect_err("missing epsilon should fail");
        assert!(error.to_string().contains("canonical Acc epsilon"));

        let error = init_sequence_rms_norm_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &zero_width,
            Some(&[]),
            Some(Acc::from_num(0.0)),
            1,
        )
        .expect_err("zero-width RMSNorm should fail");
        assert!(error.to_string().contains("non-zero width"));

        let error =
            init_sequence_gelu_row_state(&mut AuthenticatedRasterTensorStore::new(), &empty, 1)
                .expect_err("empty GELU should fail");
        assert!(error
            .to_string()
            .contains("requires at least one activation row"));

        let error = init_sequence_scale_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &input,
            None,
            1,
        )
        .expect_err("missing scalar should fail");
        assert!(error.to_string().contains("canonical Act scalar"));
    }

    #[test]
    fn sequence_unary_state_serializes_refs_not_materialized_rows() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
            vec![Act::from_num(0.25), Act::from_num(0.75)],
        ]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let state = init_sequence_gelu_row_state(&mut store, &input, 1).expect("state");

        let encoded = serde_json::to_string(&state).expect("serialize state");

        assert!(encoded.contains("input_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_rows"));
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

        let mut store = AuthenticatedRasterTensorStore::new();
        let added = run_sequence_binary_state(
            init_sequence_add_row_state(&mut store, &lhs, &rhs, 2).expect("init add"),
            &mut store,
        )
        .expect("add");
        assert_eq!(
            scaled_bits(&added),
            scaled_bits(&add_sequences(&lhs, &rhs).expect("full"))
        );

        let mut store = AuthenticatedRasterTensorStore::new();
        let multiplied = run_sequence_binary_state(
            init_sequence_mul_row_state(&mut store, &lhs, &rhs, 2).expect("init mul"),
            &mut store,
        )
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

        let error = init_sequence_add_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &lhs,
            &wrong_width,
            1,
        )
        .expect_err("width mismatch should fail");
        assert!(error.to_string().contains("right sequence row 0 has width"));

        let error = init_sequence_mul_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &lhs,
            &wrong_len,
            1,
        )
        .expect_err("length mismatch should fail");
        assert!(error.to_string().contains("sequence length mismatch"));
    }

    #[test]
    fn sequence_binary_state_serializes_refs_not_materialized_rows() {
        let lhs = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);
        let rhs = RasterActivationSequence::from_acts(vec![vec![Act::from_num(0.5)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let state = init_sequence_add_row_state(&mut store, &lhs, &rhs, 1).expect("state");

        let encoded = serde_json::to_string(&state).expect("serialize state");

        assert!(encoded.contains("lhs_ref"));
        assert!(encoded.contains("rhs_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_rows"));
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

        let mut store = AuthenticatedRasterTensorStore::new();
        let head_rms = run_head_unary_state(
            init_head_rms_norm_row_state(&mut store, &heads, Some(&weights), Some(eps), 2)
                .expect("init"),
            &mut store,
        )
        .expect("head rms");
        assert_eq!(
            head_bits(&head_rms),
            head_bits(&rms_norm_heads(&heads, Some(&weights), Some(eps)).expect("full"))
        );

        let mut store = AuthenticatedRasterTensorStore::new();
        let value_rms = run_head_unary_state(
            init_value_rms_norm_row_state(&mut store, &heads, Some(eps), 2).expect("init"),
            &mut store,
        )
        .expect("value rms");
        assert_eq!(
            head_bits(&value_rms),
            head_bits(&value_rms_norm_heads(&heads, Some(eps)).expect("full"))
        );

        let mut store = AuthenticatedRasterTensorStore::new();
        let rope = run_head_unary_state(
            init_rope_row_state(
                &mut store,
                &heads,
                2,
                2,
                Some(Acc::from_num(10_000.0)),
                3,
                2,
            )
            .expect("init rope"),
            &mut store,
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

        let error = init_head_rms_norm_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &heads,
            None,
            Some(Acc::from_num(0.0)),
            1,
        )
        .expect_err("missing head weights should fail");
        assert!(error.to_string().contains("canonical norm weights"));

        let error = init_value_rms_norm_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &heads,
            None,
            1,
        )
        .expect_err("missing value norm epsilon should fail");
        assert!(error.to_string().contains("canonical Acc epsilon"));

        let error = init_rope_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &heads,
            2,
            2,
            Some(Acc::from_num(10_000.0)),
            0,
            1,
        )
        .expect_err("rotary dim should fail");
        assert!(error.to_string().contains("exceeds attention head width"));

        let error = init_rope_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &heads,
            1,
            2,
            None,
            0,
            1,
        )
        .expect_err("missing rope base should fail");
        assert!(error.to_string().contains("canonical Acc base"));
    }

    #[test]
    fn head_unary_state_serializes_refs_not_materialized_rows() {
        let heads = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(2.0)],
        ]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let state =
            init_value_rms_norm_row_state(&mut store, &heads, Some(Acc::from_num(0.001)), 1)
                .expect("state");

        let encoded = serde_json::to_string(&state).expect("serialize state");

        assert!(encoded.contains("heads_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_heads"));
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
    fn attention_row_state_chunk_size_does_not_change_output() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
            vec![Act::from_num(1.0), Act::from_num(1.0)],
        ]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(2.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(4.0)],
            vec![Act::from_num(6.0), Act::from_num(8.0)],
        ]]);
        let full_output = causal_attention_heads_with_cache(&queries, &keys, &values, None, None)
            .expect("full attention");

        for kv_rows_per_tile in [1, 2, usize::MAX] {
            let row_output = run_attention_row_state_with_kv_rows_per_tile(
                &queries,
                &keys,
                &values,
                None,
                None,
                kv_rows_per_tile,
            )
            .expect("row attention");
            assert_eq!(head_bits(&row_output), head_bits(&full_output));
        }
    }

    #[test]
    fn attention_row_state_preserves_softmax_ties_across_chunks() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let keys = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(-1.0)],
            vec![Act::from_num(2.0)],
            vec![Act::from_num(2.0)],
        ]]);
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(2.0)],
            vec![Act::from_num(4.0)],
        ]]);
        let full_output = causal_attention_heads_with_cache(&queries, &keys, &values, None, None)
            .expect("full attention");

        let row_output =
            run_attention_row_state_with_kv_rows_per_tile(&queries, &keys, &values, None, None, 1)
                .expect("row attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn attention_row_state_does_not_materialize_score_or_weight_sequences() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(2.0)],
            vec![Act::from_num(4.0)],
            vec![Act::from_num(8.0)],
        ]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let mut state =
            init_attention_row_state(&mut store, &queries, &keys, &values, None, None, 1)
                .expect("attention state");

        while !state.is_complete() {
            let (_done, next_state) =
                compute_next_attention_row(state, &mut store).expect("bounded attention step");
            let encoded = serde_json::to_string(&next_state).expect("state should serialize");
            assert!(encoded.contains("query_ref"));
            assert!(!encoded.contains("act_bits"));
            state = next_state;
        }

        assert_eq!(store.materialization_counts(), (0, 0));
        let _output_ref =
            finalize_attention_row_state_ref(state, &mut store).expect("finalize ref");
        assert_eq!(store.materialization_counts(), (0, 0));
    }

    #[test]
    fn attention_row_state_rejects_zero_kv_rows_per_tile() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let keys = queries.clone();
        let values = queries.clone();
        let error = init_attention_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &queries,
            &keys,
            &values,
            None,
            None,
            0,
        )
        .expect_err("zero kv rows per tile should fail");

        assert!(error
            .to_string()
            .contains("raster attention KV rows per tile must be greater than zero"));
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

        let full_output =
            causal_attention_heads_with_cache(&queries, &keys, &values, None, Some(2))
                .expect("full attention");

        for kv_rows_per_tile in [1, 2, usize::MAX] {
            let row_output = run_attention_row_state_with_kv_rows_per_tile(
                &queries,
                &keys,
                &values,
                None,
                Some(2),
                kv_rows_per_tile,
            )
            .expect("row attention");
            assert_eq!(head_bits(&row_output), head_bits(&full_output));
        }
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

        let full_output = causal_attention_heads_with_cache(
            &queries,
            &current_keys,
            &current_values,
            Some(&donor_cache),
            Some(1),
        )
        .expect("full attention");

        for kv_rows_per_tile in [1, usize::MAX] {
            let row_output = run_attention_row_state_with_kv_rows_per_tile(
                &queries,
                &current_keys,
                &current_values,
                Some(&donor_cache),
                Some(1),
                kv_rows_per_tile,
            )
            .expect("row attention");
            assert_eq!(head_bits(&row_output), head_bits(&full_output));
        }
    }

    #[test]
    fn attention_row_state_serializes_refs_not_materialized_rows() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let keys = queries.clone();
        let values = queries.clone();
        let mut store = AuthenticatedRasterTensorStore::new();
        let state =
            init_attention_row_state(&mut store, &queries, &keys, &values, None, None, usize::MAX)
                .expect("attention state");

        let encoded = serde_json::to_string(&state).expect("state should serialize");

        assert!(encoded.contains("query_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_heads"));
    }

    #[test]
    fn attention_row_state_fails_closed_for_invalid_shapes() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let mismatched_width_keys = RasterAttentionHeadSequence::from_acts(vec![vec![vec![
            Act::from_num(1.0),
            Act::from_num(2.0),
        ]]]);
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let error = init_attention_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &queries,
            &mismatched_width_keys,
            &values,
            None,
            None,
            usize::MAX,
        )
        .expect_err("width mismatch should fail");
        assert!(error.to_string().contains("head width mismatch"));

        let mismatched_len_keys = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let error = init_attention_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &queries,
            &mismatched_len_keys,
            &values,
            None,
            None,
            usize::MAX,
        )
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
        let error = init_attention_row_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &grouped_queries,
            &grouped_keys,
            &grouped_values,
            None,
            None,
            usize::MAX,
        )
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
            &mut AuthenticatedRasterTensorStore::new(),
            &grouped_keys,
            &grouped_keys,
            &grouped_values,
            Some(&donor_cache),
            None,
            usize::MAX,
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

        let mut store = AuthenticatedRasterTensorStore::new();
        let reshaped = run_reshape_heads_state(
            init_reshape_heads_state(&mut store, &sequence, 2, 2).expect("init"),
            &mut store,
        )
        .expect("reshape");
        let full_reshaped = reshape_sequence_heads(&sequence, 2, 2).expect("full reshape");
        assert_eq!(head_bits(&reshaped), head_bits(&full_reshaped));

        let mut store = AuthenticatedRasterTensorStore::new();
        let combined = run_combine_heads_state(
            init_combine_heads_state(&mut store, &reshaped).expect("init"),
            &mut store,
        )
        .expect("combine");
        assert_eq!(
            scaled_bits(&combined),
            scaled_bits(&combine_attention_heads(&reshaped).expect("full combine"))
        );

        let mut store = AuthenticatedRasterTensorStore::new();
        let cache = run_kv_cache_build_state(
            init_kv_cache_build_state(&mut store, &reshaped, &full_reshaped, None)
                .expect("init cache"),
            &mut store,
        )
        .expect("cache");
        let full_cache =
            build_raster_kv_cache(&reshaped, &full_reshaped, None).expect("full cache");
        assert_eq!(cache, full_cache);

        let mut store = AuthenticatedRasterTensorStore::new();
        let sliding_cache = run_kv_cache_build_state(
            init_kv_cache_build_state(&mut store, &reshaped, &full_reshaped, Some(1))
                .expect("init cache"),
            &mut store,
        )
        .expect("sliding cache");
        let full_sliding_cache =
            build_raster_kv_cache(&reshaped, &full_reshaped, Some(1)).expect("full cache");
        assert_eq!(sliding_cache, full_sliding_cache);
    }

    #[test]
    fn chunked_layout_helpers_fail_closed() {
        let sequence = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);
        let error =
            init_reshape_heads_state(&mut AuthenticatedRasterTensorStore::new(), &sequence, 0, 1)
                .expect_err("zero heads should fail");
        assert!(error.to_string().contains("at least one head"));

        let error =
            init_reshape_heads_state(&mut AuthenticatedRasterTensorStore::new(), &sequence, 1, 0)
                .expect_err("zero dim should fail");
        assert!(error.to_string().contains("non-zero head dimension"));

        let ragged_heads = RasterAttentionHeadSequence::from_heads(vec![
            vec![RasterActivationRow::from_acts(vec![Act::from_num(1.0)])],
            vec![
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
            ],
        ]);
        let error =
            init_combine_heads_state(&mut AuthenticatedRasterTensorStore::new(), &ragged_heads)
                .expect_err("ragged heads should fail");
        assert!(error.to_string().contains("attention head 1 has 2 rows"));

        let keys = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let values = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
        ]);
        let error = init_kv_cache_build_state(
            &mut AuthenticatedRasterTensorStore::new(),
            &keys,
            &values,
            None,
        )
        .expect_err("cache head mismatch should fail");
        assert!(error.to_string().contains("head count mismatch"));
    }

    #[test]
    fn layout_and_kv_states_serialize_refs_not_materialized_rows() {
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
        let heads = reshape_sequence_heads(&sequence, 2, 2).expect("heads");

        let mut store = AuthenticatedRasterTensorStore::new();
        let reshape_state =
            init_reshape_heads_state(&mut store, &sequence, 2, 2).expect("reshape state");
        let encoded = serde_json::to_string(&reshape_state).expect("serialize reshape state");
        assert!(encoded.contains("input_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_heads"));

        let mut store = AuthenticatedRasterTensorStore::new();
        let combine_state = init_combine_heads_state(&mut store, &heads).expect("combine state");
        let encoded = serde_json::to_string(&combine_state).expect("serialize combine state");
        assert!(encoded.contains("heads_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_rows"));

        let mut store = AuthenticatedRasterTensorStore::new();
        let kv_state =
            init_kv_cache_build_state(&mut store, &heads, &heads, Some(1)).expect("kv state");
        let encoded = serde_json::to_string(&kv_state).expect("serialize kv state");
        assert!(encoded.contains("key_ref"));
        assert!(encoded.contains("value_ref"));
        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("output_keys"));
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
        store: &mut AuthenticatedRasterTensorStore,
    ) -> Result<RasterActivationSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_sequence_unary_row(state, store)?;
            state = next_state;
        }
        finalize_sequence_unary_row_state(state, store)
    }

    fn run_sequence_binary_state(
        mut state: super::RasterSequenceBinaryState,
        store: &mut AuthenticatedRasterTensorStore,
    ) -> Result<RasterActivationSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_sequence_binary_row(state, store)?;
            state = next_state;
        }
        finalize_sequence_binary_row_state(state, store)
    }

    fn run_head_unary_state(
        mut state: super::RasterHeadUnaryState,
        store: &mut AuthenticatedRasterTensorStore,
    ) -> Result<RasterAttentionHeadSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_head_unary_row(state, store)?;
            state = next_state;
        }
        finalize_head_unary_row_state(state, store)
    }

    fn run_reshape_heads_state(
        mut state: super::RasterReshapeHeadsState,
        store: &mut AuthenticatedRasterTensorStore,
    ) -> Result<RasterAttentionHeadSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_reshape_heads_row(state, store)?;
            state = next_state;
        }
        finalize_reshape_heads_state(state, store)
    }

    fn run_combine_heads_state(
        mut state: super::RasterCombineHeadsState,
        store: &mut AuthenticatedRasterTensorStore,
    ) -> Result<RasterActivationSequence> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_combine_heads_row(state, store)?;
            state = next_state;
        }
        finalize_combine_heads_state(state, store)
    }

    fn run_kv_cache_build_state(
        mut state: super::RasterKvCacheBuildState,
        store: &mut AuthenticatedRasterTensorStore,
    ) -> Result<RasterKvCache> {
        while !state.is_complete() {
            let (_done, next_state) = compute_next_kv_cache_row(state, store)?;
            state = next_state;
        }
        finalize_kv_cache_build_state(state, store)
    }

    fn run_attention_row_state(
        queries: &RasterAttentionHeadSequence,
        keys: &RasterAttentionHeadSequence,
        values: &RasterAttentionHeadSequence,
        donor_cache: Option<&RasterKvCache>,
        attention_window: Option<usize>,
    ) -> Result<RasterAttentionHeadSequence> {
        run_attention_row_state_with_kv_rows_per_tile(
            queries,
            keys,
            values,
            donor_cache,
            attention_window,
            usize::MAX,
        )
    }

    fn run_attention_row_state_with_kv_rows_per_tile(
        queries: &RasterAttentionHeadSequence,
        keys: &RasterAttentionHeadSequence,
        values: &RasterAttentionHeadSequence,
        donor_cache: Option<&RasterKvCache>,
        attention_window: Option<usize>,
        kv_rows_per_tile: usize,
    ) -> Result<RasterAttentionHeadSequence> {
        let mut store = AuthenticatedRasterTensorStore::new();
        let mut state = init_attention_row_state(
            &mut store,
            queries,
            keys,
            values,
            donor_cache,
            attention_window,
            kv_rows_per_tile,
        )?;
        while !state.is_complete() {
            let (_done, next_state) = compute_next_attention_row(state, &mut store)?;
            state = next_state;
        }
        finalize_attention_row_state(state, &mut store)
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
