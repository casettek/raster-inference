use anyhow::{anyhow, bail, Result};

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

pub fn project_sequence_with_source<S>(
    input: &RasterActivationSequence,
    source: &S,
    layer_idx: usize,
    projection_rows: usize,
) -> Result<RasterActivationSequence>
where
    S: AuthRead<GemmaPleModelProjectionRowRequest, Output = Vec<Wgt>>,
{
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }

    let mut rows = Vec::with_capacity(projection_rows);
    for row_idx in 0..projection_rows {
        rows.push(auth_read!(
            source,
            GemmaPleModelProjectionRowRequest { layer_idx, row_idx }
        )?);
    }
    project_sequence(input, &rows)
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
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }

    let mut rows = Vec::with_capacity(projection_rows);
    for row_idx in 0..projection_rows {
        rows.push(auth_read!(
            source,
            GemmaPrefillLayerMatrixRowRequest {
                layer_idx,
                matrix,
                row_idx,
            }
        )?);
    }
    project_sequence(input, &rows)
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

fn project_row(
    input: &RasterActivationRow,
    projection_rows: &[Vec<Wgt>],
) -> Result<RasterActivationRow> {
    let input_acts = input.acts();
    let width = projection_rows[0].len();
    if input_acts.len() != width {
        bail!(
            "deterministic linear input width mismatch: {} vs {}",
            input_acts.len(),
            width
        );
    }

    let mut output = Vec::with_capacity(projection_rows.len());
    for row in projection_rows {
        let mut acc_bits = 0_i64;
        for (act, weight) in input_acts.iter().zip(row) {
            acc_bits = mac_bits(acc_bits, act.to_bits(), weight.to_bits());
        }
        output.push(requantize(Acc::from_bits(acc_bits)));
    }
    Ok(RasterActivationRow::from_acts(output))
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
        gelu_sequence, mul_sequences, project_sequence, project_sequence_with_prefill_source,
        project_sequence_with_source, reshape_sequence_heads, rms_norm_sequence, scale_sequence,
        value_rms_norm_heads, RasterActivationRow, RasterActivationSequence,
        RasterAttentionHeadSequence, RasterKvCache,
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
