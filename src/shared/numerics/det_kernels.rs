//! Deterministic-only transformer kernels over flat slab buffers.
//!
//! These functions replace the deterministic branches of the dual-track
//! kernels in `transformer_kernels.rs` for the native deterministic path.
//! They operate exclusively on canonical `Act`/`Acc` values — no `f32`
//! arithmetic or storage anywhere — and use borrowed window slices plus
//! reusable scratch buffers so the per-(head, query) attention loop and the
//! per-layer decode loop perform no heap allocation.
//!
//! Canonical arithmetic, operation order, and error messages mirror the
//! legacy deterministic branches exactly; bit-identity is enforced by the
//! `det_commitment_goldens` regression gate.

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;

use crate::shared::model::transformer::{
    DetNumMatrix, Gemma4LogitsProjection, GemmaEmbeddingTensorSource, InternalActivationRow,
    InternalActivationSequence, LayerKvCache, ResolvedGemma4LayerWeights,
};
use crate::shared::numerics::det_num::{
    add_sat, attention_score as det_attention_score, attention_softmax_into,
    attention_weighted_sum_flat_into, gelu_pytorch_tanh_act, mac_bits, mul_sat, requantize,
    rms_norm_in_place, rope_rotate_pairs_in_place, scale_act, softcap_act,
    value_rms_norm_in_place, Acc, Act, Wgt,
};
use crate::shared::numerics::det_tensor::{ActSlab, DetKvCacheData, HeadSlab};

// ---------------------------------------------------------------------------
// Boundary conversions
// ---------------------------------------------------------------------------

pub(crate) fn slab_from_internal_sequence(internal: &InternalActivationSequence) -> Result<ActSlab> {
    let rows = internal
        .det_values()
        .ok_or_else(|| anyhow!("deterministic sequence operation requires canonical Act rows"))?;
    ActSlab::from_rows(rows)
}

pub(crate) fn internal_sequence_from_slab(slab: &ActSlab) -> InternalActivationSequence {
    InternalActivationSequence::from_det_values_only(slab.to_nested())
}

pub(crate) fn row_from_internal(internal: &InternalActivationRow) -> Result<Vec<Act>> {
    internal
        .det_values()
        .map(<[Act]>::to_vec)
        .ok_or_else(|| anyhow!("deterministic row operation requires canonical Act values"))
}

// ---------------------------------------------------------------------------
// Elementwise / projection primitives
// ---------------------------------------------------------------------------

fn det_linear_into(input: &[Act], weight: &DetNumMatrix, output: &mut [Act]) -> Result<()> {
    if input.len() != weight.cols {
        bail!(
            "deterministic linear input width mismatch: {} vs {}",
            input.len(),
            weight.cols
        );
    }
    debug_assert_eq!(output.len(), weight.rows);

    let weight_values = weight.values.as_slice();
    for (row_idx, out) in output.iter_mut().enumerate() {
        let row_offset = row_idx * weight.cols;
        let mut acc_bits = 0_i64;
        for (col_idx, act) in input.iter().enumerate() {
            acc_bits = mac_bits(acc_bits, act.to_bits(), weight_values[row_offset + col_idx]);
        }
        *out = requantize(Acc::from_bits(acc_bits));
    }
    Ok(())
}

fn det_linear_slab(input: &ActSlab, weight: &DetNumMatrix) -> Result<ActSlab> {
    if input.cols() != weight.cols {
        bail!(
            "deterministic linear input row 0 has width {}, expected {}",
            input.cols(),
            weight.cols
        );
    }
    let mut output = ActSlab::zeroed(input.rows(), weight.rows);
    output
        .as_flat_mut()
        .par_chunks_mut(weight.rows)
        .zip(input.as_flat().par_chunks(input.cols().max(1)))
        .try_for_each(|(out_row, in_row)| det_linear_into(in_row, weight, out_row))?;
    Ok(output)
}

fn require_det_weight<'a>(
    weight: Option<&'a DetNumMatrix>,
    error: &'static str,
) -> Result<&'a DetNumMatrix> {
    weight.ok_or_else(|| anyhow!(error))
}

fn rms_norm_weights<'a>(
    weight_det: Option<&'a [Wgt]>,
    eps_det: Option<Acc>,
) -> Result<(&'a [Wgt], Acc)> {
    let weight = weight_det
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Wgt norm weights"))?;
    let eps =
        eps_det.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    Ok((weight, eps))
}

fn rms_norm_row_checked(row: &mut [Act], weight: &[Wgt], eps: Acc) -> Result<()> {
    if row.len() != weight.len() {
        bail!("rms norm width mismatch: {} vs {}", row.len(), weight.len());
    }
    rms_norm_in_place(row, weight, eps);
    Ok(())
}

fn rms_norm_slab(
    input: &ActSlab,
    weight_det: Option<&[Wgt]>,
    eps_det: Option<Acc>,
) -> Result<ActSlab> {
    let (weight, eps) = rms_norm_weights(weight_det, eps_det)?;
    let mut output = input.clone();
    output
        .as_flat_mut()
        .par_chunks_mut(input.cols().max(1))
        .try_for_each(|row| rms_norm_row_checked(row, weight, eps))?;
    Ok(output)
}

fn add_in_place(dst: &mut [Act], rhs: &[Act]) -> Result<()> {
    if dst.len() != rhs.len() {
        bail!("row width mismatch: {} vs {}", dst.len(), rhs.len());
    }
    for (lhs_value, rhs_value) in dst.iter_mut().zip(rhs) {
        *lhs_value = add_sat(*lhs_value, *rhs_value);
    }
    Ok(())
}

fn mul_in_place(dst: &mut [Act], rhs: &[Act]) -> Result<()> {
    if dst.len() != rhs.len() {
        bail!("row width mismatch: {} vs {}", dst.len(), rhs.len());
    }
    for (lhs_value, rhs_value) in dst.iter_mut().zip(rhs) {
        *lhs_value = mul_sat(*lhs_value, *rhs_value);
    }
    Ok(())
}

fn scale_in_place(values: &mut [Act], scalar: Act) {
    for value in values.iter_mut() {
        *value = scale_act(*value, scalar);
    }
}

fn gelu_in_place(values: &mut [Act]) {
    for value in values.iter_mut() {
        *value = gelu_pytorch_tanh_act(*value);
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct AttentionWindows<'a> {
    keys: &'a [Act],
    values: &'a [Act],
    rows: usize,
}

/// Resolves the borrowed key/value window for one (kv head, query) pair,
/// clamping to the available rows exactly like the legacy `skip(start)
/// .take(len)` iteration over donor caches.
fn cache_windows<'a>(
    det: &'a DetKvCacheData,
    kv_head_idx: usize,
    start: usize,
    len: usize,
) -> AttentionWindows<'a> {
    let available = det.len();
    let start = start.min(available);
    let rows = len.min(available - start);
    AttentionWindows {
        keys: det.key_window(kv_head_idx, start, rows),
        values: det.value_window(kv_head_idx, start, rows),
        rows,
    }
}

fn attention_output_into(
    query: &[Act],
    windows: &AttentionWindows<'_>,
    head_dim: usize,
    logits_scratch: &mut Vec<Act>,
    exp_scratch: &mut Vec<Acc>,
    weights_scratch: &mut Vec<Act>,
    output: &mut [Act],
) {
    logits_scratch.clear();
    for row_idx in 0..windows.rows {
        let key_row = &windows.keys[row_idx * head_dim..(row_idx + 1) * head_dim];
        logits_scratch.push(det_attention_score(query, key_row));
    }
    attention_softmax_into(logits_scratch, exp_scratch, weights_scratch);
    attention_weighted_sum_flat_into(weights_scratch, windows.values, head_dim, output);
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

fn reshape_to_heads(projected: &ActSlab, num_heads: usize, head_dim: usize) -> Result<HeadSlab> {
    let expected_width = num_heads * head_dim;
    if projected.cols() != expected_width {
        bail!(
            "projected attention states row 0 has width {}, expected {expected_width}",
            projected.cols()
        );
    }
    let mut heads = HeadSlab::zeroed(num_heads, projected.rows(), head_dim);
    for seq_idx in 0..projected.rows() {
        let row = projected.row(seq_idx);
        for head_idx in 0..num_heads {
            heads
                .head_row_mut(head_idx, seq_idx)
                .copy_from_slice(&row[head_idx * head_dim..(head_idx + 1) * head_dim]);
        }
    }
    Ok(heads)
}

fn apply_head_rms_norm_slab(
    heads: &mut HeadSlab,
    weight_det: Option<&[Wgt]>,
    eps_det: Option<Acc>,
) -> Result<()> {
    let (weight, eps) = rms_norm_weights(weight_det, eps_det)?;
    let cols = heads.cols().max(1);
    heads
        .heads_chunks_mut()
        .par_bridge()
        .try_for_each(|head| {
            head.chunks_mut(cols)
                .try_for_each(|row| rms_norm_row_checked(row, weight, eps))
        })
}

fn apply_value_rms_norm_slab(heads: &mut HeadSlab, eps_det: Option<Acc>) -> Result<()> {
    let eps = eps_det
        .ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    let cols = heads.cols().max(1);
    heads.heads_chunks_mut().par_bridge().for_each(|head| {
        for row in head.chunks_mut(cols) {
            value_rms_norm_in_place(row, eps);
        }
    });
    Ok(())
}

fn apply_rope_slab(
    heads: &mut HeadSlab,
    rotary_dim: usize,
    freq_base_dim: usize,
    base_det: Option<Acc>,
    position_offset: usize,
) -> Result<()> {
    if rotary_dim == 0 {
        return Ok(());
    }
    let base =
        base_det.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    let cols = heads.cols().max(1);
    heads.heads_chunks_mut().par_bridge().for_each(|head| {
        for (position, row) in head.chunks_mut(cols).enumerate() {
            rope_rotate_pairs_in_place(
                row,
                rotary_dim,
                freq_base_dim,
                base,
                position_offset + position,
            );
        }
    });
    Ok(())
}

fn apply_rope_row_heads(
    rows: &mut [Act],
    head_dim: usize,
    rotary_dim: usize,
    freq_base_dim: usize,
    base_det: Option<Acc>,
    position: usize,
) -> Result<()> {
    if rotary_dim == 0 {
        return Ok(());
    }
    let base =
        base_det.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    for row in rows.chunks_mut(head_dim.max(1)) {
        rope_rotate_pairs_in_place(row, rotary_dim, freq_base_dim, base, position);
    }
    Ok(())
}

fn kv_groups(layer: &ResolvedGemma4LayerWeights) -> Result<usize> {
    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }
    Ok(kv_groups)
}

fn project_qkv_slab(
    normed: &ActSlab,
    layer: &ResolvedGemma4LayerWeights,
) -> Result<(ActSlab, ActSlab, ActSlab)> {
    let projection_error = "deterministic linear sequence projection requires canonical det_weight";
    let q = det_linear_slab(
        normed,
        require_det_weight(layer.q_proj_det.as_deref(), projection_error)?,
    )?;
    let raw_k = det_linear_slab(
        normed,
        require_det_weight(layer.k_proj_det.as_deref(), projection_error)?,
    )?;
    let raw_v = if layer.v_proj.is_some() {
        det_linear_slab(
            normed,
            require_det_weight(layer.v_proj_det.as_deref(), projection_error)?,
        )?
    } else if layer.attention_k_eq_v {
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };
    Ok((q, raw_k, raw_v))
}

fn det_attention_prefill(
    normed: &ActSlab,
    layer: &ResolvedGemma4LayerWeights,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    donor_cache: Option<&DetKvCacheData>,
) -> Result<(ActSlab, Option<DetKvCacheData>)> {
    let seq_len = normed.rows();
    let kv_groups = kv_groups(layer)?;

    let (q_projected, raw_k, raw_v) = project_qkv_slab(normed, layer)?;

    let mut q = reshape_to_heads(&q_projected, layer.num_heads, layer.head_dim)?;
    let mut k = reshape_to_heads(&raw_k, layer.num_kv_heads, layer.head_dim)?;
    let mut v = reshape_to_heads(&raw_v, layer.num_kv_heads, layer.head_dim)?;

    apply_head_rms_norm_slab(&mut q, layer.q_norm_weight_det.as_deref(), layer.rms_norm_eps_det)?;
    apply_head_rms_norm_slab(&mut k, layer.k_norm_weight_det.as_deref(), layer.rms_norm_eps_det)?;
    apply_value_rms_norm_slab(&mut v, layer.rms_norm_eps_det)?;

    apply_rope_slab(
        &mut q,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base_det,
        0,
    )?;
    apply_rope_slab(
        &mut k,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base_det,
        0,
    )?;

    let layer_cache = if donor_cache.is_some() {
        None
    } else {
        let retained = cache_window.map_or(0, |window| seq_len.saturating_sub(window));
        Some(DetKvCacheData::from_head_slabs(&k, &v, retained))
    };

    // Per-head attention into a head-major output slab; each head writes its
    // own contiguous chunk, with zero allocation inside the (head, query)
    // loop beyond the per-head scratch reuse.
    let mut head_outputs = HeadSlab::zeroed(layer.num_heads, seq_len, layer.head_dim);
    head_outputs
        .heads_chunks_mut()
        .enumerate()
        .par_bridge()
        .try_for_each(|(head_idx, head_out)| -> Result<()> {
            let kv_head_idx = head_idx / kv_groups;
            let mut logits_scratch = Vec::with_capacity(seq_len);
            let mut exp_scratch = Vec::with_capacity(seq_len);
            let mut weights_scratch = Vec::with_capacity(seq_len);
            for (query_idx, output_row) in
                head_out.chunks_mut(layer.head_dim.max(1)).enumerate()
            {
                let start = attention_window
                    .map(|window| query_idx.saturating_add(1).saturating_sub(window))
                    .unwrap_or(0);
                let row_count = query_idx + 1 - start;
                let query = q.head_row(head_idx, query_idx);
                let windows = match donor_cache {
                    Some(donor) => cache_windows(donor, kv_head_idx, start, row_count),
                    None => AttentionWindows {
                        keys: k.head_rows_window(kv_head_idx, start, row_count),
                        values: v.head_rows_window(kv_head_idx, start, row_count),
                        rows: row_count,
                    },
                };
                attention_output_into(
                    query,
                    &windows,
                    layer.head_dim,
                    &mut logits_scratch,
                    &mut exp_scratch,
                    &mut weights_scratch,
                    output_row,
                );
            }
            Ok(())
        })?;

    // Combine head-major outputs into seq-major rows for the output projection.
    let mut combined = ActSlab::zeroed(seq_len, layer.num_heads * layer.head_dim);
    for seq_idx in 0..seq_len {
        let row = combined.row_mut(seq_idx);
        for head_idx in 0..layer.num_heads {
            row[head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim]
                .copy_from_slice(head_outputs.head_row(head_idx, seq_idx));
        }
    }

    let attn_out = det_linear_slab(
        &combined,
        require_det_weight(
            layer.o_proj_det.as_deref(),
            "deterministic linear sequence projection requires canonical det_weight",
        )?,
    )?;
    Ok((attn_out, layer_cache))
}

/// Deterministic prefill for one transformer layer over flat slabs.
///
/// Mirrors the deterministic branches of `run_gemma4_layer_with_cache_internal`
/// exactly. Returns the layer output and the layer's KV cache (`None` for
/// donor layers, matching the legacy empty-cache semantics).
pub(crate) fn det_layer_prefill(
    xs: &ActSlab,
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&ActSlab>,
    donor_cache: Option<&DetKvCacheData>,
) -> Result<(ActSlab, Option<DetKvCacheData>)> {
    let _single_track = crate::shared::numerics::det_num::enter_det_single_track_region();
    if xs.rows() == 0 {
        bail!("transformer layer execution requires at least one activation row");
    }
    if xs.cols() != layer.hidden_size {
        bail!(
            "input activations row 0 has width {}, expected {}",
            xs.cols(),
            layer.hidden_size
        );
    }
    if let Some(per_layer_input) = per_layer_input {
        let expected = layer
            .ple
            .as_ref()
            .ok_or_else(|| anyhow!("transformer layer received PLE inputs without PLE weights"))?
            .input_gate
            .as_ref()
            .rows;
        if per_layer_input.cols() != expected {
            bail!(
                "per-layer inputs row 0 has width {}, expected {expected}",
                per_layer_input.cols()
            );
        }
        if per_layer_input.rows() != xs.rows() {
            bail!(
                "transformer layer execution requires per-layer inputs and activations to have matching lengths"
            );
        }
    }

    // Attention block.
    let normed = rms_norm_slab(
        xs,
        layer.input_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    let (attention_window, cache_window) = match layer.attention_kind {
        crate::shared::model::transformer::Gemma4AttentionKind::Sliding => {
            let sliding_window = layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?;
            (Some(sliding_window), layer.cache_sliding_window)
        }
        crate::shared::model::transformer::Gemma4AttentionKind::Full => (None, None),
    };
    let (attn_out, layer_cache) =
        det_attention_prefill(&normed, layer, attention_window, cache_window, donor_cache)?;
    let mut attn_out = rms_norm_slab(
        &attn_out,
        layer.post_attention_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    // Residual add: attn_out += xs (saturating add is commutative, matching
    // the legacy `residual + attn_out` order bit-for-bit).
    for (dst, src) in attn_out.rows_chunks_mut().zip(xs.iter_rows()) {
        add_in_place(dst, src)?;
    }
    let mut xs = attn_out;

    // MLP block.
    let normed = rms_norm_slab(
        &xs,
        layer.pre_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    let mut gate = det_linear_slab(
        &normed,
        require_det_weight(
            layer.gate_proj_det.as_deref(),
            "deterministic MLP gate projection requires canonical det_weight",
        )?,
    )?;
    let up = det_linear_slab(
        &normed,
        require_det_weight(
            layer.up_proj_det.as_deref(),
            "deterministic MLP up projection requires canonical det_weight",
        )?,
    )?;
    gelu_in_place(gate.as_flat_mut());
    for (dst, src) in gate.rows_chunks_mut().zip(up.iter_rows()) {
        mul_in_place(dst, src)?;
    }
    let ff_out = det_linear_slab(
        &gate,
        require_det_weight(
            layer.down_proj_det.as_deref(),
            "deterministic MLP down projection requires canonical det_weight",
        )?,
    )?;
    let mut ff_out = rms_norm_slab(
        &ff_out,
        layer.post_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    for (dst, src) in ff_out.rows_chunks_mut().zip(xs.iter_rows()) {
        add_in_place(dst, src)?;
    }
    xs = ff_out;

    // Per-layer-embedding block.
    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let projection_error =
            "deterministic linear sequence projection requires canonical det_weight";
        let mut gated = det_linear_slab(
            &xs,
            require_det_weight(ple.input_gate_det.as_deref(), projection_error)?,
        )?;
        gelu_in_place(gated.as_flat_mut());
        for (dst, src) in gated.rows_chunks_mut().zip(per_layer_input.iter_rows()) {
            mul_in_place(dst, src)?;
        }
        let projected = det_linear_slab(
            &gated,
            require_det_weight(ple.layer_projection_det.as_deref(), projection_error)?,
        )?;
        let mut projected = rms_norm_slab(
            &projected,
            ple.post_input_norm_weight_det.as_deref(),
            layer.rms_norm_eps_det,
        )?;
        for (dst, src) in projected.rows_chunks_mut().zip(xs.iter_rows()) {
            add_in_place(dst, src)?;
        }
        xs = projected;
    }

    if layer.layer_scalar.is_some() {
        let scalar = layer.layer_scalar_det.ok_or_else(|| {
            anyhow!("deterministic sequence scaling requires canonical Act scalar")
        })?;
        scale_in_place(xs.as_flat_mut(), scalar);
    }

    Ok((xs, layer_cache))
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Reusable per-step scratch for the deterministic decode layer loop. All
/// buffers retain capacity across layers and steps; inside the per-layer loop
/// no heap allocation happens once capacities are warm.
#[derive(Default)]
pub(crate) struct DetDecodeScratch {
    normed: Vec<Act>,
    q: Vec<Act>,
    k: Vec<Act>,
    v: Vec<Act>,
    attn_combined: Vec<Act>,
    attn_out: Vec<Act>,
    xs: Vec<Act>,
    ff_normed: Vec<Act>,
    gate: Vec<Act>,
    up: Vec<Act>,
    ff_out: Vec<Act>,
    ple_gate: Vec<Act>,
    ple_projected: Vec<Act>,
    logits: Vec<Act>,
    exp_terms: Vec<Acc>,
    weights: Vec<Act>,
}

impl DetDecodeScratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

fn reset(buffer: &mut Vec<Act>, len: usize) -> &mut [Act] {
    buffer.clear();
    buffer.resize(len, Act::from_bits(0));
    buffer.as_mut_slice()
}

/// Deterministic decode for one transformer layer, mutating the layer's KV
/// cache in place (no per-token cache clone). `output` is cleared and filled
/// with the layer output row.
pub(crate) fn det_layer_decode(
    input: &[Act],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[Act]>,
    cache: &mut LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    scratch: &mut DetDecodeScratch,
    output: &mut Vec<Act>,
) -> Result<()> {
    let _single_track = crate::shared::numerics::det_num::enter_det_single_track_region();
    if input.len() != layer.hidden_size {
        bail!(
            "decode input activation width mismatch: {} vs {}",
            input.len(),
            layer.hidden_size
        );
    }
    if let Some(per_layer_input) = per_layer_input {
        let expected = layer
            .ple
            .as_ref()
            .ok_or_else(|| anyhow!("transformer decode received PLE inputs without PLE weights"))?
            .input_gate
            .as_ref()
            .rows;
        if per_layer_input.len() != expected {
            bail!(
                "decode per-layer input width mismatch: {} vs {expected}",
                per_layer_input.len()
            );
        }
    }
    let kv_groups = kv_groups(layer)?;
    let validated_cache = donor_cache.unwrap_or(cache);
    if validated_cache.keys.len() != layer.num_kv_heads
        || validated_cache.values.len() != layer.num_kv_heads
    {
        bail!(
            "layer cache head count mismatch: keys {} values {} expected {}",
            validated_cache.keys.len(),
            validated_cache.values.len(),
            layer.num_kv_heads
        );
    }
    let (attention_window, cache_window) = match layer.attention_kind {
        crate::shared::model::transformer::Gemma4AttentionKind::Sliding => {
            let sliding_window = layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?;
            (Some(sliding_window), layer.cache_sliding_window)
        }
        crate::shared::model::transformer::Gemma4AttentionKind::Full => (None, None),
    };

    // Input norm.
    let (norm_weight, norm_eps) = rms_norm_weights(
        layer.input_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    let normed = reset(&mut scratch.normed, input.len());
    normed.copy_from_slice(input);
    rms_norm_row_checked(normed, norm_weight, norm_eps)?;

    // Q/K/V projections.
    let projection_error = "deterministic linear row projection requires canonical det_weight";
    let q_weight = require_det_weight(layer.q_proj_det.as_deref(), projection_error)?;
    let k_weight = require_det_weight(layer.k_proj_det.as_deref(), projection_error)?;
    let q = reset(&mut scratch.q, q_weight.rows);
    det_linear_into(&scratch.normed, q_weight, q)?;
    let k = reset(&mut scratch.k, k_weight.rows);
    det_linear_into(&scratch.normed, k_weight, k)?;
    let v_len = if layer.v_proj.is_some() {
        let v_weight = require_det_weight(layer.v_proj_det.as_deref(), projection_error)?;
        let v = reset(&mut scratch.v, v_weight.rows);
        det_linear_into(&scratch.normed, v_weight, v)?;
        v_weight.rows
    } else if layer.attention_k_eq_v {
        let v = reset(&mut scratch.v, k_weight.rows);
        v.copy_from_slice(&scratch.k);
        k_weight.rows
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };
    debug_assert_eq!(v_len, layer.num_kv_heads * layer.head_dim);

    // Per-head norms + RoPE, in place over head-dim chunks.
    let head_dim = layer.head_dim.max(1);
    let (q_norm_weight, q_norm_eps) =
        rms_norm_weights(layer.q_norm_weight_det.as_deref(), layer.rms_norm_eps_det)?;
    for row in scratch.q.chunks_mut(head_dim) {
        rms_norm_row_checked(row, q_norm_weight, q_norm_eps)?;
    }
    let (k_norm_weight, k_norm_eps) =
        rms_norm_weights(layer.k_norm_weight_det.as_deref(), layer.rms_norm_eps_det)?;
    for row in scratch.k.chunks_mut(head_dim) {
        rms_norm_row_checked(row, k_norm_weight, k_norm_eps)?;
    }
    let value_eps = layer.rms_norm_eps_det.ok_or_else(|| {
        anyhow!("deterministic value RMSNorm requires canonical Acc epsilon")
    })?;
    for row in scratch.v.chunks_mut(head_dim) {
        value_rms_norm_in_place(row, value_eps);
    }
    apply_rope_row_heads(
        &mut scratch.q,
        layer.head_dim,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base_det,
        position,
    )?;
    apply_rope_row_heads(
        &mut scratch.k,
        layer.head_dim,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base_det,
        position,
    )?;

    // KV cache append (in place; donor layers leave their own cache untouched).
    if donor_cache.is_none() {
        if cache.det.is_none() {
            if cache.keys.iter().all(|head| head.is_empty()) {
                cache.det = Some(DetKvCacheData::new(layer.num_kv_heads, layer.head_dim));
            } else {
                bail!("deterministic decode cache append requires canonical key rows");
            }
        }
        let det = cache.det.as_mut().expect("det cache should be initialized");
        det.append_rows(&scratch.k, &scratch.v, layer.head_dim, cache_window)?;
    }

    // Attention.
    let attention_cache = donor_cache.unwrap_or(cache);
    let attention_det = attention_cache.det_data().ok_or_else(|| {
        anyhow!("deterministic attention requires canonical key cache rows")
    })?;
    let key_start = attention_window
        .map(|window| attention_cache.current_len().saturating_sub(window))
        .unwrap_or(0);
    let window_len = attention_det.len().saturating_sub(key_start);
    let attn_combined = reset(&mut scratch.attn_combined, layer.num_heads * layer.head_dim);
    for head_idx in 0..layer.num_heads {
        let kv_head_idx = head_idx / kv_groups;
        let windows = cache_windows(attention_det, kv_head_idx, key_start, window_len);
        let query = &scratch.q[head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
        attention_output_into(
            query,
            &windows,
            layer.head_dim,
            &mut scratch.logits,
            &mut scratch.exp_terms,
            &mut scratch.weights,
            &mut attn_combined[head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim],
        );
    }
    let o_weight = require_det_weight(layer.o_proj_det.as_deref(), projection_error)?;
    let attn_out = reset(&mut scratch.attn_out, o_weight.rows);
    det_linear_into(&scratch.attn_combined, o_weight, attn_out)?;

    // Post-attention norm + residual.
    let (post_attn_weight, post_attn_eps) = rms_norm_weights(
        layer.post_attention_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    rms_norm_row_checked(attn_out, post_attn_weight, post_attn_eps)?;
    let xs = reset(&mut scratch.xs, input.len());
    xs.copy_from_slice(input);
    add_in_place(xs, &scratch.attn_out)?;

    // MLP block.
    let (pre_ff_weight, pre_ff_eps) = rms_norm_weights(
        layer.pre_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    let ff_normed = reset(&mut scratch.ff_normed, scratch.xs.len());
    ff_normed.copy_from_slice(&scratch.xs);
    rms_norm_row_checked(ff_normed, pre_ff_weight, pre_ff_eps)?;
    let gate_weight = require_det_weight(
        layer.gate_proj_det.as_deref(),
        "deterministic MLP gate projection requires canonical det_weight",
    )?;
    let gate = reset(&mut scratch.gate, gate_weight.rows);
    det_linear_into(&scratch.ff_normed, gate_weight, gate)?;
    gelu_in_place(gate);
    let up_weight = require_det_weight(
        layer.up_proj_det.as_deref(),
        "deterministic MLP up projection requires canonical det_weight",
    )?;
    let up = reset(&mut scratch.up, up_weight.rows);
    det_linear_into(&scratch.ff_normed, up_weight, up)?;
    mul_in_place(&mut scratch.gate, &scratch.up)?;
    let down_weight = require_det_weight(
        layer.down_proj_det.as_deref(),
        "deterministic MLP down projection requires canonical det_weight",
    )?;
    let ff_out = reset(&mut scratch.ff_out, down_weight.rows);
    det_linear_into(&scratch.gate, down_weight, ff_out)?;
    let (post_ff_weight, post_ff_eps) = rms_norm_weights(
        layer.post_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps_det,
    )?;
    rms_norm_row_checked(ff_out, post_ff_weight, post_ff_eps)?;
    add_in_place(&mut scratch.xs, &scratch.ff_out)?;

    // Per-layer-embedding block.
    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let input_gate_weight =
            require_det_weight(ple.input_gate_det.as_deref(), projection_error)?;
        let ple_gate = reset(&mut scratch.ple_gate, input_gate_weight.rows);
        det_linear_into(&scratch.xs, input_gate_weight, ple_gate)?;
        gelu_in_place(ple_gate);
        mul_in_place(&mut scratch.ple_gate, per_layer_input)?;
        let layer_projection_weight =
            require_det_weight(ple.layer_projection_det.as_deref(), projection_error)?;
        let ple_projected = reset(&mut scratch.ple_projected, layer_projection_weight.rows);
        det_linear_into(&scratch.ple_gate, layer_projection_weight, ple_projected)?;
        let (ple_norm_weight, ple_norm_eps) = rms_norm_weights(
            ple.post_input_norm_weight_det.as_deref(),
            layer.rms_norm_eps_det,
        )?;
        rms_norm_row_checked(ple_projected, ple_norm_weight, ple_norm_eps)?;
        add_in_place(&mut scratch.xs, &scratch.ple_projected)?;
    }

    if layer.layer_scalar.is_some() {
        let scalar = layer
            .layer_scalar_det
            .ok_or_else(|| anyhow!("deterministic row scaling requires canonical Act scalar"))?;
        scale_in_place(&mut scratch.xs, scalar);
    }

    output.clear();
    output.extend_from_slice(&scratch.xs);
    Ok(())
}

// ---------------------------------------------------------------------------
// Decode PLE input
// ---------------------------------------------------------------------------

/// Deterministic decode-time per-layer-embedding input, mirroring
/// `compute_decode_ple_input_internal`'s deterministic semantics. `output` is
/// cleared and filled when the layer carries PLE weights.
pub(crate) fn det_decode_ple_input(
    token_id: u32,
    input: &[Act],
    layer_idx: usize,
    layer: &crate::shared::model::transformer::Gemma4LayerWeights,
    ple_global: Option<&crate::shared::model::transformer::Gemma4PleGlobalWeights>,
    rms_norm_eps_det: Option<Acc>,
    output: &mut Vec<Act>,
) -> Result<bool> {
    let _single_track = crate::shared::numerics::det_num::enter_det_single_track_region();
    let Some(ple_global) = ple_global else {
        return Ok(false);
    };
    if layer.ple.is_none() {
        return Ok(false);
    }

    if input.len() != layer.hidden_size {
        bail!(
            "decode PLE input activation width mismatch: {} vs {}",
            input.len(),
            layer.hidden_size
        );
    }

    let embedded_row =
        crate::io::load_ple_token_embedding_row_internal(ple_global, layer_idx, token_id)?;
    let mut embedded = row_from_internal(&embedded_row)?;
    let embedding_scale = ple_global
        .embedding_scale_det
        .ok_or_else(|| anyhow!("deterministic row scaling requires canonical Act scalar"))?;
    scale_in_place(&mut embedded, embedding_scale);

    let model_projection_det =
        crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?;
    let model_projection = require_det_weight(
        model_projection_det.as_deref(),
        "deterministic linear row projection requires canonical det_weight",
    )?;
    output.clear();
    output.resize(model_projection.rows, Act::from_bits(0));
    det_linear_into(input, model_projection, output)?;
    let projection_scalar = ple_global
        .projection_scalar_det
        .ok_or_else(|| anyhow!("deterministic row scaling requires canonical Act scalar"))?;
    scale_in_place(output, projection_scalar);
    let (projection_norm_weight, projection_norm_eps) = rms_norm_weights(
        ple_global.projection_norm_weight_det.as_deref(),
        rms_norm_eps_det,
    )?;
    rms_norm_row_checked(output, projection_norm_weight, projection_norm_eps)?;

    if embedded.len() != output.len() {
        bail!(
            "decode PLE width mismatch: embedded {} vs projected {}",
            embedded.len(),
            output.len()
        );
    }
    // combined = embedded + projected (saturating add; commutative).
    add_in_place(output, &embedded)?;
    let input_scale = ple_global
        .input_scale_det
        .ok_or_else(|| anyhow!("deterministic row scaling requires canonical Act scalar"))?;
    scale_in_place(output, input_scale);
    Ok(true)
}

// ---------------------------------------------------------------------------
// Logits
// ---------------------------------------------------------------------------

/// Deterministic hidden-state → logits projection (final norm, logits matmul,
/// optional softcap), mirroring `project_internal_hidden_to_prefill_logits`'s
/// deterministic semantics without any f32 view.
pub(crate) fn det_hidden_to_logits(
    hidden_state: &[Act],
    final_norm_weight_det: Option<&[Wgt]>,
    rms_norm_eps_det: Option<Acc>,
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    final_logit_softcapping: Option<f32>,
    final_logit_softcapping_det: Option<Act>,
) -> Result<Vec<Act>> {
    let _single_track = crate::shared::numerics::det_num::enter_det_single_track_region();
    let (norm_weight, norm_eps) = rms_norm_weights(final_norm_weight_det, rms_norm_eps_det)?;
    let mut normed = hidden_state.to_vec();
    rms_norm_row_checked(&mut normed, norm_weight, norm_eps)?;

    let det_weight = match projection {
        Gemma4LogitsProjection::UntiedLmHead {
            det_weight: Some(det_weight),
            ..
        } => det_weight.clone(),
        Gemma4LogitsProjection::TiedEmbedding(_) => {
            let embedding_source = embedding_source.ok_or_else(|| {
                anyhow!("deterministic tied embedding logits require a .detwgt embedding source")
            })?;
            crate::io::materialize_det_num_embedding_matrix(embedding_source)?.ok_or_else(
                || anyhow!("deterministic tied embedding logits require a .detwgt embedding matrix"),
            )?
        }
        Gemma4LogitsProjection::UntiedLmHead {
            det_weight: None, ..
        } => bail!("deterministic untied logits projection requires canonical det_weight"),
    };
    if normed.len() != det_weight.cols {
        bail!(
            "logits projection input width mismatch: {} vs {}",
            normed.len(),
            det_weight.cols
        );
    }
    let mut logits = vec![Act::from_bits(0); det_weight.rows];
    det_linear_into(&normed, &det_weight, &mut logits)?;

    if let Some(_softcap) = final_logit_softcapping {
        let softcap = final_logit_softcapping_det.ok_or_else(|| {
            anyhow!("deterministic final logit softcapping requires canonical Act softcap")
        })?;
        for logit in logits.iter_mut() {
            *logit = softcap_act(*logit, softcap);
        }
    }
    Ok(logits)
}
