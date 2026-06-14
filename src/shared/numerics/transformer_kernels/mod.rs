use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::shared::model::transformer::{
    ActivationSequence, DetNumMatrix, Gemma4AttentionKind, Gemma4LayerWeights,
    Gemma4PleGlobalWeights, InternalActivationRow, InternalActivationSequence, InternalLogits,
    LayerKvCache, MatrixF32, PrefillLogits, ResolvedGemma4LayerWeights,
};
use crate::shared::numerics::det_num::{
    act_to_f32, act_to_le_bytes, add_sat, attention_score as det_attention_score,
    attention_softmax as det_attention_softmax,
    attention_weighted_sum as det_attention_weighted_sum, gelu_pytorch_tanh_act, mac_bits, mul_sat,
    requantize, rms_norm as det_rms_norm, rope_rotate_pairs as det_rope_rotate_pairs, scale_act,
    value_rms_norm as det_value_rms_norm, Acc, Act, Wgt,
};

pub(crate) fn compute_decode_ple_input_internal(
    token_id: u32,
    input_activation: InternalActivationRow,
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
    ple_global: Option<&Gemma4PleGlobalWeights>,
    rms_norm_eps: f32,
    rms_norm_eps_det: Option<Acc>,
) -> Result<Option<InternalActivationRow>> {
    let Some(ple_global) = ple_global else {
        return Ok(None);
    };
    if layer.ple.is_none() {
        return Ok(None);
    }

    let input_values = input_activation.as_f32_slice();
    validate_vector_width(
        input_values,
        layer.hidden_size,
        "decode PLE input activation",
    )?;
    let embedded = scale_row_buffer(
        &ActivationRowBuffer::from_internal(crate::io::load_ple_token_embedding_row_internal(
            ple_global, layer_idx, token_id,
        )?),
        ple_global.embedding_scale_det,
    )?;
    let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
    let model_projection_det =
        crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?;
    let projected = project_linear_row_buffer(
        &ActivationRowBuffer::from_internal(input_activation),
        &model_projection,
        model_projection_det.as_deref(),
    )?;
    let projected = scale_row_buffer(&projected, ple_global.projection_scalar_det)?;
    let projected = apply_rms_norm_row_buffer(
        &projected,
        &ple_global.projection_norm_weight,
        ple_global.projection_norm_weight_det.as_deref(),
        rms_norm_eps,
        rms_norm_eps_det,
    )?;

    if embedded.values.len() != projected.values.len() {
        bail!(
            "decode PLE width mismatch: embedded {} vs projected {}",
            embedded.values.len(),
            projected.values.len()
        );
    }

    let combined = add_row_buffers(&embedded, &projected)?;
    let combined = scale_row_buffer(&combined, ple_global.input_scale_det)?;

    Ok(Some(combined.into_internal()))
}

pub(crate) fn run_gemma4_layer_with_cache_internal(
    input_activations: InternalActivationSequence,
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<InternalActivationSequence>,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(ActivationSequence, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_gemma4_layer attention={:?}",
    //     layer.attention_kind
    // ));
    let input_values = input_activations.as_f32_slice();
    if input_values.is_empty() {
        bail!("transformer layer execution requires at least one activation row");
    }
    validate_sequence_width(input_values, layer.hidden_size, "input activations")?;
    if let Some(per_layer_input) = per_layer_input.as_ref() {
        validate_sequence_width(
            per_layer_input.as_f32_slice(),
            layer
                .ple
                .as_ref()
                .ok_or_else(|| {
                    anyhow!("transformer layer received PLE inputs without PLE weights")
                })?
                .input_gate
                .as_ref()
                .rows,
            "per-layer inputs",
        )?;
        if per_layer_input.as_f32_slice().len() != input_values.len() {
            bail!(
                "transformer layer execution requires per-layer inputs and activations to have matching lengths"
            );
        }
    }

    let mut xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.clone_input");
        ActivationSequenceBuffer::from_internal(input_activations)
    };

    let residual = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.clone_residual");
        xs.clone()
    };
    let normed = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.input_rms_norm");
        apply_rms_norm_to_sequence_buffer(
            &xs,
            &layer.input_layernorm_weight,
            layer.input_layernorm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
        )?
    };
    let (attn_out, layer_cache) = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.core");
        run_attention_for_layer_with_cache(&normed, layer, donor_cache)?
    };
    let attn_out = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.post_rms_norm");
        apply_rms_norm_to_sequence_buffer(
            &attn_out,
            &layer.post_attention_layernorm_weight,
            layer.post_attention_layernorm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
        )?
    };
    xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.residual_add");
        add_sequence_buffers(&residual, &attn_out)?
    };

    let residual = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.clone_residual");
        xs.clone()
    };
    let normed = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.pre_rms_norm");
        apply_rms_norm_to_sequence_buffer(
            &xs,
            &layer.pre_feedforward_layernorm_weight,
            layer.pre_feedforward_layernorm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
        )?
    };
    let (gate, up) = match (layer.gate_proj_det.as_ref(), layer.up_proj_det.as_ref()) {
        (Some(gate_weight), Some(up_weight)) => {
            let quantized_normed = sequence_buffer_acts(&normed)?;
            let gate_preactivation = ActivationSequenceBuffer::from_acts(
                det_linear_sequence_acts_from_acts(&quantized_normed, gate_weight.as_ref())?,
            );
            let up = quantized_normed
                .par_iter()
                .map(|row| det_linear_row_acts_from_acts(row, up_weight.as_ref()))
                .collect::<Vec<_>>()
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            (
                apply_gelu_to_sequence_buffer(&gate_preactivation)?,
                ActivationSequenceBuffer::from_acts(up),
            )
        }
        (gate_weight, up_weight) => {
            let gate_preactivation = match gate_weight {
                Some(weight) => project_linear_sequence_buffer(
                    &normed,
                    layer.gate_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => {
                    bail!("deterministic MLP gate projection requires canonical det_weight")
                }
            };
            let up = match up_weight {
                Some(weight) => project_linear_sequence_buffer(
                    &normed,
                    layer.up_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => {
                    bail!("deterministic MLP up projection requires canonical det_weight")
                }
            };
            (apply_gelu_to_sequence_buffer(&gate_preactivation)?, up)
        }
    };
    let ff_hidden = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.hidden_mul");
        mul_sequence_buffers(&gate, &up)?
    };
    let ff_out = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.down_proj");
        match layer.down_proj_det.as_ref() {
            Some(weight) => project_linear_sequence_buffer(
                &ff_hidden,
                layer.down_proj.as_ref(),
                Some(weight.as_ref()),
            )?,
            None => {
                bail!("deterministic MLP down projection requires canonical det_weight")
            }
        }
    };
    let ff_out = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.post_rms_norm");
        apply_rms_norm_to_sequence_buffer(
            &ff_out,
            &layer.post_feedforward_layernorm_weight,
            layer.post_feedforward_layernorm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
        )?
    };
    xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.residual_add");
        add_sequence_buffers(&residual, &ff_out)?
    };

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.clone_residual");
            xs.clone()
        };
        let gated = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.input_gate_gelu");
            let gate_preactivation = project_linear_sequence_buffer(
                &xs,
                ple.input_gate.as_ref(),
                ple.input_gate_det.as_deref(),
            )?;
            apply_gelu_to_sequence_buffer(&gate_preactivation)?
        };
        let gated = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.input_mul");
            mul_sequence_buffers(
                &gated,
                &ActivationSequenceBuffer::from_internal(per_layer_input),
            )?
        };
        let projected = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.layer_projection");
            project_linear_sequence_buffer(
                &gated,
                ple.layer_projection.as_ref(),
                ple.layer_projection_det.as_deref(),
            )?
        };
        let projected = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.post_rms_norm");
            apply_rms_norm_to_sequence_buffer(
                &projected,
                &ple.post_input_norm_weight,
                ple.post_input_norm_weight_det.as_deref(),
                layer.rms_norm_eps,
                layer.rms_norm_eps_det,
            )?
        };
        xs = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.residual_add");
            add_sequence_buffers(&residual, &projected)?
        };
    }

    if layer.layer_scalar.is_some() {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.layer_scalar");
        xs = scale_sequence_buffer(&xs, layer.layer_scalar_det)?;
    }

    Ok((activation_sequence_from_buffer(xs), layer_cache))
}

pub(crate) fn run_gemma4_layer_decode_internal(
    input_activation: InternalActivationRow,
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<InternalActivationRow>,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
) -> Result<(InternalActivationRow, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_gemma4_layer_decode attention={:?}",
    //     layer.attention_kind
    // ));
    let input_values = input_activation.as_f32_slice();
    validate_vector_width(input_values, layer.hidden_size, "decode input activation")?;
    if let Some(per_layer_input) = per_layer_input.as_ref() {
        validate_vector_width(
            per_layer_input.as_f32_slice(),
            layer
                .ple
                .as_ref()
                .ok_or_else(|| {
                    anyhow!("transformer decode received PLE inputs without PLE weights")
                })?
                .input_gate
                .as_ref()
                .rows,
            "decode per-layer input",
        )?;
    }

    let residual = ActivationRowBuffer::from_internal(input_activation);
    let normed = apply_rms_norm_row_buffer(
        &residual,
        &layer.input_layernorm_weight,
        layer.input_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
    )?;
    let (xs_values, updated_cache) =
        run_attention_for_layer_decode(&normed, layer, cache, donor_cache, position)?;
    let mut xs = apply_rms_norm_row_buffer(
        &xs_values,
        &layer.post_attention_layernorm_weight,
        layer.post_attention_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
    )?;
    xs = add_row_buffers(&residual, &xs)?;

    let residual = xs.clone();
    let normed = apply_rms_norm_row_buffer(
        &xs,
        &layer.pre_feedforward_layernorm_weight,
        layer.pre_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
    )?;
    let (gate_preactivation, up) = match (layer.gate_proj_det.as_ref(), layer.up_proj_det.as_ref())
    {
        (Some(gate_weight), Some(up_weight)) => {
            let quantized_normed = row_buffer_acts(&normed)?;
            (
                apply_gelu_to_row_buffer(&ActivationRowBuffer::from_acts(
                    det_linear_row_acts_from_acts(&quantized_normed, gate_weight.as_ref())?,
                ))?,
                ActivationRowBuffer::from_acts(det_linear_row_acts_from_acts(
                    &quantized_normed,
                    up_weight.as_ref(),
                )?),
            )
        }
        (gate_weight, up_weight) => {
            let gate_preactivation = match gate_weight {
                Some(weight) => project_linear_row_buffer(
                    &normed,
                    layer.gate_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => {
                    bail!("deterministic MLP gate projection requires canonical det_weight")
                }
            };
            let up = match up_weight {
                Some(weight) => project_linear_row_buffer(
                    &normed,
                    layer.up_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => {
                    bail!("deterministic MLP up projection requires canonical det_weight")
                }
            };
            (apply_gelu_to_row_buffer(&gate_preactivation)?, up)
        }
    };
    let ff_hidden = mul_row_buffers(&gate_preactivation, &up)?;
    let ff_out = match layer.down_proj_det.as_ref() {
        Some(weight) => {
            project_linear_row_buffer(&ff_hidden, layer.down_proj.as_ref(), Some(weight.as_ref()))?
        }
        None => {
            bail!("deterministic MLP down projection requires canonical det_weight")
        }
    };
    let ff_out = apply_rms_norm_row_buffer(
        &ff_out,
        &layer.post_feedforward_layernorm_weight,
        layer.post_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
    )?;
    xs = add_row_buffers(&residual, &ff_out)?;

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = xs.clone();
        let gated = apply_gelu_to_row_buffer(&project_linear_row_buffer(
            &xs,
            ple.input_gate.as_ref(),
            ple.input_gate_det.as_deref(),
        )?)?;
        let gated = mul_row_buffers(&gated, &ActivationRowBuffer::from_internal(per_layer_input))?;
        let projected = project_linear_row_buffer(
            &gated,
            ple.layer_projection.as_ref(),
            ple.layer_projection_det.as_deref(),
        )?;
        let projected = apply_rms_norm_row_buffer(
            &projected,
            &ple.post_input_norm_weight,
            ple.post_input_norm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
        )?;
        xs = add_row_buffers(&residual, &projected)?;
    }

    if layer.layer_scalar.is_some() {
        xs = scale_row_buffer(&xs, layer.layer_scalar_det)?;
    }

    Ok((xs.into_internal(), updated_cache))
}

pub fn select_final_position(input_activations: &[Vec<f32>]) -> Result<Vec<f32>> {
    // let _trace = trace_scope("transformer_state_transition.select_final_position");
    input_activations.last().cloned().ok_or_else(|| {
        anyhow!("transformer final-position selection requires at least one activation row")
    })
}

pub(crate) fn select_final_position_internal(
    input_activations: &InternalActivationSequence,
) -> Result<InternalActivationRow> {
    // let _trace = trace_scope("transformer_state_transition.select_final_position");
    input_activations.last_row().ok_or_else(|| {
        anyhow!("transformer final-position selection requires at least one activation row")
    })
}

pub fn extract_prefill_logits(logits: &[f32]) -> PrefillLogits {
    extract_internal_prefill_logits(InternalLogits::from_values(logits.to_vec()))
}

fn extract_internal_prefill_logits(logits: InternalLogits) -> PrefillLogits {
    // let _trace = trace_scope("transformer_state_transition.extract_prefill_logits");
    // Commitments remain anchored to the public f32 view until the trace/checkpoint
    // boundary is intentionally versioned.
    let final_logits_sha256 = build_vector_commitment(logits.as_f32_slice());
    let det_final_logits_sha256 = logits.det_values().map(build_det_vector_commitment);
    let mut prefill_logits = PrefillLogits::from_internal(logits, final_logits_sha256);
    prefill_logits.det_final_logits_sha256 = det_final_logits_sha256;
    prefill_logits
}

pub struct ActivationSequenceWithCache {
    pub activation_state: ActivationSequence,
    pub layer_caches: Vec<LayerKvCache>,
}

pub(crate) fn build_activation_commitment(activations: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    for row in activations {
        for value in row {
            hasher.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn build_vector_commitment(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn build_det_activation_commitment(activations: &[Vec<Act>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-det-num-act-v1");
    hasher.update((activations.len() as u64).to_le_bytes());
    for row in activations {
        hasher.update((row.len() as u64).to_le_bytes());
        for value in row {
            hasher.update(act_to_le_bytes(*value));
        }
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn build_det_vector_commitment(values: &[Act]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-det-num-act-vector-v1");
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(act_to_le_bytes(*value));
    }
    format!("{:x}", hasher.finalize())
}

/// Slab-flavored twin of [`build_det_activation_commitment`]; hashes identical
/// bytes (domain prefix, u64 LE length headers, canonical Act LE bytes).
pub(crate) fn build_det_activation_commitment_slab(
    slab: &crate::shared::numerics::det_tensor::ActSlab,
) -> String {
    build_det_activation_commitment_slab_range(slab, 0, slab.rows())
}

/// Row-range twin of [`build_det_activation_commitment_slab`] hashing rows
/// `start..end` as their own sequence (identical bytes to hashing the nested
/// sub-slice).
pub(crate) fn build_det_activation_commitment_slab_range(
    slab: &crate::shared::numerics::det_tensor::ActSlab,
    start: usize,
    end: usize,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-det-num-act-v1");
    hasher.update(((end - start) as u64).to_le_bytes());
    for row_idx in start..end {
        let row = slab.row(row_idx);
        hasher.update((row.len() as u64).to_le_bytes());
        for value in row {
            hasher.update(act_to_le_bytes(*value));
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Single-row twin of [`build_det_activation_commitment`] hashing one row as a
/// one-row sequence (identical bytes to `build_det_activation_commitment(&[row])`).
pub(crate) fn build_det_activation_commitment_row(row: &[Act]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-det-num-act-v1");
    hasher.update(1u64.to_le_bytes());
    hasher.update((row.len() as u64).to_le_bytes());
    for value in row {
        hasher.update(act_to_le_bytes(*value));
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn build_det_kv_cache_commitment(caches: &[LayerKvCache]) -> Option<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-det-num-kv-cache-v1");
    hasher.update((caches.len() as u64).to_le_bytes());
    for cache in caches {
        let det = cache.det_data()?;
        hasher.update((det.num_heads() as u64).to_le_bytes());
        for (kind, heads) in [(b"k", det.key_heads()), (b"v", det.value_heads())] {
            hasher.update(kind);
            hasher.update((heads.len() as u64).to_le_bytes());
            for head in heads {
                hasher.update((head.len() as u64).to_le_bytes());
                for row in head.iter_rows() {
                    hasher.update((row.len() as u64).to_le_bytes());
                    for value in row {
                        hasher.update(act_to_le_bytes(*value));
                    }
                }
            }
        }
    }
    Some(format!("{:x}", hasher.finalize()))
}

fn run_attention_for_layer_with_cache(
    inputs: &ActivationSequenceBuffer,
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(ActivationSequenceBuffer, LayerKvCache)> {
    match layer.attention_kind {
        Gemma4AttentionKind::Sliding => run_sliding_attention(inputs, layer, donor_cache),
        Gemma4AttentionKind::Full => run_full_attention(inputs, layer, donor_cache),
    }
}

fn run_attention_for_layer_decode(
    input: &ActivationRowBuffer,
    layer: &ResolvedGemma4LayerWeights,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
) -> Result<(ActivationRowBuffer, LayerKvCache)> {
    match layer.attention_kind {
        Gemma4AttentionKind::Sliding => {
            let sliding_window = layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?;
            run_causal_attention_decode_buffer(
                input,
                layer,
                cache,
                donor_cache,
                position,
                Some(sliding_window),
                layer.cache_sliding_window,
            )
        }
        Gemma4AttentionKind::Full => run_causal_attention_decode_buffer(
            input,
            layer,
            cache,
            donor_cache,
            position,
            None,
            None,
        ),
    }
}

fn run_sliding_attention(
    inputs: &ActivationSequenceBuffer,
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(ActivationSequenceBuffer, LayerKvCache)> {
    let sliding_window = layer
        .sliding_window
        .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?;
    run_causal_attention_buffer(
        inputs,
        layer,
        Some(sliding_window),
        layer.cache_sliding_window,
        donor_cache,
    )
}

fn run_full_attention(
    inputs: &ActivationSequenceBuffer,
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(ActivationSequenceBuffer, LayerKvCache)> {
    run_causal_attention_buffer(inputs, layer, None, None, donor_cache)
}

fn run_causal_attention_buffer(
    inputs: &ActivationSequenceBuffer,
    layer: &ResolvedGemma4LayerWeights,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(ActivationSequenceBuffer, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_causal_attention attention={:?}",
    //     layer.attention_kind
    // ));
    let seq_len = inputs.values.len();
    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow::anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }

    let q_projected = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.q_proj");
        project_linear_sequence_buffer(inputs, layer.q_proj.as_ref(), layer.q_proj_det.as_deref())?
    };
    let raw_k = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.k_proj");
        project_linear_sequence_buffer(inputs, layer.k_proj.as_ref(), layer.k_proj_det.as_deref())?
    };
    let raw_v = if let Some(v_proj) = &layer.v_proj {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.v_proj");
        project_linear_sequence_buffer(inputs, v_proj.as_ref(), layer.v_proj_det.as_deref())?
    } else if layer.attention_k_eq_v {
        // trace_event("transformer_state_transition.run_causal_attention.k_eq_v_reuse");
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };
    let mut q = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.reshape_q");
        reshape_sequence_head_buffer(&q_projected, layer.num_heads, layer.head_dim)?
    };
    let mut k = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.reshape_k");
        reshape_sequence_head_buffer(&raw_k, layer.num_kv_heads, layer.head_dim)?
    };
    let mut v = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.reshape_v");
        reshape_sequence_head_buffer(&raw_v, layer.num_kv_heads, layer.head_dim)?
    };

    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.q_rms_norm");
        apply_head_rms_norm(
            &mut q,
            &layer.q_norm_weight,
            layer.q_norm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
        )?;
    }
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.k_rms_norm");
        apply_head_rms_norm(
            &mut k,
            &layer.k_norm_weight,
            layer.k_norm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
        )?;
    }
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.v_rms_norm");
        apply_value_rms_norm(&mut v, layer.rms_norm_eps, layer.rms_norm_eps_det)?;
    }

    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.q_rope");
        apply_rope(
            &mut q,
            layer.partial_rotary_dim,
            layer.rope_freq_base_dim,
            layer.rope_base,
            layer.rope_base_det,
        )?;
    }
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.k_rope");
        apply_rope(
            &mut k,
            layer.partial_rotary_dim,
            layer.rope_freq_base_dim,
            layer.rope_base,
            layer.rope_base_det,
        )?;
    }

    let layer_cache = if donor_cache.is_some() {
        LayerKvCache::new(layer.num_kv_heads)
    } else {
        build_layer_kv_cache(&k, &v, cache_window)?
    };

    let head_outputs = (0..layer.num_heads)
        .into_par_iter()
        .map(|head_idx| -> Result<Vec<ActivationRowBuffer>> {
            // let _trace = trace_scope(format!("transformer_state_transition.run_causal_attention.head={head_idx}"));
            let kv_head_idx = head_idx / kv_groups;
            let mut outputs = Vec::with_capacity(seq_len);
            for query_idx in 0..seq_len {
                let start = attention_window
                    .map(|window| query_idx.saturating_add(1).saturating_sub(window))
                    .unwrap_or(0);
                let query = q.row(head_idx, query_idx);
                if let Some(donor_cache) = donor_cache {
                    let key_rows = donor_cache.keys[kv_head_idx]
                        .iter()
                        .skip(start)
                        .take(query_idx + 1 - start)
                        .cloned()
                        .collect::<Vec<_>>();
                    let value_rows = donor_cache.values[kv_head_idx]
                        .iter()
                        .skip(start)
                        .take(query_idx + 1 - start)
                        .cloned()
                        .collect::<Vec<_>>();
                    let row_count = query_idx + 1 - start;
                    let det_key_rows =
                        donor_cache.det_key_rows_window(kv_head_idx, start, row_count);
                    let det_value_rows =
                        donor_cache.det_value_rows_window(kv_head_idx, start, row_count);
                    outputs.push(attention_output(
                        &query,
                        &key_rows,
                        det_key_rows.as_deref(),
                        &value_rows,
                        det_value_rows.as_deref(),
                    )?);
                } else {
                    let key_rows = k.values[kv_head_idx][start..=query_idx].to_vec();
                    let value_rows = v.values[kv_head_idx][start..=query_idx].to_vec();
                    let det_key_rows = k
                        .acts
                        .as_ref()
                        .map(|heads| heads[kv_head_idx][start..=query_idx].to_vec());
                    let det_value_rows = v
                        .acts
                        .as_ref()
                        .map(|heads| heads[kv_head_idx][start..=query_idx].to_vec());
                    outputs.push(attention_output(
                        &query,
                        &key_rows,
                        det_key_rows.as_deref(),
                        &value_rows,
                        det_value_rows.as_deref(),
                    )?);
                }
            }
            Ok(outputs)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    let mut combined_heads = vec![vec![0.0; layer.num_heads * layer.head_dim]; seq_len];
    let mut combined_head_acts = head_outputs
        .iter()
        .all(|outputs| outputs.iter().all(|output| output.acts.is_some()))
        .then(|| vec![vec![Act::from_bits(0); layer.num_heads * layer.head_dim]; seq_len]);
    for (head_idx, outputs) in head_outputs.into_iter().enumerate() {
        for (query_idx, output) in outputs.into_iter().enumerate() {
            let dst = &mut combined_heads[query_idx]
                [head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
            dst.copy_from_slice(&output.values);
            if let (Some(combined_head_acts), Some(output_acts)) =
                (&mut combined_head_acts, output.acts)
            {
                let dst = &mut combined_head_acts[query_idx]
                    [head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
                dst.copy_from_slice(&output_acts);
            }
        }
    }

    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.o_proj");
        let combined_heads = match combined_head_acts {
            Some(acts) => ActivationSequenceBuffer::from_acts(acts),
            None => ActivationSequenceBuffer::from_values(combined_heads),
        };
        Ok((
            project_linear_sequence_buffer(
                &combined_heads,
                layer.o_proj.as_ref(),
                layer.o_proj_det.as_deref(),
            )?,
            layer_cache,
        ))
    }
}

fn run_causal_attention_decode_buffer(
    input: &ActivationRowBuffer,
    layer: &ResolvedGemma4LayerWeights,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
) -> Result<(ActivationRowBuffer, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_causal_attention_decode attention={:?}",
    //     layer.attention_kind
    // ));
    validate_vector_width(&input.values, layer.hidden_size, "decode attention input")?;

    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }
    if let Some(donor_cache) = donor_cache {
        validate_layer_cache(donor_cache, layer)?;
    } else {
        validate_layer_cache(&cache, layer)?;
    }

    let q_projected =
        project_linear_row_buffer(input, layer.q_proj.as_ref(), layer.q_proj_det.as_deref())?;
    let raw_k =
        project_linear_row_buffer(input, layer.k_proj.as_ref(), layer.k_proj_det.as_deref())?;
    let raw_v = if let Some(v_proj) = &layer.v_proj {
        project_linear_row_buffer(input, v_proj.as_ref(), layer.v_proj_det.as_deref())?
    } else if layer.attention_k_eq_v {
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };

    let mut q = reshape_row_head_buffer(&q_projected, layer.num_heads, layer.head_dim)?;
    let mut k = reshape_row_head_buffer(&raw_k, layer.num_kv_heads, layer.head_dim)?;
    let mut v = reshape_row_head_buffer(&raw_v, layer.num_kv_heads, layer.head_dim)?;

    apply_head_rms_norm_row(
        &mut q,
        &layer.q_norm_weight,
        layer.q_norm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
    )?;
    apply_head_rms_norm_row(
        &mut k,
        &layer.k_norm_weight,
        layer.k_norm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
    )?;
    apply_value_rms_norm_row(&mut v, layer.rms_norm_eps, layer.rms_norm_eps_det)?;
    apply_rope_to_rows(
        &mut q,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base,
        layer.rope_base_det,
        position,
    )?;
    apply_rope_to_rows(
        &mut k,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base,
        layer.rope_base_det,
        position,
    )?;

    let updated_cache = if donor_cache.is_some() {
        cache
    } else {
        append_kv_cache_head_buffer(cache, &k, &v, cache_window)?
    };
    let attention_cache = donor_cache.unwrap_or(&updated_cache);
    let mut combined_heads = vec![0.0; layer.num_heads * layer.head_dim];
    let mut combined_head_acts = Some(vec![Act::from_bits(0); layer.num_heads * layer.head_dim]);
    for head_idx in 0..layer.num_heads {
        let kv_head_idx = head_idx / kv_groups;
        let key_start = attention_window
            .map(|window| attention_cache.current_len().saturating_sub(window))
            .unwrap_or(0);
        let key_rows = attention_cache.keys[kv_head_idx]
            .iter()
            .skip(key_start)
            .cloned()
            .collect::<Vec<_>>();
        let value_rows = attention_cache.values[kv_head_idx]
            .iter()
            .skip(key_start)
            .cloned()
            .collect::<Vec<_>>();
        let det_key_rows = attention_cache.det_key_rows_from(kv_head_idx, key_start);
        let det_value_rows = attention_cache.det_value_rows_from(kv_head_idx, key_start);
        let output = attention_output(
            &q.row(head_idx),
            &key_rows,
            det_key_rows.as_deref(),
            &value_rows,
            det_value_rows.as_deref(),
        )?;
        let dst = &mut combined_heads[head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
        dst.copy_from_slice(&output.values);
        if let (Some(combined_head_acts), Some(output_acts)) =
            (&mut combined_head_acts, output.acts)
        {
            let dst =
                &mut combined_head_acts[head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
            dst.copy_from_slice(&output_acts);
        }
    }

    let combined_heads = match combined_head_acts {
        Some(acts) => ActivationRowBuffer::from_acts(acts),
        None => ActivationRowBuffer::from_values(combined_heads),
    };
    Ok((
        project_linear_row_buffer(
            &combined_heads,
            layer.o_proj.as_ref(),
            layer.o_proj_det.as_deref(),
        )?,
        updated_cache,
    ))
}

fn validate_sequence_width(sequence: &[Vec<f32>], width: usize, label: &str) -> Result<()> {
    if let Some((row_idx, row)) = sequence
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != width)
    {
        bail!(
            "{label} row {row_idx} has width {}, expected {width}",
            row.len()
        );
    }
    Ok(())
}

fn validate_vector_width(vector: &[f32], width: usize, label: &str) -> Result<()> {
    if vector.len() != width {
        bail!("{label} width mismatch: {} vs {width}", vector.len());
    }
    Ok(())
}

fn validate_layer_cache(cache: &LayerKvCache, layer: &ResolvedGemma4LayerWeights) -> Result<()> {
    if cache.keys.len() != layer.num_kv_heads || cache.values.len() != layer.num_kv_heads {
        bail!(
            "layer cache head count mismatch: keys {} values {} expected {}",
            cache.keys.len(),
            cache.values.len(),
            layer.num_kv_heads
        );
    }

    for (head_idx, (keys, values)) in cache.keys.iter().zip(&cache.values).enumerate() {
        if keys.len() != values.len() {
            bail!(
                "layer cache sequence length mismatch at head {head_idx}: {} vs {}",
                keys.len(),
                values.len()
            );
        }
        for row in keys {
            validate_vector_width(row, layer.head_dim, "cached key row")?;
        }
        for row in values {
            validate_vector_width(row, layer.head_dim, "cached value row")?;
        }
    }

    Ok(())
}

#[derive(Clone)]
struct ActivationSequenceBuffer {
    values: Vec<Vec<f32>>,
    acts: Option<Vec<Vec<Act>>>,
}

impl ActivationSequenceBuffer {
    fn from_values(values: Vec<Vec<f32>>) -> Self {
        Self { values, acts: None }
    }

    fn from_internal(internal: InternalActivationSequence) -> Self {
        match internal.det_values() {
            Some(det_values) => Self::from_acts(det_values.to_vec()),
            None => Self::from_values(internal.clone_f32()),
        }
    }

    fn from_acts(acts: Vec<Vec<Act>>) -> Self {
        let values = acts
            .iter()
            .map(|row| row.iter().copied().map(act_to_f32).collect())
            .collect();
        Self {
            values,
            acts: Some(acts),
        }
    }

    fn into_internal(self) -> InternalActivationSequence {
        match self.acts {
            Some(det_values) => InternalActivationSequence::from_det_values(det_values),
            None => InternalActivationSequence::from_values(self.values),
        }
    }
}

#[derive(Clone)]
struct ActivationRowBuffer {
    values: Vec<f32>,
    acts: Option<Vec<Act>>,
}

impl ActivationRowBuffer {
    fn from_values(values: Vec<f32>) -> Self {
        Self { values, acts: None }
    }

    fn from_internal(internal: InternalActivationRow) -> Self {
        match internal.det_values() {
            Some(det_values) => Self::from_acts(det_values.to_vec()),
            None => Self::from_values(internal.clone_f32()),
        }
    }

    fn from_acts(acts: Vec<Act>) -> Self {
        let values = acts.iter().copied().map(act_to_f32).collect();
        Self {
            values,
            acts: Some(acts),
        }
    }

    fn into_internal(self) -> InternalActivationRow {
        match self.acts {
            Some(det_values) => InternalActivationRow::from_det_values(det_values),
            None => InternalActivationRow::from_values(self.values),
        }
    }
}

#[derive(Clone, Debug)]
struct AttentionHeadSequenceBuffer {
    values: Vec<Vec<Vec<f32>>>,
    acts: Option<Vec<Vec<Vec<Act>>>>,
}

impl AttentionHeadSequenceBuffer {
    fn from_values(values: Vec<Vec<Vec<f32>>>) -> Self {
        Self { values, acts: None }
    }

    fn from_acts(acts: Vec<Vec<Vec<Act>>>) -> Self {
        let values = acts
            .iter()
            .map(|head| {
                head.iter()
                    .map(|row| row.iter().copied().map(act_to_f32).collect())
                    .collect()
            })
            .collect();
        Self {
            values,
            acts: Some(acts),
        }
    }

    fn row(&self, head_idx: usize, seq_idx: usize) -> ActivationRowBuffer {
        let values = self.values[head_idx][seq_idx].clone();
        let acts = self
            .acts
            .as_ref()
            .map(|heads| heads[head_idx][seq_idx].clone());
        ActivationRowBuffer { values, acts }
    }
}

#[derive(Clone, Debug)]
struct AttentionHeadRowBuffer {
    values: Vec<Vec<f32>>,
    acts: Option<Vec<Vec<Act>>>,
}

impl AttentionHeadRowBuffer {
    fn from_values(values: Vec<Vec<f32>>) -> Self {
        Self { values, acts: None }
    }

    fn from_acts(acts: Vec<Vec<Act>>) -> Self {
        let values = acts
            .iter()
            .map(|row| row.iter().copied().map(act_to_f32).collect())
            .collect();
        Self {
            values,
            acts: Some(acts),
        }
    }

    fn row(&self, head_idx: usize) -> ActivationRowBuffer {
        let values = self.values[head_idx].clone();
        let acts = self.acts.as_ref().map(|heads| heads[head_idx].clone());
        ActivationRowBuffer { values, acts }
    }
}

fn activation_sequence_from_buffer(buffer: ActivationSequenceBuffer) -> ActivationSequence {
    let activations_sha256 = build_activation_commitment(&buffer.values);
    let det_activations_sha256 = buffer
        .acts
        .as_ref()
        .map(|acts| build_det_activation_commitment(acts));
    let mut activation_sequence =
        ActivationSequence::from_internal(buffer.into_internal(), activations_sha256);
    activation_sequence.det_activations_sha256 = det_activations_sha256;
    activation_sequence
}

fn add_act_sequences(lhs: &[Vec<Act>], rhs: &[Vec<Act>]) -> Result<Vec<Vec<Act>>> {
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    lhs.iter()
        .zip(rhs)
        .map(|(lhs_row, rhs_row)| add_act_rows(lhs_row, rhs_row))
        .collect()
}

fn add_act_rows(lhs: &[Act], rhs: &[Act]) -> Result<Vec<Act>> {
    if lhs.len() != rhs.len() {
        bail!("row width mismatch: {} vs {}", lhs.len(), rhs.len());
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(lhs_value, rhs_value)| add_sat(*lhs_value, *rhs_value))
        .collect())
}

fn mul_act_sequences(lhs: &[Vec<Act>], rhs: &[Vec<Act>]) -> Result<Vec<Vec<Act>>> {
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    lhs.iter()
        .zip(rhs)
        .map(|(lhs_row, rhs_row)| mul_act_rows(lhs_row, rhs_row))
        .collect()
}

fn mul_act_rows(lhs: &[Act], rhs: &[Act]) -> Result<Vec<Act>> {
    if lhs.len() != rhs.len() {
        bail!("row width mismatch: {} vs {}", lhs.len(), rhs.len());
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(lhs_value, rhs_value)| mul_sat(*lhs_value, *rhs_value))
        .collect())
}

fn scale_act_sequences(values: &[Vec<Act>], scalar: Act) -> Vec<Vec<Act>> {
    values
        .iter()
        .map(|row| scale_act_rows(row, scalar))
        .collect()
}

fn scale_act_rows(values: &[Act], scalar: Act) -> Vec<Act> {
    values
        .iter()
        .copied()
        .map(|value| scale_act(value, scalar))
        .collect()
}

fn sequence_buffer_acts(buffer: &ActivationSequenceBuffer) -> Result<Vec<Vec<Act>>> {
    buffer
        .acts
        .clone()
        .ok_or_else(|| anyhow!("deterministic sequence operation requires canonical Act rows"))
}

fn row_buffer_acts(buffer: &ActivationRowBuffer) -> Result<Vec<Act>> {
    buffer
        .acts
        .clone()
        .ok_or_else(|| anyhow!("deterministic row operation requires canonical Act values"))
}

fn head_sequence_buffer_acts(buffer: &AttentionHeadSequenceBuffer) -> Result<Vec<Vec<Vec<Act>>>> {
    buffer
        .acts
        .clone()
        .ok_or_else(|| anyhow!("deterministic KV cache construction requires canonical head rows"))
}

fn head_row_buffer_acts(buffer: &AttentionHeadRowBuffer) -> Result<Vec<Vec<Act>>> {
    buffer
        .acts
        .clone()
        .ok_or_else(|| anyhow!("deterministic KV cache append requires canonical head rows"))
}

fn add_sequence_buffers(
    lhs: &ActivationSequenceBuffer,
    rhs: &ActivationSequenceBuffer,
) -> Result<ActivationSequenceBuffer> {
    let lhs_acts = sequence_buffer_acts(lhs)?;
    let rhs_acts = sequence_buffer_acts(rhs)?;
    Ok(ActivationSequenceBuffer::from_acts(add_act_sequences(
        &lhs_acts, &rhs_acts,
    )?))
}

fn add_row_buffers(
    lhs: &ActivationRowBuffer,
    rhs: &ActivationRowBuffer,
) -> Result<ActivationRowBuffer> {
    let lhs_acts = row_buffer_acts(lhs)?;
    let rhs_acts = row_buffer_acts(rhs)?;
    Ok(ActivationRowBuffer::from_acts(add_act_rows(
        &lhs_acts, &rhs_acts,
    )?))
}

fn mul_sequence_buffers(
    lhs: &ActivationSequenceBuffer,
    rhs: &ActivationSequenceBuffer,
) -> Result<ActivationSequenceBuffer> {
    let lhs_acts = sequence_buffer_acts(lhs)?;
    let rhs_acts = sequence_buffer_acts(rhs)?;
    Ok(ActivationSequenceBuffer::from_acts(mul_act_sequences(
        &lhs_acts, &rhs_acts,
    )?))
}

fn mul_row_buffers(
    lhs: &ActivationRowBuffer,
    rhs: &ActivationRowBuffer,
) -> Result<ActivationRowBuffer> {
    let lhs_acts = row_buffer_acts(lhs)?;
    let rhs_acts = row_buffer_acts(rhs)?;
    Ok(ActivationRowBuffer::from_acts(mul_act_rows(
        &lhs_acts, &rhs_acts,
    )?))
}

fn scale_sequence_buffer(
    values: &ActivationSequenceBuffer,
    scalar_det: Option<Act>,
) -> Result<ActivationSequenceBuffer> {
    let scalar_det = scalar_det
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    Ok(ActivationSequenceBuffer::from_acts(scale_act_sequences(
        &sequence_buffer_acts(values)?,
        scalar_det,
    )))
}

fn scale_row_buffer(
    values: &ActivationRowBuffer,
    scalar_det: Option<Act>,
) -> Result<ActivationRowBuffer> {
    let scalar_det = scalar_det
        .ok_or_else(|| anyhow!("deterministic row scaling requires canonical Act scalar"))?;
    Ok(ActivationRowBuffer::from_acts(scale_act_rows(
        &row_buffer_acts(values)?,
        scalar_det,
    )))
}

fn project_linear_sequence_buffer(
    inputs: &ActivationSequenceBuffer,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<ActivationSequenceBuffer> {
    validate_sequence_width(&inputs.values, weight.cols, "linear input")?;
    match det_weight {
        Some(det_weight) => Ok(ActivationSequenceBuffer::from_acts(
            det_linear_sequence_acts_from_acts(&sequence_buffer_acts(inputs)?, det_weight)?,
        )),
        None => bail!("deterministic linear sequence projection requires canonical det_weight"),
    }
}

fn project_linear_row_buffer(
    input: &ActivationRowBuffer,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<ActivationRowBuffer> {
    validate_vector_width(&input.values, weight.cols, "linear input")?;
    match det_weight {
        Some(det_weight) => Ok(ActivationRowBuffer::from_acts(
            det_linear_row_acts_from_acts(&row_buffer_acts(input)?, det_weight)?,
        )),
        None => bail!("deterministic linear row projection requires canonical det_weight"),
    }
}

fn det_linear_sequence_acts_from_acts(
    quantized_inputs: &[Vec<Act>],
    weight: &DetNumMatrix,
) -> Result<Vec<Vec<Act>>> {
    if let Some((row_idx, row)) = quantized_inputs
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != weight.cols)
    {
        bail!(
            "deterministic linear input row {row_idx} has width {}, expected {}",
            row.len(),
            weight.cols
        );
    }

    quantized_inputs
        .par_iter()
        .map(|input| det_linear_row_acts_from_acts(input, weight))
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

fn det_linear_row_acts_from_acts(
    quantized_input: &[Act],
    weight: &DetNumMatrix,
) -> Result<Vec<Act>> {
    if quantized_input.len() != weight.cols {
        bail!(
            "deterministic linear input width mismatch: {} vs {}",
            quantized_input.len(),
            weight.cols
        );
    }

    // Canonical scalar reference: weights are widened (sign-extended) on
    // read, so i16 storage (detwgt v2) produces bit-identical products.
    let mut output = Vec::with_capacity(weight.rows);
    match weight.values.payload() {
        crate::shared::model::transformer::WgtPayload::I32(weight_values) => {
            for row_idx in 0..weight.rows {
                let row_offset = row_idx * weight.cols;
                let mut acc_bits = 0_i64;
                for (col_idx, act) in quantized_input.iter().enumerate() {
                    acc_bits =
                        mac_bits(acc_bits, act.to_bits(), weight_values[row_offset + col_idx]);
                }
                output.push(requantize(Acc::from_bits(acc_bits)));
            }
        }
        crate::shared::model::transformer::WgtPayload::I16(weight_values) => {
            for row_idx in 0..weight.rows {
                let row_offset = row_idx * weight.cols;
                let mut acc_bits = 0_i64;
                for (col_idx, act) in quantized_input.iter().enumerate() {
                    acc_bits = mac_bits(
                        acc_bits,
                        act.to_bits(),
                        i32::from(weight_values[row_offset + col_idx]),
                    );
                }
                output.push(requantize(Acc::from_bits(acc_bits)));
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
fn det_linear_sequence(inputs: &[Vec<f32>], weight: &DetNumMatrix) -> Result<Vec<Vec<f32>>> {
    validate_sequence_width(inputs, weight.cols, "deterministic linear input")?;
    inputs
        .par_iter()
        .map(|input| det_linear_row(input, weight))
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
fn det_linear_row(input: &[f32], weight: &DetNumMatrix) -> Result<Vec<f32>> {
    validate_vector_width(input, weight.cols, "deterministic linear input")?;

    let quantized_input = input
        .iter()
        .copied()
        .map(crate::shared::numerics::det_num::f32_to_act)
        .collect::<Vec<_>>();
    det_linear_row_from_acts(&quantized_input, weight)
}

#[cfg(test)]
fn det_linear_row_from_acts(quantized_input: &[Act], weight: &DetNumMatrix) -> Result<Vec<f32>> {
    det_linear_row_acts_from_acts(quantized_input, weight)
        .map(|acts| acts.into_iter().map(act_to_f32).collect())
}

fn apply_rms_norm_to_sequence_buffer(
    inputs: &ActivationSequenceBuffer,
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
) -> Result<ActivationSequenceBuffer> {
    sequence_buffer_acts(inputs)?
        .into_par_iter()
        .map(|row| {
            apply_rms_norm_row_buffer(
                &ActivationRowBuffer::from_acts(row),
                weight,
                weight_det,
                eps,
                eps_det,
            )
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .map(|rows| {
            ActivationSequenceBuffer::from_acts(
                rows.into_iter()
                    .map(|row| row.acts.expect("deterministic RMS norm row"))
                    .collect(),
            )
        })
}

fn apply_rms_norm_row_buffer(
    input: &ActivationRowBuffer,
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    _eps: f32,
    eps_det: Option<Acc>,
) -> Result<ActivationRowBuffer> {
    if input.values.len() != weight.len() {
        bail!(
            "rms norm width mismatch: {} vs {}",
            input.values.len(),
            weight.len()
        );
    }

    let quantized_input = row_buffer_acts(input)?;
    let quantized_weight = weight_det
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Wgt norm weights"))?;
    let quantized_eps =
        eps_det.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    Ok(ActivationRowBuffer::from_acts(det_rms_norm(
        &quantized_input,
        quantized_weight,
        quantized_eps,
    )))
}

fn reshape_sequence_heads(
    projected: &[Vec<f32>],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let expected_width = num_heads * head_dim;
    validate_sequence_width(projected, expected_width, "projected attention states")?;

    let mut heads = vec![vec![vec![0.0; head_dim]; projected.len()]; num_heads];
    for (seq_idx, row) in projected.iter().enumerate() {
        for head_idx in 0..num_heads {
            let start = head_idx * head_dim;
            let end = start + head_dim;
            heads[head_idx][seq_idx].copy_from_slice(&row[start..end]);
        }
    }
    Ok(heads)
}

fn reshape_row_heads(
    projected: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected_width = num_heads * head_dim;
    validate_vector_width(projected, expected_width, "projected attention state")?;

    let mut heads = vec![vec![0.0; head_dim]; num_heads];
    for (head_idx, head) in heads.iter_mut().enumerate() {
        let start = head_idx * head_dim;
        let end = start + head_dim;
        head.copy_from_slice(&projected[start..end]);
    }
    Ok(heads)
}

fn reshape_sequence_head_buffer(
    projected: &ActivationSequenceBuffer,
    num_heads: usize,
    head_dim: usize,
) -> Result<AttentionHeadSequenceBuffer> {
    let values = reshape_sequence_heads(&projected.values, num_heads, head_dim)?;
    let Some(acts) = &projected.acts else {
        return Ok(AttentionHeadSequenceBuffer::from_values(values));
    };

    let expected_width = num_heads * head_dim;
    if let Some((row_idx, row)) = acts
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != expected_width)
    {
        bail!(
            "projected attention state acts row {row_idx} has width {}, expected {expected_width}",
            row.len()
        );
    }

    let mut head_acts = vec![vec![vec![Act::from_bits(0); head_dim]; acts.len()]; num_heads];
    for (seq_idx, row) in acts.iter().enumerate() {
        for (head_idx, head) in head_acts.iter_mut().enumerate() {
            let start = head_idx * head_dim;
            let end = start + head_dim;
            head[seq_idx].copy_from_slice(&row[start..end]);
        }
    }
    Ok(AttentionHeadSequenceBuffer {
        values,
        acts: Some(head_acts),
    })
}

fn reshape_row_head_buffer(
    projected: &ActivationRowBuffer,
    num_heads: usize,
    head_dim: usize,
) -> Result<AttentionHeadRowBuffer> {
    let values = reshape_row_heads(&projected.values, num_heads, head_dim)?;
    let Some(acts) = &projected.acts else {
        return Ok(AttentionHeadRowBuffer::from_values(values));
    };

    let expected_width = num_heads * head_dim;
    if acts.len() != expected_width {
        bail!(
            "projected attention state acts width mismatch: {} vs {expected_width}",
            acts.len()
        );
    }

    let mut head_acts = vec![vec![Act::from_bits(0); head_dim]; num_heads];
    for (head_idx, head) in head_acts.iter_mut().enumerate() {
        let start = head_idx * head_dim;
        let end = start + head_dim;
        head.copy_from_slice(&acts[start..end]);
    }
    Ok(AttentionHeadRowBuffer {
        values,
        acts: Some(head_acts),
    })
}

fn apply_head_rms_norm(
    heads: &mut AttentionHeadSequenceBuffer,
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
) -> Result<()> {
    let acts = head_sequence_buffer_acts(heads)?
        .into_par_iter()
        .map(|head| {
            head.into_iter()
                .map(|row| {
                    apply_rms_norm_row_buffer(
                        &ActivationRowBuffer::from_acts(row),
                        weight,
                        weight_det,
                        eps,
                        eps_det,
                    )
                    .map(|row| row.acts.expect("deterministic head RMSNorm row"))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    *heads = AttentionHeadSequenceBuffer::from_acts(acts);
    Ok(())
}

fn apply_head_rms_norm_row(
    heads: &mut AttentionHeadRowBuffer,
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
) -> Result<()> {
    let acts = head_row_buffer_acts(heads)?
        .into_iter()
        .map(|row| {
            apply_rms_norm_row_buffer(
                &ActivationRowBuffer::from_acts(row),
                weight,
                weight_det,
                eps,
                eps_det,
            )
            .map(|row| row.acts.expect("deterministic head RMSNorm row"))
        })
        .collect::<Result<Vec<_>>>()?;
    *heads = AttentionHeadRowBuffer::from_acts(acts);
    Ok(())
}

fn apply_value_rms_norm(
    heads: &mut AttentionHeadSequenceBuffer,
    _eps: f32,
    eps_det: Option<Acc>,
) -> Result<()> {
    let eps_det = eps_det
        .ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    let acts = head_sequence_buffer_acts(heads)?
        .into_par_iter()
        .map(|head| {
            head.into_iter()
                .map(|row| det_value_rms_norm(&row, eps_det))
                .collect::<Vec<_>>()
        })
        .collect();
    *heads = AttentionHeadSequenceBuffer::from_acts(acts);
    Ok(())
}

fn apply_value_rms_norm_row(
    heads: &mut AttentionHeadRowBuffer,
    _eps: f32,
    eps_det: Option<Acc>,
) -> Result<()> {
    let eps_det = eps_det
        .ok_or_else(|| anyhow!("deterministic value RMSNorm requires canonical Acc epsilon"))?;
    let acts = head_row_buffer_acts(heads)?
        .into_iter()
        .map(|row| det_value_rms_norm(&row, eps_det))
        .collect();
    *heads = AttentionHeadRowBuffer::from_acts(acts);
    Ok(())
}

fn apply_rope(
    heads: &mut AttentionHeadSequenceBuffer,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: f32,
    base_det: Option<Acc>,
) -> Result<()> {
    apply_rope_with_offset(heads, rotary_dim, freq_base_dim, base, base_det, 0)
}

fn apply_rope_with_offset(
    heads: &mut AttentionHeadSequenceBuffer,
    rotary_dim: usize,
    freq_base_dim: usize,
    _base: f32,
    base_det: Option<Acc>,
    position_offset: usize,
) -> Result<()> {
    if rotary_dim == 0 {
        return Ok(());
    }

    let quantized_base =
        base_det.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    let acts = head_sequence_buffer_acts(heads)?
        .into_iter()
        .map(|head| {
            head.into_iter()
                .enumerate()
                .map(|(position, row)| {
                    det_rope_rotate_pairs(
                        &row,
                        rotary_dim,
                        freq_base_dim,
                        quantized_base,
                        position_offset + position,
                    )
                })
                .collect()
        })
        .collect();
    *heads = AttentionHeadSequenceBuffer::from_acts(acts);
    Ok(())
}

fn apply_rope_to_rows(
    heads: &mut AttentionHeadRowBuffer,
    rotary_dim: usize,
    freq_base_dim: usize,
    _base: f32,
    base_det: Option<Acc>,
    position: usize,
) -> Result<()> {
    if rotary_dim == 0 {
        return Ok(());
    }

    let quantized_base =
        base_det.ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
    let acts = head_row_buffer_acts(heads)?
        .into_iter()
        .map(|row| det_rope_rotate_pairs(&row, rotary_dim, freq_base_dim, quantized_base, position))
        .collect();
    *heads = AttentionHeadRowBuffer::from_acts(acts);
    Ok(())
}

fn attention_output(
    query: &ActivationRowBuffer,
    _key_rows: &[Vec<f32>],
    det_key_rows: Option<&[Vec<Act>]>,
    _value_rows: &[Vec<f32>],
    det_value_rows: Option<&[Vec<Act>]>,
) -> Result<ActivationRowBuffer> {
    let quantized_query = row_buffer_acts(query)?;
    let quantized_keys = det_key_rows
        .map(|rows| rows.to_vec())
        .ok_or_else(|| anyhow!("deterministic attention requires canonical key cache rows"))?;
    let logits = quantized_keys
        .iter()
        .map(|key_row| det_attention_score(&quantized_query, key_row))
        .collect::<Vec<_>>();
    let weights = det_attention_softmax(&logits);
    let quantized_values = det_value_rows
        .map(|rows| rows.to_vec())
        .ok_or_else(|| anyhow!("deterministic attention requires canonical value cache rows"))?;
    Ok(ActivationRowBuffer::from_acts(det_attention_weighted_sum(
        &weights,
        &quantized_values,
    )))
}

fn build_layer_kv_cache(
    keys: &AttentionHeadSequenceBuffer,
    values: &AttentionHeadSequenceBuffer,
    sliding_window: Option<usize>,
) -> Result<LayerKvCache> {
    let retained = sliding_window.map_or(0, |window| {
        keys.values
            .first()
            .map_or(0, |head| head.len().saturating_sub(window))
    });
    Ok(LayerKvCache::from_det_heads(
        head_sequence_buffer_acts(keys)?
            .into_iter()
            .map(|head| head[retained..].iter().cloned().collect())
            .collect(),
        head_sequence_buffer_acts(values)?
            .into_iter()
            .map(|head| head[retained..].iter().cloned().collect())
            .collect(),
    ))
}

fn append_kv_cache_head_buffer(
    mut cache: LayerKvCache,
    new_keys: &AttentionHeadRowBuffer,
    new_values: &AttentionHeadRowBuffer,
    sliding_window: Option<usize>,
) -> Result<LayerKvCache> {
    if new_keys.values.len() != cache.keys.len() || new_values.values.len() != cache.values.len() {
        bail!(
            "layer cache append head count mismatch: cache {} keys {} values {}",
            cache.keys.len(),
            new_keys.values.len(),
            new_values.values.len()
        );
    }

    let new_key_acts = head_row_buffer_acts(new_keys)?;
    let new_value_acts = head_row_buffer_acts(new_values)?;
    let mut det = match cache.det.take() {
        Some(det) => det,
        None if cache.keys.iter().all(|head| head.is_empty()) => {
            crate::shared::numerics::det_tensor::DetKvCacheData::new(
                cache.keys.len(),
                new_key_acts.first().map(Vec::len).unwrap_or(0),
            )
        }
        None => bail!("deterministic decode cache append requires canonical key rows"),
    };
    det.append_step_nested(&new_key_acts, &new_value_acts, sliding_window)?;
    cache.det = Some(det);

    Ok(cache)
}

fn apply_gelu_to_sequence_buffer(
    inputs: &ActivationSequenceBuffer,
) -> Result<ActivationSequenceBuffer> {
    Ok(ActivationSequenceBuffer::from_acts(
        sequence_buffer_acts(inputs)?
            .iter()
            .map(|row| apply_det_gelu(row))
            .collect(),
    ))
}

fn apply_gelu_to_row_buffer(inputs: &ActivationRowBuffer) -> Result<ActivationRowBuffer> {
    Ok(ActivationRowBuffer::from_acts(apply_det_gelu(
        &row_buffer_acts(inputs)?,
    )))
}

fn apply_det_gelu(inputs: &[Act]) -> Vec<Act> {
    inputs.iter().copied().map(gelu_pytorch_tanh_act).collect()
}

#[cfg(test)]
mod tests;
