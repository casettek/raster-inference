use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use super::types::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4Phase2Model, Gemma4PleGlobalWeights,
    Gemma4PrefillPleInputs, MatrixF32, PrefillLogits,
};
use crate::trace::{trace_event, trace_scope};

pub fn embed_input_tokens(
    token_ids: &[u32],
    embedding_table: &EmbeddingTable,
) -> Result<EmbeddedTokenSequence> {
    let _trace = trace_scope("phase2.embed_input_tokens");
    if token_ids.is_empty() {
        bail!("phase 2 embedding requires at least one token id");
    }

    if embedding_table.rows.is_empty() {
        bail!("phase 2 embedding requires a non-empty embedding table");
    }

    let hidden_size = embedding_table.rows[0].len();
    if hidden_size == 0 {
        bail!("phase 2 embedding rows must have non-zero width");
    }

    if let Some((row_idx, row)) = embedding_table
        .rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != hidden_size)
    {
        bail!(
            "phase 2 embedding table row {row_idx} has width {}, expected {hidden_size}",
            row.len()
        );
    }

    let mut activations = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        let row = embedding_table.rows.get(row_idx).ok_or_else(|| {
            anyhow::anyhow!("token id {token_id} is out of bounds for embedding table")
        })?;
        let mut activation = row.clone();
        if embedding_table.scale != 1.0 {
            for value in &mut activation {
                *value *= embedding_table.scale;
            }
        }
        activations.push(activation);
    }

    let activations_sha256 = build_phase2_commitment(&activations);

    Ok(ActivationSequence {
        activations,
        activations_sha256,
    })
}

pub fn compute_prefill_ple_inputs(
    token_ids: &[u32],
    input_activations: &[Vec<f32>],
    layers: &[Gemma4LayerWeights],
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps: f32,
) -> Result<Gemma4PrefillPleInputs> {
    let _trace = trace_scope("phase2.compute_prefill_ple_inputs");
    if input_activations.is_empty() {
        bail!("phase 2 PLE computation requires at least one activation row");
    }
    if layers.is_empty() {
        bail!("phase 2 PLE computation requires at least one layer");
    }
    let hidden_size = layers[0].hidden_size;
    validate_sequence_width(input_activations, hidden_size, "input activations")?;

    if token_ids.len() != input_activations.len() {
        bail!(
            "phase 2 PLE computation requires token ids and activations to have matching lengths"
        );
    }
    if ple_global.token_embeddings.len() != layers.len() {
        bail!(
            "phase 2 PLE token embedding slice count mismatch: {} vs {}",
            ple_global.token_embeddings.len(),
            layers.len()
        );
    }
    if ple_global.model_projections.len() != layers.len() {
        bail!(
            "phase 2 PLE model projection slice count mismatch: {} vs {}",
            ple_global.model_projections.len(),
            layers.len()
        );
    }

    let mut per_layer_inputs = Vec::with_capacity(layers.len());
    for (layer_idx, layer) in layers.iter().enumerate() {
        trace_event(format!("phase2.compute_prefill_ple_inputs layer={layer_idx}"));
        if layer.ple.is_none() {
            per_layer_inputs.push(None);
            continue;
        }

        let token_embedding = &ple_global.token_embeddings[layer_idx];
        let model_projection = &ple_global.model_projections[layer_idx];

        let mut embedded = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
            embedded.push(
                row_from_matrix(token_embedding, row_idx)?
                    .into_iter()
                    .map(|value| value * ple_global.embedding_scale)
                    .collect::<Vec<_>>(),
            );
        }

        let mut projected = linear_sequence(input_activations, model_projection)?;
        for row in &mut projected {
            for value in row {
                *value *= ple_global.projection_scalar;
            }
        }
        let projected = apply_rms_norm_to_sequence(
            &projected,
            &ple_global.projection_norm_weight,
            rms_norm_eps,
        )?;

        let mut combined = add_sequences(&embedded, &projected)?;
        for row in &mut combined {
            for value in row {
                *value *= ple_global.input_scale;
            }
        }
        per_layer_inputs.push(Some(combined));
    }

    Ok(Gemma4PrefillPleInputs { per_layer_inputs })
}

pub fn run_gemma4_layer(
    input_activations: &[Vec<f32>],
    layer: &Gemma4LayerWeights,
    per_layer_input: Option<&[Vec<f32>]>,
) -> Result<ActivationSequence> {
    let _trace = trace_scope(format!(
        "phase2.run_gemma4_layer attention={:?}",
        layer.attention_kind
    ));
    if input_activations.is_empty() {
        bail!("phase 2 layer execution requires at least one activation row");
    }
    validate_sequence_width(input_activations, layer.hidden_size, "input activations")?;
    if let Some(per_layer_input) = per_layer_input {
        validate_sequence_width(
            per_layer_input,
            layer
                .ple
                .as_ref()
                .ok_or_else(|| anyhow!("phase 2 layer received PLE inputs without PLE weights"))?
                .input_gate
                .rows,
            "per-layer inputs",
        )?;
        if per_layer_input.len() != input_activations.len() {
            bail!(
                "phase 2 layer execution requires per-layer inputs and activations to have matching lengths"
            );
        }
    }

    let mut xs = {
        let _trace = trace_scope("phase2.run_gemma4_layer.clone_input");
        input_activations.to_vec()
    };

    let residual = {
        let _trace = trace_scope("phase2.run_gemma4_layer.attention.clone_residual");
        xs.clone()
    };
    let normed = {
        let _trace = trace_scope("phase2.run_gemma4_layer.attention.input_rms_norm");
        apply_rms_norm_to_sequence(&xs, &layer.input_layernorm_weight, layer.rms_norm_eps)?
    };
    let attn_out = {
        let _trace = trace_scope("phase2.run_gemma4_layer.attention.core");
        run_attention_for_layer(&normed, layer)?
    };
    let attn_out = {
        let _trace = trace_scope("phase2.run_gemma4_layer.attention.post_rms_norm");
        apply_rms_norm_to_sequence(
            &attn_out,
            &layer.post_attention_layernorm_weight,
            layer.rms_norm_eps,
        )?
    };
    xs = {
        let _trace = trace_scope("phase2.run_gemma4_layer.attention.residual_add");
        add_sequences(&residual, &attn_out)?
    };

    let residual = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.clone_residual");
        xs.clone()
    };
    let normed = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.pre_rms_norm");
        apply_rms_norm_to_sequence(&xs, &layer.pre_feedforward_layernorm_weight, layer.rms_norm_eps)?
    };
    let gate = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.gate_proj_gelu");
        apply_gelu_to_sequence(&linear_sequence(&normed, &layer.gate_proj)?)
    };
    let up = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.up_proj");
        linear_sequence(&normed, &layer.up_proj)?
    };
    let ff_hidden = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.hidden_mul");
        elementwise_mul_sequences(&gate, &up)?
    };
    let ff_out = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.down_proj");
        linear_sequence(&ff_hidden, &layer.down_proj)?
    };
    let ff_out = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.post_rms_norm");
        apply_rms_norm_to_sequence(
            &ff_out,
            &layer.post_feedforward_layernorm_weight,
            layer.rms_norm_eps,
        )?
    };
    xs = {
        let _trace = trace_scope("phase2.run_gemma4_layer.mlp.residual_add");
        add_sequences(&residual, &ff_out)?
    };

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = {
            let _trace = trace_scope("phase2.run_gemma4_layer.ple.clone_residual");
            xs.clone()
        };
        let gated = {
            let _trace = trace_scope("phase2.run_gemma4_layer.ple.input_gate_gelu");
            apply_gelu_to_sequence(&linear_sequence(&xs, &ple.input_gate)?)
        };
        let gated = {
            let _trace = trace_scope("phase2.run_gemma4_layer.ple.input_mul");
            elementwise_mul_sequences(&gated, per_layer_input)?
        };
        let projected = {
            let _trace = trace_scope("phase2.run_gemma4_layer.ple.layer_projection");
            linear_sequence(&gated, &ple.layer_projection)?
        };
        let projected = {
            let _trace = trace_scope("phase2.run_gemma4_layer.ple.post_rms_norm");
            apply_rms_norm_to_sequence(
                &projected,
                &ple.post_input_norm_weight,
                layer.rms_norm_eps,
            )?
        };
        xs = {
            let _trace = trace_scope("phase2.run_gemma4_layer.ple.residual_add");
            add_sequences(&residual, &projected)?
        };
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        let _trace = trace_scope("phase2.run_gemma4_layer.layer_scalar");
        for row in &mut xs {
            for value in row {
                *value *= layer_scalar;
            }
        }
    }

    Ok(ActivationSequence {
        activations_sha256: {
            let _trace = trace_scope("phase2.run_gemma4_layer.commitment_hash");
            build_phase2_commitment(&xs)
        },
        activations: xs,
    })
}

pub fn run_text_layers_prefill(
    input_activations: &[Vec<f32>],
    model: &Gemma4Phase2Model,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<ActivationSequence> {
    let _trace = trace_scope("phase2.run_text_layers_prefill");
    if model.layers.is_empty() {
        bail!("phase 2 prefill requires at least one layer");
    }

    let mut xs = input_activations.to_vec();
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        trace_event(format!(
            "phase2.run_text_layers_prefill layer={layer_idx} attention={:?}",
            layer.attention_kind
        ));
        let per_layer_input = ple_inputs
            .and_then(|inputs| inputs.per_layer_inputs.get(layer_idx))
            .and_then(|input| input.as_deref());
        xs = run_gemma4_layer(&xs, layer, per_layer_input)?.activations;
    }

    Ok(ActivationSequence {
        activations_sha256: build_phase2_commitment(&xs),
        activations: xs,
    })
}

pub fn apply_final_norm(
    input_activations: &[Vec<f32>],
    weight: &[f32],
    eps: f32,
) -> Result<ActivationSequence> {
    let _trace = trace_scope("phase2.apply_final_norm");
    let activations = apply_rms_norm_to_sequence(input_activations, weight, eps)?;
    Ok(ActivationSequence {
        activations_sha256: build_phase2_commitment(&activations),
        activations,
    })
}

pub fn select_final_position(input_activations: &[Vec<f32>]) -> Result<Vec<f32>> {
    let _trace = trace_scope("phase2.select_final_position");
    input_activations
        .last()
        .cloned()
        .ok_or_else(|| anyhow!("phase 2 final-position selection requires at least one activation row"))
}

pub fn project_to_logits(
    last_hidden_state: &[f32],
    projection: &Gemma4LogitsProjection,
) -> Result<Vec<f32>> {
    let _trace = trace_scope("phase2.project_to_logits");
    let weight = match projection {
        Gemma4LogitsProjection::UntiedLmHead(weight)
        | Gemma4LogitsProjection::TiedEmbedding(weight) => weight,
    };
    validate_vector_width(last_hidden_state, weight.cols, "logits projection input")?;

    let mut logits = vec![0.0; weight.rows];
    for (row_idx, logit) in logits.iter_mut().enumerate() {
        let row_offset = row_idx * weight.cols;
        let mut sum = 0.0;
        for col_idx in 0..weight.cols {
            sum += last_hidden_state[col_idx] * weight.values[row_offset + col_idx];
        }
        *logit = sum;
    }
    Ok(logits)
}

pub fn apply_final_logit_softcapping(logits: &[f32], softcap: f32) -> Vec<f32> {
    let _trace = trace_scope("phase2.apply_final_logit_softcapping");
    logits
        .iter()
        .map(|logit| (logit / softcap).tanh() * softcap)
        .collect()
}

pub fn extract_prefill_logits(logits: &[f32]) -> PrefillLogits {
    let _trace = trace_scope("phase2.extract_prefill_logits");
    PrefillLogits {
        logits: logits.to_vec(),
        final_logits_sha256: build_vector_commitment(logits),
    }
}

fn build_phase2_commitment(activations: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    for row in activations {
        for value in row {
            hasher.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn build_vector_commitment(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn run_attention_for_layer(
    inputs: &[Vec<f32>],
    layer: &Gemma4LayerWeights,
) -> Result<Vec<Vec<f32>>> {
    match layer.attention_kind {
        Gemma4AttentionKind::Sliding => run_sliding_attention(inputs, layer),
        Gemma4AttentionKind::Full => run_full_attention(inputs, layer),
    }
}

fn run_sliding_attention(inputs: &[Vec<f32>], layer: &Gemma4LayerWeights) -> Result<Vec<Vec<f32>>> {
    let sliding_window = layer
        .sliding_window
        .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?;
    run_causal_attention(inputs, layer, 0..0, Some(sliding_window))
}

fn run_full_attention(inputs: &[Vec<f32>], layer: &Gemma4LayerWeights) -> Result<Vec<Vec<f32>>> {
    run_causal_attention(inputs, layer, 0..0, None)
}

fn run_causal_attention(
    inputs: &[Vec<f32>],
    layer: &Gemma4LayerWeights,
    _placeholder: std::ops::Range<usize>,
    sliding_window: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    let _trace = trace_scope(format!(
        "phase2.run_causal_attention attention={:?}",
        layer.attention_kind
    ));
    let seq_len = inputs.len();
    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow::anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }

    let q_projected = {
        let _trace = trace_scope("phase2.run_causal_attention.q_proj");
        linear_sequence(inputs, &layer.q_proj)?
    };
    let raw_k = {
        let _trace = trace_scope("phase2.run_causal_attention.k_proj");
        linear_sequence(inputs, &layer.k_proj)?
    };
    let raw_v = if let Some(v_proj) = &layer.v_proj {
        let _trace = trace_scope("phase2.run_causal_attention.v_proj");
        linear_sequence(inputs, v_proj)?
    } else if layer.attention_k_eq_v {
        trace_event("phase2.run_causal_attention.k_eq_v_reuse");
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };
    let mut q = {
        let _trace = trace_scope("phase2.run_causal_attention.reshape_q");
        reshape_sequence_heads(&q_projected, layer.num_heads, layer.head_dim)?
    };
    let mut k = {
        let _trace = trace_scope("phase2.run_causal_attention.reshape_k");
        reshape_sequence_heads(&raw_k, layer.num_kv_heads, layer.head_dim)?
    };
    let mut v = {
        let _trace = trace_scope("phase2.run_causal_attention.reshape_v");
        reshape_sequence_heads(&raw_v, layer.num_kv_heads, layer.head_dim)?
    };

    {
        let _trace = trace_scope("phase2.run_causal_attention.q_rms_norm");
        apply_head_rms_norm(&mut q, &layer.q_norm_weight, layer.rms_norm_eps)?;
    }
    {
        let _trace = trace_scope("phase2.run_causal_attention.k_rms_norm");
        apply_head_rms_norm(&mut k, &layer.k_norm_weight, layer.rms_norm_eps)?;
    }
    {
        let _trace = trace_scope("phase2.run_causal_attention.v_rms_norm");
        apply_value_rms_norm(&mut v, layer.rms_norm_eps)?;
    }

    {
        let _trace = trace_scope("phase2.run_causal_attention.q_rope");
        apply_rope(&mut q, layer.partial_rotary_dim, layer.rope_base);
    }
    {
        let _trace = trace_scope("phase2.run_causal_attention.k_rope");
        apply_rope(&mut k, layer.partial_rotary_dim, layer.rope_base);
    }

    let head_outputs = (0..layer.num_heads)
        .into_par_iter()
        .map(|head_idx| {
            let _trace = trace_scope(format!("phase2.run_causal_attention.head={head_idx}"));
            let kv_head_idx = head_idx / kv_groups;
            let mut outputs = vec![vec![0.0; layer.head_dim]; seq_len];
            for (query_idx, output) in outputs.iter_mut().enumerate() {
                let start = sliding_window
                    .map(|window| query_idx.saturating_add(1).saturating_sub(window))
                    .unwrap_or(0);
                let logits = (start..=query_idx)
                    .map(|key_idx| dot(&q[head_idx][query_idx], &k[kv_head_idx][key_idx]))
                    .collect::<Vec<_>>();
                let weights = softmax(&logits);

                for (weight, key_idx) in weights.into_iter().zip(start..=query_idx) {
                    for (dim_idx, value) in output.iter_mut().enumerate() {
                        *value += weight * v[kv_head_idx][key_idx][dim_idx];
                    }
                }
            }
            outputs
        })
        .collect::<Vec<_>>();

    let mut combined_heads = vec![vec![0.0; layer.num_heads * layer.head_dim]; seq_len];
    for (head_idx, outputs) in head_outputs.into_iter().enumerate() {
        for (query_idx, output) in outputs.into_iter().enumerate() {
            let dst = &mut combined_heads[query_idx]
                [head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
            dst.copy_from_slice(&output);
        }
    }

    {
        let _trace = trace_scope("phase2.run_causal_attention.o_proj");
        linear_sequence(&combined_heads, &layer.o_proj)
    }
}

fn validate_sequence_width(sequence: &[Vec<f32>], width: usize, label: &str) -> Result<()> {
    if let Some((row_idx, row)) = sequence
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != width)
    {
        bail!("{label} row {row_idx} has width {}, expected {width}", row.len());
    }
    Ok(())
}

fn validate_vector_width(vector: &[f32], width: usize, label: &str) -> Result<()> {
    if vector.len() != width {
        bail!("{label} width mismatch: {} vs {width}", vector.len());
    }
    Ok(())
}

fn add_sequences(lhs: &[Vec<f32>], rhs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    lhs.iter()
        .zip(rhs)
        .map(|(lhs_row, rhs_row)| {
            if lhs_row.len() != rhs_row.len() {
                bail!(
                    "sequence width mismatch: {} vs {}",
                    lhs_row.len(),
                    rhs_row.len()
                );
            }
            Ok(lhs_row
                .iter()
                .zip(rhs_row)
                .map(|(lhs_value, rhs_value)| lhs_value + rhs_value)
                .collect::<Vec<_>>())
        })
        .collect()
}

fn elementwise_mul_sequences(lhs: &[Vec<f32>], rhs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    lhs.iter()
        .zip(rhs)
        .map(|(lhs_row, rhs_row)| {
            if lhs_row.len() != rhs_row.len() {
                bail!(
                    "sequence width mismatch: {} vs {}",
                    lhs_row.len(),
                    rhs_row.len()
                );
            }
            Ok(lhs_row
                .iter()
                .zip(rhs_row)
                .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
                .collect::<Vec<_>>())
        })
        .collect()
}

fn linear_sequence(inputs: &[Vec<f32>], weight: &MatrixF32) -> Result<Vec<Vec<f32>>> {
    validate_sequence_width(inputs, weight.cols, "linear input")?;

    let outputs = inputs
        .par_iter()
        .map(|input| {
            let mut output = vec![0.0; weight.rows];
            for (row_idx, value) in output.iter_mut().enumerate() {
                let mut sum = 0.0;
                let row_offset = row_idx * weight.cols;
                for col_idx in 0..weight.cols {
                    sum += input[col_idx] * weight.values[row_offset + col_idx];
                }
                *value = sum;
            }
            output
        })
        .collect();
    Ok(outputs)
}

fn row_from_matrix(matrix: &MatrixF32, row_idx: usize) -> Result<Vec<f32>> {
    if row_idx >= matrix.rows {
        bail!("matrix row index {row_idx} is out of bounds for {}", matrix.rows);
    }
    let start = row_idx * matrix.cols;
    let end = start + matrix.cols;
    Ok(matrix.values[start..end].to_vec())
}

fn apply_rms_norm_to_sequence(
    inputs: &[Vec<f32>],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<Vec<f32>>> {
    inputs
        .par_iter()
        .map(|row| apply_rms_norm(row, weight, eps))
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

fn apply_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if input.len() != weight.len() {
        bail!("rms norm width mismatch: {} vs {}", input.len(), weight.len());
    }

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, norm_weight)| value * scale * norm_weight)
        .collect())
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

fn apply_head_rms_norm(
    heads: &mut [Vec<Vec<f32>>],
    weight: &[f32],
    eps: f32,
) -> Result<()> {
    heads.par_iter_mut().try_for_each(|head| -> Result<()> {
        for row in head {
            *row = apply_rms_norm(row, weight, eps)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn apply_value_rms_norm(heads: &mut [Vec<Vec<f32>>], eps: f32) -> Result<()> {
    heads.par_iter_mut().try_for_each(|head| -> Result<()> {
        for row in head {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
            let scale = (mean_square + eps).sqrt().recip();
            for value in row {
                *value *= scale;
            }
        }
        Ok(())
    })?;
    Ok(())
}

fn apply_rope(heads: &mut [Vec<Vec<f32>>], rotary_dim: usize, base: f32) {
    if rotary_dim == 0 {
        return;
    }
    let half_dim = rotary_dim / 2;
    for head in heads {
        for (position, row) in head.iter_mut().enumerate() {
            let original = row.clone();
            for dim_idx in 0..half_dim {
                let angle = position as f32 / base.powf((2 * dim_idx) as f32 / rotary_dim as f32);
                let cos = angle.cos();
                let sin = angle.sin();
                let lhs = original[dim_idx];
                let rhs = original[dim_idx + half_dim];
                row[dim_idx] = lhs * cos - rhs * sin;
                row[dim_idx + half_dim] = rhs * cos + lhs * sin;
            }
        }
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(lhs_value, rhs_value)| lhs_value * rhs_value).sum()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / sum).collect()
}

fn apply_gelu_to_sequence(inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    inputs
        .iter()
        .map(|row| row.iter().map(|value| gelu_pytorch_tanh(*value)).collect())
        .collect()
}

fn gelu_pytorch_tanh(value: f32) -> f32 {
    let inner = std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value.powi(3));
    0.5 * value * (1.0 + inner.tanh())
}

#[cfg(test)]
mod tests {
    use super::{
        apply_final_norm, compute_prefill_ple_inputs, embed_input_tokens, extract_prefill_logits,
        project_to_logits, run_gemma4_layer, run_text_layers_prefill,
    };
    use crate::phase2::types::{
        EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights, Gemma4LogitsProjection,
        Gemma4Phase2Model, Gemma4PleGlobalWeights, Gemma4PleLayerWeights, MatrixF32,
    };

    #[test]
    fn embed_input_tokens_looks_up_rows_in_order() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![0.0, 0.5], vec![1.0, 1.5], vec![2.0, 2.5]],
            scale: 1.0,
        };

        let embedded =
            embed_input_tokens(&[2, 0], &embedding_table).expect("embedding should succeed");

        assert_eq!(embedded.activations, vec![vec![2.0, 2.5], vec![0.0, 0.5]]);
        assert_eq!(
            embedded.activations_sha256,
            "ba27ccacfb427e2f44f9a6d875abe24e064893a5ea6a76d8f6c00a29ec10be6f"
        );
    }

    #[test]
    fn embed_input_tokens_rejects_out_of_bounds_token_ids() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![0.0, 0.5]],
            scale: 1.0,
        };

        let error = embed_input_tokens(&[1], &embedding_table)
            .expect_err("out of bounds token should fail");

        assert!(error.to_string().contains("out of bounds"));
    }

    #[test]
    fn embed_input_tokens_rejects_ragged_embedding_tables() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![0.0, 0.5], vec![1.0]],
            scale: 1.0,
        };

        let error =
            embed_input_tokens(&[0], &embedding_table).expect_err("ragged table should fail");

        assert!(error.to_string().contains("expected 2"));
    }

    #[test]
    fn embed_input_tokens_applies_embedding_scale() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![1.0, 2.0]],
            scale: 3.0,
        };

        let embedded =
            embed_input_tokens(&[0], &embedding_table).expect("embedding should succeed");

        assert_eq!(embedded.activations, vec![vec![3.0, 6.0]]);
        assert_eq!(
            embedded.activations_sha256,
            "209a39e983bfd5b06df628da8981625bd58c1342e1543c3641d9873380b9d310"
        );
    }

    #[test]
    fn run_gemma4_layer_preserves_residual_when_projections_are_zero() {
        let activations = vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]];
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4),
            k_proj: zero_matrix(2, 4),
            v_proj: Some(zero_matrix(2, 4)),
            o_proj: zero_matrix(4, 4),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4),
            up_proj: zero_matrix(8, 4),
            down_proj: zero_matrix(4, 8),
            ple: None,
            layer_scalar: None,
        };

        let output = run_gemma4_layer(&activations, &layer, None).expect("layer should succeed");

        assert_eq!(output.activations, activations);
    }

    #[test]
    fn compute_prefill_ple_inputs_is_stable_for_same_inputs() {
        let layers = vec![Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4),
            k_proj: zero_matrix(2, 4),
            v_proj: Some(zero_matrix(2, 4)),
            o_proj: zero_matrix(4, 4),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4),
            up_proj: zero_matrix(8, 4),
            down_proj: zero_matrix(4, 8),
            ple: Some(Gemma4PleLayerWeights {
                input_gate: zero_matrix(2, 4),
                layer_projection: zero_matrix(4, 2),
                post_input_norm_weight: vec![1.0; 4],
            }),
            layer_scalar: None,
        }];
        let ple_global = Gemma4PleGlobalWeights {
            token_embeddings: vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            model_projections: vec![MatrixF32 {
                rows: 2,
                cols: 4,
                values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            }],
            projection_norm_weight: vec![1.0, 1.0],
            embedding_scale: 1.0,
            projection_scalar: 1.0,
            input_scale: 1.0,
        };
        let inputs = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let first =
            compute_prefill_ple_inputs(&[0, 1], &inputs, &layers, &ple_global, 1e-6).unwrap();
        let second =
            compute_prefill_ple_inputs(&[0, 1], &inputs, &layers, &ple_global, 1e-6).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn apply_final_norm_and_project_to_logits_work() {
        let final_hidden_states = vec![vec![1.0, 2.0]];
        let normed = apply_final_norm(&final_hidden_states, &[1.0, 1.0], 0.0).unwrap();
        let logits = project_to_logits(
            &normed.activations[0],
            &Gemma4LogitsProjection::UntiedLmHead(MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            }),
        )
        .unwrap();
        let extracted = extract_prefill_logits(&logits);

        assert_eq!(logits.len(), 2);
        assert_eq!(extracted.logits, logits);
        assert!(!extracted.final_logits_sha256.is_empty());
    }

    #[test]
    fn run_text_layers_prefill_threads_multiple_layers() {
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4),
            k_proj: zero_matrix(2, 4),
            v_proj: Some(zero_matrix(2, 4)),
            o_proj: zero_matrix(4, 4),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4),
            up_proj: zero_matrix(8, 4),
            down_proj: zero_matrix(4, 8),
            ple: None,
            layer_scalar: None,
        };
        let model = Gemma4Phase2Model {
            embedding_table: None,
            embedding_source: None,
            layers: vec![layer.clone(), layer],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(zero_matrix(2, 4)),
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
        };
        let activations = vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]];

        let output = run_text_layers_prefill(&activations, &model, None).unwrap();

        assert_eq!(output.activations, activations);
    }

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
