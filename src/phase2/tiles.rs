use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use super::types::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4Layer0Weights,
    Gemma4PleLayerWeights, MatrixF32,
};

pub fn embed_input_tokens(
    token_ids: &[u32],
    embedding_table: &EmbeddingTable,
) -> Result<EmbeddedTokenSequence> {
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

pub fn run_first_gemma4_layer(
    token_ids: &[u32],
    input_activations: &[Vec<f32>],
    layer: &Gemma4Layer0Weights,
) -> Result<ActivationSequence> {
    if input_activations.is_empty() {
        bail!("phase 2 layer execution requires at least one activation row");
    }
    if token_ids.len() != input_activations.len() {
        bail!(
            "phase 2 layer execution requires token ids and activations to have matching lengths"
        );
    }
    validate_sequence_width(input_activations, layer.hidden_size, "input activations")?;

    let per_layer_input = layer
        .ple
        .as_ref()
        .map(|ple| compute_per_layer_input(token_ids, input_activations, ple, layer.rms_norm_eps))
        .transpose()?;

    let mut xs = input_activations.to_vec();

    let residual = xs.clone();
    let normed = apply_rms_norm_to_sequence(&xs, &layer.input_layernorm_weight, layer.rms_norm_eps)?;
    let attn_out = run_sliding_attention(&normed, layer)?;
    let attn_out = apply_rms_norm_to_sequence(
        &attn_out,
        &layer.post_attention_layernorm_weight,
        layer.rms_norm_eps,
    )?;
    xs = add_sequences(&residual, &attn_out)?;

    let residual = xs.clone();
    let normed = apply_rms_norm_to_sequence(
        &xs,
        &layer.pre_feedforward_layernorm_weight,
        layer.rms_norm_eps,
    )?;
    let gate = apply_gelu_to_sequence(&linear_sequence(&normed, &layer.gate_proj)?);
    let up = linear_sequence(&normed, &layer.up_proj)?;
    let ff_hidden = elementwise_mul_sequences(&gate, &up)?;
    let ff_out = linear_sequence(&ff_hidden, &layer.down_proj)?;
    let ff_out = apply_rms_norm_to_sequence(
        &ff_out,
        &layer.post_feedforward_layernorm_weight,
        layer.rms_norm_eps,
    )?;
    xs = add_sequences(&residual, &ff_out)?;

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, &per_layer_input) {
        let residual = xs.clone();
        let gated = apply_gelu_to_sequence(&linear_sequence(&xs, &ple.input_gate)?);
        let gated = elementwise_mul_sequences(&gated, per_layer_input)?;
        let projected = linear_sequence(&gated, &ple.layer_projection)?;
        let projected = apply_rms_norm_to_sequence(
            &projected,
            &ple.post_input_norm_weight,
            layer.rms_norm_eps,
        )?;
        xs = add_sequences(&residual, &projected)?;
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        for row in &mut xs {
            for value in row {
                *value *= layer_scalar;
            }
        }
    }

    Ok(ActivationSequence {
        activations_sha256: build_phase2_commitment(&xs),
        activations: xs,
    })
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

fn compute_per_layer_input(
    token_ids: &[u32],
    input_activations: &[Vec<f32>],
    ple: &Gemma4PleLayerWeights,
    rms_norm_eps: f32,
) -> Result<Vec<Vec<f32>>> {
    let mut embedded = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        embedded.push(
            row_from_matrix(&ple.token_embedding, row_idx)?
                .into_iter()
                .map(|value| value * ple.embedding_scale)
                .collect::<Vec<_>>(),
        );
    }

    let mut projected = linear_sequence(input_activations, &ple.model_projection)?;
    for row in &mut projected {
        for value in row {
            *value *= ple.projection_scalar;
        }
    }
    let projected =
        apply_rms_norm_to_sequence(&projected, &ple.projection_norm_weight, rms_norm_eps)?;

    let mut combined = add_sequences(&embedded, &projected)?;
    for row in &mut combined {
        for value in row {
            *value *= ple.input_scale;
        }
    }
    Ok(combined)
}

fn run_sliding_attention(
    inputs: &[Vec<f32>],
    layer: &Gemma4Layer0Weights,
) -> Result<Vec<Vec<f32>>> {
    let seq_len = inputs.len();
    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow::anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }

    let mut q = reshape_sequence_heads(
        &linear_sequence(inputs, &layer.q_proj)?,
        layer.num_heads,
        layer.head_dim,
    )?;
    let mut k = reshape_sequence_heads(
        &linear_sequence(inputs, &layer.k_proj)?,
        layer.num_kv_heads,
        layer.head_dim,
    )?;
    let mut v = reshape_sequence_heads(
        &linear_sequence(inputs, &layer.v_proj)?,
        layer.num_kv_heads,
        layer.head_dim,
    )?;

    apply_head_rms_norm(&mut q, &layer.q_norm_weight, layer.rms_norm_eps)?;
    apply_head_rms_norm(&mut k, &layer.k_norm_weight, layer.rms_norm_eps)?;
    apply_value_rms_norm(&mut v, layer.rms_norm_eps)?;

    apply_local_rope(&mut q, layer.head_dim, 10_000.0);
    apply_local_rope(&mut k, layer.head_dim, 10_000.0);

    let mut combined_heads = vec![vec![0.0; layer.num_heads * layer.head_dim]; seq_len];
    for head_idx in 0..layer.num_heads {
        let kv_head_idx = head_idx / kv_groups;
        for query_idx in 0..seq_len {
            let start = query_idx
                .saturating_add(1)
                .saturating_sub(layer.sliding_window);
            let logits = (start..=query_idx)
                .map(|key_idx| dot(&q[head_idx][query_idx], &k[kv_head_idx][key_idx]))
                .collect::<Vec<_>>();
            let weights = softmax(&logits);

            let mut output = vec![0.0; layer.head_dim];
            for (weight, key_idx) in weights.into_iter().zip(start..=query_idx) {
                for (dim_idx, value) in output.iter_mut().enumerate() {
                    *value += weight * v[kv_head_idx][key_idx][dim_idx];
                }
            }

            let dst = &mut combined_heads[query_idx]
                [head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
            dst.copy_from_slice(&output);
        }
    }

    linear_sequence(&combined_heads, &layer.o_proj)
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

    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut output = vec![0.0; weight.rows];
        for row_idx in 0..weight.rows {
            let mut sum = 0.0;
            let row_offset = row_idx * weight.cols;
            for col_idx in 0..weight.cols {
                sum += input[col_idx] * weight.values[row_offset + col_idx];
            }
            output[row_idx] = sum;
        }
        outputs.push(output);
    }
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
        .iter()
        .map(|row| apply_rms_norm(row, weight, eps))
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
    for head in heads {
        for row in head {
            *row = apply_rms_norm(row, weight, eps)?;
        }
    }
    Ok(())
}

fn apply_value_rms_norm(heads: &mut [Vec<Vec<f32>>], eps: f32) -> Result<()> {
    for head in heads {
        for row in head {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
            let scale = (mean_square + eps).sqrt().recip();
            for value in row {
                *value *= scale;
            }
        }
    }
    Ok(())
}

fn apply_local_rope(heads: &mut [Vec<Vec<f32>>], head_dim: usize, base: f32) {
    let half_dim = head_dim / 2;
    for head in heads {
        for (position, row) in head.iter_mut().enumerate() {
            let original = row.clone();
            for dim_idx in 0..half_dim {
                let angle = position as f32 / base.powf((2 * dim_idx) as f32 / head_dim as f32);
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
    use super::{embed_input_tokens, run_first_gemma4_layer};
    use crate::phase2::types::{EmbeddingTable, Gemma4Layer0Weights, MatrixF32};

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
    fn run_first_gemma4_layer_preserves_residual_when_projections_are_zero() {
        let activations = vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]];
        let layer = Gemma4Layer0Weights {
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: 2,
            rms_norm_eps: 1e-6,
            q_proj: zero_matrix(4, 4),
            k_proj: zero_matrix(2, 4),
            v_proj: zero_matrix(2, 4),
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

        let output =
            run_first_gemma4_layer(&[1, 0], &activations, &layer).expect("layer should succeed");

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
