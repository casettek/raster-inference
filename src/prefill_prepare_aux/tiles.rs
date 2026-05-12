use anyhow::{anyhow, bail, Result};

use crate::shared::det_num::{
    act_to_f32, add_sat, mac_bits, requantize, rms_norm as det_rms_norm, scale_act, Acc, Act, Wgt,
};
use crate::shared::input::InferenceExecutionMode;
use crate::shared::transformer::{
    ActivationSequence, DetNumMatrix, Gemma4LayerWeights, Gemma4PleGlobalWeights,
    Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationRow,
    InternalActivationSequence, MatrixF32,
};

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
    execution_mode: InferenceExecutionMode,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    model.validate_execution_mode(execution_mode)?;
    model
        .ple_global
        .as_ref()
        .map(|ple_global| {
            compute_prefill_ple_inputs_internal(
                prompt_token_ids,
                token_embeddings.clone_internal(),
                &model.layers,
                ple_global,
                model.rms_norm_eps,
                model.rms_norm_eps_det,
                execution_mode,
            )
        })
        .transpose()
}

pub(crate) fn compute_prefill_ple_inputs_internal(
    token_ids: &[u32],
    input_activations: InternalActivationSequence,
    layers: &[Gemma4LayerWeights],
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps: f32,
    rms_norm_eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<Gemma4PrefillPleInputs> {
    let input_values = input_activations.as_f32_slice();
    if input_values.is_empty() {
        bail!("transformer PLE computation requires at least one activation row");
    }
    if layers.is_empty() {
        bail!("transformer PLE computation requires at least one layer");
    }

    let hidden_size = layers[0].hidden_size;
    validate_sequence_width(input_values, hidden_size, "input activations")?;
    if token_ids.len() != input_values.len() {
        bail!(
            "transformer PLE computation requires token ids and activations to have matching lengths"
        );
    }
    if ple_global.token_embedding_layer_count() != layers.len() {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            ple_global.token_embedding_layer_count(),
            layers.len()
        );
    }
    if ple_global.model_projection_layer_count() != layers.len() {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            ple_global.model_projection_layer_count(),
            layers.len()
        );
    }

    let input_buffer = ActivationSequenceBuffer::from_internal(input_activations);
    let mut per_layer_inputs = Vec::with_capacity(layers.len());
    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer.ple.is_none() {
            per_layer_inputs.push(None);
            continue;
        }

        let layer_input = compute_prefill_ple_layer_input(
            token_ids,
            &input_buffer,
            layer_idx,
            ple_global,
            rms_norm_eps,
            rms_norm_eps_det,
            execution_mode,
        )?;
        per_layer_inputs.push(Some(layer_input.into_internal()));
    }

    Ok(Gemma4PrefillPleInputs::from_internal(per_layer_inputs))
}

fn compute_prefill_ple_layer_input(
    token_ids: &[u32],
    input_buffer: &ActivationSequenceBuffer,
    layer_idx: usize,
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps: f32,
    rms_norm_eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequenceBuffer> {
    let embedded = build_scaled_token_embedding_sequence(
        token_ids,
        ple_global,
        layer_idx,
        ple_global.embedding_scale,
        ple_global.embedding_scale_det,
        execution_mode == InferenceExecutionMode::Deterministic,
    )?;

    let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
    let model_projection_det =
        crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?;
    let projected = project_linear_sequence_buffer(
        input_buffer,
        &model_projection,
        model_projection_det.as_deref(),
    )?;
    let projected_uses_det = projected.acts.is_some();
    let projected = scale_sequence_buffer(
        &projected,
        ple_global.projection_scalar,
        ple_global.projection_scalar_det,
        projected_uses_det,
    )?;
    let projected = apply_rms_norm_to_sequence_buffer(
        &projected,
        &ple_global.projection_norm_weight,
        ple_global.projection_norm_weight_det.as_deref(),
        rms_norm_eps,
        rms_norm_eps_det,
        execution_mode,
    )?;

    let combined = add_sequence_buffers(
        &embedded,
        &projected,
        execution_mode == InferenceExecutionMode::Deterministic || projected_uses_det,
    )?;
    scale_sequence_buffer(
        &combined,
        ple_global.input_scale,
        ple_global.input_scale_det,
        execution_mode == InferenceExecutionMode::Deterministic || combined.acts.is_some(),
    )
}

fn build_scaled_token_embedding_sequence(
    token_ids: &[u32],
    ple_global: &Gemma4PleGlobalWeights,
    layer_idx: usize,
    scalar: f32,
    scalar_det: Option<Act>,
    deterministic: bool,
) -> Result<ActivationSequenceBuffer> {
    let rows = token_ids
        .iter()
        .copied()
        .map(|token_id| {
            crate::io::load_ple_token_embedding_row_internal(ple_global, layer_idx, token_id)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(ActivationRowBuffer::from_internal)
        .collect();
    let embeddings = activation_sequence_buffer_from_rows(rows);
    scale_sequence_buffer(&embeddings, scalar, scalar_det, deterministic)
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
}

fn activation_sequence_buffer_from_rows(
    rows: Vec<ActivationRowBuffer>,
) -> ActivationSequenceBuffer {
    let values = rows.iter().map(|row| row.values.clone()).collect();
    let acts = rows
        .iter()
        .map(|row| row.acts.clone())
        .collect::<Option<Vec<_>>>();
    ActivationSequenceBuffer { values, acts }
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

fn add_sequence_buffers(
    lhs: &ActivationSequenceBuffer,
    rhs: &ActivationSequenceBuffer,
    deterministic: bool,
) -> Result<ActivationSequenceBuffer> {
    if deterministic {
        Ok(ActivationSequenceBuffer::from_acts(add_act_sequences(
            &sequence_buffer_acts(lhs)?,
            &sequence_buffer_acts(rhs)?,
        )?))
    } else {
        Ok(ActivationSequenceBuffer::from_values(add_sequences(
            &lhs.values,
            &rhs.values,
        )?))
    }
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
                .collect())
        })
        .collect()
}

fn add_act_sequences(lhs: &[Vec<Act>], rhs: &[Vec<Act>]) -> Result<Vec<Vec<Act>>> {
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    lhs.iter()
        .zip(rhs)
        .map(|(lhs_row, rhs_row)| {
            if lhs_row.len() != rhs_row.len() {
                bail!("row width mismatch: {} vs {}", lhs_row.len(), rhs_row.len());
            }
            Ok(lhs_row
                .iter()
                .zip(rhs_row)
                .map(|(lhs_value, rhs_value)| add_sat(*lhs_value, *rhs_value))
                .collect())
        })
        .collect()
}

fn scale_sequence_buffer(
    values: &ActivationSequenceBuffer,
    scalar: f32,
    scalar_det: Option<Act>,
    deterministic: bool,
) -> Result<ActivationSequenceBuffer> {
    if deterministic {
        let scalar_det = scalar_det.ok_or_else(|| {
            anyhow!("deterministic sequence scaling requires canonical Act scalar")
        })?;
        Ok(ActivationSequenceBuffer::from_acts(scale_act_sequences(
            &sequence_buffer_acts(values)?,
            scalar_det,
        )))
    } else {
        Ok(ActivationSequenceBuffer::from_values(
            values
                .values
                .iter()
                .map(|row| row.iter().map(|value| value * scalar).collect())
                .collect(),
        ))
    }
}

fn scale_act_sequences(values: &[Vec<Act>], scalar: Act) -> Vec<Vec<Act>> {
    values
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(|value| scale_act(value, scalar))
                .collect()
        })
        .collect()
}

fn sequence_buffer_acts(buffer: &ActivationSequenceBuffer) -> Result<Vec<Vec<Act>>> {
    buffer
        .acts
        .clone()
        .ok_or_else(|| anyhow!("deterministic sequence operation requires canonical Act rows"))
}

fn project_linear_sequence_buffer(
    inputs: &ActivationSequenceBuffer,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<ActivationSequenceBuffer> {
    match det_weight {
        Some(det_weight) => Ok(ActivationSequenceBuffer::from_acts(
            det_linear_sequence_acts_from_acts(&sequence_buffer_acts(inputs)?, det_weight)?,
        )),
        None if inputs.acts.is_some() => {
            bail!("deterministic linear sequence projection requires canonical det_weight")
        }
        None => Ok(ActivationSequenceBuffer::from_values(linear_sequence(
            &inputs.values,
            weight,
        )?)),
    }
}

fn linear_sequence(inputs: &[Vec<f32>], weight: &MatrixF32) -> Result<Vec<Vec<f32>>> {
    validate_sequence_width(inputs, weight.cols, "linear input")?;
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut output = vec![0.0; weight.rows];
        for (row_idx, value) in output.iter_mut().enumerate() {
            let row_offset = row_idx * weight.cols;
            let mut sum = 0.0;
            for col_idx in 0..weight.cols {
                sum += input[col_idx] * weight.values[row_offset + col_idx];
            }
            *value = sum;
        }
        outputs.push(output);
    }
    Ok(outputs)
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
        .iter()
        .map(|input| det_linear_row_acts_from_acts(input, weight))
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

    let mut output = Vec::with_capacity(weight.rows);
    for row_idx in 0..weight.rows {
        let row_offset = row_idx * weight.cols;
        let mut acc_bits = 0_i64;
        for (col_idx, act) in quantized_input.iter().enumerate() {
            acc_bits = mac_bits(acc_bits, act.to_bits(), weight.values[row_offset + col_idx]);
        }
        output.push(requantize(Acc::from_bits(acc_bits)));
    }
    Ok(output)
}

fn apply_rms_norm_to_sequence_buffer(
    inputs: &ActivationSequenceBuffer,
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequenceBuffer> {
    match execution_mode {
        InferenceExecutionMode::Fp32 => inputs
            .values
            .iter()
            .map(|row| apply_rms_norm_row_f32(row, weight, eps))
            .collect::<Result<Vec<_>>>()
            .map(ActivationSequenceBuffer::from_values),
        InferenceExecutionMode::Deterministic => {
            let quantized_weight = weight_det.ok_or_else(|| {
                anyhow!("deterministic RMSNorm requires canonical Wgt norm weights")
            })?;
            let quantized_eps = eps_det
                .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
            sequence_buffer_acts(inputs)?
                .into_iter()
                .map(|row| {
                    if row.len() != weight.len() {
                        bail!("rms norm width mismatch: {} vs {}", row.len(), weight.len());
                    }
                    Ok(det_rms_norm(&row, quantized_weight, quantized_eps))
                })
                .collect::<Result<Vec<_>>>()
                .map(ActivationSequenceBuffer::from_acts)
        }
    }
}

fn apply_rms_norm_row_f32(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if input.len() != weight.len() {
        bail!(
            "rms norm width mismatch: {} vs {}",
            input.len(),
            weight.len()
        );
    }

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, norm_weight)| value * scale * norm_weight)
        .collect())
}
