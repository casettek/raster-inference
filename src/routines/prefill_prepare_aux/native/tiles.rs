use anyhow::{anyhow, bail, Result};

use crate::shared::model::transformer::{
    ActivationSequence, DetNumMatrix, Gemma4LayerWeights, Gemma4PleGlobalWeights,
    Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence,
};
use crate::shared::numerics::det_num::{add_sat, mac_bits, requantize, scale_act, Acc, Act};

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    model
        .ple_global
        .as_ref()
        .map(|ple_global| {
            compute_prefill_ple_inputs_internal(
                prompt_token_ids,
                token_embeddings.clone_internal(),
                &model.layers,
                ple_global,
                model.rms_norm_eps_det,
            )
        })
        .transpose()
}

pub(crate) fn compute_prefill_ple_inputs_internal(
    token_ids: &[u32],
    input_activations: InternalActivationSequence,
    layers: &[Gemma4LayerWeights],
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps_det: Option<Acc>,
) -> Result<Gemma4PrefillPleInputs> {
    compute_prefill_ple_inputs_det(
        token_ids,
        &input_activations,
        layers,
        ple_global,
        rms_norm_eps_det,
    )
}

/// Single-track deterministic PLE input computation: identical canonical
/// arithmetic and operation order to the historical dual-track path, with no
/// f32 mirrors.
fn compute_prefill_ple_inputs_det(
    token_ids: &[u32],
    input_activations: &InternalActivationSequence,
    layers: &[Gemma4LayerWeights],
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps_det: Option<Acc>,
) -> Result<Gemma4PrefillPleInputs> {
    let input_rows = input_activations
        .det_values()
        .ok_or_else(|| anyhow!("deterministic sequence operation requires canonical Act rows"))?;
    if input_rows.is_empty() {
        bail!("transformer PLE computation requires at least one activation row");
    }
    if layers.is_empty() {
        bail!("transformer PLE computation requires at least one layer");
    }
    let hidden_size = layers[0].hidden_size;
    if let Some((row_idx, row)) = input_rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != hidden_size)
    {
        bail!(
            "input activations row {row_idx} has width {}, expected {hidden_size}",
            row.len()
        );
    }
    if token_ids.len() != input_rows.len() {
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

    let embedding_scale = ple_global
        .embedding_scale_det
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    let projection_scalar = ple_global
        .projection_scalar_det
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    let input_scale = ple_global
        .input_scale_det
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    let projection_norm_weight = ple_global
        .projection_norm_weight_det
        .as_deref()
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Wgt norm weights"))?;
    let projection_norm_eps = rms_norm_eps_det
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;

    let mut per_layer_inputs = Vec::with_capacity(layers.len());
    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer.ple.is_none() {
            per_layer_inputs.push(None);
            continue;
        }

        let model_projection_det =
            crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?
                .ok_or_else(|| {
                    anyhow!(
                        "deterministic linear sequence projection requires canonical det_weight"
                    )
                })?;

        let mut combined_rows = Vec::with_capacity(token_ids.len());
        for (row_idx, token_id) in token_ids.iter().enumerate() {
            let embedded_row =
                crate::io::load_ple_token_embedding_row_internal(ple_global, layer_idx, *token_id)?;
            let mut embedded = embedded_row
                .det_values()
                .map(<[Act]>::to_vec)
                .ok_or_else(|| {
                    anyhow!("deterministic row operation requires canonical Act values")
                })?;
            for value in embedded.iter_mut() {
                *value = scale_act(*value, embedding_scale);
            }

            let mut projected =
                det_linear_row_acts_from_acts(&input_rows[row_idx], &model_projection_det)?;
            for value in projected.iter_mut() {
                *value = scale_act(*value, projection_scalar);
            }
            if projected.len() != projection_norm_weight.len() {
                bail!(
                    "rms norm width mismatch: {} vs {}",
                    projected.len(),
                    projection_norm_weight.len()
                );
            }
            crate::shared::numerics::det_num::rms_norm_in_place(
                &mut projected,
                projection_norm_weight,
                projection_norm_eps,
            );

            if embedded.len() != projected.len() {
                bail!(
                    "row width mismatch: {} vs {}",
                    embedded.len(),
                    projected.len()
                );
            }
            let mut combined = embedded;
            for (value, projected_value) in combined.iter_mut().zip(&projected) {
                *value = add_sat(*value, *projected_value);
            }
            for value in combined.iter_mut() {
                *value = scale_act(*value, input_scale);
            }
            combined_rows.push(combined);
        }
        per_layer_inputs.push(Some(InternalActivationSequence::from_det_values_only(
            combined_rows,
        )));
    }

    Ok(Gemma4PrefillPleInputs::from_internal(per_layer_inputs))
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
