use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::shared::det_num::{
    act_to_f32, add_sat, f32_to_acc, f32_to_act, mac_bits, mul_sat, requantize,
    rms_norm as det_rms_norm, scale_act, value_rms_norm as det_value_rms_norm, Acc, Act,
};
use crate::shared::input::InferenceExecutionMode;
use crate::shared::transformer::{
    ActivationSequence, DetNumMatrix, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4PleGlobalWeights, Gemma4PrefillPleInputs,
    Gemma4TransformerModel, GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, PrefillLogits,
    ResolvedGemma4LayerWeights,
};
use crate::trace::trace_scope;

pub fn embed_input_tokens(
    token_ids: &[u32],
    embedding_table: &EmbeddingTable,
) -> Result<EmbeddedTokenSequence> {
    embed_input_tokens_with_mode(token_ids, embedding_table, InferenceExecutionMode::Fp32)
}

pub fn embed_input_tokens_with_mode(
    token_ids: &[u32],
    embedding_table: &EmbeddingTable,
    execution_mode: InferenceExecutionMode,
) -> Result<EmbeddedTokenSequence> {
    // let _trace = trace_scope("transformer_state_transition.embed_input_tokens");
    if token_ids.is_empty() {
        bail!("transformer embedding requires at least one token id");
    }

    if embedding_table.rows.is_empty() {
        bail!("transformer embedding requires a non-empty embedding table");
    }

    let hidden_size = embedding_table.rows[0].len();
    if hidden_size == 0 {
        bail!("transformer embedding rows must have non-zero width");
    }

    if let Some((row_idx, row)) = embedding_table
        .rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != hidden_size)
    {
        bail!(
            "transformer embedding table row {row_idx} has width {}, expected {hidden_size}",
            row.len()
        );
    }

    let mut activations = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        let row = embedding_table.rows.get(row_idx).ok_or_else(|| {
            anyhow::anyhow!("token id {token_id} is out of bounds for embedding table")
        })?;
        let activation = scale_row_buffer(
            &ActivationRowBuffer::from_values(row.clone()),
            embedding_table.scale,
            execution_mode == InferenceExecutionMode::Deterministic,
        );
        activations.push(activation.values);
    }

    let activations_sha256 = build_activation_commitment(&activations);

    Ok(ActivationSequence {
        activations,
        activations_sha256,
    })
}

pub fn embed_input_token(token_id: u32, embedding_table: &EmbeddingTable) -> Result<Vec<f32>> {
    embed_input_token_with_mode(token_id, embedding_table, InferenceExecutionMode::Fp32)
}

pub fn embed_input_token_with_mode(
    token_id: u32,
    embedding_table: &EmbeddingTable,
    execution_mode: InferenceExecutionMode,
) -> Result<Vec<f32>> {
    let embedded = embed_input_tokens_with_mode(&[token_id], embedding_table, execution_mode)?;
    embedded
        .activations
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("transformer embedding returned no activation rows"))
}

pub fn compute_prefill_ple_inputs(
    token_ids: &[u32],
    input_activations: &[Vec<f32>],
    layers: &[Gemma4LayerWeights],
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<Gemma4PrefillPleInputs> {
    // let _trace = trace_scope("transformer_state_transition.compute_prefill_ple_inputs");
    if input_activations.is_empty() {
        bail!("transformer PLE computation requires at least one activation row");
    }
    if layers.is_empty() {
        bail!("transformer PLE computation requires at least one layer");
    }
    let hidden_size = layers[0].hidden_size;
    validate_sequence_width(input_activations, hidden_size, "input activations")?;

    if token_ids.len() != input_activations.len() {
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

    let mut per_layer_inputs = Vec::with_capacity(layers.len());
    for (layer_idx, layer) in layers.iter().enumerate() {
        // trace_event(format!("transformer_state_transition.compute_prefill_ple_inputs layer={layer_idx}"));
        if layer.ple.is_none() {
            per_layer_inputs.push(None);
            continue;
        }

        let mut embedded = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            embedded.push(
                crate::io::load_ple_token_embedding_row(ple_global, layer_idx, *token_id)?,
            );
        }
        let embedded = scale_sequence_buffer(
            &ActivationSequenceBuffer::from_values(embedded),
            ple_global.embedding_scale,
            execution_mode == InferenceExecutionMode::Deterministic,
        );

        let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
        let model_projection_det =
            crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?;
        let projected = project_linear_sequence_buffer(
            &ActivationSequenceBuffer::from_values(input_activations.to_vec()),
            &model_projection,
            model_projection_det.as_deref(),
        )?;
        let projected_uses_det = projected.acts.is_some();
        let projected = scale_sequence_buffer(
            &projected,
            ple_global.projection_scalar,
            projected_uses_det,
        );
        let projected = ActivationSequenceBuffer::from_values(apply_rms_norm_to_sequence(
            &projected.values,
            &ple_global.projection_norm_weight,
            rms_norm_eps,
            execution_mode,
        )?);

        let combined = add_sequence_buffers(
            &embedded,
            &projected,
            execution_mode == InferenceExecutionMode::Deterministic || projected_uses_det,
        )?;
        let combined = scale_sequence_buffer(
            &combined,
            ple_global.input_scale,
            execution_mode == InferenceExecutionMode::Deterministic || combined.acts.is_some(),
        );
        per_layer_inputs.push(Some(combined.values));
    }

    Ok(Gemma4PrefillPleInputs { per_layer_inputs })
}

pub fn compute_decode_ple_input(
    token_id: u32,
    input_activation: &[f32],
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
    ple_global: Option<&Gemma4PleGlobalWeights>,
    rms_norm_eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<Option<Vec<f32>>> {
    let Some(ple_global) = ple_global else {
        return Ok(None);
    };
    if layer.ple.is_none() {
        return Ok(None);
    }

    validate_vector_width(
        input_activation,
        layer.hidden_size,
        "decode PLE input activation",
    )?;
    let embedded = scale_row_buffer(
        &ActivationRowBuffer::from_values(crate::io::load_ple_token_embedding_row(
            ple_global, layer_idx, token_id,
        )?),
        ple_global.embedding_scale,
        execution_mode == InferenceExecutionMode::Deterministic,
    );
    let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
    let model_projection_det =
        crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?;
    let projected = project_linear_row_buffer(
        &ActivationRowBuffer::from_values(input_activation.to_vec()),
        &model_projection,
        model_projection_det.as_deref(),
    )?;
    let projected_uses_det = projected.acts.is_some();
    let projected = scale_row_buffer(&projected, ple_global.projection_scalar, projected_uses_det);
    let projected = ActivationRowBuffer::from_values(apply_rms_norm(
        &projected.values,
        &ple_global.projection_norm_weight,
        rms_norm_eps,
        execution_mode,
    )?);

    if embedded.values.len() != projected.values.len() {
        bail!(
            "decode PLE width mismatch: embedded {} vs projected {}",
            embedded.values.len(),
            projected.values.len()
        );
    }

    let combined = add_row_buffers(
        &embedded,
        &projected,
        execution_mode == InferenceExecutionMode::Deterministic || projected_uses_det,
    )?;
    let combined = scale_row_buffer(
        &combined,
        ple_global.input_scale,
        execution_mode == InferenceExecutionMode::Deterministic || combined.acts.is_some(),
    );

    Ok(Some(combined.values))
}

pub fn run_gemma4_layer(
    input_activations: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[Vec<f32>]>,
) -> Result<ActivationSequence> {
    Ok(
        run_gemma4_layer_with_cache(
            input_activations,
            layer,
            per_layer_input,
            None,
            InferenceExecutionMode::Fp32,
        )?
        .0,
    )
}

pub(crate) fn run_gemma4_layer_with_cache(
    input_activations: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[Vec<f32>]>,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequence, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_gemma4_layer attention={:?}",
    //     layer.attention_kind
    // ));
    if input_activations.is_empty() {
        bail!("transformer layer execution requires at least one activation row");
    }
    validate_sequence_width(input_activations, layer.hidden_size, "input activations")?;
    if let Some(per_layer_input) = per_layer_input {
        validate_sequence_width(
            per_layer_input,
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
        if per_layer_input.len() != input_activations.len() {
            bail!(
                "transformer layer execution requires per-layer inputs and activations to have matching lengths"
            );
        }
    }

    let mut xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.clone_input");
        ActivationSequenceBuffer::from_values(input_activations.to_vec())
    };

    let residual = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.clone_residual");
        xs.clone()
    };
    let normed = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.input_rms_norm");
        apply_rms_norm_to_sequence(
            &xs.values,
            &layer.input_layernorm_weight,
            layer.rms_norm_eps,
            execution_mode,
        )?
    };
    let (attn_out, layer_cache) = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.core");
        run_attention_for_layer_with_cache(&normed, layer, donor_cache, execution_mode)?
    };
    let attn_out = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.post_rms_norm");
        ActivationSequenceBuffer::from_values(apply_rms_norm_to_sequence(
            &attn_out,
            &layer.post_attention_layernorm_weight,
            layer.rms_norm_eps,
            execution_mode,
        )?)
    };
    xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.residual_add");
        add_sequence_buffers(&residual, &attn_out, layer.o_proj_det.is_some())?
    };

    let residual = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.clone_residual");
        xs.clone()
    };
    let normed = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.pre_rms_norm");
        apply_rms_norm_to_sequence(
            &xs.values,
            &layer.pre_feedforward_layernorm_weight,
            layer.rms_norm_eps,
            execution_mode,
        )?
    };
    let (gate, up) = match (layer.gate_proj_det.as_ref(), layer.up_proj_det.as_ref()) {
        (Some(gate_weight), Some(up_weight)) => {
            let quantized_normed = normed
                .iter()
                .map(|row| row.iter().copied().map(f32_to_act).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let gate_preactivation = quantized_normed
                .par_iter()
                .map(|row| det_linear_row_from_acts(row, gate_weight.as_ref()))
                .collect::<Vec<_>>()
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let up = quantized_normed
                .par_iter()
                .map(|row| det_linear_row_acts_from_acts(row, up_weight.as_ref()))
                .collect::<Vec<_>>()
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            (
                ActivationSequenceBuffer::from_values(apply_gelu_to_sequence(&gate_preactivation)),
                ActivationSequenceBuffer::from_acts(up),
            )
        }
        (gate_weight, up_weight) => {
            let gate_preactivation = match gate_weight {
                Some(weight) => project_linear_sequence_buffer(
                    &ActivationSequenceBuffer::from_values(normed.clone()),
                    layer.gate_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => ActivationSequenceBuffer::from_values(linear_sequence(
                    &normed,
                    layer.gate_proj.as_ref(),
                )?),
            };
            let up = match up_weight {
                Some(weight) => project_linear_sequence_buffer(
                    &ActivationSequenceBuffer::from_values(normed.clone()),
                    layer.up_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => ActivationSequenceBuffer::from_values(linear_sequence(
                    &normed,
                    layer.up_proj.as_ref(),
                )?),
            };
            (
                ActivationSequenceBuffer::from_values(apply_gelu_to_sequence(
                    &gate_preactivation.values,
                )),
                up,
            )
        }
    };
    let ff_hidden = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.hidden_mul");
        mul_sequence_buffers(&gate, &up, gate.acts.is_some() || up.acts.is_some())?
    };
    let ff_out = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.down_proj");
        match layer.down_proj_det.as_ref() {
            Some(weight) => {
                project_linear_sequence_buffer(&ff_hidden, layer.down_proj.as_ref(), Some(weight.as_ref()))?
            }
            None => ActivationSequenceBuffer::from_values(linear_sequence(
                &ff_hidden.values,
                layer.down_proj.as_ref(),
            )?),
        }
    };
    let ff_out_uses_det = ff_out.acts.is_some();
    let ff_out = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.post_rms_norm");
        ActivationSequenceBuffer::from_values(apply_rms_norm_to_sequence(
            &ff_out.values,
            &layer.post_feedforward_layernorm_weight,
            layer.rms_norm_eps,
            execution_mode,
        )?)
    };
    xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.residual_add");
        add_sequence_buffers(&residual, &ff_out, ff_out_uses_det)?
    };

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.clone_residual");
            xs.clone()
        };
        let quantized_xs = ple
            .input_gate_det
            .as_ref()
            .map(|_| quantize_sequence_to_acts(&xs.values));
        let gated = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.input_gate_gelu");
            ActivationSequenceBuffer::from_values(apply_gelu_to_sequence(
                &project_linear_sequence(
                    &xs.values,
                    quantized_xs.as_deref(),
                    ple.input_gate.as_ref(),
                    ple.input_gate_det.as_deref(),
                )?,
            ))
        };
        let gated = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.input_mul");
            mul_sequence_buffers(
                &gated,
                &ActivationSequenceBuffer::from_values(per_layer_input.to_vec()),
                ple.input_gate_det.is_some(),
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
        let projected_uses_det = projected.acts.is_some();
        let projected = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.post_rms_norm");
            ActivationSequenceBuffer::from_values(apply_rms_norm_to_sequence(
                &projected.values,
                &ple.post_input_norm_weight,
                layer.rms_norm_eps,
                execution_mode,
            )?)
        };
        xs = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.residual_add");
            add_sequence_buffers(&residual, &projected, projected_uses_det)?
        };
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.layer_scalar");
        xs = scale_sequence_buffer(&xs, layer_scalar, xs.acts.is_some());
    }

    Ok((
        ActivationSequence {
            activations_sha256: {
                // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.commitment_hash");
                build_activation_commitment(&xs.values)
            },
            activations: xs.values,
        },
        layer_cache,
    ))
}

pub fn run_gemma4_layer_decode(
    input_activation: &[f32],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[f32]>,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
) -> Result<(Vec<f32>, LayerKvCache)> {
    run_gemma4_layer_decode_with_mode(
        input_activation,
        layer,
        per_layer_input,
        cache,
        donor_cache,
        position,
        InferenceExecutionMode::Fp32,
    )
}

pub fn run_gemma4_layer_decode_with_mode(
    input_activation: &[f32],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[f32]>,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<f32>, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_gemma4_layer_decode attention={:?}",
    //     layer.attention_kind
    // ));
    validate_vector_width(
        input_activation,
        layer.hidden_size,
        "decode input activation",
    )?;
    if let Some(per_layer_input) = per_layer_input {
        validate_vector_width(
            per_layer_input,
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

    let residual = ActivationRowBuffer::from_values(input_activation.to_vec());
    let normed = apply_rms_norm(
        input_activation,
        &layer.input_layernorm_weight,
        layer.rms_norm_eps,
        execution_mode,
    )?;
    let (xs_values, updated_cache) =
        run_attention_for_layer_decode(&normed, layer, cache, donor_cache, position, execution_mode)?;
    let mut xs = ActivationRowBuffer::from_values(apply_rms_norm(
        &xs_values,
        &layer.post_attention_layernorm_weight,
        layer.rms_norm_eps,
        execution_mode,
    )?);
    xs = add_row_buffers(&residual, &xs, layer.o_proj_det.is_some())?;

    let residual = xs.clone();
    let normed = apply_rms_norm(
        &xs.values,
        &layer.pre_feedforward_layernorm_weight,
        layer.rms_norm_eps,
        execution_mode,
    )?;
    let (gate_preactivation, up) = match (layer.gate_proj_det.as_ref(), layer.up_proj_det.as_ref())
    {
        (Some(gate_weight), Some(up_weight)) => {
            let quantized_normed = normed.iter().copied().map(f32_to_act).collect::<Vec<_>>();
            let gate_preactivation =
                det_linear_row_from_acts(&quantized_normed, gate_weight.as_ref())?;
            (
                ActivationRowBuffer::from_values(apply_gelu(&gate_preactivation)),
                ActivationRowBuffer::from_acts(det_linear_row_acts_from_acts(
                    &quantized_normed,
                    up_weight.as_ref(),
                )?),
            )
        }
        (gate_weight, up_weight) => {
            let gate_preactivation = match gate_weight {
                Some(weight) => project_linear_row_buffer(
                    &ActivationRowBuffer::from_values(normed.clone()),
                    layer.gate_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => ActivationRowBuffer::from_values(linear_row(
                    &normed,
                    layer.gate_proj.as_ref(),
                )?),
            };
            let up = match up_weight {
                Some(weight) => project_linear_row_buffer(
                    &ActivationRowBuffer::from_values(normed.clone()),
                    layer.up_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None => ActivationRowBuffer::from_values(linear_row(
                    &normed,
                    layer.up_proj.as_ref(),
                )?),
            };
            (
                ActivationRowBuffer::from_values(apply_gelu(&gate_preactivation.values)),
                up,
            )
        }
    };
    let ff_hidden = mul_row_buffers(
        &gate_preactivation,
        &up,
        gate_preactivation.acts.is_some() || up.acts.is_some(),
    )?;
    let ff_out = match layer.down_proj_det.as_ref() {
        Some(weight) => {
            project_linear_row_buffer(&ff_hidden, layer.down_proj.as_ref(), Some(weight.as_ref()))?
        }
        None => ActivationRowBuffer::from_values(linear_row(
            &ff_hidden.values,
            layer.down_proj.as_ref(),
        )?),
    };
    let ff_out_uses_det = ff_out.acts.is_some();
    let ff_out = ActivationRowBuffer::from_values(apply_rms_norm(
        &ff_out.values,
        &layer.post_feedforward_layernorm_weight,
        layer.rms_norm_eps,
        execution_mode,
    )?);
    xs = add_row_buffers(&residual, &ff_out, ff_out_uses_det)?;

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = xs.clone();
        let quantized_xs = ple
            .input_gate_det
            .as_ref()
            .map(|_| quantize_row_to_acts(&xs.values));
        let gated = ActivationRowBuffer::from_values(apply_gelu(&project_linear_row(
            &xs.values,
            quantized_xs.as_deref(),
            ple.input_gate.as_ref(),
            ple.input_gate_det.as_deref(),
        )?));
        let gated = mul_row_buffers(
            &gated,
            &ActivationRowBuffer::from_values(per_layer_input.to_vec()),
            ple.input_gate_det.is_some(),
        )?;
        let projected = project_linear_row_buffer(
            &gated,
            ple.layer_projection.as_ref(),
            ple.layer_projection_det.as_deref(),
        )?;
        let projected_uses_det = projected.acts.is_some();
        let projected = ActivationRowBuffer::from_values(apply_rms_norm(
            &projected.values,
            &ple.post_input_norm_weight,
            layer.rms_norm_eps,
            execution_mode,
        )?);
        xs = add_row_buffers(&residual, &projected, projected_uses_det)?;
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        xs = scale_row_buffer(&xs, layer_scalar, xs.acts.is_some());
    }

    Ok((xs.values, updated_cache))
}

pub fn run_text_layers_prefill(
    input_activations: &[Vec<f32>],
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<ActivationSequence> {
    Ok(run_text_layers_prefill_with_cache(input_activations, model, ple_inputs)?.0)
}

pub fn run_text_layers_prefill_with_cache(
    input_activations: &[Vec<f32>],
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    // let _trace = trace_scope("transformer_state_transition.run_text_layers_prefill");
    if model.layers.is_empty() {
        bail!("transformer prefill requires at least one layer");
    }

    let mut xs = input_activations.to_vec();
    let mut layer_caches = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_sha256s = Vec::with_capacity(model.layers.len());
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let _trace = trace_scope(format!(
            "prefill.layer layer={layer_idx} tokens={} attention={:?} ple={} donor={:?}",
            xs.len(),
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        // trace_event(format!(
        //     "transformer_state_transition.run_text_layers_prefill layer={layer_idx} attention={:?}",
        //     layer.attention_kind
        // ));
        // trace_event(format!(
        //     "transformer_state_transition.prefill_layer_tokens layer={layer_idx} tokens={}",
        //     xs.len()
        // ));
        let per_layer_input = ple_inputs
            .and_then(|inputs| inputs.per_layer_inputs.get(layer_idx))
            .and_then(|input| input.as_deref());
        let donor_cache = resolve_prefill_donor_cache(layer, &layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, layer_cache) =
            run_gemma4_layer_with_cache(
                &xs,
                &resolved_layer,
                per_layer_input,
                donor_cache,
                InferenceExecutionMode::Fp32,
            )?;
        xs = layer_output.activations;
        layer_caches.push(layer_cache);
        completed_layer_output_sha256s.push(layer_output.activations_sha256);
        crate::trace::trace_checkpoint(
            "prefill.layer",
            &json!({
                "next_layer_idx": layer_idx + 1,
                "current_activations": xs.clone(),
                "current_activations_sha256": build_activation_commitment(&xs),
                "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
                "completed_layer_output_sha256s": completed_layer_output_sha256s.clone(),
            }),
        );
        for (token_idx, token_activation) in xs.iter().enumerate() {
            crate::trace::trace_checkpoint(
                &format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}"),
                &json!({
                    "layer_idx": layer_idx,
                    "token_idx": token_idx,
                    "token_count": xs.len(),
                    "token_activation": token_activation,
                }),
            );
        }
    }

    Ok((
        ActivationSequence {
            activations_sha256: build_activation_commitment(&xs),
            activations: xs,
        },
        layer_caches,
    ))
}

pub fn run_text_layers_decode_step(
    input_activation: &[f32],
    token_id: u32,
    model: &Gemma4TransformerModel,
    layer_caches: Vec<LayerKvCache>,
    position: usize,
) -> Result<ActivationSequenceWithCache> {
    // let _trace = trace_scope("transformer_state_transition.run_text_layers_decode_step");
    if model.layers.is_empty() {
        bail!("transformer decode requires at least one layer");
    }
    if layer_caches.len() != model.layers.len() {
        bail!(
            "transformer decode cache count mismatch: {} vs {}",
            layer_caches.len(),
            model.layers.len()
        );
    }

    let mut xs = input_activation.to_vec();
    let mut updated_layer_caches = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_sha256s = Vec::with_capacity(model.layers.len());
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let cache = layer_caches[layer_idx].clone();
        let _trace = trace_scope(format!(
            "decode.layer layer={layer_idx} token={} position={} attention={:?} ple={} donor={:?}",
            token_id,
            position,
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        // trace_event(format!(
        //     "transformer_state_transition.run_text_layers_decode_step layer={layer_idx} attention={:?}",
        //     layer.attention_kind
        // ));
        // trace_event(format!(
        //     "transformer_state_transition.decode_layer_state layer={layer_idx} token={} position={}",
        //     token_id,
        //     position
        // ));
        let per_layer_input = compute_decode_ple_input(
            token_id,
            input_activation,
            layer_idx,
            layer,
            model.ple_global.as_ref(),
            model.rms_norm_eps,
            InferenceExecutionMode::Fp32,
        )?;
        let donor_cache = resolve_decode_donor_cache(layer, &updated_layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, updated_cache) = run_gemma4_layer_decode_with_mode(
            &xs,
            &resolved_layer,
            per_layer_input.as_deref(),
            cache,
            donor_cache,
            position,
            InferenceExecutionMode::Fp32,
        )?;
        xs = layer_output;
        updated_layer_caches.push(updated_cache);
        completed_layer_output_sha256s.push(build_vector_commitment(&xs));
        let mut checkpoint_layer_caches = updated_layer_caches.clone();
        checkpoint_layer_caches.extend(layer_caches.iter().skip(layer_idx + 1).cloned());
        crate::trace::trace_checkpoint(
            &format!("decode.layer_token.layer_{layer_idx}.position_{position}"),
            &json!({
                "token_id": token_id,
                "position": position,
                "next_layer_idx": layer_idx + 1,
                "decode_input_activation": input_activation,
                "decode_input_activation_sha256": build_vector_commitment(input_activation),
                "current_activation": xs.clone(),
                "current_activation_sha256": build_vector_commitment(&xs),
                "layer_caches": crate::trace::serialize_layer_caches(&checkpoint_layer_caches),
                "completed_layer_output_sha256s": completed_layer_output_sha256s.clone(),
            }),
        );
    }

    Ok(ActivationSequenceWithCache {
        activation_state: ActivationSequence {
            activations_sha256: build_activation_commitment(&[xs.clone()]),
            activations: vec![xs],
        },
        layer_caches: updated_layer_caches,
    })
}

fn resolve_prefill_donor_cache<'a>(
    layer: &Gemma4LayerWeights,
    layer_caches: &'a [LayerKvCache],
    layer_idx: usize,
) -> Result<Option<&'a LayerKvCache>> {
    layer
        .kv_shared_layer_index
        .map(|donor_idx| {
            if donor_idx >= layer_idx {
                bail!(
                    "transformer prefill layer {layer_idx} cannot share KV with non-prior donor {donor_idx}"
                );
            }
            layer_caches.get(donor_idx).ok_or_else(|| {
                anyhow!("transformer prefill donor cache {donor_idx} missing for layer {layer_idx}")
            })
        })
        .transpose()
}

fn resolve_decode_donor_cache<'a>(
    layer: &Gemma4LayerWeights,
    updated_layer_caches: &'a [LayerKvCache],
    layer_idx: usize,
) -> Result<Option<&'a LayerKvCache>> {
    layer
        .kv_shared_layer_index
        .map(|donor_idx| {
            if donor_idx >= layer_idx {
                bail!(
                    "transformer decode layer {layer_idx} cannot share KV with non-prior donor {donor_idx}"
                );
            }
            updated_layer_caches.get(donor_idx).ok_or_else(|| {
                anyhow!("transformer decode donor cache {donor_idx} missing for layer {layer_idx}")
            })
        })
        .transpose()
}

pub fn apply_final_norm(
    input_activations: &[Vec<f32>],
    weight: &[f32],
    eps: f32,
) -> Result<ActivationSequence> {
    apply_final_norm_with_mode(input_activations, weight, eps, InferenceExecutionMode::Fp32)
}

pub fn apply_final_norm_with_mode(
    input_activations: &[Vec<f32>],
    weight: &[f32],
    eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequence> {
    // let _trace = trace_scope("transformer_state_transition.apply_final_norm");
    let activations = apply_rms_norm_to_sequence(input_activations, weight, eps, execution_mode)?;
    Ok(ActivationSequence {
        activations_sha256: build_activation_commitment(&activations),
        activations,
    })
}

pub fn select_final_position(input_activations: &[Vec<f32>]) -> Result<Vec<f32>> {
    // let _trace = trace_scope("transformer_state_transition.select_final_position");
    input_activations.last().cloned().ok_or_else(|| {
        anyhow!("transformer final-position selection requires at least one activation row")
    })
}

pub fn project_to_logits(
    last_hidden_state: &[f32],
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    execution_mode: InferenceExecutionMode,
) -> Result<Vec<f32>> {
    // let _trace = trace_scope("transformer_state_transition.project_to_logits");
    let weight = match projection {
        Gemma4LogitsProjection::UntiedLmHead { weight, .. }
        | Gemma4LogitsProjection::TiedEmbedding(weight) => weight,
    };
    validate_vector_width(last_hidden_state, weight.cols, "logits projection input")?;

    if execution_mode == InferenceExecutionMode::Deterministic {
        let quantized_hidden_state = last_hidden_state
            .iter()
            .copied()
            .map(f32_to_act)
            .collect::<Vec<_>>();
        match projection {
            Gemma4LogitsProjection::UntiedLmHead {
                det_weight: Some(det_weight),
                ..
            } => return det_linear_row_from_acts(&quantized_hidden_state, det_weight.as_ref()),
            Gemma4LogitsProjection::TiedEmbedding(_) => {
                if let Some(embedding_source) = embedding_source {
                    if let Some(det_weight) =
                        crate::io::materialize_det_num_embedding_matrix(embedding_source)?
                    {
                        return det_linear_row_from_acts(
                            &quantized_hidden_state,
                            det_weight.as_ref(),
                        );
                    }
                }
            }
            Gemma4LogitsProjection::UntiedLmHead {
                det_weight: None, ..
            } => {}
        }
    }

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
    // let _trace = trace_scope("transformer_state_transition.apply_final_logit_softcapping");
    logits
        .iter()
        .map(|logit| (logit / softcap).tanh() * softcap)
        .collect()
}

pub fn extract_prefill_logits(logits: &[f32]) -> PrefillLogits {
    // let _trace = trace_scope("transformer_state_transition.extract_prefill_logits");
    PrefillLogits {
        logits: logits.to_vec(),
        final_logits_sha256: build_vector_commitment(logits),
    }
}

pub fn project_decode_hidden_to_logits(
    hidden_state: &[f32],
    final_norm_weight: &[f32],
    rms_norm_eps: f32,
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    execution_mode: InferenceExecutionMode,
    final_logit_softcapping: Option<f32>,
) -> Result<PrefillLogits> {
    // let _trace = trace_scope("transformer_state_transition.project_decode_hidden_to_logits");
    let normalized =
        apply_rms_norm(hidden_state, final_norm_weight, rms_norm_eps, execution_mode)?;
    let mut logits = project_to_logits(&normalized, projection, embedding_source, execution_mode)?;
    if let Some(softcap) = final_logit_softcapping {
        logits = apply_final_logit_softcapping(&logits, softcap);
    }
    Ok(extract_prefill_logits(&logits))
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

fn run_attention_for_layer_with_cache(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    match layer.attention_kind {
        Gemma4AttentionKind::Sliding => {
            run_sliding_attention(inputs, layer, donor_cache, execution_mode)
        }
        Gemma4AttentionKind::Full => run_full_attention(inputs, layer, donor_cache, execution_mode),
    }
}

fn run_attention_for_layer_decode(
    input: &[f32],
    layer: &ResolvedGemma4LayerWeights,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<f32>, LayerKvCache)> {
    match layer.attention_kind {
        Gemma4AttentionKind::Sliding => {
            let sliding_window = layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?;
            run_causal_attention_decode(
                input,
                layer,
                cache,
                donor_cache,
                position,
                Some(sliding_window),
                layer.cache_sliding_window,
                execution_mode,
            )
        }
        Gemma4AttentionKind::Full => run_causal_attention_decode(
            input,
            layer,
            cache,
            donor_cache,
            position,
            None,
            None,
            execution_mode,
        ),
    }
}

fn run_sliding_attention(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    let sliding_window = layer
        .sliding_window
        .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?;
    run_causal_attention(
        inputs,
        layer,
        Some(sliding_window),
        layer.cache_sliding_window,
        donor_cache,
        execution_mode,
    )
}

fn run_full_attention(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    run_causal_attention(inputs, layer, None, None, donor_cache, execution_mode)
}

fn run_causal_attention(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_causal_attention attention={:?}",
    //     layer.attention_kind
    // ));
    let seq_len = inputs.len();
    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow::anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }

    let quantized_inputs =
        if layer.q_proj_det.is_some() || layer.k_proj_det.is_some() || layer.v_proj_det.is_some() {
            Some(
                inputs
                    .iter()
                    .map(|row| quantize_row_to_acts(row))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
    let q_projected = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.q_proj");
        project_linear_sequence(
            inputs,
            quantized_inputs.as_deref(),
            layer.q_proj.as_ref(),
            layer.q_proj_det.as_deref(),
        )?
    };
    let raw_k = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.k_proj");
        project_linear_sequence(
            inputs,
            quantized_inputs.as_deref(),
            layer.k_proj.as_ref(),
            layer.k_proj_det.as_deref(),
        )?
    };
    let raw_v = if let Some(v_proj) = &layer.v_proj {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.v_proj");
        project_linear_sequence(
            inputs,
            quantized_inputs.as_deref(),
            v_proj.as_ref(),
            layer.v_proj_det.as_deref(),
        )?
    } else if layer.attention_k_eq_v {
        // trace_event("transformer_state_transition.run_causal_attention.k_eq_v_reuse");
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };
    let mut q = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.reshape_q");
        reshape_sequence_heads(&q_projected, layer.num_heads, layer.head_dim)?
    };
    let mut k = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.reshape_k");
        reshape_sequence_heads(&raw_k, layer.num_kv_heads, layer.head_dim)?
    };
    let mut v = {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.reshape_v");
        reshape_sequence_heads(&raw_v, layer.num_kv_heads, layer.head_dim)?
    };

    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.q_rms_norm");
        apply_head_rms_norm(&mut q, &layer.q_norm_weight, layer.rms_norm_eps, execution_mode)?;
    }
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.k_rms_norm");
        apply_head_rms_norm(&mut k, &layer.k_norm_weight, layer.rms_norm_eps, execution_mode)?;
    }
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.v_rms_norm");
        apply_value_rms_norm(&mut v, layer.rms_norm_eps, execution_mode)?;
    }

    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.q_rope");
        apply_rope(
            &mut q,
            layer.partial_rotary_dim,
            layer.rope_freq_base_dim,
            layer.rope_base,
        );
    }
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.k_rope");
        apply_rope(
            &mut k,
            layer.partial_rotary_dim,
            layer.rope_freq_base_dim,
            layer.rope_base,
        );
    }

    let layer_cache = if donor_cache.is_some() {
        LayerKvCache::new(layer.num_kv_heads)
    } else {
        build_layer_kv_cache(&k, &v, cache_window)
    };

    let head_outputs = (0..layer.num_heads)
        .into_par_iter()
        .map(|head_idx| {
            // let _trace = trace_scope(format!("transformer_state_transition.run_causal_attention.head={head_idx}"));
            let kv_head_idx = head_idx / kv_groups;
            let mut outputs = vec![vec![0.0; layer.head_dim]; seq_len];
            for (query_idx, output) in outputs.iter_mut().enumerate() {
                let start = attention_window
                    .map(|window| query_idx.saturating_add(1).saturating_sub(window))
                    .unwrap_or(0);
                if let Some(donor_cache) = donor_cache {
                    let logits = (start..=query_idx)
                        .map(|key_idx| {
                            dot(
                                &q[head_idx][query_idx],
                                &donor_cache.keys[kv_head_idx][key_idx],
                            )
                        })
                        .collect::<Vec<_>>();
                    let weights = softmax(&logits);

                    for (weight, key_idx) in weights.into_iter().zip(start..=query_idx) {
                        for (dim_idx, value) in output.iter_mut().enumerate() {
                            *value += weight * donor_cache.values[kv_head_idx][key_idx][dim_idx];
                        }
                    }
                } else {
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

    let quantized_combined_heads = layer
        .o_proj_det
        .as_ref()
        .map(|_| quantize_sequence_to_acts(&combined_heads));
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.o_proj");
        Ok((
            project_linear_sequence(
                &combined_heads,
                quantized_combined_heads.as_deref(),
                layer.o_proj.as_ref(),
                layer.o_proj_det.as_deref(),
            )?,
            layer_cache,
        ))
    }
}

fn run_causal_attention_decode(
    input: &[f32],
    layer: &ResolvedGemma4LayerWeights,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<f32>, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "transformer_state_transition.run_causal_attention_decode attention={:?}",
    //     layer.attention_kind
    // ));
    validate_vector_width(input, layer.hidden_size, "decode attention input")?;

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

    let quantized_input =
        if layer.q_proj_det.is_some() || layer.k_proj_det.is_some() || layer.v_proj_det.is_some() {
            Some(quantize_row_to_acts(input))
        } else {
            None
        };
    let q_projected = project_linear_row(
        input,
        quantized_input.as_deref(),
        layer.q_proj.as_ref(),
        layer.q_proj_det.as_deref(),
    )?;
    let raw_k = project_linear_row(
        input,
        quantized_input.as_deref(),
        layer.k_proj.as_ref(),
        layer.k_proj_det.as_deref(),
    )?;
    let raw_v = if let Some(v_proj) = &layer.v_proj {
        project_linear_row(
            input,
            quantized_input.as_deref(),
            v_proj.as_ref(),
            layer.v_proj_det.as_deref(),
        )?
    } else if layer.attention_k_eq_v {
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };

    let mut q = reshape_row_heads(&q_projected, layer.num_heads, layer.head_dim)?;
    let mut k = reshape_row_heads(&raw_k, layer.num_kv_heads, layer.head_dim)?;
    let mut v = reshape_row_heads(&raw_v, layer.num_kv_heads, layer.head_dim)?;

    apply_head_rms_norm_row(&mut q, &layer.q_norm_weight, layer.rms_norm_eps, execution_mode)?;
    apply_head_rms_norm_row(&mut k, &layer.k_norm_weight, layer.rms_norm_eps, execution_mode)?;
    apply_value_rms_norm_row(&mut v, layer.rms_norm_eps, execution_mode)?;
    apply_rope_to_rows(
        &mut q,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base,
        position,
    );
    apply_rope_to_rows(
        &mut k,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base,
        position,
    );

    let updated_cache = if donor_cache.is_some() {
        cache
    } else {
        append_kv_cache(cache, &k, &v, cache_window)?
    };
    let attention_cache = donor_cache.unwrap_or(&updated_cache);
    let mut combined_heads = vec![0.0; layer.num_heads * layer.head_dim];
    for head_idx in 0..layer.num_heads {
        let kv_head_idx = head_idx / kv_groups;
        let key_start = attention_window
            .map(|window| {
                attention_cache.keys[kv_head_idx]
                    .len()
                    .saturating_sub(window)
            })
            .unwrap_or(0);
        let logits = attention_cache.keys[kv_head_idx]
            .iter()
            .skip(key_start)
            .map(|key_row| dot(&q[head_idx], key_row))
            .collect::<Vec<_>>();
        let weights = softmax(&logits);
        let mut output = vec![0.0; layer.head_dim];
        for (weight, value_row) in weights
            .into_iter()
            .zip(attention_cache.values[kv_head_idx].iter().skip(key_start))
        {
            for (dim_idx, value) in output.iter_mut().enumerate() {
                *value += weight * value_row[dim_idx];
            }
        }
        let dst = &mut combined_heads[head_idx * layer.head_dim..(head_idx + 1) * layer.head_dim];
        dst.copy_from_slice(&output);
    }

    let quantized_combined_heads = layer
        .o_proj_det
        .as_ref()
        .map(|_| quantize_row_to_acts(&combined_heads));
    Ok((
        project_linear_row(
            &combined_heads,
            quantized_combined_heads.as_deref(),
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

    fn from_acts(acts: Vec<Act>) -> Self {
        let values = acts.iter().copied().map(act_to_f32).collect();
        Self {
            values,
            acts: Some(acts),
        }
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
                .collect::<Vec<_>>())
        })
        .collect()
}

fn add_rows(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
    if lhs.len() != rhs.len() {
        bail!("row width mismatch: {} vs {}", lhs.len(), rhs.len());
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(lhs_value, rhs_value)| lhs_value + rhs_value)
        .collect())
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

fn elementwise_mul_rows(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
    if lhs.len() != rhs.len() {
        bail!("row width mismatch: {} vs {}", lhs.len(), rhs.len());
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
        .collect())
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

fn scale_sequences(values: &[Vec<f32>], scalar: f32) -> Vec<Vec<f32>> {
    values
        .iter()
        .map(|row| row.iter().map(|value| value * scalar).collect())
        .collect()
}

fn scale_rows(values: &[f32], scalar: f32) -> Vec<f32> {
    values.iter().map(|value| value * scalar).collect()
}

fn sequence_buffer_acts(buffer: &ActivationSequenceBuffer) -> Vec<Vec<Act>> {
    buffer
        .acts
        .clone()
        .unwrap_or_else(|| quantize_sequence_to_acts(&buffer.values))
}

fn row_buffer_acts(buffer: &ActivationRowBuffer) -> Vec<Act> {
    buffer
        .acts
        .clone()
        .unwrap_or_else(|| quantize_row_to_acts(&buffer.values))
}

fn add_sequence_buffers(
    lhs: &ActivationSequenceBuffer,
    rhs: &ActivationSequenceBuffer,
    deterministic: bool,
) -> Result<ActivationSequenceBuffer> {
    if deterministic {
        Ok(ActivationSequenceBuffer::from_acts(add_act_sequences(
            &sequence_buffer_acts(lhs),
            &sequence_buffer_acts(rhs),
        )?))
    } else {
        Ok(ActivationSequenceBuffer::from_values(add_sequences(
            &lhs.values,
            &rhs.values,
        )?))
    }
}

fn add_row_buffers(
    lhs: &ActivationRowBuffer,
    rhs: &ActivationRowBuffer,
    deterministic: bool,
) -> Result<ActivationRowBuffer> {
    if deterministic {
        Ok(ActivationRowBuffer::from_acts(add_act_rows(
            &row_buffer_acts(lhs),
            &row_buffer_acts(rhs),
        )?))
    } else {
        Ok(ActivationRowBuffer::from_values(add_rows(&lhs.values, &rhs.values)?))
    }
}

fn mul_sequence_buffers(
    lhs: &ActivationSequenceBuffer,
    rhs: &ActivationSequenceBuffer,
    deterministic: bool,
) -> Result<ActivationSequenceBuffer> {
    if deterministic {
        Ok(ActivationSequenceBuffer::from_acts(mul_act_sequences(
            &sequence_buffer_acts(lhs),
            &sequence_buffer_acts(rhs),
        )?))
    } else {
        Ok(ActivationSequenceBuffer::from_values(elementwise_mul_sequences(
            &lhs.values,
            &rhs.values,
        )?))
    }
}

fn mul_row_buffers(
    lhs: &ActivationRowBuffer,
    rhs: &ActivationRowBuffer,
    deterministic: bool,
) -> Result<ActivationRowBuffer> {
    if deterministic {
        Ok(ActivationRowBuffer::from_acts(mul_act_rows(
            &row_buffer_acts(lhs),
            &row_buffer_acts(rhs),
        )?))
    } else {
        Ok(ActivationRowBuffer::from_values(elementwise_mul_rows(
            &lhs.values,
            &rhs.values,
        )?))
    }
}

fn scale_sequence_buffer(
    values: &ActivationSequenceBuffer,
    scalar: f32,
    deterministic: bool,
) -> ActivationSequenceBuffer {
    if deterministic {
        ActivationSequenceBuffer::from_acts(scale_act_sequences(
            &sequence_buffer_acts(values),
            f32_to_act(scalar),
        ))
    } else {
        ActivationSequenceBuffer::from_values(scale_sequences(&values.values, scalar))
    }
}

fn scale_row_buffer(values: &ActivationRowBuffer, scalar: f32, deterministic: bool) -> ActivationRowBuffer {
    if deterministic {
        ActivationRowBuffer::from_acts(scale_act_rows(&row_buffer_acts(values), f32_to_act(scalar)))
    } else {
        ActivationRowBuffer::from_values(scale_rows(&values.values, scalar))
    }
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

fn linear_row(input: &[f32], weight: &MatrixF32) -> Result<Vec<f32>> {
    validate_vector_width(input, weight.cols, "linear input")?;
    let mut output = vec![0.0; weight.rows];
    for (row_idx, value) in output.iter_mut().enumerate() {
        let row_offset = row_idx * weight.cols;
        let mut sum = 0.0;
        for col_idx in 0..weight.cols {
            sum += input[col_idx] * weight.values[row_offset + col_idx];
        }
        *value = sum;
    }
    Ok(output)
}

fn project_linear_sequence(
    inputs: &[Vec<f32>],
    quantized_inputs: Option<&[Vec<Act>]>,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<Vec<Vec<f32>>> {
    match det_weight {
        Some(det_weight) => match quantized_inputs {
            Some(quantized_inputs) => det_linear_sequence_from_acts(quantized_inputs, det_weight),
            None => det_linear_sequence(inputs, det_weight),
        },
        None => linear_sequence(inputs, weight),
    }
}

fn project_linear_row(
    input: &[f32],
    quantized_input: Option<&[Act]>,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<Vec<f32>> {
    match det_weight {
        Some(det_weight) => match quantized_input {
            Some(quantized_input) => det_linear_row_from_acts(quantized_input, det_weight),
            None => det_linear_row(input, det_weight),
        },
        None => linear_row(input, weight),
    }
}

fn project_linear_sequence_buffer(
    inputs: &ActivationSequenceBuffer,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<ActivationSequenceBuffer> {
    match det_weight {
        Some(det_weight) => Ok(ActivationSequenceBuffer::from_acts(
            det_linear_sequence_acts_from_acts(&sequence_buffer_acts(inputs), det_weight)?,
        )),
        None => Ok(ActivationSequenceBuffer::from_values(linear_sequence(
            &inputs.values,
            weight,
        )?)),
    }
}

fn project_linear_row_buffer(
    input: &ActivationRowBuffer,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<ActivationRowBuffer> {
    match det_weight {
        Some(det_weight) => Ok(ActivationRowBuffer::from_acts(det_linear_row_acts_from_acts(
            &row_buffer_acts(input),
            det_weight,
        )?)),
        None => Ok(ActivationRowBuffer::from_values(linear_row(&input.values, weight)?)),
    }
}

fn quantize_sequence_to_acts(inputs: &[Vec<f32>]) -> Vec<Vec<Act>> {
    inputs
        .iter()
        .map(|row| quantize_row_to_acts(row))
        .collect::<Vec<_>>()
}

fn quantize_row_to_acts(input: &[f32]) -> Vec<Act> {
    input.iter().copied().map(f32_to_act).collect::<Vec<_>>()
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

fn det_linear_row_acts_from_acts(quantized_input: &[Act], weight: &DetNumMatrix) -> Result<Vec<Act>> {
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

fn det_linear_sequence(inputs: &[Vec<f32>], weight: &DetNumMatrix) -> Result<Vec<Vec<f32>>> {
    validate_sequence_width(inputs, weight.cols, "deterministic linear input")?;
    inputs
        .par_iter()
        .map(|input| det_linear_row(input, weight))
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

fn det_linear_sequence_from_acts(
    quantized_inputs: &[Vec<Act>],
    weight: &DetNumMatrix,
) -> Result<Vec<Vec<f32>>> {
    det_linear_sequence_acts_from_acts(quantized_inputs, weight).map(|acts| {
        acts.into_iter()
            .map(|row| row.into_iter().map(act_to_f32).collect())
            .collect()
    })
}

fn det_linear_row(input: &[f32], weight: &DetNumMatrix) -> Result<Vec<f32>> {
    validate_vector_width(input, weight.cols, "deterministic linear input")?;

    let quantized_input = input.iter().copied().map(f32_to_act).collect::<Vec<_>>();
    det_linear_row_from_acts(&quantized_input, weight)
}

fn det_linear_row_from_acts(quantized_input: &[Act], weight: &DetNumMatrix) -> Result<Vec<f32>> {
    det_linear_row_acts_from_acts(quantized_input, weight)
        .map(|acts| acts.into_iter().map(act_to_f32).collect())
}

fn apply_rms_norm_to_sequence(
    inputs: &[Vec<f32>],
    weight: &[f32],
    eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<Vec<Vec<f32>>> {
    inputs
        .par_iter()
        .map(|row| apply_rms_norm(row, weight, eps, execution_mode))
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

fn apply_rms_norm(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<Vec<f32>> {
    if input.len() != weight.len() {
        bail!(
            "rms norm width mismatch: {} vs {}",
            input.len(),
            weight.len()
        );
    }

    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
            let scale = (mean_square + eps).sqrt().recip();
            Ok(input
                .iter()
                .zip(weight)
                .map(|(value, norm_weight)| value * scale * norm_weight)
                .collect())
        }
        InferenceExecutionMode::Deterministic => {
            let quantized_input = input.iter().copied().map(f32_to_act).collect::<Vec<_>>();
            let quantized_weight = weight
                .iter()
                .copied()
                .map(crate::shared::det_num::f32_to_wgt)
                .collect::<Vec<_>>();
            Ok(det_rms_norm(&quantized_input, &quantized_weight, f32_to_acc(eps))
                .into_iter()
                .map(act_to_f32)
                .collect())
        }
    }
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

fn apply_head_rms_norm(
    heads: &mut [Vec<Vec<f32>>],
    weight: &[f32],
    eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    heads.par_iter_mut().try_for_each(|head| -> Result<()> {
        for row in head {
            *row = apply_rms_norm(row, weight, eps, execution_mode)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn apply_head_rms_norm_row(
    heads: &mut [Vec<f32>],
    weight: &[f32],
    eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    for head in heads {
        *head = apply_rms_norm(head, weight, eps, execution_mode)?;
    }
    Ok(())
}

fn apply_value_rms_norm(
    heads: &mut [Vec<Vec<f32>>],
    eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    heads.par_iter_mut().try_for_each(|head| -> Result<()> {
        for row in head {
            match execution_mode {
                InferenceExecutionMode::Fp32 => {
                    let mean_square =
                        row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
                    let scale = (mean_square + eps).sqrt().recip();
                    for value in row {
                        *value *= scale;
                    }
                }
                InferenceExecutionMode::Deterministic => {
                    let quantized = row.iter().copied().map(f32_to_act).collect::<Vec<_>>();
                    *row = det_value_rms_norm(&quantized, f32_to_acc(eps))
                        .into_iter()
                        .map(act_to_f32)
                        .collect();
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

fn apply_value_rms_norm_row(
    heads: &mut [Vec<f32>],
    eps: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    for head in heads {
        match execution_mode {
            InferenceExecutionMode::Fp32 => {
                let mean_square =
                    head.iter().map(|value| value * value).sum::<f32>() / head.len() as f32;
                let scale = (mean_square + eps).sqrt().recip();
                for value in head {
                    *value *= scale;
                }
            }
            InferenceExecutionMode::Deterministic => {
                let quantized = head.iter().copied().map(f32_to_act).collect::<Vec<_>>();
                *head = det_value_rms_norm(&quantized, f32_to_acc(eps))
                    .into_iter()
                    .map(act_to_f32)
                    .collect();
            }
        }
    }
    Ok(())
}

fn apply_rope(heads: &mut [Vec<Vec<f32>>], rotary_dim: usize, freq_base_dim: usize, base: f32) {
    apply_rope_with_offset(heads, rotary_dim, freq_base_dim, base, 0);
}

fn apply_rope_with_offset(
    heads: &mut [Vec<Vec<f32>>],
    rotary_dim: usize,
    freq_base_dim: usize,
    base: f32,
    position_offset: usize,
) {
    if rotary_dim == 0 {
        return;
    }
    let half_dim = rotary_dim / 2;
    for head in heads {
        for (position, row) in head.iter_mut().enumerate() {
            let original = row.clone();
            for dim_idx in 0..half_dim {
                let absolute_position = position_offset + position;
                let angle = absolute_position as f32
                    / base.powf((2 * dim_idx) as f32 / freq_base_dim as f32);
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

fn apply_rope_to_rows(
    heads: &mut [Vec<f32>],
    rotary_dim: usize,
    freq_base_dim: usize,
    base: f32,
    position: usize,
) {
    if rotary_dim == 0 {
        return;
    }
    let half_dim = rotary_dim / 2;
    for row in heads {
        let original = row.clone();
        for dim_idx in 0..half_dim {
            let angle = position as f32 / base.powf((2 * dim_idx) as f32 / freq_base_dim as f32);
            let cos = angle.cos();
            let sin = angle.sin();
            let lhs = original[dim_idx];
            let rhs = original[dim_idx + half_dim];
            row[dim_idx] = lhs * cos - rhs * sin;
            row[dim_idx + half_dim] = rhs * cos + lhs * sin;
        }
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
        .sum()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / sum).collect()
}

fn build_layer_kv_cache(
    keys: &[Vec<Vec<f32>>],
    values: &[Vec<Vec<f32>>],
    sliding_window: Option<usize>,
) -> LayerKvCache {
    let retained = sliding_window.map_or(0, |window| {
        keys.first()
            .map_or(0, |head| head.len().saturating_sub(window))
    });
    LayerKvCache {
        keys: keys
            .iter()
            .map(|head| head[retained..].iter().cloned().collect())
            .collect(),
        values: values
            .iter()
            .map(|head| head[retained..].iter().cloned().collect())
            .collect(),
    }
}

pub fn append_kv_cache(
    mut cache: LayerKvCache,
    new_keys: &[Vec<f32>],
    new_values: &[Vec<f32>],
    sliding_window: Option<usize>,
) -> Result<LayerKvCache> {
    if new_keys.len() != cache.keys.len() || new_values.len() != cache.values.len() {
        bail!(
            "layer cache append head count mismatch: cache {} keys {} values {}",
            cache.keys.len(),
            new_keys.len(),
            new_values.len()
        );
    }

    for ((head_keys, head_values), (new_key, new_value)) in cache
        .keys
        .iter_mut()
        .zip(cache.values.iter_mut())
        .zip(new_keys.iter().zip(new_values))
    {
        head_keys.push_back(new_key.clone());
        head_values.push_back(new_value.clone());
        if let Some(window) = sliding_window {
            while head_keys.len() > window {
                head_keys.pop_front();
                head_values.pop_front();
            }
        }
    }

    Ok(cache)
}

fn apply_gelu_to_sequence(inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    inputs
        .iter()
        .map(|row| row.iter().map(|value| gelu_pytorch_tanh(*value)).collect())
        .collect()
}

fn apply_gelu(inputs: &[f32]) -> Vec<f32> {
    inputs
        .iter()
        .map(|value| gelu_pytorch_tanh(*value))
        .collect()
}

fn gelu_pytorch_tanh(value: f32) -> f32 {
    let inner = std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value.powi(3));
    0.5 * value * (1.0 + inner.tanh())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::{Arc, Mutex},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        append_kv_cache, apply_final_norm, apply_final_norm_with_mode, apply_head_rms_norm,
        apply_rms_norm_to_sequence, apply_rope_to_rows, apply_value_rms_norm,
        compute_decode_ple_input, compute_prefill_ple_inputs, det_linear_row,
        det_linear_row_from_acts, det_linear_sequence, embed_input_tokens, extract_prefill_logits,
        project_decode_hidden_to_logits, project_to_logits, run_causal_attention,
        run_causal_attention_decode, run_gemma4_layer, run_gemma4_layer_decode,
        run_text_layers_decode_step, run_text_layers_prefill, run_text_layers_prefill_with_cache,
    };
    use crate::shared::det_num::{
        act_to_f32, f32_to_wgt, wgt_to_le_bytes, Act, DET_NUM_SPEC_VERSION,
        DET_WGT_ARTIFACT_FORMAT_VERSION, DET_WGT_ARTIFACT_MAGIC,
    };
    use crate::shared::input::InferenceExecutionMode;
    use crate::shared::transformer::{
        DetNumMatrix, DetNumTensorSliceSource, EmbeddingTable, Gemma4AttentionKind,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4PleGlobalWeights, Gemma4PleLayerWeights,
        Gemma4TransformerModel, GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32,
        ResolvedGemma4LayerWeights, ResolvedGemma4PleLayerWeights,
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
    fn embed_input_tokens_with_mode_requantizes_scale_on_deterministic_path() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![1.0 / 65_536.0]],
            scale: 0.5,
        };

        let fp32 = embed_input_tokens(&[0], &embedding_table).expect("fp32 embedding should succeed");
        let det = super::embed_input_tokens_with_mode(
            &[0],
            &embedding_table,
            InferenceExecutionMode::Deterministic,
        )
        .expect("det embedding should succeed");

        assert_eq!(det.activations, vec![vec![0.0]]);
        assert!(fp32.activations[0][0] > det.activations[0][0]);
    }

    #[test]
    fn apply_rope_to_rows_uses_full_head_dim_for_frequency_base() {
        let mut heads = vec![vec![0.0, 1.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0]];

        apply_rope_to_rows(&mut heads, 4, 8, 16.0, 1);

        assert!((heads[0][1] - 0.87758255).abs() < 1e-6);
        assert!((heads[0][3] - 0.47942555).abs() < 1e-6);
        assert_eq!(heads[0][4..], [9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn det_linear_row_matches_exact_q16_dot_product() {
        let weight = DetNumMatrix {
            rows: 1,
            cols: 2,
            values: vec![Act::from_num(2).to_bits(), Act::from_num(-1).to_bits()],
        };

        let output = det_linear_row(&[1.5, -0.5], &weight).unwrap();

        assert_eq!(output, vec![3.5]);
    }

    #[test]
    fn det_linear_row_uses_ties_to_even_requantization() {
        let weight = DetNumMatrix {
            rows: 1,
            cols: 1,
            values: vec![Act::from_num(0.5).to_bits()],
        };

        let rounded_down = det_linear_row(&[1.0 / 65_536.0], &weight).unwrap();
        let rounded_up = det_linear_row(&[3.0 / 65_536.0], &weight).unwrap();

        assert_eq!(rounded_down, vec![0.0]);
        assert_eq!(rounded_up, vec![super::act_to_f32(Act::from_bits(2))]);
    }

    #[test]
    fn det_linear_row_from_acts_matches_det_linear_row() {
        let weight = DetNumMatrix {
            rows: 2,
            cols: 3,
            values: vec![
                Act::from_num(0.5).to_bits(),
                Act::from_num(-1.25).to_bits(),
                Act::from_num(2.0).to_bits(),
                Act::from_num(-0.75).to_bits(),
                Act::from_num(0.125).to_bits(),
                Act::from_num(1.5).to_bits(),
            ],
        };
        let input = [1.5, -0.5, 0.25];
        let quantized_input = input
            .iter()
            .copied()
            .map(crate::shared::det_num::f32_to_act)
            .collect::<Vec<_>>();

        let from_f32 = det_linear_row(&input, &weight).unwrap();
        let from_acts = det_linear_row_from_acts(&quantized_input, &weight).unwrap();

        assert_eq!(from_acts, from_f32);
    }

    #[test]
    fn det_linear_sequence_matches_row_by_row_results() {
        let weight = DetNumMatrix {
            rows: 2,
            cols: 3,
            values: vec![
                Act::from_num(0.5).to_bits(),
                Act::from_num(-1.0).to_bits(),
                Act::from_num(0.25).to_bits(),
                Act::from_num(-0.75).to_bits(),
                Act::from_num(1.5).to_bits(),
                Act::from_num(2.0).to_bits(),
            ],
        };
        let inputs = vec![
            vec![1.0, -0.5, 0.25],
            vec![0.0, 2.0, -1.0],
            vec![-1.5, 0.75, 0.5],
        ];

        let sequence_output = det_linear_sequence(&inputs, &weight).unwrap();
        let row_outputs = inputs
            .iter()
            .map(|row| det_linear_row(row, &weight))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(sequence_output, row_outputs);
    }

    #[test]
    fn det_linear_row_differs_from_legacy_linear_row_on_rounding_boundaries() {
        let det_weight = DetNumMatrix {
            rows: 1,
            cols: 1,
            values: vec![Act::from_num(0.5).to_bits()],
        };
        let fp32_weight = MatrixF32 {
            rows: 1,
            cols: 1,
            values: vec![0.5],
        };

        let det_output = det_linear_row(&[1.0 / 65_536.0], &det_weight).unwrap();
        let fp32_output = super::linear_row(&[1.0 / 65_536.0], &fp32_weight).unwrap();

        assert_eq!(det_output, vec![0.0]);
        assert!(fp32_output[0] > det_output[0]);
    }

    #[test]
    fn add_row_buffers_requantize_float_inputs_on_deterministic_path() {
        let lhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);
        let rhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);

        let det = super::add_row_buffers(&lhs, &rhs, true).unwrap();
        let fp32 = super::add_rows(&lhs.values, &rhs.values).unwrap();

        assert_eq!(det.values, vec![0.0]);
        assert!(fp32[0] > det.values[0]);
    }

    #[test]
    fn mul_row_buffers_requantize_float_inputs_on_deterministic_path() {
        let lhs = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);
        let rhs = super::ActivationRowBuffer::from_values(vec![0.5]);

        let det = super::mul_row_buffers(&lhs, &rhs, true).unwrap();
        let fp32 = super::elementwise_mul_rows(&lhs.values, &rhs.values).unwrap();

        assert_eq!(det.values, vec![0.0]);
        assert!(fp32[0] > det.values[0]);
    }

    #[test]
    fn scale_row_buffer_requantize_float_inputs_on_deterministic_path() {
        let values = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);

        let det = super::scale_row_buffer(&values, 0.5, true);
        let fp32 = super::scale_row_buffer(&values, 0.5, false);

        assert_eq!(det.values, vec![0.0]);
        assert!(fp32.values[0] > det.values[0]);
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
            cache_sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: None,
            layer_scalar: None,
        };

        let resolved_layer = crate::io::resolve_layer_weights(&layer).expect("resolve layer");
        let output =
            run_gemma4_layer(&activations, &resolved_layer, None).expect("layer should succeed");

        assert_eq!(output.activations, activations);
    }

    #[test]
    fn run_gemma4_layer_uses_det_up_proj_before_hidden_mul() {
        let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            cache_sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: MatrixF32 {
                rows: 8,
                cols: 4,
                values: vec![
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            }
            .into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: MatrixF32 {
                rows: 4,
                cols: 8,
                values: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            }
            .into(),
            ple: None,
            layer_scalar: None,
        };

        let resolved_without_det =
            crate::io::resolve_layer_weights(&layer).expect("resolve fp32-only layer");
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.up_proj_det = Some(Arc::new(DetNumMatrix {
            rows: 8,
            cols: 4,
            values: vec![
                Act::from_num(0.5).to_bits(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }));

        let without_det =
            run_gemma4_layer(&activations, &resolved_without_det, None).expect("run fp32 path");
        let with_det =
            run_gemma4_layer(&activations, &resolved_with_det, None).expect("run det up_proj");

        assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
        assert!(with_det.activations[0][0] > without_det.activations[0][0]);
    }

    #[test]
    fn run_gemma4_layer_uses_det_gate_proj_before_gelu_and_hidden_mul() {
        let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            cache_sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: MatrixF32 {
                rows: 8,
                cols: 4,
                values: vec![
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            }
            .into(),
            down_proj: MatrixF32 {
                rows: 4,
                cols: 8,
                values: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            }
            .into(),
            ple: None,
            layer_scalar: None,
        };

        let resolved_without_det =
            crate::io::resolve_layer_weights(&layer).expect("resolve fp32-only layer");
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.gate_proj_det = Some(Arc::new(DetNumMatrix {
            rows: 8,
            cols: 4,
            values: vec![
                Act::from_num(0.5).to_bits(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }));

        let without_det =
            run_gemma4_layer(&activations, &resolved_without_det, None).expect("run fp32 path");
        let with_det =
            run_gemma4_layer(&activations, &resolved_with_det, None).expect("run det gate_proj");

        assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
        assert!(with_det.activations[0][0] > without_det.activations[0][0]);
    }

    #[test]
    fn run_gemma4_layer_uses_det_down_proj_after_hidden_mul() {
        let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            cache_sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: MatrixF32 {
                rows: 8,
                cols: 4,
                values: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            }
            .into(),
            up_proj: MatrixF32 {
                rows: 8,
                cols: 4,
                values: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            }
            .into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: None,
            layer_scalar: None,
        };

        let resolved_without_det =
            crate::io::resolve_layer_weights(&layer).expect("resolve fp32-only layer");
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.down_proj_det = Some(Arc::new(DetNumMatrix {
            rows: 4,
            cols: 8,
            values: vec![
                Act::from_num(1.0).to_bits(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }));

        let without_det =
            run_gemma4_layer(&activations, &resolved_without_det, None).expect("run fp32 path");
        let with_det =
            run_gemma4_layer(&activations, &resolved_with_det, None).expect("run det down_proj");

        assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
        assert!(with_det.activations[0][0] > without_det.activations[0][0]);
    }

    #[test]
    fn run_causal_attention_uses_det_q_proj_during_prefill() {
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let resolved_without_det = attention_test_layer(
            zero_matrix(2, 2),
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.q_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

        let without_det =
            run_causal_attention(
                &inputs,
                &resolved_without_det,
                None,
                None,
                None,
                InferenceExecutionMode::Fp32,
            )
            .unwrap();
        let with_det = run_causal_attention(
            &inputs,
            &resolved_with_det,
            None,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert!(with_det.0[1][1] > without_det.0[1][1]);
    }

    #[test]
    fn run_causal_attention_uses_det_k_proj_during_prefill() {
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let resolved_without_det = attention_test_layer(
            identity_matrix(2),
            zero_matrix(2, 2),
            identity_matrix(2),
            identity_matrix(2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.k_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

        let without_det =
            run_causal_attention(
                &inputs,
                &resolved_without_det,
                None,
                None,
                None,
                InferenceExecutionMode::Fp32,
            )
            .unwrap();
        let with_det = run_causal_attention(
            &inputs,
            &resolved_with_det,
            None,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert!(with_det.0[1][1] > without_det.0[1][1]);
    }

    #[test]
    fn run_causal_attention_uses_det_v_proj_during_prefill() {
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let resolved_without_det = attention_test_layer(
            identity_matrix(2),
            identity_matrix(2),
            zero_matrix(2, 2),
            identity_matrix(2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.v_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

        let without_det =
            run_causal_attention(
                &inputs,
                &resolved_without_det,
                None,
                None,
                None,
                InferenceExecutionMode::Fp32,
            )
            .unwrap();
        let with_det = run_causal_attention(
            &inputs,
            &resolved_with_det,
            None,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(without_det.0, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert!(with_det.0[1][1] > 0.0);
    }

    #[test]
    fn run_causal_attention_uses_det_o_proj_during_prefill() {
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let resolved_without_det = attention_test_layer(
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
            zero_matrix(2, 2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.o_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

        let without_det =
            run_causal_attention(
                &inputs,
                &resolved_without_det,
                None,
                None,
                None,
                InferenceExecutionMode::Fp32,
            )
            .unwrap();
        let with_det = run_causal_attention(
            &inputs,
            &resolved_with_det,
            None,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(without_det.0, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert!(with_det.0[1][1] > 0.0);
    }

    #[test]
    fn run_causal_attention_decode_uses_det_q_proj() {
        let input = vec![0.0, 1.0];
        let resolved_without_det = attention_test_layer(
            zero_matrix(2, 2),
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.q_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));
        let initial_cache = append_kv_cache(
            LayerKvCache::new(1),
            &[vec![1.0, 0.0]],
            &[vec![1.0, 0.0]],
            None,
        )
        .unwrap();

        let without_det = run_causal_attention_decode(
            &input,
            &resolved_without_det,
            initial_cache.clone(),
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let with_det = run_causal_attention_decode(
            &input,
            &resolved_with_det,
            initial_cache,
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert!(with_det.0[1] > without_det.0[1]);
    }

    #[test]
    fn run_causal_attention_decode_uses_det_k_proj() {
        let input = vec![0.0, 1.0];
        let resolved_without_det = attention_test_layer(
            identity_matrix(2),
            zero_matrix(2, 2),
            identity_matrix(2),
            identity_matrix(2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.k_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));
        let initial_cache = append_kv_cache(
            LayerKvCache::new(1),
            &[vec![1.0, 0.0]],
            &[vec![1.0, 0.0]],
            None,
        )
        .unwrap();

        let without_det = run_causal_attention_decode(
            &input,
            &resolved_without_det,
            initial_cache.clone(),
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let with_det = run_causal_attention_decode(
            &input,
            &resolved_with_det,
            initial_cache,
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert!(with_det.0[1] > without_det.0[1]);
    }

    #[test]
    fn run_causal_attention_decode_uses_det_v_proj() {
        let input = vec![0.0, 1.0];
        let resolved_without_det = attention_test_layer(
            identity_matrix(2),
            identity_matrix(2),
            zero_matrix(2, 2),
            identity_matrix(2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.v_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));
        let initial_cache = append_kv_cache(
            LayerKvCache::new(1),
            &[vec![1.0, 0.0]],
            &[vec![1.0, 0.0]],
            None,
        )
        .unwrap();

        let without_det = run_causal_attention_decode(
            &input,
            &resolved_without_det,
            initial_cache.clone(),
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let with_det = run_causal_attention_decode(
            &input,
            &resolved_with_det,
            initial_cache,
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert!(with_det.0[1] > without_det.0[1]);
    }

    #[test]
    fn run_causal_attention_decode_uses_det_o_proj() {
        let input = vec![1.0, 0.0];
        let resolved_without_det = attention_test_layer(
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
            zero_matrix(2, 2),
        );
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.o_proj_det = Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]));

        let without_det = run_causal_attention_decode(
            &input,
            &resolved_without_det,
            LayerKvCache::new(1),
            None,
            0,
            None,
            None,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let with_det = run_causal_attention_decode(
            &input,
            &resolved_with_det,
            LayerKvCache::new(1),
            None,
            0,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(without_det.0, vec![0.0, 0.0]);
        assert!(with_det.0[0] > 0.0);
    }

    #[test]
    fn project_to_logits_uses_det_untied_lm_head() {
        let without_det = project_to_logits(
            &[1.0, 0.0],
            &Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 2),
                det_weight: None,
            },
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();
        let with_det = project_to_logits(
            &[1.0, 0.0],
            &Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 2),
                det_weight: Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0])),
            },
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(without_det, vec![0.0, 0.0]);
        assert_eq!(with_det, vec![1.0, 0.0]);
    }

    #[test]
    fn project_to_logits_uses_det_tied_embedding_source() {
        let embedding_source = deterministic_embedding_source(
            "tied-logits",
            "model.language_model.embed_tokens.weight",
            2,
            2,
            &[1.0, 0.0, 0.0, 1.0],
        );
        let without_det = project_to_logits(
            &[1.0, 0.0],
            &Gemma4LogitsProjection::TiedEmbedding(zero_matrix(2, 2)),
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();
        let with_det = project_to_logits(
            &[1.0, 0.0],
            &Gemma4LogitsProjection::TiedEmbedding(zero_matrix(2, 2)),
            Some(&embedding_source),
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(without_det, vec![0.0, 0.0]);
        assert_eq!(with_det, vec![1.0, 0.0]);
    }

    #[test]
    fn project_decode_hidden_to_logits_uses_det_untied_lm_head_with_softcap() {
        let without_det = project_decode_hidden_to_logits(
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.0,
            &Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 2),
                det_weight: None,
            },
            None,
            InferenceExecutionMode::Deterministic,
            Some(0.5),
        )
        .unwrap();
        let with_det = project_decode_hidden_to_logits(
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.0,
            &Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 2),
                det_weight: Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0])),
            },
            None,
            InferenceExecutionMode::Deterministic,
            Some(0.5),
        )
        .unwrap();

        assert_eq!(without_det.logits, vec![0.0, 0.0]);
        assert_eq!(
            with_det.logits,
            super::apply_final_logit_softcapping(&[1.0, 1.0], 0.5)
        );
    }

    #[test]
    fn project_decode_hidden_to_logits_uses_det_tied_embedding_source() {
        let embedding_source = deterministic_embedding_source(
            "decode-tied-logits",
            "model.language_model.embed_tokens.weight",
            2,
            2,
            &[1.0, 0.0, 0.0, 1.0],
        );
        let logits = project_decode_hidden_to_logits(
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.0,
            &Gemma4LogitsProjection::TiedEmbedding(zero_matrix(2, 2)),
            Some(&embedding_source),
            InferenceExecutionMode::Deterministic,
            None,
        )
        .unwrap();

        assert_eq!(logits.logits, vec![1.0, 1.0]);
    }

    #[test]
    fn compute_prefill_ple_inputs_uses_det_model_projection() {
        let layers = vec![ple_test_layer()];
        let ple_global = Gemma4PleGlobalWeights::from_det_num_sources(
            vec![deterministic_tensor_source(
                "prefill-ple-token",
                "model.language_model.embed_tokens_per_layer.weight",
                2,
                2,
                &[0.0, 0.0, 0.0, 0.0],
            )],
            vec![deterministic_tensor_source(
                "prefill-ple-proj",
                "model.language_model.per_layer_model_projection.weight",
                2,
                4,
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );
        let inputs = vec![vec![1.0, 0.0, 0.0, 0.0]];

        let ple_inputs = compute_prefill_ple_inputs(
            &[0],
            &inputs,
            &layers,
            &ple_global,
            0.0,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        let projected = ple_inputs.per_layer_inputs[0].as_ref().unwrap();
        assert_eq!(
            projected[0][0],
            crate::shared::det_num::act_to_f32(crate::shared::det_num::f32_to_act(2f32.sqrt()))
        );
        assert_eq!(projected[0][1], 0.0);
    }

    #[test]
    fn compute_decode_ple_input_uses_det_model_projection() {
        let layer = ple_test_layer();
        let ple_global = Gemma4PleGlobalWeights::from_det_num_sources(
            vec![deterministic_tensor_source(
                "decode-ple-token",
                "model.language_model.embed_tokens_per_layer.weight",
                2,
                2,
                &[0.0, 0.0, 0.0, 0.0],
            )],
            vec![deterministic_tensor_source(
                "decode-ple-proj",
                "model.language_model.per_layer_model_projection.weight",
                2,
                4,
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );

        let ple_input = compute_decode_ple_input(
            0,
            &[1.0, 0.0, 0.0, 0.0],
            0,
            &layer,
            Some(&ple_global),
            0.0,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        let projected = ple_input.expect("decode ple input should exist");
        assert_eq!(
            projected[0],
            crate::shared::det_num::act_to_f32(crate::shared::det_num::f32_to_act(2f32.sqrt()))
        );
        assert_eq!(projected[1], 0.0);
    }

    #[test]
    fn run_gemma4_layer_uses_det_ple_input_gate() {
        let activations = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let mut resolved_without_det = ple_resolved_test_layer();
        resolved_without_det.ple = Some(ResolvedGemma4PleLayerWeights {
            input_gate: Arc::new(zero_matrix(2, 4)),
            layer_projection: Arc::new(MatrixF32 {
                rows: 4,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }),
            input_gate_det: None,
            layer_projection_det: None,
            post_input_norm_weight: vec![1.0; 4],
        });
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.ple = Some(ResolvedGemma4PleLayerWeights {
            input_gate_det: Some(det_matrix(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            ..resolved_without_det.ple.clone().unwrap()
        });

        let without_det = run_gemma4_layer(&activations, &resolved_without_det, Some(&[vec![1.0, 1.0]]))
            .expect("run fp32 ple path");
        let with_det = run_gemma4_layer(&activations, &resolved_with_det, Some(&[vec![1.0, 1.0]]))
            .expect("run det ple input gate");

        assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
        assert!(with_det.activations[0][0] > without_det.activations[0][0]);
    }

    #[test]
    fn run_gemma4_layer_decode_uses_det_ple_layer_projection() {
        let mut resolved_without_det = ple_resolved_test_layer();
        resolved_without_det.ple = Some(ResolvedGemma4PleLayerWeights {
            input_gate: Arc::new(MatrixF32 {
                rows: 2,
                cols: 4,
                values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }),
            layer_projection: Arc::new(zero_matrix(4, 2)),
            input_gate_det: None,
            layer_projection_det: None,
            post_input_norm_weight: vec![1.0; 4],
        });
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.ple = Some(ResolvedGemma4PleLayerWeights {
            layer_projection_det: Some(det_matrix(4, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            ..resolved_without_det.ple.clone().unwrap()
        });

        let without_det = run_gemma4_layer_decode(
            &[1.0, 0.0, 0.0, 0.0],
            &resolved_without_det,
            Some(&[1.0, 1.0]),
            LayerKvCache::new(1),
            None,
            0,
        )
        .expect("run fp32 decode ple path");
        let with_det = run_gemma4_layer_decode(
            &[1.0, 0.0, 0.0, 0.0],
            &resolved_with_det,
            Some(&[1.0, 1.0]),
            LayerKvCache::new(1),
            None,
            0,
        )
        .expect("run det decode ple projection");

        assert_eq!(without_det.0, vec![1.0, 0.0, 0.0, 0.0]);
        assert!(with_det.0[0] > without_det.0[0]);
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
            cache_sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: Some(Gemma4PleLayerWeights {
                input_gate: zero_matrix(2, 4).into(),
                layer_projection: zero_matrix(4, 2).into(),
                post_input_norm_weight: vec![1.0; 4],
            }),
            layer_scalar: None,
        }];
        let ple_global = Gemma4PleGlobalWeights::from_materialized(
            vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            vec![MatrixF32 {
                rows: 2,
                cols: 4,
                values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            }],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );
        let inputs = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let first = compute_prefill_ple_inputs(
            &[0, 1],
            &inputs,
            &layers,
            &ple_global,
            1e-6,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let second = compute_prefill_ple_inputs(
            &[0, 1],
            &inputs,
            &layers,
            &ple_global,
            1e-6,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn apply_final_norm_and_project_to_logits_work() {
        let final_hidden_states = vec![vec![1.0, 2.0]];
        let normed = apply_final_norm(&final_hidden_states, &[1.0, 1.0], 0.0).unwrap();
        let logits = project_to_logits(
            &normed.activations[0],
            &Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 2,
                    cols: 2,
                    values: vec![1.0, 0.0, 0.0, 1.0],
                },
                det_weight: None,
            },
            None,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let extracted = extract_prefill_logits(&logits);

        assert_eq!(logits.len(), 2);
        assert_eq!(extracted.logits, logits);
        assert!(!extracted.final_logits_sha256.is_empty());
    }

    #[test]
    fn apply_rms_norm_to_sequence_uses_det_num_contract_in_deterministic_mode() {
        let normalized = apply_rms_norm_to_sequence(
            &[vec![1.0, 0.0]],
            &[0.5, 1.0],
            0.0,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(normalized, vec![vec![act_to_f32(Act::from_bits(46_341)), 0.0]]);
    }

    #[test]
    fn apply_head_and_value_norms_use_det_num_contract_in_deterministic_mode() {
        let mut head_normed = vec![vec![vec![1.0, 0.0]]];
        apply_head_rms_norm(
            &mut head_normed,
            &[1.0, 1.0],
            0.0,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();
        assert_eq!(
            head_normed,
            vec![vec![vec![act_to_f32(Act::from_bits(92_682)), 0.0]]]
        );

        let mut value_normed = vec![vec![vec![1.0, 0.0]]];
        apply_value_rms_norm(&mut value_normed, 0.0, InferenceExecutionMode::Deterministic)
            .unwrap();
        assert_eq!(
            value_normed,
            vec![vec![vec![act_to_f32(Act::from_bits(92_682)), 0.0]]]
        );
    }

    #[test]
    fn apply_final_norm_with_mode_uses_det_num_contract() {
        let normalized = apply_final_norm_with_mode(
            &[vec![1.0, 0.0]],
            &[0.5, 1.0],
            0.0,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(
            normalized.activations,
            vec![vec![act_to_f32(Act::from_bits(46_341)), 0.0]]
        );
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
            cache_sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: None,
            layer_scalar: None,
        };
        let model = Gemma4TransformerModel {
            embedding_table: None,
            embedding_source: None,
            layers: vec![layer.clone(), layer],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 4),
                det_weight: None,
            },
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
        };
        let activations = vec![vec![1.0, 1.5, 0.0, 0.0], vec![0.0, 0.5, 0.0, 0.0]];

        let output = run_text_layers_prefill(&activations, &model, None).unwrap();

        assert_eq!(output.activations, activations);
    }

    #[test]
    fn run_text_layers_prefill_with_cache_retains_sliding_window_entries() {
        let model = parity_test_model(Gemma4AttentionKind::Sliding, Some(2));
        let activations = model.embedding_table.as_ref().unwrap().rows.clone();

        let (_, layer_caches) =
            run_text_layers_prefill_with_cache(&activations, &model, None).unwrap();

        assert_eq!(layer_caches.len(), 1);
        assert_eq!(layer_caches[0].current_len(), 2);
    }

    #[test]
    fn append_kv_cache_keeps_newest_sliding_window_entries_in_order() {
        let cache = append_kv_cache(
            crate::shared::transformer::LayerKvCache::new(1),
            &[vec![1.0]],
            &[vec![10.0]],
            None,
        )
        .unwrap();
        let cache = append_kv_cache(cache, &[vec![2.0]], &[vec![20.0]], None).unwrap();

        let updated = append_kv_cache(cache, &[vec![3.0]], &[vec![30.0]], Some(2)).unwrap();

        assert_eq!(updated.current_len(), 2);
        assert_eq!(
            updated.keys[0].iter().cloned().collect::<Vec<_>>(),
            vec![vec![2.0], vec![3.0]]
        );
        assert_eq!(
            updated.values[0].iter().cloned().collect::<Vec<_>>(),
            vec![vec![20.0], vec![30.0]]
        );
    }

    #[test]
    fn run_text_layers_decode_step_matches_prefill_for_appended_token() {
        let model = parity_test_model(Gemma4AttentionKind::Full, None);
        let embeddings = model.embedding_table.as_ref().unwrap().rows.clone();
        let prompt_embeddings = embeddings[..2].to_vec();
        let next_embedding = embeddings[2].clone();

        let (_, layer_caches) =
            run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
        let decoded = run_text_layers_decode_step(
            &next_embedding,
            2,
            &model,
            layer_caches,
            prompt_embeddings.len(),
        )
        .unwrap();
        let replay = run_text_layers_prefill(&embeddings, &model, None).unwrap();
        let replay_last_hidden = replay.activations.last().cloned().unwrap();

        assert_eq!(decoded.activation_state.activations[0], replay_last_hidden);

        let decoded_logits = project_decode_hidden_to_logits(
            &decoded.activation_state.activations[0],
            &model.final_norm_weight,
            model.rms_norm_eps,
            &model.logits_projection,
            model.embedding_source.as_ref(),
            InferenceExecutionMode::Fp32,
            model.final_logit_softcapping,
        )
        .unwrap();
        let replay_logits = project_to_logits(
            &apply_final_norm(
                &replay.activations,
                &model.final_norm_weight,
                model.rms_norm_eps,
            )
            .unwrap()
            .activations
            .last()
            .cloned()
            .unwrap(),
            &model.logits_projection,
            model.embedding_source.as_ref(),
            InferenceExecutionMode::Fp32,
        )
        .unwrap();

        assert_eq!(decoded_logits.logits, replay_logits);
    }

    #[test]
    fn run_text_layers_decode_step_updates_full_attention_cache() {
        let model = parity_test_model(Gemma4AttentionKind::Full, None);
        let embeddings = model.embedding_table.as_ref().unwrap().rows.clone();
        let prompt_embeddings = embeddings[..2].to_vec();
        let next_embedding = embeddings[2].clone();

        let (_, layer_caches) =
            run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
        let decoded = run_text_layers_decode_step(
            &next_embedding,
            2,
            &model,
            layer_caches,
            prompt_embeddings.len(),
        )
        .unwrap();

        assert_eq!(decoded.layer_caches.len(), 1);
        assert_eq!(decoded.layer_caches[0].current_len(), 3);
    }

    #[test]
    fn run_text_layers_decode_step_rejects_non_prior_kv_donor_metadata() {
        let model = parity_test_model(Gemma4AttentionKind::Sliding, Some(2));
        let embeddings = model.embedding_table.as_ref().unwrap().rows.clone();
        let prompt_embeddings = embeddings[..2].to_vec();
        let next_embedding = embeddings[2].clone();

        let (_, layer_caches) =
            run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
        let mut invalid_model = model;
        invalid_model.layers[0].kv_shared_layer_index = Some(0);
        let error = run_text_layers_decode_step(
            &next_embedding,
            2,
            &invalid_model,
            layer_caches,
            prompt_embeddings.len(),
        )
        .err()
        .expect("non-prior donor should fail");

        assert!(error.to_string().contains("non-prior donor"));
    }

    fn parity_test_model(
        attention_kind: Gemma4AttentionKind,
        sliding_window: Option<usize>,
    ) -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            embedding_table: Some(EmbeddingTable {
                rows: vec![
                    vec![1.0, 0.0, 0.5, 0.0],
                    vec![0.0, 1.0, 0.0, 0.5],
                    vec![0.5, 0.5, 1.0, 0.0],
                ],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind,
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window,
                cache_sliding_window: sliding_window,
                rms_norm_eps: 1e-6,
                rope_base: 10_000.0,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                        1.0,
                    ],
                }
                .into(),
                k_proj: MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                }
                .into(),
                v_proj: Some(
                    MatrixF32 {
                        rows: 2,
                        cols: 4,
                        values: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    }
                    .into(),
                ),
                o_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                        1.0,
                    ],
                }
                .into(),
                q_norm_weight: vec![1.0, 1.0],
                k_norm_weight: vec![1.0, 1.0],
                input_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight: vec![1.0; 4],
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 3,
                    cols: 4,
                    values: vec![0.7, 0.1, 0.2, 0.0, 0.0, 0.8, 0.1, 0.1, 0.2, 0.0, 0.8, 0.2],
                },
                det_weight: None,
            },
            final_logit_softcapping: None,
            rms_norm_eps: 1e-6,
        }
    }

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }

    fn identity_matrix(size: usize) -> MatrixF32 {
        let mut values = vec![0.0; size * size];
        for idx in 0..size {
            values[idx * size + idx] = 1.0;
        }
        MatrixF32 {
            rows: size,
            cols: size,
            values,
        }
    }

    fn det_matrix(rows: usize, cols: usize, values: &[f32]) -> Arc<DetNumMatrix> {
        Arc::new(DetNumMatrix {
            rows,
            cols,
            values: values
                .iter()
                .copied()
                .map(|value| Act::from_num(value).to_bits())
                .collect(),
        })
    }

    fn attention_test_layer(
        q_proj: MatrixF32,
        k_proj: MatrixF32,
        v_proj: MatrixF32,
        o_proj: MatrixF32,
    ) -> ResolvedGemma4LayerWeights {
        crate::io::resolve_layer_weights(&Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Full,
            hidden_size: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: None,
            cache_sliding_window: None,
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: q_proj.into(),
            k_proj: k_proj.into(),
            v_proj: Some(v_proj.into()),
            o_proj: o_proj.into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 2],
            post_attention_layernorm_weight: vec![1.0; 2],
            pre_feedforward_layernorm_weight: vec![1.0; 2],
            post_feedforward_layernorm_weight: vec![1.0; 2],
            gate_proj: zero_matrix(4, 2).into(),
            up_proj: zero_matrix(4, 2).into(),
            down_proj: zero_matrix(2, 4).into(),
            ple: None,
            layer_scalar: None,
        })
        .expect("resolve attention test layer")
    }

    fn ple_test_layer() -> Gemma4LayerWeights {
        Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Full,
            hidden_size: 4,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: None,
            cache_sliding_window: None,
            rms_norm_eps: 1e-6,
            rope_base: 10_000.0,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(2, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 2).into(),
            q_norm_weight: vec![1.0, 1.0],
            k_norm_weight: vec![1.0, 1.0],
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight: vec![1.0; 4],
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: Some(Gemma4PleLayerWeights {
                input_gate: zero_matrix(2, 4).into(),
                layer_projection: zero_matrix(4, 2).into(),
                post_input_norm_weight: vec![1.0; 4],
            }),
            layer_scalar: None,
        }
    }

    fn ple_resolved_test_layer() -> ResolvedGemma4LayerWeights {
        let mut resolved =
            crate::io::resolve_layer_weights(&ple_test_layer()).expect("resolve ple test layer");
        resolved.ple = Some(ResolvedGemma4PleLayerWeights {
            input_gate: Arc::new(zero_matrix(2, 4)),
            layer_projection: Arc::new(zero_matrix(4, 2)),
            input_gate_det: None,
            layer_projection_det: None,
            post_input_norm_weight: vec![1.0; 4],
        });
        resolved
    }

    fn deterministic_embedding_source(
        label: &str,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        values: &[f32],
    ) -> GemmaEmbeddingTensorSource {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let weights_path =
            std::env::temp_dir().join(format!("raster-inference-{label}-{unique}.detwgt"));
        let name_bytes = tensor_name.as_bytes();
        let payload = values
            .iter()
            .flat_map(|value| wgt_to_le_bytes(f32_to_wgt(*value)))
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&(rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(cols as u64).to_le_bytes());
        bytes.extend_from_slice(&((rows * cols) as u64).to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let data_offset = bytes.len();
        bytes.extend_from_slice(&payload);
        fs::write(&weights_path, bytes).expect("write det embedding artifact");

        GemmaEmbeddingTensorSource::Deterministic {
            source: DetNumTensorSliceSource {
                weights_path,
                total_rows: rows,
                total_cols: cols,
                data_offset,
                row_offset: 0,
                row_count: rows,
                col_offset: 0,
                col_count: cols,
            },
            scale: (cols as f32).sqrt(),
            det_cache: Arc::new(Mutex::new(None)),
        }
    }

    fn deterministic_tensor_source(
        label: &str,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        values: &[f32],
    ) -> DetNumTensorSliceSource {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let weights_path =
            std::env::temp_dir().join(format!("raster-inference-{label}-{unique}.detwgt"));
        let name_bytes = tensor_name.as_bytes();
        let payload = values
            .iter()
            .flat_map(|value| wgt_to_le_bytes(f32_to_wgt(*value)))
            .collect::<Vec<_>>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&(rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(cols as u64).to_le_bytes());
        bytes.extend_from_slice(&((rows * cols) as u64).to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let data_offset = bytes.len();
        bytes.extend_from_slice(&payload);
        fs::write(&weights_path, bytes).expect("write det tensor artifact");

        DetNumTensorSliceSource {
            weights_path,
            total_rows: rows,
            total_cols: cols,
            data_offset,
            row_offset: 0,
            row_count: rows,
            col_offset: 0,
            col_count: cols,
        }
    }
}
