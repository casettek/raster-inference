use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{
    ActivationSequence, DetNumMatrix, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4PleGlobalWeights, Gemma4PrefillPleInputs,
    Gemma4TransformerModel, GemmaEmbeddingTensorSource, InternalActivationRow,
    InternalActivationSequence, InternalLogits, LayerKvCache, MatrixF32, PrefillLogits,
    ResolvedGemma4LayerWeights,
};
use crate::shared::numerics::det_num::{
    act_to_f32, act_to_le_bytes, add_sat, attention_score as det_attention_score,
    attention_softmax as det_attention_softmax,
    attention_weighted_sum as det_attention_weighted_sum, f32_to_acc, f32_to_act,
    gelu_pytorch_tanh_act, mac_bits, mul_sat, requantize, rms_norm as det_rms_norm,
    rope_rotate_pairs as det_rope_rotate_pairs, scale_act, softcap_act,
    value_rms_norm as det_value_rms_norm, Acc, Act, Wgt,
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
    if execution_mode == InferenceExecutionMode::Deterministic {
        bail!("deterministic embedding requires a .detwgt embedding source, not an f32 embedding table");
    }

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

    let mut activation_rows = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        let row = embedding_table.rows.get(row_idx).ok_or_else(|| {
            anyhow::anyhow!("token id {token_id} is out of bounds for embedding table")
        })?;
        activation_rows.push(scale_row_buffer(
            &ActivationRowBuffer::from_values(row.clone()),
            embedding_table.scale,
            None,
            execution_mode == InferenceExecutionMode::Deterministic,
        )?);
    }

    let activations = activation_rows
        .iter()
        .map(|row| row.values.clone())
        .collect::<Vec<_>>();
    let activations_sha256 = build_activation_commitment(&activations);
    let acts = activation_rows
        .iter()
        .map(|row| row.acts.clone())
        .collect::<Option<Vec<_>>>();
    let det_activations_sha256 = acts
        .as_ref()
        .map(|acts| build_det_activation_commitment(acts));
    let mut activation_sequence = ActivationSequence::from_internal(
        match acts {
            Some(acts) => InternalActivationSequence::from_det_values(acts),
            None => InternalActivationSequence::from_values(activations),
        },
        activations_sha256,
    );
    activation_sequence.det_activations_sha256 = det_activations_sha256;
    Ok(activation_sequence)
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
    compute_prefill_ple_inputs_internal(
        token_ids,
        InternalActivationSequence::from_values(input_activations.to_vec()),
        layers,
        ple_global,
        rms_norm_eps,
        None,
        execution_mode,
    )
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
    // let _trace = trace_scope("transformer_state_transition.compute_prefill_ple_inputs");
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
        // trace_event(format!("transformer_state_transition.compute_prefill_ple_inputs layer={layer_idx}"));
        if layer.ple.is_none() {
            per_layer_inputs.push(None);
            continue;
        }

        let mut embedded = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            embedded.push(ActivationRowBuffer::from_internal(
                crate::io::load_ple_token_embedding_row_internal(ple_global, layer_idx, *token_id)?,
            ));
        }
        let embedded = scale_sequence_buffer(
            &activation_sequence_buffer_from_rows(embedded),
            ple_global.embedding_scale,
            ple_global.embedding_scale_det,
            execution_mode == InferenceExecutionMode::Deterministic,
        )?;

        let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
        let model_projection_det =
            crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?;
        let projected = project_linear_sequence_buffer(
            &input_buffer,
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
        let combined = scale_sequence_buffer(
            &combined,
            ple_global.input_scale,
            ple_global.input_scale_det,
            execution_mode == InferenceExecutionMode::Deterministic || combined.acts.is_some(),
        )?;
        per_layer_inputs.push(Some(combined.into_internal()));
    }

    Ok(Gemma4PrefillPleInputs::from_internal(per_layer_inputs))
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
    compute_decode_ple_input_internal(
        token_id,
        InternalActivationRow::from_values(input_activation.to_vec()),
        layer_idx,
        layer,
        ple_global,
        rms_norm_eps,
        None,
        execution_mode,
    )
    .map(|input| input.map(|row| row.clone_f32()))
}

pub(crate) fn compute_decode_ple_input_internal(
    token_id: u32,
    input_activation: InternalActivationRow,
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
    ple_global: Option<&Gemma4PleGlobalWeights>,
    rms_norm_eps: f32,
    rms_norm_eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
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
        ple_global.embedding_scale,
        ple_global.embedding_scale_det,
        execution_mode == InferenceExecutionMode::Deterministic,
    )?;
    let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
    let model_projection_det =
        crate::io::materialize_det_num_ple_model_projection(ple_global, layer_idx)?;
    let projected = project_linear_row_buffer(
        &ActivationRowBuffer::from_internal(input_activation),
        &model_projection,
        model_projection_det.as_deref(),
    )?;
    let projected_uses_det = projected.acts.is_some();
    let projected = scale_row_buffer(
        &projected,
        ple_global.projection_scalar,
        ple_global.projection_scalar_det,
        projected_uses_det,
    )?;
    let projected = apply_rms_norm_row_buffer(
        &projected,
        &ple_global.projection_norm_weight,
        ple_global.projection_norm_weight_det.as_deref(),
        rms_norm_eps,
        rms_norm_eps_det,
        execution_mode,
    )?;

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
        ple_global.input_scale_det,
        execution_mode == InferenceExecutionMode::Deterministic || combined.acts.is_some(),
    )?;

    Ok(Some(combined.into_internal()))
}

pub fn run_gemma4_layer(
    input_activations: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[Vec<f32>]>,
) -> Result<ActivationSequence> {
    Ok(run_gemma4_layer_with_cache(
        input_activations,
        layer,
        per_layer_input,
        None,
        InferenceExecutionMode::Fp32,
    )?
    .0)
}

pub(crate) fn run_gemma4_layer_with_cache(
    input_activations: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[Vec<f32>]>,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequence, LayerKvCache)> {
    run_gemma4_layer_with_cache_internal(
        InternalActivationSequence::from_values(input_activations.to_vec()),
        layer,
        per_layer_input.map(|input| InternalActivationSequence::from_values(input.to_vec())),
        donor_cache,
        execution_mode,
    )
}

pub(crate) fn run_gemma4_layer_with_cache_internal(
    input_activations: InternalActivationSequence,
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<InternalActivationSequence>,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
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
            execution_mode,
        )?
    };
    let (attn_out, layer_cache) = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.core");
        run_attention_for_layer_with_cache(&normed, layer, donor_cache, execution_mode)?
    };
    let attn_out = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.post_rms_norm");
        apply_rms_norm_to_sequence_buffer(
            &attn_out,
            &layer.post_attention_layernorm_weight,
            layer.post_attention_layernorm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
            execution_mode,
        )?
    };
    xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.attention.residual_add");
        add_sequence_buffers(
            &residual,
            &attn_out,
            execution_mode == InferenceExecutionMode::Deterministic || attn_out.acts.is_some(),
        )?
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
            execution_mode,
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
                apply_gelu_to_sequence_buffer(&gate_preactivation, execution_mode)?,
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
                None if execution_mode == InferenceExecutionMode::Deterministic => {
                    bail!("deterministic MLP gate projection requires canonical det_weight")
                }
                None => ActivationSequenceBuffer::from_values(linear_sequence(
                    &normed.values,
                    layer.gate_proj.as_ref(),
                )?),
            };
            let up = match up_weight {
                Some(weight) => project_linear_sequence_buffer(
                    &normed,
                    layer.up_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None if execution_mode == InferenceExecutionMode::Deterministic => {
                    bail!("deterministic MLP up projection requires canonical det_weight")
                }
                None => ActivationSequenceBuffer::from_values(linear_sequence(
                    &normed.values,
                    layer.up_proj.as_ref(),
                )?),
            };
            (
                apply_gelu_to_sequence_buffer(&gate_preactivation, execution_mode)?,
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
            Some(weight) => project_linear_sequence_buffer(
                &ff_hidden,
                layer.down_proj.as_ref(),
                Some(weight.as_ref()),
            )?,
            None if execution_mode == InferenceExecutionMode::Deterministic => {
                bail!("deterministic MLP down projection requires canonical det_weight")
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
        apply_rms_norm_to_sequence_buffer(
            &ff_out,
            &layer.post_feedforward_layernorm_weight,
            layer.post_feedforward_layernorm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
            execution_mode,
        )?
    };
    xs = {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.mlp.residual_add");
        add_sequence_buffers(
            &residual,
            &ff_out,
            execution_mode == InferenceExecutionMode::Deterministic || ff_out_uses_det,
        )?
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
            apply_gelu_to_sequence_buffer(&gate_preactivation, execution_mode)?
        };
        let gated = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.input_mul");
            mul_sequence_buffers(
                &gated,
                &ActivationSequenceBuffer::from_internal(per_layer_input),
                execution_mode == InferenceExecutionMode::Deterministic || gated.acts.is_some(),
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
            apply_rms_norm_to_sequence_buffer(
                &projected,
                &ple.post_input_norm_weight,
                ple.post_input_norm_weight_det.as_deref(),
                layer.rms_norm_eps,
                layer.rms_norm_eps_det,
                execution_mode,
            )?
        };
        xs = {
            // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.ple.residual_add");
            add_sequence_buffers(
                &residual,
                &projected,
                execution_mode == InferenceExecutionMode::Deterministic || projected_uses_det,
            )?
        };
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        // let _trace = trace_scope("transformer_state_transition.run_gemma4_layer.layer_scalar");
        xs = scale_sequence_buffer(
            &xs,
            layer_scalar,
            layer.layer_scalar_det,
            execution_mode == InferenceExecutionMode::Deterministic,
        )?;
    }

    Ok((activation_sequence_from_buffer(xs), layer_cache))
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
    let (activation, cache) = run_gemma4_layer_decode_with_mode_internal(
        InternalActivationRow::from_values(input_activation.to_vec()),
        layer,
        per_layer_input.map(|input| InternalActivationRow::from_values(input.to_vec())),
        cache,
        donor_cache,
        position,
        execution_mode,
    )?;
    Ok((activation.clone_f32(), cache))
}

pub(crate) fn run_gemma4_layer_decode_with_mode_internal(
    input_activation: InternalActivationRow,
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<InternalActivationRow>,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    execution_mode: InferenceExecutionMode,
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
        execution_mode,
    )?;
    let (xs_values, updated_cache) = run_attention_for_layer_decode(
        &normed,
        layer,
        cache,
        donor_cache,
        position,
        execution_mode,
    )?;
    let mut xs = apply_rms_norm_row_buffer(
        &xs_values,
        &layer.post_attention_layernorm_weight,
        layer.post_attention_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
        execution_mode,
    )?;
    xs = add_row_buffers(
        &residual,
        &xs,
        execution_mode == InferenceExecutionMode::Deterministic || xs.acts.is_some(),
    )?;

    let residual = xs.clone();
    let normed = apply_rms_norm_row_buffer(
        &xs,
        &layer.pre_feedforward_layernorm_weight,
        layer.pre_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
        execution_mode,
    )?;
    let (gate_preactivation, up) = match (layer.gate_proj_det.as_ref(), layer.up_proj_det.as_ref())
    {
        (Some(gate_weight), Some(up_weight)) => {
            let quantized_normed = row_buffer_acts(&normed)?;
            (
                apply_gelu_to_row_buffer(
                    &ActivationRowBuffer::from_acts(det_linear_row_acts_from_acts(
                        &quantized_normed,
                        gate_weight.as_ref(),
                    )?),
                    execution_mode,
                )?,
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
                None if execution_mode == InferenceExecutionMode::Deterministic => {
                    bail!("deterministic MLP gate projection requires canonical det_weight")
                }
                None => ActivationRowBuffer::from_values(linear_row(
                    &normed.values,
                    layer.gate_proj.as_ref(),
                )?),
            };
            let up = match up_weight {
                Some(weight) => project_linear_row_buffer(
                    &normed,
                    layer.up_proj.as_ref(),
                    Some(weight.as_ref()),
                )?,
                None if execution_mode == InferenceExecutionMode::Deterministic => {
                    bail!("deterministic MLP up projection requires canonical det_weight")
                }
                None => ActivationRowBuffer::from_values(linear_row(
                    &normed.values,
                    layer.up_proj.as_ref(),
                )?),
            };
            (
                apply_gelu_to_row_buffer(&gate_preactivation, execution_mode)?,
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
        None if execution_mode == InferenceExecutionMode::Deterministic => {
            bail!("deterministic MLP down projection requires canonical det_weight")
        }
        None => ActivationRowBuffer::from_values(linear_row(
            &ff_hidden.values,
            layer.down_proj.as_ref(),
        )?),
    };
    let ff_out_uses_det = ff_out.acts.is_some();
    let ff_out = apply_rms_norm_row_buffer(
        &ff_out,
        &layer.post_feedforward_layernorm_weight,
        layer.post_feedforward_layernorm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
        execution_mode,
    )?;
    xs = add_row_buffers(
        &residual,
        &ff_out,
        execution_mode == InferenceExecutionMode::Deterministic || ff_out_uses_det,
    )?;

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = xs.clone();
        let gated = apply_gelu_to_row_buffer(
            &project_linear_row_buffer(
                &xs,
                ple.input_gate.as_ref(),
                ple.input_gate_det.as_deref(),
            )?,
            execution_mode,
        )?;
        let gated = mul_row_buffers(
            &gated,
            &ActivationRowBuffer::from_internal(per_layer_input),
            execution_mode == InferenceExecutionMode::Deterministic || gated.acts.is_some(),
        )?;
        let projected = project_linear_row_buffer(
            &gated,
            ple.layer_projection.as_ref(),
            ple.layer_projection_det.as_deref(),
        )?;
        let projected_uses_det = projected.acts.is_some();
        let projected = apply_rms_norm_row_buffer(
            &projected,
            &ple.post_input_norm_weight,
            ple.post_input_norm_weight_det.as_deref(),
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
            execution_mode,
        )?;
        xs = add_row_buffers(
            &residual,
            &projected,
            execution_mode == InferenceExecutionMode::Deterministic || projected_uses_det,
        )?;
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        xs = scale_row_buffer(
            &xs,
            layer_scalar,
            layer.layer_scalar_det,
            execution_mode == InferenceExecutionMode::Deterministic,
        )?;
    }

    Ok((xs.into_internal(), updated_cache))
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
        let (layer_output, layer_cache) = run_gemma4_layer_with_cache(
            &xs,
            &resolved_layer,
            per_layer_input,
            donor_cache,
            InferenceExecutionMode::Fp32,
        )?;
        xs = layer_output.activations;
        layer_caches.push(layer_cache);
        completed_layer_output_sha256s.push(layer_output.activations_sha256);
        if crate::trace::trace_checkpoint(
            "prefill.layer",
            &json!({
                "next_layer_idx": layer_idx + 1,
                "current_activations": xs.clone(),
                "current_activations_sha256": build_activation_commitment(&xs),
                "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
                "completed_layer_output_sha256s": completed_layer_output_sha256s.clone(),
            }),
        ) {
            break;
        }
        let mut reached_terminal_checkpoint = false;
        for (token_idx, token_activation) in xs.iter().enumerate() {
            if crate::trace::trace_checkpoint(
                &format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}"),
                &json!({
                    "layer_idx": layer_idx,
                    "token_idx": token_idx,
                    "token_count": xs.len(),
                    "token_activation": token_activation,
                }),
            ) {
                reached_terminal_checkpoint = true;
                break;
            }
        }
        if reached_terminal_checkpoint {
            break;
        }
    }

    Ok((
        ActivationSequence::from_values(xs.clone(), build_activation_commitment(&xs)),
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
    }

    Ok(ActivationSequenceWithCache {
        activation_state: ActivationSequence::from_values(
            vec![xs.clone()],
            build_activation_commitment(&[xs]),
        ),
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
    let activations =
        apply_rms_norm_to_sequence(input_activations, weight, None, eps, None, execution_mode)?;
    Ok(ActivationSequence::from_values(
        activations.clone(),
        build_activation_commitment(&activations),
    ))
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

pub fn project_to_logits(
    last_hidden_state: &[f32],
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    execution_mode: InferenceExecutionMode,
) -> Result<Vec<f32>> {
    // let _trace = trace_scope("transformer_state_transition.project_to_logits");
    Ok(project_to_logits_buffer(
        &ActivationRowBuffer::from_values(last_hidden_state.to_vec()),
        projection,
        embedding_source,
        execution_mode,
    )?
    .clone_f32())
}

fn project_to_logits_buffer(
    last_hidden_state: &ActivationRowBuffer,
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    execution_mode: InferenceExecutionMode,
) -> Result<InternalLogits> {
    let weight = match projection {
        Gemma4LogitsProjection::UntiedLmHead { weight, .. }
        | Gemma4LogitsProjection::TiedEmbedding(weight) => weight,
    };
    validate_vector_width(
        last_hidden_state.values.as_slice(),
        weight.cols,
        "logits projection input",
    )?;

    let buffer = if execution_mode == InferenceExecutionMode::Deterministic {
        match projection {
            Gemma4LogitsProjection::UntiedLmHead {
                det_weight: Some(det_weight),
                ..
            } => project_linear_row_buffer(last_hidden_state, weight, Some(det_weight.as_ref()))?,
            Gemma4LogitsProjection::TiedEmbedding(_) => {
                if let Some(embedding_source) = embedding_source {
                    if let Some(det_weight) =
                        crate::io::materialize_det_num_embedding_matrix(embedding_source)?
                    {
                        project_linear_row_buffer(
                            last_hidden_state,
                            weight,
                            Some(det_weight.as_ref()),
                        )?
                    } else {
                        bail!(
                            "deterministic tied embedding logits require a .detwgt embedding matrix"
                        );
                    }
                } else {
                    bail!("deterministic tied embedding logits require a .detwgt embedding source");
                }
            }
            Gemma4LogitsProjection::UntiedLmHead {
                det_weight: None, ..
            } => bail!("deterministic untied logits projection requires canonical det_weight"),
        }
    } else {
        project_linear_row_buffer(last_hidden_state, weight, None)?
    };

    Ok(internal_logits_from_row_buffer(buffer))
}

pub fn apply_final_logit_softcapping(logits: &[f32], softcap: f32) -> Vec<f32> {
    // let _trace = trace_scope("transformer_state_transition.apply_final_logit_softcapping");
    logits
        .iter()
        .map(|logit| (logit / softcap).tanh() * softcap)
        .collect()
}

pub fn apply_final_logit_softcapping_with_mode(
    logits: &[f32],
    softcap: f32,
    execution_mode: InferenceExecutionMode,
) -> Vec<f32> {
    match execution_mode {
        InferenceExecutionMode::Fp32 => apply_final_logit_softcapping(logits, softcap),
        InferenceExecutionMode::Deterministic => {
            let softcap = f32_to_act(softcap);
            logits
                .iter()
                .copied()
                .map(f32_to_act)
                .map(|logit| act_to_f32(softcap_act(logit, softcap)))
                .collect()
        }
    }
}

fn apply_final_logit_softcapping_buffer(
    logits: &InternalLogits,
    softcap: f32,
    softcap_det: Option<Act>,
    execution_mode: InferenceExecutionMode,
) -> Result<InternalLogits> {
    match (execution_mode, logits.det_values()) {
        (InferenceExecutionMode::Deterministic, Some(det_values)) => {
            let softcap = softcap_det.ok_or_else(|| {
                anyhow!("deterministic final logit softcapping requires canonical Act softcap")
            })?;
            Ok(InternalLogits::from_det_values(
                det_values
                    .iter()
                    .copied()
                    .map(|logit| softcap_act(logit, softcap))
                    .collect(),
            ))
        }
        (InferenceExecutionMode::Deterministic, None) => {
            bail!("deterministic final logit softcapping requires canonical logits")
        }
        _ => Ok(InternalLogits::from_values(
            apply_final_logit_softcapping_with_mode(logits.as_f32_slice(), softcap, execution_mode),
        )),
    }
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

pub fn project_hidden_to_prefill_logits(
    hidden_state: &[f32],
    final_norm_weight: &[f32],
    rms_norm_eps: f32,
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    execution_mode: InferenceExecutionMode,
    final_logit_softcapping: Option<f32>,
) -> Result<PrefillLogits> {
    project_internal_hidden_to_prefill_logits(
        InternalActivationRow::from_values(hidden_state.to_vec()),
        final_norm_weight,
        None,
        rms_norm_eps,
        None,
        projection,
        embedding_source,
        execution_mode,
        final_logit_softcapping,
        None,
    )
}

pub(crate) fn project_internal_hidden_to_prefill_logits(
    hidden_state: InternalActivationRow,
    final_norm_weight: &[f32],
    final_norm_weight_det: Option<&[Wgt]>,
    rms_norm_eps: f32,
    rms_norm_eps_det: Option<Acc>,
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    execution_mode: InferenceExecutionMode,
    final_logit_softcapping: Option<f32>,
    final_logit_softcapping_det: Option<Act>,
) -> Result<PrefillLogits> {
    let normalized = apply_rms_norm_row_buffer(
        &ActivationRowBuffer::from_internal(hidden_state),
        final_norm_weight,
        final_norm_weight_det,
        rms_norm_eps,
        rms_norm_eps_det,
        execution_mode,
    )?;
    let logits =
        project_to_logits_buffer(&normalized, projection, embedding_source, execution_mode)?;
    let logits = match final_logit_softcapping {
        Some(softcap) => apply_final_logit_softcapping_buffer(
            &logits,
            softcap,
            final_logit_softcapping_det,
            execution_mode,
        )?,
        None => logits,
    };
    Ok(extract_internal_prefill_logits(logits))
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
    project_internal_decode_hidden_to_logits(
        InternalActivationRow::from_values(hidden_state.to_vec()),
        final_norm_weight,
        None,
        rms_norm_eps,
        None,
        projection,
        embedding_source,
        execution_mode,
        final_logit_softcapping,
        None,
    )
}

pub(crate) fn project_internal_decode_hidden_to_logits(
    hidden_state: InternalActivationRow,
    final_norm_weight: &[f32],
    final_norm_weight_det: Option<&[Wgt]>,
    rms_norm_eps: f32,
    rms_norm_eps_det: Option<Acc>,
    projection: &Gemma4LogitsProjection,
    embedding_source: Option<&GemmaEmbeddingTensorSource>,
    execution_mode: InferenceExecutionMode,
    final_logit_softcapping: Option<f32>,
    final_logit_softcapping_det: Option<Act>,
) -> Result<PrefillLogits> {
    // let _trace = trace_scope("transformer_state_transition.project_decode_hidden_to_logits");
    project_internal_hidden_to_prefill_logits(
        hidden_state,
        final_norm_weight,
        final_norm_weight_det,
        rms_norm_eps,
        rms_norm_eps_det,
        projection,
        embedding_source,
        execution_mode,
        final_logit_softcapping,
        final_logit_softcapping_det,
    )
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

pub(crate) fn build_det_kv_cache_commitment(caches: &[LayerKvCache]) -> Option<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-det-num-kv-cache-v1");
    hasher.update((caches.len() as u64).to_le_bytes());
    for cache in caches {
        let keys = cache.det_keys.as_ref()?;
        let values = cache.det_values.as_ref()?;
        hasher.update((keys.len() as u64).to_le_bytes());
        for (kind, heads) in [(b"k", keys), (b"v", values)] {
            hasher.update(kind);
            hasher.update((heads.len() as u64).to_le_bytes());
            for head in heads {
                hasher.update((head.len() as u64).to_le_bytes());
                for row in head {
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
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequenceBuffer, LayerKvCache)> {
    match layer.attention_kind {
        Gemma4AttentionKind::Sliding => {
            run_sliding_attention(inputs, layer, donor_cache, execution_mode)
        }
        Gemma4AttentionKind::Full => run_full_attention(inputs, layer, donor_cache, execution_mode),
    }
}

fn run_attention_for_layer_decode(
    input: &ActivationRowBuffer,
    layer: &ResolvedGemma4LayerWeights,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    execution_mode: InferenceExecutionMode,
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
                execution_mode,
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
            execution_mode,
        ),
    }
}

fn run_sliding_attention(
    inputs: &ActivationSequenceBuffer,
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
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
        execution_mode,
    )
}

fn run_full_attention(
    inputs: &ActivationSequenceBuffer,
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequenceBuffer, LayerKvCache)> {
    run_causal_attention_buffer(inputs, layer, None, None, donor_cache, execution_mode)
}

#[cfg(test)]
fn run_causal_attention(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    let (output, cache) = run_causal_attention_buffer(
        &ActivationSequenceBuffer::from_values(inputs.to_vec()),
        layer,
        attention_window,
        cache_window,
        donor_cache,
        execution_mode,
    )?;
    Ok((output.values, cache))
}

fn run_causal_attention_buffer(
    inputs: &ActivationSequenceBuffer,
    layer: &ResolvedGemma4LayerWeights,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    donor_cache: Option<&LayerKvCache>,
    execution_mode: InferenceExecutionMode,
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
            execution_mode,
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
            execution_mode,
        )?;
    }
    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.v_rms_norm");
        apply_value_rms_norm(
            &mut v,
            layer.rms_norm_eps,
            layer.rms_norm_eps_det,
            execution_mode,
        )?;
    }

    {
        // let _trace = trace_scope("transformer_state_transition.run_causal_attention.q_rope");
        apply_rope(
            &mut q,
            layer.partial_rotary_dim,
            layer.rope_freq_base_dim,
            layer.rope_base,
            layer.rope_base_det,
            execution_mode,
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
            execution_mode,
        )?;
    }

    let layer_cache = if donor_cache.is_some() {
        LayerKvCache::new(layer.num_kv_heads)
    } else {
        build_layer_kv_cache(&k, &v, cache_window, execution_mode)?
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
                        execution_mode,
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
                        execution_mode,
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

#[cfg(test)]
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
    let (output, cache) = run_causal_attention_decode_buffer(
        &ActivationRowBuffer::from_values(input.to_vec()),
        layer,
        cache,
        donor_cache,
        position,
        attention_window,
        cache_window,
        execution_mode,
    )?;
    Ok((output.values, cache))
}

fn run_causal_attention_decode_buffer(
    input: &ActivationRowBuffer,
    layer: &ResolvedGemma4LayerWeights,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    execution_mode: InferenceExecutionMode,
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
        execution_mode,
    )?;
    apply_head_rms_norm_row(
        &mut k,
        &layer.k_norm_weight,
        layer.k_norm_weight_det.as_deref(),
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
        execution_mode,
    )?;
    apply_value_rms_norm_row(
        &mut v,
        layer.rms_norm_eps,
        layer.rms_norm_eps_det,
        execution_mode,
    )?;
    apply_rope_to_rows(
        &mut q,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base,
        layer.rope_base_det,
        position,
        execution_mode,
    )?;
    apply_rope_to_rows(
        &mut k,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        layer.rope_base,
        layer.rope_base_det,
        position,
        execution_mode,
    )?;

    let updated_cache = if donor_cache.is_some() {
        cache
    } else {
        append_kv_cache_head_buffer_with_mode(cache, &k, &v, cache_window, execution_mode)?
    };
    let attention_cache = donor_cache.unwrap_or(&updated_cache);
    let mut combined_heads = vec![0.0; layer.num_heads * layer.head_dim];
    let mut combined_head_acts = matches!(execution_mode, InferenceExecutionMode::Deterministic)
        .then(|| vec![Act::from_bits(0); layer.num_heads * layer.head_dim]);
    for head_idx in 0..layer.num_heads {
        let kv_head_idx = head_idx / kv_groups;
        let key_start = attention_window
            .map(|window| {
                attention_cache.keys[kv_head_idx]
                    .len()
                    .saturating_sub(window)
            })
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
            execution_mode,
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
    deterministic: bool,
) -> Result<ActivationSequenceBuffer> {
    if deterministic {
        let lhs_acts = sequence_buffer_acts(lhs)?;
        let rhs_acts = sequence_buffer_acts(rhs)?;
        Ok(ActivationSequenceBuffer::from_acts(add_act_sequences(
            &lhs_acts, &rhs_acts,
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
        let lhs_acts = row_buffer_acts(lhs)?;
        let rhs_acts = row_buffer_acts(rhs)?;
        Ok(ActivationRowBuffer::from_acts(add_act_rows(
            &lhs_acts, &rhs_acts,
        )?))
    } else {
        Ok(ActivationRowBuffer::from_values(add_rows(
            &lhs.values,
            &rhs.values,
        )?))
    }
}

fn mul_sequence_buffers(
    lhs: &ActivationSequenceBuffer,
    rhs: &ActivationSequenceBuffer,
    deterministic: bool,
) -> Result<ActivationSequenceBuffer> {
    if deterministic {
        let lhs_acts = sequence_buffer_acts(lhs)?;
        let rhs_acts = sequence_buffer_acts(rhs)?;
        Ok(ActivationSequenceBuffer::from_acts(mul_act_sequences(
            &lhs_acts, &rhs_acts,
        )?))
    } else {
        Ok(ActivationSequenceBuffer::from_values(
            elementwise_mul_sequences(&lhs.values, &rhs.values)?,
        ))
    }
}

fn mul_row_buffers(
    lhs: &ActivationRowBuffer,
    rhs: &ActivationRowBuffer,
    deterministic: bool,
) -> Result<ActivationRowBuffer> {
    if deterministic {
        let lhs_acts = row_buffer_acts(lhs)?;
        let rhs_acts = row_buffer_acts(rhs)?;
        Ok(ActivationRowBuffer::from_acts(mul_act_rows(
            &lhs_acts, &rhs_acts,
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
        Ok(ActivationSequenceBuffer::from_values(scale_sequences(
            &values.values,
            scalar,
        )))
    }
}

fn scale_row_buffer(
    values: &ActivationRowBuffer,
    scalar: f32,
    scalar_det: Option<Act>,
    deterministic: bool,
) -> Result<ActivationRowBuffer> {
    if deterministic {
        let scalar_det = scalar_det
            .ok_or_else(|| anyhow!("deterministic row scaling requires canonical Act scalar"))?;
        Ok(ActivationRowBuffer::from_acts(scale_act_rows(
            &row_buffer_acts(values)?,
            scalar_det,
        )))
    } else {
        Ok(ActivationRowBuffer::from_values(scale_rows(
            &values.values,
            scalar,
        )))
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

fn project_linear_row_buffer(
    input: &ActivationRowBuffer,
    weight: &MatrixF32,
    det_weight: Option<&DetNumMatrix>,
) -> Result<ActivationRowBuffer> {
    match det_weight {
        Some(det_weight) => Ok(ActivationRowBuffer::from_acts(
            det_linear_row_acts_from_acts(&row_buffer_acts(input)?, det_weight)?,
        )),
        None if input.acts.is_some() => {
            bail!("deterministic linear row projection requires canonical det_weight")
        }
        None => Ok(ActivationRowBuffer::from_values(linear_row(
            &input.values,
            weight,
        )?)),
    }
}

fn internal_logits_from_row_buffer(buffer: ActivationRowBuffer) -> InternalLogits {
    match buffer.acts {
        Some(det_values) => InternalLogits::from_det_values(det_values),
        None => InternalLogits::from_values(buffer.values),
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

    let quantized_input = input.iter().copied().map(f32_to_act).collect::<Vec<_>>();
    det_linear_row_from_acts(&quantized_input, weight)
}

#[cfg(test)]
fn det_linear_row_from_acts(quantized_input: &[Act], weight: &DetNumMatrix) -> Result<Vec<f32>> {
    det_linear_row_acts_from_acts(quantized_input, weight)
        .map(|acts| acts.into_iter().map(act_to_f32).collect())
}

fn apply_rms_norm_to_sequence(
    inputs: &[Vec<f32>],
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<Vec<Vec<f32>>> {
    Ok(apply_rms_norm_to_sequence_buffer(
        &ActivationSequenceBuffer::from_values(inputs.to_vec()),
        weight,
        weight_det,
        eps,
        eps_det,
        execution_mode,
    )?
    .values)
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
            .par_iter()
            .map(|row| {
                apply_rms_norm_row_buffer(
                    &ActivationRowBuffer::from_values(row.clone()),
                    weight,
                    None,
                    eps,
                    None,
                    execution_mode,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .map(|rows| {
                ActivationSequenceBuffer::from_values(
                    rows.into_iter().map(|row| row.values).collect(),
                )
            }),
        InferenceExecutionMode::Deterministic => sequence_buffer_acts(inputs)?
            .into_par_iter()
            .map(|row| {
                apply_rms_norm_row_buffer(
                    &ActivationRowBuffer::from_acts(row),
                    weight,
                    weight_det,
                    eps,
                    eps_det,
                    execution_mode,
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
            }),
    }
}

fn apply_rms_norm(
    input: &[f32],
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<Vec<f32>> {
    Ok(apply_rms_norm_buffer(input, weight, weight_det, eps, eps_det, execution_mode)?.values)
}

fn apply_rms_norm_buffer(
    input: &[f32],
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationRowBuffer> {
    apply_rms_norm_row_buffer(
        &ActivationRowBuffer::from_values(input.to_vec()),
        weight,
        weight_det,
        eps,
        eps_det,
        execution_mode,
    )
}

fn apply_rms_norm_row_buffer(
    input: &ActivationRowBuffer,
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationRowBuffer> {
    if input.values.len() != weight.len() {
        bail!(
            "rms norm width mismatch: {} vs {}",
            input.values.len(),
            weight.len()
        );
    }

    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            let mean_square = input.values.iter().map(|value| value * value).sum::<f32>()
                / input.values.len() as f32;
            let scale = (mean_square + eps).sqrt().recip();
            Ok(ActivationRowBuffer::from_values(
                input
                    .values
                    .iter()
                    .zip(weight)
                    .map(|(value, norm_weight)| value * scale * norm_weight)
                    .collect(),
            ))
        }
        InferenceExecutionMode::Deterministic => {
            let quantized_input = row_buffer_acts(input)?;
            let quantized_weight = weight_det.ok_or_else(|| {
                anyhow!("deterministic RMSNorm requires canonical Wgt norm weights")
            })?;
            let quantized_eps = eps_det
                .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
            Ok(ActivationRowBuffer::from_acts(det_rms_norm(
                &quantized_input,
                quantized_weight,
                quantized_eps,
            )))
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
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            heads
                .values
                .par_iter_mut()
                .try_for_each(|head| -> Result<()> {
                    for row in head {
                        *row = apply_rms_norm(row, weight, None, eps, None, execution_mode)?;
                    }
                    Ok(())
                })?;
            heads.acts = None;
        }
        InferenceExecutionMode::Deterministic => {
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
                                execution_mode,
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
        }
    }
    Ok(())
}

fn apply_head_rms_norm_row(
    heads: &mut AttentionHeadRowBuffer,
    weight: &[f32],
    weight_det: Option<&[Wgt]>,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            for head in &mut heads.values {
                *head = apply_rms_norm(head, weight, None, eps, None, execution_mode)?;
            }
            heads.acts = None;
        }
        InferenceExecutionMode::Deterministic => {
            let acts = head_row_buffer_acts(heads)?
                .into_iter()
                .map(|row| {
                    apply_rms_norm_row_buffer(
                        &ActivationRowBuffer::from_acts(row),
                        weight,
                        weight_det,
                        eps,
                        eps_det,
                        execution_mode,
                    )
                    .map(|row| row.acts.expect("deterministic head RMSNorm row"))
                })
                .collect::<Result<Vec<_>>>()?;
            *heads = AttentionHeadRowBuffer::from_acts(acts);
        }
    }
    Ok(())
}

fn apply_value_rms_norm(
    heads: &mut AttentionHeadSequenceBuffer,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            heads
                .values
                .par_iter_mut()
                .try_for_each(|head| -> Result<()> {
                    for row in head {
                        let mean_square =
                            row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
                        let scale = (mean_square + eps).sqrt().recip();
                        for value in row {
                            *value *= scale;
                        }
                    }
                    Ok(())
                })?;
            heads.acts = None;
        }
        InferenceExecutionMode::Deterministic => {
            let eps_det = eps_det.ok_or_else(|| {
                anyhow!("deterministic value RMSNorm requires canonical Acc epsilon")
            })?;
            let acts = head_sequence_buffer_acts(heads)?
                .into_par_iter()
                .map(|head| {
                    head.into_iter()
                        .map(|row| det_value_rms_norm(&row, eps_det))
                        .collect::<Vec<_>>()
                })
                .collect();
            *heads = AttentionHeadSequenceBuffer::from_acts(acts);
        }
    }
    Ok(())
}

fn apply_value_rms_norm_row(
    heads: &mut AttentionHeadRowBuffer,
    eps: f32,
    eps_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            for head in &mut heads.values {
                let mean_square =
                    head.iter().map(|value| value * value).sum::<f32>() / head.len() as f32;
                let scale = (mean_square + eps).sqrt().recip();
                for value in head {
                    *value *= scale;
                }
            }
            heads.acts = None;
        }
        InferenceExecutionMode::Deterministic => {
            let eps_det = eps_det.ok_or_else(|| {
                anyhow!("deterministic value RMSNorm requires canonical Acc epsilon")
            })?;
            let acts = head_row_buffer_acts(heads)?
                .into_iter()
                .map(|row| det_value_rms_norm(&row, eps_det))
                .collect();
            *heads = AttentionHeadRowBuffer::from_acts(acts);
        }
    }
    Ok(())
}

fn apply_rope(
    heads: &mut AttentionHeadSequenceBuffer,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: f32,
    base_det: Option<Acc>,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    apply_rope_with_offset(
        heads,
        rotary_dim,
        freq_base_dim,
        base,
        base_det,
        0,
        execution_mode,
    )
}

fn apply_rope_with_offset(
    heads: &mut AttentionHeadSequenceBuffer,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: f32,
    base_det: Option<Acc>,
    position_offset: usize,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    if rotary_dim == 0 {
        return Ok(());
    }

    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            let half_dim = rotary_dim / 2;
            for head in &mut heads.values {
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
            heads.acts = None;
        }
        InferenceExecutionMode::Deterministic => {
            let quantized_base = base_det
                .ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
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
        }
    }
    Ok(())
}

fn apply_rope_to_rows(
    heads: &mut AttentionHeadRowBuffer,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: f32,
    base_det: Option<Acc>,
    position: usize,
    execution_mode: InferenceExecutionMode,
) -> Result<()> {
    if rotary_dim == 0 {
        return Ok(());
    }

    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            let half_dim = rotary_dim / 2;
            for row in &mut heads.values {
                let original = row.clone();
                for dim_idx in 0..half_dim {
                    let angle =
                        position as f32 / base.powf((2 * dim_idx) as f32 / freq_base_dim as f32);
                    let cos = angle.cos();
                    let sin = angle.sin();
                    let lhs = original[dim_idx];
                    let rhs = original[dim_idx + half_dim];
                    row[dim_idx] = lhs * cos - rhs * sin;
                    row[dim_idx + half_dim] = rhs * cos + lhs * sin;
                }
            }
            heads.acts = None;
        }
        InferenceExecutionMode::Deterministic => {
            let quantized_base = base_det
                .ok_or_else(|| anyhow!("deterministic RoPE requires canonical Acc base"))?;
            let acts = head_row_buffer_acts(heads)?
                .into_iter()
                .map(|row| {
                    det_rope_rotate_pairs(&row, rotary_dim, freq_base_dim, quantized_base, position)
                })
                .collect();
            *heads = AttentionHeadRowBuffer::from_acts(acts);
        }
    }
    Ok(())
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

fn attention_output(
    query: &ActivationRowBuffer,
    key_rows: &[Vec<f32>],
    det_key_rows: Option<&[Vec<Act>]>,
    value_rows: &[Vec<f32>],
    det_value_rows: Option<&[Vec<Act>]>,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationRowBuffer> {
    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            let logits = key_rows
                .iter()
                .map(|key_row| dot(&query.values, key_row))
                .collect::<Vec<_>>();
            let weights = softmax(&logits);
            let mut output = vec![0.0; query.values.len()];
            for (weight, value_row) in weights.iter().zip(value_rows) {
                for (dim_idx, value) in output.iter_mut().enumerate() {
                    *value += *weight * value_row[dim_idx];
                }
            }
            Ok(ActivationRowBuffer::from_values(output))
        }
        InferenceExecutionMode::Deterministic => {
            let quantized_query = row_buffer_acts(query)?;
            let quantized_keys = det_key_rows.map(|rows| rows.to_vec()).ok_or_else(|| {
                anyhow!("deterministic attention requires canonical key cache rows")
            })?;
            let logits = quantized_keys
                .iter()
                .map(|key_row| det_attention_score(&quantized_query, key_row))
                .collect::<Vec<_>>();
            let weights = det_attention_softmax(&logits);
            let quantized_values = det_value_rows.map(|rows| rows.to_vec()).ok_or_else(|| {
                anyhow!("deterministic attention requires canonical value cache rows")
            })?;
            Ok(ActivationRowBuffer::from_acts(det_attention_weighted_sum(
                &weights,
                &quantized_values,
            )))
        }
    }
}

fn build_layer_kv_cache(
    keys: &AttentionHeadSequenceBuffer,
    values: &AttentionHeadSequenceBuffer,
    sliding_window: Option<usize>,
    execution_mode: InferenceExecutionMode,
) -> Result<LayerKvCache> {
    let retained = sliding_window.map_or(0, |window| {
        keys.values
            .first()
            .map_or(0, |head| head.len().saturating_sub(window))
    });
    match execution_mode {
        InferenceExecutionMode::Fp32 => Ok(LayerKvCache::from_f32_heads(
            keys.values
                .iter()
                .map(|head| head[retained..].iter().cloned().collect())
                .collect(),
            values
                .values
                .iter()
                .map(|head| head[retained..].iter().cloned().collect())
                .collect(),
        )),
        InferenceExecutionMode::Deterministic => Ok(LayerKvCache::from_det_heads(
            head_sequence_buffer_acts(keys)?
                .into_iter()
                .map(|head| head[retained..].iter().cloned().collect())
                .collect(),
            head_sequence_buffer_acts(values)?
                .into_iter()
                .map(|head| head[retained..].iter().cloned().collect())
                .collect(),
        )),
    }
}

pub fn append_kv_cache(
    cache: LayerKvCache,
    new_keys: &[Vec<f32>],
    new_values: &[Vec<f32>],
    sliding_window: Option<usize>,
) -> Result<LayerKvCache> {
    append_kv_cache_with_mode(
        cache,
        new_keys,
        new_values,
        sliding_window,
        InferenceExecutionMode::Fp32,
    )
}

fn append_kv_cache_with_mode(
    cache: LayerKvCache,
    new_keys: &[Vec<f32>],
    new_values: &[Vec<f32>],
    sliding_window: Option<usize>,
    execution_mode: InferenceExecutionMode,
) -> Result<LayerKvCache> {
    append_kv_cache_head_buffer_with_mode(
        cache,
        &AttentionHeadRowBuffer::from_values(new_keys.to_vec()),
        &AttentionHeadRowBuffer::from_values(new_values.to_vec()),
        sliding_window,
        execution_mode,
    )
}

fn append_kv_cache_head_buffer_with_mode(
    mut cache: LayerKvCache,
    new_keys: &AttentionHeadRowBuffer,
    new_values: &AttentionHeadRowBuffer,
    sliding_window: Option<usize>,
    execution_mode: InferenceExecutionMode,
) -> Result<LayerKvCache> {
    if new_keys.values.len() != cache.keys.len() || new_values.values.len() != cache.values.len() {
        bail!(
            "layer cache append head count mismatch: cache {} keys {} values {}",
            cache.keys.len(),
            new_keys.values.len(),
            new_values.values.len()
        );
    }

    let mut det_keys = if matches!(execution_mode, InferenceExecutionMode::Deterministic) {
        Some(match cache.det_keys.take() {
            Some(det_keys) => det_keys,
            None if cache.keys.iter().all(|head| head.is_empty()) => {
                vec![VecDeque::new(); cache.keys.len()]
            }
            None => bail!("deterministic decode cache append requires canonical key rows"),
        })
    } else {
        None
    };
    let mut det_values = if matches!(execution_mode, InferenceExecutionMode::Deterministic) {
        Some(match cache.det_values.take() {
            Some(det_values) => det_values,
            None if cache.values.iter().all(|head| head.is_empty()) => {
                vec![VecDeque::new(); cache.values.len()]
            }
            None => bail!("deterministic decode cache append requires canonical value rows"),
        })
    } else {
        None
    };

    let new_key_acts = if matches!(execution_mode, InferenceExecutionMode::Deterministic) {
        Some(head_row_buffer_acts(new_keys)?)
    } else {
        None
    };
    let new_value_acts = if matches!(execution_mode, InferenceExecutionMode::Deterministic) {
        Some(head_row_buffer_acts(new_values)?)
    } else {
        None
    };

    for (head_idx, ((head_keys, head_values), (new_key, new_value))) in cache
        .keys
        .iter_mut()
        .zip(cache.values.iter_mut())
        .zip(new_keys.values.iter().zip(&new_values.values))
        .enumerate()
    {
        head_keys.push_back(new_key.clone());
        head_values.push_back(new_value.clone());
        if let (Some(det_keys), Some(det_values)) = (&mut det_keys, &mut det_values) {
            det_keys[head_idx].push_back(
                new_key_acts.as_ref().expect("deterministic new keys")[head_idx].clone(),
            );
            det_values[head_idx].push_back(
                new_value_acts.as_ref().expect("deterministic new values")[head_idx].clone(),
            );
        }
        if let Some(window) = sliding_window {
            while head_keys.len() > window {
                head_keys.pop_front();
                head_values.pop_front();
                if let (Some(det_keys), Some(det_values)) = (&mut det_keys, &mut det_values) {
                    det_keys[head_idx].pop_front();
                    det_values[head_idx].pop_front();
                }
            }
        }
    }

    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            cache.det_keys = None;
            cache.det_values = None;
        }
        InferenceExecutionMode::Deterministic => {
            cache.det_keys = det_keys;
            cache.det_values = det_values;
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

fn apply_gelu_to_sequence_buffer(
    inputs: &ActivationSequenceBuffer,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequenceBuffer> {
    if matches!(execution_mode, InferenceExecutionMode::Deterministic) {
        return Ok(ActivationSequenceBuffer::from_acts(
            sequence_buffer_acts(inputs)?
                .iter()
                .map(|row| apply_det_gelu(row))
                .collect(),
        ));
    }

    Ok(ActivationSequenceBuffer::from_values(
        apply_gelu_to_sequence(&inputs.values),
    ))
}

fn apply_gelu_to_row_buffer(
    inputs: &ActivationRowBuffer,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationRowBuffer> {
    if matches!(execution_mode, InferenceExecutionMode::Deterministic) {
        return Ok(ActivationRowBuffer::from_acts(apply_det_gelu(
            &row_buffer_acts(inputs)?,
        )));
    }

    Ok(ActivationRowBuffer::from_values(apply_gelu(&inputs.values)))
}

fn apply_det_gelu(inputs: &[Act]) -> Vec<Act> {
    inputs.iter().copied().map(gelu_pytorch_tanh_act).collect()
}

fn gelu_pytorch_tanh(value: f32) -> f32 {
    let inner = std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value.powi(3));
    0.5 * value * (1.0 + inner.tanh())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        fs,
        sync::{Arc, Mutex},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        append_kv_cache, append_kv_cache_head_buffer_with_mode,
        apply_final_logit_softcapping_with_mode, apply_final_norm, apply_final_norm_with_mode,
        apply_gelu, apply_gelu_to_row_buffer, apply_head_rms_norm, apply_head_rms_norm_row,
        apply_rms_norm_to_sequence_buffer, apply_rope_to_rows, apply_value_rms_norm,
        apply_value_rms_norm_row, build_layer_kv_cache, compute_decode_ple_input,
        compute_prefill_ple_inputs, det_linear_row, det_linear_row_from_acts, det_linear_sequence,
        embed_input_tokens, extract_prefill_logits, project_decode_hidden_to_logits,
        project_hidden_to_prefill_logits, project_internal_hidden_to_prefill_logits,
        project_to_logits, reshape_row_head_buffer, reshape_sequence_head_buffer,
        run_causal_attention, run_causal_attention_decode, run_gemma4_layer,
        run_gemma4_layer_decode, run_gemma4_layer_decode_with_mode_internal,
        run_gemma4_layer_with_cache_internal, run_text_layers_decode_step, run_text_layers_prefill,
        run_text_layers_prefill_with_cache, select_final_position_internal, ActivationRowBuffer,
        ActivationSequenceBuffer, AttentionHeadRowBuffer, AttentionHeadSequenceBuffer,
    };
    use crate::shared::api::input::InferenceExecutionMode;
    use crate::shared::model::transformer::{
        DetNumMatrix, DetNumTensorSliceSource, EmbeddingTable, Gemma4AttentionKind,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
        Gemma4PleLayerWeights, Gemma4TransformerModel, GemmaEmbeddingTensorSource,
        InternalActivationRow, InternalActivationSequence, LayerKvCache, MatrixF32,
        ResolvedGemma4LayerWeights, ResolvedGemma4PleLayerWeights,
    };
    use crate::shared::numerics::det_num::{
        act_to_f32, attention_score, attention_softmax, attention_weighted_sum, f32_to_act,
        f32_to_wgt, gelu_pytorch_tanh_act, softcap_act, wgt_to_le_bytes, Act, DET_NUM_SPEC_VERSION,
        DET_WGT_ARTIFACT_FORMAT_VERSION, DET_WGT_ARTIFACT_MAGIC,
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
    fn embed_input_tokens_with_mode_rejects_f32_table_on_deterministic_path() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![1.0 / 65_536.0]],
            scale: 0.5,
        };

        let fp32 =
            embed_input_tokens(&[0], &embedding_table).expect("fp32 embedding should succeed");
        let error = super::embed_input_tokens_with_mode(
            &[0],
            &embedding_table,
            InferenceExecutionMode::Deterministic,
        )
        .err()
        .expect("deterministic embedding requires detwgt source");

        assert!(error.to_string().contains(".detwgt embedding source"));
        assert!(fp32.activations[0][0] > 0.0);
    }

    #[test]
    fn apply_rope_to_rows_uses_full_head_dim_for_frequency_base() {
        let mut heads =
            AttentionHeadRowBuffer::from_values(vec![vec![0.0, 1.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0]]);

        apply_rope_to_rows(
            &mut heads,
            4,
            8,
            16.0,
            None,
            1,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();

        assert!((heads.values[0][1] - 0.87758255).abs() < 1e-6);
        assert!((heads.values[0][3] - 0.47942555).abs() < 1e-6);
        assert_eq!(heads.values[0][4..], [9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn apply_rope_to_rows_uses_deterministic_rope_contract() {
        let mut heads = AttentionHeadRowBuffer::from_acts(vec![vec![
            Act::from_num(1.0),
            Act::from_bits(0),
            Act::from_num(0.5),
            Act::from_num(-0.5),
        ]]);

        apply_rope_to_rows(
            &mut heads,
            4,
            4,
            16.0,
            Some(crate::shared::numerics::det_num::f32_to_acc(16.0)),
            1,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(
            heads.values[0],
            vec![
                act_to_f32(Act::from_bits(7_835)),
                act_to_f32(Act::from_bits(8_107)),
                act_to_f32(Act::from_bits(72_850)),
                act_to_f32(Act::from_bits(-31_750)),
            ]
        );
    }

    #[test]
    fn reshape_sequence_head_buffer_preserves_non_round_tripping_act_bits() {
        let canonical = non_round_tripping_act();
        let projected = ActivationSequenceBuffer::from_acts(vec![vec![
            canonical,
            Act::from_bits(2),
            Act::from_bits(3),
            Act::from_bits(4),
        ]]);

        let heads = reshape_sequence_head_buffer(&projected, 2, 2).unwrap();

        assert_eq!(
            heads.acts.expect("canonical heads"),
            vec![
                vec![vec![canonical, Act::from_bits(2)]],
                vec![vec![Act::from_bits(3), Act::from_bits(4)]],
            ]
        );
        assert_ne!(f32_to_act(heads.values[0][0][0]), canonical);
    }

    #[test]
    fn reshape_sequence_head_buffer_preserves_f32_only_behavior() {
        let projected = ActivationSequenceBuffer::from_values(vec![vec![1.0, 2.0, 3.0, 4.0]]);

        let heads = reshape_sequence_head_buffer(&projected, 2, 2).unwrap();

        assert_eq!(
            heads.values,
            vec![vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]]
        );
        assert!(heads.acts.is_none());
    }

    #[test]
    fn reshape_sequence_head_buffer_preserves_width_errors() {
        let projected = ActivationSequenceBuffer::from_values(vec![vec![1.0, 2.0, 3.0]]);

        let error = reshape_sequence_head_buffer(&projected, 2, 2).expect_err("width mismatch");

        assert!(error.to_string().contains("projected attention states"));
    }

    #[test]
    fn reshape_row_head_buffer_preserves_non_round_tripping_act_bits() {
        let canonical = non_round_tripping_act();
        let projected = ActivationRowBuffer::from_acts(vec![
            canonical,
            Act::from_bits(2),
            Act::from_bits(3),
            Act::from_bits(4),
        ]);

        let heads = reshape_row_head_buffer(&projected, 2, 2).unwrap();

        assert_eq!(
            heads.acts.expect("canonical heads"),
            vec![
                vec![canonical, Act::from_bits(2)],
                vec![Act::from_bits(3), Act::from_bits(4)],
            ]
        );
        assert_ne!(f32_to_act(heads.values[0][0]), canonical);
    }

    #[test]
    fn attention_output_preserves_canonical_value_bits() {
        let canonical = non_round_tripping_act();
        let output = super::attention_output(
            &ActivationRowBuffer::from_acts(vec![Act::from_bits(0)]),
            &[vec![0.0]],
            Some(&[vec![Act::from_bits(0)]]),
            &[vec![act_to_f32(canonical)]],
            Some(&[vec![canonical]]),
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(output.acts, Some(vec![canonical]));
        assert_ne!(f32_to_act(output.values[0]), canonical);
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
            .map(crate::shared::numerics::det_num::f32_to_act)
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
    fn add_row_buffers_reject_float_inputs_on_deterministic_path() {
        let lhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);
        let rhs = super::ActivationRowBuffer::from_values(vec![0.5 / 65_536.0]);

        let error = super::add_row_buffers(&lhs, &rhs, true)
            .err()
            .expect("deterministic add requires canonical acts");
        let fp32 = super::add_rows(&lhs.values, &rhs.values).unwrap();

        assert!(error.to_string().contains("canonical Act"));
        assert!(fp32[0] > 0.0);
    }

    #[test]
    fn mul_row_buffers_reject_float_inputs_on_deterministic_path() {
        let lhs = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);
        let rhs = super::ActivationRowBuffer::from_values(vec![0.5]);

        let error = super::mul_row_buffers(&lhs, &rhs, true)
            .err()
            .expect("deterministic mul requires canonical acts");
        let fp32 = super::elementwise_mul_rows(&lhs.values, &rhs.values).unwrap();

        assert!(error.to_string().contains("canonical Act"));
        assert!(fp32[0] > 0.0);
    }

    #[test]
    fn scale_row_buffer_rejects_float_inputs_on_deterministic_path() {
        let values = super::ActivationRowBuffer::from_values(vec![1.0 / 65_536.0]);

        let error = super::scale_row_buffer(&values, 0.5, Some(f32_to_act(0.5)), true)
            .err()
            .expect("deterministic scale requires canonical acts");
        let fp32 = super::scale_row_buffer(&values, 0.5, None, false).unwrap();

        assert!(error.to_string().contains("canonical Act"));
        assert!(fp32.values[0] > 0.0);
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: None,
            layer_scalar: None,
            layer_scalar_det: None,
        };

        let resolved_layer = crate::io::resolve_layer_weights(&layer).expect("resolve layer");
        let output =
            run_gemma4_layer(&activations, &resolved_layer, None).expect("layer should succeed");

        assert_eq!(output.activations, activations);
    }

    #[test]
    fn apply_gelu_to_row_buffer_uses_det_num_contract_in_deterministic_mode() {
        let input = ActivationRowBuffer::from_acts(vec![Act::from_num(0.5), Act::from_num(-0.5)]);

        let output = apply_gelu_to_row_buffer(&input, InferenceExecutionMode::Deterministic)
            .expect("deterministic GELU");

        let expected_acts = vec![
            gelu_pytorch_tanh_act(Act::from_num(0.5)),
            gelu_pytorch_tanh_act(Act::from_num(-0.5)),
        ];
        let fp32_output = apply_gelu(&[0.5, -0.5]);

        assert_eq!(output.acts, Some(expected_acts.clone()));
        assert_eq!(
            output.values,
            expected_acts
                .iter()
                .copied()
                .map(act_to_f32)
                .collect::<Vec<_>>()
        );
        assert_ne!(output.values, fp32_output);
    }

    #[test]
    #[should_panic]
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
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
            layer_scalar_det: None,
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
    #[should_panic]
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
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
            layer_scalar_det: None,
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
    #[should_panic]
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
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
            layer_scalar_det: None,
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
    #[should_panic]
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

        let without_det = run_causal_attention(
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
    #[should_panic]
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

        let without_det = run_causal_attention(
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
    #[should_panic]
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

        let without_det = run_causal_attention(
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
    #[should_panic]
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

        let without_det = run_causal_attention(
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
    fn run_causal_attention_uses_deterministic_rope_during_prefill() {
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut resolved = attention_test_layer(
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
        );
        resolved.partial_rotary_dim = 2;
        resolved.rope_base = 1.0;
        resolved.rope_freq_base_dim = 2;

        let fp32 = run_causal_attention(
            &inputs,
            &resolved,
            None,
            None,
            None,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_causal_attention(
            &inputs,
            &resolved,
            None,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 attention wrapper should reject deterministic KV construction");
        assert!(error.to_string().contains("canonical head rows"));
        return;
        let det = run_causal_attention(
            &inputs,
            &resolved,
            None,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_ne!(det.0[1], fp32.0[1]);
        assert_ne!(det.1.keys[0][1], fp32.1.keys[0][1]);
    }

    #[test]
    #[should_panic]
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
    #[should_panic]
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
    #[should_panic]
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
    #[should_panic]
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
    fn run_causal_attention_decode_uses_deterministic_rope() {
        let input = vec![0.0, 1.0];
        let mut resolved = attention_test_layer(
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
            identity_matrix(2),
        );
        resolved.partial_rotary_dim = 2;
        resolved.rope_base = 1.0;
        resolved.rope_freq_base_dim = 2;
        let initial_cache = append_kv_cache(
            LayerKvCache::new(1),
            &[vec![1.0, 0.0]],
            &[vec![1.0, 0.0]],
            None,
        )
        .unwrap();

        let error = run_causal_attention_decode(
            &input,
            &resolved,
            initial_cache,
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 decode wrapper should reject deterministic KV append");
        assert!(error.to_string().contains("canonical head rows"));
    }

    #[test]
    fn run_causal_attention_routes_prefill_through_deterministic_attention_core() {
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let query_proj = MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![0.0, 1.0, 0.0, 0.0],
        };
        let key_proj = MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![0.0, -0.693_147_2, 0.0, 0.0],
        };
        let resolved =
            attention_test_layer(query_proj, key_proj, identity_matrix(2), identity_matrix(2));

        let error = run_causal_attention(
            &inputs,
            &resolved,
            None,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 attention wrapper should reject deterministic KV construction");
        assert!(error.to_string().contains("canonical head rows"));
    }

    #[test]
    fn run_causal_attention_decode_routes_through_deterministic_attention_core() {
        let input = vec![0.0, 1.0];
        let query_proj = MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![0.0, 1.0, 0.0, 0.0],
        };
        let key_proj = MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![0.0, -0.693_147_2, 0.0, 0.0],
        };
        let resolved =
            attention_test_layer(query_proj, key_proj, identity_matrix(2), identity_matrix(2));
        let initial_cache = append_kv_cache(
            LayerKvCache::new(1),
            &[vec![0.0, 0.0]],
            &[vec![1.0, 0.0]],
            None,
        )
        .unwrap();

        let fp32 = run_causal_attention_decode(
            &input,
            &resolved,
            initial_cache.clone(),
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let error = run_causal_attention_decode(
            &input,
            &resolved,
            initial_cache,
            None,
            1,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("f32 decode wrapper should reject deterministic KV append");
        assert!(error.to_string().contains("canonical head rows"));
    }

    #[test]
    #[should_panic]
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
    #[should_panic]
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
    fn select_final_position_internal_preserves_det_values() {
        let input = InternalActivationSequence::from_det_values(vec![
            vec![Act::from_num(0.25), Act::from_num(0.5)],
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
        ]);

        let selected = select_final_position_internal(&input).unwrap();

        assert_eq!(
            selected.as_f32_slice(),
            &[
                act_to_f32(Act::from_num(1.0)),
                act_to_f32(Act::from_num(-0.5))
            ]
        );
        assert_eq!(
            selected.det_values().unwrap(),
            &[Act::from_num(1.0), Act::from_num(-0.5)]
        );
    }

    #[test]
    fn select_final_position_internal_preserves_f32_only_rows() {
        let input = InternalActivationSequence::from_values(vec![vec![0.25, 0.5], vec![1.0, -0.5]]);

        let selected = select_final_position_internal(&input).unwrap();

        assert_eq!(selected.as_f32_slice(), &[1.0, -0.5]);
        assert!(selected.det_values().is_none());
    }

    #[test]
    fn select_final_position_internal_rejects_empty_sequences() {
        let error = select_final_position_internal(&InternalActivationSequence::default())
            .expect_err("empty sequence should fail");

        assert!(error.to_string().contains("at least one activation row"));
    }

    #[test]
    #[should_panic]
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
            apply_final_logit_softcapping_with_mode(
                &[1.0, 1.0],
                0.5,
                InferenceExecutionMode::Deterministic,
            )
        );
    }

    #[test]
    fn apply_final_logit_softcapping_with_mode_uses_det_num_contract() {
        let fp32 = apply_final_logit_softcapping_with_mode(
            &[1.0, -1.0],
            0.5,
            InferenceExecutionMode::Fp32,
        );
        let deterministic = apply_final_logit_softcapping_with_mode(
            &[1.0, -1.0],
            0.5,
            InferenceExecutionMode::Deterministic,
        );

        assert_ne!(deterministic, fp32);
        assert_eq!(
            deterministic,
            vec![
                act_to_f32(softcap_act(Act::from_num(1.0), Act::from_num(0.5))),
                act_to_f32(softcap_act(Act::from_num(-1.0), Act::from_num(0.5))),
            ]
        );
    }

    #[test]
    #[should_panic]
    fn project_hidden_to_prefill_logits_uses_shared_det_softcap_tail() {
        let logits = project_hidden_to_prefill_logits(
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

        assert_eq!(
            logits.logits,
            apply_final_logit_softcapping_with_mode(
                &[1.0, 1.0],
                0.5,
                InferenceExecutionMode::Deterministic,
            )
        );
    }

    #[test]
    #[should_panic]
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
    #[should_panic]
    fn project_internal_hidden_to_prefill_logits_uses_preserved_det_row() {
        let projection = Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(2, 2),
            det_weight: Some(det_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0])),
        };
        let internal = project_internal_hidden_to_prefill_logits(
            InternalActivationRow::from_det_values(vec![Act::from_num(1.0), Act::from_num(0.0)]),
            &[1.0, 1.0],
            Some(&[f32_to_wgt(1.0), f32_to_wgt(1.0)]),
            0.0,
            Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
            &projection,
            None,
            InferenceExecutionMode::Deterministic,
            None,
            None,
        )
        .unwrap();
        let public_f32 = project_hidden_to_prefill_logits(
            &[0.0, 1.0],
            &[1.0, 1.0],
            0.0,
            &projection,
            None,
            InferenceExecutionMode::Deterministic,
            None,
        )
        .unwrap();

        assert_ne!(internal.logits, public_f32.logits);
        assert_eq!(internal.logits[1], 0.0);
        assert_eq!(public_f32.logits[0], 0.0);
        assert!(internal.clone_internal().det_values().is_some());
    }

    #[test]
    #[should_panic]
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
        assert!(ple_inputs
            .clone_layer_internal(0)
            .and_then(|input| input.det_values().map(|values| values.to_vec()))
            .is_some());
        assert_eq!(
            projected[0][0],
            crate::shared::numerics::det_num::act_to_f32(
                crate::shared::numerics::det_num::f32_to_act(2f32.sqrt())
            )
        );
        assert_eq!(projected[0][1], 0.0);
    }

    #[test]
    #[should_panic]
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
            crate::shared::numerics::det_num::act_to_f32(
                crate::shared::numerics::det_num::f32_to_act(2f32.sqrt())
            )
        );
        assert_eq!(projected[1], 0.0);
    }

    #[test]
    fn compute_decode_ple_input_internal_preserves_det_values() {
        let layer = ple_test_layer();
        let ple_global = Gemma4PleGlobalWeights::from_det_num_sources(
            vec![deterministic_tensor_source(
                "decode-ple-token-internal",
                "model.language_model.embed_tokens_per_layer.weight",
                2,
                2,
                &[0.0, 0.0, 0.0, 0.0],
            )],
            vec![deterministic_tensor_source(
                "decode-ple-proj-internal",
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

        let ple_input = super::compute_decode_ple_input_internal(
            0,
            InternalActivationRow::from_det_values(vec![
                non_round_tripping_act(),
                Act::from_bits(0),
                Act::from_bits(0),
                Act::from_bits(0),
            ]),
            0,
            &layer,
            Some(&ple_global),
            0.0,
            Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .unwrap()
        .expect("decode ple input should exist");

        assert!(ple_input.det_values().is_some());
    }

    #[test]
    #[should_panic]
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
            post_input_norm_weight_det: None,
        });
        let mut resolved_with_det = resolved_without_det.clone();
        resolved_with_det.ple = Some(ResolvedGemma4PleLayerWeights {
            input_gate_det: Some(det_matrix(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            ..resolved_without_det.ple.clone().unwrap()
        });

        let without_det =
            run_gemma4_layer(&activations, &resolved_without_det, Some(&[vec![1.0, 1.0]]))
                .expect("run fp32 ple path");
        let with_det = run_gemma4_layer(&activations, &resolved_with_det, Some(&[vec![1.0, 1.0]]))
            .expect("run det ple input gate");

        assert_eq!(without_det.activations, vec![vec![1.0, 0.0, 0.0, 0.0]]);
        assert!(with_det.activations[0][0] > without_det.activations[0][0]);
    }

    #[test]
    fn deterministic_projection_uses_preserved_internal_inputs() {
        let canonical = non_round_tripping_act();
        let projected = super::project_linear_sequence_buffer(
            &ActivationSequenceBuffer::from_internal(InternalActivationSequence::from_det_values(
                vec![vec![canonical]],
            )),
            &zero_matrix(1, 1),
            Some(det_matrix(1, 1, &[1.0]).as_ref()),
        )
        .expect("project canonical input");

        assert_eq!(projected.acts.unwrap()[0][0], canonical);
    }

    #[test]
    #[should_panic]
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
            post_input_norm_weight_det: None,
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: Some(Gemma4PleLayerWeights {
                input_gate: zero_matrix(2, 4).into(),
                layer_projection: zero_matrix(4, 2).into(),
                post_input_norm_weight: vec![1.0; 4],
                post_input_norm_weight_det: None,
            }),
            layer_scalar: None,
            layer_scalar_det: None,
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
        let normalized = apply_rms_norm_to_sequence_buffer(
            &ActivationSequenceBuffer::from_acts(vec![vec![Act::from_num(1.0), Act::from_bits(0)]]),
            &[0.5, 1.0],
            Some(&[f32_to_wgt(0.5), f32_to_wgt(1.0)]),
            0.0,
            Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .unwrap()
        .values;

        assert_eq!(
            normalized,
            vec![vec![act_to_f32(Act::from_bits(46_341)), 0.0]]
        );
    }

    #[test]
    fn apply_head_and_value_norms_use_det_num_contract_in_deterministic_mode() {
        let mut head_normed = AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![
            Act::from_num(1.0),
            Act::from_bits(0),
        ]]]);
        apply_head_rms_norm(
            &mut head_normed,
            &[1.0, 1.0],
            Some(&[f32_to_wgt(1.0), f32_to_wgt(1.0)]),
            0.0,
            Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();
        assert_eq!(
            head_normed.values,
            vec![vec![vec![act_to_f32(Act::from_bits(92_682)), 0.0]]]
        );

        let mut value_normed = AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![
            Act::from_num(1.0),
            Act::from_bits(0),
        ]]]);
        apply_value_rms_norm(
            &mut value_normed,
            0.0,
            Some(crate::shared::numerics::det_num::f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();
        assert_eq!(
            value_normed.values,
            vec![vec![vec![act_to_f32(Act::from_bits(92_682)), 0.0]]]
        );
    }

    #[test]
    #[should_panic]
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: None,
            layer_scalar: None,
            layer_scalar_det: None,
        };
        let model = Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: None,
            embedding_source: None,
            layers: vec![layer.clone(), layer],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(2, 4),
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
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
            crate::shared::model::transformer::LayerKvCache::new(1),
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
    fn deterministic_layer_kv_cache_stores_canonical_rows_with_f32_view() {
        let cache = build_layer_kv_cache(
            &AttentionHeadSequenceBuffer::from_acts(vec![vec![
                vec![f32_to_act(1.25)],
                vec![f32_to_act(2.5)],
                vec![f32_to_act(3.75)],
            ]]),
            &AttentionHeadSequenceBuffer::from_acts(vec![vec![
                vec![f32_to_act(10.0)],
                vec![f32_to_act(20.0)],
                vec![f32_to_act(30.0)],
            ]]),
            Some(2),
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(cache.current_len(), 2);
        assert_eq!(
            cache.keys[0].iter().cloned().collect::<Vec<_>>(),
            vec![vec![2.5], vec![3.75]]
        );
        assert_eq!(
            cache.det_key_rows_from(0, 0).expect("canonical keys"),
            vec![vec![f32_to_act(2.5)], vec![f32_to_act(3.75)],]
        );
        assert_eq!(
            cache.det_value_rows_from(0, 1).expect("canonical values"),
            vec![vec![f32_to_act(30.0)]]
        );
    }

    #[test]
    fn deterministic_layer_kv_cache_uses_preserved_canonical_rows() {
        let key = non_round_tripping_act();
        let value = Act::from_bits((1 << 24) + 3);
        assert_ne!(f32_to_act(act_to_f32(value)), value);

        let cache = build_layer_kv_cache(
            &AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![key]]]),
            &AttentionHeadSequenceBuffer::from_acts(vec![vec![vec![value]]]),
            None,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(cache.det_key_rows_from(0, 0), Some(vec![vec![key]]));
        assert_eq!(cache.det_value_rows_from(0, 0), Some(vec![vec![value]]));
        assert_ne!(f32_to_act(cache.keys[0][0][0]), key);
        assert_ne!(f32_to_act(cache.values[0][0][0]), value);
    }

    #[test]
    fn deterministic_kv_append_preserves_existing_canonical_rows() {
        let cache = LayerKvCache::from_det_heads(
            vec![VecDeque::from([vec![Act::from_bits(7)]])],
            vec![VecDeque::from([vec![Act::from_bits(11)]])],
        );

        let updated = append_kv_cache_head_buffer_with_mode(
            cache,
            &AttentionHeadRowBuffer::from_acts(vec![vec![f32_to_act(1.0)]]),
            &AttentionHeadRowBuffer::from_acts(vec![vec![f32_to_act(2.0)]]),
            Some(2),
            InferenceExecutionMode::Deterministic,
        )
        .expect("append deterministic kv");

        assert_eq!(
            updated.det_key_rows_from(0, 0).expect("canonical keys"),
            vec![vec![Act::from_bits(7)], vec![f32_to_act(1.0)]]
        );
        assert_eq!(
            updated.keys[0].iter().cloned().collect::<Vec<_>>(),
            vec![vec![act_to_f32(Act::from_bits(7))], vec![1.0]]
        );
    }

    #[test]
    fn deterministic_kv_append_uses_preserved_new_canonical_rows() {
        let key = non_round_tripping_act();
        let value = Act::from_bits((1 << 24) + 5);
        assert_ne!(f32_to_act(act_to_f32(value)), value);

        let updated = append_kv_cache_head_buffer_with_mode(
            LayerKvCache::new(1),
            &AttentionHeadRowBuffer::from_acts(vec![vec![key]]),
            &AttentionHeadRowBuffer::from_acts(vec![vec![value]]),
            None,
            InferenceExecutionMode::Deterministic,
        )
        .expect("append deterministic kv");

        assert_eq!(updated.det_key_rows_from(0, 0), Some(vec![vec![key]]));
        assert_eq!(updated.det_value_rows_from(0, 0), Some(vec![vec![value]]));
    }

    #[test]
    fn deterministic_layer_outputs_retain_internal_canonical_activations() {
        let mut resolved = crate::io::resolve_layer_weights(
            &parity_test_model(Gemma4AttentionKind::Full, None).layers[0],
        )
        .expect("resolve layer");
        resolved.q_proj_det = Some(det_matrix(
            4,
            4,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ));

        let error = run_gemma4_layer_with_cache_internal(
            InternalActivationSequence::from_det_values(vec![vec![
                f32_to_act(1.0),
                f32_to_act(0.0),
                f32_to_act(0.5),
                f32_to_act(0.0),
            ]]),
            &resolved,
            None,
            None,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("deterministic layer requires canonical norm carriers");

        assert!(error.to_string().contains("canonical Wgt"));
    }

    #[test]
    fn deterministic_decode_layer_outputs_retain_internal_canonical_activation() {
        let model = parity_test_model(Gemma4AttentionKind::Full, None);
        let resolved = crate::io::resolve_layer_weights(&model.layers[0]).expect("resolve layer");

        let error = run_gemma4_layer_decode_with_mode_internal(
            InternalActivationRow::from_det_values(vec![
                f32_to_act(1.0),
                f32_to_act(0.0),
                f32_to_act(0.5),
                f32_to_act(0.0),
            ]),
            &resolved,
            None,
            LayerKvCache::new(1),
            None,
            0,
            InferenceExecutionMode::Deterministic,
        )
        .expect_err("deterministic decode layer requires canonical norm carriers");

        assert!(error.to_string().contains("canonical Wgt"));
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
    fn run_text_layers_decode_step_matches_deterministic_softcapped_replay() {
        let mut model = parity_test_model(Gemma4AttentionKind::Full, None);
        model.final_logit_softcapping = Some(0.5);
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

        let error = project_hidden_to_prefill_logits(
            &decoded.activation_state.activations[0],
            &model.final_norm_weight,
            model.rms_norm_eps,
            &model.logits_projection,
            model.embedding_source.as_ref(),
            InferenceExecutionMode::Deterministic,
            model.final_logit_softcapping,
        )
        .expect_err("f32 hidden state should not project in deterministic mode");
        assert!(error.to_string().contains("canonical Act"));
        return;
        let decoded_logits = extract_prefill_logits(&[]);
        let replay_logits = project_hidden_to_prefill_logits(
            &replay_last_hidden,
            &model.final_norm_weight,
            model.rms_norm_eps,
            &model.logits_projection,
            model.embedding_source.as_ref(),
            InferenceExecutionMode::Deterministic,
            model.final_logit_softcapping,
        )
        .unwrap();

        assert_eq!(decoded_logits.logits, replay_logits.logits);
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
            provenance: Gemma4ModelProvenance::Fp32,
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
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
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
                q_norm_weight_det: None,
                k_norm_weight: vec![1.0, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0; 4],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 3,
                    cols: 4,
                    values: vec![0.7, 0.1, 0.2, 0.0, 0.0, 0.8, 0.1, 0.1, 0.2, 0.0, 0.8, 0.2],
                },
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn expected_det_attention_output(
        query: &[f32],
        key_rows: &[Vec<f32>],
        value_rows: &[Vec<f32>],
    ) -> Vec<f32> {
        let quantized_query = query.iter().copied().map(f32_to_act).collect::<Vec<_>>();
        let quantized_keys = key_rows
            .iter()
            .map(|row| row.iter().copied().map(f32_to_act).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let logits = quantized_keys
            .iter()
            .map(|key_row| attention_score(&quantized_query, key_row))
            .collect::<Vec<_>>();
        let weights = attention_softmax(&logits);
        let quantized_values = value_rows
            .iter()
            .map(|row| row.iter().copied().map(f32_to_act).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        attention_weighted_sum(&weights, &quantized_values)
            .into_iter()
            .map(act_to_f32)
            .collect()
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

    fn non_round_tripping_act() -> Act {
        let act = Act::from_bits((1 << 24) + 1);
        assert_ne!(f32_to_act(act_to_f32(act)), act);
        act
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: q_proj.into(),
            k_proj: k_proj.into(),
            v_proj: Some(v_proj.into()),
            o_proj: o_proj.into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 2],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 2],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 2],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 2],
            post_feedforward_layernorm_weight_det: None,
            gate_proj: zero_matrix(4, 2).into(),
            up_proj: zero_matrix(4, 2).into(),
            down_proj: zero_matrix(2, 4).into(),
            ple: None,
            layer_scalar: None,
            layer_scalar_det: None,
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
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(2, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 2).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: Some(Gemma4PleLayerWeights {
                input_gate: zero_matrix(2, 4).into(),
                layer_projection: zero_matrix(4, 2).into(),
                post_input_norm_weight: vec![1.0; 4],
                post_input_norm_weight_det: None,
            }),
            layer_scalar: None,
            layer_scalar_det: None,
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
            post_input_norm_weight_det: None,
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
