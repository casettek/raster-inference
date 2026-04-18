use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::types::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4Phase2Model, Gemma4PleGlobalWeights,
    Gemma4PrefillPleInputs, LayerKvCache, MatrixF32, PrefillLogits,
    ResolvedGemma4LayerWeights,
};
use crate::trace::trace_scope;

pub fn embed_input_tokens(
    token_ids: &[u32],
    embedding_table: &EmbeddingTable,
) -> Result<EmbeddedTokenSequence> {
    // let _trace = trace_scope("phase2.embed_input_tokens");
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

pub fn embed_input_token(token_id: u32, embedding_table: &EmbeddingTable) -> Result<Vec<f32>> {
    let embedded = embed_input_tokens(&[token_id], embedding_table)?;
    embedded
        .activations
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("phase 2 embedding returned no activation rows"))
}

pub fn compute_prefill_ple_inputs(
    token_ids: &[u32],
    input_activations: &[Vec<f32>],
    layers: &[Gemma4LayerWeights],
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps: f32,
) -> Result<Gemma4PrefillPleInputs> {
    // let _trace = trace_scope("phase2.compute_prefill_ple_inputs");
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
    if ple_global.token_embedding_layer_count() != layers.len() {
        bail!(
            "phase 2 PLE token embedding slice count mismatch: {} vs {}",
            ple_global.token_embedding_layer_count(),
            layers.len()
        );
    }
    if ple_global.model_projection_layer_count() != layers.len() {
        bail!(
            "phase 2 PLE model projection slice count mismatch: {} vs {}",
            ple_global.model_projection_layer_count(),
            layers.len()
        );
    }

    let mut per_layer_inputs = Vec::with_capacity(layers.len());
    for (layer_idx, layer) in layers.iter().enumerate() {
        // trace_event(format!("phase2.compute_prefill_ple_inputs layer={layer_idx}"));
        if layer.ple.is_none() {
            per_layer_inputs.push(None);
            continue;
        }

        let mut embedded = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            embedded.push(
                crate::io::load_ple_token_embedding_row(ple_global, layer_idx, *token_id)?
                    .into_iter()
                    .map(|value| value * ple_global.embedding_scale)
                    .collect::<Vec<_>>(),
            );
        }

        let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
        let mut projected = linear_sequence(input_activations, &model_projection)?;
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

pub fn compute_decode_ple_input(
    token_id: u32,
    input_activation: &[f32],
    layer_idx: usize,
    layer: &Gemma4LayerWeights,
    ple_global: Option<&Gemma4PleGlobalWeights>,
    rms_norm_eps: f32,
) -> Result<Option<Vec<f32>>> {
    let Some(ple_global) = ple_global else {
        return Ok(None);
    };
    if layer.ple.is_none() {
        return Ok(None);
    }

    validate_vector_width(input_activation, layer.hidden_size, "decode PLE input activation")?;
    let embedded = crate::io::load_ple_token_embedding_row(ple_global, layer_idx, token_id)?
        .into_iter()
        .map(|value| value * ple_global.embedding_scale)
        .collect::<Vec<_>>();
    let model_projection = crate::io::load_ple_model_projection(ple_global, layer_idx)?;
    let mut projected = linear_row(input_activation, &model_projection)?;
    for value in &mut projected {
        *value *= ple_global.projection_scalar;
    }
    let projected = apply_rms_norm(&projected, &ple_global.projection_norm_weight, rms_norm_eps)?;

    if embedded.len() != projected.len() {
        bail!(
            "decode PLE width mismatch: embedded {} vs projected {}",
            embedded.len(),
            projected.len()
        );
    }

    Ok(Some(
        embedded
            .into_iter()
            .zip(projected)
            .map(|(embedded_value, projected_value)| {
                (embedded_value + projected_value) * ple_global.input_scale
            })
            .collect(),
    ))
}

pub fn run_gemma4_layer(
    input_activations: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[Vec<f32>]>,
) -> Result<ActivationSequence> {
    Ok(run_gemma4_layer_with_cache(input_activations, layer, per_layer_input, None)?.0)
}

fn run_gemma4_layer_with_cache(
    input_activations: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    per_layer_input: Option<&[Vec<f32>]>,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(ActivationSequence, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "phase2.run_gemma4_layer attention={:?}",
    //     layer.attention_kind
    // ));
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
                .as_ref()
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
        // let _trace = trace_scope("phase2.run_gemma4_layer.clone_input");
        input_activations.to_vec()
    };

    let residual = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.attention.clone_residual");
        xs.clone()
    };
    let normed = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.attention.input_rms_norm");
        apply_rms_norm_to_sequence(&xs, &layer.input_layernorm_weight, layer.rms_norm_eps)?
    };
    let (attn_out, layer_cache) = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.attention.core");
        run_attention_for_layer_with_cache(&normed, layer, donor_cache)?
    };
    let attn_out = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.attention.post_rms_norm");
        apply_rms_norm_to_sequence(
            &attn_out,
            &layer.post_attention_layernorm_weight,
            layer.rms_norm_eps,
        )?
    };
    xs = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.attention.residual_add");
        add_sequences(&residual, &attn_out)?
    };

    let residual = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.clone_residual");
        xs.clone()
    };
    let normed = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.pre_rms_norm");
        apply_rms_norm_to_sequence(&xs, &layer.pre_feedforward_layernorm_weight, layer.rms_norm_eps)?
    };
    let gate = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.gate_proj_gelu");
        apply_gelu_to_sequence(&linear_sequence(&normed, layer.gate_proj.as_ref())?)
    };
    let up = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.up_proj");
        linear_sequence(&normed, layer.up_proj.as_ref())?
    };
    let ff_hidden = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.hidden_mul");
        elementwise_mul_sequences(&gate, &up)?
    };
    let ff_out = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.down_proj");
        linear_sequence(&ff_hidden, layer.down_proj.as_ref())?
    };
    let ff_out = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.post_rms_norm");
        apply_rms_norm_to_sequence(
            &ff_out,
            &layer.post_feedforward_layernorm_weight,
            layer.rms_norm_eps,
        )?
    };
    xs = {
        // let _trace = trace_scope("phase2.run_gemma4_layer.mlp.residual_add");
        add_sequences(&residual, &ff_out)?
    };

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = {
            // let _trace = trace_scope("phase2.run_gemma4_layer.ple.clone_residual");
            xs.clone()
        };
        let gated = {
            // let _trace = trace_scope("phase2.run_gemma4_layer.ple.input_gate_gelu");
            apply_gelu_to_sequence(&linear_sequence(&xs, ple.input_gate.as_ref())?)
        };
        let gated = {
            // let _trace = trace_scope("phase2.run_gemma4_layer.ple.input_mul");
            elementwise_mul_sequences(&gated, per_layer_input)?
        };
        let projected = {
            // let _trace = trace_scope("phase2.run_gemma4_layer.ple.layer_projection");
            linear_sequence(&gated, ple.layer_projection.as_ref())?
        };
        let projected = {
            // let _trace = trace_scope("phase2.run_gemma4_layer.ple.post_rms_norm");
            apply_rms_norm_to_sequence(
                &projected,
                &ple.post_input_norm_weight,
                layer.rms_norm_eps,
            )?
        };
        xs = {
            // let _trace = trace_scope("phase2.run_gemma4_layer.ple.residual_add");
            add_sequences(&residual, &projected)?
        };
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        // let _trace = trace_scope("phase2.run_gemma4_layer.layer_scalar");
        for row in &mut xs {
            for value in row {
                *value *= layer_scalar;
            }
        }
    }

    Ok((
        ActivationSequence {
            activations_sha256: {
                // let _trace = trace_scope("phase2.run_gemma4_layer.commitment_hash");
                build_phase2_commitment(&xs)
            },
            activations: xs,
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
    // let _trace = trace_scope(format!(
    //     "phase2.run_gemma4_layer_decode attention={:?}",
    //     layer.attention_kind
    // ));
    validate_vector_width(input_activation, layer.hidden_size, "decode input activation")?;
    if let Some(per_layer_input) = per_layer_input {
        validate_vector_width(
            per_layer_input,
            layer
                .ple
                .as_ref()
                .ok_or_else(|| anyhow!("phase 2 decode received PLE inputs without PLE weights"))?
                .input_gate
                .as_ref()
                .rows,
            "decode per-layer input",
        )?;
    }

    let residual = input_activation.to_vec();
    let normed = apply_rms_norm(input_activation, &layer.input_layernorm_weight, layer.rms_norm_eps)?;
    let (mut xs, updated_cache) =
        run_attention_for_layer_decode(&normed, layer, cache, donor_cache, position)?;
    xs = apply_rms_norm(&xs, &layer.post_attention_layernorm_weight, layer.rms_norm_eps)?;
    xs = add_rows(&residual, &xs)?;

    let residual = xs.clone();
    let normed = apply_rms_norm(&xs, &layer.pre_feedforward_layernorm_weight, layer.rms_norm_eps)?;
    let gate = apply_gelu(&linear_row(&normed, layer.gate_proj.as_ref())?);
    let up = linear_row(&normed, layer.up_proj.as_ref())?;
    let ff_hidden = elementwise_mul_rows(&gate, &up)?;
    let mut ff_out = linear_row(&ff_hidden, layer.down_proj.as_ref())?;
    ff_out = apply_rms_norm(
        &ff_out,
        &layer.post_feedforward_layernorm_weight,
        layer.rms_norm_eps,
    )?;
    xs = add_rows(&residual, &ff_out)?;

    if let (Some(ple), Some(per_layer_input)) = (&layer.ple, per_layer_input) {
        let residual = xs.clone();
        let gated = apply_gelu(&linear_row(&xs, ple.input_gate.as_ref())?);
        let gated = elementwise_mul_rows(&gated, per_layer_input)?;
        let mut projected = linear_row(&gated, ple.layer_projection.as_ref())?;
        projected = apply_rms_norm(&projected, &ple.post_input_norm_weight, layer.rms_norm_eps)?;
        xs = add_rows(&residual, &projected)?;
    }

    if let Some(layer_scalar) = layer.layer_scalar {
        for value in &mut xs {
            *value *= layer_scalar;
        }
    }

    Ok((xs, updated_cache))
}

pub fn run_text_layers_prefill(
    input_activations: &[Vec<f32>],
    model: &Gemma4Phase2Model,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<ActivationSequence> {
    Ok(run_text_layers_prefill_with_cache(input_activations, model, ple_inputs)?.0)
}

pub fn run_text_layers_prefill_with_cache(
    input_activations: &[Vec<f32>],
    model: &Gemma4Phase2Model,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    // let _trace = trace_scope("phase2.run_text_layers_prefill");
    if model.layers.is_empty() {
        bail!("phase 2 prefill requires at least one layer");
    }

    let mut xs = input_activations.to_vec();
    let mut layer_caches = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_sha256s = Vec::with_capacity(model.layers.len());
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let _trace = trace_scope(format!(
            "phase2.prefill_layer layer={layer_idx} tokens={} attention={:?} ple={} donor={:?}",
            xs.len(),
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        // trace_event(format!(
        //     "phase2.run_text_layers_prefill layer={layer_idx} attention={:?}",
        //     layer.attention_kind
        // ));
        // trace_event(format!(
        //     "phase2.prefill_layer_tokens layer={layer_idx} tokens={}",
        //     xs.len()
        // ));
        let per_layer_input = ple_inputs
            .and_then(|inputs| inputs.per_layer_inputs.get(layer_idx))
            .and_then(|input| input.as_deref());
        let donor_cache = resolve_prefill_donor_cache(layer, &layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, layer_cache) =
            run_gemma4_layer_with_cache(&xs, &resolved_layer, per_layer_input, donor_cache)?;
        xs = layer_output.activations;
        layer_caches.push(layer_cache);
        completed_layer_output_sha256s.push(layer_output.activations_sha256);
        crate::trace::trace_checkpoint("phase2a_layers", &json!({
            "next_layer_idx": layer_idx + 1,
            "current_activations": xs.clone(),
            "current_activations_sha256": build_phase2_commitment(&xs),
            "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
            "completed_layer_output_sha256s": completed_layer_output_sha256s.clone(),
        }));
        for (token_idx, token_activation) in xs.iter().enumerate() {
            crate::trace::trace_checkpoint(
                &format!("phase2a_layer_token.layer_{layer_idx}.token_{token_idx}"),
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
            activations_sha256: build_phase2_commitment(&xs),
            activations: xs,
        },
        layer_caches,
    ))
}

pub fn run_text_layers_decode_step(
    input_activation: &[f32],
    token_id: u32,
    model: &Gemma4Phase2Model,
    layer_caches: Vec<LayerKvCache>,
    position: usize,
) -> Result<ActivationSequenceWithCache> {
    // let _trace = trace_scope("phase2.run_text_layers_decode_step");
    if model.layers.is_empty() {
        bail!("phase 2 decode requires at least one layer");
    }
    if layer_caches.len() != model.layers.len() {
        bail!(
            "phase 2 decode cache count mismatch: {} vs {}",
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
            "phase2.decode_layer layer={layer_idx} token={} position={} attention={:?} ple={} donor={:?}",
            token_id,
            position,
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        // trace_event(format!(
        //     "phase2.run_text_layers_decode_step layer={layer_idx} attention={:?}",
        //     layer.attention_kind
        // ));
        // trace_event(format!(
        //     "phase2.decode_layer_state layer={layer_idx} token={} position={}",
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
        )?;
        let donor_cache = resolve_decode_donor_cache(layer, &updated_layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, updated_cache) = run_gemma4_layer_decode(
            &xs,
            &resolved_layer,
            per_layer_input.as_deref(),
            cache,
            donor_cache,
            position,
        )?;
        xs = layer_output;
        updated_layer_caches.push(updated_cache);
        completed_layer_output_sha256s.push(build_vector_commitment(&xs));
        let mut checkpoint_layer_caches = updated_layer_caches.clone();
        checkpoint_layer_caches.extend(layer_caches.iter().skip(layer_idx + 1).cloned());
        crate::trace::trace_checkpoint(
            &format!("phase2b_layer_token.layer_{layer_idx}.position_{position}"),
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
            activations_sha256: build_phase2_commitment(&[xs.clone()]),
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
                    "phase 2 prefill layer {layer_idx} cannot share KV with non-prior donor {donor_idx}"
                );
            }
            layer_caches.get(donor_idx).ok_or_else(|| {
                anyhow!("phase 2 prefill donor cache {donor_idx} missing for layer {layer_idx}")
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
                    "phase 2 decode layer {layer_idx} cannot share KV with non-prior donor {donor_idx}"
                );
            }
            updated_layer_caches.get(donor_idx).ok_or_else(|| {
                anyhow!("phase 2 decode donor cache {donor_idx} missing for layer {layer_idx}")
            })
        })
        .transpose()
}

pub fn apply_final_norm(
    input_activations: &[Vec<f32>],
    weight: &[f32],
    eps: f32,
) -> Result<ActivationSequence> {
    // let _trace = trace_scope("phase2.apply_final_norm");
    let activations = apply_rms_norm_to_sequence(input_activations, weight, eps)?;
    Ok(ActivationSequence {
        activations_sha256: build_phase2_commitment(&activations),
        activations,
    })
}

pub fn select_final_position(input_activations: &[Vec<f32>]) -> Result<Vec<f32>> {
    // let _trace = trace_scope("phase2.select_final_position");
    input_activations
        .last()
        .cloned()
        .ok_or_else(|| anyhow!("phase 2 final-position selection requires at least one activation row"))
}

pub fn project_to_logits(
    last_hidden_state: &[f32],
    projection: &Gemma4LogitsProjection,
) -> Result<Vec<f32>> {
    // let _trace = trace_scope("phase2.project_to_logits");
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
    // let _trace = trace_scope("phase2.apply_final_logit_softcapping");
    logits
        .iter()
        .map(|logit| (logit / softcap).tanh() * softcap)
        .collect()
}

pub fn extract_prefill_logits(logits: &[f32]) -> PrefillLogits {
    // let _trace = trace_scope("phase2.extract_prefill_logits");
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
    final_logit_softcapping: Option<f32>,
) -> Result<PrefillLogits> {
    // let _trace = trace_scope("phase2.project_decode_hidden_to_logits");
    let normalized = apply_rms_norm(hidden_state, final_norm_weight, rms_norm_eps)?;
    let mut logits = project_to_logits(&normalized, projection)?;
    if let Some(softcap) = final_logit_softcapping {
        logits = apply_final_logit_softcapping(&logits, softcap);
    }
    Ok(extract_prefill_logits(&logits))
}

pub struct ActivationSequenceWithCache {
    pub activation_state: ActivationSequence,
    pub layer_caches: Vec<LayerKvCache>,
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

fn run_attention_for_layer_with_cache(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    match layer.attention_kind {
        Gemma4AttentionKind::Sliding => run_sliding_attention(inputs, layer, donor_cache),
        Gemma4AttentionKind::Full => run_full_attention(inputs, layer, donor_cache),
    }
}

fn run_attention_for_layer_decode(
    input: &[f32],
    layer: &ResolvedGemma4LayerWeights,
    cache: LayerKvCache,
    donor_cache: Option<&LayerKvCache>,
    position: usize,
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
            )
        }
        Gemma4AttentionKind::Full => {
            run_causal_attention_decode(input, layer, cache, donor_cache, position, None, None)
        }
    }
}

fn run_sliding_attention(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
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
    )
}

fn run_full_attention(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    run_causal_attention(inputs, layer, None, None, donor_cache)
}

fn run_causal_attention(
    inputs: &[Vec<f32>],
    layer: &ResolvedGemma4LayerWeights,
    attention_window: Option<usize>,
    cache_window: Option<usize>,
    donor_cache: Option<&LayerKvCache>,
) -> Result<(Vec<Vec<f32>>, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "phase2.run_causal_attention attention={:?}",
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

    let q_projected = {
        // let _trace = trace_scope("phase2.run_causal_attention.q_proj");
        linear_sequence(inputs, layer.q_proj.as_ref())?
    };
    let raw_k = {
        // let _trace = trace_scope("phase2.run_causal_attention.k_proj");
        linear_sequence(inputs, layer.k_proj.as_ref())?
    };
    let raw_v = if let Some(v_proj) = &layer.v_proj {
        // let _trace = trace_scope("phase2.run_causal_attention.v_proj");
        linear_sequence(inputs, v_proj.as_ref())?
    } else if layer.attention_k_eq_v {
        // trace_event("phase2.run_causal_attention.k_eq_v_reuse");
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };
    let mut q = {
        // let _trace = trace_scope("phase2.run_causal_attention.reshape_q");
        reshape_sequence_heads(&q_projected, layer.num_heads, layer.head_dim)?
    };
    let mut k = {
        // let _trace = trace_scope("phase2.run_causal_attention.reshape_k");
        reshape_sequence_heads(&raw_k, layer.num_kv_heads, layer.head_dim)?
    };
    let mut v = {
        // let _trace = trace_scope("phase2.run_causal_attention.reshape_v");
        reshape_sequence_heads(&raw_v, layer.num_kv_heads, layer.head_dim)?
    };

    {
        // let _trace = trace_scope("phase2.run_causal_attention.q_rms_norm");
        apply_head_rms_norm(&mut q, &layer.q_norm_weight, layer.rms_norm_eps)?;
    }
    {
        // let _trace = trace_scope("phase2.run_causal_attention.k_rms_norm");
        apply_head_rms_norm(&mut k, &layer.k_norm_weight, layer.rms_norm_eps)?;
    }
    {
        // let _trace = trace_scope("phase2.run_causal_attention.v_rms_norm");
        apply_value_rms_norm(&mut v, layer.rms_norm_eps)?;
    }

    {
        // let _trace = trace_scope("phase2.run_causal_attention.q_rope");
        apply_rope(
            &mut q,
            layer.partial_rotary_dim,
            layer.rope_freq_base_dim,
            layer.rope_base,
        );
    }
    {
        // let _trace = trace_scope("phase2.run_causal_attention.k_rope");
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
            // let _trace = trace_scope(format!("phase2.run_causal_attention.head={head_idx}"));
            let kv_head_idx = head_idx / kv_groups;
            let mut outputs = vec![vec![0.0; layer.head_dim]; seq_len];
            for (query_idx, output) in outputs.iter_mut().enumerate() {
                let start = attention_window
                    .map(|window| query_idx.saturating_add(1).saturating_sub(window))
                    .unwrap_or(0);
                if let Some(donor_cache) = donor_cache {
                    let logits = (start..=query_idx)
                        .map(|key_idx| dot(&q[head_idx][query_idx], &donor_cache.keys[kv_head_idx][key_idx]))
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

    {
        // let _trace = trace_scope("phase2.run_causal_attention.o_proj");
        Ok((linear_sequence(&combined_heads, layer.o_proj.as_ref())?, layer_cache))
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
) -> Result<(Vec<f32>, LayerKvCache)> {
    // let _trace = trace_scope(format!(
    //     "phase2.run_causal_attention_decode attention={:?}",
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

    let q_projected = linear_row(input, layer.q_proj.as_ref())?;
    let raw_k = linear_row(input, layer.k_proj.as_ref())?;
    let raw_v = if let Some(v_proj) = &layer.v_proj {
        linear_row(input, v_proj.as_ref())?
    } else if layer.attention_k_eq_v {
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };

    let mut q = reshape_row_heads(&q_projected, layer.num_heads, layer.head_dim)?;
    let mut k = reshape_row_heads(&raw_k, layer.num_kv_heads, layer.head_dim)?;
    let mut v = reshape_row_heads(&raw_v, layer.num_kv_heads, layer.head_dim)?;

    apply_head_rms_norm_row(&mut q, &layer.q_norm_weight, layer.rms_norm_eps)?;
    apply_head_rms_norm_row(&mut k, &layer.k_norm_weight, layer.rms_norm_eps)?;
    apply_value_rms_norm_row(&mut v, layer.rms_norm_eps)?;
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
            .map(|window| attention_cache.keys[kv_head_idx].len().saturating_sub(window))
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

    Ok((linear_row(&combined_heads, layer.o_proj.as_ref())?, updated_cache))
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

fn reshape_row_heads(projected: &[f32], num_heads: usize, head_dim: usize) -> Result<Vec<Vec<f32>>> {
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
) -> Result<()> {
    heads.par_iter_mut().try_for_each(|head| -> Result<()> {
        for row in head {
            *row = apply_rms_norm(row, weight, eps)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn apply_head_rms_norm_row(heads: &mut [Vec<f32>], weight: &[f32], eps: f32) -> Result<()> {
    for head in heads {
        *head = apply_rms_norm(head, weight, eps)?;
    }
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

fn apply_value_rms_norm_row(heads: &mut [Vec<f32>], eps: f32) -> Result<()> {
    for head in heads {
        let mean_square = head.iter().map(|value| value * value).sum::<f32>() / head.len() as f32;
        let scale = (mean_square + eps).sqrt().recip();
        for value in head {
            *value *= scale;
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

fn build_layer_kv_cache(
    keys: &[Vec<Vec<f32>>],
    values: &[Vec<Vec<f32>>],
    sliding_window: Option<usize>,
) -> LayerKvCache {
    let retained = sliding_window.map_or(0, |window| {
        keys.first().map_or(0, |head| head.len().saturating_sub(window))
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
    inputs.iter().map(|value| gelu_pytorch_tanh(*value)).collect()
}

fn gelu_pytorch_tanh(value: f32) -> f32 {
    let inner = std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value.powi(3));
    0.5 * value * (1.0 + inner.tanh())
}

#[cfg(test)]
mod tests {
    use super::{
        append_kv_cache, apply_final_norm, apply_rope_to_rows, compute_prefill_ple_inputs,
        embed_input_tokens, extract_prefill_logits, project_decode_hidden_to_logits, project_to_logits,
        run_gemma4_layer, run_text_layers_decode_step, run_text_layers_prefill,
        run_text_layers_prefill_with_cache,
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
    fn apply_rope_to_rows_uses_full_head_dim_for_frequency_base() {
        let mut heads = vec![vec![0.0, 1.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0]];

        apply_rope_to_rows(&mut heads, 4, 8, 16.0, 1);

        assert!((heads[0][1] - 0.87758255).abs() < 1e-6);
        assert!((heads[0][3] - 0.47942555).abs() < 1e-6);
        assert_eq!(heads[0][4..], [9.0, 8.0, 7.0, 6.0]);
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

    #[test]
    fn run_text_layers_prefill_with_cache_retains_sliding_window_entries() {
        let model = parity_test_model(Gemma4AttentionKind::Sliding, Some(2));
        let activations = model.embedding_table.as_ref().unwrap().rows.clone();

        let (_, layer_caches) = run_text_layers_prefill_with_cache(&activations, &model, None).unwrap();

        assert_eq!(layer_caches.len(), 1);
        assert_eq!(layer_caches[0].current_len(), 2);
    }

    #[test]
    fn append_kv_cache_keeps_newest_sliding_window_entries_in_order() {
        let cache = append_kv_cache(crate::phase2::LayerKvCache::new(1), &[vec![1.0]], &[vec![10.0]], None)
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

        let (_, layer_caches) = run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
        let decoded =
            run_text_layers_decode_step(&next_embedding, 2, &model, layer_caches, prompt_embeddings.len())
                .unwrap();
        let replay = run_text_layers_prefill(&embeddings, &model, None).unwrap();
        let replay_last_hidden = replay.activations.last().cloned().unwrap();

        assert_eq!(decoded.activation_state.activations[0], replay_last_hidden);

        let decoded_logits = project_decode_hidden_to_logits(
            &decoded.activation_state.activations[0],
            &model.final_norm_weight,
            model.rms_norm_eps,
            &model.logits_projection,
            model.final_logit_softcapping,
        )
        .unwrap();
        let replay_logits = project_to_logits(
            &apply_final_norm(&replay.activations, &model.final_norm_weight, model.rms_norm_eps)
                .unwrap()
                .activations
                .last()
                .cloned()
                .unwrap(),
            &model.logits_projection,
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

        let (_, layer_caches) = run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
        let decoded =
            run_text_layers_decode_step(&next_embedding, 2, &model, layer_caches, prompt_embeddings.len())
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

        let (_, layer_caches) = run_text_layers_prefill_with_cache(&prompt_embeddings, &model, None).unwrap();
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
    ) -> Gemma4Phase2Model {
        Gemma4Phase2Model {
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
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                }
                .into(),
                k_proj: MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                    ],
                }
                .into(),
                v_proj: Some(MatrixF32 {
                    rows: 2,
                    cols: 4,
                    values: vec![
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ],
                }
                .into()),
                o_proj: MatrixF32 {
                    rows: 4,
                    cols: 4,
                    values: vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
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
            logits_projection: Gemma4LogitsProjection::UntiedLmHead(MatrixF32 {
                rows: 3,
                cols: 4,
                values: vec![
                    0.7, 0.1, 0.2, 0.0,
                    0.0, 0.8, 0.1, 0.1,
                    0.2, 0.0, 0.8, 0.2,
                ],
            }),
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
}
