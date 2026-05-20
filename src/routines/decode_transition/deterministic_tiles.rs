use anyhow::{anyhow, bail, Result};
use serde_json::json;

use super::native_tiles::ActivationSequenceWithCache;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4LayerWeights, Gemma4TransformerModel, InternalActivationRow,
    InternalActivationSequence, LayerKvCache,
};
use crate::trace::trace_scope;

pub fn run_text_layers_decode_step(
    input_activation: &[f32],
    token_id: u32,
    model: &Gemma4TransformerModel,
    layer_caches: Vec<LayerKvCache>,
    position: usize,
) -> Result<ActivationSequenceWithCache> {
    run_text_layers_decode_step_internal(
        InternalActivationRow::from_values(input_activation.to_vec()),
        token_id,
        model,
        layer_caches,
        position,
    )
}

pub(crate) fn run_text_layers_decode_step_internal(
    input_activation: InternalActivationRow,
    token_id: u32,
    model: &Gemma4TransformerModel,
    layer_caches: Vec<LayerKvCache>,
    position: usize,
) -> Result<ActivationSequenceWithCache> {
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

    let decode_input = input_activation;
    let decode_input_values = decode_input.clone_f32();
    let mut xs = decode_input.clone();
    let mut updated_layer_caches = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_sha256s = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_det_sha256s = Vec::with_capacity(model.layers.len());
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let cache = layer_caches[layer_idx].clone();
        let _trace = trace_scope(format!(
            "decode.layer.det layer={layer_idx} token={} position={} attention={:?} ple={} donor={:?}",
            token_id,
            position,
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        let per_layer_input =
            crate::shared::numerics::transformer_kernels::compute_decode_ple_input_internal(
                token_id,
                decode_input.clone(),
                layer_idx,
                layer,
                model.ple_global.as_ref(),
                model.rms_norm_eps,
                model.rms_norm_eps_det,
                InferenceExecutionMode::Deterministic,
            )?;
        let donor_cache = resolve_decode_donor_cache(layer, &updated_layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, updated_cache) =
            crate::shared::numerics::transformer_kernels::run_gemma4_layer_decode_with_mode_internal(
                xs,
                &resolved_layer,
                per_layer_input,
                cache,
                donor_cache,
                position,
                InferenceExecutionMode::Deterministic,
            )?;
        xs = layer_output;
        let xs_values = xs.clone_f32();
        let det_current_activation_sha256 = xs
            .det_values()
            .map(crate::shared::numerics::transformer_kernels::build_det_vector_commitment);
        updated_layer_caches.push(updated_cache);
        completed_layer_output_sha256s.push(
            crate::shared::numerics::transformer_kernels::build_vector_commitment(&xs_values),
        );
        completed_layer_output_det_sha256s.push(det_current_activation_sha256.clone());
        let mut checkpoint_layer_caches = updated_layer_caches.clone();
        checkpoint_layer_caches.extend(layer_caches.iter().skip(layer_idx + 1).cloned());
        crate::trace::trace_checkpoint(
            &format!("decode.layer_token.layer_{layer_idx}.position_{position}"),
            &json!({
                "execution_mode": "deterministic",
                "token_id": token_id,
                "position": position,
                "next_layer_idx": layer_idx + 1,
                "decode_input_activation": decode_input_values.clone(),
                "decode_input_activation_sha256": crate::shared::numerics::transformer_kernels::build_vector_commitment(&decode_input_values),
                "det_decode_input_activation_sha256": decode_input.det_values().map(crate::shared::numerics::transformer_kernels::build_det_vector_commitment),
                "current_activation": xs_values.clone(),
                "current_activation_sha256": crate::shared::numerics::transformer_kernels::build_vector_commitment(&xs_values),
                "det_current_activation_sha256": det_current_activation_sha256,
                "layer_caches": crate::trace::serialize_layer_caches(&checkpoint_layer_caches),
                "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(&checkpoint_layer_caches),
                "completed_layer_output_sha256s": completed_layer_output_sha256s.clone(),
                "completed_layer_output_det_sha256s": completed_layer_output_det_sha256s.clone(),
            }),
        );
    }

    let xs_values = xs.clone_f32();
    let activation_internal = match xs.det_values() {
        Some(det_values) => InternalActivationSequence::from_det_values(vec![det_values.to_vec()]),
        None => InternalActivationSequence::from_values(vec![xs_values.clone()]),
    };
    let det_activations_sha256 = activation_internal
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_activation_commitment);
    let mut activation_state = ActivationSequence::from_internal(
        activation_internal,
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&[xs_values]),
    );
    activation_state.det_activations_sha256 = det_activations_sha256;
    Ok(ActivationSequenceWithCache {
        activation_state,
        layer_caches: updated_layer_caches,
    })
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
