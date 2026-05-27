use anyhow::{anyhow, bail, Result};

use super::ActivationSequenceWithCache;
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
    let mut xs = decode_input.clone();
    let mut updated_layer_caches = Vec::with_capacity(model.layers.len());
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
        updated_layer_caches.push(updated_cache);
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
