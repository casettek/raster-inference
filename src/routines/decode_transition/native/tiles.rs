use anyhow::{anyhow, bail, Result};

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4LayerWeights, Gemma4TransformerModel, LayerKvCache,
};
use crate::trace::trace_scope;

pub struct ActivationSequenceWithCache {
    pub activation_state: ActivationSequence,
    pub layer_caches: Vec<LayerKvCache>,
}

pub fn run_text_layers_decode_step(
    input_activation: &[f32],
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
        let per_layer_input =
            crate::shared::numerics::transformer_kernels::compute_decode_ple_input(
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
        let (layer_output, updated_cache) =
            crate::shared::numerics::transformer_kernels::run_gemma4_layer_decode_with_mode(
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
            crate::shared::numerics::transformer_kernels::build_activation_commitment(&[xs]),
        ),
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
