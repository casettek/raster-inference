use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::shared::transformer::{
    ActivationSequence, Gemma4LayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
    InternalActivationSequence, LayerKvCache,
};
use crate::trace::trace_scope;

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
    run_text_layers_prefill_with_cache_internal(
        InternalActivationSequence::from_values(input_activations.to_vec()),
        model,
        ple_inputs,
    )
}

pub(crate) fn run_text_layers_prefill_with_cache_internal(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    if model.layers.is_empty() {
        bail!("transformer prefill requires at least one layer");
    }

    let mut xs = input_activations;
    let mut layer_caches = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_sha256s = Vec::with_capacity(model.layers.len());
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let xs_values = xs.clone_f32();
        let _trace = trace_scope(format!(
            "prefill.layer.det layer={layer_idx} tokens={} attention={:?} ple={} donor={:?}",
            xs_values.len(),
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        let per_layer_input = ple_inputs.and_then(|inputs| inputs.clone_layer_internal(layer_idx));
        let donor_cache = resolve_prefill_donor_cache(layer, &layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, layer_cache) =
            crate::shared::transformer_kernels::run_gemma4_layer_with_cache_internal(
                xs,
                &resolved_layer,
                per_layer_input,
                donor_cache,
                crate::shared::input::InferenceExecutionMode::Deterministic,
            )?;
        xs = layer_output.clone_internal();
        let xs_values = xs.clone_f32();
        layer_caches.push(layer_cache);
        completed_layer_output_sha256s.push(layer_output.activations_sha256);
        crate::trace::trace_checkpoint(
            "prefill.layer",
            &json!({
                "execution_mode": "deterministic",
                "next_layer_idx": layer_idx + 1,
                "current_activations": xs_values.clone(),
                "current_activations_sha256": crate::shared::transformer_kernels::build_activation_commitment(&xs_values),
                "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
                "completed_layer_output_sha256s": completed_layer_output_sha256s.clone(),
            }),
        );
        for (token_idx, token_activation) in xs_values.iter().enumerate() {
            crate::trace::trace_checkpoint(
                &format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}"),
                &json!({
                    "execution_mode": "deterministic",
                    "layer_idx": layer_idx,
                    "token_idx": token_idx,
                    "token_count": xs_values.len(),
                    "token_activation": token_activation,
                }),
            );
        }
    }

    let xs_values = xs.clone_f32();
    Ok((
        ActivationSequence::from_internal(
            xs,
            crate::shared::transformer_kernels::build_activation_commitment(&xs_values),
        ),
        layer_caches,
    ))
}

pub fn run(
    input_activations: &[Vec<f32>],
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_text_layers_prefill_with_cache(input_activations, model, ple_inputs)
}

pub(crate) fn run_internal(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_text_layers_prefill_with_cache_internal(input_activations, model, ple_inputs)
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
