use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::runtime::checkpoints::RasterDetourController;
use crate::runtime::checkpoints::RoutineId;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4LayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
    InternalActivationSequence, LayerKvCache,
};
use crate::trace::routine_scope;

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
    run_text_layers_prefill_with_cache_internal_and_detour(
        input_activations,
        model,
        ple_inputs,
        None,
    )
}

pub(crate) fn run_text_layers_prefill_with_cache_internal_and_detour(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    mut detour_controller: Option<&mut RasterDetourController>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    if model.layers.is_empty() {
        bail!("transformer prefill requires at least one layer");
    }

    let mut xs = input_activations;
    let mut layer_caches = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_sha256s = Vec::with_capacity(model.layers.len());
    let mut completed_layer_output_det_sha256s = Vec::with_capacity(model.layers.len());
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        if let Some(controller) = detour_controller.as_deref_mut() {
            controller.reject_if_selected_unsupported(RoutineId::PrefillLayer)?;
        }
        let xs_values = xs.clone_f32();
        let _routine = routine_scope(
            RoutineId::PrefillLayer,
            format!(
                "mode=det layer={layer_idx} tokens={} attention={:?} ple={} donor={:?}",
                xs_values.len(),
                layer.attention_kind,
                layer.ple.is_some(),
                layer.kv_shared_layer_index
            ),
        );
        let per_layer_input = ple_inputs.and_then(|inputs| inputs.clone_layer_internal(layer_idx));
        let donor_cache = resolve_prefill_donor_cache(layer, &layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, layer_cache) =
            crate::shared::numerics::transformer_kernels::run_gemma4_layer_with_cache_internal(
                xs,
                &resolved_layer,
                per_layer_input,
                donor_cache,
                crate::shared::api::input::InferenceExecutionMode::Deterministic,
            )?;
        xs = layer_output.clone_internal();
        let xs_values = xs.clone_f32();
        let det_current_activations_sha256 = xs
            .det_values()
            .map(crate::shared::numerics::transformer_kernels::build_det_activation_commitment);
        layer_caches.push(layer_cache);
        completed_layer_output_sha256s.push(layer_output.activations_sha256);
        completed_layer_output_det_sha256s.push(layer_output.det_activations_sha256.clone());
        if crate::trace::trace_checkpoint(
            "prefill.layer",
            &json!({
                "execution_mode": "deterministic",
                "next_layer_idx": layer_idx + 1,
                "current_activations": xs_values.clone(),
                "current_activations_sha256": crate::shared::numerics::transformer_kernels::build_activation_commitment(&xs_values),
                "det_current_activations_sha256": det_current_activations_sha256,
                "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
                "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
                "completed_layer_output_sha256s": completed_layer_output_sha256s.clone(),
                "completed_layer_output_det_sha256s": completed_layer_output_det_sha256s.clone(),
            }),
        ) {
            break;
        }
        let mut reached_terminal_checkpoint = false;
        for (token_idx, token_activation) in xs_values.iter().enumerate() {
            let det_token_activation_sha256 = xs.det_values().and_then(|rows| {
                rows.get(token_idx).map(|row| {
                    crate::shared::numerics::transformer_kernels::build_det_vector_commitment(row)
                })
            });
            if crate::trace::trace_checkpoint(
                &format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}"),
                &json!({
                    "execution_mode": "deterministic",
                    "layer_idx": layer_idx,
                    "token_idx": token_idx,
                    "token_count": xs_values.len(),
                    "token_activation": token_activation,
                    "det_token_activation_sha256": det_token_activation_sha256,
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

    let xs_values = xs.clone_f32();
    let det_activations_sha256 = xs
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_activation_commitment);
    let mut activation_sequence = ActivationSequence::from_internal(
        xs,
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&xs_values),
    );
    activation_sequence.det_activations_sha256 = det_activations_sha256;
    Ok((activation_sequence, layer_caches))
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

pub(crate) fn run_internal_with_detour(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    detour_controller: Option<&mut RasterDetourController>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_text_layers_prefill_with_cache_internal_and_detour(
        input_activations,
        model,
        ple_inputs,
        detour_controller,
    )
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
