use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::routines::prefill_range::PrefillLayerRasterDetour;
use crate::routines::prefill_range_finalize::PrefillRangeFinalizeCheckpoint;
use crate::runtime::checkpoints::RasterDetourController;
use crate::runtime::checkpoints::RoutineId;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4LayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
    InternalActivationSequence, LayerKvCache,
};
use crate::shared::numerics::det_kernels::{
    det_layer_prefill, internal_sequence_from_slab, slab_from_internal_sequence,
};
use crate::shared::numerics::transformer_kernels::{
    build_det_activation_commitment_slab, build_det_vector_commitment,
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
    run_text_layers_prefill_with_cache_internal_with_range_width(
        input_activations,
        model,
        ple_inputs,
        crate::InferenceControls::DEFAULT_PREFILL_TOKEN_RANGE_WIDTH,
    )
}

pub(crate) fn run_text_layers_prefill_with_cache_internal_with_range_width(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    prefill_token_range_width: usize,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_text_layers_prefill_with_cache_internal_and_detour(
        input_activations,
        model,
        ple_inputs,
        None,
        None,
        prefill_token_range_width,
    )
}

pub(crate) fn run_text_layers_prefill_with_cache_internal_and_detour(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    mut detour_controller: Option<&mut RasterDetourController>,
    raster_detour: Option<PrefillLayerRasterDetour<'_>>,
    prefill_token_range_width: usize,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    if model.layers.is_empty() {
        bail!("transformer prefill requires at least one layer");
    }

    let mut xs = slab_from_internal_sequence(&input_activations)?;
    let mut layer_caches: Vec<LayerKvCache> = Vec::with_capacity(model.layers.len());
    // Deterministic mode carries only canonical commitments (spec v1); the f32
    // compatibility commitments are not collected.
    let mut completed_layer_output_sha256s: Vec<String> = Vec::new();
    let mut completed_layer_output_det_sha256s = Vec::with_capacity(model.layers.len());
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let selected_for_raster = match detour_controller.as_deref_mut() {
            Some(controller) => controller.should_detour_sim(RoutineId::PrefillRange)?,
            None => false,
        };
        if selected_for_raster {
            let detour = raster_detour.ok_or_else(|| {
                anyhow!(
                    "selective raster prefill.range detour requires an authenticated prefill layer source"
                )
            })?;
            let xs_internal = internal_sequence_from_slab(&xs);
            let output =
                crate::routines::prefill_range::run_selected_raster_detour_from_native_boundary(
                    &xs_internal,
                    layer_idx,
                    &layer_caches,
                    &completed_layer_output_sha256s,
                    &completed_layer_output_det_sha256s,
                    ple_inputs,
                    detour,
                )?;
            xs = slab_from_internal_sequence(&output.final_hidden_states.clone_internal())?;
            layer_caches = output.layer_caches;
            completed_layer_output_sha256s = output.completed_layer_output_sha256s;
            completed_layer_output_det_sha256s = output.completed_layer_output_det_sha256s;
            if crate::trace::reached_terminal_checkpoint_id().is_some() {
                break;
            }
            continue;
        }
        let _routine = routine_scope(
            RoutineId::PrefillRange,
            format!(
                "mode=det layer={layer_idx} tokens={} attention={:?} ple={} donor={:?}",
                xs.rows(),
                layer.attention_kind,
                layer.ple.is_some(),
                layer.kv_shared_layer_index
            ),
        );
        let per_layer_input = ple_inputs
            .and_then(|inputs| inputs.clone_layer_internal(layer_idx))
            .map(|input| slab_from_internal_sequence(&input))
            .transpose()?;
        let donor_cache = resolve_prefill_donor_cache(layer, &layer_caches, layer_idx)?
            .map(|cache| {
                cache.det_data().ok_or_else(|| {
                    anyhow!("deterministic attention requires canonical key cache rows")
                })
            })
            .transpose()?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, layer_cache) =
            det_layer_prefill(&xs, &resolved_layer, per_layer_input.as_ref(), donor_cache)?;
        xs = layer_output;
        let layer_output_det_sha256 = build_det_activation_commitment_slab(&xs);
        layer_caches.push(match layer_cache {
            Some(det) => LayerKvCache::from_det_data(det),
            None => LayerKvCache::new(resolved_layer.num_kv_heads),
        });
        completed_layer_output_det_sha256s.push(Some(layer_output_det_sha256.clone()));

        let current_activations = ActivationSequence::from_det_internal(
            internal_sequence_from_slab(&xs),
            Some(layer_output_det_sha256),
        );
        if crate::routines::prefill_range::trace_checkpoints(
            layer_idx,
            &current_activations,
            prefill_token_range_width,
            Some("deterministic"),
        )? {
            break;
        }
        if crate::routines::prefill_range_finalize::trace_checkpoint(
            PrefillRangeFinalizeCheckpoint {
                execution_mode: Some("deterministic"),
                layer_idx,
                current_activations: &current_activations,
                layer_caches: &layer_caches,
                completed_layer_output_sha256s: completed_layer_output_sha256s.clone(),
                completed_layer_output_det_sha256s: Some(
                    completed_layer_output_det_sha256s.clone(),
                ),
            },
        ) {
            break;
        }
        let mut reached_terminal_checkpoint = false;
        for token_idx in 0..xs.rows() {
            if crate::trace::trace_checkpoint_lazy(
                &format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}"),
                || {
                    json!({
                        "execution_mode": "deterministic",
                        "layer_idx": layer_idx,
                        "token_idx": token_idx,
                        "token_count": xs.rows(),
                        "det_token_activation_sha256": build_det_vector_commitment(xs.row(token_idx)),
                    })
                },
            ) {
                reached_terminal_checkpoint = true;
                break;
            }
        }
        if reached_terminal_checkpoint {
            break;
        }
    }

    let det_activations_sha256 = build_det_activation_commitment_slab(&xs);
    let activation_sequence = ActivationSequence::from_det_internal(
        internal_sequence_from_slab(&xs),
        Some(det_activations_sha256),
    );
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

pub(crate) fn run_internal_with_range_width(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    prefill_token_range_width: usize,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_text_layers_prefill_with_cache_internal_with_range_width(
        input_activations,
        model,
        ple_inputs,
        prefill_token_range_width,
    )
}

pub(crate) fn run_internal_with_detour(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    detour_controller: Option<&mut RasterDetourController>,
    raster_detour: Option<PrefillLayerRasterDetour<'_>>,
    prefill_token_range_width: usize,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_text_layers_prefill_with_cache_internal_and_detour(
        input_activations,
        model,
        ple_inputs,
        detour_controller,
        raster_detour,
        prefill_token_range_width,
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
