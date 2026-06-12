use crate::routines::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::runtime::checkpoints::{RasterDetourController, RoutineId};
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence,
    LayerKvCache,
};
use crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource;
use crate::RasterSizingControls;
use anyhow::{bail, Result};
use serde_json::json;

use self::raster::utils::{
    insert_activation_sequence_ref_with_roots, insert_prefill_layer_cache_with_roots,
    layer_cache_from_raster, materialize_prefill_activation_sequence_from_roots,
    materialize_prefill_layer_cache_from_roots, raster_activation_sequence_from_internal,
    raster_sequence_acts,
};

pub mod native;
pub mod raster;

pub(crate) fn range_bounds(
    token_count: usize,
    prefill_token_range_width: usize,
) -> Vec<(usize, usize)> {
    let width = prefill_token_range_width.max(1).min(token_count.max(1));
    (0..token_count)
        .step_by(width)
        .map(|start| (start, start.saturating_add(width).min(token_count)))
        .collect()
}

pub(crate) fn trace_checkpoints(
    layer_idx: usize,
    layer_output: &ActivationSequence,
    prefill_token_range_width: usize,
    execution_mode: Option<&str>,
) -> Result<bool> {
    let deterministic = execution_mode == Some("deterministic");
    let internal = layer_output.clone_internal();
    let token_count = if deterministic {
        internal.det_values().map(<[Vec<_>]>::len).unwrap_or(0)
    } else {
        layer_output.activations.len()
    };
    for (range_start, range_end) in range_bounds(token_count, prefill_token_range_width) {
        let _routine = crate::trace::routine_scope(
            RoutineId::PrefillRange,
            format!("layer={layer_idx} tokens={range_start}..{range_end}/{token_count}"),
        );
        let reached = crate::trace::trace_checkpoint_lazy("prefill.range", || {
            let det_range_activations_sha256 = internal.det_values().map(|rows| {
                crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
                    &rows[range_start..range_end],
                )
            });
            let mut payload = json!({
                "layer_idx": layer_idx,
                "range_start": range_start,
                "range_end": range_end,
                "token_count": token_count,
                "det_range_activations_sha256": det_range_activations_sha256,
            });
            if !deterministic {
                // Deterministic-mode payloads carry only canonical commitments
                // (spec v1); fp32 mode keeps the compatibility fields.
                let activations = &layer_output.activations;
                payload["range_activations"] = json!(activations[range_start..range_end].to_vec());
                payload["range_activations_sha256"] = json!(
                    crate::shared::numerics::transformer_kernels::build_activation_commitment(
                        &activations[range_start..range_end]
                    )
                );
            }
            if let Some(execution_mode) = execution_mode {
                payload["execution_mode"] = json!(execution_mode);
            }
            payload
        });
        if reached {
            return Ok(true);
        }
    }
    Ok(false)
}

#[derive(Clone, Copy)]
pub(crate) struct PrefillLayerRasterDetour<'a> {
    pub(crate) layer_source: &'a AuthenticatedGemmaPrefillLayerSource,
    pub(crate) raster_sizing: RasterSizingControls,
}

pub(crate) struct PrefillLayerNativeDetourOutput {
    pub(crate) final_hidden_states: ActivationSequence,
    pub(crate) layer_caches: Vec<LayerKvCache>,
    pub(crate) completed_layer_output_sha256s: Vec<String>,
    pub(crate) completed_layer_output_det_sha256s: Vec<Option<String>>,
}

pub fn run(
    input_activations: &[Vec<f32>],
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_with_mode(
        input_activations,
        model,
        ple_inputs,
        InferenceExecutionMode::Fp32,
    )
}

pub fn run_with_mode(
    input_activations: &[Vec<f32>],
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_with_mode_internal(
        InternalActivationSequence::from_values(input_activations.to_vec()),
        model,
        ple_inputs,
        execution_mode,
    )
}

pub fn materialize_prefill_layer_output_refs_from_roots_for_trace(
    roots: &RasterArtifactStoreRoots,
    refs: &raster::PrefillLayerOutputRefs,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    // Public/dev compatibility boundary. Route-based prefill carries refs and
    // roots through zkVM-shaped work; this materializes only for public results
    // and checkpoint payloads.
    let current_activations =
        materialize_prefill_activation_sequence_from_roots(roots, &refs.final_hidden_states_ref)?;
    let det_activations = raster_sequence_acts(&current_activations);
    let activation_sequence = ActivationSequence::from_det_internal(
        InternalActivationSequence::from_det_values_only(det_activations.clone()),
        Some(
            crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
                &det_activations,
            ),
        ),
    );

    Ok((
        activation_sequence,
        refs.layer_caches
            .iter()
            .map(|cache| {
                materialize_prefill_layer_cache_from_roots(roots, cache)
                    .map(layer_cache_from_raster)
            })
            .collect::<Result<Vec<_>>>()?,
    ))
}

pub fn run_raster(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, raster::PrefillLayerOutputRefs)> {
    let layer_source =
        crate::shared::raster_contracts::prefill_layer::RasterPrefillLayerSource::for_current_integrity_mode(layer_source)?;
    raster::main(
        artifact_store_roots,
        input_embedding_refs,
        &layer_source,
        ple_input_manifest_root,
        raster_sizing,
    )
}

pub(crate) fn insert_prefill_layer_output_refs_from_native(
    final_hidden_states: &ActivationSequence,
    layer_caches: &[LayerKvCache],
    source_name_prefix: &str,
    detour_routine: &str,
) -> Result<(RasterArtifactStoreRoots, raster::PrefillLayerOutputRefs)> {
    let mut artifact_store_roots = ArtifactIo::export_store_roots();
    let final_hidden_states_sequence = raster_activation_sequence_from_internal(
        &final_hidden_states.clone_internal(),
        detour_routine,
        "final hidden states",
    )?;
    let (next_roots, final_hidden_states_ref) = insert_activation_sequence_ref_with_roots(
        &artifact_store_roots,
        format!("{source_name_prefix}.final_hidden_states"),
        final_hidden_states_sequence,
    )?;
    artifact_store_roots = next_roots;

    let mut raster_layer_caches = Vec::with_capacity(layer_caches.len());
    for (cache_idx, cache) in layer_caches.iter().enumerate() {
        let (next_roots, cache_slot) = insert_prefill_layer_cache_with_roots(
            &artifact_store_roots,
            &format!("{source_name_prefix}.cache.{cache_idx}"),
            cache,
            detour_routine,
        )?;
        artifact_store_roots = next_roots;
        raster_layer_caches.push(cache_slot);
    }

    Ok((
        artifact_store_roots,
        raster::PrefillLayerOutputRefs {
            final_hidden_states_ref,
            layer_caches: raster_layer_caches,
        },
    ))
}

pub(crate) fn run_selected_raster_detour_from_native_boundary(
    input_activations: &InternalActivationSequence,
    layer_idx: usize,
    layer_caches: &[LayerKvCache],
    completed_layer_output_sha256s: &[String],
    completed_layer_output_det_sha256s: &[Option<String>],
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    detour: PrefillLayerRasterDetour<'_>,
) -> Result<PrefillLayerNativeDetourOutput> {
    if layer_caches.len() != layer_idx {
        bail!(
            "selective raster prefill.layer detour at layer {layer_idx} received {} prior caches",
            layer_caches.len()
        );
    }
    // Deterministic mode no longer collects f32 compatibility commitments, so
    // an empty f32 commitment list is accepted alongside the canonical list.
    if (!completed_layer_output_sha256s.is_empty()
        && completed_layer_output_sha256s.len() != layer_idx)
        || completed_layer_output_det_sha256s.len() != layer_idx
    {
        bail!(
            "selective raster prefill.layer detour at layer {layer_idx} received inconsistent completed layer commitments"
        );
    }

    let layer_source =
        crate::shared::raster_contracts::prefill_layer::RasterPrefillLayerSource::for_current_integrity_mode(detour.layer_source)?;
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let input_sequence = raster_activation_sequence_from_internal(
        input_activations,
        "prefill.layer",
        "current activations",
    )?;
    let (mut artifact_store_roots, current_activations_ref) =
        insert_activation_sequence_ref_with_roots(
            &artifact_store_roots,
            format!("prefill.layer.detour.input.{layer_idx}"),
            input_sequence,
        )?;

    let mut raster_layer_caches = Vec::with_capacity(layer_caches.len());
    for (cache_idx, cache) in layer_caches.iter().enumerate() {
        let (next_roots, cache_slot) = insert_prefill_layer_cache_with_roots(
            &artifact_store_roots,
            &format!("prefill.layer.detour.cache.{cache_idx}"),
            cache,
            "prefill.layer",
        )?;
        artifact_store_roots = next_roots;
        raster_layer_caches.push(cache_slot);
    }

    let mut per_layer_inputs = vec![None; layer_idx + 1];
    if let Some(per_layer_input) =
        ple_inputs.and_then(|inputs| inputs.clone_layer_internal(layer_idx))
    {
        let input_sequence = raster_activation_sequence_from_internal(
            &per_layer_input,
            "prefill.layer",
            "PLE input rows",
        )?;
        let (next_roots, input_ref) = insert_activation_sequence_ref_with_roots(
            &artifact_store_roots,
            format!("prefill.layer.detour.ple.{layer_idx}"),
            input_sequence,
        )?;
        artifact_store_roots = next_roots;
        per_layer_inputs[layer_idx] = Some(input_ref);
    }

    let layer_state = raster::PrefillLayerRasterState {
        current_activations_ref,
        next_layer_idx: layer_idx,
        layer_count: layer_idx + 1,
        layer_caches: raster_layer_caches,
        per_layer_inputs,
        completed_layer_output_sha256s: completed_layer_output_sha256s.to_vec(),
        completed_layer_output_det_sha256s: completed_layer_output_det_sha256s.to_vec(),
        projection_rows_per_tile: detour.raster_sizing.projection_rows_per_tile,
        attention_kv_rows_per_tile: detour.raster_sizing.attention_kv_rows_per_tile,
        sequence_rows_per_tile: detour.raster_sizing.sequence_rows_per_tile,
        head_rows_per_tile: detour.raster_sizing.head_rows_per_tile,
        prefill_token_range_width: detour.raster_sizing.prefill_token_range_width,
    };

    let mut complete = false;
    let mut layer_state = layer_state;
    while !complete {
        let (next_complete, next_roots, next_state) =
            raster::compute_next_prefill_layer_sequence_with_roots(
                artifact_store_roots,
                layer_state,
                &layer_source,
            )?;
        complete = next_complete;
        artifact_store_roots = next_roots;
        layer_state = next_state;
    }
    if layer_state.next_layer_idx != layer_idx + 1 {
        bail!(
            "selective raster prefill.layer detour finalized after {} layers, expected {}",
            layer_state.next_layer_idx,
            layer_idx + 1
        );
    }

    let completed_layer_output_sha256s = layer_state.completed_layer_output_sha256s.clone();
    let completed_layer_output_det_sha256s = layer_state.completed_layer_output_det_sha256s.clone();
    let refs = raster::finalize_prefill_layer_refs(layer_state)?;
    let (final_hidden_states, layer_caches) =
        materialize_prefill_layer_output_refs_from_roots_for_trace(&artifact_store_roots, &refs)?;

    Ok(PrefillLayerNativeDetourOutput {
        final_hidden_states,
        layer_caches,
        completed_layer_output_sha256s,
        completed_layer_output_det_sha256s,
    })
}

pub(crate) fn run_with_mode_internal(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_with_mode_internal_with_detour(
        input_activations,
        model,
        ple_inputs,
        execution_mode,
        None,
        None,
        crate::InferenceControls::DEFAULT_PREFILL_TOKEN_RANGE_WIDTH,
    )
}

pub(crate) fn run_with_mode_internal_with_detour(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    execution_mode: InferenceExecutionMode,
    detour_controller: Option<&mut RasterDetourController>,
    raster_detour: Option<PrefillLayerRasterDetour<'_>>,
    prefill_token_range_width: usize,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    model.validate_execution_mode(execution_mode)?;
    match execution_mode {
        InferenceExecutionMode::Fp32 => native::run_with_range_width(
            input_activations.as_f32_slice(),
            model,
            ple_inputs,
            prefill_token_range_width,
        ),
        InferenceExecutionMode::Deterministic => match detour_controller {
            Some(detour_controller) => native::deterministic_tiles::run_internal_with_detour(
                input_activations,
                model,
                ple_inputs,
                Some(detour_controller),
                raster_detour,
                prefill_token_range_width,
            ),
            None => native::deterministic_tiles::run_internal_with_range_width(
                input_activations,
                model,
                ple_inputs,
                prefill_token_range_width,
            ),
        },
    }
}
