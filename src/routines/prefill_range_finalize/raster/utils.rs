use anyhow::Result;
use serde_json::json;

use crate::routines::prefill_range::raster::utils::{
    layer_caches_from_raster, materialize_prefill_activation_sequence_from_roots,
    materialize_prefill_layer_caches_from_roots, raster_sequence_acts,
};
use crate::routines::prefill_range::raster::{PrefillLayerCacheSlot, PrefillLayerRasterState};
use crate::routines::prefill_range_finalize::PrefillRangeFinalizeCheckpoint;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{ActivationSequence, InternalActivationSequence};
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
};

pub(crate) fn update_prefill_range_state_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut layer_state: PrefillLayerRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    layer_cache: PrefillLayerCacheSlot,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    if layer_idx != layer_state.next_layer_idx {
        anyhow::bail!(
            "cannot update prefill range layer {layer_idx} while next layer is {}",
            layer_state.next_layer_idx
        );
    }

    layer_state.current_activations_ref = layer_output_ref;
    layer_state.layer_caches.push(layer_cache);

    let completed_layer_output = trace_prefill_range_finalize_checkpoint_with_roots(
        &artifact_store_roots,
        &layer_state,
        layer_idx,
    )?;
    if let Some((sha256, det_sha256)) = completed_layer_output {
        layer_state.completed_layer_output_sha256s.push(sha256);
        layer_state
            .completed_layer_output_det_sha256s
            .push(det_sha256);
    }

    if trace_prefill_layer_token_checkpoints_with_roots(
        &artifact_store_roots,
        &layer_state,
        layer_idx,
    )? {
        layer_state.next_layer_idx += 1;
        layer_state.layer_count = layer_state.next_layer_idx;
        return Ok((true, artifact_store_roots, layer_state));
    }
    layer_state.next_layer_idx += 1;
    Ok((false, artifact_store_roots, layer_state))
}

fn trace_prefill_range_finalize_checkpoint_with_roots(
    artifact_store_roots: &RasterArtifactStoreRoots,
    state: &PrefillLayerRasterState,
    layer_idx: usize,
) -> Result<Option<(String, Option<String>)>> {
    let current_activations = materialize_prefill_activation_sequence_from_roots(
        artifact_store_roots,
        &state.current_activations_ref,
    )?;
    let current_values = current_activations.to_f32_values();
    let current_det_activations = raster_sequence_acts(&current_activations);
    let current_sha256 =
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&current_values);
    let current_det_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
            &current_det_activations,
        ),
    );
    let mut activation_sequence = ActivationSequence::from_internal(
        InternalActivationSequence::from_det_values(current_det_activations.clone()),
        current_sha256.clone(),
    );
    activation_sequence.det_activations_sha256 = current_det_sha256.clone();
    if crate::routines::prefill_range::trace_checkpoints(
        layer_idx,
        &activation_sequence,
        state.prefill_token_range_width,
        Some("deterministic"),
    )? {
        return Ok(Some((current_sha256, current_det_sha256)));
    }

    let raster_layer_caches =
        materialize_prefill_layer_caches_from_roots(artifact_store_roots, &state.layer_caches)?;
    let layer_caches = layer_caches_from_raster(&raster_layer_caches);
    let mut completed_layer_output_sha256s = state.completed_layer_output_sha256s.clone();
    completed_layer_output_sha256s.push(current_sha256.clone());
    let mut completed_layer_output_det_sha256s = state.completed_layer_output_det_sha256s.clone();
    completed_layer_output_det_sha256s.push(current_det_sha256.clone());
    let completed_layer_output = Some((current_sha256, current_det_sha256));
    let reached =
        crate::routines::prefill_range_finalize::trace_checkpoint(PrefillRangeFinalizeCheckpoint {
            execution_mode: Some("deterministic"),
            layer_idx,
            current_activations: &activation_sequence,
            layer_caches: &layer_caches,
            completed_layer_output_sha256s,
            completed_layer_output_det_sha256s: Some(completed_layer_output_det_sha256s),
        });
    if reached {
        return Ok(completed_layer_output);
    }
    Ok(completed_layer_output)
}

fn trace_prefill_layer_token_checkpoints_with_roots(
    artifact_store_roots: &RasterArtifactStoreRoots,
    state: &PrefillLayerRasterState,
    layer_idx: usize,
) -> Result<bool> {
    let (token_count, _) = state
        .current_activations_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    for token_idx in 0..token_count {
        let checkpoint_name = format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}");
        if crate::trace::trace_checkpoint_lazy_result(&checkpoint_name, || {
            let token_row = read_sequence_row_from_roots(
                artifact_store_roots,
                RasterSequenceRowRequest {
                    tensor_ref: state.current_activations_ref.clone(),
                    row_idx: token_idx,
                },
            )?;
            let token_activation = token_row.to_f32_values();
            let det_token_activation = token_row.acts();
            Ok(json!({
                "execution_mode": "deterministic",
                "layer_idx": layer_idx,
                "token_idx": token_idx,
                "token_count": token_count,
                "token_activation": token_activation,
                "det_token_activation_sha256": crate::shared::numerics::transformer_kernels::build_det_vector_commitment(&det_token_activation),
            }))
        })? {
            return Ok(true);
        }
    }
    Ok(false)
}
