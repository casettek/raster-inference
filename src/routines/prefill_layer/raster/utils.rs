use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use super::types::*;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::LayerKvCache;
use crate::shared::raster_contracts::prefill_layer::GemmaPrefillLayerMetadata;
use crate::shared::raster_kernels::transformer::{RasterActivationSequence, RasterKvCache};
use crate::shared::tensors::raster_tensor_artifacts::{
    read_kv_row_from_roots, read_sequence_row_from_roots, RasterActivationSequenceRef,
    RasterAttentionHeadsRef, RasterKvRowKind, RasterKvRowRequest, RasterSequenceRowRequest,
};

use super::PrefillLayerCacheSlot;

pub(in super::super) fn validate_prefill_layer_ple_input_ref(
    current_activations_ref: &RasterActivationSequenceRef,
    layer: &GemmaPrefillLayerMetadata,
    per_layer_input: Option<&RasterActivationSequenceRef>,
) -> Result<()> {
    match (layer.has_ple, per_layer_input) {
        (false, Some(_)) => bail!("transformer layer received PLE inputs without PLE weights"),
        (true, None) => bail!("transformer layer requires PLE inputs but none were provided"),
        (false, None) => Ok(()),
        (true, Some(input_ref)) => {
            let (token_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
            let (expected_token_count, _) = current_activations_ref
                .tensor_ref()
                .shape()
                .sequence_metadata()?;
            let expected_width = expected_prefill_ple_input_width(layer)?;
            if token_count != expected_token_count {
                bail!(
                    "transformer layer PLE input has {token_count} rows, expected {expected_token_count}"
                );
            }
            if width != expected_width {
                bail!(
                    "transformer layer PLE input width {width}, expected {}",
                    expected_width
                );
            }
            Ok(())
        }
    }
}

fn expected_prefill_ple_input_width(layer: &GemmaPrefillLayerMetadata) -> Result<usize> {
    let input_gate_shape = layer
        .ple_input_gate_shape
        .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing PLE input gate shape"))?;
    let layer_projection_shape = layer.ple_layer_projection_shape.ok_or_else(|| {
        anyhow!("Gemma prefill layer metadata is missing PLE layer projection shape")
    })?;
    if input_gate_shape.rows != layer_projection_shape.cols {
        bail!(
            "Gemma prefill layer PLE width mismatch: input gate rows {} vs layer projection cols {}",
            input_gate_shape.rows,
            layer_projection_shape.cols
        );
    }
    Ok(input_gate_shape.rows)
}

pub(in super::super) fn retained_prefill_kv_cache_len(
    key_ref: &RasterAttentionHeadsRef,
    sliding_window: Option<usize>,
) -> Result<usize> {
    let (_, sequence_len, _) = key_ref.tensor_ref().shape().heads_metadata()?;
    Ok(sliding_window.map_or(sequence_len, |window| window.min(sequence_len)))
}

pub(in super::super) fn resolve_prefill_donor_cache_index(
    layer_caches: &[PrefillLayerCacheSlot],
    layer_idx: usize,
    layer: &GemmaPrefillLayerMetadata,
) -> Result<Option<usize>> {
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
            })?;
            Ok(donor_idx)
        })
        .transpose()
}

pub(in super::super) fn materialize_prefill_activation_sequence_from_roots(
    roots: &RasterArtifactStoreRoots,
    sequence_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationSequence> {
    // Public/dev compatibility boundary. zkVM-target substeps should consume the
    // ref directly and avoid this materializer.
    let (row_count, _) = sequence_ref.tensor_ref().shape().sequence_metadata()?;
    let rows = (0..row_count)
        .map(|row_idx| {
            read_sequence_row_from_roots(
                roots,
                RasterSequenceRowRequest {
                    tensor_ref: sequence_ref.clone(),
                    row_idx,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RasterActivationSequence::from_rows(rows))
}

pub(in super::super) fn materialize_prefill_layer_cache_from_roots(
    roots: &RasterArtifactStoreRoots,
    cache: &PrefillLayerCacheSlot,
) -> Result<RasterKvCache> {
    // Public/dev compatibility boundary. Shared-store layer substeps use
    // `PrefillLayerCacheSlot::Ref` directly when replaying zkVM-shaped work.
    match cache {
        PrefillLayerCacheSlot::Empty { num_kv_heads } => Ok(RasterKvCache::empty(*num_kv_heads)),
        PrefillLayerCacheSlot::Ref(cache_ref) => {
            let (head_count, current_len, _) = cache_ref.shape().kv_cache_metadata()?;
            let mut keys = vec![Vec::with_capacity(current_len); head_count];
            let mut values = vec![Vec::with_capacity(current_len); head_count];
            for head_idx in 0..head_count {
                for token_idx in 0..current_len {
                    keys[head_idx].push(read_kv_row_from_roots(
                        roots,
                        RasterKvRowRequest {
                            cache_ref: cache_ref.clone(),
                            row_kind: RasterKvRowKind::Key,
                            head_idx,
                            token_idx,
                        },
                    )?);
                    values[head_idx].push(read_kv_row_from_roots(
                        roots,
                        RasterKvRowRequest {
                            cache_ref: cache_ref.clone(),
                            row_kind: RasterKvRowKind::Value,
                            head_idx,
                            token_idx,
                        },
                    )?);
                }
            }
            RasterKvCache::from_heads(keys, values)
        }
    }
}

pub(in super::super) fn materialize_prefill_layer_caches_from_roots(
    roots: &RasterArtifactStoreRoots,
    caches: &[PrefillLayerCacheSlot],
) -> Result<Vec<RasterKvCache>> {
    caches
        .iter()
        .map(|cache| materialize_prefill_layer_cache_from_roots(roots, cache))
        .collect()
}

pub(in super::super) fn raster_sequence_acts(
    sequence: &RasterActivationSequence,
) -> Vec<Vec<crate::shared::numerics::det_num::Act>> {
    sequence.rows().iter().map(|row| row.acts()).collect()
}

pub(in super::super) fn layer_cache_from_raster(cache: RasterKvCache) -> LayerKvCache {
    if cache.current_len() == 0 {
        return LayerKvCache::new(cache.head_count());
    }

    LayerKvCache::from_det_heads(
        cache
            .keys()
            .iter()
            .map(|head| head.iter().map(|row| row.acts()).collect::<VecDeque<_>>())
            .collect(),
        cache
            .values()
            .iter()
            .map(|head| head.iter().map(|row| row.acts()).collect::<VecDeque<_>>())
            .collect(),
    )
}

pub(in super::super) fn layer_caches_from_raster(caches: &[RasterKvCache]) -> Vec<LayerKvCache> {
    caches
        .iter()
        .cloned()
        .map(layer_cache_from_raster)
        .collect()
}

pub(in super::super) fn ensure_artifact_root_present(
    roots: &RasterArtifactStoreRoots,
    root: &str,
) -> Result<()> {
    if roots.artifacts.iter().any(|entry| entry.root() == root) {
        return Ok(());
    }
    bail!("raster artifact root {root} is not present in the store roots snapshot")
}

pub(in super::super) fn trace_prefill_layer_checkpoint(
    artifact_store_roots: &RasterArtifactStoreRoots,
    state: &PrefillLayerRasterState,
    layer_idx: usize,
) -> Result<Option<(String, Option<String>)>> {
    let mut completed_layer_output = None;
    let reached = crate::trace::trace_checkpoint_lazy_result("prefill.layer", || {
        let current_activations = materialize_prefill_activation_sequence_from_roots(
            artifact_store_roots,
            &state.current_activations_ref,
        )?;
        let current_values = current_activations.to_f32_values();
        let current_det_activations = raster_sequence_acts(&current_activations);
        let current_sha256 =
            crate::shared::numerics::transformer_kernels::build_activation_commitment(
                &current_values,
            );
        let current_det_sha256 = Some(
            crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
                &current_det_activations,
            ),
        );
        let raster_layer_caches =
            materialize_prefill_layer_caches_from_roots(artifact_store_roots, &state.layer_caches)?;
        let layer_caches = layer_caches_from_raster(&raster_layer_caches);
        let mut completed_layer_output_sha256s = state.completed_layer_output_sha256s.clone();
        completed_layer_output_sha256s.push(current_sha256.clone());
        let mut completed_layer_output_det_sha256s =
            state.completed_layer_output_det_sha256s.clone();
        completed_layer_output_det_sha256s.push(current_det_sha256.clone());
        completed_layer_output = Some((current_sha256.clone(), current_det_sha256.clone()));
        Ok(json!({
            "execution_mode": "deterministic",
            "next_layer_idx": layer_idx + 1,
            "current_activations": current_values,
            "current_activations_sha256": current_sha256,
            "det_current_activations_sha256": current_det_sha256,
            "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
            "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
            "completed_layer_output_sha256s": completed_layer_output_sha256s,
            "completed_layer_output_det_sha256s": completed_layer_output_det_sha256s,
        }))
    })?;
    if reached {
        return Ok(completed_layer_output);
    }
    Ok(completed_layer_output)
}

pub(in super::super) fn trace_prefill_layer_checkpoint_with_roots(
    artifact_store_roots: &RasterArtifactStoreRoots,
    state: &PrefillLayerRasterState,
    layer_idx: usize,
) -> Result<Option<(String, Option<String>)>> {
    trace_prefill_layer_checkpoint(artifact_store_roots, state, layer_idx)
}

pub(in super::super) fn trace_prefill_layer_token_checkpoints_with_roots(
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
