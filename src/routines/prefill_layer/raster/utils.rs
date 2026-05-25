use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use super::types::*;
use crate::dsl::prelude::auth_read;
use crate::shared::artifacts::external_artifacts::CommittedExternalSource;
use crate::shared::artifacts::raster_artifact_store::RasterActivationSequenceArtifactRef;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::LayerKvCache;
use crate::shared::raster_contracts::prefill_layer::{
    GemmaPrefillLayerMetadata, GemmaPrefillLayerMetadataRequest,
    GemmaPrefillLayerSourceMetadataRequest,
};
use crate::shared::raster_contracts::prefill_ple::read_prefill_ple_input_manifest_from_roots;
use crate::shared::raster_kernels::transformer::{
    validate_projection_rows_per_tile, RasterActivationSequence, RasterKvCache,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, read_kv_row_from_roots, read_sequence_row_from_roots,
    RasterActivationSequenceRef, RasterAttentionHeadsRef, RasterKvRowKind, RasterKvRowRequest,
    RasterSequenceRowRequest, RasterTensorId,
};
use crate::RasterSizingControls;

use super::PrefillLayerCacheSlot;

pub(in super::super) fn prepare_next_prefill_layer_context(
    layer_state: &PrefillLayerRasterState,
    layer_source: &CommittedExternalSource,
) -> Result<PrefillLayerContext> {
    if layer_state.next_layer_idx >= layer_state.layer_count {
        bail!(
            "cannot prepare prefill layer {} after completing {} layers",
            layer_state.next_layer_idx,
            layer_state.layer_count
        );
    }

    let layer_idx = layer_state.next_layer_idx;
    let layer = auth_read!(layer_source, GemmaPrefillLayerMetadataRequest { layer_idx })?;
    let donor_cache =
        resolve_prefill_donor_cache_index(&layer_state.layer_caches, layer_idx, &layer)?
            .map(|donor_idx| {
                layer_state.layer_caches.get(donor_idx).cloned().ok_or_else(|| {
                anyhow!("transformer prefill donor cache {donor_idx} missing for layer {layer_idx}")
            })
            })
            .transpose()?;
    let per_layer_input = layer_state
        .per_layer_inputs
        .get(layer_idx)
        .and_then(Option::as_ref)
        .cloned();
    validate_prefill_layer_ple_input_ref(
        &layer_state.current_activations_ref,
        &layer,
        per_layer_input.as_ref(),
    )?;

    Ok(PrefillLayerContext {
        layer_idx,
        layer,
        donor_cache,
        per_layer_input,
    })
}

pub(in super::super) fn update_prefill_layer_state_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut layer_state: PrefillLayerRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    layer_cache: PrefillLayerCacheSlot,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    if layer_idx != layer_state.next_layer_idx {
        bail!(
            "cannot update prefill layer {layer_idx} while next layer is {}",
            layer_state.next_layer_idx
        );
    }

    layer_state.current_activations_ref = layer_output_ref;
    layer_state.layer_caches.push(layer_cache);

    let completed_layer_output =
        trace_prefill_layer_checkpoint_with_roots(&artifact_store_roots, &layer_state, layer_idx)?;
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

pub(in super::super) fn init_prefill_layer_state_from_activation_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_activations_ref: RasterActivationSequenceArtifactRef,
    layer_source: &CommittedExternalSource,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    crate::shared::raster_kernels::transformer::validate_attention_kv_rows_per_tile(
        raster_sizing.attention_kv_rows_per_tile,
    )?;
    crate::shared::raster_kernels::transformer::validate_sequence_rows_per_tile(
        raster_sizing.sequence_rows_per_tile,
    )?;
    crate::shared::raster_kernels::transformer::validate_head_rows_per_tile(
        raster_sizing.head_rows_per_tile,
    )?;
    ensure_artifact_root_present(&artifact_store_roots, input_activations_ref.root())?;
    let metadata = auth_read!(layer_source, GemmaPrefillLayerSourceMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer prefill requires at least one layer");
    }

    if input_activations_ref.row_count() == 0 {
        bail!("transformer layer execution requires at least one activation row");
    }
    let first_layer = auth_read!(
        layer_source,
        GemmaPrefillLayerMetadataRequest { layer_idx: 0 }
    )?;
    if input_activations_ref.width() != first_layer.hidden_size {
        bail!(
            "transformer layer input width {}, expected {}",
            input_activations_ref.width(),
            first_layer.hidden_size
        );
    }
    let token_count = input_activations_ref.row_count();
    let current_activations_ref = activation_sequence_ref_from_artifact(
        RasterTensorId::new(format!(
            "prefill.layer.current.initial.{}",
            input_activations_ref.root()
        ))?,
        input_activations_ref,
    )?;

    let ple_input_refs = ple_input_manifest_root
        .map(|root| {
            read_prefill_ple_input_manifest_from_roots(&artifact_store_roots, root)?
                .into_prefill_ple_input_refs(artifact_store_roots.clone())
        })
        .transpose()?;
    let per_layer_inputs = match ple_input_refs.as_ref() {
        Some(ple_input_refs) => {
            if ple_input_refs.source_id() != metadata.source_id {
                bail!(
                    "raster PLE input refs source {} does not match prefill layer source {}",
                    ple_input_refs.source_id(),
                    metadata.source_id
                );
            }
            if ple_input_refs.layer_count() != metadata.layer_count {
                bail!(
                    "raster PLE input refs contain {} layers, expected {}",
                    ple_input_refs.layer_count(),
                    metadata.layer_count
                );
            }
            if ple_input_refs.token_count() != token_count {
                bail!(
                    "raster PLE input refs contain {} tokens, expected {token_count}",
                    ple_input_refs.token_count()
                );
            }
            ple_input_refs
                .per_layer_inputs()
                .iter()
                .enumerate()
                .map(|(layer_idx, input_ref)| {
                    input_ref
                        .as_ref()
                        .map(|input_ref| {
                            ensure_artifact_root_present(&artifact_store_roots, input_ref.root())?;
                            activation_sequence_ref_from_artifact(
                                RasterTensorId::new(format!(
                                    "prefill.layer.per_layer_input.{layer_idx}.{}",
                                    input_ref.root()
                                ))?,
                                input_ref.clone(),
                            )
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => vec![None; metadata.layer_count],
    };

    Ok((
        artifact_store_roots,
        PrefillLayerRasterState {
            current_activations_ref,
            next_layer_idx: 0,
            layer_count: metadata.layer_count,
            layer_caches: Vec::with_capacity(metadata.layer_count),
            per_layer_inputs,
            completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
            projection_rows_per_tile: raster_sizing.projection_rows_per_tile,
            attention_kv_rows_per_tile: raster_sizing.attention_kv_rows_per_tile,
            sequence_rows_per_tile: raster_sizing.sequence_rows_per_tile,
            head_rows_per_tile: raster_sizing.head_rows_per_tile,
        },
    ))
}

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

pub(in super::super) fn active_prefill_layer_work(
    work: PrefillLayerArtifactWork,
) -> Result<Box<PrefillLayerActiveWork>> {
    match work {
        PrefillLayerArtifactWork::Active(work) => Ok(work),
        PrefillLayerArtifactWork::Passthrough(_) => {
            bail!("prefill layer artifact work unexpectedly skipped")
        }
    }
}

pub(in super::super) fn require_activation_ref(
    value: Option<RasterActivationSequenceRef>,
    description: &str,
) -> Result<RasterActivationSequenceRef> {
    value.ok_or_else(|| anyhow!("prefill layer artifact work is missing {description}"))
}

pub(in super::super) fn require_heads_ref(
    value: Option<RasterAttentionHeadsRef>,
    description: &str,
) -> Result<RasterAttentionHeadsRef> {
    value.ok_or_else(|| anyhow!("prefill layer artifact work is missing {description}"))
}
