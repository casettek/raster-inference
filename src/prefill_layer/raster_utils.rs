use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};

use crate::raster_authoring::prelude::auth_read;
use crate::shared::raster_artifact_store::RasterActivationSequenceArtifactRef;
use crate::shared::raster_prefill_layer::{
    AuthenticatedGemmaPrefillLayerSource, GemmaPrefillLayerMetadata,
    GemmaPrefillLayerSourceMetadataRequest,
};
use crate::shared::raster_prefill_ple::RasterPrefillPleInputRefs;
use crate::shared::raster_row_store::{
    insert_activation_sequence_artifact_ref, AuthenticatedRasterTensorStore,
    RasterActivationSequenceRef, RasterAttentionHeadsRef, RasterTensorId,
};
use crate::shared::raster_transformer_kernels::{RasterActivationSequence, RasterKvCache};
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, InternalActivationSequence, LayerKvCache,
};

use super::raster_tiles::PrefillLayerCacheSlot;

pub(super) fn validate_prefill_layer_ple_input_ref(
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

pub(super) fn retained_prefill_kv_cache_len(
    key_ref: &RasterAttentionHeadsRef,
    sliding_window: Option<usize>,
) -> Result<usize> {
    let (_, sequence_len, _) = key_ref.tensor_ref().shape().heads_metadata()?;
    Ok(sliding_window.map_or(sequence_len, |window| window.min(sequence_len)))
}

pub(super) fn resolve_prefill_donor_cache_index(
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

#[cfg(test)]
pub(super) fn register_prefill_layer_cache(
    store: &mut AuthenticatedRasterTensorStore,
    layer_idx: usize,
    cache: RasterKvCache,
) -> Result<PrefillLayerCacheSlot> {
    if cache.current_len() == 0 {
        return Ok(PrefillLayerCacheSlot::Empty {
            num_kv_heads: cache.head_count(),
        });
    }

    Ok(PrefillLayerCacheSlot::Ref(store.insert_kv_cache(
        RasterTensorId::new(format!("prefill.layer.cache.{layer_idx}.keys"))?,
        RasterTensorId::new(format!("prefill.layer.cache.{layer_idx}.values"))?,
        cache,
    )?))
}

pub(super) fn import_materialized_ple_inputs(
    _store: &mut AuthenticatedRasterTensorStore,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    // Host/dev compatibility bridge only. This deliberately lives outside any
    // authored tile or sequence so proof-shaped code cannot accidentally ingest
    // all PLE layer inputs as one materialized argument.
    let Some(ple_inputs) = ple_inputs else {
        return Ok(None);
    };
    let metadata = auth_read!(layer_source, GemmaPrefillLayerSourceMetadataRequest)?;
    let mut token_count = None;
    let per_layer_inputs = (0..metadata.layer_count)
        .map(|layer_idx| {
            let input = ple_inputs
                .clone_layer_internal(layer_idx)
                .map(|input| raster_activation_sequence_from_internal(&input))
                .transpose()?;
            input
                .map(|input| {
                    let input_len = input.len();
                    match token_count {
                        Some(expected) if expected != input_len => bail!(
                            "materialized PLE input layer {layer_idx} contains {input_len} tokens, expected {expected}"
                        ),
                        None => token_count = Some(input_len),
                        _ => {}
                    }
                    insert_ple_input_artifact(layer_idx, input)
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;

    if per_layer_inputs.iter().all(Option::is_none) {
        return Ok(None);
    }

    Ok(Some(RasterPrefillPleInputRefs::new(
        metadata.source_id,
        metadata.layer_count,
        token_count.ok_or_else(|| anyhow!("materialized PLE inputs contained no layer rows"))?,
        per_layer_inputs,
    )?))
}

pub(super) fn register_ple_input_artifact_refs(
    store: &mut AuthenticatedRasterTensorStore,
    ple_input_refs: &RasterPrefillPleInputRefs,
) -> Result<Vec<Option<RasterActivationSequenceRef>>> {
    ple_input_refs
        .per_layer_inputs()
        .iter()
        .enumerate()
        .map(|(layer_idx, input_ref)| {
            input_ref
                .as_ref()
                .map(|input_ref| {
                    store.register_activation_sequence_artifact(
                        RasterTensorId::new(format!(
                            "prefill.layer.per_layer_input.{layer_idx}.{}",
                            input_ref.root()
                        ))?,
                        input_ref.clone(),
                    )
                })
                .transpose()
        })
        .collect()
}

pub(super) fn insert_ple_input_artifact(
    layer_idx: usize,
    input: RasterActivationSequence,
) -> Result<RasterActivationSequenceArtifactRef> {
    insert_activation_sequence_artifact_ref(
        &format!("prefill.layer.materialized_ple.{layer_idx}"),
        input,
    )
}

pub(super) fn materialize_prefill_activation_sequence_from_store(
    store: &AuthenticatedRasterTensorStore,
    sequence_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationSequence> {
    // Public/dev compatibility boundary. zkVM-target substeps should consume the
    // ref directly and avoid this materializer.
    store.materialize_sequence(sequence_ref)
}

pub(super) fn materialize_prefill_layer_cache_from_store(
    store: &AuthenticatedRasterTensorStore,
    cache: &PrefillLayerCacheSlot,
) -> Result<RasterKvCache> {
    // Public/dev compatibility boundary. Shared-store layer substeps use
    // `PrefillLayerCacheSlot::Ref` directly when replaying zkVM-shaped work.
    match cache {
        PrefillLayerCacheSlot::Empty { num_kv_heads } => Ok(RasterKvCache::empty(*num_kv_heads)),
        PrefillLayerCacheSlot::Ref(cache_ref) => store.materialize_kv_cache(cache_ref),
    }
}

pub(super) fn materialize_prefill_layer_caches(
    store: &AuthenticatedRasterTensorStore,
    caches: &[PrefillLayerCacheSlot],
) -> Result<Vec<RasterKvCache>> {
    caches
        .iter()
        .map(|cache| materialize_prefill_layer_cache_from_store(store, cache))
        .collect()
}

pub(super) fn raster_activation_sequence_from_activation(
    input_activations: &ActivationSequence,
) -> Result<RasterActivationSequence> {
    raster_activation_sequence_from_internal(&input_activations.clone_internal())
}

pub(super) fn raster_activation_sequence_from_internal(
    input_activations: &InternalActivationSequence,
) -> Result<RasterActivationSequence> {
    let det_rows = input_activations.det_values().ok_or_else(|| {
        anyhow!("deterministic raster prefill layer input requires canonical activations")
    })?;
    Ok(RasterActivationSequence::from_acts(det_rows.to_vec()))
}

pub(super) fn raster_sequence_acts(
    sequence: &RasterActivationSequence,
) -> Vec<Vec<crate::shared::det_num::Act>> {
    sequence.rows().iter().map(|row| row.acts()).collect()
}

pub(super) fn layer_cache_from_raster(cache: RasterKvCache) -> LayerKvCache {
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

pub(super) fn layer_caches_from_raster(caches: &[RasterKvCache]) -> Vec<LayerKvCache> {
    caches
        .iter()
        .cloned()
        .map(layer_cache_from_raster)
        .collect()
}
