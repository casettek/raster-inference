use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};

use crate::shared::model::transformer::LayerKvCache;
use crate::shared::raster_contracts::prefill_layer::GemmaPrefillLayerMetadata;
use crate::shared::raster_kernels::transformer::{RasterActivationSequence, RasterKvCache};
use crate::shared::tensors::raster_row_store::{
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterAttentionHeadsRef,
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

pub(super) fn raster_sequence_acts(
    sequence: &RasterActivationSequence,
) -> Vec<Vec<crate::shared::numerics::det_num::Act>> {
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
