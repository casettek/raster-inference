use anyhow::{bail, Result};

use crate::routines::prefill_finalize::raster::{RasterPrefillFinalizeRefs, PREFILL_LOGITS_ARTIFACT_NAME};
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{
    ActivationSequence, InternalLogits, LayerKvCache, PrefillLogits, TransformerPrefillResult,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
};

pub fn build_prefill_result_from_root_refs(
    roots: &RasterArtifactStoreRoots,
    refs: &RasterPrefillFinalizeRefs,
) -> Result<TransformerPrefillResult> {
    let layer_refs = crate::routines::prefill_range::raster::PrefillLayerOutputRefs {
        final_hidden_states_ref: refs.final_hidden_states_ref.clone(),
        layer_caches: refs.layer_caches.clone(),
    };
    let (final_hidden_states, layer_caches) =
        crate::routines::prefill_range::materialize_prefill_layer_output_refs_from_roots_for_trace(
            roots,
            &layer_refs,
        )?;
    let det_logits = materialize_prefill_logits_from_roots(roots, &refs.logits_ref)?;
    // Deterministic logits carry only the canonical commitment (spec v1).
    let prefill_logits = PrefillLogits::from_det_internal(
        InternalLogits::from_det_values_only(det_logits.clone()),
        Some(
            crate::shared::numerics::transformer_kernels::build_det_vector_commitment(&det_logits),
        ),
    );

    build_prefill_result(
        refs.prompt_token_count,
        final_hidden_states,
        layer_caches,
        prefill_logits,
    )
}

pub fn build_prefill_result(
    prompt_token_count: usize,
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
    prefill_logits: PrefillLogits,
) -> Result<TransformerPrefillResult> {
    // Delegates to the shared checkpoint/result builder so native and raster
    // prefill.finalize payloads stay in lockstep.
    crate::routines::prefill_finalize::native::build_prefill_result(
        prompt_token_count,
        final_hidden_states,
        layer_caches,
        prefill_logits,
    )
}

fn materialize_prefill_logits_from_roots(
    roots: &RasterArtifactStoreRoots,
    logits_ref: &RasterActivationSequenceRef,
) -> Result<Vec<crate::shared::numerics::det_num::Act>> {
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    if width != 1 {
        bail!(
            "raster prefill logits artifact width {width}, expected 1 for {PREFILL_LOGITS_ARTIFACT_NAME}"
        );
    }
    let mut logits = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let row = read_sequence_row_from_roots(
            roots,
            RasterSequenceRowRequest {
                tensor_ref: logits_ref.clone(),
                row_idx,
            },
        )?;
        let acts = row.acts();
        let [logit] = acts.as_slice() else {
            bail!(
                "raster prefill logit row {row_idx} has width {}",
                acts.len()
            );
        };
        logits.push(*logit);
    }
    Ok(logits)
}

pub(in super::super) fn validate_layer_cache_roots(
    roots: &RasterArtifactStoreRoots,
    layer_caches: &[crate::routines::prefill_range::raster::PrefillLayerCacheSlot],
) -> Result<()> {
    for cache in layer_caches {
        if let crate::routines::prefill_range::raster::PrefillLayerCacheSlot::Ref(cache_ref) = cache {
            ensure_artifact_root_present(roots, cache_ref.keys().det_commitment())?;
            ensure_artifact_root_present(roots, cache_ref.values().det_commitment())?;
        }
    }
    Ok(())
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
