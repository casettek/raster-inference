use anyhow::{bail, Result};
use serde_json::json;

use crate::prefill_finalize::raster::{RasterPrefillFinalizeRefs, PREFILL_LOGITS_ARTIFACT_NAME};
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{
    ActivationSequence, InternalLogits, LayerKvCache, PrefillLogits, TransformerDecodeState,
    TransformerPrefillResult, TransformerStateTransitionState,
};
use crate::shared::tensors::raster_row_store::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
};

pub fn build_prefill_result_from_root_refs(
    roots: &RasterArtifactStoreRoots,
    refs: &RasterPrefillFinalizeRefs,
) -> Result<TransformerPrefillResult> {
    let layer_refs = crate::prefill_layer::raster::PrefillLayerOutputRefs {
        final_hidden_states_ref: refs.final_hidden_states_ref.clone(),
        layer_caches: refs.layer_caches.clone(),
    };
    let (final_hidden_states, layer_caches) =
        crate::prefill_layer::materialize_prefill_layer_output_refs_from_roots(roots, &layer_refs)?;
    let det_logits = materialize_prefill_logits_from_roots(roots, &refs.logits_ref)?;
    let internal_logits = InternalLogits::from_det_values(det_logits.clone());
    let final_logits_sha256 = crate::shared::numerics::transformer_kernels::build_vector_commitment(
        internal_logits.as_f32_slice(),
    );
    let mut prefill_logits = PrefillLogits::from_internal(internal_logits, final_logits_sha256);
    prefill_logits.det_final_logits_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_vector_commitment(&det_logits),
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
    crate::trace::trace_checkpoint(
        "prefill.finalize",
        &json!({
            "final_hidden_states": final_hidden_states.activations.clone(),
            "final_hidden_states_sha256": final_hidden_states.activations_sha256.clone(),
            "det_final_hidden_states_sha256": final_hidden_states.det_activations_sha256.clone(),
            "prefill_logits": prefill_logits.logits.clone(),
            "prefill_logits_sha256": prefill_logits.final_logits_sha256.clone(),
            "det_prefill_logits_sha256": prefill_logits.det_final_logits_sha256.clone(),
            "decode_position": prompt_token_count,
            "decode_token_count": prompt_token_count,
            "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
            "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
        }),
    );

    Ok(TransformerPrefillResult {
        transformer_decode_state: TransformerDecodeState {
            layer_caches,
            position: prompt_token_count,
            token_count: prompt_token_count,
        },
        transformer_state: TransformerStateTransitionState {
            activation_states: vec![final_hidden_states],
            prefill_logits,
        },
    })
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
