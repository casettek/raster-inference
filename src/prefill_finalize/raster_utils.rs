use anyhow::{anyhow, Result};
use serde_json::json;

use crate::shared::raster_row_store::{
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterTensorId,
};
use crate::shared::raster_transformer_kernels::RasterActivationSequence;
use crate::shared::transformer::{
    ActivationSequence, LayerKvCache, PrefillLogits, TransformerDecodeState,
    TransformerPrefillResult, TransformerStateTransitionState,
};

pub fn import_materialized_final_hidden_states(
    store: &mut AuthenticatedRasterTensorStore,
    final_hidden_states: &ActivationSequence,
) -> Result<RasterActivationSequenceRef> {
    // Compatibility bridge for public/dev callers that still hold full final
    // hidden states. Projection still enters the ref-backed tile path.
    let internal = final_hidden_states.clone_internal();
    let det_rows = internal.det_values().ok_or_else(|| {
        anyhow!("deterministic raster prefill finalize requires canonical final hidden activations")
    })?;
    store.insert_activation_sequence(
        RasterTensorId::new("prefill.finalize.final_hidden_states")?,
        RasterActivationSequence::from_acts(det_rows.to_vec()),
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
            "det_layer_caches_sha256": crate::shared::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
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
