use anyhow::Result;
use serde_json::json;

use crate::shared::input::InferenceExecutionMode;
use crate::shared::transformer::{
    ActivationSequence, Gemma4TransformerModel, LayerKvCache, PrefillLogits,
    TransformerDecodeState, TransformerPrefillResult, TransformerStateTransitionState,
};
use crate::trace::trace_event;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<TransformerPrefillResult> {
    trace_event("prefill.select_final_position");
    let final_position = crate::shared::transformer_kernels::select_final_position_internal(
        &final_hidden_states.clone_internal(),
    )?;
    trace_event("prefill.project_to_logits");
    let prefill_logits =
        crate::shared::transformer_kernels::project_internal_hidden_to_prefill_logits(
            final_position,
            &model.final_norm_weight,
            model.final_norm_weight_det.as_deref(),
            model.rms_norm_eps,
            model.rms_norm_eps_det,
            &model.logits_projection,
            model.embedding_source.as_ref(),
            execution_mode,
            model.final_logit_softcapping,
            model.final_logit_softcapping_det,
        )?;
    build_prefill_result(
        prompt_token_ids.len(),
        final_hidden_states,
        layer_caches,
        prefill_logits,
    )
}

pub(crate) fn build_prefill_result(
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
