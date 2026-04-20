use anyhow::Result;
use serde_json::json;

use crate::shared::transformer::{
    ActivationSequence, Gemma4TransformerModel, LayerKvCache, TransformerDecodeState,
    TransformerPrefillResult, TransformerStateTransitionState,
};
use crate::trace::trace_event;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
) -> Result<TransformerPrefillResult> {
    trace_event("prefill.apply_final_norm");
    let normalized_hidden_states = crate::transformer_state_transition::apply_final_norm(
        &final_hidden_states.activations,
        &model.final_norm_weight,
        model.rms_norm_eps,
    )?;
    let final_position =
        crate::transformer_state_transition::select_final_position(&normalized_hidden_states.activations)?;
    trace_event("prefill.project_to_logits");
    let mut logits =
        crate::transformer_state_transition::project_to_logits(&final_position, &model.logits_projection)?;
    if let Some(softcap) = model.final_logit_softcapping {
        logits = crate::transformer_state_transition::apply_final_logit_softcapping(&logits, softcap);
    }
    let prefill_logits = crate::transformer_state_transition::extract_prefill_logits(&logits);
    crate::trace::trace_checkpoint(
        "prefill.finalize",
        &json!({
            "final_hidden_states": final_hidden_states.activations.clone(),
            "final_hidden_states_sha256": final_hidden_states.activations_sha256.clone(),
            "prefill_logits": prefill_logits.logits.clone(),
            "prefill_logits_sha256": prefill_logits.final_logits_sha256.clone(),
            "decode_position": prompt_token_ids.len(),
            "decode_token_count": prompt_token_ids.len(),
            "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
        }),
    );

    Ok(TransformerPrefillResult {
        transformer_decode_state: TransformerDecodeState {
            layer_caches,
            position: prompt_token_ids.len(),
            token_count: prompt_token_ids.len(),
        },
        transformer_state: TransformerStateTransitionState {
            activation_states: vec![final_hidden_states],
            prefill_logits,
        },
    })
}
