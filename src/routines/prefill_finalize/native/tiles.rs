use anyhow::Result;
use serde_json::json;

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{
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
    let final_position =
        crate::shared::numerics::transformer_kernels::select_final_position_internal(
            &final_hidden_states.clone_internal(),
        )?;
    trace_event("prefill.project_to_logits");
    let prefill_logits = match execution_mode {
        InferenceExecutionMode::Deterministic => {
            let hidden_state =
                crate::shared::numerics::det_kernels::row_from_internal(&final_position)?;
            let logits = crate::shared::numerics::det_kernels::det_hidden_to_logits(
                &hidden_state,
                model.final_norm_weight_det.as_deref(),
                model.rms_norm_eps_det,
                &model.logits_projection,
                model.embedding_source.as_ref(),
                model.final_logit_softcapping,
                model.final_logit_softcapping_det,
            )?;
            let det_final_logits_sha256 = Some(
                crate::shared::numerics::transformer_kernels::build_det_vector_commitment(&logits),
            );
            crate::shared::model::transformer::PrefillLogits::from_det_internal(
                crate::shared::model::transformer::InternalLogits::from_det_values_only(logits),
                det_final_logits_sha256,
            )
        }
        InferenceExecutionMode::Fp32 => {
            crate::shared::numerics::transformer_kernels::project_internal_hidden_to_prefill_logits(
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
            )?
        }
    };
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
    // Deterministic-mode payloads carry only canonical commitments (spec v1);
    // fp32 mode keeps the compatibility fields.
    let deterministic = prefill_logits.det_final_logits_sha256.is_some();
    crate::trace::trace_checkpoint_lazy("prefill.finalize", || {
        let mut payload = json!({
            "det_final_hidden_states_sha256": final_hidden_states.det_activations_sha256.clone(),
            "det_prefill_logits_sha256": prefill_logits.det_final_logits_sha256.clone(),
            "decode_position": prompt_token_count,
            "decode_token_count": prompt_token_count,
            "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
        });
        if !deterministic {
            payload["final_hidden_states"] = json!(final_hidden_states.activations.clone());
            payload["final_hidden_states_sha256"] =
                json!(final_hidden_states.activations_sha256.clone());
            payload["prefill_logits"] = json!(prefill_logits.logits.clone());
            payload["prefill_logits_sha256"] = json!(prefill_logits.final_logits_sha256.clone());
            payload["layer_caches"] = json!(crate::trace::serialize_layer_caches(&layer_caches));
        }
        payload
    });

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
