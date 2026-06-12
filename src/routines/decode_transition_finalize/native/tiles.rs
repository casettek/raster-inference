use anyhow::Result;

use crate::routines::decode_layer_range::DecodeLayerRangeState;
use crate::shared::model::transformer::{
    Gemma4TransformerModel, TransformerDecodeState, TransformerDecodeStepResult,
};

pub(crate) fn run(
    state: DecodeLayerRangeState,
    model: &Gemma4TransformerModel,
) -> Result<TransformerDecodeStepResult> {
    let activation_state = state.activation_state();
    let hidden_state =
        crate::shared::numerics::det_kernels::row_from_internal(&state.current_activation)?;
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
    let prefill_logits = crate::shared::model::transformer::PrefillLogits::from_det_internal(
        crate::shared::model::transformer::InternalLogits::from_det_values_only(logits),
        det_final_logits_sha256,
    );
    Ok(TransformerDecodeStepResult {
        transformer_decode_state: TransformerDecodeState {
            layer_caches: state.completed_layer_caches()?,
            position: state.position + 1,
            token_count: state.token_count + 1,
        },
        activation_state,
        prefill_logits,
    })
}
