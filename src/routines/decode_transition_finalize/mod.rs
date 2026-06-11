use anyhow::Result;
use serde_json::json;

use crate::runtime::checkpoints::RoutineId;
use crate::shared::api::output::DecodeState;
use crate::shared::model::transformer::TransformerDecodeStepResult;
use crate::RasterSizingControls;

pub mod native;
pub mod raster;

use self::raster::auth_source::AuthenticatedGemmaDecodeTransitionSource;

pub fn run_raster(
    prior_decode_state: crate::shared::raster_contracts::pipeline::RasterDecodeLoopState,
    completed_range_state: raster::RasterDecodeTransitionFinalizeInput,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<crate::shared::raster_contracts::pipeline::RasterDecodeLoopState> {
    let source =
        raster::auth_source::RasterDecodeTransitionSource::for_current_integrity_mode(source)?;
    raster::run_raster(prior_decode_state, completed_range_state, &source)
}

pub(crate) fn run_selected_raster_detour_from_native_boundary(
    decode_state: &DecodeState,
    completed_range_state: crate::decode_layer_range::DecodeLayerRangeState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    raster_sizing: RasterSizingControls,
) -> Result<TransformerDecodeStepResult> {
    let source_prefix = format!(
        "decode.transition_finalize.detour.position_{}",
        completed_range_state.position
    );
    let prior_raster_state =
        crate::decode_layer_range::prepare_raster_decode_loop_state_from_native(
            decode_state,
            &source_prefix,
        )?;
    let completed_range_state = crate::decode_layer_range::raster_state_from_native_state(
        completed_range_state,
        raster_sizing,
    )?;
    let raster_state = run_raster(prior_raster_state, completed_range_state, source)?;
    let activation_ref = raster_state.activation_state_ref.as_ref().ok_or_else(|| {
        anyhow::anyhow!("decode transition finalize detour requires activation state ref")
    })?;
    let activation_state =
        crate::decode_layer_range::raster::materialize_activation_sequence_from_ref(
            &raster_state.artifact_store_roots,
            activation_ref,
        )?;
    let materialized =
        crate::decode_layer_range::materialize_decode_state_from_raster_state_for_trace(
            &raster_state,
        )?;
    let prefill_logits = crate::decode_layer_range::prefill_logits_from_internal(
        materialized.clone_internal_logits(),
    );

    Ok(TransformerDecodeStepResult {
        transformer_decode_state: materialized.transformer_decode_state,
        activation_state,
        prefill_logits,
    })
}

pub fn trace_checkpoint(decode_state: &DecodeState) -> Result<()> {
    let _routine = crate::trace::routine_scope(
        RoutineId::DecodeTransitionFinalize,
        format!(
            "position={} token_count={}",
            decode_state.transformer_decode_state.position,
            decode_state.transformer_decode_state.token_count
        ),
    );
    // Deterministic-mode payloads carry only canonical commitments (spec v1);
    // fp32 mode keeps the compatibility fields.
    let deterministic = decode_state.clone_internal_logits().det_values().is_some();
    crate::trace::trace_checkpoint_lazy_result("decode.transition_finalize", || {
        let mut payload = json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": generated_token_ids_commitment(decode_state)?,
            "det_current_logits_sha256": current_det_logits_commitment(decode_state),
            "decode_position": decode_state.transformer_decode_state.position,
            "decode_token_count": decode_state.transformer_decode_state.token_count,
        });
        if !deterministic {
            payload["current_logits"] = json!(decode_state.current_logits.clone());
            payload["current_logits_sha256"] = json!(current_logits_commitment(decode_state));
            payload["layer_caches"] = json!(crate::trace::serialize_layer_caches(
                &decode_state.transformer_decode_state.layer_caches
            ));
        }
        Ok(payload)
    })?;
    Ok(())
}

pub(crate) fn finalize_raster_state_for_trace(
    decode_state: &crate::shared::raster_contracts::pipeline::RasterDecodeLoopState,
) -> Result<()> {
    let decode_state =
        crate::decode_layer_range::materialize_decode_state_from_raster_state_for_trace(
            decode_state,
        )?;
    trace_checkpoint(&decode_state)
}

fn generated_token_ids_commitment(decode_state: &DecodeState) -> Result<String> {
    crate::output_finalize::native::build_output_decode_commitment(
        &decode_state.generated_token_ids,
    )
}

fn current_logits_commitment(decode_state: &DecodeState) -> String {
    let logits = decode_state.clone_internal_logits();
    crate::shared::numerics::transformer_kernels::build_vector_commitment(logits.as_f32_slice())
}

fn current_det_logits_commitment(decode_state: &DecodeState) -> Option<String> {
    decode_state
        .clone_internal_logits()
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_vector_commitment)
}
