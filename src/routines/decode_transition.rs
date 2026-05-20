use anyhow::Result;
use serde_json::json;

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::api::output::DecodeState;
use crate::shared::model::transformer::{
    Gemma4TransformerModel, TransformerDecodeState, TransformerDecodeStepResult,
};
use crate::RasterSizingControls;

pub mod native;
pub mod raster;

use self::raster::auth_source::AuthenticatedGemmaDecodeTransitionSource;

pub fn run(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
) -> Result<TransformerDecodeStepResult> {
    run_with_mode(
        transformer_decode_state,
        next_token,
        model,
        InferenceExecutionMode::Fp32,
    )
}

pub fn run_with_mode(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<TransformerDecodeStepResult> {
    model.validate_execution_mode(execution_mode)?;
    let TransformerDecodeState {
        layer_caches,
        position,
        token_count,
    } = transformer_decode_state;
    let embedded_token = if let Some(ref embedding_table) = model.embedding_table {
        crate::shared::numerics::transformer_kernels::embed_input_tokens_with_mode(
            &[next_token],
            embedding_table,
            execution_mode,
        )?
    } else if let Some(ref embedding_source) = model.embedding_source {
        crate::io::embed_input_tokens_from_gemma_source_with_mode(
            &[next_token],
            embedding_source,
            execution_mode,
        )?
    } else {
        anyhow::bail!(
            "transformer state model is missing both embedding_table and embedding_source"
        )
    };
    let final_hidden_state = match execution_mode {
        InferenceExecutionMode::Fp32 => {
            let embedded_token = embedded_token.activations.first().ok_or_else(|| {
                anyhow::anyhow!("transformer embedding returned no activation rows")
            })?;
            native::run_text_layers_decode_step(
                embedded_token,
                next_token,
                model,
                layer_caches,
                position,
            )?
        }
        InferenceExecutionMode::Deterministic => {
            let embedded_token = embedded_token.clone_internal().last_row().ok_or_else(|| {
                anyhow::anyhow!("transformer embedding returned no activation rows")
            })?;
            native::deterministic_tiles::run_text_layers_decode_step_internal(
                embedded_token,
                next_token,
                model,
                layer_caches,
                position,
            )?
        }
    };
    let final_position =
        crate::shared::numerics::transformer_kernels::select_final_position_internal(
            &final_hidden_state.activation_state.clone_internal(),
        )?;
    let prefill_logits =
        crate::shared::numerics::transformer_kernels::project_internal_decode_hidden_to_logits(
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

    Ok(TransformerDecodeStepResult {
        transformer_decode_state: TransformerDecodeState {
            layer_caches: final_hidden_state.layer_caches,
            position: position + 1,
            token_count: token_count + 1,
        },
        activation_state: final_hidden_state.activation_state,
        prefill_logits,
    })
}

pub fn run_raster(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    raster_sizing: RasterSizingControls,
) -> Result<TransformerDecodeStepResult> {
    raster::run(transformer_decode_state, next_token, source, raster_sizing)
}

pub fn run_raster_with_roots(
    input_roots: raster::RasterDecodeTransitionInputRoots,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<raster::RasterDecodeTransitionOutputRefs> {
    raster::main(input_roots, source)
}

pub fn finalize(decode_state: &DecodeState) -> Result<()> {
    crate::trace::trace_checkpoint(
        "decode.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": generated_token_ids_commitment(decode_state)?,
            "current_logits": decode_state.current_logits.clone(),
            "current_logits_sha256": current_logits_commitment(decode_state),
            "det_current_logits_sha256": current_det_logits_commitment(decode_state),
            "decode_position": decode_state.transformer_decode_state.position,
            "decode_token_count": decode_state.transformer_decode_state.token_count,
            "layer_caches": crate::trace::serialize_layer_caches(&decode_state.transformer_decode_state.layer_caches),
        }),
    );
    Ok(())
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
    let logits = decode_state.clone_internal_logits();
    logits
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_vector_commitment)
}

#[cfg(test)]
mod tests;
