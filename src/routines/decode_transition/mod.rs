use anyhow::Result;
use serde_json::json;

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::api::output::DecodeState;
use crate::shared::artifacts::raster_artifact_store::{
    read_token_id_from_ref_roots, RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{
    Gemma4TransformerModel, InternalLogits, TransformerDecodeState, TransformerDecodeStepResult,
};
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
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

/// Refs-first raster path: consumes selected-token/cache refs and returns an
/// updated `RasterDecodeLoopState` without building `TransformerDecodeStepResult`.
pub fn run_raster(
    decode_state: RasterDecodeLoopState,
    selected_token_ref: RasterSelectedTokenRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    raster_sizing: RasterSizingControls,
) -> Result<RasterDecodeLoopState> {
    let source =
        raster::auth_source::RasterDecodeTransitionSource::for_current_integrity_mode(source)?;
    let output_source_prefix = format!("decode.transition.position_{}", decode_state.position);
    let output = raster::main_state_refs(
        raster::RasterDecodeTransitionInputRefs {
            artifact_store_roots: decode_state.artifact_store_roots,
            position: decode_state.position,
            token_count: decode_state.token_count,
            layer_caches: decode_state.layer_caches,
            selected_token_ref,
            decode_transition_source_root: source.root().to_string(),
            output_source_prefix,
            raster_sizing,
        },
        &source,
    )?;
    RasterDecodeLoopState::new(
        output.artifact_store_roots,
        decode_state.full_token_ids_ref,
        decode_state.full_token_count,
        decode_state.generated_token_ids_ref,
        decode_state.generated_token_count,
        output.logits_ref,
        output.logit_count,
        output.layer_caches,
        output.position,
        output.token_count,
        Some(output.final_hidden_state_ref),
    )
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

pub(crate) fn finalize_raster_state_for_trace(decode_state: &RasterDecodeLoopState) -> Result<()> {
    let decode_state = materialize_decode_state_from_raster_state_for_trace(decode_state)?;
    finalize(&decode_state)
}

pub(crate) fn materialize_decode_state_from_raster_state_for_trace(
    decode_state: &RasterDecodeLoopState,
) -> Result<DecodeState> {
    let full_token_ids = materialize_token_ids_from_optional_ref(
        &decode_state.artifact_store_roots,
        decode_state.full_token_ids_ref.as_ref(),
    )?;
    let generated_token_ids = materialize_token_ids_from_optional_ref(
        &decode_state.artifact_store_roots,
        decode_state.generated_token_ids_ref.as_ref(),
    )?;
    let internal_logits = materialize_internal_logits_from_ref(
        &decode_state.artifact_store_roots,
        &decode_state.current_logits_ref,
    )?;
    let layer_caches = raster::materialize_decode_layer_caches_from_roots(
        &decode_state.artifact_store_roots,
        &decode_state.layer_caches,
    )?;
    let mut materialized = DecodeState::new(
        full_token_ids,
        internal_logits.clone_f32(),
        TransformerDecodeState {
            layer_caches,
            position: decode_state.position,
            token_count: decode_state.token_count,
        },
    );
    materialized.generated_token_ids = generated_token_ids;
    materialized.set_internal_logits(internal_logits);
    Ok(materialized)
}

fn materialize_token_ids_from_optional_ref(
    roots: &crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots,
    token_ids_ref: Option<&RasterTokenIdSequenceRef>,
) -> Result<Vec<u32>> {
    let Some(token_ids_ref) = token_ids_ref else {
        return Ok(Vec::new());
    };
    (0..token_ids_ref.token_count())
        .map(|token_idx| read_token_id_from_ref_roots(roots, token_ids_ref, token_idx))
        .collect()
}

fn materialize_internal_logits_from_ref(
    roots: &crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots,
    logits_ref: &RasterActivationSequenceRef,
) -> Result<InternalLogits> {
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    let det_logits = match (row_count, width) {
        (_, 1) => (0..row_count)
            .map(|row_idx| {
                let row = read_sequence_row_from_roots(
                    roots,
                    RasterSequenceRowRequest {
                        tensor_ref: logits_ref.clone(),
                        row_idx,
                    },
                )?;
                row.acts()
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("raster logits row {row_idx} is empty"))
            })
            .collect::<Result<Vec<_>>>()?,
        (1, _) => read_sequence_row_from_roots(
            roots,
            RasterSequenceRowRequest {
                tensor_ref: logits_ref.clone(),
                row_idx: 0,
            },
        )?
        .acts(),
        _ => anyhow::bail!("raster logits shape {row_count}x{width} must be Nx1 or 1xN"),
    };
    Ok(InternalLogits::from_det_values(det_logits))
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
