use anyhow::{Context, Result};
use serde_json::json;

use crate::runtime::checkpoints::RoutineId;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::api::output::DecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, read_token_id_from_ref_roots, token_id_leaf,
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4TransformerModel, InternalActivationSequence, InternalLogits,
    LayerKvCache, PrefillLogits, TransformerDecodeState, TransformerDecodeStepResult,
};
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::raster_kernels::transformer::RasterActivationRow;
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, read_sequence_row_from_roots,
    RasterActivationSequenceRef, RasterSequenceRowRequest, RasterTensorId,
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
    let _routine = crate::trace::routine_scope(
        RoutineId::DecodeTransition,
        format!("position={position} token_count={token_count}"),
    );
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
    let _routine = crate::trace::routine_scope(
        RoutineId::DecodeTransition,
        format!("mode=raster position={}", decode_state.position),
    );
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

pub(crate) fn run_selected_raster_detour_from_native_boundary(
    decode_state: &DecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
    raster_sizing: RasterSizingControls,
) -> Result<TransformerDecodeStepResult> {
    let position = decode_state.transformer_decode_state.position;
    let generated_count = decode_state.generated_token_ids.len();
    let source_prefix =
        format!("decode.transition.detour.position_{position}.step_{generated_count}");
    let mut raster_state =
        prepare_raster_decode_loop_state_from_native(decode_state, &source_prefix)?;
    let (artifact_store_roots, selected_token_ref) =
        raster::insert_decode_selected_token_with_roots(
            &raster_state.artifact_store_roots,
            format!("{source_prefix}.input.selected_token"),
            next_token,
        )?;
    raster_state.artifact_store_roots = artifact_store_roots;

    let source = AuthenticatedGemmaDecodeTransitionSource::from_model(
        format!("decode.transition.position_{position}"),
        model,
    )?;
    let raster_state = run_raster(raster_state, selected_token_ref, &source, raster_sizing)?;
    let activation_ref = raster_state
        .activation_state_ref
        .as_ref()
        .context("selective raster decode.transition detour requires activation state ref")?;
    let activation_state = materialize_activation_sequence_from_ref(
        &raster_state.artifact_store_roots,
        activation_ref,
    )?;
    let materialized = materialize_decode_state_from_raster_state_for_trace(&raster_state)?;
    let prefill_logits = prefill_logits_from_internal(materialized.clone_internal_logits());

    Ok(TransformerDecodeStepResult {
        transformer_decode_state: materialized.transformer_decode_state,
        activation_state,
        prefill_logits,
    })
}

fn prepare_raster_decode_loop_state_from_native(
    decode_state: &DecodeState,
    source_prefix: &str,
) -> Result<RasterDecodeLoopState> {
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let (artifact_store_roots, full_token_ids_ref) = insert_decode_token_ids_artifact_with_roots(
        artifact_store_roots,
        format!("{source_prefix}.input.full_token_ids"),
        &decode_state.full_token_ids,
    )?;
    let (artifact_store_roots, generated_token_ids_ref) =
        insert_decode_token_ids_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.generated_token_ids"),
            &decode_state.generated_token_ids,
        )?;
    let (artifact_store_roots, logits_ref, logit_count) = insert_decode_logits_artifact_with_roots(
        artifact_store_roots,
        format!("{source_prefix}.input.logits"),
        decode_state,
        "raster decode transition",
    )?;
    let (artifact_store_roots, layer_caches) = insert_decode_layer_cache_refs_from_native(
        artifact_store_roots,
        &decode_state.transformer_decode_state.layer_caches,
        &format!("{source_prefix}.input.layer_cache"),
    )?;

    RasterDecodeLoopState::new(
        artifact_store_roots,
        full_token_ids_ref,
        decode_state.full_token_ids.len(),
        generated_token_ids_ref,
        decode_state.generated_token_ids.len(),
        logits_ref,
        logit_count,
        layer_caches,
        decode_state.transformer_decode_state.position,
        decode_state.transformer_decode_state.token_count,
        None,
    )
}

pub(crate) fn insert_decode_token_ids_artifact_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    source_name: String,
    token_ids: &[u32],
) -> Result<(RasterArtifactStoreRoots, Option<RasterTokenIdSequenceRef>)> {
    if token_ids.is_empty() {
        return Ok((artifact_store_roots, None));
    }
    let leaves = token_ids.iter().copied().map(token_id_leaf).collect();
    let (artifact_store_roots, token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?;
    Ok((
        artifact_store_roots,
        Some(RasterTokenIdSequenceRef::new(token_ids_ref)?),
    ))
}

pub(crate) fn insert_decode_logits_artifact_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    source_name: String,
    decode_state: &DecodeState,
    label: &str,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef, usize)> {
    let logits = decode_state.clone_internal_logits();
    let det_logits = logits
        .det_values()
        .ok_or_else(|| anyhow::anyhow!("{label} requires canonical deterministic logits"))?;
    if det_logits.is_empty() {
        anyhow::bail!("{label} requires at least one canonical logit");
    }

    let leaves = det_logits
        .iter()
        .map(|logit| activation_row_leaf(&RasterActivationRow::from_acts(vec![*logit])))
        .collect::<Vec<_>>();
    let (artifact_store_roots, logits_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(source_name.clone())?,
        RasterArtifactMetadata::activation_rows(det_logits.len(), 1)?,
        leaves,
    )?;
    let logits_ref = activation_sequence_ref_from_artifact(
        RasterTensorId::new(source_name)?,
        RasterActivationSequenceArtifactRef::new(logits_artifact_ref)?,
    )?;
    Ok((artifact_store_roots, logits_ref, det_logits.len()))
}

pub(crate) fn insert_decode_layer_cache_refs_from_native(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_caches: &[LayerKvCache],
    source_name_prefix: &str,
) -> Result<(RasterArtifactStoreRoots, Vec<raster::DecodeLayerCacheSlot>)> {
    let mut artifact_store_roots = artifact_store_roots;
    let mut raster_layer_caches = Vec::with_capacity(layer_caches.len());
    for (layer_idx, cache) in layer_caches.iter().enumerate() {
        let raster_cache = raster::raster_cache_from_layer_cache(cache)?;
        let (next_roots, cache_slot) = raster::register_decode_layer_cache_with_roots(
            &artifact_store_roots,
            source_name_prefix,
            layer_idx,
            raster_cache,
        )?;
        artifact_store_roots = next_roots;
        raster_layer_caches.push(cache_slot);
    }
    Ok((artifact_store_roots, raster_layer_caches))
}

pub fn finalize(decode_state: &DecodeState) -> Result<()> {
    crate::trace::trace_checkpoint(
        "decode.transition",
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

pub(crate) fn materialize_activation_sequence_from_ref(
    roots: &RasterArtifactStoreRoots,
    activation_ref: &RasterActivationSequenceRef,
) -> Result<ActivationSequence> {
    let (row_count, _) = activation_ref.tensor_ref().shape().sequence_metadata()?;
    let rows = (0..row_count)
        .map(|row_idx| {
            read_sequence_row_from_roots(
                roots,
                RasterSequenceRowRequest {
                    tensor_ref: activation_ref.clone(),
                    row_idx,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let det_rows = rows.iter().map(|row| row.acts()).collect::<Vec<_>>();
    let values = rows
        .iter()
        .map(|row| row.to_f32_values())
        .collect::<Vec<_>>();
    let mut activation_sequence = ActivationSequence::from_internal(
        InternalActivationSequence::from_det_values(det_rows.clone()),
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&values),
    );
    activation_sequence.det_activations_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_activation_commitment(&det_rows),
    );
    Ok(activation_sequence)
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

fn prefill_logits_from_internal(internal_logits: InternalLogits) -> PrefillLogits {
    let final_logits_sha256 = crate::shared::numerics::transformer_kernels::build_vector_commitment(
        internal_logits.as_f32_slice(),
    );
    let det_final_logits_sha256 = internal_logits
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_vector_commitment);
    let mut prefill_logits = PrefillLogits::from_internal(internal_logits, final_logits_sha256);
    prefill_logits.det_final_logits_sha256 = det_final_logits_sha256;
    prefill_logits
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
