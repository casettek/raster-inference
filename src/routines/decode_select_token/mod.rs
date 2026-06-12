use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::runtime::checkpoints::RoutineId;
use crate::shared::api::output::DecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
#[cfg(test)]
use crate::shared::artifacts::raster_artifact_store::read_token_id_from_ref_roots;
#[cfg(test)]
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
#[cfg(test)]
use crate::shared::artifacts::raster_artifact_store::RasterTokenIdSequenceRef;
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::RasterSizingControls;

pub mod native;
pub mod raster;

pub fn run(decode_state: &mut DecodeState, max_new_tokens: usize) -> Result<Option<u32>> {
    let _routine = crate::trace::routine_scope(
        RoutineId::SelectOutputToken,
        format!(
            "generated_tokens={}",
            decode_state.generated_token_ids.len()
        ),
    );
    if native::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens)
        .is_some()
    {
        return Ok(None);
    }

    let logits = decode_state.clone_internal_logits();
    let next_token = native::select_next_token_internal(&logits)?;
    decode_state.full_token_ids = native::append_token(&decode_state.full_token_ids, next_token);
    decode_state.generated_token_ids =
        native::append_token(&decode_state.generated_token_ids, next_token);
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &decode_select_checkpoint_state(decode_state, next_token, max_new_tokens)?,
    );
    Ok(Some(next_token))
}

#[cfg(test)]
pub(crate) fn materialize_run_raster_for_api(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
) -> Result<Option<u32>> {
    Ok(run_raster_refs_for_api(decode_state, max_new_tokens)?.map(|output| output.next_token))
}

#[cfg(test)]
pub(crate) fn run_raster_refs_for_api(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
) -> Result<Option<raster::RasterDecodeSelectOutputRefs>> {
    run_raster_refs_with_roots_for_api(
        decode_state,
        max_new_tokens,
        ArtifactIo::export_store_roots(),
    )
}

#[cfg(test)]
pub(crate) fn run_raster_refs_with_roots_for_api(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
    artifact_store_roots: RasterArtifactStoreRoots,
) -> Result<Option<raster::RasterDecodeSelectOutputRefs>> {
    if raster::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens)
        .is_some()
    {
        return Ok(None);
    }

    let input_roots = prepare_raster_decode_select_input_roots(artifact_store_roots, decode_state)?;
    let output = raster::main(input_roots)?;

    decode_state.full_token_ids =
        materialize_token_ids(&output.artifact_store_roots, &output.full_token_ids_ref)?;
    decode_state.generated_token_ids = materialize_token_ids(
        &output.artifact_store_roots,
        &output.generated_token_ids_ref,
    )?;
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &decode_select_checkpoint_state(decode_state, output.next_token, max_new_tokens)?,
    );
    Ok(Some(output))
}

/// Refs-first raster path: advances token refs in `RasterDecodeLoopState`
/// without materializing or mutating host `DecodeState`.
pub fn run_raster(
    decode_state: RasterDecodeLoopState,
    max_new_tokens: usize,
) -> Result<(
    RasterDecodeLoopState,
    Option<raster::RasterDecodeSelectOutputRefs>,
)> {
    run_raster_with_decode_select_sizing(
        decode_state,
        max_new_tokens,
        DecodeSelectRasterSizing::default(),
    )
}

pub(crate) fn run_raster_with_sizing(
    decode_state: RasterDecodeLoopState,
    max_new_tokens: usize,
    raster_sizing: RasterSizingControls,
) -> Result<(
    RasterDecodeLoopState,
    Option<raster::RasterDecodeSelectOutputRefs>,
)> {
    run_raster_with_decode_select_sizing(
        decode_state,
        max_new_tokens,
        DecodeSelectRasterSizing::from(raster_sizing),
    )
}

fn run_raster_with_decode_select_sizing(
    decode_state: RasterDecodeLoopState,
    max_new_tokens: usize,
    raster_sizing: DecodeSelectRasterSizing,
) -> Result<(
    RasterDecodeLoopState,
    Option<raster::RasterDecodeSelectOutputRefs>,
)> {
    let _routine = crate::trace::routine_scope(
        RoutineId::SelectOutputToken,
        format!(
            "mode=raster generated_tokens={}",
            decode_state.generated_token_count
        ),
    );
    if raster::check_stop_condition(decode_state.generated_token_count, max_new_tokens).is_some() {
        return Ok((decode_state, None));
    }

    let input_roots =
        prepare_raster_decode_select_input_roots_from_state(decode_state.clone(), raster_sizing);
    let output = raster::main(input_roots)?;
    let next_state = RasterDecodeLoopState::new(
        output.artifact_store_roots.clone(),
        Some(output.full_token_ids_ref.clone()),
        output.full_token_ids_ref.token_count(),
        Some(output.generated_token_ids_ref.clone()),
        output.generated_token_ids_ref.token_count(),
        output.logits_ref.clone(),
        output.logit_count,
        decode_state.layer_caches,
        decode_state.position,
        decode_state.token_count,
        decode_state.activation_state_ref,
    )?;
    Ok((next_state, Some(output)))
}

pub(crate) fn run_selected_raster_detour_from_native_boundary(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
    raster_sizing: RasterSizingControls,
) -> Result<u32> {
    let original_current_logits = decode_state.current_logits.clone();
    let raster_state = prepare_raster_decode_loop_state_from_native(decode_state)?;
    let (next_state, output) = run_raster_with_sizing(raster_state, max_new_tokens, raster_sizing)?;
    let output = output.context(
        "selective raster decode.select_token detour reached stop condition unexpectedly",
    )?;

    let mut materialized =
        crate::routines::decode_layer_range::materialize_decode_state_from_raster_state_for_trace(
            &next_state,
        )?;
    materialized.current_logits = original_current_logits;
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &decode_select_checkpoint_state(&materialized, output.next_token, max_new_tokens)?,
    );
    *decode_state = materialized;
    Ok(output.next_token)
}

pub(crate) fn trace_raster_checkpoint_from_state(
    decode_state: &RasterDecodeLoopState,
    selected_next_token: u32,
    max_new_tokens: usize,
) -> Result<()> {
    let decode_state =
        crate::routines::decode_layer_range::materialize_decode_state_from_raster_state_for_trace(
            decode_state,
        )?;
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &decode_select_checkpoint_state(&decode_state, selected_next_token, max_new_tokens)?,
    );
    Ok(())
}

fn prepare_raster_decode_select_input_roots_from_state(
    decode_state: RasterDecodeLoopState,
    raster_sizing: DecodeSelectRasterSizing,
) -> raster::RasterDecodeSelectInputRoots {
    let source_prefix = format!(
        "decode.select_token.position_{}.step_{}",
        decode_state.position, decode_state.generated_token_count
    );
    raster::RasterDecodeSelectInputRoots {
        artifact_store_roots: decode_state.artifact_store_roots,
        logits_ref: decode_state.current_logits_ref,
        full_token_ids_ref: decode_state.full_token_ids_ref,
        full_token_count: decode_state.full_token_count,
        generated_token_ids_ref: decode_state.generated_token_ids_ref,
        generated_token_count: decode_state.generated_token_count,
        logits_per_tile: raster_sizing.logits_per_tile,
        token_ids_per_tile: raster_sizing.token_ids_per_tile,
        output_full_token_ids_source_name: format!("{source_prefix}.output.full_token_ids"),
        output_generated_token_ids_source_name: format!(
            "{source_prefix}.output.generated_token_ids"
        ),
        output_selected_token_source_name: format!("{source_prefix}.output.selected_token"),
    }
}

#[derive(Clone, Copy)]
struct DecodeSelectRasterSizing {
    logits_per_tile: usize,
    token_ids_per_tile: usize,
}

impl Default for DecodeSelectRasterSizing {
    fn default() -> Self {
        Self {
            logits_per_tile: raster::DEFAULT_DECODE_SELECT_LOGITS_PER_TILE,
            token_ids_per_tile: raster::DEFAULT_DECODE_SELECT_TOKEN_IDS_PER_TILE,
        }
    }
}

impl From<RasterSizingControls> for DecodeSelectRasterSizing {
    fn from(raster_sizing: RasterSizingControls) -> Self {
        Self {
            logits_per_tile: raster_sizing.sequence_rows_per_tile,
            token_ids_per_tile: raster_sizing.sequence_rows_per_tile,
        }
    }
}

fn prepare_raster_decode_loop_state_from_native(
    decode_state: &DecodeState,
) -> Result<RasterDecodeLoopState> {
    let position = decode_state.transformer_decode_state.position;
    let generated_count = decode_state.generated_token_ids.len();
    let source_prefix =
        format!("decode.select_token.detour.position_{position}.step_{generated_count}");
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let (artifact_store_roots, full_token_ids_ref) =
        crate::routines::decode_layer_range::insert_decode_token_ids_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.full_token_ids"),
            &decode_state.full_token_ids,
        )?;
    let (artifact_store_roots, generated_token_ids_ref) =
        crate::routines::decode_layer_range::insert_decode_token_ids_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.generated_token_ids"),
            &decode_state.generated_token_ids,
        )?;
    let (artifact_store_roots, logits_ref, logit_count) =
        crate::routines::decode_layer_range::insert_decode_logits_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.logits"),
            decode_state,
            "raster decode select token",
        )?;
    let (artifact_store_roots, layer_caches) =
        crate::routines::decode_layer_range::insert_decode_layer_cache_refs_from_native(
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

#[cfg(test)]
fn prepare_raster_decode_select_input_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    decode_state: &DecodeState,
) -> Result<raster::RasterDecodeSelectInputRoots> {
    let position = decode_state.transformer_decode_state.position;
    let generated_count = decode_state.generated_token_ids.len();
    let source_prefix = format!("decode.select_token.position_{position}.step_{generated_count}");
    let (artifact_store_roots, full_token_ids_ref) =
        crate::routines::decode_layer_range::insert_decode_token_ids_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.full_token_ids"),
            &decode_state.full_token_ids,
        )?;
    let (artifact_store_roots, generated_token_ids_ref) =
        crate::routines::decode_layer_range::insert_decode_token_ids_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.generated_token_ids"),
            &decode_state.generated_token_ids,
        )?;
    let (artifact_store_roots, logits_ref, _) =
        crate::routines::decode_layer_range::insert_decode_logits_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.logits"),
            decode_state,
            "raster decode select token",
        )?;

    Ok(raster::RasterDecodeSelectInputRoots {
        artifact_store_roots,
        logits_ref,
        full_token_ids_ref,
        full_token_count: decode_state.full_token_ids.len(),
        generated_token_ids_ref,
        generated_token_count: decode_state.generated_token_ids.len(),
        logits_per_tile: raster::DEFAULT_DECODE_SELECT_LOGITS_PER_TILE,
        token_ids_per_tile: raster::DEFAULT_DECODE_SELECT_TOKEN_IDS_PER_TILE,
        output_full_token_ids_source_name: format!("{source_prefix}.output.full_token_ids"),
        output_generated_token_ids_source_name: format!(
            "{source_prefix}.output.generated_token_ids"
        ),
        output_selected_token_source_name: format!("{source_prefix}.output.selected_token"),
    })
}

#[cfg(test)]
fn materialize_token_ids(
    artifact_store_roots: &RasterArtifactStoreRoots,
    token_ids_ref: &RasterTokenIdSequenceRef,
) -> Result<Vec<u32>> {
    (0..token_ids_ref.token_count())
        .map(|token_idx| {
            read_token_id_from_ref_roots(artifact_store_roots, token_ids_ref, token_idx)
        })
        .collect()
}

fn decode_select_checkpoint_state(
    decode_state: &DecodeState,
    selected_next_token: u32,
    max_new_tokens: usize,
) -> Result<Value> {
    // Deterministic-mode payloads carry only canonical commitments (spec v1);
    // fp32 mode keeps the compatibility fields.
    let deterministic = decode_state.clone_internal_logits().det_values().is_some();
    let mut payload = json!({
        "full_token_ids": decode_state.full_token_ids.clone(),
        "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
        "generated_token_ids": decode_state.generated_token_ids.clone(),
        "generated_token_ids_sha256": crate::routines::output_finalize::native::build_output_decode_commitment(&decode_state.generated_token_ids)?,
        "det_current_logits_sha256": current_det_logits_commitment(decode_state),
        "selected_next_token": selected_next_token,
        "decode_position": decode_state.transformer_decode_state.position,
        "decode_token_count": decode_state.transformer_decode_state.token_count,
        "max_new_tokens": max_new_tokens,
    });
    if !deterministic {
        payload["current_logits"] = json!(decode_state.current_logits.clone());
        payload["current_logits_sha256"] =
            json!(crate::trace::sha256_hex(&decode_state.current_logits));
        payload["layer_caches"] = json!(crate::trace::serialize_layer_caches(
            &decode_state.transformer_decode_state.layer_caches
        ));
    }
    Ok(payload)
}

fn current_det_logits_commitment(decode_state: &DecodeState) -> Option<String> {
    let logits = decode_state.clone_internal_logits();
    logits
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_vector_commitment)
}

#[cfg(test)]
mod tests;
