use anyhow::{anyhow, bail, Result};

use crate::dsl::prelude::{auth_read, call_recur_tile, call_seq, call_tile, sequence, tile};
use crate::routines::prefill_finalize::raster::auth_source::{
    GemmaPrefillFinalizeMetadataRequest, GemmaPrefillFinalizeNormWeightsRequest,
    GemmaPrefillFinalizeProjectionRowRequest, GemmaPrefillFinalizeScalarsRequest,
    RasterPrefillFinalizeSource,
};
use crate::shared::artifacts::raster_artifact_store::{RasterArtifactId, RasterArtifactStoreRoots};
use crate::shared::model::transformer::TransformerPrefillResult;
use crate::shared::numerics::det_num::{softcap_act, Act};
use crate::shared::raster_kernels::transformer::{
    project_row_with_weights, rms_norm_sequence, validate_projection_rows_per_tile,
    RasterActivationRow, RasterActivationSequence,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    append_sequence_row_by_source_name_with_roots,
    finalize_sequence_builder_by_source_name_with_roots, read_sequence_row_from_roots,
    start_sequence_builder_with_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
    RasterTensorId,
};

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    input_roots: RasterPrefillFinalizeInputRoots,
    finalize_source: &RasterPrefillFinalizeSource<'_>,
) -> Result<RasterPrefillFinalizeOutput> {
    crate::trace::trace_event("prefill.select_final_position");
    let finalize_state = call_tile!(
        init_prefill_finalize_state,
        input_roots.artifact_store_roots,
        input_roots.prompt_token_count,
        input_roots.finalize_source_root,
        input_roots.final_hidden_states_ref,
        &input_roots.layer_caches,
        input_roots.projection_rows_per_tile,
        finalize_source
    )?;
    let finalize_state = call_tile!(
        normalize_final_position_to_artifact,
        finalize_state,
        finalize_source
    )?;
    crate::trace::trace_event("prefill.project_to_logits");
    let finalize_state = call_recur_tile!(
        project_next_prefill_logit_chunk,
        finalize_state,
        finalize_source
    )?;
    let (artifact_store_roots, refs) = call_tile!(
        finalize_prefill_finalize_refs,
        finalize_state,
        input_roots.layer_caches
    )?;
    Ok(RasterPrefillFinalizeOutput::new(artifact_store_roots, refs))
}

#[sequence]
pub fn materialize_prefill_result_for_api(
    input_roots: RasterPrefillFinalizeInputRoots,
) -> Result<TransformerPrefillResult> {
    let finalize_source =
        RasterPrefillFinalizeSource::from_committed_root(&input_roots.finalize_source_root)?;
    let output = call_seq!(main, input_roots, &finalize_source)?;
    call_tile!(
        build_prefill_result_from_refs,
        output.artifact_store_roots,
        output.refs
    )
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
pub fn init_prefill_finalize_state(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
    finalize_source_root: String,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: &[crate::routines::prefill_range::raster::PrefillLayerCacheSlot],
    projection_rows_per_tile: usize,
    finalize_source: &RasterPrefillFinalizeSource<'_>,
) -> Result<PrefillFinalizeRasterState> {
    validate_projection_rows_per_tile(projection_rows_per_tile)?;
    if prompt_token_count == 0 {
        bail!("raster prefill finalize requires at least one prompt token");
    }
    let (row_count, hidden_width) = final_hidden_states_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    if row_count == 0 {
        bail!("transformer final-position selection requires at least one activation row");
    }
    ensure_artifact_root_present(
        &artifact_store_roots,
        final_hidden_states_ref.tensor_ref().det_commitment(),
    )?;
    validate_layer_cache_roots(&artifact_store_roots, layer_caches)?;

    if finalize_source_root != finalize_source.root() {
        bail!(
            "raster prefill finalize source root {} does not match input source root {}",
            finalize_source.root(),
            finalize_source_root
        );
    }
    let metadata = auth_read!(finalize_source, GemmaPrefillFinalizeMetadataRequest)?;
    if metadata.projection_rows == 0 {
        bail!("deterministic logits projection requires at least one projection row");
    }
    if hidden_width != metadata.hidden_width {
        bail!(
            "deterministic final logits projection input has width {}, expected {}",
            hidden_width,
            metadata.hidden_width,
        );
    }
    if metadata.projection_cols != metadata.hidden_width {
        bail!(
            "deterministic final logits projection metadata width mismatch: {} vs {}",
            metadata.projection_cols,
            metadata.hidden_width
        );
    }
    let scalars = auth_read!(finalize_source, GemmaPrefillFinalizeScalarsRequest)?;
    artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(NORMALIZED_FINAL_POSITION_ARTIFACT_NAME)?,
        1,
        hidden_width,
    )?;
    artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(PREFILL_LOGITS_ARTIFACT_NAME)?,
        metadata.projection_rows,
        1,
    )?;

    Ok(PrefillFinalizeRasterState {
        artifact_store_roots,
        source_id: metadata.source_id,
        finalize_source_root,
        prompt_token_count,
        final_hidden_states_ref,
        normalized_final_position_ref: None,
        next_logit_idx: 0,
        logit_count: metadata.projection_rows,
        hidden_width,
        softcap_bits: scalars.final_logit_softcapping.map(Act::to_bits),
        projection_rows_per_tile,
    })
}

#[tile]
pub fn normalize_final_position_to_artifact(
    mut finalize_state: PrefillFinalizeRasterState,
    finalize_source: &RasterPrefillFinalizeSource<'_>,
) -> Result<PrefillFinalizeRasterState> {
    if finalize_state.finalize_source_root != finalize_source.root() {
        bail!(
            "raster prefill finalize source root {} does not match state source root {}",
            finalize_source.root(),
            finalize_state.finalize_source_root
        );
    }
    let (row_count, _) = finalize_state
        .final_hidden_states_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    let final_position = read_sequence_row_from_roots(
        &finalize_state.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: finalize_state.final_hidden_states_ref.clone(),
            row_idx: row_count - 1,
        },
    )?;
    let norm_weights = auth_read!(finalize_source, GemmaPrefillFinalizeNormWeightsRequest)?;
    let scalars = auth_read!(finalize_source, GemmaPrefillFinalizeScalarsRequest)?;
    let normalized = rms_norm_sequence(
        &RasterActivationSequence::from_rows(vec![final_position]),
        Some(&norm_weights),
        Some(scalars.rms_norm_eps),
    )?
    .into_rows()
    .into_iter()
    .next()
    .ok_or_else(|| anyhow!("deterministic final RMSNorm returned no rows"))?;
    if normalized.width() != finalize_state.hidden_width {
        bail!(
            "deterministic final RMSNorm produced width {}, expected {}",
            normalized.width(),
            finalize_state.hidden_width
        );
    }
    finalize_state.artifact_store_roots = append_sequence_row_by_source_name_with_roots(
        &finalize_state.artifact_store_roots,
        NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
        0,
        normalized,
    )?;
    let (artifact_store_roots, normalized_final_position_ref) =
        finalize_sequence_builder_by_source_name_with_roots(
            &finalize_state.artifact_store_roots,
            NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
            RasterTensorId::new(NORMALIZED_FINAL_POSITION_ARTIFACT_NAME)?,
        )?;
    finalize_state.artifact_store_roots = artifact_store_roots;
    finalize_state.normalized_final_position_ref = Some(normalized_final_position_ref);
    Ok(finalize_state)
}

#[tile(kind = recursive)]
pub fn project_next_prefill_logit_chunk(
    mut finalize_state: PrefillFinalizeRasterState,
    finalize_source: &RasterPrefillFinalizeSource<'_>,
) -> Result<(bool, PrefillFinalizeRasterState)> {
    if finalize_state.is_complete() {
        return Ok((true, finalize_state));
    }
    if finalize_state.finalize_source_root != finalize_source.root() {
        bail!(
            "raster prefill finalize source root {} does not match state source root {}",
            finalize_source.root(),
            finalize_state.finalize_source_root
        );
    }
    let normalized_final_position_ref = finalize_state
        .normalized_final_position_ref
        .clone()
        .ok_or_else(|| anyhow!("raster prefill finalize projection missing normalized row ref"))?;
    let normalized_final_position = read_sequence_row_from_roots(
        &finalize_state.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: normalized_final_position_ref,
            row_idx: 0,
        },
    )?;
    if normalized_final_position.width() != finalize_state.hidden_width {
        bail!(
            "deterministic final logits projection input has width {}, expected {}",
            normalized_final_position.width(),
            finalize_state.hidden_width
        );
    }

    let end = finalize_state
        .next_logit_idx
        .saturating_add(finalize_state.projection_rows_per_tile)
        .min(finalize_state.logit_count);
    while finalize_state.next_logit_idx < end {
        let projection_row = auth_read!(
            finalize_source,
            GemmaPrefillFinalizeProjectionRowRequest {
                row_idx: finalize_state.next_logit_idx,
            },
        )?;
        let mut logit = project_row_with_weights(&normalized_final_position, &projection_row)?;
        if let Some(softcap_bits) = finalize_state.softcap_bits {
            logit = softcap_act(logit, Act::from_bits(softcap_bits));
        }
        finalize_state.artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &finalize_state.artifact_store_roots,
            PREFILL_LOGITS_ARTIFACT_NAME,
            finalize_state.next_logit_idx,
            RasterActivationRow::from_acts(vec![logit]),
        )?;
        finalize_state.next_logit_idx += 1;
    }
    Ok((finalize_state.is_complete(), finalize_state))
}

#[tile]
pub fn finalize_prefill_finalize_refs(
    finalize_state: PrefillFinalizeRasterState,
    layer_caches: Vec<crate::routines::prefill_range::raster::PrefillLayerCacheSlot>,
) -> Result<(RasterArtifactStoreRoots, RasterPrefillFinalizeRefs)> {
    let normalized_final_position_ref = finalize_state
        .normalized_final_position_ref
        .clone()
        .ok_or_else(|| anyhow!("raster prefill finalize missing normalized row ref"))?;
    if finalize_state.next_logit_idx != finalize_state.logit_count {
        bail!(
            "raster prefill finalize completed {} logits, expected {}",
            finalize_state.next_logit_idx,
            finalize_state.logit_count
        );
    }
    let (artifact_store_roots, logits_ref) = finalize_sequence_builder_by_source_name_with_roots(
        &finalize_state.artifact_store_roots,
        PREFILL_LOGITS_ARTIFACT_NAME,
        RasterTensorId::new(PREFILL_LOGITS_ARTIFACT_NAME)?,
    )?;
    Ok((
        artifact_store_roots,
        RasterPrefillFinalizeRefs {
            source_id: finalize_state.source_id,
            finalize_source_root: finalize_state.finalize_source_root,
            prompt_token_count: finalize_state.prompt_token_count,
            final_hidden_states_ref: finalize_state.final_hidden_states_ref,
            layer_caches,
            normalized_final_position_ref,
            logits_ref,
            logit_count: finalize_state.logit_count,
        },
    ))
}

#[tile]
pub fn build_prefill_result_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    refs: RasterPrefillFinalizeRefs,
) -> Result<TransformerPrefillResult> {
    build_prefill_result_from_root_refs(&artifact_store_roots, &refs)
}
