use anyhow::{anyhow, bail, Result};

use super::utils::build_prefill_result_from_root_refs;
use crate::dsl::prelude::{auth_read, call_recur_tile, call_seq, call_tile, sequence, tile};
use crate::prefill_finalize::raster::auth_source::{
    GemmaPrefillFinalizeMetadataRequest, GemmaPrefillFinalizeNormWeightsRequest,
    GemmaPrefillFinalizeProjectionRowRequest, GemmaPrefillFinalizeScalarsRequest,
};
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactId, RasterArtifactStoreRoots, RasterRoutineOutput,
};
use crate::shared::model::transformer::TransformerPrefillResult;
use crate::shared::numerics::det_num::{softcap_act, Act};
use crate::shared::raster_kernels::transformer::{
    project_row_with_weights, rms_norm_sequence, validate_projection_rows_per_tile,
    RasterActivationRow, RasterActivationSequence,
};
use crate::shared::tensors::raster_row_store::{
    append_sequence_row_by_source_name_with_roots,
    finalize_sequence_builder_by_source_name_with_roots, read_sequence_row_from_roots,
    start_sequence_builder_with_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
    RasterTensorId,
};

pub const NORMALIZED_FINAL_POSITION_ARTIFACT_NAME: &str =
    "prefill.finalize.normalized_final_position";
pub const PREFILL_LOGITS_ARTIFACT_NAME: &str = "prefill.finalize.logits";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillFinalizeInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub prompt_token_count: usize,
    pub finalize_source_root: String,
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<crate::prefill_layer::raster::PrefillLayerCacheSlot>,
    pub projection_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillFinalizeRefs {
    pub source_id: String,
    pub finalize_source_root: String,
    pub prompt_token_count: usize,
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<crate::prefill_layer::raster::PrefillLayerCacheSlot>,
    pub normalized_final_position_ref: RasterActivationSequenceRef,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
}

pub type RasterPrefillFinalizeOutput = RasterRoutineOutput<RasterPrefillFinalizeRefs>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillFinalizeRasterState {
    artifact_store_roots: RasterArtifactStoreRoots,
    source_id: String,
    finalize_source_root: String,
    prompt_token_count: usize,
    final_hidden_states_ref: RasterActivationSequenceRef,
    normalized_final_position_ref: Option<RasterActivationSequenceRef>,
    next_logit_idx: usize,
    logit_count: usize,
    hidden_width: usize,
    softcap_bits: Option<i32>,
    projection_rows_per_tile: usize,
}

impl PrefillFinalizeRasterState {
    fn is_complete(&self) -> bool {
        self.next_logit_idx >= self.logit_count
    }
}

#[tile]
pub fn init_prefill_finalize_state(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
    finalize_source_root: String,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: &[crate::prefill_layer::raster::PrefillLayerCacheSlot],
    projection_rows_per_tile: usize,
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

    let metadata = auth_read!(
        finalize_source_root.as_str(),
        GemmaPrefillFinalizeMetadataRequest
    )?;
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
    let scalars = auth_read!(
        finalize_source_root.as_str(),
        GemmaPrefillFinalizeScalarsRequest
    )?;
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
    mut state: PrefillFinalizeRasterState,
) -> Result<PrefillFinalizeRasterState> {
    let (row_count, _) = state
        .final_hidden_states_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    let final_position = read_sequence_row_from_roots(
        &state.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: state.final_hidden_states_ref.clone(),
            row_idx: row_count - 1,
        },
    )?;
    let norm_weights = auth_read!(
        state.finalize_source_root.as_str(),
        GemmaPrefillFinalizeNormWeightsRequest
    )?;
    let scalars = auth_read!(
        state.finalize_source_root.as_str(),
        GemmaPrefillFinalizeScalarsRequest
    )?;
    let normalized = rms_norm_sequence(
        &RasterActivationSequence::from_rows(vec![final_position]),
        Some(&norm_weights),
        Some(scalars.rms_norm_eps),
    )?
    .into_rows()
    .into_iter()
    .next()
    .ok_or_else(|| anyhow!("deterministic final RMSNorm returned no rows"))?;
    if normalized.width() != state.hidden_width {
        bail!(
            "deterministic final RMSNorm produced width {}, expected {}",
            normalized.width(),
            state.hidden_width
        );
    }
    state.artifact_store_roots = append_sequence_row_by_source_name_with_roots(
        &state.artifact_store_roots,
        NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
        0,
        normalized,
    )?;
    let (artifact_store_roots, normalized_final_position_ref) =
        finalize_sequence_builder_by_source_name_with_roots(
            &state.artifact_store_roots,
            NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
            RasterTensorId::new(NORMALIZED_FINAL_POSITION_ARTIFACT_NAME)?,
        )?;
    state.artifact_store_roots = artifact_store_roots;
    state.normalized_final_position_ref = Some(normalized_final_position_ref);
    Ok(state)
}

#[tile(kind = recursive)]
pub fn project_next_prefill_logit_chunk(
    mut state: PrefillFinalizeRasterState,
) -> Result<(bool, PrefillFinalizeRasterState)> {
    if state.is_complete() {
        return Ok((true, state));
    }
    let normalized_final_position_ref = state
        .normalized_final_position_ref
        .clone()
        .ok_or_else(|| anyhow!("raster prefill finalize projection missing normalized row ref"))?;
    let normalized_final_position = read_sequence_row_from_roots(
        &state.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: normalized_final_position_ref,
            row_idx: 0,
        },
    )?;
    if normalized_final_position.width() != state.hidden_width {
        bail!(
            "deterministic final logits projection input has width {}, expected {}",
            normalized_final_position.width(),
            state.hidden_width
        );
    }

    let end = state
        .next_logit_idx
        .saturating_add(state.projection_rows_per_tile)
        .min(state.logit_count);
    while state.next_logit_idx < end {
        let projection_row = auth_read!(
            state.finalize_source_root.as_str(),
            GemmaPrefillFinalizeProjectionRowRequest {
                row_idx: state.next_logit_idx,
            },
        )?;
        let mut logit = project_row_with_weights(&normalized_final_position, &projection_row)?;
        if let Some(softcap_bits) = state.softcap_bits {
            logit = softcap_act(logit, Act::from_bits(softcap_bits));
        }
        state.artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &state.artifact_store_roots,
            PREFILL_LOGITS_ARTIFACT_NAME,
            state.next_logit_idx,
            RasterActivationRow::from_acts(vec![logit]),
        )?;
        state.next_logit_idx += 1;
    }
    Ok((state.is_complete(), state))
}

#[tile]
pub fn finalize_prefill_finalize_refs(
    state: PrefillFinalizeRasterState,
    layer_caches: Vec<crate::prefill_layer::raster::PrefillLayerCacheSlot>,
) -> Result<(RasterArtifactStoreRoots, RasterPrefillFinalizeRefs)> {
    let normalized_final_position_ref = state
        .normalized_final_position_ref
        .clone()
        .ok_or_else(|| anyhow!("raster prefill finalize missing normalized row ref"))?;
    if state.next_logit_idx != state.logit_count {
        bail!(
            "raster prefill finalize completed {} logits, expected {}",
            state.next_logit_idx,
            state.logit_count
        );
    }
    let (artifact_store_roots, logits_ref) = finalize_sequence_builder_by_source_name_with_roots(
        &state.artifact_store_roots,
        PREFILL_LOGITS_ARTIFACT_NAME,
        RasterTensorId::new(PREFILL_LOGITS_ARTIFACT_NAME)?,
    )?;
    Ok((
        artifact_store_roots,
        RasterPrefillFinalizeRefs {
            source_id: state.source_id,
            finalize_source_root: state.finalize_source_root,
            prompt_token_count: state.prompt_token_count,
            final_hidden_states_ref: state.final_hidden_states_ref,
            layer_caches,
            normalized_final_position_ref,
            logits_ref,
            logit_count: state.logit_count,
        },
    ))
}

#[sequence]
pub fn main_refs(
    input_roots: RasterPrefillFinalizeInputRoots,
) -> Result<RasterPrefillFinalizeOutput> {
    crate::trace::trace_event("prefill.select_final_position");
    let state = call_tile!(
        init_prefill_finalize_state,
        input_roots.artifact_store_roots,
        input_roots.prompt_token_count,
        input_roots.finalize_source_root,
        input_roots.final_hidden_states_ref,
        &input_roots.layer_caches,
        input_roots.projection_rows_per_tile
    )?;
    let state = call_tile!(normalize_final_position_to_artifact, state)?;
    crate::trace::trace_event("prefill.project_to_logits");
    let state = call_recur_tile!(project_next_prefill_logit_chunk, state)?;
    let (artifact_store_roots, refs) = call_tile!(
        finalize_prefill_finalize_refs,
        state,
        input_roots.layer_caches
    )?;
    Ok(RasterPrefillFinalizeOutput::new(artifact_store_roots, refs))
}

#[sequence]
pub fn main(input_roots: RasterPrefillFinalizeInputRoots) -> Result<TransformerPrefillResult> {
    let output = call_seq!(main_refs, input_roots)?;
    call_tile!(
        build_prefill_result_from_refs,
        output.artifact_store_roots,
        output.refs
    )
}

#[tile]
pub fn build_prefill_result_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    refs: RasterPrefillFinalizeRefs,
) -> Result<TransformerPrefillResult> {
    build_prefill_result_from_root_refs(&artifact_store_roots, &refs)
}

fn validate_layer_cache_roots(
    roots: &RasterArtifactStoreRoots,
    layer_caches: &[crate::prefill_layer::raster::PrefillLayerCacheSlot],
) -> Result<()> {
    for cache in layer_caches {
        if let crate::prefill_layer::raster::PrefillLayerCacheSlot::Ref(cache_ref) = cache {
            ensure_artifact_root_present(roots, cache_ref.keys().det_commitment())?;
            ensure_artifact_root_present(roots, cache_ref.values().det_commitment())?;
        }
    }
    Ok(())
}

fn ensure_artifact_root_present(roots: &RasterArtifactStoreRoots, root: &str) -> Result<()> {
    if roots.artifacts.iter().any(|entry| entry.root() == root) {
        return Ok(());
    }
    bail!("raster artifact root {root} is not present in the store roots snapshot")
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
