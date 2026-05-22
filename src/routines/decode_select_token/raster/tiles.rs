use anyhow::{anyhow, bail, Result};

use crate::dsl::prelude::{call_recur_tile, call_tile, sequence, tile};
use crate::shared::api::output::OutputDecodeStopReason;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    read_token_id_from_ref_roots, token_id_leaf, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef;

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    input_roots: RasterDecodeSelectInputRoots,
) -> Result<Option<RasterDecodeSelectOutputRefs>> {
    if call_tile!(
        check_stop_condition,
        input_roots.generated_token_count,
        input_roots.max_new_tokens
    )
    .is_some()
    {
        return Ok(None);
    }

    let argmax_state = call_tile!(
        init_select_next_token,
        input_roots.artifact_store_roots.clone(),
        input_roots.logits_ref.clone(),
        input_roots.logits_per_tile
    )?;
    let argmax_state = call_recur_tile!(scan_next_token_logit, argmax_state)?;
    let next_token = call_tile!(finalize_selected_token, argmax_state)?;
    let append_state = call_tile!(
        init_decode_select_append_state,
        input_roots.artifact_store_roots.clone(),
        &input_roots,
        next_token
    )?;
    let append_state = call_recur_tile!(copy_next_full_token_chunk, append_state)?;
    let append_state = call_recur_tile!(copy_next_generated_token_chunk, append_state)?;
    let append_state = call_tile!(append_selected_token, append_state)?;
    call_tile!(finalize_decode_select_refs, append_state).map(Some)
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
pub fn check_stop_condition(
    generated_token_count: usize,
    max_new_tokens: usize,
) -> Option<OutputDecodeStopReason> {
    (generated_token_count >= max_new_tokens).then_some(OutputDecodeStopReason::MaxNewTokens)
}

#[tile]
pub fn init_select_next_token(
    artifact_store_roots: RasterArtifactStoreRoots,
    logits_ref: RasterActivationSequenceRef,
    logits_per_tile: usize,
) -> Result<DecodeSelectArgmaxState> {
    if logits_per_tile == 0 {
        bail!("raster decode select logits per tile must be greater than zero");
    }
    let logit_count = decode_select_logit_count(&logits_ref)?;

    let best_logit_bits = read_logit_bits(&artifact_store_roots, &logits_ref, 0)?;
    Ok(DecodeSelectArgmaxState {
        artifact_store_roots,
        logits_ref,
        next_token_idx: 1,
        logit_count,
        best_token_id: 0,
        best_logit_bits,
        logits_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn scan_next_token_logit(
    mut state: DecodeSelectArgmaxState,
) -> Result<(bool, DecodeSelectArgmaxState)> {
    if state.next_token_idx >= state.logit_count {
        return Ok((true, state));
    }
    if state.logits_per_tile == 0 {
        bail!("raster decode select logits per tile must be greater than zero");
    }

    let end = state
        .next_token_idx
        .saturating_add(state.logits_per_tile)
        .min(state.logit_count);
    while state.next_token_idx < end {
        let candidate_bits = read_logit_bits(
            &state.artifact_store_roots,
            &state.logits_ref,
            state.next_token_idx,
        )?;
        if candidate_wins(state.best_logit_bits, candidate_bits) {
            state.best_token_id = u32::try_from(state.next_token_idx)
                .map_err(|_| anyhow!("raster decode selected token index exceeds u32"))?;
            state.best_logit_bits = candidate_bits;
        }
        state.next_token_idx += 1;
    }

    Ok((state.next_token_idx >= state.logit_count, state))
}

#[tile]
pub fn finalize_selected_token(state: DecodeSelectArgmaxState) -> Result<u32> {
    if state.logit_count == 0 {
        bail!("raster decode select token cannot finalize empty logits");
    }
    if state.next_token_idx != state.logit_count {
        bail!(
            "raster decode select token scanned {} logits, expected {}",
            state.next_token_idx,
            state.logit_count
        );
    }
    Ok(state.best_token_id)
}

#[tile]
pub fn init_decode_select_append_state(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: &RasterDecodeSelectInputRoots,
    next_token: u32,
) -> Result<DecodeSelectAppendState> {
    if input_roots.token_ids_per_tile == 0 {
        bail!("raster decode select token ids per tile must be greater than zero");
    }
    validate_token_input(
        &artifact_store_roots,
        input_roots.full_token_ids_ref.as_ref(),
        input_roots.full_token_count,
        "full",
    )?;
    validate_token_input(
        &artifact_store_roots,
        input_roots.generated_token_ids_ref.as_ref(),
        input_roots.generated_token_count,
        "generated",
    )?;

    let (next_roots, _builder) = ArtifactIo::start_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(input_roots.output_full_token_ids_source_name.clone())?,
        RasterArtifactMetadata::token_ids(input_roots.full_token_count + 1),
    )?;
    artifact_store_roots = next_roots;
    let (next_roots, _builder) = ArtifactIo::start_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(input_roots.output_generated_token_ids_source_name.clone())?,
        RasterArtifactMetadata::token_ids(input_roots.generated_token_count + 1),
    )?;
    artifact_store_roots = next_roots;

    Ok(DecodeSelectAppendState {
        artifact_store_roots,
        full_token_ids_ref: input_roots.full_token_ids_ref.clone(),
        full_token_count: input_roots.full_token_count,
        generated_token_ids_ref: input_roots.generated_token_ids_ref.clone(),
        generated_token_count: input_roots.generated_token_count,
        next_full_token_idx: 0,
        next_generated_token_idx: 0,
        next_token,
        token_ids_per_tile: input_roots.token_ids_per_tile,
        output_full_token_ids_source_name: input_roots.output_full_token_ids_source_name.clone(),
        output_generated_token_ids_source_name: input_roots
            .output_generated_token_ids_source_name
            .clone(),
        output_selected_token_source_name: input_roots.output_selected_token_source_name.clone(),
        logits_ref: input_roots.logits_ref.clone(),
    })
}

#[tile(kind = recursive)]
pub fn copy_next_full_token_chunk(
    mut state: DecodeSelectAppendState,
) -> Result<(bool, DecodeSelectAppendState)> {
    if state.next_full_token_idx >= state.full_token_count {
        return Ok((true, state));
    }
    if state.token_ids_per_tile == 0 {
        bail!("raster decode select token ids per tile must be greater than zero");
    }
    let Some(token_ids_ref) = state.full_token_ids_ref.clone() else {
        bail!("raster decode select full token ids root is missing");
    };
    let end = state
        .next_full_token_idx
        .saturating_add(state.token_ids_per_tile)
        .min(state.full_token_count);
    while state.next_full_token_idx < end {
        let token_id = read_token_id_from_ref_roots(
            &state.artifact_store_roots,
            &token_ids_ref,
            state.next_full_token_idx,
        )?;
        let (next_roots, _builder_root) =
            ArtifactIo::append_leaf_by_builder_source_name_with_roots(
                &state.artifact_store_roots,
                &state.output_full_token_ids_source_name,
                state.next_full_token_idx,
                token_id_leaf(token_id),
            )?;
        state.artifact_store_roots = next_roots;
        state.next_full_token_idx += 1;
    }
    Ok((state.next_full_token_idx >= state.full_token_count, state))
}

#[tile(kind = recursive)]
pub fn copy_next_generated_token_chunk(
    mut state: DecodeSelectAppendState,
) -> Result<(bool, DecodeSelectAppendState)> {
    if state.next_generated_token_idx >= state.generated_token_count {
        return Ok((true, state));
    }
    if state.token_ids_per_tile == 0 {
        bail!("raster decode select token ids per tile must be greater than zero");
    }
    let Some(token_ids_ref) = state.generated_token_ids_ref.clone() else {
        bail!("raster decode select generated token ids root is missing");
    };
    let end = state
        .next_generated_token_idx
        .saturating_add(state.token_ids_per_tile)
        .min(state.generated_token_count);
    while state.next_generated_token_idx < end {
        let token_id = read_token_id_from_ref_roots(
            &state.artifact_store_roots,
            &token_ids_ref,
            state.next_generated_token_idx,
        )?;
        let (next_roots, _builder_root) =
            ArtifactIo::append_leaf_by_builder_source_name_with_roots(
                &state.artifact_store_roots,
                &state.output_generated_token_ids_source_name,
                state.next_generated_token_idx,
                token_id_leaf(token_id),
            )?;
        state.artifact_store_roots = next_roots;
        state.next_generated_token_idx += 1;
    }
    Ok((
        state.next_generated_token_idx >= state.generated_token_count,
        state,
    ))
}

#[tile]
pub fn append_selected_token(
    mut state: DecodeSelectAppendState,
) -> Result<DecodeSelectAppendState> {
    if state.next_full_token_idx != state.full_token_count {
        bail!(
            "raster decode select copied {} full tokens, expected {}",
            state.next_full_token_idx,
            state.full_token_count
        );
    }
    if state.next_generated_token_idx != state.generated_token_count {
        bail!(
            "raster decode select copied {} generated tokens, expected {}",
            state.next_generated_token_idx,
            state.generated_token_count
        );
    }
    let (next_roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        &state.artifact_store_roots,
        &state.output_full_token_ids_source_name,
        state.full_token_count,
        token_id_leaf(state.next_token),
    )?;
    state.artifact_store_roots = next_roots;
    let (next_roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        &state.artifact_store_roots,
        &state.output_generated_token_ids_source_name,
        state.generated_token_count,
        token_id_leaf(state.next_token),
    )?;
    state.artifact_store_roots = next_roots;
    Ok(state)
}

#[tile]
pub fn finalize_decode_select_refs(
    state: DecodeSelectAppendState,
) -> Result<RasterDecodeSelectOutputRefs> {
    if state.next_full_token_idx != state.full_token_count {
        bail!(
            "raster decode select finalized after copying {} full tokens, expected {}",
            state.next_full_token_idx,
            state.full_token_count
        );
    }
    if state.next_generated_token_idx != state.generated_token_count {
        bail!(
            "raster decode select finalized after copying {} generated tokens, expected {}",
            state.next_generated_token_idx,
            state.generated_token_count
        );
    }
    let (artifact_store_roots, full_token_ids_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(
            &state.artifact_store_roots,
            &state.output_full_token_ids_source_name,
        )?;
    let full_token_ids_ref = RasterTokenIdSequenceRef::new(full_token_ids_ref)?;
    let (artifact_store_roots, generated_token_ids_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(
            &artifact_store_roots,
            &state.output_generated_token_ids_source_name,
        )?;
    let generated_token_ids_ref = RasterTokenIdSequenceRef::new(generated_token_ids_ref)?;
    let (artifact_store_roots, _builder) = ArtifactIo::start_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(state.output_selected_token_source_name.clone())?,
        RasterArtifactMetadata::token_ids(1),
    )?;
    let (artifact_store_roots, _builder_root) =
        ArtifactIo::append_leaf_by_builder_source_name_with_roots(
            &artifact_store_roots,
            &state.output_selected_token_source_name,
            0,
            token_id_leaf(state.next_token),
        )?;
    let (artifact_store_roots, selected_token_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(
            &artifact_store_roots,
            &state.output_selected_token_source_name,
        )?;
    let selected_token_ref =
        RasterSelectedTokenRef::new(RasterTokenIdSequenceRef::new(selected_token_ref)?)?;

    Ok(RasterDecodeSelectOutputRefs {
        artifact_store_roots,
        next_token: state.next_token,
        selected_token_ref,
        full_token_ids_ref,
        generated_token_ids_ref,
        logit_count: decode_select_logit_count(&state.logits_ref)?,
        logits_ref: state.logits_ref,
    })
}
