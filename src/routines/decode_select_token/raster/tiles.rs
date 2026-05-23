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
    mut argmax_state: DecodeSelectArgmaxState,
) -> Result<(bool, DecodeSelectArgmaxState)> {
    if argmax_state.next_token_idx >= argmax_state.logit_count {
        return Ok((true, argmax_state));
    }
    if argmax_state.logits_per_tile == 0 {
        bail!("raster decode select logits per tile must be greater than zero");
    }

    let end = argmax_state
        .next_token_idx
        .saturating_add(argmax_state.logits_per_tile)
        .min(argmax_state.logit_count);
    while argmax_state.next_token_idx < end {
        let candidate_bits = read_logit_bits(
            &argmax_state.artifact_store_roots,
            &argmax_state.logits_ref,
            argmax_state.next_token_idx,
        )?;
        if candidate_wins(argmax_state.best_logit_bits, candidate_bits) {
            argmax_state.best_token_id = u32::try_from(argmax_state.next_token_idx)
                .map_err(|_| anyhow!("raster decode selected token index exceeds u32"))?;
            argmax_state.best_logit_bits = candidate_bits;
        }
        argmax_state.next_token_idx += 1;
    }

    Ok((
        argmax_state.next_token_idx >= argmax_state.logit_count,
        argmax_state,
    ))
}

#[tile]
pub fn finalize_selected_token(argmax_state: DecodeSelectArgmaxState) -> Result<u32> {
    if argmax_state.logit_count == 0 {
        bail!("raster decode select token cannot finalize empty logits");
    }
    if argmax_state.next_token_idx != argmax_state.logit_count {
        bail!(
            "raster decode select token scanned {} logits, expected {}",
            argmax_state.next_token_idx,
            argmax_state.logit_count
        );
    }
    Ok(argmax_state.best_token_id)
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
    mut append_state: DecodeSelectAppendState,
) -> Result<(bool, DecodeSelectAppendState)> {
    if append_state.next_full_token_idx >= append_state.full_token_count {
        return Ok((true, append_state));
    }
    if append_state.token_ids_per_tile == 0 {
        bail!("raster decode select token ids per tile must be greater than zero");
    }
    let Some(token_ids_ref) = append_state.full_token_ids_ref.clone() else {
        bail!("raster decode select full token ids root is missing");
    };
    let end = append_state
        .next_full_token_idx
        .saturating_add(append_state.token_ids_per_tile)
        .min(append_state.full_token_count);
    while append_state.next_full_token_idx < end {
        let token_id = read_token_id_from_ref_roots(
            &append_state.artifact_store_roots,
            &token_ids_ref,
            append_state.next_full_token_idx,
        )?;
        let (next_roots, _builder_root) =
            ArtifactIo::append_leaf_by_builder_source_name_with_roots(
                &append_state.artifact_store_roots,
                &append_state.output_full_token_ids_source_name,
                append_state.next_full_token_idx,
                token_id_leaf(token_id),
            )?;
        append_state.artifact_store_roots = next_roots;
        append_state.next_full_token_idx += 1;
    }
    Ok((
        append_state.next_full_token_idx >= append_state.full_token_count,
        append_state,
    ))
}

#[tile(kind = recursive)]
pub fn copy_next_generated_token_chunk(
    mut append_state: DecodeSelectAppendState,
) -> Result<(bool, DecodeSelectAppendState)> {
    if append_state.next_generated_token_idx >= append_state.generated_token_count {
        return Ok((true, append_state));
    }
    if append_state.token_ids_per_tile == 0 {
        bail!("raster decode select token ids per tile must be greater than zero");
    }
    let Some(token_ids_ref) = append_state.generated_token_ids_ref.clone() else {
        bail!("raster decode select generated token ids root is missing");
    };
    let end = append_state
        .next_generated_token_idx
        .saturating_add(append_state.token_ids_per_tile)
        .min(append_state.generated_token_count);
    while append_state.next_generated_token_idx < end {
        let token_id = read_token_id_from_ref_roots(
            &append_state.artifact_store_roots,
            &token_ids_ref,
            append_state.next_generated_token_idx,
        )?;
        let (next_roots, _builder_root) =
            ArtifactIo::append_leaf_by_builder_source_name_with_roots(
                &append_state.artifact_store_roots,
                &append_state.output_generated_token_ids_source_name,
                append_state.next_generated_token_idx,
                token_id_leaf(token_id),
            )?;
        append_state.artifact_store_roots = next_roots;
        append_state.next_generated_token_idx += 1;
    }
    Ok((
        append_state.next_generated_token_idx >= append_state.generated_token_count,
        append_state,
    ))
}

#[tile]
pub fn append_selected_token(
    mut append_state: DecodeSelectAppendState,
) -> Result<DecodeSelectAppendState> {
    if append_state.next_full_token_idx != append_state.full_token_count {
        bail!(
            "raster decode select copied {} full tokens, expected {}",
            append_state.next_full_token_idx,
            append_state.full_token_count
        );
    }
    if append_state.next_generated_token_idx != append_state.generated_token_count {
        bail!(
            "raster decode select copied {} generated tokens, expected {}",
            append_state.next_generated_token_idx,
            append_state.generated_token_count
        );
    }
    let (next_roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        &append_state.artifact_store_roots,
        &append_state.output_full_token_ids_source_name,
        append_state.full_token_count,
        token_id_leaf(append_state.next_token),
    )?;
    append_state.artifact_store_roots = next_roots;
    let (next_roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        &append_state.artifact_store_roots,
        &append_state.output_generated_token_ids_source_name,
        append_state.generated_token_count,
        token_id_leaf(append_state.next_token),
    )?;
    append_state.artifact_store_roots = next_roots;
    Ok(append_state)
}

#[tile]
pub fn finalize_decode_select_refs(
    append_state: DecodeSelectAppendState,
) -> Result<RasterDecodeSelectOutputRefs> {
    if append_state.next_full_token_idx != append_state.full_token_count {
        bail!(
            "raster decode select finalized after copying {} full tokens, expected {}",
            append_state.next_full_token_idx,
            append_state.full_token_count
        );
    }
    if append_state.next_generated_token_idx != append_state.generated_token_count {
        bail!(
            "raster decode select finalized after copying {} generated tokens, expected {}",
            append_state.next_generated_token_idx,
            append_state.generated_token_count
        );
    }
    let (artifact_store_roots, full_token_ids_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(
            &append_state.artifact_store_roots,
            &append_state.output_full_token_ids_source_name,
        )?;
    let full_token_ids_ref = RasterTokenIdSequenceRef::new(full_token_ids_ref)?;
    let (artifact_store_roots, generated_token_ids_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(
            &artifact_store_roots,
            &append_state.output_generated_token_ids_source_name,
        )?;
    let generated_token_ids_ref = RasterTokenIdSequenceRef::new(generated_token_ids_ref)?;
    let (artifact_store_roots, _builder) = ArtifactIo::start_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(append_state.output_selected_token_source_name.clone())?,
        RasterArtifactMetadata::token_ids(1),
    )?;
    let (artifact_store_roots, _builder_root) =
        ArtifactIo::append_leaf_by_builder_source_name_with_roots(
            &artifact_store_roots,
            &append_state.output_selected_token_source_name,
            0,
            token_id_leaf(append_state.next_token),
        )?;
    let (artifact_store_roots, selected_token_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(
            &artifact_store_roots,
            &append_state.output_selected_token_source_name,
        )?;
    let selected_token_ref =
        RasterSelectedTokenRef::new(RasterTokenIdSequenceRef::new(selected_token_ref)?)?;

    Ok(RasterDecodeSelectOutputRefs {
        artifact_store_roots,
        next_token: append_state.next_token,
        selected_token_ref,
        full_token_ids_ref,
        generated_token_ids_ref,
        logit_count: decode_select_logit_count(&append_state.logits_ref)?,
        logits_ref: append_state.logits_ref,
    })
}
