use anyhow::{anyhow, bail, Result};

use crate::dsl::prelude::{call_recur_tile, call_tile, sequence, tile};
use crate::shared::api::output::OutputDecodeStopReason;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    read_token_id_from_ref_roots, token_id_leaf, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::numerics::det_num::{argmax_first, Act};
use crate::shared::tensors::raster_row_store::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
};

pub const DEFAULT_DECODE_SELECT_LOGITS_PER_TILE: usize = 32;
pub const DEFAULT_DECODE_SELECT_TOKEN_IDS_PER_TILE: usize = 64;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeSelectInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub logits_ref: RasterActivationSequenceRef,
    pub full_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub full_token_count: usize,
    pub generated_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub generated_token_count: usize,
    pub max_new_tokens: usize,
    pub logits_per_tile: usize,
    pub token_ids_per_tile: usize,
    pub output_full_token_ids_source_name: String,
    pub output_generated_token_ids_source_name: String,
    pub output_selected_token_source_name: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectArgmaxState {
    artifact_store_roots: RasterArtifactStoreRoots,
    logits_ref: RasterActivationSequenceRef,
    next_token_idx: usize,
    logit_count: usize,
    best_token_id: u32,
    best_logit_bits: i32,
    logits_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectAppendState {
    artifact_store_roots: RasterArtifactStoreRoots,
    full_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    full_token_count: usize,
    generated_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    generated_token_count: usize,
    next_full_token_idx: usize,
    next_generated_token_idx: usize,
    next_token: u32,
    token_ids_per_tile: usize,
    output_full_token_ids_source_name: String,
    output_generated_token_ids_source_name: String,
    output_selected_token_source_name: String,
    logits_ref: RasterActivationSequenceRef,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeSelectOutputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub next_token: u32,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub full_token_ids_ref: RasterTokenIdSequenceRef,
    pub generated_token_ids_ref: RasterTokenIdSequenceRef,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
}

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
    let (logit_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    if logit_count == 0 {
        bail!("raster decode select token requires at least one canonical logit");
    }
    if width != 1 {
        bail!("raster decode select logits artifact width {width}, expected 1");
    }

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

fn read_logit_bits(
    artifact_store_roots: &RasterArtifactStoreRoots,
    logits_ref: &RasterActivationSequenceRef,
    token_idx: usize,
) -> Result<i32> {
    let row = read_sequence_row_from_roots(
        artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: logits_ref.clone(),
            row_idx: token_idx,
        },
    )?;
    let [logit_bits] = row.act_bits() else {
        bail!(
            "raster decode select logit row {token_idx} has width {}",
            row.width()
        );
    };
    Ok(*logit_bits)
}

fn candidate_wins(best_logit_bits: i32, candidate_bits: i32) -> bool {
    let candidates = [
        Act::from_bits(best_logit_bits),
        Act::from_bits(candidate_bits),
    ];
    argmax_first(&candidates) == 1
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
        logit_count: state.logits_ref.tensor_ref().shape().sequence_metadata()?.0,
        logits_ref: state.logits_ref,
    })
}

fn validate_token_input(
    artifact_store_roots: &RasterArtifactStoreRoots,
    token_ids_ref: Option<&RasterTokenIdSequenceRef>,
    token_count: usize,
    label: &str,
) -> Result<()> {
    match (token_ids_ref, token_count) {
        (Some(token_ids_ref), count) => {
            if token_ids_ref.token_count() != count {
                bail!(
                    "raster decode select {label} token count mismatch: ref has {}, expected {count}",
                    token_ids_ref.token_count()
                );
            }
            if count > 0 {
                read_token_id_from_ref_roots(artifact_store_roots, token_ids_ref, 0)?;
            }
            Ok(())
        }
        (None, 0) => Ok(()),
        (None, _) => bail!("raster decode select {label} token ids root is missing"),
    }
}

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

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
