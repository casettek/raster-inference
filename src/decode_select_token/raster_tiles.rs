use anyhow::{anyhow, bail, Result};

use crate::raster_authoring::prelude::{call_recur_tile, call_tile, sequence, tile};
use crate::shared::artifact_io::ArtifactIo;
use crate::shared::det_num::{argmax_first, Act};
use crate::shared::output::OutputDecodeStopReason;
use crate::shared::raster_artifact_store::{
    read_token_id_from_ref_roots, token_id_leaf, RasterArtifactId,
    RasterArtifactMetadata, RasterArtifactStoreRoots, RasterSelectedTokenRef,
    RasterTokenIdSequenceRef,
};
use crate::shared::raster_row_store::{
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
mod tests {
    use super::{
        copy_next_full_token_chunk, init_decode_select_append_state, init_select_next_token, main,
        scan_next_token_logit, DecodeSelectArgmaxState, RasterDecodeSelectInputRoots,
    };
    use crate::shared::artifact_io::ArtifactIo;
    use crate::shared::det_num::Act;
    use crate::shared::input::InferenceExecutionMode;
    use crate::shared::raster_artifact_store::{
        activation_row_leaf, read_selected_token_from_roots, read_token_id_from_ref_roots,
        token_id_leaf, RasterActivationSequenceArtifactRef, RasterArtifactId,
        RasterArtifactMetadata, RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
    };
    use crate::shared::raster_row_store::{
        activation_sequence_ref_from_artifact, RasterActivationSequenceRef, RasterTensorId,
    };
    use crate::shared::raster_transformer_kernels::RasterActivationRow;
    use crate::shared::transformer::InternalLogits;
    use anyhow::Result;

    #[test]
    fn main_selects_highest_logit_and_appends_token_refs() {
        let input = input_roots(
            vec![10],
            vec![],
            vec![
                Act::from_bits(-2),
                Act::from_bits(0),
                Act::from_bits(7),
                Act::from_bits(3),
            ],
            1,
            2,
            1,
            "basic",
        )
        .expect("input roots");

        let output = main(input)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(output.next_token, 2);
        assert_eq!(output.selected_token_ref.token_ids_ref().token_count(), 1);
        assert_eq!(
            read_selected_token_from_roots(
                &output.artifact_store_roots,
                &output.selected_token_ref,
            )
            .expect("selected token artifact"),
            2
        );
        assert_eq!(
            materialize_token_ids(&output.artifact_store_roots, &output.full_token_ids_ref,)
                .expect("full token ids"),
            vec![10, 2]
        );
        assert_eq!(
            materialize_token_ids(
                &output.artifact_store_roots,
                &output.generated_token_ids_ref,
            )
            .expect("generated token ids"),
            vec![2]
        );
    }

    #[test]
    fn main_breaks_equal_logits_by_lowest_token_id() {
        let input = input_roots(
            vec![10],
            vec![],
            vec![Act::from_bits(1), Act::from_bits(5), Act::from_bits(5)],
            1,
            1,
            1,
            "ties",
        )
        .expect("input roots");

        let output = main(input)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(output.next_token, 1);
    }

    #[test]
    fn main_matches_native_deterministic_selection() {
        let det_logits = vec![
            Act::from_bits(-10),
            Act::from_bits(4),
            Act::from_bits(7),
            Act::from_bits(7),
        ];
        let internal = InternalLogits::from_det_values(det_logits.clone());
        let input =
            input_roots(vec![10], vec![], det_logits, 1, 1, 1, "native").expect("input roots");

        let native = crate::decode_select_token::tiles::select_next_token_internal(
            &internal,
            InferenceExecutionMode::Deterministic,
        )
        .expect("native deterministic selection should run");
        let output = main(input)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(output.next_token, native);
    }

    #[test]
    fn chunk_sizes_do_not_change_output() {
        let single = main(
            input_roots(
                vec![1, 2],
                vec![2],
                (0..40).map(Act::from_bits).collect(),
                4,
                1,
                1,
                "single",
            )
            .expect("single input roots"),
        )
        .expect("single chunk should run")
        .expect("single chunk should select");
        let single_full_tokens =
            materialize_token_ids(&single.artifact_store_roots, &single.full_token_ids_ref)
                .expect("single full tokens");
        let single_generated_tokens = materialize_token_ids(
            &single.artifact_store_roots,
            &single.generated_token_ids_ref,
        )
        .expect("single generated tokens");
        let multi = main(
            input_roots(
                vec![1, 2],
                vec![2],
                (0..40).map(Act::from_bits).collect(),
                4,
                11,
                8,
                "multi",
            )
            .expect("multi input roots"),
        )
        .expect("multi chunk should run")
        .expect("multi chunk should select");

        assert_eq!(single.next_token, multi.next_token);
        assert_eq!(
            single_full_tokens,
            materialize_token_ids(&multi.artifact_store_roots, &multi.full_token_ids_ref,)
                .expect("multi full tokens")
        );
        assert_eq!(
            single_generated_tokens,
            materialize_token_ids(&multi.artifact_store_roots, &multi.generated_token_ids_ref,)
                .expect("multi generated tokens")
        );
    }

    #[test]
    fn states_serialize_refs_and_cursors_not_payloads() {
        let input = input_roots(
            vec![9, 4],
            vec![4],
            vec![Act::from_bits(1), Act::from_bits(3)],
            2,
            1,
            1,
            "compact",
        )
        .expect("input roots");
        let argmax_state = init_select_next_token(
            input.artifact_store_roots.clone(),
            input.logits_ref.clone(),
            input.logits_per_tile,
        )
        .expect("argmax init");
        let append_state =
            init_decode_select_append_state(input.artifact_store_roots.clone(), &input, 1)
                .expect("append init");

        let encoded_argmax = serde_json::to_string(&argmax_state).expect("serialize argmax");
        let encoded_append = serde_json::to_string(&append_state).expect("serialize append");

        assert!(encoded_argmax.contains("artifact_store_roots"));
        assert!(encoded_argmax.contains("logits_ref"));
        assert!(encoded_append.contains("full_token_ids_ref"));
        for encoded in [&encoded_argmax, &encoded_append] {
            assert!(!encoded.contains("current_logits"));
            assert!(!encoded.contains("\"logit_bits\":["));
            assert!(!encoded.contains("full_token_ids\":["));
            assert!(!encoded.contains("generated_token_ids\":["));
            assert!(!encoded.contains("layer_caches"));
            assert!(!encoded.contains("activation_states"));
        }
    }

    #[test]
    fn missing_logits_root_fails_closed() {
        let input = input_roots(
            vec![1],
            vec![],
            vec![Act::from_bits(1), Act::from_bits(2)],
            1,
            1,
            1,
            "missing",
        )
        .expect("input roots");

        let error = init_select_next_token(
            RasterArtifactStoreRoots::default(),
            input.logits_ref,
            input.logits_per_tile,
        )
        .expect_err("missing root should fail");

        assert!(error.to_string().contains("not present"));
    }

    #[test]
    fn stale_builder_roots_fail_closed() {
        let input = input_roots(
            vec![1, 2],
            vec![],
            vec![Act::from_bits(1), Act::from_bits(2)],
            1,
            1,
            1,
            "stale",
        )
        .expect("input roots");
        let state = init_decode_select_append_state(input.artifact_store_roots.clone(), &input, 1)
            .expect("append init");
        ArtifactIo::append_leaf_by_builder_source_name_with_roots(
            &state.artifact_store_roots,
            &state.output_full_token_ids_source_name,
            0,
            token_id_leaf(99),
        )
        .expect("external append should advance builder roots");

        let error = copy_next_full_token_chunk(state).expect_err("stale roots should fail");

        assert!(error.to_string().contains("snapshot"));
    }

    #[test]
    fn selected_token_ref_fails_closed_with_missing_roots() {
        let input = input_roots(
            vec![1],
            vec![],
            vec![Act::from_bits(1), Act::from_bits(9)],
            1,
            1,
            1,
            "selected-missing",
        )
        .expect("input roots");
        let output = main(input)
            .expect("raster select should run")
            .expect("should select token");

        let error = read_selected_token_from_roots(
            &RasterArtifactStoreRoots::default(),
            &output.selected_token_ref,
        )
        .expect_err("missing selected-token root should fail");

        assert!(error.to_string().contains("not present"));
    }

    #[test]
    fn stop_condition_does_not_inspect_logits_or_start_builders() {
        let mut input = input_roots(
            vec![1],
            vec![2],
            vec![Act::from_bits(1), Act::from_bits(2)],
            1,
            1,
            1,
            "stop",
        )
        .expect("input roots");
        input.artifact_store_roots = RasterArtifactStoreRoots::default();

        let output = main(input).expect("stop should not inspect roots");

        assert_eq!(output, None);
    }

    #[test]
    fn zero_chunk_sizes_fail() {
        let input = input_roots(
            vec![1],
            vec![],
            vec![Act::from_bits(1), Act::from_bits(2)],
            1,
            0,
            1,
            "zero-logits",
        )
        .expect("input roots");
        let error = main(input).expect_err("zero logits per tile should fail");
        assert!(error.to_string().contains("greater than zero"));

        let input = input_roots(
            vec![1],
            vec![],
            vec![Act::from_bits(1), Act::from_bits(2)],
            1,
            1,
            0,
            "zero-tokens",
        )
        .expect("input roots");
        let error = main(input).expect_err("zero token ids per tile should fail");
        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn scan_next_token_logit_bounds_work_per_recursive_step() {
        let input = input_roots(
            vec![1],
            vec![],
            (0..40).map(Act::from_bits).collect(),
            1,
            7,
            1,
            "bounds",
        )
        .expect("input roots");
        let state = DecodeSelectArgmaxState {
            artifact_store_roots: input.artifact_store_roots,
            logits_ref: input.logits_ref,
            next_token_idx: 1,
            logit_count: 40,
            best_token_id: 0,
            best_logit_bits: 0,
            logits_per_tile: 7,
        };

        let (done, state) = scan_next_token_logit(state).expect("scan chunk should succeed");

        assert!(!done);
        assert_eq!(state.next_token_idx, 8);
        assert_eq!(state.best_token_id, 7);
        assert_eq!(state.best_logit_bits, 7);
    }

    fn input_roots(
        full_token_ids: Vec<u32>,
        generated_token_ids: Vec<u32>,
        logits: Vec<Act>,
        max_new_tokens: usize,
        logits_per_tile: usize,
        token_ids_per_tile: usize,
        source_prefix: &str,
    ) -> Result<RasterDecodeSelectInputRoots> {
        ArtifactIo::reset_store();
        let roots = ArtifactIo::export_store_roots();
        let (roots, full_token_ids_ref) = insert_token_ids(
            roots,
            format!("{source_prefix}.input.full"),
            &full_token_ids,
        )?;
        let (roots, generated_token_ids_ref) = insert_token_ids(
            roots,
            format!("{source_prefix}.input.generated"),
            &generated_token_ids,
        )?;
        let (roots, logits_ref) =
            insert_logits(roots, format!("{source_prefix}.input.logits"), &logits)?;

        Ok(RasterDecodeSelectInputRoots {
            artifact_store_roots: roots,
            logits_ref,
            full_token_ids_ref,
            full_token_count: full_token_ids.len(),
            generated_token_ids_ref,
            generated_token_count: generated_token_ids.len(),
            max_new_tokens,
            logits_per_tile,
            token_ids_per_tile,
            output_full_token_ids_source_name: format!("{source_prefix}.output.full"),
            output_generated_token_ids_source_name: format!("{source_prefix}.output.generated"),
            output_selected_token_source_name: format!("{source_prefix}.output.selected"),
        })
    }

    fn insert_token_ids(
        roots: RasterArtifactStoreRoots,
        source_name: String,
        token_ids: &[u32],
    ) -> Result<(RasterArtifactStoreRoots, Option<RasterTokenIdSequenceRef>)> {
        if token_ids.is_empty() {
            return Ok((roots, None));
        }
        let leaves = token_ids.iter().copied().map(token_id_leaf).collect();
        let (roots, token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
            &roots,
            RasterArtifactId::new(source_name)?,
            RasterArtifactMetadata::token_ids(token_ids.len()),
            leaves,
        )?;
        Ok((roots, Some(RasterTokenIdSequenceRef::new(token_ids_ref)?)))
    }

    fn insert_logits(
        roots: RasterArtifactStoreRoots,
        source_name: String,
        logits: &[Act],
    ) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
        let leaves = logits
            .iter()
            .map(|logit| activation_row_leaf(&RasterActivationRow::from_acts(vec![*logit])))
            .collect::<Vec<_>>();
        let (roots, logits_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
            &roots,
            RasterArtifactId::new(source_name.clone())?,
            RasterArtifactMetadata::activation_rows(logits.len(), 1)?,
            leaves,
        )?;
        let logits_ref = activation_sequence_ref_from_artifact(
            RasterTensorId::new(source_name)?,
            RasterActivationSequenceArtifactRef::new(logits_artifact_ref)?,
        )?;
        Ok((roots, logits_ref))
    }

    fn materialize_token_ids(
        roots: &RasterArtifactStoreRoots,
        token_ids_ref: &RasterTokenIdSequenceRef,
    ) -> Result<Vec<u32>> {
        (0..token_ids_ref.token_count())
            .map(|token_idx| read_token_id_from_ref_roots(roots, token_ids_ref, token_idx))
            .collect()
    }
}
