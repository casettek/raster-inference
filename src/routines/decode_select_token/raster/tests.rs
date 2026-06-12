use super::{
    check_stop_condition, copy_next_full_token_chunk, init_decode_select_append_state,
    init_select_next_token, main, scan_next_token_logit, DecodeSelectArgmaxState,
    DecodeSelectSelectedState, RasterDecodeSelectInputRoots,
};
use crate::shared::api::output::OutputDecodeStopReason;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, read_selected_token_from_roots, read_token_id_from_ref_roots,
    token_id_leaf, RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::InternalLogits;
use crate::shared::numerics::det_num::Act;
use crate::shared::raster_kernels::transformer::RasterActivationRow;
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, RasterActivationSequenceRef, RasterTensorId,
};
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
        2,
        1,
        "basic",
    )
    .expect("input roots");

    let output = main(input).expect("raster select should run");

    assert_eq!(output.next_token, 2);
    assert_eq!(output.selected_token_ref.token_ids_ref().token_count(), 1);
    assert_eq!(
        read_selected_token_from_roots(&output.artifact_store_roots, &output.selected_token_ref,)
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
        "ties",
    )
    .expect("input roots");

    let output = main(input).expect("raster select should run");

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
    let input = input_roots(vec![10], vec![], det_logits, 1, 1, "native").expect("input roots");

    let native = crate::routines::decode_select_token::native::select_next_token_internal(&internal)
        .expect("native deterministic selection should run");
    let output = main(input).expect("raster select should run");

    assert_eq!(output.next_token, native);
}

#[test]
fn chunk_sizes_do_not_change_output() {
    let single = main(
        input_roots(
            vec![1, 2],
            vec![2],
            (0..40).map(Act::from_bits).collect(),
            1,
            1,
            "single",
        )
        .expect("single input roots"),
    )
    .expect("single chunk should run");
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
            11,
            8,
            "multi",
        )
        .expect("multi input roots"),
    )
    .expect("multi chunk should run");

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
        1,
        1,
        "compact",
    )
    .expect("input roots");
    let argmax_state = init_select_next_token(input.clone()).expect("argmax init");
    let append_state = init_decode_select_append_state(DecodeSelectSelectedState {
        input_roots: input,
        next_token: 1,
    })
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
        "missing",
    )
    .expect("input roots");

    let mut input = input;
    input.artifact_store_roots = RasterArtifactStoreRoots::default();
    let error = init_select_next_token(input).expect_err("missing root should fail");

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
        "stale",
    )
    .expect("input roots");
    let state = init_decode_select_append_state(DecodeSelectSelectedState {
        input_roots: input,
        next_token: 1,
    })
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
        "selected-missing",
    )
    .expect("input roots");
    let output = main(input).expect("raster select should run");

    let error = read_selected_token_from_roots(
        &RasterArtifactStoreRoots::default(),
        &output.selected_token_ref,
    )
    .expect_err("missing selected-token root should fail");

    assert!(error.to_string().contains("not present"));
}

#[test]
fn stop_condition_does_not_inspect_logits_or_start_builders() {
    assert_eq!(
        check_stop_condition(1, 1),
        Some(OutputDecodeStopReason::MaxNewTokens)
    );
}

#[test]
fn zero_chunk_sizes_fail() {
    let input = input_roots(
        vec![1],
        vec![],
        vec![Act::from_bits(1), Act::from_bits(2)],
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
        7,
        1,
        "bounds",
    )
    .expect("input roots");
    let state = DecodeSelectArgmaxState {
        input_roots: input,
        next_token_idx: 1,
        logit_count: 40,
        best_token_id: 0,
        best_logit_bits: 0,
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
