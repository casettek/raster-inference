use super::{decode_select_checkpoint_state, run_raster, run_raster_state};
use crate::shared::api::output::DecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, read_token_id_from_ref_roots, token_id_leaf,
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{InternalLogits, TransformerDecodeState};
use crate::shared::numerics::det_num::Act;
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::raster_kernels::transformer::RasterActivationRow;
use crate::shared::tensors::raster_row_store::{
    activation_sequence_ref_from_artifact, RasterActivationSequenceRef, RasterTensorId,
};

#[test]
fn run_raster_appends_selected_token_to_decode_state() {
    let mut decode_state = decode_state_with_logits(vec![
        Act::from_bits(1),
        Act::from_bits(9),
        Act::from_bits(3),
    ]);

    let selected = run_raster(&mut decode_state, 1)
        .expect("raster select should run")
        .expect("should select token");

    assert_eq!(selected, 1);
    assert_eq!(decode_state.full_token_ids, vec![7, 1]);
    assert_eq!(decode_state.generated_token_ids, vec![1]);
}

#[test]
fn run_raster_uses_canonical_logits_over_public_f32_view() {
    let mut decode_state = decode_state_with_logits(vec![Act::from_bits(100), Act::from_bits(1)]);
    decode_state.current_logits = vec![0.0, 1000.0];

    let selected = run_raster(&mut decode_state, 1)
        .expect("raster select should run")
        .expect("should select token");

    assert_eq!(selected, 0);
    assert_eq!(decode_state.generated_token_ids, vec![0]);
}

#[test]
fn run_raster_rejects_f32_only_logits_without_mutating_decode_state() {
    let mut decode_state =
        DecodeState::new(vec![7], vec![0.0, 1.0], TransformerDecodeState::default());

    let error =
        run_raster(&mut decode_state, 1).expect_err("f32-only logits should fail in raster");

    assert!(error.to_string().contains("canonical deterministic logits"));
    assert_eq!(decode_state.full_token_ids, vec![7]);
    assert!(decode_state.generated_token_ids.is_empty());
}

#[test]
fn checkpoint_uses_canonical_deterministic_logits_commitment() {
    let internal = InternalLogits::from_det_values(vec![Act::from_bits(100), Act::from_bits(1)]);
    let mut decode_state = DecodeState::new(
        vec![7],
        internal.clone_f32(),
        TransformerDecodeState::default(),
    );
    decode_state.set_internal_logits(internal.clone());
    decode_state.current_logits = vec![0.0, 1000.0];

    let payload = decode_select_checkpoint_state(&decode_state, 0, 1).expect("checkpoint");
    let expected_det_commitment =
        crate::shared::numerics::transformer_kernels::build_det_vector_commitment(
            internal.det_values().expect("canonical logits"),
        );
    let expected_public_commitment = crate::trace::sha256_hex(&decode_state.current_logits);

    assert_eq!(
        payload
            .get("det_current_logits_sha256")
            .and_then(|value| value.as_str()),
        Some(expected_det_commitment.as_str())
    );
    assert_eq!(
        payload
            .get("current_logits_sha256")
            .and_then(|value| value.as_str()),
        Some(expected_public_commitment.as_str())
    );
}

#[test]
fn run_raster_stops_without_requiring_canonical_logits() {
    let mut decode_state =
        DecodeState::new(vec![7], vec![0.0, 1.0], TransformerDecodeState::default());

    let selected = run_raster(&mut decode_state, 0).expect("stop should not inspect logits");

    assert_eq!(selected, None);
    assert_eq!(decode_state.full_token_ids, vec![7]);
    assert!(decode_state.generated_token_ids.is_empty());
}

#[test]
fn run_raster_state_threads_refs_without_host_decode_state_mutation() {
    ArtifactIo::reset_store();
    let roots = ArtifactIo::export_store_roots();
    let (roots, full_token_ids_ref) = insert_token_ids(roots, "decode.input.full", &[7]).unwrap();
    let (roots, logits_ref) = insert_logits(
        roots,
        "decode.input.logits",
        &[Act::from_bits(1), Act::from_bits(9), Act::from_bits(3)],
    )
    .unwrap();
    let decode_state = RasterDecodeLoopState::new(
        roots,
        Some(full_token_ids_ref),
        1,
        None,
        0,
        logits_ref,
        3,
        Vec::new(),
        1,
        1,
        None,
    )
    .unwrap();

    let (next_state, output) =
        run_raster_state(decode_state, 1).expect("raster state select should run");
    let output = output.expect("should select token");

    assert_eq!(output.next_token, 1);
    assert_eq!(next_state.full_token_count, 2);
    assert_eq!(next_state.generated_token_count, 1);
    assert_eq!(next_state.position, 1);
    assert_eq!(next_state.token_count, 1);
    assert_eq!(
        materialize_token_ids(
            &next_state.artifact_store_roots,
            next_state.full_token_ids_ref.as_ref().unwrap()
        )
        .unwrap(),
        vec![7, 1]
    );
    assert_eq!(
        materialize_token_ids(
            &next_state.artifact_store_roots,
            next_state.generated_token_ids_ref.as_ref().unwrap()
        )
        .unwrap(),
        vec![1]
    );
}

fn decode_state_with_logits(det_logits: Vec<Act>) -> DecodeState {
    let internal = InternalLogits::from_det_values(det_logits);
    let mut decode_state = DecodeState::new(
        vec![7],
        internal.clone_f32(),
        TransformerDecodeState::default(),
    );
    decode_state.set_internal_logits(internal);
    decode_state
}

fn insert_token_ids(
    roots: RasterArtifactStoreRoots,
    source_name: &str,
    token_ids: &[u32],
) -> anyhow::Result<(RasterArtifactStoreRoots, RasterTokenIdSequenceRef)> {
    let leaves = token_ids.iter().copied().map(token_id_leaf).collect();
    let (roots, token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
        &roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?;
    Ok((roots, RasterTokenIdSequenceRef::new(token_ids_ref)?))
}

fn insert_logits(
    roots: RasterArtifactStoreRoots,
    source_name: &str,
    logits: &[Act],
) -> anyhow::Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let leaves = logits
        .iter()
        .map(|logit| activation_row_leaf(&RasterActivationRow::from_acts(vec![*logit])))
        .collect::<Vec<_>>();
    let (roots, logits_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        &roots,
        RasterArtifactId::new(source_name)?,
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
) -> anyhow::Result<Vec<u32>> {
    (0..token_ids_ref.token_count())
        .map(|token_idx| read_token_id_from_ref_roots(roots, token_ids_ref, token_idx))
        .collect()
}
