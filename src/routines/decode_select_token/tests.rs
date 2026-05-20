use super::{decode_select_checkpoint_state, run_raster};
use crate::shared::api::output::DecodeState;
use crate::shared::model::transformer::{InternalLogits, TransformerDecodeState};
use crate::shared::numerics::det_num::Act;

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
