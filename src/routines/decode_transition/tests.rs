use super::{
    current_det_logits_commitment, current_logits_commitment, generated_token_ids_commitment,
};
use crate::shared::api::output::DecodeState;
use crate::shared::model::transformer::{InternalLogits, TransformerDecodeState};
use crate::shared::numerics::det_num::Act;

#[test]
fn decode_finalize_uses_internal_logits_commitment_methodology() {
    let mut decode_state = DecodeState::new(
        vec![7],
        vec![999.0, -999.0],
        TransformerDecodeState::default(),
    );
    decode_state.set_internal_logits(InternalLogits::from_det_values(vec![
        Act::from_bits(3),
        Act::from_bits(5),
    ]));

    let internal = decode_state.clone_internal_logits();
    assert_eq!(
        current_logits_commitment(&decode_state),
        crate::shared::numerics::transformer_kernels::build_vector_commitment(
            internal.as_f32_slice()
        )
    );
    assert_eq!(
        current_det_logits_commitment(&decode_state),
        Some(
            crate::shared::numerics::transformer_kernels::build_det_vector_commitment(
                internal.det_values().expect("det logits")
            )
        )
    );
}

#[test]
fn decode_finalize_uses_output_finalize_token_commitment_methodology() {
    let mut decode_state = DecodeState::new(vec![7], vec![0.0], TransformerDecodeState::default());
    decode_state.generated_token_ids = vec![4, 3, 6, 7];

    assert_eq!(
        generated_token_ids_commitment(&decode_state).expect("commitment should build"),
        crate::output_finalize::native::build_output_decode_commitment(
            &decode_state.generated_token_ids
        )
        .expect("output commitment should build")
    );
}
