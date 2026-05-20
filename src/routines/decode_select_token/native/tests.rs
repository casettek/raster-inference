use super::{append_token, check_stop_condition, select_next_token, select_next_token_internal};
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::api::output::OutputDecodeStopReason;
use crate::shared::model::transformer::InternalLogits;
use crate::shared::numerics::det_num::Act;

#[test]
fn select_next_token_returns_highest_logit_token_id() {
    let token_id = select_next_token(&[-2.0, 0.25, 4.0, 0.5]).expect("token selection");
    assert_eq!(token_id, 2);
}

#[test]
fn select_next_token_breaks_equal_logits_by_lowest_token_id() {
    let token_id = select_next_token(&[1.0, 3.0, 3.0, 2.0]).expect("token selection");
    assert_eq!(token_id, 1);
}

#[test]
fn select_next_token_resolves_near_ties_by_strictly_higher_logit() {
    let token_id = select_next_token(&[0.4820099, 0.4820100, 0.4820098]).expect("token selection");
    assert_eq!(token_id, 1);
}

#[test]
fn select_next_token_rejects_empty_logits() {
    let error = select_next_token(&[]).expect_err("empty logits should fail");
    assert!(error.to_string().contains("at least one logit"));
}

#[test]
fn select_next_token_internal_prefers_canonical_deterministic_values() {
    let token_id = select_next_token_internal(
        &InternalLogits::from_det_values(vec![
            Act::from_bits(3),
            Act::from_bits(5),
            Act::from_bits(5),
        ]),
        InferenceExecutionMode::Deterministic,
    )
    .expect("token selection");
    assert_eq!(token_id, 1);
}

#[test]
fn append_token_returns_new_sequence() {
    let original = vec![1, 2];
    let appended = append_token(&original, 3);
    assert_eq!(original, vec![1, 2]);
    assert_eq!(appended, vec![1, 2, 3]);
}

#[test]
fn check_stop_condition_finishes_at_max_new_tokens() {
    assert_eq!(
        check_stop_condition(2, 2),
        Some(OutputDecodeStopReason::MaxNewTokens)
    );
}

#[test]
fn check_stop_condition_immediately_finishes_when_limit_is_zero() {
    assert_eq!(
        check_stop_condition(0, 0),
        Some(OutputDecodeStopReason::MaxNewTokens)
    );
}
