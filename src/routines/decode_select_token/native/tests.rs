use super::{append_token, check_stop_condition, select_next_token_internal};
use crate::shared::api::output::OutputDecodeStopReason;
use crate::shared::model::transformer::InternalLogits;
use crate::shared::numerics::det_num::Act;

#[test]
fn select_next_token_internal_returns_highest_logit_token_id() {
    let token_id = select_next_token_internal(&InternalLogits::from_det_values(vec![
        Act::from_bits(-2),
        Act::from_bits(1),
        Act::from_bits(4),
        Act::from_bits(2),
    ]))
    .expect("token selection");
    assert_eq!(token_id, 2);
}

#[test]
fn select_next_token_internal_breaks_equal_logits_by_lowest_token_id() {
    let token_id = select_next_token_internal(&InternalLogits::from_det_values(vec![
        Act::from_bits(3),
        Act::from_bits(5),
        Act::from_bits(5),
    ]))
    .expect("token selection");
    assert_eq!(token_id, 1);
}

#[test]
fn select_next_token_internal_rejects_empty_logits() {
    let error = select_next_token_internal(&InternalLogits::from_det_values(vec![]))
        .expect_err("empty logits should fail");
    assert!(error.to_string().contains("at least one logit"));
}

#[test]
fn select_next_token_internal_rejects_missing_canonical_logits() {
    let error = select_next_token_internal(&InternalLogits::from_values(vec![1.0, 2.0]))
        .expect_err("f32-only logits should fail");
    assert!(error.to_string().contains("canonical logits"));
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
