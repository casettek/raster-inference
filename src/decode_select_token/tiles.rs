use anyhow::{bail, Result};

pub fn select_next_token(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() {
        bail!("output decode requires at least one logit to select the next token");
    }

    let mut best_token = 0usize;
    for (token_id, logit) in logits.iter().enumerate().skip(1) {
        if logit.total_cmp(&logits[best_token]).is_gt() {
            best_token = token_id;
        }
    }

    Ok(best_token as u32)
}

pub fn append_token(token_ids: &[u32], next_token: u32) -> Vec<u32> {
    let mut appended = token_ids.to_vec();
    appended.push(next_token);
    appended
}

pub fn check_stop_condition(
    generated_token_count: usize,
    max_new_tokens: usize,
) -> Option<crate::shared::output::OutputDecodeStopReason> {
    (generated_token_count >= max_new_tokens)
        .then_some(crate::shared::output::OutputDecodeStopReason::MaxNewTokens)
}

#[cfg(test)]
mod tests {
    use super::{append_token, check_stop_condition, select_next_token};
    use crate::shared::output::OutputDecodeStopReason;

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
        let token_id =
            select_next_token(&[0.4820099, 0.4820100, 0.4820098]).expect("token selection");
        assert_eq!(token_id, 1);
    }

    #[test]
    fn select_next_token_rejects_empty_logits() {
        let error = select_next_token(&[]).expect_err("empty logits should fail");
        assert!(error.to_string().contains("at least one logit"));
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
}
