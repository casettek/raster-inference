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
