use anyhow::{bail, Result};

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::InternalLogits;
use crate::shared::numerics::det_num::argmax_first;

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

pub(crate) fn select_next_token_internal(
    logits: &InternalLogits,
    execution_mode: InferenceExecutionMode,
) -> Result<u32> {
    if let Some(det_values) = logits.det_values() {
        if det_values.is_empty() {
            bail!("output decode requires at least one logit to select the next token");
        }
        return Ok(argmax_first(det_values) as u32);
    }

    if execution_mode == InferenceExecutionMode::Deterministic {
        bail!("deterministic token selection requires canonical logits");
    }

    select_next_token(logits.as_f32_slice())
}

pub fn append_token(token_ids: &[u32], next_token: u32) -> Vec<u32> {
    let mut appended = token_ids.to_vec();
    appended.push(next_token);
    appended
}

pub fn check_stop_condition(
    generated_token_count: usize,
    max_new_tokens: usize,
) -> Option<crate::shared::api::output::OutputDecodeStopReason> {
    (generated_token_count >= max_new_tokens)
        .then_some(crate::shared::api::output::OutputDecodeStopReason::MaxNewTokens)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
