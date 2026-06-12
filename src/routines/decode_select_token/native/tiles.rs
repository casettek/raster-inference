use anyhow::{bail, Result};

use crate::shared::model::transformer::InternalLogits;
use crate::shared::numerics::det_num::argmax_first;

pub(crate) fn select_next_token_internal(logits: &InternalLogits) -> Result<u32> {
    let Some(det_values) = logits.det_values() else {
        bail!("deterministic token selection requires canonical logits");
    };
    if det_values.is_empty() {
        bail!("output decode requires at least one logit to select the next token");
    }
    Ok(argmax_first(det_values) as u32)
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
