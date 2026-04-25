use anyhow::Result;
use serde_json::json;

use crate::shared::input::InferenceExecutionMode;
use crate::shared::output::DecodeState;

pub mod tiles;

pub fn run(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
    execution_mode: InferenceExecutionMode,
) -> Result<Option<u32>> {
    if tiles::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens).is_some()
    {
        return Ok(None);
    }

    let next_token =
        tiles::select_next_token_internal(&decode_state.clone_internal_logits(), execution_mode)?;
    decode_state.full_token_ids = tiles::append_token(&decode_state.full_token_ids, next_token);
    decode_state.generated_token_ids =
        tiles::append_token(&decode_state.generated_token_ids, next_token);
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": crate::output_finalize::tiles::build_output_decode_commitment(&decode_state.generated_token_ids)?,
            "current_logits": decode_state.current_logits.clone(),
            "current_logits_sha256": crate::trace::sha256_hex(&decode_state.current_logits),
            "selected_next_token": next_token,
            "decode_position": decode_state.transformer_decode_state.position,
            "decode_token_count": decode_state.transformer_decode_state.token_count,
            "layer_caches": crate::trace::serialize_layer_caches(&decode_state.transformer_decode_state.layer_caches),
            "max_new_tokens": max_new_tokens,
        }),
    );
    Ok(Some(next_token))
}
