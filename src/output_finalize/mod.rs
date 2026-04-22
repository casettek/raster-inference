use anyhow::Result;
use serde_json::json;
use tokenizers::Tokenizer;

use crate::shared::output::{DecodeState, OutputDecodeState};

pub mod tiles;

pub fn run(decode_state: DecodeState, tokenizer: &Tokenizer) -> Result<OutputDecodeState> {
    let generated_token_count = decode_state.generated_token_ids.len();
    let generated_text =
        tiles::detokenize_output_tokens(tokenizer, &decode_state.generated_token_ids)?;
    let generated_token_ids_sha256 =
        tiles::build_output_decode_commitment(&decode_state.generated_token_ids)?;
    let stop_reason = crate::shared::output::OutputDecodeStopReason::MaxNewTokens;
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": generated_token_ids_sha256.clone(),
            "generated_text": generated_text.clone(),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason.clone(),
        }),
    );

    Ok(OutputDecodeState {
        generated_token_ids: decode_state.generated_token_ids,
        generated_token_ids_sha256,
        generated_text,
        generated_token_count,
        stop_reason,
        decode_transition_states: Vec::new(),
    })
}
