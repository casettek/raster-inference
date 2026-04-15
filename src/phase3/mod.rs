use anyhow::Result;
use tokenizers::Tokenizer;

use crate::trace::{trace_event, trace_scope};

pub mod tiles;
pub mod types;

pub use tiles::{
    append_token, build_phase3_commitment, check_stop_condition, detokenize_output_tokens,
    select_next_token, validate_sampling_config,
};
pub use types::{DecodeState, Phase3State, Phase3StopReason};

pub fn run_phase3(
    prompt_token_ids: &[u32],
    initial_phase2_state: &crate::phase2::Phase2State,
    sampling: &crate::phase1::SamplingConfig,
    tokenizer: &Tokenizer,
    phase2_model: &crate::phase2::Gemma4Phase2Model,
) -> Result<Phase3State> {
    let _trace = trace_scope("phase3.run_phase3");
    let max_new_tokens = validate_sampling_config(sampling)?;
    let mut decode_state = DecodeState::new(
        prompt_token_ids.to_vec(),
        initial_phase2_state.prefill_logits.logits.clone(),
    );

    loop {
        trace_event("phase3.check_stop_condition");
        if let Some(stop_reason) =
            check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens)
        {
            trace_event("phase3.detokenize_output_tokens");
            let generated_token_count = decode_state.generated_token_ids.len();
            let generated_text =
                detokenize_output_tokens(tokenizer, &decode_state.generated_token_ids)?;
            let generated_token_ids_sha256 =
                build_phase3_commitment(&decode_state.generated_token_ids)?;

            return Ok(Phase3State {
                generated_token_ids: decode_state.generated_token_ids,
                generated_token_ids_sha256,
                generated_text,
                generated_token_count,
                stop_reason,
            });
        }

        trace_event("phase3.select_next_token");
        let next_token = select_next_token(&decode_state.current_logits)?;

        trace_event("phase3.append_token");
        decode_state.full_token_ids = append_token(&decode_state.full_token_ids, next_token);
        decode_state.generated_token_ids =
            append_token(&decode_state.generated_token_ids, next_token);

        trace_event("phase3.run_phase2_for_token_ids");
        let phase2_state =
            crate::phase2::run_phase2_for_token_ids(&decode_state.full_token_ids, phase2_model)?;
        decode_state.current_logits = phase2_state.prefill_logits.logits;
    }
}
