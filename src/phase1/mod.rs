use anyhow::Result;
use tokenizers::Tokenizer;

use crate::trace::{trace_event, trace_scope};

use self::tiles::{
    build_gemma4_messages, build_phase1_commitment, decode_prompt_bytes, render_prompt,
    tokenize_prompt,
};

pub mod tiles;
pub mod types;

pub use types::{
    Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, Phase1State, SamplingConfig,
    TextDecodingPolicy, TextMessage,
};

pub fn run_phase1(
    request: &types::InferenceRequest,
    model: &types::ModelSpec,
    tokenizer: &Tokenizer,
) -> Result<types::Phase1State> {
    let _trace = trace_scope("prompt.prepare");
    trace_event("prompt.decode_bytes");
    let prompt_text = decode_prompt_bytes(&request.prompt_bytes, request.text_decoding_policy)?;
    trace_event("prompt.build_messages");
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request.add_generation_prompt)?;
    trace_event("prompt.render");
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    trace_event("prompt.tokenize");
    let prompt_token_ids =
        tokenize_prompt(&rendered_prompt, tokenizer, request.add_special_tokens)?;
    trace_event("prompt.commitment");
    let prompt_token_ids_sha256 = build_phase1_commitment(&prompt_token_ids)?;

    Ok(types::Phase1State {
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}
