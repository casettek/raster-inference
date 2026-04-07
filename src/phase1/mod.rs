use anyhow::Result;
use tokenizers::Tokenizer;

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
    let prompt_text = decode_prompt_bytes(&request.prompt_bytes, request.text_decoding_policy)?;
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request.add_generation_prompt)?;
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    let prompt_token_ids =
        tokenize_prompt(&rendered_prompt, tokenizer, request.add_special_tokens)?;
    let prompt_token_ids_sha256 = build_phase1_commitment(&prompt_token_ids)?;

    Ok(types::Phase1State {
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}
