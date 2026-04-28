use anyhow::Result;
use tokenizers::Tokenizer;

use crate::shared::input::{InferenceRequest, PromptPreparationState};
use crate::trace::{trace_event, trace_scope};

use self::tiles::{
    build_gemma4_messages, build_prompt_commitment, decode_prompt_bytes, render_prompt,
    tokenize_prompt,
};

pub mod raster_tiles;
pub mod tiles;

pub fn run(
    request: &InferenceRequest,
    model: &crate::shared::input::ModelSpec,
    tokenizer: &Tokenizer,
) -> Result<PromptPreparationState> {
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
    let prompt_token_ids_sha256 = build_prompt_commitment(&prompt_token_ids)?;

    Ok(PromptPreparationState {
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}

pub fn run_raster(
    request: &InferenceRequest,
    model: &crate::shared::input::ModelSpec,
    tokenizer: &Tokenizer,
) -> Result<PromptPreparationState> {
    raster_tiles::run(request, model, tokenizer)
}
