use std::{fs, path::Path};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::{
    tiles::{
        build_gemma4_messages, build_phase1_commitment, decode_prompt_bytes, render_prompt,
        tokenize_prompt,
    },
    types::{InferenceRequest, ModelSpec, Phase1State},
};

pub fn load_chat_template<P: AsRef<Path>>(path: P) -> Result<String> {
    fs::read_to_string(path.as_ref()).with_context(|| {
        format!(
            "failed to read chat template from {}",
            path.as_ref().display()
        )
    })
}

pub fn load_tokenizer_from_path<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    let raw = fs::read(path.as_ref())
        .with_context(|| format!("failed to read tokenizer from {}", path.as_ref().display()))?;

    Tokenizer::from_bytes(raw.as_slice()).map_err(anyhow::Error::msg)
}

pub fn run_phase1(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
) -> Result<Phase1State> {
    let prompt_text = decode_prompt_bytes(&request.prompt_bytes, request.text_decoding_policy)?;
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request.add_generation_prompt)?;
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    let prompt_tokens = tokenize_prompt(&rendered_prompt, tokenizer, request.add_special_tokens)?;
    let commitment = Some(build_phase1_commitment(
        model,
        request,
        &prompt_text,
        &gemma4_prompt,
        &rendered_prompt,
        &prompt_tokens,
    )?);

    Ok(Phase1State {
        model: model.clone(),
        request: request.clone(),
        prompt_text,
        gemma4_prompt,
        rendered_prompt,
        prompt_tokens,
        commitment,
    })
}
