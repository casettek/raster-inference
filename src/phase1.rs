use std::{fs, path::Path};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::{
    tiles::{build_phase1_commitment, canonicalize_request, render_prompt, tokenize_prompt},
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
    let request = canonicalize_request(request)?;
    let prompt = render_prompt(&request, model)?;
    let prompt_tokens = tokenize_prompt(&prompt, tokenizer, request.add_special_tokens)?;
    let commitment = Some(build_phase1_commitment(
        model,
        &request,
        &prompt,
        &prompt_tokens,
    )?);

    Ok(Phase1State {
        model: model.clone(),
        request,
        prompt,
        prompt_tokens,
        commitment,
    })
}
