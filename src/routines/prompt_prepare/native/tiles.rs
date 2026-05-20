use anyhow::{bail, Context, Result};
use minijinja::{context, Environment};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::shared::api::input::{
    Gemma4Prompt, MessageRole, ModelSpec, TextDecodingPolicy, TextMessage,
};

#[derive(Debug, Clone, serde::Serialize)]
struct TemplateMessage {
    role: String,
    content: String,
}

impl From<&TextMessage> for TemplateMessage {
    fn from(message: &TextMessage) -> Self {
        Self {
            role: message.role.as_template_role().to_string(),
            content: message.content.clone(),
        }
    }
}

pub fn decode_prompt_bytes(prompt_bytes: &[u8], policy: TextDecodingPolicy) -> Result<String> {
    match policy {
        TextDecodingPolicy::Utf8 => String::from_utf8(prompt_bytes.to_vec())
            .context("failed to decode prompt bytes as utf-8"),
    }
}

pub fn build_gemma4_messages(
    prompt_text: &str,
    add_generation_prompt: bool,
) -> Result<Gemma4Prompt> {
    if prompt_text.is_empty() {
        bail!("input embedding requires a non-empty prompt");
    }

    Ok(Gemma4Prompt {
        messages: vec![TextMessage {
            role: MessageRole::User,
            content: prompt_text.to_string(),
        }],
        add_generation_prompt,
    })
}

pub fn render_prompt(prompt: &Gemma4Prompt, model: &ModelSpec) -> Result<String> {
    let mut environment = Environment::new();
    environment
        .add_template("chat", &model.chat_template)
        .context("failed to register chat template")?;

    let template = environment
        .get_template("chat")
        .context("failed to load chat template")?;

    let messages = prompt
        .messages
        .iter()
        .map(TemplateMessage::from)
        .collect::<Vec<_>>();

    template
        .render(context! {
            messages => messages,
            add_generation_prompt => prompt.add_generation_prompt,
            bos_token => model.bos_token.clone(),
            eos_token => model.eos_token.clone(),
            unk_token => model.unk_token.clone(),
        })
        .context("failed to render chat template")
}

pub fn tokenize_prompt(
    prompt: &str,
    tokenizer: &Tokenizer,
    add_special_tokens: bool,
) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode_fast(prompt, add_special_tokens)
        .map_err(anyhow::Error::msg)
        .context("failed to tokenize rendered prompt")?;

    Ok(encoding.get_ids().to_vec())
}

pub fn build_prompt_commitment(prompt_token_ids: &[u32]) -> Result<String> {
    let payload = serde_json::to_vec(prompt_token_ids)
        .context("failed to serialize input-embedding prompt token ids")?;

    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
