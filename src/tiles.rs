use anyhow::{bail, Context, Result};
use minijinja::{context, Environment};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::types::{CanonicalRequest, InferenceRequest, ModelSpec, TextMessage};

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

pub fn canonicalize_request(request: &InferenceRequest) -> Result<CanonicalRequest> {
    if request.messages.is_empty() {
        bail!("phase 1 requires at least one message");
    }

    let messages = request
        .messages
        .iter()
        .map(|message| {
            let content = message.content.trim().to_string();
            if content.is_empty() {
                bail!("messages must contain non-empty content");
            }

            Ok(TextMessage {
                role: message.role.clone(),
                content,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(CanonicalRequest {
        messages,
        add_generation_prompt: request.add_generation_prompt,
        add_special_tokens: request.add_special_tokens,
        sampling: request.sampling.clone(),
    })
}

pub fn render_prompt(request: &CanonicalRequest, model: &ModelSpec) -> Result<String> {
    let mut environment = Environment::new();
    environment
        .add_template("chat", &model.chat_template)
        .context("failed to register chat template")?;

    let template = environment
        .get_template("chat")
        .context("failed to load chat template")?;

    let messages = request
        .messages
        .iter()
        .map(TemplateMessage::from)
        .collect::<Vec<_>>();

    template
        .render(context! {
            messages => messages,
            add_generation_prompt => request.add_generation_prompt,
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

pub fn build_phase1_commitment(
    model: &ModelSpec,
    request: &CanonicalRequest,
    prompt: &str,
    prompt_tokens: &[u32],
) -> Result<String> {
    let payload = serde_json::to_vec(&serde_json::json!({
        "model_id": model.model_id,
        "request": request,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
    }))
    .context("failed to serialize phase 1 commitment payload")?;

    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}

#[cfg(test)]
mod tests {
    use super::{canonicalize_request, render_prompt};
    use crate::types::{InferenceRequest, MessageRole, ModelSpec, SamplingConfig, TextMessage};

    #[test]
    fn canonicalize_trims_message_content() {
        let request = InferenceRequest {
            messages: vec![TextMessage {
                role: MessageRole::User,
                content: "  hello world  ".to_string(),
            }],
            add_generation_prompt: true,
            add_special_tokens: true,
            sampling: SamplingConfig::default(),
        };

        let canonical = canonicalize_request(&request).expect("request should canonicalize");

        assert_eq!(canonical.messages[0].content, "hello world");
    }

    #[test]
    fn render_prompt_uses_messages_and_generation_flag() {
        let model = ModelSpec {
            model_id: "gemma-4-test".to_string(),
            tokenizer_path: "tokenizer.json".into(),
            chat_template: "{{ bos_token }}{% for message in messages %}[{{ message.role }}] {{ message.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}".to_string(),
            bos_token: Some("<bos>".to_string()),
            eos_token: None,
            unk_token: None,
        };
        let request = InferenceRequest {
            messages: vec![TextMessage {
                role: MessageRole::User,
                content: "hello".to_string(),
            }],
            add_generation_prompt: true,
            add_special_tokens: true,
            sampling: SamplingConfig::default(),
        };

        let canonical = canonicalize_request(&request).expect("request should canonicalize");
        let prompt = render_prompt(&canonical, &model).expect("prompt should render");

        assert_eq!(prompt, "<bos>[user] hello[assistant]");
    }
}
