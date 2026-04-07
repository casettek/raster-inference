use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl MessageRole {
    pub fn as_template_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplingConfig {
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(128),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSpec {
    pub model_id: String,
    pub tokenizer_path: PathBuf,
    pub chat_template: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceRequest {
    pub messages: Vec<TextMessage>,
    pub add_generation_prompt: bool,
    pub add_special_tokens: bool,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanonicalRequest {
    pub messages: Vec<TextMessage>,
    pub add_generation_prompt: bool,
    pub add_special_tokens: bool,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Phase1State {
    pub model: ModelSpec,
    pub request: CanonicalRequest,
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub commitment: Option<String>,
}
