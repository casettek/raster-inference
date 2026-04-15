use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase3StopReason {
    MaxNewTokens,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeState {
    pub full_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub current_logits: Vec<f32>,
}

impl DecodeState {
    pub fn new(full_token_ids: Vec<u32>, current_logits: Vec<f32>) -> Self {
        Self {
            full_token_ids,
            generated_token_ids: Vec::new(),
            current_logits,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Phase3State {
    #[serde(skip_serializing, default)]
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    pub generated_token_count: usize,
    pub stop_reason: Phase3StopReason,
}
