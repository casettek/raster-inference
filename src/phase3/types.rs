use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Phase3StopReason {
    #[default]
    MaxNewTokens,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeState {
    pub full_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub current_logits: Vec<f32>,
    pub phase2_decode_state: crate::phase2::Phase2DecodeState,
}

impl DecodeState {
    pub fn new(
        full_token_ids: Vec<u32>,
        current_logits: Vec<f32>,
        phase2_decode_state: crate::phase2::Phase2DecodeState,
    ) -> Self {
        Self {
            full_token_ids,
            generated_token_ids: Vec::new(),
            current_logits,
            phase2_decode_state,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Phase3State {
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    #[serde(skip_serializing, default)]
    pub generated_token_count: usize,
    #[serde(skip_serializing, default)]
    pub stop_reason: Phase3StopReason,
    #[serde(skip_serializing, default)]
    pub phase2_activation_states: Vec<crate::phase2::ActivationSequence>,
}
