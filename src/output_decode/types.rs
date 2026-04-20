use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputDecodeStopReason {
    #[default]
    MaxNewTokens,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeState {
    pub full_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub current_logits: Vec<f32>,
    pub transformer_decode_state: crate::transformer_state_transition::TransformerDecodeState,
}

impl DecodeState {
    pub fn new(
        full_token_ids: Vec<u32>,
        current_logits: Vec<f32>,
        transformer_decode_state: crate::transformer_state_transition::TransformerDecodeState,
    ) -> Self {
        Self {
            full_token_ids,
            generated_token_ids: Vec::new(),
            current_logits,
            transformer_decode_state,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputDecodeState {
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    #[serde(skip_serializing, default)]
    pub generated_token_count: usize,
    #[serde(skip_serializing, default)]
    pub stop_reason: OutputDecodeStopReason,
    #[serde(skip_serializing, default)]
    pub decode_transition_states: Vec<crate::transformer_state_transition::ActivationSequence>,
}
