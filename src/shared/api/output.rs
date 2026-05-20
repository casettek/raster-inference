use serde::{Deserialize, Serialize};

use crate::shared::model::transformer::InternalLogits;

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
    pub(crate) internal_logits: InternalLogits,
    pub transformer_decode_state: crate::shared::model::transformer::TransformerDecodeState,
}

impl DecodeState {
    pub fn new(
        full_token_ids: Vec<u32>,
        current_logits: Vec<f32>,
        transformer_decode_state: crate::shared::model::transformer::TransformerDecodeState,
    ) -> Self {
        Self {
            full_token_ids,
            generated_token_ids: Vec::new(),
            internal_logits: InternalLogits::from_values(current_logits.clone()),
            current_logits,
            transformer_decode_state,
        }
    }

    pub(crate) fn clone_internal_logits(&self) -> InternalLogits {
        if self.internal_logits.as_f32_slice().is_empty() && !self.current_logits.is_empty() {
            return InternalLogits::from_values(self.current_logits.clone());
        }
        self.internal_logits.clone()
    }

    pub(crate) fn set_internal_logits(&mut self, logits: InternalLogits) {
        self.current_logits = logits.clone_f32();
        self.internal_logits = logits;
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
    pub decode_transition_states: Vec<crate::shared::model::transformer::ActivationSequence>,
}
