pub mod phase1;
pub mod tiles;
pub mod types;

pub use phase1::{load_chat_template, load_tokenizer_from_path, run_phase1};
pub use types::{
    Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, Phase1State, SamplingConfig,
    TextDecodingPolicy, TextMessage,
};

/// Reserved module boundary for the eventual transformer state transition phase.
pub mod phase2 {}

/// Reserved module boundary for the eventual logits-to-token decode phase.
pub mod phase3 {}
