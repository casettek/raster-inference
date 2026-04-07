pub mod io;
pub mod phase1;

pub use io::{load_chat_template, load_tokenizer_from_path};
pub use phase1::{
    run_phase1, Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, Phase1State,
    SamplingConfig, TextDecodingPolicy, TextMessage,
};

/// Reserved module boundary for the eventual transformer state transition phase.
pub mod phase2 {}

/// Reserved module boundary for the eventual logits-to-token decode phase.
pub mod phase3 {}
