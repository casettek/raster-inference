pub mod io;
pub mod phase1;
pub mod phase2;

pub use io::{load_chat_template, load_tokenizer_from_path};
pub use phase1::{
    run_phase1, Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, Phase1State,
    SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use phase2::{embed_input_tokens, EmbeddedTokenSequence, EmbeddingTable};

/// Reserved module boundary for the eventual logits-to-token decode phase.
pub mod phase3 {}
