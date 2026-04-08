use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

pub mod io;
pub mod phase1;
pub mod phase2;

pub use io::{
    load_chat_template, load_embedding_table_from_gemma_model_path,
    load_embedding_table_from_path, load_phase2_model_from_gemma_model_path,
    load_tokenizer_from_path,
};
pub use phase1::{
    run_phase1, Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, Phase1State,
    SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use phase2::{
    embed_input_tokens, run_first_gemma4_layer, run_phase2, ActivationSequence,
    EmbeddedTokenSequence, EmbeddingTable, Gemma4Layer0Weights, Gemma4Phase2Model,
    Gemma4PleLayerWeights, GemmaEmbeddingTensorSource, MatrixF32, Phase2State,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceState {
    pub phase1: Phase1State,
    pub phase2: Phase2State,
}

pub fn run_inference(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    phase2_model: &Gemma4Phase2Model,
) -> Result<InferenceState> {
    let phase1 = run_phase1(request, model, tokenizer)?;
    let phase2 = run_phase2(&phase1, phase2_model)?;

    Ok(InferenceState { phase1, phase2 })
}

/// Reserved module boundary for the eventual logits-to-token decode phase.
pub mod phase3 {}
