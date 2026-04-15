use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

pub mod io;
pub mod phase1;
pub mod phase2;
pub mod trace;

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
    apply_final_logit_softcapping, apply_final_norm, compute_prefill_ple_inputs,
    embed_input_tokens, extract_prefill_logits, project_to_logits, run_gemma4_layer, run_phase2,
    run_prefill_pass, run_text_layers_prefill, select_final_position, ActivationSequence,
    EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights,
    Gemma4LogitsProjection, Gemma4Phase2Model, Gemma4PleGlobalWeights, Gemma4PleLayerWeights,
    Gemma4PrefillPleInputs, GemmaEmbeddingTensorSource, MatrixF32, Phase2State, PrefillLogits,
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
