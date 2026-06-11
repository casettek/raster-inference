pub mod dsl;
pub mod io;
pub mod routines;
pub mod runtime;
pub mod shared;

pub use decode_select_token::run as run_decode_select_token;
pub use decode_transition_finalize::trace_checkpoint as finalize_decode_transition;
pub use input_embedding::run as run_input_embedding;
pub use io::{
    load_chat_template, load_embedding_table_from_gemma_model_path, load_embedding_table_from_path,
    load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path,
};
pub use output_finalize::run as run_output_finalize;
pub use prefill_finalize::run as run_prefill_finalize;
pub use prefill_prepare_aux::run as run_prefill_prepare_aux;
pub use prefill_range::native::{run_text_layers_prefill, run_text_layers_prefill_with_cache};
pub use prefill_range::run as run_prefill_range;
pub use prefill_range::run_with_mode as run_prefill_range_with_mode;
pub use prompt_prepare::run as run_prompt_prepare;
pub use routines::{
    decode_layer_range, decode_select_token, decode_transition_finalize, input_embedding,
    output_finalize, prefill_finalize, prefill_prepare_aux, prefill_range, prefill_range_finalize,
    prompt_prepare,
};
pub use runtime::checkpoints;
pub use runtime::checkpoints::{
    classify_checkpoint, CheckpointTaxonomy, PhaseId, RasterDetourController, RasterDetourSpec,
    RoutineId,
};
pub use runtime::inference::{
    run_inference, run_inference_with_controls, InferenceControls, InferenceRunOutcome,
    InferenceState, InputEmbeddingState, PausedInferenceState, RasterPromptPreparedState,
    RasterSizingControls,
};
pub use runtime::pipeline::{
    decode_step, decode_step_with_mode, run_output_decode, run_output_decode_with_mode,
    run_prefill_pass, run_prefill_pass_with_mode, run_transformer_state_transition,
    run_transformer_state_transition_for_token_ids, validate_sampling_config,
};
pub use runtime::trace;
pub use shared::api::input::{
    InferenceExecutionMode, InferenceRequest, MessageRole, ModelSpec, PromptPreparationState,
    RasterPromptPreparationState, SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use shared::api::output::{DecodeState, OutputDecodeState, OutputDecodeStopReason};
pub use shared::artifacts::integrity_mode::RasterIntegrityMode;
pub use shared::model::gemma::Gemma4Prompt;
pub use shared::model::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeState, GemmaTokenizerSpec,
    GemmaVocabEntry,
};
pub use shared::model::transformer::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
    Gemma4PleLayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, PrefillLogits, TransformerDecodeState,
    TransformerDecodeStepResult, TransformerPrefillResult, TransformerStateTransitionState,
};
pub use shared::numerics::transformer_kernels::{
    append_kv_cache, apply_final_logit_softcapping, apply_final_norm, compute_decode_ple_input,
    compute_prefill_ple_inputs, embed_input_token, embed_input_token_with_mode, embed_input_tokens,
    embed_input_tokens_with_mode, extract_prefill_logits, project_decode_hidden_to_logits,
    project_to_logits, run_gemma4_layer, run_gemma4_layer_decode, select_final_position,
};
