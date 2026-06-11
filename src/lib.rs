pub mod dsl;
pub mod io;
pub mod routines;
pub mod runtime;
pub mod shared;

// ---------------------------------------------------------------------------
// Stable surface
//
// Protocol-facing types and entry points. Everything below this section is
// either a compatibility re-export slated for removal or internal machinery
// reachable under its module path.
// ---------------------------------------------------------------------------

// Role entry points (claimer / challenger) and audit report types.
pub use runtime::roles::claimer::ClaimerOutcome;
pub use runtime::roles::{challenger, claimer};
pub use shared::api::audit::{
    AuditOutcome, CheckpointDivergence, ClaimedTrace, ClaimedTraceEntry, DetourArtifact,
};

// Inference sequencing: request/outcome types and the sequence engine.
pub use runtime::inference::{
    InferenceControls, InferenceRunOutcome, InferenceState, InputEmbeddingState,
    PausedInferenceState, RasterPromptPreparedState, RasterSizingControls,
};
pub use runtime::sequence;

// Checkpoint taxonomy and trace artifact layer.
pub use runtime::checkpoints;
pub use runtime::checkpoints::{
    classify_checkpoint, CheckpointTaxonomy, PhaseId, RasterDetourController, RasterDetourSpec,
    RoutineId,
};
pub use runtime::trace;

// Request/response API types.
pub use shared::api::input::{
    InferenceExecutionMode, InferenceRequest, MessageRole, ModelSpec, PromptPreparationState,
    RasterPromptPreparationState, SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use shared::api::output::{DecodeState, OutputDecodeState, OutputDecodeStopReason};
pub use shared::artifacts::integrity_mode::RasterIntegrityMode;

// Asset loaders.
pub use io::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path,
};

// Legacy entry points (deprecated shims over `runtime::sequence::run`).
#[allow(deprecated)]
pub use runtime::inference::{run_inference, run_inference_with_controls};

// ---------------------------------------------------------------------------
// Compatibility re-exports
//
// Kept for one release cycle so existing imports continue to work; prefer
// the module paths. Gemma model-family types live under
// `shared::model::gemma` (see docs/model-agnostic-layers.md).
// ---------------------------------------------------------------------------

pub use routines::{
    decode_layer_range, decode_select_token, decode_transition_finalize, input_embedding,
    output_finalize, prefill_finalize, prefill_prepare_aux, prefill_range, prefill_range_finalize,
    prompt_prepare,
};
pub use runtime::pipeline::{
    decode_step, decode_step_with_mode, run_output_decode, run_output_decode_with_mode,
    run_prefill_pass, run_prefill_pass_with_mode, run_transformer_state_transition,
    run_transformer_state_transition_for_token_ids, validate_sampling_config,
};
pub use shared::model::gemma::tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeState, GemmaTokenizerSpec,
    GemmaVocabEntry,
};
pub use shared::model::gemma::transformer::{
    Gemma4AttentionKind, Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance,
    Gemma4PleGlobalWeights, Gemma4PleLayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource,
};
pub use shared::model::gemma::Gemma4Prompt;
pub use shared::model::transformer::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, LayerKvCache, MatrixF32,
    PrefillLogits, TransformerDecodeState, TransformerDecodeStepResult, TransformerPrefillResult,
    TransformerStateTransitionState,
};
