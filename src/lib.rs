pub mod dsl;
pub mod io;
pub mod routines;
pub mod runtime;
pub mod shared;

// ---------------------------------------------------------------------------
// Stable surface
//
// Protocol-facing types and entry points. Everything else is internal
// machinery reachable under its module path.
// ---------------------------------------------------------------------------

// Role entry points (claimer / challenger / detour) and audit report types.
pub use runtime::roles::claimer::{ClaimerOptions, ClaimerOutcome, ClaimerRunOutcome};
pub use runtime::roles::detour::DetourOutcome;
pub use runtime::roles::{challenger, claimer, detour, ExecutionTuning};
pub use shared::api::audit::{
    AuditOutcome, CheckpointDivergence, ClaimedTrace, ClaimedTraceEntry, DetourArtifact,
};
pub use shared::api::protocol;
pub use shared::model::gemma::adapter::GemmaModelBundle;
pub use shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
pub use shared::model::runtime::LoadedModel;

// Inference sequencing: request/outcome types and the sequence engine.
pub use runtime::inference::{
    InferenceControls, InferenceRunOutcome, InferenceState, InputEmbeddingState,
    PausedInferenceState, RasterPromptPreparedState, RasterSizingControls,
};
pub use runtime::sequence;

// Checkpoint taxonomy and trace artifact layer.
pub use runtime::checkpoints;
pub use runtime::checkpoints::{
    classify_checkpoint, CheckpointTaxonomy, DetourBackend, PhaseId, RasterDetourController,
    RasterDetourSpec, RoutineId,
};
pub use runtime::trace;

// Request/response API types.
pub use shared::api::input::{
    InferenceRequest, MessageRole, ModelSpec, PromptPreparationState, RasterPromptPreparationState,
    SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use shared::api::output::{DecodeState, OutputDecodeState, OutputDecodeStopReason};
pub use shared::artifacts::integrity_mode::RasterIntegrityMode;

// Asset loaders.
pub use io::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
};
