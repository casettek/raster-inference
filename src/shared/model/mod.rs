pub mod gemma;
pub mod transformer;

/// Compatibility alias for the pre-containment module path; new code should
/// import from `shared::model::gemma::tokenizer`.
pub use gemma::tokenizer as gemma_tokenizer;
