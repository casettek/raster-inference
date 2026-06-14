//! Gemma model-family code.
//!
//! All Gemma-specific model code (transformer weight types, tokenizer,
//! weight/tokenizer loaders) lives under this module. The checkpoint
//! taxonomy, trace format, artifact commitment contract, `runtime/` layer,
//! and role APIs must remain model-agnostic; see
//! `docs/model-agnostic-layers.md`.

use serde::{Deserialize, Serialize};

use crate::shared::api::input::TextMessage;

pub mod adapter;
pub mod io;
pub mod tokenizer;
pub mod transformer;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gemma4Prompt {
    pub messages: Vec<TextMessage>,
    pub add_generation_prompt: bool,
}
