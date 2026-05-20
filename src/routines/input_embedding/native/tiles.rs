use anyhow::{bail, Result};

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{ActivationSequence, Gemma4TransformerModel};

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequence> {
    model.validate_execution_mode(execution_mode)?;
    if let Some(embedding_table) = model.embedding_table.as_ref() {
        crate::shared::numerics::transformer_kernels::embed_input_tokens_with_mode(
            prompt_token_ids,
            embedding_table,
            execution_mode,
        )
    } else if let Some(embedding_source) = model.embedding_source.as_ref() {
        crate::io::embed_input_tokens_from_gemma_source_with_mode(
            prompt_token_ids,
            embedding_source,
            execution_mode,
        )
    } else {
        bail!("transformer state model is missing both embedding_table and embedding_source")
    }
}
