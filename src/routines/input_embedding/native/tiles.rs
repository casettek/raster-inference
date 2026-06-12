use anyhow::{bail, Result};

use crate::shared::model::transformer::{ActivationSequence, Gemma4TransformerModel};

pub fn run(prompt_token_ids: &[u32], model: &Gemma4TransformerModel) -> Result<ActivationSequence> {
    if let Some(embedding_source) = model.embedding_source.as_ref() {
        crate::io::embed_input_tokens_from_gemma_source(prompt_token_ids, embedding_source)
    } else {
        bail!("transformer state model is missing an embedding_source")
    }
}
