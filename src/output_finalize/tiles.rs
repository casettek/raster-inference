use anyhow::Result;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

pub fn detokenize_output_tokens(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    if token_ids.is_empty() {
        return Ok(String::new());
    }

    tokenizer
        .decode(token_ids, true)
        .map_err(anyhow::Error::msg)
}

pub fn build_output_decode_commitment(token_ids: &[u32]) -> Result<String> {
    let payload =
        serde_json::to_vec(token_ids).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}
