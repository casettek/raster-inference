//! Prompt-token committed storage reads for `input.embedding`.

use alloc::string::String;

use crate::types::{field_selector, read_prompt_selection, PromptTokenSource};

pub(crate) fn read_prompt_token_count(source: &PromptTokenSource) -> u32 {
    read_prompt_selection::<u32>(source, field_selector("token_count"), "prompt token count")
}

pub(crate) fn read_prompt_token_ids_sha256(source: &PromptTokenSource) -> String {
    read_prompt_selection::<String>(
        source,
        field_selector("token_ids_sha256"),
        "prompt token ids sha256",
    )
}

pub(crate) fn read_prompt_token_id(source: &PromptTokenSource, token_idx: u32) -> u32 {
    read_prompt_selection::<u32>(
        source,
        crate::types::field_index_selector("token_ids", token_idx),
        "prompt token id",
    )
}
