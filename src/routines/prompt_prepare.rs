use anyhow::Result;
use tokenizers::Tokenizer;

use crate::shared::api::input::{
    InferenceRequest, ModelSpec, PromptPreparationState, RasterPromptPreparationState,
};
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::trace::{trace_event, trace_scope};

use self::native::{
    build_gemma4_messages, build_prompt_commitment, decode_prompt_bytes, render_prompt,
    tokenize_prompt,
};
use self::raster::utils::{
    init_artifact_store, init_tokenize_prompt, insert_token_id_artifact, normalize_tokenize_prompt,
    prepare_raster_prompt_input_roots, store_byte_artifact, store_text_artifact,
    NORMALIZED_PROMPT_ARTIFACT_DOMAIN, NORMALIZED_PROMPT_ARTIFACT_KIND,
    PROMPT_BYTES_ARTIFACT_DOMAIN, PROMPT_BYTES_ARTIFACT_KIND, PROMPT_TEXT_ARTIFACT_DOMAIN,
    PROMPT_TEXT_ARTIFACT_KIND, RENDERED_PROMPT_ARTIFACT_DOMAIN, RENDERED_PROMPT_ARTIFACT_KIND,
};

pub mod native;
pub mod raster;

pub fn run(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
) -> Result<PromptPreparationState> {
    let _trace = trace_scope("prompt.prepare");
    trace_event("prompt.decode_bytes");
    let prompt_text = decode_prompt_bytes(&request.prompt_bytes, request.text_decoding_policy)?;
    trace_event("prompt.build_messages");
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request.add_generation_prompt)?;
    trace_event("prompt.render");
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    trace_event("prompt.tokenize");
    let prompt_token_ids =
        tokenize_prompt(&rendered_prompt, tokenizer, request.add_special_tokens)?;
    trace_event("prompt.commitment");
    let prompt_token_ids_sha256 = build_prompt_commitment(&prompt_token_ids)?;

    Ok(PromptPreparationState {
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}

pub fn format_native_prompt_as_raster_checkpoint(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
    prompt_preparation: &PromptPreparationState,
) -> Result<RasterPromptPreparationState> {
    init_artifact_store();

    let prompt_bytes_ref = store_byte_artifact(
        "prompt-bytes",
        PROMPT_BYTES_ARTIFACT_KIND,
        PROMPT_BYTES_ARTIFACT_DOMAIN,
        &request.prompt_bytes,
    )?;
    let prompt_text_ref = store_text_artifact(
        "prompt-text",
        PROMPT_TEXT_ARTIFACT_KIND,
        PROMPT_TEXT_ARTIFACT_DOMAIN,
        &prompt_preparation.prompt_text,
    )?;
    let gemma4_prompt = build_gemma4_messages(
        &prompt_preparation.prompt_text,
        request.add_generation_prompt,
    )?;
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    let rendered_prompt_ref = store_text_artifact(
        "rendered-prompt",
        RENDERED_PROMPT_ARTIFACT_KIND,
        RENDERED_PROMPT_ARTIFACT_DOMAIN,
        &rendered_prompt,
    )?;
    let input = init_tokenize_prompt(&rendered_prompt, request.add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer)?;
    let normalized_prompt_ref = store_text_artifact(
        "normalized-prompt",
        NORMALIZED_PROMPT_ARTIFACT_KIND,
        NORMALIZED_PROMPT_ARTIFACT_DOMAIN,
        &normalized.text,
    )?;
    let token_ids_ref =
        insert_token_id_artifact("prompt-token-ids", &prompt_preparation.prompt_token_ids)?;

    Ok(RasterPromptPreparationState {
        prompt_bytes_root: prompt_bytes_ref.root().to_string(),
        prompt_text_root: prompt_text_ref.root().to_string(),
        rendered_prompt_root: rendered_prompt_ref.root().to_string(),
        normalized_prompt_root: normalized_prompt_ref.root().to_string(),
        prompt_token_ids_root: token_ids_ref.root().to_string(),
        prompt_token_count: token_ids_ref.token_count(),
    })
}

pub fn run_raster(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<raster::RasterPromptPreparationResult> {
    run_raster_with_tokenizer_controls(
        request,
        model,
        tokenizer,
        raster::DEFAULT_BPE_PAIRS_PER_TILE,
        raster::DEFAULT_BPE_PIECES_PER_TILE,
    )
}

pub fn run_raster_with_tokenizer_controls(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<raster::RasterPromptPreparationResult> {
    let prepared_inputs = prepare_raster_prompt_input_roots(
        request,
        model,
        tokenizer,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;
    raster::main(
        prepared_inputs.artifact_store_roots,
        prepared_inputs.input_roots,
    )
}
