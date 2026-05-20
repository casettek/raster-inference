use anyhow::Result;
use serde_json::json;
use tokenizers::Tokenizer;

use crate::shared::api::output::{DecodeState, OutputDecodeState};
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;

pub mod native;
pub mod raster;

pub fn run(decode_state: DecodeState, tokenizer: &Tokenizer) -> Result<OutputDecodeState> {
    let generated_token_count = decode_state.generated_token_ids.len();
    let generated_text =
        native::detokenize_output_tokens(tokenizer, &decode_state.generated_token_ids)?;
    let generated_token_ids_sha256 =
        native::build_output_decode_commitment(&decode_state.generated_token_ids)?;
    let stop_reason = crate::shared::api::output::OutputDecodeStopReason::MaxNewTokens;
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": generated_token_ids_sha256.clone(),
            "generated_text": generated_text.clone(),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason.clone(),
        }),
    );

    Ok(OutputDecodeState {
        generated_token_ids: decode_state.generated_token_ids,
        generated_token_ids_sha256,
        generated_text,
        generated_token_count,
        stop_reason,
        decode_transition_states: Vec::new(),
    })
}

pub fn run_raster(
    decode_state: DecodeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDecodeState> {
    run_raster_with_byte_flush_bytes_per_tile(
        decode_state,
        tokenizer,
        raster::DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    )
}

pub fn run_raster_with_byte_flush_bytes_per_tile(
    decode_state: DecodeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDecodeState> {
    let generated_token_count = decode_state.generated_token_ids.len();
    let output = raster::run_with_byte_flush_bytes_per_tile(
        &decode_state.generated_token_ids,
        tokenizer,
        byte_flush_bytes_per_tile,
    )?;
    let stop_reason = output.stop_reason.clone();
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": output.generated_token_ids.clone(),
            "generated_token_ids_sha256": output.generated_token_ids_sha256.clone(),
            "generated_text": output.generated_text.clone(),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason,
        }),
    );

    Ok(output)
}

pub fn run_raster_with_roots(
    decode_state: DecodeState,
    input_roots: raster::RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDecodeState> {
    let generated_token_count = input_roots.generated_token_ids_ref.token_count();
    let refs = raster::main(input_roots, tokenizer)?;
    let generated_token_ids = raster::materialize_token_ids_from_roots(
        &refs.artifact_store_roots,
        &refs.refs.generated_token_ids_ref,
    )?;
    let generated_text = raster::auth_source::materialize_text_from_roots(
        &refs.artifact_store_roots,
        &refs.refs.generated_text_ref,
    )?;
    let stop_reason = refs.refs.stop_reason.clone();
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": generated_token_ids.clone(),
            "generated_token_ids_sha256": refs.refs.generated_token_ids_sha256.clone(),
            "generated_text": generated_text.clone(),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason,
        }),
    );

    Ok(OutputDecodeState {
        generated_token_count,
        generated_token_ids,
        generated_token_ids_sha256: refs.refs.generated_token_ids_sha256,
        generated_text,
        stop_reason: refs.refs.stop_reason,
        decode_transition_states: Vec::new(),
    })
}

/// Public output boundary for the refs-first raster decode loop.
/// This is intentionally allowed to materialize token refs into `OutputDecodeState`.
pub fn run_raster_from_decode_state_refs(
    decode_state: RasterDecodeLoopState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDecodeState> {
    let Some(generated_token_ids_ref) = decode_state.generated_token_ids_ref.clone() else {
        raster::validate_output_byte_flush_bytes_per_tile(byte_flush_bytes_per_tile)?;
        let output = OutputDecodeState {
            generated_token_count: 0,
            generated_token_ids: Vec::new(),
            generated_token_ids_sha256: native::build_output_decode_commitment(&[])?,
            generated_text: String::new(),
            stop_reason: crate::shared::api::output::OutputDecodeStopReason::MaxNewTokens,
            decode_transition_states: Vec::new(),
        };
        trace_raster_output_finalize_from_refs(
            &decode_state,
            &output.generated_token_ids,
            &output,
        )?;
        return Ok(output);
    };

    let generated_token_count = generated_token_ids_ref.token_count();
    let refs = raster::main(
        raster::RasterOutputFinalizeInputRoots {
            artifact_store_roots: decode_state.artifact_store_roots.clone(),
            generated_token_ids_ref,
            tokenizer_source_root: tokenizer.committed_source_ref()?.root().to_string(),
            output_text_source_name: "output.finalize.output.text".to_string(),
            pending_bytes_source_prefix: "output.finalize.output.pending_bytes".to_string(),
            byte_flush_bytes_per_tile,
            stop_reason: crate::shared::api::output::OutputDecodeStopReason::MaxNewTokens,
        },
        tokenizer,
    )?;
    let generated_token_ids = raster::materialize_token_ids_from_roots(
        &refs.artifact_store_roots,
        &refs.refs.generated_token_ids_ref,
    )?;
    let generated_text = raster::auth_source::materialize_text_from_roots(
        &refs.artifact_store_roots,
        &refs.refs.generated_text_ref,
    )?;
    let stop_reason = refs.refs.stop_reason.clone();
    let output = OutputDecodeState {
        generated_token_count,
        generated_token_ids,
        generated_token_ids_sha256: refs.refs.generated_token_ids_sha256,
        generated_text,
        stop_reason,
        decode_transition_states: Vec::new(),
    };
    trace_raster_output_finalize_from_refs(
        &decode_state.with_roots(refs.artifact_store_roots),
        &output.generated_token_ids,
        &output,
    )?;
    Ok(output)
}

fn trace_raster_output_finalize_from_refs(
    decode_state: &RasterDecodeLoopState,
    generated_token_ids: &[u32],
    output: &OutputDecodeState,
) -> Result<()> {
    let full_token_ids = match decode_state.full_token_ids_ref.as_ref() {
        Some(full_token_ids_ref) => raster::materialize_token_ids_from_roots(
            &decode_state.artifact_store_roots,
            full_token_ids_ref,
        )?,
        None => Vec::new(),
    };
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": full_token_ids,
            "full_token_ids_sha256": crate::trace::sha256_hex(&full_token_ids),
            "generated_token_ids": generated_token_ids,
            "generated_token_ids_sha256": output.generated_token_ids_sha256.clone(),
            "generated_text": output.generated_text.clone(),
            "generated_token_count": output.generated_token_count,
            "stop_reason": output.stop_reason.clone(),
        }),
    );
    Ok(())
}

#[cfg(test)]
mod tests;
