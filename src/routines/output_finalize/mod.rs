use anyhow::Result;
use serde_json::json;
use tokenizers::Tokenizer;

use crate::runtime::checkpoints::RoutineId;
use crate::shared::api::output::{DecodeState, OutputDecodeState};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactId, RasterArtifactMetadata, RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;

pub mod native;
pub mod raster;

pub fn run(decode_state: DecodeState, tokenizer: &Tokenizer) -> Result<OutputDecodeState> {
    let _routine = crate::trace::routine_scope(
        RoutineId::FinalizeOutput,
        format!(
            "generated_tokens={}",
            decode_state.generated_token_ids.len()
        ),
    );
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

#[cfg(test)]
pub(crate) fn materialize_run_raster_for_api(
    decode_state: DecodeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDecodeState> {
    materialize_run_raster_with_byte_flush_bytes_per_tile_for_api(
        decode_state,
        tokenizer,
        raster::DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    )
}

#[cfg(test)]
pub(crate) fn materialize_run_raster_with_byte_flush_bytes_per_tile_for_api(
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

#[cfg(test)]
pub(crate) fn materialize_run_raster_with_roots_for_api(
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

pub fn run_raster(
    decode_state: RasterDecodeLoopState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<raster::RasterOutputFinalizeOutput> {
    let _routine = crate::trace::routine_scope(
        RoutineId::FinalizeOutput,
        format!(
            "mode=raster generated_tokens={}",
            decode_state.generated_token_count
        ),
    );
    let input_roots = prepare_raster_output_finalize_input_roots_from_state(
        decode_state,
        tokenizer,
        byte_flush_bytes_per_tile,
    )?;
    raster::main(input_roots, tokenizer)
}

/// Public API boundary for the refs-first raster decode loop.
/// This is intentionally allowed to materialize token/text refs into `OutputDecodeState`.
pub fn materialize_output_decode_state_for_api(
    decode_state: RasterDecodeLoopState,
    output_refs: raster::RasterOutputFinalizeOutput,
) -> Result<OutputDecodeState> {
    let generated_token_ids = raster::materialize_token_ids_from_roots(
        &output_refs.artifact_store_roots,
        &output_refs.refs.generated_token_ids_ref,
    )?;
    let generated_text = raster::auth_source::materialize_text_from_roots(
        &output_refs.artifact_store_roots,
        &output_refs.refs.generated_text_ref,
    )?;
    let stop_reason = output_refs.refs.stop_reason.clone();
    let output = OutputDecodeState {
        generated_token_count: output_refs.refs.generated_token_count,
        generated_token_ids,
        generated_token_ids_sha256: output_refs.refs.generated_token_ids_sha256,
        generated_text,
        stop_reason,
        decode_transition_states: Vec::new(),
    };
    trace_raster_output_finalize_from_refs(
        &decode_state.with_roots(output_refs.artifact_store_roots),
        &output.generated_token_ids,
        &output,
    )?;
    Ok(output)
}

fn prepare_raster_output_finalize_input_roots_from_state(
    mut decode_state: RasterDecodeLoopState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<raster::RasterOutputFinalizeInputRoots> {
    raster::validate_output_byte_flush_bytes_per_tile(byte_flush_bytes_per_tile)?;
    let generated_token_ids_ref = match decode_state.generated_token_ids_ref.take() {
        Some(generated_token_ids_ref) => generated_token_ids_ref,
        None => {
            let (roots, generated_ref) = ArtifactIo::insert_artifact_with_roots(
                &decode_state.artifact_store_roots,
                RasterArtifactId::new("output.finalize.input.generated_token_ids")?,
                RasterArtifactMetadata::token_ids(0),
                Vec::new(),
            )?;
            decode_state.artifact_store_roots = roots;
            RasterTokenIdSequenceRef::new(generated_ref)?
        }
    };

    Ok(raster::RasterOutputFinalizeInputRoots {
        artifact_store_roots: decode_state.artifact_store_roots,
        generated_token_ids_ref,
        tokenizer_source_root: tokenizer.raster_source_root_for_current_integrity_mode()?,
        output_text_source_name: "output.finalize.output.text".to_string(),
        pending_bytes_source_prefix: "output.finalize.output.pending_bytes".to_string(),
        byte_flush_bytes_per_tile,
        stop_reason: crate::shared::api::output::OutputDecodeStopReason::MaxNewTokens,
    })
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
