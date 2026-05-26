use anyhow::{bail, Context, Result};

use crate::dsl::prelude::{auth_read, call_recur_tile, call_seq, call_tile, sequence, tile};
use crate::output_finalize::raster::auth_source::{
    build_output_token_ids_commitment, OutputTextRef, OutputTokenIdsCommitmentState,
};
use crate::shared::api::output::{OutputDecodeState, OutputDecodeStopReason};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    read_token_id_from_ref_roots, RasterArtifactId, RasterArtifactStoreRoots,
};
use crate::shared::model::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaDecoderMetadataRequest, GemmaTokenByIdRequest,
};

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    input_roots: RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<RasterOutputFinalizeOutput> {
    detokenize_output_tokens_ref_with_roots(input_roots, tokenizer)
}

#[sequence]
pub fn detokenize_output_tokens_ref_with_roots(
    input_roots: RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<RasterOutputFinalizeOutput> {
    let (_artifact_store_roots, detokenize_state) = call_tile!(
        init_raster_output_detokenize,
        input_roots.artifact_store_roots.clone(),
        input_roots,
        tokenizer
    )?;
    let detokenize_state = call_recur_tile!(
        decode_next_output_token_with_roots,
        detokenize_state,
        tokenizer
    )?;
    let (_artifact_store_roots, refs) =
        call_tile!(finalize_raster_output_detokenize_refs, detokenize_state)?;
    Ok(RasterOutputFinalizeOutput::new(_artifact_store_roots, refs))
}

#[sequence]
pub fn run(
    generated_token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDecodeState> {
    run_with_byte_flush_bytes_per_tile(
        generated_token_ids,
        tokenizer,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    )
}

#[sequence]
pub fn run_with_byte_flush_bytes_per_tile(
    generated_token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDecodeState> {
    ArtifactIo::reset_store();
    let input_roots = prepare_raster_output_finalize_input_roots(
        generated_token_ids,
        tokenizer,
        byte_flush_bytes_per_tile,
        "output.finalize",
    )?;
    let refs = call_seq!(main, input_roots, tokenizer)?;
    let generated_token_ids = materialize_token_ids_from_roots(
        &refs.artifact_store_roots,
        &refs.refs.generated_token_ids_ref,
    )?;
    let generated_text = crate::output_finalize::raster::auth_source::materialize_text_from_roots(
        &refs.artifact_store_roots,
        &refs.refs.generated_text_ref,
    )?;
    Ok(call_tile!(
        finalize_output_decode,
        generated_token_ids,
        refs.refs.generated_token_ids_sha256,
        generated_text
    ))
}

#[sequence]
pub fn detokenize_output_tokens_with_byte_flush_bytes_per_tile(
    token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<String> {
    ArtifactIo::reset_store();
    let input_roots = prepare_raster_output_finalize_input_roots(
        token_ids,
        tokenizer,
        byte_flush_bytes_per_tile,
        "output.finalize.detokenize",
    )?;
    let refs = call_seq!(main, input_roots, tokenizer)?;
    crate::output_finalize::raster::auth_source::materialize_text_from_roots(
        &refs.artifact_store_roots,
        &refs.refs.generated_text_ref,
    )
}

#[sequence]
pub fn detokenize_output_tokens(
    token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<String> {
    detokenize_output_tokens_with_byte_flush_bytes_per_tile(
        token_ids,
        tokenizer,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    )
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
pub fn init_raster_output_detokenize(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<(RasterArtifactStoreRoots, RasterOutputDetokenizeState)> {
    validate_output_byte_flush_bytes_per_tile(input_roots.byte_flush_bytes_per_tile)?;
    let tokenizer_source_root = tokenizer.raster_source_root_for_current_integrity_mode()?;
    if input_roots.tokenizer_source_root != tokenizer_source_root {
        bail!(
            "raster output finalize tokenizer source root {} does not match input source root {}",
            tokenizer_source_root,
            input_roots.tokenizer_source_root
        );
    }
    let entry = artifact_store_roots
        .artifact_entry_for_source_name(input_roots.generated_token_ids_ref.id().source_name())?;
    if entry.root() != input_roots.generated_token_ids_ref.root() {
        bail!(
            "raster output finalize token ids root mismatch for {}",
            input_roots.generated_token_ids_ref.id().source_name()
        );
    }
    let metadata = auth_read!(tokenizer, GemmaDecoderMetadataRequest)?;
    if !metadata.byte_fallback {
        bail!("raster output finalize requires Gemma byte fallback decoder");
    }
    if !metadata.fuse {
        bail!("raster output finalize requires Gemma fuse decoder");
    }

    let (next_roots, _builder) = ArtifactIo::start_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(input_roots.output_text_source_name.clone())?,
        crate::output_finalize::raster::auth_source::output_text_metadata()?,
    )?;
    artifact_store_roots = next_roots;

    Ok((
        artifact_store_roots.clone(),
        RasterOutputDetokenizeState {
            artifact_store_roots,
            token_ids_ref: input_roots.generated_token_ids_ref.clone(),
            next_token_idx: 0,
            token_count: input_roots.generated_token_ids_ref.token_count(),
            tokenizer_source_root: input_roots.tokenizer_source_root,
            text_builder_source_name: input_roots.output_text_source_name,
            pending_bytes_source_prefix: input_roots.pending_bytes_source_prefix,
            pending_bytes_builder_source_name: None,
            pending_bytes_written: 0,
            pending_segment_idx: 0,
            text_chunk_count: 0,
            text_byte_len: 0,
            text_char_count: 0,
            token_commitment: OutputTokenIdsCommitmentState::new(),
            replacement_pattern: metadata.replacement_pattern,
            replacement_content: metadata.replacement_content,
            byte_fallback: metadata.byte_fallback,
            phase: RasterOutputDetokenizePhase::ReadNextToken,
            byte_flush_bytes_per_tile: input_roots.byte_flush_bytes_per_tile,
            stop_reason: input_roots.stop_reason,
        },
    ))
}

#[tile(kind = recursive)]
pub fn decode_next_output_token_with_roots(
    mut detokenize_state: RasterOutputDetokenizeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<(bool, RasterOutputDetokenizeState)> {
    let tokenizer_source_root = tokenizer.raster_source_root_for_current_integrity_mode()?;
    if detokenize_state.tokenizer_source_root != tokenizer_source_root {
        bail!(
            "raster output finalize tokenizer source root {} does not match state source root {}",
            tokenizer_source_root,
            detokenize_state.tokenizer_source_root
        );
    }
    match detokenize_state.phase.clone() {
        RasterOutputDetokenizePhase::ReadNextToken => {
            if detokenize_state.next_token_idx >= detokenize_state.token_count {
                if detokenize_state.pending_bytes_written > 0 {
                    detokenize_state = begin_pending_byte_flush_with_roots(
                        detokenize_state,
                        RasterOutputPendingFlushContinuation::Complete,
                    )?;
                    return Ok((false, detokenize_state));
                }

                detokenize_state.phase = RasterOutputDetokenizePhase::Complete;
                return Ok((true, detokenize_state));
            }

            let token_id = read_token_id_from_ref_roots(
                &detokenize_state.artifact_store_roots,
                &detokenize_state.token_ids_ref,
                detokenize_state.next_token_idx,
            )?;
            let token =
                auth_read!(tokenizer, GemmaTokenByIdRequest { token_id })?.with_context(|| {
                    format!("Gemma tokenizer output token id {token_id} is missing")
                })?;
            if token.special {
                detokenize_state
                    .token_commitment
                    .update_token(token_id, detokenize_state.next_token_idx)?;
                detokenize_state.next_token_idx += 1;
                return Ok((false, detokenize_state));
            }

            detokenize_state = append_decoded_token_with_roots(detokenize_state, token)?;
            Ok((false, detokenize_state))
        }
        RasterOutputDetokenizePhase::ValidatePendingBytes {
            continuation,
            pending_ref,
            next_byte_idx,
            validation_state,
        } => {
            detokenize_state = validate_pending_byte_chunk_with_roots(
                detokenize_state,
                continuation,
                pending_ref,
                next_byte_idx,
                validation_state,
            )?;
            Ok((false, detokenize_state))
        }
        RasterOutputDetokenizePhase::FlushPendingBytes {
            continuation,
            pending_ref,
            next_byte_idx,
            valid_utf8,
        } => {
            detokenize_state = flush_pending_byte_chunk_with_roots(
                detokenize_state,
                continuation,
                pending_ref,
                next_byte_idx,
                valid_utf8,
            )?;
            Ok((false, detokenize_state))
        }
        RasterOutputDetokenizePhase::Complete => Ok((true, detokenize_state)),
    }
}

#[tile]
pub fn finalize_raster_output_detokenize_refs(
    mut detokenize_state: RasterOutputDetokenizeState,
) -> Result<(RasterArtifactStoreRoots, RasterOutputFinalizeRefs)> {
    if detokenize_state.next_token_idx != detokenize_state.token_count {
        bail!(
            "raster output finalize decoded {} tokens, expected {}",
            detokenize_state.next_token_idx,
            detokenize_state.token_count
        );
    }
    if detokenize_state.phase != RasterOutputDetokenizePhase::Complete {
        bail!("raster output finalize reached incomplete detokenize phase");
    }
    if detokenize_state.pending_bytes_written != 0
        || detokenize_state.pending_bytes_builder_source_name.is_some()
    {
        bail!("raster output finalize reached completion with pending bytes");
    }

    let (roots, text_ref) = ArtifactIo::finalize_builder_by_source_name_with_roots(
        &detokenize_state.artifact_store_roots,
        &detokenize_state.text_builder_source_name,
    )?;
    detokenize_state.artifact_store_roots = roots;
    let text_commitment = text_ref.root().to_string();
    let text_ref = OutputTextRef::from_artifact(
        text_ref,
        detokenize_state.text_byte_len,
        detokenize_state.text_char_count,
        text_commitment,
    )?;
    let generated_token_ids_sha256 = detokenize_state.token_commitment.finish();

    Ok((
        detokenize_state.artifact_store_roots.clone(),
        RasterOutputFinalizeRefs {
            generated_token_ids_ref: detokenize_state.token_ids_ref,
            generated_text_ref: text_ref,
            generated_token_ids_sha256,
            generated_token_count: detokenize_state.token_count,
            stop_reason: detokenize_state.stop_reason,
        },
    ))
}

#[tile]
pub fn finalize_output_decode(
    generated_token_ids: Vec<u32>,
    generated_token_ids_sha256: String,
    generated_text: String,
) -> OutputDecodeState {
    OutputDecodeState {
        generated_token_count: generated_token_ids.len(),
        generated_token_ids,
        generated_token_ids_sha256,
        generated_text,
        stop_reason: OutputDecodeStopReason::MaxNewTokens,
        decode_transition_states: Vec::new(),
    }
}

#[tile]
pub fn build_output_decode_commitment(token_ids: &[u32]) -> Result<String> {
    build_output_token_ids_commitment(token_ids)
}
