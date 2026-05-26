use super::types::*;

use anyhow::{bail, Result};

use crate::output_finalize::raster::auth_source::{
    OutputPendingBytesRef, OutputUtf8ValidationState,
};
use crate::shared::api::output::OutputDecodeStopReason;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    read_token_id_from_ref_roots, token_id_leaf, RasterArtifactId, RasterArtifactStoreRoots,
    RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma_tokenizer::{AuthenticatedGemmaTokenizer, GemmaDecodedToken};

pub fn validate_output_byte_flush_bytes_per_tile(bytes_per_tile: usize) -> Result<()> {
    if bytes_per_tile == 0 {
        bail!("raster output byte flush bytes per tile must be greater than zero");
    }
    Ok(())
}

pub(in super::super) fn append_decoded_token_with_roots(
    mut state: RasterOutputDetokenizeState,
    token: GemmaDecodedToken,
) -> Result<RasterOutputDetokenizeState> {
    let piece = token
        .content
        .replace(&state.replacement_pattern, &state.replacement_content);
    if state.byte_fallback {
        if let Some(byte) = byte_fallback_value(&piece)? {
            state = append_pending_byte_with_roots(state, byte)?;
            state
                .token_commitment
                .update_token(token.id, state.next_token_idx)?;
            state.next_token_idx += 1;
            return Ok(state);
        }
    }

    if state.pending_bytes_written > 0 {
        return begin_pending_byte_flush_with_roots(
            state,
            RasterOutputPendingFlushContinuation::ReplayCurrentToken,
        );
    }

    state = append_text_chunk_with_roots(state, &piece)?;
    state
        .token_commitment
        .update_token(token.id, state.next_token_idx)?;
    state.next_token_idx += 1;
    Ok(state)
}

pub(in super::super) fn append_pending_byte_with_roots(
    mut state: RasterOutputDetokenizeState,
    byte: u8,
) -> Result<RasterOutputDetokenizeState> {
    if state.pending_bytes_builder_source_name.is_none() {
        let source_name = pending_bytes_source_name(&state);
        let (roots, _builder) = ArtifactIo::start_builder_with_roots(
            &state.artifact_store_roots,
            RasterArtifactId::new(source_name.clone())?,
            crate::output_finalize::raster::auth_source::output_pending_bytes_metadata()?,
        )?;
        state.artifact_store_roots = roots;
        state.pending_bytes_builder_source_name = Some(source_name);
    }
    let source_name = state
        .pending_bytes_builder_source_name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("raster output pending byte builder missing"))?;
    let (roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        &state.artifact_store_roots,
        &source_name,
        state.pending_bytes_written,
        crate::output_finalize::raster::auth_source::pending_byte_leaf(byte),
    )?;
    state.artifact_store_roots = roots;
    state.pending_bytes_written += 1;
    Ok(state)
}

pub(in super::super) fn begin_pending_byte_flush_with_roots(
    mut state: RasterOutputDetokenizeState,
    continuation: RasterOutputPendingFlushContinuation,
) -> Result<RasterOutputDetokenizeState> {
    if state.pending_bytes_written == 0 {
        state.phase = pending_flush_continuation_phase(continuation);
        return Ok(state);
    }
    let source_name = state
        .pending_bytes_builder_source_name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("raster output pending byte builder missing"))?;
    let (roots, pending_ref) = ArtifactIo::finalize_builder_by_source_name_with_roots(
        &state.artifact_store_roots,
        &source_name,
    )?;
    let pending_ref = OutputPendingBytesRef::new(pending_ref)?;
    if pending_ref.byte_count() != state.pending_bytes_written {
        bail!(
            "raster output pending byte count {}, expected {}",
            pending_ref.byte_count(),
            state.pending_bytes_written
        );
    }
    state.artifact_store_roots = roots;
    state.pending_bytes_builder_source_name = None;
    state.phase = RasterOutputDetokenizePhase::ValidatePendingBytes {
        continuation,
        pending_ref,
        next_byte_idx: 0,
        validation_state: OutputUtf8ValidationState::new(),
    };
    Ok(state)
}

pub(in super::super) fn validate_pending_byte_chunk_with_roots(
    mut state: RasterOutputDetokenizeState,
    continuation: RasterOutputPendingFlushContinuation,
    pending_ref: OutputPendingBytesRef,
    next_byte_idx: usize,
    mut validation_state: OutputUtf8ValidationState,
) -> Result<RasterOutputDetokenizeState> {
    if next_byte_idx >= pending_ref.byte_count() {
        state.phase = RasterOutputDetokenizePhase::FlushPendingBytes {
            continuation,
            pending_ref,
            next_byte_idx: 0,
            valid_utf8: validation_state.is_complete(),
        };
        return Ok(state);
    }
    let end = next_byte_idx
        .saturating_add(state.byte_flush_bytes_per_tile)
        .min(pending_ref.byte_count());
    for byte_idx in next_byte_idx..end {
        let byte =
            read_pending_byte_from_ref_roots(&state.artifact_store_roots, &pending_ref, byte_idx)?;
        if !validation_state.push(byte) {
            state.phase = RasterOutputDetokenizePhase::FlushPendingBytes {
                continuation,
                pending_ref,
                next_byte_idx: 0,
                valid_utf8: false,
            };
            return Ok(state);
        }
    }
    state.phase = RasterOutputDetokenizePhase::ValidatePendingBytes {
        continuation,
        pending_ref,
        next_byte_idx: end,
        validation_state,
    };
    Ok(state)
}

pub(in super::super) fn flush_pending_byte_chunk_with_roots(
    mut state: RasterOutputDetokenizeState,
    continuation: RasterOutputPendingFlushContinuation,
    pending_ref: OutputPendingBytesRef,
    next_byte_idx: usize,
    valid_utf8: bool,
) -> Result<RasterOutputDetokenizeState> {
    if next_byte_idx >= pending_ref.byte_count() {
        state.pending_bytes_written = 0;
        state.pending_segment_idx += 1;
        state.phase = pending_flush_continuation_phase(continuation);
        return Ok(state);
    }
    let chunk_len = state
        .byte_flush_bytes_per_tile
        .min(pending_ref.byte_count() - next_byte_idx);
    let advanced = if valid_utf8 {
        let bytes = read_pending_byte_range_from_ref_roots(
            &state.artifact_store_roots,
            &pending_ref,
            next_byte_idx,
            chunk_len,
        )?;
        let mut end = bytes.len();
        while end > 0 && std::str::from_utf8(&bytes[..end]).is_err() {
            end -= 1;
        }
        if end == 0 {
            bail!("raster output UTF-8 flush could not find a valid chunk boundary");
        }
        let chunk = std::str::from_utf8(&bytes[..end])?;
        state = append_text_chunk_with_roots(state, chunk)?;
        end
    } else {
        state = append_text_chunk_with_roots(state, &INVALID_UTF8_REPLACEMENT.repeat(chunk_len))?;
        chunk_len
    };
    let next_byte_idx = next_byte_idx + advanced;
    state.phase = RasterOutputDetokenizePhase::FlushPendingBytes {
        continuation,
        pending_ref,
        next_byte_idx,
        valid_utf8,
    };
    Ok(state)
}

pub(in super::super) fn append_text_chunk_with_roots(
    mut state: RasterOutputDetokenizeState,
    chunk: &str,
) -> Result<RasterOutputDetokenizeState> {
    let (roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        &state.artifact_store_roots,
        &state.text_builder_source_name,
        state.text_chunk_count,
        crate::output_finalize::raster::auth_source::text_chunk_leaf(chunk),
    )?;
    state.artifact_store_roots = roots;
    state.text_chunk_count += 1;
    state.text_byte_len += chunk.len();
    state.text_char_count += chunk.chars().count();
    Ok(state)
}

pub(in super::super) fn read_pending_byte_from_ref_roots(
    roots: &RasterArtifactStoreRoots,
    pending_ref: &OutputPendingBytesRef,
    byte_idx: usize,
) -> Result<u8> {
    if byte_idx >= pending_ref.byte_count() {
        bail!(
            "raster output pending byte index {byte_idx} is out of range for {} bytes",
            pending_ref.byte_count()
        );
    }
    let artifact_ref = pending_ref.artifact_ref();
    let read = ArtifactIo::read_authenticated_leaf_from_roots(roots, artifact_ref, byte_idx)?;
    crate::output_finalize::raster::auth_source::decode_pending_byte_leaf(read.bytes())
}

pub(in super::super) fn read_pending_byte_range_from_ref_roots(
    roots: &RasterArtifactStoreRoots,
    pending_ref: &OutputPendingBytesRef,
    start_byte_idx: usize,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    (start_byte_idx..start_byte_idx + max_bytes)
        .map(|byte_idx| read_pending_byte_from_ref_roots(roots, pending_ref, byte_idx))
        .collect()
}

pub(in super::super) fn pending_flush_continuation_phase(
    continuation: RasterOutputPendingFlushContinuation,
) -> RasterOutputDetokenizePhase {
    match continuation {
        RasterOutputPendingFlushContinuation::ReadNextToken
        | RasterOutputPendingFlushContinuation::ReplayCurrentToken => {
            RasterOutputDetokenizePhase::ReadNextToken
        }
        RasterOutputPendingFlushContinuation::Complete => RasterOutputDetokenizePhase::Complete,
    }
}

pub(in super::super) fn pending_bytes_source_name(state: &RasterOutputDetokenizeState) -> String {
    format!(
        "{}.segment_{}",
        state.pending_bytes_source_prefix, state.pending_segment_idx
    )
}

pub(in super::super) fn byte_fallback_value(piece: &str) -> Result<Option<u8>> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        return Ok(u8::from_str_radix(&piece[3..5], 16).ok());
    }

    Ok(None)
}

pub fn prepare_raster_output_finalize_input_roots(
    generated_token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
    source_prefix: &str,
) -> Result<RasterOutputFinalizeInputRoots> {
    validate_output_byte_flush_bytes_per_tile(byte_flush_bytes_per_tile)?;
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let leaves = generated_token_ids
        .iter()
        .copied()
        .map(token_id_leaf)
        .collect::<Vec<_>>();
    let generated_source_name = format!("{source_prefix}.input.generated_token_ids");
    let (artifact_store_roots, generated_token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(generated_source_name)?,
        crate::shared::artifacts::raster_artifact_store::RasterArtifactMetadata::token_ids(
            generated_token_ids.len(),
        ),
        leaves,
    )?;
    Ok(RasterOutputFinalizeInputRoots {
        artifact_store_roots,
        generated_token_ids_ref: RasterTokenIdSequenceRef::new(generated_token_ids_ref)?,
        tokenizer_source_root: tokenizer.committed_source_ref()?.root().to_string(),
        output_text_source_name: format!("{source_prefix}.output.text"),
        pending_bytes_source_prefix: format!("{source_prefix}.output.pending_bytes"),
        byte_flush_bytes_per_tile,
        stop_reason: OutputDecodeStopReason::MaxNewTokens,
    })
}

pub fn materialize_token_ids_from_roots(
    roots: &RasterArtifactStoreRoots,
    token_ids_ref: &RasterTokenIdSequenceRef,
) -> Result<Vec<u32>> {
    (0..token_ids_ref.token_count())
        .map(|token_idx| read_token_id_from_ref_roots(roots, token_ids_ref, token_idx))
        .collect()
}
