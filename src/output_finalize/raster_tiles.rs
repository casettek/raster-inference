use anyhow::{bail, Context, Result};

use crate::output_finalize::authenticated_source::{
    build_output_token_ids_commitment, OutputPendingBytesRef, OutputTextRef,
    OutputTokenIdsCommitmentState, OutputUtf8ValidationState,
};
use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::api::output::{OutputDecodeState, OutputDecodeStopReason};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    read_token_id_from_ref_roots, token_id_leaf, RasterArtifactId, RasterArtifactStoreRoots,
    RasterRoutineOutput, RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaDecodedToken, GemmaDecoderMetadataRequest,
    GemmaTokenByIdRequest,
};

pub const DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE: usize = 16;
const INVALID_UTF8_REPLACEMENT: &str = "�";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterOutputFinalizeInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub generated_token_ids_ref: RasterTokenIdSequenceRef,
    pub tokenizer_source_root: String,
    pub output_text_source_name: String,
    pub pending_bytes_source_prefix: String,
    pub byte_flush_bytes_per_tile: usize,
    pub stop_reason: OutputDecodeStopReason,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterOutputFinalizeRefs {
    pub generated_token_ids_ref: RasterTokenIdSequenceRef,
    pub generated_text_ref: OutputTextRef,
    pub generated_token_ids_sha256: String,
    pub generated_token_count: usize,
    pub stop_reason: OutputDecodeStopReason,
}

pub type RasterOutputFinalizeOutput = RasterRoutineOutput<RasterOutputFinalizeRefs>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterOutputDetokenizeState {
    artifact_store_roots: RasterArtifactStoreRoots,
    token_ids_ref: RasterTokenIdSequenceRef,
    next_token_idx: usize,
    token_count: usize,
    tokenizer_source_root: String,
    text_builder_source_name: String,
    pending_bytes_source_prefix: String,
    pending_bytes_builder_source_name: Option<String>,
    pending_bytes_written: usize,
    pending_segment_idx: usize,
    text_chunk_count: usize,
    text_byte_len: usize,
    text_char_count: usize,
    token_commitment: OutputTokenIdsCommitmentState,
    replacement_pattern: String,
    replacement_content: String,
    byte_fallback: bool,
    phase: RasterOutputDetokenizePhase,
    byte_flush_bytes_per_tile: usize,
    stop_reason: OutputDecodeStopReason,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum RasterOutputDetokenizePhase {
    ReadNextToken,
    ValidatePendingBytes {
        continuation: RasterOutputPendingFlushContinuation,
        pending_ref: OutputPendingBytesRef,
        next_byte_idx: usize,
        validation_state: OutputUtf8ValidationState,
    },
    FlushPendingBytes {
        continuation: RasterOutputPendingFlushContinuation,
        pending_ref: OutputPendingBytesRef,
        next_byte_idx: usize,
        valid_utf8: bool,
    },
    Complete,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum RasterOutputPendingFlushContinuation {
    ReadNextToken,
    ReplayCurrentToken,
    Complete,
}

pub fn validate_output_byte_flush_bytes_per_tile(bytes_per_tile: usize) -> Result<()> {
    if bytes_per_tile == 0 {
        bail!("raster output byte flush bytes per tile must be greater than zero");
    }
    Ok(())
}

#[tile]
pub fn init_raster_output_detokenize(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<(RasterArtifactStoreRoots, RasterOutputDetokenizeState)> {
    validate_output_byte_flush_bytes_per_tile(input_roots.byte_flush_bytes_per_tile)?;
    let tokenizer_source_root = tokenizer.committed_source_ref()?.root().to_string();
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
        crate::output_finalize::authenticated_source::output_text_metadata()?,
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
    mut state: RasterOutputDetokenizeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<(bool, RasterOutputDetokenizeState)> {
    let tokenizer_source_root = tokenizer.committed_source_ref()?.root().to_string();
    if state.tokenizer_source_root != tokenizer_source_root {
        bail!(
            "raster output finalize tokenizer source root {} does not match state source root {}",
            tokenizer_source_root,
            state.tokenizer_source_root
        );
    }
    match state.phase.clone() {
        RasterOutputDetokenizePhase::ReadNextToken => {
            if state.next_token_idx >= state.token_count {
                if state.pending_bytes_written > 0 {
                    state = begin_pending_byte_flush_with_roots(
                        state,
                        RasterOutputPendingFlushContinuation::Complete,
                    )?;
                    return Ok((false, state));
                }

                state.phase = RasterOutputDetokenizePhase::Complete;
                return Ok((true, state));
            }

            let token_id = read_token_id_from_ref_roots(
                &state.artifact_store_roots,
                &state.token_ids_ref,
                state.next_token_idx,
            )?;
            let token =
                auth_read!(tokenizer, GemmaTokenByIdRequest { token_id })?.with_context(|| {
                    format!("Gemma tokenizer output token id {token_id} is missing")
                })?;
            if token.special {
                state
                    .token_commitment
                    .update_token(token_id, state.next_token_idx)?;
                state.next_token_idx += 1;
                return Ok((false, state));
            }

            state = append_decoded_token_with_roots(state, token)?;
            Ok((false, state))
        }
        RasterOutputDetokenizePhase::ValidatePendingBytes {
            continuation,
            pending_ref,
            next_byte_idx,
            validation_state,
        } => {
            state = validate_pending_byte_chunk_with_roots(
                state,
                continuation,
                pending_ref,
                next_byte_idx,
                validation_state,
            )?;
            Ok((false, state))
        }
        RasterOutputDetokenizePhase::FlushPendingBytes {
            continuation,
            pending_ref,
            next_byte_idx,
            valid_utf8,
        } => {
            state = flush_pending_byte_chunk_with_roots(
                state,
                continuation,
                pending_ref,
                next_byte_idx,
                valid_utf8,
            )?;
            Ok((false, state))
        }
        RasterOutputDetokenizePhase::Complete => Ok((true, state)),
    }
}

fn append_decoded_token_with_roots(
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

fn append_pending_byte_with_roots(
    mut state: RasterOutputDetokenizeState,
    byte: u8,
) -> Result<RasterOutputDetokenizeState> {
    if state.pending_bytes_builder_source_name.is_none() {
        let source_name = pending_bytes_source_name(&state);
        let (roots, _builder) = ArtifactIo::start_builder_with_roots(
            &state.artifact_store_roots,
            RasterArtifactId::new(source_name.clone())?,
            crate::output_finalize::authenticated_source::output_pending_bytes_metadata()?,
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
        crate::output_finalize::authenticated_source::pending_byte_leaf(byte),
    )?;
    state.artifact_store_roots = roots;
    state.pending_bytes_written += 1;
    Ok(state)
}

fn begin_pending_byte_flush_with_roots(
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

fn validate_pending_byte_chunk_with_roots(
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

fn flush_pending_byte_chunk_with_roots(
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

fn append_text_chunk_with_roots(
    mut state: RasterOutputDetokenizeState,
    chunk: &str,
) -> Result<RasterOutputDetokenizeState> {
    let (roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        &state.artifact_store_roots,
        &state.text_builder_source_name,
        state.text_chunk_count,
        crate::output_finalize::authenticated_source::text_chunk_leaf(chunk),
    )?;
    state.artifact_store_roots = roots;
    state.text_chunk_count += 1;
    state.text_byte_len += chunk.len();
    state.text_char_count += chunk.chars().count();
    Ok(state)
}

fn read_pending_byte_from_ref_roots(
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
    let entry = roots.artifact_entry_for_source_name(artifact_ref.id().source_name())?;
    if entry.root() != artifact_ref.root() {
        bail!(
            "raster output pending byte root mismatch for {}",
            artifact_ref.id().source_name()
        );
    }
    let read = ArtifactIo::read_leaf(artifact_ref, byte_idx)?;
    ArtifactIo::verify_artifact_read(artifact_ref, &read)?;
    crate::output_finalize::authenticated_source::decode_pending_byte_leaf(read.payload())
}

fn read_pending_byte_range_from_ref_roots(
    roots: &RasterArtifactStoreRoots,
    pending_ref: &OutputPendingBytesRef,
    start_byte_idx: usize,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    (start_byte_idx..start_byte_idx + max_bytes)
        .map(|byte_idx| read_pending_byte_from_ref_roots(roots, pending_ref, byte_idx))
        .collect()
}

fn pending_flush_continuation_phase(
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

fn pending_bytes_source_name(state: &RasterOutputDetokenizeState) -> String {
    format!(
        "{}.segment_{}",
        state.pending_bytes_source_prefix, state.pending_segment_idx
    )
}

#[tile]
pub fn finalize_raster_output_detokenize_refs(
    mut state: RasterOutputDetokenizeState,
) -> Result<(RasterArtifactStoreRoots, RasterOutputFinalizeRefs)> {
    if state.next_token_idx != state.token_count {
        bail!(
            "raster output finalize decoded {} tokens, expected {}",
            state.next_token_idx,
            state.token_count
        );
    }
    if state.phase != RasterOutputDetokenizePhase::Complete {
        bail!("raster output finalize reached incomplete detokenize phase");
    }
    if state.pending_bytes_written != 0 || state.pending_bytes_builder_source_name.is_some() {
        bail!("raster output finalize reached completion with pending bytes");
    }

    let (roots, text_ref) = ArtifactIo::finalize_builder_by_source_name_with_roots(
        &state.artifact_store_roots,
        &state.text_builder_source_name,
    )?;
    state.artifact_store_roots = roots;
    let text_commitment = text_ref.root().to_string();
    let text_ref = OutputTextRef::from_artifact(
        text_ref,
        state.text_byte_len,
        state.text_char_count,
        text_commitment,
    )?;
    let generated_token_ids_sha256 = state.token_commitment.finish();

    Ok((
        state.artifact_store_roots.clone(),
        RasterOutputFinalizeRefs {
            generated_token_ids_ref: state.token_ids_ref,
            generated_text_ref: text_ref,
            generated_token_ids_sha256,
            generated_token_count: state.token_count,
            stop_reason: state.stop_reason,
        },
    ))
}

fn byte_fallback_value(piece: &str) -> Result<Option<u8>> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        return Ok(u8::from_str_radix(&piece[3..5], 16).ok());
    }

    Ok(None)
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
    crate::output_finalize::authenticated_source::materialize_text_from_roots(
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

#[sequence]
pub fn detokenize_output_tokens_ref_with_roots(
    input_roots: RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<RasterOutputFinalizeOutput> {
    let (_artifact_store_roots, state) = call_tile!(
        init_raster_output_detokenize,
        input_roots.artifact_store_roots.clone(),
        input_roots,
        tokenizer
    )?;
    let state = call_recur_tile!(decode_next_output_token_with_roots, state, tokenizer)?;
    let (_artifact_store_roots, refs) = call_tile!(finalize_raster_output_detokenize_refs, state)?;
    Ok(RasterOutputFinalizeOutput::new(_artifact_store_roots, refs))
}

#[sequence]
pub fn main(
    input_roots: RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<RasterOutputFinalizeOutput> {
    detokenize_output_tokens_ref_with_roots(input_roots, tokenizer)
}

#[tile]
pub fn build_output_decode_commitment(token_ids: &[u32]) -> Result<String> {
    build_output_token_ids_commitment(token_ids)
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
    let generated_text = crate::output_finalize::authenticated_source::materialize_text_from_roots(
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

#[cfg(test)]
mod tests {
    use super::{
        build_output_decode_commitment, decode_next_output_token_with_roots,
        detokenize_output_tokens, detokenize_output_tokens_with_byte_flush_bytes_per_tile,
        init_raster_output_detokenize, main, materialize_token_ids_from_roots,
        prepare_raster_output_finalize_input_roots, run, DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    };
    use crate::output_finalize::authenticated_source::materialize_text_from_roots;
    use crate::raster_authoring::{start_tile_invocation_counting, stop_tile_invocation_counting};
    use crate::shared::artifacts::artifact_io::ArtifactIo;
    use crate::shared::artifacts::raster_artifact_store::{
        token_id_leaf, RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots,
    };
    use crate::shared::model::gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaTokenizerSpec,
        GemmaVocabEntry,
    };

    #[test]
    fn detokenize_output_tokens_returns_empty_text_for_empty_ids() {
        let text = detokenize_output_tokens(&[], &test_tokenizer_source())
            .expect("empty ids should decode");

        assert_eq!(text, "");
    }

    #[test]
    fn detokenize_output_tokens_decodes_supported_gemma_pieces() {
        let text = detokenize_output_tokens(&[4, 3], &test_tokenizer_source())
            .expect("generated ids should decode");

        assert_eq!(text, " ab");
    }

    #[test]
    fn detokenize_output_tokens_skips_special_tokens() {
        let text = detokenize_output_tokens(&[5, 4, 3], &test_tokenizer_source())
            .expect("generated ids should decode");

        assert_eq!(text, " ab");
    }

    #[test]
    fn detokenize_output_tokens_decodes_byte_fallback_sequences() {
        let text = detokenize_output_tokens(&[6, 7], &test_tokenizer_source())
            .expect("byte fallback ids should decode");

        assert_eq!(text, "é");
    }

    #[test]
    fn detokenize_output_tokens_rejects_missing_token_ids() {
        let error = detokenize_output_tokens(&[99], &test_tokenizer_source())
            .expect_err("missing token id should fail");

        assert!(error.to_string().contains("token id 99 is missing"));
    }

    #[test]
    fn detokenize_output_tokens_replaces_invalid_byte_fallback_utf8() {
        let text = detokenize_output_tokens(&[6], &test_tokenizer_source())
            .expect("invalid utf-8 byte fallback should be replaced");

        assert_eq!(text, "�");
    }

    #[test]
    fn detokenize_output_tokens_replaces_each_invalid_byte_fallback_byte() {
        let text = detokenize_output_tokens(&[6, 6], &test_tokenizer_source())
            .expect("invalid utf-8 byte fallback should be replaced");

        assert_eq!(text, "��");
    }

    #[test]
    fn detokenize_output_tokens_replaces_truncated_byte_fallback_before_normal_token() {
        let text = detokenize_output_tokens(&[6, 4, 3], &test_tokenizer_source())
            .expect("truncated byte fallback should decode with replacement");

        assert_eq!(text, "� ab");
    }

    #[test]
    fn detokenize_output_tokens_flushes_long_byte_fallback_in_chunks() {
        let token_ids = std::iter::repeat([6, 7])
            .take(20)
            .flatten()
            .collect::<Vec<_>>();

        start_tile_invocation_counting();
        let text = detokenize_output_tokens(&token_ids, &test_tokenizer_source())
            .expect("long valid byte fallback should decode");
        let invocations = stop_tile_invocation_counting().expect("tile counting should be active");

        assert_eq!(text, "é".repeat(20));
        assert!(invocations > token_ids.len() as u64);
    }

    #[test]
    fn detokenize_output_tokens_honors_byte_flush_chunk_size() {
        let token_ids = std::iter::repeat([6, 7])
            .take(20)
            .flatten()
            .collect::<Vec<_>>();

        start_tile_invocation_counting();
        let default_text = detokenize_output_tokens(&token_ids, &test_tokenizer_source())
            .expect("default chunking should decode");
        let default_invocations =
            stop_tile_invocation_counting().expect("tile counting should be active");

        start_tile_invocation_counting();
        let wide_text = detokenize_output_tokens_with_byte_flush_bytes_per_tile(
            &token_ids,
            &test_tokenizer_source(),
            64,
        )
        .expect("wide chunking should decode");
        let wide_invocations =
            stop_tile_invocation_counting().expect("tile counting should be active");

        assert_eq!(default_text, wide_text);
        assert_eq!(wide_text, "é".repeat(20));
        assert!(
            wide_invocations < default_invocations,
            "larger byte flush chunks should require fewer tile invocations"
        );
    }

    #[test]
    fn detokenize_output_tokens_rejects_zero_byte_flush_chunk_size() {
        let error = detokenize_output_tokens_with_byte_flush_bytes_per_tile(
            &[6],
            &test_tokenizer_source(),
            0,
        )
        .expect_err("zero byte flush chunk size should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn build_output_decode_commitment_hashes_generated_token_ids_only() {
        let digest = build_output_decode_commitment(&[4, 5]).expect("commitment should build");
        assert_eq!(
            digest,
            "d4c7a98da55490b0a5a65cc5057db99aa708a436609b177748505342d569457b"
        );
    }

    #[test]
    fn run_builds_output_decode_state() {
        let output = run(&[4, 3], &test_tokenizer_source()).expect("raster finalize should run");

        assert_eq!(output.generated_token_ids, vec![4, 3]);
        assert_eq!(output.generated_text, " ab");
        assert_eq!(output.generated_token_count, 2);
        assert!(output.decode_transition_states.is_empty());
    }

    #[test]
    fn root_backed_ref_path_returns_materializable_refs_with_stable_commitments() {
        ArtifactIo::reset_store();
        let tokenizer = test_tokenizer_source();
        let input_roots = prepare_raster_output_finalize_input_roots(
            &[4, 3],
            &tokenizer,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
            "output.finalize.test",
        )
        .expect("input roots should prepare");

        let refs = main(input_roots, &tokenizer).expect("root-backed finalize should run");

        assert_eq!(refs.refs.generated_token_count, 2);
        assert_eq!(
            refs.refs.generated_token_ids_sha256,
            build_output_decode_commitment(&[4, 3]).expect("commitment should build")
        );
        assert_eq!(
            materialize_token_ids_from_roots(
                &refs.artifact_store_roots,
                &refs.refs.generated_token_ids_ref
            )
            .expect("token ids should materialize"),
            vec![4, 3]
        );
        assert_eq!(
            materialize_text_from_roots(&refs.artifact_store_roots, &refs.refs.generated_text_ref)
                .expect("text should materialize"),
            " ab"
        );
        assert!(refs.refs.generated_text_ref.root().is_some());
    }

    #[test]
    fn root_backed_chunk_size_changes_do_not_change_output() {
        let token_ids = std::iter::repeat([6, 7])
            .take(20)
            .flatten()
            .collect::<Vec<_>>();
        let tokenizer = test_tokenizer_source();

        ArtifactIo::reset_store();
        let default_refs = main(
            prepare_raster_output_finalize_input_roots(
                &token_ids,
                &tokenizer,
                DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
                "output.finalize.default",
            )
            .expect("input roots should prepare"),
            &tokenizer,
        )
        .expect("default chunking should run");
        let default_text = materialize_text_from_roots(
            &default_refs.artifact_store_roots,
            &default_refs.refs.generated_text_ref,
        )
        .expect("default text should materialize");

        ArtifactIo::reset_store();
        let wide_refs = main(
            prepare_raster_output_finalize_input_roots(
                &token_ids,
                &tokenizer,
                64,
                "output.finalize.wide",
            )
            .expect("input roots should prepare"),
            &tokenizer,
        )
        .expect("wide chunking should run");
        let wide_text = materialize_text_from_roots(
            &wide_refs.artifact_store_roots,
            &wide_refs.refs.generated_text_ref,
        )
        .expect("wide text should materialize");

        assert_eq!(default_text, wide_text);
        assert_eq!(wide_text, "é".repeat(20));
        assert_eq!(
            default_refs.refs.generated_token_ids_sha256,
            wide_refs.refs.generated_token_ids_sha256
        );
    }

    #[test]
    fn root_backed_state_serializes_refs_without_payloads() {
        ArtifactIo::reset_store();
        let tokenizer = test_tokenizer_source();
        let token_ids = vec![4; 128];
        let input_roots = prepare_raster_output_finalize_input_roots(
            &token_ids,
            &tokenizer,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
            "output.finalize.state",
        )
        .expect("input roots should prepare");
        let (_roots, state) = init_raster_output_detokenize(
            input_roots.artifact_store_roots.clone(),
            input_roots,
            &tokenizer,
        )
        .expect("state should initialize");

        let value = serde_json::to_value(&state).expect("state should serialize");

        assert!(value.get("token_ids").is_none());
        assert!(value.get("generated_token_ids").is_none());
        assert!(value.get("text").is_none());
        assert!(value.get("bytes").is_none());
        assert!(value.get("token_ids_ref").is_some());
        assert!(value.get("artifact_store_roots").is_some());
        assert_eq!(value["token_count"], 128);
    }

    #[test]
    fn root_backed_state_does_not_embed_pending_byte_arrays() {
        ArtifactIo::reset_store();
        let tokenizer = test_tokenizer_source();
        let input_roots = prepare_raster_output_finalize_input_roots(
            &[6],
            &tokenizer,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
            "output.finalize.pending_state",
        )
        .expect("input roots should prepare");
        let (_roots, state) = init_raster_output_detokenize(
            input_roots.artifact_store_roots.clone(),
            input_roots,
            &tokenizer,
        )
        .expect("state should initialize");
        let (_done, state) =
            decode_next_output_token_with_roots(state, &tokenizer).expect("first token should run");

        let value = serde_json::to_value(&state).expect("state should serialize");

        assert!(value.get("pending_byte_fallback").is_none());
        assert!(value.get("bytes").is_none());
        assert_eq!(value["pending_bytes_written"], 1);
        assert!(value["pending_bytes_builder_source_name"].is_string());
    }

    #[test]
    fn root_backed_finalize_fails_closed_on_missing_stale_and_bad_routes() {
        ArtifactIo::reset_store();
        let tokenizer = test_tokenizer_source();
        let input_roots = prepare_raster_output_finalize_input_roots(
            &[4],
            &tokenizer,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
            "output.finalize.fail_closed",
        )
        .expect("input roots should prepare");

        let mut missing_roots = input_roots.clone();
        missing_roots.artifact_store_roots = RasterArtifactStoreRoots::default();
        assert!(main(missing_roots, &tokenizer)
            .expect_err("missing roots should fail")
            .to_string()
            .contains("not present"));

        let (stale_store_roots, _extra) = ArtifactIo::insert_artifact_with_roots(
            &input_roots.artifact_store_roots,
            RasterArtifactId::new("output.finalize.fail_closed.extra").expect("id"),
            RasterArtifactMetadata::token_ids(1),
            vec![token_id_leaf(9)],
        )
        .expect("extra artifact should insert");
        assert_ne!(stale_store_roots, input_roots.artifact_store_roots);
        assert!(main(input_roots.clone(), &tokenizer)
            .expect_err("stale roots should fail")
            .to_string()
            .contains("snapshot"));

        ArtifactIo::reset_store();
        let mut bad_source = prepare_raster_output_finalize_input_roots(
            &[4],
            &tokenizer,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
            "output.finalize.bad_source",
        )
        .expect("input roots should prepare");
        bad_source.tokenizer_source_root = "wrong-tokenizer-root".to_string();
        assert!(main(bad_source, &tokenizer)
            .expect_err("bad tokenizer route should fail")
            .to_string()
            .contains("does not match"));
    }

    #[test]
    fn root_backed_token_commitment_matches_decode_finalize_methodology() {
        ArtifactIo::reset_store();
        let tokenizer = test_tokenizer_source();
        let token_ids = vec![4, 3, 6, 7];
        let refs = main(
            prepare_raster_output_finalize_input_roots(
                &token_ids,
                &tokenizer,
                DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
                "output.finalize.commitment",
            )
            .expect("input roots should prepare"),
            &tokenizer,
        )
        .expect("root-backed finalize should run");

        assert_eq!(
            refs.refs.generated_token_ids_sha256,
            build_output_decode_commitment(&token_ids).expect("commitment should build")
        );
    }

    fn test_tokenizer_source() -> AuthenticatedGemmaTokenizer {
        AuthenticatedGemmaTokenizer::new(test_tokenizer_spec())
    }

    fn test_tokenizer_spec() -> GemmaTokenizerSpec {
        GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![
                GemmaVocabEntry {
                    token: "<unk>".to_string(),
                    id: 0,
                },
                GemmaVocabEntry {
                    token: "a".to_string(),
                    id: 1,
                },
                GemmaVocabEntry {
                    token: "b".to_string(),
                    id: 2,
                },
                GemmaVocabEntry {
                    token: "ab".to_string(),
                    id: 3,
                },
                GemmaVocabEntry {
                    token: "▁".to_string(),
                    id: 4,
                },
                GemmaVocabEntry {
                    token: "<bos>".to_string(),
                    id: 5,
                },
                GemmaVocabEntry {
                    token: "<0xC3>".to_string(),
                    id: 6,
                },
                GemmaVocabEntry {
                    token: "<0xA9>".to_string(),
                    id: 7,
                },
            ],
            vec![GemmaBpeMerge {
                left: "a".to_string(),
                right: "b".to_string(),
                merged: "ab".to_string(),
                rank: 0,
            }],
            vec![GemmaAddedToken {
                id: 5,
                content: "<bos>".to_string(),
                special: true,
            }],
            "<unk>".to_string(),
            true,
            "▁".to_string(),
            " ".to_string(),
        )
        .expect("test tokenizer spec should build")
    }
}
