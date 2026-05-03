use anyhow::{bail, Context, Result};

use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile_result, call_seq, call_tile, sequence, tile,
};
use crate::shared::{
    gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaDecodedToken, GemmaDecoderMetadataRequest,
        GemmaTokenByIdRequest,
    },
    output::{OutputDecodeState, OutputDecodeStopReason},
    raster_output_finalize::{
        build_output_token_ids_commitment, AuthenticatedOutputFinalizeStore,
        AuthenticatedOutputTokenIdsSource, OutputTextBuilderRef, OutputTextRef,
        OutputTokenIdRequest, OutputTokenIdsMetadataRequest, OutputTokenIdsRef,
        PendingByteBuilderRef,
    },
};

pub const DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE: usize = 16;
const INVALID_UTF8_REPLACEMENT: &str = "�";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputDetokenizeRefs {
    pub token_ids_ref: OutputTokenIdsRef,
    pub text_ref: OutputTextRef,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputDecodeRefs {
    pub generated_token_ids_ref: OutputTokenIdsRef,
    pub generated_text_ref: OutputTextRef,
    pub generated_token_ids_sha256: String,
    pub generated_token_count: usize,
    pub stop_reason: OutputDecodeStopReason,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputDetokenizeState {
    token_ids_ref: OutputTokenIdsRef,
    next_token_idx: usize,
    token_count: usize,
    text_builder_ref: OutputTextBuilderRef,
    pending_byte_builder_ref: PendingByteBuilderRef,
    replacement_pattern: String,
    replacement_content: String,
    byte_fallback: bool,
    phase: OutputDetokenizePhase,
    byte_flush_bytes_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum OutputDetokenizePhase {
    ReadNextToken,
    FlushPendingBytes {
        continuation: OutputPendingFlushContinuation,
        next_byte_idx: usize,
        byte_count: usize,
        valid_utf8: bool,
    },
    AppendText {
        chunk: String,
    },
    Complete,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum OutputPendingFlushContinuation {
    ReadNextToken,
    AppendText { chunk: String },
    Complete,
}

#[tile]
pub fn empty_detokenized_output(token_ids: &[u32]) -> Option<String> {
    token_ids.is_empty().then(String::new)
}

#[tile]
pub fn empty_detokenized_output_ref(
    token_source: &AuthenticatedOutputTokenIdsSource,
    store: &mut AuthenticatedOutputFinalizeStore,
) -> Result<Option<OutputDetokenizeRefs>> {
    let metadata = auth_read!(token_source, OutputTokenIdsMetadataRequest)?;
    if metadata.token_count != 0 {
        return Ok(None);
    }

    let text_builder_ref = store.start_text_builder("output.text")?;
    let text_ref = store.finalize_text_builder(text_builder_ref)?;
    Ok(Some(OutputDetokenizeRefs {
        token_ids_ref: metadata.token_ids_ref(),
        text_ref,
    }))
}

#[tile]
pub fn init_output_detokenize(
    token_source: &AuthenticatedOutputTokenIdsSource,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedOutputFinalizeStore,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDetokenizeState> {
    validate_output_byte_flush_bytes_per_tile(byte_flush_bytes_per_tile)?;
    let token_metadata = auth_read!(token_source, OutputTokenIdsMetadataRequest)?;
    let metadata = auth_read!(tokenizer, GemmaDecoderMetadataRequest)?;
    if !metadata.byte_fallback {
        bail!("raster output finalize requires Gemma byte fallback decoder");
    }
    if !metadata.fuse {
        bail!("raster output finalize requires Gemma fuse decoder");
    }

    Ok(OutputDetokenizeState {
        token_ids_ref: token_metadata.token_ids_ref(),
        next_token_idx: 0,
        token_count: token_metadata.token_count,
        text_builder_ref: store.start_text_builder("output.text")?,
        pending_byte_builder_ref: store.start_pending_byte_builder("output.pending_bytes")?,
        replacement_pattern: metadata.replacement_pattern,
        replacement_content: metadata.replacement_content,
        byte_fallback: metadata.byte_fallback,
        phase: OutputDetokenizePhase::ReadNextToken,
        byte_flush_bytes_per_tile,
    })
}

pub fn validate_output_byte_flush_bytes_per_tile(bytes_per_tile: usize) -> Result<()> {
    if bytes_per_tile == 0 {
        bail!("raster output byte flush bytes per tile must be greater than zero");
    }
    Ok(())
}

#[tile(kind = recursive)]
pub fn decode_next_output_token(
    mut state: OutputDetokenizeState,
    token_source: &AuthenticatedOutputTokenIdsSource,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedOutputFinalizeStore,
) -> Result<(bool, OutputDetokenizeState)> {
    match state.phase.clone() {
        OutputDetokenizePhase::ReadNextToken => {
            if state.next_token_idx >= state.token_count {
                if state.pending_byte_builder_ref.bytes_written() > 0 {
                    begin_pending_byte_flush(
                        &mut state,
                        OutputPendingFlushContinuation::Complete,
                        store,
                    )?;
                    return Ok((false, state));
                }

                state.phase = OutputDetokenizePhase::Complete;
                return Ok((true, state));
            }

            let token_id = auth_read!(
                token_source,
                OutputTokenIdRequest {
                    token_idx: state.next_token_idx
                }
            )?;
            let token =
                auth_read!(tokenizer, GemmaTokenByIdRequest { token_id })?.with_context(|| {
                    format!("Gemma tokenizer output token id {token_id} is missing")
                })?;
            state.next_token_idx += 1;
            if token.special {
                return Ok((false, state));
            }

            append_decoded_token(&mut state, token, store)?;
            Ok((false, state))
        }
        OutputDetokenizePhase::FlushPendingBytes {
            continuation,
            next_byte_idx,
            byte_count,
            valid_utf8,
        } => {
            flush_pending_byte_chunk(
                &mut state,
                continuation,
                next_byte_idx,
                byte_count,
                valid_utf8,
                store,
            )?;
            Ok((false, state))
        }
        OutputDetokenizePhase::AppendText { chunk } => {
            store.append_text_chunk(&mut state.text_builder_ref, &chunk)?;
            state.phase = OutputDetokenizePhase::ReadNextToken;
            Ok((false, state))
        }
        OutputDetokenizePhase::Complete => Ok((true, state)),
    }
}

fn append_decoded_token(
    state: &mut OutputDetokenizeState,
    token: GemmaDecodedToken,
    store: &mut AuthenticatedOutputFinalizeStore,
) -> Result<()> {
    let piece = token
        .content
        .replace(&state.replacement_pattern, &state.replacement_content);
    if state.byte_fallback {
        if let Some(byte) = byte_fallback_value(&piece)? {
            store.append_pending_byte(&mut state.pending_byte_builder_ref, byte)?;
            return Ok(());
        }
    }

    if state.pending_byte_builder_ref.bytes_written() > 0 {
        begin_pending_byte_flush(
            state,
            OutputPendingFlushContinuation::AppendText { chunk: piece },
            store,
        )?;
        return Ok(());
    }

    store.append_text_chunk(&mut state.text_builder_ref, &piece)?;
    Ok(())
}

fn byte_fallback_value(piece: &str) -> Result<Option<u8>> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        return Ok(u8::from_str_radix(&piece[3..5], 16).ok());
    }

    Ok(None)
}

fn begin_pending_byte_flush(
    state: &mut OutputDetokenizeState,
    continuation: OutputPendingFlushContinuation,
    store: &AuthenticatedOutputFinalizeStore,
) -> Result<()> {
    let byte_count = state.pending_byte_builder_ref.bytes_written();
    if byte_count == 0 {
        state.phase = continuation.into_phase();
        return Ok(());
    }

    state.phase = OutputDetokenizePhase::FlushPendingBytes {
        continuation,
        next_byte_idx: 0,
        byte_count,
        valid_utf8: store.pending_bytes_are_valid_utf8(&state.pending_byte_builder_ref)?,
    };
    Ok(())
}

#[tile]
fn flush_pending_byte_chunk(
    state: &mut OutputDetokenizeState,
    continuation: OutputPendingFlushContinuation,
    next_byte_idx: usize,
    byte_count: usize,
    valid_utf8: bool,
    store: &mut AuthenticatedOutputFinalizeStore,
) -> Result<()> {
    if byte_count == 0 {
        state.phase = continuation.into_phase();
        return Ok(());
    }
    if next_byte_idx >= byte_count {
        store.clear_pending_bytes(&mut state.pending_byte_builder_ref)?;
        state.phase = continuation.into_phase();
        return Ok(());
    }

    let max_bytes = state.byte_flush_bytes_per_tile.max(1);
    let chunk_len = max_bytes.min(byte_count - next_byte_idx);
    let advanced = if valid_utf8 {
        store.append_pending_utf8_chunk_to_text(
            &state.pending_byte_builder_ref,
            &mut state.text_builder_ref,
            next_byte_idx,
            chunk_len,
        )?
    } else {
        store.append_text_chunk(
            &mut state.text_builder_ref,
            &INVALID_UTF8_REPLACEMENT.repeat(chunk_len),
        )?;
        chunk_len
    };
    let next_byte_idx = next_byte_idx + advanced;
    if next_byte_idx < byte_count {
        state.phase = OutputDetokenizePhase::FlushPendingBytes {
            continuation,
            next_byte_idx,
            byte_count,
            valid_utf8,
        };
    } else {
        store.clear_pending_bytes(&mut state.pending_byte_builder_ref)?;
        state.phase = continuation.into_phase();
    }
    Ok(())
}

impl OutputPendingFlushContinuation {
    fn into_phase(self) -> OutputDetokenizePhase {
        match self {
            Self::ReadNextToken => OutputDetokenizePhase::ReadNextToken,
            Self::AppendText { chunk } => OutputDetokenizePhase::AppendText { chunk },
            Self::Complete => OutputDetokenizePhase::Complete,
        }
    }
}

#[tile]
pub fn finalize_output_detokenize_refs(
    state: OutputDetokenizeState,
    store: &mut AuthenticatedOutputFinalizeStore,
) -> Result<OutputDetokenizeRefs> {
    if state.next_token_idx != state.token_count {
        bail!(
            "raster output finalize decoded {} tokens, expected {}",
            state.next_token_idx,
            state.token_count
        );
    }
    if state.phase != OutputDetokenizePhase::Complete {
        bail!("raster output finalize reached incomplete detokenize phase");
    }
    if state.pending_byte_builder_ref.bytes_written() != 0 {
        bail!("raster output finalize reached completion with pending bytes");
    }

    let text_ref = store.finalize_text_builder(state.text_builder_ref)?;
    Ok(OutputDetokenizeRefs {
        token_ids_ref: state.token_ids_ref,
        text_ref,
    })
}

#[sequence]
pub fn detokenize_output_tokens_ref_with_byte_flush_bytes_per_tile(
    token_source: &AuthenticatedOutputTokenIdsSource,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedOutputFinalizeStore,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDetokenizeRefs> {
    validate_output_byte_flush_bytes_per_tile(byte_flush_bytes_per_tile)?;
    if let Some(refs) = call_tile!(empty_detokenized_output_ref, token_source, store)? {
        return Ok(refs);
    }

    let state = call_tile!(
        init_output_detokenize,
        token_source,
        tokenizer,
        store,
        byte_flush_bytes_per_tile
    )?;
    let state = call_recur_tile_result!(
        decode_next_output_token,
        state,
        token_source,
        tokenizer,
        store
    )?;
    call_tile!(finalize_output_detokenize_refs, state, store)
}

#[sequence]
pub fn detokenize_output_tokens_ref(
    token_source: &AuthenticatedOutputTokenIdsSource,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedOutputFinalizeStore,
) -> Result<OutputDetokenizeRefs> {
    detokenize_output_tokens_ref_with_byte_flush_bytes_per_tile(
        token_source,
        tokenizer,
        store,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    )
}

#[sequence]
pub fn detokenize_output_tokens_with_byte_flush_bytes_per_tile(
    token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<String> {
    validate_output_byte_flush_bytes_per_tile(byte_flush_bytes_per_tile)?;
    if let Some(text) = call_tile!(empty_detokenized_output, token_ids) {
        return Ok(text);
    }

    let token_source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", token_ids)?;
    let mut store = AuthenticatedOutputFinalizeStore::new();
    let refs = call_seq!(
        detokenize_output_tokens_ref_with_byte_flush_bytes_per_tile,
        &token_source,
        tokenizer,
        &mut store,
        byte_flush_bytes_per_tile
    )?;
    store.materialize_text(&refs.text_ref)
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

#[tile]
pub fn build_output_decode_commitment(token_ids: &[u32]) -> Result<String> {
    build_output_token_ids_commitment(token_ids)
}

#[tile]
pub fn finalize_output_decode_refs(detokenized: OutputDetokenizeRefs) -> OutputDecodeRefs {
    OutputDecodeRefs {
        generated_token_count: detokenized.token_ids_ref.token_count(),
        generated_token_ids_sha256: detokenized.token_ids_ref.det_token_ids_sha256().to_string(),
        generated_token_ids_ref: detokenized.token_ids_ref,
        generated_text_ref: detokenized.text_ref,
        stop_reason: OutputDecodeStopReason::MaxNewTokens,
    }
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
pub fn run_ref_with_store_with_byte_flush_bytes_per_tile(
    token_source: &AuthenticatedOutputTokenIdsSource,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedOutputFinalizeStore,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDecodeRefs> {
    let detokenized = call_seq!(
        detokenize_output_tokens_ref_with_byte_flush_bytes_per_tile,
        token_source,
        tokenizer,
        store,
        byte_flush_bytes_per_tile
    )?;
    Ok(call_tile!(finalize_output_decode_refs, detokenized))
}

#[sequence]
pub fn run_ref_with_store(
    token_source: &AuthenticatedOutputTokenIdsSource,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedOutputFinalizeStore,
) -> Result<OutputDecodeRefs> {
    run_ref_with_store_with_byte_flush_bytes_per_tile(
        token_source,
        tokenizer,
        store,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    )
}

#[sequence]
pub fn run_with_byte_flush_bytes_per_tile(
    generated_token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDecodeState> {
    let token_source =
        AuthenticatedOutputTokenIdsSource::from_token_ids("generated", generated_token_ids)?;
    let mut store = AuthenticatedOutputFinalizeStore::new();
    let refs = call_seq!(
        run_ref_with_store_with_byte_flush_bytes_per_tile,
        &token_source,
        tokenizer,
        &mut store,
        byte_flush_bytes_per_tile
    )?;
    let generated_token_ids = token_source.materialize_token_ids(&refs.generated_token_ids_ref)?;
    let generated_text = store.materialize_text(&refs.generated_text_ref)?;
    Ok(call_tile!(
        finalize_output_decode,
        generated_token_ids,
        refs.generated_token_ids_sha256,
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

#[cfg(test)]
mod tests {
    use super::{
        build_output_decode_commitment, decode_next_output_token, detokenize_output_tokens,
        detokenize_output_tokens_with_byte_flush_bytes_per_tile, init_output_detokenize, run,
        run_ref_with_store, DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    };
    use crate::raster_authoring::{start_tile_invocation_counting, stop_tile_invocation_counting};
    use crate::shared::gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaTokenizerSpec,
        GemmaVocabEntry,
    };
    use crate::shared::raster_output_finalize::{
        AuthenticatedOutputFinalizeStore, AuthenticatedOutputTokenIdsSource,
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
    fn ref_path_returns_materializable_refs_with_stable_commitments() {
        let token_source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[4, 3])
            .expect("token source should build");
        let mut store = AuthenticatedOutputFinalizeStore::new();

        let refs = run_ref_with_store(&token_source, &test_tokenizer_source(), &mut store)
            .expect("ref path should run");

        assert_eq!(refs.generated_token_count, 2);
        assert_eq!(
            refs.generated_token_ids_sha256,
            build_output_decode_commitment(&[4, 3]).expect("commitment should build")
        );
        assert_eq!(
            token_source
                .materialize_token_ids(&refs.generated_token_ids_ref)
                .expect("token ids should materialize"),
            vec![4, 3]
        );
        assert_eq!(
            store
                .materialize_text(&refs.generated_text_ref)
                .expect("text should materialize"),
            " ab"
        );
    }

    #[test]
    fn recursive_state_serializes_refs_without_generated_values_or_text() {
        let token_ids = vec![4; 128];
        let token_source =
            AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &token_ids)
                .expect("token source should build");
        let mut store = AuthenticatedOutputFinalizeStore::new();
        let state = init_output_detokenize(
            &token_source,
            &test_tokenizer_source(),
            &mut store,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        )
        .expect("state should initialize");

        let value = serde_json::to_value(&state).expect("state should serialize");

        assert!(value.get("token_ids").is_none());
        assert!(value.get("text").is_none());
        assert!(value.get("pending_byte_fallback").is_none());
        assert!(value.get("token_ids_ref").is_some());
        assert!(value.get("text_builder_ref").is_some());
        assert!(value.get("pending_byte_builder_ref").is_some());
        assert_eq!(value["token_count"], 128);
    }

    #[test]
    fn recursive_state_does_not_embed_pending_byte_arrays() {
        let token_source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[6])
            .expect("token source should build");
        let mut store = AuthenticatedOutputFinalizeStore::new();
        let state = init_output_detokenize(
            &token_source,
            &test_tokenizer_source(),
            &mut store,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        )
        .expect("state should initialize");
        let (_done, state) =
            decode_next_output_token(state, &token_source, &test_tokenizer_source(), &mut store)
                .expect("first token should decode");

        let value = serde_json::to_value(&state).expect("state should serialize");

        assert!(value.get("pending_byte_fallback").is_none());
        assert!(value["pending_byte_builder_ref"].get("bytes").is_none());
        assert_eq!(value["pending_byte_builder_ref"]["bytes_written"], 1);
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
