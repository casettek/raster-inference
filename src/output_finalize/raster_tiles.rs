use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile_result, call_seq, call_tile, sequence, tile,
};
use crate::shared::{
    gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaDecodedToken, GemmaDecoderMetadataRequest,
        GemmaTokenByIdRequest,
    },
    output::{OutputDecodeState, OutputDecodeStopReason},
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OutputDetokenizeState {
    token_ids: Vec<u32>,
    next_token_idx: usize,
    text: String,
    pending_byte_fallback: Vec<u8>,
    replacement_pattern: String,
    replacement_content: String,
    byte_fallback: bool,
}

#[tile]
pub fn empty_detokenized_output(token_ids: &[u32]) -> Option<String> {
    token_ids.is_empty().then(String::new)
}

#[tile]
pub fn init_output_detokenize(
    token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDetokenizeState> {
    let metadata = auth_read!(tokenizer, GemmaDecoderMetadataRequest)?;
    if !metadata.byte_fallback {
        bail!("raster output finalize requires Gemma byte fallback decoder");
    }
    if !metadata.fuse {
        bail!("raster output finalize requires Gemma fuse decoder");
    }

    Ok(OutputDetokenizeState {
        token_ids: token_ids.to_vec(),
        next_token_idx: 0,
        text: String::new(),
        pending_byte_fallback: Vec::new(),
        replacement_pattern: metadata.replacement_pattern,
        replacement_content: metadata.replacement_content,
        byte_fallback: metadata.byte_fallback,
    })
}

#[tile(kind = recursive)]
pub fn decode_next_output_token(
    mut state: OutputDetokenizeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<(bool, OutputDetokenizeState)> {
    if state.next_token_idx >= state.token_ids.len() {
        return Ok((true, state));
    }

    let token_id = state.token_ids[state.next_token_idx];
    let token = auth_read!(tokenizer, GemmaTokenByIdRequest { token_id })?
        .with_context(|| format!("Gemma tokenizer output token id {token_id} is missing"))?;
    state.next_token_idx += 1;
    if token.special {
        return Ok((false, state));
    }

    append_decoded_token(&mut state, token)?;
    Ok((false, state))
}

fn append_decoded_token(state: &mut OutputDetokenizeState, token: GemmaDecodedToken) -> Result<()> {
    let piece = token
        .content
        .replace(&state.replacement_pattern, &state.replacement_content);
    if state.byte_fallback {
        if let Some(byte) = byte_fallback_value(&piece)? {
            state.pending_byte_fallback.push(byte);
            return Ok(());
        }
    }

    flush_pending_byte_fallback(state)?;
    state.text.push_str(&piece);
    Ok(())
}

fn byte_fallback_value(piece: &str) -> Result<Option<u8>> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        return Ok(u8::from_str_radix(&piece[3..5], 16).ok());
    }

    Ok(None)
}

fn flush_pending_byte_fallback(state: &mut OutputDetokenizeState) -> Result<()> {
    if state.pending_byte_fallback.is_empty() {
        return Ok(());
    }

    let bytes = std::mem::take(&mut state.pending_byte_fallback);
    match String::from_utf8(bytes) {
        Ok(decoded) => state.text.push_str(&decoded),
        Err(error) => {
            for _ in 0..error.into_bytes().len() {
                state.text.push('�');
            }
        }
    }
    Ok(())
}

#[tile]
pub fn finalize_output_detokenize(mut state: OutputDetokenizeState) -> Result<String> {
    if state.next_token_idx != state.token_ids.len() {
        bail!(
            "raster output finalize decoded {} tokens, expected {}",
            state.next_token_idx,
            state.token_ids.len()
        );
    }
    flush_pending_byte_fallback(&mut state)?;
    Ok(state.text)
}

#[sequence]
pub fn detokenize_output_tokens(
    token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<String> {
    if let Some(text) = call_tile!(empty_detokenized_output, token_ids) {
        return Ok(text);
    }

    let state = call_tile!(init_output_detokenize, token_ids, tokenizer)?;
    let state = call_recur_tile_result!(decode_next_output_token, state, tokenizer)?;
    call_tile!(finalize_output_detokenize, state)
}

#[tile]
pub fn build_output_decode_commitment(token_ids: &[u32]) -> Result<String> {
    let payload =
        serde_json::to_vec(token_ids).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
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
pub fn run(
    generated_token_ids: &[u32],
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDecodeState> {
    let generated_text = call_seq!(detokenize_output_tokens, generated_token_ids, tokenizer)?;
    let generated_token_ids_sha256 =
        call_tile!(build_output_decode_commitment, generated_token_ids)?;
    Ok(call_tile!(
        finalize_output_decode,
        generated_token_ids.to_vec(),
        generated_token_ids_sha256,
        generated_text
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        build_output_decode_commitment, detokenize_output_tokens, finalize_output_detokenize, run,
        OutputDetokenizeState,
    };
    use crate::shared::gemma_tokenizer::{
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
    fn finalize_output_detokenize_replaces_invalid_byte_fallback_utf8() {
        let text = finalize_output_detokenize(OutputDetokenizeState {
            token_ids: vec![6],
            next_token_idx: 1,
            text: String::new(),
            pending_byte_fallback: vec![0xC3],
            replacement_pattern: "▁".to_string(),
            replacement_content: " ".to_string(),
            byte_fallback: true,
        })
        .expect("invalid utf-8 byte fallback should be replaced");

        assert_eq!(text, "�");
    }

    #[test]
    fn detokenize_output_tokens_replaces_truncated_byte_fallback_before_normal_token() {
        let text = detokenize_output_tokens(&[6, 4, 3], &test_tokenizer_source())
            .expect("truncated byte fallback should decode with replacement");

        assert_eq!(text, "� ab");
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
