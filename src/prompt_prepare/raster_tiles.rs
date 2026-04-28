use anyhow::{bail, Context, Result};
use minijinja::{context, Environment};
use sha2::{Digest, Sha256};

use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile_result, call_seq, call_tile, sequence, tile,
};
use crate::shared::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaBpeMergeRequest, GemmaBpeMergedTokenRequest, GemmaBpeOutput,
    GemmaBpeState, GemmaNormalizedText, GemmaPreTokenizedText, GemmaSpecialTokenAtRequest,
    GemmaTokenIdRequest, GemmaTokenizerMetadata, GemmaTokenizerMetadataRequest, GemmaTokenizerSpec,
};
use crate::shared::input::{
    Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, PromptPreparationState,
    TextDecodingPolicy, TextMessage,
};

#[derive(Debug, Clone, serde::Serialize)]
struct TemplateMessage {
    role: String,
    content: String,
}

impl From<&TextMessage> for TemplateMessage {
    fn from(message: &TextMessage) -> Self {
        Self {
            role: message.role.as_template_role().to_string(),
            content: message.content.clone(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TokenizePromptInput {
    pub rendered_prompt: String,
    pub add_special_tokens: bool,
}

#[tile]
pub fn decode_prompt_bytes(prompt_bytes: &[u8], policy: TextDecodingPolicy) -> Result<String> {
    match policy {
        TextDecodingPolicy::Utf8 => String::from_utf8(prompt_bytes.to_vec())
            .context("failed to decode prompt bytes as utf-8"),
    }
}

#[tile]
pub fn build_gemma4_messages(
    prompt_text: &str,
    add_generation_prompt: bool,
) -> Result<Gemma4Prompt> {
    if prompt_text.is_empty() {
        bail!("input embedding requires a non-empty prompt");
    }

    Ok(Gemma4Prompt {
        messages: vec![TextMessage {
            role: MessageRole::User,
            content: prompt_text.to_string(),
        }],
        add_generation_prompt,
    })
}

#[tile]
pub fn render_prompt(prompt: &Gemma4Prompt, model: &ModelSpec) -> Result<String> {
    let mut environment = Environment::new();
    environment
        .add_template("chat", &model.chat_template)
        .context("failed to register chat template")?;

    let template = environment
        .get_template("chat")
        .context("failed to load chat template")?;
    let messages = prompt
        .messages
        .iter()
        .map(TemplateMessage::from)
        .collect::<Vec<_>>();

    template
        .render(context! {
            messages => messages,
            add_generation_prompt => prompt.add_generation_prompt,
            bos_token => model.bos_token.clone(),
            eos_token => model.eos_token.clone(),
            unk_token => model.unk_token.clone(),
        })
        .context("failed to render chat template")
}

#[tile]
pub fn init_tokenize_prompt(prompt: &str, add_special_tokens: bool) -> Result<TokenizePromptInput> {
    Ok(TokenizePromptInput {
        rendered_prompt: prompt.to_string(),
        add_special_tokens,
    })
}

#[tile]
pub fn normalize_tokenize_prompt(
    input: &TokenizePromptInput,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<GemmaNormalizedText> {
    let metadata = auth_read!(tokenizer, GemmaTokenizerMetadataRequest)?;

    Ok(GemmaNormalizedText {
        text: input
            .rendered_prompt
            .replace(' ', &metadata.space_replacement),
        add_special_tokens: input.add_special_tokens,
    })
}

#[tile]
pub fn split_tokenize_prompt(
    normalized: GemmaNormalizedText,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<GemmaPreTokenizedText> {
    let metadata = auth_read!(tokenizer, GemmaTokenizerMetadataRequest)?;
    if metadata.split_pattern != " " {
        bail!(
            "Gemma tokenizer split pattern {} is not supported",
            metadata.split_pattern
        );
    }

    Ok(GemmaPreTokenizedText {
        segments: split_merged_with_previous(&normalized.text, &metadata.split_pattern),
        add_special_tokens: normalized.add_special_tokens,
    })
}

#[tile]
pub fn init_bpe_tokenize_prompt(
    pre_tokenized: GemmaPreTokenizedText,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<GemmaBpeState> {
    let metadata = auth_read!(tokenizer, GemmaTokenizerMetadataRequest)?;
    let mut pieces = Vec::new();
    for segment in pre_tokenized.segments {
        pieces.extend(initial_bpe_pieces(&segment, tokenizer, &metadata)?);
    }

    Ok(GemmaBpeState::new(pieces, pre_tokenized.add_special_tokens))
}

#[tile(kind = recursive)]
pub fn merge_bpe_tokenize_prompt(
    mut state: GemmaBpeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<(bool, GemmaBpeState)> {
    let Some((piece_idx, merged)) = best_merge_candidate(&state, tokenizer)? else {
        return Ok((true, state));
    };

    state.pieces.splice(piece_idx..=piece_idx + 1, [merged]);
    state.iteration += 1;
    Ok((false, state))
}

#[tile]
pub fn finalize_bpe_tokenize_prompt(state: GemmaBpeState) -> Result<GemmaBpeOutput> {
    Ok(state.into_output())
}

#[tile]
pub fn finalize_tokenize_prompt(
    output: GemmaBpeOutput,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<Vec<u32>> {
    let mut token_ids = Vec::with_capacity(output.pieces.len());
    for piece in output.pieces {
        let token_id = auth_read!(tokenizer, GemmaTokenIdRequest { token: &piece })?
            .with_context(|| format!("Gemma tokenizer piece {piece:?} is missing from vocab"))?;
        token_ids.push(token_id);
    }

    Ok(token_ids)
}

fn split_merged_with_previous(text: &str, pattern: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    if pattern != " " {
        return vec![text.to_string()];
    }

    let mut segments = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == ' ' {
            segments.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

fn initial_bpe_pieces(
    segment: &str,
    tokenizer: &AuthenticatedGemmaTokenizer,
    metadata: &GemmaTokenizerMetadata,
) -> Result<Vec<String>> {
    let mut pieces = Vec::new();
    let mut byte_idx = 0;

    while byte_idx < segment.len() {
        if let Some(token) = auth_read!(
            tokenizer,
            GemmaSpecialTokenAtRequest {
                input: segment,
                byte_idx,
            },
        )? {
            pieces.push(token.content.clone());
            byte_idx += token.content.len();
            continue;
        }

        let ch = segment[byte_idx..]
            .chars()
            .next()
            .expect("byte_idx should point at a char boundary");
        let piece = ch.to_string();
        if auth_read!(tokenizer, GemmaTokenIdRequest { token: &piece })?.is_some() {
            pieces.push(piece);
        } else if metadata.byte_fallback {
            for byte in piece.as_bytes() {
                pieces.push(GemmaTokenizerSpec::byte_fallback_token(*byte));
            }
        } else {
            pieces.push(metadata.unk_token.clone());
        }
        byte_idx += ch.len_utf8();
    }

    Ok(pieces)
}

fn best_merge_candidate(
    state: &GemmaBpeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<Option<(usize, String)>> {
    let mut best = None::<(usize, usize, usize)>;

    for piece_idx in 0..state.pieces.len().saturating_sub(1) {
        let left = &state.pieces[piece_idx];
        let right = &state.pieces[piece_idx + 1];
        if let Some(rule) = auth_read!(tokenizer, GemmaBpeMergeRequest { left, right },)? {
            match &best {
                Some((_, best_rank, _)) if *best_rank <= rule.rank => {}
                _ => best = Some((piece_idx, rule.rank, rule.merge_index)),
            }
        }
    }

    let Some((piece_idx, _, merge_index)) = best else {
        return Ok(None);
    };
    let merged = auth_read!(tokenizer, GemmaBpeMergedTokenRequest { merge_index })?
        .with_context(|| format!("Gemma tokenizer BPE merge index {merge_index} is missing"))?;

    Ok(Some((piece_idx, merged)))
}

#[sequence]
pub fn tokenize_prompt(
    prompt: &str,
    tokenizer: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
) -> Result<Vec<u32>> {
    let input = call_tile!(init_tokenize_prompt, prompt, add_special_tokens)?;
    let normalized = call_tile!(normalize_tokenize_prompt, &input, tokenizer)?;
    let pre_tokenized = call_tile!(split_tokenize_prompt, normalized, tokenizer)?;
    let state = call_tile!(init_bpe_tokenize_prompt, pre_tokenized, tokenizer)?;
    let state = call_recur_tile_result!(merge_bpe_tokenize_prompt, state, tokenizer)?;
    let output = call_tile!(finalize_bpe_tokenize_prompt, state)?;
    call_tile!(finalize_tokenize_prompt, output, tokenizer)
}

#[tile]
pub fn build_prompt_commitment(prompt_token_ids: &[u32]) -> Result<String> {
    let payload = serde_json::to_vec(prompt_token_ids)
        .context("failed to serialize input-embedding prompt token ids")?;

    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}

#[tile]
pub fn finalize_prompt_preparation(
    prompt_text: String,
    prompt_token_ids: Vec<u32>,
    prompt_token_ids_sha256: String,
) -> Result<PromptPreparationState> {
    Ok(PromptPreparationState {
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}

#[sequence]
pub fn run(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<PromptPreparationState> {
    let prompt_text = call_tile!(
        decode_prompt_bytes,
        &request.prompt_bytes,
        request.text_decoding_policy
    )?;
    let gemma4_prompt = call_tile!(
        build_gemma4_messages,
        &prompt_text,
        request.add_generation_prompt
    )?;
    let rendered_prompt = call_tile!(render_prompt, &gemma4_prompt, model)?;
    let prompt_token_ids = call_seq!(
        tokenize_prompt,
        &rendered_prompt,
        tokenizer,
        request.add_special_tokens
    )?;
    let prompt_token_ids_sha256 = call_tile!(build_prompt_commitment, &prompt_token_ids)?;

    call_tile!(
        finalize_prompt_preparation,
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256
    )
}

#[cfg(test)]
mod tests {
    use super::{
        build_gemma4_messages, build_prompt_commitment, decode_prompt_bytes,
        finalize_tokenize_prompt, init_tokenize_prompt, render_prompt, tokenize_prompt,
    };
    use crate::shared::gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeOutput,
        GemmaTokenizerSpec, GemmaVocabEntry,
    };
    use crate::shared::input::{MessageRole, ModelSpec, TextDecodingPolicy};

    #[test]
    fn decode_prompt_bytes_preserves_prompt_text() {
        let prompt = decode_prompt_bytes(b"  hello world  ", TextDecodingPolicy::Utf8)
            .expect("prompt should decode");

        assert_eq!(prompt, "  hello world  ");
    }

    #[test]
    fn decode_prompt_bytes_rejects_invalid_utf8() {
        let error = decode_prompt_bytes(&[0xFF], TextDecodingPolicy::Utf8)
            .expect_err("invalid utf-8 should fail");

        assert!(error.to_string().contains("utf-8"));
    }

    #[test]
    fn build_gemma4_messages_wraps_prompt_as_single_user_message() {
        let prompt = build_gemma4_messages("hello", true).expect("messages should build");

        assert_eq!(prompt.messages.len(), 1);
        assert_eq!(prompt.messages[0].role, MessageRole::User);
        assert_eq!(prompt.messages[0].content, "hello");
        assert!(prompt.add_generation_prompt);
    }

    #[test]
    fn render_prompt_uses_messages_and_generation_flag() {
        let model = ModelSpec {
            model_id: "gemma-4-test".to_string(),
            tokenizer_path: "tokenizer.json".into(),
            chat_template: "{{ bos_token }}{% for message in messages %}[{{ message.role }}] {{ message.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}".to_string(),
            bos_token: Some("<bos>".to_string()),
            eos_token: None,
            unk_token: None,
        };
        let prompt = build_gemma4_messages("hello", true).expect("messages should build");
        let prompt = render_prompt(&prompt, &model).expect("prompt should render");

        assert_eq!(prompt, "<bos>[user] hello[assistant]");
    }

    #[test]
    fn init_tokenize_prompt_captures_rendered_prompt_and_special_token_policy() {
        let input =
            init_tokenize_prompt("<bos>[user] hello[assistant]", true).expect("input should build");

        assert_eq!(input.rendered_prompt, "<bos>[user] hello[assistant]");
        assert!(input.add_special_tokens);
    }

    #[test]
    fn finalize_tokenize_prompt_returns_token_ids() {
        let token_ids = finalize_tokenize_prompt(
            GemmaBpeOutput {
                pieces: vec!["a".to_string(), "ab".to_string()],
                add_special_tokens: false,
            },
            &test_tokenizer_source(),
        )
        .expect("token ids should finalize");

        assert_eq!(token_ids, vec![1, 3]);
    }

    #[test]
    fn tokenize_prompt_applies_recursive_bpe_merges() {
        let token_ids =
            tokenize_prompt("ab", &test_tokenizer_source(), false).expect("prompt should tokenize");

        assert_eq!(token_ids, vec![3]);
    }

    #[test]
    fn tokenize_prompt_uses_byte_fallback_for_unknown_chars() {
        let token_ids =
            tokenize_prompt("é", &test_tokenizer_source(), false).expect("prompt should tokenize");

        assert_eq!(token_ids, vec![10, 11]);
    }

    #[test]
    fn build_prompt_commitment_hashes_prompt_token_ids_only() {
        let digest = build_prompt_commitment(&[1, 2, 3]).expect("commitment should build");

        assert_eq!(
            digest,
            "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4"
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
                    token: GemmaTokenizerSpec::byte_fallback_token(0xC3),
                    id: 10,
                },
                GemmaVocabEntry {
                    token: GemmaTokenizerSpec::byte_fallback_token(0xA9),
                    id: 11,
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
