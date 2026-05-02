use anyhow::{bail, Context, Result};
use minijinja::{context, Environment};
use sha2::{Digest, Sha256};

use crate::raster_authoring::prelude::{
    auth_read, call_recur_seq_result, call_recur_tile_result, call_seq, call_tile, sequence, tile,
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
use crate::shared::raster_tokenizer_store::{
    AuthenticatedRasterTokenizerStore, RasterBpePairRequest, RasterBpePieceRequest,
    RasterBpePieceSequenceBuilderRef, RasterBpePieceSequenceRef, RasterTokenIdSequenceBuilderRef,
    RasterTokenizerSequenceId,
};

pub const DEFAULT_BPE_PAIRS_PER_TILE: usize = 64;
pub const DEFAULT_BPE_PIECES_PER_TILE: usize = 64;

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeMergeSelection {
    pub piece_idx: usize,
    pub merge_index: usize,
    pub merged: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanCandidate {
    pub piece_idx: usize,
    pub rank: usize,
    pub merge_index: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanState {
    pub pieces_ref: RasterBpePieceSequenceRef,
    pub piece_count: usize,
    pub next_pair_idx: usize,
    pub best_candidate: Option<GemmaBpeScanCandidate>,
    pub bpe_pairs_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyState {
    pub input_pieces_ref: RasterBpePieceSequenceRef,
    pub output_builder_ref: RasterBpePieceSequenceBuilderRef,
    pub input_piece_count: usize,
    pub merge_piece_idx: usize,
    pub merged: String,
    pub input_cursor: usize,
    pub output_cursor: usize,
    pub add_special_tokens: bool,
    pub iteration: u64,
    pub bpe_pairs_per_tile: usize,
    pub bpe_pieces_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdFinalizeState {
    pub pieces_ref: RasterBpePieceSequenceRef,
    pub piece_count: usize,
    pub token_ids_builder_ref: RasterTokenIdSequenceBuilderRef,
    pub next_piece_idx: usize,
    pub pieces_per_tile: usize,
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
pub fn init_tokenizer_store() -> AuthenticatedRasterTokenizerStore {
    AuthenticatedRasterTokenizerStore::new()
}

#[tile]
pub fn init_bpe_tokenize_prompt(
    pre_tokenized: GemmaPreTokenizedText,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedRasterTokenizerStore,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<GemmaBpeState> {
    ensure_tokenizer_controls(bpe_pairs_per_tile, bpe_pieces_per_tile)?;
    let metadata = auth_read!(tokenizer, GemmaTokenizerMetadataRequest)?;
    let mut pieces = Vec::new();
    for segment in pre_tokenized.segments {
        pieces.extend(initial_bpe_pieces(&segment, tokenizer, &metadata)?);
    }

    let pieces_ref =
        store.insert_bpe_piece_sequence(tokenizer_sequence_id("bpe-pieces-0")?, pieces)?;
    Ok(GemmaBpeState::new(
        pieces_ref,
        pre_tokenized.add_special_tokens,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    ))
}

#[tile]
pub fn init_bpe_merge_scan(state: &GemmaBpeState) -> Result<GemmaBpeScanState> {
    if state.bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }
    Ok(GemmaBpeScanState {
        pieces_ref: state.pieces_ref.clone(),
        piece_count: state.piece_count,
        next_pair_idx: 0,
        best_candidate: None,
        bpe_pairs_per_tile: state.bpe_pairs_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn scan_bpe_merge_candidates(
    mut state: GemmaBpeScanState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    tokenizer_store: &AuthenticatedRasterTokenizerStore,
) -> Result<(bool, GemmaBpeScanState)> {
    let pair_count = state.piece_count.saturating_sub(1);
    if state.next_pair_idx >= pair_count {
        return Ok((true, state));
    }
    if state.bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }

    let end_pair_idx = state
        .next_pair_idx
        .saturating_add(state.bpe_pairs_per_tile)
        .min(pair_count);
    for pair_idx in state.next_pair_idx..end_pair_idx {
        let pair = auth_read!(
            tokenizer_store,
            RasterBpePairRequest {
                pieces_ref: state.pieces_ref.clone(),
                pair_idx,
            },
        )?;
        if let Some(rule) = auth_read!(
            tokenizer,
            GemmaBpeMergeRequest {
                left: &pair.left,
                right: &pair.right,
            },
        )? {
            match &state.best_candidate {
                Some(best) if best.rank <= rule.rank => {}
                _ => {
                    state.best_candidate = Some(GemmaBpeScanCandidate {
                        piece_idx: pair_idx,
                        rank: rule.rank,
                        merge_index: rule.merge_index,
                    });
                }
            }
        }
    }

    state.next_pair_idx = end_pair_idx;
    Ok((state.next_pair_idx >= pair_count, state))
}

#[tile]
pub fn finalize_bpe_merge_scan(
    state: GemmaBpeScanState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<Option<GemmaBpeMergeSelection>> {
    let pair_count = state.piece_count.saturating_sub(1);
    if state.next_pair_idx != pair_count {
        bail!(
            "BPE merge scan finalized at pair {}, expected {pair_count}",
            state.next_pair_idx
        );
    }
    let Some(candidate) = state.best_candidate else {
        return Ok(None);
    };
    let merged = auth_read!(
        tokenizer,
        GemmaBpeMergedTokenRequest {
            merge_index: candidate.merge_index,
        },
    )?
    .with_context(|| {
        format!(
            "Gemma tokenizer BPE merge index {} is missing",
            candidate.merge_index
        )
    })?;

    Ok(Some(GemmaBpeMergeSelection {
        piece_idx: candidate.piece_idx,
        merge_index: candidate.merge_index,
        merged,
    }))
}

#[tile]
pub fn init_apply_bpe_merge(
    state: &GemmaBpeState,
    selection: GemmaBpeMergeSelection,
    store: &mut AuthenticatedRasterTokenizerStore,
) -> Result<GemmaBpeApplyState> {
    if state.bpe_pieces_per_tile == 0 {
        bail!("raster tokenizer BPE pieces per tile must be greater than zero");
    }
    if selection.piece_idx + 1 >= state.piece_count {
        bail!(
            "BPE merge index {} is out of range for {} pieces",
            selection.piece_idx,
            state.piece_count
        );
    }

    let output_builder_ref = store.start_bpe_piece_sequence_builder(
        tokenizer_sequence_id(format!("bpe-pieces-{}", state.iteration + 1))?,
        state.piece_count.saturating_sub(1),
    )?;

    Ok(GemmaBpeApplyState {
        input_pieces_ref: state.pieces_ref.clone(),
        output_builder_ref,
        input_piece_count: state.piece_count,
        merge_piece_idx: selection.piece_idx,
        merged: selection.merged,
        input_cursor: 0,
        output_cursor: 0,
        add_special_tokens: state.add_special_tokens,
        iteration: state.iteration,
        bpe_pairs_per_tile: state.bpe_pairs_per_tile,
        bpe_pieces_per_tile: state.bpe_pieces_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn apply_bpe_merge_chunk(
    mut state: GemmaBpeApplyState,
    store: &mut AuthenticatedRasterTokenizerStore,
) -> Result<(bool, GemmaBpeApplyState)> {
    if state.bpe_pieces_per_tile == 0 {
        bail!("raster tokenizer BPE pieces per tile must be greater than zero");
    }

    let max_output_cursor = state.input_piece_count.saturating_sub(1);
    if state.output_cursor >= max_output_cursor {
        return Ok((true, state));
    }

    let output_limit = state
        .output_cursor
        .saturating_add(state.bpe_pieces_per_tile)
        .min(max_output_cursor);
    while state.output_cursor < output_limit {
        if state.input_cursor == state.merge_piece_idx {
            store.append_bpe_piece(
                &mut state.output_builder_ref,
                state.output_cursor,
                state.merged.clone(),
            )?;
            state.input_cursor += 2;
            state.output_cursor += 1;
            continue;
        }

        let piece = auth_read!(
            store,
            RasterBpePieceRequest {
                pieces_ref: state.input_pieces_ref.clone(),
                piece_idx: state.input_cursor,
            },
        )?;
        store.append_bpe_piece(&mut state.output_builder_ref, state.output_cursor, piece)?;
        state.input_cursor += 1;
        state.output_cursor += 1;
    }

    Ok((state.output_cursor >= max_output_cursor, state))
}

#[tile]
pub fn finalize_apply_bpe_merge(
    state: GemmaBpeApplyState,
    store: &mut AuthenticatedRasterTokenizerStore,
) -> Result<GemmaBpeState> {
    let expected_piece_count = state.input_piece_count.saturating_sub(1);
    if state.output_cursor != expected_piece_count {
        bail!(
            "BPE merge apply finalized with {} pieces, expected {expected_piece_count}",
            state.output_cursor
        );
    }
    let pieces_ref = store.finalize_bpe_piece_sequence_builder(state.output_builder_ref)?;
    let mut next_state = GemmaBpeState::new(
        pieces_ref,
        state.add_special_tokens,
        state.bpe_pairs_per_tile,
        state.bpe_pieces_per_tile,
    );
    next_state.iteration = state.iteration + 1;
    Ok(next_state)
}

#[sequence]
pub fn merge_bpe_tokenize_prompt(
    state: GemmaBpeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    tokenizer_store: &mut AuthenticatedRasterTokenizerStore,
) -> Result<(bool, GemmaBpeState)> {
    let scan_state = call_tile!(init_bpe_merge_scan, &state)?;
    let scan_state = call_recur_tile_result!(
        scan_bpe_merge_candidates,
        scan_state,
        tokenizer,
        tokenizer_store
    )?;
    let Some(selection) = call_tile!(finalize_bpe_merge_scan, scan_state, tokenizer)? else {
        return Ok((true, state));
    };
    let apply_state = call_tile!(init_apply_bpe_merge, &state, selection, tokenizer_store)?;
    let apply_state = call_recur_tile_result!(apply_bpe_merge_chunk, apply_state, tokenizer_store)?;
    let state = call_tile!(finalize_apply_bpe_merge, apply_state, tokenizer_store)?;
    Ok((false, state))
}

#[tile]
pub fn finalize_bpe_tokenize_prompt(state: GemmaBpeState) -> Result<GemmaBpeOutput> {
    Ok(state.into_output())
}

#[tile]
pub fn init_token_id_finalization(
    output: GemmaBpeOutput,
    store: &mut AuthenticatedRasterTokenizerStore,
) -> Result<GemmaTokenIdFinalizeState> {
    let token_ids_builder_ref = store.start_token_id_sequence_builder(
        tokenizer_sequence_id("prompt-token-ids")?,
        output.piece_count,
    )?;
    Ok(GemmaTokenIdFinalizeState {
        pieces_ref: output.pieces_ref,
        piece_count: output.piece_count,
        token_ids_builder_ref,
        next_piece_idx: 0,
        pieces_per_tile: output.bpe_pieces_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn finalize_next_token_ids(
    mut state: GemmaTokenIdFinalizeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    store: &mut AuthenticatedRasterTokenizerStore,
) -> Result<(bool, GemmaTokenIdFinalizeState)> {
    if state.pieces_per_tile == 0 {
        bail!("raster tokenizer token-id pieces per tile must be greater than zero");
    }
    if state.next_piece_idx >= state.piece_count {
        return Ok((true, state));
    }

    let end_piece_idx = state
        .next_piece_idx
        .saturating_add(state.pieces_per_tile)
        .min(state.piece_count);
    for piece_idx in state.next_piece_idx..end_piece_idx {
        let piece = auth_read!(
            store,
            RasterBpePieceRequest {
                pieces_ref: state.pieces_ref.clone(),
                piece_idx,
            },
        )?;
        let token_id = auth_read!(tokenizer, GemmaTokenIdRequest { token: &piece })?
            .with_context(|| format!("Gemma tokenizer piece {piece:?} is missing from vocab"))?;
        store.append_token_id(&mut state.token_ids_builder_ref, piece_idx, token_id)?;
    }
    state.next_piece_idx = end_piece_idx;

    Ok((state.next_piece_idx >= state.piece_count, state))
}

#[tile]
pub fn finalize_tokenize_prompt(
    state: GemmaTokenIdFinalizeState,
    store: &mut AuthenticatedRasterTokenizerStore,
) -> Result<Vec<u32>> {
    if state.next_piece_idx != state.piece_count {
        bail!(
            "token-id finalization stopped at piece {}, expected {}",
            state.next_piece_idx,
            state.piece_count
        );
    }
    let token_ids_ref = store.finalize_token_id_sequence_builder(state.token_ids_builder_ref)?;
    store.materialize_token_ids(&token_ids_ref)
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

fn ensure_tokenizer_controls(bpe_pairs_per_tile: usize, bpe_pieces_per_tile: usize) -> Result<()> {
    if bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }
    if bpe_pieces_per_tile == 0 {
        bail!("raster tokenizer BPE pieces per tile must be greater than zero");
    }
    Ok(())
}

fn tokenizer_sequence_id(name: impl Into<String>) -> Result<RasterTokenizerSequenceId> {
    RasterTokenizerSequenceId::new(name)
}

#[sequence]
pub fn tokenize_prompt(
    prompt: &str,
    tokenizer: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
) -> Result<Vec<u32>> {
    call_seq!(
        tokenize_prompt_with_controls,
        prompt,
        tokenizer,
        add_special_tokens,
        DEFAULT_BPE_PAIRS_PER_TILE,
        DEFAULT_BPE_PIECES_PER_TILE
    )
}

#[sequence]
pub fn tokenize_prompt_with_controls(
    prompt: &str,
    tokenizer: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<Vec<u32>> {
    let input = call_tile!(init_tokenize_prompt, prompt, add_special_tokens)?;
    let normalized = call_tile!(normalize_tokenize_prompt, &input, tokenizer)?;
    let pre_tokenized = call_tile!(split_tokenize_prompt, normalized, tokenizer)?;
    let mut tokenizer_store = call_tile!(init_tokenizer_store);
    let state = call_tile!(
        init_bpe_tokenize_prompt,
        pre_tokenized,
        tokenizer,
        &mut tokenizer_store,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile
    )?;
    let state = call_recur_seq_result!(
        merge_bpe_tokenize_prompt,
        state,
        tokenizer,
        &mut tokenizer_store
    )?;
    let output = call_tile!(finalize_bpe_tokenize_prompt, state)?;
    let token_id_state = call_tile!(init_token_id_finalization, output, &mut tokenizer_store)?;
    let token_id_state = call_recur_tile_result!(
        finalize_next_token_ids,
        token_id_state,
        tokenizer,
        &mut tokenizer_store
    )?;
    call_tile!(
        finalize_tokenize_prompt,
        token_id_state,
        &mut tokenizer_store
    )
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
    call_seq!(
        run_with_tokenizer_controls,
        request,
        model,
        tokenizer,
        DEFAULT_BPE_PAIRS_PER_TILE,
        DEFAULT_BPE_PIECES_PER_TILE
    )
}

#[sequence]
pub fn run_with_tokenizer_controls(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
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
        tokenize_prompt_with_controls,
        &rendered_prompt,
        tokenizer,
        request.add_special_tokens,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile
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
        finalize_next_token_ids, finalize_tokenize_prompt, init_token_id_finalization,
        init_tokenize_prompt, render_prompt, tokenize_prompt, tokenize_prompt_with_controls,
    };
    use crate::shared::gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeOutput,
        GemmaPreTokenizedText, GemmaTokenizerSpec, GemmaVocabEntry,
    };
    use crate::shared::input::{MessageRole, ModelSpec, TextDecodingPolicy};
    use crate::shared::raster_tokenizer_store::{
        AuthenticatedRasterTokenizerStore, RasterTokenizerSequenceId,
    };

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
        let tokenizer = test_tokenizer_source();
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let pieces_ref = store
            .insert_bpe_piece_sequence(
                RasterTokenizerSequenceId::new("pieces").expect("sequence id"),
                vec!["a".to_string(), "ab".to_string()],
            )
            .expect("pieces should insert");
        let mut state = init_token_id_finalization(
            GemmaBpeOutput {
                pieces_ref,
                piece_count: 2,
                add_special_tokens: false,
                bpe_pieces_per_tile: 1,
            },
            &mut store,
        )
        .expect("token id finalization should init");
        loop {
            let (done, next_state) = finalize_next_token_ids(state, &tokenizer, &mut store)
                .expect("token ids should advance");
            state = next_state;
            if done {
                break;
            }
        }
        let token_ids =
            finalize_tokenize_prompt(state, &mut store).expect("token ids should finalize");

        assert_eq!(token_ids, vec![1, 3]);
    }

    #[test]
    fn tokenize_prompt_applies_recursive_bpe_merges() {
        let token_ids =
            tokenize_prompt("ab", &test_tokenizer_source(), false).expect("prompt should tokenize");

        assert_eq!(token_ids, vec![3]);
    }

    #[test]
    fn init_bpe_tokenize_prompt_returns_compact_ref_state() {
        let tokenizer = test_tokenizer_source();
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let state = super::init_bpe_tokenize_prompt(
            GemmaPreTokenizedText {
                segments: vec!["ab".to_string()],
                add_special_tokens: false,
            },
            &tokenizer,
            &mut store,
            1,
            1,
        )
        .expect("BPE init should build ref state");

        let serialized = serde_json::to_string(&state).expect("state should serialize");
        assert!(!serialized.contains(r#""a""#));
        assert!(!serialized.contains(r#""b""#));
        assert_eq!(
            store
                .materialize_bpe_pieces(&state.pieces_ref)
                .expect("pieces should materialize"),
            vec!["a", "b"]
        );
    }

    #[test]
    fn tokenize_prompt_uses_byte_fallback_for_unknown_chars() {
        let token_ids =
            tokenize_prompt("é", &test_tokenizer_source(), false).expect("prompt should tokenize");

        assert_eq!(token_ids, vec![10, 11]);
    }

    #[test]
    fn tokenize_prompt_chunk_sizes_do_not_change_results() {
        let tokenizer = test_tokenizer_source();
        let tiny_chunks =
            tokenize_prompt_with_controls("aba", &tokenizer, false, 1, 1).expect("tiny chunks");
        let larger_chunks =
            tokenize_prompt_with_controls("aba", &tokenizer, false, 8, 8).expect("larger chunks");

        assert_eq!(tiny_chunks, larger_chunks);
        assert_eq!(tiny_chunks, vec![12]);
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
                    token: "aba".to_string(),
                    id: 12,
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
            vec![
                GemmaBpeMerge {
                    left: "a".to_string(),
                    right: "b".to_string(),
                    merged: "ab".to_string(),
                    rank: 0,
                },
                GemmaBpeMerge {
                    left: "ab".to_string(),
                    right: "a".to_string(),
                    merged: "aba".to_string(),
                    rank: 1,
                },
            ],
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
