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
    Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, RasterPromptPreparationState,
    TextDecodingPolicy, TextMessage,
};
use crate::shared::raster_artifact_store::{
    self, RasterArtifactBuilderRef, RasterArtifactId, RasterArtifactMetadata, RasterArtifactRef,
    RasterBpePieceSequenceRef, RasterTokenIdSequenceRef,
};

pub const DEFAULT_BPE_PAIRS_PER_TILE: usize = 64;
pub const DEFAULT_BPE_PIECES_PER_TILE: usize = 64;

const PROMPT_BYTES_ARTIFACT_KIND: &str = "prompt_bytes";
const PROMPT_TEXT_ARTIFACT_KIND: &str = "prompt_text";
const RENDERED_PROMPT_ARTIFACT_KIND: &str = "rendered_prompt";
const NORMALIZED_PROMPT_ARTIFACT_KIND: &str = "normalized_prompt";

const PROMPT_BYTES_ARTIFACT_DOMAIN: &str = "raster-artifact-prompt-bytes-merkle-v1";
const PROMPT_TEXT_ARTIFACT_DOMAIN: &str = "raster-artifact-prompt-text-merkle-v1";
const RENDERED_PROMPT_ARTIFACT_DOMAIN: &str = "raster-artifact-rendered-prompt-merkle-v1";
const NORMALIZED_PROMPT_ARTIFACT_DOMAIN: &str = "raster-artifact-normalized-prompt-merkle-v1";

#[derive(Debug, Clone, serde::Serialize)]
struct TemplateMessage {
    role: String,
    content: String,
}

impl From<&TextMessage> for TemplateMessage {
    fn from(message_ref: &TextMessage) -> Self {
        Self {
            role: message_ref.role.as_template_role().to_string(),
            content: message_ref.content.clone(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TokenizePromptInput {
    pub rendered_prompt: String,
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone)]
pub struct RasterPromptPreparationResult {
    pub state: RasterPromptPreparationState,
}

#[derive(Debug, Clone)]
pub struct RasterTokenizationResult {
    pub token_ids_ref: RasterTokenIdSequenceRef,
}

#[derive(Debug, Clone)]
struct RasterPromptBootstrap {
    prompt_bytes_ref: RasterArtifactRef,
    prompt_text_ref: RasterArtifactRef,
    rendered_prompt_ref: RasterArtifactRef,
    normalized_prompt_ref: RasterArtifactRef,
    bpe_state: GemmaBpeState,
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
    pub output_builder_ref: RasterArtifactBuilderRef,
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
    pub token_ids_builder_ref: RasterArtifactBuilderRef,
    pub next_piece_idx: usize,
    pub pieces_per_tile: usize,
}

pub fn decode_prompt_bytes(prompt_bytes_ref: &[u8], policy: TextDecodingPolicy) -> Result<String> {
    match policy {
        TextDecodingPolicy::Utf8 => String::from_utf8(prompt_bytes_ref.to_vec())
            .context("failed to decode prompt bytes as utf-8"),
    }
}

pub fn build_gemma4_messages(
    prompt_text_ref: &str,
    add_generation_prompt: bool,
) -> Result<Gemma4Prompt> {
    if prompt_text_ref.is_empty() {
        bail!("input embedding requires a non-empty prompt");
    }

    Ok(Gemma4Prompt {
        messages: vec![TextMessage {
            role: MessageRole::User,
            content: prompt_text_ref.to_string(),
        }],
        add_generation_prompt,
    })
}

pub fn render_prompt(prompt_ref: &Gemma4Prompt, model_ref: &ModelSpec) -> Result<String> {
    let mut environment = Environment::new();
    environment
        .add_template("chat", &model_ref.chat_template)
        .context("failed to register chat template")?;

    let template = environment
        .get_template("chat")
        .context("failed to load chat template")?;
    let messages = prompt_ref
        .messages
        .iter()
        .map(TemplateMessage::from)
        .collect::<Vec<_>>();

    template
        .render(context! {
            messages => messages,
            add_generation_prompt => prompt_ref.add_generation_prompt,
            bos_token => model_ref.bos_token.clone(),
            eos_token => model_ref.eos_token.clone(),
            unk_token => model_ref.unk_token.clone(),
        })
        .context("failed to render chat template")
}

pub fn init_tokenize_prompt(
    prompt_ref: &str,
    add_special_tokens: bool,
) -> Result<TokenizePromptInput> {
    Ok(TokenizePromptInput {
        rendered_prompt: prompt_ref.to_string(),
        add_special_tokens,
    })
}

pub fn normalize_tokenize_prompt(
    input_ref: &TokenizePromptInput,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
) -> Result<GemmaNormalizedText> {
    let metadata = auth_read!(tokenizer_ref, GemmaTokenizerMetadataRequest)?;

    Ok(GemmaNormalizedText {
        text: input_ref
            .rendered_prompt
            .replace(' ', &metadata.space_replacement),
        add_special_tokens: input_ref.add_special_tokens,
    })
}

pub fn split_tokenize_prompt(
    normalized: GemmaNormalizedText,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
) -> Result<GemmaPreTokenizedText> {
    let metadata = auth_read!(tokenizer_ref, GemmaTokenizerMetadataRequest)?;
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
pub fn init_artifact_store() {
    raster_artifact_store::reset_artifact_store();
}

pub fn init_bpe_tokenize_prompt(
    pre_tokenized: GemmaPreTokenizedText,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<GemmaBpeState> {
    ensure_tokenizer_controls(bpe_pairs_per_tile, bpe_pieces_per_tile)?;
    let metadata = auth_read!(tokenizer_ref, GemmaTokenizerMetadataRequest)?;
    let mut pieces = Vec::new();
    for segment in pre_tokenized.segments {
        pieces.extend(initial_bpe_pieces(&segment, tokenizer_ref, &metadata)?);
    }

    let mut pieces_builder = raster_artifact_store::start_builder(
        artifact_id("bpe-pieces-0")?,
        RasterArtifactMetadata::open_bpe_pieces(),
    )?;
    for (piece_idx, piece) in pieces.iter().enumerate() {
        raster_artifact_store::append_leaf(
            &mut pieces_builder,
            piece_idx,
            raster_artifact_store::bpe_piece_leaf(piece),
        )?;
    }
    let pieces_ref =
        RasterBpePieceSequenceRef::new(raster_artifact_store::finalize_builder(pieces_builder)?)?;
    Ok(GemmaBpeState::new(
        pieces_ref,
        pre_tokenized.add_special_tokens,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    ))
}

fn bootstrap_raster_prompt(
    request_ref: &InferenceRequest,
    model_ref: &ModelSpec,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterPromptBootstrap> {
    let prompt_bytes_ref = store_byte_artifact(
        "prompt-bytes",
        PROMPT_BYTES_ARTIFACT_KIND,
        PROMPT_BYTES_ARTIFACT_DOMAIN,
        &request_ref.prompt_bytes,
    )?;
    let prompt_text =
        decode_prompt_bytes(&request_ref.prompt_bytes, request_ref.text_decoding_policy)?;
    let prompt_text_ref = store_text_artifact(
        "prompt-text",
        PROMPT_TEXT_ARTIFACT_KIND,
        PROMPT_TEXT_ARTIFACT_DOMAIN,
        &prompt_text,
    )?;
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request_ref.add_generation_prompt)?;
    let rendered_prompt = render_prompt(&gemma4_prompt, model_ref)?;
    let rendered_prompt_ref = store_text_artifact(
        "rendered-prompt",
        RENDERED_PROMPT_ARTIFACT_KIND,
        RENDERED_PROMPT_ARTIFACT_DOMAIN,
        &rendered_prompt,
    )?;
    let input = init_tokenize_prompt(&rendered_prompt, request_ref.add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer_ref)?;
    let normalized_prompt_ref = store_text_artifact(
        "normalized-prompt",
        NORMALIZED_PROMPT_ARTIFACT_KIND,
        NORMALIZED_PROMPT_ARTIFACT_DOMAIN,
        &normalized.text,
    )?;
    let pre_tokenized = split_tokenize_prompt(normalized, tokenizer_ref)?;
    let bpe_state = init_bpe_tokenize_prompt(
        pre_tokenized,
        tokenizer_ref,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;

    Ok(RasterPromptBootstrap {
        prompt_bytes_ref,
        prompt_text_ref,
        rendered_prompt_ref,
        normalized_prompt_ref,
        bpe_state,
    })
}

#[tile]
pub fn init_bpe_merge_scan(state_ref: &GemmaBpeState) -> Result<GemmaBpeScanState> {
    if state_ref.bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }
    Ok(GemmaBpeScanState {
        pieces_ref: state_ref.pieces_ref.clone(),
        piece_count: state_ref.piece_count,
        next_pair_idx: 0,
        best_candidate: None,
        bpe_pairs_per_tile: state_ref.bpe_pairs_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn scan_bpe_merge_candidates(
    mut state: GemmaBpeScanState,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
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
        let pair = read_bpe_pair(&state.pieces_ref, pair_idx)?;
        if let Some(rule) = auth_read!(
            tokenizer_ref,
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
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
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
        tokenizer_ref,
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
    state_ref: &GemmaBpeState,
    selection: GemmaBpeMergeSelection,
) -> Result<GemmaBpeApplyState> {
    if state_ref.bpe_pieces_per_tile == 0 {
        bail!("raster tokenizer BPE pieces per tile must be greater than zero");
    }
    if selection.piece_idx + 1 >= state_ref.piece_count {
        bail!(
            "BPE merge index {} is out of range for {} pieces",
            selection.piece_idx,
            state_ref.piece_count
        );
    }

    let output_builder_ref = raster_artifact_store::start_builder(
        artifact_id(format!("bpe-pieces-{}", state_ref.iteration + 1))?,
        RasterArtifactMetadata::open_bpe_pieces(),
    )?;

    Ok(GemmaBpeApplyState {
        input_pieces_ref: state_ref.pieces_ref.clone(),
        output_builder_ref,
        input_piece_count: state_ref.piece_count,
        merge_piece_idx: selection.piece_idx,
        merged: selection.merged,
        input_cursor: 0,
        output_cursor: 0,
        add_special_tokens: state_ref.add_special_tokens,
        iteration: state_ref.iteration,
        bpe_pairs_per_tile: state_ref.bpe_pairs_per_tile,
        bpe_pieces_per_tile: state_ref.bpe_pieces_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn apply_bpe_merge_chunk(mut state: GemmaBpeApplyState) -> Result<(bool, GemmaBpeApplyState)> {
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
            raster_artifact_store::append_leaf(
                &mut state.output_builder_ref,
                state.output_cursor,
                raster_artifact_store::bpe_piece_leaf(&state.merged),
            )?;
            state.input_cursor += 2;
            state.output_cursor += 1;
            continue;
        }

        let piece = read_bpe_piece(&state.input_pieces_ref, state.input_cursor)?;
        raster_artifact_store::append_leaf(
            &mut state.output_builder_ref,
            state.output_cursor,
            raster_artifact_store::bpe_piece_leaf(&piece),
        )?;
        state.input_cursor += 1;
        state.output_cursor += 1;
    }

    Ok((state.output_cursor >= max_output_cursor, state))
}

#[tile]
pub fn finalize_apply_bpe_merge(state: GemmaBpeApplyState) -> Result<GemmaBpeState> {
    let expected_piece_count = state.input_piece_count.saturating_sub(1);
    if state.output_cursor != expected_piece_count {
        bail!(
            "BPE merge apply finalized with {} pieces, expected {expected_piece_count}",
            state.output_cursor
        );
    }
    let pieces_ref = RasterBpePieceSequenceRef::new(raster_artifact_store::finalize_builder(
        state.output_builder_ref,
    )?)?;
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
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
) -> Result<(bool, GemmaBpeState)> {
    let scan_state = call_tile!(init_bpe_merge_scan, &state)?;
    let scan_state = call_recur_tile_result!(scan_bpe_merge_candidates, scan_state, tokenizer_ref)?;
    let Some(selection) = call_tile!(finalize_bpe_merge_scan, scan_state, tokenizer_ref)? else {
        return Ok((true, state));
    };
    let apply_state = call_tile!(init_apply_bpe_merge, &state, selection)?;
    let apply_state = call_recur_tile_result!(apply_bpe_merge_chunk, apply_state)?;
    let state = call_tile!(finalize_apply_bpe_merge, apply_state)?;
    Ok((false, state))
}

#[tile]
pub fn finalize_bpe_tokenize_prompt(state: GemmaBpeState) -> Result<GemmaBpeOutput> {
    Ok(state.into_output())
}

#[tile]
pub fn init_token_id_finalization(output: GemmaBpeOutput) -> Result<GemmaTokenIdFinalizeState> {
    let token_ids_builder_ref = raster_artifact_store::start_builder(
        artifact_id("prompt-token-ids")?,
        RasterArtifactMetadata::open_token_ids(),
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
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
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
        let piece = read_bpe_piece(&state.pieces_ref, piece_idx)?;
        let token_id = auth_read!(tokenizer_ref, GemmaTokenIdRequest { token: &piece })?
            .with_context(|| format!("Gemma tokenizer piece {piece:?} is missing from vocab"))?;
        raster_artifact_store::append_leaf(
            &mut state.token_ids_builder_ref,
            piece_idx,
            raster_artifact_store::token_id_leaf(token_id),
        )?;
    }
    state.next_piece_idx = end_piece_idx;

    Ok((state.next_piece_idx >= state.piece_count, state))
}

#[tile]
pub fn finalize_tokenize_prompt(
    state: GemmaTokenIdFinalizeState,
) -> Result<RasterTokenIdSequenceRef> {
    if state.next_piece_idx != state.piece_count {
        bail!(
            "token-id finalization stopped at piece {}, expected {}",
            state.next_piece_idx,
            state.piece_count
        );
    }
    RasterTokenIdSequenceRef::new(raster_artifact_store::finalize_builder(
        state.token_ids_builder_ref,
    )?)
}

struct BpePair {
    left: String,
    right: String,
}

fn read_bpe_piece(pieces_ref: &RasterBpePieceSequenceRef, piece_idx: usize) -> Result<String> {
    let read = raster_artifact_store::read_leaf(pieces_ref.artifact_ref(), piece_idx)?;
    raster_artifact_store::verify_artifact_read(pieces_ref.artifact_ref(), &read)?;
    raster_artifact_store::decode_bpe_piece_leaf(read.payload())
}

fn read_bpe_pair(pieces_ref: &RasterBpePieceSequenceRef, pair_idx: usize) -> Result<BpePair> {
    if pair_idx >= pieces_ref.piece_count().saturating_sub(1) {
        bail!(
            "BPE pair {} is out of range for {} pieces",
            pair_idx,
            pieces_ref.piece_count()
        );
    }
    Ok(BpePair {
        left: read_bpe_piece(pieces_ref, pair_idx)?,
        right: read_bpe_piece(pieces_ref, pair_idx + 1)?,
    })
}

fn store_byte_artifact(
    name: &str,
    kind: &str,
    domain: &str,
    bytes: &[u8],
) -> Result<RasterArtifactRef> {
    let leaves = bytes.iter().map(|byte| vec![*byte]).collect::<Vec<_>>();
    raster_artifact_store::insert_artifact(
        artifact_id(name)?,
        RasterArtifactMetadata::open(kind, domain, Vec::new())?,
        leaves,
    )
}

fn store_text_artifact(
    name: &str,
    kind: &str,
    domain: &str,
    text: &str,
) -> Result<RasterArtifactRef> {
    let leaves = text.chars().map(text_char_leaf).collect::<Vec<_>>();
    raster_artifact_store::insert_artifact(
        artifact_id(name)?,
        RasterArtifactMetadata::open(kind, domain, Vec::new())?,
        leaves,
    )
}

fn text_char_leaf(ch: char) -> Vec<u8> {
    let mut buffer = [0; 4];
    let text = ch.encode_utf8(&mut buffer);
    raster_artifact_store::bpe_piece_leaf(text)
}

fn split_merged_with_previous(text_ref: &str, pattern_ref: &str) -> Vec<String> {
    if text_ref.is_empty() {
        return Vec::new();
    }
    if pattern_ref != " " {
        return vec![text_ref.to_string()];
    }

    let mut segments = Vec::new();
    let mut current = String::new();
    for ch in text_ref.chars() {
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
    segment_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    metadata_ref: &GemmaTokenizerMetadata,
) -> Result<Vec<String>> {
    let mut pieces = Vec::new();
    let mut byte_idx = 0;

    while byte_idx < segment_ref.len() {
        if let Some(token) = auth_read!(
            tokenizer_ref,
            GemmaSpecialTokenAtRequest {
                input: segment_ref,
                byte_idx,
            },
        )? {
            pieces.push(token.content.clone());
            byte_idx += token.content.len();
            continue;
        }

        let ch = segment_ref[byte_idx..]
            .chars()
            .next()
            .expect("byte_idx should point at a char boundary");
        let piece = ch.to_string();
        if auth_read!(tokenizer_ref, GemmaTokenIdRequest { token: &piece })?.is_some() {
            pieces.push(piece);
        } else if metadata_ref.byte_fallback {
            for byte in piece.as_bytes() {
                pieces.push(GemmaTokenizerSpec::byte_fallback_token(*byte));
            }
        } else {
            pieces.push(metadata_ref.unk_token.clone());
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

fn artifact_id(name: impl Into<String>) -> Result<RasterArtifactId> {
    RasterArtifactId::new(name)
}

#[sequence]
pub fn tokenize_prompt(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
) -> Result<RasterTokenizationResult> {
    call_seq!(
        tokenize_prompt_with_controls,
        prompt_ref,
        tokenizer_ref,
        add_special_tokens,
        DEFAULT_BPE_PAIRS_PER_TILE,
        DEFAULT_BPE_PIECES_PER_TILE
    )
}

#[sequence]
pub fn tokenize_prompt_with_controls(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterTokenizationResult> {
    call_tile!(init_artifact_store);
    let input = init_tokenize_prompt(prompt_ref, add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer_ref)?;
    let pre_tokenized = split_tokenize_prompt(normalized, tokenizer_ref)?;
    let state = init_bpe_tokenize_prompt(
        pre_tokenized,
        tokenizer_ref,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;
    let state = call_recur_seq_result!(merge_bpe_tokenize_prompt, state, tokenizer_ref)?;
    let output = call_tile!(finalize_bpe_tokenize_prompt, state)?;
    let token_id_state = call_tile!(init_token_id_finalization, output)?;
    let token_id_state =
        call_recur_tile_result!(finalize_next_token_ids, token_id_state, tokenizer_ref)?;
    let token_ids_ref = call_tile!(finalize_tokenize_prompt, token_id_state)?;
    Ok(RasterTokenizationResult { token_ids_ref })
}

pub fn build_prompt_commitment(prompt_token_ids_ref: &[u32]) -> Result<String> {
    let payload = serde_json::to_vec(prompt_token_ids_ref)
        .context("failed to serialize input-embedding prompt token ids")?;

    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}

#[tile]
pub fn build_prompt_commitment_from_ref(
    token_ids_ref: &RasterTokenIdSequenceRef,
) -> Result<String> {
    Ok(token_ids_ref.root().to_string())
}

#[tile]
pub fn finalize_raster_prompt_preparation(
    prompt_bytes_ref: RasterArtifactRef,
    prompt_text_ref: RasterArtifactRef,
    rendered_prompt_ref: RasterArtifactRef,
    normalized_prompt_ref: RasterArtifactRef,
    prompt_token_ids_ref: RasterTokenIdSequenceRef,
    prompt_token_ids_root: String,
) -> Result<RasterPromptPreparationState> {
    Ok(RasterPromptPreparationState {
        prompt_bytes_ref,
        prompt_text_ref,
        rendered_prompt_ref,
        normalized_prompt_ref,
        prompt_token_count: prompt_token_ids_ref.token_count(),
        prompt_token_ids_ref,
        prompt_token_ids_root,
    })
}

#[sequence]
pub fn run(
    request_ref: &InferenceRequest,
    model_ref: &ModelSpec,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
) -> Result<RasterPromptPreparationResult> {
    call_seq!(
        run_with_tokenizer_controls,
        request_ref,
        model_ref,
        tokenizer_ref,
        DEFAULT_BPE_PAIRS_PER_TILE,
        DEFAULT_BPE_PIECES_PER_TILE
    )
}

#[sequence]
pub fn run_with_tokenizer_controls(
    request_ref: &InferenceRequest,
    model_ref: &ModelSpec,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterPromptPreparationResult> {
    call_tile!(init_artifact_store);
    let bootstrap = bootstrap_raster_prompt(
        request_ref,
        model_ref,
        tokenizer_ref,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;
    let bpe_state = call_recur_seq_result!(
        merge_bpe_tokenize_prompt,
        bootstrap.bpe_state,
        tokenizer_ref
    )?;
    let output = call_tile!(finalize_bpe_tokenize_prompt, bpe_state)?;
    let token_id_state = call_tile!(init_token_id_finalization, output)?;
    let token_id_state =
        call_recur_tile_result!(finalize_next_token_ids, token_id_state, tokenizer_ref)?;
    let token_ids_ref = call_tile!(finalize_tokenize_prompt, token_id_state)?;
    let prompt_token_ids_root = call_tile!(build_prompt_commitment_from_ref, &token_ids_ref)?;

    let state = call_tile!(
        finalize_raster_prompt_preparation,
        bootstrap.prompt_bytes_ref,
        bootstrap.prompt_text_ref,
        bootstrap.rendered_prompt_ref,
        bootstrap.normalized_prompt_ref,
        token_ids_ref,
        prompt_token_ids_root
    )?;
    Ok(RasterPromptPreparationResult { state })
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{
        build_gemma4_messages, build_prompt_commitment, decode_prompt_bytes,
        finalize_next_token_ids, finalize_tokenize_prompt, init_token_id_finalization,
        init_tokenize_prompt, render_prompt, run_with_tokenizer_controls, tokenize_prompt,
        tokenize_prompt_with_controls,
    };
    use crate::shared::gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeOutput,
        GemmaPreTokenizedText, GemmaTokenizerSpec, GemmaVocabEntry,
    };
    use crate::shared::input::{MessageRole, ModelSpec, TextDecodingPolicy};
    use crate::shared::raster_artifact_store::{
        self, RasterArtifactId, RasterBpePieceSequenceRef, RasterTokenIdSequenceRef,
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
    fn finalize_tokenize_prompt_returns_token_id_ref() {
        let tokenizer = test_tokenizer_source();
        super::init_artifact_store();
        let pieces_ref =
            insert_bpe_piece_sequence_for_test("pieces", vec!["a".to_string(), "ab".to_string()])
                .expect("pieces should insert");
        let mut state = init_token_id_finalization(GemmaBpeOutput {
            pieces_ref,
            piece_count: 2,
            add_special_tokens: false,
            bpe_pieces_per_tile: 1,
        })
        .expect("token id finalization should init");
        loop {
            let (done, next_state) =
                finalize_next_token_ids(state, &tokenizer).expect("token ids should advance");
            state = next_state;
            if done {
                break;
            }
        }
        let token_ids_ref = finalize_tokenize_prompt(state).expect("token ids should finalize");
        let token_ids =
            materialize_token_ids(&token_ids_ref).expect("token ids should materialize");

        assert_eq!(token_ids, vec![1, 3]);
    }

    #[test]
    fn tokenize_prompt_applies_recursive_bpe_merges() {
        let tokenization =
            tokenize_prompt("ab", &test_tokenizer_source(), false).expect("prompt should tokenize");
        let token_ids = materialize_token_ids(&tokenization.token_ids_ref)
            .expect("token ids should materialize");

        assert_eq!(token_ids, vec![3]);
    }

    #[test]
    fn init_bpe_tokenize_prompt_returns_compact_ref_state() {
        let tokenizer = test_tokenizer_source();
        super::init_artifact_store();
        let state = super::init_bpe_tokenize_prompt(
            GemmaPreTokenizedText {
                segments: vec!["ab".to_string()],
                add_special_tokens: false,
            },
            &tokenizer,
            1,
            1,
        )
        .expect("BPE init should build ref state");

        let serialized = serde_json::to_string(&state).expect("state should serialize");
        assert!(!serialized.contains(r#""a""#));
        assert!(!serialized.contains(r#""b""#));
        assert_eq!(
            materialize_bpe_pieces(&state.pieces_ref).expect("pieces should materialize"),
            vec!["a", "b"]
        );
    }

    #[test]
    fn tokenize_prompt_uses_byte_fallback_for_unknown_chars() {
        let tokenization =
            tokenize_prompt("é", &test_tokenizer_source(), false).expect("prompt should tokenize");
        let token_ids = materialize_token_ids(&tokenization.token_ids_ref)
            .expect("token ids should materialize");

        assert_eq!(token_ids, vec![10, 11]);
    }

    #[test]
    fn tokenize_prompt_chunk_sizes_do_not_change_results() {
        let tokenizer = test_tokenizer_source();
        let tiny_chunks =
            tokenize_prompt_with_controls("aba", &tokenizer, false, 1, 1).expect("tiny chunks");
        let larger_chunks =
            tokenize_prompt_with_controls("aba", &tokenizer, false, 8, 8).expect("larger chunks");

        let tiny_token_ids = materialize_token_ids(&tiny_chunks.token_ids_ref)
            .expect("tiny token ids should materialize");
        let larger_token_ids = materialize_token_ids(&larger_chunks.token_ids_ref)
            .expect("larger token ids should materialize");
        assert_eq!(tiny_token_ids, larger_token_ids);
        assert_eq!(tiny_token_ids, vec![12]);
    }

    #[test]
    fn run_with_tokenizer_controls_returns_ref_backed_prompt_state() {
        let request = crate::shared::input::InferenceRequest {
            prompt_bytes: b"ab".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: crate::shared::input::InferenceExecutionMode::Deterministic,
            sampling: crate::shared::input::SamplingConfig::default(),
        };
        let model = ModelSpec {
            model_id: "gemma-4-test".to_string(),
            tokenizer_path: "tokenizer.json".into(),
            chat_template: "{% for message in messages %}{{ message.content }}{% endfor %}"
                .to_string(),
            bos_token: None,
            eos_token: None,
            unk_token: None,
        };

        let result = run_with_tokenizer_controls(&request, &model, &test_tokenizer_source(), 1, 1)
            .expect("raster prompt refs should build");
        let token_ids = materialize_token_ids(&result.state.prompt_token_ids_ref)
            .expect("token ids should materialize for test assertions");
        let serialized = serde_json::to_string(&result.state).expect("state should serialize");

        assert_eq!(token_ids, vec![3]);
        assert_eq!(result.state.prompt_token_count, 1);
        assert_eq!(
            result.state.prompt_token_ids_root,
            result.state.prompt_token_ids_ref.root()
        );
        assert!(!serialized.contains("prompt_token_ids\":["));
    }

    #[test]
    fn build_prompt_commitment_hashes_prompt_token_ids_only() {
        let digest = build_prompt_commitment(&[1, 2, 3]).expect("commitment should build");

        assert_eq!(
            digest,
            "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4"
        );
    }

    fn insert_bpe_piece_sequence_for_test(
        name: &str,
        pieces: Vec<String>,
    ) -> Result<RasterBpePieceSequenceRef> {
        let mut builder = raster_artifact_store::start_builder(
            RasterArtifactId::new(name).expect("artifact id"),
            raster_artifact_store::RasterArtifactMetadata::bpe_pieces(pieces.len()),
        )?;
        for (piece_idx, piece) in pieces.iter().enumerate() {
            raster_artifact_store::append_leaf(
                &mut builder,
                piece_idx,
                raster_artifact_store::bpe_piece_leaf(piece),
            )?;
        }
        RasterBpePieceSequenceRef::new(raster_artifact_store::finalize_builder(builder)?)
    }

    fn materialize_bpe_pieces(pieces_ref: &RasterBpePieceSequenceRef) -> Result<Vec<String>> {
        (0..pieces_ref.piece_count())
            .map(|piece_idx| super::read_bpe_piece(pieces_ref, piece_idx))
            .collect()
    }

    fn materialize_token_ids(token_ids_ref: &RasterTokenIdSequenceRef) -> Result<Vec<u32>> {
        (0..token_ids_ref.token_count())
            .map(|token_idx| {
                let read =
                    raster_artifact_store::read_leaf(token_ids_ref.artifact_ref(), token_idx)?;
                raster_artifact_store::verify_artifact_read(token_ids_ref.artifact_ref(), &read)?;
                raster_artifact_store::decode_token_id_leaf(read.payload())
            })
            .collect()
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
