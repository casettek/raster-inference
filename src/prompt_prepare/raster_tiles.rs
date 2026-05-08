use anyhow::{bail, Context, Result};
use minijinja::{context, Environment};
use sha2::{Digest, Sha256};

use crate::raster_authoring::prelude::{
    call_recur_seq_result, call_recur_tile_result, call_tile, sequence, tile,
};
use crate::shared::artifact_io::ArtifactIo;
use crate::shared::external_artifacts::{CommittedExternalSource, ExternalSourceRef};
use crate::shared::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaBpeMergeRequest, GemmaBpeMergedTokenRequest, GemmaBpeOutput,
    GemmaBpeState, GemmaNormalizedText, GemmaPreTokenizedText, GemmaTokenIdRequest,
    GemmaTokenizerMetadataRequest,
};
use crate::shared::input::{
    Gemma4Prompt, InferenceRequest, MessageRole, ModelSpec, RasterPromptPreparationState,
    TextDecodingPolicy, TextMessage,
};
use crate::shared::raster_artifact_store::{
    RasterArtifactMetadata, RasterBpePieceSequenceRef, RasterTokenIdSequenceRef,
};

use super::raster_utils::{
    artifact_id, bpe_piece_leaf, ensure_tokenizer_controls, initial_bpe_pieces, read_bpe_pair,
    read_bpe_piece, split_merged_with_previous, store_byte_artifact, store_text_artifact,
    token_id_leaf,
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPromptInputRoots {
    pub prompt_bytes_root: String,
    pub prompt_text_root: String,
    pub rendered_prompt_root: String,
    pub normalized_prompt_root: String,
    pub tokenizer_source_ref: ExternalSourceRef,
    pub bpe_state: GemmaBpeState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterTokenizationResult {
    pub token_ids_root: String,
    pub token_count: usize,
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
    pub pieces_root: String,
    pub piece_count: usize,
    pub next_pair_idx: usize,
    pub best_candidate: Option<GemmaBpeScanCandidate>,
    pub bpe_pairs_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyState {
    pub input_pieces_root: String,
    pub output_builder_root: String,
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
    pub pieces_root: String,
    pub piece_count: usize,
    pub token_ids_builder_root: String,
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
    let metadata = ArtifactIo::auth_read(tokenizer_ref, GemmaTokenizerMetadataRequest)?;

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
    let metadata = ArtifactIo::auth_read(tokenizer_ref, GemmaTokenizerMetadataRequest)?;
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
    ArtifactIo::reset_store();
}

pub fn init_bpe_tokenize_prompt(
    pre_tokenized: GemmaPreTokenizedText,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<GemmaBpeState> {
    ensure_tokenizer_controls(bpe_pairs_per_tile, bpe_pieces_per_tile)?;
    let metadata = ArtifactIo::auth_read(tokenizer_ref, GemmaTokenizerMetadataRequest)?;
    let mut pieces = Vec::new();
    for segment in pre_tokenized.segments {
        pieces.extend(initial_bpe_pieces(&segment, tokenizer_ref, &metadata)?);
    }

    let mut pieces_builder = ArtifactIo::start_builder(
        artifact_id("bpe-pieces-0")?,
        RasterArtifactMetadata::open_bpe_pieces(),
    )?;
    for (piece_idx, piece) in pieces.iter().enumerate() {
        ArtifactIo::append_leaf(&mut pieces_builder, piece_idx, bpe_piece_leaf(piece))?;
    }
    let pieces_ref = RasterBpePieceSequenceRef::new(ArtifactIo::finalize_builder(pieces_builder)?)?;
    Ok(GemmaBpeState::new(
        pieces_ref,
        pre_tokenized.add_special_tokens,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    ))
}

pub fn prepare_raster_prompt_input_roots(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterPromptInputRoots> {
    ArtifactIo::reset_store();

    let prompt_bytes_ref = store_byte_artifact(
        "prompt-bytes",
        PROMPT_BYTES_ARTIFACT_KIND,
        PROMPT_BYTES_ARTIFACT_DOMAIN,
        &request.prompt_bytes,
    )?;
    let prompt_text = decode_prompt_bytes(&request.prompt_bytes, request.text_decoding_policy)?;
    let prompt_text_ref = store_text_artifact(
        "prompt-text",
        PROMPT_TEXT_ARTIFACT_KIND,
        PROMPT_TEXT_ARTIFACT_DOMAIN,
        &prompt_text,
    )?;
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request.add_generation_prompt)?;
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    let rendered_prompt_ref = store_text_artifact(
        "rendered-prompt",
        RENDERED_PROMPT_ARTIFACT_KIND,
        RENDERED_PROMPT_ARTIFACT_DOMAIN,
        &rendered_prompt,
    )?;
    let input = init_tokenize_prompt(&rendered_prompt, request.add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer)?;
    let normalized_prompt_ref = store_text_artifact(
        "normalized-prompt",
        NORMALIZED_PROMPT_ARTIFACT_KIND,
        NORMALIZED_PROMPT_ARTIFACT_DOMAIN,
        &normalized.text,
    )?;
    let pre_tokenized = split_tokenize_prompt(normalized, tokenizer)?;
    let bpe_state = init_bpe_tokenize_prompt(
        pre_tokenized,
        tokenizer,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;
    let tokenizer_source_ref = tokenizer.committed_source_ref()?;

    Ok(RasterPromptInputRoots {
        prompt_bytes_root: prompt_bytes_ref.root().to_string(),
        prompt_text_root: prompt_text_ref.root().to_string(),
        rendered_prompt_root: rendered_prompt_ref.root().to_string(),
        normalized_prompt_root: normalized_prompt_ref.root().to_string(),
        tokenizer_source_ref,
        bpe_state,
    })
}

#[tile]
pub fn init_bpe_merge_scan(state: &GemmaBpeState) -> Result<GemmaBpeScanState> {
    if state.bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }
    Ok(GemmaBpeScanState {
        pieces_root: state.pieces_root.clone(),
        piece_count: state.piece_count,
        next_pair_idx: 0,
        best_candidate: None,
        bpe_pairs_per_tile: state.bpe_pairs_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn scan_bpe_merge_candidates(
    mut state: GemmaBpeScanState,
    tokenizer: &CommittedExternalSource,
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
        let pair = read_bpe_pair(&state.pieces_root, state.piece_count, pair_idx)?;
        if let Some(rule) = ArtifactIo::auth_read(
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
    tokenizer: &CommittedExternalSource,
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
    let merged = ArtifactIo::auth_read(
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

    let output_builder_ref = ArtifactIo::start_builder(
        artifact_id(format!("bpe-pieces-{}", state.iteration + 1))?,
        RasterArtifactMetadata::open_bpe_pieces(),
    )?;

    Ok(GemmaBpeApplyState {
        input_pieces_root: state.pieces_root.clone(),
        output_builder_root: output_builder_ref.running_root().to_string(),
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
            state.output_builder_root = ArtifactIo::append_leaf_by_builder_root(
                &state.output_builder_root,
                state.output_cursor,
                bpe_piece_leaf(&state.merged),
            )?;
            state.input_cursor += 2;
            state.output_cursor += 1;
            continue;
        }

        let piece = read_bpe_piece(&state.input_pieces_root, state.input_cursor)?;
        state.output_builder_root = ArtifactIo::append_leaf_by_builder_root(
            &state.output_builder_root,
            state.output_cursor,
            bpe_piece_leaf(&piece),
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
    let pieces_ref = RasterBpePieceSequenceRef::new(ArtifactIo::finalize_builder_by_root(
        &state.output_builder_root,
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
    tokenizer: &CommittedExternalSource,
) -> Result<(bool, GemmaBpeState)> {
    let scan_state = call_tile!(init_bpe_merge_scan, &state)?;
    let scan_state = call_recur_tile_result!(scan_bpe_merge_candidates, scan_state, tokenizer)?;
    let Some(selection) = call_tile!(finalize_bpe_merge_scan, scan_state, tokenizer)? else {
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
    let token_ids_builder_ref = ArtifactIo::start_builder(
        artifact_id("prompt-token-ids")?,
        RasterArtifactMetadata::open_token_ids(),
    )?;
    Ok(GemmaTokenIdFinalizeState {
        pieces_root: output.pieces_root,
        piece_count: output.piece_count,
        token_ids_builder_root: token_ids_builder_ref.running_root().to_string(),
        next_piece_idx: 0,
        pieces_per_tile: output.bpe_pieces_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn finalize_next_token_ids(
    mut state: GemmaTokenIdFinalizeState,
    tokenizer: &CommittedExternalSource,
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
        let piece = read_bpe_piece(&state.pieces_root, piece_idx)?;
        let token_id = ArtifactIo::auth_read(tokenizer, GemmaTokenIdRequest { token: &piece })?
            .with_context(|| format!("Gemma tokenizer piece {piece:?} is missing from vocab"))?;
        state.token_ids_builder_root = ArtifactIo::append_leaf_by_builder_root(
            &state.token_ids_builder_root,
            piece_idx,
            token_id_leaf(token_id),
        )?;
    }
    state.next_piece_idx = end_piece_idx;

    Ok((state.next_piece_idx >= state.piece_count, state))
}

#[tile]
pub fn finalize_tokenize_prompt(
    state: GemmaTokenIdFinalizeState,
) -> Result<RasterTokenizationResult> {
    if state.next_piece_idx != state.piece_count {
        bail!(
            "token-id finalization stopped at piece {}, expected {}",
            state.next_piece_idx,
            state.piece_count
        );
    }
    let token_ids_ref = RasterTokenIdSequenceRef::new(ArtifactIo::finalize_builder_by_root(
        &state.token_ids_builder_root,
    )?)?;
    Ok(RasterTokenizationResult {
        token_ids_root: token_ids_ref.root().to_string(),
        token_count: token_ids_ref.token_count(),
    })
}

pub fn tokenize_prompt(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
) -> Result<RasterTokenizationResult> {
    tokenize_prompt_with_controls(
        prompt_ref,
        tokenizer_ref,
        add_special_tokens,
        DEFAULT_BPE_PAIRS_PER_TILE,
        DEFAULT_BPE_PIECES_PER_TILE,
    )
}

pub fn tokenize_prompt_with_controls(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterTokenizationResult> {
    ArtifactIo::reset_store();
    let input = init_tokenize_prompt(prompt_ref, add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer_ref)?;
    let pre_tokenized = split_tokenize_prompt(normalized, tokenizer_ref)?;
    let state = init_bpe_tokenize_prompt(
        pre_tokenized,
        tokenizer_ref,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;
    let tokenizer_source_ref = tokenizer_ref.committed_source_ref()?;
    tokenize_bpe_state(state, tokenizer_source_ref)
}

#[sequence]
pub fn tokenize_bpe_state(
    state: GemmaBpeState,
    tokenizer_source_ref: ExternalSourceRef,
) -> Result<RasterTokenizationResult> {
    let tokenizer = CommittedExternalSource::new(tokenizer_source_ref);
    let state = call_recur_seq_result!(merge_bpe_tokenize_prompt, state, &tokenizer)?;
    let output = call_tile!(finalize_bpe_tokenize_prompt, state)?;
    let token_id_state = call_tile!(init_token_id_finalization, output)?;
    let token_id_state =
        call_recur_tile_result!(finalize_next_token_ids, token_id_state, &tokenizer)?;
    call_tile!(finalize_tokenize_prompt, token_id_state)
}

pub fn build_prompt_commitment(prompt_token_ids_ref: &[u32]) -> Result<String> {
    let payload = serde_json::to_vec(prompt_token_ids_ref)
        .context("failed to serialize input-embedding prompt token ids")?;

    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}

#[tile]
pub fn build_prompt_commitment_from_root(token_ids_root: &str) -> Result<String> {
    Ok(token_ids_root.to_string())
}

#[tile]
pub fn finalize_raster_prompt_preparation(
    prompt_bytes_root: String,
    prompt_text_root: String,
    rendered_prompt_root: String,
    normalized_prompt_root: String,
    prompt_token_ids_root: String,
    prompt_token_count: usize,
) -> Result<RasterPromptPreparationState> {
    Ok(RasterPromptPreparationState {
        prompt_bytes_root,
        prompt_text_root,
        rendered_prompt_root,
        normalized_prompt_root,
        prompt_token_ids_root,
        prompt_token_count,
    })
}

pub fn run(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<RasterPromptPreparationResult> {
    run_with_tokenizer_controls(
        request,
        model,
        tokenizer,
        DEFAULT_BPE_PAIRS_PER_TILE,
        DEFAULT_BPE_PIECES_PER_TILE,
    )
}

pub fn run_with_tokenizer_controls(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterPromptPreparationResult> {
    let input_roots = prepare_raster_prompt_input_roots(
        request,
        model,
        tokenizer,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;
    run_from_input_roots(input_roots)
}

#[sequence]
pub fn run_from_input_roots(
    input_roots: RasterPromptInputRoots,
) -> Result<RasterPromptPreparationResult> {
    let tokenizer = CommittedExternalSource::new(input_roots.tokenizer_source_ref);
    let bpe_state =
        call_recur_seq_result!(merge_bpe_tokenize_prompt, input_roots.bpe_state, &tokenizer)?;
    let output = call_tile!(finalize_bpe_tokenize_prompt, bpe_state)?;
    let token_id_state = call_tile!(init_token_id_finalization, output)?;
    let token_id_state =
        call_recur_tile_result!(finalize_next_token_ids, token_id_state, &tokenizer)?;
    let tokenization = call_tile!(finalize_tokenize_prompt, token_id_state)?;
    let prompt_token_ids_root = call_tile!(
        build_prompt_commitment_from_root,
        &tokenization.token_ids_root
    )?;

    let state = call_tile!(
        finalize_raster_prompt_preparation,
        input_roots.prompt_bytes_root,
        input_roots.prompt_text_root,
        input_roots.rendered_prompt_root,
        input_roots.normalized_prompt_root,
        prompt_token_ids_root,
        tokenization.token_count
    )?;
    Ok(RasterPromptPreparationResult { state })
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};

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
    fn finalize_tokenize_prompt_returns_token_id_root() {
        let tokenizer = test_tokenizer_source();
        let tokenizer = tokenizer
            .committed_source()
            .expect("tokenizer source should commit");
        super::init_artifact_store();
        let pieces_ref =
            insert_bpe_piece_sequence_for_test("pieces", vec!["a".to_string(), "ab".to_string()])
                .expect("pieces should insert");
        let mut state = init_token_id_finalization(GemmaBpeOutput {
            pieces_root: pieces_ref.root().to_string(),
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
        let tokenization = finalize_tokenize_prompt(state).expect("token ids should finalize");
        let token_ids =
            materialize_token_ids(&tokenization.token_ids_root, tokenization.token_count)
                .expect("token ids should materialize");

        assert_eq!(token_ids, vec![1, 3]);
        let serialized =
            serde_json::to_string(&tokenization).expect("tokenization should serialize");
        assert!(!serialized.contains("token_ids_ref"));
    }

    #[test]
    fn tokenize_prompt_applies_recursive_bpe_merges() {
        let tokenization =
            tokenize_prompt("ab", &test_tokenizer_source(), false).expect("prompt should tokenize");
        let token_ids =
            materialize_token_ids(&tokenization.token_ids_root, tokenization.token_count)
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
        assert!(!serialized.contains("pieces_ref"));
        assert_eq!(
            materialize_bpe_pieces(&state.pieces_root, state.piece_count)
                .expect("pieces should materialize"),
            vec!["a", "b"]
        );
    }

    #[test]
    fn tokenize_prompt_uses_byte_fallback_for_unknown_chars() {
        let tokenization =
            tokenize_prompt("é", &test_tokenizer_source(), false).expect("prompt should tokenize");
        let token_ids =
            materialize_token_ids(&tokenization.token_ids_root, tokenization.token_count)
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

        let tiny_token_ids =
            materialize_token_ids(&tiny_chunks.token_ids_root, tiny_chunks.token_count)
                .expect("tiny token ids should materialize");
        let larger_token_ids =
            materialize_token_ids(&larger_chunks.token_ids_root, larger_chunks.token_count)
                .expect("larger token ids should materialize");
        assert_eq!(tiny_token_ids, larger_token_ids);
        assert_eq!(tiny_token_ids, vec![12]);
    }

    #[test]
    fn run_with_tokenizer_controls_returns_root_backed_prompt_state() {
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
        let token_ids = materialize_token_ids(
            &result.state.prompt_token_ids_root,
            result.state.prompt_token_count,
        )
        .expect("token ids should materialize for test assertions");
        let serialized = serde_json::to_string(&result.state).expect("state should serialize");

        assert_eq!(token_ids, vec![3]);
        assert_eq!(result.state.prompt_token_count, 1);
        assert!(!serialized.contains("prompt_token_ids\":["));
        assert!(!serialized.contains("prompt_token_ids_ref"));
        assert!(!serialized.contains("prompt_bytes_ref"));
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
                super::bpe_piece_leaf(piece),
            )?;
        }
        RasterBpePieceSequenceRef::new(raster_artifact_store::finalize_builder(builder)?)
    }

    fn materialize_bpe_pieces(pieces_root: &str, piece_count: usize) -> Result<Vec<String>> {
        (0..piece_count)
            .map(|piece_idx| super::read_bpe_piece(pieces_root, piece_idx))
            .collect()
    }

    fn materialize_token_ids(token_ids_root: &str, token_count: usize) -> Result<Vec<u32>> {
        let token_ids_ref = RasterTokenIdSequenceRef::new(
            raster_artifact_store::artifact_ref_for_root(token_ids_root)?,
        )?;
        (0..token_count)
            .map(|token_idx| {
                let read =
                    raster_artifact_store::read_leaf(token_ids_ref.artifact_ref(), token_idx)?;
                raster_artifact_store::verify_artifact_read(token_ids_ref.artifact_ref(), &read)?;
                decode_token_id_leaf(read.payload())
            })
            .collect()
    }

    fn decode_token_id_leaf(payload: &[u8]) -> Result<u32> {
        let bytes: [u8; 4] = payload
            .try_into()
            .context("token-id leaf payload must be exactly four bytes")?;
        Ok(u32::from_le_bytes(bytes))
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
