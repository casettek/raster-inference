use anyhow::{bail, Context, Result};

use crate::shared::api::input::{InferenceRequest, ModelSpec};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    token_id_leaf, RasterArtifactId, RasterArtifactMetadata, RasterArtifactRef,
    RasterBpePieceSequenceRef, RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaBpeState, GemmaNormalizedText, GemmaPreTokenizedText,
    GemmaSpecialTokenAtRequest, GemmaTokenIdRequest, GemmaTokenizerMetadata,
    GemmaTokenizerMetadataRequest, GemmaTokenizerSpec,
};

use super::raster_tiles::{
    RasterPromptInputRoots, RasterPromptPreparedInputs, RasterTokenizationResult,
    TokenizePromptInput,
};
use super::tiles::{build_gemma4_messages, decode_prompt_bytes, render_prompt};

pub(super) const PROMPT_BYTES_ARTIFACT_KIND: &str = "prompt_bytes";
pub(super) const PROMPT_TEXT_ARTIFACT_KIND: &str = "prompt_text";
pub(super) const RENDERED_PROMPT_ARTIFACT_KIND: &str = "rendered_prompt";
pub(super) const NORMALIZED_PROMPT_ARTIFACT_KIND: &str = "normalized_prompt";
pub(super) const PROMPT_BYTES_ARTIFACT_NAME: &str = "prompt-bytes";
pub(super) const PROMPT_TEXT_ARTIFACT_NAME: &str = "prompt-text";
pub(super) const RENDERED_PROMPT_ARTIFACT_NAME: &str = "rendered-prompt";
pub(super) const NORMALIZED_PROMPT_ARTIFACT_NAME: &str = "normalized-prompt";
pub(super) const PROMPT_TOKEN_IDS_ARTIFACT_NAME: &str = "prompt-token-ids";

pub(super) const PROMPT_BYTES_ARTIFACT_DOMAIN: &str = "raster-artifact-prompt-bytes-merkle-v1";
pub(super) const PROMPT_TEXT_ARTIFACT_DOMAIN: &str = "raster-artifact-prompt-text-merkle-v1";
pub(super) const RENDERED_PROMPT_ARTIFACT_DOMAIN: &str =
    "raster-artifact-rendered-prompt-merkle-v1";
pub(super) const NORMALIZED_PROMPT_ARTIFACT_DOMAIN: &str =
    "raster-artifact-normalized-prompt-merkle-v1";

pub(super) struct BpePair {
    pub(super) left: String,
    pub(super) right: String,
}

pub(super) fn read_bpe_piece(pieces_root: &str, piece_idx: usize) -> Result<String> {
    let pieces_ref =
        RasterBpePieceSequenceRef::new(ArtifactIo::artifact_ref_for_root(pieces_root)?)?;
    let read = ArtifactIo::read_leaf(pieces_ref.artifact_ref(), piece_idx)?;
    ArtifactIo::verify_artifact_read(pieces_ref.artifact_ref(), &read)?;
    decode_bpe_piece_leaf(read.payload())
}

pub(super) fn read_bpe_pair(
    pieces_root: &str,
    piece_count: usize,
    pair_idx: usize,
) -> Result<BpePair> {
    if pair_idx >= piece_count.saturating_sub(1) {
        bail!(
            "BPE pair {} is out of range for {} pieces",
            pair_idx,
            piece_count
        );
    }
    Ok(BpePair {
        left: read_bpe_piece(pieces_root, pair_idx)?,
        right: read_bpe_piece(pieces_root, pair_idx + 1)?,
    })
}

pub(super) fn store_byte_artifact(
    name: &str,
    kind: &str,
    domain: &str,
    bytes: &[u8],
) -> Result<RasterArtifactRef> {
    let leaves = bytes.iter().map(|byte| vec![*byte]).collect::<Vec<_>>();
    ArtifactIo::insert_artifact(
        artifact_id(name)?,
        RasterArtifactMetadata::open(kind, domain, Vec::new())?,
        leaves,
    )
}

pub(super) fn store_text_artifact(
    name: &str,
    kind: &str,
    domain: &str,
    text: &str,
) -> Result<RasterArtifactRef> {
    let leaves = text.chars().map(text_char_leaf).collect::<Vec<_>>();
    ArtifactIo::insert_artifact(
        artifact_id(name)?,
        RasterArtifactMetadata::open(kind, domain, Vec::new())?,
        leaves,
    )
}

pub(super) fn init_artifact_store() {
    ArtifactIo::reset_store();
}

pub(super) fn prepare_raster_prompt_input_roots(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &AuthenticatedGemmaTokenizer,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterPromptPreparedInputs> {
    init_artifact_store();

    store_byte_artifact(
        PROMPT_BYTES_ARTIFACT_NAME,
        PROMPT_BYTES_ARTIFACT_KIND,
        PROMPT_BYTES_ARTIFACT_DOMAIN,
        &request.prompt_bytes,
    )?;
    let prompt_text = decode_prompt_bytes(&request.prompt_bytes, request.text_decoding_policy)?;
    store_text_artifact(
        PROMPT_TEXT_ARTIFACT_NAME,
        PROMPT_TEXT_ARTIFACT_KIND,
        PROMPT_TEXT_ARTIFACT_DOMAIN,
        &prompt_text,
    )?;
    let gemma4_prompt = build_gemma4_messages(&prompt_text, request.add_generation_prompt)?;
    let rendered_prompt = render_prompt(&gemma4_prompt, model)?;
    store_text_artifact(
        RENDERED_PROMPT_ARTIFACT_NAME,
        RENDERED_PROMPT_ARTIFACT_KIND,
        RENDERED_PROMPT_ARTIFACT_DOMAIN,
        &rendered_prompt,
    )?;
    let input = init_tokenize_prompt(&rendered_prompt, request.add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer)?;
    store_text_artifact(
        NORMALIZED_PROMPT_ARTIFACT_NAME,
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
    let tokenizer_source_root = tokenizer.committed_source_ref()?.root().to_string();

    let input_roots = RasterPromptInputRoots {
        tokenizer_source_root,
        bpe_state,
    };

    Ok(RasterPromptPreparedInputs {
        artifact_store_roots: ArtifactIo::export_store_roots(),
        input_roots,
    })
}

pub(super) fn init_tokenize_prompt(
    prompt_ref: &str,
    add_special_tokens: bool,
) -> Result<TokenizePromptInput> {
    Ok(TokenizePromptInput {
        rendered_prompt: prompt_ref.to_string(),
        add_special_tokens,
    })
}

pub(super) fn normalize_tokenize_prompt(
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

pub(super) fn split_tokenize_prompt(
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

pub(super) fn init_bpe_tokenize_prompt(
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

pub(super) fn tokenize_prompt(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
) -> Result<RasterTokenizationResult> {
    tokenize_prompt_with_controls(
        prompt_ref,
        tokenizer_ref,
        add_special_tokens,
        super::raster_tiles::DEFAULT_BPE_PAIRS_PER_TILE,
        super::raster_tiles::DEFAULT_BPE_PIECES_PER_TILE,
    )
}

pub(super) fn tokenize_prompt_with_controls(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterTokenizationResult> {
    init_artifact_store();
    let input = init_tokenize_prompt(prompt_ref, add_special_tokens)?;
    let normalized = normalize_tokenize_prompt(&input, tokenizer_ref)?;
    let pre_tokenized = split_tokenize_prompt(normalized, tokenizer_ref)?;
    let state = init_bpe_tokenize_prompt(
        pre_tokenized,
        tokenizer_ref,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )?;
    let tokenizer_source_root = tokenizer_ref.committed_source_ref()?.root().to_string();
    let (_artifact_store_roots, tokenization) = super::raster_tiles::tokenize_bpe_state(
        ArtifactIo::export_store_roots(),
        state,
        tokenizer_source_root,
    )?;
    Ok(tokenization)
}

pub(super) fn insert_token_id_artifact(
    artifact_name: &str,
    token_ids: &[u32],
) -> Result<RasterTokenIdSequenceRef> {
    let leaves = token_ids
        .iter()
        .map(|token_id| token_id_leaf(*token_id))
        .collect::<Vec<_>>();
    RasterTokenIdSequenceRef::new(ArtifactIo::insert_artifact(
        artifact_id(artifact_name)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?)
}

fn text_char_leaf(ch: char) -> Vec<u8> {
    let mut buffer = [0; 4];
    let text = ch.encode_utf8(&mut buffer);
    bpe_piece_leaf(text)
}

pub(super) fn bpe_piece_leaf(piece: &str) -> Vec<u8> {
    let bytes = piece.as_bytes();
    let mut payload = Vec::with_capacity(8 + bytes.len());
    payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(bytes);
    payload
}

fn decode_bpe_piece_leaf(payload: &[u8]) -> Result<String> {
    if payload.len() < 8 {
        bail!("BPE piece leaf payload is too short");
    }
    let len = u64::from_le_bytes(
        payload[0..8]
            .try_into()
            .expect("slice length checked above"),
    ) as usize;
    let bytes = &payload[8..];
    if bytes.len() != len {
        bail!("BPE piece leaf length mismatch: {} vs {len}", bytes.len());
    }
    String::from_utf8(bytes.to_vec()).context("BPE piece leaf is not valid UTF-8")
}

pub(super) fn split_merged_with_previous(text_ref: &str, pattern_ref: &str) -> Vec<String> {
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

pub(super) fn initial_bpe_pieces(
    segment_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    metadata_ref: &GemmaTokenizerMetadata,
) -> Result<Vec<String>> {
    let mut pieces = Vec::new();
    let mut byte_idx = 0;

    while byte_idx < segment_ref.len() {
        if let Some(token) = ArtifactIo::auth_read(
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
        if ArtifactIo::auth_read(tokenizer_ref, GemmaTokenIdRequest { token: &piece })?.is_some() {
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

pub(super) fn ensure_tokenizer_controls(
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<()> {
    if bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }
    if bpe_pieces_per_tile == 0 {
        bail!("raster tokenizer BPE pieces per tile must be greater than zero");
    }
    Ok(())
}

pub(super) fn artifact_id(name: impl Into<String>) -> Result<RasterArtifactId> {
    RasterArtifactId::new(name)
}
