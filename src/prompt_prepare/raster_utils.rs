use anyhow::{bail, Context, Result};

use crate::shared::artifact_io::ArtifactIo;
use crate::shared::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaSpecialTokenAtRequest, GemmaTokenIdRequest,
    GemmaTokenizerMetadata, GemmaTokenizerSpec,
};
use crate::shared::raster_artifact_store::{
    RasterArtifactId, RasterArtifactMetadata, RasterArtifactRef, RasterBpePieceSequenceRef,
};

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

fn text_char_leaf(ch: char) -> Vec<u8> {
    let mut buffer = [0; 4];
    let text = ch.encode_utf8(&mut buffer);
    bpe_piece_leaf(text)
}

pub(super) fn token_id_leaf(token_id: u32) -> Vec<u8> {
    token_id.to_le_bytes().to_vec()
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
