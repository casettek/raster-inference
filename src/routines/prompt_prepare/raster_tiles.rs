use anyhow::{bail, Context, Result};

use crate::dsl::prelude::{call_recur_seq, call_recur_tile, call_tile, sequence, tile};
use crate::shared::api::input::RasterPromptPreparationState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    token_id_leaf, RasterArtifactMetadata, RasterArtifactStoreRoots, RasterBpePieceSequenceRef,
    RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaBpeMergeRequest, GemmaBpeMergedTokenRequest, GemmaBpeOutput,
    GemmaBpeState, GemmaTokenIdRequest,
};

use super::raster_utils::{
    artifact_id, bpe_piece_leaf, read_bpe_pair, read_bpe_piece, NORMALIZED_PROMPT_ARTIFACT_NAME,
    PROMPT_BYTES_ARTIFACT_NAME, PROMPT_TEXT_ARTIFACT_NAME, PROMPT_TOKEN_IDS_ARTIFACT_NAME,
    RENDERED_PROMPT_ARTIFACT_NAME,
};

pub const DEFAULT_BPE_PAIRS_PER_TILE: usize = 64;
pub const DEFAULT_BPE_PIECES_PER_TILE: usize = 64;

fn bpe_pieces_artifact_name(iteration: u64) -> String {
    format!("bpe-pieces-{iteration}")
}

fn bpe_pieces_root(
    artifact_store_roots: &RasterArtifactStoreRoots,
    iteration: u64,
) -> Result<&str> {
    let source_name = bpe_pieces_artifact_name(iteration);
    artifact_store_roots.artifact_root_for_source_name(&source_name)
}

fn bpe_pieces_builder_root(
    artifact_store_roots: &RasterArtifactStoreRoots,
    iteration: u64,
) -> Result<&str> {
    let source_name = bpe_pieces_artifact_name(iteration);
    artifact_store_roots.builder_root_for_source_name(&source_name)
}

fn prompt_token_ids_root(artifact_store_roots: &RasterArtifactStoreRoots) -> Result<&str> {
    artifact_store_roots.artifact_root_for_source_name(PROMPT_TOKEN_IDS_ARTIFACT_NAME)
}

fn prompt_token_ids_builder_root(artifact_store_roots: &RasterArtifactStoreRoots) -> Result<&str> {
    artifact_store_roots.builder_root_for_source_name(PROMPT_TOKEN_IDS_ARTIFACT_NAME)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TokenizePromptInput {
    pub rendered_prompt: String,
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone)]
pub struct RasterPromptPreparationResult {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub state: RasterPromptPreparationState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPromptPreparedInputs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub input_roots: RasterPromptInputRoots,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPromptInputRoots {
    pub tokenizer_source_root: String,
    pub bpe_state: GemmaBpeState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterTokenizationResult {
    pub token_ids_root: String,
    pub token_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeTokenizeSequenceState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub bpe_state: GemmaBpeState,
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
    pub piece_count: usize,
    pub iteration: u64,
    pub next_pair_idx: usize,
    pub best_candidate: Option<GemmaBpeScanCandidate>,
    pub bpe_pairs_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeScanTileState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub scan_state: GemmaBpeScanState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaBpeApplyState {
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
pub struct GemmaBpeApplyTileState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub apply_state: GemmaBpeApplyState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdFinalizeState {
    pub piece_count: usize,
    pub pieces_iteration: u64,
    pub next_piece_idx: usize,
    pub pieces_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaTokenIdFinalizeTileState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub token_id_state: GemmaTokenIdFinalizeState,
}

#[tile]
pub fn init_bpe_merge_scan(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: &GemmaBpeState,
) -> Result<GemmaBpeScanTileState> {
    if state.bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }
    Ok(GemmaBpeScanTileState {
        artifact_store_roots,
        scan_state: GemmaBpeScanState {
            piece_count: state.piece_count,
            iteration: state.iteration,
            next_pair_idx: 0,
            best_candidate: None,
            bpe_pairs_per_tile: state.bpe_pairs_per_tile,
        },
    })
}

#[tile(kind = recursive)]
pub fn scan_bpe_merge_candidates(
    mut state: GemmaBpeScanTileState,
    tokenizer_source_root: &str,
) -> Result<(bool, GemmaBpeScanTileState)> {
    let pair_count = state.scan_state.piece_count.saturating_sub(1);
    if state.scan_state.next_pair_idx >= pair_count {
        return Ok((true, state));
    }
    if state.scan_state.bpe_pairs_per_tile == 0 {
        bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }

    let end_pair_idx = state
        .scan_state
        .next_pair_idx
        .saturating_add(state.scan_state.bpe_pairs_per_tile)
        .min(pair_count);
    let pieces_root =
        bpe_pieces_root(&state.artifact_store_roots, state.scan_state.iteration)?.to_string();
    for pair_idx in state.scan_state.next_pair_idx..end_pair_idx {
        let pair = read_bpe_pair(&pieces_root, state.scan_state.piece_count, pair_idx)?;
        if let Some(rule) = ArtifactIo::auth_read(
            tokenizer_source_root,
            GemmaBpeMergeRequest {
                left: &pair.left,
                right: &pair.right,
            },
        )? {
            match &state.scan_state.best_candidate {
                Some(best) if best.rank <= rule.rank => {}
                _ => {
                    state.scan_state.best_candidate = Some(GemmaBpeScanCandidate {
                        piece_idx: pair_idx,
                        rank: rule.rank,
                        merge_index: rule.merge_index,
                    });
                }
            }
        }
    }

    state.scan_state.next_pair_idx = end_pair_idx;
    Ok((state.scan_state.next_pair_idx >= pair_count, state))
}

#[tile]
pub fn finalize_bpe_merge_scan(
    state: GemmaBpeScanTileState,
    tokenizer_source_root: &str,
) -> Result<(RasterArtifactStoreRoots, Option<GemmaBpeMergeSelection>)> {
    let pair_count = state.scan_state.piece_count.saturating_sub(1);
    if state.scan_state.next_pair_idx != pair_count {
        bail!(
            "BPE merge scan finalized at pair {}, expected {pair_count}",
            state.scan_state.next_pair_idx
        );
    }
    let Some(candidate) = state.scan_state.best_candidate else {
        return Ok((state.artifact_store_roots, None));
    };
    let merged = ArtifactIo::auth_read(
        tokenizer_source_root,
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

    Ok((
        state.artifact_store_roots,
        Some(GemmaBpeMergeSelection {
            piece_idx: candidate.piece_idx,
            merge_index: candidate.merge_index,
            merged,
        }),
    ))
}

#[tile]
pub fn init_apply_bpe_merge(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: &GemmaBpeState,
    selection: GemmaBpeMergeSelection,
) -> Result<GemmaBpeApplyTileState> {
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

    let (artifact_store_roots, _output_builder_ref) = ArtifactIo::start_builder_with_roots(
        &artifact_store_roots,
        artifact_id(bpe_pieces_artifact_name(state.iteration + 1))?,
        RasterArtifactMetadata::open_bpe_pieces(),
    )?;

    Ok(GemmaBpeApplyTileState {
        artifact_store_roots,
        apply_state: GemmaBpeApplyState {
            input_piece_count: state.piece_count,
            merge_piece_idx: selection.piece_idx,
            merged: selection.merged,
            input_cursor: 0,
            output_cursor: 0,
            add_special_tokens: state.add_special_tokens,
            iteration: state.iteration,
            bpe_pairs_per_tile: state.bpe_pairs_per_tile,
            bpe_pieces_per_tile: state.bpe_pieces_per_tile,
        },
    })
}

#[tile(kind = recursive)]
pub fn apply_bpe_merge_chunk(
    mut state: GemmaBpeApplyTileState,
) -> Result<(bool, GemmaBpeApplyTileState)> {
    if state.apply_state.bpe_pieces_per_tile == 0 {
        bail!("raster tokenizer BPE pieces per tile must be greater than zero");
    }

    let max_output_cursor = state.apply_state.input_piece_count.saturating_sub(1);
    if state.apply_state.output_cursor >= max_output_cursor {
        return Ok((true, state));
    }

    let output_limit = state
        .apply_state
        .output_cursor
        .saturating_add(state.apply_state.bpe_pieces_per_tile)
        .min(max_output_cursor);
    while state.apply_state.output_cursor < output_limit {
        if state.apply_state.input_cursor == state.apply_state.merge_piece_idx {
            let output_builder_root = bpe_pieces_builder_root(
                &state.artifact_store_roots,
                state.apply_state.iteration + 1,
            )?
            .to_string();
            let (next_roots, _next_builder_root) =
                ArtifactIo::append_leaf_by_builder_root_with_roots(
                    &state.artifact_store_roots,
                    &output_builder_root,
                    state.apply_state.output_cursor,
                    bpe_piece_leaf(&state.apply_state.merged),
                )?;
            state.artifact_store_roots = next_roots;
            state.apply_state.input_cursor += 2;
            state.apply_state.output_cursor += 1;
            continue;
        }

        let input_pieces_root =
            bpe_pieces_root(&state.artifact_store_roots, state.apply_state.iteration)?.to_string();
        let piece = read_bpe_piece(&input_pieces_root, state.apply_state.input_cursor)?;
        let output_builder_root =
            bpe_pieces_builder_root(&state.artifact_store_roots, state.apply_state.iteration + 1)?
                .to_string();
        let (next_roots, _next_builder_root) = ArtifactIo::append_leaf_by_builder_root_with_roots(
            &state.artifact_store_roots,
            &output_builder_root,
            state.apply_state.output_cursor,
            bpe_piece_leaf(&piece),
        )?;
        state.artifact_store_roots = next_roots;
        state.apply_state.input_cursor += 1;
        state.apply_state.output_cursor += 1;
    }

    Ok((state.apply_state.output_cursor >= max_output_cursor, state))
}

#[tile]
pub fn finalize_apply_bpe_merge(
    state: GemmaBpeApplyTileState,
) -> Result<(RasterArtifactStoreRoots, GemmaBpeState)> {
    let expected_piece_count = state.apply_state.input_piece_count.saturating_sub(1);
    if state.apply_state.output_cursor != expected_piece_count {
        bail!(
            "BPE merge apply finalized with {} pieces, expected {expected_piece_count}",
            state.apply_state.output_cursor
        );
    }
    let output_builder_root =
        bpe_pieces_builder_root(&state.artifact_store_roots, state.apply_state.iteration + 1)?
            .to_string();
    let (artifact_store_roots, artifact_ref) = ArtifactIo::finalize_builder_by_root_with_roots(
        &state.artifact_store_roots,
        &output_builder_root,
    )?;
    let pieces_ref = RasterBpePieceSequenceRef::new(artifact_ref)?;
    let mut next_state = GemmaBpeState::new(
        pieces_ref,
        state.apply_state.add_special_tokens,
        state.apply_state.bpe_pairs_per_tile,
        state.apply_state.bpe_pieces_per_tile,
    );
    next_state.iteration = state.apply_state.iteration + 1;
    Ok((artifact_store_roots, next_state))
}

#[sequence(kind = recursive)]
pub fn merge_bpe_tokenize_prompt(
    state: GemmaBpeTokenizeSequenceState,
    tokenizer_source_root: &str,
) -> Result<(bool, GemmaBpeTokenizeSequenceState)> {
    let GemmaBpeTokenizeSequenceState {
        artifact_store_roots,
        bpe_state,
    } = state;
    let scan_state = call_tile!(init_bpe_merge_scan, artifact_store_roots, &bpe_state)?;
    let scan_state =
        call_recur_tile!(scan_bpe_merge_candidates, scan_state, tokenizer_source_root)?;
    let (artifact_store_roots, selection) =
        call_tile!(finalize_bpe_merge_scan, scan_state, tokenizer_source_root)?;
    let Some(selection) = selection else {
        return Ok((
            true,
            GemmaBpeTokenizeSequenceState {
                artifact_store_roots,
                bpe_state,
            },
        ));
    };
    let apply_state = call_tile!(
        init_apply_bpe_merge,
        artifact_store_roots,
        &bpe_state,
        selection
    )?;
    let apply_state = call_recur_tile!(apply_bpe_merge_chunk, apply_state)?;
    let (artifact_store_roots, bpe_state) = call_tile!(finalize_apply_bpe_merge, apply_state)?;
    Ok((
        false,
        GemmaBpeTokenizeSequenceState {
            artifact_store_roots,
            bpe_state,
        },
    ))
}

#[tile]
pub fn finalize_bpe_tokenize_prompt(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: GemmaBpeState,
) -> Result<(RasterArtifactStoreRoots, GemmaBpeOutput)> {
    Ok((artifact_store_roots, state.into_output()))
}

#[tile]
pub fn init_token_id_finalization(
    artifact_store_roots: RasterArtifactStoreRoots,
    output: GemmaBpeOutput,
) -> Result<GemmaTokenIdFinalizeTileState> {
    let (artifact_store_roots, _token_ids_builder_ref) = ArtifactIo::start_builder_with_roots(
        &artifact_store_roots,
        artifact_id(PROMPT_TOKEN_IDS_ARTIFACT_NAME)?,
        RasterArtifactMetadata::open_token_ids(),
    )?;
    Ok(GemmaTokenIdFinalizeTileState {
        artifact_store_roots,
        token_id_state: GemmaTokenIdFinalizeState {
            piece_count: output.piece_count,
            pieces_iteration: output.iteration,
            next_piece_idx: 0,
            pieces_per_tile: output.bpe_pieces_per_tile,
        },
    })
}

#[tile(kind = recursive)]
pub fn finalize_next_token_ids(
    mut state: GemmaTokenIdFinalizeTileState,
    tokenizer_source_root: &str,
) -> Result<(bool, GemmaTokenIdFinalizeTileState)> {
    if state.token_id_state.pieces_per_tile == 0 {
        bail!("raster tokenizer token-id pieces per tile must be greater than zero");
    }
    if state.token_id_state.next_piece_idx >= state.token_id_state.piece_count {
        return Ok((true, state));
    }

    let end_piece_idx = state
        .token_id_state
        .next_piece_idx
        .saturating_add(state.token_id_state.pieces_per_tile)
        .min(state.token_id_state.piece_count);
    for piece_idx in state.token_id_state.next_piece_idx..end_piece_idx {
        let pieces_root = bpe_pieces_root(
            &state.artifact_store_roots,
            state.token_id_state.pieces_iteration,
        )?
        .to_string();
        let piece = read_bpe_piece(&pieces_root, piece_idx)?;
        let token_id =
            ArtifactIo::auth_read(tokenizer_source_root, GemmaTokenIdRequest { token: &piece })?
                .with_context(|| {
                    format!("Gemma tokenizer piece {piece:?} is missing from vocab")
                })?;
        let token_ids_builder_root =
            prompt_token_ids_builder_root(&state.artifact_store_roots)?.to_string();
        let (next_roots, _next_builder_root) = ArtifactIo::append_leaf_by_builder_root_with_roots(
            &state.artifact_store_roots,
            &token_ids_builder_root,
            piece_idx,
            token_id_leaf(token_id),
        )?;
        state.artifact_store_roots = next_roots;
    }
    state.token_id_state.next_piece_idx = end_piece_idx;

    Ok((
        state.token_id_state.next_piece_idx >= state.token_id_state.piece_count,
        state,
    ))
}

#[tile]
pub fn finalize_tokenize_prompt(
    state: GemmaTokenIdFinalizeTileState,
) -> Result<(RasterArtifactStoreRoots, RasterTokenizationResult)> {
    if state.token_id_state.next_piece_idx != state.token_id_state.piece_count {
        bail!(
            "token-id finalization stopped at piece {}, expected {}",
            state.token_id_state.next_piece_idx,
            state.token_id_state.piece_count
        );
    }
    let token_ids_builder_root =
        prompt_token_ids_builder_root(&state.artifact_store_roots)?.to_string();
    let (artifact_store_roots, artifact_ref) = ArtifactIo::finalize_builder_by_root_with_roots(
        &state.artifact_store_roots,
        &token_ids_builder_root,
    )?;
    let token_ids_ref = RasterTokenIdSequenceRef::new(artifact_ref)?;
    Ok((
        artifact_store_roots,
        RasterTokenizationResult {
            token_ids_root: token_ids_ref.root().to_string(),
            token_count: token_ids_ref.token_count(),
        },
    ))
}

#[sequence]
pub fn tokenize_bpe_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: GemmaBpeState,
    tokenizer_source_root: String,
) -> Result<(RasterArtifactStoreRoots, RasterTokenizationResult)> {
    let state = call_recur_seq!(
        merge_bpe_tokenize_prompt,
        GemmaBpeTokenizeSequenceState {
            artifact_store_roots,
            bpe_state: state,
        },
        tokenizer_source_root.as_str()
    )?;
    let GemmaBpeTokenizeSequenceState {
        artifact_store_roots,
        bpe_state,
    } = state;
    let (artifact_store_roots, output) = call_tile!(
        finalize_bpe_tokenize_prompt,
        artifact_store_roots,
        bpe_state
    )?;
    let token_id_state = call_tile!(init_token_id_finalization, artifact_store_roots, output)?;
    let token_id_state = call_recur_tile!(
        finalize_next_token_ids,
        token_id_state,
        tokenizer_source_root.as_str()
    )?;
    call_tile!(finalize_tokenize_prompt, token_id_state)
}

pub fn tokenize_prompt(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
) -> Result<RasterTokenizationResult> {
    super::raster_utils::tokenize_prompt(prompt_ref, tokenizer_ref, add_special_tokens)
}

pub fn tokenize_prompt_with_controls(
    prompt_ref: &str,
    tokenizer_ref: &AuthenticatedGemmaTokenizer,
    add_special_tokens: bool,
    bpe_pairs_per_tile: usize,
    bpe_pieces_per_tile: usize,
) -> Result<RasterTokenizationResult> {
    super::raster_utils::tokenize_prompt_with_controls(
        prompt_ref,
        tokenizer_ref,
        add_special_tokens,
        bpe_pairs_per_tile,
        bpe_pieces_per_tile,
    )
}

#[tile]
pub fn finalize_raster_prompt_preparation(
    artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
) -> Result<(RasterArtifactStoreRoots, RasterPromptPreparationState)> {
    let prompt_bytes_root = artifact_store_roots
        .artifact_root_for_source_name(PROMPT_BYTES_ARTIFACT_NAME)?
        .to_string();
    let prompt_text_root = artifact_store_roots
        .artifact_root_for_source_name(PROMPT_TEXT_ARTIFACT_NAME)?
        .to_string();
    let rendered_prompt_root = artifact_store_roots
        .artifact_root_for_source_name(RENDERED_PROMPT_ARTIFACT_NAME)?
        .to_string();
    let normalized_prompt_root = artifact_store_roots
        .artifact_root_for_source_name(NORMALIZED_PROMPT_ARTIFACT_NAME)?
        .to_string();
    let prompt_token_ids_root = prompt_token_ids_root(&artifact_store_roots)?.to_string();

    Ok((
        artifact_store_roots,
        RasterPromptPreparationState {
            prompt_bytes_root,
            prompt_text_root,
            rendered_prompt_root,
            normalized_prompt_root,
            prompt_token_ids_root,
            prompt_token_count,
        },
    ))
}

#[sequence]
pub fn main(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterPromptInputRoots,
) -> Result<RasterPromptPreparationResult> {
    let bpe_state = call_recur_seq!(
        merge_bpe_tokenize_prompt,
        GemmaBpeTokenizeSequenceState {
            artifact_store_roots,
            bpe_state: input_roots.bpe_state,
        },
        input_roots.tokenizer_source_root.as_str()
    )?;
    let GemmaBpeTokenizeSequenceState {
        artifact_store_roots,
        bpe_state,
    } = bpe_state;
    let (artifact_store_roots, output) = call_tile!(
        finalize_bpe_tokenize_prompt,
        artifact_store_roots,
        bpe_state
    )?;
    let token_id_state = call_tile!(init_token_id_finalization, artifact_store_roots, output)?;
    let token_id_state = call_recur_tile!(
        finalize_next_token_ids,
        token_id_state,
        input_roots.tokenizer_source_root.as_str()
    )?;
    let (artifact_store_roots, tokenization) =
        call_tile!(finalize_tokenize_prompt, token_id_state)?;
    let (artifact_store_roots, state) = call_tile!(
        finalize_raster_prompt_preparation,
        artifact_store_roots,
        tokenization.token_count
    )?;
    Ok(RasterPromptPreparationResult {
        artifact_store_roots,
        state,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::super::native_tiles::{
        build_gemma4_messages, build_prompt_commitment, decode_prompt_bytes, render_prompt,
    };
    use super::super::raster_utils::{
        bpe_piece_leaf, init_artifact_store, init_bpe_tokenize_prompt, init_tokenize_prompt,
        prepare_raster_prompt_input_roots, read_bpe_piece, tokenize_prompt,
        tokenize_prompt_with_controls,
    };
    use super::{
        bpe_pieces_artifact_name, finalize_next_token_ids, finalize_tokenize_prompt,
        init_token_id_finalization, main, prompt_token_ids_root,
    };
    use crate::shared::api::input::{MessageRole, ModelSpec, TextDecodingPolicy};
    use crate::shared::artifacts::artifact_io::ArtifactIo;
    use crate::shared::artifacts::raster_artifact_store::{
        self, decode_token_id_leaf, RasterArtifactId, RasterBpePieceSequenceRef,
        RasterTokenIdSequenceRef,
    };
    use crate::shared::model::gemma_tokenizer::{
        AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeOutput,
        GemmaPreTokenizedText, GemmaTokenizerSpec, GemmaVocabEntry,
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
        let tokenizer_source_root = tokenizer
            .committed_source_ref()
            .expect("tokenizer source should commit")
            .root()
            .to_string();
        init_artifact_store();
        let _pieces_ref = insert_bpe_piece_sequence_for_test(
            &bpe_pieces_artifact_name(0),
            vec!["a".to_string(), "ab".to_string()],
        )
        .expect("pieces should insert");
        let artifact_store_roots = ArtifactIo::export_store_roots();
        let mut state = init_token_id_finalization(
            artifact_store_roots,
            GemmaBpeOutput {
                piece_count: 2,
                iteration: 0,
                add_special_tokens: false,
                bpe_pieces_per_tile: 1,
            },
        )
        .expect("token id finalization should init");
        loop {
            let (done, next_state) = finalize_next_token_ids(state, &tokenizer_source_root)
                .expect("token ids should advance");
            state = next_state;
            if done {
                break;
            }
        }
        let (artifact_store_roots, tokenization) =
            finalize_tokenize_prompt(state).expect("token ids should finalize");
        let token_ids_root =
            prompt_token_ids_root(&artifact_store_roots).expect("token ids root should be present");
        let token_ids = materialize_token_ids(token_ids_root, tokenization.token_count)
            .expect("token ids should materialize");

        assert_eq!(token_ids, vec![1, 3]);
        assert!(artifact_store_roots
            .artifact_entry_for_root(token_ids_root)
            .is_ok());
        let serialized =
            serde_json::to_string(&tokenization).expect("tokenization should serialize");
        assert!(!serialized.contains("token_ids_ref"));
    }

    #[test]
    fn tokenize_prompt_applies_recursive_bpe_merges() {
        let tokenization =
            tokenize_prompt("ab", &test_tokenizer_source(), false).expect("prompt should tokenize");
        let token_ids_root = ArtifactIo::export_store_roots()
            .artifact_root_for_source_name("prompt-token-ids")
            .expect("token ids root should be present")
            .to_string();
        let token_ids = materialize_token_ids(&token_ids_root, tokenization.token_count)
            .expect("token ids should materialize");

        assert_eq!(token_ids, vec![3]);
    }

    #[test]
    fn init_bpe_tokenize_prompt_returns_compact_ref_state() {
        let tokenizer = test_tokenizer_source();
        init_artifact_store();
        let state = init_bpe_tokenize_prompt(
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
            materialize_bpe_pieces(
                ArtifactIo::export_store_roots()
                    .artifact_root_for_source_name("bpe-pieces-0")
                    .expect("BPE pieces root should be present"),
                state.piece_count,
            )
            .expect("pieces should materialize"),
            vec!["a", "b"]
        );
    }

    #[test]
    fn tokenize_prompt_uses_byte_fallback_for_unknown_chars() {
        let tokenization =
            tokenize_prompt("é", &test_tokenizer_source(), false).expect("prompt should tokenize");
        let token_ids_root = ArtifactIo::export_store_roots()
            .artifact_root_for_source_name("prompt-token-ids")
            .expect("token ids root should be present")
            .to_string();
        let token_ids = materialize_token_ids(&token_ids_root, tokenization.token_count)
            .expect("token ids should materialize");

        assert_eq!(token_ids, vec![10, 11]);
    }

    #[test]
    fn tokenize_prompt_chunk_sizes_do_not_change_results() {
        let tokenizer = test_tokenizer_source();
        let tiny_chunks =
            tokenize_prompt_with_controls("aba", &tokenizer, false, 1, 1).expect("tiny chunks");
        let tiny_root = ArtifactIo::export_store_roots()
            .artifact_root_for_source_name("prompt-token-ids")
            .expect("tiny token ids root should be present")
            .to_string();
        let tiny_token_ids = materialize_token_ids(&tiny_root, tiny_chunks.token_count)
            .expect("tiny token ids should materialize");

        let larger_chunks =
            tokenize_prompt_with_controls("aba", &tokenizer, false, 8, 8).expect("larger chunks");
        let larger_root = ArtifactIo::export_store_roots()
            .artifact_root_for_source_name("prompt-token-ids")
            .expect("larger token ids root should be present")
            .to_string();
        let larger_token_ids = materialize_token_ids(&larger_root, larger_chunks.token_count)
            .expect("larger token ids should materialize");
        assert_eq!(tiny_token_ids, larger_token_ids);
        assert_eq!(tiny_token_ids, vec![12]);
    }

    #[test]
    fn run_returns_root_backed_prompt_state() {
        let request = crate::shared::api::input::InferenceRequest {
            prompt_bytes: b"ab".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: crate::shared::api::input::InferenceExecutionMode::Deterministic,
            sampling: crate::shared::api::input::SamplingConfig::default(),
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

        let prepared_inputs =
            prepare_raster_prompt_input_roots(&request, &model, &test_tokenizer_source(), 1, 1)
                .expect("raster prompt roots should build");
        let result = main(
            prepared_inputs.artifact_store_roots,
            prepared_inputs.input_roots,
        )
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
        assert!(result
            .artifact_store_roots
            .artifact_entry_for_root(&result.state.prompt_token_ids_root)
            .is_ok());
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
            raster_artifact_store::append_leaf(&mut builder, piece_idx, bpe_piece_leaf(piece))?;
        }
        RasterBpePieceSequenceRef::new(raster_artifact_store::finalize_builder(builder)?)
    }

    fn materialize_bpe_pieces(pieces_root: &str, piece_count: usize) -> Result<Vec<String>> {
        (0..piece_count)
            .map(|piece_idx| read_bpe_piece(pieces_root, piece_idx))
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
