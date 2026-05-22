use anyhow::{bail, Context, Result};

use crate::dsl::prelude::{call_recur_seq, call_recur_tile, call_tile, sequence, tile};
use crate::shared::api::input::RasterPromptPreparationState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    token_id_leaf, RasterArtifactMetadata, RasterArtifactStoreRoots, RasterBpePieceSequenceRef,
    RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma_tokenizer::{
    GemmaBpeMergeRequest, GemmaBpeMergedTokenRequest, GemmaBpeOutput, GemmaBpeState,
    GemmaTokenIdRequest,
};

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

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

// Raster execution tiles, ordered by the sequence calls that reach them.

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
