use anyhow::{bail, Result};

use crate::dsl::prelude::{call_recur_tile, call_tile, sequence, tile};
use crate::routines::input_embedding::raster::auth_source::{
    GemmaInputEmbeddingMetadataRequest, GemmaInputEmbeddingRowRequest, RasterInputEmbeddingSource,
};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{RasterArtifactId, RasterArtifactStoreRoots};
use crate::shared::raster_kernels::transformer::RasterActivationRow;

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterInputEmbeddingInputRoots,
    embedding_source: &RasterInputEmbeddingSource<'_>,
) -> Result<RasterInputEmbeddingOutput> {
    let (artifact_store_roots, input_embedding_state) = call_tile!(
        init_input_embedding_state,
        artifact_store_roots,
        input_roots,
        embedding_source
    )?;
    let (artifact_store_roots, input_embedding_state) = call_recur_tile!(
        append_next_input_embedding_row,
        (artifact_store_roots, input_embedding_state),
        embedding_source
    )?;
    let (_artifact_store_roots, refs) = call_tile!(
        finalize_input_embedding_refs,
        artifact_store_roots,
        input_embedding_state
    )?;
    Ok(RasterInputEmbeddingOutput::new(_artifact_store_roots, refs))
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
pub fn init_input_embedding_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterInputEmbeddingInputRoots,
    embedding_source: &RasterInputEmbeddingSource<'_>,
) -> Result<(RasterArtifactStoreRoots, InputEmbeddingRasterState)> {
    if input_roots.prompt_token_count == 0 {
        bail!("transformer embedding requires at least one token id");
    }
    if input_roots.embedding_source_root != embedding_source.root() {
        bail!(
            "raster input embedding source root {} does not match input source root {}",
            embedding_source.root(),
            input_roots.embedding_source_root
        );
    }
    let metadata = ArtifactIo::auth_read(embedding_source, GemmaInputEmbeddingMetadataRequest)?;
    if metadata.hidden_size == 0 {
        bail!("Gemma input embedding source must have non-zero hidden size");
    }
    let (artifact_store_roots, _output_builder) = start_activation_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(EMBEDDED_PROMPT_ARTIFACT_NAME)?,
        input_roots.prompt_token_count,
        metadata.hidden_size,
    )?;

    Ok((
        artifact_store_roots,
        InputEmbeddingRasterState {
            source_id: metadata.source_id,
            embedding_source_root: input_roots.embedding_source_root,
            prompt_token_ids_root: input_roots.prompt_token_ids_root,
            prompt_token_count: input_roots.prompt_token_count,
            hidden_size: metadata.hidden_size,
            next_token_idx: 0,
        },
    ))
}

#[tile(kind = recursive)]
pub fn append_next_input_embedding_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut input_embedding_state: InputEmbeddingRasterState,
    embedding_source: &RasterInputEmbeddingSource<'_>,
) -> Result<(bool, RasterArtifactStoreRoots, InputEmbeddingRasterState)> {
    if input_embedding_state.is_complete() {
        return Ok((true, artifact_store_roots, input_embedding_state));
    }

    let token_id = read_prompt_token_id(
        &artifact_store_roots,
        &input_embedding_state.prompt_token_ids_root,
        input_embedding_state.prompt_token_count,
        input_embedding_state.next_token_idx,
    )?;
    if input_embedding_state.embedding_source_root != embedding_source.root() {
        bail!(
            "raster input embedding source root {} does not match state source root {}",
            embedding_source.root(),
            input_embedding_state.embedding_source_root
        );
    }
    let row = ArtifactIo::auth_read(embedding_source, GemmaInputEmbeddingRowRequest { token_id })?;
    if row.len() != input_embedding_state.hidden_size {
        bail!(
            "input embedding row {} has width {}, expected {}",
            input_embedding_state.next_token_idx,
            row.len(),
            input_embedding_state.hidden_size
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(EMBEDDED_PROMPT_ARTIFACT_NAME)?
        .to_string();
    let (next_roots, _next_builder_root) = append_activation_row_by_builder_root_with_roots(
        &artifact_store_roots,
        &output_builder_root,
        input_embedding_state.next_token_idx,
        RasterActivationRow::from_acts(row),
    )?;
    artifact_store_roots = next_roots;
    input_embedding_state.next_token_idx += 1;

    Ok((
        input_embedding_state.is_complete(),
        artifact_store_roots,
        input_embedding_state,
    ))
}

#[tile]
pub fn finalize_input_embedding_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_state: InputEmbeddingRasterState,
) -> Result<(RasterArtifactStoreRoots, RasterInputEmbeddingRefs)> {
    if !input_embedding_state.is_complete() {
        bail!(
            "input embedding finalized at token {}, expected {}",
            input_embedding_state.next_token_idx,
            input_embedding_state.prompt_token_count
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(EMBEDDED_PROMPT_ARTIFACT_NAME)?
        .to_string();
    let (artifact_store_roots, embedded_prompt_activations_ref) =
        finalize_activation_sequence_builder_by_root_with_roots(
            &artifact_store_roots,
            &output_builder_root,
        )?;

    Ok((
        artifact_store_roots.clone(),
        RasterInputEmbeddingRefs {
            source_id: input_embedding_state.source_id,
            embedding_source_root: input_embedding_state.embedding_source_root,
            prompt_token_ids_root: input_embedding_state.prompt_token_ids_root,
            prompt_token_count: input_embedding_state.prompt_token_count,
            embedded_prompt_activations_ref,
        },
    ))
}
