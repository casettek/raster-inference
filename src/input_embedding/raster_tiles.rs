use anyhow::{bail, Result};

use crate::input_embedding::authenticated_source::{
    GemmaInputEmbeddingMetadataRequest, GemmaInputEmbeddingRowRequest,
};
use crate::raster_authoring::prelude::{call_recur_tile, call_tile, sequence, tile};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactStoreRoots,
    RasterRoutineOutput,
};
use crate::shared::raster_kernels::transformer::RasterActivationRow;

use super::raster_utils::{
    append_activation_row_by_builder_root_with_roots,
    finalize_activation_sequence_builder_by_root_with_roots, read_prompt_token_id,
    start_activation_sequence_builder_with_roots,
};

const EMBEDDED_PROMPT_ARTIFACT_NAME: &str = "input.embedding.embedded_prompt";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterInputEmbeddingInputRoots {
    pub prompt_token_ids_root: String,
    pub prompt_token_count: usize,
    pub embedding_source_root: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterInputEmbeddingRefs {
    pub source_id: String,
    pub embedding_source_root: String,
    pub prompt_token_ids_root: String,
    pub prompt_token_count: usize,
    pub embedded_prompt_activations_ref: RasterActivationSequenceArtifactRef,
}

pub type RasterInputEmbeddingOutput = RasterRoutineOutput<RasterInputEmbeddingRefs>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct InputEmbeddingRasterState {
    source_id: String,
    embedding_source_root: String,
    prompt_token_ids_root: String,
    prompt_token_count: usize,
    hidden_size: usize,
    next_token_idx: usize,
}

impl InputEmbeddingRasterState {
    fn is_complete(&self) -> bool {
        self.next_token_idx >= self.prompt_token_count
    }
}

#[tile]
pub fn init_input_embedding_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterInputEmbeddingInputRoots,
) -> Result<(RasterArtifactStoreRoots, InputEmbeddingRasterState)> {
    if input_roots.prompt_token_count == 0 {
        bail!("transformer embedding requires at least one token id");
    }
    let metadata = ArtifactIo::auth_read(
        input_roots.embedding_source_root.as_str(),
        GemmaInputEmbeddingMetadataRequest,
    )?;
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
    mut state: InputEmbeddingRasterState,
) -> Result<(bool, RasterArtifactStoreRoots, InputEmbeddingRasterState)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let token_id = read_prompt_token_id(
        &artifact_store_roots,
        &state.prompt_token_ids_root,
        state.prompt_token_count,
        state.next_token_idx,
    )?;
    let row = ArtifactIo::auth_read(
        state.embedding_source_root.as_str(),
        GemmaInputEmbeddingRowRequest { token_id },
    )?;
    if row.len() != state.hidden_size {
        bail!(
            "input embedding row {} has width {}, expected {}",
            state.next_token_idx,
            row.len(),
            state.hidden_size
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(EMBEDDED_PROMPT_ARTIFACT_NAME)?
        .to_string();
    let (next_roots, _next_builder_root) = append_activation_row_by_builder_root_with_roots(
        &artifact_store_roots,
        &output_builder_root,
        state.next_token_idx,
        RasterActivationRow::from_acts(row),
    )?;
    artifact_store_roots = next_roots;
    state.next_token_idx += 1;

    Ok((state.is_complete(), artifact_store_roots, state))
}

#[tile]
pub fn finalize_input_embedding_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: InputEmbeddingRasterState,
) -> Result<(RasterArtifactStoreRoots, RasterInputEmbeddingRefs)> {
    if !state.is_complete() {
        bail!(
            "input embedding finalized at token {}, expected {}",
            state.next_token_idx,
            state.prompt_token_count
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
            source_id: state.source_id,
            embedding_source_root: state.embedding_source_root,
            prompt_token_ids_root: state.prompt_token_ids_root,
            prompt_token_count: state.prompt_token_count,
            embedded_prompt_activations_ref,
        },
    ))
}

#[sequence]
pub fn main(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterInputEmbeddingInputRoots,
) -> Result<RasterInputEmbeddingOutput> {
    let (artifact_store_roots, state) = call_tile!(
        init_input_embedding_state,
        artifact_store_roots,
        input_roots
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        append_next_input_embedding_row,
        (artifact_store_roots, state)
    )?;
    let (_artifact_store_roots, refs) =
        call_tile!(finalize_input_embedding_refs, artifact_store_roots, state)?;
    Ok(RasterInputEmbeddingOutput::new(_artifact_store_roots, refs))
}
