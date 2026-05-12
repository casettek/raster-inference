use anyhow::{bail, Result};

use crate::raster_authoring::prelude::{call_recur_tile, call_tile, sequence, tile};
use crate::shared::artifact_io::ArtifactIo;
use crate::shared::raster_artifact_store::{RasterActivationSequenceArtifactRef, RasterArtifactId};
use crate::shared::raster_input_embedding::{
    GemmaInputEmbeddingMetadataRequest, GemmaInputEmbeddingRowRequest,
};
use crate::shared::raster_transformer_kernels::RasterActivationRow;

use super::raster_utils::{
    append_activation_row_by_builder_root, finalize_activation_sequence_builder_by_root,
    read_prompt_token_id, start_activation_sequence_builder,
};

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct InputEmbeddingRasterState {
    source_id: String,
    embedding_source_root: String,
    prompt_token_ids_root: String,
    prompt_token_count: usize,
    hidden_size: usize,
    next_token_idx: usize,
    output_builder_root: String,
}

impl InputEmbeddingRasterState {
    fn is_complete(&self) -> bool {
        self.next_token_idx >= self.prompt_token_count
    }
}

#[tile]
pub fn init_input_embedding_state(
    input_roots: RasterInputEmbeddingInputRoots,
) -> Result<InputEmbeddingRasterState> {
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
    let output_builder = start_activation_sequence_builder(
        RasterArtifactId::new("input.embedding.embedded_prompt")?,
        input_roots.prompt_token_count,
        metadata.hidden_size,
    )?;

    Ok(InputEmbeddingRasterState {
        source_id: metadata.source_id,
        embedding_source_root: input_roots.embedding_source_root,
        prompt_token_ids_root: input_roots.prompt_token_ids_root,
        prompt_token_count: input_roots.prompt_token_count,
        hidden_size: metadata.hidden_size,
        next_token_idx: 0,
        output_builder_root: output_builder.running_root().to_string(),
    })
}

#[tile(kind = recursive)]
pub fn append_next_input_embedding_row(
    mut state: InputEmbeddingRasterState,
) -> Result<(bool, InputEmbeddingRasterState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let token_id = read_prompt_token_id(
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
    state.output_builder_root = append_activation_row_by_builder_root(
        &state.output_builder_root,
        state.next_token_idx,
        RasterActivationRow::from_acts(row),
    )?;
    state.next_token_idx += 1;

    Ok((state.is_complete(), state))
}

#[tile]
pub fn finalize_input_embedding_refs(
    state: InputEmbeddingRasterState,
) -> Result<RasterInputEmbeddingRefs> {
    if !state.is_complete() {
        bail!(
            "input embedding finalized at token {}, expected {}",
            state.next_token_idx,
            state.prompt_token_count
        );
    }
    let embedded_prompt_activations_ref =
        finalize_activation_sequence_builder_by_root(&state.output_builder_root)?;

    Ok(RasterInputEmbeddingRefs {
        source_id: state.source_id,
        embedding_source_root: state.embedding_source_root,
        prompt_token_ids_root: state.prompt_token_ids_root,
        prompt_token_count: state.prompt_token_count,
        embedded_prompt_activations_ref,
    })
}

#[sequence]
pub fn main(input_roots: RasterInputEmbeddingInputRoots) -> Result<RasterInputEmbeddingRefs> {
    let state = call_tile!(init_input_embedding_state, input_roots)?;
    let state = call_recur_tile!(append_next_input_embedding_row, state)?;
    call_tile!(finalize_input_embedding_refs, state)
}
