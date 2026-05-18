use anyhow::Result;

use crate::shared::input::InferenceExecutionMode;
use crate::shared::raster_prefill_finalize::AuthenticatedGemmaPrefillFinalizeSource;
use crate::shared::raster_row_store::{
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef,
};
use crate::shared::transformer::{
    ActivationSequence, Gemma4TransformerModel, LayerKvCache, TransformerPrefillResult,
};

pub mod raster_tiles;
mod raster_utils;
pub mod tiles;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<TransformerPrefillResult> {
    tiles::run(
        prompt_token_ids,
        model,
        final_hidden_states,
        layer_caches,
        execution_mode,
    )
}

pub fn run_raster(
    prompt_token_ids: &[u32],
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
    projection_rows_per_tile: usize,
) -> Result<TransformerPrefillResult> {
    raster_tiles::run(
        prompt_token_ids,
        final_hidden_states,
        layer_caches,
        finalize_source,
        projection_rows_per_tile,
    )
}

pub fn run_raster_refs_with_store(
    store: &mut AuthenticatedRasterTensorStore,
    prompt_token_ids: &[u32],
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: Vec<crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot>,
    projection_rows_per_tile: usize,
) -> Result<TransformerPrefillResult> {
    raster_tiles::main(
        store,
        prompt_token_ids,
        final_hidden_states_ref,
        layer_caches,
        finalize_source,
        projection_rows_per_tile,
    )
}
