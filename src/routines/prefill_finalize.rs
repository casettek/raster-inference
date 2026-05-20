use anyhow::Result;

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4TransformerModel, LayerKvCache, TransformerPrefillResult,
};
use crate::shared::tensors::raster_row_store::RasterActivationSequenceRef;

pub mod native;
pub mod raster;

use self::raster::auth_source::AuthenticatedGemmaPrefillFinalizeSource;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
    execution_mode: InferenceExecutionMode,
) -> Result<TransformerPrefillResult> {
    native::run(
        prompt_token_ids,
        model,
        final_hidden_states,
        layer_caches,
        execution_mode,
    )
}

pub fn run_raster_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: Vec<crate::prefill_layer::raster::PrefillLayerCacheSlot>,
    projection_rows_per_tile: usize,
) -> Result<TransformerPrefillResult> {
    let finalize_source_ref = finalize_source.committed_source_ref()?;
    raster::main(raster::RasterPrefillFinalizeInputRoots {
        artifact_store_roots,
        prompt_token_count,
        finalize_source_root: finalize_source_ref.root().to_string(),
        final_hidden_states_ref,
        layer_caches,
        projection_rows_per_tile,
    })
}

pub fn run_raster_output_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: Vec<crate::prefill_layer::raster::PrefillLayerCacheSlot>,
    projection_rows_per_tile: usize,
) -> Result<raster::RasterPrefillFinalizeOutput> {
    let finalize_source_ref = finalize_source.committed_source_ref()?;
    raster::main_refs(raster::RasterPrefillFinalizeInputRoots {
        artifact_store_roots,
        prompt_token_count,
        finalize_source_root: finalize_source_ref.root().to_string(),
        final_hidden_states_ref,
        layer_caches,
        projection_rows_per_tile,
    })
}
