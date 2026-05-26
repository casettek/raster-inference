use anyhow::Result;

use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4TransformerModel, LayerKvCache, TransformerPrefillResult,
};
use crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef;

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

#[cfg(test)]
pub(crate) fn materialize_raster_input_roots_for_api(
    artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: Vec<crate::prefill_layer::raster::PrefillLayerCacheSlot>,
    projection_rows_per_tile: usize,
) -> Result<TransformerPrefillResult> {
    let finalize_source_ref = finalize_source.committed_source_ref()?;
    raster::materialize_prefill_result_for_api(raster::RasterPrefillFinalizeInputRoots {
        artifact_store_roots,
        prompt_token_count,
        finalize_source_root: finalize_source_ref.root().to_string(),
        final_hidden_states_ref,
        layer_caches,
        projection_rows_per_tile,
    })
}

pub fn run_raster(
    artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: Vec<crate::prefill_layer::raster::PrefillLayerCacheSlot>,
    projection_rows_per_tile: usize,
) -> Result<raster::RasterPrefillFinalizeOutput> {
    let finalize_source =
        raster::auth_source::RasterPrefillFinalizeSource::for_current_integrity_mode(
            finalize_source,
        )?;
    raster::main(
        raster::RasterPrefillFinalizeInputRoots {
            artifact_store_roots,
            prompt_token_count,
            finalize_source_root: finalize_source.root().to_string(),
            final_hidden_states_ref,
            layer_caches,
            projection_rows_per_tile,
        },
        &finalize_source,
    )
}

/// Compatibility boundary: materializes raster prefill refs into the public
/// `TransformerPrefillResult` shape used by checkpoints and non-ref callers.
pub fn materialize_raster_output_refs_for_api(
    output: &raster::RasterPrefillFinalizeOutput,
) -> Result<TransformerPrefillResult> {
    raster::utils::build_prefill_result_from_root_refs(&output.artifact_store_roots, &output.refs)
}
