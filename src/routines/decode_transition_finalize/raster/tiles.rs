use anyhow::Result;

use crate::decode_transition_finalize::raster::auth_source::RasterDecodeTransitionSource;
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;

use super::types::RasterDecodeTransitionFinalizeInput;

pub fn run_raster(
    prior_decode_state: RasterDecodeLoopState,
    completed_range_state: RasterDecodeTransitionFinalizeInput,
    source: &RasterDecodeTransitionSource<'_>,
) -> Result<RasterDecodeLoopState> {
    let output = crate::decode_layer_range::raster::finalize_state_refs_with_roots(
        completed_range_state.artifact_store_roots().clone(),
        completed_range_state,
        source,
    )?;
    RasterDecodeLoopState::new(
        output.artifact_store_roots,
        prior_decode_state.full_token_ids_ref,
        prior_decode_state.full_token_count,
        prior_decode_state.generated_token_ids_ref,
        prior_decode_state.generated_token_count,
        output.logits_ref,
        output.logit_count,
        output.layer_caches,
        output.position,
        output.token_count,
        Some(output.final_hidden_state_ref),
    )
}
