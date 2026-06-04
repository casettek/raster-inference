use anyhow::{bail, Result};

use crate::dsl::prelude::tile;
use crate::prefill_range::raster::{PrefillLayerRasterState, PrefillLayerStep};
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;

use super::utils::update_prefill_range_state_refs_with_roots;

#[tile]
pub(crate) fn finalize_prefill_range_step_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillLayerStep,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    match layer_step {
        PrefillLayerStep::Complete { layer_state } => Ok((true, artifact_store_roots, layer_state)),
        PrefillLayerStep::Compute { layer_state, .. } => {
            bail!(
                "prefill range layer {} reached finalization before compute completed",
                layer_state.next_layer_idx
            )
        }
        PrefillLayerStep::Computed {
            layer_state,
            layer_idx,
            layer_output_ref,
            layer_cache,
        } => update_prefill_range_state_refs_with_roots(
            artifact_store_roots,
            layer_state,
            layer_idx,
            layer_output_ref,
            layer_cache,
        ),
    }
}
