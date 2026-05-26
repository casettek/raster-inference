use anyhow::{anyhow, bail, Result};

use crate::dsl::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::artifacts::raster_artifact_store::{
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactStoreRoots,
};
use crate::shared::numerics::det_num::{
    add_sat, rms_norm as det_rms_norm, scale_act, Acc, Act, Wgt,
};
use crate::shared::raster_contracts::prefill_ple::{
    store_prefill_ple_input_manifest_with_roots, GemmaPleLayerMetadataRequest,
    GemmaPleMetadataRequest, GemmaPleModelProjectionRowRequest,
    GemmaPleProjectionNormWeightsRequest, GemmaPleScalarsRequest, GemmaPleTokenEmbeddingRowRequest,
    RasterPrefillPleSource,
};
use crate::shared::raster_kernels::transformer::{
    project_row_with_weights, validate_projection_rows_per_tile, validate_sequence_rows_per_tile,
    RasterActivationRow,
};

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterPrefillPleInputRoots,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    let (artifact_store_roots, ple_state) = call_tile!(
        init_prefill_ple_state,
        artifact_store_roots,
        input_roots,
        ple_source
    )?;
    let (artifact_store_roots, ple_state) = call_recur_seq!(
        compute_next_prefill_ple_layer_sequence,
        (artifact_store_roots, ple_state),
        ple_source
    )?;
    let (artifact_store_roots, manifest_root) = call_tile!(
        finalize_prefill_ple_input_refs,
        artifact_store_roots,
        ple_state
    )?;
    Ok((artifact_store_roots, manifest_root))
}

#[sequence(kind = recursive)]
pub fn compute_next_prefill_ple_layer_sequence(
    artifact_store_roots: RasterArtifactStoreRoots,
    ple_state: PrefillPleRasterState,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleRasterState)> {
    let (artifact_store_roots, layer_step) = call_tile!(
        init_prefill_ple_layer_step,
        artifact_store_roots,
        ple_state,
        ple_source
    )?;
    let (artifact_store_roots, layer_step) = call_seq!(
        run_prefill_ple_layer_step_sequence_ref,
        artifact_store_roots,
        layer_step,
        ple_source
    )?;
    call_tile!(
        finalize_prefill_ple_layer_step,
        artifact_store_roots,
        layer_step
    )
}

#[sequence]
fn run_prefill_ple_layer_step_sequence_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    let layer_step = call_tile!(
        prepare_prefill_ple_layer_compute_inputs,
        layer_step,
        ple_source
    )?;
    let (artifact_store_roots, layer_step) = call_seq!(
        build_scaled_token_embedding_for_layer_step,
        artifact_store_roots,
        layer_step,
        ple_source
    )?;
    let (artifact_store_roots, layer_step) = call_seq!(
        project_prefill_ple_input_for_layer_step,
        artifact_store_roots,
        layer_step,
        ple_source
    )?;
    let (artifact_store_roots, layer_step) = call_seq!(
        scale_prefill_ple_projection_for_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, layer_step) = call_seq!(
        normalize_prefill_ple_projection_for_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, layer_step) = call_seq!(
        add_prefill_ple_embedding_and_projection_for_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    call_seq!(
        scale_prefill_ple_combined_input_for_layer_step,
        artifact_store_roots,
        layer_step
    )
}

#[sequence]
fn build_scaled_token_embedding_for_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    let (artifact_store_roots, token_embedding_step) = call_tile!(
        init_scaled_token_embedding_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, token_embedding_step) = call_recur_tile!(
        append_next_scaled_token_embedding_layer_step_row,
        (artifact_store_roots, token_embedding_step),
        ple_source
    )?;
    call_tile!(
        finalize_scaled_token_embedding_layer_step,
        artifact_store_roots,
        token_embedding_step
    )
}

#[sequence]
fn project_prefill_ple_input_for_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    let (artifact_store_roots, projection_step) = call_tile!(
        init_project_prefill_ple_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, projection_step) = call_recur_tile!(
        project_next_prefill_ple_layer_step_rows,
        (artifact_store_roots, projection_step),
        ple_source
    )?;
    call_tile!(
        finalize_project_prefill_ple_layer_step,
        artifact_store_roots,
        projection_step
    )
}

#[sequence]
fn scale_prefill_ple_projection_for_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    let (artifact_store_roots, unary_step) = call_tile!(
        init_scale_prefill_ple_projection_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, unary_step) = call_recur_tile!(
        compute_next_prefill_ple_layer_unary_step,
        (artifact_store_roots, unary_step)
    )?;
    call_tile!(
        finalize_scale_prefill_ple_projection_layer_step,
        artifact_store_roots,
        unary_step
    )
}

#[sequence]
fn normalize_prefill_ple_projection_for_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    let (artifact_store_roots, unary_step) = call_tile!(
        init_normalize_prefill_ple_projection_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, unary_step) = call_recur_tile!(
        compute_next_prefill_ple_layer_unary_step,
        (artifact_store_roots, unary_step)
    )?;
    call_tile!(
        finalize_normalize_prefill_ple_projection_layer_step,
        artifact_store_roots,
        unary_step
    )
}

#[sequence]
fn add_prefill_ple_embedding_and_projection_for_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    let (artifact_store_roots, binary_step) = call_tile!(
        init_add_prefill_ple_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, binary_step) = call_recur_tile!(
        compute_next_prefill_ple_layer_binary_step,
        (artifact_store_roots, binary_step)
    )?;
    call_tile!(
        finalize_add_prefill_ple_layer_step,
        artifact_store_roots,
        binary_step
    )
}

#[sequence]
fn scale_prefill_ple_combined_input_for_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    let (artifact_store_roots, unary_step) = call_tile!(
        init_scale_prefill_ple_combined_input_layer_step,
        artifact_store_roots,
        layer_step
    )?;
    let (artifact_store_roots, unary_step) = call_recur_tile!(
        compute_next_prefill_ple_layer_unary_step,
        (artifact_store_roots, unary_step)
    )?;
    call_tile!(
        finalize_scale_prefill_ple_combined_input_layer_step,
        artifact_store_roots,
        unary_step
    )
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
pub fn init_prefill_ple_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterPrefillPleInputRoots,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillPleRasterState)> {
    if input_roots.ple_source_root != ple_source.root() {
        bail!(
            "raster PLE source root {} does not match input source root {}",
            ple_source.root(),
            input_roots.ple_source_root
        );
    }
    let metadata = auth_read!(ple_source, GemmaPleMetadataRequest)?;
    Ok((
        artifact_store_roots,
        PrefillPleRasterState {
            source_id: metadata.source_id,
            ple_source_root: input_roots.ple_source_root,
            token_ids_source_name: input_roots.token_ids_source_name,
            token_count: input_roots.token_count,
            input_activations_ref: input_roots.input_activations_ref,
            next_layer_idx: 0,
            layer_count: metadata.layer_count,
            per_layer_inputs: Vec::with_capacity(metadata.layer_count),
            has_ple_global: metadata.has_ple_global,
            raster_sizing: input_roots.raster_sizing,
        },
    ))
}

#[tile]
pub fn finalize_prefill_ple_input_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    ple_state: PrefillPleRasterState,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    if !ple_state.has_ple_global {
        return Ok((artifact_store_roots, None));
    }
    if ple_state.per_layer_inputs.len() != ple_state.layer_count {
        bail!(
            "raster PLE finalized with {} layers, expected {}",
            ple_state.per_layer_inputs.len(),
            ple_state.layer_count
        );
    }

    let (artifact_store_roots, manifest_root) = store_prefill_ple_input_manifest_with_roots(
        &artifact_store_roots,
        ple_state.source_id,
        ple_state.layer_count,
        ple_state.token_count,
        &ple_state.per_layer_inputs,
    )?;
    Ok((artifact_store_roots, Some(manifest_root)))
}

#[tile]
pub(in super::super) fn init_prefill_ple_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    ple_state: PrefillPleRasterState,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    if !ple_state.has_ple_global || ple_state.next_layer_idx >= ple_state.layer_count {
        return Ok((
            artifact_store_roots,
            PrefillPleLayerStep::Complete { ple_state },
        ));
    }

    let layer_idx = ple_state.next_layer_idx;
    let layer = auth_read!(ple_source, GemmaPleLayerMetadataRequest { layer_idx })?;
    crate::trace::trace_event(format!(
        "progress prefill.prepare_aux layer={}/{} ple={} tokens={}",
        layer_idx + 1,
        ple_state.layer_count,
        layer.has_ple,
        ple_state.token_count
    ));

    if !layer.has_ple {
        return Ok((
            artifact_store_roots,
            PrefillPleLayerStep::Skip {
                ple_state,
                layer_idx,
            },
        ));
    }

    let input_activations_ref = ple_state
        .input_activations_ref
        .as_ref()
        .ok_or_else(|| anyhow!("raster PLE state is missing input activation ref"))?
        .clone();
    let projection_rows = layer.model_projection_rows.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing model projection row metadata")
    })?;
    let ple_width = layer
        .ple_width
        .ok_or_else(|| anyhow!("Gemma PLE layer {layer_idx} is missing token embedding width"))?;

    Ok((
        artifact_store_roots,
        PrefillPleLayerStep::Compute(PrefillPleLayerComputeState {
            ple_state,
            layer_idx,
            input_activations_ref,
            ple_width,
            projection_rows,
            embedding_scale_bits: None,
            projection_scalar_bits: None,
            input_scale_bits: None,
            rms_norm_eps_bits: None,
            norm_weight_bits: None,
            embedded_ref: None,
            projected_ref: None,
            combined_ref: None,
            layer_input_ref: None,
        }),
    ))
}

#[tile]
pub(in super::super) fn prepare_prefill_ple_layer_compute_inputs(
    layer_step: PrefillPleLayerStep,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<PrefillPleLayerStep> {
    match layer_step {
        PrefillPleLayerStep::Compute(mut compute_state) => {
            let scalars = auth_read!(ple_source, GemmaPleScalarsRequest)?;
            let norm_weights = auth_read!(ple_source, GemmaPleProjectionNormWeightsRequest)?;
            compute_state.embedding_scale_bits = Some(scalars.embedding_scale.to_bits());
            compute_state.projection_scalar_bits = Some(scalars.projection_scalar.to_bits());
            compute_state.input_scale_bits = Some(scalars.input_scale.to_bits());
            compute_state.rms_norm_eps_bits = Some(scalars.rms_norm_eps.to_bits());
            compute_state.norm_weight_bits =
                Some(norm_weights.iter().map(|weight| weight.to_bits()).collect());
            Ok(PrefillPleLayerStep::Compute(compute_state))
        }
        layer_step => Ok(layer_step),
    }
}

#[tile]
fn finalize_prefill_ple_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleRasterState)> {
    match layer_step {
        PrefillPleLayerStep::Complete { ple_state } => Ok((true, artifact_store_roots, ple_state)),
        PrefillPleLayerStep::Skip {
            ple_state,
            layer_idx,
        } => update_prefill_ple_state_refs(artifact_store_roots, ple_state, layer_idx, None),
        PrefillPleLayerStep::Compute(mut compute_state) => {
            let layer_input_ref = compute_state.layer_input_ref.take().ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} did not produce a per-layer input ref",
                    compute_state.layer_idx
                )
            })?;
            update_prefill_ple_state_refs(
                artifact_store_roots,
                compute_state.ple_state,
                compute_state.layer_idx,
                Some(layer_input_ref),
            )
        }
    }
}

#[tile]
pub fn update_prefill_ple_state_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut ple_state: PrefillPleRasterState,
    layer_idx: usize,
    per_layer_input: Option<RasterActivationSequenceArtifactRef>,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleRasterState)> {
    if layer_idx != ple_state.next_layer_idx {
        bail!(
            "cannot update PLE layer {layer_idx} while next layer is {}",
            ple_state.next_layer_idx
        );
    }
    ple_state.per_layer_inputs.push(per_layer_input);
    ple_state.next_layer_idx += 1;
    Ok((
        ple_state.next_layer_idx >= ple_state.layer_count,
        artifact_store_roots,
        ple_state,
    ))
}

#[tile]
pub(in super::super) fn init_scaled_token_embedding_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerTokenEmbeddingStep)> {
    match layer_step {
        PrefillPleLayerStep::Compute(compute_state) => {
            let embedding_scale_bits = compute_state.embedding_scale_bits.ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing embedding scale metadata",
                    compute_state.layer_idx
                )
            })?;
            let (artifact_store_roots, token_embedding_state) =
                init_scaled_token_embedding_sequence_ref(
                    artifact_store_roots,
                    &compute_state.ple_state.token_ids_source_name,
                    compute_state.ple_state.token_count,
                    compute_state.layer_idx,
                    Act::from_bits(embedding_scale_bits),
                    compute_state.ple_width,
                )?;
            Ok((
                artifact_store_roots,
                PrefillPleLayerTokenEmbeddingStep::Compute {
                    compute_state,
                    operation_state: token_embedding_state,
                },
            ))
        }
        layer_step => Ok((
            artifact_store_roots,
            PrefillPleLayerTokenEmbeddingStep::Noop(layer_step),
        )),
    }
}

#[tile(kind = recursive)]
fn append_next_scaled_token_embedding_layer_step_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    token_embedding_step: PrefillPleLayerTokenEmbeddingStep,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleLayerTokenEmbeddingStep,
)> {
    match token_embedding_step {
        PrefillPleLayerTokenEmbeddingStep::Noop(layer_step) => Ok((
            true,
            artifact_store_roots,
            PrefillPleLayerTokenEmbeddingStep::Noop(layer_step),
        )),
        PrefillPleLayerTokenEmbeddingStep::Compute {
            compute_state,
            operation_state: token_embedding_state,
        } => {
            let (done, artifact_store_roots, token_embedding_state) =
                append_next_scaled_token_embedding_row(
                    artifact_store_roots,
                    token_embedding_state,
                    ple_source,
                )?;
            Ok((
                done,
                artifact_store_roots,
                PrefillPleLayerTokenEmbeddingStep::Compute {
                    compute_state,
                    operation_state: token_embedding_state,
                },
            ))
        }
    }
}

#[tile]
fn finalize_scaled_token_embedding_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    token_embedding_step: PrefillPleLayerTokenEmbeddingStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    match token_embedding_step {
        PrefillPleLayerTokenEmbeddingStep::Noop(layer_step) => {
            Ok((artifact_store_roots, layer_step))
        }
        PrefillPleLayerTokenEmbeddingStep::Compute {
            mut compute_state,
            operation_state: token_embedding_state,
        } => {
            let (artifact_store_roots, embedded_ref) =
                finalize_scaled_token_embedding_sequence_ref(
                    artifact_store_roots,
                    token_embedding_state,
                )?;
            compute_state.embedded_ref = Some(embedded_ref);
            Ok((
                artifact_store_roots,
                PrefillPleLayerStep::Compute(compute_state),
            ))
        }
    }
}

#[tile]
fn init_project_prefill_ple_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerProjectionStep)> {
    match layer_step {
        PrefillPleLayerStep::Compute(compute_state) => {
            let output_id = RasterArtifactId::new(format!(
                "prefill.prepare_aux.projected.{}",
                compute_state.layer_idx
            ))?;
            let (artifact_store_roots, projection_state) = init_ple_sequence_projection_from_ref(
                artifact_store_roots,
                compute_state.input_activations_ref.clone(),
                output_id,
                compute_state.projection_rows,
                compute_state
                    .ple_state
                    .raster_sizing
                    .projection_rows_per_tile,
            )?;
            Ok((
                artifact_store_roots,
                PrefillPleLayerProjectionStep::Compute {
                    compute_state,
                    operation_state: projection_state,
                },
            ))
        }
        layer_step => Ok((
            artifact_store_roots,
            PrefillPleLayerProjectionStep::Noop(layer_step),
        )),
    }
}

#[tile(kind = recursive)]
fn project_next_prefill_ple_layer_step_rows(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_step: PrefillPleLayerProjectionStep,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleLayerProjectionStep,
)> {
    match projection_step {
        PrefillPleLayerProjectionStep::Noop(layer_step) => Ok((
            true,
            artifact_store_roots,
            PrefillPleLayerProjectionStep::Noop(layer_step),
        )),
        PrefillPleLayerProjectionStep::Compute {
            compute_state,
            operation_state: projection_state,
        } => {
            let layer_idx = compute_state.layer_idx;
            let (done, artifact_store_roots, projection_state) = project_next_ple_sequence_rows(
                artifact_store_roots,
                projection_state,
                ple_source,
                layer_idx,
            )?;
            Ok((
                done,
                artifact_store_roots,
                PrefillPleLayerProjectionStep::Compute {
                    compute_state,
                    operation_state: projection_state,
                },
            ))
        }
    }
}

#[tile]
fn finalize_project_prefill_ple_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_step: PrefillPleLayerProjectionStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    match projection_step {
        PrefillPleLayerProjectionStep::Noop(layer_step) => Ok((artifact_store_roots, layer_step)),
        PrefillPleLayerProjectionStep::Compute {
            mut compute_state,
            operation_state: projection_state,
        } => {
            let (artifact_store_roots, projected_ref) =
                finalize_ple_sequence_projection_ref(artifact_store_roots, projection_state)?;
            compute_state.projected_ref = Some(projected_ref);
            Ok((
                artifact_store_roots,
                PrefillPleLayerStep::Compute(compute_state),
            ))
        }
    }
}

#[tile]
fn init_scale_prefill_ple_projection_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerUnaryStep)> {
    match layer_step {
        PrefillPleLayerStep::Compute(compute_state) => {
            let projected_ref = compute_state.projected_ref.clone().ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing projected activation ref",
                    compute_state.layer_idx
                )
            })?;
            let projection_scalar_bits = compute_state.projection_scalar_bits.ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing projection scalar metadata",
                    compute_state.layer_idx
                )
            })?;
            let output_id = RasterArtifactId::new(format!(
                "prefill.prepare_aux.projected.scaled.{}",
                compute_state.layer_idx
            ))?;
            let (artifact_store_roots, unary_state) = init_sequence_scale_ref_state(
                artifact_store_roots,
                projected_ref,
                output_id,
                Some(Act::from_bits(projection_scalar_bits)),
                compute_state.ple_state.raster_sizing.sequence_rows_per_tile,
            )?;
            Ok((
                artifact_store_roots,
                PrefillPleLayerUnaryStep::Compute {
                    compute_state,
                    operation_state: unary_state,
                },
            ))
        }
        layer_step => Ok((
            artifact_store_roots,
            PrefillPleLayerUnaryStep::Noop(layer_step),
        )),
    }
}

#[tile(kind = recursive)]
fn compute_next_prefill_ple_layer_unary_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_step: PrefillPleLayerUnaryStep,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleLayerUnaryStep)> {
    match unary_step {
        PrefillPleLayerUnaryStep::Noop(layer_step) => Ok((
            true,
            artifact_store_roots,
            PrefillPleLayerUnaryStep::Noop(layer_step),
        )),
        PrefillPleLayerUnaryStep::Compute {
            compute_state,
            operation_state: unary_state,
        } => {
            let (done, artifact_store_roots, unary_state) =
                compute_next_sequence_unary_ref(artifact_store_roots, unary_state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillPleLayerUnaryStep::Compute {
                    compute_state,
                    operation_state: unary_state,
                },
            ))
        }
    }
}

#[tile]
fn finalize_scale_prefill_ple_projection_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_step: PrefillPleLayerUnaryStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    finalize_prefill_ple_layer_unary_step(
        artifact_store_roots,
        unary_step,
        |compute_state, output_ref| {
            compute_state.projected_ref = Some(output_ref);
        },
    )
}

#[tile]
fn init_normalize_prefill_ple_projection_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerUnaryStep)> {
    match layer_step {
        PrefillPleLayerStep::Compute(compute_state) => {
            let projected_ref = compute_state.projected_ref.clone().ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing scaled projected activation ref",
                    compute_state.layer_idx
                )
            })?;
            let norm_weight_bits = compute_state.norm_weight_bits.as_ref().ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing norm weights",
                    compute_state.layer_idx
                )
            })?;
            let norm_weights = norm_weight_bits
                .iter()
                .copied()
                .map(Wgt::from_bits)
                .collect::<Vec<_>>();
            let rms_norm_eps_bits = compute_state.rms_norm_eps_bits.ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing RMSNorm epsilon metadata",
                    compute_state.layer_idx
                )
            })?;
            let output_id = RasterArtifactId::new(format!(
                "prefill.prepare_aux.projected.normed.{}",
                compute_state.layer_idx
            ))?;
            let (artifact_store_roots, unary_state) = init_sequence_rms_norm_ref_state(
                artifact_store_roots,
                projected_ref,
                output_id,
                Some(norm_weights.as_slice()),
                Some(Acc::from_bits(rms_norm_eps_bits)),
                compute_state.ple_state.raster_sizing.sequence_rows_per_tile,
            )?;
            Ok((
                artifact_store_roots,
                PrefillPleLayerUnaryStep::Compute {
                    compute_state,
                    operation_state: unary_state,
                },
            ))
        }
        layer_step => Ok((
            artifact_store_roots,
            PrefillPleLayerUnaryStep::Noop(layer_step),
        )),
    }
}

#[tile]
fn finalize_normalize_prefill_ple_projection_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_step: PrefillPleLayerUnaryStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    finalize_prefill_ple_layer_unary_step(
        artifact_store_roots,
        unary_step,
        |compute_state, output_ref| {
            compute_state.projected_ref = Some(output_ref);
        },
    )
}

#[tile]
fn init_add_prefill_ple_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerBinaryStep)> {
    match layer_step {
        PrefillPleLayerStep::Compute(compute_state) => {
            let embedded_ref = compute_state.embedded_ref.clone().ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing embedded activation ref",
                    compute_state.layer_idx
                )
            })?;
            let projected_ref = compute_state.projected_ref.clone().ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing normalized projected activation ref",
                    compute_state.layer_idx
                )
            })?;
            let output_id = RasterArtifactId::new(format!(
                "prefill.prepare_aux.combined.{}",
                compute_state.layer_idx
            ))?;
            let (artifact_store_roots, binary_state) = init_sequence_add_ref_state(
                artifact_store_roots,
                embedded_ref,
                projected_ref,
                output_id,
                compute_state.ple_state.raster_sizing.sequence_rows_per_tile,
            )?;
            Ok((
                artifact_store_roots,
                PrefillPleLayerBinaryStep::Compute {
                    compute_state,
                    operation_state: binary_state,
                },
            ))
        }
        layer_step => Ok((
            artifact_store_roots,
            PrefillPleLayerBinaryStep::Noop(layer_step),
        )),
    }
}

#[tile(kind = recursive)]
fn compute_next_prefill_ple_layer_binary_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_step: PrefillPleLayerBinaryStep,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleLayerBinaryStep)> {
    match binary_step {
        PrefillPleLayerBinaryStep::Noop(layer_step) => Ok((
            true,
            artifact_store_roots,
            PrefillPleLayerBinaryStep::Noop(layer_step),
        )),
        PrefillPleLayerBinaryStep::Compute {
            compute_state,
            operation_state: binary_state,
        } => {
            let (done, artifact_store_roots, binary_state) =
                compute_next_sequence_binary_ref(artifact_store_roots, binary_state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillPleLayerBinaryStep::Compute {
                    compute_state,
                    operation_state: binary_state,
                },
            ))
        }
    }
}

#[tile]
fn finalize_add_prefill_ple_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_step: PrefillPleLayerBinaryStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    match binary_step {
        PrefillPleLayerBinaryStep::Noop(layer_step) => Ok((artifact_store_roots, layer_step)),
        PrefillPleLayerBinaryStep::Compute {
            mut compute_state,
            operation_state: binary_state,
        } => {
            let (artifact_store_roots, combined_ref) =
                finalize_sequence_binary_ref_state(artifact_store_roots, binary_state)?;
            compute_state.combined_ref = Some(combined_ref);
            Ok((
                artifact_store_roots,
                PrefillPleLayerStep::Compute(compute_state),
            ))
        }
    }
}

#[tile]
fn init_scale_prefill_ple_combined_input_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillPleLayerStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerUnaryStep)> {
    match layer_step {
        PrefillPleLayerStep::Compute(compute_state) => {
            let combined_ref = compute_state.combined_ref.clone().ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing combined activation ref",
                    compute_state.layer_idx
                )
            })?;
            let input_scale_bits = compute_state.input_scale_bits.ok_or_else(|| {
                anyhow!(
                    "Gemma PLE layer {} is missing input scale metadata",
                    compute_state.layer_idx
                )
            })?;
            let output_id = RasterArtifactId::new(format!(
                "prefill.prepare_aux.per_layer_input.{}",
                compute_state.layer_idx
            ))?;
            let (artifact_store_roots, unary_state) = init_sequence_scale_ref_state(
                artifact_store_roots,
                combined_ref,
                output_id,
                Some(Act::from_bits(input_scale_bits)),
                compute_state.ple_state.raster_sizing.sequence_rows_per_tile,
            )?;
            Ok((
                artifact_store_roots,
                PrefillPleLayerUnaryStep::Compute {
                    compute_state,
                    operation_state: unary_state,
                },
            ))
        }
        layer_step => Ok((
            artifact_store_roots,
            PrefillPleLayerUnaryStep::Noop(layer_step),
        )),
    }
}

#[tile]
fn finalize_scale_prefill_ple_combined_input_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_step: PrefillPleLayerUnaryStep,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    finalize_prefill_ple_layer_unary_step(
        artifact_store_roots,
        unary_step,
        |compute_state, output_ref| {
            compute_state.layer_input_ref = Some(output_ref);
        },
    )
}

fn finalize_prefill_ple_layer_unary_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_step: PrefillPleLayerUnaryStep,
    update_compute_state: impl FnOnce(
        &mut PrefillPleLayerComputeState,
        RasterActivationSequenceArtifactRef,
    ),
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerStep)> {
    match unary_step {
        PrefillPleLayerUnaryStep::Noop(layer_step) => Ok((artifact_store_roots, layer_step)),
        PrefillPleLayerUnaryStep::Compute {
            mut compute_state,
            operation_state: unary_state,
        } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_ref_state(artifact_store_roots, unary_state)?;
            update_compute_state(&mut compute_state, output_ref);
            Ok((
                artifact_store_roots,
                PrefillPleLayerStep::Compute(compute_state),
            ))
        }
    }
}

#[tile]
pub(in super::super) fn init_scaled_token_embedding_sequence_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    token_ids_source_name: &str,
    token_count: usize,
    layer_idx: usize,
    scale: crate::shared::numerics::det_num::Act,
    row_width: usize,
) -> Result<(RasterArtifactStoreRoots, PrefillPleTokenEmbeddingState)> {
    if token_count == 0 {
        bail!("transformer PLE token embedding sequence requires at least one token");
    }
    let output_id = RasterArtifactId::new(format!("prefill.prepare_aux.embedded.{layer_idx}"))?;
    let output_source_name = output_id.source_name().to_string();
    let (artifact_store_roots, _output_builder_ref) = start_sequence_builder_with_roots(
        &artifact_store_roots,
        output_id,
        token_count,
        row_width,
    )?;
    Ok((
        artifact_store_roots,
        PrefillPleTokenEmbeddingState {
            token_ids_source_name: token_ids_source_name.to_string(),
            layer_idx,
            scale_bits: scale.to_bits(),
            next_token_idx: 0,
            token_count,
            row_width,
            output_source_name,
        },
    ))
}

#[tile(kind = recursive)]
pub(in super::super) fn append_next_scaled_token_embedding_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut token_embedding_state: PrefillPleTokenEmbeddingState,
    ple_source: &RasterPrefillPleSource<'_>,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleTokenEmbeddingState,
)> {
    if token_embedding_state.is_complete() {
        return Ok((true, artifact_store_roots, token_embedding_state));
    }

    let token_id = read_prefill_token_id(
        &artifact_store_roots,
        token_ids_root(
            &artifact_store_roots,
            &token_embedding_state.token_ids_source_name,
        )?,
        token_embedding_state.token_count,
        token_embedding_state.next_token_idx,
    )?;
    let row = auth_read!(
        ple_source,
        GemmaPleTokenEmbeddingRowRequest {
            layer_idx: token_embedding_state.layer_idx,
            token_id,
        },
    )?;
    if row.len() != token_embedding_state.row_width {
        bail!(
            "PLE token embedding row has width {}, expected {}",
            row.len(),
            token_embedding_state.row_width
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&token_embedding_state.output_source_name)?
        .to_string();
    let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
        &artifact_store_roots,
        &output_builder_root,
        token_embedding_state.next_token_idx,
        RasterActivationRow::from_acts(
            row.into_iter()
                .map(|value| {
                    scale_act(
                        value,
                        crate::shared::numerics::det_num::Act::from_bits(
                            token_embedding_state.scale_bits,
                        ),
                    )
                })
                .collect(),
        ),
    )?;
    artifact_store_roots = next_roots;
    token_embedding_state.next_token_idx += 1;
    Ok((false, artifact_store_roots, token_embedding_state))
}

#[tile]
pub(in super::super) fn finalize_scaled_token_embedding_sequence_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    token_embedding_state: PrefillPleTokenEmbeddingState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if !token_embedding_state.is_complete() {
        bail!(
            "PLE token embedding finalized at token {}, expected {}",
            token_embedding_state.next_token_idx,
            token_embedding_state.token_count
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&token_embedding_state.output_source_name)?
        .to_string();
    finalize_sequence_builder_by_root_with_roots(&artifact_store_roots, &output_builder_root)
}

#[tile]
fn init_ple_sequence_projection_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceArtifactRef,
    output_id: RasterArtifactId,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, PrefillPleSequenceProjectionState)> {
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }
    validate_projection_rows_per_tile(projection_rows_per_tile)?;
    let token_count = input_ref.row_count();
    let input_width = input_ref.width();
    let output_source_name = output_id.source_name().to_string();
    let (artifact_store_roots, _output_builder) = start_sequence_builder_with_roots(
        &artifact_store_roots,
        output_id,
        token_count,
        projection_rows,
    )?;
    Ok((
        artifact_store_roots,
        PrefillPleSequenceProjectionState {
            input_ref,
            output_source_name,
            current_row_bits: Vec::new(),
            next_token_idx: 0,
            next_projection_row_idx: 0,
            token_count,
            input_width,
            projection_rows,
            rows_per_tile: projection_rows_per_tile,
        },
    ))
}

#[tile(kind = recursive)]
pub fn project_next_ple_sequence_rows(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut projection_state: PrefillPleSequenceProjectionState,
    ple_source: &RasterPrefillPleSource<'_>,
    layer_idx: usize,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleSequenceProjectionState,
)> {
    if projection_state.is_complete() {
        return Ok((true, artifact_store_roots, projection_state));
    }

    let end = projection_state
        .next_projection_row_idx
        .saturating_add(projection_state.rows_per_tile)
        .min(projection_state.projection_rows);
    let start_projection_row_idx = projection_state.next_projection_row_idx;
    let start_token_idx = projection_state.next_token_idx;
    let mut rows = Vec::with_capacity(end - projection_state.next_projection_row_idx);
    for row_idx in projection_state.next_projection_row_idx..end {
        rows.push(auth_read!(
            ple_source,
            GemmaPleModelProjectionRowRequest { layer_idx, row_idx },
        )?);
    }
    let input_row =
        read_activation_row_from_ref(&projection_state.input_ref, projection_state.next_token_idx)?;
    let output_bits = rows
        .iter()
        .map(|projection_row| {
            project_row_with_weights(&input_row, projection_row)
                .map(|projected| projected.to_bits())
        })
        .collect::<Result<Vec<_>>>()?;
    projection_state.current_row_bits.extend(output_bits);
    projection_state.next_projection_row_idx = end;
    if projection_state.next_projection_row_idx == projection_state.projection_rows {
        if projection_state.current_row_bits.len() != projection_state.projection_rows {
            bail!(
                "raster projection output row has width {}, expected {}",
                projection_state.current_row_bits.len(),
                projection_state.projection_rows
            );
        }
        let output_builder_root = artifact_store_roots
            .builder_root_for_source_name(&projection_state.output_source_name)?
            .to_string();
        let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
            &artifact_store_roots,
            &output_builder_root,
            projection_state.next_token_idx,
            RasterActivationRow::from_act_bits(std::mem::take(
                &mut projection_state.current_row_bits,
            )),
        )?;
        artifact_store_roots = next_roots;
        projection_state.next_token_idx += 1;
        projection_state.next_projection_row_idx = 0;
    }
    crate::trace::trace_event(format!(
        "progress prefill.prepare_aux.projection layer={} token={}/{} projection_rows={}..{} of {} input_width={}",
        layer_idx,
        start_token_idx + 1,
        projection_state.token_count,
        start_projection_row_idx,
        end,
        projection_state.projection_rows,
        projection_state.input_width
    ));
    Ok((false, artifact_store_roots, projection_state))
}

#[tile]
fn finalize_ple_sequence_projection_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_state: PrefillPleSequenceProjectionState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if projection_state.next_token_idx != projection_state.token_count
        || projection_state.next_projection_row_idx != 0
    {
        bail!(
            "raster projection completed token {}, projection row {}, expected {} complete token rows",
            projection_state.next_token_idx,
            projection_state.next_projection_row_idx,
            projection_state.token_count
        );
    }
    if !projection_state.current_row_bits.is_empty() {
        bail!(
            "raster projection finalized with partial row width {}",
            projection_state.current_row_bits.len()
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&projection_state.output_source_name)?
        .to_string();
    finalize_sequence_builder_by_root_with_roots(&artifact_store_roots, &output_builder_root)
}

#[tile]
fn init_sequence_scale_ref_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceArtifactRef,
    output_id: RasterArtifactId,
    scalar: Option<crate::shared::numerics::det_num::Act>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, PrefillPleSequenceUnaryState)> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let scalar = scalar
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    let row_count = input_ref.row_count();
    let width = input_ref.width();
    let output_source_name = output_id.source_name().to_string();
    let (artifact_store_roots, _output_builder) =
        start_sequence_builder_with_roots(&artifact_store_roots, output_id, row_count, width)?;
    Ok((
        artifact_store_roots,
        PrefillPleSequenceUnaryState {
            input_ref,
            output_source_name,
            op: PrefillPleSequenceUnaryOp::Scale {
                scalar_bits: scalar.to_bits(),
            },
            next_row_idx: 0,
            row_count,
            width,
            rows_per_tile,
        },
    ))
}

#[tile(kind = recursive)]
fn compute_next_sequence_unary_ref(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut unary_state: PrefillPleSequenceUnaryState,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleSequenceUnaryState)> {
    if unary_state.is_complete() {
        return Ok((true, artifact_store_roots, unary_state));
    }
    let end = unary_state
        .next_row_idx
        .saturating_add(unary_state.rows_per_tile)
        .min(unary_state.row_count);
    while unary_state.next_row_idx < end {
        let row = read_activation_row_from_ref(&unary_state.input_ref, unary_state.next_row_idx)?;
        let output_row = match &unary_state.op {
            PrefillPleSequenceUnaryOp::RmsNorm {
                norm_weight_bits,
                eps_bits,
            } => {
                let norm_weights = norm_weight_bits
                    .iter()
                    .copied()
                    .map(Wgt::from_bits)
                    .collect::<Vec<_>>();
                RasterActivationRow::from_acts(det_rms_norm(
                    &row.acts(),
                    &norm_weights,
                    Acc::from_bits(*eps_bits),
                ))
            }
            PrefillPleSequenceUnaryOp::Scale { scalar_bits } => RasterActivationRow::from_acts(
                row.acts()
                    .into_iter()
                    .map(|value| scale_act(value, Act::from_bits(*scalar_bits)))
                    .collect(),
            ),
        };
        let output_builder_root = artifact_store_roots
            .builder_root_for_source_name(&unary_state.output_source_name)?
            .to_string();
        let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
            &artifact_store_roots,
            &output_builder_root,
            unary_state.next_row_idx,
            output_row,
        )?;
        artifact_store_roots = next_roots;
        unary_state.next_row_idx += 1;
    }
    Ok((false, artifact_store_roots, unary_state))
}

#[tile]
fn finalize_sequence_unary_ref_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_state: PrefillPleSequenceUnaryState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if !unary_state.is_complete() {
        bail!(
            "sequence unary state completed {} rows, expected {}",
            unary_state.next_row_idx,
            unary_state.row_count
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&unary_state.output_source_name)?
        .to_string();
    finalize_sequence_builder_by_root_with_roots(&artifact_store_roots, &output_builder_root)
}

#[tile]
fn init_sequence_rms_norm_ref_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceArtifactRef,
    output_id: RasterArtifactId,
    norm_weights: Option<&[crate::shared::numerics::det_num::Wgt]>,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, PrefillPleSequenceUnaryState)> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical norm weights"))?;
    if norm_weights.len() != input_ref.width() {
        bail!(
            "RMSNorm weight length {} does not match sequence width {}",
            norm_weights.len(),
            input_ref.width()
        );
    }
    let eps = eps.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical epsilon"))?;
    let row_count = input_ref.row_count();
    let width = input_ref.width();
    let output_source_name = output_id.source_name().to_string();
    let (artifact_store_roots, _output_builder) =
        start_sequence_builder_with_roots(&artifact_store_roots, output_id, row_count, width)?;
    Ok((
        artifact_store_roots,
        PrefillPleSequenceUnaryState {
            input_ref,
            output_source_name,
            op: PrefillPleSequenceUnaryOp::RmsNorm {
                norm_weight_bits: norm_weights.iter().map(|weight| weight.to_bits()).collect(),
                eps_bits: eps.to_bits(),
            },
            next_row_idx: 0,
            row_count,
            width,
            rows_per_tile,
        },
    ))
}

#[tile]
fn init_sequence_add_ref_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceArtifactRef,
    rhs_ref: RasterActivationSequenceArtifactRef,
    output_id: RasterArtifactId,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, PrefillPleSequenceBinaryState)> {
    validate_sequence_rows_per_tile(rows_per_tile)?;
    if lhs_ref.row_count() != rhs_ref.row_count() {
        bail!(
            "sequence length mismatch: {} vs {}",
            lhs_ref.row_count(),
            rhs_ref.row_count()
        );
    }
    if lhs_ref.width() != rhs_ref.width() {
        bail!(
            "right sequence row 0 has width {}, expected {}",
            rhs_ref.width(),
            lhs_ref.width()
        );
    }
    let row_count = lhs_ref.row_count();
    let width = lhs_ref.width();
    let output_source_name = output_id.source_name().to_string();
    let (artifact_store_roots, _output_builder) =
        start_sequence_builder_with_roots(&artifact_store_roots, output_id, row_count, width)?;
    Ok((
        artifact_store_roots,
        PrefillPleSequenceBinaryState {
            lhs_ref,
            rhs_ref,
            output_source_name,
            next_row_idx: 0,
            row_count,
            width,
            rows_per_tile,
        },
    ))
}

#[tile(kind = recursive)]
fn compute_next_sequence_binary_ref(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut binary_state: PrefillPleSequenceBinaryState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleSequenceBinaryState,
)> {
    if binary_state.is_complete() {
        return Ok((true, artifact_store_roots, binary_state));
    }
    let end = binary_state
        .next_row_idx
        .saturating_add(binary_state.rows_per_tile)
        .min(binary_state.row_count);
    while binary_state.next_row_idx < end {
        let lhs_row =
            read_activation_row_from_ref(&binary_state.lhs_ref, binary_state.next_row_idx)?;
        let rhs_row =
            read_activation_row_from_ref(&binary_state.rhs_ref, binary_state.next_row_idx)?;
        let output_row = RasterActivationRow::from_acts(
            lhs_row
                .acts()
                .into_iter()
                .zip(rhs_row.acts())
                .map(|(lhs_value, rhs_value)| add_sat(lhs_value, rhs_value))
                .collect(),
        );
        let output_builder_root = artifact_store_roots
            .builder_root_for_source_name(&binary_state.output_source_name)?
            .to_string();
        let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
            &artifact_store_roots,
            &output_builder_root,
            binary_state.next_row_idx,
            output_row,
        )?;
        artifact_store_roots = next_roots;
        binary_state.next_row_idx += 1;
    }
    Ok((false, artifact_store_roots, binary_state))
}

#[tile]
fn finalize_sequence_binary_ref_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_state: PrefillPleSequenceBinaryState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if !binary_state.is_complete() {
        bail!(
            "sequence binary state completed {} rows, expected {}",
            binary_state.next_row_idx,
            binary_state.row_count
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&binary_state.output_source_name)?
        .to_string();
    finalize_sequence_builder_by_root_with_roots(&artifact_store_roots, &output_builder_root)
}
