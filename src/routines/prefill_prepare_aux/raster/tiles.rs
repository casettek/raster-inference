use anyhow::{anyhow, bail, Result};

use crate::dsl::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactStoreRoots,
    RasterRoutineOutput,
};
use crate::shared::model::transformer::ActivationSequence;
use crate::shared::numerics::det_num::{
    add_sat, rms_norm as det_rms_norm, scale_act, Acc, Act, Wgt,
};
use crate::shared::raster_contracts::prefill_ple::{
    store_prefill_ple_input_manifest_with_roots, AuthenticatedGemmaPleSource,
    GemmaPleLayerMetadataRequest, GemmaPleMetadataRequest, GemmaPleModelProjectionRowRequest,
    GemmaPleProjectionNormWeightsRequest, GemmaPleScalarsRequest, GemmaPleTokenEmbeddingRowRequest,
};
use crate::shared::raster_kernels::transformer::{
    project_row_with_weights, validate_projection_rows_per_tile, validate_sequence_rows_per_tile,
    RasterActivationRow,
};
use crate::RasterSizingControls;

use super::utils::{
    append_sequence_row_by_builder_root_with_roots, finalize_sequence_builder_by_root_with_roots,
    insert_activation_sequence_with_roots, raster_activation_sequence_from_embedding,
    read_activation_row_from_ref, read_prefill_token_id, reset_artifact_store,
    start_sequence_builder_with_roots, store_prefill_token_ids_artifact_with_roots,
};

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterPrefillPleInputRoots,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    let (artifact_store_roots, state) =
        call_tile!(init_prefill_ple_state, artifact_store_roots, input_roots)?;
    let (artifact_store_roots, state) = call_recur_seq!(
        compute_next_prefill_ple_layer_sequence,
        (artifact_store_roots, state),
        ple_source
    )?;
    let (artifact_store_roots, manifest_root) =
        call_tile!(finalize_prefill_ple_input_refs, artifact_store_roots, state)?;
    Ok((artifact_store_roots, manifest_root))
}

#[sequence(kind = recursive)]
pub fn compute_next_prefill_ple_layer_sequence(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: PrefillPleRasterState,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleRasterState)> {
    if !state.has_ple_global || state.next_layer_idx >= state.layer_count {
        return Ok((true, artifact_store_roots, state));
    }

    let (artifact_store_roots, context) = call_tile!(
        prepare_next_prefill_ple_context,
        artifact_store_roots,
        &state,
        ple_source
    )?;
    crate::trace::trace_event(format!(
        "progress prefill.prepare_aux layer={}/{} ple={} tokens={}",
        context.layer_idx + 1,
        state.layer_count,
        context.has_ple,
        state.token_count
    ));

    if !context.has_ple {
        return call_tile!(
            update_prefill_ple_state_refs,
            artifact_store_roots,
            state,
            context.layer_idx,
            None
        );
    }

    let input_activations_ref = state
        .input_activations_ref
        .as_ref()
        .ok_or_else(|| anyhow!("raster PLE state is missing input activation ref"))?
        .clone();
    let (artifact_store_roots, layer_input_ref) = call_seq!(
        run_prefill_ple_layer_sequence_ref,
        artifact_store_roots,
        &state.token_ids_source_name,
        state.token_count,
        input_activations_ref,
        ple_source,
        &context,
        state.raster_sizing
    )?;

    call_tile!(
        update_prefill_ple_state_refs,
        artifact_store_roots,
        state,
        context.layer_idx,
        Some(layer_input_ref)
    )
}

#[sequence]
fn run_prefill_ple_layer_sequence_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    token_ids_source_name: &str,
    token_count: usize,
    input_activations_ref: RasterActivationSequenceArtifactRef,
    ple_source: &AuthenticatedGemmaPleSource,
    context: &PrefillPleLayerContext,
    raster_sizing: RasterSizingControls,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let layer_idx = context.layer_idx;
    let embedding_scale = Act::from_bits(context.embedding_scale_bits.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing embedding scale metadata")
    })?);
    let projection_scalar = Act::from_bits(context.projection_scalar_bits.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing projection scalar metadata")
    })?);
    let input_scale =
        Act::from_bits(context.input_scale_bits.ok_or_else(|| {
            anyhow!("Gemma PLE layer {layer_idx} is missing input scale metadata")
        })?);
    let rms_norm_eps = Acc::from_bits(context.rms_norm_eps_bits.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing RMSNorm epsilon metadata")
    })?);
    let norm_weights = context
        .norm_weight_bits
        .as_ref()
        .ok_or_else(|| anyhow!("Gemma PLE layer {layer_idx} is missing norm weights"))?
        .iter()
        .map(|bits| Wgt::from_bits(*bits))
        .collect::<Vec<_>>();
    let ple_width = context
        .ple_width
        .ok_or_else(|| anyhow!("Gemma PLE layer {layer_idx} is missing token embedding width"))?;
    let projection_rows = context.projection_rows.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing model projection row metadata")
    })?;
    let (artifact_store_roots, embedded_ref) = call_seq!(
        build_scaled_token_embedding_sequence_ref,
        artifact_store_roots,
        token_ids_source_name,
        token_count,
        layer_idx,
        ple_source,
        embedding_scale,
        ple_width
    )?;
    let (artifact_store_roots, projected_ref) = call_seq!(
        project_ple_sequence_with_source_ref,
        artifact_store_roots,
        input_activations_ref.clone(),
        ple_source,
        layer_idx,
        projection_rows,
        raster_sizing.projection_rows_per_tile,
        RasterArtifactId::new(format!("prefill.prepare_aux.projected.{layer_idx}"))?
    )?;
    let (artifact_store_roots, projected_ref) = call_seq!(
        compute_sequence_scale_ref,
        artifact_store_roots,
        projected_ref,
        RasterArtifactId::new(format!("prefill.prepare_aux.projected.scaled.{layer_idx}"))?,
        Some(projection_scalar),
        raster_sizing.sequence_rows_per_tile
    )?;
    let (artifact_store_roots, projected_ref) = call_seq!(
        compute_sequence_rms_norm_ref,
        artifact_store_roots,
        projected_ref,
        RasterArtifactId::new(format!("prefill.prepare_aux.projected.normed.{layer_idx}"))?,
        Some(norm_weights.as_slice()),
        Some(rms_norm_eps),
        raster_sizing.sequence_rows_per_tile
    )?;
    let (artifact_store_roots, combined_ref) = call_seq!(
        compute_sequence_add_ref,
        artifact_store_roots,
        embedded_ref,
        projected_ref,
        RasterArtifactId::new(format!("prefill.prepare_aux.combined.{layer_idx}"))?,
        raster_sizing.sequence_rows_per_tile
    )?;
    call_seq!(
        compute_sequence_scale_ref,
        artifact_store_roots,
        combined_ref,
        RasterArtifactId::new(format!("prefill.prepare_aux.per_layer_input.{layer_idx}"))?,
        Some(input_scale),
        raster_sizing.sequence_rows_per_tile
    )
}

#[sequence]
fn build_scaled_token_embedding_sequence_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    token_ids_source_name: &str,
    token_count: usize,
    layer_idx: usize,
    ple_source: &AuthenticatedGemmaPleSource,
    scale: crate::shared::numerics::det_num::Act,
    row_width: usize,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let (artifact_store_roots, state) = call_tile!(
        init_scaled_token_embedding_sequence_ref,
        artifact_store_roots,
        token_ids_source_name,
        token_count,
        layer_idx,
        scale,
        row_width
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        append_next_scaled_token_embedding_row,
        (artifact_store_roots, state),
        ple_source
    )?;
    call_tile!(
        finalize_scaled_token_embedding_sequence_ref,
        artifact_store_roots,
        state
    )
}

#[sequence]
fn project_ple_sequence_with_source_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceArtifactRef,
    ple_source: &AuthenticatedGemmaPleSource,
    layer_idx: usize,
    projection_rows: usize,
    projection_rows_per_tile: usize,
    output_id: RasterArtifactId,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let (artifact_store_roots, state) = call_tile!(
        init_ple_sequence_projection_from_ref,
        artifact_store_roots,
        input_ref,
        output_id,
        projection_rows,
        projection_rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        project_next_ple_sequence_rows,
        (artifact_store_roots, state),
        ple_source,
        layer_idx
    )?;
    call_tile!(
        finalize_ple_sequence_projection_ref,
        artifact_store_roots,
        state
    )
}

#[sequence]
fn compute_sequence_scale_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceArtifactRef,
    output_id: RasterArtifactId,
    scalar: Option<crate::shared::numerics::det_num::Act>,
    rows_per_tile: usize,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let (artifact_store_roots, state) = call_tile!(
        init_sequence_scale_ref_state,
        artifact_store_roots,
        input_ref,
        output_id,
        scalar,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        compute_next_sequence_unary_ref,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_sequence_unary_ref_state,
        artifact_store_roots,
        state
    )
}

#[sequence]
fn compute_sequence_rms_norm_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceArtifactRef,
    output_id: RasterArtifactId,
    norm_weights: Option<&[crate::shared::numerics::det_num::Wgt]>,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let (artifact_store_roots, state) = call_tile!(
        init_sequence_rms_norm_ref_state,
        artifact_store_roots,
        input_ref,
        output_id,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        compute_next_sequence_unary_ref,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_sequence_unary_ref_state,
        artifact_store_roots,
        state
    )
}

#[sequence]
fn compute_sequence_add_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceArtifactRef,
    rhs_ref: RasterActivationSequenceArtifactRef,
    output_id: RasterArtifactId,
    rows_per_tile: usize,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let (artifact_store_roots, state) = call_tile!(
        init_sequence_add_ref_state,
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        output_id,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        compute_next_sequence_binary_ref,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_sequence_binary_ref_state,
        artifact_store_roots,
        state
    )
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
pub fn init_prefill_ple_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_roots: RasterPrefillPleInputRoots,
) -> Result<(RasterArtifactStoreRoots, PrefillPleRasterState)> {
    Ok((
        artifact_store_roots,
        PrefillPleRasterState {
            source_id: input_roots.source_id,
            token_ids_source_name: input_roots.token_ids_source_name,
            token_count: input_roots.token_count,
            input_activations_ref: input_roots.input_activations_ref,
            next_layer_idx: 0,
            layer_count: input_roots.layer_count,
            per_layer_inputs: Vec::with_capacity(input_roots.layer_count),
            has_ple_global: input_roots.has_ple_global,
            raster_sizing: input_roots.raster_sizing,
        },
    ))
}

#[tile]
pub fn finalize_prefill_ple_input_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: PrefillPleRasterState,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    if !state.has_ple_global {
        return Ok((artifact_store_roots, None));
    }
    if state.per_layer_inputs.len() != state.layer_count {
        bail!(
            "raster PLE finalized with {} layers, expected {}",
            state.per_layer_inputs.len(),
            state.layer_count
        );
    }

    let (artifact_store_roots, manifest_root) = store_prefill_ple_input_manifest_with_roots(
        &artifact_store_roots,
        state.source_id,
        state.layer_count,
        state.token_count,
        &state.per_layer_inputs,
    )?;
    Ok((artifact_store_roots, Some(manifest_root)))
}

#[tile]
pub fn prepare_next_prefill_ple_context(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: &PrefillPleRasterState,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<(RasterArtifactStoreRoots, PrefillPleLayerContext)> {
    if !state.has_ple_global {
        bail!("cannot prepare PLE layer context without global PLE weights");
    }
    if state.next_layer_idx >= state.layer_count {
        bail!(
            "cannot prepare PLE layer {} after completing {} layers",
            state.next_layer_idx,
            state.layer_count
        );
    }
    let layer_idx = state.next_layer_idx;
    let layer = auth_read!(ple_source, GemmaPleLayerMetadataRequest { layer_idx })?;
    if !layer.has_ple {
        return Ok((
            artifact_store_roots,
            PrefillPleLayerContext {
                layer_idx,
                has_ple: false,
                ple_width: None,
                projection_rows: None,
                embedding_scale_bits: None,
                projection_scalar_bits: None,
                input_scale_bits: None,
                rms_norm_eps_bits: None,
                norm_weight_bits: None,
            },
        ));
    }

    state
        .input_activations_ref
        .as_ref()
        .ok_or_else(|| anyhow!("raster PLE state is missing input activation ref"))?;
    let projection_rows = layer.model_projection_rows.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing model projection row metadata")
    })?;
    let scalars = auth_read!(ple_source, GemmaPleScalarsRequest)?;
    let norm_weights = auth_read!(ple_source, GemmaPleProjectionNormWeightsRequest)?;
    let ple_width = layer
        .ple_width
        .ok_or_else(|| anyhow!("Gemma PLE layer {layer_idx} is missing token embedding width"))?;

    Ok((
        artifact_store_roots,
        PrefillPleLayerContext {
            layer_idx,
            has_ple: true,
            ple_width: Some(ple_width),
            projection_rows: Some(projection_rows),
            embedding_scale_bits: Some(scalars.embedding_scale.to_bits()),
            projection_scalar_bits: Some(scalars.projection_scalar.to_bits()),
            input_scale_bits: Some(scalars.input_scale.to_bits()),
            rms_norm_eps_bits: Some(scalars.rms_norm_eps.to_bits()),
            norm_weight_bits: Some(norm_weights.iter().map(|weight| weight.to_bits()).collect()),
        },
    ))
}

#[tile]
pub fn update_prefill_ple_state_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut state: PrefillPleRasterState,
    layer_idx: usize,
    per_layer_input: Option<RasterActivationSequenceArtifactRef>,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleRasterState)> {
    if layer_idx != state.next_layer_idx {
        bail!(
            "cannot update PLE layer {layer_idx} while next layer is {}",
            state.next_layer_idx
        );
    }
    state.per_layer_inputs.push(per_layer_input);
    state.next_layer_idx += 1;
    Ok((
        state.next_layer_idx >= state.layer_count,
        artifact_store_roots,
        state,
    ))
}

#[tile]
fn init_scaled_token_embedding_sequence_ref(
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
fn append_next_scaled_token_embedding_row(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    mut state: PrefillPleTokenEmbeddingState,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleTokenEmbeddingState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let token_id = read_prefill_token_id(
        &artifact_store_roots,
        token_ids_root(&artifact_store_roots, &state.token_ids_source_name)?,
        state.token_count,
        state.next_token_idx,
    )?;
    let row = auth_read!(
        ple_source,
        GemmaPleTokenEmbeddingRowRequest {
            layer_idx: state.layer_idx,
            token_id,
        },
    )?;
    if row.len() != state.row_width {
        bail!(
            "PLE token embedding row has width {}, expected {}",
            row.len(),
            state.row_width
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&state.output_source_name)?
        .to_string();
    let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
        &artifact_store_roots,
        &output_builder_root,
        state.next_token_idx,
        RasterActivationRow::from_acts(
            row.into_iter()
                .map(|value| {
                    scale_act(
                        value,
                        crate::shared::numerics::det_num::Act::from_bits(state.scale_bits),
                    )
                })
                .collect(),
        ),
    )?;
    artifact_store_roots = next_roots;
    state.next_token_idx += 1;
    Ok((false, artifact_store_roots, state))
}

#[tile]
fn finalize_scaled_token_embedding_sequence_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: PrefillPleTokenEmbeddingState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if !state.is_complete() {
        bail!(
            "PLE token embedding finalized at token {}, expected {}",
            state.next_token_idx,
            state.token_count
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&state.output_source_name)?
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
    mut state: PrefillPleSequenceProjectionState,
    ple_source: &AuthenticatedGemmaPleSource,
    layer_idx: usize,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleSequenceProjectionState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }

    let end = state
        .next_projection_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.projection_rows);
    let start_projection_row_idx = state.next_projection_row_idx;
    let start_token_idx = state.next_token_idx;
    let mut rows = Vec::with_capacity(end - state.next_projection_row_idx);
    for row_idx in state.next_projection_row_idx..end {
        rows.push(auth_read!(
            ple_source,
            GemmaPleModelProjectionRowRequest { layer_idx, row_idx },
        )?);
    }
    let input_row = read_activation_row_from_ref(&state.input_ref, state.next_token_idx)?;
    let output_bits = rows
        .iter()
        .map(|projection_row| {
            project_row_with_weights(&input_row, projection_row)
                .map(|projected| projected.to_bits())
        })
        .collect::<Result<Vec<_>>>()?;
    state.current_row_bits.extend(output_bits);
    state.next_projection_row_idx = end;
    if state.next_projection_row_idx == state.projection_rows {
        if state.current_row_bits.len() != state.projection_rows {
            bail!(
                "raster projection output row has width {}, expected {}",
                state.current_row_bits.len(),
                state.projection_rows
            );
        }
        let output_builder_root = artifact_store_roots
            .builder_root_for_source_name(&state.output_source_name)?
            .to_string();
        let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
            &artifact_store_roots,
            &output_builder_root,
            state.next_token_idx,
            RasterActivationRow::from_act_bits(std::mem::take(&mut state.current_row_bits)),
        )?;
        artifact_store_roots = next_roots;
        state.next_token_idx += 1;
        state.next_projection_row_idx = 0;
    }
    crate::trace::trace_event(format!(
        "progress prefill.prepare_aux.projection layer={} token={}/{} projection_rows={}..{} of {} input_width={}",
        layer_idx,
        start_token_idx + 1,
        state.token_count,
        start_projection_row_idx,
        end,
        state.projection_rows,
        state.input_width
    ));
    Ok((false, artifact_store_roots, state))
}

#[tile]
fn finalize_ple_sequence_projection_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: PrefillPleSequenceProjectionState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if state.next_token_idx != state.token_count || state.next_projection_row_idx != 0 {
        bail!(
            "raster projection completed token {}, projection row {}, expected {} complete token rows",
            state.next_token_idx,
            state.next_projection_row_idx,
            state.token_count
        );
    }
    if !state.current_row_bits.is_empty() {
        bail!(
            "raster projection finalized with partial row width {}",
            state.current_row_bits.len()
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&state.output_source_name)?
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
    mut state: PrefillPleSequenceUnaryState,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillPleSequenceUnaryState)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }
    let end = state
        .next_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.row_count);
    while state.next_row_idx < end {
        let row = read_activation_row_from_ref(&state.input_ref, state.next_row_idx)?;
        let output_row = match &state.op {
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
            .builder_root_for_source_name(&state.output_source_name)?
            .to_string();
        let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
            &artifact_store_roots,
            &output_builder_root,
            state.next_row_idx,
            output_row,
        )?;
        artifact_store_roots = next_roots;
        state.next_row_idx += 1;
    }
    Ok((false, artifact_store_roots, state))
}

#[tile]
fn finalize_sequence_unary_ref_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: PrefillPleSequenceUnaryState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if !state.is_complete() {
        bail!(
            "sequence unary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&state.output_source_name)?
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
    mut state: PrefillPleSequenceBinaryState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillPleSequenceBinaryState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
    }
    let end = state
        .next_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.row_count);
    while state.next_row_idx < end {
        let lhs_row = read_activation_row_from_ref(&state.lhs_ref, state.next_row_idx)?;
        let rhs_row = read_activation_row_from_ref(&state.rhs_ref, state.next_row_idx)?;
        let output_row = RasterActivationRow::from_acts(
            lhs_row
                .acts()
                .into_iter()
                .zip(rhs_row.acts())
                .map(|(lhs_value, rhs_value)| add_sat(lhs_value, rhs_value))
                .collect(),
        );
        let output_builder_root = artifact_store_roots
            .builder_root_for_source_name(&state.output_source_name)?
            .to_string();
        let (next_roots, _next_builder_root) = append_sequence_row_by_builder_root_with_roots(
            &artifact_store_roots,
            &output_builder_root,
            state.next_row_idx,
            output_row,
        )?;
        artifact_store_roots = next_roots;
        state.next_row_idx += 1;
    }
    Ok((false, artifact_store_roots, state))
}

#[tile]
fn finalize_sequence_binary_ref_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: PrefillPleSequenceBinaryState,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    if !state.is_complete() {
        bail!(
            "sequence binary state completed {} rows, expected {}",
            state.next_row_idx,
            state.row_count
        );
    }
    let output_builder_root = artifact_store_roots
        .builder_root_for_source_name(&state.output_source_name)?
        .to_string();
    finalize_sequence_builder_by_root_with_roots(&artifact_store_roots, &output_builder_root)
}

// Supporting definitions used by the sequences and tiles.

#[cfg(test)]
use super::utils::{insert_activation_sequence, store_prefill_token_ids_artifact};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleRasterState {
    source_id: String,
    token_ids_source_name: String,
    token_count: usize,
    input_activations_ref: Option<RasterActivationSequenceArtifactRef>,
    next_layer_idx: usize,
    layer_count: usize,
    per_layer_inputs: Vec<Option<RasterActivationSequenceArtifactRef>>,
    has_ple_global: bool,
    raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillPleInputRoots {
    source_id: String,
    token_ids_source_name: String,
    token_count: usize,
    input_activations_ref: Option<RasterActivationSequenceArtifactRef>,
    layer_count: usize,
    has_ple_global: bool,
    raster_sizing: RasterSizingControls,
}

pub type RasterPrefillPleOutput = RasterRoutineOutput<Option<String>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleLayerContext {
    layer_idx: usize,
    has_ple: bool,
    ple_width: Option<usize>,
    projection_rows: Option<usize>,
    embedding_scale_bits: Option<i32>,
    projection_scalar_bits: Option<i32>,
    input_scale_bits: Option<i32>,
    rms_norm_eps_bits: Option<i64>,
    norm_weight_bits: Option<Vec<i32>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleTokenEmbeddingState {
    token_ids_source_name: String,
    layer_idx: usize,
    scale_bits: i32,
    next_token_idx: usize,
    token_count: usize,
    row_width: usize,
    output_source_name: String,
}

impl PrefillPleTokenEmbeddingState {
    fn is_complete(&self) -> bool {
        self.next_token_idx >= self.token_count
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleSequenceProjectionState {
    input_ref: RasterActivationSequenceArtifactRef,
    output_source_name: String,
    current_row_bits: Vec<i32>,
    next_token_idx: usize,
    next_projection_row_idx: usize,
    token_count: usize,
    input_width: usize,
    projection_rows: usize,
    rows_per_tile: usize,
}

impl PrefillPleSequenceProjectionState {
    fn is_complete(&self) -> bool {
        self.next_token_idx >= self.token_count
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum PrefillPleSequenceUnaryOp {
    RmsNorm {
        norm_weight_bits: Vec<i32>,
        eps_bits: i64,
    },
    Scale {
        scalar_bits: i32,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleSequenceUnaryState {
    input_ref: RasterActivationSequenceArtifactRef,
    output_source_name: String,
    op: PrefillPleSequenceUnaryOp,
    next_row_idx: usize,
    row_count: usize,
    width: usize,
    rows_per_tile: usize,
}

impl PrefillPleSequenceUnaryState {
    fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleSequenceBinaryState {
    lhs_ref: RasterActivationSequenceArtifactRef,
    rhs_ref: RasterActivationSequenceArtifactRef,
    output_source_name: String,
    next_row_idx: usize,
    row_count: usize,
    width: usize,
    rows_per_tile: usize,
}

impl PrefillPleSequenceBinaryState {
    fn is_complete(&self) -> bool {
        self.next_row_idx >= self.row_count
    }
}

fn artifact_source_name_for_root(
    artifact_store_roots: &RasterArtifactStoreRoots,
    root: &str,
) -> Result<String> {
    Ok(artifact_store_roots
        .artifact_entry_for_root(root)?
        .id()
        .source_name()
        .to_string())
}

fn token_ids_root<'a>(
    artifact_store_roots: &'a RasterArtifactStoreRoots,
    source_name: &str,
) -> Result<&'a str> {
    artifact_store_roots.artifact_root_for_source_name(source_name)
}

pub fn prepare_raster_prefill_ple_input_roots(
    token_ids: &[u32],
    input_activations: &ActivationSequence,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, RasterPrefillPleInputRoots)> {
    reset_artifact_store();
    let artifact_store_roots = ArtifactIo::export_store_roots();
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    validate_sequence_rows_per_tile(raster_sizing.sequence_rows_per_tile)?;
    let (artifact_store_roots, token_ids_ref) =
        store_prefill_token_ids_artifact_with_roots(&artifact_store_roots, token_ids)?;
    let token_count = token_ids_ref.token_count();
    let token_ids_source_name = token_ids_ref.id().source_name().to_string();
    let metadata = auth_read!(ple_source, GemmaPleMetadataRequest)?;
    if !metadata.has_ple_global {
        return Ok((
            artifact_store_roots,
            RasterPrefillPleInputRoots {
                source_id: metadata.source_id,
                token_ids_source_name,
                token_count: token_ids_ref.token_count(),
                input_activations_ref: None,
                layer_count: metadata.layer_count,
                has_ple_global: false,
                raster_sizing,
            },
        ));
    }

    if metadata.layer_count == 0 {
        bail!("transformer PLE computation requires at least one layer");
    }
    if metadata.token_embedding_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            metadata.token_embedding_layer_count,
            metadata.layer_count
        );
    }
    if metadata.model_projection_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            metadata.model_projection_layer_count,
            metadata.layer_count
        );
    }

    let input_activations = raster_activation_sequence_from_embedding(input_activations)?;
    if input_activations.is_empty() {
        bail!("transformer PLE computation requires at least one activation row");
    }
    if token_count != input_activations.len() {
        bail!(
            "transformer PLE computation requires token ids and activations to have matching lengths"
        );
    }

    let first_layer = auth_read!(ple_source, GemmaPleLayerMetadataRequest { layer_idx: 0 })?;
    let activation_width = input_activations.width()?;
    if activation_width != first_layer.hidden_width {
        bail!(
            "input activations row 0 has width {}, expected {}",
            activation_width,
            first_layer.hidden_width
        );
    }
    let (artifact_store_roots, input_activations_ref) = insert_activation_sequence_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new("prefill.prepare_aux.input.initial")?,
        input_activations,
    )?;

    Ok((
        artifact_store_roots,
        RasterPrefillPleInputRoots {
            source_id: metadata.source_id,
            token_ids_source_name,
            token_count: token_ids_ref.token_count(),
            input_activations_ref: Some(input_activations_ref),
            layer_count: metadata.layer_count,
            has_ple_global: true,
            raster_sizing,
        },
    ))
}

pub fn prepare_raster_prefill_ple_input_roots_from_embedding_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, RasterPrefillPleInputRoots)> {
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    validate_sequence_rows_per_tile(raster_sizing.sequence_rows_per_tile)?;
    let token_ids_source_name = artifact_source_name_for_root(
        &artifact_store_roots,
        &input_embedding_refs.prompt_token_ids_root,
    )?;
    let metadata = auth_read!(ple_source, GemmaPleMetadataRequest)?;
    if !metadata.has_ple_global {
        return Ok((
            artifact_store_roots,
            RasterPrefillPleInputRoots {
                source_id: metadata.source_id,
                token_ids_source_name,
                token_count: input_embedding_refs.prompt_token_count,
                input_activations_ref: None,
                layer_count: metadata.layer_count,
                has_ple_global: false,
                raster_sizing,
            },
        ));
    }

    if metadata.layer_count == 0 {
        bail!("transformer PLE computation requires at least one layer");
    }
    if metadata.token_embedding_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            metadata.token_embedding_layer_count,
            metadata.layer_count
        );
    }
    if metadata.model_projection_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            metadata.model_projection_layer_count,
            metadata.layer_count
        );
    }
    if input_embedding_refs.prompt_token_count
        != input_embedding_refs
            .embedded_prompt_activations_ref
            .row_count()
    {
        bail!("transformer PLE computation requires token ids and activations to have matching lengths");
    }

    let first_layer = auth_read!(ple_source, GemmaPleLayerMetadataRequest { layer_idx: 0 })?;
    if input_embedding_refs.embedded_prompt_activations_ref.width() != first_layer.hidden_width {
        bail!(
            "input activations row 0 has width {}, expected {}",
            input_embedding_refs.embedded_prompt_activations_ref.width(),
            first_layer.hidden_width
        );
    }
    artifact_store_roots
        .artifact_entry_for_root(input_embedding_refs.embedded_prompt_activations_ref.root())?;

    Ok((
        artifact_store_roots,
        RasterPrefillPleInputRoots {
            source_id: metadata.source_id,
            token_ids_source_name,
            token_count: input_embedding_refs.prompt_token_count,
            input_activations_ref: Some(
                input_embedding_refs.embedded_prompt_activations_ref.clone(),
            ),
            layer_count: metadata.layer_count,
            has_ple_global: true,
            raster_sizing,
        },
    ))
}

pub fn run(
    token_ids: &[u32],
    input_activations: &ActivationSequence,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    let (artifact_store_roots, input_roots) = prepare_raster_prefill_ple_input_roots(
        token_ids,
        input_activations,
        ple_source,
        raster_sizing,
    )?;
    main(artifact_store_roots, input_roots, ple_source)
}

pub fn run_with_input_embedding_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<RasterPrefillPleOutput> {
    let (artifact_store_roots, input_roots) =
        prepare_raster_prefill_ple_input_roots_from_embedding_refs(
            artifact_store_roots,
            input_embedding_refs,
            ple_source,
            raster_sizing,
        )?;
    let (artifact_store_roots, refs) = main(artifact_store_roots, input_roots, ple_source)?;
    Ok(RasterPrefillPleOutput::new(artifact_store_roots, refs))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
