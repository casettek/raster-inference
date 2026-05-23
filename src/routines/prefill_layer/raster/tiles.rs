use anyhow::{anyhow, bail, Result};

use crate::dsl::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::shared::artifacts::raster_artifact_store::{
    RasterActivationSequenceArtifactRef, RasterArtifactStoreRoots,
};
use crate::shared::raster_contracts::prefill_layer::{
    AuthenticatedGemmaPrefillLayerSource, GemmaPrefillAttentionKind, GemmaPrefillLayerMatrixKind,
    GemmaPrefillLayerMetadata, GemmaPrefillLayerMetadataRequest, GemmaPrefillLayerNormKind,
    GemmaPrefillLayerNormWeightsRequest, GemmaPrefillLayerScalars, GemmaPrefillLayerScalarsRequest,
    GemmaPrefillLayerSourceMetadataRequest,
};
use crate::shared::raster_contracts::prefill_ple::read_prefill_ple_input_manifest_from_roots;
use crate::shared::raster_kernels::transformer::{
    append_projection_chunk_to_artifact_state, compute_next_attention_artifact_row,
    compute_next_combine_heads_artifact_row, compute_next_head_unary_artifact_row,
    compute_next_kv_cache_artifact_row, compute_next_reshape_heads_artifact_row,
    compute_next_sequence_binary_artifact_row, compute_next_sequence_unary_artifact_row,
    finalize_attention_artifact_row_state_ref, finalize_combine_heads_artifact_state_ref,
    finalize_head_unary_artifact_state_ref, finalize_kv_cache_build_artifact_state_ref,
    finalize_reshape_heads_artifact_state_ref, finalize_sequence_binary_artifact_state_ref,
    finalize_sequence_projection_artifact_state_ref, finalize_sequence_unary_artifact_state_ref,
    init_attention_artifact_row_state_from_refs, init_combine_heads_artifact_state_from_ref,
    init_head_rms_norm_artifact_state_from_ref, init_kv_cache_build_artifact_state_from_refs,
    init_reshape_heads_artifact_state_from_ref, init_rope_artifact_state_from_ref,
    init_sequence_add_artifact_state_from_refs, init_sequence_gelu_artifact_state_from_ref,
    init_sequence_mul_artifact_state_from_refs, init_sequence_projection_artifact_state_from_ref,
    init_sequence_rms_norm_artifact_state_from_ref, init_sequence_scale_artifact_state_from_ref,
    init_value_rms_norm_artifact_state_from_ref, validate_projection_rows_per_tile,
    RasterAttentionArtifactRowState, RasterCombineHeadsArtifactState, RasterHeadUnaryArtifactState,
    RasterKvCacheBuildArtifactState, RasterReshapeHeadsArtifactState,
    RasterSequenceBinaryArtifactState, RasterSequenceProjectionArtifactState,
    RasterSequenceUnaryArtifactState,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, read_sequence_row_from_roots,
    RasterActivationSequenceRef, RasterAttentionHeadsRef, RasterKvCacheRef,
    RasterSequenceRowRequest, RasterTensorId,
};
use crate::trace::{trace_event, trace_scope};
use crate::RasterSizingControls;

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerOutputRefs)> {
    ensure_artifact_root_present(
        &artifact_store_roots,
        input_embedding_refs.embedded_prompt_activations_ref.root(),
    )?;
    let (artifact_store_roots, layer_state) = call_tile!(
        init_prefill_layer_state_from_input_embedding_refs_with_roots,
        artifact_store_roots,
        input_embedding_refs,
        layer_source,
        ple_input_manifest_root,
        raster_sizing
    )?;
    let (artifact_store_roots, layer_state) = call_recur_seq!(
        compute_next_prefill_layer_sequence_with_roots,
        (artifact_store_roots, layer_state),
        layer_source
    )?;
    let refs = call_tile!(finalize_prefill_layer_refs, layer_state)?;
    Ok((artifact_store_roots, refs))
}

#[sequence(kind = recursive)]
pub fn compute_next_prefill_layer_sequence_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_state: PrefillLayerRasterState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    if layer_state.next_layer_idx >= layer_state.layer_count {
        return Ok((true, artifact_store_roots, layer_state));
    }

    let context = call_tile!(
        prepare_next_prefill_layer_context,
        &layer_state,
        layer_source
    )?;
    if let Some(per_layer_input) = context.per_layer_input.as_ref() {
        read_sequence_row_from_roots(
            &artifact_store_roots,
            RasterSequenceRowRequest {
                tensor_ref: per_layer_input.clone(),
                row_idx: 0,
            },
        )?;
    }
    let (token_count, _) = layer_state
        .current_activations_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    let _trace = trace_scope(format!(
        "prefill.layer.raster layer={} tokens={} attention={:?} ple={} donor={:?}",
        context.layer_idx,
        token_count,
        context.layer.attention_kind,
        context.layer.has_ple,
        context.layer.kv_shared_layer_index
    ));
    trace_event(format!(
        "progress prefill.layer layer={}/{} tokens={} attention={:?} ple={} donor={:?}",
        context.layer_idx + 1,
        layer_state.layer_count,
        token_count,
        context.layer.attention_kind,
        context.layer.has_ple,
        context.layer.kv_shared_layer_index
    ));
    let (artifact_store_roots, layer_output_ref, layer_cache) = call_seq!(
        run_prefill_layer_sequence_artifact_ref,
        artifact_store_roots,
        layer_state.current_activations_ref.clone(),
        layer_source,
        &context.layer,
        context.donor_cache.as_ref(),
        context.per_layer_input.clone(),
        layer_state.projection_rows_per_tile,
        layer_state.attention_kv_rows_per_tile,
        layer_state.sequence_rows_per_tile,
        layer_state.head_rows_per_tile,
    )?;

    call_tile!(
        update_prefill_layer_state_refs_with_roots,
        artifact_store_roots,
        layer_state,
        context.layer_idx,
        layer_output_ref,
        layer_cache
    )
}

#[sequence]
pub fn run_prefill_layer_sequence_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &GemmaPrefillLayerMetadata,
    donor_cache: Option<&PrefillLayerCacheSlot>,
    per_layer_input_ref: Option<RasterActivationSequenceRef>,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
    head_rows_per_tile: usize,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceRef,
    PrefillLayerCacheSlot,
)> {
    let scalars = call_tile!(read_prefill_layer_scalars, layer_source, layer.layer_idx)?;
    let input_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::InputLayer
    )?;
    let (artifact_store_roots, normed_ref) = call_seq!(
        compute_sequence_rms_norm_artifact_ref,
        artifact_store_roots,
        input_ref.clone(),
        format!("prefill.layer.{}.attention.input_norm", layer.layer_idx),
        Some(&input_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, q_projected_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        normed_ref.clone(),
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Query,
        layer.q_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.q_proj", layer.layer_idx)
    )?;
    let (artifact_store_roots, k_projected_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        normed_ref.clone(),
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Key,
        layer.k_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.k_proj", layer.layer_idx)
    )?;
    let (artifact_store_roots, v_projected_ref) = if layer.has_v_proj {
        call_seq!(
            project_sequence_with_prefill_source_artifact_ref,
            artifact_store_roots,
            normed_ref,
            layer_source,
            layer.layer_idx,
            GemmaPrefillLayerMatrixKind::Value,
            layer
                .v_proj_shape
                .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing v_proj shape"))?
                .rows,
            projection_rows_per_tile,
            format!("prefill.layer.{}.v_proj", layer.layer_idx)
        )?
    } else if layer.attention_k_eq_v {
        (artifact_store_roots, k_projected_ref.clone())
    } else {
        bail!("Gemma prefill layer is missing v_proj without attention_k_eq_v enabled");
    };

    let (artifact_store_roots, q_heads_ref) = call_seq!(
        reshape_heads_artifact_ref,
        artifact_store_roots,
        q_projected_ref,
        format!("prefill.layer.{}.q_heads", layer.layer_idx),
        layer.num_heads,
        layer.head_dim
    )?;
    let (artifact_store_roots, k_heads_ref) = call_seq!(
        reshape_heads_artifact_ref,
        artifact_store_roots,
        k_projected_ref,
        format!("prefill.layer.{}.k_heads", layer.layer_idx),
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let q_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::Query
    )?;
    let (artifact_store_roots, q_heads_ref) = call_seq!(
        compute_head_rms_norm_artifact_ref,
        artifact_store_roots,
        q_heads_ref,
        format!("prefill.layer.{}.q_norm", layer.layer_idx),
        Some(&q_norm_weights),
        Some(scalars.rms_norm_eps),
        head_rows_per_tile
    )?;
    let k_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::Key
    )?;
    let (artifact_store_roots, k_heads_ref) = call_seq!(
        compute_head_rms_norm_artifact_ref,
        artifact_store_roots,
        k_heads_ref,
        format!("prefill.layer.{}.k_norm", layer.layer_idx),
        Some(&k_norm_weights),
        Some(scalars.rms_norm_eps),
        head_rows_per_tile
    )?;
    let (artifact_store_roots, v_heads_ref) = call_seq!(
        reshape_heads_artifact_ref,
        artifact_store_roots,
        v_projected_ref,
        format!("prefill.layer.{}.v_heads", layer.layer_idx),
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let (artifact_store_roots, v_heads_ref) = call_seq!(
        compute_value_rms_norm_artifact_ref,
        artifact_store_roots,
        v_heads_ref,
        format!("prefill.layer.{}.v_norm", layer.layer_idx),
        Some(scalars.rms_norm_eps),
        head_rows_per_tile
    )?;
    let (artifact_store_roots, q_heads_ref) = call_seq!(
        compute_rope_artifact_ref,
        artifact_store_roots,
        q_heads_ref,
        format!("prefill.layer.{}.q_rope", layer.layer_idx),
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        0,
        head_rows_per_tile
    )?;
    let (artifact_store_roots, k_heads_ref) = call_seq!(
        compute_rope_artifact_ref,
        artifact_store_roots,
        k_heads_ref,
        format!("prefill.layer.{}.k_rope", layer.layer_idx),
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        0,
        head_rows_per_tile
    )?;

    let retained_cache_len =
        retained_prefill_kv_cache_len(&k_heads_ref, layer.cache_sliding_window)?;
    let (artifact_store_roots, layer_cache) = if donor_cache.is_some() || retained_cache_len == 0 {
        (
            artifact_store_roots,
            PrefillLayerCacheSlot::Empty {
                num_kv_heads: layer.num_kv_heads,
            },
        )
    } else {
        let (artifact_store_roots, cache_ref) = call_seq!(
            build_kv_cache_artifact_ref,
            artifact_store_roots,
            k_heads_ref.clone(),
            v_heads_ref.clone(),
            format!("prefill.layer.cache.{}", layer.layer_idx),
            layer.cache_sliding_window
        )?;
        (artifact_store_roots, PrefillLayerCacheSlot::Ref(cache_ref))
    };
    let attention_window = call_tile!(resolve_prefill_attention_window, layer)?;
    let donor_cache_ref = donor_cache
        .map(|cache| match cache {
            PrefillLayerCacheSlot::Empty { .. } => {
                bail!("transformer prefill donor cache is empty")
            }
            PrefillLayerCacheSlot::Ref(cache_ref) => Ok(cache_ref.clone()),
        })
        .transpose()?;
    let (artifact_store_roots, attention_heads_ref) = call_seq!(
        compute_attention_artifact_ref,
        artifact_store_roots,
        q_heads_ref,
        k_heads_ref,
        v_heads_ref,
        donor_cache_ref,
        format!("prefill.layer.{}.attention", layer.layer_idx),
        attention_window,
        attention_kv_rows_per_tile
    )?;
    let (artifact_store_roots, attention_sequence_ref) = call_seq!(
        combine_heads_artifact_ref,
        artifact_store_roots,
        attention_heads_ref,
        format!("prefill.layer.{}.attention.combine", layer.layer_idx)
    )?;
    let (artifact_store_roots, attention_output_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        attention_sequence_ref,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Output,
        layer.o_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.o_proj", layer.layer_idx)
    )?;
    let post_attention_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PostAttention
    )?;
    let (artifact_store_roots, attention_output_ref) = call_seq!(
        compute_sequence_rms_norm_artifact_ref,
        artifact_store_roots,
        attention_output_ref,
        format!("prefill.layer.{}.attention.post_norm", layer.layer_idx),
        Some(&post_attention_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, xs_ref) = call_seq!(
        compute_sequence_add_artifact_ref,
        artifact_store_roots,
        input_ref,
        attention_output_ref,
        format!("prefill.layer.{}.attention.residual", layer.layer_idx),
        sequence_rows_per_tile
    )?;

    let (artifact_store_roots, xs_ref) = call_seq!(
        run_prefill_mlp_block_artifact_ref,
        artifact_store_roots,
        xs_ref,
        layer_source,
        layer,
        &scalars,
        projection_rows_per_tile,
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, mut xs_ref) = if let Some(per_layer_input_ref) = per_layer_input_ref
    {
        call_seq!(
            run_prefill_ple_block_artifact_ref,
            artifact_store_roots,
            xs_ref,
            per_layer_input_ref,
            layer_source,
            layer,
            &scalars,
            projection_rows_per_tile,
            sequence_rows_per_tile
        )?
    } else {
        (artifact_store_roots, xs_ref)
    };
    if scalars.layer_scalar.is_some() {
        let scaled = call_seq!(
            compute_sequence_scale_artifact_ref,
            artifact_store_roots,
            xs_ref,
            format!("prefill.layer.{}.layer_scalar", layer.layer_idx),
            scalars.layer_scalar,
            sequence_rows_per_tile
        )?;
        xs_ref = scaled.1;
        return Ok((scaled.0, xs_ref, layer_cache));
    }

    Ok((artifact_store_roots, xs_ref, layer_cache))
}

#[sequence]
fn compute_sequence_rms_norm_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    norm_weights: Option<&[crate::shared::numerics::det_num::Wgt]>,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, sequence_unary_state) = call_tile!(
        init_prefill_sequence_rms_norm_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let (artifact_store_roots, sequence_unary_state) = call_recur_tile!(
        transform_next_prefill_sequence_unary_artifact_row,
        (artifact_store_roots, sequence_unary_state)
    )?;
    call_tile!(
        finalize_prefill_sequence_unary_artifact_state_ref,
        artifact_store_roots,
        sequence_unary_state
    )
}

#[sequence]
fn project_sequence_with_prefill_source_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
    projection_rows: usize,
    rows_per_tile: usize,
    id_prefix: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, projection_state) = call_tile!(
        init_prefill_sequence_projection_artifact_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        projection_rows,
        rows_per_tile
    )?;
    let (artifact_store_roots, projection_state) = call_recur_tile!(
        project_next_prefill_sequence_artifact_rows,
        (artifact_store_roots, projection_state),
        layer_source,
        layer_idx,
        matrix
    )?;
    call_tile!(
        finalize_prefill_sequence_projection_artifact_ref,
        artifact_store_roots,
        projection_state
    )
}

#[sequence]
fn reshape_heads_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    num_heads: usize,
    head_dim: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let (artifact_store_roots, reshape_state) = call_tile!(
        init_prefill_reshape_heads_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        num_heads,
        head_dim
    )?;
    let (artifact_store_roots, reshape_state) = call_recur_tile!(
        transform_next_prefill_reshape_artifact_row,
        (artifact_store_roots, reshape_state)
    )?;
    call_tile!(
        finalize_prefill_reshape_heads_artifact_state_ref,
        artifact_store_roots,
        reshape_state
    )
}

#[sequence]
fn compute_head_rms_norm_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    norm_weights: Option<&[crate::shared::numerics::det_num::Wgt]>,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let (artifact_store_roots, head_state) = call_tile!(
        init_prefill_head_rms_norm_artifact_state_from_ref,
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let (artifact_store_roots, head_state) = call_recur_tile!(
        transform_next_prefill_head_artifact_row,
        (artifact_store_roots, head_state)
    )?;
    call_tile!(
        finalize_prefill_head_artifact_state_ref,
        artifact_store_roots,
        head_state
    )
}

#[sequence]
fn compute_value_rms_norm_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let (artifact_store_roots, head_state) = call_tile!(
        init_prefill_value_rms_norm_artifact_state_from_ref,
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        eps,
        rows_per_tile
    )?;
    let (artifact_store_roots, head_state) = call_recur_tile!(
        transform_next_prefill_head_artifact_row,
        (artifact_store_roots, head_state)
    )?;
    call_tile!(
        finalize_prefill_head_artifact_state_ref,
        artifact_store_roots,
        head_state
    )
}

#[sequence]
fn compute_rope_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<crate::shared::numerics::det_num::Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let (artifact_store_roots, head_state) = call_tile!(
        init_prefill_rope_artifact_state_from_ref,
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rotary_dim,
        freq_base_dim,
        base,
        position_offset,
        rows_per_tile
    )?;
    let (artifact_store_roots, head_state) = call_recur_tile!(
        transform_next_prefill_head_artifact_row,
        (artifact_store_roots, head_state)
    )?;
    call_tile!(
        finalize_prefill_head_artifact_state_ref,
        artifact_store_roots,
        head_state
    )
}

#[sequence]
fn build_kv_cache_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    sliding_window: Option<usize>,
) -> Result<(RasterArtifactStoreRoots, RasterKvCacheRef)> {
    let (artifact_store_roots, kv_cache_state) = call_tile!(
        init_prefill_kv_cache_artifact_state_from_refs,
        artifact_store_roots,
        key_ref,
        value_ref,
        RasterTensorId::new(format!("{id_prefix}.keys"))?,
        RasterTensorId::new(format!("{id_prefix}.values"))?,
        sliding_window
    )?;
    let (artifact_store_roots, kv_cache_state) = call_recur_tile!(
        transform_next_prefill_kv_cache_artifact_row,
        (artifact_store_roots, kv_cache_state)
    )?;
    call_tile!(
        finalize_prefill_kv_cache_artifact_state_ref,
        artifact_store_roots,
        kv_cache_state
    )
}

#[sequence]
fn compute_attention_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    id_prefix: String,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let (artifact_store_roots, attention_state) = call_tile!(
        init_prefill_attention_artifact_state_from_refs,
        artifact_store_roots,
        query_ref,
        key_ref,
        value_ref,
        donor_cache_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        attention_window,
        kv_rows_per_tile
    )?;
    let (artifact_store_roots, attention_state) = call_recur_tile!(
        project_next_prefill_attention_artifact_row,
        (artifact_store_roots, attention_state)
    )?;
    call_tile!(
        finalize_prefill_attention_artifact_state_ref,
        artifact_store_roots,
        attention_state
    )
}

#[sequence]
fn combine_heads_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, combine_state) = call_tile!(
        init_prefill_combine_heads_artifact_state_from_ref,
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?
    )?;
    let (artifact_store_roots, combine_state) = call_recur_tile!(
        transform_next_prefill_combine_artifact_row,
        (artifact_store_roots, combine_state)
    )?;
    call_tile!(
        finalize_prefill_combine_heads_artifact_state_ref,
        artifact_store_roots,
        combine_state
    )
}

#[sequence]
fn compute_sequence_add_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    id_prefix: String,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, sequence_binary_state) = call_tile!(
        init_prefill_sequence_add_artifact_state_from_refs,
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let (artifact_store_roots, sequence_binary_state) = call_recur_tile!(
        transform_next_prefill_sequence_binary_artifact_row,
        (artifact_store_roots, sequence_binary_state)
    )?;
    call_tile!(
        finalize_prefill_sequence_binary_artifact_state_ref,
        artifact_store_roots,
        sequence_binary_state
    )
}

#[sequence]
fn run_prefill_mlp_block_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    xs_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &GemmaPrefillLayerMetadata,
    scalars: &GemmaPrefillLayerScalars,
    projection_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let pre_feedforward_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PreFeedForward
    )?;
    let (artifact_store_roots, normed_ref) = call_seq!(
        compute_sequence_rms_norm_artifact_ref,
        artifact_store_roots,
        xs_ref.clone(),
        format!("prefill.layer.{}.mlp.pre_norm", layer.layer_idx),
        Some(&pre_feedforward_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, gate_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        normed_ref.clone(),
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Gate,
        layer.gate_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.gate_proj", layer.layer_idx)
    )?;
    let (artifact_store_roots, gate_ref) = call_seq!(
        compute_sequence_gelu_artifact_ref,
        artifact_store_roots,
        gate_ref,
        format!("prefill.layer.{}.gate_gelu", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, up_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        normed_ref,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Up,
        layer.up_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.up_proj", layer.layer_idx)
    )?;
    let (artifact_store_roots, ff_hidden_ref) = call_seq!(
        compute_sequence_mul_artifact_ref,
        artifact_store_roots,
        gate_ref,
        up_ref,
        format!("prefill.layer.{}.ff_hidden", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, ff_out_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        ff_hidden_ref,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Down,
        layer.down_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.down_proj", layer.layer_idx)
    )?;
    let post_feedforward_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PostFeedForward
    )?;
    let (artifact_store_roots, ff_out_ref) = call_seq!(
        compute_sequence_rms_norm_artifact_ref,
        artifact_store_roots,
        ff_out_ref,
        format!("prefill.layer.{}.ff_post_norm", layer.layer_idx),
        Some(&post_feedforward_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    call_seq!(
        compute_sequence_add_artifact_ref,
        artifact_store_roots,
        xs_ref,
        ff_out_ref,
        format!("prefill.layer.{}.mlp.residual", layer.layer_idx),
        sequence_rows_per_tile
    )
}

#[sequence]
fn compute_sequence_gelu_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, sequence_unary_state) = call_tile!(
        init_prefill_sequence_gelu_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let (artifact_store_roots, sequence_unary_state) = call_recur_tile!(
        transform_next_prefill_sequence_unary_artifact_row,
        (artifact_store_roots, sequence_unary_state)
    )?;
    call_tile!(
        finalize_prefill_sequence_unary_artifact_state_ref,
        artifact_store_roots,
        sequence_unary_state
    )
}

#[sequence]
fn compute_sequence_mul_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    id_prefix: String,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, sequence_binary_state) = call_tile!(
        init_prefill_sequence_mul_artifact_state_from_refs,
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let (artifact_store_roots, sequence_binary_state) = call_recur_tile!(
        transform_next_prefill_sequence_binary_artifact_row,
        (artifact_store_roots, sequence_binary_state)
    )?;
    call_tile!(
        finalize_prefill_sequence_binary_artifact_state_ref,
        artifact_store_roots,
        sequence_binary_state
    )
}

#[sequence]
fn run_prefill_ple_block_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    xs_ref: RasterActivationSequenceRef,
    per_layer_input_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &GemmaPrefillLayerMetadata,
    scalars: &GemmaPrefillLayerScalars,
    projection_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, gated_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        xs_ref.clone(),
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::PleInputGate,
        layer
            .ple_input_gate_shape
            .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing PLE input gate shape"))?
            .rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.ple_gate", layer.layer_idx)
    )?;
    let (artifact_store_roots, gated_ref) = call_seq!(
        compute_sequence_gelu_artifact_ref,
        artifact_store_roots,
        gated_ref,
        format!("prefill.layer.{}.ple_gate_gelu", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, gated_ref) = call_seq!(
        compute_sequence_mul_artifact_ref,
        artifact_store_roots,
        gated_ref,
        per_layer_input_ref,
        format!("prefill.layer.{}.ple_input_mul", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let (artifact_store_roots, projected_ref) = call_seq!(
        project_sequence_with_prefill_source_artifact_ref,
        artifact_store_roots,
        gated_ref,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::PleLayerProjection,
        layer
            .ple_layer_projection_shape
            .ok_or_else(|| {
                anyhow!("Gemma prefill layer metadata is missing PLE layer projection shape")
            })?
            .rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.ple_layer_projection", layer.layer_idx)
    )?;
    let ple_post_input_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PlePostInput
    )?;
    let (artifact_store_roots, projected_ref) = call_seq!(
        compute_sequence_rms_norm_artifact_ref,
        artifact_store_roots,
        projected_ref,
        format!("prefill.layer.{}.ple_post_norm", layer.layer_idx),
        Some(&ple_post_input_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    call_seq!(
        compute_sequence_add_artifact_ref,
        artifact_store_roots,
        xs_ref,
        projected_ref,
        format!("prefill.layer.{}.ple_residual", layer.layer_idx),
        sequence_rows_per_tile
    )
}

#[sequence]
fn compute_sequence_scale_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    scalar: Option<crate::shared::numerics::det_num::Act>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, sequence_unary_state) = call_tile!(
        init_prefill_sequence_scale_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        scalar,
        rows_per_tile
    )?;
    let (artifact_store_roots, sequence_unary_state) = call_recur_tile!(
        transform_next_prefill_sequence_unary_artifact_row,
        (artifact_store_roots, sequence_unary_state)
    )?;
    call_tile!(
        finalize_prefill_sequence_unary_artifact_state_ref,
        artifact_store_roots,
        sequence_unary_state
    )
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
pub fn init_prefill_layer_state_from_input_embedding_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    if input_embedding_refs.prompt_token_count
        != input_embedding_refs
            .embedded_prompt_activations_ref
            .row_count()
    {
        bail!("prefill layer requires input embedding token count and activation rows to match");
    }
    init_prefill_layer_state_from_activation_ref_with_roots(
        artifact_store_roots,
        input_embedding_refs.embedded_prompt_activations_ref.clone(),
        layer_source,
        ple_input_manifest_root,
        raster_sizing,
    )
}

#[tile]
pub fn finalize_prefill_layer_refs(
    layer_state: PrefillLayerRasterState,
) -> Result<PrefillLayerOutputRefs> {
    if layer_state.next_layer_idx != layer_state.layer_count {
        bail!(
            "raster prefill layer finalized after {} layers, expected {}",
            layer_state.next_layer_idx,
            layer_state.layer_count
        );
    }

    Ok(PrefillLayerOutputRefs {
        final_hidden_states_ref: layer_state.current_activations_ref,
        layer_caches: layer_state.layer_caches,
    })
}

#[tile]
pub fn prepare_next_prefill_layer_context(
    layer_state: &PrefillLayerRasterState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
) -> Result<PrefillLayerContext> {
    if layer_state.next_layer_idx >= layer_state.layer_count {
        bail!(
            "cannot prepare prefill layer {} after completing {} layers",
            layer_state.next_layer_idx,
            layer_state.layer_count
        );
    }

    let layer_idx = layer_state.next_layer_idx;
    let layer = auth_read!(layer_source, GemmaPrefillLayerMetadataRequest { layer_idx })?;
    let donor_cache =
        resolve_prefill_donor_cache_index(&layer_state.layer_caches, layer_idx, &layer)?
            .map(|donor_idx| {
                layer_state.layer_caches.get(donor_idx).cloned().ok_or_else(|| {
                anyhow!("transformer prefill donor cache {donor_idx} missing for layer {layer_idx}")
            })
            })
            .transpose()?;
    let per_layer_input = layer_state
        .per_layer_inputs
        .get(layer_idx)
        .and_then(Option::as_ref)
        .cloned();
    validate_prefill_layer_ple_input_ref(
        &layer_state.current_activations_ref,
        &layer,
        per_layer_input.as_ref(),
    )?;

    Ok(PrefillLayerContext {
        layer_idx,
        layer,
        donor_cache,
        per_layer_input,
    })
}

#[tile]
pub fn update_prefill_layer_state_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut layer_state: PrefillLayerRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    layer_cache: PrefillLayerCacheSlot,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    if layer_idx != layer_state.next_layer_idx {
        bail!(
            "cannot update prefill layer {layer_idx} while next layer is {}",
            layer_state.next_layer_idx
        );
    }

    layer_state.current_activations_ref = layer_output_ref;
    layer_state.layer_caches.push(layer_cache);

    let completed_layer_output =
        trace_prefill_layer_checkpoint_with_roots(&artifact_store_roots, &layer_state, layer_idx)?;
    if let Some((sha256, det_sha256)) = completed_layer_output {
        layer_state.completed_layer_output_sha256s.push(sha256);
        layer_state
            .completed_layer_output_det_sha256s
            .push(det_sha256);
    }

    if trace_prefill_layer_token_checkpoints_with_roots(
        &artifact_store_roots,
        &layer_state,
        layer_idx,
    )? {
        layer_state.next_layer_idx += 1;
        layer_state.layer_count = layer_state.next_layer_idx;
        return Ok((true, artifact_store_roots, layer_state));
    }
    layer_state.next_layer_idx += 1;
    Ok((false, artifact_store_roots, layer_state))
}

#[tile]
pub fn read_prefill_layer_scalars(
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
) -> Result<GemmaPrefillLayerScalars> {
    auth_read!(layer_source, GemmaPrefillLayerScalarsRequest { layer_idx })
}

#[tile]
pub fn read_prefill_layer_norm_weights(
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
    norm: GemmaPrefillLayerNormKind,
) -> Result<Vec<crate::shared::numerics::det_num::Wgt>> {
    auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest { layer_idx, norm }
    )
}

#[tile]
pub fn resolve_prefill_attention_window(
    layer: &GemmaPrefillLayerMetadata,
) -> Result<Option<usize>> {
    match layer.attention_kind {
        GemmaPrefillAttentionKind::Full => Ok(None),
        GemmaPrefillAttentionKind::Sliding => {
            Ok(Some(layer.sliding_window.ok_or_else(|| {
                anyhow!("sliding attention layer is missing a sliding window")
            })?))
        }
    }
}

#[tile]
pub fn init_prefill_sequence_rms_norm_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[crate::shared::numerics::det_num::Wgt]>,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceUnaryArtifactState)> {
    init_sequence_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        output_id,
        norm_weights,
        eps,
        rows_per_tile,
    )
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_sequence_unary_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    sequence_unary_state: RasterSequenceUnaryArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceUnaryArtifactState,
)> {
    compute_next_sequence_unary_artifact_row(artifact_store_roots, sequence_unary_state)
}

#[tile]
pub fn finalize_prefill_sequence_unary_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    sequence_unary_state: RasterSequenceUnaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_sequence_unary_artifact_state_ref(artifact_store_roots, sequence_unary_state)
}

#[tile]
pub fn init_prefill_sequence_projection_artifact_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterSequenceProjectionArtifactState,
)> {
    init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        output_id,
        projection_rows,
        projection_rows_per_tile,
    )
}

#[tile(kind = recursive)]
pub fn project_next_prefill_sequence_artifact_rows(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_state: RasterSequenceProjectionArtifactState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceProjectionArtifactState,
)> {
    if projection_state.is_complete() {
        return Ok((true, artifact_store_roots, projection_state));
    }

    let end = projection_state
        .next_projection_row_idx()
        .saturating_add(projection_state.rows_per_tile())
        .min(projection_state.projection_rows());
    let start_projection_row_idx = projection_state.next_projection_row_idx();
    let start_token_idx = projection_state.next_token_idx();
    let mut rows = Vec::with_capacity(end - projection_state.next_projection_row_idx());
    for row_idx in projection_state.next_projection_row_idx()..end {
        rows.push(auth_read!(
            layer_source,
            crate::shared::raster_contracts::prefill_layer::GemmaPrefillLayerMatrixRowRequest {
                layer_idx,
                matrix,
                row_idx,
            },
        )?);
    }
    let (artifact_store_roots, projection_state) =
        append_projection_chunk_to_artifact_state(artifact_store_roots, projection_state, &rows)?;
    trace_event(format!(
        "progress prefill.layer.projection layer={} matrix={:?} token={}/{} projection_rows={}..{} of {} input_width={}",
        layer_idx,
        matrix,
        start_token_idx + 1,
        projection_state.token_count(),
        start_projection_row_idx,
        end,
        projection_state.projection_rows(),
        projection_state.input_width()
    ));
    Ok((false, artifact_store_roots, projection_state))
}

#[tile]
pub fn finalize_prefill_sequence_projection_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_state: RasterSequenceProjectionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_sequence_projection_artifact_state_ref(artifact_store_roots, projection_state)
}

#[tile]
pub fn init_prefill_reshape_heads_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    num_heads: usize,
    head_dim: usize,
) -> Result<(RasterArtifactStoreRoots, RasterReshapeHeadsArtifactState)> {
    init_reshape_heads_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        output_id,
        num_heads,
        head_dim,
    )
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_reshape_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    reshape_state: RasterReshapeHeadsArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterReshapeHeadsArtifactState,
)> {
    compute_next_reshape_heads_artifact_row(artifact_store_roots, reshape_state)
}

#[tile]
pub fn finalize_prefill_reshape_heads_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    reshape_state: RasterReshapeHeadsArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    finalize_reshape_heads_artifact_state_ref(artifact_store_roots, reshape_state)
}

#[tile]
pub fn init_prefill_head_rms_norm_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[crate::shared::numerics::det_num::Wgt]>,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    init_head_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        output_id,
        norm_weights,
        eps,
        rows_per_tile,
    )
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_head_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_state: RasterHeadUnaryArtifactState,
) -> Result<(bool, RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    compute_next_head_unary_artifact_row(artifact_store_roots, head_state)
}

#[tile]
pub fn finalize_prefill_head_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_state: RasterHeadUnaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    finalize_head_unary_artifact_state_ref(artifact_store_roots, head_state)
}

#[tile]
pub fn init_prefill_value_rms_norm_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    eps: Option<crate::shared::numerics::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    init_value_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        output_id,
        eps,
        rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_rope_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<crate::shared::numerics::det_num::Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    init_rope_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        output_id,
        rotary_dim,
        freq_base_dim,
        base,
        position_offset,
        rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_kv_cache_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    sliding_window: Option<usize>,
) -> Result<(RasterArtifactStoreRoots, RasterKvCacheBuildArtifactState)> {
    init_kv_cache_build_artifact_state_from_refs(
        artifact_store_roots,
        key_ref,
        value_ref,
        keys_id,
        values_id,
        sliding_window,
    )
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_kv_cache_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    kv_cache_state: RasterKvCacheBuildArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterKvCacheBuildArtifactState,
)> {
    compute_next_kv_cache_artifact_row(artifact_store_roots, kv_cache_state)
}

#[tile]
pub fn finalize_prefill_kv_cache_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    kv_cache_state: RasterKvCacheBuildArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterKvCacheRef)> {
    finalize_kv_cache_build_artifact_state_ref(artifact_store_roots, kv_cache_state)
}

#[tile]
pub fn init_prefill_attention_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    output_id: RasterTensorId,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionArtifactRowState)> {
    init_attention_artifact_row_state_from_refs(
        artifact_store_roots,
        query_ref,
        key_ref,
        value_ref,
        donor_cache_ref,
        output_id,
        attention_window,
        kv_rows_per_tile,
    )
}

#[tile(kind = recursive)]
pub fn project_next_prefill_attention_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    attention_state: RasterAttentionArtifactRowState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterAttentionArtifactRowState,
)> {
    compute_next_attention_artifact_row(artifact_store_roots, attention_state)
}

#[tile]
pub fn finalize_prefill_attention_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    attention_state: RasterAttentionArtifactRowState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    finalize_attention_artifact_row_state_ref(artifact_store_roots, attention_state)
}

#[tile]
pub fn init_prefill_combine_heads_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
) -> Result<(RasterArtifactStoreRoots, RasterCombineHeadsArtifactState)> {
    init_combine_heads_artifact_state_from_ref(artifact_store_roots, heads_ref, output_id)
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_combine_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    combine_state: RasterCombineHeadsArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterCombineHeadsArtifactState,
)> {
    compute_next_combine_heads_artifact_row(artifact_store_roots, combine_state)
}

#[tile]
pub fn finalize_prefill_combine_heads_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    combine_state: RasterCombineHeadsArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_combine_heads_artifact_state_ref(artifact_store_roots, combine_state)
}

#[tile]
pub fn init_prefill_sequence_add_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceBinaryArtifactState)> {
    init_sequence_add_artifact_state_from_refs(
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        output_id,
        rows_per_tile,
    )
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_sequence_binary_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    sequence_binary_state: RasterSequenceBinaryArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceBinaryArtifactState,
)> {
    compute_next_sequence_binary_artifact_row(artifact_store_roots, sequence_binary_state)
}

#[tile]
pub fn finalize_prefill_sequence_binary_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    sequence_binary_state: RasterSequenceBinaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_sequence_binary_artifact_state_ref(artifact_store_roots, sequence_binary_state)
}

#[tile]
pub fn init_prefill_sequence_gelu_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceUnaryArtifactState)> {
    init_sequence_gelu_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        output_id,
        rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_sequence_mul_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceBinaryArtifactState)> {
    init_sequence_mul_artifact_state_from_refs(
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        output_id,
        rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_sequence_scale_artifact_state_from_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    scalar: Option<crate::shared::numerics::det_num::Act>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterSequenceUnaryArtifactState)> {
    init_sequence_scale_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        output_id,
        scalar,
        rows_per_tile,
    )
}

#[tile]
pub(crate) fn init_prefill_layer_state_from_activation_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_activations_ref: RasterActivationSequenceArtifactRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    crate::shared::raster_kernels::transformer::validate_attention_kv_rows_per_tile(
        raster_sizing.attention_kv_rows_per_tile,
    )?;
    crate::shared::raster_kernels::transformer::validate_sequence_rows_per_tile(
        raster_sizing.sequence_rows_per_tile,
    )?;
    crate::shared::raster_kernels::transformer::validate_head_rows_per_tile(
        raster_sizing.head_rows_per_tile,
    )?;
    ensure_artifact_root_present(&artifact_store_roots, input_activations_ref.root())?;
    let metadata = auth_read!(layer_source, GemmaPrefillLayerSourceMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer prefill requires at least one layer");
    }

    if input_activations_ref.row_count() == 0 {
        bail!("transformer layer execution requires at least one activation row");
    }
    let first_layer = auth_read!(
        layer_source,
        GemmaPrefillLayerMetadataRequest { layer_idx: 0 }
    )?;
    if input_activations_ref.width() != first_layer.hidden_size {
        bail!(
            "transformer layer input width {}, expected {}",
            input_activations_ref.width(),
            first_layer.hidden_size
        );
    }
    let token_count = input_activations_ref.row_count();
    let current_activations_ref = activation_sequence_ref_from_artifact(
        RasterTensorId::new(format!(
            "prefill.layer.current.initial.{}",
            input_activations_ref.root()
        ))?,
        input_activations_ref,
    )?;

    let ple_input_refs = ple_input_manifest_root
        .map(|root| {
            read_prefill_ple_input_manifest_from_roots(&artifact_store_roots, root)?
                .into_prefill_ple_input_refs(artifact_store_roots.clone())
        })
        .transpose()?;
    let per_layer_inputs = match ple_input_refs.as_ref() {
        Some(ple_input_refs) => {
            if ple_input_refs.source_id() != metadata.source_id {
                bail!(
                    "raster PLE input refs source {} does not match prefill layer source {}",
                    ple_input_refs.source_id(),
                    metadata.source_id
                );
            }
            if ple_input_refs.layer_count() != metadata.layer_count {
                bail!(
                    "raster PLE input refs contain {} layers, expected {}",
                    ple_input_refs.layer_count(),
                    metadata.layer_count
                );
            }
            if ple_input_refs.token_count() != token_count {
                bail!(
                    "raster PLE input refs contain {} tokens, expected {token_count}",
                    ple_input_refs.token_count()
                );
            }
            ple_input_refs
                .per_layer_inputs()
                .iter()
                .enumerate()
                .map(|(layer_idx, input_ref)| {
                    input_ref
                        .as_ref()
                        .map(|input_ref| {
                            ensure_artifact_root_present(&artifact_store_roots, input_ref.root())?;
                            activation_sequence_ref_from_artifact(
                                RasterTensorId::new(format!(
                                    "prefill.layer.per_layer_input.{layer_idx}.{}",
                                    input_ref.root()
                                ))?,
                                input_ref.clone(),
                            )
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => vec![None; metadata.layer_count],
    };

    Ok((
        artifact_store_roots,
        PrefillLayerRasterState {
            current_activations_ref,
            next_layer_idx: 0,
            layer_count: metadata.layer_count,
            layer_caches: Vec::with_capacity(metadata.layer_count),
            per_layer_inputs,
            completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
            projection_rows_per_tile: raster_sizing.projection_rows_per_tile,
            attention_kv_rows_per_tile: raster_sizing.attention_kv_rows_per_tile,
            sequence_rows_per_tile: raster_sizing.sequence_rows_per_tile,
            head_rows_per_tile: raster_sizing.head_rows_per_tile,
        },
    ))
}
