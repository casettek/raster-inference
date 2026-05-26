use anyhow::{anyhow, bail, Result};

use crate::dsl::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::raster_contracts::prefill_layer::{
    GemmaPrefillAttentionKind, GemmaPrefillLayerMatrixKind, GemmaPrefillLayerMetadata,
    GemmaPrefillLayerNormKind, GemmaPrefillLayerNormWeightsRequest, GemmaPrefillLayerScalars,
    GemmaPrefillLayerScalarsRequest, RasterPrefillLayerSource,
};
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
    init_value_rms_norm_artifact_state_from_ref, RasterAttentionArtifactRowState,
    RasterCombineHeadsArtifactState, RasterHeadUnaryArtifactState, RasterKvCacheBuildArtifactState,
    RasterReshapeHeadsArtifactState, RasterSequenceBinaryArtifactState,
    RasterSequenceProjectionArtifactState, RasterSequenceUnaryArtifactState,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterAttentionHeadsRef,
    RasterKvCacheRef, RasterSequenceRowRequest, RasterTensorId,
};
use crate::trace::trace_event;
use crate::RasterSizingControls;

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    layer_source: &RasterPrefillLayerSource<'_>,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerOutputRefs)> {
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
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    let (artifact_store_roots, layer_step) = call_tile!(
        init_prefill_layer_step,
        artifact_store_roots,
        layer_state,
        layer_source
    )?;
    let (artifact_store_roots, layer_step) = call_seq!(
        run_prefill_layer_sequence_artifact_ref,
        artifact_store_roots,
        layer_step,
        layer_source
    )?;
    call_tile!(
        finalize_prefill_layer_step,
        artifact_store_roots,
        layer_step
    )
}

#[sequence]
fn run_prefill_layer_sequence_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillLayerStep,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerStep)> {
    let (artifact_store_roots, layer_work) = call_seq!(
        run_prefill_layer_artifact_ref_body,
        artifact_store_roots,
        layer_step,
        layer_source
    )?;
    call_tile!(
        finalize_prefill_layer_artifact_work,
        artifact_store_roots,
        layer_work
    )
}

#[sequence]
fn run_prefill_layer_artifact_ref_body(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillLayerStep,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, layer_work) = call_tile!(
        init_prefill_layer_artifact_work,
        artifact_store_roots,
        layer_step,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        compute_prefill_layer_attention_input_norm_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_query_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_key_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_value_work,
        artifact_store_roots,
        layer_work,
        layer_source,
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        reshape_prefill_layer_query_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        reshape_prefill_layer_key_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        normalize_prefill_layer_query_heads_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        normalize_prefill_layer_key_heads_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        reshape_prefill_layer_value_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        normalize_prefill_layer_value_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        rope_prefill_layer_query_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        rope_prefill_layer_key_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        build_prefill_layer_kv_cache_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        prepare_prefill_layer_attention_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        compute_prefill_layer_attention_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        combine_prefill_layer_attention_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_attention_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        normalize_prefill_layer_attention_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        add_prefill_layer_attention_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        normalize_prefill_layer_mlp_input_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_mlp_gate_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        gelu_prefill_layer_mlp_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_mlp_up_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        multiply_prefill_layer_mlp_hidden_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_mlp_down_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        normalize_prefill_layer_mlp_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        add_prefill_layer_mlp_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_ple_gate_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        gelu_prefill_layer_ple_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        multiply_prefill_layer_ple_input_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_prefill_layer_ple_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        normalize_prefill_layer_ple_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        add_prefill_layer_ple_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    call_seq!(
        scale_prefill_layer_output_work,
        artifact_store_roots,
        layer_work
    )
}

#[sequence]
fn compute_prefill_layer_attention_input_norm_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_prefill_layer_attention_input_norm_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_prefill_layer_attention_input_norm_work,
        artifact_store_roots,
        unary_work
    )
}

#[sequence]
fn project_prefill_layer_query_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_query_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_query_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn project_prefill_layer_key_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_key_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_key_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn project_prefill_layer_value_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_value_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_value_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn reshape_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, reshape_work) = call_tile!(
        init_reshape_prefill_layer_query_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, reshape_work) = call_recur_tile!(
        transform_next_prefill_layer_reshape_heads_work_row,
        (artifact_store_roots, reshape_work)
    )?;
    call_tile!(
        finalize_reshape_prefill_layer_query_heads_work,
        artifact_store_roots,
        reshape_work
    )
}

#[sequence]
fn reshape_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, reshape_work) = call_tile!(
        init_reshape_prefill_layer_key_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, reshape_work) = call_recur_tile!(
        transform_next_prefill_layer_reshape_heads_work_row,
        (artifact_store_roots, reshape_work)
    )?;
    call_tile!(
        finalize_reshape_prefill_layer_key_heads_work,
        artifact_store_roots,
        reshape_work
    )
}

#[sequence]
fn normalize_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, head_work) = call_tile!(
        init_normalize_prefill_layer_query_heads_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, head_work) = call_recur_tile!(
        transform_next_prefill_layer_head_unary_work_row,
        (artifact_store_roots, head_work)
    )?;
    call_tile!(
        finalize_normalize_prefill_layer_query_heads_work,
        artifact_store_roots,
        head_work
    )
}

#[sequence]
fn normalize_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, head_work) = call_tile!(
        init_normalize_prefill_layer_key_heads_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, head_work) = call_recur_tile!(
        transform_next_prefill_layer_head_unary_work_row,
        (artifact_store_roots, head_work)
    )?;
    call_tile!(
        finalize_normalize_prefill_layer_key_heads_work,
        artifact_store_roots,
        head_work
    )
}

#[sequence]
fn reshape_prefill_layer_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, reshape_work) = call_tile!(
        init_reshape_prefill_layer_value_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, reshape_work) = call_recur_tile!(
        transform_next_prefill_layer_reshape_heads_work_row,
        (artifact_store_roots, reshape_work)
    )?;
    call_tile!(
        finalize_reshape_prefill_layer_value_heads_work,
        artifact_store_roots,
        reshape_work
    )
}

#[sequence]
fn normalize_prefill_layer_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, head_work) = call_tile!(
        init_normalize_prefill_layer_value_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, head_work) = call_recur_tile!(
        transform_next_prefill_layer_head_unary_work_row,
        (artifact_store_roots, head_work)
    )?;
    call_tile!(
        finalize_normalize_prefill_layer_value_heads_work,
        artifact_store_roots,
        head_work
    )
}

#[sequence]
fn rope_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, head_work) = call_tile!(
        init_rope_prefill_layer_query_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, head_work) = call_recur_tile!(
        transform_next_prefill_layer_head_unary_work_row,
        (artifact_store_roots, head_work)
    )?;
    call_tile!(
        finalize_rope_prefill_layer_query_heads_work,
        artifact_store_roots,
        head_work
    )
}

#[sequence]
fn rope_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, head_work) = call_tile!(
        init_rope_prefill_layer_key_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, head_work) = call_recur_tile!(
        transform_next_prefill_layer_head_unary_work_row,
        (artifact_store_roots, head_work)
    )?;
    call_tile!(
        finalize_rope_prefill_layer_key_heads_work,
        artifact_store_roots,
        head_work
    )
}

#[sequence]
fn build_prefill_layer_kv_cache_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, cache_work) = call_tile!(
        init_build_prefill_layer_kv_cache_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, cache_work) = call_recur_tile!(
        transform_next_prefill_layer_kv_cache_work_row,
        (artifact_store_roots, cache_work)
    )?;
    call_tile!(
        finalize_build_prefill_layer_kv_cache_work,
        artifact_store_roots,
        cache_work
    )
}

#[sequence]
fn compute_prefill_layer_attention_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, attention_work) = call_tile!(
        init_compute_prefill_layer_attention_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, attention_work) = call_recur_tile!(
        project_next_prefill_layer_attention_work_row,
        (artifact_store_roots, attention_work)
    )?;
    call_tile!(
        finalize_compute_prefill_layer_attention_work,
        artifact_store_roots,
        attention_work
    )
}

#[sequence]
fn combine_prefill_layer_attention_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, combine_work) = call_tile!(
        init_combine_prefill_layer_attention_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, combine_work) = call_recur_tile!(
        transform_next_prefill_layer_combine_heads_work_row,
        (artifact_store_roots, combine_work)
    )?;
    call_tile!(
        finalize_combine_prefill_layer_attention_heads_work,
        artifact_store_roots,
        combine_work
    )
}

#[sequence]
fn project_prefill_layer_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_attention_output_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_attention_output_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn normalize_prefill_layer_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_normalize_prefill_layer_attention_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_normalize_prefill_layer_attention_output_work,
        artifact_store_roots,
        unary_work
    )
}

#[sequence]
fn add_prefill_layer_attention_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, binary_work) = call_tile!(
        init_add_prefill_layer_attention_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, binary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_binary_work_row,
        (artifact_store_roots, binary_work)
    )?;
    call_tile!(
        finalize_add_prefill_layer_attention_residual_work,
        artifact_store_roots,
        binary_work
    )
}

#[sequence]
fn normalize_prefill_layer_mlp_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_normalize_prefill_layer_mlp_input_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_normalize_prefill_layer_mlp_input_work,
        artifact_store_roots,
        unary_work
    )
}

#[sequence]
fn project_prefill_layer_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_mlp_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_mlp_gate_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn gelu_prefill_layer_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_gelu_prefill_layer_mlp_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_gelu_prefill_layer_mlp_gate_work,
        artifact_store_roots,
        unary_work
    )
}

#[sequence]
fn project_prefill_layer_mlp_up_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_mlp_up_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_mlp_up_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn multiply_prefill_layer_mlp_hidden_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, binary_work) = call_tile!(
        init_multiply_prefill_layer_mlp_hidden_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, binary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_binary_work_row,
        (artifact_store_roots, binary_work)
    )?;
    call_tile!(
        finalize_multiply_prefill_layer_mlp_hidden_work,
        artifact_store_roots,
        binary_work
    )
}

#[sequence]
fn project_prefill_layer_mlp_down_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_mlp_down_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_mlp_down_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn normalize_prefill_layer_mlp_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_normalize_prefill_layer_mlp_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_normalize_prefill_layer_mlp_output_work,
        artifact_store_roots,
        unary_work
    )
}

#[sequence]
fn add_prefill_layer_mlp_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, binary_work) = call_tile!(
        init_add_prefill_layer_mlp_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, binary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_binary_work_row,
        (artifact_store_roots, binary_work)
    )?;
    call_tile!(
        finalize_add_prefill_layer_mlp_residual_work,
        artifact_store_roots,
        binary_work
    )
}

#[sequence]
fn project_prefill_layer_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_ple_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_ple_gate_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn gelu_prefill_layer_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_gelu_prefill_layer_ple_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_gelu_prefill_layer_ple_gate_work,
        artifact_store_roots,
        unary_work
    )
}

#[sequence]
fn multiply_prefill_layer_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, binary_work) = call_tile!(
        init_multiply_prefill_layer_ple_input_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, binary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_binary_work_row,
        (artifact_store_roots, binary_work)
    )?;
    call_tile!(
        finalize_multiply_prefill_layer_ple_input_work,
        artifact_store_roots,
        binary_work
    )
}

#[sequence]
fn project_prefill_layer_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_prefill_layer_ple_output_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_prefill_layer_sequence_projection_work_rows,
        (artifact_store_roots, projection_work),
        layer_source
    )?;
    call_tile!(
        finalize_project_prefill_layer_ple_output_work,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn normalize_prefill_layer_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_normalize_prefill_layer_ple_output_work,
        artifact_store_roots,
        layer_work,
        layer_source
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_normalize_prefill_layer_ple_output_work,
        artifact_store_roots,
        unary_work
    )
}

#[sequence]
fn add_prefill_layer_ple_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, binary_work) = call_tile!(
        init_add_prefill_layer_ple_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, binary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_binary_work_row,
        (artifact_store_roots, binary_work)
    )?;
    call_tile!(
        finalize_add_prefill_layer_ple_residual_work,
        artifact_store_roots,
        binary_work
    )
}

#[sequence]
fn scale_prefill_layer_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let (artifact_store_roots, unary_work) = call_tile!(
        init_scale_prefill_layer_output_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, unary_work) = call_recur_tile!(
        transform_next_prefill_layer_sequence_unary_work_row,
        (artifact_store_roots, unary_work)
    )?;
    call_tile!(
        finalize_scale_prefill_layer_output_work,
        artifact_store_roots,
        unary_work
    )
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
fn init_prefill_layer_artifact_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillLayerStep,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let PrefillLayerStep::Compute {
        layer_state,
        context,
    } = layer_step
    else {
        return Ok((
            artifact_store_roots,
            PrefillLayerArtifactWork::Passthrough(layer_step),
        ));
    };

    let input_ref = layer_state.current_activations_ref.clone();
    let scalars = auth_read!(
        layer_source,
        GemmaPrefillLayerScalarsRequest {
            layer_idx: context.layer_idx
        }
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerArtifactWork::Active(Box::new(PrefillLayerActiveWork {
            projection_rows_per_tile: layer_state.projection_rows_per_tile,
            attention_kv_rows_per_tile: layer_state.attention_kv_rows_per_tile,
            sequence_rows_per_tile: layer_state.sequence_rows_per_tile,
            head_rows_per_tile: layer_state.head_rows_per_tile,
            input_ref,
            layer_state,
            layer_idx: context.layer_idx,
            layer: context.layer,
            donor_cache: context.donor_cache,
            per_layer_input_ref: context.per_layer_input,
            scalars,
            normed_ref: None,
            q_projected_ref: None,
            k_projected_ref: None,
            v_projected_ref: None,
            q_heads_ref: None,
            k_heads_ref: None,
            v_heads_ref: None,
            layer_cache: None,
            donor_cache_ref: None,
            attention_heads_ref: None,
            attention_sequence_ref: None,
            attention_output_ref: None,
            xs_ref: None,
            mlp_normed_ref: None,
            mlp_gate_ref: None,
            mlp_up_ref: None,
            mlp_hidden_ref: None,
            mlp_out_ref: None,
            ple_gate_ref: None,
            ple_projected_ref: None,
            layer_output_ref: None,
        })),
    ))
}

#[tile]
fn finalize_prefill_layer_artifact_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerStep)> {
    match layer_work {
        PrefillLayerArtifactWork::Passthrough(layer_step) => Ok((artifact_store_roots, layer_step)),
        PrefillLayerArtifactWork::Active(work) => Ok((
            artifact_store_roots,
            PrefillLayerStep::Computed {
                layer_state: work.layer_state,
                layer_idx: work.layer_idx,
                layer_output_ref: work.layer_output_ref.ok_or_else(|| {
                    anyhow!("prefill layer compute did not produce an output ref")
                })?,
                layer_cache: work
                    .layer_cache
                    .ok_or_else(|| anyhow!("prefill layer compute did not produce a cache slot"))?,
            },
        )),
    }
}

#[tile(kind = recursive)]
fn transform_next_prefill_layer_sequence_unary_work_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillLayerSequenceUnaryWork,
)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerSequenceUnaryWork::Skip(work),
        )),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_sequence_unary_artifact_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Active { work, state },
            ))
        }
    }
}

#[tile(kind = recursive)]
fn project_next_prefill_layer_sequence_projection_work_rows(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillLayerSequenceProjectionWork,
)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerSequenceProjectionWork::Skip(work),
        )),
        PrefillLayerSequenceProjectionWork::Active {
            work,
            state,
            layer_idx,
            matrix,
        } => {
            if state.is_complete() {
                return Ok((
                    true,
                    artifact_store_roots,
                    PrefillLayerSequenceProjectionWork::Active {
                        work,
                        state,
                        layer_idx,
                        matrix,
                    },
                ));
            }

            let end = state
                .next_projection_row_idx()
                .saturating_add(state.rows_per_tile())
                .min(state.projection_rows());
            let start_projection_row_idx = state.next_projection_row_idx();
            let start_token_idx = state.next_token_idx();
            let mut rows = Vec::with_capacity(end - state.next_projection_row_idx());
            for row_idx in state.next_projection_row_idx()..end {
                rows.push(auth_read!(
                    layer_source,
                    crate::shared::raster_contracts::prefill_layer::GemmaPrefillLayerMatrixRowRequest {
                        layer_idx,
                        matrix,
                        row_idx,
                    },
                )?);
            }
            let (artifact_store_roots, state) =
                append_projection_chunk_to_artifact_state(artifact_store_roots, state, &rows)?;
            trace_event(format!(
                "progress prefill.layer.projection layer={} matrix={:?} token={}/{} projection_rows={}..{} of {} input_width={}",
                layer_idx,
                matrix,
                start_token_idx + 1,
                state.token_count(),
                start_projection_row_idx,
                end,
                state.projection_rows(),
                state.input_width()
            ));
            Ok((
                false,
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Active {
                    work,
                    state,
                    layer_idx,
                    matrix,
                },
            ))
        }
    }
}

#[tile(kind = recursive)]
fn transform_next_prefill_layer_reshape_heads_work_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    reshape_work: PrefillLayerReshapeHeadsWork,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerReshapeHeadsWork)> {
    match reshape_work {
        PrefillLayerReshapeHeadsWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerReshapeHeadsWork::Skip(work),
        )),
        PrefillLayerReshapeHeadsWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_reshape_heads_artifact_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillLayerReshapeHeadsWork::Active { work, state },
            ))
        }
    }
}

#[tile(kind = recursive)]
fn transform_next_prefill_layer_head_unary_work_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_work: PrefillLayerHeadUnaryWork,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerHeadUnaryWork)> {
    match head_work {
        PrefillLayerHeadUnaryWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerHeadUnaryWork::Skip(work),
        )),
        PrefillLayerHeadUnaryWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_head_unary_artifact_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillLayerHeadUnaryWork::Active { work, state },
            ))
        }
    }
}

#[tile(kind = recursive)]
fn transform_next_prefill_layer_kv_cache_work_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    cache_work: PrefillLayerKvCacheWork,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerKvCacheWork)> {
    match cache_work {
        PrefillLayerKvCacheWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerKvCacheWork::Skip(work),
        )),
        PrefillLayerKvCacheWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_kv_cache_artifact_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillLayerKvCacheWork::Active { work, state },
            ))
        }
    }
}

#[tile(kind = recursive)]
fn project_next_prefill_layer_attention_work_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    attention_work: PrefillLayerAttentionWork,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerAttentionWork)> {
    match attention_work {
        PrefillLayerAttentionWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerAttentionWork::Skip(work),
        )),
        PrefillLayerAttentionWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_attention_artifact_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillLayerAttentionWork::Active { work, state },
            ))
        }
    }
}

#[tile(kind = recursive)]
fn transform_next_prefill_layer_combine_heads_work_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    combine_work: PrefillLayerCombineHeadsWork,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerCombineHeadsWork)> {
    match combine_work {
        PrefillLayerCombineHeadsWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerCombineHeadsWork::Skip(work),
        )),
        PrefillLayerCombineHeadsWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_combine_heads_artifact_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillLayerCombineHeadsWork::Active { work, state },
            ))
        }
    }
}

#[tile(kind = recursive)]
fn transform_next_prefill_layer_sequence_binary_work_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_work: PrefillLayerSequenceBinaryWork,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    PrefillLayerSequenceBinaryWork,
)> {
    match binary_work {
        PrefillLayerSequenceBinaryWork::Skip(work) => Ok((
            true,
            artifact_store_roots,
            PrefillLayerSequenceBinaryWork::Skip(work),
        )),
        PrefillLayerSequenceBinaryWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_sequence_binary_artifact_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                PrefillLayerSequenceBinaryWork::Active { work, state },
            ))
        }
    }
}

#[tile]
fn init_prefill_layer_attention_input_norm_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let norm_weights = auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaPrefillLayerNormKind::InputLayer
        }
    )?;
    let (artifact_store_roots, state) = init_sequence_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        work.input_ref.clone(),
        RasterTensorId::new(format!(
            "prefill.layer.{}.attention.input_norm.output",
            work.layer_idx
        ))?,
        Some(&norm_weights),
        Some(work.scalars.rms_norm_eps),
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_prefill_layer_attention_input_norm_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.normed_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_query_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let input_ref = require_activation_ref(work.normed_ref.clone(), "attention input norm ref")?;
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("prefill.layer.{}.q_proj.output", work.layer_idx))?,
        work.layer.q_proj_shape.rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::Query,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_query_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.q_projected_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_key_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let input_ref = require_activation_ref(work.normed_ref.clone(), "attention input norm ref")?;
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("prefill.layer.{}.k_proj.output", work.layer_idx))?,
        work.layer.k_proj_shape.rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::Key,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_key_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.k_projected_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_value_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let mut work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    if !work.layer.has_v_proj {
        if work.layer.attention_k_eq_v {
            work.v_projected_ref = Some(require_activation_ref(
                work.k_projected_ref.clone(),
                "key projection ref",
            )?);
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(PrefillLayerArtifactWork::Active(work)),
            ));
        }
        bail!("Gemma prefill layer is missing v_proj without attention_k_eq_v enabled");
    }
    let input_ref = require_activation_ref(work.normed_ref.clone(), "attention input norm ref")?;
    let projection_rows = work
        .layer
        .v_proj_shape
        .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing v_proj shape"))?
        .rows;
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("prefill.layer.{}.v_proj.output", work.layer_idx))?,
        projection_rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::Value,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_value_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.v_projected_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_reshape_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerReshapeHeadsWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerReshapeHeadsWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let input_ref = require_activation_ref(work.q_projected_ref.clone(), "query projection ref")?;
    let (artifact_store_roots, state) = init_reshape_heads_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("prefill.layer.{}.q_heads.output", work.layer_idx))?,
        work.layer.num_heads,
        work.layer.head_dim,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerReshapeHeadsWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_reshape_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    reshape_work: PrefillLayerReshapeHeadsWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match reshape_work {
        PrefillLayerReshapeHeadsWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerReshapeHeadsWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_reshape_heads_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.q_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_reshape_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerReshapeHeadsWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerReshapeHeadsWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let input_ref = require_activation_ref(work.k_projected_ref.clone(), "key projection ref")?;
    let (artifact_store_roots, state) = init_reshape_heads_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("prefill.layer.{}.k_heads.output", work.layer_idx))?,
        work.layer.num_kv_heads,
        work.layer.head_dim,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerReshapeHeadsWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_reshape_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    reshape_work: PrefillLayerReshapeHeadsWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match reshape_work {
        PrefillLayerReshapeHeadsWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerReshapeHeadsWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_reshape_heads_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.k_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_normalize_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerHeadUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerHeadUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let heads_ref = require_heads_ref(work.q_heads_ref.clone(), "query heads ref")?;
    let norm_weights = auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaPrefillLayerNormKind::Query
        }
    )?;
    let (artifact_store_roots, state) = init_head_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("prefill.layer.{}.q_norm.output", work.layer_idx))?,
        Some(&norm_weights),
        Some(work.scalars.rms_norm_eps),
        work.head_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerHeadUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_normalize_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_work: PrefillLayerHeadUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match head_work {
        PrefillLayerHeadUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerHeadUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_head_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.q_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_normalize_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerHeadUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerHeadUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let heads_ref = require_heads_ref(work.k_heads_ref.clone(), "key heads ref")?;
    let norm_weights = auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaPrefillLayerNormKind::Key
        }
    )?;
    let (artifact_store_roots, state) = init_head_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("prefill.layer.{}.k_norm.output", work.layer_idx))?,
        Some(&norm_weights),
        Some(work.scalars.rms_norm_eps),
        work.head_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerHeadUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_normalize_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_work: PrefillLayerHeadUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match head_work {
        PrefillLayerHeadUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerHeadUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_head_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.k_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_reshape_prefill_layer_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerReshapeHeadsWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerReshapeHeadsWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let input_ref = require_activation_ref(work.v_projected_ref.clone(), "value projection ref")?;
    let (artifact_store_roots, state) = init_reshape_heads_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("prefill.layer.{}.v_heads.output", work.layer_idx))?,
        work.layer.num_kv_heads,
        work.layer.head_dim,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerReshapeHeadsWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_reshape_prefill_layer_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    reshape_work: PrefillLayerReshapeHeadsWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match reshape_work {
        PrefillLayerReshapeHeadsWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerReshapeHeadsWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_reshape_heads_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.v_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_normalize_prefill_layer_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerHeadUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerHeadUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let heads_ref = require_heads_ref(work.v_heads_ref.clone(), "value heads ref")?;
    let (artifact_store_roots, state) = init_value_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("prefill.layer.{}.v_norm.output", work.layer_idx))?,
        Some(work.scalars.rms_norm_eps),
        work.head_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerHeadUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_normalize_prefill_layer_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_work: PrefillLayerHeadUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match head_work {
        PrefillLayerHeadUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerHeadUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_head_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.v_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_rope_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerHeadUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerHeadUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let heads_ref = require_heads_ref(work.q_heads_ref.clone(), "query heads ref")?;
    let (artifact_store_roots, state) = init_rope_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("prefill.layer.{}.q_rope.output", work.layer_idx))?,
        work.layer.partial_rotary_dim,
        work.layer.rope_freq_base_dim,
        work.scalars.rope_base,
        0,
        work.head_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerHeadUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_rope_prefill_layer_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_work: PrefillLayerHeadUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match head_work {
        PrefillLayerHeadUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerHeadUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_head_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.q_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_rope_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerHeadUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerHeadUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let heads_ref = require_heads_ref(work.k_heads_ref.clone(), "key heads ref")?;
    let (artifact_store_roots, state) = init_rope_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("prefill.layer.{}.k_rope.output", work.layer_idx))?,
        work.layer.partial_rotary_dim,
        work.layer.rope_freq_base_dim,
        work.scalars.rope_base,
        0,
        work.head_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerHeadUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_rope_prefill_layer_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    head_work: PrefillLayerHeadUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match head_work {
        PrefillLayerHeadUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerHeadUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_head_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.k_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_build_prefill_layer_kv_cache_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerKvCacheWork)> {
    let mut work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerKvCacheWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let key_ref = require_heads_ref(work.k_heads_ref.clone(), "key heads ref")?;
    let retained_cache_len =
        retained_prefill_kv_cache_len(&key_ref, work.layer.cache_sliding_window)?;
    if work.donor_cache.is_some() || retained_cache_len == 0 {
        work.layer_cache = Some(PrefillLayerCacheSlot::Empty {
            num_kv_heads: work.layer.num_kv_heads,
        });
        return Ok((
            artifact_store_roots,
            PrefillLayerKvCacheWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    }
    let value_ref = require_heads_ref(work.v_heads_ref.clone(), "value heads ref")?;
    let (artifact_store_roots, state) = init_kv_cache_build_artifact_state_from_refs(
        artifact_store_roots,
        key_ref,
        value_ref,
        RasterTensorId::new(format!("prefill.layer.cache.{}.keys", work.layer_idx))?,
        RasterTensorId::new(format!("prefill.layer.cache.{}.values", work.layer_idx))?,
        work.layer.cache_sliding_window,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerKvCacheWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_build_prefill_layer_kv_cache_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    cache_work: PrefillLayerKvCacheWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match cache_work {
        PrefillLayerKvCacheWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerKvCacheWork::Active { work, state } => {
            let (artifact_store_roots, cache_ref) =
                finalize_kv_cache_build_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.layer_cache = Some(PrefillLayerCacheSlot::Ref(cache_ref));
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn prepare_prefill_layer_attention_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    let mut work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => return Ok((artifact_store_roots, layer_work)),
        PrefillLayerArtifactWork::Active(work) => work,
    };
    work.donor_cache_ref = work
        .donor_cache
        .as_ref()
        .map(|cache| match cache {
            PrefillLayerCacheSlot::Empty { .. } => {
                bail!("transformer prefill donor cache is empty")
            }
            PrefillLayerCacheSlot::Ref(cache_ref) => Ok(cache_ref.clone()),
        })
        .transpose()?;
    Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
}

#[tile]
fn init_compute_prefill_layer_attention_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerAttentionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerAttentionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let attention_window = match work.layer.attention_kind {
        GemmaPrefillAttentionKind::Full => None,
        GemmaPrefillAttentionKind::Sliding => Some(
            work.layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?,
        ),
    };
    let (artifact_store_roots, state) = init_attention_artifact_row_state_from_refs(
        artifact_store_roots,
        require_heads_ref(work.q_heads_ref.clone(), "query heads ref")?,
        require_heads_ref(work.k_heads_ref.clone(), "key heads ref")?,
        require_heads_ref(work.v_heads_ref.clone(), "value heads ref")?,
        work.donor_cache_ref.clone(),
        RasterTensorId::new(format!("prefill.layer.{}.attention.output", work.layer_idx))?,
        attention_window,
        work.attention_kv_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerAttentionWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_compute_prefill_layer_attention_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    attention_work: PrefillLayerAttentionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match attention_work {
        PrefillLayerAttentionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerAttentionWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_attention_artifact_row_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.attention_heads_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_combine_prefill_layer_attention_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerCombineHeadsWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerCombineHeadsWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let heads_ref = require_heads_ref(work.attention_heads_ref.clone(), "attention heads ref")?;
    let (artifact_store_roots, state) = init_combine_heads_artifact_state_from_ref(
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!(
            "prefill.layer.{}.attention.combine.output",
            work.layer_idx
        ))?,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerCombineHeadsWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_combine_prefill_layer_attention_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    combine_work: PrefillLayerCombineHeadsWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match combine_work {
        PrefillLayerCombineHeadsWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerCombineHeadsWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_combine_heads_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.attention_sequence_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let input_ref = require_activation_ref(
        work.attention_sequence_ref.clone(),
        "attention sequence ref",
    )?;
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("prefill.layer.{}.o_proj.output", work.layer_idx))?,
        work.layer.o_proj_shape.rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::Output,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.attention_output_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_normalize_prefill_layer_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let norm_weights = auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaPrefillLayerNormKind::PostAttention
        }
    )?;
    let input_ref =
        require_activation_ref(work.attention_output_ref.clone(), "attention output ref")?;
    let (artifact_store_roots, state) = init_sequence_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!(
            "prefill.layer.{}.attention.post_norm.output",
            work.layer_idx
        ))?,
        Some(&norm_weights),
        Some(work.scalars.rms_norm_eps),
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_normalize_prefill_layer_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.attention_output_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_add_prefill_layer_attention_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceBinaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceBinaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_sequence_add_artifact_state_from_refs(
        artifact_store_roots,
        work.input_ref.clone(),
        require_activation_ref(work.attention_output_ref.clone(), "attention output ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.attention.residual.output",
            work.layer_idx
        ))?,
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceBinaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_add_prefill_layer_attention_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_work: PrefillLayerSequenceBinaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match binary_work {
        PrefillLayerSequenceBinaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceBinaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_binary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.xs_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_normalize_prefill_layer_mlp_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let norm_weights = auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaPrefillLayerNormKind::PreFeedForward
        }
    )?;
    let (artifact_store_roots, state) = init_sequence_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.xs_ref.clone(), "attention residual ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.mlp.pre_norm.output",
            work.layer_idx
        ))?,
        Some(&norm_weights),
        Some(work.scalars.rms_norm_eps),
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_normalize_prefill_layer_mlp_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.mlp_normed_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.mlp_normed_ref.clone(), "MLP normed ref")?,
        RasterTensorId::new(format!("prefill.layer.{}.gate_proj.output", work.layer_idx))?,
        work.layer.gate_proj_shape.rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::Gate,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.mlp_gate_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_gelu_prefill_layer_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_sequence_gelu_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.mlp_gate_ref.clone(), "MLP gate ref")?,
        RasterTensorId::new(format!("prefill.layer.{}.gate_gelu.output", work.layer_idx))?,
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_gelu_prefill_layer_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.mlp_gate_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_mlp_up_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.mlp_normed_ref.clone(), "MLP normed ref")?,
        RasterTensorId::new(format!("prefill.layer.{}.up_proj.output", work.layer_idx))?,
        work.layer.up_proj_shape.rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::Up,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_mlp_up_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.mlp_up_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_multiply_prefill_layer_mlp_hidden_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceBinaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceBinaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_sequence_mul_artifact_state_from_refs(
        artifact_store_roots,
        require_activation_ref(work.mlp_gate_ref.clone(), "MLP gate ref")?,
        require_activation_ref(work.mlp_up_ref.clone(), "MLP up ref")?,
        RasterTensorId::new(format!("prefill.layer.{}.ff_hidden.output", work.layer_idx))?,
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceBinaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_multiply_prefill_layer_mlp_hidden_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_work: PrefillLayerSequenceBinaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match binary_work {
        PrefillLayerSequenceBinaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceBinaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_binary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.mlp_hidden_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_mlp_down_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.mlp_hidden_ref.clone(), "MLP hidden ref")?,
        RasterTensorId::new(format!("prefill.layer.{}.down_proj.output", work.layer_idx))?,
        work.layer.down_proj_shape.rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::Down,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_mlp_down_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.mlp_out_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_normalize_prefill_layer_mlp_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let norm_weights = auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaPrefillLayerNormKind::PostFeedForward
        }
    )?;
    let (artifact_store_roots, state) = init_sequence_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.mlp_out_ref.clone(), "MLP output ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.ff_post_norm.output",
            work.layer_idx
        ))?,
        Some(&norm_weights),
        Some(work.scalars.rms_norm_eps),
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_normalize_prefill_layer_mlp_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.mlp_out_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_add_prefill_layer_mlp_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceBinaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceBinaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_sequence_add_artifact_state_from_refs(
        artifact_store_roots,
        require_activation_ref(work.xs_ref.clone(), "attention residual ref")?,
        require_activation_ref(work.mlp_out_ref.clone(), "MLP output ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.mlp.residual.output",
            work.layer_idx
        ))?,
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceBinaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_add_prefill_layer_mlp_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_work: PrefillLayerSequenceBinaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match binary_work {
        PrefillLayerSequenceBinaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceBinaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_binary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.xs_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((
            artifact_store_roots,
            PrefillLayerSequenceProjectionWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    }
    let projection_rows = work
        .layer
        .ple_input_gate_shape
        .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing PLE input gate shape"))?
        .rows;
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.xs_ref.clone(), "MLP residual ref")?,
        RasterTensorId::new(format!("prefill.layer.{}.ple_gate.output", work.layer_idx))?,
        projection_rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::PleInputGate,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.ple_gate_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_gelu_prefill_layer_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((
            artifact_store_roots,
            PrefillLayerSequenceUnaryWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    }
    let (artifact_store_roots, state) = init_sequence_gelu_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.ple_gate_ref.clone(), "PLE gate ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.ple_gate_gelu.output",
            work.layer_idx
        ))?,
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_gelu_prefill_layer_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.ple_gate_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_multiply_prefill_layer_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceBinaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceBinaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let Some(per_layer_input_ref) = work.per_layer_input_ref.clone() else {
        return Ok((
            artifact_store_roots,
            PrefillLayerSequenceBinaryWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    };
    let (artifact_store_roots, state) = init_sequence_mul_artifact_state_from_refs(
        artifact_store_roots,
        require_activation_ref(work.ple_gate_ref.clone(), "PLE gate ref")?,
        per_layer_input_ref,
        RasterTensorId::new(format!(
            "prefill.layer.{}.ple_input_mul.output",
            work.layer_idx
        ))?,
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceBinaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_multiply_prefill_layer_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_work: PrefillLayerSequenceBinaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match binary_work {
        PrefillLayerSequenceBinaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceBinaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_binary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.ple_gate_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_project_prefill_layer_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceProjectionWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceProjectionWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((
            artifact_store_roots,
            PrefillLayerSequenceProjectionWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    }
    let projection_rows = work
        .layer
        .ple_layer_projection_shape
        .ok_or_else(|| {
            anyhow!("Gemma prefill layer metadata is missing PLE layer projection shape")
        })?
        .rows;
    let (artifact_store_roots, state) = init_sequence_projection_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.ple_gate_ref.clone(), "PLE gated input ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.ple_layer_projection.output",
            work.layer_idx
        ))?,
        projection_rows,
        work.projection_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceProjectionWork::Active {
            layer_idx: work.layer_idx,
            matrix: GemmaPrefillLayerMatrixKind::PleLayerProjection,
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_project_prefill_layer_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: PrefillLayerSequenceProjectionWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match projection_work {
        PrefillLayerSequenceProjectionWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceProjectionWork::Active { work, state, .. } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.ple_projected_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_normalize_prefill_layer_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((
            artifact_store_roots,
            PrefillLayerSequenceUnaryWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    }
    let norm_weights = auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaPrefillLayerNormKind::PlePostInput
        }
    )?;
    let (artifact_store_roots, state) = init_sequence_rms_norm_artifact_state_from_ref(
        artifact_store_roots,
        require_activation_ref(work.ple_projected_ref.clone(), "PLE projected ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.ple_post_norm.output",
            work.layer_idx
        ))?,
        Some(&norm_weights),
        Some(work.scalars.rms_norm_eps),
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_normalize_prefill_layer_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.ple_projected_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_add_prefill_layer_ple_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceBinaryWork)> {
    let work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceBinaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((
            artifact_store_roots,
            PrefillLayerSequenceBinaryWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    }
    let (artifact_store_roots, state) = init_sequence_add_artifact_state_from_refs(
        artifact_store_roots,
        require_activation_ref(work.xs_ref.clone(), "MLP residual ref")?,
        require_activation_ref(work.ple_projected_ref.clone(), "PLE projected ref")?,
        RasterTensorId::new(format!(
            "prefill.layer.{}.ple_residual.output",
            work.layer_idx
        ))?,
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceBinaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_add_prefill_layer_ple_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    binary_work: PrefillLayerSequenceBinaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match binary_work {
        PrefillLayerSequenceBinaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceBinaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_binary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.xs_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
fn init_scale_prefill_layer_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: PrefillLayerArtifactWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerSequenceUnaryWork)> {
    let mut work = match layer_work {
        PrefillLayerArtifactWork::Passthrough(_) => {
            return Ok((
                artifact_store_roots,
                PrefillLayerSequenceUnaryWork::Skip(layer_work),
            ));
        }
        PrefillLayerArtifactWork::Active(work) => work,
    };
    let input_ref = require_activation_ref(work.xs_ref.clone(), "layer output ref")?;
    let Some(scalar) = work.scalars.layer_scalar else {
        work.layer_output_ref = Some(input_ref);
        return Ok((
            artifact_store_roots,
            PrefillLayerSequenceUnaryWork::Skip(PrefillLayerArtifactWork::Active(work)),
        ));
    };
    let (artifact_store_roots, state) = init_sequence_scale_artifact_state_from_ref(
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!(
            "prefill.layer.{}.layer_scalar.output",
            work.layer_idx
        ))?,
        Some(scalar),
        work.sequence_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        PrefillLayerSequenceUnaryWork::Active {
            work: PrefillLayerArtifactWork::Active(work),
            state,
        },
    ))
}

#[tile]
fn finalize_scale_prefill_layer_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    unary_work: PrefillLayerSequenceUnaryWork,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerArtifactWork)> {
    match unary_work {
        PrefillLayerSequenceUnaryWork::Skip(work) => Ok((artifact_store_roots, work)),
        PrefillLayerSequenceUnaryWork::Active { work, state } => {
            let (artifact_store_roots, output_ref) =
                finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)?;
            let mut work = active_prefill_layer_work(work)?;
            work.layer_output_ref = Some(output_ref);
            Ok((artifact_store_roots, PrefillLayerArtifactWork::Active(work)))
        }
    }
}

#[tile]
pub fn init_prefill_layer_state_from_input_embedding_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    layer_source: &RasterPrefillLayerSource<'_>,
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
fn init_prefill_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_state: PrefillLayerRasterState,
    layer_source: &RasterPrefillLayerSource<'_>,
) -> Result<(RasterArtifactStoreRoots, PrefillLayerStep)> {
    if layer_state.next_layer_idx >= layer_state.layer_count {
        return Ok((
            artifact_store_roots,
            PrefillLayerStep::Complete { layer_state },
        ));
    }

    let context = prepare_next_prefill_layer_context(&layer_state, layer_source)?;
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
    trace_event(format!(
        "progress prefill.layer layer={}/{} tokens={} attention={:?} ple={} donor={:?}",
        context.layer_idx + 1,
        layer_state.layer_count,
        token_count,
        context.layer.attention_kind,
        context.layer.has_ple,
        context.layer.kv_shared_layer_index
    ));

    Ok((
        artifact_store_roots,
        PrefillLayerStep::Compute {
            layer_state,
            context,
        },
    ))
}

#[tile]
fn finalize_prefill_layer_step(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_step: PrefillLayerStep,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    match layer_step {
        PrefillLayerStep::Complete { layer_state } => Ok((true, artifact_store_roots, layer_state)),
        PrefillLayerStep::Compute { layer_state, .. } => {
            bail!(
                "prefill layer {} reached finalization before compute completed",
                layer_state.next_layer_idx
            )
        }
        PrefillLayerStep::Computed {
            layer_state,
            layer_idx,
            layer_output_ref,
            layer_cache,
        } => update_prefill_layer_state_refs_with_roots(
            artifact_store_roots,
            layer_state,
            layer_idx,
            layer_output_ref,
            layer_cache,
        ),
    }
}

#[tile]
pub fn read_prefill_layer_scalars(
    layer_source: &RasterPrefillLayerSource<'_>,
    layer_idx: usize,
) -> Result<GemmaPrefillLayerScalars> {
    auth_read!(layer_source, GemmaPrefillLayerScalarsRequest { layer_idx })
}

#[tile]
pub fn read_prefill_layer_norm_weights(
    layer_source: &RasterPrefillLayerSource<'_>,
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
    layer_source: &RasterPrefillLayerSource<'_>,
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
