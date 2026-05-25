use anyhow::{anyhow, bail, Result};

use crate::decode_transition::raster::auth_source::{
    GemmaDecodeAttentionKind, GemmaDecodeEmbeddingRowRequest, GemmaDecodeFinalNormWeightsRequest,
    GemmaDecodeFinalScalarsRequest, GemmaDecodeLayerMatrixKind, GemmaDecodeLayerMetadata,
    GemmaDecodeLayerMetadataRequest, GemmaDecodeLayerNormKind, GemmaDecodeLayerNormWeightsRequest,
    GemmaDecodeLayerScalarsRequest, GemmaDecodePleProjectionNormWeightsRequest,
    GemmaDecodePleScalarsRequest, GemmaDecodePleTokenEmbeddingRowRequest,
    GemmaDecodeTransitionMetadataRequest,
};
use crate::dsl::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::external_artifacts::CommittedExternalSource;
use crate::shared::artifacts::raster_artifact_store::{
    read_selected_token_from_roots, RasterArtifactId, RasterArtifactStoreRoots,
    RasterSelectedTokenRef,
};
use crate::shared::model::transformer::{
    ActivationSequence, InternalActivationSequence, TransformerDecodeState,
    TransformerDecodeStepResult,
};
use crate::shared::numerics::det_num::{
    acc_add_sat, add_sat, attention_score as det_attention_score, attention_softmax_exp_term,
    attention_softmax_raw_weight, attention_softmax_residual, mac_bits, requantize, Acc, Act,
};
use crate::shared::raster_kernels::transformer::{
    apply_rope_to_heads, rms_norm_heads, rms_norm_sequence, validate_attention_kv_rows_per_tile,
    validate_projection_rows_per_tile, value_rms_norm_heads, RasterActivationRow,
    RasterActivationSequence, RasterAttentionHeadSequence,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    append_head_row_by_source_name_with_roots, append_sequence_row_by_source_name_with_roots,
    finalize_heads_builder_by_source_name_with_roots,
    finalize_kv_cache_builders_by_source_name_with_roots,
    finalize_sequence_builder_by_source_name_with_roots, read_head_row_from_roots,
    read_kv_row_from_roots, start_sequence_builder_with_roots, RasterActivationSequenceRef,
    RasterAttentionHeadsRef, RasterHeadRowRequest, RasterKvCacheRef, RasterKvRowKind,
    RasterKvRowRequest, RasterTensorId,
};
use crate::RasterSizingControls;

use super::super::native::ActivationSequenceWithCache;

use super::types::*;
use super::utils::*;

// Raster execution sequences, ordered from the primary entry point outward.

#[sequence]
pub fn main(
    input_roots: RasterDecodeTransitionInputRoots,
    source: &CommittedExternalSource,
) -> Result<RasterDecodeTransitionOutputRefs> {
    let (_artifact_store_roots, decode_state) = call_tile!(
        init_decode_transition_state_refs_from_input_roots,
        input_roots,
        source
    )?;
    let (artifact_store_roots, decode_state) = call_recur_seq!(
        compute_next_decode_layer_with_roots,
        (_artifact_store_roots, decode_state),
        source
    )?;
    let final_work = call_tile!(
        init_decode_transition_final_work_with_roots,
        artifact_store_roots,
        decode_state
    )?;
    let final_work = call_seq!(
        compute_decode_final_logits_work_with_roots,
        final_work,
        source
    )?;
    call_tile!(finalize_decode_transition_output_refs, final_work)
}

#[sequence(kind = recursive)]
pub fn compute_next_decode_layer_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    decode_state: DecodeTransitionRasterState,
    source: &CommittedExternalSource,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    let (artifact_store_roots, layer_work) = call_tile!(
        init_next_decode_layer_work,
        artifact_store_roots,
        decode_state,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        run_decode_layer_work_with_roots,
        artifact_store_roots,
        layer_work,
        source
    )?;
    call_tile!(
        finalize_next_decode_layer_work,
        artifact_store_roots,
        layer_work
    )
}

#[sequence]
fn run_decode_layer_work_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, layer_work) = call_seq!(
        compute_decode_ple_input_work_with_roots,
        artifact_store_roots,
        layer_work,
        source
    )?;
    call_seq!(
        run_basic_decode_layer_work_with_roots,
        artifact_store_roots,
        layer_work,
        source
    )
}

#[sequence]
fn compute_decode_ple_input_work_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, ple_work) = call_tile!(
        init_decode_ple_input_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_ple_input_work,
        artifact_store_roots,
        ple_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    let (artifact_store_roots, ple_work) = call_tile!(
        finalize_project_decode_ple_input_work,
        artifact_store_roots,
        continuation
    )?;
    call_tile!(
        finalize_decode_ple_input_work,
        artifact_store_roots,
        ple_work
    )
}

#[sequence]
fn project_ref_with_decode_source_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source: &CommittedExternalSource,
    projection_kind: DecodeProjectionKind,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: String,
    softcap_bits: Option<i32>,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, projection_state) = call_tile!(
        init_decode_row_projection_artifact,
        artifact_store_roots,
        input_ref,
        projection_kind,
        projection_rows,
        rows_per_tile,
        output_id,
        softcap_bits
    )?;
    let (artifact_store_roots, projection_state) = call_recur_tile!(
        project_next_decode_projection_chunk_with_roots,
        (artifact_store_roots, projection_state),
        source
    )?;
    call_tile!(
        finalize_decode_row_projection_ref_with_roots,
        artifact_store_roots,
        projection_state
    )
}

#[sequence]
fn project_decode_projection_work_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: DecodeProjectionWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionContinuation)> {
    let (artifact_store_roots, projection_work) = call_recur_tile!(
        project_next_decode_projection_work_chunk_with_roots,
        (artifact_store_roots, projection_work),
        source
    )?;
    call_tile!(
        finalize_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work
    )
}

#[sequence]
fn compute_decode_final_logits_work_with_roots(
    final_work: DecodeTransitionFinalWork,
    source: &CommittedExternalSource,
) -> Result<DecodeTransitionFinalWork> {
    let final_work = call_tile!(normalize_decode_final_work_with_roots, final_work, source)?;
    let (artifact_store_roots, projection_work) =
        call_tile!(init_decode_final_logits_projection_work, final_work, source)?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_final_logits_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn run_basic_decode_layer_work_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, layer_work) = call_tile!(
        prepare_basic_decode_layer_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_attention_input_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_attention_query_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_attention_key_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_attention_value_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        reshape_decode_attention_query_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        reshape_decode_attention_key_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        reshape_decode_attention_value_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_attention_query_heads_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_attention_key_heads_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_attention_value_heads_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        rope_decode_attention_query_heads_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        rope_decode_attention_key_heads_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        update_decode_attention_cache_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        compute_decode_attention_scores_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        combine_decode_attention_heads_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_attention_output_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_attention_output_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        add_decode_attention_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_mlp_input_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_mlp_gate_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) =
        call_tile!(gelu_decode_mlp_gate_work, artifact_store_roots, layer_work)?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_mlp_up_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        multiply_decode_mlp_hidden_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_mlp_down_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_mlp_output_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        add_decode_mlp_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_ple_gate_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) =
        call_tile!(gelu_decode_ple_gate_work, artifact_store_roots, layer_work)?;
    let (artifact_store_roots, layer_work) = call_tile!(
        multiply_decode_ple_input_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, layer_work) = call_seq!(
        project_decode_ple_output_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        normalize_decode_ple_output_work,
        artifact_store_roots,
        layer_work,
        source
    )?;
    let (artifact_store_roots, layer_work) = call_tile!(
        add_decode_ple_residual_work,
        artifact_store_roots,
        layer_work
    )?;
    call_tile!(
        scale_decode_layer_output_work,
        artifact_store_roots,
        layer_work,
        source
    )
}

#[sequence]
fn project_decode_attention_query_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_attention_query_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn project_decode_attention_key_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_attention_key_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn project_decode_attention_value_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_attention_value_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn update_decode_attention_cache_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, cache_work) = call_tile!(
        init_update_decode_attention_cache_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, cache_work) = call_recur_tile!(
        compute_next_decode_kv_cache_append_work_row_with_roots,
        (artifact_store_roots, cache_work)
    )?;
    call_tile!(
        finalize_update_decode_attention_cache_work,
        artifact_store_roots,
        cache_work
    )
}

#[sequence]
fn compute_decode_attention_scores_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, attention_work) = call_tile!(
        init_compute_decode_attention_scores_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, attention_work) = call_recur_tile!(
        compute_next_decode_attention_work_head_with_roots,
        (artifact_store_roots, attention_work)
    )?;
    call_tile!(
        finalize_compute_decode_attention_scores_work,
        artifact_store_roots,
        attention_work
    )
}

#[sequence]
fn project_decode_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_attention_output_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn project_decode_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_mlp_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn project_decode_mlp_up_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_mlp_up_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn project_decode_mlp_down_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_mlp_down_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn project_decode_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_ple_gate_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
fn project_decode_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let (artifact_store_roots, projection_work) = call_tile!(
        init_project_decode_ple_output_work,
        artifact_store_roots,
        layer_work
    )?;
    let (artifact_store_roots, continuation) = call_seq!(
        project_decode_projection_work_with_roots,
        artifact_store_roots,
        projection_work,
        source
    )?;
    call_tile!(
        finalize_decode_layer_projection_work,
        artifact_store_roots,
        continuation
    )
}

#[sequence]
pub fn run(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &CommittedExternalSource,
    raster_sizing: RasterSizingControls,
) -> Result<TransformerDecodeStepResult> {
    let input_roots = call_tile!(
        init_decode_transition_run_input_roots,
        transformer_decode_state,
        next_token,
        source,
        raster_sizing
    )?;
    let output = call_seq!(main, input_roots, source)?;
    call_tile!(finalize_decode_transition_run_output, output)
}

#[sequence]
pub fn main_state_refs(
    input_roots: RasterDecodeTransitionInputRefs,
    source: &CommittedExternalSource,
) -> Result<RasterDecodeTransitionOutputStateRefs> {
    let (_artifact_store_roots, decode_state) = call_tile!(
        init_decode_transition_state_from_input_refs,
        input_roots,
        source
    )?;
    let (artifact_store_roots, decode_state) = call_recur_seq!(
        compute_next_decode_layer_with_roots,
        (_artifact_store_roots, decode_state),
        source
    )?;
    let final_work = call_tile!(
        init_decode_transition_final_work_with_roots,
        artifact_store_roots,
        decode_state
    )?;
    let final_work = call_seq!(
        compute_decode_final_logits_work_with_roots,
        final_work,
        source
    )?;
    call_tile!(finalize_decode_transition_output_state_refs, final_work)
}

// Raster execution tiles, ordered by the sequence calls that reach them.

#[tile]
fn prepare_basic_decode_layer_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let input = read_activation_row_from_ref_roots(
        &artifact_store_roots,
        &work.decode_state.current_activation_ref,
    )?;
    validate_row_width(&input, work.layer.hidden_size, "decode attention input")?;
    let kv_groups = work
        .layer
        .num_heads
        .checked_div(work.layer.num_kv_heads)
        .ok_or_else(|| anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn normalize_decode_attention_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let norm_weights = auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaDecodeLayerNormKind::InputLayer
        }
    )?;
    let (artifact_store_roots, output_ref) = rms_norm_decode_ref(
        artifact_store_roots,
        work.decode_state.current_activation_ref.clone(),
        &norm_weights,
        scalars.rms_norm_eps,
        "decode layer RMSNorm",
        format!("{}.attention_block.input_norm", work.output_prefix),
    )?;
    work.attention_normed_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_project_decode_attention_query_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerAttentionQuery(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(
            work.attention_normed_ref.clone(),
            "attention input norm ref",
        )?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Query,
        },
        work.layer.q_proj_shape.rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.attention_block.attention.q_proj", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerAttentionQuery(
                DecodeLayerWork::Active(work),
            ),
            state,
        },
    ))
}

#[tile]
fn init_project_decode_attention_key_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerAttentionKey(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(
            work.attention_normed_ref.clone(),
            "attention input norm ref",
        )?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Key,
        },
        work.layer.k_proj_shape.rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.attention_block.attention.k_proj", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerAttentionKey(DecodeLayerWork::Active(
                work,
            )),
            state,
        },
    ))
}

#[tile]
fn init_project_decode_attention_value_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerAttentionValue(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    if !work.layer.has_v_proj {
        if work.layer.attention_k_eq_v {
            work.v_projected_ref = Some(require_decode_activation_ref(
                work.k_projected_ref.clone(),
                "key projection ref",
            )?);
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerAttentionValue(
                        DecodeLayerWork::Active(work),
                    ),
                },
            ));
        }
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    }
    let projection_rows = work
        .layer
        .v_proj_shape
        .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing v_proj shape"))?
        .rows;
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(
            work.attention_normed_ref.clone(),
            "attention input norm ref",
        )?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Value,
        },
        projection_rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.attention_block.attention.v_proj", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerAttentionValue(
                DecodeLayerWork::Active(work),
            ),
            state,
        },
    ))
}

#[tile]
fn finalize_decode_layer_projection_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    continuation: DecodeProjectionContinuation,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    match continuation {
        DecodeProjectionContinuation::LayerAttentionQuery(work)
        | DecodeProjectionContinuation::LayerAttentionKey(work)
        | DecodeProjectionContinuation::LayerAttentionValue(work)
        | DecodeProjectionContinuation::LayerAttentionOutput(work)
        | DecodeProjectionContinuation::LayerMlpGate(work)
        | DecodeProjectionContinuation::LayerMlpUp(work)
        | DecodeProjectionContinuation::LayerMlpDown(work)
        | DecodeProjectionContinuation::LayerPleGate(work)
        | DecodeProjectionContinuation::LayerPleOutput(work) => Ok((artifact_store_roots, work)),
        _ => bail!("decode layer projection returned unexpected continuation"),
    }
}

#[tile]
fn reshape_decode_attention_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let row = read_activation_row_from_ref_roots(
        &artifact_store_roots,
        &require_decode_activation_ref(work.q_projected_ref.clone(), "query projection ref")?,
    )?;
    let heads = reshape_row_heads(row, work.layer.num_heads, work.layer.head_dim)?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.q_heads", work.output_prefix),
        heads,
    )?;
    work.q_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn reshape_decode_attention_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let row = read_activation_row_from_ref_roots(
        &artifact_store_roots,
        &require_decode_activation_ref(work.k_projected_ref.clone(), "key projection ref")?,
    )?;
    let heads = reshape_row_heads(row, work.layer.num_kv_heads, work.layer.head_dim)?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.k_heads", work.output_prefix),
        heads,
    )?;
    work.k_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn reshape_decode_attention_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let row = read_activation_row_from_ref_roots(
        &artifact_store_roots,
        &require_decode_activation_ref(work.v_projected_ref.clone(), "value projection ref")?,
    )?;
    let heads = reshape_row_heads(row, work.layer.num_kv_heads, work.layer.head_dim)?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.v_heads", work.output_prefix),
        heads,
    )?;
    work.v_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn normalize_decode_attention_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let norm_weights = auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaDecodeLayerNormKind::Query
        }
    )?;
    let heads = materialize_attention_heads_from_roots(
        &artifact_store_roots,
        &require_decode_heads_ref(work.q_heads_ref.clone(), "query heads ref")?,
    )?;
    let heads = rms_norm_heads(&heads, Some(&norm_weights), Some(scalars.rms_norm_eps))?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.q_norm", work.output_prefix),
        heads,
    )?;
    work.q_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn normalize_decode_attention_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let norm_weights = auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaDecodeLayerNormKind::Key
        }
    )?;
    let heads = materialize_attention_heads_from_roots(
        &artifact_store_roots,
        &require_decode_heads_ref(work.k_heads_ref.clone(), "key heads ref")?,
    )?;
    let heads = rms_norm_heads(&heads, Some(&norm_weights), Some(scalars.rms_norm_eps))?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.k_norm", work.output_prefix),
        heads,
    )?;
    work.k_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn normalize_decode_attention_value_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let heads = materialize_attention_heads_from_roots(
        &artifact_store_roots,
        &require_decode_heads_ref(work.v_heads_ref.clone(), "value heads ref")?,
    )?;
    let heads = value_rms_norm_heads(&heads, Some(scalars.rms_norm_eps))?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.v_norm", work.output_prefix),
        heads,
    )?;
    work.v_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn rope_decode_attention_query_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let heads = materialize_attention_heads_from_roots(
        &artifact_store_roots,
        &require_decode_heads_ref(work.q_heads_ref.clone(), "query heads ref")?,
    )?;
    let heads = apply_rope_to_heads(
        &heads,
        work.layer.partial_rotary_dim,
        work.layer.rope_freq_base_dim,
        scalars.rope_base,
        work.decode_state.position,
    )?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.q_rope", work.output_prefix),
        heads,
    )?;
    work.q_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn rope_decode_attention_key_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let heads = materialize_attention_heads_from_roots(
        &artifact_store_roots,
        &require_decode_heads_ref(work.k_heads_ref.clone(), "key heads ref")?,
    )?;
    let heads = apply_rope_to_heads(
        &heads,
        work.layer.partial_rotary_dim,
        work.layer.rope_freq_base_dim,
        scalars.rope_base,
        work.decode_state.position,
    )?;
    let (artifact_store_roots, output_ref) = insert_attention_heads_with_roots(
        &artifact_store_roots,
        format!("{}.attention_block.attention.k_rope", work.output_prefix),
        heads,
    )?;
    work.k_heads_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_update_decode_attention_cache_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeKvCacheAppendWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeKvCacheAppendWork::Skip(layer_work),
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    if work.donor_cache_slot.is_some() {
        work.updated_cache = Some(work.cache_slot.clone());
        return Ok((
            artifact_store_roots,
            DecodeKvCacheAppendWork::Skip(DecodeLayerWork::Active(work)),
        ));
    }
    let state = init_decode_kv_cache_append_artifact_work_state(
        artifact_store_roots.clone(),
        work.cache_slot.clone(),
        require_decode_heads_ref(work.k_heads_ref.clone(), "key heads ref")?,
        require_decode_heads_ref(work.v_heads_ref.clone(), "value heads ref")?,
        RasterTensorId::new(format!(
            "{}.attention_block.attention.updated_cache.{}.keys",
            work.output_prefix, work.layer_idx
        ))?,
        RasterTensorId::new(format!(
            "{}.attention_block.attention.updated_cache.{}.values",
            work.output_prefix, work.layer_idx
        ))?,
        work.layer.cache_sliding_window,
        work.decode_state.attention_kv_rows_per_tile,
    )?;
    Ok((
        state.0,
        DecodeKvCacheAppendWork::Active {
            work: DecodeLayerWork::Active(work),
            state: state.1,
        },
    ))
}

#[tile(kind = recursive)]
fn compute_next_decode_kv_cache_append_work_row_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    cache_work: DecodeKvCacheAppendWork,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeKvCacheAppendWork)> {
    match cache_work {
        DecodeKvCacheAppendWork::Skip(layer_work) => Ok((
            true,
            artifact_store_roots,
            DecodeKvCacheAppendWork::Skip(layer_work),
        )),
        DecodeKvCacheAppendWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_decode_kv_cache_append_artifact_work_row(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                DecodeKvCacheAppendWork::Active { work, state },
            ))
        }
    }
}

#[tile]
fn finalize_update_decode_attention_cache_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    cache_work: DecodeKvCacheAppendWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    match cache_work {
        DecodeKvCacheAppendWork::Skip(layer_work) => Ok((artifact_store_roots, layer_work)),
        DecodeKvCacheAppendWork::Active { work, state } => {
            let (artifact_store_roots, cache_slot) =
                finalize_decode_kv_cache_append_artifact_work_state(artifact_store_roots, state)?;
            let mut work = active_decode_layer_work(work)?;
            work.updated_cache = Some(cache_slot);
            Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
        }
    }
}

#[tile]
fn init_compute_decode_attention_scores_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeAttentionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((artifact_store_roots, DecodeAttentionWork::Skip(layer_work)));
        }
        DecodeLayerWork::Active(work) => work,
    };
    let attention_cache_slot = work
        .donor_cache_slot
        .as_ref()
        .or(work.updated_cache.as_ref())
        .ok_or_else(|| anyhow!("decode attention requires an updated cache slot"))?;
    let attention_cache_ref = match attention_cache_slot {
        DecodeLayerCacheSlot::Empty { .. } => bail!("decode attention cache is empty"),
        DecodeLayerCacheSlot::Ref(cache_ref) => cache_ref.clone(),
    };
    let attention_window = match work.layer.attention_kind {
        GemmaDecodeAttentionKind::Full => None,
        GemmaDecodeAttentionKind::Sliding => Some(
            work.layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?,
        ),
    };
    let (artifact_store_roots, state) = init_decode_attention_artifact_work_state(
        artifact_store_roots,
        require_decode_heads_ref(work.q_heads_ref.clone(), "query heads ref")?,
        attention_cache_ref,
        RasterTensorId::new(format!(
            "{}.attention_block.attention.scores.output",
            work.output_prefix
        ))?,
        format!("{}.attention_block.attention.scores", work.output_prefix),
        attention_window,
        work.decode_state.attention_kv_rows_per_tile,
    )?;
    Ok((
        artifact_store_roots,
        DecodeAttentionWork::Active {
            work: DecodeLayerWork::Active(work),
            state,
        },
    ))
}

#[tile(kind = recursive)]
fn compute_next_decode_attention_work_head_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    attention_work: DecodeAttentionWork,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeAttentionWork)> {
    match attention_work {
        DecodeAttentionWork::Skip(layer_work) => Ok((
            true,
            artifact_store_roots,
            DecodeAttentionWork::Skip(layer_work),
        )),
        DecodeAttentionWork::Active { work, state } => {
            let (done, artifact_store_roots, state) =
                compute_next_decode_attention_artifact_work_head(artifact_store_roots, state)?;
            Ok((
                done,
                artifact_store_roots,
                DecodeAttentionWork::Active { work, state },
            ))
        }
    }
}

#[tile]
fn finalize_compute_decode_attention_scores_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    attention_work: DecodeAttentionWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    match attention_work {
        DecodeAttentionWork::Skip(layer_work) => Ok((artifact_store_roots, layer_work)),
        DecodeAttentionWork::Active { work, state } => {
            let (artifact_store_roots, heads_ref) =
                finalize_decode_attention_artifact_work_state(artifact_store_roots, state)?;
            let mut work = active_decode_layer_work(work)?;
            work.attention_heads_ref = Some(heads_ref);
            Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
        }
    }
}

#[tile]
fn combine_decode_attention_heads_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let (mut artifact_store_roots, mut combine_state) =
        crate::shared::raster_kernels::transformer::init_combine_heads_artifact_state_from_ref(
            artifact_store_roots,
            require_decode_heads_ref(work.attention_heads_ref.clone(), "attention heads ref")?,
            RasterTensorId::new(format!(
                "{}.attention_block.attention.combined",
                work.output_prefix
            ))?,
        )?;
    loop {
        let (done, next_roots, next_state) =
            crate::shared::raster_kernels::transformer::compute_next_combine_heads_artifact_row(
                artifact_store_roots,
                combine_state,
            )?;
        artifact_store_roots = next_roots;
        combine_state = next_state;
        if done {
            break;
        }
    }
    let (artifact_store_roots, output_ref) =
        crate::shared::raster_kernels::transformer::finalize_combine_heads_artifact_state_ref(
            artifact_store_roots,
            combine_state,
        )?;
    work.attention_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_project_decode_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerAttentionOutput(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(work.attention_ref.clone(), "attention combined ref")?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Output,
        },
        work.layer.o_proj_shape.rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.attention_block.attention.o_proj", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerAttentionOutput(
                DecodeLayerWork::Active(work),
            ),
            state,
        },
    ))
}

#[tile]
fn normalize_decode_attention_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let norm_weights = auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaDecodeLayerNormKind::PostAttention
        }
    )?;
    let (artifact_store_roots, output_ref) = rms_norm_decode_ref(
        artifact_store_roots,
        require_decode_activation_ref(work.attention_output_ref.clone(), "attention output ref")?,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode layer RMSNorm",
        format!("{}.attention_block.post_attention_norm", work.output_prefix),
    )?;
    work.attention_output_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn add_decode_attention_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, output_ref) = add_decode_refs(
        artifact_store_roots,
        work.decode_state.current_activation_ref.clone(),
        require_decode_activation_ref(work.attention_output_ref.clone(), "attention output ref")?,
        format!("{}.attention_block.residual", work.output_prefix),
    )?;
    work.xs_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn normalize_decode_mlp_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let norm_weights = auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaDecodeLayerNormKind::PreFeedForward
        }
    )?;
    let (artifact_store_roots, output_ref) = rms_norm_decode_ref(
        artifact_store_roots,
        require_decode_activation_ref(work.xs_ref.clone(), "attention residual ref")?,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode MLP pre-feedforward RMSNorm",
        format!("{}.mlp.pre_ff_norm", work.output_prefix),
    )?;
    work.mlp_normed_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_project_decode_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerMlpGate(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(work.mlp_normed_ref.clone(), "MLP normed ref")?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Gate,
        },
        work.layer.gate_proj_shape.rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.mlp.gate_proj", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerMlpGate(DecodeLayerWork::Active(work)),
            state,
        },
    ))
}

#[tile]
fn gelu_decode_mlp_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, output_ref) = gelu_decode_ref(
        artifact_store_roots,
        require_decode_activation_ref(work.mlp_gate_ref.clone(), "MLP gate ref")?,
        format!("{}.mlp.gate_gelu", work.output_prefix),
    )?;
    work.mlp_gate_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_project_decode_mlp_up_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerMlpUp(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(work.mlp_normed_ref.clone(), "MLP normed ref")?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Up,
        },
        work.layer.up_proj_shape.rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.mlp.up_proj", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerMlpUp(DecodeLayerWork::Active(work)),
            state,
        },
    ))
}

#[tile]
fn multiply_decode_mlp_hidden_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, output_ref) = mul_decode_refs(
        artifact_store_roots,
        require_decode_activation_ref(work.mlp_gate_ref.clone(), "MLP gate ref")?,
        require_decode_activation_ref(work.mlp_up_ref.clone(), "MLP up ref")?,
        format!("{}.mlp.ff_hidden", work.output_prefix),
    )?;
    work.mlp_hidden_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_project_decode_mlp_down_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerMlpDown(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(work.mlp_hidden_ref.clone(), "MLP hidden ref")?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Down,
        },
        work.layer.down_proj_shape.rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.mlp.down_proj", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerMlpDown(DecodeLayerWork::Active(work)),
            state,
        },
    ))
}

#[tile]
fn normalize_decode_mlp_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let norm_weights = auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaDecodeLayerNormKind::PostFeedForward
        }
    )?;
    let (artifact_store_roots, output_ref) = rms_norm_decode_ref(
        artifact_store_roots,
        require_decode_activation_ref(work.mlp_out_ref.clone(), "MLP output ref")?,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode MLP post-feedforward RMSNorm",
        format!("{}.mlp.post_ff_norm", work.output_prefix),
    )?;
    work.mlp_out_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn add_decode_mlp_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let (artifact_store_roots, output_ref) = add_decode_refs(
        artifact_store_roots,
        require_decode_activation_ref(work.xs_ref.clone(), "attention residual ref")?,
        require_decode_activation_ref(work.mlp_out_ref.clone(), "MLP output ref")?,
        format!("{}.mlp.residual", work.output_prefix),
    )?;
    work.xs_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_project_decode_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerPleGate(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((
            artifact_store_roots,
            DecodeProjectionWork::Skip {
                continuation: DecodeProjectionContinuation::LayerPleGate(DecodeLayerWork::Active(
                    work,
                )),
            },
        ));
    }
    let projection_rows = work
        .layer
        .ple_input_gate_shape
        .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing PLE input gate shape"))?
        .rows;
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(work.xs_ref.clone(), "MLP residual ref")?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::PleInputGate,
        },
        projection_rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.ple.input_gate", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerPleGate(DecodeLayerWork::Active(work)),
            state,
        },
    ))
}

#[tile]
fn gelu_decode_ple_gate_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((artifact_store_roots, DecodeLayerWork::Active(work)));
    }
    let (artifact_store_roots, output_ref) = gelu_decode_ref(
        artifact_store_roots,
        require_decode_activation_ref(work.ple_gate_ref.clone(), "PLE gate ref")?,
        format!("{}.ple.input_gate_gelu", work.output_prefix),
    )?;
    work.ple_gate_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn multiply_decode_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let Some(per_layer_input_ref) = work.per_layer_input_ref.clone() else {
        return Ok((artifact_store_roots, DecodeLayerWork::Active(work)));
    };
    let (artifact_store_roots, output_ref) = mul_decode_refs(
        artifact_store_roots,
        require_decode_activation_ref(work.ple_gate_ref.clone(), "PLE gate ref")?,
        per_layer_input_ref,
        format!("{}.ple.gated", work.output_prefix),
    )?;
    work.ple_gate_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn init_project_decode_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((
                artifact_store_roots,
                DecodeProjectionWork::Skip {
                    continuation: DecodeProjectionContinuation::LayerPleOutput(layer_work),
                },
            ));
        }
        DecodeLayerWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((
            artifact_store_roots,
            DecodeProjectionWork::Skip {
                continuation: DecodeProjectionContinuation::LayerPleOutput(
                    DecodeLayerWork::Active(work),
                ),
            },
        ));
    }
    let projection_rows = work
        .layer
        .ple_layer_projection_shape
        .ok_or_else(|| {
            anyhow!("Gemma decode layer metadata is missing PLE layer projection shape")
        })?
        .rows;
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        require_decode_activation_ref(work.ple_gate_ref.clone(), "PLE gated input ref")?,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: work.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::PleLayerProjection,
        },
        projection_rows,
        work.decode_state.projection_rows_per_tile,
        format!("{}.ple.layer_projection", work.output_prefix),
        None,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::LayerPleOutput(DecodeLayerWork::Active(
                work,
            )),
            state,
        },
    ))
}

#[tile]
fn normalize_decode_ple_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((artifact_store_roots, DecodeLayerWork::Active(work)));
    }
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let norm_weights = auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest {
            layer_idx: work.layer_idx,
            norm: GemmaDecodeLayerNormKind::PlePostInput
        }
    )?;
    let (artifact_store_roots, output_ref) = rms_norm_decode_ref(
        artifact_store_roots,
        require_decode_activation_ref(work.ple_projected_ref.clone(), "PLE projected ref")?,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode PLE post-input RMSNorm",
        format!("{}.ple.post_input_norm", work.output_prefix),
    )?;
    work.ple_projected_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn add_decode_ple_residual_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    if work.per_layer_input_ref.is_none() {
        return Ok((artifact_store_roots, DecodeLayerWork::Active(work)));
    }
    let (artifact_store_roots, output_ref) = add_decode_refs(
        artifact_store_roots,
        require_decode_activation_ref(work.xs_ref.clone(), "MLP residual ref")?,
        require_decode_activation_ref(work.ple_projected_ref.clone(), "PLE projected ref")?,
        format!("{}.ple.residual", work.output_prefix),
    )?;
    work.xs_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn scale_decode_layer_output_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    let mut work = match layer_work {
        DecodeLayerWork::Complete(_) => return Ok((artifact_store_roots, layer_work)),
        DecodeLayerWork::Active(work) => work,
    };
    let input_ref = require_decode_activation_ref(work.xs_ref.clone(), "layer output ref")?;
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: work.layer_idx
        }
    )?;
    let (artifact_store_roots, output_ref) = scale_decode_ref(
        artifact_store_roots,
        input_ref,
        scalars.layer_scalar,
        format!("{}.scaled", work.output_prefix),
    )?;
    work.layer_output_ref = Some(output_ref);
    Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
}

#[tile]
fn read_decode_selected_token(
    artifact_store_roots: &RasterArtifactStoreRoots,
    selected_token_ref: &RasterSelectedTokenRef,
) -> Result<u32> {
    read_selected_token_from_roots(artifact_store_roots, selected_token_ref)
}

#[tile]
fn init_decode_transition_state_refs_from_input_roots(
    input_roots: RasterDecodeTransitionInputRoots,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    if input_roots.decode_transition_source_root != source.root() {
        bail!(
            "raster decode transition source root {} does not match input source root {}",
            source.root(),
            input_roots.decode_transition_source_root
        );
    }
    let next_token = read_selected_token_from_roots(
        &input_roots.artifact_store_roots,
        &input_roots.selected_token_ref,
    )?;
    validate_projection_rows_per_tile(input_roots.raster_sizing.projection_rows_per_tile)?;
    validate_attention_kv_rows_per_tile(input_roots.raster_sizing.attention_kv_rows_per_tile)?;
    let metadata = auth_read!(source, GemmaDecodeTransitionMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer decode requires at least one layer");
    }
    if input_roots.transformer_decode_state.layer_caches.len() != metadata.layer_count {
        bail!(
            "transformer decode cache count mismatch: {} vs {}",
            input_roots.transformer_decode_state.layer_caches.len(),
            metadata.layer_count
        );
    }

    let embedded = RasterActivationRow::from_acts(auth_read!(
        source,
        GemmaDecodeEmbeddingRowRequest {
            token_id: next_token
        },
    )?);
    if embedded.width() != metadata.embedding_width {
        bail!(
            "decode embedded token width {}, expected {}",
            embedded.width(),
            metadata.embedding_width
        );
    }
    let mut artifact_store_roots = input_roots.artifact_store_roots;
    let (roots, decode_input_ref) = insert_decode_activation_row_with_roots(
        &artifact_store_roots,
        format!(
            "{}.input.selected_token_embedding",
            input_roots.output_source_prefix
        ),
        &embedded,
    )?;
    artifact_store_roots = roots;
    let mut original_layer_caches =
        Vec::with_capacity(input_roots.transformer_decode_state.layer_caches.len());
    for (layer_idx, cache) in input_roots
        .transformer_decode_state
        .layer_caches
        .iter()
        .enumerate()
    {
        let cache = raster_cache_from_layer_cache(cache)?;
        let (roots, cache_slot) = register_decode_layer_cache_with_roots(
            &artifact_store_roots,
            &format!("{}.original.cache", input_roots.output_source_prefix),
            layer_idx,
            cache,
        )?;
        artifact_store_roots = roots;
        original_layer_caches.push(cache_slot);
    }

    Ok((
        artifact_store_roots.clone(),
        DecodeTransitionRasterState {
            artifact_store_roots,
            decode_input_ref: decode_input_ref.clone(),
            current_activation_ref: decode_input_ref,
            next_token,
            position: input_roots.transformer_decode_state.position,
            token_count: input_roots.transformer_decode_state.token_count,
            next_layer_idx: 0,
            layer_count: metadata.layer_count,
            original_layer_caches,
            updated_layer_caches: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
            projection_rows_per_tile: input_roots.raster_sizing.projection_rows_per_tile,
            attention_kv_rows_per_tile: input_roots.raster_sizing.attention_kv_rows_per_tile,
            output_source_prefix: input_roots.output_source_prefix,
        },
    ))
}

#[tile]
fn init_decode_transition_state_from_input_refs(
    input_roots: RasterDecodeTransitionInputRefs,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    if input_roots.decode_transition_source_root != source.root() {
        bail!(
            "raster decode transition source root {} does not match input source root {}",
            source.root(),
            input_roots.decode_transition_source_root
        );
    }
    let next_token = read_selected_token_from_roots(
        &input_roots.artifact_store_roots,
        &input_roots.selected_token_ref,
    )?;
    validate_projection_rows_per_tile(input_roots.raster_sizing.projection_rows_per_tile)?;
    validate_attention_kv_rows_per_tile(input_roots.raster_sizing.attention_kv_rows_per_tile)?;
    let metadata = auth_read!(source, GemmaDecodeTransitionMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer decode requires at least one layer");
    }
    if input_roots.layer_caches.len() != metadata.layer_count {
        bail!(
            "transformer decode cache count mismatch: {} vs {}",
            input_roots.layer_caches.len(),
            metadata.layer_count
        );
    }

    let embedded = RasterActivationRow::from_acts(auth_read!(
        source,
        GemmaDecodeEmbeddingRowRequest {
            token_id: next_token
        },
    )?);
    if embedded.width() != metadata.embedding_width {
        bail!(
            "decode embedded token width {}, expected {}",
            embedded.width(),
            metadata.embedding_width
        );
    }
    let (artifact_store_roots, decode_input_ref) = insert_decode_activation_row_with_roots(
        &input_roots.artifact_store_roots,
        format!(
            "{}.input.selected_token_embedding",
            input_roots.output_source_prefix
        ),
        &embedded,
    )?;

    Ok((
        artifact_store_roots.clone(),
        DecodeTransitionRasterState {
            artifact_store_roots,
            decode_input_ref: decode_input_ref.clone(),
            current_activation_ref: decode_input_ref,
            next_token,
            position: input_roots.position,
            token_count: input_roots.token_count,
            next_layer_idx: 0,
            layer_count: metadata.layer_count,
            original_layer_caches: input_roots.layer_caches,
            updated_layer_caches: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
            projection_rows_per_tile: input_roots.raster_sizing.projection_rows_per_tile,
            attention_kv_rows_per_tile: input_roots.raster_sizing.attention_kv_rows_per_tile,
            output_source_prefix: input_roots.output_source_prefix,
        },
    ))
}

#[tile]
fn init_decode_transition_run_input_roots(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &CommittedExternalSource,
    raster_sizing: RasterSizingControls,
) -> Result<RasterDecodeTransitionInputRoots> {
    ArtifactIo::reset_store();
    let roots = ArtifactIo::export_store_roots();
    let output_source_prefix = format!(
        "decode.transition.position_{}.token_count_{}",
        transformer_decode_state.position, transformer_decode_state.token_count
    );
    let (artifact_store_roots, selected_token_ref) = insert_decode_selected_token_with_roots(
        &roots,
        format!("{output_source_prefix}.input.selected_token"),
        next_token,
    )?;
    Ok(RasterDecodeTransitionInputRoots {
        artifact_store_roots,
        transformer_decode_state,
        selected_token_ref,
        decode_transition_source_root: source.root().to_string(),
        output_source_prefix,
        raster_sizing,
    })
}

#[tile]
fn finalize_decode_transition_run_output(
    output: RasterDecodeTransitionOutputRefs,
) -> Result<TransformerDecodeStepResult> {
    Ok(output.transition_result)
}

#[tile]
pub fn init_decode_transition_state_refs_with_roots(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &CommittedExternalSource,
    raster_sizing: RasterSizingControls,
    output_source_prefix: String,
) -> Result<(RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    validate_attention_kv_rows_per_tile(raster_sizing.attention_kv_rows_per_tile)?;
    let metadata = auth_read!(source, GemmaDecodeTransitionMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer decode requires at least one layer");
    }
    if transformer_decode_state.layer_caches.len() != metadata.layer_count {
        bail!(
            "transformer decode cache count mismatch: {} vs {}",
            transformer_decode_state.layer_caches.len(),
            metadata.layer_count
        );
    }

    let embedded = RasterActivationRow::from_acts(auth_read!(
        source,
        GemmaDecodeEmbeddingRowRequest {
            token_id: next_token
        },
    )?);
    if embedded.width() != metadata.embedding_width {
        bail!(
            "decode embedded token width {}, expected {}",
            embedded.width(),
            metadata.embedding_width
        );
    }
    let (roots, decode_input_ref) = insert_decode_activation_row_with_roots(
        &artifact_store_roots,
        format!("{output_source_prefix}.input.selected_token_embedding"),
        &embedded,
    )?;
    artifact_store_roots = roots;
    let mut original_layer_caches = Vec::with_capacity(transformer_decode_state.layer_caches.len());
    for (layer_idx, cache) in transformer_decode_state.layer_caches.iter().enumerate() {
        let cache = raster_cache_from_layer_cache(cache)?;
        let (roots, cache_slot) = register_decode_layer_cache_with_roots(
            &artifact_store_roots,
            &format!("{output_source_prefix}.original.cache"),
            layer_idx,
            cache,
        )?;
        artifact_store_roots = roots;
        original_layer_caches.push(cache_slot);
    }

    Ok((
        artifact_store_roots.clone(),
        DecodeTransitionRasterState {
            artifact_store_roots,
            decode_input_ref: decode_input_ref.clone(),
            current_activation_ref: decode_input_ref,
            next_token,
            position: transformer_decode_state.position,
            token_count: transformer_decode_state.token_count,
            next_layer_idx: 0,
            layer_count: metadata.layer_count,
            original_layer_caches,
            updated_layer_caches: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
            projection_rows_per_tile: raster_sizing.projection_rows_per_tile,
            attention_kv_rows_per_tile: raster_sizing.attention_kv_rows_per_tile,
            output_source_prefix,
        },
    ))
}

#[tile]
fn normalize_decode_final_position_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    final_hidden_states_ref: RasterActivationSequenceRef,
    source: &CommittedExternalSource,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let final_hidden_state =
        read_activation_row_from_ref_roots(&artifact_store_roots, &final_hidden_states_ref)?;
    let norm_weights = auth_read!(source, GemmaDecodeFinalNormWeightsRequest)?;
    let scalars = auth_read!(source, GemmaDecodeFinalScalarsRequest)?;
    let normalized = first_row(
        rms_norm_sequence(
            &RasterActivationSequence::from_rows(vec![final_hidden_state]),
            Some(&norm_weights),
            Some(scalars.rms_norm_eps),
        )?,
        "deterministic decode final RMSNorm",
    )?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &normalized)
}

#[tile]
fn init_decode_transition_final_work_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    decode_state: DecodeTransitionRasterState,
) -> Result<DecodeTransitionFinalWork> {
    if decode_state.next_layer_idx != decode_state.layer_count {
        bail!(
            "raster decode finalized after {} layers, expected {}",
            decode_state.next_layer_idx,
            decode_state.layer_count
        );
    }
    if decode_state.updated_layer_caches.len() != decode_state.layer_count {
        bail!(
            "raster decode stored {} layer caches, expected {}",
            decode_state.updated_layer_caches.len(),
            decode_state.layer_count
        );
    }
    Ok(DecodeTransitionFinalWork {
        artifact_store_roots,
        final_hidden_state_ref: decode_state.current_activation_ref,
        normalized_ref: None,
        logits_ref: None,
        output_source_prefix: decode_state.output_source_prefix,
        projection_rows_per_tile: decode_state.projection_rows_per_tile,
        layer_caches: decode_state.updated_layer_caches,
        position: decode_state.position,
        token_count: decode_state.token_count,
    })
}

#[tile]
fn normalize_decode_final_work_with_roots(
    mut final_work: DecodeTransitionFinalWork,
    source: &CommittedExternalSource,
) -> Result<DecodeTransitionFinalWork> {
    let final_hidden_state = read_activation_row_from_ref_roots(
        &final_work.artifact_store_roots,
        &final_work.final_hidden_state_ref,
    )?;
    let norm_weights = auth_read!(source, GemmaDecodeFinalNormWeightsRequest)?;
    let scalars = auth_read!(source, GemmaDecodeFinalScalarsRequest)?;
    let normalized = first_row(
        rms_norm_sequence(
            &RasterActivationSequence::from_rows(vec![final_hidden_state]),
            Some(&norm_weights),
            Some(scalars.rms_norm_eps),
        )?,
        "deterministic decode final RMSNorm",
    )?;
    let (artifact_store_roots, normalized_ref) = insert_decode_activation_row_with_roots(
        &final_work.artifact_store_roots,
        format!("{}.final_norm", final_work.output_source_prefix),
        &normalized,
    )?;
    final_work.artifact_store_roots = artifact_store_roots;
    final_work.normalized_ref = Some(normalized_ref);
    Ok(final_work)
}

#[tile]
fn init_decode_final_logits_projection_work(
    final_work: DecodeTransitionFinalWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    let normalized_ref = final_work
        .normalized_ref
        .clone()
        .ok_or_else(|| anyhow!("decode final logits projection requires normalized ref"))?;
    let projection_rows = auth_read!(source, GemmaDecodeTransitionMetadataRequest)?.projection_rows;
    let softcap_bits = auth_read!(source, GemmaDecodeFinalScalarsRequest)?
        .final_logit_softcapping
        .map(Act::to_bits);
    let output_source_name = format!("{}.final_logits", final_work.output_source_prefix);
    let projection_rows_per_tile = final_work.projection_rows_per_tile;
    let artifact_store_roots = final_work.artifact_store_roots.clone();
    let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
        artifact_store_roots,
        normalized_ref,
        DecodeProjectionKind::FinalLogits,
        projection_rows,
        projection_rows_per_tile,
        output_source_name,
        softcap_bits,
    )?;
    Ok((
        artifact_store_roots,
        DecodeProjectionWork::Active {
            continuation: DecodeProjectionContinuation::FinalLogits(final_work),
            state,
        },
    ))
}

#[tile]
fn finalize_decode_final_logits_projection_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    continuation: DecodeProjectionContinuation,
) -> Result<DecodeTransitionFinalWork> {
    match continuation {
        DecodeProjectionContinuation::FinalLogits(mut final_work) => {
            final_work.artifact_store_roots = artifact_store_roots;
            if final_work.logits_ref.is_none() {
                bail!("decode final logits projection did not produce logits ref");
            }
            Ok(final_work)
        }
        _ => bail!("decode final logits projection returned unexpected continuation"),
    }
}

#[tile]
fn finalize_decode_transition_output_refs(
    final_work: DecodeTransitionFinalWork,
) -> Result<RasterDecodeTransitionOutputRefs> {
    let current_activation = read_activation_row_from_ref_roots(
        &final_work.artifact_store_roots,
        &final_work.final_hidden_state_ref,
    )?;
    let det_row = current_activation.acts();
    let values = vec![current_activation.to_f32_values()];
    let internal = InternalActivationSequence::from_det_values(vec![det_row.clone()]);
    let mut activation_state = ActivationSequence::from_internal(
        internal,
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&values),
    );
    activation_state.det_activations_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_activation_commitment(&[det_row]),
    );
    let logits_ref = final_work
        .logits_ref
        .ok_or_else(|| anyhow!("decode transition output requires logits ref"))?;
    let layer_caches = final_work
        .layer_caches
        .iter()
        .map(|cache| {
            materialize_decode_layer_cache_from_roots(&final_work.artifact_store_roots, cache)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(layer_cache_from_raster)
        .collect();
    let transition_result = finalize_decode_transition_result_values_from_roots(
        &final_work.artifact_store_roots,
        logits_ref,
        TransformerDecodeState {
            layer_caches,
            position: final_work.position,
            token_count: final_work.token_count,
        },
        activation_state,
    )?;
    Ok(RasterDecodeTransitionOutputRefs {
        artifact_store_roots: final_work.artifact_store_roots,
        transition_result,
    })
}

#[tile]
fn finalize_decode_transition_output_state_refs(
    final_work: DecodeTransitionFinalWork,
) -> Result<RasterDecodeTransitionOutputStateRefs> {
    let logits_ref = final_work
        .logits_ref
        .ok_or_else(|| anyhow!("decode transition state output requires logits ref"))?;
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    let logit_count = match (row_count, width) {
        (rows, 1) => rows,
        (1, cols) => cols,
        _ => bail!("raster decode transition logits shape {row_count}x{width} must be Nx1 or 1xN"),
    };
    Ok(RasterDecodeTransitionOutputStateRefs {
        artifact_store_roots: final_work.artifact_store_roots,
        final_hidden_state_ref: final_work.final_hidden_state_ref,
        logits_ref,
        logit_count,
        layer_caches: final_work.layer_caches,
        position: final_work.position + 1,
        token_count: final_work.token_count + 1,
    })
}

#[tile]
fn decode_projection_row_count(source: &CommittedExternalSource) -> Result<usize> {
    Ok(auth_read!(source, GemmaDecodeTransitionMetadataRequest)?.projection_rows)
}

#[tile]
fn decode_final_logit_softcap_bits(source: &CommittedExternalSource) -> Result<Option<i32>> {
    Ok(auth_read!(source, GemmaDecodeFinalScalarsRequest)?
        .final_logit_softcapping
        .map(Act::to_bits))
}

#[tile]
pub fn finalize_decode_layer_state_with_roots(
    roots: &RasterArtifactStoreRoots,
    decode_state: DecodeTransitionRasterState,
) -> Result<ActivationSequenceWithCache> {
    if decode_state.next_layer_idx != decode_state.layer_count {
        bail!(
            "raster decode finalized after {} layers, expected {}",
            decode_state.next_layer_idx,
            decode_state.layer_count
        );
    }
    if decode_state.updated_layer_caches.len() != decode_state.layer_count {
        bail!(
            "raster decode stored {} layer caches, expected {}",
            decode_state.updated_layer_caches.len(),
            decode_state.layer_count
        );
    }

    let current_activation =
        read_activation_row_from_ref_roots(roots, &decode_state.current_activation_ref)?;
    let det_row = current_activation.acts();
    let values = vec![current_activation.to_f32_values()];
    let internal = InternalActivationSequence::from_det_values(vec![det_row.clone()]);
    let mut activation_state = ActivationSequence::from_internal(
        internal,
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&values),
    );
    activation_state.det_activations_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_activation_commitment(&[det_row]),
    );

    Ok(ActivationSequenceWithCache {
        activation_state,
        layer_caches: decode_state
            .updated_layer_caches
            .iter()
            .map(|cache| materialize_decode_layer_cache_from_roots(roots, cache))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(layer_cache_from_raster)
            .collect(),
    })
}

#[tile]
fn finalize_decode_transition_result_from_roots(
    artifact_store_roots: &RasterArtifactStoreRoots,
    logits_ref: RasterActivationSequenceRef,
    transformer_decode_state: TransformerDecodeState,
    final_hidden_state: ActivationSequence,
) -> Result<TransformerDecodeStepResult> {
    finalize_decode_transition_result_values_from_roots(
        artifact_store_roots,
        logits_ref,
        transformer_decode_state,
        final_hidden_state,
    )
}

#[tile]
pub(in super::super) fn prepare_next_decode_layer_context(
    decode_state: &DecodeTransitionRasterState,
    source: &CommittedExternalSource,
) -> Result<DecodeLayerContext> {
    let layer_idx = decode_state.next_layer_idx;
    let layer = auth_read!(source, GemmaDecodeLayerMetadataRequest { layer_idx })?;
    let cache_slot = decode_state
        .original_layer_caches
        .get(layer_idx)
        .cloned()
        .ok_or_else(|| anyhow!("transformer decode cache {layer_idx} missing"))?;
    let donor_cache_slot =
        resolve_decode_donor_cache_slot(&decode_state.updated_layer_caches, layer_idx, &layer)?
            .cloned();
    Ok(DecodeLayerContext {
        layer_idx,
        layer,
        cache_slot,
        donor_cache_slot,
    })
}

#[tile]
fn init_next_decode_layer_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    decode_state: DecodeTransitionRasterState,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    if decode_state.next_layer_idx >= decode_state.layer_count {
        return Ok((
            artifact_store_roots,
            DecodeLayerWork::Complete(decode_state),
        ));
    }
    let layer_idx = decode_state.next_layer_idx;
    let layer = auth_read!(source, GemmaDecodeLayerMetadataRequest { layer_idx })?;
    let cache_slot = decode_state
        .original_layer_caches
        .get(layer_idx)
        .cloned()
        .ok_or_else(|| anyhow!("transformer decode cache {layer_idx} missing"))?;
    let donor_cache_slot =
        resolve_decode_donor_cache_slot(&decode_state.updated_layer_caches, layer_idx, &layer)?
            .cloned();
    let _trace = crate::trace::trace_scope(format!(
        "decode.layer.det layer={} token={} position={} attention={:?} ple={} donor={:?}",
        layer_idx,
        decode_state.next_token,
        decode_state.position,
        layer.attention_kind,
        layer.has_ple,
        layer.kv_shared_layer_index,
    ));
    Ok((
        artifact_store_roots,
        DecodeLayerWork::Active(DecodeActiveLayerWork {
            output_prefix: format!("{}.layer_{}", decode_state.output_source_prefix, layer_idx),
            decode_state,
            layer_idx,
            layer,
            cache_slot,
            donor_cache_slot,
            per_layer_input_ref: None,
            attention_normed_ref: None,
            q_projected_ref: None,
            k_projected_ref: None,
            v_projected_ref: None,
            q_heads_ref: None,
            k_heads_ref: None,
            v_heads_ref: None,
            attention_heads_ref: None,
            attention_ref: None,
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
            updated_cache: None,
        }),
    ))
}

#[tile]
fn finalize_next_decode_layer_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    match layer_work {
        DecodeLayerWork::Complete(decode_state) => Ok((true, artifact_store_roots, decode_state)),
        DecodeLayerWork::Active(work) => {
            let layer_output_ref = work
                .layer_output_ref
                .ok_or_else(|| anyhow!("decode layer work finalized without layer output ref"))?;
            let updated_cache = work
                .updated_cache
                .ok_or_else(|| anyhow!("decode layer work finalized without updated cache"))?;
            update_decode_layer_state_refs(
                artifact_store_roots,
                work.decode_state,
                work.layer_idx,
                layer_output_ref,
                updated_cache,
            )
        }
    }
}

#[tile]
fn update_decode_layer_state_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    decode_state: DecodeTransitionRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    updated_cache: DecodeLayerCacheSlot,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    update_decode_layer_state_refs(
        artifact_store_roots,
        decode_state,
        layer_idx,
        layer_output_ref,
        updated_cache,
    )
}

#[tile]
fn decode_ple_input_gate_rows(layer: &GemmaDecodeLayerMetadata) -> Result<usize> {
    Ok(layer
        .ple_input_gate_shape
        .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing PLE input gate shape"))?
        .rows)
}

#[tile]
fn init_decode_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_work: DecodeLayerWork,
    source: &CommittedExternalSource,
) -> Result<(RasterArtifactStoreRoots, DecodePleInputWork)> {
    let work = match layer_work {
        DecodeLayerWork::Complete(_) => {
            return Ok((artifact_store_roots, DecodePleInputWork::Skip(layer_work)));
        }
        DecodeLayerWork::Active(work) => work,
    };
    if !work.layer.has_ple {
        return Ok((
            artifact_store_roots,
            DecodePleInputWork::Skip(DecodeLayerWork::Active(work)),
        ));
    }
    let scalars = auth_read!(source, GemmaDecodePleScalarsRequest)?;
    let norm_weights = auth_read!(source, GemmaDecodePleProjectionNormWeightsRequest)?;
    let embedded = RasterActivationRow::from_acts(auth_read!(
        source,
        GemmaDecodePleTokenEmbeddingRowRequest {
            layer_idx: work.layer.layer_idx,
            token_id: work.decode_state.next_token,
        },
    )?);
    let embedded = scale_row(&embedded, Some(scalars.embedding_scale))?;
    let output_prefix = format!("{}.ple.input", work.output_prefix);
    let (artifact_store_roots, embedded_ref) = insert_decode_activation_row_with_roots(
        &artifact_store_roots,
        format!("{output_prefix}.token_embedding"),
        &embedded,
    )?;
    Ok((
        artifact_store_roots,
        DecodePleInputWork::Active {
            work,
            scalars,
            norm_weights,
            embedded_ref,
            projected_ref: None,
            output_prefix,
        },
    ))
}

#[tile]
fn init_project_decode_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    ple_work: DecodePleInputWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionWork)> {
    match ple_work {
        DecodePleInputWork::Skip(layer_work) => Ok((
            artifact_store_roots,
            DecodeProjectionWork::Skip {
                continuation: DecodeProjectionContinuation::PleInput(DecodePleInputWork::Skip(
                    layer_work,
                )),
            },
        )),
        DecodePleInputWork::Active {
            work,
            scalars,
            norm_weights,
            embedded_ref,
            projected_ref,
            output_prefix,
        } => {
            let projection_rows = work
                .layer
                .ple_input_gate_shape
                .ok_or_else(|| {
                    anyhow!("Gemma decode layer metadata is missing PLE input gate shape")
                })?
                .rows;
            let (artifact_store_roots, state) = init_decode_row_projection_artifact_state(
                artifact_store_roots,
                work.decode_state.decode_input_ref.clone(),
                DecodeProjectionKind::PleModel {
                    layer_idx: work.layer.layer_idx,
                },
                projection_rows,
                work.decode_state.projection_rows_per_tile,
                format!("{output_prefix}.model_projection"),
                None,
            )?;
            Ok((
                artifact_store_roots,
                DecodeProjectionWork::Active {
                    continuation: DecodeProjectionContinuation::PleInput(
                        DecodePleInputWork::Active {
                            work,
                            scalars,
                            norm_weights,
                            embedded_ref,
                            projected_ref,
                            output_prefix,
                        },
                    ),
                    state,
                },
            ))
        }
    }
}

#[tile]
fn finalize_project_decode_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    continuation: DecodeProjectionContinuation,
) -> Result<(RasterArtifactStoreRoots, DecodePleInputWork)> {
    match continuation {
        DecodeProjectionContinuation::PleInput(ple_work) => Ok((artifact_store_roots, ple_work)),
        _ => bail!("decode PLE input projection returned unexpected continuation"),
    }
}

#[tile]
fn finalize_decode_ple_input_work(
    artifact_store_roots: RasterArtifactStoreRoots,
    ple_work: DecodePleInputWork,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerWork)> {
    match ple_work {
        DecodePleInputWork::Skip(layer_work) => Ok((artifact_store_roots, layer_work)),
        DecodePleInputWork::Active {
            mut work,
            scalars,
            norm_weights,
            embedded_ref,
            projected_ref,
            output_prefix,
        } => {
            let projected_ref = projected_ref
                .ok_or_else(|| anyhow!("decode PLE input projection did not produce ref"))?;
            let (artifact_store_roots, projected_ref) = scale_decode_ref(
                artifact_store_roots,
                projected_ref,
                Some(scalars.projection_scalar),
                format!("{output_prefix}.projection_scaled"),
            )?;
            let (artifact_store_roots, projected_ref) = rms_norm_decode_ref(
                artifact_store_roots,
                projected_ref,
                &norm_weights,
                scalars.rms_norm_eps,
                "decode PLE input RMSNorm",
                format!("{output_prefix}.projection_norm"),
            )?;
            let (artifact_store_roots, combined_ref) = add_decode_refs(
                artifact_store_roots,
                embedded_ref,
                projected_ref,
                format!("{output_prefix}.combined"),
            )?;
            let (artifact_store_roots, input_ref) = scale_decode_ref(
                artifact_store_roots,
                combined_ref,
                Some(scalars.input_scale),
                format!("{output_prefix}.scaled"),
            )?;
            work.per_layer_input_ref = Some(input_ref);
            Ok((artifact_store_roots, DecodeLayerWork::Active(work)))
        }
    }
}

#[tile]
fn read_decode_ple_scalars(
    source: &CommittedExternalSource,
) -> Result<crate::decode_transition::raster::auth_source::GemmaDecodePleScalars> {
    auth_read!(source, GemmaDecodePleScalarsRequest)
}

#[tile]
fn read_decode_ple_projection_norm_weights(
    source: &CommittedExternalSource,
) -> Result<Vec<crate::shared::numerics::det_num::Wgt>> {
    auth_read!(source, GemmaDecodePleProjectionNormWeightsRequest)
}

#[tile]
fn read_decode_ple_token_embedding(
    source: &CommittedExternalSource,
    layer_idx: usize,
    token_id: u32,
) -> Result<RasterActivationRow> {
    Ok(RasterActivationRow::from_acts(auth_read!(
        source,
        GemmaDecodePleTokenEmbeddingRowRequest {
            layer_idx,
            token_id,
        },
    )?))
}

#[tile]
fn scale_decode_row_optional(
    row: &RasterActivationRow,
    scalar: Option<Act>,
) -> Result<RasterActivationRow> {
    if scalar.is_none() {
        return Ok(row.clone());
    }
    scale_row(row, scalar)
}

#[tile]
fn scale_decode_ref_optional_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    scalar: Option<Act>,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    scale_decode_ref(artifact_store_roots, input_ref, scalar, output_source_name)
}

#[tile]
fn rms_norm_decode_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    norm_weights: &[crate::shared::numerics::det_num::Wgt],
    eps: crate::shared::numerics::det_num::Acc,
    label: &str,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    rms_norm_decode_ref(
        artifact_store_roots,
        input_ref,
        norm_weights,
        eps,
        label,
        output_source_name,
    )
}

#[tile]
fn add_decode_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    add_decode_refs(artifact_store_roots, lhs_ref, rhs_ref, output_source_name)
}

#[tile]
pub fn init_decode_row_projection_artifact(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    projection_kind: DecodeProjectionKind,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: String,
    softcap_bits: Option<i32>,
) -> Result<(RasterArtifactStoreRoots, DecodeRowProjectionArtifactState)> {
    init_decode_row_projection_artifact_state(
        artifact_store_roots,
        input_ref,
        projection_kind,
        projection_rows,
        rows_per_tile,
        output_id,
        softcap_bits,
    )
}

#[tile(kind = recursive)]
pub fn project_next_decode_projection_chunk_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_state: DecodeRowProjectionArtifactState,
    source: &CommittedExternalSource,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    DecodeRowProjectionArtifactState,
)> {
    project_next_decode_projection_artifact_chunk(artifact_store_roots, projection_state, source)
}

#[tile]
pub fn finalize_decode_row_projection_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_state: DecodeRowProjectionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_decode_row_projection_artifact_state_ref(artifact_store_roots, projection_state)
}

#[tile(kind = recursive)]
fn project_next_decode_projection_work_chunk_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: DecodeProjectionWork,
    source: &CommittedExternalSource,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeProjectionWork)> {
    match projection_work {
        DecodeProjectionWork::Skip { continuation } => Ok((
            true,
            artifact_store_roots,
            DecodeProjectionWork::Skip { continuation },
        )),
        DecodeProjectionWork::Active {
            continuation,
            state,
        } => {
            let (done, artifact_store_roots, state) =
                project_next_decode_projection_artifact_chunk(artifact_store_roots, state, source)?;
            Ok((
                done,
                artifact_store_roots,
                DecodeProjectionWork::Active {
                    continuation,
                    state,
                },
            ))
        }
    }
}

#[tile]
fn finalize_decode_projection_work_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_work: DecodeProjectionWork,
) -> Result<(RasterArtifactStoreRoots, DecodeProjectionContinuation)> {
    match projection_work {
        DecodeProjectionWork::Skip { continuation } => Ok((artifact_store_roots, continuation)),
        DecodeProjectionWork::Active {
            continuation,
            state,
        } => {
            let (artifact_store_roots, output_ref) =
                finalize_decode_row_projection_artifact_state_ref(artifact_store_roots, state)?;
            Ok((
                artifact_store_roots,
                attach_decode_projection_output(continuation, output_ref),
            ))
        }
    }
}

#[tile]
fn read_decode_layer_scalars(
    source: &CommittedExternalSource,
    layer_idx: usize,
) -> Result<crate::decode_transition::raster::auth_source::GemmaDecodeLayerScalars> {
    auth_read!(source, GemmaDecodeLayerScalarsRequest { layer_idx })
}

#[tile]
fn validate_decode_attention_context(
    input: &RasterActivationRow,
    layer: &GemmaDecodeLayerMetadata,
) -> Result<()> {
    validate_row_width(input, layer.hidden_size, "decode attention input")?;
    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }
    Ok(())
}

#[tile]
fn read_decode_layer_norm_weights(
    source: &CommittedExternalSource,
    layer_idx: usize,
    norm: GemmaDecodeLayerNormKind,
) -> Result<Vec<crate::shared::numerics::det_num::Wgt>> {
    auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest { layer_idx, norm }
    )
}

#[tile]
fn reshape_decode_ref_heads_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source_name: String,
    num_heads: usize,
    head_dim: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let row = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    let heads = reshape_row_heads(row, num_heads, head_dim)?;
    insert_attention_heads_with_roots(&artifact_store_roots, source_name, heads)
}

#[tile]
fn apply_decode_rope_to_heads_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    partial_rotary_dim: usize,
    rope_freq_base_dim: usize,
    rope_base: Option<Acc>,
    position: usize,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let heads = materialize_attention_heads_from_roots(&artifact_store_roots, &heads_ref)?;
    let heads = apply_rope_to_heads(
        &heads,
        partial_rotary_dim,
        rope_freq_base_dim,
        rope_base,
        position,
    )?;
    insert_attention_heads_with_roots(&artifact_store_roots, output_source_name, heads)
}

#[tile]
fn resolve_decode_attention_cache_ref(
    attention_cache_slot: &DecodeLayerCacheSlot,
) -> Result<RasterKvCacheRef> {
    match attention_cache_slot {
        DecodeLayerCacheSlot::Empty { .. } => bail!("decode attention cache is empty"),
        DecodeLayerCacheSlot::Ref(cache_ref) => Ok(cache_ref.clone()),
    }
}

#[tile]
fn resolve_decode_attention_window(layer: &GemmaDecodeLayerMetadata) -> Result<Option<usize>> {
    match layer.attention_kind {
        GemmaDecodeAttentionKind::Full => Ok(None),
        GemmaDecodeAttentionKind::Sliding => {
            Ok(Some(layer.sliding_window.ok_or_else(|| {
                anyhow!("sliding attention layer is missing a sliding window")
            })?))
        }
    }
}

#[tile]
fn rms_norm_decode_heads(
    heads: &RasterAttentionHeadSequence,
    norm_weights: &[crate::shared::numerics::det_num::Wgt],
    eps: crate::shared::numerics::det_num::Acc,
) -> Result<RasterAttentionHeadSequence> {
    rms_norm_heads(heads, Some(norm_weights), Some(eps))
}

#[tile]
fn value_rms_norm_decode_heads(
    heads: &RasterAttentionHeadSequence,
    eps: crate::shared::numerics::det_num::Acc,
) -> Result<RasterAttentionHeadSequence> {
    value_rms_norm_heads(heads, Some(eps))
}

#[tile]
fn init_decode_kv_cache_append_state_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    cache_slot: DecodeLayerCacheSlot,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    cache_window: Option<usize>,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, DecodeKvCacheAppendArtifactState)> {
    validate_attention_kv_rows_per_tile(rows_per_tile)?;
    if cache_window == Some(0) {
        bail!("decode cache sliding window must retain at least one row");
    }
    let (head_count, key_sequence_len, head_dim) = key_ref.tensor_ref().shape().heads_metadata()?;
    let (value_head_count, value_sequence_len, value_head_dim) =
        value_ref.tensor_ref().shape().heads_metadata()?;
    if head_count != value_head_count
        || key_sequence_len != value_sequence_len
        || head_dim != value_head_dim
    {
        bail!("decode cache append key/value attention heads shape mismatch");
    }
    if key_sequence_len != 1 {
        bail!("decode cache append expects one-token K/V heads, got {key_sequence_len} rows");
    }

    let (old_cache_ref, old_len) = match cache_slot {
        DecodeLayerCacheSlot::Empty { num_kv_heads } => {
            if num_kv_heads != head_count {
                bail!(
                    "decode empty cache head count {num_kv_heads}, expected projected {head_count}"
                );
            }
            (None, 0)
        }
        DecodeLayerCacheSlot::Ref(cache_ref) => {
            let (cache_head_count, cache_len, cache_head_dim) =
                cache_ref.shape().kv_cache_metadata()?;
            if cache_head_count != head_count {
                bail!(
                    "decode cache head count {cache_head_count}, expected projected {head_count}"
                );
            }
            if cache_head_dim != head_dim {
                bail!("decode cache head width {cache_head_dim}, expected projected {head_dim}");
            }
            (Some(cache_ref), cache_len)
        }
    };

    let retained_start = cache_window
        .map(|window| old_len.saturating_add(1).saturating_sub(window))
        .unwrap_or(0)
        .min(old_len);
    let retained_old_len = old_len.saturating_sub(retained_start);
    let current_len = retained_old_len + 1;
    let keys_source_name = keys_id.source_name().to_string();
    let values_source_name = values_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&keys_source_name)?,
        head_count * current_len,
        head_dim,
    )?;
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&values_source_name)?,
        head_count * current_len,
        head_dim,
    )?;

    Ok((
        artifact_store_roots,
        DecodeKvCacheAppendArtifactState {
            old_cache_ref,
            key_ref,
            value_ref,
            keys_source_name,
            values_source_name,
            retained_old_start: retained_start,
            retained_old_len,
            next_head_idx: 0,
            next_old_offset: 0,
            head_count,
            current_len,
            head_dim,
            rows_per_tile,
        },
    ))
}

#[tile(kind = recursive)]
fn compute_next_decode_kv_cache_append_row_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut kv_cache_append_state: DecodeKvCacheAppendArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    DecodeKvCacheAppendArtifactState,
)> {
    if kv_cache_append_state.next_head_idx >= kv_cache_append_state.head_count {
        return Ok((true, artifact_store_roots, kv_cache_append_state));
    }
    let mut artifact_store_roots = artifact_store_roots;

    if kv_cache_append_state.next_old_offset < kv_cache_append_state.retained_old_len {
        let old_cache_ref = kv_cache_append_state
            .old_cache_ref
            .as_ref()
            .ok_or_else(|| {
                anyhow!("decode cache append has retained rows without old cache ref")
            })?;
        let end = kv_cache_append_state
            .next_old_offset
            .saturating_add(kv_cache_append_state.rows_per_tile)
            .min(kv_cache_append_state.retained_old_len);
        for old_offset in kv_cache_append_state.next_old_offset..end {
            let input_token_idx = kv_cache_append_state.retained_old_start + old_offset;
            let key_row = read_kv_row_from_roots(
                &artifact_store_roots,
                RasterKvRowRequest {
                    cache_ref: old_cache_ref.clone(),
                    row_kind: RasterKvRowKind::Key,
                    head_idx: kv_cache_append_state.next_head_idx,
                    token_idx: input_token_idx,
                },
            )?;
            let value_row = read_kv_row_from_roots(
                &artifact_store_roots,
                RasterKvRowRequest {
                    cache_ref: old_cache_ref.clone(),
                    row_kind: RasterKvRowKind::Value,
                    head_idx: kv_cache_append_state.next_head_idx,
                    token_idx: input_token_idx,
                },
            )?;
            artifact_store_roots = append_head_row_by_source_name_with_roots(
                &artifact_store_roots,
                &kv_cache_append_state.keys_source_name,
                kv_cache_append_state.next_head_idx,
                old_offset,
                kv_cache_append_state.current_len,
                key_row,
            )?;
            artifact_store_roots = append_head_row_by_source_name_with_roots(
                &artifact_store_roots,
                &kv_cache_append_state.values_source_name,
                kv_cache_append_state.next_head_idx,
                old_offset,
                kv_cache_append_state.current_len,
                value_row,
            )?;
        }
        kv_cache_append_state.next_old_offset = end;
        if kv_cache_append_state.next_old_offset < kv_cache_append_state.retained_old_len {
            return Ok((false, artifact_store_roots, kv_cache_append_state));
        }
    }

    let output_token_idx = kv_cache_append_state.retained_old_len;
    let key_row = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: kv_cache_append_state.key_ref.clone(),
            head_idx: kv_cache_append_state.next_head_idx,
            token_idx: 0,
        },
    )?;
    let value_row = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: kv_cache_append_state.value_ref.clone(),
            head_idx: kv_cache_append_state.next_head_idx,
            token_idx: 0,
        },
    )?;
    artifact_store_roots = append_head_row_by_source_name_with_roots(
        &artifact_store_roots,
        &kv_cache_append_state.keys_source_name,
        kv_cache_append_state.next_head_idx,
        output_token_idx,
        kv_cache_append_state.current_len,
        key_row,
    )?;
    artifact_store_roots = append_head_row_by_source_name_with_roots(
        &artifact_store_roots,
        &kv_cache_append_state.values_source_name,
        kv_cache_append_state.next_head_idx,
        output_token_idx,
        kv_cache_append_state.current_len,
        value_row,
    )?;
    kv_cache_append_state.next_head_idx += 1;
    kv_cache_append_state.next_old_offset = 0;
    Ok((false, artifact_store_roots, kv_cache_append_state))
}

#[tile]
fn finalize_decode_kv_cache_append_state_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    kv_cache_append_state: DecodeKvCacheAppendArtifactState,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerCacheSlot)> {
    if kv_cache_append_state.next_head_idx != kv_cache_append_state.head_count {
        bail!(
            "decode cache append finalized at head {}, expected {} heads",
            kv_cache_append_state.next_head_idx,
            kv_cache_append_state.head_count
        );
    }
    let (artifact_store_roots, cache_ref) = finalize_kv_cache_builders_by_source_name_with_roots(
        &artifact_store_roots,
        &kv_cache_append_state.keys_source_name,
        &kv_cache_append_state.values_source_name,
        RasterTensorId::new(kv_cache_append_state.keys_source_name.clone())?,
        RasterTensorId::new(kv_cache_append_state.values_source_name.clone())?,
        kv_cache_append_state.head_count,
        kv_cache_append_state.current_len,
        kv_cache_append_state.head_dim,
    )?;
    Ok((artifact_store_roots, DecodeLayerCacheSlot::Ref(cache_ref)))
}

#[tile]
pub fn init_decode_attention_artifact_state_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    query_ref: RasterAttentionHeadsRef,
    cache_ref: RasterKvCacheRef,
    output_id: RasterTensorId,
    attention_id_prefix: String,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, DecodeAttentionArtifactState)> {
    validate_attention_kv_rows_per_tile(kv_rows_per_tile)?;
    let (query_head_count, query_sequence_len, head_dim) =
        query_ref.tensor_ref().shape().heads_metadata()?;
    if query_sequence_len != 1 {
        bail!(
            "decode attention query must contain exactly one token row, got {query_sequence_len}"
        );
    }
    let (kv_head_count, cache_len, cache_head_dim) = cache_ref.shape().kv_cache_metadata()?;
    if cache_head_dim != head_dim {
        bail!("decode attention cache head width {cache_head_dim}, expected {head_dim}");
    }
    if query_head_count % kv_head_count != 0 {
        bail!(
            "decode attention query head count {} must be divisible by KV head count {}",
            query_head_count,
            kv_head_count
        );
    }
    let key_start = attention_window
        .map(|window| cache_len.saturating_sub(window))
        .unwrap_or(0);
    let row_count = cache_len.saturating_sub(key_start);
    if row_count == 0 {
        bail!("decode attention requires at least one visible KV row");
    }
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        query_head_count,
        head_dim,
    )?;
    let (artifact_store_roots, phase) = init_decode_attention_score_phase_with_roots(
        artifact_store_roots,
        &attention_id_prefix,
        0,
        row_count,
    )?;
    Ok((
        artifact_store_roots,
        DecodeAttentionArtifactState {
            query_ref,
            cache_ref,
            output_source_name,
            phase,
            attention_id_prefix,
            next_query_head_idx: 0,
            query_head_count,
            kv_head_count,
            kv_groups: query_head_count / kv_head_count,
            key_start,
            row_count,
            head_dim,
            kv_rows_per_tile,
        },
    ))
}

#[tile(kind = recursive)]
pub fn compute_next_decode_attention_head_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut attention_state: DecodeAttentionArtifactState,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeAttentionArtifactState)> {
    if attention_state.next_query_head_idx >= attention_state.query_head_count {
        return Ok((true, artifact_store_roots, attention_state));
    }

    let mut artifact_store_roots = artifact_store_roots;
    let query_head_idx = attention_state.next_query_head_idx;
    let kv_head_idx = query_head_idx / attention_state.kv_groups;
    let query = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: attention_state.query_ref.clone(),
            head_idx: query_head_idx,
            token_idx: 0,
        },
    )?;

    match attention_state.phase.clone() {
        DecodeAttentionArtifactPhase::CollectScores {
            score_source_name,
            next_kv_offset,
        } => {
            let end = next_kv_offset
                .saturating_add(attention_state.kv_rows_per_tile)
                .min(attention_state.row_count);
            for offset in next_kv_offset..end {
                let token_idx = attention_state.key_start + offset;
                let key_row = read_decode_attention_kv_row_from_roots(
                    &artifact_store_roots,
                    &attention_state.cache_ref,
                    RasterKvRowKind::Key,
                    kv_head_idx,
                    token_idx,
                )?;
                if key_row.width() != query.width() {
                    bail!(
                        "decode attention key row ({kv_head_idx}, {token_idx}) has width {}, expected {}",
                        key_row.width(),
                        query.width()
                    );
                }
                let score = det_attention_score(&query.acts(), &key_row.acts());
                artifact_store_roots = append_sequence_row_by_source_name_with_roots(
                    &artifact_store_roots,
                    &score_source_name,
                    offset,
                    RasterActivationRow::from_acts(vec![score]),
                )?;
            }
            if end < attention_state.row_count {
                attention_state.phase = DecodeAttentionArtifactPhase::CollectScores {
                    score_source_name,
                    next_kv_offset: end,
                };
                return Ok((false, artifact_store_roots, attention_state));
            }
            let (roots, score_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &score_source_name,
                RasterTensorId::new(score_source_name.clone())?,
            )?;
            artifact_store_roots = roots;
            attention_state.phase = DecodeAttentionArtifactPhase::FindSoftmaxMax {
                score_ref,
                next_score_row_idx: 0,
                max_index: None,
                max_logit_bits: 0,
            };
            Ok((false, artifact_store_roots, attention_state))
        }
        DecodeAttentionArtifactPhase::FindSoftmaxMax {
            score_ref,
            next_score_row_idx,
            mut max_index,
            mut max_logit_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(attention_state.kv_rows_per_tile)
                .min(attention_state.row_count);
            for row_idx in next_score_row_idx..end {
                let score = read_decode_attention_scalar_row_from_roots(
                    &artifact_store_roots,
                    &score_ref,
                    row_idx,
                    "score",
                )?;
                let score_bits = score.to_bits();
                if max_index.is_none() || score_bits > max_logit_bits {
                    max_index = Some(row_idx);
                    max_logit_bits = score_bits;
                }
            }
            if end < attention_state.row_count {
                attention_state.phase = DecodeAttentionArtifactPhase::FindSoftmaxMax {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                };
                return Ok((false, artifact_store_roots, attention_state));
            }
            let max_index = max_index
                .ok_or_else(|| anyhow!("decode attention softmax requires at least one score"))?;
            attention_state.phase = DecodeAttentionArtifactPhase::SumSoftmaxExp {
                score_ref,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits: 0,
            };
            Ok((false, artifact_store_roots, attention_state))
        }
        DecodeAttentionArtifactPhase::SumSoftmaxExp {
            score_ref,
            next_score_row_idx,
            max_index,
            max_logit_bits,
            mut sum_exp_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(attention_state.kv_rows_per_tile)
                .min(attention_state.row_count);
            let max_logit = Act::from_bits(max_logit_bits);
            let mut sum_exp = Acc::from_bits(sum_exp_bits);
            for row_idx in next_score_row_idx..end {
                let score = read_decode_attention_scalar_row_from_roots(
                    &artifact_store_roots,
                    &score_ref,
                    row_idx,
                    "score",
                )?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                sum_exp = acc_add_sat(sum_exp, exp_term);
            }
            sum_exp_bits = sum_exp.to_bits();
            if end < attention_state.row_count {
                attention_state.phase = DecodeAttentionArtifactPhase::SumSoftmaxExp {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                };
                return Ok((false, artifact_store_roots, attention_state));
            }
            if sum_exp_bits == 0 {
                bail!("decode attention softmax exp sum is zero");
            }
            let raw_weight_source_name = format!(
                "{}.raw_weights.head_{query_head_idx}",
                attention_state.attention_id_prefix
            );
            artifact_store_roots = start_sequence_builder_with_roots(
                &artifact_store_roots,
                RasterArtifactId::new(&raw_weight_source_name)?,
                attention_state.row_count,
                1,
            )?;
            attention_state.phase = DecodeAttentionArtifactPhase::BuildRawSoftmaxWeights {
                score_ref,
                raw_weight_source_name,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits,
                summed_weight_bits: 0,
            };
            Ok((false, artifact_store_roots, attention_state))
        }
        DecodeAttentionArtifactPhase::BuildRawSoftmaxWeights {
            score_ref,
            raw_weight_source_name,
            next_score_row_idx,
            max_index,
            max_logit_bits,
            sum_exp_bits,
            mut summed_weight_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(attention_state.kv_rows_per_tile)
                .min(attention_state.row_count);
            let max_logit = Act::from_bits(max_logit_bits);
            let sum_exp = Acc::from_bits(sum_exp_bits);
            let mut summed_weight = Act::from_bits(summed_weight_bits);
            for row_idx in next_score_row_idx..end {
                let score = read_decode_attention_scalar_row_from_roots(
                    &artifact_store_roots,
                    &score_ref,
                    row_idx,
                    "score",
                )?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                let weight = attention_softmax_raw_weight(exp_term, sum_exp);
                artifact_store_roots = append_sequence_row_by_source_name_with_roots(
                    &artifact_store_roots,
                    &raw_weight_source_name,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
                summed_weight = add_sat(summed_weight, weight);
            }
            summed_weight_bits = summed_weight.to_bits();
            if end < attention_state.row_count {
                attention_state.phase = DecodeAttentionArtifactPhase::BuildRawSoftmaxWeights {
                    score_ref,
                    raw_weight_source_name,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                    summed_weight_bits,
                };
                return Ok((false, artifact_store_roots, attention_state));
            }
            let (roots, raw_weight_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &raw_weight_source_name,
                RasterTensorId::new(raw_weight_source_name.clone())?,
            )?;
            artifact_store_roots = roots;
            let final_weight_source_name = format!(
                "{}.weights.head_{query_head_idx}",
                attention_state.attention_id_prefix
            );
            artifact_store_roots = start_sequence_builder_with_roots(
                &artifact_store_roots,
                RasterArtifactId::new(&final_weight_source_name)?,
                attention_state.row_count,
                1,
            )?;
            let residual = attention_softmax_residual(Act::from_bits(summed_weight_bits));
            attention_state.phase = DecodeAttentionArtifactPhase::CorrectSoftmaxResidual {
                raw_weight_ref,
                final_weight_source_name,
                next_weight_row_idx: 0,
                max_index,
                residual_bits: residual.to_bits(),
            };
            Ok((false, artifact_store_roots, attention_state))
        }
        DecodeAttentionArtifactPhase::CorrectSoftmaxResidual {
            raw_weight_ref,
            final_weight_source_name,
            next_weight_row_idx,
            max_index,
            residual_bits,
        } => {
            let end = next_weight_row_idx
                .saturating_add(attention_state.kv_rows_per_tile)
                .min(attention_state.row_count);
            let residual = Act::from_bits(residual_bits);
            for row_idx in next_weight_row_idx..end {
                let mut weight = read_decode_attention_scalar_row_from_roots(
                    &artifact_store_roots,
                    &raw_weight_ref,
                    row_idx,
                    "weight",
                )?;
                if row_idx == max_index {
                    weight = add_sat(weight, residual);
                }
                artifact_store_roots = append_sequence_row_by_source_name_with_roots(
                    &artifact_store_roots,
                    &final_weight_source_name,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
            }
            if end < attention_state.row_count {
                attention_state.phase = DecodeAttentionArtifactPhase::CorrectSoftmaxResidual {
                    raw_weight_ref,
                    final_weight_source_name,
                    next_weight_row_idx: end,
                    max_index,
                    residual_bits,
                };
                return Ok((false, artifact_store_roots, attention_state));
            }
            let (roots, weight_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &final_weight_source_name,
                RasterTensorId::new(final_weight_source_name.clone())?,
            )?;
            artifact_store_roots = roots;
            attention_state.phase = DecodeAttentionArtifactPhase::ApplyValues {
                weight_ref,
                next_kv_offset: 0,
                weighted_sum_acc_bits: vec![0; query.width()],
            };
            Ok((false, artifact_store_roots, attention_state))
        }
        DecodeAttentionArtifactPhase::ApplyValues {
            weight_ref,
            next_kv_offset,
            mut weighted_sum_acc_bits,
        } => {
            let end = next_kv_offset
                .saturating_add(attention_state.kv_rows_per_tile)
                .min(attention_state.row_count);
            for offset in next_kv_offset..end {
                let token_idx = attention_state.key_start + offset;
                let weight = read_decode_attention_scalar_row_from_roots(
                    &artifact_store_roots,
                    &weight_ref,
                    offset,
                    "weight",
                )?;
                let value_row = read_decode_attention_kv_row_from_roots(
                    &artifact_store_roots,
                    &attention_state.cache_ref,
                    RasterKvRowKind::Value,
                    kv_head_idx,
                    token_idx,
                )?;
                if value_row.width() != weighted_sum_acc_bits.len() {
                    bail!(
                        "decode attention value row ({kv_head_idx}, {token_idx}) has width {}, expected {}",
                        value_row.width(),
                        weighted_sum_acc_bits.len()
                    );
                }
                for (acc_bits, value) in weighted_sum_acc_bits.iter_mut().zip(value_row.acts()) {
                    *acc_bits = mac_bits(*acc_bits, value.to_bits(), weight.to_bits());
                }
            }
            if end < attention_state.row_count {
                attention_state.phase = DecodeAttentionArtifactPhase::ApplyValues {
                    weight_ref,
                    next_kv_offset: end,
                    weighted_sum_acc_bits,
                };
                return Ok((false, artifact_store_roots, attention_state));
            }
            let output_row = RasterActivationRow::from_acts(
                weighted_sum_acc_bits
                    .into_iter()
                    .map(|bits| requantize(Acc::from_bits(bits)))
                    .collect(),
            );
            artifact_store_roots = append_head_row_by_source_name_with_roots(
                &artifact_store_roots,
                &attention_state.output_source_name,
                query_head_idx,
                0,
                1,
                output_row,
            )?;
            attention_state.next_query_head_idx += 1;
            if attention_state.next_query_head_idx < attention_state.query_head_count {
                let (roots, phase) = init_decode_attention_score_phase_with_roots(
                    artifact_store_roots,
                    &attention_state.attention_id_prefix,
                    attention_state.next_query_head_idx,
                    attention_state.row_count,
                )?;
                artifact_store_roots = roots;
                attention_state.phase = phase;
            }
            Ok((false, artifact_store_roots, attention_state))
        }
    }
}

#[tile]
pub fn finalize_decode_attention_state_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    attention_state: DecodeAttentionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    if attention_state.next_query_head_idx != attention_state.query_head_count {
        bail!(
            "decode attention finalized at head {}, expected {} heads",
            attention_state.next_query_head_idx,
            attention_state.query_head_count
        );
    }
    finalize_heads_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &attention_state.output_source_name,
        RasterTensorId::new(attention_state.output_source_name.clone())?,
        attention_state.query_head_count,
        1,
        attention_state.head_dim,
    )
}

#[tile]
fn gelu_decode_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    gelu_decode_ref(artifact_store_roots, input_ref, output_source_name)
}

#[tile]
fn mul_decode_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    mul_decode_refs(artifact_store_roots, lhs_ref, rhs_ref, output_source_name)
}

#[tile]
fn decode_ple_layer_projection_rows(layer: &GemmaDecodeLayerMetadata) -> Result<usize> {
    Ok(layer
        .ple_layer_projection_shape
        .ok_or_else(|| {
            anyhow!("Gemma decode layer metadata is missing PLE layer projection shape")
        })?
        .rows)
}

#[tile]
pub fn init_decode_transition_state_from_refs_with_roots(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    position: usize,
    token_count: usize,
    original_layer_caches: Vec<DecodeLayerCacheSlot>,
    next_token: u32,
    source: &CommittedExternalSource,
    raster_sizing: RasterSizingControls,
    output_source_prefix: String,
) -> Result<(RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    validate_attention_kv_rows_per_tile(raster_sizing.attention_kv_rows_per_tile)?;
    let metadata = auth_read!(source, GemmaDecodeTransitionMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer decode requires at least one layer");
    }
    if original_layer_caches.len() != metadata.layer_count {
        bail!(
            "transformer decode cache count mismatch: {} vs {}",
            original_layer_caches.len(),
            metadata.layer_count
        );
    }

    let embedded = RasterActivationRow::from_acts(auth_read!(
        source,
        GemmaDecodeEmbeddingRowRequest {
            token_id: next_token
        },
    )?);
    if embedded.width() != metadata.embedding_width {
        bail!(
            "decode embedded token width {}, expected {}",
            embedded.width(),
            metadata.embedding_width
        );
    }
    let (roots, decode_input_ref) = insert_decode_activation_row_with_roots(
        &artifact_store_roots,
        format!("{output_source_prefix}.input.selected_token_embedding"),
        &embedded,
    )?;
    artifact_store_roots = roots;

    Ok((
        artifact_store_roots.clone(),
        DecodeTransitionRasterState {
            artifact_store_roots,
            decode_input_ref: decode_input_ref.clone(),
            current_activation_ref: decode_input_ref,
            next_token,
            position,
            token_count,
            next_layer_idx: 0,
            layer_count: metadata.layer_count,
            original_layer_caches,
            updated_layer_caches: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
            completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
            projection_rows_per_tile: raster_sizing.projection_rows_per_tile,
            attention_kv_rows_per_tile: raster_sizing.attention_kv_rows_per_tile,
            output_source_prefix,
        },
    ))
}

#[tile]
pub fn finalize_decode_layer_refs_with_roots(
    decode_state: DecodeTransitionRasterState,
) -> Result<(Vec<DecodeLayerCacheSlot>, usize, usize)> {
    if decode_state.next_layer_idx != decode_state.layer_count {
        bail!(
            "raster decode finalized after {} layers, expected {}",
            decode_state.next_layer_idx,
            decode_state.layer_count
        );
    }
    if decode_state.updated_layer_caches.len() != decode_state.layer_count {
        bail!(
            "raster decode stored {} layer caches, expected {}",
            decode_state.updated_layer_caches.len(),
            decode_state.layer_count
        );
    }

    Ok((
        decode_state.updated_layer_caches,
        decode_state.position + 1,
        decode_state.token_count + 1,
    ))
}

#[tile]
fn rms_norm_decode_row(
    row: &RasterActivationRow,
    norm_weights: &[crate::shared::numerics::det_num::Wgt],
    eps: crate::shared::numerics::det_num::Acc,
    label: &str,
) -> Result<RasterActivationRow> {
    first_row(
        rms_norm_sequence(
            &RasterActivationSequence::from_rows(vec![row.clone()]),
            Some(norm_weights),
            Some(eps),
        )?,
        label,
    )
}

#[tile]
fn read_decode_single_activation_row(
    decode_state: &DecodeTransitionRasterState,
    activation_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationRow> {
    read_activation_row_from_ref_roots(&decode_state.artifact_store_roots, activation_ref)
}

#[tile]
pub fn normalize_decode_final_position(
    final_hidden_state: &ActivationSequence,
    source: &CommittedExternalSource,
) -> Result<RasterActivationRow> {
    let internal = final_hidden_state.clone_internal();
    let det_rows = internal.det_values().ok_or_else(|| {
        anyhow!("deterministic raster decode finalize requires canonical final hidden activations")
    })?;
    let final_row = det_rows.last().ok_or_else(|| {
        anyhow!("transformer final-position selection requires at least one activation row")
    })?;
    let norm_weights = auth_read!(source, GemmaDecodeFinalNormWeightsRequest)?;
    let scalars = auth_read!(source, GemmaDecodeFinalScalarsRequest)?;
    let normalized = rms_norm_sequence(
        &RasterActivationSequence::from_rows(vec![RasterActivationRow::from_acts(
            final_row.clone(),
        )]),
        Some(&norm_weights),
        Some(scalars.rms_norm_eps),
    )?;
    first_row(normalized, "deterministic decode final RMSNorm")
}

#[tile]
fn add_decode_rows(
    lhs: &RasterActivationRow,
    rhs: &RasterActivationRow,
) -> Result<RasterActivationRow> {
    add_rows(lhs, rhs)
}

#[tile]
fn mul_decode_rows(
    lhs: &RasterActivationRow,
    rhs: &RasterActivationRow,
) -> Result<RasterActivationRow> {
    mul_rows(lhs, rhs)
}

#[tile]
fn gelu_decode_row(row: &RasterActivationRow) -> Result<RasterActivationRow> {
    gelu_row(row)
}

#[tile]
fn clone_decode_key_as_value(raw_k: &RasterActivationRow) -> Result<RasterActivationRow> {
    Ok(raw_k.clone())
}

#[tile]
fn reshape_decode_row_heads(
    row: RasterActivationRow,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterAttentionHeadSequence> {
    reshape_row_heads(row, num_heads, head_dim)
}

#[tile]
fn apply_decode_rope_to_heads(
    heads: &RasterAttentionHeadSequence,
    partial_rotary_dim: usize,
    rope_freq_base_dim: usize,
    rope_base: Option<Acc>,
    position: usize,
) -> Result<RasterAttentionHeadSequence> {
    apply_rope_to_heads(
        heads,
        partial_rotary_dim,
        rope_freq_base_dim,
        rope_base,
        position,
    )
}
