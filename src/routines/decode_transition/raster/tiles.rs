use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::decode_transition::raster::auth_source::{
    AuthenticatedGemmaDecodeTransitionSource, GemmaDecodeAttentionKind,
    GemmaDecodeEmbeddingRowRequest, GemmaDecodeFinalNormWeightsRequest,
    GemmaDecodeFinalScalarsRequest, GemmaDecodeLayerMatrixKind, GemmaDecodeLayerMatrixRowRequest,
    GemmaDecodeLayerMetadata, GemmaDecodeLayerMetadataRequest, GemmaDecodeLayerNormKind,
    GemmaDecodeLayerNormWeightsRequest, GemmaDecodeLayerScalarsRequest,
    GemmaDecodePleModelProjectionRowRequest, GemmaDecodePleProjectionNormWeightsRequest,
    GemmaDecodePleScalarsRequest, GemmaDecodePleTokenEmbeddingRowRequest,
    GemmaDecodeProjectionRowRequest, GemmaDecodeTransitionMetadataRequest,
};
use crate::dsl::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, read_selected_token_from_roots, token_id_leaf,
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{
    ActivationSequence, InternalActivationSequence, InternalLogits, LayerKvCache, PrefillLogits,
    TransformerDecodeState, TransformerDecodeStepResult,
};
use crate::shared::numerics::det_num::{
    acc_add_sat, add_sat, attention_score as det_attention_score, attention_softmax_exp_term,
    attention_softmax_raw_weight, attention_softmax_residual, mac_bits, requantize, softcap_act,
    Acc, Act,
};
use crate::shared::raster_kernels::transformer::{
    add_sequences, apply_rope_to_heads, gelu_sequence, mul_sequences, project_row_with_weights,
    rms_norm_heads, rms_norm_sequence, scale_sequence, validate_attention_kv_rows_per_tile,
    validate_projection_rows_per_tile, value_rms_norm_heads, RasterActivationRow,
    RasterActivationSequence, RasterAttentionHeadSequence, RasterKvCache,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, append_head_row_by_source_name_with_roots,
    append_sequence_row_by_source_name_with_roots, attention_heads_ref_from_artifact,
    finalize_heads_builder_by_source_name_with_roots,
    finalize_kv_cache_builders_by_source_name_with_roots,
    finalize_sequence_builder_by_source_name_with_roots, kv_cache_ref_from_artifacts,
    read_head_row_from_roots, read_kv_row_from_roots, read_sequence_row_from_roots,
    start_sequence_builder_with_roots, RasterActivationSequenceRef, RasterAttentionHeadsRef,
    RasterHeadRowRequest, RasterKvCacheBuilderRef, RasterKvCacheRef, RasterKvRowKind,
    RasterKvRowRequest, RasterProjectionOutputBuilderRef, RasterSequenceRowRequest,
    RasterTensorBuilderRef, RasterTensorId,
};
use crate::RasterSizingControls;

use super::super::native::ActivationSequenceWithCache;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeTransitionRasterState {
    artifact_store_roots: RasterArtifactStoreRoots,
    decode_input_ref: RasterActivationSequenceRef,
    current_activation_ref: RasterActivationSequenceRef,
    next_token: u32,
    position: usize,
    token_count: usize,
    next_layer_idx: usize,
    layer_count: usize,
    original_layer_caches: Vec<DecodeLayerCacheSlot>,
    updated_layer_caches: Vec<DecodeLayerCacheSlot>,
    completed_layer_output_sha256s: Vec<String>,
    completed_layer_output_det_sha256s: Vec<Option<String>>,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    output_source_prefix: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterDecodeTransitionInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub transformer_decode_state: TransformerDecodeState,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub decode_transition_source_root: String,
    pub output_source_prefix: String,
    pub raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeTransitionInputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub position: usize,
    pub token_count: usize,
    pub layer_caches: Vec<DecodeLayerCacheSlot>,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub decode_transition_source_root: String,
    pub output_source_prefix: String,
    pub raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterDecodeTransitionOutputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub transition_result: TransformerDecodeStepResult,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeTransitionOutputStateRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub final_hidden_state_ref: RasterActivationSequenceRef,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
    pub layer_caches: Vec<DecodeLayerCacheSlot>,
    pub position: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum DecodeLayerCacheSlot {
    Empty { num_kv_heads: usize },
    Ref(RasterKvCacheRef),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeLogitsRasterState {
    normalized_final_position: RasterActivationRow,
    next_logit_idx: usize,
    logit_count: usize,
    output_builder_ref: RasterProjectionOutputBuilderRef,
    softcap_bits: Option<i32>,
    projection_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum DecodeProjectionKind {
    LayerMatrix {
        layer_idx: usize,
        matrix: GemmaDecodeLayerMatrixKind,
    },
    PleModel {
        layer_idx: usize,
    },
    FinalLogits,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeRowProjectionState {
    input: RasterActivationRow,
    projection_kind: DecodeProjectionKind,
    next_projection_row_idx: usize,
    projection_rows: usize,
    input_width: usize,
    output_builder_ref: RasterProjectionOutputBuilderRef,
    rows_per_tile: usize,
    softcap_bits: Option<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeRowProjectionArtifactState {
    input_ref: RasterActivationSequenceRef,
    projection_kind: DecodeProjectionKind,
    output_source_name: String,
    current_row_bits: Vec<i32>,
    next_projection_row_idx: usize,
    projection_rows: usize,
    input_width: usize,
    rows_per_tile: usize,
    softcap_bits: Option<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeAttentionState {
    query_ref: RasterAttentionHeadsRef,
    cache_ref: RasterKvCacheRef,
    output_builder_ref: RasterTensorBuilderRef,
    phase: DecodeAttentionPhase,
    attention_id_prefix: String,
    next_query_head_idx: usize,
    query_head_count: usize,
    kv_head_count: usize,
    kv_groups: usize,
    key_start: usize,
    row_count: usize,
    head_dim: usize,
    kv_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeAttentionArtifactState {
    query_ref: RasterAttentionHeadsRef,
    cache_ref: RasterKvCacheRef,
    output_source_name: String,
    phase: DecodeAttentionArtifactPhase,
    attention_id_prefix: String,
    next_query_head_idx: usize,
    query_head_count: usize,
    kv_head_count: usize,
    kv_groups: usize,
    key_start: usize,
    row_count: usize,
    head_dim: usize,
    kv_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct DecodeLayerContext {
    layer_idx: usize,
    layer: GemmaDecodeLayerMetadata,
    cache_slot: DecodeLayerCacheSlot,
    donor_cache_slot: Option<DecodeLayerCacheSlot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeKvCacheAppendState {
    old_cache_ref: Option<RasterKvCacheRef>,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    output_builder_ref: RasterKvCacheBuilderRef,
    retained_old_start: usize,
    retained_old_len: usize,
    next_head_idx: usize,
    next_old_offset: usize,
    head_count: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeKvCacheAppendArtifactState {
    old_cache_ref: Option<RasterKvCacheRef>,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_source_name: String,
    values_source_name: String,
    retained_old_start: usize,
    retained_old_len: usize,
    next_head_idx: usize,
    next_old_offset: usize,
    head_count: usize,
    current_len: usize,
    head_dim: usize,
    rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum DecodeAttentionPhase {
    CollectScores {
        score_builder_ref: RasterTensorBuilderRef,
        next_kv_offset: usize,
    },
    FindSoftmaxMax {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: Option<usize>,
        max_logit_bits: i32,
    },
    SumSoftmaxExp {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
    },
    BuildRawSoftmaxWeights {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        raw_weight_builder_ref: RasterTensorBuilderRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
        summed_weight_bits: i32,
    },
    CorrectSoftmaxResidual {
        raw_weight_ref:
            crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        final_weight_builder_ref: RasterTensorBuilderRef,
        next_weight_row_idx: usize,
        max_index: usize,
        residual_bits: i32,
    },
    ApplyValues {
        weight_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_kv_offset: usize,
        weighted_sum_acc_bits: Vec<i64>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
enum DecodeAttentionArtifactPhase {
    CollectScores {
        score_source_name: String,
        next_kv_offset: usize,
    },
    FindSoftmaxMax {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: Option<usize>,
        max_logit_bits: i32,
    },
    SumSoftmaxExp {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
    },
    BuildRawSoftmaxWeights {
        score_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        raw_weight_source_name: String,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
        summed_weight_bits: i32,
    },
    CorrectSoftmaxResidual {
        raw_weight_ref:
            crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        final_weight_source_name: String,
        next_weight_row_idx: usize,
        max_index: usize,
        residual_bits: i32,
    },
    ApplyValues {
        weight_ref: crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
        next_kv_offset: usize,
        weighted_sum_acc_bits: Vec<i64>,
    },
}

#[tile]
fn read_decode_selected_token(
    artifact_store_roots: &RasterArtifactStoreRoots,
    selected_token_ref: &RasterSelectedTokenRef,
) -> Result<u32> {
    read_selected_token_from_roots(artifact_store_roots, selected_token_ref)
}

#[tile]
fn read_decode_single_activation_row(
    state: &DecodeTransitionRasterState,
    activation_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationRow> {
    read_activation_row_from_ref_roots(&state.artifact_store_roots, activation_ref)
}

#[tile]
fn prepare_next_decode_layer_context(
    state: &DecodeTransitionRasterState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<DecodeLayerContext> {
    let layer_idx = state.next_layer_idx;
    let layer = auth_read!(source, GemmaDecodeLayerMetadataRequest { layer_idx })?;
    let cache_slot = state
        .original_layer_caches
        .get(layer_idx)
        .cloned()
        .ok_or_else(|| anyhow!("transformer decode cache {layer_idx} missing"))?;
    let donor_cache_slot =
        resolve_decode_donor_cache_slot(&state.updated_layer_caches, layer_idx, &layer)?.cloned();
    Ok(DecodeLayerContext {
        layer_idx,
        layer,
        cache_slot,
        donor_cache_slot,
    })
}

#[tile]
pub fn normalize_decode_final_position(
    final_hidden_state: &ActivationSequence,
    source: &AuthenticatedGemmaDecodeTransitionSource,
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

#[sequence]
pub fn run(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    raster_sizing: RasterSizingControls,
) -> Result<TransformerDecodeStepResult> {
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
    let output = main(
        RasterDecodeTransitionInputRoots {
            artifact_store_roots,
            transformer_decode_state,
            selected_token_ref,
            decode_transition_source_root: source.static_source_root(),
            output_source_prefix,
            raster_sizing,
        },
        source,
    )?;
    Ok(output.transition_result)
}

#[sequence]
pub fn main(
    input_roots: RasterDecodeTransitionInputRoots,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<RasterDecodeTransitionOutputRefs> {
    if input_roots.decode_transition_source_root != source.static_source_root() {
        bail!(
            "raster decode transition source root {} does not match input source root {}",
            source.static_source_root(),
            input_roots.decode_transition_source_root
        );
    }
    let next_token = call_tile!(
        read_decode_selected_token,
        &input_roots.artifact_store_roots,
        &input_roots.selected_token_ref
    )?;
    let (_artifact_store_roots, state) = call_tile!(
        init_decode_transition_state_refs_with_roots,
        input_roots.artifact_store_roots,
        input_roots.transformer_decode_state.clone(),
        next_token,
        source,
        input_roots.raster_sizing,
        input_roots.output_source_prefix
    )?;
    let (artifact_store_roots, state) = call_recur_seq!(
        compute_next_decode_layer_with_roots,
        (_artifact_store_roots, state),
        source
    )?;
    let final_hidden_states_ref = state.current_activation_ref.clone();
    let (artifact_store_roots, normalized_ref) = call_tile!(
        normalize_decode_final_position_ref_with_roots,
        artifact_store_roots,
        final_hidden_states_ref,
        source,
        format!("{}.final_norm", state.output_source_prefix)
    )?;
    let (artifact_store_roots, logits_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        normalized_ref,
        source,
        DecodeProjectionKind::FinalLogits,
        call_tile!(decode_projection_row_count, source)?,
        input_roots.raster_sizing.projection_rows_per_tile,
        format!("{}.final_logits", state.output_source_prefix),
        call_tile!(decode_final_logit_softcap_bits, source)?
    )?;
    let layer_output = call_tile!(
        finalize_decode_layer_state_with_roots,
        &artifact_store_roots,
        state
    )?;
    let transition_result = call_tile!(
        finalize_decode_transition_result_from_roots,
        &artifact_store_roots,
        logits_ref,
        TransformerDecodeState {
            layer_caches: layer_output.layer_caches,
            position: input_roots.transformer_decode_state.position,
            token_count: input_roots.transformer_decode_state.token_count,
        },
        layer_output.activation_state
    )?;
    Ok(RasterDecodeTransitionOutputRefs {
        artifact_store_roots,
        transition_result,
    })
}

#[sequence]
pub fn main_state_refs(
    input_roots: RasterDecodeTransitionInputRefs,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<RasterDecodeTransitionOutputStateRefs> {
    if input_roots.decode_transition_source_root != source.static_source_root() {
        bail!(
            "raster decode transition source root {} does not match input source root {}",
            source.static_source_root(),
            input_roots.decode_transition_source_root
        );
    }
    let next_token = call_tile!(
        read_decode_selected_token,
        &input_roots.artifact_store_roots,
        &input_roots.selected_token_ref
    )?;
    let (_artifact_store_roots, state) = call_tile!(
        init_decode_transition_state_from_refs_with_roots,
        input_roots.artifact_store_roots,
        input_roots.position,
        input_roots.token_count,
        input_roots.layer_caches,
        next_token,
        source,
        input_roots.raster_sizing,
        input_roots.output_source_prefix
    )?;
    let (artifact_store_roots, state) = call_recur_seq!(
        compute_next_decode_layer_with_roots,
        (_artifact_store_roots, state),
        source
    )?;
    let final_hidden_state_ref = state.current_activation_ref.clone();
    let (artifact_store_roots, normalized_ref) = call_tile!(
        normalize_decode_final_position_ref_with_roots,
        artifact_store_roots,
        final_hidden_state_ref.clone(),
        source,
        format!("{}.final_norm", state.output_source_prefix)
    )?;
    let (artifact_store_roots, logits_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        normalized_ref,
        source,
        DecodeProjectionKind::FinalLogits,
        call_tile!(decode_projection_row_count, source)?,
        input_roots.raster_sizing.projection_rows_per_tile,
        format!("{}.final_logits", state.output_source_prefix),
        call_tile!(decode_final_logit_softcap_bits, source)?
    )?;
    let (layer_caches, position, token_count) =
        call_tile!(finalize_decode_layer_refs_with_roots, state)?;
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    let logit_count = match (row_count, width) {
        (rows, 1) => rows,
        (1, cols) => cols,
        _ => bail!("raster decode transition logits shape {row_count}x{width} must be Nx1 or 1xN"),
    };
    Ok(RasterDecodeTransitionOutputStateRefs {
        artifact_store_roots,
        final_hidden_state_ref,
        logits_ref,
        logit_count,
        layer_caches,
        position,
        token_count,
    })
}

#[tile]
pub fn init_decode_transition_state_refs_with_roots(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &AuthenticatedGemmaDecodeTransitionSource,
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
pub fn init_decode_transition_state_from_refs_with_roots(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    position: usize,
    token_count: usize,
    original_layer_caches: Vec<DecodeLayerCacheSlot>,
    next_token: u32,
    source: &AuthenticatedGemmaDecodeTransitionSource,
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

#[sequence(kind = recursive)]
pub fn compute_next_decode_layer_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: DecodeTransitionRasterState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    if state.next_layer_idx >= state.layer_count {
        return Ok((true, artifact_store_roots, state));
    }

    let context = call_tile!(prepare_next_decode_layer_context, &state, source)?;
    let _trace = crate::trace::trace_scope(format!(
        "decode.layer.det layer={layer_idx} token={} position={} attention={:?} ple={} donor={:?}",
        state.next_token,
        state.position,
        context.layer.attention_kind,
        context.layer.has_ple,
        context.layer.kv_shared_layer_index,
        layer_idx = context.layer_idx
    ));
    let (artifact_store_roots, per_layer_input_ref) = call_seq!(
        compute_decode_ple_input_with_roots,
        artifact_store_roots,
        state.next_token,
        state.decode_input_ref.clone(),
        source,
        &context.layer,
        state.projection_rows_per_tile,
        format!(
            "{}.layer_{}.ple.input",
            state.output_source_prefix, context.layer_idx
        )
    )?;
    let (artifact_store_roots, layer_output_ref, updated_cache) = call_seq!(
        run_basic_decode_layer_with_roots,
        artifact_store_roots,
        state.current_activation_ref.clone(),
        source,
        &context.layer,
        context.cache_slot,
        context.donor_cache_slot.as_ref(),
        per_layer_input_ref,
        state.position,
        state.projection_rows_per_tile,
        state.attention_kv_rows_per_tile,
        format!("{}.layer_{}", state.output_source_prefix, context.layer_idx)
    )?;

    call_tile!(
        update_decode_layer_state_refs_with_roots,
        artifact_store_roots,
        state,
        context.layer_idx,
        layer_output_ref,
        updated_cache
    )
}

#[tile]
fn update_decode_layer_state_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut state: DecodeTransitionRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    updated_cache: DecodeLayerCacheSlot,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    if layer_idx != state.next_layer_idx {
        bail!(
            "cannot update decode layer {layer_idx} while next layer is {}",
            state.next_layer_idx
        );
    }
    let layer_output =
        read_activation_row_from_ref_roots(&artifact_store_roots, &layer_output_ref)?;
    state.artifact_store_roots = artifact_store_roots.clone();
    state.current_activation_ref = layer_output_ref;
    state.updated_layer_caches.push(updated_cache);
    let current_activation_values = layer_output.to_f32_values();
    let current_activation_acts = layer_output.acts();
    state.completed_layer_output_sha256s.push(
        crate::shared::numerics::transformer_kernels::build_vector_commitment(
            &current_activation_values,
        ),
    );
    state.completed_layer_output_det_sha256s.push(Some(
        crate::shared::numerics::transformer_kernels::build_det_vector_commitment(
            &current_activation_acts,
        ),
    ));
    trace_decode_layer_checkpoint_with_roots(&artifact_store_roots, &state, layer_idx)?;
    state.next_layer_idx += 1;
    Ok((false, artifact_store_roots, state))
}

#[sequence]
fn run_basic_decode_layer_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache_slot: DecodeLayerCacheSlot,
    donor_cache_slot: Option<&DecodeLayerCacheSlot>,
    per_layer_input_ref: Option<RasterActivationSequenceRef>,
    position: usize,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    output_prefix: String,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceRef,
    DecodeLayerCacheSlot,
)> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer.layer_idx)?;
    let (artifact_store_roots, xs_ref, updated_cache) = call_seq!(
        run_decode_attention_block_with_roots,
        artifact_store_roots,
        input_ref,
        source,
        layer,
        cache_slot,
        donor_cache_slot,
        position,
        projection_rows_per_tile,
        attention_kv_rows_per_tile,
        format!("{output_prefix}.attention_block")
    )?;
    let (artifact_store_roots, xs_ref) = call_seq!(
        run_decode_mlp_block_with_roots,
        artifact_store_roots,
        xs_ref,
        source,
        layer,
        scalars.rms_norm_eps,
        projection_rows_per_tile,
        format!("{output_prefix}.mlp")
    )?;
    let (artifact_store_roots, xs_ref) = if let Some(per_layer_input_ref) = per_layer_input_ref {
        call_seq!(
            run_decode_ple_block_with_roots,
            artifact_store_roots,
            xs_ref,
            per_layer_input_ref,
            source,
            layer,
            scalars.rms_norm_eps,
            projection_rows_per_tile,
            format!("{output_prefix}.ple")
        )?
    } else {
        (artifact_store_roots, xs_ref)
    };
    let (artifact_store_roots, xs_ref) = call_tile!(
        scale_decode_ref_optional_with_roots,
        artifact_store_roots,
        xs_ref,
        scalars.layer_scalar,
        format!("{output_prefix}.scaled")
    )?;

    Ok((artifact_store_roots, xs_ref, updated_cache))
}

#[sequence]
fn run_decode_attention_block_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache_slot: DecodeLayerCacheSlot,
    donor_cache_slot: Option<&DecodeLayerCacheSlot>,
    position: usize,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    output_prefix: String,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceRef,
    DecodeLayerCacheSlot,
)> {
    let input = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    call_tile!(validate_decode_attention_context, &input, layer)?;
    let (artifact_store_roots, normed_ref) = call_seq!(
        rms_norm_decode_layer_ref_with_roots,
        artifact_store_roots,
        input_ref.clone(),
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::InputLayer,
        format!("{output_prefix}.input_norm")
    )?;
    let (artifact_store_roots, attention_output_ref, updated_cache) = call_seq!(
        run_decode_attention_with_roots,
        artifact_store_roots,
        normed_ref,
        source,
        layer,
        cache_slot,
        donor_cache_slot,
        position,
        projection_rows_per_tile,
        attention_kv_rows_per_tile,
        format!("{output_prefix}.attention")
    )?;
    let (artifact_store_roots, attention_output_ref) = call_seq!(
        rms_norm_decode_layer_ref_with_roots,
        artifact_store_roots,
        attention_output_ref,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PostAttention,
        format!("{output_prefix}.post_attention_norm")
    )?;
    let (artifact_store_roots, xs_ref) = call_tile!(
        add_decode_refs_with_roots,
        artifact_store_roots,
        input_ref,
        attention_output_ref,
        format!("{output_prefix}.residual")
    )?;
    Ok((artifact_store_roots, xs_ref, updated_cache))
}

#[sequence]
fn run_decode_mlp_block_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    rms_norm_eps: Acc,
    projection_rows_per_tile: usize,
    output_prefix: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let residual_ref = input_ref.clone();
    let norm_weights = call_tile!(
        read_decode_layer_norm_weights,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PreFeedForward
    )?;
    let (artifact_store_roots, normed_ref) = call_tile!(
        rms_norm_decode_ref_with_roots,
        artifact_store_roots,
        input_ref,
        &norm_weights,
        rms_norm_eps,
        "decode MLP pre-feedforward RMSNorm",
        format!("{output_prefix}.pre_ff_norm")
    )?;
    let (artifact_store_roots, gate_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        normed_ref.clone(),
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Gate,
        },
        layer.gate_proj_shape.rows,
        projection_rows_per_tile,
        format!("{output_prefix}.gate_proj"),
        None
    )?;
    let (artifact_store_roots, gate_ref) = call_tile!(
        gelu_decode_ref_with_roots,
        artifact_store_roots,
        gate_ref,
        format!("{output_prefix}.gate_gelu")
    )?;
    let (artifact_store_roots, up_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        normed_ref,
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Up,
        },
        layer.up_proj_shape.rows,
        projection_rows_per_tile,
        format!("{output_prefix}.up_proj"),
        None
    )?;
    let (artifact_store_roots, ff_hidden_ref) = call_tile!(
        mul_decode_refs_with_roots,
        artifact_store_roots,
        gate_ref,
        up_ref,
        format!("{output_prefix}.ff_hidden")
    )?;
    let (artifact_store_roots, ff_out_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        ff_hidden_ref,
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Down,
        },
        layer.down_proj_shape.rows,
        projection_rows_per_tile,
        format!("{output_prefix}.down_proj"),
        None
    )?;
    let norm_weights = call_tile!(
        read_decode_layer_norm_weights,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PostFeedForward
    )?;
    let (artifact_store_roots, ff_out_ref) = call_tile!(
        rms_norm_decode_ref_with_roots,
        artifact_store_roots,
        ff_out_ref,
        &norm_weights,
        rms_norm_eps,
        "decode MLP post-feedforward RMSNorm",
        format!("{output_prefix}.post_ff_norm")
    )?;
    call_tile!(
        add_decode_refs_with_roots,
        artifact_store_roots,
        residual_ref,
        ff_out_ref,
        format!("{output_prefix}.residual")
    )
}

#[sequence]
fn run_decode_ple_block_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    per_layer_input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    rms_norm_eps: Acc,
    projection_rows_per_tile: usize,
    output_prefix: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let residual_ref = input_ref.clone();
    let (artifact_store_roots, gated_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        input_ref,
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::PleInputGate,
        },
        call_tile!(decode_ple_input_gate_rows, layer)?,
        projection_rows_per_tile,
        format!("{output_prefix}.input_gate"),
        None
    )?;
    let (artifact_store_roots, gated_ref) = call_tile!(
        gelu_decode_ref_with_roots,
        artifact_store_roots,
        gated_ref,
        format!("{output_prefix}.input_gate_gelu")
    )?;
    let (artifact_store_roots, gated_ref) = call_tile!(
        mul_decode_refs_with_roots,
        artifact_store_roots,
        gated_ref,
        per_layer_input_ref,
        format!("{output_prefix}.gated")
    )?;
    let (artifact_store_roots, projected_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        gated_ref,
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::PleLayerProjection,
        },
        call_tile!(decode_ple_layer_projection_rows, layer)?,
        projection_rows_per_tile,
        format!("{output_prefix}.layer_projection"),
        None
    )?;
    let norm_weights = call_tile!(
        read_decode_layer_norm_weights,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PlePostInput
    )?;
    let (artifact_store_roots, projected_ref) = call_tile!(
        rms_norm_decode_ref_with_roots,
        artifact_store_roots,
        projected_ref,
        &norm_weights,
        rms_norm_eps,
        "decode PLE post-input RMSNorm",
        format!("{output_prefix}.post_input_norm")
    )?;
    call_tile!(
        add_decode_refs_with_roots,
        artifact_store_roots,
        residual_ref,
        projected_ref,
        format!("{output_prefix}.residual")
    )
}

#[sequence]
fn compute_decode_ple_input_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    token_id: u32,
    decode_input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    projection_rows_per_tile: usize,
    output_prefix: String,
) -> Result<(
    RasterArtifactStoreRoots,
    Option<RasterActivationSequenceRef>,
)> {
    if !layer.has_ple {
        return Ok((artifact_store_roots, None));
    }

    let ple_width = call_tile!(decode_ple_input_gate_rows, layer)?;
    let scalars = call_tile!(read_decode_ple_scalars, source)?;
    let norm_weights = call_tile!(read_decode_ple_projection_norm_weights, source)?;
    let embedded = call_tile!(
        read_decode_ple_token_embedding,
        source,
        layer.layer_idx,
        token_id
    )?;
    let embedded = call_tile!(
        scale_decode_row_optional,
        &embedded,
        Some(scalars.embedding_scale)
    )?;
    let (artifact_store_roots, embedded_ref) = insert_decode_activation_row_with_roots(
        &artifact_store_roots,
        format!("{output_prefix}.token_embedding"),
        &embedded,
    )?;

    let (artifact_store_roots, projected_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        decode_input_ref,
        source,
        DecodeProjectionKind::PleModel {
            layer_idx: layer.layer_idx,
        },
        ple_width,
        projection_rows_per_tile,
        format!("{output_prefix}.model_projection"),
        None
    )?;
    let (artifact_store_roots, projected_ref) = call_tile!(
        scale_decode_ref_optional_with_roots,
        artifact_store_roots,
        projected_ref,
        Some(scalars.projection_scalar),
        format!("{output_prefix}.projection_scaled")
    )?;
    let (artifact_store_roots, projected_ref) = call_tile!(
        rms_norm_decode_ref_with_roots,
        artifact_store_roots,
        projected_ref,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode PLE input RMSNorm",
        format!("{output_prefix}.projection_norm")
    )?;
    let (artifact_store_roots, combined_ref) = call_tile!(
        add_decode_refs_with_roots,
        artifact_store_roots,
        embedded_ref,
        projected_ref,
        format!("{output_prefix}.combined")
    )?;
    let (artifact_store_roots, input_ref) = call_tile!(
        scale_decode_ref_optional_with_roots,
        artifact_store_roots,
        combined_ref,
        Some(scalars.input_scale),
        format!("{output_prefix}.scaled")
    )?;
    Ok((artifact_store_roots, Some(input_ref)))
}

#[sequence]
fn run_decode_attention_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache_slot: DecodeLayerCacheSlot,
    donor_cache_slot: Option<&DecodeLayerCacheSlot>,
    position: usize,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    output_prefix: String,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceRef,
    DecodeLayerCacheSlot,
)> {
    let (artifact_store_roots, q_projected_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        input_ref.clone(),
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Query,
        },
        layer.q_proj_shape.rows,
        projection_rows_per_tile,
        format!("{output_prefix}.q_proj"),
        None
    )?;
    let (artifact_store_roots, raw_k_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        input_ref.clone(),
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Key,
        },
        layer.k_proj_shape.rows,
        projection_rows_per_tile,
        format!("{output_prefix}.k_proj"),
        None
    )?;
    let (artifact_store_roots, raw_v_ref) = if layer.has_v_proj {
        call_seq!(
            project_ref_with_decode_source_with_roots,
            artifact_store_roots,
            input_ref,
            source,
            DecodeProjectionKind::LayerMatrix {
                layer_idx: layer.layer_idx,
                matrix: GemmaDecodeLayerMatrixKind::Value,
            },
            layer
                .v_proj_shape
                .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing v_proj shape"))?
                .rows,
            projection_rows_per_tile,
            format!("{output_prefix}.v_proj"),
            None
        )?
    } else if layer.attention_k_eq_v {
        (artifact_store_roots, raw_k_ref.clone())
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };

    let (artifact_store_roots, q_heads_ref) = call_tile!(
        reshape_decode_ref_heads_with_roots,
        artifact_store_roots,
        q_projected_ref,
        format!("{output_prefix}.q_heads"),
        layer.num_heads,
        layer.head_dim
    )?;
    let (artifact_store_roots, k_heads_ref) = call_tile!(
        reshape_decode_ref_heads_with_roots,
        artifact_store_roots,
        raw_k_ref,
        format!("{output_prefix}.k_heads"),
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let (artifact_store_roots, v_heads_ref) = call_tile!(
        reshape_decode_ref_heads_with_roots,
        artifact_store_roots,
        raw_v_ref,
        format!("{output_prefix}.v_heads"),
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let (artifact_store_roots, q_heads_ref) = call_seq!(
        rms_norm_decode_attention_heads_ref_with_roots,
        artifact_store_roots,
        q_heads_ref,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::Query,
        format!("{output_prefix}.q_norm")
    )?;
    let (artifact_store_roots, k_heads_ref) = call_seq!(
        rms_norm_decode_attention_heads_ref_with_roots,
        artifact_store_roots,
        k_heads_ref,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::Key,
        format!("{output_prefix}.k_norm")
    )?;
    let (artifact_store_roots, v_heads_ref) = call_seq!(
        value_rms_norm_decode_attention_heads_ref_with_roots,
        artifact_store_roots,
        v_heads_ref,
        source,
        layer.layer_idx,
        format!("{output_prefix}.v_norm")
    )?;
    let scalars = call_tile!(read_decode_layer_scalars, source, layer.layer_idx)?;
    let (artifact_store_roots, q_heads_ref) = call_tile!(
        apply_decode_rope_to_heads_ref_with_roots,
        artifact_store_roots,
        q_heads_ref,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        position,
        format!("{output_prefix}.q_rope")
    )?;
    let (artifact_store_roots, k_heads_ref) = call_tile!(
        apply_decode_rope_to_heads_ref_with_roots,
        artifact_store_roots,
        k_heads_ref,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        position,
        format!("{output_prefix}.k_rope")
    )?;

    let (artifact_store_roots, updated_cache_slot) = call_seq!(
        update_decode_attention_cache_with_roots,
        artifact_store_roots,
        cache_slot,
        donor_cache_slot.is_some(),
        k_heads_ref,
        v_heads_ref,
        layer.layer_idx,
        layer.cache_sliding_window,
        attention_kv_rows_per_tile,
        format!("{output_prefix}.updated_cache")
    )?;
    let attention_cache_slot = donor_cache_slot.unwrap_or(&updated_cache_slot);
    let attention_cache_ref = call_tile!(resolve_decode_attention_cache_ref, attention_cache_slot)?;
    let attention_window = call_tile!(resolve_decode_attention_window, layer)?;
    let (artifact_store_roots, attention_heads_ref) = call_seq!(
        compute_decode_attention_ref_with_roots,
        artifact_store_roots,
        q_heads_ref,
        attention_cache_ref,
        format!("{output_prefix}.scores"),
        attention_window,
        attention_kv_rows_per_tile
    )?;
    let (artifact_store_roots, attention_ref) = call_seq!(
        combine_decode_attention_heads_ref_with_roots,
        artifact_store_roots,
        attention_heads_ref,
        format!("{output_prefix}.combined")
    )?;
    let (artifact_store_roots, projected_ref) = call_seq!(
        project_ref_with_decode_source_with_roots,
        artifact_store_roots,
        attention_ref,
        source,
        DecodeProjectionKind::LayerMatrix {
            layer_idx: layer.layer_idx,
            matrix: GemmaDecodeLayerMatrixKind::Output,
        },
        layer.o_proj_shape.rows,
        projection_rows_per_tile,
        format!("{output_prefix}.o_proj"),
        None
    )?;

    Ok((artifact_store_roots, projected_ref, updated_cache_slot))
}

#[sequence]
fn project_ref_with_decode_source_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    projection_kind: DecodeProjectionKind,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: String,
    softcap_bits: Option<i32>,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, state) = call_tile!(
        init_decode_row_projection_artifact,
        artifact_store_roots,
        input_ref,
        projection_kind,
        projection_rows,
        rows_per_tile,
        RasterTensorId::new(output_id)?,
        softcap_bits
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        project_next_decode_projection_chunk_with_roots,
        (artifact_store_roots, state),
        source
    )?;
    call_tile!(
        finalize_decode_row_projection_ref_with_roots,
        artifact_store_roots,
        state
    )
}

#[tile]
pub fn init_decode_row_projection_artifact(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    projection_kind: DecodeProjectionKind,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: RasterTensorId,
    softcap_bits: Option<i32>,
) -> Result<(RasterArtifactStoreRoots, DecodeRowProjectionArtifactState)> {
    if projection_rows == 0 {
        bail!("deterministic decode projection requires at least one projection row");
    }
    validate_projection_rows_per_tile(rows_per_tile)?;
    let (row_count, input_width) = input_ref.tensor_ref().shape().sequence_metadata()?;
    if row_count != 1 {
        bail!("decode row projection expects a one-row input ref, got {row_count}");
    }
    let output_source_name = output_id.source_name().to_string();
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&output_source_name)?,
        1,
        projection_rows,
    )?;
    Ok((
        artifact_store_roots,
        DecodeRowProjectionArtifactState {
            input_ref,
            projection_kind,
            output_source_name,
            current_row_bits: Vec::new(),
            next_projection_row_idx: 0,
            projection_rows,
            input_width,
            rows_per_tile,
            softcap_bits,
        },
    ))
}

#[tile(kind = recursive)]
pub fn project_next_decode_projection_chunk_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut state: DecodeRowProjectionArtifactState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    DecodeRowProjectionArtifactState,
)> {
    if state.next_projection_row_idx >= state.projection_rows {
        return Ok((true, artifact_store_roots, state));
    }
    let input = read_activation_row_from_ref_roots(&artifact_store_roots, &state.input_ref)?;
    if input.width() != state.input_width {
        bail!(
            "decode projection input row has width {}, expected {}",
            input.width(),
            state.input_width
        );
    }
    let end = state
        .next_projection_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.projection_rows);
    while state.next_projection_row_idx < end {
        let projection_row = read_decode_projection_row(
            source,
            &state.projection_kind,
            state.next_projection_row_idx,
        )?;
        if projection_row.len() != state.input_width {
            bail!(
                "decode projection row {} has width {}, expected {}",
                state.next_projection_row_idx,
                projection_row.len(),
                state.input_width
            );
        }
        let mut projected = project_row_with_weights(&input, &projection_row)?;
        if let Some(softcap_bits) = state.softcap_bits {
            projected = softcap_act(projected, Act::from_bits(softcap_bits));
        }
        state.current_row_bits.push(projected.to_bits());
        state.next_projection_row_idx += 1;
    }
    if state.next_projection_row_idx == state.projection_rows {
        let row_bits = std::mem::take(&mut state.current_row_bits);
        let artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &artifact_store_roots,
            &state.output_source_name,
            0,
            RasterActivationRow::from_act_bits(row_bits),
        )?;
        return Ok((true, artifact_store_roots, state));
    }
    Ok((false, artifact_store_roots, state))
}

#[tile]
pub fn finalize_decode_row_projection_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: DecodeRowProjectionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if state.next_projection_row_idx != state.projection_rows {
        bail!(
            "raster decode projection completed {} rows, expected {}",
            state.next_projection_row_idx,
            state.projection_rows
        );
    }
    if !state.current_row_bits.is_empty() {
        bail!(
            "raster decode projection finalized with partial row width {}",
            state.current_row_bits.len()
        );
    }
    finalize_sequence_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        RasterTensorId::new(state.output_source_name.clone())?,
    )
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
    let row = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    let row = call_tile!(rms_norm_decode_row, &row, norm_weights, eps, label)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

#[sequence]
fn rms_norm_decode_layer_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    norm: GemmaDecodeLayerNormKind,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer_idx)?;
    let norm_weights = call_tile!(read_decode_layer_norm_weights, source, layer_idx, norm)?;
    call_tile!(
        rms_norm_decode_ref_with_roots,
        artifact_store_roots,
        input_ref,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode layer RMSNorm",
        output_source_name
    )
}

#[tile]
fn scale_decode_ref_optional_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    scalar: Option<Act>,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if scalar.is_none() {
        return Ok((artifact_store_roots, input_ref));
    }
    let row = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    let row = call_tile!(scale_decode_row_optional, &row, scalar)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

#[tile]
fn add_decode_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let lhs = read_activation_row_from_ref_roots(&artifact_store_roots, &lhs_ref)?;
    let rhs = read_activation_row_from_ref_roots(&artifact_store_roots, &rhs_ref)?;
    let row = call_tile!(add_decode_rows, &lhs, &rhs)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

#[tile]
fn mul_decode_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let lhs = read_activation_row_from_ref_roots(&artifact_store_roots, &lhs_ref)?;
    let rhs = read_activation_row_from_ref_roots(&artifact_store_roots, &rhs_ref)?;
    let row = call_tile!(mul_decode_rows, &lhs, &rhs)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

#[tile]
fn gelu_decode_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let row = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    let row = call_tile!(gelu_decode_row, &row)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
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

#[sequence]
fn rms_norm_decode_attention_heads_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    norm: GemmaDecodeLayerNormKind,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer_idx)?;
    let norm_weights = call_tile!(read_decode_layer_norm_weights, source, layer_idx, norm)?;
    let heads = materialize_attention_heads_from_roots(&artifact_store_roots, &heads_ref)?;
    let heads = call_tile!(
        rms_norm_decode_heads,
        &heads,
        &norm_weights,
        scalars.rms_norm_eps
    )?;
    insert_attention_heads_with_roots(&artifact_store_roots, output_source_name, heads)
}

#[sequence]
fn value_rms_norm_decode_attention_heads_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer_idx)?;
    let heads = materialize_attention_heads_from_roots(&artifact_store_roots, &heads_ref)?;
    let heads = call_tile!(value_rms_norm_decode_heads, &heads, scalars.rms_norm_eps)?;
    insert_attention_heads_with_roots(&artifact_store_roots, output_source_name, heads)
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
    let heads = call_tile!(
        apply_decode_rope_to_heads,
        &heads,
        partial_rotary_dim,
        rope_freq_base_dim,
        rope_base,
        position
    )?;
    insert_attention_heads_with_roots(&artifact_store_roots, output_source_name, heads)
}

#[sequence]
fn combine_decode_attention_heads_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    output_id: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, state) =
        crate::shared::raster_kernels::transformer::init_combine_heads_artifact_state_from_ref(
            artifact_store_roots,
            heads_ref,
            RasterTensorId::new(output_id)?,
        )?;
    let mut artifact_store_roots = artifact_store_roots;
    let mut state = state;
    loop {
        let (done, next_roots, next_state) =
            crate::shared::raster_kernels::transformer::compute_next_combine_heads_artifact_row(
                artifact_store_roots,
                state,
            )?;
        artifact_store_roots = next_roots;
        state = next_state;
        if done {
            break;
        }
    }
    crate::shared::raster_kernels::transformer::finalize_combine_heads_artifact_state_ref(
        artifact_store_roots,
        state,
    )
}

#[sequence]
fn compute_decode_attention_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    query_ref: RasterAttentionHeadsRef,
    cache_ref: RasterKvCacheRef,
    id_prefix: String,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let (artifact_store_roots, state) = call_tile!(
        init_decode_attention_artifact_state_from_refs,
        artifact_store_roots,
        query_ref,
        cache_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        id_prefix,
        attention_window,
        kv_rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        compute_next_decode_attention_head_with_roots,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_decode_attention_state_ref_with_roots,
        artifact_store_roots,
        state
    )
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

fn init_decode_attention_score_phase_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    id_prefix: &str,
    query_head_idx: usize,
    visible_row_count: usize,
) -> Result<(RasterArtifactStoreRoots, DecodeAttentionArtifactPhase)> {
    let score_source_name = format!("{id_prefix}.scores.head_{query_head_idx}");
    let artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(&score_source_name)?,
        visible_row_count,
        1,
    )?;
    Ok((
        artifact_store_roots,
        DecodeAttentionArtifactPhase::CollectScores {
            score_source_name,
            next_kv_offset: 0,
        },
    ))
}

#[tile(kind = recursive)]
pub fn compute_next_decode_attention_head_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut state: DecodeAttentionArtifactState,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeAttentionArtifactState)> {
    if state.next_query_head_idx >= state.query_head_count {
        return Ok((true, artifact_store_roots, state));
    }

    let mut artifact_store_roots = artifact_store_roots;
    let query_head_idx = state.next_query_head_idx;
    let kv_head_idx = query_head_idx / state.kv_groups;
    let query = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: state.query_ref.clone(),
            head_idx: query_head_idx,
            token_idx: 0,
        },
    )?;

    match state.phase.clone() {
        DecodeAttentionArtifactPhase::CollectScores {
            score_source_name,
            next_kv_offset,
        } => {
            let end = next_kv_offset
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
            for offset in next_kv_offset..end {
                let token_idx = state.key_start + offset;
                let key_row = read_decode_attention_kv_row_from_roots(
                    &artifact_store_roots,
                    &state.cache_ref,
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
            if end < state.row_count {
                state.phase = DecodeAttentionArtifactPhase::CollectScores {
                    score_source_name,
                    next_kv_offset: end,
                };
                return Ok((false, artifact_store_roots, state));
            }
            let (roots, score_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &score_source_name,
                RasterTensorId::new(score_source_name.clone())?,
            )?;
            artifact_store_roots = roots;
            state.phase = DecodeAttentionArtifactPhase::FindSoftmaxMax {
                score_ref,
                next_score_row_idx: 0,
                max_index: None,
                max_logit_bits: 0,
            };
            Ok((false, artifact_store_roots, state))
        }
        DecodeAttentionArtifactPhase::FindSoftmaxMax {
            score_ref,
            next_score_row_idx,
            mut max_index,
            mut max_logit_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
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
            if end < state.row_count {
                state.phase = DecodeAttentionArtifactPhase::FindSoftmaxMax {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }
            let max_index = max_index
                .ok_or_else(|| anyhow!("decode attention softmax requires at least one score"))?;
            state.phase = DecodeAttentionArtifactPhase::SumSoftmaxExp {
                score_ref,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits: 0,
            };
            Ok((false, artifact_store_roots, state))
        }
        DecodeAttentionArtifactPhase::SumSoftmaxExp {
            score_ref,
            next_score_row_idx,
            max_index,
            max_logit_bits,
            mut sum_exp_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
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
            if end < state.row_count {
                state.phase = DecodeAttentionArtifactPhase::SumSoftmaxExp {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }
            if sum_exp_bits == 0 {
                bail!("decode attention softmax exp sum is zero");
            }
            let raw_weight_source_name = format!(
                "{}.raw_weights.head_{query_head_idx}",
                state.attention_id_prefix
            );
            artifact_store_roots = start_sequence_builder_with_roots(
                &artifact_store_roots,
                RasterArtifactId::new(&raw_weight_source_name)?,
                state.row_count,
                1,
            )?;
            state.phase = DecodeAttentionArtifactPhase::BuildRawSoftmaxWeights {
                score_ref,
                raw_weight_source_name,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits,
                summed_weight_bits: 0,
            };
            Ok((false, artifact_store_roots, state))
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
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
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
            if end < state.row_count {
                state.phase = DecodeAttentionArtifactPhase::BuildRawSoftmaxWeights {
                    score_ref,
                    raw_weight_source_name,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                    summed_weight_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }
            let (roots, raw_weight_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &raw_weight_source_name,
                RasterTensorId::new(raw_weight_source_name.clone())?,
            )?;
            artifact_store_roots = roots;
            let final_weight_source_name = format!(
                "{}.weights.head_{query_head_idx}",
                state.attention_id_prefix
            );
            artifact_store_roots = start_sequence_builder_with_roots(
                &artifact_store_roots,
                RasterArtifactId::new(&final_weight_source_name)?,
                state.row_count,
                1,
            )?;
            let residual = attention_softmax_residual(Act::from_bits(summed_weight_bits));
            state.phase = DecodeAttentionArtifactPhase::CorrectSoftmaxResidual {
                raw_weight_ref,
                final_weight_source_name,
                next_weight_row_idx: 0,
                max_index,
                residual_bits: residual.to_bits(),
            };
            Ok((false, artifact_store_roots, state))
        }
        DecodeAttentionArtifactPhase::CorrectSoftmaxResidual {
            raw_weight_ref,
            final_weight_source_name,
            next_weight_row_idx,
            max_index,
            residual_bits,
        } => {
            let end = next_weight_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
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
            if end < state.row_count {
                state.phase = DecodeAttentionArtifactPhase::CorrectSoftmaxResidual {
                    raw_weight_ref,
                    final_weight_source_name,
                    next_weight_row_idx: end,
                    max_index,
                    residual_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }
            let (roots, weight_ref) = finalize_sequence_builder_by_source_name_with_roots(
                &artifact_store_roots,
                &final_weight_source_name,
                RasterTensorId::new(final_weight_source_name.clone())?,
            )?;
            artifact_store_roots = roots;
            state.phase = DecodeAttentionArtifactPhase::ApplyValues {
                weight_ref,
                next_kv_offset: 0,
                weighted_sum_acc_bits: vec![0; query.width()],
            };
            Ok((false, artifact_store_roots, state))
        }
        DecodeAttentionArtifactPhase::ApplyValues {
            weight_ref,
            next_kv_offset,
            mut weighted_sum_acc_bits,
        } => {
            let end = next_kv_offset
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
            for offset in next_kv_offset..end {
                let token_idx = state.key_start + offset;
                let weight = read_decode_attention_scalar_row_from_roots(
                    &artifact_store_roots,
                    &weight_ref,
                    offset,
                    "weight",
                )?;
                let value_row = read_decode_attention_kv_row_from_roots(
                    &artifact_store_roots,
                    &state.cache_ref,
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
            if end < state.row_count {
                state.phase = DecodeAttentionArtifactPhase::ApplyValues {
                    weight_ref,
                    next_kv_offset: end,
                    weighted_sum_acc_bits,
                };
                return Ok((false, artifact_store_roots, state));
            }
            let output_row = RasterActivationRow::from_acts(
                weighted_sum_acc_bits
                    .into_iter()
                    .map(|bits| requantize(Acc::from_bits(bits)))
                    .collect(),
            );
            artifact_store_roots = append_head_row_by_source_name_with_roots(
                &artifact_store_roots,
                &state.output_source_name,
                query_head_idx,
                0,
                1,
                output_row,
            )?;
            state.next_query_head_idx += 1;
            if state.next_query_head_idx < state.query_head_count {
                let (roots, phase) = init_decode_attention_score_phase_with_roots(
                    artifact_store_roots,
                    &state.attention_id_prefix,
                    state.next_query_head_idx,
                    state.row_count,
                )?;
                artifact_store_roots = roots;
                state.phase = phase;
            }
            Ok((false, artifact_store_roots, state))
        }
    }
}

#[tile]
pub fn finalize_decode_attention_state_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: DecodeAttentionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    if state.next_query_head_idx != state.query_head_count {
        bail!(
            "decode attention finalized at head {}, expected {} heads",
            state.next_query_head_idx,
            state.query_head_count
        );
    }
    finalize_heads_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &state.output_source_name,
        RasterTensorId::new(state.output_source_name.clone())?,
        state.query_head_count,
        1,
        state.head_dim,
    )
}

#[sequence]
fn update_decode_attention_cache_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    cache_slot: DecodeLayerCacheSlot,
    use_donor_cache: bool,
    k_heads_ref: RasterAttentionHeadsRef,
    v_heads_ref: RasterAttentionHeadsRef,
    layer_idx: usize,
    cache_window: Option<usize>,
    rows_per_tile: usize,
    output_prefix: String,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerCacheSlot)> {
    if use_donor_cache {
        return Ok((artifact_store_roots, cache_slot));
    }
    call_seq!(
        append_decode_kv_cache_ref_with_roots,
        artifact_store_roots,
        cache_slot,
        k_heads_ref,
        v_heads_ref,
        layer_idx,
        cache_window,
        rows_per_tile,
        output_prefix
    )
}

#[sequence]
fn append_decode_kv_cache_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    cache_slot: DecodeLayerCacheSlot,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    layer_idx: usize,
    cache_window: Option<usize>,
    rows_per_tile: usize,
    output_prefix: String,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerCacheSlot)> {
    let (artifact_store_roots, state) = call_tile!(
        init_decode_kv_cache_append_state_with_roots,
        artifact_store_roots,
        cache_slot,
        key_ref,
        value_ref,
        RasterTensorId::new(format!("{output_prefix}.{layer_idx}.keys"))?,
        RasterTensorId::new(format!("{output_prefix}.{layer_idx}.values"))?,
        cache_window,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        compute_next_decode_kv_cache_append_row_with_roots,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_decode_kv_cache_append_state_with_roots,
        artifact_store_roots,
        state
    )
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
    mut state: DecodeKvCacheAppendArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    DecodeKvCacheAppendArtifactState,
)> {
    if state.next_head_idx >= state.head_count {
        return Ok((true, artifact_store_roots, state));
    }
    let mut artifact_store_roots = artifact_store_roots;

    if state.next_old_offset < state.retained_old_len {
        let old_cache_ref = state.old_cache_ref.as_ref().ok_or_else(|| {
            anyhow!("decode cache append has retained rows without old cache ref")
        })?;
        let end = state
            .next_old_offset
            .saturating_add(state.rows_per_tile)
            .min(state.retained_old_len);
        for old_offset in state.next_old_offset..end {
            let input_token_idx = state.retained_old_start + old_offset;
            let key_row = read_kv_row_from_roots(
                &artifact_store_roots,
                RasterKvRowRequest {
                    cache_ref: old_cache_ref.clone(),
                    row_kind: RasterKvRowKind::Key,
                    head_idx: state.next_head_idx,
                    token_idx: input_token_idx,
                },
            )?;
            let value_row = read_kv_row_from_roots(
                &artifact_store_roots,
                RasterKvRowRequest {
                    cache_ref: old_cache_ref.clone(),
                    row_kind: RasterKvRowKind::Value,
                    head_idx: state.next_head_idx,
                    token_idx: input_token_idx,
                },
            )?;
            artifact_store_roots = append_head_row_by_source_name_with_roots(
                &artifact_store_roots,
                &state.keys_source_name,
                state.next_head_idx,
                old_offset,
                state.current_len,
                key_row,
            )?;
            artifact_store_roots = append_head_row_by_source_name_with_roots(
                &artifact_store_roots,
                &state.values_source_name,
                state.next_head_idx,
                old_offset,
                state.current_len,
                value_row,
            )?;
        }
        state.next_old_offset = end;
        if state.next_old_offset < state.retained_old_len {
            return Ok((false, artifact_store_roots, state));
        }
    }

    let output_token_idx = state.retained_old_len;
    let key_row = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: state.key_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: 0,
        },
    )?;
    let value_row = read_head_row_from_roots(
        &artifact_store_roots,
        RasterHeadRowRequest {
            tensor_ref: state.value_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: 0,
        },
    )?;
    artifact_store_roots = append_head_row_by_source_name_with_roots(
        &artifact_store_roots,
        &state.keys_source_name,
        state.next_head_idx,
        output_token_idx,
        state.current_len,
        key_row,
    )?;
    artifact_store_roots = append_head_row_by_source_name_with_roots(
        &artifact_store_roots,
        &state.values_source_name,
        state.next_head_idx,
        output_token_idx,
        state.current_len,
        value_row,
    )?;
    state.next_head_idx += 1;
    state.next_old_offset = 0;
    Ok((false, artifact_store_roots, state))
}

#[tile]
fn finalize_decode_kv_cache_append_state_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: DecodeKvCacheAppendArtifactState,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerCacheSlot)> {
    if state.next_head_idx != state.head_count {
        bail!(
            "decode cache append finalized at head {}, expected {} heads",
            state.next_head_idx,
            state.head_count
        );
    }
    let (artifact_store_roots, cache_ref) = finalize_kv_cache_builders_by_source_name_with_roots(
        &artifact_store_roots,
        &state.keys_source_name,
        &state.values_source_name,
        RasterTensorId::new(state.keys_source_name.clone())?,
        RasterTensorId::new(state.values_source_name.clone())?,
        state.head_count,
        state.current_len,
        state.head_dim,
    )?;
    Ok((artifact_store_roots, DecodeLayerCacheSlot::Ref(cache_ref)))
}

#[tile]
fn normalize_decode_final_position_ref_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    final_hidden_states_ref: RasterActivationSequenceRef,
    source: &AuthenticatedGemmaDecodeTransitionSource,
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
fn decode_projection_row_count(source: &AuthenticatedGemmaDecodeTransitionSource) -> Result<usize> {
    Ok(auth_read!(source, GemmaDecodeTransitionMetadataRequest)?.projection_rows)
}

#[tile]
fn decode_final_logit_softcap_bits(
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<Option<i32>> {
    Ok(auth_read!(source, GemmaDecodeFinalScalarsRequest)?
        .final_logit_softcapping
        .map(Act::to_bits))
}

#[tile]
fn finalize_decode_transition_result_from_roots(
    artifact_store_roots: &RasterArtifactStoreRoots,
    logits_ref: RasterActivationSequenceRef,
    transformer_decode_state: TransformerDecodeState,
    final_hidden_state: ActivationSequence,
) -> Result<TransformerDecodeStepResult> {
    let logits_row = read_activation_row_from_ref_roots(artifact_store_roots, &logits_ref)?;
    let det_logits = logits_row.acts();
    let internal_logits = InternalLogits::from_det_values(det_logits.clone());
    let final_logits_sha256 = crate::shared::numerics::transformer_kernels::build_vector_commitment(
        internal_logits.as_f32_slice(),
    );
    let mut prefill_logits = PrefillLogits::from_internal(internal_logits, final_logits_sha256);
    prefill_logits.det_final_logits_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_vector_commitment(&det_logits),
    );

    Ok(TransformerDecodeStepResult {
        transformer_decode_state: TransformerDecodeState {
            layer_caches: transformer_decode_state.layer_caches,
            position: transformer_decode_state.position + 1,
            token_count: transformer_decode_state.token_count + 1,
        },
        activation_state: final_hidden_state,
        prefill_logits,
    })
}

#[tile]
pub fn finalize_decode_layer_state_with_roots(
    roots: &RasterArtifactStoreRoots,
    state: DecodeTransitionRasterState,
) -> Result<ActivationSequenceWithCache> {
    if state.next_layer_idx != state.layer_count {
        bail!(
            "raster decode finalized after {} layers, expected {}",
            state.next_layer_idx,
            state.layer_count
        );
    }
    if state.updated_layer_caches.len() != state.layer_count {
        bail!(
            "raster decode stored {} layer caches, expected {}",
            state.updated_layer_caches.len(),
            state.layer_count
        );
    }

    let current_activation =
        read_activation_row_from_ref_roots(roots, &state.current_activation_ref)?;
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
        layer_caches: state
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
pub fn finalize_decode_layer_refs_with_roots(
    state: DecodeTransitionRasterState,
) -> Result<(Vec<DecodeLayerCacheSlot>, usize, usize)> {
    if state.next_layer_idx != state.layer_count {
        bail!(
            "raster decode finalized after {} layers, expected {}",
            state.next_layer_idx,
            state.layer_count
        );
    }
    if state.updated_layer_caches.len() != state.layer_count {
        bail!(
            "raster decode stored {} layer caches, expected {}",
            state.updated_layer_caches.len(),
            state.layer_count
        );
    }

    Ok((
        state.updated_layer_caches,
        state.position + 1,
        state.token_count + 1,
    ))
}

fn trace_decode_layer_checkpoint_with_roots(
    roots: &RasterArtifactStoreRoots,
    state: &DecodeTransitionRasterState,
    layer_idx: usize,
) -> Result<()> {
    let current_activation =
        read_activation_row_from_ref_roots(roots, &state.current_activation_ref)?;
    let current_activation_values = current_activation.to_f32_values();
    let decode_input = read_activation_row_from_ref_roots(roots, &state.decode_input_ref)?;
    let decode_input_values = decode_input.to_f32_values();
    let decode_input_acts = decode_input.acts();
    let current_activation_sha256 = state
        .completed_layer_output_sha256s
        .last()
        .cloned()
        .expect("current layer output commitment should exist");
    let det_current_activation_sha256 = state
        .completed_layer_output_det_sha256s
        .last()
        .cloned()
        .unwrap_or(None);
    crate::trace::trace_checkpoint_lazy_result(
        &format!(
            "decode.layer_token.layer_{layer_idx}.position_{}",
            state.position
        ),
        || {
            let checkpoint_layer_caches =
                materialize_decode_checkpoint_caches_from_roots(roots, state, layer_idx)?;
            Ok(json!({
                "execution_mode": "deterministic",
                "token_id": state.next_token,
                "position": state.position,
                "next_layer_idx": layer_idx + 1,
                "decode_input_activation": decode_input_values.clone(),
                "decode_input_activation_sha256": crate::shared::numerics::transformer_kernels::build_vector_commitment(&decode_input_values),
                "det_decode_input_activation_sha256": Some(crate::shared::numerics::transformer_kernels::build_det_vector_commitment(&decode_input_acts)),
                "current_activation": current_activation_values,
                "current_activation_sha256": current_activation_sha256,
                "det_current_activation_sha256": det_current_activation_sha256,
                "layer_caches": crate::trace::serialize_layer_caches(&checkpoint_layer_caches),
                "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(&checkpoint_layer_caches),
                "completed_layer_output_sha256s": state.completed_layer_output_sha256s.clone(),
                "completed_layer_output_det_sha256s": state.completed_layer_output_det_sha256s.clone(),
            }))
        },
    )?;
    Ok(())
}

#[sequence]
fn rms_norm_decode_layer_row(
    row: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    norm: GemmaDecodeLayerNormKind,
) -> Result<RasterActivationRow> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer_idx)?;
    let norm_weights = call_tile!(read_decode_layer_norm_weights, source, layer_idx, norm)?;
    call_tile!(
        rms_norm_decode_row,
        row,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode layer RMSNorm"
    )
}

#[tile]
fn read_decode_layer_scalars(
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
) -> Result<crate::decode_transition::raster::auth_source::GemmaDecodeLayerScalars> {
    auth_read!(source, GemmaDecodeLayerScalarsRequest { layer_idx })
}

#[tile]
fn read_decode_layer_norm_weights(
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    norm: GemmaDecodeLayerNormKind,
) -> Result<Vec<crate::shared::numerics::det_num::Wgt>> {
    auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest { layer_idx, norm }
    )
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
fn decode_ple_input_gate_rows(layer: &GemmaDecodeLayerMetadata) -> Result<usize> {
    Ok(layer
        .ple_input_gate_shape
        .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing PLE input gate shape"))?
        .rows)
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

#[sequence]
fn rms_norm_decode_attention_heads(
    heads: &RasterAttentionHeadSequence,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    norm: GemmaDecodeLayerNormKind,
) -> Result<RasterAttentionHeadSequence> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer_idx)?;
    let norm_weights = call_tile!(read_decode_layer_norm_weights, source, layer_idx, norm)?;
    call_tile!(
        rms_norm_decode_heads,
        heads,
        &norm_weights,
        scalars.rms_norm_eps
    )
}

#[tile]
fn rms_norm_decode_heads(
    heads: &RasterAttentionHeadSequence,
    norm_weights: &[crate::shared::numerics::det_num::Wgt],
    eps: crate::shared::numerics::det_num::Acc,
) -> Result<RasterAttentionHeadSequence> {
    rms_norm_heads(heads, Some(norm_weights), Some(eps))
}

#[sequence]
fn value_rms_norm_decode_attention_heads(
    heads: &RasterAttentionHeadSequence,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
) -> Result<RasterAttentionHeadSequence> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer_idx)?;
    call_tile!(value_rms_norm_decode_heads, heads, scalars.rms_norm_eps)
}

#[tile]
fn value_rms_norm_decode_heads(
    heads: &RasterAttentionHeadSequence,
    eps: crate::shared::numerics::det_num::Acc,
) -> Result<RasterAttentionHeadSequence> {
    value_rms_norm_heads(heads, Some(eps))
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
fn read_decode_ple_scalars(
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<crate::decode_transition::raster::auth_source::GemmaDecodePleScalars> {
    auth_read!(source, GemmaDecodePleScalarsRequest)
}

#[tile]
fn read_decode_ple_projection_norm_weights(
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<Vec<crate::shared::numerics::det_num::Wgt>> {
    auth_read!(source, GemmaDecodePleProjectionNormWeightsRequest)
}

#[tile]
fn read_decode_ple_token_embedding(
    source: &AuthenticatedGemmaDecodeTransitionSource,
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

fn read_decode_projection_row(
    source: &AuthenticatedGemmaDecodeTransitionSource,
    projection_kind: &DecodeProjectionKind,
    row_idx: usize,
) -> Result<Vec<crate::shared::numerics::det_num::Wgt>> {
    match *projection_kind {
        DecodeProjectionKind::LayerMatrix { layer_idx, matrix } => auth_read!(
            source,
            GemmaDecodeLayerMatrixRowRequest {
                layer_idx,
                matrix,
                row_idx,
            },
        ),
        DecodeProjectionKind::PleModel { layer_idx } => auth_read!(
            source,
            GemmaDecodePleModelProjectionRowRequest { layer_idx, row_idx },
        ),
        DecodeProjectionKind::FinalLogits => {
            auth_read!(source, GemmaDecodeProjectionRowRequest { row_idx })
        }
    }
}

fn reshape_row_heads(
    row: RasterActivationRow,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterAttentionHeadSequence> {
    crate::shared::raster_kernels::transformer::reshape_sequence_heads(
        &RasterActivationSequence::from_rows(vec![row]),
        num_heads,
        head_dim,
    )
}

fn register_decode_layer_cache_with_roots(
    roots: &RasterArtifactStoreRoots,
    id_prefix: &str,
    layer_idx: usize,
    cache: RasterKvCache,
) -> Result<(RasterArtifactStoreRoots, DecodeLayerCacheSlot)> {
    if cache.current_len() == 0 {
        return Ok((
            roots.clone(),
            DecodeLayerCacheSlot::Empty {
                num_kv_heads: cache.head_count(),
            },
        ));
    }

    let head_count = cache.head_count();
    let current_len = cache.current_len();
    let head_dim = cache
        .keys()
        .first()
        .and_then(|head| head.first())
        .map(RasterActivationRow::width)
        .ok_or_else(|| anyhow!("decode cache registration requires a non-empty cache"))?;
    let keys_source_name = format!("{id_prefix}.{layer_idx}.keys");
    let values_source_name = format!("{id_prefix}.{layer_idx}.values");
    let key_leaves = cache
        .keys()
        .iter()
        .flat_map(|head| head.iter().map(activation_row_leaf))
        .collect::<Vec<_>>();
    let value_leaves = cache
        .values()
        .iter()
        .flat_map(|head| head.iter().map(activation_row_leaf))
        .collect::<Vec<_>>();
    let (roots, keys_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        roots,
        RasterArtifactId::new(keys_source_name.clone())?,
        RasterArtifactMetadata::activation_rows(head_count * current_len, head_dim)?,
        key_leaves,
    )?;
    let (roots, values_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        &roots,
        RasterArtifactId::new(values_source_name.clone())?,
        RasterArtifactMetadata::activation_rows(head_count * current_len, head_dim)?,
        value_leaves,
    )?;
    let cache_ref = kv_cache_ref_from_artifacts(
        RasterTensorId::new(keys_source_name)?,
        RasterTensorId::new(values_source_name)?,
        RasterActivationSequenceArtifactRef::new(keys_artifact_ref)?,
        RasterActivationSequenceArtifactRef::new(values_artifact_ref)?,
        head_count,
        current_len,
        head_dim,
    )?;
    Ok((roots, DecodeLayerCacheSlot::Ref(cache_ref)))
}

fn materialize_decode_layer_cache_from_roots(
    roots: &RasterArtifactStoreRoots,
    cache: &DecodeLayerCacheSlot,
) -> Result<RasterKvCache> {
    match cache {
        DecodeLayerCacheSlot::Empty { num_kv_heads } => Ok(RasterKvCache::empty(*num_kv_heads)),
        DecodeLayerCacheSlot::Ref(cache_ref) => {
            let (head_count, current_len, _) = cache_ref.shape().kv_cache_metadata()?;
            let mut keys = vec![Vec::with_capacity(current_len); head_count];
            let mut values = vec![Vec::with_capacity(current_len); head_count];
            for head_idx in 0..head_count {
                for token_idx in 0..current_len {
                    keys[head_idx].push(read_kv_row_from_roots(
                        roots,
                        RasterKvRowRequest {
                            cache_ref: cache_ref.clone(),
                            row_kind: RasterKvRowKind::Key,
                            head_idx,
                            token_idx,
                        },
                    )?);
                    values[head_idx].push(read_kv_row_from_roots(
                        roots,
                        RasterKvRowRequest {
                            cache_ref: cache_ref.clone(),
                            row_kind: RasterKvRowKind::Value,
                            head_idx,
                            token_idx,
                        },
                    )?);
                }
            }
            RasterKvCache::from_heads(keys, values)
        }
    }
}

pub fn materialize_decode_layer_caches_from_roots(
    roots: &RasterArtifactStoreRoots,
    layer_caches: &[DecodeLayerCacheSlot],
) -> Result<Vec<LayerKvCache>> {
    Ok(layer_caches
        .iter()
        .map(|cache| materialize_decode_layer_cache_from_roots(roots, cache))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(layer_cache_from_raster)
        .collect::<Vec<_>>())
}

fn materialize_decode_checkpoint_caches_from_roots(
    roots: &RasterArtifactStoreRoots,
    state: &DecodeTransitionRasterState,
    layer_idx: usize,
) -> Result<Vec<LayerKvCache>> {
    let mut caches = state
        .updated_layer_caches
        .iter()
        .map(|cache| {
            materialize_decode_layer_cache_from_roots(roots, cache).map(layer_cache_from_raster)
        })
        .collect::<Result<Vec<_>>>()?;
    caches.extend(
        state
            .original_layer_caches
            .iter()
            .skip(layer_idx + 1)
            .map(|cache| {
                materialize_decode_layer_cache_from_roots(roots, cache).map(layer_cache_from_raster)
            })
            .collect::<Result<Vec<_>>>()?,
    );
    Ok(caches)
}

fn resolve_decode_donor_cache_slot<'a>(
    layer_caches: &'a [DecodeLayerCacheSlot],
    layer_idx: usize,
    layer: &GemmaDecodeLayerMetadata,
) -> Result<Option<&'a DecodeLayerCacheSlot>> {
    layer
        .kv_shared_layer_index
        .map(|donor_idx| {
            if donor_idx >= layer_idx {
                bail!(
                    "transformer decode layer {layer_idx} cannot share KV with non-prior donor {donor_idx}"
                );
            }
            layer_caches.get(donor_idx).ok_or_else(|| {
                anyhow!("transformer decode donor cache {donor_idx} missing for layer {layer_idx}")
            })
        })
        .transpose()
}

fn raster_cache_from_layer_cache(cache: &LayerKvCache) -> Result<RasterKvCache> {
    if cache.current_len() == 0 {
        return Ok(RasterKvCache::empty(cache.keys.len()));
    }
    let det_keys = cache.det_keys.as_ref().ok_or_else(|| {
        anyhow!("deterministic raster decode requires canonical layer cache keys")
    })?;
    let det_values = cache.det_values.as_ref().ok_or_else(|| {
        anyhow!("deterministic raster decode requires canonical layer cache values")
    })?;
    if det_keys.len() != det_values.len() {
        bail!(
            "layer cache head count mismatch: keys {} values {}",
            det_keys.len(),
            det_values.len()
        );
    }
    RasterKvCache::from_heads(
        det_keys
            .iter()
            .map(|head| {
                head.iter()
                    .cloned()
                    .map(RasterActivationRow::from_acts)
                    .collect()
            })
            .collect(),
        det_values
            .iter()
            .map(|head| {
                head.iter()
                    .cloned()
                    .map(RasterActivationRow::from_acts)
                    .collect()
            })
            .collect(),
    )
}

fn layer_cache_from_raster(cache: RasterKvCache) -> LayerKvCache {
    if cache.current_len() == 0 {
        return LayerKvCache::new(cache.head_count());
    }

    LayerKvCache::from_det_heads(
        cache
            .keys()
            .iter()
            .map(|head| head.iter().map(|row| row.acts()).collect::<VecDeque<_>>())
            .collect(),
        cache
            .values()
            .iter()
            .map(|head| head.iter().map(|row| row.acts()).collect::<VecDeque<_>>())
            .collect(),
    )
}

fn insert_decode_activation_row_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: String,
    row: &RasterActivationRow,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (roots, artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        roots,
        RasterArtifactId::new(source_name.clone())?,
        RasterArtifactMetadata::activation_rows(1, row.width())?,
        vec![activation_row_leaf(row)],
    )?;
    let activation_ref = activation_sequence_ref_from_artifact(
        RasterTensorId::new(source_name)?,
        RasterActivationSequenceArtifactRef::new(artifact_ref)?,
    )?;
    Ok((roots, activation_ref))
}

fn insert_decode_selected_token_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: String,
    token_id: u32,
) -> Result<(RasterArtifactStoreRoots, RasterSelectedTokenRef)> {
    let (roots, token_ref) = ArtifactIo::insert_artifact_with_roots(
        roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::token_ids(1),
        vec![token_id_leaf(token_id)],
    )?;
    Ok((
        roots,
        RasterSelectedTokenRef::new(RasterTokenIdSequenceRef::new(token_ref)?)?,
    ))
}

fn insert_attention_heads_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: String,
    heads: RasterAttentionHeadSequence,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let head_count = heads.head_count();
    let sequence_len = heads.sequence_len()?;
    let head_dim = heads
        .heads()
        .first()
        .and_then(|head| head.first())
        .map(RasterActivationRow::width)
        .ok_or_else(|| anyhow!("raster attention heads require at least one row"))?;
    let leaves = heads
        .heads()
        .iter()
        .flat_map(|head| head.iter().map(activation_row_leaf))
        .collect::<Vec<_>>();
    let (roots, artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        roots,
        RasterArtifactId::new(source_name.clone())?,
        RasterArtifactMetadata::activation_rows(head_count * sequence_len, head_dim)?,
        leaves,
    )?;
    let heads_ref = attention_heads_ref_from_artifact(
        RasterTensorId::new(source_name)?,
        RasterActivationSequenceArtifactRef::new(artifact_ref)?,
        head_count,
        sequence_len,
        head_dim,
    )?;
    Ok((roots, heads_ref))
}

fn materialize_attention_heads_from_roots(
    roots: &RasterArtifactStoreRoots,
    heads_ref: &RasterAttentionHeadsRef,
) -> Result<RasterAttentionHeadSequence> {
    let (head_count, sequence_len, _) = heads_ref.tensor_ref().shape().heads_metadata()?;
    let mut heads = vec![Vec::with_capacity(sequence_len); head_count];
    for head_idx in 0..head_count {
        for token_idx in 0..sequence_len {
            heads[head_idx].push(read_head_row_from_roots(
                roots,
                RasterHeadRowRequest {
                    tensor_ref: heads_ref.clone(),
                    head_idx,
                    token_idx,
                },
            )?);
        }
    }
    Ok(RasterAttentionHeadSequence::from_heads(heads))
}

fn read_activation_row_from_ref_roots(
    roots: &RasterArtifactStoreRoots,
    activation_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationRow> {
    let (row_count, _width) = activation_ref.tensor_ref().shape().sequence_metadata()?;
    if row_count != 1 {
        bail!("raster decode transition activation ref contains {row_count} rows, expected one");
    }
    read_sequence_row_from_roots(
        roots,
        RasterSequenceRowRequest {
            tensor_ref: activation_ref.clone(),
            row_idx: 0,
        },
    )
}

fn read_decode_attention_kv_row_from_roots(
    roots: &RasterArtifactStoreRoots,
    cache_ref: &RasterKvCacheRef,
    row_kind: RasterKvRowKind,
    head_idx: usize,
    token_idx: usize,
) -> Result<RasterActivationRow> {
    read_kv_row_from_roots(
        roots,
        RasterKvRowRequest {
            cache_ref: cache_ref.clone(),
            row_kind,
            head_idx,
            token_idx,
        },
    )
}

fn read_decode_attention_scalar_row_from_roots(
    roots: &RasterArtifactStoreRoots,
    tensor_ref: &crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef,
    row_idx: usize,
    label: &str,
) -> Result<Act> {
    let row = read_sequence_row_from_roots(
        roots,
        RasterSequenceRowRequest {
            tensor_ref: tensor_ref.clone(),
            row_idx,
        },
    )?;
    if row.width() != 1 {
        bail!(
            "decode attention {label} row {row_idx} has width {}, expected 1",
            row.width()
        );
    }
    Ok(row.acts()[0])
}

fn scale_row(row: &RasterActivationRow, scalar: Option<Act>) -> Result<RasterActivationRow> {
    first_row(
        scale_sequence(
            &RasterActivationSequence::from_rows(vec![row.clone()]),
            scalar,
        )?,
        "deterministic decode scaling",
    )
}

fn add_rows(lhs: &RasterActivationRow, rhs: &RasterActivationRow) -> Result<RasterActivationRow> {
    first_row(
        add_sequences(
            &RasterActivationSequence::from_rows(vec![lhs.clone()]),
            &RasterActivationSequence::from_rows(vec![rhs.clone()]),
        )?,
        "deterministic decode row add",
    )
}

fn mul_rows(lhs: &RasterActivationRow, rhs: &RasterActivationRow) -> Result<RasterActivationRow> {
    first_row(
        mul_sequences(
            &RasterActivationSequence::from_rows(vec![lhs.clone()]),
            &RasterActivationSequence::from_rows(vec![rhs.clone()]),
        )?,
        "deterministic decode row multiply",
    )
}

fn gelu_row(row: &RasterActivationRow) -> Result<RasterActivationRow> {
    first_row(
        gelu_sequence(&RasterActivationSequence::from_rows(vec![row.clone()]))?,
        "deterministic decode GELU",
    )
}

fn first_row(sequence: RasterActivationSequence, label: &str) -> Result<RasterActivationRow> {
    sequence
        .into_rows()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("{label} returned no rows"))
}

fn validate_row_width(row: &RasterActivationRow, expected_width: usize, label: &str) -> Result<()> {
    if row.width() != expected_width {
        bail!(
            "{label} width mismatch: {} vs {}",
            row.width(),
            expected_width
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
