use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::raster_authoring::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::artifact_io::ArtifactIo;
use crate::shared::det_num::{
    acc_add_sat, add_sat, attention_score as det_attention_score, attention_softmax_exp_term,
    attention_softmax_raw_weight, attention_softmax_residual, mac_bits, requantize, softcap_act,
    Acc, Act,
};
use crate::shared::raster_artifact_store::{
    activation_row_leaf, read_selected_token_from_roots, RasterActivationSequenceArtifactRef,
    RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots, RasterSelectedTokenRef,
};
use crate::shared::raster_decode_transition::{
    AuthenticatedGemmaDecodeTransitionSource, GemmaDecodeAttentionKind,
    GemmaDecodeEmbeddingRowRequest, GemmaDecodeFinalNormWeightsRequest,
    GemmaDecodeFinalScalarsRequest, GemmaDecodeLayerMatrixKind, GemmaDecodeLayerMatrixRowRequest,
    GemmaDecodeLayerMetadata, GemmaDecodeLayerMetadataRequest, GemmaDecodeLayerNormKind,
    GemmaDecodeLayerNormWeightsRequest, GemmaDecodeLayerScalarsRequest,
    GemmaDecodePleModelProjectionRowRequest, GemmaDecodePleProjectionNormWeightsRequest,
    GemmaDecodePleScalarsRequest, GemmaDecodePleTokenEmbeddingRowRequest,
    GemmaDecodeProjectionRowRequest, GemmaDecodeTransitionMetadataRequest,
};
use crate::shared::raster_row_store::{
    activation_sequence_ref_from_artifact, read_sequence_row_from_roots,
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterAttentionHeadsRef,
    RasterHeadRowRequest, RasterKvCacheBuilderRef, RasterKvCacheRef, RasterKvRowKind,
    RasterKvRowRequest, RasterProjectionOutputBuilderRef, RasterSequenceRowRequest,
    RasterTensorBuilderRef, RasterTensorId,
};
use crate::shared::raster_transformer_kernels::{
    add_sequences, apply_rope_to_heads, combine_attention_heads, gelu_sequence, mul_sequences,
    project_row_with_weights, rms_norm_heads, rms_norm_sequence, scale_sequence,
    validate_attention_kv_rows_per_tile, validate_projection_rows_per_tile, value_rms_norm_heads,
    RasterActivationRow, RasterActivationSequence, RasterAttentionHeadSequence, RasterKvCache,
};
use crate::shared::transformer::{
    ActivationSequence, InternalActivationSequence, InternalLogits, LayerKvCache, PrefillLogits,
    TransformerDecodeState, TransformerDecodeStepResult,
};
use crate::RasterSizingControls;

use super::tiles::ActivationSequenceWithCache;

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
    pub decode_transition_source_name: String,
    pub output_source_prefix: String,
    pub raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterDecodeTransitionOutputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub transition_result: TransformerDecodeStepResult,
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
enum DecodeAttentionPhase {
    CollectScores {
        score_builder_ref: RasterTensorBuilderRef,
        next_kv_offset: usize,
    },
    FindSoftmaxMax {
        score_ref: crate::shared::raster_row_store::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: Option<usize>,
        max_logit_bits: i32,
    },
    SumSoftmaxExp {
        score_ref: crate::shared::raster_row_store::RasterActivationSequenceRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
    },
    BuildRawSoftmaxWeights {
        score_ref: crate::shared::raster_row_store::RasterActivationSequenceRef,
        raw_weight_builder_ref: RasterTensorBuilderRef,
        next_score_row_idx: usize,
        max_index: usize,
        max_logit_bits: i32,
        sum_exp_bits: i64,
        summed_weight_bits: i32,
    },
    CorrectSoftmaxResidual {
        raw_weight_ref: crate::shared::raster_row_store::RasterActivationSequenceRef,
        final_weight_builder_ref: RasterTensorBuilderRef,
        next_weight_row_idx: usize,
        max_index: usize,
        residual_bits: i32,
    },
    ApplyValues {
        weight_ref: crate::shared::raster_row_store::RasterActivationSequenceRef,
        next_kv_offset: usize,
        weighted_sum_acc_bits: Vec<i64>,
    },
}

#[tile]
pub fn init_decode_transition_store() -> AuthenticatedRasterTensorStore {
    AuthenticatedRasterTensorStore::new()
}

#[tile]
pub fn init_decode_transition_state(
    store: &mut AuthenticatedRasterTensorStore,
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    raster_sizing: RasterSizingControls,
) -> Result<DecodeTransitionRasterState> {
    let output_source_prefix = format!(
        "decode.transition.position_{}.token_count_{}",
        transformer_decode_state.position, transformer_decode_state.token_count
    );
    ArtifactIo::reset_store();
    let roots = ArtifactIo::export_store_roots();
    let (_, state) = init_decode_transition_state_with_roots(
        store,
        roots,
        transformer_decode_state,
        next_token,
        source,
        raster_sizing,
        output_source_prefix,
    )?;
    Ok(state)
}

#[tile]
pub fn init_decode_transition_state_with_roots(
    store: &mut AuthenticatedRasterTensorStore,
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
    let (_activation_roots, decode_input_ref) = insert_decode_activation_row_with_roots(
        &artifact_store_roots,
        format!("{output_source_prefix}.input.selected_token_embedding"),
        &embedded,
    )?;
    let original_layer_caches = transformer_decode_state
        .layer_caches
        .iter()
        .enumerate()
        .map(|(layer_idx, cache)| {
            let cache = raster_cache_from_layer_cache(cache)?;
            register_decode_layer_cache(store, "decode.original.cache", layer_idx, cache)
        })
        .collect::<Result<Vec<_>>>()?;
    artifact_store_roots = ArtifactIo::export_store_roots();

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

#[sequence(kind = recursive)]
pub fn compute_next_decode_layer(
    state: DecodeTransitionRasterState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, DecodeTransitionRasterState)> {
    if state.next_layer_idx >= state.layer_count {
        return Ok((true, state));
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
    let decode_input = call_tile!(
        read_decode_single_activation_row,
        &state,
        &state.decode_input_ref
    )?;
    let current_activation = call_tile!(
        read_decode_single_activation_row,
        &state,
        &state.current_activation_ref
    )?;
    let per_layer_input = call_seq!(
        compute_decode_ple_input,
        store,
        state.next_token,
        &decode_input,
        source,
        &context.layer,
        state.projection_rows_per_tile
    )?;
    let (layer_output, updated_cache) = call_seq!(
        run_basic_decode_layer,
        store,
        &current_activation,
        source,
        &context.layer,
        context.cache_slot,
        context.donor_cache_slot.as_ref(),
        per_layer_input.as_ref(),
        state.position,
        state.projection_rows_per_tile,
        state.attention_kv_rows_per_tile
    )?;

    let state = call_tile!(
        update_decode_layer_state,
        store,
        state,
        context.layer_idx,
        layer_output,
        updated_cache
    )?;
    Ok((false, state))
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
fn update_decode_layer_state(
    store: &AuthenticatedRasterTensorStore,
    mut state: DecodeTransitionRasterState,
    layer_idx: usize,
    layer_output: RasterActivationRow,
    updated_cache: DecodeLayerCacheSlot,
) -> Result<DecodeTransitionRasterState> {
    state.artifact_store_roots = ArtifactIo::export_store_roots();
    let (artifact_store_roots, current_activation_ref) = insert_decode_activation_row_with_roots(
        &state.artifact_store_roots,
        format!(
            "{}.layer_{}.position_{}.output",
            state.output_source_prefix, layer_idx, state.position
        ),
        &layer_output,
    )?;
    state.artifact_store_roots = artifact_store_roots;
    state.current_activation_ref = current_activation_ref;
    state.updated_layer_caches.push(updated_cache);
    let current_activation_values = layer_output.to_f32_values();
    let current_activation_acts = layer_output.acts();
    state.completed_layer_output_sha256s.push(
        crate::shared::transformer_kernels::build_vector_commitment(&current_activation_values),
    );
    state.completed_layer_output_det_sha256s.push(Some(
        crate::shared::transformer_kernels::build_det_vector_commitment(&current_activation_acts),
    ));
    let decode_input =
        read_activation_row_from_ref_roots(&state.artifact_store_roots, &state.decode_input_ref)?;
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
                materialize_decode_checkpoint_caches(store, &state, layer_idx)?;
            Ok(json!({
                "execution_mode": "deterministic",
                "token_id": state.next_token,
                "position": state.position,
                "next_layer_idx": layer_idx + 1,
                "decode_input_activation": decode_input_values.clone(),
                "decode_input_activation_sha256": crate::shared::transformer_kernels::build_vector_commitment(&decode_input_values),
                "det_decode_input_activation_sha256": Some(crate::shared::transformer_kernels::build_det_vector_commitment(&decode_input_acts)),
                "current_activation": current_activation_values,
                "current_activation_sha256": current_activation_sha256,
                "det_current_activation_sha256": det_current_activation_sha256,
                "layer_caches": crate::trace::serialize_layer_caches(&checkpoint_layer_caches),
                "det_layer_caches_sha256": crate::shared::transformer_kernels::build_det_kv_cache_commitment(&checkpoint_layer_caches),
                "completed_layer_output_sha256s": state.completed_layer_output_sha256s.clone(),
                "completed_layer_output_det_sha256s": state.completed_layer_output_det_sha256s.clone(),
            }))
        },
    )?;

    state.next_layer_idx += 1;
    Ok(state)
}

#[tile]
pub fn finalize_decode_layer_state(
    store: &AuthenticatedRasterTensorStore,
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

    let current_activation = read_activation_row_from_ref_roots(
        &state.artifact_store_roots,
        &state.current_activation_ref,
    )?;
    let det_row = current_activation.acts();
    let values = vec![current_activation.to_f32_values()];
    let internal = InternalActivationSequence::from_det_values(vec![det_row.clone()]);
    let mut activation_state = ActivationSequence::from_internal(
        internal,
        crate::shared::transformer_kernels::build_activation_commitment(&values),
    );
    activation_state.det_activations_sha256 =
        Some(crate::shared::transformer_kernels::build_det_activation_commitment(&[det_row]));

    Ok(ActivationSequenceWithCache {
        activation_state,
        // Public decode outputs still expose materialized layer caches for compatibility.
        layer_caches: state
            .updated_layer_caches
            .iter()
            .map(|cache| materialize_decode_layer_cache_from_store(store, cache))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(layer_cache_from_raster)
            .collect(),
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

#[tile]
pub fn init_decode_logits_projection(
    store: &mut AuthenticatedRasterTensorStore,
    normalized_final_position: RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    projection_rows_per_tile: usize,
) -> Result<DecodeLogitsRasterState> {
    validate_projection_rows_per_tile(projection_rows_per_tile)?;
    let metadata = auth_read!(source, GemmaDecodeTransitionMetadataRequest)?;
    if metadata.projection_rows == 0 {
        bail!("deterministic decode logits projection requires at least one projection row");
    }
    if normalized_final_position.width() != metadata.final_norm_width {
        bail!(
            "deterministic decode logits projection input has width {}, expected {}",
            normalized_final_position.width(),
            metadata.final_norm_width
        );
    }
    if metadata.projection_cols != metadata.final_norm_width {
        bail!(
            "deterministic decode logits projection metadata width mismatch: {} vs {}",
            metadata.projection_cols,
            metadata.final_norm_width
        );
    }

    let scalars = auth_read!(source, GemmaDecodeFinalScalarsRequest)?;
    let output_builder_ref = store.start_projection_output_builder(
        RasterTensorId::new("decode.final.logits")?,
        1,
        metadata.projection_rows,
    )?;
    Ok(DecodeLogitsRasterState {
        normalized_final_position,
        next_logit_idx: 0,
        logit_count: metadata.projection_rows,
        output_builder_ref,
        softcap_bits: scalars.final_logit_softcapping.map(Act::to_bits),
        projection_rows_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn project_next_decode_logit(
    mut state: DecodeLogitsRasterState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, DecodeLogitsRasterState)> {
    if state.next_logit_idx >= state.logit_count {
        return Ok((true, state));
    }

    let end = state
        .next_logit_idx
        .saturating_add(state.projection_rows_per_tile)
        .min(state.logit_count);
    let start_logit_idx = state.next_logit_idx;
    let mut logit_bits = Vec::with_capacity(end - start_logit_idx);
    while state.next_logit_idx < end {
        let projection_row = auth_read!(
            source,
            GemmaDecodeProjectionRowRequest {
                row_idx: state.next_logit_idx,
            },
        )?;
        let mut logit =
            project_row_with_weights(&state.normalized_final_position, &projection_row)?;
        if let Some(softcap_bits) = state.softcap_bits {
            logit = softcap_act(logit, Act::from_bits(softcap_bits));
        }
        logit_bits.push(logit.to_bits());
        state.next_logit_idx += 1;
    }
    store.append_projection_output_chunk(
        &mut state.output_builder_ref,
        0,
        start_logit_idx,
        &logit_bits,
    )?;
    Ok((false, state))
}

#[tile]
pub fn finalize_decode_transition_result(
    store: &mut AuthenticatedRasterTensorStore,
    state: DecodeLogitsRasterState,
    transformer_decode_state: TransformerDecodeState,
    final_hidden_state: ActivationSequence,
) -> Result<TransformerDecodeStepResult> {
    if state.next_logit_idx != state.logit_count {
        bail!(
            "raster decode projection completed {} logits, expected {}",
            state.next_logit_idx,
            state.logit_count
        );
    }
    let logits_ref = store.finalize_projection_output_builder(state.output_builder_ref)?;
    let logits_sequence = store.materialize_sequence(&logits_ref)?;
    let det_logits = logits_sequence
        .into_rows()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("raster decode projection produced no logits row"))?
        .acts();
    let internal_logits = InternalLogits::from_det_values(det_logits.clone());
    let final_logits_sha256 =
        crate::shared::transformer_kernels::build_vector_commitment(internal_logits.as_f32_slice());
    let mut prefill_logits = PrefillLogits::from_internal(internal_logits, final_logits_sha256);
    prefill_logits.det_final_logits_sha256 =
        Some(crate::shared::transformer_kernels::build_det_vector_commitment(&det_logits));

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

#[sequence]
pub fn run(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    raster_sizing: RasterSizingControls,
) -> Result<TransformerDecodeStepResult> {
    let mut store = call_tile!(init_decode_transition_store);
    let state = call_tile!(
        init_decode_transition_state,
        &mut store,
        transformer_decode_state.clone(),
        next_token,
        source,
        raster_sizing
    )?;
    let state = call_recur_seq!(compute_next_decode_layer, state, source, &mut store)?;
    let layer_output = call_tile!(finalize_decode_layer_state, &store, state)?;
    let normalized = call_tile!(
        normalize_decode_final_position,
        &layer_output.activation_state,
        source
    )?;
    let logits_state = call_tile!(
        init_decode_logits_projection,
        &mut store,
        normalized,
        source,
        raster_sizing.projection_rows_per_tile
    )?;
    let logits_state =
        call_recur_tile!(project_next_decode_logit, logits_state, source, &mut store)?;
    call_tile!(
        finalize_decode_transition_result,
        &mut store,
        logits_state,
        TransformerDecodeState {
            layer_caches: layer_output.layer_caches,
            position: transformer_decode_state.position,
            token_count: transformer_decode_state.token_count,
        },
        layer_output.activation_state
    )
}

#[sequence]
pub fn main(
    input_roots: RasterDecodeTransitionInputRoots,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<RasterDecodeTransitionOutputRefs> {
    if input_roots.decode_transition_source_name != source.identifier() {
        bail!(
            "raster decode transition source {} does not match input source {}",
            source.identifier(),
            input_roots.decode_transition_source_name
        );
    }
    let next_token = call_tile!(
        read_decode_selected_token,
        &input_roots.artifact_store_roots,
        &input_roots.selected_token_ref
    )?;
    let mut store = call_tile!(init_decode_transition_store);
    let (_artifact_store_roots, state) = call_tile!(
        init_decode_transition_state_with_roots,
        &mut store,
        input_roots.artifact_store_roots,
        input_roots.transformer_decode_state.clone(),
        next_token,
        source,
        input_roots.raster_sizing,
        input_roots.output_source_prefix
    )?;
    let state = call_recur_seq!(compute_next_decode_layer, state, source, &mut store)?;
    let layer_output = call_tile!(finalize_decode_layer_state, &store, state)?;
    let normalized = call_tile!(
        normalize_decode_final_position,
        &layer_output.activation_state,
        source
    )?;
    let logits_state = call_tile!(
        init_decode_logits_projection,
        &mut store,
        normalized,
        source,
        input_roots.raster_sizing.projection_rows_per_tile
    )?;
    let logits_state =
        call_recur_tile!(project_next_decode_logit, logits_state, source, &mut store)?;
    let transition_result = call_tile!(
        finalize_decode_transition_result,
        &mut store,
        logits_state,
        TransformerDecodeState {
            layer_caches: layer_output.layer_caches,
            position: input_roots.transformer_decode_state.position,
            token_count: input_roots.transformer_decode_state.token_count,
        },
        layer_output.activation_state
    )?;
    let artifact_store_roots = ArtifactIo::export_store_roots();
    Ok(RasterDecodeTransitionOutputRefs {
        artifact_store_roots,
        transition_result,
    })
}

#[sequence]
fn run_basic_decode_layer(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache_slot: DecodeLayerCacheSlot,
    donor_cache_slot: Option<&DecodeLayerCacheSlot>,
    per_layer_input: Option<&RasterActivationRow>,
    position: usize,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
) -> Result<(RasterActivationRow, DecodeLayerCacheSlot)> {
    let scalars = call_tile!(read_decode_layer_scalars, source, layer.layer_idx)?;
    let (xs, updated_cache) = call_seq!(
        run_decode_attention_block,
        store,
        input,
        source,
        layer,
        cache_slot,
        donor_cache_slot,
        position,
        projection_rows_per_tile,
        attention_kv_rows_per_tile
    )?;
    let xs = call_seq!(
        run_decode_mlp_block,
        store,
        &xs,
        source,
        layer,
        scalars.rms_norm_eps,
        projection_rows_per_tile
    )?;
    let xs = if let Some(per_layer_input) = per_layer_input {
        call_seq!(
            run_decode_ple_block,
            store,
            &xs,
            per_layer_input,
            source,
            layer,
            scalars.rms_norm_eps,
            projection_rows_per_tile
        )?
    } else {
        xs
    };
    let xs = call_tile!(scale_decode_row_optional, &xs, scalars.layer_scalar)?;

    Ok((xs, updated_cache))
}

#[sequence]
fn run_decode_attention_block(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache_slot: DecodeLayerCacheSlot,
    donor_cache_slot: Option<&DecodeLayerCacheSlot>,
    position: usize,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
) -> Result<(RasterActivationRow, DecodeLayerCacheSlot)> {
    let residual = input.clone();
    let normed = call_seq!(
        rms_norm_decode_layer_row,
        input,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::InputLayer
    )?;
    let (attention_output, updated_cache) = call_seq!(
        run_decode_attention,
        store,
        &normed,
        source,
        layer,
        cache_slot,
        donor_cache_slot,
        position,
        projection_rows_per_tile,
        attention_kv_rows_per_tile
    )?;
    let attention_output = call_seq!(
        rms_norm_decode_layer_row,
        &attention_output,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PostAttention
    )?;
    let xs = call_tile!(add_decode_rows, &residual, &attention_output)?;
    Ok((xs, updated_cache))
}

#[sequence]
fn run_decode_mlp_block(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    rms_norm_eps: Acc,
    projection_rows_per_tile: usize,
) -> Result<RasterActivationRow> {
    let residual = input.clone();
    let norm_weights = call_tile!(
        read_decode_layer_norm_weights,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PreFeedForward
    )?;
    let normed = call_tile!(
        rms_norm_decode_row,
        input,
        &norm_weights,
        rms_norm_eps,
        "decode MLP pre-feedforward RMSNorm"
    )?;
    let gate = call_seq!(
        project_row_with_decode_source,
        store,
        &normed,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Gate,
        layer.gate_proj_shape.rows,
        projection_rows_per_tile,
        format!("decode.layer.{}.gate_proj", layer.layer_idx)
    )?;
    let gate = call_tile!(gelu_decode_row, &gate)?;
    let up = call_seq!(
        project_row_with_decode_source,
        store,
        &normed,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Up,
        layer.up_proj_shape.rows,
        projection_rows_per_tile,
        format!("decode.layer.{}.up_proj", layer.layer_idx)
    )?;
    let ff_hidden = call_tile!(mul_decode_rows, &gate, &up)?;
    let ff_out = call_seq!(
        project_row_with_decode_source,
        store,
        &ff_hidden,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Down,
        layer.down_proj_shape.rows,
        projection_rows_per_tile,
        format!("decode.layer.{}.down_proj", layer.layer_idx)
    )?;
    let norm_weights = call_tile!(
        read_decode_layer_norm_weights,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PostFeedForward
    )?;
    let ff_out = call_tile!(
        rms_norm_decode_row,
        &ff_out,
        &norm_weights,
        rms_norm_eps,
        "decode MLP post-feedforward RMSNorm"
    )?;
    call_tile!(add_decode_rows, &residual, &ff_out)
}

#[sequence]
fn run_decode_ple_block(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationRow,
    per_layer_input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    rms_norm_eps: Acc,
    projection_rows_per_tile: usize,
) -> Result<RasterActivationRow> {
    let residual = input.clone();
    let gated = call_seq!(
        project_row_with_decode_source,
        store,
        input,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::PleInputGate,
        call_tile!(decode_ple_input_gate_rows, layer)?,
        projection_rows_per_tile,
        format!("decode.layer.{}.ple.input_gate", layer.layer_idx)
    )?;
    let gated = call_tile!(gelu_decode_row, &gated)?;
    let gated = call_tile!(mul_decode_rows, &gated, per_layer_input)?;
    let projected = call_seq!(
        project_row_with_decode_source,
        store,
        &gated,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::PleLayerProjection,
        call_tile!(decode_ple_layer_projection_rows, layer)?,
        projection_rows_per_tile,
        format!("decode.layer.{}.ple.layer_projection", layer.layer_idx)
    )?;
    let norm_weights = call_tile!(
        read_decode_layer_norm_weights,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::PlePostInput
    )?;
    let projected = call_tile!(
        rms_norm_decode_row,
        &projected,
        &norm_weights,
        rms_norm_eps,
        "decode PLE post-input RMSNorm"
    )?;
    call_tile!(add_decode_rows, &residual, &projected)
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
) -> Result<crate::shared::raster_decode_transition::GemmaDecodeLayerScalars> {
    auth_read!(source, GemmaDecodeLayerScalarsRequest { layer_idx })
}

#[tile]
fn read_decode_layer_norm_weights(
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    norm: GemmaDecodeLayerNormKind,
) -> Result<Vec<crate::shared::det_num::Wgt>> {
    auth_read!(
        source,
        GemmaDecodeLayerNormWeightsRequest { layer_idx, norm }
    )
}

#[tile]
fn rms_norm_decode_row(
    row: &RasterActivationRow,
    norm_weights: &[crate::shared::det_num::Wgt],
    eps: crate::shared::det_num::Acc,
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
    norm_weights: &[crate::shared::det_num::Wgt],
    eps: crate::shared::det_num::Acc,
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
    eps: crate::shared::det_num::Acc,
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

#[sequence]
fn update_decode_attention_cache(
    store: &mut AuthenticatedRasterTensorStore,
    cache_slot: DecodeLayerCacheSlot,
    use_donor_cache: bool,
    k_heads_ref: RasterAttentionHeadsRef,
    v_heads_ref: RasterAttentionHeadsRef,
    layer_idx: usize,
    cache_window: Option<usize>,
    rows_per_tile: usize,
) -> Result<DecodeLayerCacheSlot> {
    if use_donor_cache {
        return Ok(cache_slot);
    }
    call_seq!(
        append_decode_kv_cache_ref,
        store,
        cache_slot,
        k_heads_ref,
        v_heads_ref,
        layer_idx,
        cache_window,
        rows_per_tile
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
fn insert_decode_attention_query_heads(
    store: &mut AuthenticatedRasterTensorStore,
    layer_idx: usize,
    q_heads: RasterAttentionHeadSequence,
) -> Result<RasterAttentionHeadsRef> {
    store.insert_attention_heads(
        RasterTensorId::new(format!("decode.layer.{layer_idx}.q_heads"))?,
        q_heads,
    )
}

#[tile]
fn insert_decode_attention_key_heads(
    store: &mut AuthenticatedRasterTensorStore,
    layer_idx: usize,
    k_heads: RasterAttentionHeadSequence,
) -> Result<RasterAttentionHeadsRef> {
    store.insert_attention_heads(
        RasterTensorId::new(format!("decode.layer.{layer_idx}.k_heads"))?,
        k_heads,
    )
}

#[tile]
fn insert_decode_attention_value_heads(
    store: &mut AuthenticatedRasterTensorStore,
    layer_idx: usize,
    v_heads: RasterAttentionHeadSequence,
) -> Result<RasterAttentionHeadsRef> {
    store.insert_attention_heads(
        RasterTensorId::new(format!("decode.layer.{layer_idx}.v_heads"))?,
        v_heads,
    )
}

#[tile]
fn materialize_decode_attention_output_row(
    store: &AuthenticatedRasterTensorStore,
    attention_heads_ref: &RasterAttentionHeadsRef,
) -> Result<RasterActivationRow> {
    let output_heads = store.materialize_heads(attention_heads_ref)?;
    let attention_sequence = combine_attention_heads(&output_heads)?;
    first_row(attention_sequence, "decode attention combine")
}

#[sequence]
fn run_decode_attention(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache_slot: DecodeLayerCacheSlot,
    donor_cache_slot: Option<&DecodeLayerCacheSlot>,
    position: usize,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
) -> Result<(RasterActivationRow, DecodeLayerCacheSlot)> {
    call_tile!(validate_decode_attention_context, input, layer)?;

    let q_projected = call_seq!(
        project_row_with_decode_source,
        store,
        input,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Query,
        layer.q_proj_shape.rows,
        projection_rows_per_tile,
        format!("decode.layer.{}.q_proj", layer.layer_idx)
    )?;
    let raw_k = call_seq!(
        project_row_with_decode_source,
        store,
        input,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Key,
        layer.k_proj_shape.rows,
        projection_rows_per_tile,
        format!("decode.layer.{}.k_proj", layer.layer_idx)
    )?;
    let raw_v = if layer.has_v_proj {
        call_seq!(
            project_row_with_decode_source,
            store,
            input,
            source,
            layer.layer_idx,
            GemmaDecodeLayerMatrixKind::Value,
            layer
                .v_proj_shape
                .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing v_proj shape"))?
                .rows,
            projection_rows_per_tile,
            format!("decode.layer.{}.v_proj", layer.layer_idx)
        )?
    } else if layer.attention_k_eq_v {
        call_tile!(clone_decode_key_as_value, &raw_k)?
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };

    let q_heads = call_tile!(
        reshape_decode_row_heads,
        q_projected,
        layer.num_heads,
        layer.head_dim
    )?;
    let k_heads = call_tile!(
        reshape_decode_row_heads,
        raw_k,
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let v_heads = call_tile!(
        reshape_decode_row_heads,
        raw_v,
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let q_heads = call_seq!(
        rms_norm_decode_attention_heads,
        &q_heads,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::Query
    )?;
    let k_heads = call_seq!(
        rms_norm_decode_attention_heads,
        &k_heads,
        source,
        layer.layer_idx,
        GemmaDecodeLayerNormKind::Key
    )?;
    let v_heads = call_seq!(
        value_rms_norm_decode_attention_heads,
        &v_heads,
        source,
        layer.layer_idx
    )?;
    let scalars = call_tile!(read_decode_layer_scalars, source, layer.layer_idx)?;
    let q_heads = call_tile!(
        apply_decode_rope_to_heads,
        &q_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        position
    )?;
    let k_heads = call_tile!(
        apply_decode_rope_to_heads,
        &k_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        position
    )?;

    let q_heads_ref = call_tile!(
        insert_decode_attention_query_heads,
        store,
        layer.layer_idx,
        q_heads
    )?;
    let k_heads_ref = call_tile!(
        insert_decode_attention_key_heads,
        store,
        layer.layer_idx,
        k_heads
    )?;
    let v_heads_ref = call_tile!(
        insert_decode_attention_value_heads,
        store,
        layer.layer_idx,
        v_heads
    )?;

    let updated_cache_slot = call_seq!(
        update_decode_attention_cache,
        store,
        cache_slot,
        donor_cache_slot.is_some(),
        k_heads_ref,
        v_heads_ref,
        layer.layer_idx,
        layer.cache_sliding_window,
        attention_kv_rows_per_tile
    )?;
    let attention_cache_slot = donor_cache_slot.unwrap_or(&updated_cache_slot);
    let attention_cache_ref = call_tile!(resolve_decode_attention_cache_ref, attention_cache_slot)?;
    let attention_window = call_tile!(resolve_decode_attention_window, layer)?;
    let attention_heads_ref = call_seq!(
        compute_decode_attention_ref,
        store,
        q_heads_ref,
        attention_cache_ref,
        format!("decode.layer.{}.attention", layer.layer_idx),
        attention_window,
        attention_kv_rows_per_tile
    )?;
    let attention_row = call_tile!(
        materialize_decode_attention_output_row,
        store,
        &attention_heads_ref
    )?;
    let projected = call_seq!(
        project_row_with_decode_source,
        store,
        &attention_row,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Output,
        layer.o_proj_shape.rows,
        projection_rows_per_tile,
        format!("decode.layer.{}.o_proj", layer.layer_idx)
    )?;

    Ok((projected, updated_cache_slot))
}

#[sequence]
fn compute_decode_ple_input(
    store: &mut AuthenticatedRasterTensorStore,
    token_id: u32,
    decode_input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    projection_rows_per_tile: usize,
) -> Result<Option<RasterActivationRow>> {
    if !layer.has_ple {
        return Ok(None);
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

    let projected = call_seq!(
        project_row_with_ple_source,
        store,
        decode_input,
        source,
        layer.layer_idx,
        ple_width,
        projection_rows_per_tile,
        format!("decode.layer.{}.ple.model_projection", layer.layer_idx)
    )?;
    let projected = call_tile!(
        scale_decode_row_optional,
        &projected,
        Some(scalars.projection_scalar)
    )?;
    let projected = call_tile!(
        rms_norm_decode_row,
        &projected,
        &norm_weights,
        scalars.rms_norm_eps,
        "decode PLE input RMSNorm"
    )?;
    let combined = call_tile!(add_decode_rows, &embedded, &projected)?;
    call_tile!(
        scale_decode_row_optional,
        &combined,
        Some(scalars.input_scale)
    )
    .map(Some)
}

#[tile]
fn read_decode_ple_scalars(
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<crate::shared::raster_decode_transition::GemmaDecodePleScalars> {
    auth_read!(source, GemmaDecodePleScalarsRequest)
}

#[tile]
fn read_decode_ple_projection_norm_weights(
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<Vec<crate::shared::det_num::Wgt>> {
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

#[sequence]
fn project_row_with_decode_source(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    matrix: GemmaDecodeLayerMatrixKind,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: String,
) -> Result<RasterActivationRow> {
    let state = call_tile!(
        init_decode_row_projection,
        store,
        input.clone(),
        DecodeProjectionKind::LayerMatrix { layer_idx, matrix },
        projection_rows,
        rows_per_tile,
        RasterTensorId::new(output_id)?,
        None
    )?;
    let state = call_recur_tile!(project_next_decode_projection_chunk, state, source, store)?;
    call_tile!(finalize_decode_row_projection, store, state)
}

#[sequence]
fn project_row_with_ple_source(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: String,
) -> Result<RasterActivationRow> {
    let state = call_tile!(
        init_decode_row_projection,
        store,
        input.clone(),
        DecodeProjectionKind::PleModel { layer_idx },
        projection_rows,
        rows_per_tile,
        RasterTensorId::new(output_id)?,
        None
    )?;
    let state = call_recur_tile!(project_next_decode_projection_chunk, state, source, store)?;
    call_tile!(finalize_decode_row_projection, store, state)
}

#[tile]
pub fn init_decode_row_projection(
    store: &mut AuthenticatedRasterTensorStore,
    input: RasterActivationRow,
    projection_kind: DecodeProjectionKind,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: RasterTensorId,
    softcap_bits: Option<i32>,
) -> Result<DecodeRowProjectionState> {
    if projection_rows == 0 {
        bail!("deterministic decode projection requires at least one projection row");
    }
    validate_projection_rows_per_tile(rows_per_tile)?;
    let input_width = input.width();
    let output_builder_ref =
        store.start_projection_output_builder(output_id, 1, projection_rows)?;
    Ok(DecodeRowProjectionState {
        input,
        projection_kind,
        next_projection_row_idx: 0,
        projection_rows,
        input_width,
        output_builder_ref,
        rows_per_tile,
        softcap_bits,
    })
}

#[tile(kind = recursive)]
pub fn project_next_decode_projection_chunk(
    mut state: DecodeRowProjectionState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, DecodeRowProjectionState)> {
    if state.next_projection_row_idx >= state.projection_rows {
        return Ok((true, state));
    }
    let end = state
        .next_projection_row_idx
        .saturating_add(state.rows_per_tile)
        .min(state.projection_rows);
    let start_projection_row_idx = state.next_projection_row_idx;
    let mut output_bits = Vec::with_capacity(end - start_projection_row_idx);
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
        let mut projected = project_row_with_weights(&state.input, &projection_row)?;
        if let Some(softcap_bits) = state.softcap_bits {
            projected = softcap_act(projected, Act::from_bits(softcap_bits));
        }
        output_bits.push(projected.to_bits());
        state.next_projection_row_idx += 1;
    }
    store.append_projection_output_chunk(
        &mut state.output_builder_ref,
        0,
        start_projection_row_idx,
        &output_bits,
    )?;
    Ok((false, state))
}

fn read_decode_projection_row(
    source: &AuthenticatedGemmaDecodeTransitionSource,
    projection_kind: &DecodeProjectionKind,
    row_idx: usize,
) -> Result<Vec<crate::shared::det_num::Wgt>> {
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

#[tile]
pub fn finalize_decode_row_projection(
    store: &mut AuthenticatedRasterTensorStore,
    state: DecodeRowProjectionState,
) -> Result<RasterActivationRow> {
    if state.next_projection_row_idx != state.projection_rows {
        bail!(
            "raster decode projection completed {} rows, expected {}",
            state.next_projection_row_idx,
            state.projection_rows
        );
    }
    let output_ref = store.finalize_projection_output_builder(state.output_builder_ref)?;
    let output = store.materialize_sequence(&output_ref)?;
    output
        .into_rows()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("raster decode projection produced no output row"))
}

fn reshape_row_heads(
    row: RasterActivationRow,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterAttentionHeadSequence> {
    crate::shared::raster_transformer_kernels::reshape_sequence_heads(
        &RasterActivationSequence::from_rows(vec![row]),
        num_heads,
        head_dim,
    )
}

#[sequence]
fn compute_decode_attention_ref(
    store: &mut AuthenticatedRasterTensorStore,
    query_ref: RasterAttentionHeadsRef,
    cache_ref: RasterKvCacheRef,
    id_prefix: String,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<RasterAttentionHeadsRef> {
    let state = call_tile!(
        init_decode_attention_state_from_refs,
        store,
        query_ref,
        cache_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        id_prefix,
        attention_window,
        kv_rows_per_tile
    )?;
    let state = call_recur_tile!(compute_next_decode_attention_head, state, store)?;
    call_tile!(finalize_decode_attention_state_ref, store, state)
}

#[tile]
pub fn init_decode_attention_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    query_ref: RasterAttentionHeadsRef,
    cache_ref: RasterKvCacheRef,
    output_id: RasterTensorId,
    attention_id_prefix: String,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<DecodeAttentionState> {
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
    let output_builder_ref = store.start_heads_builder(output_id, query_head_count, 1, head_dim)?;
    let phase = init_decode_attention_score_phase(store, &attention_id_prefix, 0, row_count)?;
    Ok(DecodeAttentionState {
        query_ref,
        cache_ref,
        output_builder_ref,
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
    })
}

fn init_decode_attention_score_phase(
    store: &mut AuthenticatedRasterTensorStore,
    id_prefix: &str,
    query_head_idx: usize,
    visible_row_count: usize,
) -> Result<DecodeAttentionPhase> {
    let score_builder_ref = store.start_sequence_builder(
        RasterTensorId::new(format!("{id_prefix}.scores.head_{query_head_idx}"))?,
        visible_row_count,
        1,
    )?;
    Ok(DecodeAttentionPhase::CollectScores {
        score_builder_ref,
        next_kv_offset: 0,
    })
}

#[tile(kind = recursive)]
pub fn compute_next_decode_attention_head(
    mut state: DecodeAttentionState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, DecodeAttentionState)> {
    if state.next_query_head_idx >= state.query_head_count {
        return Ok((true, state));
    }

    let query_head_idx = state.next_query_head_idx;
    let kv_head_idx = query_head_idx / state.kv_groups;
    let query = auth_read!(
        store,
        RasterHeadRowRequest {
            tensor_ref: state.query_ref.clone(),
            head_idx: query_head_idx,
            token_idx: 0,
        }
    )?;

    match state.phase.clone() {
        DecodeAttentionPhase::CollectScores {
            mut score_builder_ref,
            next_kv_offset,
        } => {
            let end = next_kv_offset
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
            for offset in next_kv_offset..end {
                let token_idx = state.key_start + offset;
                let key_row = read_decode_attention_kv_row(
                    store,
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
                store.append_sequence_row(
                    &mut score_builder_ref,
                    offset,
                    RasterActivationRow::from_acts(vec![score]),
                )?;
            }
            if end < state.row_count {
                state.phase = DecodeAttentionPhase::CollectScores {
                    score_builder_ref,
                    next_kv_offset: end,
                };
                return Ok((false, state));
            }
            let score_ref = store.finalize_sequence_builder(score_builder_ref)?;
            state.phase = DecodeAttentionPhase::FindSoftmaxMax {
                score_ref,
                next_score_row_idx: 0,
                max_index: None,
                max_logit_bits: 0,
            };
            Ok((false, state))
        }
        DecodeAttentionPhase::FindSoftmaxMax {
            score_ref,
            next_score_row_idx,
            mut max_index,
            mut max_logit_bits,
        } => {
            let end = next_score_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
            for row_idx in next_score_row_idx..end {
                let score = read_decode_attention_scalar_row(store, &score_ref, row_idx, "score")?;
                let score_bits = score.to_bits();
                if max_index.is_none() || score_bits > max_logit_bits {
                    max_index = Some(row_idx);
                    max_logit_bits = score_bits;
                }
            }
            if end < state.row_count {
                state.phase = DecodeAttentionPhase::FindSoftmaxMax {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                };
                return Ok((false, state));
            }
            let max_index = max_index
                .ok_or_else(|| anyhow!("decode attention softmax requires at least one score"))?;
            state.phase = DecodeAttentionPhase::SumSoftmaxExp {
                score_ref,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits: 0,
            };
            Ok((false, state))
        }
        DecodeAttentionPhase::SumSoftmaxExp {
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
                let score = read_decode_attention_scalar_row(store, &score_ref, row_idx, "score")?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                sum_exp = acc_add_sat(sum_exp, exp_term);
            }
            sum_exp_bits = sum_exp.to_bits();
            if end < state.row_count {
                state.phase = DecodeAttentionPhase::SumSoftmaxExp {
                    score_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                };
                return Ok((false, state));
            }
            if sum_exp_bits == 0 {
                bail!("decode attention softmax exp sum is zero");
            }
            let raw_weight_builder_ref = store.start_sequence_builder(
                RasterTensorId::new(format!(
                    "{}.raw_weights.head_{query_head_idx}",
                    state.attention_id_prefix
                ))?,
                state.row_count,
                1,
            )?;
            state.phase = DecodeAttentionPhase::BuildRawSoftmaxWeights {
                score_ref,
                raw_weight_builder_ref,
                next_score_row_idx: 0,
                max_index,
                max_logit_bits,
                sum_exp_bits,
                summed_weight_bits: 0,
            };
            Ok((false, state))
        }
        DecodeAttentionPhase::BuildRawSoftmaxWeights {
            score_ref,
            mut raw_weight_builder_ref,
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
                let score = read_decode_attention_scalar_row(store, &score_ref, row_idx, "score")?;
                let exp_term = attention_softmax_exp_term(score, max_logit);
                let weight = attention_softmax_raw_weight(exp_term, sum_exp);
                store.append_sequence_row(
                    &mut raw_weight_builder_ref,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
                summed_weight = add_sat(summed_weight, weight);
            }
            summed_weight_bits = summed_weight.to_bits();
            if end < state.row_count {
                state.phase = DecodeAttentionPhase::BuildRawSoftmaxWeights {
                    score_ref,
                    raw_weight_builder_ref,
                    next_score_row_idx: end,
                    max_index,
                    max_logit_bits,
                    sum_exp_bits,
                    summed_weight_bits,
                };
                return Ok((false, state));
            }
            let raw_weight_ref = store.finalize_sequence_builder(raw_weight_builder_ref)?;
            let final_weight_builder_ref = store.start_sequence_builder(
                RasterTensorId::new(format!(
                    "{}.weights.head_{query_head_idx}",
                    state.attention_id_prefix
                ))?,
                state.row_count,
                1,
            )?;
            let residual = attention_softmax_residual(Act::from_bits(summed_weight_bits));
            state.phase = DecodeAttentionPhase::CorrectSoftmaxResidual {
                raw_weight_ref,
                final_weight_builder_ref,
                next_weight_row_idx: 0,
                max_index,
                residual_bits: residual.to_bits(),
            };
            Ok((false, state))
        }
        DecodeAttentionPhase::CorrectSoftmaxResidual {
            raw_weight_ref,
            mut final_weight_builder_ref,
            next_weight_row_idx,
            max_index,
            residual_bits,
        } => {
            let end = next_weight_row_idx
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
            let residual = Act::from_bits(residual_bits);
            for row_idx in next_weight_row_idx..end {
                let mut weight =
                    read_decode_attention_scalar_row(store, &raw_weight_ref, row_idx, "weight")?;
                if row_idx == max_index {
                    weight = add_sat(weight, residual);
                }
                store.append_sequence_row(
                    &mut final_weight_builder_ref,
                    row_idx,
                    RasterActivationRow::from_acts(vec![weight]),
                )?;
            }
            if end < state.row_count {
                state.phase = DecodeAttentionPhase::CorrectSoftmaxResidual {
                    raw_weight_ref,
                    final_weight_builder_ref,
                    next_weight_row_idx: end,
                    max_index,
                    residual_bits,
                };
                return Ok((false, state));
            }
            let weight_ref = store.finalize_sequence_builder(final_weight_builder_ref)?;
            state.phase = DecodeAttentionPhase::ApplyValues {
                weight_ref,
                next_kv_offset: 0,
                weighted_sum_acc_bits: vec![0; query.width()],
            };
            Ok((false, state))
        }
        DecodeAttentionPhase::ApplyValues {
            weight_ref,
            next_kv_offset,
            mut weighted_sum_acc_bits,
        } => {
            let end = next_kv_offset
                .saturating_add(state.kv_rows_per_tile)
                .min(state.row_count);
            for offset in next_kv_offset..end {
                let token_idx = state.key_start + offset;
                let weight =
                    read_decode_attention_scalar_row(store, &weight_ref, offset, "weight")?;
                let value_row = read_decode_attention_kv_row(
                    store,
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
                state.phase = DecodeAttentionPhase::ApplyValues {
                    weight_ref,
                    next_kv_offset: end,
                    weighted_sum_acc_bits,
                };
                return Ok((false, state));
            }
            let output_row = RasterActivationRow::from_acts(
                weighted_sum_acc_bits
                    .into_iter()
                    .map(|bits| requantize(Acc::from_bits(bits)))
                    .collect(),
            );
            store.append_head_row(&mut state.output_builder_ref, query_head_idx, 0, output_row)?;
            state.next_query_head_idx += 1;
            if state.next_query_head_idx < state.query_head_count {
                state.phase = init_decode_attention_score_phase(
                    store,
                    &state.attention_id_prefix,
                    state.next_query_head_idx,
                    state.row_count,
                )?;
            }
            Ok((false, state))
        }
    }
}

fn read_decode_attention_kv_row(
    store: &AuthenticatedRasterTensorStore,
    cache_ref: &RasterKvCacheRef,
    row_kind: RasterKvRowKind,
    head_idx: usize,
    token_idx: usize,
) -> Result<RasterActivationRow> {
    auth_read!(
        store,
        RasterKvRowRequest {
            cache_ref: cache_ref.clone(),
            row_kind,
            head_idx,
            token_idx,
        }
    )
}

fn read_decode_attention_scalar_row(
    store: &AuthenticatedRasterTensorStore,
    tensor_ref: &crate::shared::raster_row_store::RasterActivationSequenceRef,
    row_idx: usize,
    label: &str,
) -> Result<Act> {
    let row = auth_read!(
        store,
        RasterSequenceRowRequest {
            tensor_ref: tensor_ref.clone(),
            row_idx,
        }
    )?;
    if row.width() != 1 {
        bail!(
            "decode attention {label} row {row_idx} has width {}, expected 1",
            row.width()
        );
    }
    Ok(row.acts()[0])
}

#[tile]
pub fn finalize_decode_attention_state_ref(
    store: &mut AuthenticatedRasterTensorStore,
    state: DecodeAttentionState,
) -> Result<RasterAttentionHeadsRef> {
    if state.next_query_head_idx != state.query_head_count {
        bail!(
            "decode attention finalized at head {}, expected {} heads",
            state.next_query_head_idx,
            state.query_head_count
        );
    }
    store.finalize_heads_builder(state.output_builder_ref)
}

#[sequence]
fn append_decode_kv_cache_ref(
    store: &mut AuthenticatedRasterTensorStore,
    cache_slot: DecodeLayerCacheSlot,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    layer_idx: usize,
    cache_window: Option<usize>,
    rows_per_tile: usize,
) -> Result<DecodeLayerCacheSlot> {
    let state = call_tile!(
        init_decode_kv_cache_append_state,
        store,
        cache_slot,
        key_ref,
        value_ref,
        RasterTensorId::new(format!("decode.updated.cache.{layer_idx}.keys"))?,
        RasterTensorId::new(format!("decode.updated.cache.{layer_idx}.values"))?,
        cache_window,
        rows_per_tile
    )?;
    let state = call_recur_tile!(compute_next_decode_kv_cache_append_row, state, store)?;
    call_tile!(finalize_decode_kv_cache_append_state, store, state)
}

#[tile]
fn init_decode_kv_cache_append_state(
    store: &mut AuthenticatedRasterTensorStore,
    cache_slot: DecodeLayerCacheSlot,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    cache_window: Option<usize>,
    rows_per_tile: usize,
) -> Result<DecodeKvCacheAppendState> {
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
    let output_builder_ref = store.start_kv_cache_builder(
        keys_id,
        values_id,
        head_count,
        retained_old_len + 1,
        head_dim,
    )?;

    Ok(DecodeKvCacheAppendState {
        old_cache_ref,
        key_ref,
        value_ref,
        output_builder_ref,
        retained_old_start: retained_start,
        retained_old_len,
        next_head_idx: 0,
        next_old_offset: 0,
        head_count,
        rows_per_tile,
    })
}

#[tile(kind = recursive)]
fn compute_next_decode_kv_cache_append_row(
    mut state: DecodeKvCacheAppendState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, DecodeKvCacheAppendState)> {
    if state.next_head_idx >= state.head_count {
        return Ok((true, state));
    }

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
            let key_row = auth_read!(
                store,
                RasterKvRowRequest {
                    cache_ref: old_cache_ref.clone(),
                    row_kind: RasterKvRowKind::Key,
                    head_idx: state.next_head_idx,
                    token_idx: input_token_idx,
                }
            )?;
            let value_row = auth_read!(
                store,
                RasterKvRowRequest {
                    cache_ref: old_cache_ref.clone(),
                    row_kind: RasterKvRowKind::Value,
                    head_idx: state.next_head_idx,
                    token_idx: input_token_idx,
                }
            )?;
            store.append_kv_row(
                state.output_builder_ref.keys_mut(),
                RasterKvRowKind::Key,
                state.next_head_idx,
                old_offset,
                key_row,
            )?;
            store.append_kv_row(
                state.output_builder_ref.values_mut(),
                RasterKvRowKind::Value,
                state.next_head_idx,
                old_offset,
                value_row,
            )?;
        }
        state.next_old_offset = end;
        if state.next_old_offset < state.retained_old_len {
            return Ok((false, state));
        }
    }

    let output_token_idx = state.retained_old_len;
    let key_row = auth_read!(
        store,
        RasterHeadRowRequest {
            tensor_ref: state.key_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: 0,
        }
    )?;
    let value_row = auth_read!(
        store,
        RasterHeadRowRequest {
            tensor_ref: state.value_ref.clone(),
            head_idx: state.next_head_idx,
            token_idx: 0,
        }
    )?;
    store.append_kv_row(
        state.output_builder_ref.keys_mut(),
        RasterKvRowKind::Key,
        state.next_head_idx,
        output_token_idx,
        key_row,
    )?;
    store.append_kv_row(
        state.output_builder_ref.values_mut(),
        RasterKvRowKind::Value,
        state.next_head_idx,
        output_token_idx,
        value_row,
    )?;
    state.next_head_idx += 1;
    state.next_old_offset = 0;
    Ok((false, state))
}

#[tile]
fn finalize_decode_kv_cache_append_state(
    store: &mut AuthenticatedRasterTensorStore,
    state: DecodeKvCacheAppendState,
) -> Result<DecodeLayerCacheSlot> {
    if state.next_head_idx != state.head_count {
        bail!(
            "decode cache append finalized at head {}, expected {} heads",
            state.next_head_idx,
            state.head_count
        );
    }
    Ok(DecodeLayerCacheSlot::Ref(
        store.finalize_kv_cache_builder(state.output_builder_ref)?,
    ))
}

fn register_decode_layer_cache(
    store: &mut AuthenticatedRasterTensorStore,
    id_prefix: &str,
    layer_idx: usize,
    cache: RasterKvCache,
) -> Result<DecodeLayerCacheSlot> {
    if cache.current_len() == 0 {
        return Ok(DecodeLayerCacheSlot::Empty {
            num_kv_heads: cache.head_count(),
        });
    }
    let cache_ref = store.insert_kv_cache(
        RasterTensorId::new(format!("{id_prefix}.{layer_idx}.keys"))?,
        RasterTensorId::new(format!("{id_prefix}.{layer_idx}.values"))?,
        cache,
    )?;
    Ok(DecodeLayerCacheSlot::Ref(cache_ref))
}

fn materialize_decode_layer_cache_from_store(
    store: &AuthenticatedRasterTensorStore,
    cache: &DecodeLayerCacheSlot,
) -> Result<RasterKvCache> {
    match cache {
        DecodeLayerCacheSlot::Empty { num_kv_heads } => Ok(RasterKvCache::empty(*num_kv_heads)),
        DecodeLayerCacheSlot::Ref(cache_ref) => store.materialize_kv_cache(cache_ref),
    }
}

fn materialize_decode_checkpoint_caches(
    store: &AuthenticatedRasterTensorStore,
    state: &DecodeTransitionRasterState,
    layer_idx: usize,
) -> Result<Vec<LayerKvCache>> {
    // Checkpoint payloads keep the legacy full-cache shape; replay paths stay ref-backed.
    let mut caches = state
        .updated_layer_caches
        .iter()
        .map(|cache| {
            materialize_decode_layer_cache_from_store(store, cache).map(layer_cache_from_raster)
        })
        .collect::<Result<Vec<_>>>()?;
    caches.extend(
        state
            .original_layer_caches
            .iter()
            .skip(layer_idx + 1)
            .map(|cache| {
                materialize_decode_layer_cache_from_store(store, cache).map(layer_cache_from_raster)
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
mod tests {
    use super::{
        append_decode_kv_cache_ref, compute_next_decode_layer, finalize_decode_layer_state,
        init_decode_logits_projection, init_decode_transition_state, init_decode_transition_store,
        main, materialize_decode_layer_cache_from_store, prepare_next_decode_layer_context,
        register_decode_layer_cache, run, DecodeLayerCacheSlot, RasterDecodeTransitionInputRoots,
    };
    use crate::shared::artifact_io::ArtifactIo;
    use crate::shared::det_num::{Acc, Act, Wgt};
    use crate::shared::input::InferenceExecutionMode;
    use crate::shared::raster_artifact_store::{
        token_id_leaf, RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots,
        RasterSelectedTokenRef, RasterTokenIdSequenceRef,
    };
    use crate::shared::raster_decode_transition::AuthenticatedGemmaDecodeTransitionSource;
    use crate::shared::raster_row_store::RasterTensorId;
    use crate::shared::raster_transformer_kernels::{
        RasterActivationRow, RasterAttentionHeadSequence, RasterKvCache,
    };
    use crate::shared::transformer::{
        DetNumMatrix, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4TransformerModel,
        GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, TransformerDecodeState,
    };
    use crate::RasterSizingControls;
    use anyhow::{Context, Result};
    use std::collections::VecDeque;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    #[test]
    fn raster_decode_transition_matches_deterministic_no_ple() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(1);

        let raster = run(decode_state.clone(), 1, &source, raster_sizing(1))
            .expect("raster decode should run");
        let deterministic = crate::decode_transition::run_with_mode(
            decode_state,
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("deterministic decode should run");

        assert_eq!(raster.activation_state, deterministic.activation_state);
        assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
        assert_eq!(
            raster.transformer_decode_state,
            deterministic.transformer_decode_state
        );
    }

    #[test]
    fn root_backed_main_reads_selected_token_ref() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(1);
        let (roots, selected_token_ref) =
            selected_token_ref("decode.transition.selected", 1).expect("selected token ref");

        let raster = main(
            RasterDecodeTransitionInputRoots {
                artifact_store_roots: roots,
                transformer_decode_state: decode_state.clone(),
                selected_token_ref,
                decode_transition_source_name: source.identifier().to_string(),
                output_source_prefix: "decode.transition.root-backed".to_string(),
                raster_sizing: raster_sizing(1),
            },
            &source,
        )
        .expect("root-backed raster decode should run")
        .transition_result;
        let deterministic = crate::decode_transition::run_with_mode(
            decode_state,
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("deterministic decode should run");

        assert_eq!(raster.activation_state, deterministic.activation_state);
        assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
        assert_eq!(
            raster.transformer_decode_state,
            deterministic.transformer_decode_state
        );
    }

    #[test]
    fn root_backed_main_rejects_missing_selected_token_root() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let (_roots, selected_token_ref) =
            selected_token_ref("decode.transition.missing.selected", 1)
                .expect("selected token ref");

        let error = main(
            RasterDecodeTransitionInputRoots {
                artifact_store_roots: RasterArtifactStoreRoots::default(),
                transformer_decode_state: decode_state_with_cache(1),
                selected_token_ref,
                decode_transition_source_name: source.identifier().to_string(),
                output_source_prefix: "decode.transition.missing".to_string(),
                raster_sizing: raster_sizing(1),
            },
            &source,
        )
        .expect_err("missing selected-token root should fail");

        assert!(error.to_string().contains("not present"));
    }

    #[test]
    fn raster_decode_transition_matches_across_projection_and_attention_chunks() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(3);
        let deterministic = crate::decode_transition::run_with_mode(
            decode_state.clone(),
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("deterministic decode should run");

        for sizing in [
            raster_sizing_with_attention(1, 1),
            raster_sizing_with_attention(2, 1),
            raster_sizing_with_attention(8, 2),
        ] {
            let raster =
                run(decode_state.clone(), 1, &source, sizing).expect("raster decode should run");

            assert_eq!(raster.activation_state, deterministic.activation_state);
            assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
            assert_eq!(
                raster.transformer_decode_state,
                deterministic.transformer_decode_state
            );
        }
    }

    #[test]
    fn raster_decode_rejects_zero_sizing_controls() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(1);

        let projection_error = run(
            decode_state.clone(),
            1,
            &source,
            raster_sizing_with_attention(0, 1),
        )
        .expect_err("zero projection chunk should fail");
        assert!(projection_error
            .to_string()
            .contains("projection rows per tile"));

        let attention_error = run(decode_state, 1, &source, raster_sizing_with_attention(1, 0))
            .expect_err("zero attention chunk should fail");
        assert!(attention_error
            .to_string()
            .contains("attention KV rows per tile"));
    }

    #[test]
    fn decode_recursive_state_serializes_refs_and_builders() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(2);
        let mut store = init_decode_transition_store();
        let state =
            init_decode_transition_state(&mut store, decode_state, 1, &source, raster_sizing(1))
                .expect("state should initialize");
        let encoded = serde_json::to_string(&state).expect("state should serialize");

        assert!(encoded.contains("artifact_store_roots"));
        assert!(encoded.contains("decode_input_ref"));
        assert!(encoded.contains("current_activation_ref"));
        assert!(encoded.contains("original_layer_caches"));
        assert!(encoded.contains("Ref"));
        assert!(!encoded.contains("decode_input_activation"));
        assert!(!encoded.contains("current_activation\":["));
        assert!(!encoded.contains("\"keys\":[[["));
        assert!(!encoded.contains("\"values\":[[["));

        let normalized =
            RasterActivationRow::from_acts(vec![Act::from_num(0.0), Act::from_num(0.0)]);
        let logits_state =
            init_decode_logits_projection(&mut store, normalized, &source, 1).expect("logits init");
        let encoded = serde_json::to_string(&logits_state).expect("logits state should serialize");

        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("logit_bits"));
    }

    #[test]
    fn decode_layer_context_serializes_cache_refs_without_materialized_rows() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(2);
        let mut store = init_decode_transition_store();
        let state =
            init_decode_transition_state(&mut store, decode_state, 1, &source, raster_sizing(1))
                .expect("state should initialize");

        let context =
            prepare_next_decode_layer_context(&state, &source).expect("context should prepare");
        let encoded = serde_json::to_string(&context).expect("context should serialize");

        assert!(encoded.contains("cache_slot"));
        assert!(encoded.contains("Ref"));
        assert!(!encoded.contains("\"keys\":[[["));
        assert!(!encoded.contains("\"values\":[[["));
        assert!(!encoded.contains("act_bits"));
        assert_eq!(store.materialization_counts().1, 0);
    }

    #[test]
    fn decode_layer_replay_avoids_kv_materialization_until_public_boundary() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(2);
        let mut store = init_decode_transition_store();
        let state =
            init_decode_transition_state(&mut store, decode_state, 1, &source, raster_sizing(1))
                .expect("state should initialize");
        assert_eq!(store.materialization_counts().1, 0);

        let (_done, state) =
            compute_next_decode_layer(state, &source, &mut store).expect("layer should compute");
        assert_eq!(
            store.materialization_counts().1,
            0,
            "normal decode layer replay should not materialize full KV caches"
        );

        finalize_decode_layer_state(&store, state).expect("public layer output should finalize");
        assert!(
            store.materialization_counts().1 > 0,
            "public/cache compatibility boundary should materialize KV caches"
        );
    }

    #[test]
    fn decode_ref_cache_append_handles_empty_and_sliding_windows() {
        let mut store = init_decode_transition_store();
        let key_ref = store
            .insert_attention_heads(
                RasterTensorId::new("test.empty.key").expect("key id"),
                heads_from_rows(&[&[7]]),
            )
            .expect("key ref");
        let value_ref = store
            .insert_attention_heads(
                RasterTensorId::new("test.empty.value").expect("value id"),
                heads_from_rows(&[&[8]]),
            )
            .expect("value ref");
        let empty_slot = DecodeLayerCacheSlot::Empty { num_kv_heads: 1 };

        let appended =
            append_decode_kv_cache_ref(&mut store, empty_slot, key_ref, value_ref, 0, None, 1)
                .expect("empty append should build ref");
        let appended = materialize_decode_layer_cache_from_store(&store, &appended).expect("cache");
        assert_eq!(cache_key_bits(&appended), vec![vec![7]]);
        assert_eq!(cache_value_bits(&appended), vec![vec![8]]);

        let old_cache = RasterKvCache::from_heads(
            vec![rows_from_bits(&[1, 2, 3])],
            vec![rows_from_bits(&[11, 12, 13])],
        )
        .expect("old cache");
        let old_slot = register_decode_layer_cache(&mut store, "test.old", 0, old_cache)
            .expect("old cache slot");
        let key_ref = store
            .insert_attention_heads(
                RasterTensorId::new("test.sliding.key").expect("key id"),
                heads_from_rows(&[&[4]]),
            )
            .expect("key ref");
        let value_ref = store
            .insert_attention_heads(
                RasterTensorId::new("test.sliding.value").expect("value id"),
                heads_from_rows(&[&[14]]),
            )
            .expect("value ref");

        let appended =
            append_decode_kv_cache_ref(&mut store, old_slot, key_ref, value_ref, 1, Some(2), 1)
                .expect("sliding append should build ref");
        let appended = materialize_decode_layer_cache_from_store(&store, &appended).expect("cache");
        assert_eq!(cache_key_bits(&appended), vec![vec![3, 4]]);
        assert_eq!(cache_value_bits(&appended), vec![vec![13, 14]]);
    }

    #[test]
    fn decode_ref_cache_append_rejects_invalid_shapes() {
        let mut store = init_decode_transition_store();
        let key_ref = store
            .insert_attention_heads(
                RasterTensorId::new("test.bad.key").expect("key id"),
                heads_from_rows(&[&[1]]),
            )
            .expect("key ref");
        let value_ref = store
            .insert_attention_heads(
                RasterTensorId::new("test.bad.value").expect("value id"),
                heads_from_rows(&[&[2]]),
            )
            .expect("value ref");

        let error = append_decode_kv_cache_ref(
            &mut store,
            DecodeLayerCacheSlot::Empty { num_kv_heads: 2 },
            key_ref,
            value_ref,
            0,
            None,
            1,
        )
        .expect_err("head mismatch should fail");

        assert!(error.to_string().contains("empty cache head count"));
    }

    #[test]
    fn raster_decode_transition_matches_sliding_cache_window() {
        let (_path, model) = no_ple_model(true);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");

        for cache_len in [0, 1, 2, 3] {
            let decode_state = decode_state_with_cache(cache_len);
            let raster = run(decode_state.clone(), 1, &source, raster_sizing(1))
                .expect("raster decode should run");
            let deterministic = crate::decode_transition::run_with_mode(
                decode_state,
                1,
                &model,
                InferenceExecutionMode::Deterministic,
            )
            .expect("deterministic decode should run");

            assert_eq!(
                raster.transformer_decode_state.layer_caches[0].current_len(),
                1,
                "sliding cache should retain one row for cache length {cache_len}"
            );
            assert_eq!(
                raster.transformer_decode_state, deterministic.transformer_decode_state,
                "sliding decode parity failed for cache length {cache_len}"
            );
        }
    }

    #[test]
    fn raster_decode_transition_matches_deterministic_with_attention_k_eq_v() {
        let (_path, mut model) = no_ple_model(false);
        model.layers[0].v_proj = None;
        model.layers[0].attention_k_eq_v = true;
        assert_raster_matches_deterministic(&model, decode_state_with_cache(2));
    }

    #[test]
    fn raster_decode_transition_matches_deterministic_with_layer_scalar() {
        let (_path, mut model) = no_ple_model(false);
        model.layers[0].layer_scalar = Some(0.5);
        model.layers[0].layer_scalar_det = Some(Act::from_num(0.5));
        assert_raster_matches_deterministic(&model, decode_state_with_cache(2));
    }

    #[test]
    fn raster_decode_transition_matches_deterministic_with_ple() {
        let (_paths, model) = ple_model();
        assert_raster_matches_deterministic(&model, decode_state_with_cache(2));
    }

    #[test]
    fn raster_decode_transition_matches_deterministic_with_donor_cache() {
        let (_path, mut model) = no_ple_model(false);
        let mut donor_layer = model.layers[0].clone();
        donor_layer.kv_shared_layer_index = Some(0);
        model.layers.push(donor_layer);
        assert_raster_matches_deterministic(&model, decode_state_with_layer_count(2, 2));
    }

    #[test]
    fn raster_decode_rejects_f32_only_cache() {
        let (_path, model) = no_ple_model(false);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = TransformerDecodeState {
            layer_caches: vec![LayerKvCache::from_f32_heads(
                vec![VecDeque::from([vec![0.0, 0.0]])],
                vec![VecDeque::from([vec![0.0, 0.0]])],
            )],
            position: 1,
            token_count: 1,
        };

        let error =
            run(decode_state, 1, &source, raster_sizing(1)).expect_err("f32 cache should fail");

        assert!(error.to_string().contains("canonical layer cache keys"));
    }

    #[test]
    fn decode_source_rejects_non_deterministic_model() {
        let (_path, mut model) = no_ple_model(false);
        model.provenance = Gemma4ModelProvenance::Fp32;

        let error = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect_err("fp32 model should fail");

        assert!(error.to_string().contains(".detwgt artifact"));
    }

    fn assert_raster_matches_deterministic(
        model: &Gemma4TransformerModel,
        decode_state: TransformerDecodeState,
    ) {
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", model)
            .expect("source should build");
        let raster = run(
            decode_state.clone(),
            1,
            &source,
            raster_sizing_with_attention(2, 1),
        )
        .expect("raster decode should run");
        let deterministic = crate::decode_transition::run_with_mode(
            decode_state,
            1,
            model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("deterministic decode should run");

        assert_eq!(raster.activation_state, deterministic.activation_state);
        assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
        assert_eq!(
            raster.transformer_decode_state,
            deterministic.transformer_decode_state
        );
    }

    fn selected_token_ref(
        source_name: &str,
        token_id: u32,
    ) -> Result<(RasterArtifactStoreRoots, RasterSelectedTokenRef)> {
        ArtifactIo::reset_store();
        let roots = ArtifactIo::export_store_roots();
        let (roots, artifact_ref) = ArtifactIo::insert_artifact_with_roots(
            &roots,
            RasterArtifactId::new(source_name)?,
            RasterArtifactMetadata::token_ids(1),
            vec![token_id_leaf(token_id)],
        )?;
        let selected_token_ref =
            RasterSelectedTokenRef::new(RasterTokenIdSequenceRef::new(artifact_ref)?)?;
        Ok((roots, selected_token_ref))
    }

    fn no_ple_model(sliding: bool) -> (PathBuf, Gemma4TransformerModel) {
        let hidden_width = 2;
        let matrices = vec![
            vec![
                vec![Wgt::from_num(0.0), Wgt::from_num(0.0)],
                vec![Wgt::from_num(1.0), Wgt::from_num(-0.5)],
            ],
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
        ];
        let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
        let mut sources = sources.into_iter();
        let embedding_source = sources.next().expect("embedding source");
        let layer = Gemma4LayerWeights {
            attention_kind: if sliding {
                Gemma4AttentionKind::Sliding
            } else {
                Gemma4AttentionKind::Full
            },
            hidden_size: hidden_width,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden_width,
            sliding_window: sliding.then_some(1),
            cache_sliding_window: sliding.then_some(1),
            rms_norm_eps: 0.0,
            rms_norm_eps_det: Some(Acc::from_num(0.0)),
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: det_matrix(sources.next().expect("q source")),
            k_proj: det_matrix(sources.next().expect("k source")),
            v_proj: Some(det_matrix(sources.next().expect("v source"))),
            o_proj: det_matrix(sources.next().expect("o source")),
            q_norm_weight: vec![1.0; hidden_width],
            q_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            k_norm_weight: vec![1.0; hidden_width],
            k_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            input_layernorm_weight: vec![1.0; hidden_width],
            input_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            post_attention_layernorm_weight: vec![1.0; hidden_width],
            post_attention_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            pre_feedforward_layernorm_weight: vec![1.0; hidden_width],
            pre_feedforward_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            post_feedforward_layernorm_weight: vec![1.0; hidden_width],
            post_feedforward_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            gate_proj: det_matrix(sources.next().expect("gate source")),
            up_proj: det_matrix(sources.next().expect("up source")),
            down_proj: det_matrix(sources.next().expect("down source")),
            ple: None,
            layer_scalar: None,
            layer_scalar_det: None,
        };

        (
            path,
            Gemma4TransformerModel {
                provenance: Gemma4ModelProvenance::DetNumWgt,
                embedding_table: None,
                embedding_source: Some(GemmaEmbeddingTensorSource::Deterministic {
                    source: embedding_source,
                    scale: 1.0,
                    det_cache: Arc::new(Mutex::new(None)),
                }),
                layers: vec![layer],
                ple_global: None,
                final_norm_weight: vec![1.0; hidden_width],
                final_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
                logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                    weight: MatrixF32 {
                        rows: 2,
                        cols: hidden_width,
                        values: vec![1.0, 0.0, 0.0, 1.0],
                    },
                    det_weight: Some(Arc::new(det_num_matrix(identity_matrix()))),
                },
                final_logit_softcapping: None,
                final_logit_softcapping_det: None,
                rms_norm_eps: 0.0,
                rms_norm_eps_det: Some(Acc::from_num(0.0)),
            },
        )
    }

    fn ple_model() -> (Vec<PathBuf>, Gemma4TransformerModel) {
        let (base_path, mut model) = no_ple_model(false);
        let hidden_width = 2;
        let (ple_layer_path, ple_layer_sources) =
            write_det_matrices(vec![identity_matrix(), identity_matrix()])
                .expect("fixture PLE layer weights should write");
        let mut ple_layer_sources = ple_layer_sources.into_iter();
        model.layers[0].ple = Some(crate::Gemma4PleLayerWeights {
            input_gate: det_matrix(ple_layer_sources.next().expect("PLE input gate")),
            layer_projection: det_matrix(ple_layer_sources.next().expect("PLE layer projection")),
            post_input_norm_weight: vec![1.0; hidden_width],
            post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        });

        let (ple_global_path, ple_global_sources) =
            write_det_matrices(vec![identity_matrix(), identity_matrix()])
                .expect("fixture PLE global weights should write");
        let mut ple_global_sources = ple_global_sources.into_iter();
        model.ple_global = Some(crate::Gemma4PleGlobalWeights::from_det_num_sources(
            vec![ple_global_sources.next().expect("PLE token embeddings")],
            vec![ple_global_sources.next().expect("PLE model projection")],
            vec![1.0; hidden_width],
            1.0,
            1.0,
            1.0,
        ));

        (vec![base_path, ple_layer_path, ple_global_path], model)
    }

    fn raster_sizing(projection_rows_per_tile: usize) -> RasterSizingControls {
        raster_sizing_with_attention(projection_rows_per_tile, 1)
    }

    fn raster_sizing_with_attention(
        projection_rows_per_tile: usize,
        attention_kv_rows_per_tile: usize,
    ) -> RasterSizingControls {
        RasterSizingControls {
            projection_rows_per_tile,
            attention_kv_rows_per_tile,
            sequence_rows_per_tile: 1,
            head_rows_per_tile: 1,
            tokenizer_bpe_pairs_per_tile:
                crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE,
            tokenizer_bpe_pieces_per_tile:
                crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE,
            output_byte_flush_bytes_per_tile:
                crate::InferenceControls::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        }
    }

    fn decode_state_with_cache(cache_len: usize) -> TransformerDecodeState {
        decode_state_with_layer_count(1, cache_len)
    }

    fn decode_state_with_layer_count(
        layer_count: usize,
        cache_len: usize,
    ) -> TransformerDecodeState {
        let key_rows = (0..cache_len)
            .map(|_| vec![Act::from_num(0.0), Act::from_num(0.0)])
            .collect::<VecDeque<_>>();
        let value_rows = key_rows.clone();
        TransformerDecodeState {
            layer_caches: (0..layer_count)
                .map(|_| {
                    LayerKvCache::from_det_heads(vec![key_rows.clone()], vec![value_rows.clone()])
                })
                .collect(),
            position: cache_len,
            token_count: cache_len,
        }
    }

    fn heads_from_rows(heads: &[&[i32]]) -> RasterAttentionHeadSequence {
        RasterAttentionHeadSequence::from_heads(
            heads
                .iter()
                .map(|rows| rows_from_bits(rows))
                .collect::<Vec<_>>(),
        )
    }

    fn rows_from_bits(bits: &[i32]) -> Vec<RasterActivationRow> {
        bits.iter()
            .map(|bits| RasterActivationRow::from_acts(vec![Act::from_bits(*bits)]))
            .collect()
    }

    fn cache_key_bits(cache: &RasterKvCache) -> Vec<Vec<i32>> {
        cache
            .keys()
            .iter()
            .map(|head| head.iter().map(first_act_bits).collect())
            .collect()
    }

    fn cache_value_bits(cache: &RasterKvCache) -> Vec<Vec<i32>> {
        cache
            .values()
            .iter()
            .map(|head| head.iter().map(first_act_bits).collect())
            .collect()
    }

    fn first_act_bits(row: &RasterActivationRow) -> i32 {
        row.acts()
            .first()
            .expect("test row should have one value")
            .to_bits()
    }

    fn identity_matrix() -> Vec<Vec<Wgt>> {
        vec![
            vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
            vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
        ]
    }

    fn zero_matrix(width: usize) -> Vec<Vec<Wgt>> {
        vec![vec![Wgt::from_num(0.0); width]; width]
    }

    fn det_matrix(source: DetNumTensorSliceSource) -> Gemma4LayerMatrixSource {
        Gemma4LayerMatrixSource::from_det_num_source(source)
    }

    fn det_num_matrix(rows: Vec<Vec<Wgt>>) -> DetNumMatrix {
        DetNumMatrix {
            rows: rows.len(),
            cols: rows.first().map(Vec::len).unwrap_or(0),
            values: rows
                .into_iter()
                .flat_map(|row| row.into_iter().map(|value| value.to_bits()))
                .collect(),
        }
    }

    fn write_det_matrices(
        matrices: Vec<Vec<Vec<Wgt>>>,
    ) -> Result<(PathBuf, Vec<DetNumTensorSliceSource>)> {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-decode-transition-{}-{}-{}.detwgt",
            std::process::id(),
            unique_suffix,
            crate::trace::sha256_hex(&format!("{:?}", matrices))
        ));
        let mut bytes = Vec::new();
        let mut sources = Vec::new();

        for rows in matrices {
            let data_offset = bytes.len();
            for row in &rows {
                for value in row {
                    bytes.extend(value.to_bits().to_le_bytes());
                }
            }
            sources.push(det_source(&path, rows.len(), rows[0].len(), data_offset));
        }

        std::fs::write(&path, bytes)
            .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
        Ok((path, sources))
    }

    fn det_source(
        path: &Path,
        rows: usize,
        cols: usize,
        data_offset: usize,
    ) -> DetNumTensorSliceSource {
        DetNumTensorSliceSource {
            weights_path: path.to_path_buf(),
            total_rows: rows,
            total_cols: cols,
            data_offset,
            row_offset: 0,
            row_count: rows,
            col_offset: 0,
            col_count: cols,
        }
    }
}
