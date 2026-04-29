use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile_result, call_tile, sequence, tile,
};
use crate::shared::det_num::{softcap_act, Act};
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
use crate::shared::raster_transformer_kernels::{
    add_sequences, apply_rope_to_heads, attention_output_row, combine_attention_heads,
    gelu_sequence, mul_sequences, project_row_with_weights, rms_norm_heads, rms_norm_sequence,
    scale_sequence, value_rms_norm_heads, RasterActivationRow, RasterActivationSequence,
    RasterAttentionHeadSequence, RasterKvCache,
};
use crate::shared::transformer::{
    ActivationSequence, InternalActivationSequence, InternalLogits, LayerKvCache, PrefillLogits,
    TransformerDecodeState, TransformerDecodeStepResult,
};

use super::tiles::ActivationSequenceWithCache;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeTransitionRasterState {
    decode_input: RasterActivationRow,
    current_activation: RasterActivationRow,
    next_token: u32,
    position: usize,
    token_count: usize,
    next_layer_idx: usize,
    layer_count: usize,
    original_layer_caches: Vec<RasterKvCache>,
    updated_layer_caches: Vec<RasterKvCache>,
    completed_layer_output_sha256s: Vec<String>,
    completed_layer_output_det_sha256s: Vec<Option<String>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeLogitsRasterState {
    normalized_final_position: RasterActivationRow,
    next_logit_idx: usize,
    logit_count: usize,
    logit_bits: Vec<i32>,
    softcap_bits: Option<i32>,
}

#[tile]
pub fn init_decode_transition_state(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<DecodeTransitionRasterState> {
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
    let original_layer_caches = transformer_decode_state
        .layer_caches
        .iter()
        .map(raster_cache_from_layer_cache)
        .collect::<Result<Vec<_>>>()?;

    Ok(DecodeTransitionRasterState {
        decode_input: embedded.clone(),
        current_activation: embedded,
        next_token,
        position: transformer_decode_state.position,
        token_count: transformer_decode_state.token_count,
        next_layer_idx: 0,
        layer_count: metadata.layer_count,
        original_layer_caches,
        updated_layer_caches: Vec::with_capacity(metadata.layer_count),
        completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
        completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
    })
}

#[tile(kind = recursive)]
pub fn compute_next_decode_layer(
    mut state: DecodeTransitionRasterState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<(bool, DecodeTransitionRasterState)> {
    if state.next_layer_idx >= state.layer_count {
        return Ok((true, state));
    }

    let layer_idx = state.next_layer_idx;
    let layer = auth_read!(source, GemmaDecodeLayerMetadataRequest { layer_idx })?;
    let _trace = crate::trace::trace_scope(format!(
        "decode.layer.det layer={layer_idx} token={} position={} attention={:?} ple={} donor={:?}",
        state.next_token,
        state.position,
        layer.attention_kind,
        layer.has_ple,
        layer.kv_shared_layer_index
    ));
    let cache = state
        .original_layer_caches
        .get(layer_idx)
        .cloned()
        .ok_or_else(|| anyhow!("transformer decode cache {layer_idx} missing"))?;
    let donor_cache = resolve_decode_donor_cache(&state.updated_layer_caches, layer_idx, &layer)?;
    let per_layer_input =
        compute_decode_ple_input(state.next_token, &state.decode_input, source, &layer)?;
    let (layer_output, updated_cache) = run_basic_decode_layer(
        &state.current_activation,
        source,
        &layer,
        cache,
        donor_cache,
        per_layer_input.as_ref(),
        state.position,
    )?;

    state.current_activation = layer_output;
    state.updated_layer_caches.push(updated_cache);
    let current_activation_values = state.current_activation.to_f32_values();
    let current_activation_acts = state.current_activation.acts();
    state.completed_layer_output_sha256s.push(
        crate::shared::transformer_kernels::build_vector_commitment(&current_activation_values),
    );
    state.completed_layer_output_det_sha256s.push(Some(
        crate::shared::transformer_kernels::build_det_vector_commitment(&current_activation_acts),
    ));
    let mut checkpoint_layer_caches = state
        .updated_layer_caches
        .iter()
        .cloned()
        .map(layer_cache_from_raster)
        .collect::<Vec<_>>();
    checkpoint_layer_caches.extend(
        state
            .original_layer_caches
            .iter()
            .skip(layer_idx + 1)
            .cloned()
            .map(layer_cache_from_raster),
    );
    let decode_input_values = state.decode_input.to_f32_values();
    let decode_input_acts = state.decode_input.acts();
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
    crate::trace::trace_checkpoint(
        &format!(
            "decode.layer_token.layer_{layer_idx}.position_{}",
            state.position
        ),
        &json!({
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
        }),
    );

    state.next_layer_idx += 1;
    Ok((false, state))
}

#[tile]
pub fn finalize_decode_layer_state(
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

    let det_row = state.current_activation.acts();
    let values = vec![state.current_activation.to_f32_values()];
    let internal = InternalActivationSequence::from_det_values(vec![det_row.clone()]);
    let mut activation_state = ActivationSequence::from_internal(
        internal,
        crate::shared::transformer_kernels::build_activation_commitment(&values),
    );
    activation_state.det_activations_sha256 =
        Some(crate::shared::transformer_kernels::build_det_activation_commitment(&[det_row]));

    Ok(ActivationSequenceWithCache {
        activation_state,
        layer_caches: state
            .updated_layer_caches
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
    normalized_final_position: RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<DecodeLogitsRasterState> {
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
    Ok(DecodeLogitsRasterState {
        normalized_final_position,
        next_logit_idx: 0,
        logit_count: metadata.projection_rows,
        logit_bits: Vec::with_capacity(metadata.projection_rows),
        softcap_bits: scalars.final_logit_softcapping.map(Act::to_bits),
    })
}

#[tile(kind = recursive)]
pub fn project_next_decode_logit(
    mut state: DecodeLogitsRasterState,
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> Result<(bool, DecodeLogitsRasterState)> {
    if state.next_logit_idx >= state.logit_count {
        return Ok((true, state));
    }
    if state.logit_bits.len() != state.next_logit_idx {
        bail!(
            "raster decode projection state has {} logits before row {}",
            state.logit_bits.len(),
            state.next_logit_idx
        );
    }

    let projection_row = auth_read!(
        source,
        GemmaDecodeProjectionRowRequest {
            row_idx: state.next_logit_idx,
        },
    )?;
    let mut logit = project_row_with_weights(&state.normalized_final_position, &projection_row)?;
    if let Some(softcap_bits) = state.softcap_bits {
        logit = softcap_act(logit, Act::from_bits(softcap_bits));
    }
    state.logit_bits.push(logit.to_bits());
    state.next_logit_idx += 1;
    Ok((false, state))
}

#[tile]
pub fn finalize_decode_transition_result(
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
    if state.logit_bits.len() != state.logit_count {
        bail!(
            "raster decode projection stored {} logits, expected {}",
            state.logit_bits.len(),
            state.logit_count
        );
    }

    let det_logits = state
        .logit_bits
        .into_iter()
        .map(Act::from_bits)
        .collect::<Vec<_>>();
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
) -> Result<TransformerDecodeStepResult> {
    let state = call_tile!(
        init_decode_transition_state,
        transformer_decode_state.clone(),
        next_token,
        source
    )?;
    let state = call_recur_tile_result!(compute_next_decode_layer, state, source)?;
    let layer_output = call_tile!(finalize_decode_layer_state, state)?;
    let normalized = call_tile!(
        normalize_decode_final_position,
        &layer_output.activation_state,
        source
    )?;
    let logits_state = call_tile!(init_decode_logits_projection, normalized, source)?;
    let logits_state = call_recur_tile_result!(project_next_decode_logit, logits_state, source)?;
    call_tile!(
        finalize_decode_transition_result,
        logits_state,
        TransformerDecodeState {
            layer_caches: layer_output.layer_caches,
            position: transformer_decode_state.position,
            token_count: transformer_decode_state.token_count,
        },
        layer_output.activation_state
    )
}

fn run_basic_decode_layer(
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache: RasterKvCache,
    donor_cache: Option<&RasterKvCache>,
    per_layer_input: Option<&RasterActivationRow>,
    position: usize,
) -> Result<(RasterActivationRow, RasterKvCache)> {
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: layer.layer_idx,
        },
    )?;

    let residual = input.clone();
    let normed = rms_norm_row(
        input,
        &auth_read!(
            source,
            GemmaDecodeLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaDecodeLayerNormKind::InputLayer,
            },
        )?,
        scalars.rms_norm_eps,
    )?;
    let (attention_output, updated_cache) =
        run_decode_attention(&normed, source, layer, cache, donor_cache, position)?;
    let attention_output = rms_norm_row(
        &attention_output,
        &auth_read!(
            source,
            GemmaDecodeLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaDecodeLayerNormKind::PostAttention,
            },
        )?,
        scalars.rms_norm_eps,
    )?;
    let mut xs = add_rows(&residual, &attention_output)?;

    let residual = xs.clone();
    let normed = rms_norm_row(
        &xs,
        &auth_read!(
            source,
            GemmaDecodeLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaDecodeLayerNormKind::PreFeedForward,
            },
        )?,
        scalars.rms_norm_eps,
    )?;
    let gate = project_row_with_decode_source(
        &normed,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Gate,
        layer.gate_proj_shape.rows,
    )?;
    let gate = gelu_row(&gate)?;
    let up = project_row_with_decode_source(
        &normed,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Up,
        layer.up_proj_shape.rows,
    )?;
    let ff_hidden = mul_rows(&gate, &up)?;
    let ff_out = project_row_with_decode_source(
        &ff_hidden,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Down,
        layer.down_proj_shape.rows,
    )?;
    let ff_out = rms_norm_row(
        &ff_out,
        &auth_read!(
            source,
            GemmaDecodeLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaDecodeLayerNormKind::PostFeedForward,
            },
        )?,
        scalars.rms_norm_eps,
    )?;
    xs = add_rows(&residual, &ff_out)?;

    if let Some(per_layer_input) = per_layer_input {
        let residual = xs.clone();
        let gated = project_row_with_decode_source(
            &xs,
            source,
            layer.layer_idx,
            GemmaDecodeLayerMatrixKind::PleInputGate,
            layer
                .ple_input_gate_shape
                .ok_or_else(|| {
                    anyhow!("Gemma decode layer metadata is missing PLE input gate shape")
                })?
                .rows,
        )?;
        let gated = gelu_row(&gated)?;
        let gated = mul_rows(&gated, per_layer_input)?;
        let projected = project_row_with_decode_source(
            &gated,
            source,
            layer.layer_idx,
            GemmaDecodeLayerMatrixKind::PleLayerProjection,
            layer
                .ple_layer_projection_shape
                .ok_or_else(|| {
                    anyhow!("Gemma decode layer metadata is missing PLE layer projection shape")
                })?
                .rows,
        )?;
        let projected = rms_norm_row(
            &projected,
            &auth_read!(
                source,
                GemmaDecodeLayerNormWeightsRequest {
                    layer_idx: layer.layer_idx,
                    norm: GemmaDecodeLayerNormKind::PlePostInput,
                },
            )?,
            scalars.rms_norm_eps,
        )?;
        xs = add_rows(&residual, &projected)?;
    }

    if scalars.layer_scalar.is_some() {
        xs = scale_row(&xs, scalars.layer_scalar)?;
    }

    Ok((xs, updated_cache))
}

fn run_decode_attention(
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
    cache: RasterKvCache,
    donor_cache: Option<&RasterKvCache>,
    position: usize,
) -> Result<(RasterActivationRow, RasterKvCache)> {
    validate_row_width(input, layer.hidden_size, "decode attention input")?;
    let kv_groups = layer
        .num_heads
        .checked_div(layer.num_kv_heads)
        .ok_or_else(|| anyhow!("invalid Gemma head configuration"))?;
    if kv_groups == 0 {
        bail!("Gemma layer must have at least one KV group");
    }

    let q_projected = project_row_with_decode_source(
        input,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Query,
        layer.q_proj_shape.rows,
    )?;
    let raw_k = project_row_with_decode_source(
        input,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Key,
        layer.k_proj_shape.rows,
    )?;
    let raw_v = if layer.has_v_proj {
        project_row_with_decode_source(
            input,
            source,
            layer.layer_idx,
            GemmaDecodeLayerMatrixKind::Value,
            layer
                .v_proj_shape
                .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing v_proj shape"))?
                .rows,
        )?
    } else if layer.attention_k_eq_v {
        raw_k.clone()
    } else {
        bail!("Gemma layer is missing v_proj without attention_k_eq_v enabled");
    };

    let q_heads = reshape_row_heads(q_projected, layer.num_heads, layer.head_dim)?;
    let k_heads = reshape_row_heads(raw_k, layer.num_kv_heads, layer.head_dim)?;
    let v_heads = reshape_row_heads(raw_v, layer.num_kv_heads, layer.head_dim)?;
    let q_heads = rms_norm_heads(
        &q_heads,
        Some(&auth_read!(
            source,
            GemmaDecodeLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaDecodeLayerNormKind::Query,
            },
        )?),
        Some(
            auth_read!(
                source,
                GemmaDecodeLayerScalarsRequest {
                    layer_idx: layer.layer_idx,
                },
            )?
            .rms_norm_eps,
        ),
    )?;
    let k_heads = rms_norm_heads(
        &k_heads,
        Some(&auth_read!(
            source,
            GemmaDecodeLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaDecodeLayerNormKind::Key,
            },
        )?),
        Some(
            auth_read!(
                source,
                GemmaDecodeLayerScalarsRequest {
                    layer_idx: layer.layer_idx,
                },
            )?
            .rms_norm_eps,
        ),
    )?;
    let v_heads = value_rms_norm_heads(
        &v_heads,
        Some(
            auth_read!(
                source,
                GemmaDecodeLayerScalarsRequest {
                    layer_idx: layer.layer_idx,
                },
            )?
            .rms_norm_eps,
        ),
    )?;
    let scalars = auth_read!(
        source,
        GemmaDecodeLayerScalarsRequest {
            layer_idx: layer.layer_idx,
        },
    )?;
    let q_heads = apply_rope_to_heads(
        &q_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        position,
    )?;
    let k_heads = apply_rope_to_heads(
        &k_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        position,
    )?;

    let updated_cache = if donor_cache.is_some() {
        cache
    } else {
        append_decode_kv_cache(cache, &k_heads, &v_heads, layer.cache_sliding_window)?
    };
    let attention_cache = donor_cache.unwrap_or(&updated_cache);
    let attention_window = match layer.attention_kind {
        GemmaDecodeAttentionKind::Full => None,
        GemmaDecodeAttentionKind::Sliding => Some(
            layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?,
        ),
    };
    let mut output_heads = Vec::with_capacity(layer.num_heads);
    for head_idx in 0..layer.num_heads {
        let kv_head_idx = head_idx / kv_groups;
        let key_start = attention_window
            .map(|window| attention_cache.current_len().saturating_sub(window))
            .unwrap_or(0);
        let row_count = attention_cache.current_len().saturating_sub(key_start);
        let key_rows = attention_cache.key_rows_window(kv_head_idx, key_start, row_count)?;
        let value_rows = attention_cache.value_rows_window(kv_head_idx, key_start, row_count)?;
        let query = q_heads
            .heads()
            .get(head_idx)
            .and_then(|head| head.first())
            .ok_or_else(|| anyhow!("decode query head {head_idx} is missing"))?;
        output_heads.push(vec![attention_output_row(query, &key_rows, &value_rows)?]);
    }
    let attention_sequence =
        combine_attention_heads(&RasterAttentionHeadSequence::from_heads(output_heads))?;
    let attention_row = first_row(attention_sequence, "decode attention combine")?;
    let projected = project_row_with_decode_source(
        &attention_row,
        source,
        layer.layer_idx,
        GemmaDecodeLayerMatrixKind::Output,
        layer.o_proj_shape.rows,
    )?;

    Ok((projected, updated_cache))
}

fn compute_decode_ple_input(
    token_id: u32,
    decode_input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer: &GemmaDecodeLayerMetadata,
) -> Result<Option<RasterActivationRow>> {
    if !layer.has_ple {
        return Ok(None);
    }

    let ple_width = layer
        .ple_input_gate_shape
        .ok_or_else(|| anyhow!("Gemma decode layer metadata is missing PLE input gate shape"))?
        .rows;
    let scalars = auth_read!(source, GemmaDecodePleScalarsRequest)?;
    let norm_weights = auth_read!(source, GemmaDecodePleProjectionNormWeightsRequest)?;
    let embedded = RasterActivationRow::from_acts(auth_read!(
        source,
        GemmaDecodePleTokenEmbeddingRowRequest {
            layer_idx: layer.layer_idx,
            token_id,
        },
    )?);
    let embedded = scale_row(&embedded, Some(scalars.embedding_scale))?;

    let projected = project_row_with_ple_source(decode_input, source, layer.layer_idx, ple_width)?;
    let projected = scale_row(&projected, Some(scalars.projection_scalar))?;
    let projected = rms_norm_row(&projected, &norm_weights, scalars.rms_norm_eps)?;
    let combined = add_rows(&embedded, &projected)?;
    scale_row(&combined, Some(scalars.input_scale)).map(Some)
}

fn project_row_with_decode_source(
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    matrix: GemmaDecodeLayerMatrixKind,
    projection_rows: usize,
) -> Result<RasterActivationRow> {
    if projection_rows == 0 {
        bail!("deterministic decode linear projection requires at least one projection row");
    }
    let mut output = Vec::with_capacity(projection_rows);
    for row_idx in 0..projection_rows {
        let row = auth_read!(
            source,
            GemmaDecodeLayerMatrixRowRequest {
                layer_idx,
                matrix,
                row_idx,
            },
        )?;
        output.push(project_row_with_weights(input, &row)?);
    }
    Ok(RasterActivationRow::from_acts(output))
}

fn project_row_with_ple_source(
    input: &RasterActivationRow,
    source: &AuthenticatedGemmaDecodeTransitionSource,
    layer_idx: usize,
    projection_rows: usize,
) -> Result<RasterActivationRow> {
    if projection_rows == 0 {
        bail!("deterministic decode PLE projection requires at least one projection row");
    }
    let mut output = Vec::with_capacity(projection_rows);
    for row_idx in 0..projection_rows {
        let row = auth_read!(
            source,
            GemmaDecodePleModelProjectionRowRequest { layer_idx, row_idx },
        )?;
        output.push(project_row_with_weights(input, &row)?);
    }
    Ok(RasterActivationRow::from_acts(output))
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

fn append_decode_kv_cache(
    cache: RasterKvCache,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    cache_window: Option<usize>,
) -> Result<RasterKvCache> {
    if keys.head_count() != values.head_count() || keys.head_count() != cache.head_count() {
        bail!(
            "decode cache head count mismatch: cache {} keys {} values {}",
            cache.head_count(),
            keys.head_count(),
            values.head_count()
        );
    }
    let mut updated_keys = cache.keys().to_vec();
    let mut updated_values = cache.values().to_vec();
    for head_idx in 0..keys.head_count() {
        let key_row = keys
            .heads()
            .get(head_idx)
            .and_then(|head| head.first())
            .ok_or_else(|| anyhow!("decode key head {head_idx} is missing"))?
            .clone();
        let value_row = values
            .heads()
            .get(head_idx)
            .and_then(|head| head.first())
            .ok_or_else(|| anyhow!("decode value head {head_idx} is missing"))?
            .clone();
        updated_keys[head_idx].push(key_row);
        updated_values[head_idx].push(value_row);
        if let Some(window) = cache_window {
            let retained = updated_keys[head_idx].len().saturating_sub(window);
            updated_keys[head_idx] = updated_keys[head_idx][retained..].to_vec();
            updated_values[head_idx] = updated_values[head_idx][retained..].to_vec();
        }
    }
    RasterKvCache::from_heads(updated_keys, updated_values)
}

fn resolve_decode_donor_cache<'a>(
    layer_caches: &'a [RasterKvCache],
    layer_idx: usize,
    layer: &GemmaDecodeLayerMetadata,
) -> Result<Option<&'a RasterKvCache>> {
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

fn rms_norm_row(
    row: &RasterActivationRow,
    norm_weights: &[crate::shared::det_num::Wgt],
    eps: crate::shared::det_num::Acc,
) -> Result<RasterActivationRow> {
    first_row(
        rms_norm_sequence(
            &RasterActivationSequence::from_rows(vec![row.clone()]),
            Some(norm_weights),
            Some(eps),
        )?,
        "deterministic decode RMSNorm",
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
    use super::run;
    use crate::shared::det_num::{Acc, Act, Wgt};
    use crate::shared::input::InferenceExecutionMode;
    use crate::shared::raster_decode_transition::AuthenticatedGemmaDecodeTransitionSource;
    use crate::shared::transformer::{
        DetNumMatrix, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4TransformerModel,
        GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, TransformerDecodeState,
    };
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

        let raster = run(decode_state.clone(), 1, &source).expect("raster decode should run");
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
    fn raster_decode_transition_matches_sliding_cache_window() {
        let (_path, model) = no_ple_model(true);
        let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
            .expect("source should build");
        let decode_state = decode_state_with_cache(2);

        let raster = run(decode_state.clone(), 1, &source).expect("raster decode should run");
        let deterministic = crate::decode_transition::run_with_mode(
            decode_state,
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("deterministic decode should run");

        assert_eq!(
            raster.transformer_decode_state.layer_caches[0].current_len(),
            1
        );
        assert_eq!(
            raster.transformer_decode_state,
            deterministic.transformer_decode_state
        );
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

        let error = run(decode_state, 1, &source).expect_err("f32 cache should fail");

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

    fn decode_state_with_cache(cache_len: usize) -> TransformerDecodeState {
        let key_rows = (0..cache_len)
            .map(|_| vec![Act::from_num(0.0), Act::from_num(0.0)])
            .collect::<VecDeque<_>>();
        let value_rows = key_rows.clone();
        TransformerDecodeState {
            layer_caches: vec![LayerKvCache::from_det_heads(
                vec![key_rows],
                vec![value_rows],
            )],
            position: cache_len,
            token_count: cache_len,
        }
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
