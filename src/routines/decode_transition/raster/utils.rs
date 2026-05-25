use super::types::*;

use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::decode_transition::raster::auth_source::{
    GemmaDecodeLayerMatrixRowRequest, GemmaDecodeLayerMetadata,
    GemmaDecodePleModelProjectionRowRequest, GemmaDecodeProjectionRowRequest,
};
use crate::dsl::prelude::auth_read;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::external_artifacts::CommittedExternalSource;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, token_id_leaf, RasterActivationSequenceArtifactRef, RasterArtifactId,
    RasterArtifactMetadata, RasterArtifactStoreRoots, RasterSelectedTokenRef,
    RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{
    ActivationSequence, InternalLogits, LayerKvCache, PrefillLogits, TransformerDecodeState,
    TransformerDecodeStepResult,
};
use crate::shared::numerics::det_num::{
    acc_add_sat, add_sat, attention_score as det_attention_score, attention_softmax_exp_term,
    attention_softmax_raw_weight, attention_softmax_residual, mac_bits, requantize, Acc, Act,
};
use crate::shared::raster_kernels::transformer::{
    add_sequences, gelu_sequence, mul_sequences, project_row_with_weights, scale_sequence,
    validate_attention_kv_rows_per_tile, validate_projection_rows_per_tile, RasterActivationRow,
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
    RasterHeadRowRequest, RasterKvCacheRef, RasterKvRowKind, RasterKvRowRequest,
    RasterSequenceRowRequest, RasterTensorId,
};

pub(in super::super) fn init_decode_attention_score_phase_with_roots(
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

pub(in super::super) fn trace_decode_layer_checkpoint_with_roots(
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

pub(in super::super) fn read_decode_projection_row(
    source: &CommittedExternalSource,
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

pub(in super::super) fn init_decode_row_projection_artifact_state(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    projection_kind: DecodeProjectionKind,
    projection_rows: usize,
    rows_per_tile: usize,
    output_id: String,
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
    let output_id = RasterTensorId::new(output_id)?;
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

pub(in super::super) fn project_next_decode_projection_artifact_chunk(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut projection_state: DecodeRowProjectionArtifactState,
    source: &CommittedExternalSource,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    DecodeRowProjectionArtifactState,
)> {
    if projection_state.next_projection_row_idx >= projection_state.projection_rows {
        return Ok((true, artifact_store_roots, projection_state));
    }
    let input =
        read_activation_row_from_ref_roots(&artifact_store_roots, &projection_state.input_ref)?;
    if input.width() != projection_state.input_width {
        bail!(
            "decode projection input row has width {}, expected {}",
            input.width(),
            projection_state.input_width
        );
    }
    let end = projection_state
        .next_projection_row_idx
        .saturating_add(projection_state.rows_per_tile)
        .min(projection_state.projection_rows);
    while projection_state.next_projection_row_idx < end {
        let projection_row = read_decode_projection_row(
            source,
            &projection_state.projection_kind,
            projection_state.next_projection_row_idx,
        )?;
        if projection_row.len() != projection_state.input_width {
            bail!(
                "decode projection row {} has width {}, expected {}",
                projection_state.next_projection_row_idx,
                projection_row.len(),
                projection_state.input_width
            );
        }
        let mut projected = project_row_with_weights(&input, &projection_row)?;
        if let Some(softcap_bits) = projection_state.softcap_bits {
            projected = crate::shared::numerics::det_num::softcap_act(
                projected,
                Act::from_bits(softcap_bits),
            );
        }
        projection_state.current_row_bits.push(projected.to_bits());
        projection_state.next_projection_row_idx += 1;
    }
    if projection_state.next_projection_row_idx == projection_state.projection_rows {
        let row_bits = std::mem::take(&mut projection_state.current_row_bits);
        let artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &artifact_store_roots,
            &projection_state.output_source_name,
            0,
            RasterActivationRow::from_act_bits(row_bits),
        )?;
        return Ok((true, artifact_store_roots, projection_state));
    }
    Ok((false, artifact_store_roots, projection_state))
}

pub(in super::super) fn finalize_decode_row_projection_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    projection_state: DecodeRowProjectionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if projection_state.next_projection_row_idx != projection_state.projection_rows {
        bail!(
            "raster decode projection completed {} rows, expected {}",
            projection_state.next_projection_row_idx,
            projection_state.projection_rows
        );
    }
    if !projection_state.current_row_bits.is_empty() {
        bail!(
            "raster decode projection finalized with partial row width {}",
            projection_state.current_row_bits.len()
        );
    }
    finalize_sequence_builder_by_source_name_with_roots(
        &artifact_store_roots,
        &projection_state.output_source_name,
        RasterTensorId::new(projection_state.output_source_name.clone())?,
    )
}

pub(in super::super) fn attach_decode_projection_output(
    continuation: DecodeProjectionContinuation,
    output_ref: RasterActivationSequenceRef,
) -> DecodeProjectionContinuation {
    match continuation {
        DecodeProjectionContinuation::PleInput(work) => match work {
            DecodePleInputWork::Skip(layer_work) => {
                DecodeProjectionContinuation::PleInput(DecodePleInputWork::Skip(layer_work))
            }
            DecodePleInputWork::Active {
                work,
                scalars,
                norm_weights,
                embedded_ref,
                output_prefix,
                ..
            } => DecodeProjectionContinuation::PleInput(DecodePleInputWork::Active {
                work,
                scalars,
                norm_weights,
                embedded_ref,
                projected_ref: Some(output_ref),
                output_prefix,
            }),
        },
        DecodeProjectionContinuation::FinalLogits(mut work) => {
            work.logits_ref = Some(output_ref);
            DecodeProjectionContinuation::FinalLogits(work)
        }
        DecodeProjectionContinuation::LayerAttentionQuery(layer_work) => {
            DecodeProjectionContinuation::LayerAttentionQuery(
                attach_decode_layer_projection_output(
                    layer_work,
                    output_ref,
                    |work, output_ref| work.q_projected_ref = Some(output_ref),
                ),
            )
        }
        DecodeProjectionContinuation::LayerAttentionKey(layer_work) => {
            DecodeProjectionContinuation::LayerAttentionKey(attach_decode_layer_projection_output(
                layer_work,
                output_ref,
                |work, output_ref| work.k_projected_ref = Some(output_ref),
            ))
        }
        DecodeProjectionContinuation::LayerAttentionValue(layer_work) => {
            DecodeProjectionContinuation::LayerAttentionValue(
                attach_decode_layer_projection_output(
                    layer_work,
                    output_ref,
                    |work, output_ref| work.v_projected_ref = Some(output_ref),
                ),
            )
        }
        DecodeProjectionContinuation::LayerAttentionOutput(layer_work) => {
            DecodeProjectionContinuation::LayerAttentionOutput(
                attach_decode_layer_projection_output(
                    layer_work,
                    output_ref,
                    |work, output_ref| work.attention_output_ref = Some(output_ref),
                ),
            )
        }
        DecodeProjectionContinuation::LayerMlpGate(layer_work) => {
            DecodeProjectionContinuation::LayerMlpGate(attach_decode_layer_projection_output(
                layer_work,
                output_ref,
                |work, output_ref| work.mlp_gate_ref = Some(output_ref),
            ))
        }
        DecodeProjectionContinuation::LayerMlpUp(layer_work) => {
            DecodeProjectionContinuation::LayerMlpUp(attach_decode_layer_projection_output(
                layer_work,
                output_ref,
                |work, output_ref| work.mlp_up_ref = Some(output_ref),
            ))
        }
        DecodeProjectionContinuation::LayerMlpDown(layer_work) => {
            DecodeProjectionContinuation::LayerMlpDown(attach_decode_layer_projection_output(
                layer_work,
                output_ref,
                |work, output_ref| work.mlp_out_ref = Some(output_ref),
            ))
        }
        DecodeProjectionContinuation::LayerPleGate(layer_work) => {
            DecodeProjectionContinuation::LayerPleGate(attach_decode_layer_projection_output(
                layer_work,
                output_ref,
                |work, output_ref| work.ple_gate_ref = Some(output_ref),
            ))
        }
        DecodeProjectionContinuation::LayerPleOutput(layer_work) => {
            DecodeProjectionContinuation::LayerPleOutput(attach_decode_layer_projection_output(
                layer_work,
                output_ref,
                |work, output_ref| work.ple_projected_ref = Some(output_ref),
            ))
        }
    }
}

pub(in super::super) fn attach_decode_layer_projection_output(
    layer_work: DecodeLayerWork,
    output_ref: RasterActivationSequenceRef,
    attach: impl FnOnce(&mut DecodeActiveLayerWork, RasterActivationSequenceRef),
) -> DecodeLayerWork {
    match layer_work {
        DecodeLayerWork::Complete(decode_state) => DecodeLayerWork::Complete(decode_state),
        DecodeLayerWork::Active(mut work) => {
            attach(&mut work, output_ref);
            DecodeLayerWork::Active(work)
        }
    }
}

pub(in super::super) fn active_decode_layer_work(
    work: DecodeLayerWork,
) -> Result<DecodeActiveLayerWork> {
    match work {
        DecodeLayerWork::Active(work) => Ok(work),
        DecodeLayerWork::Complete(_) => {
            bail!("decode layer work should be active at this stage")
        }
    }
}

pub(in super::super) fn require_decode_activation_ref(
    value: Option<RasterActivationSequenceRef>,
    description: &str,
) -> Result<RasterActivationSequenceRef> {
    value.ok_or_else(|| anyhow!("decode layer work is missing {description}"))
}

pub(in super::super) fn require_decode_heads_ref(
    value: Option<RasterAttentionHeadsRef>,
    description: &str,
) -> Result<RasterAttentionHeadsRef> {
    value.ok_or_else(|| anyhow!("decode layer work is missing {description}"))
}

pub(in super::super) fn init_decode_kv_cache_append_artifact_work_state(
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

pub(in super::super) fn compute_next_decode_kv_cache_append_artifact_work_row(
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

pub(in super::super) fn finalize_decode_kv_cache_append_artifact_work_state(
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

pub(in super::super) fn init_decode_attention_artifact_work_state(
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

pub(in super::super) fn compute_next_decode_attention_artifact_work_head(
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

pub(in super::super) fn finalize_decode_attention_artifact_work_state(
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

pub(in super::super) fn reshape_row_heads(
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

pub(in super::super) fn register_decode_layer_cache_with_roots(
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

pub(in super::super) fn materialize_decode_layer_cache_from_roots(
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

pub(in super::super) fn materialize_decode_checkpoint_caches_from_roots(
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

pub(in super::super) fn resolve_decode_donor_cache_slot<'a>(
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

pub(in super::super) fn raster_cache_from_layer_cache(
    cache: &LayerKvCache,
) -> Result<RasterKvCache> {
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

pub(in super::super) fn layer_cache_from_raster(cache: RasterKvCache) -> LayerKvCache {
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

pub(in super::super) fn insert_decode_activation_row_with_roots(
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

pub(in super::super) fn insert_decode_selected_token_with_roots(
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

pub(in super::super) fn insert_attention_heads_with_roots(
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

pub(in super::super) fn materialize_attention_heads_from_roots(
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

pub(in super::super) fn read_activation_row_from_ref_roots(
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

pub(in super::super) fn read_decode_attention_kv_row_from_roots(
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

pub(in super::super) fn read_decode_attention_scalar_row_from_roots(
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

pub(in super::super) fn scale_decode_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    scalar: Option<Act>,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    if scalar.is_none() {
        return Ok((artifact_store_roots, input_ref));
    }
    let row = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    let row = scale_row(&row, scalar)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

pub(in super::super) fn rms_norm_decode_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    norm_weights: &[crate::shared::numerics::det_num::Wgt],
    eps: crate::shared::numerics::det_num::Acc,
    label: &str,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let row = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    let row = first_row(
        crate::shared::raster_kernels::transformer::rms_norm_sequence(
            &RasterActivationSequence::from_rows(vec![row]),
            Some(norm_weights),
            Some(eps),
        )?,
        label,
    )?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

pub(in super::super) fn add_decode_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let lhs = read_activation_row_from_ref_roots(&artifact_store_roots, &lhs_ref)?;
    let rhs = read_activation_row_from_ref_roots(&artifact_store_roots, &rhs_ref)?;
    let row = add_rows(&lhs, &rhs)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

pub(in super::super) fn mul_decode_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let lhs = read_activation_row_from_ref_roots(&artifact_store_roots, &lhs_ref)?;
    let rhs = read_activation_row_from_ref_roots(&artifact_store_roots, &rhs_ref)?;
    let row = mul_rows(&lhs, &rhs)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

pub(in super::super) fn gelu_decode_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    output_source_name: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let row = read_activation_row_from_ref_roots(&artifact_store_roots, &input_ref)?;
    let row = gelu_row(&row)?;
    insert_decode_activation_row_with_roots(&artifact_store_roots, output_source_name, &row)
}

pub(in super::super) fn scale_row(
    row: &RasterActivationRow,
    scalar: Option<Act>,
) -> Result<RasterActivationRow> {
    first_row(
        scale_sequence(
            &RasterActivationSequence::from_rows(vec![row.clone()]),
            scalar,
        )?,
        "deterministic decode scaling",
    )
}

pub(in super::super) fn add_rows(
    lhs: &RasterActivationRow,
    rhs: &RasterActivationRow,
) -> Result<RasterActivationRow> {
    first_row(
        add_sequences(
            &RasterActivationSequence::from_rows(vec![lhs.clone()]),
            &RasterActivationSequence::from_rows(vec![rhs.clone()]),
        )?,
        "deterministic decode row add",
    )
}

pub(in super::super) fn mul_rows(
    lhs: &RasterActivationRow,
    rhs: &RasterActivationRow,
) -> Result<RasterActivationRow> {
    first_row(
        mul_sequences(
            &RasterActivationSequence::from_rows(vec![lhs.clone()]),
            &RasterActivationSequence::from_rows(vec![rhs.clone()]),
        )?,
        "deterministic decode row multiply",
    )
}

pub(in super::super) fn gelu_row(row: &RasterActivationRow) -> Result<RasterActivationRow> {
    first_row(
        gelu_sequence(&RasterActivationSequence::from_rows(vec![row.clone()]))?,
        "deterministic decode GELU",
    )
}

pub(in super::super) fn first_row(
    sequence: RasterActivationSequence,
    label: &str,
) -> Result<RasterActivationRow> {
    sequence
        .into_rows()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("{label} returned no rows"))
}

pub(in super::super) fn validate_row_width(
    row: &RasterActivationRow,
    expected_width: usize,
    label: &str,
) -> Result<()> {
    if row.width() != expected_width {
        bail!(
            "{label} width mismatch: {} vs {}",
            row.width(),
            expected_width
        );
    }
    Ok(())
}

pub(in super::super) fn finalize_decode_transition_result_values_from_roots(
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

pub(in super::super) fn update_decode_layer_state_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut decode_state: DecodeTransitionRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    updated_cache: DecodeLayerCacheSlot,
) -> Result<(bool, RasterArtifactStoreRoots, DecodeTransitionRasterState)> {
    if layer_idx != decode_state.next_layer_idx {
        bail!(
            "cannot update decode layer {layer_idx} while next layer is {}",
            decode_state.next_layer_idx
        );
    }
    let layer_output =
        read_activation_row_from_ref_roots(&artifact_store_roots, &layer_output_ref)?;
    decode_state.artifact_store_roots = artifact_store_roots.clone();
    decode_state.current_activation_ref = layer_output_ref;
    decode_state.updated_layer_caches.push(updated_cache);
    let current_activation_values = layer_output.to_f32_values();
    let current_activation_acts = layer_output.acts();
    decode_state.completed_layer_output_sha256s.push(
        crate::shared::numerics::transformer_kernels::build_vector_commitment(
            &current_activation_values,
        ),
    );
    decode_state.completed_layer_output_det_sha256s.push(Some(
        crate::shared::numerics::transformer_kernels::build_det_vector_commitment(
            &current_activation_acts,
        ),
    ));
    trace_decode_layer_checkpoint_with_roots(&artifact_store_roots, &decode_state, layer_idx)?;
    decode_state.next_layer_idx += 1;
    Ok((false, artifact_store_roots, decode_state))
}
