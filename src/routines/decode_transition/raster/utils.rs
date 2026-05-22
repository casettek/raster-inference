use super::types::*;

use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::decode_transition::raster::auth_source::{
    AuthenticatedGemmaDecodeTransitionSource, GemmaDecodeLayerMatrixRowRequest,
    GemmaDecodeLayerMetadata, GemmaDecodePleModelProjectionRowRequest,
    GemmaDecodeProjectionRowRequest,
};
use crate::dsl::prelude::auth_read;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, token_id_leaf, RasterActivationSequenceArtifactRef, RasterArtifactId,
    RasterArtifactMetadata, RasterArtifactStoreRoots, RasterSelectedTokenRef,
    RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::LayerKvCache;
use crate::shared::numerics::det_num::Act;
use crate::shared::raster_kernels::transformer::{
    add_sequences, gelu_sequence, mul_sequences, scale_sequence, RasterActivationRow,
    RasterActivationSequence, RasterAttentionHeadSequence, RasterKvCache,
};
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, attention_heads_ref_from_artifact,
    kv_cache_ref_from_artifacts, read_head_row_from_roots, read_kv_row_from_roots,
    read_sequence_row_from_roots, start_sequence_builder_with_roots, RasterActivationSequenceRef,
    RasterAttentionHeadsRef, RasterHeadRowRequest, RasterKvCacheRef, RasterKvRowKind,
    RasterKvRowRequest, RasterSequenceRowRequest, RasterTensorId,
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
