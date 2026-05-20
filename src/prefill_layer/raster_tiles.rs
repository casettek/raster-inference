use anyhow::{anyhow, bail, Result};
use serde_json::json;

use super::raster_utils::{
    layer_caches_from_raster, materialize_prefill_activation_sequence_from_store,
    materialize_prefill_layer_caches, raster_sequence_acts, resolve_prefill_donor_cache_index,
    retained_prefill_kv_cache_len, validate_prefill_layer_ple_input_ref,
};
use crate::input_embedding::raster_tiles::RasterInputEmbeddingRefs;
use crate::raster_authoring::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
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
use crate::shared::tensors::raster_row_store::{
    activation_sequence_ref_from_artifact, read_sequence_row_from_roots,
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterAttentionHeadsRef,
    RasterKvCacheRef, RasterSequenceRowRequest, RasterTensorId,
};
use crate::trace::{trace_event, trace_scope};
use crate::RasterSizingControls;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerRasterState {
    current_activations_ref: RasterActivationSequenceRef,
    next_layer_idx: usize,
    layer_count: usize,
    layer_caches: Vec<PrefillLayerCacheSlot>,
    per_layer_inputs: Vec<Option<RasterActivationSequenceRef>>,
    completed_layer_output_sha256s: Vec<String>,
    completed_layer_output_det_sha256s: Vec<Option<String>>,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
    head_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum PrefillLayerCacheSlot {
    Empty { num_kv_heads: usize },
    Ref(RasterKvCacheRef),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerOutputRefs {
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<PrefillLayerCacheSlot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerContext {
    layer_idx: usize,
    layer: GemmaPrefillLayerMetadata,
    donor_cache: Option<PrefillLayerCacheSlot>,
    per_layer_input: Option<RasterActivationSequenceRef>,
}

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

#[tile]
pub fn prepare_next_prefill_layer_context(
    state: &PrefillLayerRasterState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
) -> Result<PrefillLayerContext> {
    if state.next_layer_idx >= state.layer_count {
        bail!(
            "cannot prepare prefill layer {} after completing {} layers",
            state.next_layer_idx,
            state.layer_count
        );
    }

    let layer_idx = state.next_layer_idx;
    let layer = auth_read!(layer_source, GemmaPrefillLayerMetadataRequest { layer_idx })?;
    let donor_cache = resolve_prefill_donor_cache_index(&state.layer_caches, layer_idx, &layer)?
        .map(|donor_idx| {
            state.layer_caches.get(donor_idx).cloned().ok_or_else(|| {
                anyhow!("transformer prefill donor cache {donor_idx} missing for layer {layer_idx}")
            })
        })
        .transpose()?;
    let per_layer_input = state
        .per_layer_inputs
        .get(layer_idx)
        .and_then(Option::as_ref)
        .cloned();
    validate_prefill_layer_ple_input_ref(
        &state.current_activations_ref,
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

#[sequence(kind = recursive)]
pub fn compute_next_prefill_layer_sequence_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: PrefillLayerRasterState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    if state.next_layer_idx >= state.layer_count {
        return Ok((true, artifact_store_roots, state));
    }

    let context = call_tile!(prepare_next_prefill_layer_context, &state, layer_source)?;
    if let Some(per_layer_input) = context.per_layer_input.as_ref() {
        read_sequence_row_from_roots(
            &artifact_store_roots,
            RasterSequenceRowRequest {
                tensor_ref: per_layer_input.clone(),
                row_idx: 0,
            },
        )?;
    }
    let (token_count, _) = state
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
        state.layer_count,
        token_count,
        context.layer.attention_kind,
        context.layer.has_ple,
        context.layer.kv_shared_layer_index
    ));
    let (artifact_store_roots, layer_output_ref, layer_cache) = call_seq!(
        run_prefill_layer_sequence_artifact_ref,
        artifact_store_roots,
        state.current_activations_ref.clone(),
        layer_source,
        &context.layer,
        context.donor_cache.as_ref(),
        context.per_layer_input.clone(),
        state.projection_rows_per_tile,
        state.attention_kv_rows_per_tile,
        state.sequence_rows_per_tile,
        state.head_rows_per_tile,
    )?;

    call_tile!(
        update_prefill_layer_state_refs_with_roots,
        artifact_store_roots,
        state,
        context.layer_idx,
        layer_output_ref,
        layer_cache
    )
}

#[tile]
pub fn update_prefill_layer_state_refs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    mut state: PrefillLayerRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    layer_cache: PrefillLayerCacheSlot,
) -> Result<(bool, RasterArtifactStoreRoots, PrefillLayerRasterState)> {
    if layer_idx != state.next_layer_idx {
        bail!(
            "cannot update prefill layer {layer_idx} while next layer is {}",
            state.next_layer_idx
        );
    }

    state.current_activations_ref = layer_output_ref;
    state.layer_caches.push(layer_cache);

    let completed_layer_output = trace_prefill_layer_checkpoint_with_roots(&state, layer_idx)?;
    if let Some((sha256, det_sha256)) = completed_layer_output {
        state.completed_layer_output_sha256s.push(sha256);
        state.completed_layer_output_det_sha256s.push(det_sha256);
    }

    if trace_prefill_layer_token_checkpoints_with_roots(&artifact_store_roots, &state, layer_idx)? {
        state.next_layer_idx += 1;
        state.layer_count = state.next_layer_idx;
        return Ok((true, artifact_store_roots, state));
    }
    state.next_layer_idx += 1;
    Ok((false, artifact_store_roots, state))
}

fn ensure_artifact_root_present(roots: &RasterArtifactStoreRoots, root: &str) -> Result<()> {
    if roots.artifacts.iter().any(|entry| entry.root() == root) {
        return Ok(());
    }
    bail!("raster artifact root {root} is not present in the store roots snapshot")
}

fn trace_prefill_layer_checkpoint(
    store: &AuthenticatedRasterTensorStore,
    state: &PrefillLayerRasterState,
    layer_idx: usize,
) -> Result<Option<(String, Option<String>)>> {
    let mut completed_layer_output = None;
    let reached = crate::trace::trace_checkpoint_lazy_result("prefill.layer", || {
        let current_activations = materialize_prefill_activation_sequence_from_store(
            store,
            &state.current_activations_ref,
        )?;
        let current_values = current_activations.to_f32_values();
        let current_det_activations = raster_sequence_acts(&current_activations);
        let current_sha256 =
            crate::shared::numerics::transformer_kernels::build_activation_commitment(
                &current_values,
            );
        let current_det_sha256 = Some(
            crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
                &current_det_activations,
            ),
        );
        let raster_layer_caches = materialize_prefill_layer_caches(store, &state.layer_caches)?;
        let layer_caches = layer_caches_from_raster(&raster_layer_caches);
        let mut completed_layer_output_sha256s = state.completed_layer_output_sha256s.clone();
        completed_layer_output_sha256s.push(current_sha256.clone());
        let mut completed_layer_output_det_sha256s =
            state.completed_layer_output_det_sha256s.clone();
        completed_layer_output_det_sha256s.push(current_det_sha256.clone());
        completed_layer_output = Some((current_sha256.clone(), current_det_sha256.clone()));
        Ok(json!({
            "execution_mode": "deterministic",
            "next_layer_idx": layer_idx + 1,
            "current_activations": current_values,
            "current_activations_sha256": current_sha256,
            "det_current_activations_sha256": current_det_sha256,
            "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
            "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
            "completed_layer_output_sha256s": completed_layer_output_sha256s,
            "completed_layer_output_det_sha256s": completed_layer_output_det_sha256s,
        }))
    })?;
    if reached {
        return Ok(completed_layer_output);
    }
    Ok(completed_layer_output)
}

fn trace_prefill_layer_checkpoint_with_roots(
    state: &PrefillLayerRasterState,
    layer_idx: usize,
) -> Result<Option<(String, Option<String>)>> {
    let store = AuthenticatedRasterTensorStore::artifact_backed();
    trace_prefill_layer_checkpoint(&store, state, layer_idx)
}

fn trace_prefill_layer_token_checkpoints_with_roots(
    artifact_store_roots: &RasterArtifactStoreRoots,
    state: &PrefillLayerRasterState,
    layer_idx: usize,
) -> Result<bool> {
    let (token_count, _) = state
        .current_activations_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    for token_idx in 0..token_count {
        let checkpoint_name = format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}");
        if crate::trace::trace_checkpoint_lazy_result(&checkpoint_name, || {
            let token_row = read_sequence_row_from_roots(
                artifact_store_roots,
                RasterSequenceRowRequest {
                    tensor_ref: state.current_activations_ref.clone(),
                    row_idx: token_idx,
                },
            )?;
            let token_activation = token_row.to_f32_values();
            let det_token_activation = token_row.acts();
            Ok(json!({
                "execution_mode": "deterministic",
                "layer_idx": layer_idx,
                "token_idx": token_idx,
                "token_count": token_count,
                "token_activation": token_activation,
                "det_token_activation_sha256": crate::shared::numerics::transformer_kernels::build_det_vector_commitment(&det_token_activation),
            }))
        })? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[tile]
pub fn finalize_prefill_layer_refs(
    state: PrefillLayerRasterState,
) -> Result<PrefillLayerOutputRefs> {
    if state.next_layer_idx != state.layer_count {
        bail!(
            "raster prefill layer finalized after {} layers, expected {}",
            state.next_layer_idx,
            state.layer_count
        );
    }

    Ok(PrefillLayerOutputRefs {
        final_hidden_states_ref: state.current_activations_ref,
        layer_caches: state.layer_caches,
    })
}

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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_layer_state_from_input_embedding_refs_with_roots,
        artifact_store_roots,
        input_embedding_refs,
        layer_source,
        ple_input_manifest_root,
        raster_sizing
    )?;
    let (artifact_store_roots, state) = call_recur_seq!(
        compute_next_prefill_layer_sequence_with_roots,
        (artifact_store_roots, state),
        layer_source
    )?;
    let refs = call_tile!(finalize_prefill_layer_refs, state)?;
    Ok((artifact_store_roots, refs))
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
    state: RasterSequenceProjectionArtifactState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceProjectionArtifactState,
)> {
    if state.is_complete() {
        return Ok((true, artifact_store_roots, state));
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
    Ok((false, artifact_store_roots, state))
}

#[tile]
pub fn finalize_prefill_sequence_projection_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceProjectionArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_sequence_projection_artifact_state_ref(artifact_store_roots, state)
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
    state: RasterAttentionArtifactRowState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterAttentionArtifactRowState,
)> {
    compute_next_attention_artifact_row(artifact_store_roots, state)
}

#[tile]
pub fn finalize_prefill_attention_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterAttentionArtifactRowState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    finalize_attention_artifact_row_state_ref(artifact_store_roots, state)
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

#[tile(kind = recursive)]
pub fn transform_next_prefill_sequence_unary_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceUnaryArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceUnaryArtifactState,
)> {
    compute_next_sequence_unary_artifact_row(artifact_store_roots, state)
}

#[tile]
pub fn finalize_prefill_sequence_unary_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceUnaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_sequence_unary_artifact_state_ref(artifact_store_roots, state)
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

#[tile(kind = recursive)]
pub fn transform_next_prefill_sequence_binary_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceBinaryArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterSequenceBinaryArtifactState,
)> {
    compute_next_sequence_binary_artifact_row(artifact_store_roots, state)
}

#[tile]
pub fn finalize_prefill_sequence_binary_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterSequenceBinaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_sequence_binary_artifact_state_ref(artifact_store_roots, state)
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

#[tile(kind = recursive)]
pub fn transform_next_prefill_head_artifact_row(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterHeadUnaryArtifactState,
) -> Result<(bool, RasterArtifactStoreRoots, RasterHeadUnaryArtifactState)> {
    compute_next_head_unary_artifact_row(artifact_store_roots, state)
}

#[tile]
pub fn finalize_prefill_head_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterHeadUnaryArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    finalize_head_unary_artifact_state_ref(artifact_store_roots, state)
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
    state: RasterReshapeHeadsArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterReshapeHeadsArtifactState,
)> {
    compute_next_reshape_heads_artifact_row(artifact_store_roots, state)
}

#[tile]
pub fn finalize_prefill_reshape_heads_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterReshapeHeadsArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    finalize_reshape_heads_artifact_state_ref(artifact_store_roots, state)
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
    state: RasterCombineHeadsArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterCombineHeadsArtifactState,
)> {
    compute_next_combine_heads_artifact_row(artifact_store_roots, state)
}

#[tile]
pub fn finalize_prefill_combine_heads_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterCombineHeadsArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    finalize_combine_heads_artifact_state_ref(artifact_store_roots, state)
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
    state: RasterKvCacheBuildArtifactState,
) -> Result<(
    bool,
    RasterArtifactStoreRoots,
    RasterKvCacheBuildArtifactState,
)> {
    compute_next_kv_cache_artifact_row(artifact_store_roots, state)
}

#[tile]
pub fn finalize_prefill_kv_cache_artifact_state_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    state: RasterKvCacheBuildArtifactState,
) -> Result<(RasterArtifactStoreRoots, RasterKvCacheRef)> {
    finalize_kv_cache_build_artifact_state_ref(artifact_store_roots, state)
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_sequence_projection_artifact_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        projection_rows,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        project_next_prefill_sequence_artifact_rows,
        (artifact_store_roots, state),
        layer_source,
        layer_idx,
        matrix
    )?;
    call_tile!(
        finalize_prefill_sequence_projection_artifact_ref,
        artifact_store_roots,
        state
    )
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_sequence_rms_norm_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_sequence_unary_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_sequence_unary_artifact_state_ref,
        artifact_store_roots,
        state
    )
}

#[sequence]
fn compute_sequence_gelu_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    rows_per_tile: usize,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_sequence_gelu_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_sequence_unary_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_sequence_unary_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_sequence_scale_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        scalar,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_sequence_unary_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_sequence_unary_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_sequence_add_artifact_state_from_refs,
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_sequence_binary_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_sequence_binary_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_sequence_mul_artifact_state_from_refs,
        artifact_store_roots,
        lhs_ref,
        rhs_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_sequence_binary_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_sequence_binary_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_reshape_heads_artifact_state_from_ref,
        artifact_store_roots,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        num_heads,
        head_dim
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_reshape_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_reshape_heads_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_head_rms_norm_artifact_state_from_ref,
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_head_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_head_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_value_rms_norm_artifact_state_from_ref,
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        eps,
        rows_per_tile
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_head_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_head_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
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
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_head_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_head_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_kv_cache_artifact_state_from_refs,
        artifact_store_roots,
        key_ref,
        value_ref,
        RasterTensorId::new(format!("{id_prefix}.keys"))?,
        RasterTensorId::new(format!("{id_prefix}.values"))?,
        sliding_window
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_kv_cache_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_kv_cache_artifact_state_ref,
        artifact_store_roots,
        state
    )
}

#[sequence]
fn combine_heads_artifact_ref(
    artifact_store_roots: RasterArtifactStoreRoots,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (artifact_store_roots, state) = call_tile!(
        init_prefill_combine_heads_artifact_state_from_ref,
        artifact_store_roots,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?
    )?;
    let (artifact_store_roots, state) = call_recur_tile!(
        transform_next_prefill_combine_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_combine_heads_artifact_state_ref,
        artifact_store_roots,
        state
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
    let (artifact_store_roots, state) = call_tile!(
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
    let (artifact_store_roots, state) = call_recur_tile!(
        project_next_prefill_attention_artifact_row,
        (artifact_store_roots, state)
    )?;
    call_tile!(
        finalize_prefill_attention_artifact_state_ref,
        artifact_store_roots,
        state
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

#[cfg(test)]
mod tests {
    use crate::input_embedding::raster_tiles::RasterInputEmbeddingRefs;
    use crate::prefill_layer::deterministic_tiles;
    use crate::prefill_layer::{
        materialize_prefill_layer_output_refs, run_raster_refs_from_input_embedding,
    };
    use crate::raster_authoring::prelude::auth_read;
    use crate::shared::artifacts::artifact_io::ArtifactIo;
    use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
    use crate::shared::model::transformer::{
        ActivationSequence, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleLayerWeights,
        Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence, MatrixF32,
    };
    use crate::shared::numerics::det_num::{Acc, Act, Wgt};
    use crate::shared::raster_contracts::prefill_layer::{
        AuthenticatedGemmaPrefillLayerSource, GemmaPrefillLayerSourceMetadataRequest,
    };
    use crate::shared::raster_contracts::prefill_ple::store_prefill_ple_input_manifest_with_roots;
    use crate::shared::raster_kernels::transformer::RasterActivationSequence;
    use crate::shared::tensors::raster_row_store::{
        insert_activation_sequence_artifact_ref, AuthenticatedRasterTensorStore,
    };
    use crate::RasterSizingControls;
    use anyhow::{Context, Result};
    use std::path::{Path, PathBuf};

    fn raster_sizing(projection_rows_per_tile: usize) -> RasterSizingControls {
        RasterSizingControls {
            projection_rows_per_tile,
            attention_kv_rows_per_tile: usize::MAX,
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

    #[test]
    fn ref_backed_prefill_helpers_do_not_hide_completion_loops() {
        let source = include_str!("raster_tiles.rs");
        let forbidden = concat!("while !", "state.is_complete()");

        assert!(
            !source.contains(forbidden),
            "ref-backed prefill helpers must expose dynamic loops through recursive tile calls"
        );
    }

    #[test]
    fn authored_prefill_layer_surfaces_do_not_accept_materialized_stage_inputs() {
        let source = include_str!("raster_tiles.rs");
        let materialized_ple_type = concat!("Gemma4", "PrefillPleInputs");
        let materialized_activation_arg = concat!("&", "ActivationSequence");
        let lines = source.lines().collect::<Vec<_>>();
        let mut index = 0;

        while index < lines.len() {
            let marker = lines[index].trim();
            if marker == "#[tile]" || marker == "#[sequence]" {
                let mut signature = String::new();
                index += 1;
                while index < lines.len() {
                    signature.push_str(lines[index]);
                    signature.push('\n');
                    if lines[index].contains('{') {
                        break;
                    }
                    index += 1;
                }
                assert!(
                    !signature.contains(materialized_ple_type),
                    "authored raster tile/sequence must not accept materialized PLE inputs: {signature}"
                );
                assert!(
                    !signature.contains(materialized_activation_arg),
                    "authored raster tile/sequence must not accept materialized activation inputs: {signature}"
                );
            }
            index += 1;
        }

        let materialized_compat_fn = concat!("pub fn ", "run_materialized_compat(");
        let compatibility_comment = concat!("Compatibility adapter", " for dev/tests");
        assert!(!source.contains(materialized_compat_fn));
        assert!(!source.contains(compatibility_comment));
    }

    #[test]
    fn prefill_layer_main_threads_roots_without_local_tensor_store() {
        let source = include_str!("raster_tiles.rs");
        let main_start = source
            .find("pub fn main(\n    artifact_store_roots: RasterArtifactStoreRoots,")
            .expect("prefill layer main should exist");
        let after_main = &source[main_start..];
        let next_tile = after_main
            .find("\n#[tile]\npub fn init_prefill_sequence_projection")
            .expect("next authored helper marks end of main");
        let main_body = &after_main[..next_tile];

        assert!(main_body.contains("compute_next_prefill_layer_sequence_with_roots"));
        assert!(
            !main_body.contains("AuthenticatedRasterTensorStore::new()"),
            "proof-shaped prefill main must thread artifact roots instead of creating a tensor store"
        );
    }

    #[test]
    fn single_layer_no_ple_matches_deterministic_prefill_layer() {
        let (_path, model) = no_ple_model();

        assert_raster_matches_deterministic(
            &model,
            vec![vec![Act::from_num(1.0), Act::from_num(-0.5)]],
        );
    }

    #[test]
    fn sliding_attention_matches_deterministic_prefill_layer() {
        let (_path, mut model) = no_ple_model();
        model.layers[0].attention_kind = Gemma4AttentionKind::Sliding;
        model.layers[0].sliding_window = Some(1);
        model.layers[0].cache_sliding_window = Some(1);

        let raster = assert_raster_matches_deterministic(
            &model,
            vec![
                vec![Act::from_num(1.0), Act::from_num(0.0)],
                vec![Act::from_num(0.5), Act::from_num(-0.5)],
                vec![Act::from_num(-1.0), Act::from_num(1.0)],
            ],
        );
        assert_eq!(raster.1[0].current_len(), 1);
    }

    #[test]
    fn attention_k_equals_v_matches_deterministic_prefill_layer() {
        let (_path, mut model) = no_ple_model();
        model.layers[0].v_proj = None;
        model.layers[0].attention_k_eq_v = true;

        assert_raster_matches_deterministic(
            &model,
            vec![
                vec![Act::from_num(1.0), Act::from_num(0.0)],
                vec![Act::from_num(0.0), Act::from_num(1.0)],
            ],
        );
    }

    #[test]
    fn donor_kv_sharing_matches_deterministic_prefill_layer() {
        let (_path, mut model) = no_ple_model();
        let mut donor_layer = model.layers[0].clone();
        donor_layer.kv_shared_layer_index = Some(0);
        model.layers.push(donor_layer);

        let raster = assert_raster_matches_deterministic(
            &model,
            vec![
                vec![Act::from_num(1.0), Act::from_num(0.0)],
                vec![Act::from_num(0.0), Act::from_num(1.0)],
            ],
        );
        assert_eq!(raster.1.len(), 2);
        assert_eq!(raster.1[1].current_len(), 0);
    }

    #[test]
    fn zero_length_self_cache_materializes_as_empty_slot() {
        let (_path, mut model) = no_ple_model();
        model.layers[0].attention_kind = Gemma4AttentionKind::Sliding;
        model.layers[0].sliding_window = Some(1);
        model.layers[0].cache_sliding_window = Some(0);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let raster = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
            .expect("zero-length self cache should succeed");

        assert_eq!(raster.1.len(), 1);
        assert_eq!(raster.1[0].current_len(), 0);
    }

    #[test]
    fn empty_donor_cache_fails_closed() {
        let (_path, mut model) = no_ple_model();
        let mut shared_layer = model.layers[0].clone();
        shared_layer.kv_shared_layer_index = Some(0);
        let mut chained_layer = model.layers[0].clone();
        chained_layer.kv_shared_layer_index = Some(1);
        model.layers.push(shared_layer);
        model.layers.push(chained_layer);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

        let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
            .expect_err("empty donor cache should fail");

        assert!(error.to_string().contains("donor cache is empty"));
    }

    #[test]
    fn multi_head_sliding_attention_matches_deterministic_prefill_layer() {
        let (_path, model) = multi_head_sliding_model();

        let raster = assert_raster_matches_deterministic(
            &model,
            vec![
                vec![
                    Act::from_num(1.0),
                    Act::from_num(0.0),
                    Act::from_num(-0.5),
                    Act::from_num(0.25),
                ],
                vec![
                    Act::from_num(0.25),
                    Act::from_num(0.75),
                    Act::from_num(0.5),
                    Act::from_num(-0.25),
                ],
                vec![
                    Act::from_num(-1.0),
                    Act::from_num(1.0),
                    Act::from_num(0.0),
                    Act::from_num(0.5),
                ],
            ],
        );
        assert_eq!(raster.1[0].current_len(), 2);
    }

    #[test]
    fn non_prior_donor_cache_fails_closed() {
        let (_path, mut model) = no_ple_model();
        model.layers[0].kv_shared_layer_index = Some(0);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

        let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
            .expect_err("self donor should fail");

        assert!(error
            .to_string()
            .contains("cannot share KV with non-prior donor"));
    }

    #[test]
    fn nonzero_mlp_and_layer_scalar_match_deterministic_prefill_layer() {
        let (_path, model) = nonzero_model(false, true);

        assert_raster_matches_deterministic(
            &model,
            vec![
                vec![Act::from_num(1.0), Act::from_num(-0.5)],
                vec![Act::from_num(0.25), Act::from_num(0.75)],
            ],
        );
    }

    #[test]
    fn chunked_projection_matches_deterministic_prefill_layer() {
        let (_path, model) = nonzero_model(false, true);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let rows = vec![
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
            vec![Act::from_num(0.25), Act::from_num(0.75)],
        ];
        let input_internal = InternalActivationSequence::from_det_values(rows);
        let input = activation_sequence_from_internal(input_internal.clone());

        let raster = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(2))
            .expect("raster prefill layer should run");
        let deterministic = deterministic_tiles::run_internal(input_internal, &model, None)
            .expect("deterministic prefill layer should run");

        assert_eq!(raster.0.activations, deterministic.0.activations);
        assert_eq!(
            raster.0.det_activations_sha256,
            deterministic.0.det_activations_sha256
        );
        assert_eq!(raster.1, deterministic.1);
    }

    #[test]
    fn ple_layer_with_matching_input_matches_deterministic_prefill_layer() {
        let (_path, model) = nonzero_model(true, false);
        let ple_inputs = ple_inputs(vec![
            vec![Act::from_num(0.5), Act::from_num(-0.25)],
            vec![Act::from_num(1.0), Act::from_num(0.25)],
        ]);

        assert_raster_matches_deterministic_with_ple(
            &model,
            vec![
                vec![Act::from_num(1.0), Act::from_num(-0.5)],
                vec![Act::from_num(0.25), Act::from_num(0.75)],
            ],
            Some(&ple_inputs),
        );
    }

    #[test]
    fn ple_input_width_can_differ_from_hidden_size() {
        let (_path, model) = ple_width_differs_from_hidden_model();
        let ple_inputs = ple_inputs(vec![
            vec![Act::from_num(0.5), Act::from_num(-0.25)],
            vec![Act::from_num(1.0), Act::from_num(0.25)],
        ]);

        assert_raster_matches_deterministic_with_ple(
            &model,
            vec![
                vec![
                    Act::from_num(1.0),
                    Act::from_num(-0.5),
                    Act::from_num(0.25),
                    Act::from_num(0.75),
                ],
                vec![
                    Act::from_num(0.25),
                    Act::from_num(0.75),
                    Act::from_num(-0.5),
                    Act::from_num(1.0),
                ],
            ],
            Some(&ple_inputs),
        );
    }

    #[test]
    fn ple_layer_missing_input_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

        let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
            .expect_err("missing PLE input should fail");

        assert!(error
            .to_string()
            .contains("requires PLE inputs but none were provided"));
    }

    #[test]
    fn ple_input_on_non_ple_layer_fails_closed() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let ple_inputs = ple_inputs(vec![vec![Act::from_num(0.5), Act::from_num(0.25)]]);

        let error =
            run_roots_path_with_optional_ple(&input, &source, Some(&ple_inputs), raster_sizing(1))
                .expect_err("PLE input should fail");

        assert!(error
            .to_string()
            .contains("received PLE inputs without PLE weights"));
    }

    #[test]
    fn ple_input_shape_mismatch_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let ple_inputs = ple_inputs(vec![vec![
            Act::from_num(0.5),
            Act::from_num(0.25),
            Act::from_num(0.125),
        ]]);

        let error =
            run_roots_path_with_optional_ple(&input, &source, Some(&ple_inputs), raster_sizing(1))
                .expect_err("PLE width should fail");

        assert!(error
            .to_string()
            .contains("transformer layer PLE input width 3, expected 2"));
    }

    #[test]
    fn zero_layer_source_fails_closed() {
        let model = Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::DetNumWgt,
            embedding_table: None,
            embedding_source: None,
            layers: Vec::new(),
            ple_global: None,
            final_norm_weight: vec![1.0, 1.0],
            final_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(1.0)]),
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 1,
                    cols: 2,
                    values: vec![0.0, 0.0],
                },
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 0.0,
            rms_norm_eps_det: Some(Acc::from_num(0.0)),
        };
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("empty", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

        let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
            .expect_err("zero layers should fail");

        assert!(error
            .to_string()
            .contains("transformer prefill requires at least one layer"));
    }

    #[test]
    fn empty_activation_sequence_fails_closed() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(Vec::new());

        let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
            .expect_err("empty input should fail");

        assert!(error
            .to_string()
            .contains("requires at least one activation row"));
    }

    #[test]
    fn non_deterministic_activation_input_fails_closed() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = ActivationSequence::from_values(
            vec![vec![1.0, 0.0]],
            crate::shared::numerics::transformer_kernels::build_activation_commitment(&[vec![
                1.0, 0.0,
            ]]),
        );

        let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
            .expect_err("f32-only input should fail");

        assert!(error.to_string().contains("requires canonical activations"));
    }

    fn assert_raster_matches_deterministic(
        model: &Gemma4TransformerModel,
        rows: Vec<Vec<Act>>,
    ) -> (
        ActivationSequence,
        Vec<crate::shared::model::transformer::LayerKvCache>,
    ) {
        assert_raster_matches_deterministic_with_ple(model, rows, None)
    }

    fn assert_raster_matches_deterministic_with_ple(
        model: &Gemma4TransformerModel,
        rows: Vec<Vec<Act>>,
        ple_inputs: Option<&Gemma4PrefillPleInputs>,
    ) -> (
        ActivationSequence,
        Vec<crate::shared::model::transformer::LayerKvCache>,
    ) {
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", model)
            .expect("source should build");
        let input_internal = InternalActivationSequence::from_det_values(rows);
        let input = activation_sequence_from_internal(input_internal.clone());

        let raster =
            run_roots_path_with_optional_ple(&input, &source, ple_inputs, raster_sizing(1))
                .expect("raster prefill layer should run");
        let deterministic = deterministic_tiles::run_internal(input_internal, model, ple_inputs)
            .expect("deterministic prefill layer should run");

        assert_eq!(raster.0.activations, deterministic.0.activations);
        assert_eq!(
            raster.0.det_activations_sha256,
            deterministic.0.det_activations_sha256
        );
        assert_eq!(raster.1, deterministic.1);
        raster
    }

    fn run_roots_path_with_optional_ple(
        input: &ActivationSequence,
        source: &AuthenticatedGemmaPrefillLayerSource,
        ple_inputs: Option<&Gemma4PrefillPleInputs>,
        raster_sizing: RasterSizingControls,
    ) -> Result<(
        ActivationSequence,
        Vec<crate::shared::model::transformer::LayerKvCache>,
    )> {
        ArtifactIo::reset_store();
        let input_internal = input.clone_internal();
        let input_rows = input_internal.det_values().ok_or_else(|| {
            anyhow::anyhow!("raster prefill layer requires canonical activations")
        })?;
        let input_ref = insert_activation_sequence_artifact_ref(
            "prefill.layer.test.input_embedding",
            RasterActivationSequence::from_acts(input_rows.to_vec()),
        )?;
        let artifact_store_roots = ArtifactIo::export_store_roots();
        let input_embedding_refs = RasterInputEmbeddingRefs {
            source_id: "embedding-fixture".to_string(),
            embedding_source_root: "embedding-root".to_string(),
            prompt_token_ids_root: "token-root".to_string(),
            prompt_token_count: input_ref.row_count(),
            embedded_prompt_activations_ref: input_ref,
        };
        let (artifact_store_roots, ple_input_manifest_root) =
            store_materialized_ple_inputs_with_roots(artifact_store_roots, source, ple_inputs)?;
        let (_artifact_store_roots, refs) = run_raster_refs_from_input_embedding(
            artifact_store_roots,
            &input_embedding_refs,
            source,
            ple_input_manifest_root.as_deref(),
            raster_sizing,
        )?;
        let store = AuthenticatedRasterTensorStore::artifact_backed();
        materialize_prefill_layer_output_refs(&store, &refs)
    }

    fn store_materialized_ple_inputs_with_roots(
        artifact_store_roots: RasterArtifactStoreRoots,
        source: &AuthenticatedGemmaPrefillLayerSource,
        ple_inputs: Option<&Gemma4PrefillPleInputs>,
    ) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
        let Some(ple_inputs) = ple_inputs else {
            return Ok((artifact_store_roots, None));
        };
        let metadata = auth_read!(source, GemmaPrefillLayerSourceMetadataRequest)?;
        let mut token_count = None;
        let mut per_layer_inputs = Vec::with_capacity(metadata.layer_count);

        for layer_idx in 0..metadata.layer_count {
            let input = ple_inputs.clone_layer_internal(layer_idx);
            let Some(input) = input else {
                per_layer_inputs.push(None);
                continue;
            };
            let rows = input.det_values().ok_or_else(|| {
                anyhow::anyhow!("raster PLE inputs require canonical activations")
            })?;
            match token_count {
                Some(expected) if expected != rows.len() => {
                    anyhow::bail!(
                        "materialized PLE input layer {layer_idx} contains {} tokens, expected {expected}",
                        rows.len()
                    );
                }
                None => token_count = Some(rows.len()),
                _ => {}
            }
            per_layer_inputs.push(Some(insert_activation_sequence_artifact_ref(
                &format!("prefill.layer.test.ple.{layer_idx}"),
                RasterActivationSequence::from_acts(rows.to_vec()),
            )?));
        }

        if per_layer_inputs.iter().all(Option::is_none) {
            return Ok((ArtifactIo::export_store_roots(), None));
        }

        let artifact_store_roots = ArtifactIo::export_store_roots();
        let (artifact_store_roots, manifest_root) = store_prefill_ple_input_manifest_with_roots(
            &artifact_store_roots,
            metadata.source_id,
            metadata.layer_count,
            token_count.ok_or_else(|| {
                anyhow::anyhow!("materialized PLE inputs contained no layer rows")
            })?,
            &per_layer_inputs,
        )?;
        Ok((artifact_store_roots, Some(manifest_root)))
    }

    fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
        activation_sequence_from_internal(InternalActivationSequence::from_det_values(rows))
    }

    fn activation_sequence_from_internal(
        input_internal: InternalActivationSequence,
    ) -> ActivationSequence {
        let mut input = ActivationSequence::from_internal(
            input_internal.clone(),
            crate::shared::numerics::transformer_kernels::build_activation_commitment(
                input_internal.as_f32_slice(),
            ),
        );
        input.det_activations_sha256 = Some(
            crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
                input_internal.det_values().expect("det input"),
            ),
        );
        input
    }

    fn ple_inputs(rows: Vec<Vec<Act>>) -> Gemma4PrefillPleInputs {
        Gemma4PrefillPleInputs::from_internal(vec![Some(
            InternalActivationSequence::from_det_values(rows),
        )])
    }

    fn no_ple_model() -> (PathBuf, Gemma4TransformerModel) {
        let hidden_width = 2;
        let matrices = vec![
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
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Full,
            hidden_size: hidden_width,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden_width,
            sliding_window: None,
            cache_sliding_window: None,
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
                embedding_source: None,
                layers: vec![layer],
                ple_global: None,
                final_norm_weight: vec![1.0; hidden_width],
                final_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
                logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                    weight: MatrixF32 {
                        rows: 1,
                        cols: hidden_width,
                        values: vec![0.0; hidden_width],
                    },
                    det_weight: None,
                },
                final_logit_softcapping: None,
                final_logit_softcapping_det: None,
                rms_norm_eps: 0.0,
                rms_norm_eps_det: Some(Acc::from_num(0.0)),
            },
        )
    }

    fn nonzero_model(has_ple: bool, has_layer_scalar: bool) -> (PathBuf, Gemma4TransformerModel) {
        let hidden_width = 2;
        let matrices = vec![
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            zero_matrix(hidden_width),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
            identity_matrix(),
        ];
        let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
        let mut sources = sources.into_iter();
        let q_proj = det_matrix(sources.next().expect("q source"));
        let k_proj = det_matrix(sources.next().expect("k source"));
        let v_proj = det_matrix(sources.next().expect("v source"));
        let o_proj = det_matrix(sources.next().expect("o source"));
        let gate_proj = det_matrix(sources.next().expect("gate source"));
        let up_proj = det_matrix(sources.next().expect("up source"));
        let down_proj = det_matrix(sources.next().expect("down source"));
        let ple_input_gate = det_matrix(sources.next().expect("PLE input gate source"));
        let ple_layer_projection = det_matrix(sources.next().expect("PLE projection source"));

        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Full,
            hidden_size: hidden_width,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden_width,
            sliding_window: None,
            cache_sliding_window: None,
            rms_norm_eps: 0.001,
            rms_norm_eps_det: Some(Acc::from_num(0.001)),
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj,
            k_proj,
            v_proj: Some(v_proj),
            o_proj,
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
            gate_proj,
            up_proj,
            down_proj,
            ple: has_ple.then(|| Gemma4PleLayerWeights {
                input_gate: ple_input_gate,
                layer_projection: ple_layer_projection,
                post_input_norm_weight: vec![1.0; hidden_width],
                post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            }),
            layer_scalar: has_layer_scalar.then_some(0.5),
            layer_scalar_det: has_layer_scalar.then_some(Act::from_num(0.5)),
        };

        (
            path,
            Gemma4TransformerModel {
                provenance: Gemma4ModelProvenance::DetNumWgt,
                embedding_table: None,
                embedding_source: None,
                layers: vec![layer],
                ple_global: None,
                final_norm_weight: vec![1.0; hidden_width],
                final_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
                logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                    weight: MatrixF32 {
                        rows: 1,
                        cols: hidden_width,
                        values: vec![0.0; hidden_width],
                    },
                    det_weight: None,
                },
                final_logit_softcapping: None,
                final_logit_softcapping_det: None,
                rms_norm_eps: 0.001,
                rms_norm_eps_det: Some(Acc::from_num(0.001)),
            },
        )
    }

    fn ple_width_differs_from_hidden_model() -> (PathBuf, Gemma4TransformerModel) {
        let hidden_width = 4;
        let ple_width = 2;
        let matrices = vec![
            zero_matrix_rect(hidden_width, hidden_width),
            zero_matrix_rect(hidden_width, hidden_width),
            zero_matrix_rect(hidden_width, hidden_width),
            zero_matrix_rect(hidden_width, hidden_width),
            zero_matrix_rect(hidden_width, hidden_width),
            zero_matrix_rect(hidden_width, hidden_width),
            zero_matrix_rect(hidden_width, hidden_width),
            zero_matrix_rect(ple_width, hidden_width),
            zero_matrix_rect(hidden_width, ple_width),
        ];
        let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
        let mut sources = sources.into_iter();
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Full,
            hidden_size: hidden_width,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden_width,
            sliding_window: None,
            cache_sliding_window: None,
            rms_norm_eps: 0.001,
            rms_norm_eps_det: Some(Acc::from_num(0.001)),
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
            ple: Some(Gemma4PleLayerWeights {
                input_gate: det_matrix(sources.next().expect("PLE input gate source")),
                layer_projection: det_matrix(sources.next().expect("PLE projection source")),
                post_input_norm_weight: vec![1.0; hidden_width],
                post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            }),
            layer_scalar: None,
            layer_scalar_det: None,
        };

        (
            path,
            Gemma4TransformerModel {
                provenance: Gemma4ModelProvenance::DetNumWgt,
                embedding_table: None,
                embedding_source: None,
                layers: vec![layer],
                ple_global: None,
                final_norm_weight: vec![1.0; hidden_width],
                final_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
                logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                    weight: MatrixF32 {
                        rows: 1,
                        cols: hidden_width,
                        values: vec![0.0; hidden_width],
                    },
                    det_weight: None,
                },
                final_logit_softcapping: None,
                final_logit_softcapping_det: None,
                rms_norm_eps: 0.001,
                rms_norm_eps_det: Some(Acc::from_num(0.001)),
            },
        )
    }

    fn multi_head_sliding_model() -> (PathBuf, Gemma4TransformerModel) {
        let hidden_width = 4;
        let matrices = vec![
            zero_matrix_rect(4, 4),
            zero_matrix_rect(2, 4),
            zero_matrix_rect(2, 4),
            zero_matrix_rect(4, 4),
            zero_matrix_rect(8, 4),
            zero_matrix_rect(8, 4),
            zero_matrix_rect(4, 8),
        ];
        let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
        let mut sources = sources.into_iter();
        let layer = Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: hidden_width,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            cache_sliding_window: Some(2),
            rms_norm_eps: 0.0,
            rms_norm_eps_det: Some(Acc::from_num(0.0)),
            rope_base: 10_000.0,
            rope_base_det: Some(Acc::from_num(10_000.0)),
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: det_matrix(sources.next().expect("q source")),
            k_proj: det_matrix(sources.next().expect("k source")),
            v_proj: Some(det_matrix(sources.next().expect("v source"))),
            o_proj: det_matrix(sources.next().expect("o source")),
            q_norm_weight: vec![1.0; 2],
            q_norm_weight_det: Some(vec![Wgt::from_num(1.0); 2]),
            k_norm_weight: vec![1.0; 2],
            k_norm_weight_det: Some(vec![Wgt::from_num(1.0); 2]),
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
                embedding_source: None,
                layers: vec![layer],
                ple_global: None,
                final_norm_weight: vec![1.0; hidden_width],
                final_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
                logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                    weight: MatrixF32 {
                        rows: 1,
                        cols: hidden_width,
                        values: vec![0.0; hidden_width],
                    },
                    det_weight: None,
                },
                final_logit_softcapping: None,
                final_logit_softcapping_det: None,
                rms_norm_eps: 0.0,
                rms_norm_eps_det: Some(Acc::from_num(0.0)),
            },
        )
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

    fn zero_matrix_rect(rows: usize, cols: usize) -> Vec<Vec<Wgt>> {
        vec![vec![Wgt::from_num(0.0); cols]; rows]
    }

    fn det_matrix(source: DetNumTensorSliceSource) -> Gemma4LayerMatrixSource {
        Gemma4LayerMatrixSource::from_det_num_source(source)
    }

    fn write_det_matrices(
        matrices: Vec<Vec<Vec<Wgt>>>,
    ) -> Result<(PathBuf, Vec<DetNumTensorSliceSource>)> {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-prefill-layer-state-{}-{}-{}.detwgt",
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
