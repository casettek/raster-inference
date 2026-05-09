use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::raster_authoring::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::raster_prefill_layer::{
    AuthenticatedGemmaPrefillLayerSource, GemmaPrefillAttentionKind, GemmaPrefillLayerMatrixKind,
    GemmaPrefillLayerMetadata, GemmaPrefillLayerMetadataRequest, GemmaPrefillLayerNormKind,
    GemmaPrefillLayerNormWeightsRequest, GemmaPrefillLayerScalars, GemmaPrefillLayerScalarsRequest,
    GemmaPrefillLayerSourceMetadataRequest,
};
use crate::shared::raster_prefill_ple::RasterPrefillPleInputRefs;
use crate::shared::raster_row_store::{
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterAttentionHeadsRef,
    RasterKvCacheRef, RasterSequenceRowRequest, RasterTensorId,
};
use crate::shared::raster_transformer_kernels::{
    append_projection_chunk_to_state, compute_next_attention_row, compute_next_combine_heads_row,
    compute_next_head_unary_row, compute_next_kv_cache_row, compute_next_reshape_heads_row,
    compute_next_sequence_binary_row, compute_next_sequence_unary_row,
    finalize_attention_row_state, finalize_attention_row_state_ref, finalize_combine_heads_state,
    finalize_combine_heads_state_ref, finalize_head_unary_row_state,
    finalize_head_unary_row_state_ref, finalize_kv_cache_build_state,
    finalize_kv_cache_build_state_ref, finalize_reshape_heads_state,
    finalize_reshape_heads_state_ref, finalize_sequence_binary_row_state,
    finalize_sequence_binary_row_state_ref, finalize_sequence_projection_state,
    finalize_sequence_projection_state_ref, finalize_sequence_unary_row_state,
    finalize_sequence_unary_row_state_ref, init_attention_row_state,
    init_attention_row_state_from_refs, init_combine_heads_state,
    init_combine_heads_state_from_ref, init_head_rms_norm_row_state,
    init_head_rms_norm_row_state_from_ref, init_kv_cache_build_state,
    init_kv_cache_build_state_from_refs, init_reshape_heads_state,
    init_reshape_heads_state_from_ref, init_rope_row_state, init_rope_row_state_from_ref,
    init_sequence_add_row_state, init_sequence_add_row_state_from_refs,
    init_sequence_gelu_row_state, init_sequence_gelu_row_state_from_ref,
    init_sequence_mul_row_state, init_sequence_mul_row_state_from_refs,
    init_sequence_projection_state, init_sequence_projection_state_from_ref,
    init_sequence_rms_norm_row_state, init_sequence_rms_norm_row_state_from_ref,
    init_sequence_scale_row_state, init_sequence_scale_row_state_from_ref,
    init_value_rms_norm_row_state, init_value_rms_norm_row_state_from_ref,
    validate_projection_rows_per_tile, RasterActivationSequence, RasterAttentionHeadSequence,
    RasterAttentionRowState, RasterCombineHeadsState, RasterHeadUnaryState, RasterKvCache,
    RasterKvCacheBuildState, RasterReshapeHeadsState, RasterSequenceBinaryState,
    RasterSequenceProjectionState, RasterSequenceUnaryState,
};
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, InternalActivationSequence, LayerKvCache,
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
pub fn init_prefill_layer_state(
    store: &mut AuthenticatedRasterTensorStore,
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
    raster_sizing: RasterSizingControls,
) -> Result<PrefillLayerRasterState> {
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    crate::shared::raster_transformer_kernels::validate_attention_kv_rows_per_tile(
        raster_sizing.attention_kv_rows_per_tile,
    )?;
    crate::shared::raster_transformer_kernels::validate_sequence_rows_per_tile(
        raster_sizing.sequence_rows_per_tile,
    )?;
    crate::shared::raster_transformer_kernels::validate_head_rows_per_tile(
        raster_sizing.head_rows_per_tile,
    )?;
    let metadata = auth_read!(layer_source, GemmaPrefillLayerSourceMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer prefill requires at least one layer");
    }

    let current_activations = raster_activation_sequence_from_activation(input_activations)?;
    if current_activations.is_empty() {
        bail!("transformer layer execution requires at least one activation row");
    }
    let token_count = current_activations.len();
    let current_activations_ref = store.insert_activation_sequence(
        RasterTensorId::new("prefill.layer.current.initial")?,
        current_activations,
    )?;

    let per_layer_inputs = match ple_input_refs {
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
            ple_input_refs.per_layer_inputs().to_vec()
        }
        None => vec![None; metadata.layer_count],
    };

    Ok(PrefillLayerRasterState {
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
    })
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
    validate_prefill_layer_ple_input_ref(state, &layer, per_layer_input.as_ref())?;

    Ok(PrefillLayerContext {
        layer_idx,
        layer,
        donor_cache,
        per_layer_input,
    })
}

fn validate_prefill_layer_ple_input_ref(
    state: &PrefillLayerRasterState,
    layer: &GemmaPrefillLayerMetadata,
    per_layer_input: Option<&RasterActivationSequenceRef>,
) -> Result<()> {
    match (layer.has_ple, per_layer_input) {
        (false, Some(_)) => bail!("transformer layer received PLE inputs without PLE weights"),
        (true, None) => bail!("transformer layer requires PLE inputs but none were provided"),
        (false, None) => Ok(()),
        (true, Some(input_ref)) => {
            let (token_count, width) = input_ref.tensor_ref().shape().sequence_metadata()?;
            let (expected_token_count, _) = state
                .current_activations_ref
                .tensor_ref()
                .shape()
                .sequence_metadata()?;
            let expected_width = expected_prefill_ple_input_width(layer)?;
            if token_count != expected_token_count {
                bail!(
                    "transformer layer PLE input has {token_count} rows, expected {expected_token_count}"
                );
            }
            if width != expected_width {
                bail!(
                    "transformer layer PLE input width {width}, expected {}",
                    expected_width
                );
            }
            Ok(())
        }
    }
}

fn expected_prefill_ple_input_width(layer: &GemmaPrefillLayerMetadata) -> Result<usize> {
    let input_gate_shape = layer
        .ple_input_gate_shape
        .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing PLE input gate shape"))?;
    let layer_projection_shape = layer.ple_layer_projection_shape.ok_or_else(|| {
        anyhow!("Gemma prefill layer metadata is missing PLE layer projection shape")
    })?;
    if input_gate_shape.rows != layer_projection_shape.cols {
        bail!(
            "Gemma prefill layer PLE width mismatch: input gate rows {} vs layer projection cols {}",
            input_gate_shape.rows,
            layer_projection_shape.cols
        );
    }
    Ok(input_gate_shape.rows)
}

#[sequence(kind = recursive)]
pub fn compute_next_prefill_layer_sequence(
    state: PrefillLayerRasterState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, PrefillLayerRasterState)> {
    if state.next_layer_idx >= state.layer_count {
        return Ok((true, state));
    }

    let context = call_tile!(prepare_next_prefill_layer_context, &state, layer_source)?;
    if let Some(per_layer_input) = context.per_layer_input.as_ref() {
        auth_read!(
            store,
            RasterSequenceRowRequest {
                tensor_ref: per_layer_input.clone(),
                row_idx: 0,
            }
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
    let (layer_output_ref, layer_cache) = call_seq!(
        run_prefill_layer_sequence_ref,
        store,
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
        update_prefill_layer_state_refs,
        store,
        state,
        context.layer_idx,
        layer_output_ref,
        layer_cache
    )
}

#[tile]
pub fn update_prefill_layer_state_refs(
    store: &mut AuthenticatedRasterTensorStore,
    mut state: PrefillLayerRasterState,
    layer_idx: usize,
    layer_output_ref: RasterActivationSequenceRef,
    layer_cache: PrefillLayerCacheSlot,
) -> Result<(bool, PrefillLayerRasterState)> {
    if layer_idx != state.next_layer_idx {
        bail!(
            "cannot update prefill layer {layer_idx} while next layer is {}",
            state.next_layer_idx
        );
    }

    state.current_activations_ref = layer_output_ref;
    state.layer_caches.push(layer_cache);

    let completed_layer_output = trace_prefill_layer_checkpoint(store, &state, layer_idx)?;
    if let Some((sha256, det_sha256)) = completed_layer_output {
        state.completed_layer_output_sha256s.push(sha256);
        state.completed_layer_output_det_sha256s.push(det_sha256);
    }

    if trace_prefill_layer_token_checkpoints(store, &state, layer_idx)? {
        state.next_layer_idx += 1;
        state.layer_count = state.next_layer_idx;
        return Ok((true, state));
    }
    state.next_layer_idx += 1;
    Ok((false, state))
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
            crate::shared::transformer_kernels::build_activation_commitment(&current_values);
        let current_det_sha256 = Some(
            crate::shared::transformer_kernels::build_det_activation_commitment(
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
            "det_layer_caches_sha256": crate::shared::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
            "completed_layer_output_sha256s": completed_layer_output_sha256s,
            "completed_layer_output_det_sha256s": completed_layer_output_det_sha256s,
        }))
    })?;
    if reached {
        return Ok(completed_layer_output);
    }
    Ok(completed_layer_output)
}

fn trace_prefill_layer_token_checkpoints(
    store: &AuthenticatedRasterTensorStore,
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
            let token_row = auth_read!(
                store,
                RasterSequenceRowRequest {
                    tensor_ref: state.current_activations_ref.clone(),
                    row_idx: token_idx,
                }
            )?;
            let token_activation = token_row.to_f32_values();
            let det_token_activation = token_row.acts();
            Ok(json!({
                "execution_mode": "deterministic",
                "layer_idx": layer_idx,
                "token_idx": token_idx,
                "token_count": token_count,
                "token_activation": token_activation,
                "det_token_activation_sha256": crate::shared::transformer_kernels::build_det_vector_commitment(&det_token_activation),
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

#[tile]
pub fn materialize_prefill_layer_output_refs(
    store: &AuthenticatedRasterTensorStore,
    refs: &PrefillLayerOutputRefs,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    // Public/dev compatibility boundary. The proof-shaped prefill path carries
    // `PrefillLayerOutputRefs` forward and materializes only for public results
    // and checkpoint payloads.
    let current_activations =
        materialize_prefill_activation_sequence_from_store(store, &refs.final_hidden_states_ref)?;
    let det_activations = raster_sequence_acts(&current_activations);
    let values = current_activations.to_f32_values();
    let mut activation_sequence = ActivationSequence::from_internal(
        InternalActivationSequence::from_det_values(det_activations.clone()),
        crate::shared::transformer_kernels::build_activation_commitment(&values),
    );
    activation_sequence.det_activations_sha256 =
        Some(crate::shared::transformer_kernels::build_det_activation_commitment(&det_activations));

    Ok((
        activation_sequence,
        refs.layer_caches
            .iter()
            .map(|cache| {
                materialize_prefill_layer_cache_from_store(store, cache)
                    .map(layer_cache_from_raster)
            })
            .collect::<Result<Vec<_>>>()?,
    ))
}

#[tile]
pub fn finalize_prefill_layer_state(
    store: &AuthenticatedRasterTensorStore,
    state: PrefillLayerRasterState,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    let refs = finalize_prefill_layer_refs(state)?;
    materialize_prefill_layer_output_refs(store, &refs)
}

#[tile]
pub fn init_prefill_layer_store() -> AuthenticatedRasterTensorStore {
    AuthenticatedRasterTensorStore::new()
}

#[tile]
pub fn materialize_prefill_activation_sequence(
    store: &AuthenticatedRasterTensorStore,
    sequence_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationSequence> {
    materialize_prefill_activation_sequence_from_store(store, sequence_ref)
}

#[tile]
pub fn materialize_prefill_layer_cache(
    store: &AuthenticatedRasterTensorStore,
    cache: &PrefillLayerCacheSlot,
) -> Result<RasterKvCache> {
    materialize_prefill_layer_cache_from_store(store, cache)
}

pub fn run_materialized_compat(
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    raster_sizing: RasterSizingControls,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    // Compatibility adapter for dev/tests that still hold materialized
    // `Gemma4PrefillPleInputs`. ZkVM-shaped raster replay should enter through
    // `run_with_store` with refs already written by prepare-aux.
    let mut store = AuthenticatedRasterTensorStore::new();
    let ple_input_refs = import_materialized_ple_inputs(&mut store, layer_source, ple_inputs)?;
    run_with_store(
        &mut store,
        input_activations,
        layer_source,
        ple_input_refs.as_ref(),
        raster_sizing,
    )
}

#[sequence]
pub fn run_refs_with_store(
    store: &mut AuthenticatedRasterTensorStore,
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
    raster_sizing: RasterSizingControls,
) -> Result<PrefillLayerOutputRefs> {
    let state = call_tile!(
        init_prefill_layer_state,
        store,
        input_activations,
        layer_source,
        ple_input_refs,
        raster_sizing
    )?;
    let state = call_recur_seq!(
        compute_next_prefill_layer_sequence,
        state,
        layer_source,
        store
    )?;
    call_tile!(finalize_prefill_layer_refs, state)
}

#[sequence]
pub fn run_with_store(
    store: &mut AuthenticatedRasterTensorStore,
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
    raster_sizing: RasterSizingControls,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    let refs = call_seq!(
        run_refs_with_store,
        store,
        input_activations,
        layer_source,
        ple_input_refs,
        raster_sizing
    )?;
    call_tile!(materialize_prefill_layer_output_refs, store, &refs)
}

#[tile]
pub fn init_prefill_sequence_projection(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    init_sequence_projection_state(store, input, projection_rows, projection_rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_projection_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    init_sequence_projection_state_from_ref(
        store,
        input_ref,
        output_id,
        projection_rows,
        projection_rows_per_tile,
    )
}

#[tile(kind = recursive)]
pub fn project_next_prefill_sequence_rows(
    mut state: RasterSequenceProjectionState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterSequenceProjectionState)> {
    if state.is_complete() {
        return Ok((true, state));
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
            crate::shared::raster_prefill_layer::GemmaPrefillLayerMatrixRowRequest {
                layer_idx,
                matrix,
                row_idx,
            },
        )?);
    }
    append_projection_chunk_to_state(&mut state, store, &rows)?;
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
    Ok((false, state))
}

#[tile]
pub fn finalize_prefill_sequence_projection(
    state: RasterSequenceProjectionState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    finalize_sequence_projection_state(state, store)
}

#[tile]
pub fn finalize_prefill_sequence_projection_ref(
    state: RasterSequenceProjectionState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    finalize_sequence_projection_state_ref(state, store)
}

#[sequence]
pub fn project_sequence_with_prefill_source(
    input: &RasterActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterActivationSequence> {
    let mut store = call_tile!(init_prefill_layer_store);
    let state = call_tile!(
        init_prefill_sequence_projection,
        &mut store,
        input,
        projection_rows,
        projection_rows_per_tile
    )?;
    let state = call_recur_tile!(
        project_next_prefill_sequence_rows,
        state,
        layer_source,
        layer_idx,
        matrix,
        &mut store
    )?;
    call_tile!(finalize_prefill_sequence_projection, state, &mut store)
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
) -> Result<Vec<crate::shared::det_num::Wgt>> {
    auth_read!(
        layer_source,
        GemmaPrefillLayerNormWeightsRequest { layer_idx, norm }
    )
}

#[tile]
pub fn resolve_prefill_value_projection(
    layer: &GemmaPrefillLayerMetadata,
    key_projection: &RasterActivationSequence,
    value_projection: Option<RasterActivationSequence>,
) -> Result<RasterActivationSequence> {
    if layer.has_v_proj {
        value_projection
            .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing v_proj shape"))
    } else if layer.attention_k_eq_v {
        Ok(key_projection.clone())
    } else {
        bail!("Gemma prefill layer is missing v_proj without attention_k_eq_v enabled");
    }
}

#[tile]
pub fn empty_prefill_layer_cache(num_kv_heads: usize) -> RasterKvCache {
    RasterKvCache::empty(num_kv_heads)
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
pub fn init_prefill_attention_store() -> AuthenticatedRasterTensorStore {
    AuthenticatedRasterTensorStore::new()
}

#[tile]
pub fn init_prefill_attention_state(
    store: &mut AuthenticatedRasterTensorStore,
    queries: &RasterAttentionHeadSequence,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    donor_cache: Option<&RasterKvCache>,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<RasterAttentionRowState> {
    init_attention_row_state(
        store,
        queries,
        keys,
        values,
        donor_cache,
        attention_window,
        kv_rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_attention_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    output_id: RasterTensorId,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<RasterAttentionRowState> {
    init_attention_row_state_from_refs(
        store,
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
pub fn project_next_prefill_attention_row(
    state: RasterAttentionRowState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterAttentionRowState)> {
    compute_next_attention_row(state, store)
}

#[tile]
pub fn finalize_prefill_attention_state(
    store: &mut AuthenticatedRasterTensorStore,
    state: RasterAttentionRowState,
) -> Result<RasterAttentionHeadSequence> {
    finalize_attention_row_state(state, store)
}

#[tile]
pub fn finalize_prefill_attention_state_ref(
    store: &mut AuthenticatedRasterTensorStore,
    state: RasterAttentionRowState,
) -> Result<RasterAttentionHeadsRef> {
    finalize_attention_row_state_ref(state, store)
}

#[sequence]
pub fn run_prefill_attention_rows(
    queries: &RasterAttentionHeadSequence,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    donor_cache: Option<&RasterKvCache>,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<RasterAttentionHeadSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_attention_state,
        &mut store,
        queries,
        keys,
        values,
        donor_cache,
        attention_window,
        kv_rows_per_tile
    )?;
    let state = call_recur_tile!(project_next_prefill_attention_row, state, &mut store)?;
    call_tile!(finalize_prefill_attention_state, &mut store, state)
}

#[tile]
pub fn init_prefill_sequence_rms_norm_state(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_sequence_rms_norm_row_state(store, input, norm_weights, eps, rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_rms_norm_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_sequence_rms_norm_row_state_from_ref(
        store,
        input_ref,
        output_id,
        norm_weights,
        eps,
        rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_sequence_gelu_state(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_sequence_gelu_row_state(store, input, rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_gelu_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_sequence_gelu_row_state_from_ref(store, input_ref, output_id, rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_scale_state(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    scalar: Option<crate::shared::det_num::Act>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_sequence_scale_row_state(store, input, scalar, rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_scale_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    scalar: Option<crate::shared::det_num::Act>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_sequence_scale_row_state_from_ref(store, input_ref, output_id, scalar, rows_per_tile)
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_sequence_unary_row(
    state: RasterSequenceUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterSequenceUnaryState)> {
    compute_next_sequence_unary_row(state, store)
}

#[tile]
pub fn finalize_prefill_sequence_unary_state(
    state: RasterSequenceUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    finalize_sequence_unary_row_state(state, store)
}

#[tile]
pub fn finalize_prefill_sequence_unary_state_ref(
    state: RasterSequenceUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    finalize_sequence_unary_row_state_ref(state, store)
}

#[sequence]
pub fn run_prefill_sequence_rms_norm(
    input: &RasterActivationSequence,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
) -> Result<RasterActivationSequence> {
    run_prefill_sequence_rms_norm_with_rows_per_tile(input, norm_weights, eps, 1)
}

#[sequence]
pub fn run_prefill_sequence_rms_norm_with_rows_per_tile(
    input: &RasterActivationSequence,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterActivationSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_sequence_rms_norm_state,
        &mut store,
        input,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_unary_row, state, &mut store)?;
    call_tile!(finalize_prefill_sequence_unary_state, state, &mut store)
}

#[sequence]
pub fn run_prefill_sequence_gelu(
    input: &RasterActivationSequence,
) -> Result<RasterActivationSequence> {
    run_prefill_sequence_gelu_with_rows_per_tile(input, 1)
}

#[sequence]
pub fn run_prefill_sequence_gelu_with_rows_per_tile(
    input: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterActivationSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_sequence_gelu_state,
        &mut store,
        input,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_unary_row, state, &mut store)?;
    call_tile!(finalize_prefill_sequence_unary_state, state, &mut store)
}

#[sequence]
pub fn run_prefill_sequence_scale(
    input: &RasterActivationSequence,
    scalar: Option<crate::shared::det_num::Act>,
) -> Result<RasterActivationSequence> {
    run_prefill_sequence_scale_with_rows_per_tile(input, scalar, 1)
}

#[sequence]
pub fn run_prefill_sequence_scale_with_rows_per_tile(
    input: &RasterActivationSequence,
    scalar: Option<crate::shared::det_num::Act>,
    rows_per_tile: usize,
) -> Result<RasterActivationSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_sequence_scale_state,
        &mut store,
        input,
        scalar,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_unary_row, state, &mut store)?;
    call_tile!(finalize_prefill_sequence_unary_state, state, &mut store)
}

#[tile]
pub fn init_prefill_sequence_add_state(
    store: &mut AuthenticatedRasterTensorStore,
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_add_row_state(store, lhs, rhs, rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_add_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_add_row_state_from_refs(store, lhs_ref, rhs_ref, output_id, rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_mul_state(
    store: &mut AuthenticatedRasterTensorStore,
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_mul_row_state(store, lhs, rhs, rows_per_tile)
}

#[tile]
pub fn init_prefill_sequence_mul_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_sequence_mul_row_state_from_refs(store, lhs_ref, rhs_ref, output_id, rows_per_tile)
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_sequence_binary_row(
    state: RasterSequenceBinaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterSequenceBinaryState)> {
    compute_next_sequence_binary_row(state, store)
}

#[tile]
pub fn finalize_prefill_sequence_binary_state(
    state: RasterSequenceBinaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    finalize_sequence_binary_row_state(state, store)
}

#[tile]
pub fn finalize_prefill_sequence_binary_state_ref(
    state: RasterSequenceBinaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    finalize_sequence_binary_row_state_ref(state, store)
}

#[sequence]
pub fn run_prefill_sequence_add(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
) -> Result<RasterActivationSequence> {
    run_prefill_sequence_add_with_rows_per_tile(lhs, rhs, 1)
}

#[sequence]
pub fn run_prefill_sequence_add_with_rows_per_tile(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterActivationSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_sequence_add_state,
        &mut store,
        lhs,
        rhs,
        rows_per_tile
    )?;
    let state = call_recur_tile!(
        transform_next_prefill_sequence_binary_row,
        state,
        &mut store
    )?;
    call_tile!(finalize_prefill_sequence_binary_state, state, &mut store)
}

#[sequence]
pub fn run_prefill_sequence_mul(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
) -> Result<RasterActivationSequence> {
    run_prefill_sequence_mul_with_rows_per_tile(lhs, rhs, 1)
}

#[sequence]
pub fn run_prefill_sequence_mul_with_rows_per_tile(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
    rows_per_tile: usize,
) -> Result<RasterActivationSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_sequence_mul_state,
        &mut store,
        lhs,
        rhs,
        rows_per_tile
    )?;
    let state = call_recur_tile!(
        transform_next_prefill_sequence_binary_row,
        state,
        &mut store
    )?;
    call_tile!(finalize_prefill_sequence_binary_state, state, &mut store)
}

#[tile]
pub fn init_prefill_head_rms_norm_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    init_head_rms_norm_row_state(store, heads, norm_weights, eps, rows_per_tile)
}

#[tile]
pub fn init_prefill_head_rms_norm_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    init_head_rms_norm_row_state_from_ref(
        store,
        heads_ref,
        output_id,
        norm_weights,
        eps,
        rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_value_rms_norm_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    init_value_rms_norm_row_state(store, heads, eps, rows_per_tile)
}

#[tile]
pub fn init_prefill_value_rms_norm_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    init_value_rms_norm_row_state_from_ref(store, heads_ref, output_id, eps, rows_per_tile)
}

#[tile]
pub fn init_prefill_rope_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<crate::shared::det_num::Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    init_rope_row_state(
        store,
        heads,
        rotary_dim,
        freq_base_dim,
        base,
        position_offset,
        rows_per_tile,
    )
}

#[tile]
pub fn init_prefill_rope_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<crate::shared::det_num::Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<RasterHeadUnaryState> {
    init_rope_row_state_from_ref(
        store,
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
pub fn transform_next_prefill_head_row(
    state: RasterHeadUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterHeadUnaryState)> {
    compute_next_head_unary_row(state, store)
}

#[tile]
pub fn finalize_prefill_head_state(
    state: RasterHeadUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadSequence> {
    finalize_head_unary_row_state(state, store)
}

#[tile]
pub fn finalize_prefill_head_state_ref(
    state: RasterHeadUnaryState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadsRef> {
    finalize_head_unary_row_state_ref(state, store)
}

#[sequence]
pub fn run_prefill_head_rms_norm(
    heads: &RasterAttentionHeadSequence,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
) -> Result<RasterAttentionHeadSequence> {
    run_prefill_head_rms_norm_with_rows_per_tile(heads, norm_weights, eps, 1)
}

#[sequence]
pub fn run_prefill_head_rms_norm_with_rows_per_tile(
    heads: &RasterAttentionHeadSequence,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterAttentionHeadSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_head_rms_norm_state,
        &mut store,
        heads,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_head_row, state, &mut store)?;
    call_tile!(finalize_prefill_head_state, state, &mut store)
}

#[sequence]
pub fn run_prefill_value_rms_norm(
    heads: &RasterAttentionHeadSequence,
    eps: Option<crate::shared::det_num::Acc>,
) -> Result<RasterAttentionHeadSequence> {
    run_prefill_value_rms_norm_with_rows_per_tile(heads, eps, 1)
}

#[sequence]
pub fn run_prefill_value_rms_norm_with_rows_per_tile(
    heads: &RasterAttentionHeadSequence,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterAttentionHeadSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_value_rms_norm_state,
        &mut store,
        heads,
        eps,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_head_row, state, &mut store)?;
    call_tile!(finalize_prefill_head_state, state, &mut store)
}

#[sequence]
pub fn run_prefill_rope_heads(
    heads: &RasterAttentionHeadSequence,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<crate::shared::det_num::Acc>,
    position_offset: usize,
) -> Result<RasterAttentionHeadSequence> {
    run_prefill_rope_heads_with_rows_per_tile(
        heads,
        rotary_dim,
        freq_base_dim,
        base,
        position_offset,
        1,
    )
}

#[sequence]
pub fn run_prefill_rope_heads_with_rows_per_tile(
    heads: &RasterAttentionHeadSequence,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<crate::shared::det_num::Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<RasterAttentionHeadSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_rope_state,
        &mut store,
        heads,
        rotary_dim,
        freq_base_dim,
        base,
        position_offset,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_head_row, state, &mut store)?;
    call_tile!(finalize_prefill_head_state, state, &mut store)
}

#[tile]
pub fn init_prefill_reshape_heads_state(
    store: &mut AuthenticatedRasterTensorStore,
    input: &RasterActivationSequence,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterReshapeHeadsState> {
    init_reshape_heads_state(store, input, num_heads, head_dim)
}

#[tile]
pub fn init_prefill_reshape_heads_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterReshapeHeadsState> {
    init_reshape_heads_state_from_ref(store, input_ref, output_id, num_heads, head_dim)
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_reshape_row(
    state: RasterReshapeHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterReshapeHeadsState)> {
    compute_next_reshape_heads_row(state, store)
}

#[tile]
pub fn finalize_prefill_reshape_heads_state(
    state: RasterReshapeHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadSequence> {
    finalize_reshape_heads_state(state, store)
}

#[tile]
pub fn finalize_prefill_reshape_heads_state_ref(
    state: RasterReshapeHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterAttentionHeadsRef> {
    finalize_reshape_heads_state_ref(state, store)
}

#[sequence]
pub fn run_prefill_reshape_heads(
    input: &RasterActivationSequence,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterAttentionHeadSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_reshape_heads_state,
        &mut store,
        input,
        num_heads,
        head_dim
    )?;
    let state = call_recur_tile!(transform_next_prefill_reshape_row, state, &mut store)?;
    call_tile!(finalize_prefill_reshape_heads_state, state, &mut store)
}

#[tile]
pub fn init_prefill_combine_heads_state(
    store: &mut AuthenticatedRasterTensorStore,
    heads: &RasterAttentionHeadSequence,
) -> Result<RasterCombineHeadsState> {
    init_combine_heads_state(store, heads)
}

#[tile]
pub fn init_prefill_combine_heads_state_from_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    output_id: RasterTensorId,
) -> Result<RasterCombineHeadsState> {
    init_combine_heads_state_from_ref(store, heads_ref, output_id)
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_combine_row(
    state: RasterCombineHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterCombineHeadsState)> {
    compute_next_combine_heads_row(state, store)
}

#[tile]
pub fn finalize_prefill_combine_heads_state(
    state: RasterCombineHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequence> {
    finalize_combine_heads_state(state, store)
}

#[tile]
pub fn finalize_prefill_combine_heads_state_ref(
    state: RasterCombineHeadsState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterActivationSequenceRef> {
    finalize_combine_heads_state_ref(state, store)
}

#[sequence]
pub fn run_prefill_combine_heads(
    heads: &RasterAttentionHeadSequence,
) -> Result<RasterActivationSequence> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(init_prefill_combine_heads_state, &mut store, heads)?;
    let state = call_recur_tile!(transform_next_prefill_combine_row, state, &mut store)?;
    call_tile!(finalize_prefill_combine_heads_state, state, &mut store)
}

#[tile]
pub fn init_prefill_kv_cache_state(
    store: &mut AuthenticatedRasterTensorStore,
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    sliding_window: Option<usize>,
) -> Result<RasterKvCacheBuildState> {
    init_kv_cache_build_state(store, keys, values, sliding_window)
}

#[tile]
pub fn init_prefill_kv_cache_state_from_refs(
    store: &mut AuthenticatedRasterTensorStore,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    sliding_window: Option<usize>,
) -> Result<RasterKvCacheBuildState> {
    init_kv_cache_build_state_from_refs(
        store,
        key_ref,
        value_ref,
        keys_id,
        values_id,
        sliding_window,
    )
}

#[tile(kind = recursive)]
pub fn transform_next_prefill_kv_cache_row(
    state: RasterKvCacheBuildState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, RasterKvCacheBuildState)> {
    compute_next_kv_cache_row(state, store)
}

#[tile]
pub fn finalize_prefill_kv_cache_state(
    state: RasterKvCacheBuildState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterKvCache> {
    finalize_kv_cache_build_state(state, store)
}

#[tile]
pub fn finalize_prefill_kv_cache_state_ref(
    state: RasterKvCacheBuildState,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<RasterKvCacheRef> {
    finalize_kv_cache_build_state_ref(state, store)
}

#[sequence]
pub fn run_prefill_kv_cache(
    keys: &RasterAttentionHeadSequence,
    values: &RasterAttentionHeadSequence,
    sliding_window: Option<usize>,
) -> Result<RasterKvCache> {
    let mut store = call_tile!(init_prefill_attention_store);
    let state = call_tile!(
        init_prefill_kv_cache_state,
        &mut store,
        keys,
        values,
        sliding_window
    )?;
    let state = call_recur_tile!(transform_next_prefill_kv_cache_row, state, &mut store)?;
    call_tile!(finalize_prefill_kv_cache_state, state, &mut store)
}

#[sequence]
pub fn run_prefill_layer_sequence_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &GemmaPrefillLayerMetadata,
    donor_cache: Option<&PrefillLayerCacheSlot>,
    per_layer_input_ref: Option<RasterActivationSequenceRef>,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
    head_rows_per_tile: usize,
) -> Result<(RasterActivationSequenceRef, PrefillLayerCacheSlot)> {
    let scalars = call_tile!(read_prefill_layer_scalars, layer_source, layer.layer_idx)?;
    let input_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::InputLayer
    )?;
    let normed_ref = call_seq!(
        compute_sequence_rms_norm_ref,
        store,
        input_ref.clone(),
        format!("prefill.layer.{}.attention.input_norm", layer.layer_idx),
        Some(&input_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    let q_projected_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
        normed_ref.clone(),
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Query,
        layer.q_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.q_proj", layer.layer_idx)
    )?;
    let k_projected_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
        normed_ref.clone(),
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Key,
        layer.k_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.k_proj", layer.layer_idx)
    )?;
    let v_projected_ref = if layer.has_v_proj {
        call_seq!(
            project_sequence_with_prefill_source_ref,
            store,
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
        k_projected_ref.clone()
    } else {
        bail!("Gemma prefill layer is missing v_proj without attention_k_eq_v enabled");
    };

    let q_heads_ref = call_seq!(
        reshape_heads_ref,
        store,
        q_projected_ref,
        format!("prefill.layer.{}.q_heads", layer.layer_idx),
        layer.num_heads,
        layer.head_dim
    )?;
    let k_heads_ref = call_seq!(
        reshape_heads_ref,
        store,
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
    let q_heads_ref = call_seq!(
        compute_head_rms_norm_ref,
        store,
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
    let k_heads_ref = call_seq!(
        compute_head_rms_norm_ref,
        store,
        k_heads_ref,
        format!("prefill.layer.{}.k_norm", layer.layer_idx),
        Some(&k_norm_weights),
        Some(scalars.rms_norm_eps),
        head_rows_per_tile
    )?;
    let v_heads_ref = call_seq!(
        reshape_heads_ref,
        store,
        v_projected_ref,
        format!("prefill.layer.{}.v_heads", layer.layer_idx),
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let v_heads_ref = call_seq!(
        compute_value_rms_norm_ref,
        store,
        v_heads_ref,
        format!("prefill.layer.{}.v_norm", layer.layer_idx),
        Some(scalars.rms_norm_eps),
        head_rows_per_tile
    )?;
    let q_heads_ref = call_seq!(
        compute_rope_ref,
        store,
        q_heads_ref,
        format!("prefill.layer.{}.q_rope", layer.layer_idx),
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        0,
        head_rows_per_tile
    )?;
    let k_heads_ref = call_seq!(
        compute_rope_ref,
        store,
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
    let layer_cache = if donor_cache.is_some() || retained_cache_len == 0 {
        PrefillLayerCacheSlot::Empty {
            num_kv_heads: layer.num_kv_heads,
        }
    } else {
        PrefillLayerCacheSlot::Ref(call_seq!(
            build_kv_cache_ref,
            store,
            k_heads_ref.clone(),
            v_heads_ref.clone(),
            format!("prefill.layer.cache.{}", layer.layer_idx),
            layer.cache_sliding_window
        )?)
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
    let attention_heads_ref = call_seq!(
        compute_attention_ref,
        store,
        q_heads_ref,
        k_heads_ref,
        v_heads_ref,
        donor_cache_ref,
        format!("prefill.layer.{}.attention", layer.layer_idx),
        attention_window,
        attention_kv_rows_per_tile
    )?;
    let attention_sequence_ref = call_seq!(
        combine_heads_ref,
        store,
        attention_heads_ref,
        format!("prefill.layer.{}.attention.combine", layer.layer_idx)
    )?;
    let attention_output_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
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
    let attention_output_ref = call_seq!(
        compute_sequence_rms_norm_ref,
        store,
        attention_output_ref,
        format!("prefill.layer.{}.attention.post_norm", layer.layer_idx),
        Some(&post_attention_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    let xs_ref = call_seq!(
        compute_sequence_add_ref,
        store,
        input_ref,
        attention_output_ref,
        format!("prefill.layer.{}.attention.residual", layer.layer_idx),
        sequence_rows_per_tile
    )?;

    let xs_ref = call_seq!(
        run_prefill_mlp_block_ref,
        store,
        xs_ref,
        layer_source,
        layer,
        &scalars,
        projection_rows_per_tile,
        sequence_rows_per_tile
    )?;
    let mut xs_ref = if let Some(per_layer_input_ref) = per_layer_input_ref {
        call_seq!(
            run_prefill_ple_block_ref,
            store,
            xs_ref,
            per_layer_input_ref,
            layer_source,
            layer,
            &scalars,
            projection_rows_per_tile,
            sequence_rows_per_tile
        )?
    } else {
        xs_ref
    };
    if scalars.layer_scalar.is_some() {
        xs_ref = call_seq!(
            compute_sequence_scale_ref,
            store,
            xs_ref,
            format!("prefill.layer.{}.layer_scalar", layer.layer_idx),
            scalars.layer_scalar,
            sequence_rows_per_tile
        )?;
    }

    Ok((xs_ref, layer_cache))
}

#[sequence]
pub fn run_prefill_layer_sequence(
    input: &RasterActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &GemmaPrefillLayerMetadata,
    donor_cache: Option<&RasterKvCache>,
    per_layer_input: Option<&RasterActivationSequence>,
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
    head_rows_per_tile: usize,
) -> Result<(RasterActivationSequence, RasterKvCache)> {
    let scalars = call_tile!(read_prefill_layer_scalars, layer_source, layer.layer_idx)?;
    let input_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::InputLayer
    )?;
    let normed = call_seq!(
        run_prefill_sequence_rms_norm_with_rows_per_tile,
        input,
        Some(&input_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile,
    )?;

    let q_projected = call_seq!(
        project_sequence_with_prefill_source,
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Query,
        layer.q_proj_shape.rows,
        projection_rows_per_tile,
    )?;
    let k_projected = call_seq!(
        project_sequence_with_prefill_source,
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Key,
        layer.k_proj_shape.rows,
        projection_rows_per_tile,
    )?;
    let v_projected = if layer.has_v_proj {
        Some(call_seq!(
            project_sequence_with_prefill_source,
            &normed,
            layer_source,
            layer.layer_idx,
            GemmaPrefillLayerMatrixKind::Value,
            layer
                .v_proj_shape
                .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing v_proj shape"))?
                .rows,
            projection_rows_per_tile,
        )?)
    } else {
        None
    };
    let v_projected = call_tile!(
        resolve_prefill_value_projection,
        layer,
        &k_projected,
        v_projected
    )?;

    let q_heads = call_seq!(
        run_prefill_reshape_heads,
        &q_projected,
        layer.num_heads,
        layer.head_dim
    )?;
    let k_heads = call_seq!(
        run_prefill_reshape_heads,
        &k_projected,
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let q_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::Query
    )?;
    let q_heads = call_seq!(
        run_prefill_head_rms_norm_with_rows_per_tile,
        &q_heads,
        Some(&q_norm_weights),
        Some(scalars.rms_norm_eps),
        head_rows_per_tile,
    )?;
    let k_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::Key
    )?;
    let k_heads = call_seq!(
        run_prefill_head_rms_norm_with_rows_per_tile,
        &k_heads,
        Some(&k_norm_weights),
        Some(scalars.rms_norm_eps),
        head_rows_per_tile,
    )?;
    let v_heads = call_seq!(
        run_prefill_reshape_heads,
        &v_projected,
        layer.num_kv_heads,
        layer.head_dim
    )?;
    let v_heads = call_seq!(
        run_prefill_value_rms_norm_with_rows_per_tile,
        &v_heads,
        Some(scalars.rms_norm_eps),
        head_rows_per_tile
    )?;
    let q_heads = call_seq!(
        run_prefill_rope_heads_with_rows_per_tile,
        &q_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        0,
        head_rows_per_tile,
    )?;
    let k_heads = call_seq!(
        run_prefill_rope_heads_with_rows_per_tile,
        &k_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        0,
        head_rows_per_tile,
    )?;

    let layer_cache = if donor_cache.is_some() {
        call_tile!(empty_prefill_layer_cache, layer.num_kv_heads)
    } else {
        call_seq!(
            run_prefill_kv_cache,
            &k_heads,
            &v_heads,
            layer.cache_sliding_window
        )?
    };
    let attention_window = call_tile!(resolve_prefill_attention_window, layer)?;
    let attention_heads = call_seq!(
        run_prefill_attention_rows,
        &q_heads,
        &k_heads,
        &v_heads,
        donor_cache,
        attention_window,
        attention_kv_rows_per_tile,
    )?;
    let attention_sequence = call_seq!(run_prefill_combine_heads, &attention_heads)?;
    let attention_output = call_seq!(
        project_sequence_with_prefill_source,
        &attention_sequence,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Output,
        layer.o_proj_shape.rows,
        projection_rows_per_tile,
    )?;
    let post_attention_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PostAttention
    )?;
    let attention_output = call_seq!(
        run_prefill_sequence_rms_norm_with_rows_per_tile,
        &attention_output,
        Some(&post_attention_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile,
    )?;
    let xs = call_seq!(
        run_prefill_sequence_add_with_rows_per_tile,
        input,
        &attention_output,
        sequence_rows_per_tile
    )?;

    let pre_feedforward_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PreFeedForward
    )?;
    let normed = call_seq!(
        run_prefill_sequence_rms_norm_with_rows_per_tile,
        &xs,
        Some(&pre_feedforward_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile,
    )?;
    let gate = call_seq!(
        project_sequence_with_prefill_source,
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Gate,
        layer.gate_proj_shape.rows,
        projection_rows_per_tile,
    )?;
    let gate = call_seq!(
        run_prefill_sequence_gelu_with_rows_per_tile,
        &gate,
        sequence_rows_per_tile
    )?;
    let up = call_seq!(
        project_sequence_with_prefill_source,
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Up,
        layer.up_proj_shape.rows,
        projection_rows_per_tile,
    )?;
    let ff_hidden = call_seq!(
        run_prefill_sequence_mul_with_rows_per_tile,
        &gate,
        &up,
        sequence_rows_per_tile
    )?;
    let ff_out = call_seq!(
        project_sequence_with_prefill_source,
        &ff_hidden,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Down,
        layer.down_proj_shape.rows,
        projection_rows_per_tile,
    )?;
    let post_feedforward_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PostFeedForward
    )?;
    let ff_out = call_seq!(
        run_prefill_sequence_rms_norm_with_rows_per_tile,
        &ff_out,
        Some(&post_feedforward_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile,
    )?;
    let mut xs = call_seq!(
        run_prefill_sequence_add_with_rows_per_tile,
        &xs,
        &ff_out,
        sequence_rows_per_tile
    )?;

    if let Some(per_layer_input) = per_layer_input {
        let gated = call_seq!(
            project_sequence_with_prefill_source,
            &xs,
            layer_source,
            layer.layer_idx,
            GemmaPrefillLayerMatrixKind::PleInputGate,
            layer
                .ple_input_gate_shape
                .ok_or_else(|| {
                    anyhow!("Gemma prefill layer metadata is missing PLE input gate shape")
                })?
                .rows,
            projection_rows_per_tile,
        )?;
        let gated = call_seq!(
            run_prefill_sequence_gelu_with_rows_per_tile,
            &gated,
            sequence_rows_per_tile
        )?;
        let gated = call_seq!(
            run_prefill_sequence_mul_with_rows_per_tile,
            &gated,
            per_layer_input,
            sequence_rows_per_tile
        )?;
        let projected = call_seq!(
            project_sequence_with_prefill_source,
            &gated,
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
        )?;
        let ple_post_input_norm_weights = call_tile!(
            read_prefill_layer_norm_weights,
            layer_source,
            layer.layer_idx,
            GemmaPrefillLayerNormKind::PlePostInput
        )?;
        let projected = call_seq!(
            run_prefill_sequence_rms_norm_with_rows_per_tile,
            &projected,
            Some(&ple_post_input_norm_weights),
            Some(scalars.rms_norm_eps),
            sequence_rows_per_tile,
        )?;
        xs = call_seq!(
            run_prefill_sequence_add_with_rows_per_tile,
            &xs,
            &projected,
            sequence_rows_per_tile
        )?;
    }

    if scalars.layer_scalar.is_some() {
        xs = call_seq!(
            run_prefill_sequence_scale_with_rows_per_tile,
            &xs,
            scalars.layer_scalar,
            sequence_rows_per_tile
        )?;
    }

    Ok((xs, layer_cache))
}

#[sequence]
fn project_sequence_with_prefill_source_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer_idx: usize,
    matrix: GemmaPrefillLayerMatrixKind,
    projection_rows: usize,
    rows_per_tile: usize,
    id_prefix: String,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_prefill_sequence_projection_from_ref,
        store,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        projection_rows,
        rows_per_tile
    )?;
    let state = call_recur_tile!(
        project_next_prefill_sequence_rows,
        state,
        layer_source,
        layer_idx,
        matrix,
        store
    )?;
    call_tile!(finalize_prefill_sequence_projection_ref, state, store)
}

#[sequence]
fn compute_sequence_rms_norm_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_prefill_sequence_rms_norm_state_from_ref,
        store,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_unary_row, state, store)?;
    call_tile!(finalize_prefill_sequence_unary_state_ref, state, store)
}

#[sequence]
fn compute_sequence_gelu_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_prefill_sequence_gelu_state_from_ref,
        store,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_unary_row, state, store)?;
    call_tile!(finalize_prefill_sequence_unary_state_ref, state, store)
}

#[sequence]
fn compute_sequence_scale_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    scalar: Option<crate::shared::det_num::Act>,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_prefill_sequence_scale_state_from_ref,
        store,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        scalar,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_unary_row, state, store)?;
    call_tile!(finalize_prefill_sequence_unary_state_ref, state, store)
}

#[sequence]
fn compute_sequence_add_ref(
    store: &mut AuthenticatedRasterTensorStore,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    id_prefix: String,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_prefill_sequence_add_state_from_refs,
        store,
        lhs_ref,
        rhs_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_binary_row, state, store)?;
    call_tile!(finalize_prefill_sequence_binary_state_ref, state, store)
}

#[sequence]
fn compute_sequence_mul_ref(
    store: &mut AuthenticatedRasterTensorStore,
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    id_prefix: String,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_prefill_sequence_mul_state_from_refs,
        store,
        lhs_ref,
        rhs_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_sequence_binary_row, state, store)?;
    call_tile!(finalize_prefill_sequence_binary_state_ref, state, store)
}

#[sequence]
fn reshape_heads_ref(
    store: &mut AuthenticatedRasterTensorStore,
    input_ref: RasterActivationSequenceRef,
    id_prefix: String,
    num_heads: usize,
    head_dim: usize,
) -> Result<RasterAttentionHeadsRef> {
    let state = call_tile!(
        init_prefill_reshape_heads_state_from_ref,
        store,
        input_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        num_heads,
        head_dim
    )?;
    let state = call_recur_tile!(transform_next_prefill_reshape_row, state, store)?;
    call_tile!(finalize_prefill_reshape_heads_state_ref, state, store)
}

#[sequence]
fn compute_head_rms_norm_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterAttentionHeadsRef> {
    let state = call_tile!(
        init_prefill_head_rms_norm_state_from_ref,
        store,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_head_row, state, store)?;
    call_tile!(finalize_prefill_head_state_ref, state, store)
}

#[sequence]
fn compute_value_rms_norm_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterAttentionHeadsRef> {
    let state = call_tile!(
        init_prefill_value_rms_norm_state_from_ref,
        store,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        eps,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_head_row, state, store)?;
    call_tile!(finalize_prefill_head_state_ref, state, store)
}

#[sequence]
fn compute_rope_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    rotary_dim: usize,
    freq_base_dim: usize,
    base: Option<crate::shared::det_num::Acc>,
    position_offset: usize,
    rows_per_tile: usize,
) -> Result<RasterAttentionHeadsRef> {
    let state = call_tile!(
        init_prefill_rope_state_from_ref,
        store,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        rotary_dim,
        freq_base_dim,
        base,
        position_offset,
        rows_per_tile
    )?;
    let state = call_recur_tile!(transform_next_prefill_head_row, state, store)?;
    call_tile!(finalize_prefill_head_state_ref, state, store)
}

#[sequence]
fn build_kv_cache_ref(
    store: &mut AuthenticatedRasterTensorStore,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    id_prefix: String,
    sliding_window: Option<usize>,
) -> Result<RasterKvCacheRef> {
    let state = call_tile!(
        init_prefill_kv_cache_state_from_refs,
        store,
        key_ref,
        value_ref,
        RasterTensorId::new(format!("{id_prefix}.keys"))?,
        RasterTensorId::new(format!("{id_prefix}.values"))?,
        sliding_window
    )?;
    let state = call_recur_tile!(transform_next_prefill_kv_cache_row, state, store)?;
    call_tile!(finalize_prefill_kv_cache_state_ref, state, store)
}

#[sequence]
fn compute_attention_ref(
    store: &mut AuthenticatedRasterTensorStore,
    query_ref: RasterAttentionHeadsRef,
    key_ref: RasterAttentionHeadsRef,
    value_ref: RasterAttentionHeadsRef,
    donor_cache_ref: Option<RasterKvCacheRef>,
    id_prefix: String,
    attention_window: Option<usize>,
    kv_rows_per_tile: usize,
) -> Result<RasterAttentionHeadsRef> {
    let state = call_tile!(
        init_prefill_attention_state_from_refs,
        store,
        query_ref,
        key_ref,
        value_ref,
        donor_cache_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?,
        attention_window,
        kv_rows_per_tile
    )?;
    let state = call_recur_tile!(project_next_prefill_attention_row, state, store)?;
    call_tile!(finalize_prefill_attention_state_ref, store, state)
}

#[sequence]
fn combine_heads_ref(
    store: &mut AuthenticatedRasterTensorStore,
    heads_ref: RasterAttentionHeadsRef,
    id_prefix: String,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_prefill_combine_heads_state_from_ref,
        store,
        heads_ref,
        RasterTensorId::new(format!("{id_prefix}.output"))?
    )?;
    let state = call_recur_tile!(transform_next_prefill_combine_row, state, store)?;
    call_tile!(finalize_prefill_combine_heads_state_ref, state, store)
}

#[sequence]
fn run_prefill_mlp_block_ref(
    store: &mut AuthenticatedRasterTensorStore,
    xs_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &GemmaPrefillLayerMetadata,
    scalars: &GemmaPrefillLayerScalars,
    projection_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let pre_feedforward_norm_weights = call_tile!(
        read_prefill_layer_norm_weights,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerNormKind::PreFeedForward
    )?;
    let normed_ref = call_seq!(
        compute_sequence_rms_norm_ref,
        store,
        xs_ref.clone(),
        format!("prefill.layer.{}.mlp.pre_norm", layer.layer_idx),
        Some(&pre_feedforward_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    let gate_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
        normed_ref.clone(),
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Gate,
        layer.gate_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.gate_proj", layer.layer_idx)
    )?;
    let gate_ref = call_seq!(
        compute_sequence_gelu_ref,
        store,
        gate_ref,
        format!("prefill.layer.{}.gate_gelu", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let up_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
        normed_ref,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Up,
        layer.up_proj_shape.rows,
        projection_rows_per_tile,
        format!("prefill.layer.{}.up_proj", layer.layer_idx)
    )?;
    let ff_hidden_ref = call_seq!(
        compute_sequence_mul_ref,
        store,
        gate_ref,
        up_ref,
        format!("prefill.layer.{}.ff_hidden", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let ff_out_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
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
    let ff_out_ref = call_seq!(
        compute_sequence_rms_norm_ref,
        store,
        ff_out_ref,
        format!("prefill.layer.{}.ff_post_norm", layer.layer_idx),
        Some(&post_feedforward_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    call_seq!(
        compute_sequence_add_ref,
        store,
        xs_ref,
        ff_out_ref,
        format!("prefill.layer.{}.mlp.residual", layer.layer_idx),
        sequence_rows_per_tile
    )
}

#[sequence]
fn run_prefill_ple_block_ref(
    store: &mut AuthenticatedRasterTensorStore,
    xs_ref: RasterActivationSequenceRef,
    per_layer_input_ref: RasterActivationSequenceRef,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &GemmaPrefillLayerMetadata,
    scalars: &GemmaPrefillLayerScalars,
    projection_rows_per_tile: usize,
    sequence_rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let gated_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
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
    let gated_ref = call_seq!(
        compute_sequence_gelu_ref,
        store,
        gated_ref,
        format!("prefill.layer.{}.ple_gate_gelu", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let gated_ref = call_seq!(
        compute_sequence_mul_ref,
        store,
        gated_ref,
        per_layer_input_ref,
        format!("prefill.layer.{}.ple_input_mul", layer.layer_idx),
        sequence_rows_per_tile
    )?;
    let projected_ref = call_seq!(
        project_sequence_with_prefill_source_ref,
        store,
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
    let projected_ref = call_seq!(
        compute_sequence_rms_norm_ref,
        store,
        projected_ref,
        format!("prefill.layer.{}.ple_post_norm", layer.layer_idx),
        Some(&ple_post_input_norm_weights),
        Some(scalars.rms_norm_eps),
        sequence_rows_per_tile
    )?;
    call_seq!(
        compute_sequence_add_ref,
        store,
        xs_ref,
        projected_ref,
        format!("prefill.layer.{}.ple_residual", layer.layer_idx),
        sequence_rows_per_tile
    )
}

fn retained_prefill_kv_cache_len(
    key_ref: &RasterAttentionHeadsRef,
    sliding_window: Option<usize>,
) -> Result<usize> {
    let (_, sequence_len, _) = key_ref.tensor_ref().shape().heads_metadata()?;
    Ok(sliding_window.map_or(sequence_len, |window| window.min(sequence_len)))
}

fn resolve_prefill_donor_cache_index(
    layer_caches: &[PrefillLayerCacheSlot],
    layer_idx: usize,
    layer: &GemmaPrefillLayerMetadata,
) -> Result<Option<usize>> {
    layer
        .kv_shared_layer_index
        .map(|donor_idx| {
            if donor_idx >= layer_idx {
                bail!(
                    "transformer prefill layer {layer_idx} cannot share KV with non-prior donor {donor_idx}"
                );
            }
            layer_caches.get(donor_idx).ok_or_else(|| {
                anyhow!("transformer prefill donor cache {donor_idx} missing for layer {layer_idx}")
            })?;
            Ok(donor_idx)
        })
        .transpose()
}

fn register_prefill_layer_cache(
    store: &mut AuthenticatedRasterTensorStore,
    layer_idx: usize,
    cache: RasterKvCache,
) -> Result<PrefillLayerCacheSlot> {
    if cache.current_len() == 0 {
        return Ok(PrefillLayerCacheSlot::Empty {
            num_kv_heads: cache.head_count(),
        });
    }

    Ok(PrefillLayerCacheSlot::Ref(store.insert_kv_cache(
        RasterTensorId::new(format!("prefill.layer.cache.{layer_idx}.keys"))?,
        RasterTensorId::new(format!("prefill.layer.cache.{layer_idx}.values"))?,
        cache,
    )?))
}

fn import_materialized_ple_inputs(
    store: &mut AuthenticatedRasterTensorStore,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    // Host/dev compatibility bridge only. This deliberately lives outside any
    // authored tile or sequence so proof-shaped code cannot accidentally ingest
    // all PLE layer inputs as one materialized argument.
    let Some(ple_inputs) = ple_inputs else {
        return Ok(None);
    };
    let metadata = auth_read!(layer_source, GemmaPrefillLayerSourceMetadataRequest)?;
    let mut token_count = None;
    let per_layer_inputs = (0..metadata.layer_count)
        .map(|layer_idx| {
            let input = ple_inputs
                .clone_layer_internal(layer_idx)
                .map(|input| raster_activation_sequence_from_internal(&input))
                .transpose()?;
            input
                .map(|input| {
                    let input_len = input.len();
                    match token_count {
                        Some(expected) if expected != input_len => bail!(
                            "materialized PLE input layer {layer_idx} contains {input_len} tokens, expected {expected}"
                        ),
                        None => token_count = Some(input_len),
                        _ => {}
                    }
                    store.insert_activation_sequence(
                        RasterTensorId::new(format!("prefill.layer.per_layer_input.{layer_idx}"))?,
                        input,
                    )
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;

    if per_layer_inputs.iter().all(Option::is_none) {
        return Ok(None);
    }

    Ok(Some(RasterPrefillPleInputRefs::new(
        metadata.source_id,
        metadata.layer_count,
        token_count.ok_or_else(|| anyhow!("materialized PLE inputs contained no layer rows"))?,
        per_layer_inputs,
    )?))
}

fn materialize_prefill_activation_sequence_from_store(
    store: &AuthenticatedRasterTensorStore,
    sequence_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationSequence> {
    // Public/dev compatibility boundary. zkVM-target substeps should consume the
    // ref directly and avoid this materializer.
    store.materialize_sequence(sequence_ref)
}

fn materialize_prefill_layer_cache_from_store(
    store: &AuthenticatedRasterTensorStore,
    cache: &PrefillLayerCacheSlot,
) -> Result<RasterKvCache> {
    // Public/dev compatibility boundary. Shared-store layer substeps use
    // `PrefillLayerCacheSlot::Ref` directly when replaying zkVM-shaped work.
    match cache {
        PrefillLayerCacheSlot::Empty { num_kv_heads } => Ok(RasterKvCache::empty(*num_kv_heads)),
        PrefillLayerCacheSlot::Ref(cache_ref) => store.materialize_kv_cache(cache_ref),
    }
}

fn materialize_prefill_layer_caches(
    store: &AuthenticatedRasterTensorStore,
    caches: &[PrefillLayerCacheSlot],
) -> Result<Vec<RasterKvCache>> {
    caches
        .iter()
        .map(|cache| materialize_prefill_layer_cache_from_store(store, cache))
        .collect()
}

fn raster_activation_sequence_from_activation(
    input_activations: &ActivationSequence,
) -> Result<RasterActivationSequence> {
    raster_activation_sequence_from_internal(&input_activations.clone_internal())
}

fn raster_activation_sequence_from_internal(
    input_activations: &InternalActivationSequence,
) -> Result<RasterActivationSequence> {
    let det_rows = input_activations.det_values().ok_or_else(|| {
        anyhow!("deterministic raster prefill layer input requires canonical activations")
    })?;
    Ok(RasterActivationSequence::from_acts(det_rows.to_vec()))
}

fn raster_sequence_acts(
    sequence: &RasterActivationSequence,
) -> Vec<Vec<crate::shared::det_num::Act>> {
    sequence.rows().iter().map(|row| row.acts()).collect()
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

fn layer_caches_from_raster(caches: &[RasterKvCache]) -> Vec<LayerKvCache> {
    caches
        .iter()
        .cloned()
        .map(layer_cache_from_raster)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        finalize_prefill_attention_state, init_prefill_attention_state,
        project_next_prefill_attention_row, run_materialized_compat, run_prefill_attention_rows,
        run_prefill_combine_heads, run_prefill_head_rms_norm, run_prefill_kv_cache,
        run_prefill_reshape_heads, run_prefill_rope_heads, run_prefill_sequence_add,
        run_prefill_sequence_gelu, run_prefill_sequence_mul, run_prefill_sequence_rms_norm,
        run_prefill_sequence_scale, run_prefill_value_rms_norm,
    };
    use crate::prefill_layer::deterministic_tiles;
    use crate::shared::det_num::{Acc, Act, Wgt};
    use crate::shared::raster_prefill_layer::AuthenticatedGemmaPrefillLayerSource;
    use crate::shared::raster_prefill_ple::RasterPrefillPleInputRefs;
    use crate::shared::raster_row_store::{
        AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterTensorId,
        RasterTensorKind, RasterTensorRef, RasterTensorShape,
    };
    use crate::shared::raster_transformer_kernels::{
        add_sequences, apply_rope_to_heads, build_raster_kv_cache,
        causal_attention_heads_with_cache, combine_attention_heads, gelu_sequence, mul_sequences,
        reshape_sequence_heads, rms_norm_heads, rms_norm_sequence, scale_sequence,
        value_rms_norm_heads, RasterActivationRow, RasterActivationSequence,
        RasterAttentionHeadSequence, RasterKvCache,
    };
    use crate::shared::transformer::{
        ActivationSequence, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleLayerWeights,
        Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence, MatrixF32,
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
    fn authored_prefill_layer_surfaces_do_not_accept_materialized_ple_inputs() {
        let source = include_str!("raster_tiles.rs");
        let materialized_ple_type = concat!("Gemma4", "PrefillPleInputs");
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
            }
            index += 1;
        }

        assert!(source.contains("pub fn run_materialized_compat("));
        assert!(source.contains("Compatibility adapter for dev/tests"));
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
    fn prefill_layer_state_serializes_refs_not_materialized_rows() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();

        let state =
            super::init_prefill_layer_state(&mut store, &input, &source, None, raster_sizing(1))
                .expect("state");

        let encoded = serde_json::to_string(&state).expect("serialize state");
        assert!(encoded.contains("current_activations_ref"));
        assert!(encoded.contains("layer_caches"));
        assert!(encoded.contains("per_layer_inputs"));
        assert!(!encoded.contains("act_bits"));
        assert!(!encoded.contains("\"keys\""));
        assert!(!encoded.contains("\"values\""));
    }

    #[test]
    fn prefill_layer_state_stays_ref_backed_after_layer_update() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let state =
            super::init_prefill_layer_state(&mut store, &input, &source, None, raster_sizing(1))
                .expect("state");

        let (_done, state) = super::compute_next_prefill_layer_sequence(state, &source, &mut store)
            .expect("layer should compute");
        let encoded = serde_json::to_string(&state).expect("serialize state");

        assert!(encoded.contains("current_activations_ref"));
        assert!(!encoded.contains("prefill.layer.current.after.0"));
        assert!(encoded.contains("layer_caches"));
        assert!(encoded.contains("prefill.layer.cache.0.keys"));
        assert!(encoded.contains("prefill.layer.cache.0.values"));
        assert!(!encoded.contains("act_bits"));
    }

    #[test]
    fn prefill_layer_ref_entrypoint_materializes_to_existing_result() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ]);
        let mut store = AuthenticatedRasterTensorStore::new();

        let refs = super::run_refs_with_store(&mut store, &input, &source, None, raster_sizing(1))
            .expect("ref-backed layer should run");
        assert_eq!(store.materialization_counts(), (0, 0));

        let materialized = super::materialize_prefill_layer_output_refs(&store, &refs)
            .expect("materialize ref-backed layer output");
        let deterministic = deterministic_tiles::run_internal(input.clone_internal(), &model, None)
            .expect("deterministic prefill layer should run");

        assert_eq!(materialized.0.activations, deterministic.0.activations);
        assert_eq!(
            materialized.0.det_activations_sha256,
            deterministic.0.det_activations_sha256
        );
        assert_eq!(materialized.1, deterministic.1);
        assert_eq!(refs.layer_caches.len(), deterministic.1.len());
    }

    #[test]
    fn recursive_layer_update_does_not_materialize_when_checkpointing_is_off() {
        let (mut store, state, output_ref, cache_slot) = update_state_fixture();

        let (_done, state) =
            super::update_prefill_layer_state_refs(&mut store, state, 0, output_ref, cache_slot)
                .expect("state update should succeed");

        assert_eq!(store.materialization_counts(), (0, 0));
        assert_eq!(state.next_layer_idx, 1);
        assert!(state.completed_layer_output_sha256s.is_empty());
        assert!(state.completed_layer_output_det_sha256s.is_empty());

        let _ = super::finalize_prefill_layer_state(&store, state).expect("finalize");
        let (sequence_materializations, cache_materializations) = store.materialization_counts();
        assert!(sequence_materializations > 0);
        assert!(cache_materializations > 0);
    }

    #[test]
    fn checkpoint_enabled_layer_update_materializes_lazily_for_compat_payload() {
        let (mut store, state, output_ref, cache_slot) = update_state_fixture();

        let (_done, state) = crate::trace::with_checkpointing_enabled(true, || {
            crate::trace::start_inference_trace(&serde_json::json!({ "test": "prefill.layer" }));
            super::update_prefill_layer_state_refs(&mut store, state, 0, output_ref, cache_slot)
                .expect("state update should succeed")
        });

        let (sequence_materializations, cache_materializations) = store.materialization_counts();
        assert!(sequence_materializations > 0);
        assert!(cache_materializations > 0);
        assert_eq!(state.completed_layer_output_sha256s.len(), 1);
        assert_eq!(state.completed_layer_output_det_sha256s.len(), 1);
    }

    #[test]
    fn layer_token_terminal_checkpoint_pauses_without_full_materialization() {
        let (mut store, state, output_ref, cache_slot) = update_state_fixture();
        let terminal =
            crate::trace::TerminalCheckpointSpec::parse("prefill.layer_token.layer_0.token_0")
                .expect("terminal checkpoint should parse");

        let (done, state) = crate::trace::with_terminal_checkpoint(Some(terminal), || {
            super::update_prefill_layer_state_refs(&mut store, state, 0, output_ref, cache_slot)
                .expect("state update should succeed")
        });

        assert!(done);
        assert_eq!(state.next_layer_idx, 1);
        assert_eq!(state.layer_count, 1);
        assert_eq!(store.materialization_counts(), (0, 0));
    }

    #[test]
    fn prefill_layer_cache_slots_materialize_empty_and_ref_caches() {
        let mut store = AuthenticatedRasterTensorStore::new();
        let empty = super::materialize_prefill_layer_cache_from_store(
            &store,
            &super::PrefillLayerCacheSlot::Empty { num_kv_heads: 2 },
        )
        .expect("empty cache");
        assert_eq!(empty.head_count(), 2);
        assert_eq!(empty.current_len(), 0);

        let cache = RasterKvCache::from_heads(
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                1.0,
            )])]],
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                2.0,
            )])]],
        )
        .expect("cache");
        let slot =
            super::register_prefill_layer_cache(&mut store, 0, cache.clone()).expect("cache slot");
        let materialized =
            super::materialize_prefill_layer_cache_from_store(&store, &slot).expect("cache");
        assert_eq!(materialized, cache);
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
        let raster =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(1))
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

        let error =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(1))
                .expect_err("empty donor cache should fail");

        assert!(error.to_string().contains("donor cache is empty"));
    }

    #[test]
    fn prefill_attention_rows_match_full_attention() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(2.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(4.0)],
        ]]);

        let row_output =
            run_prefill_attention_rows(&queries, &keys, &values, None, None, usize::MAX)
                .expect("row attention");
        let full_output = causal_attention_heads_with_cache(&queries, &keys, &values, None, None)
            .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn prefill_attention_rows_match_sliding_window_attention() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
            vec![Act::from_num(1.0), Act::from_num(0.0)],
        ]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(2.0)],
            vec![Act::from_num(3.0), Act::from_num(0.0)],
        ]]);

        let row_output =
            run_prefill_attention_rows(&queries, &keys, &values, None, Some(2), usize::MAX)
                .expect("row attention");
        let full_output =
            causal_attention_heads_with_cache(&queries, &keys, &values, None, Some(2))
                .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn prefill_attention_rows_match_donor_cache_attention() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(1.0)],
            vec![Act::from_num(1.0)],
        ]]);
        let current_keys = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(9.0)],
            vec![Act::from_num(9.0)],
        ]]);
        let current_values = RasterAttentionHeadSequence::from_acts(vec![vec![
            vec![Act::from_num(9.0)],
            vec![Act::from_num(9.0)],
        ]]);
        let donor_cache = RasterKvCache::from_heads(
            vec![vec![
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
            ]],
            vec![vec![
                RasterActivationRow::from_acts(vec![Act::from_num(3.0)]),
                RasterActivationRow::from_acts(vec![Act::from_num(4.0)]),
            ]],
        )
        .expect("cache should build");

        let row_output = run_prefill_attention_rows(
            &queries,
            &current_keys,
            &current_values,
            Some(&donor_cache),
            Some(1),
            usize::MAX,
        )
        .expect("row attention");
        let full_output = causal_attention_heads_with_cache(
            &queries,
            &current_keys,
            &current_values,
            Some(&donor_cache),
            Some(1),
        )
        .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn prefill_attention_rows_match_grouped_kv_attention() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
            vec![vec![Act::from_num(3.0)]],
            vec![vec![Act::from_num(4.0)]],
        ]);
        let keys = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(1.0)]],
            vec![vec![Act::from_num(2.0)]],
        ]);
        let values = RasterAttentionHeadSequence::from_acts(vec![
            vec![vec![Act::from_num(5.0)]],
            vec![vec![Act::from_num(7.0)]],
        ]);

        let row_output =
            run_prefill_attention_rows(&queries, &keys, &values, None, None, usize::MAX)
                .expect("row attention");
        let full_output = causal_attention_heads_with_cache(&queries, &keys, &values, None, None)
            .expect("full attention");

        assert_eq!(head_bits(&row_output), head_bits(&full_output));
    }

    #[test]
    fn prefill_attention_row_tile_reports_done_after_completion() {
        let queries = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(1.0)]]]);
        let keys = queries.clone();
        let values = RasterAttentionHeadSequence::from_acts(vec![vec![vec![Act::from_num(2.0)]]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let mut state = init_prefill_attention_state(
            &mut store,
            &queries,
            &keys,
            &values,
            None,
            None,
            usize::MAX,
        )
        .expect("attention state");
        let mut steps = 0;
        loop {
            if state.is_complete() {
                break;
            }
            let (done, next_state) = project_next_prefill_attention_row(state, &mut store)
                .expect("bounded attention phase should compute");
            assert!(!done);
            state = next_state;
            steps += 1;
            assert!(steps <= 10);
        }
        assert!(state.is_complete());
        assert!(steps > 2);

        let (done, state) = project_next_prefill_attention_row(state, &mut store)
            .expect("complete state should return done");
        assert!(done);
        let output = finalize_prefill_attention_state(&mut store, state).expect("finalize");
        assert_eq!(
            output.heads()[0][0].act_bits(),
            &[Act::from_num(2.0).to_bits()]
        );
    }

    #[test]
    fn phase_c_prefill_wrappers_match_full_helpers() {
        let sequence = RasterActivationSequence::from_acts(vec![
            vec![
                Act::from_num(1.0),
                Act::from_num(-0.5),
                Act::from_num(0.25),
                Act::from_num(0.75),
            ],
            vec![
                Act::from_num(0.5),
                Act::from_num(0.25),
                Act::from_num(-0.25),
                Act::from_num(1.0),
            ],
        ]);
        let rhs = RasterActivationSequence::from_acts(vec![
            vec![
                Act::from_num(0.25),
                Act::from_num(0.5),
                Act::from_num(0.75),
                Act::from_num(1.0),
            ],
            vec![
                Act::from_num(1.0),
                Act::from_num(-0.25),
                Act::from_num(0.5),
                Act::from_num(0.25),
            ],
        ]);
        let sequence_weights = vec![
            Wgt::from_num(1.0),
            Wgt::from_num(0.5),
            Wgt::from_num(0.25),
            Wgt::from_num(0.75),
        ];
        let eps = Acc::from_num(0.001);

        assert_eq!(
            scaled_bits(
                &run_prefill_sequence_rms_norm(&sequence, Some(&sequence_weights), Some(eps))
                    .expect("rms")
            ),
            scaled_bits(
                &rms_norm_sequence(&sequence, Some(&sequence_weights), Some(eps))
                    .expect("full rms")
            )
        );
        assert_eq!(
            scaled_bits(&run_prefill_sequence_gelu(&sequence).expect("gelu")),
            scaled_bits(&gelu_sequence(&sequence).expect("full gelu"))
        );
        assert_eq!(
            scaled_bits(
                &run_prefill_sequence_scale(&sequence, Some(Act::from_num(0.5))).expect("scale")
            ),
            scaled_bits(&scale_sequence(&sequence, Some(Act::from_num(0.5))).expect("full scale"))
        );
        assert_eq!(
            scaled_bits(&run_prefill_sequence_add(&sequence, &rhs).expect("add")),
            scaled_bits(&add_sequences(&sequence, &rhs).expect("full add"))
        );
        assert_eq!(
            scaled_bits(&run_prefill_sequence_mul(&sequence, &rhs).expect("mul")),
            scaled_bits(&mul_sequences(&sequence, &rhs).expect("full mul"))
        );

        let heads = run_prefill_reshape_heads(&sequence, 2, 2).expect("reshape");
        let full_heads = reshape_sequence_heads(&sequence, 2, 2).expect("full reshape");
        assert_eq!(head_bits(&heads), head_bits(&full_heads));

        let head_weights = vec![Wgt::from_num(1.0), Wgt::from_num(0.5)];
        assert_eq!(
            head_bits(
                &run_prefill_head_rms_norm(&heads, Some(&head_weights), Some(eps))
                    .expect("head rms")
            ),
            head_bits(
                &rms_norm_heads(&heads, Some(&head_weights), Some(eps)).expect("full head rms")
            )
        );
        assert_eq!(
            head_bits(&run_prefill_value_rms_norm(&heads, Some(eps)).expect("value rms")),
            head_bits(&value_rms_norm_heads(&heads, Some(eps)).expect("full value rms"))
        );
        assert_eq!(
            head_bits(
                &run_prefill_rope_heads(&heads, 2, 2, Some(Acc::from_num(10_000.0)), 1)
                    .expect("rope")
            ),
            head_bits(
                &apply_rope_to_heads(&heads, 2, 2, Some(Acc::from_num(10_000.0)), 1)
                    .expect("full rope")
            )
        );

        assert_eq!(
            scaled_bits(&run_prefill_combine_heads(&heads).expect("combine")),
            scaled_bits(&combine_attention_heads(&heads).expect("full combine"))
        );
        assert_eq!(
            run_prefill_kv_cache(&heads, &full_heads, Some(1)).expect("cache"),
            build_raster_kv_cache(&heads, &full_heads, Some(1)).expect("full cache")
        );
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

        let error =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(1))
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

        let raster =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(2))
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
    fn prefill_layer_init_from_refs_keeps_original_ple_refs() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let ple_ref = store
            .insert_activation_sequence(
                RasterTensorId::new("bridge.ple.0").expect("id"),
                RasterActivationSequence::from_acts(vec![vec![
                    Act::from_num(0.5),
                    Act::from_num(0.25),
                ]]),
            )
            .expect("insert PLE ref");
        let ple_refs =
            RasterPrefillPleInputRefs::new("prefill-layer", 1, 1, vec![Some(ple_ref.clone())])
                .expect("PLE refs");

        let state = super::init_prefill_layer_state(
            &mut store,
            &input,
            &source,
            Some(&ple_refs),
            raster_sizing(1),
        )
        .expect("state");

        assert_eq!(state.per_layer_inputs, vec![Some(ple_ref)]);
        assert_eq!(
            state.per_layer_inputs[0]
                .as_ref()
                .expect("PLE ref")
                .tensor_ref()
                .id()
                .source_name(),
            "bridge.ple.0"
        );
    }

    #[test]
    fn ple_layer_missing_input_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

        let error =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(1))
                .expect_err("missing PLE input should fail");

        assert!(error
            .to_string()
            .contains("requires PLE inputs but none were provided"));
    }

    #[test]
    fn stale_ple_input_ref_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let raw_ref = RasterTensorRef::new(
            RasterTensorId::new("missing.ple.0").expect("id"),
            RasterTensorKind::ActivationSequence,
            RasterTensorShape::sequence(1, 2).expect("shape"),
            "missing-commitment",
        )
        .expect("raw ref");
        let ple_ref = RasterActivationSequenceRef::new(raw_ref).expect("typed ref");
        let ple_refs = RasterPrefillPleInputRefs::new("prefill-layer", 1, 1, vec![Some(ple_ref)])
            .expect("PLE refs");

        let error = super::run_with_store(
            &mut store,
            &input,
            &source,
            Some(&ple_refs),
            raster_sizing(1),
        )
        .expect_err("stale PLE ref should fail");

        assert!(error.to_string().contains("is not registered"));
    }

    #[test]
    fn ple_input_manifest_source_mismatch_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let ple_ref = store
            .insert_activation_sequence(
                RasterTensorId::new("bridge.ple.0").expect("id"),
                RasterActivationSequence::from_acts(vec![vec![
                    Act::from_num(0.5),
                    Act::from_num(0.25),
                ]]),
            )
            .expect("insert PLE ref");
        let ple_refs = RasterPrefillPleInputRefs::new("other-source", 1, 1, vec![Some(ple_ref)])
            .expect("PLE refs");

        let error = super::init_prefill_layer_state(
            &mut store,
            &input,
            &source,
            Some(&ple_refs),
            raster_sizing(1),
        )
        .expect_err("source mismatch should fail");

        assert!(error
            .to_string()
            .contains("does not match prefill layer source"));
    }

    #[test]
    fn ple_input_manifest_layer_count_mismatch_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let ple_ref = store
            .insert_activation_sequence(
                RasterTensorId::new("bridge.ple.0").expect("id"),
                RasterActivationSequence::from_acts(vec![vec![
                    Act::from_num(0.5),
                    Act::from_num(0.25),
                ]]),
            )
            .expect("insert PLE ref");
        let ple_refs =
            RasterPrefillPleInputRefs::new("prefill-layer", 2, 1, vec![Some(ple_ref), None])
                .expect("PLE refs");

        let error = super::init_prefill_layer_state(
            &mut store,
            &input,
            &source,
            Some(&ple_refs),
            raster_sizing(1),
        )
        .expect_err("layer count mismatch should fail");

        assert!(error.to_string().contains("contain 2 layers, expected 1"));
    }

    #[test]
    fn ple_input_manifest_token_count_mismatch_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let ple_ref = store
            .insert_activation_sequence(
                RasterTensorId::new("bridge.ple.0").expect("id"),
                RasterActivationSequence::from_acts(vec![vec![
                    Act::from_num(0.5),
                    Act::from_num(0.25),
                ]]),
            )
            .expect("insert PLE ref");
        let ple_refs = RasterPrefillPleInputRefs::new("prefill-layer", 1, 2, vec![Some(ple_ref)])
            .expect("PLE refs");

        let error = super::init_prefill_layer_state(
            &mut store,
            &input,
            &source,
            Some(&ple_refs),
            raster_sizing(1),
        )
        .expect_err("token count mismatch should fail");

        assert!(error.to_string().contains("contain 2 tokens, expected 1"));
    }

    #[test]
    fn ple_input_ref_row_count_mismatch_fails_closed() {
        let (_path, model) = nonzero_model(true, false);
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let ple_ref = store
            .insert_activation_sequence(
                RasterTensorId::new("bridge.ple.0").expect("id"),
                RasterActivationSequence::from_acts(vec![
                    vec![Act::from_num(0.5), Act::from_num(0.25)],
                    vec![Act::from_num(1.0), Act::from_num(-0.25)],
                ]),
            )
            .expect("insert PLE ref");
        let ple_refs = RasterPrefillPleInputRefs::new("prefill-layer", 1, 1, vec![Some(ple_ref)])
            .expect("PLE refs");

        let error = super::run_with_store(
            &mut store,
            &input,
            &source,
            Some(&ple_refs),
            raster_sizing(1),
        )
        .expect_err("row count mismatch should fail");

        assert!(error
            .to_string()
            .contains("PLE input has 2 rows, expected 1"));
    }

    #[test]
    fn ple_input_on_non_ple_layer_fails_closed() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let ple_inputs = ple_inputs(vec![vec![Act::from_num(0.5), Act::from_num(0.25)]]);

        let error = run_materialized_compat(&input, &source, Some(&ple_inputs), raster_sizing(1))
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

        let error = run_materialized_compat(&input, &source, Some(&ple_inputs), raster_sizing(1))
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

        let error =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(1))
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

        let error =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(1))
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
            crate::shared::transformer_kernels::build_activation_commitment(&[vec![1.0, 0.0]]),
        );

        let error =
            run_ref_path_with_optional_materialized_compat(&input, &source, None, raster_sizing(1))
                .expect_err("f32-only input should fail");

        assert!(error.to_string().contains("requires canonical activations"));
    }

    fn assert_raster_matches_deterministic(
        model: &Gemma4TransformerModel,
        rows: Vec<Vec<Act>>,
    ) -> (
        ActivationSequence,
        Vec<crate::shared::transformer::LayerKvCache>,
    ) {
        assert_raster_matches_deterministic_with_ple(model, rows, None)
    }

    fn assert_raster_matches_deterministic_with_ple(
        model: &Gemma4TransformerModel,
        rows: Vec<Vec<Act>>,
        ple_inputs: Option<&Gemma4PrefillPleInputs>,
    ) -> (
        ActivationSequence,
        Vec<crate::shared::transformer::LayerKvCache>,
    ) {
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", model)
            .expect("source should build");
        let input_internal = InternalActivationSequence::from_det_values(rows);
        let input = activation_sequence_from_internal(input_internal.clone());

        let raster = run_ref_path_with_optional_materialized_compat(
            &input,
            &source,
            ple_inputs,
            raster_sizing(1),
        )
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

    fn run_ref_path_with_optional_materialized_compat(
        input: &ActivationSequence,
        source: &AuthenticatedGemmaPrefillLayerSource,
        ple_inputs: Option<&Gemma4PrefillPleInputs>,
        raster_sizing: RasterSizingControls,
    ) -> Result<(
        ActivationSequence,
        Vec<crate::shared::transformer::LayerKvCache>,
    )> {
        let mut store = AuthenticatedRasterTensorStore::new();
        let ple_input_refs = super::import_materialized_ple_inputs(&mut store, source, ple_inputs)?;
        super::run_with_store(
            &mut store,
            input,
            source,
            ple_input_refs.as_ref(),
            raster_sizing,
        )
    }

    fn update_state_fixture() -> (
        AuthenticatedRasterTensorStore,
        super::PrefillLayerRasterState,
        RasterActivationSequenceRef,
        super::PrefillLayerCacheSlot,
    ) {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let mut store = AuthenticatedRasterTensorStore::new();
        let state =
            super::init_prefill_layer_state(&mut store, &input, &source, None, raster_sizing(1))
                .expect("state");
        let output_ref = store
            .insert_activation_sequence(
                RasterTensorId::new("test.prefill.layer.output").expect("output id"),
                RasterActivationSequence::from_acts(vec![vec![
                    Act::from_num(0.5),
                    Act::from_num(0.25),
                ]]),
            )
            .expect("output ref");
        let cache = RasterKvCache::from_heads(
            vec![vec![RasterActivationRow::from_acts(vec![
                Act::from_num(1.0),
                Act::from_num(0.0),
            ])]],
            vec![vec![RasterActivationRow::from_acts(vec![
                Act::from_num(0.0),
                Act::from_num(1.0),
            ])]],
        )
        .expect("cache");
        let cache_ref = store
            .insert_kv_cache(
                RasterTensorId::new("test.prefill.layer.cache.keys").expect("keys id"),
                RasterTensorId::new("test.prefill.layer.cache.values").expect("values id"),
                cache,
            )
            .expect("cache ref");
        (
            store,
            state,
            output_ref,
            super::PrefillLayerCacheSlot::Ref(cache_ref),
        )
    }

    fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
        activation_sequence_from_internal(InternalActivationSequence::from_det_values(rows))
    }

    fn scaled_bits(sequence: &RasterActivationSequence) -> Vec<Vec<i32>> {
        sequence
            .rows()
            .iter()
            .map(|row| row.act_bits().to_vec())
            .collect()
    }

    fn head_bits(sequence: &RasterAttentionHeadSequence) -> Vec<Vec<Vec<i32>>> {
        sequence
            .heads()
            .iter()
            .map(|head| head.iter().map(|row| row.act_bits().to_vec()).collect())
            .collect()
    }

    fn activation_sequence_from_internal(
        input_internal: InternalActivationSequence,
    ) -> ActivationSequence {
        let mut input = ActivationSequence::from_internal(
            input_internal.clone(),
            crate::shared::transformer_kernels::build_activation_commitment(
                input_internal.as_f32_slice(),
            ),
        );
        input.det_activations_sha256 = Some(
            crate::shared::transformer_kernels::build_det_activation_commitment(
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
