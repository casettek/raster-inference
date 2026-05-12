use anyhow::{anyhow, bail, Result};

use crate::raster_authoring::prelude::{
    auth_read, call_recur_seq, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::det_num::{scale_act, Acc, Act, Wgt};
use crate::shared::raster_prefill_ple::{
    AuthenticatedGemmaPleSource, GemmaPleLayerMetadataRequest, GemmaPleMetadataRequest,
    GemmaPleModelProjectionRowRequest, GemmaPleProjectionNormWeightsRequest,
    GemmaPleScalarsRequest, GemmaPleTokenEmbeddingRowRequest, RasterPrefillPleInputRefs,
};
use crate::shared::raster_row_store::{
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterTensorBuilderRef,
    RasterTensorId,
};
use crate::shared::raster_transformer_kernels::{
    validate_projection_rows_per_tile, validate_sequence_rows_per_tile, RasterActivationRow,
    RasterActivationSequence, RasterSequenceBinaryState, RasterSequenceProjectionState,
    RasterSequenceUnaryState,
};
use crate::shared::transformer::{ActivationSequence, Gemma4PrefillPleInputs};
use crate::RasterSizingControls;

use super::raster_utils::{
    append_projection_chunk, append_sequence_row, compute_next_binary_row, compute_next_unary_row,
    finalize_binary_row_state_ref, finalize_projection_state, finalize_projection_state_ref,
    finalize_sequence_builder, finalize_unary_row_state_ref, init_add_row_state_from_refs,
    init_projection_state, init_projection_state_from_ref, init_rms_norm_row_state_from_ref,
    init_scale_row_state_from_ref, insert_activation_sequence, internal_sequence_from_raster,
    materialize_sequence, raster_activation_sequence_from_embedding, read_prefill_token_id,
    reset_tensor_store, start_sequence_builder, store_prefill_token_ids_artifact,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleRasterState {
    source_id: String,
    token_ids_artifact_root: String,
    token_count: usize,
    input_activations_ref: Option<RasterActivationSequenceRef>,
    next_layer_idx: usize,
    layer_count: usize,
    per_layer_inputs: Vec<Option<RasterActivationSequenceRef>>,
    has_ple_global: bool,
    raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillPleInputRoots {
    source_id: String,
    token_ids_artifact_root: String,
    token_count: usize,
    input_activations_ref: Option<RasterActivationSequenceRef>,
    layer_count: usize,
    has_ple_global: bool,
    raster_sizing: RasterSizingControls,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleLayerContext {
    layer_idx: usize,
    has_ple: bool,
    ple_width: Option<usize>,
    projection_rows: Option<usize>,
    embedding_scale_bits: Option<i32>,
    projection_scalar_bits: Option<i32>,
    input_scale_bits: Option<i32>,
    rms_norm_eps_bits: Option<i64>,
    norm_weight_bits: Option<Vec<i32>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillPleTokenEmbeddingState {
    token_ids_artifact_root: String,
    layer_idx: usize,
    scale_bits: i32,
    next_token_idx: usize,
    token_count: usize,
    row_width: usize,
    output_builder_ref: RasterTensorBuilderRef,
}

impl PrefillPleTokenEmbeddingState {
    fn is_complete(&self) -> bool {
        self.next_token_idx >= self.token_count
    }
}

pub fn init_prefill_ple_tensor_store() {
    reset_tensor_store();
}

pub fn tensor_store_snapshot() -> AuthenticatedRasterTensorStore {
    super::raster_utils::tensor_store_snapshot()
}

pub fn prepare_raster_prefill_ple_input_roots(
    token_ids: &[u32],
    input_activations: &ActivationSequence,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<RasterPrefillPleInputRoots> {
    init_prefill_ple_tensor_store();
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    validate_sequence_rows_per_tile(raster_sizing.sequence_rows_per_tile)?;
    let token_ids_ref = store_prefill_token_ids_artifact(token_ids)?;
    let token_count = token_ids_ref.token_count();
    let metadata = auth_read!(ple_source, GemmaPleMetadataRequest)?;
    if !metadata.has_ple_global {
        return Ok(RasterPrefillPleInputRoots {
            source_id: metadata.source_id,
            token_ids_artifact_root: token_ids_ref.root().to_string(),
            token_count: token_ids_ref.token_count(),
            input_activations_ref: None,
            layer_count: metadata.layer_count,
            has_ple_global: false,
            raster_sizing,
        });
    }

    if metadata.layer_count == 0 {
        bail!("transformer PLE computation requires at least one layer");
    }
    if metadata.token_embedding_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            metadata.token_embedding_layer_count,
            metadata.layer_count
        );
    }
    if metadata.model_projection_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            metadata.model_projection_layer_count,
            metadata.layer_count
        );
    }

    let input_activations = raster_activation_sequence_from_embedding(input_activations)?;
    if input_activations.is_empty() {
        bail!("transformer PLE computation requires at least one activation row");
    }
    if token_count != input_activations.len() {
        bail!(
            "transformer PLE computation requires token ids and activations to have matching lengths"
        );
    }

    let first_layer = auth_read!(ple_source, GemmaPleLayerMetadataRequest { layer_idx: 0 })?;
    let activation_width = input_activations.width()?;
    if activation_width != first_layer.hidden_width {
        bail!(
            "input activations row 0 has width {}, expected {}",
            activation_width,
            first_layer.hidden_width
        );
    }
    let input_activations_ref = insert_activation_sequence(
        RasterTensorId::new("prefill.prepare_aux.input.initial")?,
        input_activations,
    )?;

    Ok(RasterPrefillPleInputRoots {
        source_id: metadata.source_id,
        token_ids_artifact_root: token_ids_ref.root().to_string(),
        token_count: token_ids_ref.token_count(),
        input_activations_ref: Some(input_activations_ref),
        layer_count: metadata.layer_count,
        has_ple_global: true,
        raster_sizing,
    })
}

#[tile]
pub fn init_prefill_ple_state(
    input_roots: RasterPrefillPleInputRoots,
) -> Result<PrefillPleRasterState> {
    Ok(PrefillPleRasterState {
        source_id: input_roots.source_id,
        token_ids_artifact_root: input_roots.token_ids_artifact_root,
        token_count: input_roots.token_count,
        input_activations_ref: input_roots.input_activations_ref,
        next_layer_idx: 0,
        layer_count: input_roots.layer_count,
        per_layer_inputs: Vec::with_capacity(input_roots.layer_count),
        has_ple_global: input_roots.has_ple_global,
        raster_sizing: input_roots.raster_sizing,
    })
}

#[tile]
pub fn prepare_next_prefill_ple_context(
    state: &PrefillPleRasterState,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<PrefillPleLayerContext> {
    if !state.has_ple_global {
        bail!("cannot prepare PLE layer context without global PLE weights");
    }
    if state.next_layer_idx >= state.layer_count {
        bail!(
            "cannot prepare PLE layer {} after completing {} layers",
            state.next_layer_idx,
            state.layer_count
        );
    }
    let layer_idx = state.next_layer_idx;
    let layer = auth_read!(ple_source, GemmaPleLayerMetadataRequest { layer_idx })?;
    if !layer.has_ple {
        return Ok(PrefillPleLayerContext {
            layer_idx,
            has_ple: false,
            ple_width: None,
            projection_rows: None,
            embedding_scale_bits: None,
            projection_scalar_bits: None,
            input_scale_bits: None,
            rms_norm_eps_bits: None,
            norm_weight_bits: None,
        });
    }

    state
        .input_activations_ref
        .as_ref()
        .ok_or_else(|| anyhow!("raster PLE state is missing input activation ref"))?;
    let projection_rows = layer.model_projection_rows.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing model projection row metadata")
    })?;
    let scalars = auth_read!(ple_source, GemmaPleScalarsRequest)?;
    let norm_weights = auth_read!(ple_source, GemmaPleProjectionNormWeightsRequest)?;
    let ple_width = layer
        .ple_width
        .ok_or_else(|| anyhow!("Gemma PLE layer {layer_idx} is missing token embedding width"))?;

    Ok(PrefillPleLayerContext {
        layer_idx,
        has_ple: true,
        ple_width: Some(ple_width),
        projection_rows: Some(projection_rows),
        embedding_scale_bits: Some(scalars.embedding_scale.to_bits()),
        projection_scalar_bits: Some(scalars.projection_scalar.to_bits()),
        input_scale_bits: Some(scalars.input_scale.to_bits()),
        rms_norm_eps_bits: Some(scalars.rms_norm_eps.to_bits()),
        norm_weight_bits: Some(norm_weights.iter().map(|weight| weight.to_bits()).collect()),
    })
}

#[sequence(kind = recursive)]
pub fn compute_next_prefill_ple_layer_sequence(
    state: PrefillPleRasterState,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<(bool, PrefillPleRasterState)> {
    if !state.has_ple_global || state.next_layer_idx >= state.layer_count {
        return Ok((true, state));
    }

    let context = call_tile!(prepare_next_prefill_ple_context, &state, ple_source)?;
    crate::trace::trace_event(format!(
        "progress prefill.prepare_aux layer={}/{} ple={} tokens={}",
        context.layer_idx + 1,
        state.layer_count,
        context.has_ple,
        state.token_count
    ));

    if !context.has_ple {
        return call_tile!(
            update_prefill_ple_state_refs,
            state,
            context.layer_idx,
            None
        );
    }

    let input_activations_ref = state
        .input_activations_ref
        .as_ref()
        .ok_or_else(|| anyhow!("raster PLE state is missing input activation ref"))?
        .clone();
    let layer_input_ref = call_seq!(
        run_prefill_ple_layer_sequence_ref,
        &state.token_ids_artifact_root,
        state.token_count,
        input_activations_ref,
        ple_source,
        &context,
        state.raster_sizing
    )?;

    call_tile!(
        update_prefill_ple_state_refs,
        state,
        context.layer_idx,
        Some(layer_input_ref)
    )
}

#[sequence]
fn run_prefill_ple_layer_sequence_ref(
    token_ids_artifact_root: &str,
    token_count: usize,
    input_activations_ref: RasterActivationSequenceRef,
    ple_source: &AuthenticatedGemmaPleSource,
    context: &PrefillPleLayerContext,
    raster_sizing: RasterSizingControls,
) -> Result<RasterActivationSequenceRef> {
    let layer_idx = context.layer_idx;
    let embedding_scale = Act::from_bits(context.embedding_scale_bits.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing embedding scale metadata")
    })?);
    let projection_scalar = Act::from_bits(context.projection_scalar_bits.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing projection scalar metadata")
    })?);
    let input_scale =
        Act::from_bits(context.input_scale_bits.ok_or_else(|| {
            anyhow!("Gemma PLE layer {layer_idx} is missing input scale metadata")
        })?);
    let rms_norm_eps = Acc::from_bits(context.rms_norm_eps_bits.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing RMSNorm epsilon metadata")
    })?);
    let norm_weights = context
        .norm_weight_bits
        .as_ref()
        .ok_or_else(|| anyhow!("Gemma PLE layer {layer_idx} is missing norm weights"))?
        .iter()
        .map(|bits| Wgt::from_bits(*bits))
        .collect::<Vec<_>>();
    let ple_width = context
        .ple_width
        .ok_or_else(|| anyhow!("Gemma PLE layer {layer_idx} is missing token embedding width"))?;
    let projection_rows = context.projection_rows.ok_or_else(|| {
        anyhow!("Gemma PLE layer {layer_idx} is missing model projection row metadata")
    })?;
    let embedded_ref = call_seq!(
        build_scaled_token_embedding_sequence_ref,
        token_ids_artifact_root,
        token_count,
        layer_idx,
        ple_source,
        embedding_scale,
        ple_width
    )?;
    let projected_ref = call_seq!(
        project_ple_sequence_with_source_ref,
        input_activations_ref.clone(),
        ple_source,
        layer_idx,
        projection_rows,
        raster_sizing.projection_rows_per_tile,
        RasterTensorId::new(format!("prefill.prepare_aux.projected.{layer_idx}"))?
    )?;
    let projected_ref = call_seq!(
        compute_sequence_scale_ref,
        projected_ref,
        RasterTensorId::new(format!("prefill.prepare_aux.projected.scaled.{layer_idx}"))?,
        Some(projection_scalar),
        raster_sizing.sequence_rows_per_tile
    )?;
    let projected_ref = call_seq!(
        compute_sequence_rms_norm_ref,
        projected_ref,
        RasterTensorId::new(format!("prefill.prepare_aux.projected.normed.{layer_idx}"))?,
        Some(norm_weights.as_slice()),
        Some(rms_norm_eps),
        raster_sizing.sequence_rows_per_tile
    )?;
    let combined_ref = call_seq!(
        compute_sequence_add_ref,
        embedded_ref,
        projected_ref,
        RasterTensorId::new(format!("prefill.prepare_aux.combined.{layer_idx}"))?,
        raster_sizing.sequence_rows_per_tile
    )?;
    call_seq!(
        compute_sequence_scale_ref,
        combined_ref,
        RasterTensorId::new(format!("prefill.prepare_aux.per_layer_input.{layer_idx}"))?,
        Some(input_scale),
        raster_sizing.sequence_rows_per_tile
    )
}

#[tile]
pub fn update_prefill_ple_state_refs(
    mut state: PrefillPleRasterState,
    layer_idx: usize,
    per_layer_input: Option<RasterActivationSequenceRef>,
) -> Result<(bool, PrefillPleRasterState)> {
    if layer_idx != state.next_layer_idx {
        bail!(
            "cannot update PLE layer {layer_idx} while next layer is {}",
            state.next_layer_idx
        );
    }
    state.per_layer_inputs.push(per_layer_input);
    state.next_layer_idx += 1;
    Ok((state.next_layer_idx >= state.layer_count, state))
}

#[tile]
pub fn finalize_prefill_ple_input_refs(
    state: PrefillPleRasterState,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    if !state.has_ple_global {
        return Ok(None);
    }
    if state.per_layer_inputs.len() != state.layer_count {
        bail!(
            "raster PLE finalized with {} layers, expected {}",
            state.per_layer_inputs.len(),
            state.layer_count
        );
    }

    Ok(Some(RasterPrefillPleInputRefs::new(
        state.source_id,
        state.layer_count,
        state.token_count,
        state.per_layer_inputs,
    )?))
}

#[tile]
pub fn materialize_prefill_ple_input_refs(
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let Some(ple_input_refs) = ple_input_refs else {
        return Ok(None);
    };

    // Public compatibility boundary. Recursive prepare-aux work stores only
    // refs/cursors; materialization happens here only to preserve the existing
    // `Gemma4PrefillPleInputs` return shape outside zkVM replay.
    Ok(Some(Gemma4PrefillPleInputs::from_internal(
        ple_input_refs
            .per_layer_inputs()
            .iter()
            .map(|input| {
                input
                    .as_ref()
                    .map(|input_ref| {
                        materialize_sequence(&input_ref).map(internal_sequence_from_raster)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?,
    )))
}

#[tile]
pub fn finalize_prefill_ple_inputs(
    state: PrefillPleRasterState,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let refs = finalize_prefill_ple_input_refs(state)?;
    materialize_prefill_ple_input_refs(refs.as_ref())
}

#[tile]
pub fn init_ple_sequence_projection(
    input: &RasterActivationSequence,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    init_projection_state(input, projection_rows, projection_rows_per_tile)
}

#[tile(kind = recursive)]
pub fn project_next_ple_sequence_rows(
    mut state: RasterSequenceProjectionState,
    ple_source: &AuthenticatedGemmaPleSource,
    layer_idx: usize,
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
            ple_source,
            GemmaPleModelProjectionRowRequest { layer_idx, row_idx },
        )?);
    }
    append_projection_chunk(&mut state, &rows)?;
    crate::trace::trace_event(format!(
        "progress prefill.prepare_aux.projection layer={} token={}/{} projection_rows={}..{} of {} input_width={}",
        layer_idx,
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
pub fn finalize_ple_sequence_projection(
    state: RasterSequenceProjectionState,
) -> Result<RasterActivationSequence> {
    finalize_projection_state(state)
}

#[sequence]
pub fn project_sequence_with_source(
    input: &RasterActivationSequence,
    ple_source: &AuthenticatedGemmaPleSource,
    layer_idx: usize,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterActivationSequence> {
    let state = call_tile!(
        init_ple_sequence_projection,
        input,
        projection_rows,
        projection_rows_per_tile
    )?;
    let state = call_recur_tile!(project_next_ple_sequence_rows, state, ple_source, layer_idx)?;
    call_tile!(finalize_ple_sequence_projection, state)
}

pub fn run(
    token_ids: &[u32],
    input_activations: &ActivationSequence,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    let input_roots = prepare_raster_prefill_ple_input_roots(
        token_ids,
        input_activations,
        ple_source,
        raster_sizing,
    )?;
    main(input_roots, ple_source)
}

#[sequence]
pub fn main(
    input_roots: RasterPrefillPleInputRoots,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    let state = call_tile!(init_prefill_ple_state, input_roots)?;
    let state = call_recur_seq!(compute_next_prefill_ple_layer_sequence, state, ple_source)?;
    call_tile!(finalize_prefill_ple_input_refs, state)
}

fn raster_sizing_with_projection_rows(projection_rows_per_tile: usize) -> RasterSizingControls {
    RasterSizingControls {
        projection_rows_per_tile,
        attention_kv_rows_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE,
        sequence_rows_per_tile: crate::InferenceControls::DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE,
        head_rows_per_tile: crate::InferenceControls::DEFAULT_RASTER_HEAD_ROWS_PER_TILE,
        tokenizer_bpe_pairs_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE,
        tokenizer_bpe_pieces_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE,
        output_byte_flush_bytes_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    }
}

#[tile]
fn init_scaled_token_embedding_sequence_ref(
    token_ids_artifact_root: &str,
    token_count: usize,
    layer_idx: usize,
    scale: crate::shared::det_num::Act,
    row_width: usize,
) -> Result<PrefillPleTokenEmbeddingState> {
    if token_count == 0 {
        bail!("transformer PLE token embedding sequence requires at least one token");
    }
    let output_builder_ref = start_sequence_builder(
        RasterTensorId::new(format!("prefill.prepare_aux.embedded.{layer_idx}"))?,
        token_count,
        row_width,
    )?;
    Ok(PrefillPleTokenEmbeddingState {
        token_ids_artifact_root: token_ids_artifact_root.to_string(),
        layer_idx,
        scale_bits: scale.to_bits(),
        next_token_idx: 0,
        token_count,
        row_width,
        output_builder_ref,
    })
}

#[tile(kind = recursive)]
fn append_next_scaled_token_embedding_row(
    mut state: PrefillPleTokenEmbeddingState,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<(bool, PrefillPleTokenEmbeddingState)> {
    if state.is_complete() {
        return Ok((true, state));
    }

    let token_id = read_prefill_token_id(
        &state.token_ids_artifact_root,
        state.token_count,
        state.next_token_idx,
    )?;
    let row = auth_read!(
        ple_source,
        GemmaPleTokenEmbeddingRowRequest {
            layer_idx: state.layer_idx,
            token_id,
        },
    )?;
    if row.len() != state.row_width {
        bail!(
            "PLE token embedding row has width {}, expected {}",
            row.len(),
            state.row_width
        );
    }
    append_sequence_row(
        &mut state.output_builder_ref,
        state.next_token_idx,
        RasterActivationRow::from_acts(
            row.into_iter()
                .map(|value| {
                    scale_act(
                        value,
                        crate::shared::det_num::Act::from_bits(state.scale_bits),
                    )
                })
                .collect(),
        ),
    )?;
    state.next_token_idx += 1;
    Ok((false, state))
}

#[tile]
fn finalize_scaled_token_embedding_sequence_ref(
    state: PrefillPleTokenEmbeddingState,
) -> Result<RasterActivationSequenceRef> {
    if !state.is_complete() {
        bail!(
            "PLE token embedding finalized at token {}, expected {}",
            state.next_token_idx,
            state.token_count
        );
    }
    finalize_sequence_builder(state.output_builder_ref)
}

#[sequence]
fn build_scaled_token_embedding_sequence_ref(
    token_ids_artifact_root: &str,
    token_count: usize,
    layer_idx: usize,
    ple_source: &AuthenticatedGemmaPleSource,
    scale: crate::shared::det_num::Act,
    row_width: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_scaled_token_embedding_sequence_ref,
        token_ids_artifact_root,
        token_count,
        layer_idx,
        scale,
        row_width
    )?;
    let state = call_recur_tile!(append_next_scaled_token_embedding_row, state, ple_source)?;
    call_tile!(finalize_scaled_token_embedding_sequence_ref, state)
}

#[tile]
fn init_ple_sequence_projection_from_ref(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    init_projection_state_from_ref(
        input_ref,
        output_id,
        projection_rows,
        projection_rows_per_tile,
    )
}

#[tile]
fn finalize_ple_sequence_projection_ref(
    state: RasterSequenceProjectionState,
) -> Result<RasterActivationSequenceRef> {
    finalize_projection_state_ref(state)
}

#[sequence]
fn project_ple_sequence_with_source_ref(
    input_ref: RasterActivationSequenceRef,
    ple_source: &AuthenticatedGemmaPleSource,
    layer_idx: usize,
    projection_rows: usize,
    projection_rows_per_tile: usize,
    output_id: RasterTensorId,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_ple_sequence_projection_from_ref,
        input_ref,
        output_id,
        projection_rows,
        projection_rows_per_tile
    )?;
    let state = call_recur_tile!(project_next_ple_sequence_rows, state, ple_source, layer_idx)?;
    call_tile!(finalize_ple_sequence_projection_ref, state)
}

#[tile]
fn init_sequence_scale_ref_state(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    scalar: Option<crate::shared::det_num::Act>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_scale_row_state_from_ref(input_ref, output_id, scalar, rows_per_tile)
}

#[tile]
fn init_sequence_rms_norm_ref_state(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    init_rms_norm_row_state_from_ref(input_ref, output_id, norm_weights, eps, rows_per_tile)
}

#[tile]
fn finalize_sequence_unary_ref_state(
    state: RasterSequenceUnaryState,
) -> Result<RasterActivationSequenceRef> {
    finalize_unary_row_state_ref(state)
}

#[sequence]
fn compute_sequence_scale_ref(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    scalar: Option<crate::shared::det_num::Act>,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_sequence_scale_ref_state,
        input_ref,
        output_id,
        scalar,
        rows_per_tile
    )?;
    let state = call_recur_tile!(compute_next_sequence_unary_ref, state)?;
    call_tile!(finalize_sequence_unary_ref_state, state)
}

#[sequence]
fn compute_sequence_rms_norm_ref(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_sequence_rms_norm_ref_state,
        input_ref,
        output_id,
        norm_weights,
        eps,
        rows_per_tile
    )?;
    let state = call_recur_tile!(compute_next_sequence_unary_ref, state)?;
    call_tile!(finalize_sequence_unary_ref_state, state)
}

#[tile(kind = recursive)]
fn compute_next_sequence_unary_ref(
    state: RasterSequenceUnaryState,
) -> Result<(bool, RasterSequenceUnaryState)> {
    compute_next_unary_row(state)
}

#[tile]
fn init_sequence_add_ref_state(
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    init_add_row_state_from_refs(lhs_ref, rhs_ref, output_id, rows_per_tile)
}

#[tile]
fn finalize_sequence_binary_ref_state(
    state: RasterSequenceBinaryState,
) -> Result<RasterActivationSequenceRef> {
    finalize_binary_row_state_ref(state)
}

#[sequence]
fn compute_sequence_add_ref(
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterActivationSequenceRef> {
    let state = call_tile!(
        init_sequence_add_ref_state,
        lhs_ref,
        rhs_ref,
        output_id,
        rows_per_tile
    )?;
    let state = call_recur_tile!(compute_next_sequence_binary_ref, state)?;
    call_tile!(finalize_sequence_binary_ref_state, state)
}

#[tile(kind = recursive)]
fn compute_next_sequence_binary_ref(
    state: RasterSequenceBinaryState,
) -> Result<(bool, RasterSequenceBinaryState)> {
    compute_next_binary_row(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::det_num::{f32_to_acc, Act, Wgt};
    use crate::shared::raster_prefill_ple::{
        AuthenticatedGemmaPleSource, GemmaPleLayerConfig, GemmaPleScalars,
    };
    use crate::shared::transformer::{
        ActivationSequence, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
        Gemma4PleLayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
        InternalActivationSequence, MatrixF32,
    };
    use crate::shared::{det_num::act_to_f32, input::InferenceExecutionMode};
    use anyhow::{Context, Result};
    use std::path::PathBuf;

    #[test]
    fn ref_backed_prepare_aux_helpers_do_not_hide_completion_loops() {
        let source = include_str!("raster_tiles.rs");
        let forbidden = concat!("while !", "state.is_complete()");

        assert!(
            !source.contains(forbidden),
            "ref-backed prepare-aux helpers must expose dynamic loops through recursive tile calls"
        );
    }

    #[test]
    fn scaled_token_embedding_rows_advance_one_recursive_step_at_a_time() {
        let fixture = PleFixture::single_layer().expect("fixture should build");
        init_prefill_ple_tensor_store();
        let token_ids_ref = store_prefill_token_ids_artifact(&[0, 1]).expect("token ids ref");
        let state = init_scaled_token_embedding_sequence_ref(
            token_ids_ref.root(),
            token_ids_ref.token_count(),
            0,
            Act::from_num(1.0),
            2,
        )
        .expect("init token embedding state");
        let encoded = serde_json::to_string(&state).expect("serialize token embedding state");
        assert!(encoded.contains("token_ids_artifact_root"));
        assert!(!encoded.contains("[0,1]"));
        assert_eq!(state.next_token_idx, 0);

        let (done, state) = append_next_scaled_token_embedding_row(state, &fixture.source)
            .expect("append first row");
        assert!(!done);
        assert_eq!(state.next_token_idx, 1);

        let (done, state) = append_next_scaled_token_embedding_row(state, &fixture.source)
            .expect("append second row");
        assert!(!done);
        assert_eq!(state.next_token_idx, 2);

        let (done, state) = append_next_scaled_token_embedding_row(state, &fixture.source)
            .expect("observe completion");
        assert!(done);

        let sequence_ref = finalize_scaled_token_embedding_sequence_ref(state)
            .expect("finalize embedded sequence");
        let sequence = materialize_sequence(&sequence_ref).expect("materialize embedded sequence");
        assert_eq!(sequence.len(), 2);
        assert_eq!(sequence.width().expect("width"), 2);
    }

    #[test]
    fn no_ple_globals_return_none_without_requiring_deterministic_inputs() {
        let source = AuthenticatedGemmaPleSource::no_ple(
            "no-ple",
            vec![GemmaPleLayerConfig {
                has_ple: false,
                hidden_width: 2,
            }],
        )
        .expect("source should build");
        let input = ActivationSequence::from_values(vec![vec![1.0, 2.0]], "digest".to_string());

        let output = run_materialized(&[0], &input, &source, 1).expect("raster PLE should run");

        assert!(output.is_none());
    }

    #[test]
    fn no_ple_globals_match_native_none_output() {
        let source = AuthenticatedGemmaPleSource::no_ple(
            "no-ple",
            vec![GemmaPleLayerConfig {
                has_ple: false,
                hidden_width: 2,
            }],
        )
        .expect("source should build");
        let input = ActivationSequence::from_values(vec![vec![1.0, 2.0]], "digest".to_string());
        let native_model = no_ple_model(2);

        let raster = run_materialized(&[0], &input, &source, 1).expect("raster PLE should run");
        let native = crate::prefill_prepare_aux::run(
            &[0],
            &native_model,
            &input,
            InferenceExecutionMode::Deterministic,
        )
        .expect("native PLE should run");

        assert_eq!(raster, native);
        assert!(raster.is_none());
    }

    #[test]
    fn single_token_single_ple_layer_matches_native_prefill_ple_computation() {
        let fixture = PleFixture::new(
            vec![true],
            vec![vec![
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
                vec![Act::from_num(1.0), Act::from_num(0.5)],
            ]],
            vec![vec![
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ]],
        )
        .expect("fixture should build");
        let token_ids = [0];
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let (raster, native) = compare_raster_and_native(&fixture, &token_ids, &input);

        assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
        assert_eq!(
            raster
                .clone_layer_internal(0)
                .expect("raster layer")
                .det_values(),
            native
                .clone_layer_internal(0)
                .expect("native layer")
                .det_values()
        );
    }

    #[test]
    fn multiple_prompt_tokens_single_ple_layer_matches_native_prefill_ple_computation() {
        let fixture = PleFixture::new(
            vec![true],
            vec![vec![
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
                vec![Act::from_num(1.0), Act::from_num(0.5)],
            ]],
            vec![vec![
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ]],
        )
        .expect("fixture should build");
        let token_ids = [0, 1];
        let input = activation_sequence(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);

        let (raster, native) = compare_raster_and_native(&fixture, &token_ids, &input);

        assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
        assert_eq!(
            raster
                .clone_layer_internal(0)
                .expect("raster layer")
                .det_values(),
            native
                .clone_layer_internal(0)
                .expect("native layer")
                .det_values()
        );
    }

    #[test]
    fn chunked_prefill_ple_projection_matches_native_prefill_ple_computation() {
        let fixture = PleFixture::new(
            vec![true],
            vec![vec![
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
                vec![Act::from_num(1.0), Act::from_num(0.5)],
            ]],
            vec![vec![
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ]],
        )
        .expect("fixture should build");
        let token_ids = [0, 1];
        let input = activation_sequence(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);

        let raster = run_materialized(&token_ids, &input, &fixture.source, 2)
            .expect("raster PLE should run")
            .expect("raster PLE inputs");
        let native = crate::shared::transformer_kernels::compute_prefill_ple_inputs_internal(
            &token_ids,
            input.clone_internal(),
            &fixture.layers,
            &fixture.native_ple_global,
            0.0,
            Some(f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .expect("native PLE should run");

        assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
    }

    #[test]
    fn chunked_prefill_ple_sequence_rows_do_not_change_output() {
        let fixture = PleFixture::new(
            vec![true],
            vec![vec![
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
                vec![Act::from_num(1.0), Act::from_num(0.5)],
                vec![Act::from_num(-0.25), Act::from_num(0.75)],
            ]],
            vec![vec![
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ]],
        )
        .expect("fixture should build");
        let token_ids = [0, 1, 2];
        let input = activation_sequence(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
            vec![Act::from_num(0.25), Act::from_num(-0.75)],
        ]);
        let one_row = run_materialized_with_sizing(
            &token_ids,
            &input,
            &fixture.source,
            raster_sizing_with_projection_rows(1),
        )
        .expect("one-row raster PLE should run")
        .expect("one-row PLE inputs");
        let mut two_rows_sizing = raster_sizing_with_projection_rows(1);
        two_rows_sizing.sequence_rows_per_tile = 2;
        let two_rows =
            run_materialized_with_sizing(&token_ids, &input, &fixture.source, two_rows_sizing)
                .expect("two-row raster PLE should run")
                .expect("two-row PLE inputs");

        assert_eq!(one_row.per_layer_inputs, two_rows.per_layer_inputs);
    }

    #[test]
    fn prefill_ple_state_serializes_refs_not_activation_rows() {
        let fixture = PleFixture::single_layer().expect("fixture should build");
        let token_ids = [0, 1];
        let input = activation_sequence(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);
        let input_roots = prepare_raster_prefill_ple_input_roots(
            &token_ids,
            &input,
            &fixture.source,
            raster_sizing_with_projection_rows(1),
        )
        .expect("prepare input roots");
        let state = init_prefill_ple_state(input_roots).expect("init state");

        let encoded = serde_json::to_string(&state).expect("serialize initial state");
        assert!(encoded.contains("token_ids_artifact_root"));
        assert!(encoded.contains("input_activations_ref"));
        assert!(encoded.contains("per_layer_inputs"));
        assert!(!encoded.contains("[0,1]"));
        assert!(!encoded.contains("act_bits"));

        let (_complete, state) =
            compute_next_prefill_ple_layer_sequence(state, &fixture.source).expect("compute layer");
        let encoded = serde_json::to_string(&state).expect("serialize computed state");
        assert!(encoded.contains("prefill.prepare_aux.input.initial"));
        assert!(encoded.contains("prefill.prepare_aux.per_layer_input.0"));
        assert!(!encoded.contains("act_bits"));

        let finalized = finalize_prefill_ple_inputs(state)
            .expect("finalize")
            .expect("PLE inputs");
        let native = crate::shared::transformer_kernels::compute_prefill_ple_inputs_internal(
            &token_ids,
            input.clone_internal(),
            &fixture.layers,
            &fixture.native_ple_global,
            0.0,
            Some(f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .expect("native PLE should run");
        assert_eq!(finalized.per_layer_inputs, native.per_layer_inputs);
    }

    #[test]
    fn prefill_ple_ref_manifest_serializes_refs_not_activation_rows() {
        let fixture = PleFixture::single_layer().expect("fixture should build");
        let token_ids = [0, 1];
        let input = activation_sequence(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);
        let refs = run(
            &token_ids,
            &input,
            &fixture.source,
            raster_sizing_with_projection_rows(1),
        )
        .expect("raster PLE refs should run")
        .expect("PLE refs");

        let encoded = serde_json::to_string(&refs).expect("serialize refs");
        assert!(encoded.contains("prefill.prepare_aux.per_layer_input.0"));
        assert!(encoded.contains("per_layer_inputs"));
        assert!(!encoded.contains("act_bits"));

        let materialized = materialize_prefill_ple_input_refs(Some(&refs))
            .expect("materialize refs")
            .expect("PLE inputs");
        let native = crate::shared::transformer_kernels::compute_prefill_ple_inputs_internal(
            &token_ids,
            input.clone_internal(),
            &fixture.layers,
            &fixture.native_ple_global,
            0.0,
            Some(f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .expect("native PLE should run");
        assert_eq!(materialized.per_layer_inputs, native.per_layer_inputs);
    }

    #[test]
    fn missing_input_ref_fails_closed() {
        let fixture = PleFixture::single_layer().expect("fixture should build");
        let token_ids = [0];
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);
        let input_roots = prepare_raster_prefill_ple_input_roots(
            &token_ids,
            &input,
            &fixture.source,
            raster_sizing_with_projection_rows(1),
        )
        .expect("prepare input roots");
        let mut state = init_prefill_ple_state(input_roots).expect("init state");
        state.input_activations_ref = None;

        let error = compute_next_prefill_ple_layer_sequence(state, &fixture.source)
            .expect_err("missing input ref should fail");

        assert!(error
            .to_string()
            .contains("raster PLE state is missing input activation ref"));
    }

    #[test]
    fn mixed_ple_and_non_ple_layers_match_native_layout() {
        let fixture = PleFixture::new(
            vec![true, false],
            vec![
                vec![
                    vec![Act::from_num(0.25), Act::from_num(-0.5)],
                    vec![Act::from_num(1.0), Act::from_num(0.5)],
                ],
                vec![
                    vec![Act::from_num(0.0), Act::from_num(0.0)],
                    vec![Act::from_num(0.0), Act::from_num(0.0)],
                ],
            ],
            vec![
                vec![
                    vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                    vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
                ],
                vec![
                    vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
                    vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
                ],
            ],
        )
        .expect("fixture should build");
        let token_ids = [0];
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let (raster, native) = compare_raster_and_native(&fixture, &token_ids, &input);

        assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
        assert_eq!(raster.per_layer_inputs.len(), 2);
        assert!(raster.per_layer_inputs[0].is_some());
        assert!(raster.per_layer_inputs[1].is_none());
        assert!(raster
            .clone_layer_internal(0)
            .expect("layer 0")
            .det_values()
            .is_some());
        assert!(raster.clone_layer_internal(1).is_none());
        assert!(native.clone_layer_internal(1).is_none());
    }

    #[test]
    fn empty_activation_sequence_fails_like_native_ple_computation() {
        let fixture = PleFixture::single_layer().expect("fixture should build");
        let input = activation_sequence(Vec::new());

        let error = run_materialized(&[], &input, &fixture.source, 1)
            .expect_err("empty activations should fail");

        assert!(error
            .to_string()
            .contains("requires at least one activation row"));
    }

    #[test]
    fn token_and_activation_count_mismatch_fails_clearly() {
        let fixture = PleFixture::single_layer().expect("fixture should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let error = run_materialized(&[0, 1], &input, &fixture.source, 1)
            .expect_err("mismatch should fail");

        assert!(error
            .to_string()
            .contains("token ids and activations to have matching lengths"));
    }

    #[test]
    fn source_construction_reports_token_embedding_layer_mismatch() {
        let error = AuthenticatedGemmaPleSource::from_canonical_parts(
            "bad-token-layers",
            vec![
                GemmaPleLayerConfig {
                    has_ple: true,
                    hidden_width: 2,
                },
                GemmaPleLayerConfig {
                    has_ple: true,
                    hidden_width: 2,
                },
            ],
            vec![vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]],
            vec![identity_projection(), identity_projection()],
            vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
            test_scalars(),
        )
        .expect_err("token embedding layer mismatch should fail");

        assert!(error
            .to_string()
            .contains("token embedding slice count mismatch"));
    }

    #[test]
    fn source_construction_reports_model_projection_layer_mismatch() {
        let error = AuthenticatedGemmaPleSource::from_canonical_parts(
            "bad-projection-layers",
            vec![
                GemmaPleLayerConfig {
                    has_ple: true,
                    hidden_width: 2,
                },
                GemmaPleLayerConfig {
                    has_ple: true,
                    hidden_width: 2,
                },
            ],
            vec![
                vec![vec![Act::from_num(1.0), Act::from_num(0.0)]],
                vec![vec![Act::from_num(0.0), Act::from_num(1.0)]],
            ],
            vec![identity_projection()],
            vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
            test_scalars(),
        )
        .expect_err("projection layer mismatch should fail");

        assert!(error
            .to_string()
            .contains("model projection slice count mismatch"));
    }

    #[test]
    fn unsupported_fp32_only_ple_source_fails_closed() {
        let ple_global = Gemma4PleGlobalWeights::from_materialized(
            vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            }],
            vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            }],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );

        let error = AuthenticatedGemmaPleSource::from_ple_global(
            "fp32-ple",
            Gemma4ModelProvenance::DetNumWgt,
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 2,
            }],
            Some(ple_global),
            Some(f32_to_acc(0.0)),
        )
        .expect_err("fp32-only PLE source should fail");

        assert!(error.to_string().contains(".detwgt token embedding source"));
    }

    struct PleFixture {
        source: AuthenticatedGemmaPleSource,
        native_ple_global: Gemma4PleGlobalWeights,
        layers: Vec<Gemma4LayerWeights>,
        _weights_file: PathBuf,
    }

    impl Drop for PleFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self._weights_file);
        }
    }

    impl PleFixture {
        fn single_layer() -> Result<Self> {
            Self::new(
                vec![true],
                vec![vec![
                    vec![Act::from_num(0.25), Act::from_num(-0.5)],
                    vec![Act::from_num(1.0), Act::from_num(0.5)],
                ]],
                vec![vec![
                    vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                    vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
                ]],
            )
        }

        fn new(
            has_ple_layers: Vec<bool>,
            token_embeddings: Vec<Vec<Vec<Act>>>,
            model_projections: Vec<Vec<Vec<Wgt>>>,
        ) -> Result<Self> {
            let hidden_width = model_projections
                .first()
                .and_then(|layer| layer.first())
                .map(Vec::len)
                .context("fixture requires at least one projection row")?;
            let ple_width = token_embeddings
                .first()
                .and_then(|layer| layer.first())
                .map(Vec::len)
                .context("fixture requires at least one token embedding row")?;
            let layer_configs = has_ple_layers
                .iter()
                .copied()
                .map(|has_ple| GemmaPleLayerConfig {
                    has_ple,
                    hidden_width,
                })
                .collect::<Vec<_>>();
            let source = AuthenticatedGemmaPleSource::from_canonical_parts(
                "ple-raster-fixture",
                layer_configs,
                token_embeddings.clone(),
                model_projections.clone(),
                vec![Wgt::from_num(1.0); ple_width],
                test_scalars(),
            )?;
            let (weights_file, token_sources, projection_sources) =
                write_det_weights(token_embeddings.clone(), model_projections.clone())?;
            let native_ple_global = Gemma4PleGlobalWeights::from_det_num_sources_with_canonical(
                token_sources,
                projection_sources,
                vec![1.0; ple_width],
                vec![Wgt::from_num(1.0); ple_width],
                1.0,
                Act::from_num(1.0),
                1.0,
                Act::from_num(1.0),
                1.0,
                Act::from_num(1.0),
            );
            let layers = has_ple_layers
                .into_iter()
                .map(|has_ple| test_layer(hidden_width, has_ple))
                .collect();

            Ok(Self {
                source,
                native_ple_global,
                layers,
                _weights_file: weights_file,
            })
        }
    }

    fn compare_raster_and_native(
        fixture: &PleFixture,
        token_ids: &[u32],
        input: &ActivationSequence,
    ) -> (Gemma4PrefillPleInputs, Gemma4PrefillPleInputs) {
        let raster = run_materialized(token_ids, input, &fixture.source, 1)
            .expect("raster PLE should run")
            .expect("raster PLE inputs");
        let native = crate::shared::transformer_kernels::compute_prefill_ple_inputs_internal(
            token_ids,
            input.clone_internal(),
            &fixture.layers,
            &fixture.native_ple_global,
            0.0,
            Some(f32_to_acc(0.0)),
            InferenceExecutionMode::Deterministic,
        )
        .expect("native PLE should run");
        (raster, native)
    }

    fn run_materialized(
        token_ids: &[u32],
        input: &ActivationSequence,
        source: &AuthenticatedGemmaPleSource,
        projection_rows_per_tile: usize,
    ) -> Result<Option<Gemma4PrefillPleInputs>> {
        run_materialized_with_sizing(
            token_ids,
            input,
            source,
            raster_sizing_with_projection_rows(projection_rows_per_tile),
        )
    }

    fn run_materialized_with_sizing(
        token_ids: &[u32],
        input: &ActivationSequence,
        source: &AuthenticatedGemmaPleSource,
        raster_sizing: RasterSizingControls,
    ) -> Result<Option<Gemma4PrefillPleInputs>> {
        let refs = run(token_ids, input, source, raster_sizing)?;
        materialize_prefill_ple_input_refs(refs.as_ref())
    }

    fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
        let values = rows
            .iter()
            .map(|row| row.iter().copied().map(act_to_f32).collect())
            .collect::<Vec<Vec<f32>>>();
        ActivationSequence::from_internal(
            InternalActivationSequence::from_det_values(rows),
            crate::shared::transformer_kernels::build_activation_commitment(&values),
        )
    }

    fn test_scalars() -> GemmaPleScalars {
        GemmaPleScalars {
            embedding_scale: Act::from_num(1.0),
            projection_scalar: Act::from_num(1.0),
            input_scale: Act::from_num(1.0),
            rms_norm_eps: f32_to_acc(0.0),
        }
    }

    fn identity_projection() -> Vec<Vec<Wgt>> {
        vec![
            vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
            vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
        ]
    }

    fn write_det_weights(
        token_embeddings: Vec<Vec<Vec<Act>>>,
        model_projections: Vec<Vec<Vec<Wgt>>>,
    ) -> Result<(
        PathBuf,
        Vec<DetNumTensorSliceSource>,
        Vec<DetNumTensorSliceSource>,
    )> {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-prefill-ple-{}-{}-{}.detwgt",
            std::process::id(),
            unique_suffix,
            crate::trace::sha256_hex(&format!("{:?}{:?}", token_embeddings, model_projections))
        ));
        let mut bytes = Vec::new();
        let mut token_sources = Vec::new();
        let mut projection_sources = Vec::new();

        for matrix in &token_embeddings {
            let data_offset = bytes.len();
            for row in matrix {
                for value in row {
                    bytes.extend(value.to_bits().to_le_bytes());
                }
            }
            token_sources.push(det_source(
                &path,
                matrix.len(),
                matrix[0].len(),
                data_offset,
            ));
        }

        for matrix in &model_projections {
            let data_offset = bytes.len();
            for row in matrix {
                for value in row {
                    bytes.extend(value.to_bits().to_le_bytes());
                }
            }
            projection_sources.push(det_source(
                &path,
                matrix.len(),
                matrix[0].len(),
                data_offset,
            ));
        }

        std::fs::write(&path, bytes)
            .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
        Ok((path, token_sources, projection_sources))
    }

    fn det_source(
        path: &std::path::Path,
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

    fn test_layer(hidden_width: usize, has_ple: bool) -> Gemma4LayerWeights {
        Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Full,
            hidden_size: hidden_width,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden_width,
            sliding_window: None,
            cache_sliding_window: None,
            rms_norm_eps: 0.0,
            rms_norm_eps_det: Some(f32_to_acc(0.0)),
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_layer_matrix(hidden_width),
            k_proj: zero_layer_matrix(hidden_width),
            v_proj: Some(zero_layer_matrix(hidden_width)),
            o_proj: zero_layer_matrix(hidden_width),
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
            gate_proj: zero_layer_matrix(hidden_width),
            up_proj: zero_layer_matrix(hidden_width),
            down_proj: zero_layer_matrix(hidden_width),
            ple: has_ple.then(|| Gemma4PleLayerWeights {
                input_gate: zero_layer_matrix(hidden_width),
                layer_projection: zero_layer_matrix(hidden_width),
                post_input_norm_weight: vec![1.0; hidden_width],
                post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            }),
            layer_scalar: None,
            layer_scalar_det: None,
        }
    }

    fn no_ple_model(hidden_width: usize) -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::DetNumWgt,
            embedding_table: None,
            embedding_source: None,
            layers: vec![test_layer(hidden_width, false)],
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
            rms_norm_eps_det: Some(f32_to_acc(0.0)),
        }
    }

    fn zero_layer_matrix(width: usize) -> Gemma4LayerMatrixSource {
        Gemma4LayerMatrixSource::from(MatrixF32 {
            rows: width,
            cols: width,
            values: vec![0.0; width * width],
        })
    }
}
