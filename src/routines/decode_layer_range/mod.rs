use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::runtime::checkpoints::RoutineId;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::api::output::DecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, read_token_id_from_ref_roots, token_id_leaf,
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4TransformerModel, InternalActivationRow, InternalActivationSequence,
    InternalLogits, LayerKvCache, PrefillLogits, TransformerDecodeState,
};
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::raster_kernels::transformer::RasterActivationRow;
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, read_sequence_row_from_roots,
    RasterActivationSequenceRef, RasterSequenceRowRequest, RasterTensorId,
};
use crate::RasterSizingControls;

pub mod native;
pub mod raster;

use self::raster::auth_source::AuthenticatedGemmaDecodeLayerRangeSource;

pub(crate) fn init_raster_state_from_decode_loop(
    decode_state: RasterDecodeLoopState,
    selected_token_ref: RasterSelectedTokenRef,
    source: &AuthenticatedGemmaDecodeLayerRangeSource,
    raster_sizing: RasterSizingControls,
) -> Result<raster::RasterDecodeLayerRangeState> {
    let source =
        raster::auth_source::RasterDecodeLayerRangeSource::for_current_integrity_mode(source)?;
    let output_source_prefix = format!("decode.layer_range.position_{}", decode_state.position);
    let next_token =
        crate::shared::artifacts::raster_artifact_store::read_selected_token_from_roots(
            &decode_state.artifact_store_roots,
            &selected_token_ref,
        )?;
    let (_artifact_store_roots, state) =
        raster::init_decode_layer_range_state_from_refs_with_roots(
            decode_state.artifact_store_roots,
            decode_state.position,
            decode_state.token_count,
            decode_state.layer_caches,
            next_token,
            &source,
            raster_sizing,
            output_source_prefix,
        )?;
    Ok(state)
}

pub fn run_raster(
    state: raster::RasterDecodeLayerRangeState,
    source: &AuthenticatedGemmaDecodeLayerRangeSource,
    decode_layer_range_width: usize,
) -> Result<raster::RasterDecodeLayerRangeState> {
    let source =
        raster::auth_source::RasterDecodeLayerRangeSource::for_current_integrity_mode(source)?;
    run_raster_with_source(state, &source, decode_layer_range_width)
}

pub(crate) fn run_selected_raster_detour_from_native_boundary(
    state: DecodeLayerRangeState,
    source: &AuthenticatedGemmaDecodeLayerRangeSource,
    raster_sizing: RasterSizingControls,
) -> Result<DecodeLayerRangeState> {
    let source_for_mode =
        raster::auth_source::RasterDecodeLayerRangeSource::for_current_integrity_mode(source)?;
    let raster_state = raster_state_from_native_state(state, raster_sizing)?;
    let raster_state = run_raster_with_source(
        raster_state,
        &source_for_mode,
        raster_sizing.decode_layer_range_width,
    )?;
    native_state_from_raster_state(raster_state)
}

fn run_raster_with_source(
    mut state: raster::RasterDecodeLayerRangeState,
    source: &raster::auth_source::RasterDecodeLayerRangeSource<'_>,
    decode_layer_range_width: usize,
) -> Result<raster::RasterDecodeLayerRangeState> {
    if state.is_complete() {
        return Ok(state);
    }
    let layer_start = state.next_layer_idx();
    let layer_limit = layer_start
        .saturating_add(layer_range_width(
            decode_layer_range_width,
            state.layer_count(),
        ))
        .min(state.layer_count());
    while state.next_layer_idx() < layer_limit {
        let roots = state.artifact_store_roots().clone();
        let (done, _roots, next_state) =
            raster::compute_next_decode_layer_with_roots(roots, state, source)?;
        state = next_state;
        if done {
            break;
        }
    }
    raster::utils::trace_raster_checkpoint(&state, layer_start)?;
    Ok(state)
}

pub(crate) fn raster_state_from_native_state(
    state: DecodeLayerRangeState,
    raster_sizing: RasterSizingControls,
) -> Result<raster::RasterDecodeLayerRangeState> {
    let output_source_prefix = format!(
        "decode.layer_range.detour.position_{}.layer_{}",
        state.position, state.next_layer_idx
    );
    let mut roots = ArtifactIo::export_store_roots();
    let decode_input = raster_row_from_internal(&state.decode_input, "decode layer range input")?;
    let (next_roots, decode_input_ref) = raster::insert_decode_activation_row_with_roots(
        &roots,
        format!("{output_source_prefix}.input.selected_token_embedding"),
        &decode_input,
    )?;
    roots = next_roots;
    let current_activation =
        raster_row_from_internal(&state.current_activation, "decode layer range activation")?;
    let (next_roots, current_activation_ref) = raster::insert_decode_activation_row_with_roots(
        &roots,
        format!("{output_source_prefix}.input.current_activation"),
        &current_activation,
    )?;
    roots = next_roots;
    let original_layer_caches = insert_layer_cache_slots(
        roots,
        &state.original_layer_caches,
        &format!("{output_source_prefix}.original.cache"),
    )?;
    roots = original_layer_caches.0;
    let original_layer_caches = original_layer_caches.1;
    let updated_layer_caches = insert_layer_cache_slots(
        roots,
        &state.updated_layer_caches,
        &format!("{output_source_prefix}.updated.cache"),
    )?;
    roots = updated_layer_caches.0;
    let updated_layer_caches = updated_layer_caches.1;

    Ok(raster::RasterDecodeLayerRangeState::from_parts(
        roots,
        decode_input_ref,
        current_activation_ref,
        state.next_token,
        state.position,
        state.token_count,
        state.next_layer_idx,
        state.layer_count,
        original_layer_caches,
        updated_layer_caches,
        state.completed_layer_output_sha256s,
        state.completed_layer_output_det_sha256s,
        raster_sizing.projection_rows_per_tile,
        raster_sizing.attention_kv_rows_per_tile,
        output_source_prefix,
    ))
}

pub(crate) fn native_state_from_raster_state(
    raster_state: raster::RasterDecodeLayerRangeState,
) -> Result<DecodeLayerRangeState> {
    let current_activation = raster::materialize_activation_sequence_from_ref(
        raster_state.artifact_store_roots(),
        raster_state.current_activation_ref(),
    )?
    .clone_internal()
    .last_row()
    .ok_or_else(|| anyhow!("raster decode layer range returned no activation row"))?;
    let decode_input = raster::materialize_activation_sequence_from_ref(
        raster_state.artifact_store_roots(),
        raster_state.decode_input_ref(),
    )?
    .clone_internal()
    .last_row()
    .ok_or_else(|| anyhow!("raster decode layer range returned no decode input row"))?;
    let updated_layer_caches = raster::materialize_decode_layer_caches_from_roots(
        raster_state.artifact_store_roots(),
        raster_state.updated_layer_caches(),
    )?;
    let effective_layer_caches = raster::materialize_decode_layer_caches_from_roots(
        raster_state.artifact_store_roots(),
        &raster_state.effective_layer_caches(),
    )?;

    Ok(DecodeLayerRangeState {
        decode_input,
        current_activation,
        next_token: raster_state.next_token(),
        position: raster_state.position(),
        token_count: raster_state.token_count(),
        next_layer_idx: raster_state.next_layer_idx(),
        layer_count: raster_state.layer_count(),
        original_layer_caches: effective_layer_caches,
        updated_layer_caches,
        completed_layer_output_sha256s: raster_state.completed_layer_output_sha256s().to_vec(),
        completed_layer_output_det_sha256s: raster_state
            .completed_layer_output_det_sha256s()
            .to_vec(),
    })
}

fn insert_layer_cache_slots(
    mut roots: crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots,
    layer_caches: &[LayerKvCache],
    source_name_prefix: &str,
) -> Result<(
    crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots,
    Vec<raster::DecodeLayerCacheSlot>,
)> {
    let mut slots = Vec::with_capacity(layer_caches.len());
    for (layer_idx, cache) in layer_caches.iter().enumerate() {
        let raster_cache = raster::raster_cache_from_layer_cache(cache)?;
        let (next_roots, slot) = raster::register_decode_layer_cache_with_roots(
            &roots,
            source_name_prefix,
            layer_idx,
            raster_cache,
        )?;
        roots = next_roots;
        slots.push(slot);
    }
    Ok((roots, slots))
}

fn raster_row_from_internal(
    row: &InternalActivationRow,
    label: &str,
) -> Result<RasterActivationRow> {
    let acts = row
        .det_values()
        .ok_or_else(|| anyhow!("raster {label} requires canonical deterministic activations"))?;
    Ok(RasterActivationRow::from_acts(acts.to_vec()))
}

pub(crate) fn prepare_raster_decode_loop_state_from_native(
    decode_state: &DecodeState,
    source_prefix: &str,
) -> Result<RasterDecodeLoopState> {
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let (artifact_store_roots, full_token_ids_ref) = insert_decode_token_ids_artifact_with_roots(
        artifact_store_roots,
        format!("{source_prefix}.input.full_token_ids"),
        &decode_state.full_token_ids,
    )?;
    let (artifact_store_roots, generated_token_ids_ref) =
        insert_decode_token_ids_artifact_with_roots(
            artifact_store_roots,
            format!("{source_prefix}.input.generated_token_ids"),
            &decode_state.generated_token_ids,
        )?;
    let (artifact_store_roots, logits_ref, logit_count) = insert_decode_logits_artifact_with_roots(
        artifact_store_roots,
        format!("{source_prefix}.input.logits"),
        decode_state,
        "raster decode transition finalize",
    )?;
    let (artifact_store_roots, layer_caches) = insert_decode_layer_cache_refs_from_native(
        artifact_store_roots,
        &decode_state.transformer_decode_state.layer_caches,
        &format!("{source_prefix}.input.layer_cache"),
    )?;

    RasterDecodeLoopState::new(
        artifact_store_roots,
        full_token_ids_ref,
        decode_state.full_token_ids.len(),
        generated_token_ids_ref,
        decode_state.generated_token_ids.len(),
        logits_ref,
        logit_count,
        layer_caches,
        decode_state.transformer_decode_state.position,
        decode_state.transformer_decode_state.token_count,
        None,
    )
}

pub(crate) fn insert_decode_token_ids_artifact_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    source_name: String,
    token_ids: &[u32],
) -> Result<(RasterArtifactStoreRoots, Option<RasterTokenIdSequenceRef>)> {
    if token_ids.is_empty() {
        return Ok((artifact_store_roots, None));
    }
    let leaves = token_ids.iter().copied().map(token_id_leaf).collect();
    let (artifact_store_roots, token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?;
    Ok((
        artifact_store_roots,
        Some(RasterTokenIdSequenceRef::new(token_ids_ref)?),
    ))
}

pub(crate) fn insert_decode_logits_artifact_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    source_name: String,
    decode_state: &DecodeState,
    label: &str,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef, usize)> {
    let logits = decode_state.clone_internal_logits();
    let det_logits = logits
        .det_values()
        .ok_or_else(|| anyhow!("{label} requires canonical deterministic logits"))?;
    if det_logits.is_empty() {
        bail!("{label} requires at least one canonical logit");
    }

    let leaves = det_logits
        .iter()
        .map(|logit| activation_row_leaf(&RasterActivationRow::from_acts(vec![*logit])))
        .collect::<Vec<_>>();
    let (artifact_store_roots, logits_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(source_name.clone())?,
        RasterArtifactMetadata::activation_rows(det_logits.len(), 1)?,
        leaves,
    )?;
    let logits_ref = activation_sequence_ref_from_artifact(
        RasterTensorId::new(source_name)?,
        RasterActivationSequenceArtifactRef::new(logits_artifact_ref)?,
    )?;
    Ok((artifact_store_roots, logits_ref, det_logits.len()))
}

pub(crate) fn insert_decode_layer_cache_refs_from_native(
    artifact_store_roots: RasterArtifactStoreRoots,
    layer_caches: &[LayerKvCache],
    source_name_prefix: &str,
) -> Result<(RasterArtifactStoreRoots, Vec<raster::DecodeLayerCacheSlot>)> {
    insert_layer_cache_slots(artifact_store_roots, layer_caches, source_name_prefix)
}

pub(crate) fn materialize_decode_state_from_raster_state_for_trace(
    decode_state: &RasterDecodeLoopState,
) -> Result<DecodeState> {
    let full_token_ids = materialize_token_ids_from_optional_ref(
        &decode_state.artifact_store_roots,
        decode_state.full_token_ids_ref.as_ref(),
    )?;
    let generated_token_ids = materialize_token_ids_from_optional_ref(
        &decode_state.artifact_store_roots,
        decode_state.generated_token_ids_ref.as_ref(),
    )?;
    let internal_logits = materialize_internal_logits_from_ref(
        &decode_state.artifact_store_roots,
        &decode_state.current_logits_ref,
    )?;
    let layer_caches = raster::materialize_decode_layer_caches_from_roots(
        &decode_state.artifact_store_roots,
        &decode_state.layer_caches,
    )?;
    let mut materialized = DecodeState::new(
        full_token_ids,
        internal_logits.clone_f32(),
        TransformerDecodeState {
            layer_caches,
            position: decode_state.position,
            token_count: decode_state.token_count,
        },
    );
    materialized.generated_token_ids = generated_token_ids;
    materialized.set_internal_logits(internal_logits);
    Ok(materialized)
}

fn materialize_token_ids_from_optional_ref(
    roots: &RasterArtifactStoreRoots,
    token_ids_ref: Option<&RasterTokenIdSequenceRef>,
) -> Result<Vec<u32>> {
    let Some(token_ids_ref) = token_ids_ref else {
        return Ok(Vec::new());
    };
    (0..token_ids_ref.token_count())
        .map(|token_idx| read_token_id_from_ref_roots(roots, token_ids_ref, token_idx))
        .collect()
}

fn materialize_internal_logits_from_ref(
    roots: &RasterArtifactStoreRoots,
    logits_ref: &RasterActivationSequenceRef,
) -> Result<InternalLogits> {
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    let det_logits = match (row_count, width) {
        (_, 1) => (0..row_count)
            .map(|row_idx| {
                let row = read_sequence_row_from_roots(
                    roots,
                    RasterSequenceRowRequest {
                        tensor_ref: logits_ref.clone(),
                        row_idx,
                    },
                )?;
                row.acts()
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("raster logits row {row_idx} is empty"))
            })
            .collect::<Result<Vec<_>>>()?,
        (1, _) => read_sequence_row_from_roots(
            roots,
            RasterSequenceRowRequest {
                tensor_ref: logits_ref.clone(),
                row_idx: 0,
            },
        )?
        .acts(),
        _ => bail!("raster logits shape {row_count}x{width} must be Nx1 or 1xN"),
    };
    Ok(InternalLogits::from_det_values(det_logits))
}

pub(crate) fn prefill_logits_from_internal(internal_logits: InternalLogits) -> PrefillLogits {
    let det_final_logits_sha256 = internal_logits
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_vector_commitment);
    if det_final_logits_sha256.is_some() {
        // Deterministic logits carry only the canonical commitment (spec v1).
        return PrefillLogits::from_det_internal(internal_logits, det_final_logits_sha256);
    }
    let final_logits_sha256 = crate::shared::numerics::transformer_kernels::build_vector_commitment(
        internal_logits.as_f32_slice(),
    );
    PrefillLogits::from_internal(internal_logits, final_logits_sha256)
}

#[derive(Debug, Clone)]
pub(crate) struct DecodeLayerRangeState {
    pub(crate) decode_input: InternalActivationRow,
    pub(crate) current_activation: InternalActivationRow,
    pub(crate) next_token: u32,
    pub(crate) position: usize,
    pub(crate) token_count: usize,
    pub(crate) next_layer_idx: usize,
    pub(crate) layer_count: usize,
    pub(crate) original_layer_caches: Vec<LayerKvCache>,
    pub(crate) updated_layer_caches: Vec<LayerKvCache>,
    pub(crate) completed_layer_output_sha256s: Vec<String>,
    pub(crate) completed_layer_output_det_sha256s: Vec<Option<String>>,
}

impl DecodeLayerRangeState {
    pub(crate) fn new(
        decode_input: InternalActivationRow,
        next_token: u32,
        transformer_decode_state: TransformerDecodeState,
        layer_count: usize,
    ) -> Result<Self> {
        if layer_count == 0 {
            bail!("transformer decode requires at least one layer");
        }
        if transformer_decode_state.layer_caches.len() != layer_count {
            bail!(
                "transformer decode cache count mismatch: {} vs {}",
                transformer_decode_state.layer_caches.len(),
                layer_count
            );
        }
        Ok(Self {
            current_activation: decode_input.clone(),
            decode_input,
            next_token,
            position: transformer_decode_state.position,
            token_count: transformer_decode_state.token_count,
            next_layer_idx: 0,
            layer_count,
            original_layer_caches: transformer_decode_state.layer_caches,
            updated_layer_caches: Vec::with_capacity(layer_count),
            completed_layer_output_sha256s: Vec::with_capacity(layer_count),
            completed_layer_output_det_sha256s: Vec::with_capacity(layer_count),
        })
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.next_layer_idx >= self.layer_count
    }

    pub(crate) fn activation_state(&self) -> ActivationSequence {
        activation_state_from_row(&self.current_activation)
    }

    pub(crate) fn effective_layer_caches(&self) -> Vec<LayerKvCache> {
        let mut caches = self.updated_layer_caches.clone();
        caches.extend(
            self.original_layer_caches
                .iter()
                .skip(self.updated_layer_caches.len())
                .cloned(),
        );
        caches
    }

    pub(crate) fn completed_layer_caches(&self) -> Result<Vec<LayerKvCache>> {
        if !self.is_complete() {
            bail!(
                "decode transition finalize requires all layers, got {} of {}",
                self.next_layer_idx,
                self.layer_count
            );
        }
        if self.updated_layer_caches.len() != self.layer_count {
            bail!(
                "decode transition finalized with {} caches, expected {}",
                self.updated_layer_caches.len(),
                self.layer_count
            );
        }
        Ok(self.updated_layer_caches.clone())
    }
}

pub(crate) fn layer_range_width(width: usize, layer_count: usize) -> usize {
    width.max(1).min(layer_count.max(1))
}

pub(crate) fn activation_state_from_row(row: &InternalActivationRow) -> ActivationSequence {
    match row.det_values() {
        Some(det_values) => {
            // Deterministic rows carry only the canonical commitment (spec v1).
            let internal = InternalActivationSequence::from_det_values(vec![det_values.to_vec()]);
            let det_activations_sha256 = internal
                .det_values()
                .map(crate::shared::numerics::transformer_kernels::build_det_activation_commitment);
            ActivationSequence::from_det_internal(internal, det_activations_sha256)
        }
        None => {
            let values = row.clone_f32();
            ActivationSequence::from_internal(
                InternalActivationSequence::from_values(vec![values.clone()]),
                crate::shared::numerics::transformer_kernels::build_activation_commitment(&[
                    values,
                ]),
            )
        }
    }
}

pub(crate) fn trace_checkpoint(
    state: &DecodeLayerRangeState,
    layer_start: usize,
    execution_mode: Option<&str>,
) -> Result<bool> {
    let _routine = crate::trace::routine_scope(
        RoutineId::DecodeLayerRange,
        format!(
            "{}layers={layer_start}..{} position={}",
            execution_mode
                .map(|mode| format!("mode={mode} "))
                .unwrap_or_default(),
            state.next_layer_idx,
            state.position
        ),
    );
    Ok(crate::trace::trace_checkpoint_lazy(
        "decode.layer_range",
        || {
            decode_layer_range_checkpoint_json(
                state.next_token,
                state.position,
                state.token_count,
                layer_start,
                state.next_layer_idx,
                state.layer_count,
                &state.activation_state(),
                &state.effective_layer_caches(),
                state.completed_layer_output_sha256s.clone(),
                Some(state.completed_layer_output_det_sha256s.clone()),
                execution_mode,
            )
        },
    ))
}

pub(crate) fn trace_checkpoint_payload(
    next_token: u32,
    position: usize,
    token_count: usize,
    layer_start: usize,
    layer_end: usize,
    layer_count: usize,
    current_activation: &ActivationSequence,
    layer_caches: &[LayerKvCache],
    completed_layer_output_sha256s: Vec<String>,
    completed_layer_output_det_sha256s: Option<Vec<Option<String>>>,
    execution_mode: Option<&str>,
) -> Result<bool> {
    let payload = decode_layer_range_checkpoint_json(
        next_token,
        position,
        token_count,
        layer_start,
        layer_end,
        layer_count,
        current_activation,
        layer_caches,
        completed_layer_output_sha256s,
        completed_layer_output_det_sha256s,
        execution_mode,
    );
    Ok(crate::trace::trace_checkpoint(
        "decode.layer_range",
        &payload,
    ))
}

#[allow(clippy::too_many_arguments)]
fn decode_layer_range_checkpoint_json(
    next_token: u32,
    position: usize,
    token_count: usize,
    layer_start: usize,
    layer_end: usize,
    layer_count: usize,
    current_activation: &ActivationSequence,
    layer_caches: &[LayerKvCache],
    completed_layer_output_sha256s: Vec<String>,
    completed_layer_output_det_sha256s: Option<Vec<Option<String>>>,
    execution_mode: Option<&str>,
) -> serde_json::Value {
    let deterministic = execution_mode == Some("deterministic");
    let current_internal = current_activation.clone_internal();
    let mut payload = json!({
        "next_token": next_token,
        "decode_position": position,
        "decode_token_count": token_count,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "layer_count": layer_count,
        "det_current_activation_sha256": current_internal
            .det_values()
            .map(crate::shared::numerics::transformer_kernels::build_det_activation_commitment),
        "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(layer_caches),
    });
    if !deterministic {
        // Deterministic-mode payloads carry only canonical commitments
        // (spec v1); fp32 mode keeps the compatibility fields.
        payload["current_activation"] = json!(current_activation.activations.clone());
        payload["current_activation_sha256"] = json!(current_activation.activations_sha256);
        payload["layer_caches"] = json!(crate::trace::serialize_layer_caches(layer_caches));
        payload["completed_layer_output_sha256s"] = json!(completed_layer_output_sha256s);
    }
    if let Some(execution_mode) = execution_mode {
        payload["execution_mode"] = json!(execution_mode);
    }
    if let Some(completed_layer_output_det_sha256s) = completed_layer_output_det_sha256s {
        payload["completed_layer_output_det_sha256s"] = json!(completed_layer_output_det_sha256s);
    }
    payload
}

pub(crate) fn embed_decode_token(
    next_token: u32,
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<InternalActivationRow> {
    let embedded_token = if let Some(ref embedding_table) = model.embedding_table {
        crate::shared::numerics::transformer_kernels::embed_input_tokens_with_mode(
            &[next_token],
            embedding_table,
            execution_mode,
        )?
    } else if let Some(ref embedding_source) = model.embedding_source {
        crate::io::embed_input_tokens_from_gemma_source_with_mode(
            &[next_token],
            embedding_source,
            execution_mode,
        )?
    } else {
        bail!("transformer state model is missing both embedding_table and embedding_source")
    };
    embedded_token
        .clone_internal()
        .last_row()
        .ok_or_else(|| anyhow!("transformer embedding returned no activation rows"))
}
