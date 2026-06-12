use anyhow::{anyhow, bail, Result};

use crate::routines::decode_layer_range::{
    embed_decode_token, layer_range_width, trace_checkpoint, DecodeLayerRangeState,
};
use crate::shared::model::transformer::{
    Gemma4LayerWeights, Gemma4TransformerModel, InternalActivationRow, LayerKvCache,
    TransformerDecodeState,
};
use crate::shared::numerics::det_kernels::{
    det_decode_ple_input, det_layer_decode, row_from_internal, DetDecodeScratch,
};
use crate::shared::numerics::det_num::Act;
use crate::shared::numerics::transformer_kernels::build_det_activation_commitment_row;
use crate::trace::trace_scope;

pub(crate) fn init_state(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
) -> Result<DecodeLayerRangeState> {
    let decode_input = embed_decode_token(next_token, model)?;
    DecodeLayerRangeState::new(
        decode_input,
        next_token,
        transformer_decode_state,
        model.layers.len(),
    )
}

pub(crate) fn run_range(
    mut state: DecodeLayerRangeState,
    model: &Gemma4TransformerModel,
    decode_layer_range_width: usize,
    execution_mode_label: Option<&str>,
) -> Result<(DecodeLayerRangeState, bool)> {
    if state.layer_count != model.layers.len() {
        bail!(
            "decode layer range state has {} layers, model has {}",
            state.layer_count,
            model.layers.len()
        );
    }
    if state.is_complete() {
        return Ok((state, false));
    }

    let layer_start = state.next_layer_idx;
    let layer_end = layer_start
        .saturating_add(layer_range_width(
            decode_layer_range_width,
            state.layer_count,
        ))
        .min(state.layer_count);

    run_layers_det(&mut state, model, layer_end)?;

    let reached_terminal = trace_checkpoint(&state, layer_start, execution_mode_label)?;
    Ok((state, reached_terminal))
}

fn run_layers_det(
    state: &mut DecodeLayerRangeState,
    model: &Gemma4TransformerModel,
    layer_end: usize,
) -> Result<()> {
    let mut scratch = DetDecodeScratch::new();
    let decode_input = row_from_internal(&state.decode_input)?;
    let mut xs = row_from_internal(&state.current_activation)?;
    let mut next_xs: Vec<Act> = Vec::new();
    let mut ple_row: Vec<Act> = Vec::new();

    while state.next_layer_idx < layer_end {
        let layer_idx = state.next_layer_idx;
        let layer = &model.layers[layer_idx];
        let _trace = trace_scope(format!(
            "decode.layer_range layer={layer_idx} token={} position={} attention={:?} ple={} donor={:?}",
            state.next_token,
            state.position,
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        // Move the layer's cache out of the original slot (no clone); slots
        // before `next_layer_idx` are never read again — `effective_layer_caches`
        // only reads original slots at indices >= `updated_layer_caches.len()`.
        let mut cache = state
            .original_layer_caches
            .get_mut(layer_idx)
            .map(std::mem::take)
            .ok_or_else(|| anyhow!("decode layer cache {layer_idx} missing"))?;
        let has_ple = det_decode_ple_input(
            state.next_token,
            &decode_input,
            layer_idx,
            layer,
            model.ple_global.as_ref(),
            model.rms_norm_eps_det,
            &mut ple_row,
        )?;
        let donor_cache =
            resolve_decode_donor_cache(layer, &state.updated_layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        det_layer_decode(
            &xs,
            &resolved_layer,
            has_ple.then_some(ple_row.as_slice()),
            &mut cache,
            donor_cache,
            state.position,
            &mut scratch,
            &mut next_xs,
        )?;
        std::mem::swap(&mut xs, &mut next_xs);
        // Deterministic mode carries only canonical commitments (spec v1).
        state
            .completed_layer_output_det_sha256s
            .push(Some(build_det_activation_commitment_row(&xs)));
        state.updated_layer_caches.push(cache);
        state.next_layer_idx += 1;
    }

    state.current_activation = InternalActivationRow::from_det_values_only(xs);
    Ok(())
}

fn resolve_decode_donor_cache<'a>(
    layer: &Gemma4LayerWeights,
    updated_layer_caches: &'a [LayerKvCache],
    layer_idx: usize,
) -> Result<Option<&'a LayerKvCache>> {
    layer
        .kv_shared_layer_index
        .map(|donor_idx| {
            if donor_idx >= layer_idx {
                bail!(
                    "transformer decode layer {layer_idx} cannot share KV with non-prior donor {donor_idx}"
                );
            }
            updated_layer_caches.get(donor_idx).ok_or_else(|| {
                anyhow!("transformer decode donor cache {donor_idx} missing for layer {layer_idx}")
            })
        })
        .transpose()
}
