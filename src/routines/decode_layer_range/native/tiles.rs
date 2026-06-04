use anyhow::{anyhow, bail, Result};

use crate::routines::decode_layer_range::{
    activation_state_from_row, embed_decode_token, layer_range_width, trace_checkpoint,
    DecodeLayerRangeState,
};
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{
    Gemma4LayerWeights, Gemma4TransformerModel, LayerKvCache, TransformerDecodeState,
};
use crate::trace::trace_scope;

pub(crate) fn init_state_with_mode(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<DecodeLayerRangeState> {
    model.validate_execution_mode(execution_mode)?;
    let decode_input = embed_decode_token(next_token, model, execution_mode)?;
    DecodeLayerRangeState::new(
        decode_input,
        next_token,
        transformer_decode_state,
        model.layers.len(),
    )
}

pub(crate) fn run_range_with_mode(
    mut state: DecodeLayerRangeState,
    model: &Gemma4TransformerModel,
    decode_layer_range_width: usize,
    execution_mode: InferenceExecutionMode,
    execution_mode_label: Option<&str>,
) -> Result<(DecodeLayerRangeState, bool)> {
    model.validate_execution_mode(execution_mode)?;
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

    while state.next_layer_idx < layer_end {
        let layer_idx = state.next_layer_idx;
        let layer = &model.layers[layer_idx];
        let cache = state
            .original_layer_caches
            .get(layer_idx)
            .cloned()
            .ok_or_else(|| anyhow!("decode layer cache {layer_idx} missing"))?;
        let _trace = trace_scope(format!(
            "decode.layer_range layer={layer_idx} token={} position={} attention={:?} ple={} donor={:?}",
            state.next_token,
            state.position,
            layer.attention_kind,
            layer.ple.is_some(),
            layer.kv_shared_layer_index
        ));
        let per_layer_input =
            crate::shared::numerics::transformer_kernels::compute_decode_ple_input_internal(
                state.next_token,
                state.decode_input.clone(),
                layer_idx,
                layer,
                model.ple_global.as_ref(),
                model.rms_norm_eps,
                model.rms_norm_eps_det,
                execution_mode,
            )?;
        let donor_cache =
            resolve_decode_donor_cache(layer, &state.updated_layer_caches, layer_idx)?;
        let resolved_layer = crate::io::resolve_layer_weights(layer)?;
        let (layer_output, updated_cache) =
            crate::shared::numerics::transformer_kernels::run_gemma4_layer_decode_with_mode_internal(
                state.current_activation,
                &resolved_layer,
                per_layer_input,
                cache,
                donor_cache,
                state.position,
                execution_mode,
            )?;
        state.current_activation = layer_output;
        let activation_state = activation_state_from_row(&state.current_activation);
        state
            .completed_layer_output_sha256s
            .push(activation_state.activations_sha256.clone());
        state
            .completed_layer_output_det_sha256s
            .push(activation_state.det_activations_sha256.clone());
        state.updated_layer_caches.push(updated_cache);
        state.next_layer_idx += 1;
    }

    let reached_terminal = trace_checkpoint(&state, layer_start, execution_mode_label)?;
    Ok((state, reached_terminal))
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
