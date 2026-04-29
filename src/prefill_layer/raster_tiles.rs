use std::collections::VecDeque;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile_result, call_tile, sequence, tile,
};
use crate::shared::raster_prefill_layer::{
    AuthenticatedGemmaPrefillLayerSource, GemmaPrefillAttentionKind, GemmaPrefillLayerMatrixKind,
    GemmaPrefillLayerMetadataRequest, GemmaPrefillLayerNormKind,
    GemmaPrefillLayerNormWeightsRequest, GemmaPrefillLayerScalarsRequest,
    GemmaPrefillLayerSourceMetadataRequest,
};
use crate::shared::raster_transformer_kernels::{
    add_sequences, apply_rope_to_heads, build_raster_kv_cache, causal_attention_heads_with_cache,
    combine_attention_heads, gelu_sequence, mul_sequences, project_sequence_with_prefill_source,
    reshape_sequence_heads, rms_norm_heads, rms_norm_sequence, scale_sequence,
    value_rms_norm_heads, RasterActivationSequence, RasterKvCache,
};
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, InternalActivationSequence, LayerKvCache,
};
use crate::trace::trace_scope;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillLayerRasterState {
    current_activations: RasterActivationSequence,
    next_layer_idx: usize,
    layer_count: usize,
    layer_caches: Vec<RasterKvCache>,
    per_layer_inputs: Vec<Option<RasterActivationSequence>>,
    completed_layer_output_sha256s: Vec<String>,
    completed_layer_output_det_sha256s: Vec<Option<String>>,
}

#[tile]
pub fn init_prefill_layer_state(
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<PrefillLayerRasterState> {
    let metadata = auth_read!(layer_source, GemmaPrefillLayerSourceMetadataRequest)?;
    if metadata.layer_count == 0 {
        bail!("transformer prefill requires at least one layer");
    }

    let current_activations = raster_activation_sequence_from_activation(input_activations)?;
    if current_activations.is_empty() {
        bail!("transformer layer execution requires at least one activation row");
    }

    let per_layer_inputs = (0..metadata.layer_count)
        .map(|layer_idx| {
            ple_inputs
                .and_then(|inputs| inputs.clone_layer_internal(layer_idx))
                .map(|input| raster_activation_sequence_from_internal(&input))
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(PrefillLayerRasterState {
        current_activations,
        next_layer_idx: 0,
        layer_count: metadata.layer_count,
        layer_caches: Vec::with_capacity(metadata.layer_count),
        per_layer_inputs,
        completed_layer_output_sha256s: Vec::with_capacity(metadata.layer_count),
        completed_layer_output_det_sha256s: Vec::with_capacity(metadata.layer_count),
    })
}

#[tile(kind = recursive)]
pub fn compute_next_prefill_layer(
    mut state: PrefillLayerRasterState,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
) -> Result<(bool, PrefillLayerRasterState)> {
    if state.next_layer_idx >= state.layer_count {
        return Ok((true, state));
    }

    let layer_idx = state.next_layer_idx;
    let layer = auth_read!(layer_source, GemmaPrefillLayerMetadataRequest { layer_idx })?;
    let _trace = trace_scope(format!(
        "prefill.layer.raster layer={layer_idx} tokens={} attention={:?} ple={} donor={:?}",
        state.current_activations.len(),
        layer.attention_kind,
        layer.has_ple,
        layer.kv_shared_layer_index
    ));
    if !layer.has_ple
        && state
            .per_layer_inputs
            .get(layer_idx)
            .and_then(Option::as_ref)
            .is_some()
    {
        bail!("transformer layer received PLE inputs without PLE weights");
    }

    let donor_cache = resolve_prefill_donor_cache(&state.layer_caches, layer_idx, &layer)?;
    let per_layer_input = state
        .per_layer_inputs
        .get(layer_idx)
        .and_then(Option::as_ref);
    let (layer_output, layer_cache) = run_basic_prefill_layer(
        &state.current_activations,
        layer_source,
        &layer,
        donor_cache,
        per_layer_input,
    )?;
    state.current_activations = layer_output;
    state.layer_caches.push(layer_cache);
    state.completed_layer_output_sha256s.push(
        crate::shared::transformer_kernels::build_activation_commitment(
            &state.current_activations.to_f32_values(),
        ),
    );
    state.completed_layer_output_det_sha256s.push(Some(
        crate::shared::transformer_kernels::build_det_activation_commitment(&raster_sequence_acts(
            &state.current_activations,
        )),
    ));
    let current_activations = state.current_activations.to_f32_values();
    let current_det_activations = raster_sequence_acts(&state.current_activations);
    let layer_caches = layer_caches_from_raster(&state.layer_caches);
    crate::trace::trace_checkpoint(
        "prefill.layer",
        &json!({
            "execution_mode": "raster",
            "next_layer_idx": layer_idx + 1,
            "current_activations": current_activations.clone(),
            "current_activations_sha256": crate::shared::transformer_kernels::build_activation_commitment(&current_activations),
            "det_current_activations_sha256": crate::shared::transformer_kernels::build_det_activation_commitment(&current_det_activations),
            "layer_caches": crate::trace::serialize_layer_caches(&layer_caches),
            "det_layer_caches_sha256": crate::shared::transformer_kernels::build_det_kv_cache_commitment(&layer_caches),
            "completed_layer_output_sha256s": state.completed_layer_output_sha256s.clone(),
            "completed_layer_output_det_sha256s": state.completed_layer_output_det_sha256s.clone(),
        }),
    );
    for (token_idx, token_activation) in current_activations.iter().enumerate() {
        let det_token_activation_sha256 = current_det_activations
            .get(token_idx)
            .map(|row| crate::shared::transformer_kernels::build_det_vector_commitment(row));
        crate::trace::trace_checkpoint(
            &format!("prefill.layer_token.layer_{layer_idx}.token_{token_idx}"),
            &json!({
                "execution_mode": "raster",
                "layer_idx": layer_idx,
                "token_idx": token_idx,
                "token_count": current_activations.len(),
                "token_activation": token_activation,
                "det_token_activation_sha256": det_token_activation_sha256,
            }),
        );
    }
    state.next_layer_idx += 1;
    Ok((false, state))
}

#[tile]
pub fn finalize_prefill_layer_state(
    state: PrefillLayerRasterState,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    if state.next_layer_idx != state.layer_count {
        bail!(
            "raster prefill layer finalized after {} layers, expected {}",
            state.next_layer_idx,
            state.layer_count
        );
    }

    let det_activations = raster_sequence_acts(&state.current_activations);
    let values = state.current_activations.to_f32_values();
    let mut activation_sequence = ActivationSequence::from_internal(
        InternalActivationSequence::from_det_values(det_activations.clone()),
        crate::shared::transformer_kernels::build_activation_commitment(&values),
    );
    activation_sequence.det_activations_sha256 =
        Some(crate::shared::transformer_kernels::build_det_activation_commitment(&det_activations));

    Ok((
        activation_sequence,
        state
            .layer_caches
            .into_iter()
            .map(layer_cache_from_raster)
            .collect(),
    ))
}

#[sequence]
pub fn run(
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    let state = call_tile!(
        init_prefill_layer_state,
        input_activations,
        layer_source,
        ple_inputs
    )?;
    let state = call_recur_tile_result!(compute_next_prefill_layer, state, layer_source)?;
    call_tile!(finalize_prefill_layer_state, state)
}

fn run_basic_prefill_layer(
    input: &RasterActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    layer: &crate::shared::raster_prefill_layer::GemmaPrefillLayerMetadata,
    donor_cache: Option<&RasterKvCache>,
    per_layer_input: Option<&RasterActivationSequence>,
) -> Result<(RasterActivationSequence, RasterKvCache)> {
    let scalars = auth_read!(
        layer_source,
        GemmaPrefillLayerScalarsRequest {
            layer_idx: layer.layer_idx,
        },
    )?;

    let residual = input.clone();
    let normed = rms_norm_sequence(
        input,
        Some(&auth_read!(
            layer_source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaPrefillLayerNormKind::InputLayer,
            },
        )?),
        Some(scalars.rms_norm_eps),
    )?;

    let q_projected = project_sequence_with_prefill_source(
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Query,
        layer.q_proj_shape.rows,
    )?;
    let k_projected = project_sequence_with_prefill_source(
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Key,
        layer.k_proj_shape.rows,
    )?;
    let v_projected = if layer.has_v_proj {
        project_sequence_with_prefill_source(
            &normed,
            layer_source,
            layer.layer_idx,
            GemmaPrefillLayerMatrixKind::Value,
            layer
                .v_proj_shape
                .ok_or_else(|| anyhow!("Gemma prefill layer metadata is missing v_proj shape"))?
                .rows,
        )?
    } else if layer.attention_k_eq_v {
        k_projected.clone()
    } else {
        bail!("Gemma prefill layer is missing v_proj without attention_k_eq_v enabled");
    };

    let q_heads = reshape_sequence_heads(&q_projected, layer.num_heads, layer.head_dim)?;
    let k_heads = reshape_sequence_heads(&k_projected, layer.num_kv_heads, layer.head_dim)?;
    let q_heads = rms_norm_heads(
        &q_heads,
        Some(&auth_read!(
            layer_source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaPrefillLayerNormKind::Query,
            },
        )?),
        Some(scalars.rms_norm_eps),
    )?;
    let k_heads = rms_norm_heads(
        &k_heads,
        Some(&auth_read!(
            layer_source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaPrefillLayerNormKind::Key,
            },
        )?),
        Some(scalars.rms_norm_eps),
    )?;
    let v_heads = reshape_sequence_heads(&v_projected, layer.num_kv_heads, layer.head_dim)?;
    let v_heads = value_rms_norm_heads(&v_heads, Some(scalars.rms_norm_eps))?;
    let q_heads = apply_rope_to_heads(
        &q_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        0,
    )?;
    let k_heads = apply_rope_to_heads(
        &k_heads,
        layer.partial_rotary_dim,
        layer.rope_freq_base_dim,
        scalars.rope_base,
        0,
    )?;

    let layer_cache = if donor_cache.is_some() {
        RasterKvCache::empty(layer.num_kv_heads)
    } else {
        build_raster_kv_cache(&k_heads, &v_heads, layer.cache_sliding_window)?
    };
    let attention_window = match layer.attention_kind {
        GemmaPrefillAttentionKind::Full => None,
        GemmaPrefillAttentionKind::Sliding => Some(
            layer
                .sliding_window
                .ok_or_else(|| anyhow!("sliding attention layer is missing a sliding window"))?,
        ),
    };
    let attention_heads = causal_attention_heads_with_cache(
        &q_heads,
        &k_heads,
        &v_heads,
        donor_cache,
        attention_window,
    )?;
    let attention_sequence = combine_attention_heads(&attention_heads)?;
    let attention_output = project_sequence_with_prefill_source(
        &attention_sequence,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Output,
        layer.o_proj_shape.rows,
    )?;
    let attention_output = rms_norm_sequence(
        &attention_output,
        Some(&auth_read!(
            layer_source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaPrefillLayerNormKind::PostAttention,
            },
        )?),
        Some(scalars.rms_norm_eps),
    )?;
    let xs = add_sequences(&residual, &attention_output)?;

    let residual = xs.clone();
    let normed = rms_norm_sequence(
        &xs,
        Some(&auth_read!(
            layer_source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaPrefillLayerNormKind::PreFeedForward,
            },
        )?),
        Some(scalars.rms_norm_eps),
    )?;
    let gate = project_sequence_with_prefill_source(
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Gate,
        layer.gate_proj_shape.rows,
    )?;
    let gate = gelu_sequence(&gate)?;
    let up = project_sequence_with_prefill_source(
        &normed,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Up,
        layer.up_proj_shape.rows,
    )?;
    let ff_hidden = mul_sequences(&gate, &up)?;
    let ff_out = project_sequence_with_prefill_source(
        &ff_hidden,
        layer_source,
        layer.layer_idx,
        GemmaPrefillLayerMatrixKind::Down,
        layer.down_proj_shape.rows,
    )?;
    let ff_out = rms_norm_sequence(
        &ff_out,
        Some(&auth_read!(
            layer_source,
            GemmaPrefillLayerNormWeightsRequest {
                layer_idx: layer.layer_idx,
                norm: GemmaPrefillLayerNormKind::PostFeedForward,
            },
        )?),
        Some(scalars.rms_norm_eps),
    )?;
    let mut xs = add_sequences(&residual, &ff_out)?;

    if let Some(per_layer_input) = per_layer_input {
        let residual = xs.clone();
        let gated = project_sequence_with_prefill_source(
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
        )?;
        let gated = gelu_sequence(&gated)?;
        let gated = mul_sequences(&gated, per_layer_input)?;
        let projected = project_sequence_with_prefill_source(
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
        )?;
        let projected = rms_norm_sequence(
            &projected,
            Some(&auth_read!(
                layer_source,
                GemmaPrefillLayerNormWeightsRequest {
                    layer_idx: layer.layer_idx,
                    norm: GemmaPrefillLayerNormKind::PlePostInput,
                },
            )?),
            Some(scalars.rms_norm_eps),
        )?;
        xs = add_sequences(&residual, &projected)?;
    }

    if scalars.layer_scalar.is_some() {
        xs = scale_sequence(&xs, scalars.layer_scalar)?;
    }

    Ok((xs, layer_cache))
}

fn resolve_prefill_donor_cache<'a>(
    layer_caches: &'a [RasterKvCache],
    layer_idx: usize,
    layer: &crate::shared::raster_prefill_layer::GemmaPrefillLayerMetadata,
) -> Result<Option<&'a RasterKvCache>> {
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
            })
        })
        .transpose()
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
    use super::run;
    use crate::prefill_layer::deterministic_tiles;
    use crate::shared::det_num::{Acc, Act, Wgt};
    use crate::shared::raster_prefill_layer::AuthenticatedGemmaPrefillLayerSource;
    use crate::shared::transformer::{
        ActivationSequence, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleLayerWeights,
        Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence, MatrixF32,
    };
    use anyhow::{Context, Result};
    use std::path::{Path, PathBuf};

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

        let error = run(&input, &source, None).expect_err("self donor should fail");

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
    fn ple_input_on_non_ple_layer_fails_closed() {
        let (_path, model) = no_ple_model();
        let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
            .expect("source should build");
        let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let ple_inputs = ple_inputs(vec![vec![Act::from_num(0.5), Act::from_num(0.25)]]);

        let error = run(&input, &source, Some(&ple_inputs)).expect_err("PLE input should fail");

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

        let error = run(&input, &source, Some(&ple_inputs)).expect_err("PLE width should fail");

        assert!(error
            .to_string()
            .contains("right sequence row 0 has width 3"));
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

        let error = run(&input, &source, None).expect_err("zero layers should fail");

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

        let error = run(&input, &source, None).expect_err("empty input should fail");

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

        let error = run(&input, &source, None).expect_err("f32-only input should fail");

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

        let raster = run(&input, &source, ple_inputs).expect("raster prefill layer should run");
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

    fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
        activation_sequence_from_internal(InternalActivationSequence::from_det_values(rows))
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
