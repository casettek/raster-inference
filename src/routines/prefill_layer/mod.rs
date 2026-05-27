use anyhow::Result;

use crate::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::runtime::checkpoints::RasterDetourController;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence,
    LayerKvCache,
};
use crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource;
use crate::RasterSizingControls;

use self::raster::utils::{
    layer_cache_from_raster, materialize_prefill_activation_sequence_from_roots,
    materialize_prefill_layer_cache_from_roots, raster_sequence_acts,
};

pub mod native;
pub mod raster;

pub fn run(
    input_activations: &[Vec<f32>],
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_with_mode(
        input_activations,
        model,
        ple_inputs,
        InferenceExecutionMode::Fp32,
    )
}

pub fn run_with_mode(
    input_activations: &[Vec<f32>],
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_with_mode_internal(
        InternalActivationSequence::from_values(input_activations.to_vec()),
        model,
        ple_inputs,
        execution_mode,
    )
}

pub fn materialize_prefill_layer_output_refs_from_roots_for_trace(
    roots: &RasterArtifactStoreRoots,
    refs: &raster::PrefillLayerOutputRefs,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    // Public/dev compatibility boundary. Route-based prefill carries refs and
    // roots through zkVM-shaped work; this materializes only for public results
    // and checkpoint payloads.
    let current_activations =
        materialize_prefill_activation_sequence_from_roots(roots, &refs.final_hidden_states_ref)?;
    let det_activations = raster_sequence_acts(&current_activations);
    let values = current_activations.to_f32_values();
    let mut activation_sequence = ActivationSequence::from_internal(
        InternalActivationSequence::from_det_values(det_activations.clone()),
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&values),
    );
    activation_sequence.det_activations_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
            &det_activations,
        ),
    );

    Ok((
        activation_sequence,
        refs.layer_caches
            .iter()
            .map(|cache| {
                materialize_prefill_layer_cache_from_roots(roots, cache)
                    .map(layer_cache_from_raster)
            })
            .collect::<Result<Vec<_>>>()?,
    ))
}

pub fn run_raster(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, raster::PrefillLayerOutputRefs)> {
    let layer_source =
        crate::shared::raster_contracts::prefill_layer::RasterPrefillLayerSource::for_current_integrity_mode(layer_source)?;
    raster::main(
        artifact_store_roots,
        input_embedding_refs,
        &layer_source,
        ple_input_manifest_root,
        raster_sizing,
    )
}

pub(crate) fn run_with_mode_internal(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    execution_mode: InferenceExecutionMode,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    run_with_mode_internal_with_detour(input_activations, model, ple_inputs, execution_mode, None)
}

pub(crate) fn run_with_mode_internal_with_detour(
    input_activations: InternalActivationSequence,
    model: &Gemma4TransformerModel,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    execution_mode: InferenceExecutionMode,
    detour_controller: Option<&mut RasterDetourController>,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    model.validate_execution_mode(execution_mode)?;
    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            native::run(input_activations.as_f32_slice(), model, ple_inputs)
        }
        InferenceExecutionMode::Deterministic => match detour_controller {
            Some(detour_controller) => native::deterministic_tiles::run_internal_with_detour(
                input_activations,
                model,
                ple_inputs,
                Some(detour_controller),
            ),
            None => native::deterministic_tiles::run_internal(input_activations, model, ple_inputs),
        },
    }
}
