use anyhow::Result;

use crate::input_embedding::raster_tiles::RasterInputEmbeddingRefs;
use crate::shared::input::InferenceExecutionMode;
use crate::shared::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::raster_prefill_layer::AuthenticatedGemmaPrefillLayerSource;
use crate::shared::raster_row_store::AuthenticatedRasterTensorStore;
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence,
    LayerKvCache,
};
use crate::RasterSizingControls;

use self::raster_utils::{
    layer_cache_from_raster, materialize_prefill_activation_sequence_from_store,
    materialize_prefill_layer_cache_from_store, raster_sequence_acts,
};

pub mod deterministic_tiles;
pub mod raster_tiles;
mod raster_utils;
pub mod tiles;

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

pub fn materialize_prefill_layer_output_refs(
    store: &AuthenticatedRasterTensorStore,
    refs: &raster_tiles::PrefillLayerOutputRefs,
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

pub fn run_raster_refs_from_input_embedding(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_manifest_root: Option<&str>,
    raster_sizing: RasterSizingControls,
) -> Result<(
    RasterArtifactStoreRoots,
    raster_tiles::PrefillLayerOutputRefs,
)> {
    raster_tiles::main(
        artifact_store_roots,
        input_embedding_refs,
        layer_source,
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
    model.validate_execution_mode(execution_mode)?;
    match execution_mode {
        InferenceExecutionMode::Fp32 => {
            tiles::run(input_activations.as_f32_slice(), model, ple_inputs)
        }
        InferenceExecutionMode::Deterministic => {
            deterministic_tiles::run_internal(input_activations, model, ple_inputs)
        }
    }
}
