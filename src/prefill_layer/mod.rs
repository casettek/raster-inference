use anyhow::Result;

use crate::shared::input::InferenceExecutionMode;
use crate::shared::raster_prefill_layer::AuthenticatedGemmaPrefillLayerSource;
use crate::shared::raster_prefill_ple::RasterPrefillPleInputRefs;
use crate::shared::raster_row_store::AuthenticatedRasterTensorStore;
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence,
    LayerKvCache,
};
use crate::RasterSizingControls;

pub mod deterministic_tiles;
pub mod raster_tiles;
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

pub fn run_raster(
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    raster_sizing: RasterSizingControls,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    // Compatibility entry point for callers that still hold materialized PLE
    // inputs. The proof-shaped raster path should use `run_raster_with_store`.
    raster_tiles::run_materialized_compat(
        input_activations,
        layer_source,
        ple_inputs,
        raster_sizing,
    )
}

pub fn run_raster_with_store(
    store: &mut AuthenticatedRasterTensorStore,
    input_activations: &ActivationSequence,
    layer_source: &AuthenticatedGemmaPrefillLayerSource,
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
    raster_sizing: RasterSizingControls,
) -> Result<(ActivationSequence, Vec<LayerKvCache>)> {
    raster_tiles::run_with_store(
        store,
        input_activations,
        layer_source,
        ple_input_refs,
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
