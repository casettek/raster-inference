use anyhow::Result;

use crate::shared::input::InferenceExecutionMode;
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel, LayerKvCache,
};

pub mod deterministic_tiles;
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
    match execution_mode {
        InferenceExecutionMode::Fp32 => tiles::run(input_activations, model, ple_inputs),
        InferenceExecutionMode::Deterministic => {
            deterministic_tiles::run(input_activations, model, ple_inputs)
        }
    }
}
