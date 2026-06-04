use anyhow::Result;

use crate::routines::decode_layer_range::DecodeLayerRangeState;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::model::transformer::{Gemma4TransformerModel, TransformerDecodeState};

pub(crate) fn init_state(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
) -> Result<DecodeLayerRangeState> {
    super::tiles::init_state_with_mode(
        transformer_decode_state,
        next_token,
        model,
        InferenceExecutionMode::Deterministic,
    )
}

pub(crate) fn run_range(
    state: DecodeLayerRangeState,
    model: &Gemma4TransformerModel,
    decode_layer_range_width: usize,
) -> Result<(DecodeLayerRangeState, bool)> {
    super::tiles::run_range_with_mode(
        state,
        model,
        decode_layer_range_width,
        InferenceExecutionMode::Deterministic,
        Some("deterministic"),
    )
}
