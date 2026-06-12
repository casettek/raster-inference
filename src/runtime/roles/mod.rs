//! Protocol role entry points.
//!
//! The protocol defines two roles with distinct flows, both built on the
//! phase-sequencing skeleton (`runtime::sequence`) and the executor seam:
//!
//! - [`claimer`]: runs native deterministic inference end-to-end with
//!   checkpoint commitment on, producing the trace artifact for on-chain
//!   commitment.
//! - [`challenger`]: replays a claimed inference natively, locates the first
//!   divergent committed checkpoint, and re-executes the spanning routine
//!   occurrence at raster (tile) level, producing the dispute's raster
//!   detour trace.
//!
//! Both roles use the process-global trace collector and artifact stores;
//! runs must not be interleaved across threads (the same constraint as the
//! legacy entry points).

pub mod challenger;
pub mod claimer;
pub mod detour;

use crate::runtime::inference::InferenceControls;

/// Execution tuning knobs shared by every role entry point: raster tile
/// sizing plus prefill/decode range widths. `None` means the library
/// default, so `ExecutionTuning::default()` reproduces the untuned behavior
/// exactly — committed checkpoint bytes are identical for every tuning.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionTuning {
    pub prefill_token_range_width: Option<usize>,
    pub decode_layer_range_width: Option<usize>,
    pub projection_rows_per_tile: Option<usize>,
    pub attention_kv_rows_per_tile: Option<usize>,
    pub sequence_rows_per_tile: Option<usize>,
    pub head_rows_per_tile: Option<usize>,
    pub tokenizer_bpe_pairs_per_tile: Option<usize>,
    pub tokenizer_bpe_pieces_per_tile: Option<usize>,
    pub output_byte_flush_bytes_per_tile: Option<usize>,
}

impl ExecutionTuning {
    /// Copies the tuning knobs into a set of inference controls.
    pub(crate) fn apply(&self, controls: &mut InferenceControls) {
        controls.prefill_token_range_width = self.prefill_token_range_width;
        controls.decode_layer_range_width = self.decode_layer_range_width;
        controls.raster_projection_rows_per_tile = self.projection_rows_per_tile;
        controls.raster_attention_kv_rows_per_tile = self.attention_kv_rows_per_tile;
        controls.raster_sequence_rows_per_tile = self.sequence_rows_per_tile;
        controls.raster_head_rows_per_tile = self.head_rows_per_tile;
        controls.raster_tokenizer_bpe_pairs_per_tile = self.tokenizer_bpe_pairs_per_tile;
        controls.raster_tokenizer_bpe_pieces_per_tile = self.tokenizer_bpe_pieces_per_tile;
        controls.raster_output_byte_flush_bytes_per_tile = self.output_byte_flush_bytes_per_tile;
    }
}
