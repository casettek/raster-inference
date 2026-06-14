use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::runtime::checkpoints::RasterDetourSpec;
use crate::runtime::trace;
use crate::shared::api::input::{
    PromptPreparationState, RasterPromptPreparationState, SamplingConfig,
};
use crate::shared::api::output::OutputDecodeState;
use crate::shared::model::transformer::TransformerStateTransitionState;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputEmbeddingState {
    #[serde(flatten)]
    pub prompt_preparation: PromptPreparationState,
    /// f32 compatibility commitment. Always `Some` in fp32 mode; `None` for
    /// deterministic-mode runs (spec v1 retires f32 compatibility commitments
    /// on the deterministic path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedded_prompt_activations_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub det_embedded_prompt_activations_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceState {
    pub input_embedding: InputEmbeddingState,
    pub transformer_state_transition: TransformerStateTransitionState,
    pub output_decode: OutputDecodeState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_tile_invocations: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferenceControls {
    pub commit_checkpoints: bool,
    pub terminal_checkpoint: Option<String>,
    pub raster: bool,
    pub raster_detour: Option<RasterDetourSpec>,
    pub raster_tokenizer_enabled: bool,
    pub raster_projection_rows_per_tile: Option<usize>,
    pub raster_attention_kv_rows_per_tile: Option<usize>,
    pub raster_sequence_rows_per_tile: Option<usize>,
    pub raster_head_rows_per_tile: Option<usize>,
    pub prefill_token_range_width: Option<usize>,
    pub decode_layer_range_width: Option<usize>,
    pub raster_tokenizer_bpe_pairs_per_tile: Option<usize>,
    pub raster_tokenizer_bpe_pieces_per_tile: Option<usize>,
    pub raster_output_byte_flush_bytes_per_tile: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RasterSizingControls {
    pub projection_rows_per_tile: usize,
    pub attention_kv_rows_per_tile: usize,
    pub sequence_rows_per_tile: usize,
    pub head_rows_per_tile: usize,
    pub prefill_token_range_width: usize,
    pub decode_layer_range_width: usize,
    pub tokenizer_bpe_pairs_per_tile: usize,
    pub tokenizer_bpe_pieces_per_tile: usize,
    pub output_byte_flush_bytes_per_tile: usize,
}

impl InferenceControls {
    pub const DEFAULT_RASTER_PROJECTION_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE: usize = 32;
    pub const DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_HEAD_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_PREFILL_TOKEN_RANGE_WIDTH: usize = usize::MAX;
    pub const DEFAULT_DECODE_LAYER_RANGE_WIDTH: usize = usize::MAX;
    pub const DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE: usize =
        crate::routines::prompt_prepare::raster::DEFAULT_BPE_PAIRS_PER_TILE;
    pub const DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE: usize =
        crate::routines::prompt_prepare::raster::DEFAULT_BPE_PIECES_PER_TILE;
    pub const DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE: usize =
        crate::routines::output_finalize::raster::DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE;

    pub fn raster_projection_rows_per_tile(&self) -> Result<usize> {
        match self.raster_projection_rows_per_tile {
            Some(0) => anyhow::bail!("raster projection rows per tile must be greater than zero"),
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_PROJECTION_ROWS_PER_TILE),
        }
    }

    pub fn raster_attention_kv_rows_per_tile(&self) -> Result<usize> {
        match self.raster_attention_kv_rows_per_tile {
            Some(0) => {
                anyhow::bail!("raster attention KV rows per tile must be greater than zero")
            }
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE),
        }
    }

    pub fn raster_sequence_rows_per_tile(&self) -> Result<usize> {
        match self.raster_sequence_rows_per_tile {
            Some(0) => anyhow::bail!("raster sequence rows per tile must be greater than zero"),
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE),
        }
    }

    pub fn raster_head_rows_per_tile(&self) -> Result<usize> {
        match self.raster_head_rows_per_tile {
            Some(0) => anyhow::bail!("raster head rows per tile must be greater than zero"),
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_HEAD_ROWS_PER_TILE),
        }
    }

    pub fn prefill_token_range_width(&self) -> Result<usize> {
        match self.prefill_token_range_width {
            Some(0) => anyhow::bail!("prefill token range width must be greater than zero"),
            Some(width) => Ok(width),
            None => Ok(Self::DEFAULT_PREFILL_TOKEN_RANGE_WIDTH),
        }
    }

    pub fn decode_layer_range_width(&self) -> Result<usize> {
        match self.decode_layer_range_width {
            Some(0) => anyhow::bail!("decode layer range width must be greater than zero"),
            Some(width) => Ok(width),
            None => Ok(Self::DEFAULT_DECODE_LAYER_RANGE_WIDTH),
        }
    }

    pub fn raster_tokenizer_bpe_pairs_per_tile(&self) -> Result<usize> {
        match self.raster_tokenizer_bpe_pairs_per_tile {
            Some(0) => {
                anyhow::bail!("raster tokenizer BPE pairs per tile must be greater than zero")
            }
            Some(pairs) => Ok(pairs),
            None => Ok(Self::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE),
        }
    }

    pub fn raster_tokenizer_bpe_pieces_per_tile(&self) -> Result<usize> {
        match self.raster_tokenizer_bpe_pieces_per_tile {
            Some(0) => {
                anyhow::bail!("raster tokenizer BPE pieces per tile must be greater than zero")
            }
            Some(pieces) => Ok(pieces),
            None => Ok(Self::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE),
        }
    }

    pub fn raster_output_byte_flush_bytes_per_tile(&self) -> Result<usize> {
        match self.raster_output_byte_flush_bytes_per_tile {
            Some(0) => {
                anyhow::bail!("raster output byte flush bytes per tile must be greater than zero")
            }
            Some(bytes) => Ok(bytes),
            None => Ok(Self::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE),
        }
    }

    pub fn raster_sizing_controls(&self) -> Result<RasterSizingControls> {
        Ok(RasterSizingControls {
            projection_rows_per_tile: self.raster_projection_rows_per_tile()?,
            attention_kv_rows_per_tile: self.raster_attention_kv_rows_per_tile()?,
            sequence_rows_per_tile: self.raster_sequence_rows_per_tile()?,
            head_rows_per_tile: self.raster_head_rows_per_tile()?,
            prefill_token_range_width: self.prefill_token_range_width()?,
            decode_layer_range_width: self.decode_layer_range_width()?,
            tokenizer_bpe_pairs_per_tile: self.raster_tokenizer_bpe_pairs_per_tile()?,
            tokenizer_bpe_pieces_per_tile: self.raster_tokenizer_bpe_pieces_per_tile()?,
            output_byte_flush_bytes_per_tile: self.raster_output_byte_flush_bytes_per_tile()?,
        })
    }

    pub(crate) fn terminal_checkpoint_spec(&self) -> Result<Option<trace::TerminalCheckpointSpec>> {
        let spec = self
            .terminal_checkpoint
            .as_deref()
            .map(trace::TerminalCheckpointSpec::parse)
            .transpose()?;
        if spec
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.checkpoint_id() == "decode.layer_range")
        {
            anyhow::bail!(
                "terminal checkpoint decode.layer_range is not supported because paused state cannot yet carry an in-progress decode transition"
            );
        }
        Ok(spec)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PausedInferenceState {
    pub terminal_checkpoint_id: String,
    pub input_embedding: InputEmbeddingState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformer_state_transition: Option<TransformerStateTransitionState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_decode: Option<OutputDecodeState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_tile_invocations: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RasterPromptPreparedState {
    pub terminal_checkpoint_id: String,
    pub prompt_preparation: RasterPromptPreparationState,
    pub sampling: SamplingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_tile_invocations: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceRunOutcome {
    Completed(InferenceState),
    Paused(PausedInferenceState),
    RasterPromptPrepared(RasterPromptPreparedState),
}

#[cfg(test)]
mod tests;
