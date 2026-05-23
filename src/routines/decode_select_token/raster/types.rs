use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef;

// Types and constants used by the raster sequences and tiles.

pub const DEFAULT_DECODE_SELECT_LOGITS_PER_TILE: usize = 32;

pub const DEFAULT_DECODE_SELECT_TOKEN_IDS_PER_TILE: usize = 64;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeSelectInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub logits_ref: RasterActivationSequenceRef,
    pub full_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub full_token_count: usize,
    pub generated_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub generated_token_count: usize,
    pub logits_per_tile: usize,
    pub token_ids_per_tile: usize,
    pub output_full_token_ids_source_name: String,
    pub output_generated_token_ids_source_name: String,
    pub output_selected_token_source_name: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectArgmaxState {
    pub(in super::super) input_roots: RasterDecodeSelectInputRoots,
    pub(in super::super) next_token_idx: usize,
    pub(in super::super) logit_count: usize,
    pub(in super::super) best_token_id: u32,
    pub(in super::super) best_logit_bits: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectSelectedState {
    pub(in super::super) input_roots: RasterDecodeSelectInputRoots,
    pub(in super::super) next_token: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectAppendState {
    pub(in super::super) artifact_store_roots: RasterArtifactStoreRoots,
    pub(in super::super) full_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub(in super::super) full_token_count: usize,
    pub(in super::super) generated_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub(in super::super) generated_token_count: usize,
    pub(in super::super) next_full_token_idx: usize,
    pub(in super::super) next_generated_token_idx: usize,
    pub(in super::super) next_token: u32,
    pub(in super::super) token_ids_per_tile: usize,
    pub(in super::super) output_full_token_ids_source_name: String,
    pub(in super::super) output_generated_token_ids_source_name: String,
    pub(in super::super) output_selected_token_source_name: String,
    pub(in super::super) logits_ref: RasterActivationSequenceRef,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeSelectOutputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub next_token: u32,
    pub selected_token_ref: RasterSelectedTokenRef,
    pub full_token_ids_ref: RasterTokenIdSequenceRef,
    pub generated_token_ids_ref: RasterTokenIdSequenceRef,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
}
