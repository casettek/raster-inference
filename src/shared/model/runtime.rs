use anyhow::Result;
use tokenizers::Tokenizer;

use crate::routines::decode_layer_range::raster::auth_source::AuthenticatedDecoderDecodeLayerRangeSource;
use crate::routines::decode_transition_finalize::raster::auth_source::AuthenticatedDecoderDecodeTransitionSource;
use crate::routines::input_embedding::raster::auth_source::AuthenticatedDecoderEmbeddingSource;
use crate::routines::prefill_finalize::raster::auth_source::AuthenticatedDecoderPrefillFinalizeSource;
use crate::shared::api::input::ModelSpec;
use crate::shared::model::gemma::adapter::GemmaModelBundle;
use crate::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::gemma::transformer::Gemma4TransformerModel;
use crate::shared::model::transformer::ActivationSequence;
use crate::shared::raster_contracts::prefill_layer::AuthenticatedDecoderPrefillLayerSource;
use crate::shared::raster_contracts::prefill_ple::AuthenticatedDecoderPrefillPleSource;

#[derive(Clone)]
pub enum LoadedModel {
    Gemma(GemmaModelBundle),
}

impl LoadedModel {
    pub fn model_spec(&self) -> &ModelSpec {
        match self {
            Self::Gemma(model) => model.model_spec(),
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Gemma(model) => model.tokenizer(),
        }
    }

    pub fn transformer_layer_count(&self) -> usize {
        match self {
            Self::Gemma(model) => model.transformer_layer_count(),
        }
    }

    pub(crate) fn transformer_model(&self) -> &Gemma4TransformerModel {
        match self {
            Self::Gemma(model) => model.transformer_model(),
        }
    }

    pub(crate) fn raster_tokenizer(&self) -> Result<&AuthenticatedGemmaTokenizer> {
        match self {
            Self::Gemma(model) => model.raster_tokenizer(),
        }
    }

    pub(crate) fn embed_token_ids(&self, token_ids: &[u32]) -> Result<ActivationSequence> {
        match self {
            Self::Gemma(model) => model.embed_token_ids(token_ids),
        }
    }

    pub(crate) fn input_embedding_source(&self) -> Result<AuthenticatedDecoderEmbeddingSource> {
        match self {
            Self::Gemma(model) => model.input_embedding_source(),
        }
    }

    pub(crate) fn prefill_ple_source(&self) -> Result<AuthenticatedDecoderPrefillPleSource> {
        match self {
            Self::Gemma(model) => model.prefill_ple_source(),
        }
    }

    pub(crate) fn prefill_layer_source(&self) -> Result<AuthenticatedDecoderPrefillLayerSource> {
        match self {
            Self::Gemma(model) => model.prefill_layer_source(),
        }
    }

    pub(crate) fn prefill_finalize_source(
        &self,
    ) -> Result<AuthenticatedDecoderPrefillFinalizeSource> {
        match self {
            Self::Gemma(model) => model.prefill_finalize_source(),
        }
    }

    pub(crate) fn decode_layer_range_source(
        &self,
        identifier: impl Into<String>,
    ) -> Result<AuthenticatedDecoderDecodeLayerRangeSource> {
        match self {
            Self::Gemma(model) => model.decode_layer_range_source(identifier),
        }
    }

    pub(crate) fn decode_transition_source(
        &self,
        identifier: impl Into<String>,
    ) -> Result<AuthenticatedDecoderDecodeTransitionSource> {
        match self {
            Self::Gemma(model) => model.decode_transition_source(identifier),
        }
    }
}
