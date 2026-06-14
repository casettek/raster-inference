use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::routines::decode_layer_range::raster::auth_source::AuthenticatedDecoderDecodeLayerRangeSource;
use crate::routines::decode_transition_finalize::raster::auth_source::AuthenticatedDecoderDecodeTransitionSource;
use crate::routines::input_embedding::raster::auth_source::AuthenticatedDecoderEmbeddingSource;
use crate::routines::prefill_finalize::raster::auth_source::AuthenticatedDecoderPrefillFinalizeSource;
use crate::shared::api::input::ModelSpec;
use crate::shared::model::gemma::io::embed_input_tokens_from_gemma_source;
use crate::shared::model::gemma::sources;
use crate::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::gemma::transformer::Gemma4TransformerModel;
use crate::shared::model::transformer::ActivationSequence;
use crate::shared::raster_contracts::prefill_layer::AuthenticatedDecoderPrefillLayerSource;
use crate::shared::raster_contracts::prefill_ple::AuthenticatedDecoderPrefillPleSource;

#[derive(Clone)]
pub struct GemmaModelBundle {
    model_spec: ModelSpec,
    tokenizer: Tokenizer,
    transformer_model: Gemma4TransformerModel,
    raster_tokenizer: Option<AuthenticatedGemmaTokenizer>,
}

impl GemmaModelBundle {
    pub fn new(
        model_spec: ModelSpec,
        tokenizer: Tokenizer,
        transformer_model: Gemma4TransformerModel,
        raster_tokenizer: Option<AuthenticatedGemmaTokenizer>,
    ) -> Self {
        Self {
            model_spec,
            tokenizer,
            transformer_model,
            raster_tokenizer,
        }
    }

    pub fn model_spec(&self) -> &ModelSpec {
        &self.model_spec
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub(crate) fn transformer_model(&self) -> &Gemma4TransformerModel {
        &self.transformer_model
    }

    pub fn transformer_layer_count(&self) -> usize {
        self.transformer_model.layers.len()
    }

    pub(crate) fn raster_tokenizer(&self) -> Result<&AuthenticatedGemmaTokenizer> {
        self.raster_tokenizer
            .as_ref()
            .context("raster tokenizer capability is not available for this loaded model")
    }

    pub(crate) fn embed_token_ids(&self, token_ids: &[u32]) -> Result<ActivationSequence> {
        let embedding_source = self
            .transformer_model
            .embedding_source
            .as_ref()
            .context("transformer state model is missing an embedding_source")?;
        embed_input_tokens_from_gemma_source(token_ids, embedding_source)
    }

    pub(crate) fn input_embedding_source(&self) -> Result<AuthenticatedDecoderEmbeddingSource> {
        sources::input_embedding_source(self.model_spec.model_id.clone(), &self.transformer_model)
    }

    pub(crate) fn prefill_ple_source(&self) -> Result<AuthenticatedDecoderPrefillPleSource> {
        sources::prefill_ple_source(self.model_spec.model_id.clone(), &self.transformer_model)
    }

    pub(crate) fn prefill_layer_source(&self) -> Result<AuthenticatedDecoderPrefillLayerSource> {
        sources::prefill_layer_source(self.model_spec.model_id.clone(), &self.transformer_model)
    }

    pub(crate) fn prefill_finalize_source(
        &self,
    ) -> Result<AuthenticatedDecoderPrefillFinalizeSource> {
        sources::prefill_finalize_source(self.model_spec.model_id.clone(), &self.transformer_model)
    }

    pub(crate) fn decode_layer_range_source(
        &self,
        identifier: impl Into<String>,
    ) -> Result<AuthenticatedDecoderDecodeLayerRangeSource> {
        sources::decode_layer_range_source(identifier, &self.transformer_model)
    }

    pub(crate) fn decode_transition_source(
        &self,
        identifier: impl Into<String>,
    ) -> Result<AuthenticatedDecoderDecodeTransitionSource> {
        sources::decode_transition_source(identifier, &self.transformer_model)
    }
}
