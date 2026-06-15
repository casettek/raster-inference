use anyhow::Result;

use crate::routines::decode_layer_range::raster::auth_source::AuthenticatedDecoderDecodeLayerRangeSource;
use crate::routines::decode_transition_finalize::raster::auth_source::AuthenticatedDecoderDecodeTransitionSource;
use crate::routines::input_embedding::raster::auth_source::AuthenticatedDecoderEmbeddingSource;
use crate::routines::prefill_finalize::raster::auth_source::AuthenticatedDecoderPrefillFinalizeSource;
use crate::shared::model::gemma::transformer::Gemma4TransformerModel;
use crate::shared::raster_contracts::prefill_layer::AuthenticatedDecoderPrefillLayerSource;
use crate::shared::raster_contracts::prefill_ple::AuthenticatedDecoderPrefillPleSource;

pub(crate) fn input_embedding_source(
    model_id: impl Into<String>,
    model: &Gemma4TransformerModel,
) -> Result<AuthenticatedDecoderEmbeddingSource> {
    AuthenticatedDecoderEmbeddingSource::from_decoder_view(model_id, &model.decoder_view())
}

pub(crate) fn prefill_ple_source(
    model_id: impl Into<String>,
    model: &Gemma4TransformerModel,
) -> Result<AuthenticatedDecoderPrefillPleSource> {
    AuthenticatedDecoderPrefillPleSource::from_decoder_view(
        model_id,
        &model.decoder_view(),
        model.ple_global.clone(),
        model.rms_norm_eps_det,
    )
}

pub(crate) fn prefill_layer_source(
    model_id: impl Into<String>,
    model: &Gemma4TransformerModel,
) -> Result<AuthenticatedDecoderPrefillLayerSource> {
    AuthenticatedDecoderPrefillLayerSource::from_decoder_view(model_id, &model.decoder_view())
}

pub(crate) fn prefill_finalize_source(
    model_id: impl Into<String>,
    model: &Gemma4TransformerModel,
) -> Result<AuthenticatedDecoderPrefillFinalizeSource> {
    AuthenticatedDecoderPrefillFinalizeSource::from_decoder_view(model_id, &model.decoder_view())
}

pub(crate) fn decode_layer_range_source(
    identifier: impl Into<String>,
    model: &Gemma4TransformerModel,
) -> Result<AuthenticatedDecoderDecodeLayerRangeSource> {
    AuthenticatedDecoderDecodeLayerRangeSource::from_decoder_view(identifier, &model.decoder_view())
}

pub(crate) fn decode_transition_source(
    identifier: impl Into<String>,
    model: &Gemma4TransformerModel,
) -> Result<AuthenticatedDecoderDecodeTransitionSource> {
    AuthenticatedDecoderDecodeTransitionSource::from_decoder_view(identifier, &model.decoder_view())
}
