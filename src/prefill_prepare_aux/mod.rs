use anyhow::Result;
use serde_json::json;

use crate::input_embedding::raster_tiles::RasterInputEmbeddingRefs;
use crate::shared::input::InferenceExecutionMode;
use crate::shared::raster_prefill_ple::{AuthenticatedGemmaPleSource, RasterPrefillPleInputRefs};
use crate::shared::raster_row_store::AuthenticatedRasterTensorStore;
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel,
};
use crate::RasterSizingControls;

pub mod raster_tiles;
mod raster_utils;
pub mod tiles;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
    execution_mode: InferenceExecutionMode,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let ple_inputs = tiles::run(prompt_token_ids, model, token_embeddings, execution_mode)?;
    trace_prefill_prepare_aux_checkpoint(prompt_token_ids, token_embeddings, ple_inputs.as_ref());
    Ok(ple_inputs)
}

pub fn run_raster(
    prompt_token_ids: &[u32],
    ple_source: &AuthenticatedGemmaPleSource,
    token_embeddings: &ActivationSequence,
    projection_rows_per_tile: usize,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let ple_input_refs = run_raster_refs(
        prompt_token_ids,
        ple_source,
        token_embeddings,
        raster_sizing_with_projection_rows(projection_rows_per_tile),
    )?;
    let ple_inputs = materialize_prefill_ple_input_refs(ple_input_refs.as_ref())?;
    Ok(ple_inputs)
}

fn raster_sizing_with_projection_rows(projection_rows_per_tile: usize) -> RasterSizingControls {
    RasterSizingControls {
        projection_rows_per_tile,
        attention_kv_rows_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE,
        sequence_rows_per_tile: crate::InferenceControls::DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE,
        head_rows_per_tile: crate::InferenceControls::DEFAULT_RASTER_HEAD_ROWS_PER_TILE,
        tokenizer_bpe_pairs_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE,
        tokenizer_bpe_pieces_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE,
        output_byte_flush_bytes_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    }
}

pub fn run_raster_refs(
    prompt_token_ids: &[u32],
    ple_source: &AuthenticatedGemmaPleSource,
    token_embeddings: &ActivationSequence,
    raster_sizing: RasterSizingControls,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    let ple_input_refs = raster_tiles::run(
        prompt_token_ids,
        token_embeddings,
        ple_source,
        raster_sizing,
    )?;
    trace_prefill_prepare_aux_raster_checkpoint(
        prompt_token_ids,
        token_embeddings,
        ple_input_refs.as_ref(),
    )?;
    Ok(ple_input_refs)
}

pub fn run_raster_refs_from_input_embedding(
    prompt_token_ids: &[u32],
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    token_embeddings: &ActivationSequence,
    raster_sizing: RasterSizingControls,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    let ple_input_refs = raster_tiles::run_with_input_embedding_refs(
        input_embedding_refs,
        ple_source,
        raster_sizing,
    )?;
    trace_prefill_prepare_aux_raster_checkpoint(
        prompt_token_ids,
        token_embeddings,
        ple_input_refs.as_ref(),
    )?;
    Ok(ple_input_refs)
}

pub fn raster_tensor_store_snapshot() -> AuthenticatedRasterTensorStore {
    raster_utils::tensor_store_snapshot()
}

pub fn materialize_prefill_ple_input_refs(
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let Some(ple_input_refs) = ple_input_refs else {
        return Ok(None);
    };

    Ok(Some(Gemma4PrefillPleInputs::from_internal(
        ple_input_refs
            .per_layer_inputs()
            .iter()
            .map(|input| {
                input
                    .as_ref()
                    .map(|input_ref| {
                        raster_utils::materialize_sequence(input_ref)
                            .map(raster_utils::internal_sequence_from_raster)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?,
    )))
}

fn trace_prefill_prepare_aux_checkpoint(
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) {
    crate::trace::trace_checkpoint(
        "prefill.prepare_aux",
        &prefill_prepare_aux_checkpoint_payload(prompt_token_ids, token_embeddings, ple_inputs),
    );
}

fn prefill_prepare_aux_checkpoint_payload(
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> serde_json::Value {
    json!({
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": crate::trace::sha256_hex(&prompt_token_ids),
        "embedded_prompt_activations": token_embeddings.activations.clone(),
        "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
        "det_embedded_prompt_activations_sha256": token_embeddings.det_activations_sha256.clone(),
        "per_layer_prefill_inputs": ple_inputs.map(|inputs| inputs.per_layer_inputs.clone()),
        "per_layer_prefill_input_sha256s": ple_inputs.map(|inputs| {
            inputs
                .per_layer_inputs
                .iter()
                .map(|input| input.as_ref().map(crate::trace::sha256_hex))
                .collect::<Vec<_>>()
        }),
    })
}

fn trace_prefill_prepare_aux_raster_checkpoint(
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
) -> Result<()> {
    crate::trace::trace_checkpoint_lazy_result("prefill.prepare_aux", || {
        let ple_inputs = materialize_prefill_ple_input_refs(ple_input_refs)?;
        Ok(prefill_prepare_aux_checkpoint_payload(
            prompt_token_ids,
            token_embeddings,
            ple_inputs.as_ref(),
        ))
    })?;
    Ok(())
}
