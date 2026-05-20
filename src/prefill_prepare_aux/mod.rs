use anyhow::Result;
use serde_json::json;

use crate::input_embedding::raster_tiles::RasterInputEmbeddingRefs;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{RasterArtifactId, RasterArtifactStoreRoots};
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel,
};
use crate::shared::raster_contracts::prefill_ple::{
    read_prefill_ple_input_manifest_from_roots, AuthenticatedGemmaPleSource,
    GemmaPleMetadataRequest, RasterPrefillPleInputRefs,
};
use crate::shared::tensors::raster_row_store::AuthenticatedRasterTensorStore;
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

pub fn run_with_input_embedding_checkpoint(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
    execution_mode: InferenceExecutionMode,
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let ple_inputs = tiles::run(prompt_token_ids, model, token_embeddings, execution_mode)?;
    let ple_input_refs = format_native_prefill_prepare_aux_as_raster_checkpoint(
        artifact_store_roots,
        input_embedding_refs,
        ple_source,
        ple_inputs.as_ref(),
    )?;
    trace_prefill_prepare_aux_raster_checkpoint_from_input_embedding(
        input_embedding_refs,
        ple_input_refs.as_ref(),
    );
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
    let ple_input_refs =
        prefill_ple_input_refs_from_manifest(ple_input_refs.0, ple_input_refs.1.as_deref())?;
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
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    let (artifact_store_roots, ple_input_manifest_root) = raster_tiles::run(
        prompt_token_ids,
        token_embeddings,
        ple_source,
        raster_sizing,
    )?;
    let ple_input_refs = prefill_ple_input_refs_from_manifest(
        artifact_store_roots.clone(),
        ple_input_manifest_root.as_deref(),
    )?;
    trace_prefill_prepare_aux_raster_checkpoint(
        prompt_token_ids,
        token_embeddings,
        ple_input_refs.as_ref(),
    )?;
    Ok((artifact_store_roots, ple_input_manifest_root))
}

pub fn run_raster_refs_from_input_embedding(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    let output = raster_tiles::run_with_input_embedding_refs(
        artifact_store_roots,
        input_embedding_refs,
        ple_source,
        raster_sizing,
    )?;
    let (artifact_store_roots, ple_input_manifest_root) = output.into_parts();
    let ple_input_refs = prefill_ple_input_refs_from_manifest(
        artifact_store_roots.clone(),
        ple_input_manifest_root.as_deref(),
    )?;
    trace_prefill_prepare_aux_raster_checkpoint_from_input_embedding(
        input_embedding_refs,
        ple_input_refs.as_ref(),
    );
    Ok((artifact_store_roots, ple_input_manifest_root))
}

pub fn prefill_ple_input_refs_from_manifest(
    artifact_store_roots: RasterArtifactStoreRoots,
    ple_input_manifest_root: Option<&str>,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    let Some(ple_input_manifest_root) = ple_input_manifest_root else {
        return Ok(None);
    };
    Ok(Some(
        read_prefill_ple_input_manifest_from_roots(&artifact_store_roots, ple_input_manifest_root)?
            .into_prefill_ple_input_refs(artifact_store_roots)?,
    ))
}

pub fn format_native_prefill_prepare_aux_as_raster_checkpoint(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<Option<RasterPrefillPleInputRefs>> {
    let metadata = ArtifactIo::auth_read(ple_source, GemmaPleMetadataRequest)?;
    let Some(ple_inputs) = ple_inputs else {
        return Ok(None);
    };

    let per_layer_inputs = ple_inputs
        .internal_per_layer_inputs
        .iter()
        .enumerate()
        .map(|(layer_idx, input)| {
            input
                .as_ref()
                .map(|input| {
                    let sequence = raster_utils::raster_activation_sequence_from_internal(input)?;
                    let (next_roots, input_ref) =
                        raster_utils::insert_activation_sequence_with_roots(
                            &artifact_store_roots,
                            RasterArtifactId::new(format!(
                                "prefill.prepare_aux.per_layer_input.{layer_idx}"
                            ))?,
                            sequence,
                        )?;
                    artifact_store_roots = next_roots;
                    Ok(input_ref)
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(RasterPrefillPleInputRefs::new_with_roots(
        artifact_store_roots,
        metadata.source_id,
        metadata.layer_count,
        input_embedding_refs.prompt_token_count,
        per_layer_inputs,
    )?))
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

fn trace_prefill_prepare_aux_raster_checkpoint_from_input_embedding(
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
) {
    crate::trace::trace_checkpoint(
        "prefill.prepare_aux",
        &prefill_prepare_aux_raster_checkpoint_payload(input_embedding_refs, ple_input_refs),
    );
}

fn prefill_prepare_aux_raster_checkpoint_payload(
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
) -> serde_json::Value {
    let per_layer_prefill_input_refs = ple_input_refs.map(|refs| {
        refs.per_layer_inputs()
            .iter()
            .map(|input_ref| {
                input_ref.as_ref().map(|input_ref| {
                    json!({
                        "root": input_ref.root(),
                        "row_count": input_ref.row_count(),
                        "width": input_ref.width(),
                    })
                })
            })
            .collect::<Vec<_>>()
    });

    json!({
        "input_embedding": {
            "source_id": input_embedding_refs.source_id.as_str(),
            "embedding_source_root": input_embedding_refs.embedding_source_root.as_str(),
            "prompt_token_ids_root": input_embedding_refs.prompt_token_ids_root.as_str(),
            "prompt_token_count": input_embedding_refs.prompt_token_count,
            "embedded_prompt_activations_root": input_embedding_refs.embedded_prompt_activations_ref.root(),
            "embedded_prompt_activation_row_count": input_embedding_refs.embedded_prompt_activations_ref.row_count(),
            "embedded_prompt_activation_width": input_embedding_refs.embedded_prompt_activations_ref.width(),
        },
        "ple": ple_input_refs.map(|refs| {
            json!({
                "source_id": refs.source_id(),
                "layer_count": refs.layer_count(),
                "token_count": refs.token_count(),
                "per_layer_prefill_input_refs": per_layer_prefill_input_refs,
            })
        }),
    })
}
