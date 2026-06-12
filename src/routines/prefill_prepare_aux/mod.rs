use anyhow::Result;
use serde_json::json;

use crate::routines::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::runtime::checkpoints::RoutineId;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{RasterArtifactId, RasterArtifactStoreRoots};
use crate::shared::model::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel,
};
use crate::shared::raster_contracts::prefill_ple::{
    read_prefill_ple_input_manifest_from_roots, AuthenticatedGemmaPleSource,
    GemmaPleMetadataRequest, RasterPrefillPleInputRefs,
};
use crate::RasterSizingControls;

pub mod native;
pub mod raster;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let _routine = crate::trace::routine_scope(
        RoutineId::PrefillPrepareAux,
        format!("tokens={}", prompt_token_ids.len()),
    );
    let ple_inputs = native::run(prompt_token_ids, model, token_embeddings)?;
    trace_prefill_prepare_aux_checkpoint(prompt_token_ids, token_embeddings, ple_inputs.as_ref());
    Ok(ple_inputs)
}

pub fn run_with_input_embedding_checkpoint(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let _routine = crate::trace::routine_scope(
        RoutineId::PrefillPrepareAux,
        format!("tokens={}", prompt_token_ids.len()),
    );
    let ple_inputs = native::run(prompt_token_ids, model, token_embeddings)?;
    let ple_input_refs = format_native_prefill_prepare_aux_as_raster_checkpoint_for_trace(
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
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<raster::RasterPrefillPleOutput> {
    let _routine = crate::trace::routine_scope(RoutineId::PrefillPrepareAux, "mode=raster");
    let output = raster::run_with_input_embedding_refs(
        artifact_store_roots,
        input_embedding_refs,
        ple_source,
        raster_sizing,
    )?;
    let ple_input_refs = prefill_ple_input_refs_from_manifest(
        output.artifact_store_roots.clone(),
        output.refs.as_deref(),
    )?;
    trace_prefill_prepare_aux_raster_checkpoint_from_input_embedding(
        input_embedding_refs,
        ple_input_refs.as_ref(),
    );
    Ok(output)
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

pub fn format_native_prefill_prepare_aux_as_raster_checkpoint_for_trace(
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
                    let sequence = raster::utils::raster_activation_sequence_from_internal(input)?;
                    let (next_roots, input_ref) =
                        raster::utils::insert_activation_sequence_with_roots(
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

pub fn materialize_prefill_ple_input_refs_for_trace(
    ple_input_refs: Option<&RasterPrefillPleInputRefs>,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    materialize_prefill_ple_inputs(ple_input_refs)
}

pub(crate) fn materialize_prefill_ple_inputs(
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
                        raster::utils::materialize_sequence(input_ref)
                            .map(raster::utils::internal_sequence_from_raster)
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
    let mut payload = json!({
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": crate::trace::sha256_hex(&prompt_token_ids),
        "det_embedded_prompt_activations_sha256": token_embeddings.det_activations_sha256.clone(),
    });
    if let Some(embedded_prompt_activations_sha256) = token_embeddings.activations_sha256.as_ref() {
        // Deterministic-mode payloads carry only canonical commitments
        // (spec v1); fp32 mode keeps the compatibility fields.
        payload["embedded_prompt_activations"] = json!(token_embeddings.activations.clone());
        payload["embedded_prompt_activations_sha256"] =
            json!(embedded_prompt_activations_sha256.clone());
        payload["per_layer_prefill_inputs"] =
            json!(ple_inputs.map(|inputs| inputs.per_layer_inputs.clone()));
        payload["per_layer_prefill_input_sha256s"] = json!(ple_inputs.map(|inputs| {
            inputs
                .per_layer_inputs
                .iter()
                .map(|input| input.as_ref().map(crate::trace::sha256_hex))
                .collect::<Vec<_>>()
        }));
    }
    payload
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
