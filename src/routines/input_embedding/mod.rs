use anyhow::Result;
use serde_json::json;

use crate::shared::api::input::{InferenceExecutionMode, RasterPromptPreparationState};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterRoutineOutput,
};
use crate::shared::model::transformer::{ActivationSequence, Gemma4TransformerModel};

pub mod native;
pub mod raster;

use self::raster::auth_source::AuthenticatedGemmaInputEmbeddingSource;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequence> {
    native::run(prompt_token_ids, model, execution_mode)
}

pub fn run_raster(
    artifact_store_roots: RasterArtifactStoreRoots,
    prompt_preparation: &RasterPromptPreparationState,
    embedding_source: &AuthenticatedGemmaInputEmbeddingSource,
) -> Result<raster::RasterInputEmbeddingOutput> {
    let embedding_source =
        raster::auth_source::RasterInputEmbeddingSource::for_current_integrity_mode(
            embedding_source,
        )?;
    raster::main(
        artifact_store_roots,
        raster::RasterInputEmbeddingInputRoots {
            prompt_token_ids_root: prompt_preparation.prompt_token_ids_root.clone(),
            prompt_token_count: prompt_preparation.prompt_token_count,
            embedding_source_root: embedding_source.root().to_string(),
        },
        &embedding_source,
    )
}

pub fn materialize_input_embedding_refs_for_trace(
    refs: &raster::RasterInputEmbeddingRefs,
) -> Result<ActivationSequence> {
    let sequence = raster::utils::materialize_sequence(&refs.embedded_prompt_activations_ref)?;
    let internal = raster::utils::internal_sequence_from_raster(sequence);
    let activations = internal.clone_f32();
    let det_activations_sha256 = internal
        .det_values()
        .map(crate::shared::numerics::transformer_kernels::build_det_activation_commitment);
    let mut activation_sequence = ActivationSequence::from_internal(
        internal,
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&activations),
    );
    activation_sequence.det_activations_sha256 = det_activations_sha256;
    Ok(activation_sequence)
}

pub fn format_native_input_embedding_as_raster_checkpoint_for_trace(
    source_id: impl Into<String>,
    embedding_source_root: impl Into<String>,
    prompt_preparation: &RasterPromptPreparationState,
    token_embeddings: &ActivationSequence,
) -> Result<raster::RasterInputEmbeddingOutput> {
    let embedded_prompt_activations_ref = raster::utils::insert_activation_sequence(
        crate::shared::artifacts::raster_artifact_store::RasterArtifactId::new(
            "input.embedding.embedded_prompt",
        )?,
        raster::utils::raster_activation_sequence_from_embedding(token_embeddings)?,
    )?;

    Ok(RasterRoutineOutput::new(
        ArtifactIo::export_store_roots(),
        raster::RasterInputEmbeddingRefs {
            source_id: source_id.into(),
            embedding_source_root: embedding_source_root.into(),
            prompt_token_ids_root: prompt_preparation.prompt_token_ids_root.clone(),
            prompt_token_count: prompt_preparation.prompt_token_count,
            embedded_prompt_activations_ref,
        },
    ))
}

pub fn trace_input_embedding_checkpoint(
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    raster_refs: Option<&raster::RasterInputEmbeddingRefs>,
) {
    crate::trace::trace_checkpoint(
        "input.embedding",
        &input_embedding_checkpoint_payload(prompt_token_ids, token_embeddings, raster_refs),
    );
}

fn input_embedding_checkpoint_payload(
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    raster_refs: Option<&raster::RasterInputEmbeddingRefs>,
) -> serde_json::Value {
    let raster_payload = raster_refs.map(|refs| {
        json!({
            "source_id": refs.source_id,
            "embedding_source_root": refs.embedding_source_root,
            "prompt_token_ids_root": refs.prompt_token_ids_root,
            "prompt_token_count": refs.prompt_token_count,
            "embedded_prompt_activations_root": refs.embedded_prompt_activations_ref.root(),
            "embedded_prompt_activation_row_count": refs.embedded_prompt_activations_ref.row_count(),
            "embedded_prompt_activation_width": refs.embedded_prompt_activations_ref.width(),
        })
    });

    json!({
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": crate::trace::sha256_hex(&prompt_token_ids),
        "embedded_prompt_activations": token_embeddings.activations.clone(),
        "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
        "det_embedded_prompt_activations_sha256": token_embeddings.det_activations_sha256.clone(),
        "raster": raster_payload,
    })
}

#[cfg(test)]
mod tests;
