use anyhow::Result;
use serde_json::json;

use crate::shared::input::{InferenceExecutionMode, RasterPromptPreparationState};
use crate::shared::raster_input_embedding::AuthenticatedGemmaInputEmbeddingSource;
use crate::shared::transformer::{ActivationSequence, Gemma4TransformerModel};

pub mod raster_tiles;
mod raster_utils;
pub mod tiles;

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequence> {
    tiles::run(prompt_token_ids, model, execution_mode)
}

pub fn run_raster_refs(
    prompt_preparation: &RasterPromptPreparationState,
    embedding_source: &AuthenticatedGemmaInputEmbeddingSource,
) -> Result<raster_tiles::RasterInputEmbeddingRefs> {
    let embedding_source_ref = embedding_source.committed_source_ref()?;
    run_raster_refs_for_roots(
        prompt_preparation.prompt_token_ids_root.clone(),
        prompt_preparation.prompt_token_count,
        embedding_source_ref.root().to_string(),
    )
}

pub fn run_raster_refs_for_roots(
    prompt_token_ids_root: String,
    prompt_token_count: usize,
    embedding_source_root: String,
) -> Result<raster_tiles::RasterInputEmbeddingRefs> {
    raster_tiles::main(raster_tiles::RasterInputEmbeddingInputRoots {
        prompt_token_ids_root,
        prompt_token_count,
        embedding_source_root,
    })
}

pub fn materialize_input_embedding_refs(
    refs: &raster_tiles::RasterInputEmbeddingRefs,
) -> Result<ActivationSequence> {
    let sequence = raster_utils::materialize_sequence(&refs.embedded_prompt_activations_ref)?;
    let internal = raster_utils::internal_sequence_from_raster(sequence);
    let activations = internal.clone_f32();
    let det_activations_sha256 = internal
        .det_values()
        .map(crate::shared::transformer_kernels::build_det_activation_commitment);
    let mut activation_sequence = ActivationSequence::from_internal(
        internal,
        crate::shared::transformer_kernels::build_activation_commitment(&activations),
    );
    activation_sequence.det_activations_sha256 = det_activations_sha256;
    Ok(activation_sequence)
}

pub fn format_native_input_embedding_as_raster_checkpoint(
    source_id: impl Into<String>,
    embedding_source_root: impl Into<String>,
    prompt_preparation: &RasterPromptPreparationState,
    token_embeddings: &ActivationSequence,
) -> Result<raster_tiles::RasterInputEmbeddingRefs> {
    let embedded_prompt_activations_ref = raster_utils::insert_activation_sequence(
        crate::shared::raster_artifact_store::RasterArtifactId::new(
            "input.embedding.embedded_prompt",
        )?,
        raster_utils::raster_activation_sequence_from_embedding(token_embeddings)?,
    )?;

    Ok(raster_tiles::RasterInputEmbeddingRefs {
        source_id: source_id.into(),
        embedding_source_root: embedding_source_root.into(),
        prompt_token_ids_root: prompt_preparation.prompt_token_ids_root.clone(),
        prompt_token_count: prompt_preparation.prompt_token_count,
        embedded_prompt_activations_ref,
    })
}

pub fn trace_input_embedding_checkpoint(
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    raster_refs: Option<&raster_tiles::RasterInputEmbeddingRefs>,
) {
    crate::trace::trace_checkpoint(
        "input.embedding",
        &input_embedding_checkpoint_payload(prompt_token_ids, token_embeddings, raster_refs),
    );
}

fn input_embedding_checkpoint_payload(
    prompt_token_ids: &[u32],
    token_embeddings: &ActivationSequence,
    raster_refs: Option<&raster_tiles::RasterInputEmbeddingRefs>,
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
mod tests {
    use super::{materialize_input_embedding_refs, run_raster_refs};
    use crate::shared::artifact_io::ArtifactIo;
    use crate::shared::det_num::Act;
    use crate::shared::input::RasterPromptPreparationState;
    use crate::shared::raster_artifact_store::{
        RasterArtifactId, RasterArtifactMetadata, RasterTokenIdSequenceRef,
    };
    use crate::shared::raster_input_embedding::AuthenticatedGemmaInputEmbeddingSource;

    #[test]
    fn raster_input_embedding_consumes_prompt_token_root() {
        ArtifactIo::reset_store();
        let token_ids = insert_token_ids(&[1, 0]);
        let prompt_preparation = RasterPromptPreparationState {
            prompt_bytes_root: "prompt-bytes".to_string(),
            prompt_text_root: "prompt-text".to_string(),
            rendered_prompt_root: "rendered-prompt".to_string(),
            normalized_prompt_root: "normalized-prompt".to_string(),
            prompt_token_ids_root: token_ids.root().to_string(),
            prompt_token_count: token_ids.token_count(),
        };
        let source = AuthenticatedGemmaInputEmbeddingSource::from_canonical_rows(
            "embedding-fixture",
            vec![
                vec![Act::from_num(1.0), Act::from_num(2.0)],
                vec![Act::from_num(3.0), Act::from_num(4.0)],
            ],
            Act::from_num(1.0),
        )
        .expect("embedding source should build");

        let refs = run_raster_refs(&prompt_preparation, &source)
            .expect("raster input embedding should run");
        let materialized =
            materialize_input_embedding_refs(&refs).expect("embedding refs should materialize");

        assert_eq!(refs.prompt_token_ids_root, token_ids.root());
        assert_eq!(refs.prompt_token_count, 2);
        assert_eq!(
            materialized.clone_internal().det_values().unwrap(),
            &[
                vec![Act::from_num(3.0), Act::from_num(4.0)],
                vec![Act::from_num(1.0), Act::from_num(2.0)]
            ]
        );
        assert!(materialized.det_activations_sha256.is_some());
    }

    #[test]
    fn native_input_embedding_checkpoint_refs_match_raster_refs() {
        ArtifactIo::reset_store();
        let token_ids = insert_token_ids(&[1, 0]);
        let prompt_preparation = RasterPromptPreparationState {
            prompt_bytes_root: "prompt-bytes".to_string(),
            prompt_text_root: "prompt-text".to_string(),
            rendered_prompt_root: "rendered-prompt".to_string(),
            normalized_prompt_root: "normalized-prompt".to_string(),
            prompt_token_ids_root: token_ids.root().to_string(),
            prompt_token_count: token_ids.token_count(),
        };
        let source = AuthenticatedGemmaInputEmbeddingSource::from_canonical_rows(
            "embedding-fixture",
            vec![
                vec![Act::from_num(1.0), Act::from_num(2.0)],
                vec![Act::from_num(3.0), Act::from_num(4.0)],
            ],
            Act::from_num(1.0),
        )
        .expect("embedding source should build");
        let raster_refs = run_raster_refs(&prompt_preparation, &source)
            .expect("raster input embedding should run");
        let materialized = materialize_input_embedding_refs(&raster_refs)
            .expect("embedding refs should materialize");

        ArtifactIo::reset_store();
        let native_refs = super::format_native_input_embedding_as_raster_checkpoint(
            "embedding-fixture",
            raster_refs.embedding_source_root.clone(),
            &prompt_preparation,
            &materialized,
        )
        .expect("native checkpoint refs");

        assert_eq!(native_refs.source_id, raster_refs.source_id);
        assert_eq!(
            native_refs.prompt_token_ids_root,
            raster_refs.prompt_token_ids_root
        );
        assert_eq!(
            native_refs.prompt_token_count,
            raster_refs.prompt_token_count
        );
        assert_eq!(
            native_refs.embedded_prompt_activations_ref.root(),
            raster_refs.embedded_prompt_activations_ref.root()
        );
        assert_eq!(
            native_refs.embedded_prompt_activations_ref.row_count(),
            raster_refs.embedded_prompt_activations_ref.row_count()
        );
        assert_eq!(
            native_refs.embedded_prompt_activations_ref.width(),
            raster_refs.embedded_prompt_activations_ref.width()
        );
    }

    fn insert_token_ids(token_ids: &[u32]) -> RasterTokenIdSequenceRef {
        let leaves = token_ids
            .iter()
            .map(|token_id| token_id.to_le_bytes().to_vec())
            .collect();
        RasterTokenIdSequenceRef::new(
            ArtifactIo::insert_artifact(
                RasterArtifactId::new("input.embedding.test.prompt-token-ids")
                    .expect("artifact id"),
                RasterArtifactMetadata::token_ids(token_ids.len()),
                leaves,
            )
            .expect("token artifact"),
        )
        .expect("token ids ref")
    }
}
