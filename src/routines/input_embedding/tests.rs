use super::{materialize_input_embedding_refs_for_trace, run_raster};
use crate::input_embedding::raster::auth_source::AuthenticatedGemmaInputEmbeddingSource;
use crate::shared::api::input::RasterPromptPreparationState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactId, RasterArtifactMetadata, RasterTokenIdSequenceRef,
};
use crate::shared::numerics::det_num::Act;

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

    let output = run_raster(
        ArtifactIo::export_store_roots(),
        &prompt_preparation,
        &source,
    )
    .expect("raster input embedding should run");
    let refs = output.refs;
    let materialized = materialize_input_embedding_refs_for_trace(&refs)
        .expect("embedding refs should materialize");

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
    let raster_refs = run_raster(
        ArtifactIo::export_store_roots(),
        &prompt_preparation,
        &source,
    )
    .expect("raster input embedding should run")
    .refs;
    let materialized = materialize_input_embedding_refs_for_trace(&raster_refs)
        .expect("embedding refs should materialize");

    ArtifactIo::reset_store();
    let native_refs = super::format_native_input_embedding_as_raster_checkpoint_for_trace(
        "embedding-fixture",
        raster_refs.embedding_source_root.clone(),
        &prompt_preparation,
        &materialized,
    )
    .expect("native checkpoint refs");
    let native_refs = native_refs.refs;

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
            RasterArtifactId::new("input.embedding.test.prompt-token-ids").expect("artifact id"),
            RasterArtifactMetadata::token_ids(token_ids.len()),
            leaves,
        )
        .expect("token artifact"),
    )
    .expect("token ids ref")
}
