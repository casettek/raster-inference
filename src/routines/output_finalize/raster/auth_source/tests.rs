use super::{
    AuthenticatedOutputFinalizeStore, AuthenticatedOutputTokenIdsSource, OutputTextBuilderRef,
    OutputTokenIdRequest, OutputTokenIdsMetadataRequest,
};
use crate::shared::artifacts::artifact_io::AuthRead;

#[test]
fn token_source_reads_by_index_with_metadata() {
    let source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[4, 7, 9])
        .expect("source should build");

    let metadata = source
        .auth_read(OutputTokenIdsMetadataRequest)
        .expect("metadata should read");
    assert_eq!(metadata.source_id, "generated");
    assert_eq!(metadata.token_count, 3);
    assert_eq!(
        source
            .auth_read(OutputTokenIdRequest { token_idx: 1 })
            .expect("token should read"),
        7
    );
}

#[test]
fn token_source_allows_empty_metadata_and_rejects_reads() {
    let source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[])
        .expect("empty source should build");

    let metadata = source
        .auth_read(OutputTokenIdsMetadataRequest)
        .expect("metadata should read");
    assert_eq!(metadata.token_count, 0);
    assert!(source
        .auth_read(OutputTokenIdRequest { token_idx: 0 })
        .expect_err("empty source read should fail")
        .to_string()
        .contains("out of range"));
}

#[test]
fn token_source_materialization_rejects_tampered_refs() {
    let source = AuthenticatedOutputTokenIdsSource::from_token_ids("generated", &[4])
        .expect("source should build");
    let mut token_ids_ref = source.token_ids_ref();
    token_ids_ref.det_token_ids_sha256 = "tampered".to_string();

    assert!(source.materialize_token_ids(&token_ids_ref).is_err());
}

#[test]
fn text_builder_materializes_text_and_commitment() {
    let mut store = AuthenticatedOutputFinalizeStore::new();
    let mut builder = store
        .start_text_builder("output")
        .expect("builder should start");

    store
        .append_text_chunk(&mut builder, " ab")
        .expect("first chunk should append");
    store
        .append_text_chunk(&mut builder, "c")
        .expect("second chunk should append");
    let text_ref = store
        .finalize_text_builder(builder)
        .expect("builder should finalize");

    assert_eq!(
        store
            .materialize_text(&text_ref)
            .expect("text should materialize"),
        " abc"
    );
    assert_eq!(text_ref.chunk_count(), 2);
    assert_eq!(text_ref.byte_len(), 4);
}

#[test]
fn text_builder_materialization_rejects_tampered_refs() {
    let mut store = AuthenticatedOutputFinalizeStore::new();
    let builder = store
        .start_text_builder("output")
        .expect("builder should start");
    let mut text_ref = store
        .finalize_text_builder(builder)
        .expect("builder should finalize");
    text_ref.det_text_sha256 = "tampered".to_string();

    assert!(store.materialize_text(&text_ref).is_err());
}

#[test]
fn text_builder_materialization_rejects_tampered_chunk_counts() {
    let mut store = AuthenticatedOutputFinalizeStore::new();
    let mut builder = store
        .start_text_builder("output")
        .expect("builder should start");
    store
        .append_text_chunk(&mut builder, "text")
        .expect("chunk should append");
    let mut text_ref = store
        .finalize_text_builder(builder)
        .expect("builder should finalize");
    text_ref.chunk_count += 1;

    assert!(store.materialize_text(&text_ref).is_err());
}

#[test]
fn pending_byte_builder_tracks_bytes_without_exposing_them_in_ref() {
    let mut store = AuthenticatedOutputFinalizeStore::new();
    let mut builder = store
        .start_pending_byte_builder("pending")
        .expect("builder should start");

    store
        .append_pending_byte(&mut builder, 0xC3)
        .expect("first byte should append");
    store
        .append_pending_byte(&mut builder, 0xA9)
        .expect("second byte should append");

    let serialized = serde_json::to_string(&builder).expect("builder should serialize");
    assert_eq!(builder.bytes_written(), 2);
    assert!(!serialized.contains("195"));
    assert!(!serialized.contains("169"));
    assert!(!serialized.contains("bytes:"));
}

#[test]
fn builder_refs_fail_closed_on_metadata_mismatch() {
    let mut store = AuthenticatedOutputFinalizeStore::new();
    let mut builder = store
        .start_text_builder("output")
        .expect("builder should start");
    store
        .append_text_chunk(&mut builder, "text")
        .expect("chunk should append");

    let tampered = OutputTextBuilderRef {
        running_commitment: "tampered".to_string(),
        ..builder
    };
    assert!(store.finalize_text_builder(tampered).is_err());
}
