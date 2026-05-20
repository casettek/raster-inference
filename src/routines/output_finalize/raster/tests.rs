use super::{
    build_output_decode_commitment, decode_next_output_token_with_roots, detokenize_output_tokens,
    detokenize_output_tokens_with_byte_flush_bytes_per_tile, init_raster_output_detokenize, main,
    materialize_token_ids_from_roots, prepare_raster_output_finalize_input_roots, run,
    DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
};
use crate::dsl::{start_tile_invocation_counting, stop_tile_invocation_counting};
use crate::output_finalize::raster::auth_source::materialize_text_from_roots;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    token_id_leaf, RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots,
};
use crate::shared::model::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaTokenizerSpec,
    GemmaVocabEntry,
};

#[test]
fn detokenize_output_tokens_returns_empty_text_for_empty_ids() {
    let text =
        detokenize_output_tokens(&[], &test_tokenizer_source()).expect("empty ids should decode");

    assert_eq!(text, "");
}

#[test]
fn detokenize_output_tokens_decodes_supported_gemma_pieces() {
    let text = detokenize_output_tokens(&[4, 3], &test_tokenizer_source())
        .expect("generated ids should decode");

    assert_eq!(text, " ab");
}

#[test]
fn detokenize_output_tokens_skips_special_tokens() {
    let text = detokenize_output_tokens(&[5, 4, 3], &test_tokenizer_source())
        .expect("generated ids should decode");

    assert_eq!(text, " ab");
}

#[test]
fn detokenize_output_tokens_decodes_byte_fallback_sequences() {
    let text = detokenize_output_tokens(&[6, 7], &test_tokenizer_source())
        .expect("byte fallback ids should decode");

    assert_eq!(text, "é");
}

#[test]
fn detokenize_output_tokens_rejects_missing_token_ids() {
    let error = detokenize_output_tokens(&[99], &test_tokenizer_source())
        .expect_err("missing token id should fail");

    assert!(error.to_string().contains("token id 99 is missing"));
}

#[test]
fn detokenize_output_tokens_replaces_invalid_byte_fallback_utf8() {
    let text = detokenize_output_tokens(&[6], &test_tokenizer_source())
        .expect("invalid utf-8 byte fallback should be replaced");

    assert_eq!(text, "�");
}

#[test]
fn detokenize_output_tokens_replaces_each_invalid_byte_fallback_byte() {
    let text = detokenize_output_tokens(&[6, 6], &test_tokenizer_source())
        .expect("invalid utf-8 byte fallback should be replaced");

    assert_eq!(text, "��");
}

#[test]
fn detokenize_output_tokens_replaces_truncated_byte_fallback_before_normal_token() {
    let text = detokenize_output_tokens(&[6, 4, 3], &test_tokenizer_source())
        .expect("truncated byte fallback should decode with replacement");

    assert_eq!(text, "� ab");
}

#[test]
fn detokenize_output_tokens_flushes_long_byte_fallback_in_chunks() {
    let token_ids = std::iter::repeat([6, 7])
        .take(20)
        .flatten()
        .collect::<Vec<_>>();

    start_tile_invocation_counting();
    let text = detokenize_output_tokens(&token_ids, &test_tokenizer_source())
        .expect("long valid byte fallback should decode");
    let invocations = stop_tile_invocation_counting().expect("tile counting should be active");

    assert_eq!(text, "é".repeat(20));
    assert!(invocations > token_ids.len() as u64);
}

#[test]
fn detokenize_output_tokens_honors_byte_flush_chunk_size() {
    let token_ids = std::iter::repeat([6, 7])
        .take(20)
        .flatten()
        .collect::<Vec<_>>();

    start_tile_invocation_counting();
    let default_text = detokenize_output_tokens(&token_ids, &test_tokenizer_source())
        .expect("default chunking should decode");
    let default_invocations =
        stop_tile_invocation_counting().expect("tile counting should be active");

    start_tile_invocation_counting();
    let wide_text = detokenize_output_tokens_with_byte_flush_bytes_per_tile(
        &token_ids,
        &test_tokenizer_source(),
        64,
    )
    .expect("wide chunking should decode");
    let wide_invocations = stop_tile_invocation_counting().expect("tile counting should be active");

    assert_eq!(default_text, wide_text);
    assert_eq!(wide_text, "é".repeat(20));
    assert!(
        wide_invocations < default_invocations,
        "larger byte flush chunks should require fewer tile invocations"
    );
}

#[test]
fn detokenize_output_tokens_rejects_zero_byte_flush_chunk_size() {
    let error =
        detokenize_output_tokens_with_byte_flush_bytes_per_tile(&[6], &test_tokenizer_source(), 0)
            .expect_err("zero byte flush chunk size should fail");

    assert!(error.to_string().contains("greater than zero"));
}

#[test]
fn build_output_decode_commitment_hashes_generated_token_ids_only() {
    let digest = build_output_decode_commitment(&[4, 5]).expect("commitment should build");
    assert_eq!(
        digest,
        "d4c7a98da55490b0a5a65cc5057db99aa708a436609b177748505342d569457b"
    );
}

#[test]
fn run_builds_output_decode_state() {
    let output = run(&[4, 3], &test_tokenizer_source()).expect("raster finalize should run");

    assert_eq!(output.generated_token_ids, vec![4, 3]);
    assert_eq!(output.generated_text, " ab");
    assert_eq!(output.generated_token_count, 2);
    assert!(output.decode_transition_states.is_empty());
}

#[test]
fn root_backed_ref_path_returns_materializable_refs_with_stable_commitments() {
    ArtifactIo::reset_store();
    let tokenizer = test_tokenizer_source();
    let input_roots = prepare_raster_output_finalize_input_roots(
        &[4, 3],
        &tokenizer,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        "output.finalize.test",
    )
    .expect("input roots should prepare");

    let refs = main(input_roots, &tokenizer).expect("root-backed finalize should run");

    assert_eq!(refs.refs.generated_token_count, 2);
    assert_eq!(
        refs.refs.generated_token_ids_sha256,
        build_output_decode_commitment(&[4, 3]).expect("commitment should build")
    );
    assert_eq!(
        materialize_token_ids_from_roots(
            &refs.artifact_store_roots,
            &refs.refs.generated_token_ids_ref
        )
        .expect("token ids should materialize"),
        vec![4, 3]
    );
    assert_eq!(
        materialize_text_from_roots(&refs.artifact_store_roots, &refs.refs.generated_text_ref)
            .expect("text should materialize"),
        " ab"
    );
    assert!(refs.refs.generated_text_ref.root().is_some());
}

#[test]
fn root_backed_chunk_size_changes_do_not_change_output() {
    let token_ids = std::iter::repeat([6, 7])
        .take(20)
        .flatten()
        .collect::<Vec<_>>();
    let tokenizer = test_tokenizer_source();

    ArtifactIo::reset_store();
    let default_refs = main(
        prepare_raster_output_finalize_input_roots(
            &token_ids,
            &tokenizer,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
            "output.finalize.default",
        )
        .expect("input roots should prepare"),
        &tokenizer,
    )
    .expect("default chunking should run");
    let default_text = materialize_text_from_roots(
        &default_refs.artifact_store_roots,
        &default_refs.refs.generated_text_ref,
    )
    .expect("default text should materialize");

    ArtifactIo::reset_store();
    let wide_refs = main(
        prepare_raster_output_finalize_input_roots(
            &token_ids,
            &tokenizer,
            64,
            "output.finalize.wide",
        )
        .expect("input roots should prepare"),
        &tokenizer,
    )
    .expect("wide chunking should run");
    let wide_text = materialize_text_from_roots(
        &wide_refs.artifact_store_roots,
        &wide_refs.refs.generated_text_ref,
    )
    .expect("wide text should materialize");

    assert_eq!(default_text, wide_text);
    assert_eq!(wide_text, "é".repeat(20));
    assert_eq!(
        default_refs.refs.generated_token_ids_sha256,
        wide_refs.refs.generated_token_ids_sha256
    );
}

#[test]
fn root_backed_state_serializes_refs_without_payloads() {
    ArtifactIo::reset_store();
    let tokenizer = test_tokenizer_source();
    let token_ids = vec![4; 128];
    let input_roots = prepare_raster_output_finalize_input_roots(
        &token_ids,
        &tokenizer,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        "output.finalize.state",
    )
    .expect("input roots should prepare");
    let (_roots, state) = init_raster_output_detokenize(
        input_roots.artifact_store_roots.clone(),
        input_roots,
        &tokenizer,
    )
    .expect("state should initialize");

    let value = serde_json::to_value(&state).expect("state should serialize");

    assert!(value.get("token_ids").is_none());
    assert!(value.get("generated_token_ids").is_none());
    assert!(value.get("text").is_none());
    assert!(value.get("bytes").is_none());
    assert!(value.get("token_ids_ref").is_some());
    assert!(value.get("artifact_store_roots").is_some());
    assert_eq!(value["token_count"], 128);
}

#[test]
fn root_backed_state_does_not_embed_pending_byte_arrays() {
    ArtifactIo::reset_store();
    let tokenizer = test_tokenizer_source();
    let input_roots = prepare_raster_output_finalize_input_roots(
        &[6],
        &tokenizer,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        "output.finalize.pending_state",
    )
    .expect("input roots should prepare");
    let (_roots, state) = init_raster_output_detokenize(
        input_roots.artifact_store_roots.clone(),
        input_roots,
        &tokenizer,
    )
    .expect("state should initialize");
    let (_done, state) =
        decode_next_output_token_with_roots(state, &tokenizer).expect("first token should run");

    let value = serde_json::to_value(&state).expect("state should serialize");

    assert!(value.get("pending_byte_fallback").is_none());
    assert!(value.get("bytes").is_none());
    assert_eq!(value["pending_bytes_written"], 1);
    assert!(value["pending_bytes_builder_source_name"].is_string());
}

#[test]
fn root_backed_finalize_fails_closed_on_missing_stale_and_bad_routes() {
    ArtifactIo::reset_store();
    let tokenizer = test_tokenizer_source();
    let input_roots = prepare_raster_output_finalize_input_roots(
        &[4],
        &tokenizer,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        "output.finalize.fail_closed",
    )
    .expect("input roots should prepare");

    let mut missing_roots = input_roots.clone();
    missing_roots.artifact_store_roots = RasterArtifactStoreRoots::default();
    assert!(main(missing_roots, &tokenizer)
        .expect_err("missing roots should fail")
        .to_string()
        .contains("not present"));

    let (stale_store_roots, _extra) = ArtifactIo::insert_artifact_with_roots(
        &input_roots.artifact_store_roots,
        RasterArtifactId::new("output.finalize.fail_closed.extra").expect("id"),
        RasterArtifactMetadata::token_ids(1),
        vec![token_id_leaf(9)],
    )
    .expect("extra artifact should insert");
    assert_ne!(stale_store_roots, input_roots.artifact_store_roots);
    assert!(main(input_roots.clone(), &tokenizer)
        .expect_err("stale roots should fail")
        .to_string()
        .contains("snapshot"));

    ArtifactIo::reset_store();
    let mut bad_source = prepare_raster_output_finalize_input_roots(
        &[4],
        &tokenizer,
        DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
        "output.finalize.bad_source",
    )
    .expect("input roots should prepare");
    bad_source.tokenizer_source_root = "wrong-tokenizer-root".to_string();
    assert!(main(bad_source, &tokenizer)
        .expect_err("bad tokenizer route should fail")
        .to_string()
        .contains("does not match"));
}

#[test]
fn root_backed_token_commitment_matches_decode_finalize_methodology() {
    ArtifactIo::reset_store();
    let tokenizer = test_tokenizer_source();
    let token_ids = vec![4, 3, 6, 7];
    let refs = main(
        prepare_raster_output_finalize_input_roots(
            &token_ids,
            &tokenizer,
            DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
            "output.finalize.commitment",
        )
        .expect("input roots should prepare"),
        &tokenizer,
    )
    .expect("root-backed finalize should run");

    assert_eq!(
        refs.refs.generated_token_ids_sha256,
        build_output_decode_commitment(&token_ids).expect("commitment should build")
    );
}

fn test_tokenizer_source() -> AuthenticatedGemmaTokenizer {
    AuthenticatedGemmaTokenizer::new(test_tokenizer_spec())
}

fn test_tokenizer_spec() -> GemmaTokenizerSpec {
    GemmaTokenizerSpec::new(
        "digest".to_string(),
        vec![
            GemmaVocabEntry {
                token: "<unk>".to_string(),
                id: 0,
            },
            GemmaVocabEntry {
                token: "a".to_string(),
                id: 1,
            },
            GemmaVocabEntry {
                token: "b".to_string(),
                id: 2,
            },
            GemmaVocabEntry {
                token: "ab".to_string(),
                id: 3,
            },
            GemmaVocabEntry {
                token: "▁".to_string(),
                id: 4,
            },
            GemmaVocabEntry {
                token: "<bos>".to_string(),
                id: 5,
            },
            GemmaVocabEntry {
                token: "<0xC3>".to_string(),
                id: 6,
            },
            GemmaVocabEntry {
                token: "<0xA9>".to_string(),
                id: 7,
            },
        ],
        vec![GemmaBpeMerge {
            left: "a".to_string(),
            right: "b".to_string(),
            merged: "ab".to_string(),
            rank: 0,
        }],
        vec![GemmaAddedToken {
            id: 5,
            content: "<bos>".to_string(),
            special: true,
        }],
        "<unk>".to_string(),
        true,
        "▁".to_string(),
        " ".to_string(),
    )
    .expect("test tokenizer spec should build")
}
