use super::{
    materialize_output_decode_state_for_api, materialize_run_raster_for_api, run, run_raster,
    run_selected_raster_detour_from_native_boundary,
};
use crate::io::parse_gemma_tokenizer_spec_bytes;
use crate::runtime::inference::InferenceControls;
use crate::shared::api::output::DecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, token_id_leaf, RasterActivationSequenceArtifactRef, RasterArtifactId,
    RasterArtifactMetadata, RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::{InternalLogits, TransformerDecodeState};
use crate::shared::numerics::det_num::Act;
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::raster_kernels::transformer::RasterActivationRow;
use crate::shared::tensors::raster_tensor_artifacts::{
    activation_sequence_ref_from_artifact, RasterActivationSequenceRef, RasterTensorId,
};
use tokenizers::Tokenizer;

#[test]
fn run_raster_matches_native_for_supported_gemma_decode() {
    let tokenizer_json = minimal_gemma_tokenizer_json();
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json.as_bytes()).expect("HF tokenizer should parse");
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(tokenizer_json.as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );
    let decode_state = decode_state(vec![9, 4, 3], vec![4, 3]);

    let native = run(decode_state.clone(), &tokenizer).expect("native finalize should run");
    let raster = materialize_run_raster_for_api(decode_state, &tokenizer_source)
        .expect("raster finalize should run");

    assert_eq!(raster.generated_token_ids, native.generated_token_ids);
    assert_eq!(
        raster.generated_token_ids_sha256,
        native.generated_token_ids_sha256
    );
    assert_eq!(raster.generated_text, native.generated_text);
    assert_eq!(raster.generated_token_count, native.generated_token_count);
    assert_eq!(raster.stop_reason, native.stop_reason);
}

#[test]
fn run_raster_returns_empty_generation_for_zero_tokens() {
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );

    let output = materialize_run_raster_for_api(decode_state(vec![9], vec![]), &tokenizer_source)
        .expect("raster finalize should run");

    assert_eq!(output.generated_token_ids, Vec::<u32>::new());
    assert_eq!(output.generated_text, "");
    assert_eq!(output.generated_token_count, 0);
}

#[test]
fn run_raster_surfaces_detokenization_errors() {
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );

    let error =
        materialize_run_raster_for_api(decode_state(vec![9, 99], vec![99]), &tokenizer_source)
            .expect_err("missing output token should fail");

    assert!(error.to_string().contains("token id 99 is missing"));
}

#[test]
fn run_raster_from_decode_state_refs_consumes_generated_token_ref() {
    let tokenizer_json = minimal_gemma_tokenizer_json();
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json.as_bytes()).expect("HF tokenizer should parse");
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(tokenizer_json.as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );
    let host_state = decode_state(vec![9, 4, 3], vec![4, 3]);
    let expected = run(host_state.clone(), &tokenizer).expect("native finalize should run");
    let raster_state = raster_decode_state_refs(&host_state).expect("refs should build");

    let output_refs = run_raster(raster_state.clone(), &tokenizer_source, 2)
        .expect("raster refs finalize should run");
    let output = materialize_output_decode_state_for_api(raster_state, output_refs)
        .expect("raster refs output should materialize");

    assert_eq!(output.generated_token_ids, expected.generated_token_ids);
    assert_eq!(output.generated_text, expected.generated_text);
    assert_eq!(
        output.generated_token_ids_sha256,
        expected.generated_token_ids_sha256
    );
    assert_eq!(output.generated_token_count, expected.generated_token_count);
}

#[test]
fn run_raster_from_decode_state_refs_handles_missing_generated_ref_as_empty_output(
) -> anyhow::Result<()> {
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );
    ArtifactIo::reset_store();
    let roots = ArtifactIo::export_store_roots();
    let (roots, full_token_ids_ref) = insert_token_ids(roots, "output.finalize.full.empty", &[9])?;
    let (roots, logits_ref) = insert_logits(roots, "output.finalize.logits.empty")?;
    let raster_state = RasterDecodeLoopState::new(
        roots,
        Some(full_token_ids_ref),
        1,
        None,
        0,
        logits_ref,
        1,
        Vec::new(),
        1,
        1,
        None,
    )?;

    let output_refs = run_raster(raster_state.clone(), &tokenizer_source, 2)
        .expect("raster refs finalize should run");
    let output = materialize_output_decode_state_for_api(raster_state, output_refs)
        .expect("raster refs output should materialize");

    assert_eq!(output.generated_token_ids, Vec::<u32>::new());
    assert_eq!(output.generated_text, "");
    assert_eq!(output.generated_token_count, 0);
    Ok(())
}

#[test]
fn selected_raster_detour_matches_native_finalize() {
    let tokenizer_json = minimal_gemma_tokenizer_json();
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json.as_bytes()).expect("HF tokenizer should parse");
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(tokenizer_json.as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );
    let decode_state = decode_state(vec![9, 4, 3], vec![4, 3]);

    ArtifactIo::reset_store();
    let native = run(decode_state.clone(), &tokenizer).expect("native finalize should run");
    ArtifactIo::reset_store();
    let detour = run_selected_raster_detour_from_native_boundary(
        decode_state,
        &tokenizer_source,
        raster_sizing(2),
    )
    .expect("selected output finalize detour should run");

    assert_eq!(detour.generated_token_ids, native.generated_token_ids);
    assert_eq!(
        detour.generated_token_ids_sha256,
        native.generated_token_ids_sha256
    );
    assert_eq!(detour.generated_text, native.generated_text);
    assert_eq!(detour.generated_token_count, native.generated_token_count);
    assert_eq!(detour.stop_reason, native.stop_reason);
}

#[test]
fn selected_raster_detour_returns_empty_generation_for_zero_tokens() {
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );

    ArtifactIo::reset_store();
    let output = run_selected_raster_detour_from_native_boundary(
        decode_state(vec![9], vec![]),
        &tokenizer_source,
        raster_sizing(2),
    )
    .expect("selected output finalize detour should run");

    assert_eq!(output.generated_token_ids, Vec::<u32>::new());
    assert_eq!(output.generated_text, "");
    assert_eq!(output.generated_token_count, 0);
}

#[test]
fn selected_raster_detour_surfaces_detokenization_errors() {
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );

    ArtifactIo::reset_store();
    let error = run_selected_raster_detour_from_native_boundary(
        decode_state(vec![9, 99], vec![99]),
        &tokenizer_source,
        raster_sizing(2),
    )
    .expect_err("missing output token should fail");

    assert!(error.to_string().contains("token id 99 is missing"));
}

#[test]
fn selected_raster_detour_rejects_zero_byte_flush_chunk_size() {
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );

    ArtifactIo::reset_store();
    let error = run_selected_raster_detour_from_native_boundary(
        decode_state(vec![9, 6], vec![6]),
        &tokenizer_source,
        raster_sizing(0),
    )
    .expect_err("zero byte flush chunk size should fail");

    assert!(error.to_string().contains("greater than zero"));
}

#[test]
fn selected_raster_detour_byte_flush_chunk_size_does_not_change_output() {
    let tokenizer_source = AuthenticatedGemmaTokenizer::new(
        parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("Gemma tokenizer spec should parse"),
    );
    let generated_token_ids = std::iter::repeat([6, 7])
        .take(20)
        .flatten()
        .collect::<Vec<_>>();
    let full_token_ids = [vec![9], generated_token_ids.clone()].concat();

    ArtifactIo::reset_store();
    let narrow = run_selected_raster_detour_from_native_boundary(
        decode_state(full_token_ids.clone(), generated_token_ids.clone()),
        &tokenizer_source,
        raster_sizing(1),
    )
    .expect("narrow byte flush chunks should run");
    ArtifactIo::reset_store();
    let wide = run_selected_raster_detour_from_native_boundary(
        decode_state(full_token_ids, generated_token_ids),
        &tokenizer_source,
        raster_sizing(64),
    )
    .expect("wide byte flush chunks should run");

    assert_eq!(narrow.generated_token_ids, wide.generated_token_ids);
    assert_eq!(
        narrow.generated_token_ids_sha256,
        wide.generated_token_ids_sha256
    );
    assert_eq!(narrow.generated_text, wide.generated_text);
    assert_eq!(wide.generated_text, "é".repeat(20));
    assert_eq!(narrow.generated_token_count, wide.generated_token_count);
}

fn decode_state(full_token_ids: Vec<u32>, generated_token_ids: Vec<u32>) -> DecodeState {
    let token_count = full_token_ids.len();
    let mut state = DecodeState::new(
        full_token_ids,
        vec![0.0],
        TransformerDecodeState {
            layer_caches: Vec::new(),
            position: token_count,
            token_count,
        },
    );
    state.set_internal_logits(InternalLogits::from_det_values(vec![Act::from_bits(0)]));
    state.generated_token_ids = generated_token_ids;
    state
}

fn raster_sizing(output_byte_flush_bytes_per_tile: usize) -> crate::RasterSizingControls {
    if output_byte_flush_bytes_per_tile == 0 {
        return crate::RasterSizingControls {
            projection_rows_per_tile: InferenceControls::DEFAULT_RASTER_PROJECTION_ROWS_PER_TILE,
            attention_kv_rows_per_tile:
                InferenceControls::DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE,
            sequence_rows_per_tile: InferenceControls::DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE,
            head_rows_per_tile: InferenceControls::DEFAULT_RASTER_HEAD_ROWS_PER_TILE,
            prefill_token_range_width: InferenceControls::DEFAULT_PREFILL_TOKEN_RANGE_WIDTH,
            decode_layer_range_width: InferenceControls::DEFAULT_DECODE_LAYER_RANGE_WIDTH,
            tokenizer_bpe_pairs_per_tile:
                InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE,
            tokenizer_bpe_pieces_per_tile:
                InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE,
            output_byte_flush_bytes_per_tile,
        };
    }

    InferenceControls {
        raster_output_byte_flush_bytes_per_tile: Some(output_byte_flush_bytes_per_tile),
        ..InferenceControls::default()
    }
    .raster_sizing_controls()
    .expect("raster sizing controls should build")
}

fn raster_decode_state_refs(state: &DecodeState) -> anyhow::Result<RasterDecodeLoopState> {
    ArtifactIo::reset_store();
    let roots = ArtifactIo::export_store_roots();
    let (roots, full_token_ids_ref) =
        insert_token_ids(roots, "output.finalize.full", &state.full_token_ids)?;
    let (roots, generated_token_ids_ref) = insert_token_ids(
        roots,
        "output.finalize.generated",
        &state.generated_token_ids,
    )?;
    let (roots, logits_ref) = insert_logits(roots, "output.finalize.logits")?;
    RasterDecodeLoopState::new(
        roots,
        Some(full_token_ids_ref),
        state.full_token_ids.len(),
        Some(generated_token_ids_ref),
        state.generated_token_ids.len(),
        logits_ref,
        1,
        Vec::new(),
        state.full_token_ids.len(),
        state.full_token_ids.len(),
        None,
    )
}

fn insert_token_ids(
    roots: RasterArtifactStoreRoots,
    source_name: &str,
    token_ids: &[u32],
) -> anyhow::Result<(RasterArtifactStoreRoots, RasterTokenIdSequenceRef)> {
    let leaves = token_ids.iter().copied().map(token_id_leaf).collect();
    let (roots, token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
        &roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?;
    Ok((roots, RasterTokenIdSequenceRef::new(token_ids_ref)?))
}

fn insert_logits(
    roots: RasterArtifactStoreRoots,
    source_name: &str,
) -> anyhow::Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (roots, logits_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        &roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::activation_rows(1, 1)?,
        vec![activation_row_leaf(&RasterActivationRow::from_acts(vec![
            Act::from_bits(0),
        ]))],
    )?;
    let logits_ref = activation_sequence_ref_from_artifact(
        RasterTensorId::new(source_name)?,
        RasterActivationSequenceArtifactRef::new(logits_artifact_ref)?,
    )?;
    Ok((roots, logits_ref))
}

fn minimal_gemma_tokenizer_json() -> String {
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {
                "id": 5,
                "content": "<bos>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ],
        "normalizer": {
            "type": "Replace",
            "pattern": { "String": " " },
            "content": "▁"
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": { "String": " " },
            "behavior": "MergedWithPrevious",
            "invert": false
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                }
            ],
            "pair": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                },
                {
                    "Sequence": {
                        "id": "B",
                        "type_id": 1
                    }
                }
            ],
            "special_tokens": {}
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": { "String": "▁" },
                    "content": " "
                },
                {
                    "type": "ByteFallback"
                },
                {
                    "type": "Fuse"
                }
            ]
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": {
                "<unk>": 0,
                "a": 1,
                "b": 2,
                "ab": 3,
                "▁": 4,
                "<bos>": 5,
                "<0xC3>": 6,
                "<0xA9>": 7
            },
            "merges": [
                ["a", "b"]
            ]
        }
    })
    .to_string()
}
