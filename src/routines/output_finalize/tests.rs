use super::{run, run_raster, run_raster_from_decode_state_refs};
use crate::io::parse_gemma_tokenizer_spec_bytes;
use crate::shared::api::output::DecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, token_id_leaf, RasterActivationSequenceArtifactRef, RasterArtifactId,
    RasterArtifactMetadata, RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::TransformerDecodeState;
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
    let raster = run_raster(decode_state, &tokenizer_source).expect("raster finalize should run");

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

    let output = run_raster(decode_state(vec![9], vec![]), &tokenizer_source)
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

    let error = run_raster(decode_state(vec![9, 99], vec![99]), &tokenizer_source)
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

    let output = run_raster_from_decode_state_refs(raster_state, &tokenizer_source, 2)
        .expect("raster refs finalize should run");

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

    let output = run_raster_from_decode_state_refs(raster_state, &tokenizer_source, 2)
        .expect("raster refs finalize should run");

    assert_eq!(output.generated_token_ids, Vec::<u32>::new());
    assert_eq!(output.generated_text, "");
    assert_eq!(output.generated_token_count, 0);
    Ok(())
}

fn decode_state(full_token_ids: Vec<u32>, generated_token_ids: Vec<u32>) -> DecodeState {
    let mut state = DecodeState::new(full_token_ids, vec![], TransformerDecodeState::default());
    state.generated_token_ids = generated_token_ids;
    state
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
