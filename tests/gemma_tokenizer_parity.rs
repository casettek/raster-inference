use anyhow::Result;
use raster_inference::io::parse_gemma_tokenizer_spec_bytes;
use raster_inference::load_gemma_tokenizer_spec_from_path;
use raster_inference::output_finalize::raster_tiles::detokenize_output_tokens;
use raster_inference::prompt_prepare::raster_tiles::{
    tokenize_prompt, tokenize_prompt_with_controls,
};
use raster_inference::shared::artifacts::raster_artifact_store::{self, RasterTokenIdSequenceRef};
use raster_inference::shared::model::gemma_tokenizer::GemmaDecoderMetadataRequest;
use raster_inference::AuthenticatedGemmaTokenizer;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[test]
fn raster_gemma_tokenizer_matches_huggingface_for_supported_subset() {
    let tokenizer_json = minimal_gemma_tokenizer_json();
    let spec = parse_gemma_tokenizer_spec_bytes(tokenizer_json.as_bytes())
        .expect("Gemma tokenizer spec should parse");
    let source = AuthenticatedGemmaTokenizer::new(spec);
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json.as_bytes()).expect("HF tokenizer should parse");

    for prompt in ["ab", "a b", "<bos>ab", "é"] {
        let hf_ids = tokenizer
            .encode_fast(prompt, false)
            .expect("HF tokenizer should encode")
            .get_ids()
            .to_vec();
        let raster =
            tokenize_prompt(prompt, &source, false).expect("Raster tokenizer should encode");
        let raster_ids = materialize_token_ids(&raster.token_ids_root, raster.token_count)
            .expect("Raster token ids should materialize");

        assert_eq!(raster_ids, hf_ids, "token ids should match for {prompt:?}");
    }
}

#[test]
fn raster_gemma_tokenizer_chunk_controls_do_not_change_supported_subset() {
    let tokenizer_json = minimal_gemma_tokenizer_json();
    let spec = parse_gemma_tokenizer_spec_bytes(tokenizer_json.as_bytes())
        .expect("Gemma tokenizer spec should parse");
    let source = AuthenticatedGemmaTokenizer::new(spec);

    for prompt in ["abab", "a b ab", "<bos>abab", "éab"] {
        let tiny_chunk =
            tokenize_prompt_with_controls(prompt, &source, false, 1, 1).expect("tiny chunks");
        let tiny_chunk_ids =
            materialize_token_ids(&tiny_chunk.token_ids_root, tiny_chunk.token_count)
                .expect("tiny chunk token ids should materialize");
        let oversized_chunk =
            tokenize_prompt_with_controls(prompt, &source, false, 64, 64).expect("large chunks");
        let oversized_chunk_ids =
            materialize_token_ids(&oversized_chunk.token_ids_root, oversized_chunk.token_count)
                .expect("oversized chunk token ids should materialize");

        assert_eq!(
            tiny_chunk_ids, oversized_chunk_ids,
            "chunking should not change token ids for {prompt:?}"
        );
    }
}

#[test]
fn raster_gemma_detokenizer_matches_huggingface_for_supported_subset() {
    let tokenizer_json = minimal_gemma_tokenizer_json();
    let spec = parse_gemma_tokenizer_spec_bytes(tokenizer_json.as_bytes())
        .expect("Gemma tokenizer spec should parse");
    let source = AuthenticatedGemmaTokenizer::new(spec);
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json.as_bytes()).expect("HF tokenizer should parse");

    for token_ids in [
        vec![],
        vec![4, 3],
        vec![5, 4, 3],
        vec![6],
        vec![6, 4, 3],
        vec![6, 7],
        vec![6, 7, 4, 3],
        vec![4, 3, 6, 7],
    ] {
        let hf_text = tokenizer
            .decode(&token_ids, true)
            .expect("HF tokenizer should decode");
        let raster_text =
            detokenize_output_tokens(&token_ids, &source).expect("Raster tokenizer should decode");

        assert_eq!(
            raster_text, hf_text,
            "decoded text should match for {token_ids:?}"
        );
    }
}

#[test]
fn real_gemma_tokenizer_asset_converts_to_spec() {
    let mut tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    tokenizer_path.push("assets/gemma-4-E4B-it/tokenizer.json");

    let spec = load_gemma_tokenizer_spec_from_path(tokenizer_path)
        .expect("checked-in Gemma tokenizer asset should convert to spec");

    assert_eq!(spec.token_id("<unk>"), Some(spec.unk_token_id));
    assert!(spec.byte_fallback);
    assert!(!spec.merges.is_empty());
}

#[test]
fn unsupported_gemma_tokenizer_decoder_is_rejected() {
    let cases: Vec<(&str, Box<dyn Fn(&mut serde_json::Value)>, &str)> = vec![
        (
            "decoder type",
            Box::new(|json| json["decoder"]["type"] = serde_json::json!("Replace")),
            "unsupported Gemma tokenizer decoder",
        ),
        (
            "decoder length",
            Box::new(|json| {
                json["decoder"]["decoders"]
                    .as_array_mut()
                    .expect("decoder list")
                    .pop();
            }),
            "unsupported Gemma tokenizer decoder sequence length",
        ),
        (
            "replacement step",
            Box::new(|json| {
                json["decoder"]["decoders"][0]["pattern"]["String"] = serde_json::json!("_")
            }),
            "unsupported Gemma tokenizer decoder replacement",
        ),
        (
            "byte fallback step",
            Box::new(|json| {
                json["decoder"]["decoders"][1]["type"] = serde_json::json!("Metaspace")
            }),
            "unsupported Gemma tokenizer decoder byte fallback",
        ),
        (
            "fuse step",
            Box::new(|json| json["decoder"]["decoders"][2]["type"] = serde_json::json!("Strip")),
            "unsupported Gemma tokenizer decoder fuse",
        ),
    ];

    for (name, mutate, expected_error) in cases {
        let mut tokenizer_json: serde_json::Value =
            serde_json::from_str(&minimal_gemma_tokenizer_json()).expect("fixture should parse");
        mutate(&mut tokenizer_json);

        let error = parse_gemma_tokenizer_spec_bytes(tokenizer_json.to_string().as_bytes())
            .expect_err(&format!("{name} should fail with a supported error"));

        assert!(
            error.to_string().contains(expected_error),
            "{name} should contain {expected_error:?}, got {error}"
        );
    }
}

#[test]
fn gemma_tokenizer_without_decoder_still_loads_for_encode_only_use() {
    let mut tokenizer_json: serde_json::Value =
        serde_json::from_str(&minimal_gemma_tokenizer_json()).expect("fixture should parse");
    tokenizer_json
        .as_object_mut()
        .expect("fixture should be an object")
        .remove("decoder");

    let spec = parse_gemma_tokenizer_spec_bytes(tokenizer_json.to_string().as_bytes())
        .expect("encode-only tokenizer spec should parse without decoder metadata");
    let source = AuthenticatedGemmaTokenizer::new(spec);

    let encoded_ref = tokenize_prompt("ab", &source, false).expect("encode path should still work");
    let encoded = materialize_token_ids(&encoded_ref.token_ids_root, encoded_ref.token_count)
        .expect("encoded token ids should materialize");
    assert_eq!(encoded, vec![3]);
    let error = raster_inference::auth_read!(&source, GemmaDecoderMetadataRequest)
        .expect_err("decoder metadata should be required for raster output decode");
    assert!(error
        .to_string()
        .contains("missing supported decoder metadata"));
}

fn materialize_token_ids(token_ids_root: &str, token_count: usize) -> Result<Vec<u32>> {
    let token_ids_ref = RasterTokenIdSequenceRef::new(
        raster_artifact_store::artifact_ref_for_root(token_ids_root)?,
    )?;
    (0..token_count)
        .map(|token_idx| {
            let read = raster_artifact_store::read_leaf(token_ids_ref.artifact_ref(), token_idx)?;
            raster_artifact_store::verify_artifact_read(token_ids_ref.artifact_ref(), &read)?;
            decode_token_id_leaf(read.payload())
        })
        .collect()
}

fn decode_token_id_leaf(payload: &[u8]) -> Result<u32> {
    if payload.len() != 4 {
        anyhow::bail!("token-id leaf payload must be exactly four bytes");
    }
    Ok(u32::from_le_bytes(
        payload.try_into().expect("payload length checked above"),
    ))
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
