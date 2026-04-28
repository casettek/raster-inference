use raster_inference::io::parse_gemma_tokenizer_spec_bytes;
use raster_inference::load_gemma_tokenizer_spec_from_path;
use raster_inference::prompt_prepare::raster_tiles::tokenize_prompt;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[test]
fn raster_gemma_tokenizer_matches_huggingface_for_supported_subset() {
    let tokenizer_json = minimal_gemma_tokenizer_json();
    let spec = parse_gemma_tokenizer_spec_bytes(tokenizer_json.as_bytes())
        .expect("Gemma tokenizer spec should parse");
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json.as_bytes()).expect("HF tokenizer should parse");

    for prompt in ["ab", "a b", "<bos>ab", "é"] {
        let hf_ids = tokenizer
            .encode_fast(prompt, false)
            .expect("HF tokenizer should encode")
            .get_ids()
            .to_vec();
        let raster_ids =
            tokenize_prompt(prompt, &spec, false).expect("Raster tokenizer should encode");

        assert_eq!(raster_ids, hf_ids, "token ids should match for {prompt:?}");
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
