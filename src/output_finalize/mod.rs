use anyhow::Result;
use serde_json::json;
use tokenizers::Tokenizer;

use crate::shared::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::output::{DecodeState, OutputDecodeState};

pub mod raster_tiles;
pub mod tiles;

pub fn run(decode_state: DecodeState, tokenizer: &Tokenizer) -> Result<OutputDecodeState> {
    let generated_token_count = decode_state.generated_token_ids.len();
    let generated_text =
        tiles::detokenize_output_tokens(tokenizer, &decode_state.generated_token_ids)?;
    let generated_token_ids_sha256 =
        tiles::build_output_decode_commitment(&decode_state.generated_token_ids)?;
    let stop_reason = crate::shared::output::OutputDecodeStopReason::MaxNewTokens;
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": generated_token_ids_sha256.clone(),
            "generated_text": generated_text.clone(),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason.clone(),
        }),
    );

    Ok(OutputDecodeState {
        generated_token_ids: decode_state.generated_token_ids,
        generated_token_ids_sha256,
        generated_text,
        generated_token_count,
        stop_reason,
        decode_transition_states: Vec::new(),
    })
}

pub fn run_raster(
    decode_state: DecodeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDecodeState> {
    run_raster_with_byte_flush_bytes_per_tile(
        decode_state,
        tokenizer,
        raster_tiles::DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    )
}

pub fn run_raster_with_byte_flush_bytes_per_tile(
    decode_state: DecodeState,
    tokenizer: &AuthenticatedGemmaTokenizer,
    byte_flush_bytes_per_tile: usize,
) -> Result<OutputDecodeState> {
    let generated_token_count = decode_state.generated_token_ids.len();
    let output = raster_tiles::run_with_byte_flush_bytes_per_tile(
        &decode_state.generated_token_ids,
        tokenizer,
        byte_flush_bytes_per_tile,
    )?;
    let stop_reason = output.stop_reason.clone();
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": output.generated_token_ids.clone(),
            "generated_token_ids_sha256": output.generated_token_ids_sha256.clone(),
            "generated_text": output.generated_text.clone(),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason,
        }),
    );

    Ok(output)
}

pub fn run_raster_with_roots(
    decode_state: DecodeState,
    input_roots: raster_tiles::RasterOutputFinalizeInputRoots,
    tokenizer: &AuthenticatedGemmaTokenizer,
) -> Result<OutputDecodeState> {
    let generated_token_count = input_roots.generated_token_ids_ref.token_count();
    let refs = raster_tiles::main(input_roots, tokenizer)?;
    let generated_token_ids = raster_tiles::materialize_token_ids_from_roots(
        &refs.artifact_store_roots,
        &refs.generated_token_ids_ref,
    )?;
    let generated_text = crate::shared::raster_output_finalize::materialize_text_from_roots(
        &refs.artifact_store_roots,
        &refs.generated_text_ref,
    )?;
    let stop_reason = refs.stop_reason.clone();
    crate::trace::trace_checkpoint(
        "output.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": generated_token_ids.clone(),
            "generated_token_ids_sha256": refs.generated_token_ids_sha256.clone(),
            "generated_text": generated_text.clone(),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason,
        }),
    );

    Ok(OutputDecodeState {
        generated_token_count,
        generated_token_ids,
        generated_token_ids_sha256: refs.generated_token_ids_sha256,
        generated_text,
        stop_reason: refs.stop_reason,
        decode_transition_states: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::{run, run_raster};
    use crate::io::parse_gemma_tokenizer_spec_bytes;
    use crate::shared::{
        gemma_tokenizer::AuthenticatedGemmaTokenizer, output::DecodeState,
        transformer::TransformerDecodeState,
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
        let raster =
            run_raster(decode_state, &tokenizer_source).expect("raster finalize should run");

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

    fn decode_state(full_token_ids: Vec<u32>, generated_token_ids: Vec<u32>) -> DecodeState {
        let mut state = DecodeState::new(full_token_ids, vec![], TransformerDecodeState::default());
        state.generated_token_ids = generated_token_ids;
        state
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
}
