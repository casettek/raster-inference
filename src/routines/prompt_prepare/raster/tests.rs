use anyhow::Result;

use super::super::native::{
    build_gemma4_messages, build_prompt_commitment, decode_prompt_bytes, render_prompt,
};
use super::utils::{
    bpe_piece_leaf, init_artifact_store, init_bpe_tokenize_prompt, init_tokenize_prompt,
    prepare_raster_prompt_input_roots, read_bpe_piece, tokenize_prompt,
    tokenize_prompt_with_controls,
};
use super::{
    bpe_pieces_artifact_name, finalize_next_token_ids, finalize_tokenize_prompt,
    init_token_id_finalization, main, prompt_token_ids_root,
};
use crate::shared::api::input::{MessageRole, ModelSpec, TextDecodingPolicy};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    self, RasterArtifactId, RasterBpePieceSequenceRef, RasterTokenIdSequenceRef,
};
use crate::shared::model::gemma::tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeOutput,
    GemmaPreTokenizedText, GemmaTokenizerSpec, GemmaVocabEntry,
};

#[test]
fn decode_prompt_bytes_preserves_prompt_text() {
    let prompt = decode_prompt_bytes(b"  hello world  ", TextDecodingPolicy::Utf8)
        .expect("prompt should decode");

    assert_eq!(prompt, "  hello world  ");
}

#[test]
fn decode_prompt_bytes_rejects_invalid_utf8() {
    let error = decode_prompt_bytes(&[0xFF], TextDecodingPolicy::Utf8)
        .expect_err("invalid utf-8 should fail");

    assert!(error.to_string().contains("utf-8"));
}

#[test]
fn build_gemma4_messages_wraps_prompt_as_single_user_message() {
    let prompt = build_gemma4_messages("hello", true).expect("messages should build");

    assert_eq!(prompt.messages.len(), 1);
    assert_eq!(prompt.messages[0].role, MessageRole::User);
    assert_eq!(prompt.messages[0].content, "hello");
    assert!(prompt.add_generation_prompt);
}

#[test]
fn render_prompt_uses_messages_and_generation_flag() {
    let model = ModelSpec {
        model_id: "gemma-4-test".to_string(),
        tokenizer_path: "tokenizer.json".into(),
        chat_template: "{{ bos_token }}{% for message in messages %}[{{ message.role }}] {{ message.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}".to_string(),
        bos_token: Some("<bos>".to_string()),
        eos_token: None,
        unk_token: None,
    };
    let prompt = build_gemma4_messages("hello", true).expect("messages should build");
    let prompt = render_prompt(&prompt, &model).expect("prompt should render");

    assert_eq!(prompt, "<bos>[user] hello[assistant]");
}

#[test]
fn init_tokenize_prompt_captures_rendered_prompt_and_special_token_policy() {
    let input =
        init_tokenize_prompt("<bos>[user] hello[assistant]", true).expect("input should build");

    assert_eq!(input.rendered_prompt, "<bos>[user] hello[assistant]");
    assert!(input.add_special_tokens);
}

#[test]
fn finalize_tokenize_prompt_returns_token_id_root() {
    let tokenizer = test_tokenizer_source();
    init_artifact_store();
    let _pieces_ref = insert_bpe_piece_sequence_for_test(
        &bpe_pieces_artifact_name(0),
        vec!["a".to_string(), "ab".to_string()],
    )
    .expect("pieces should insert");
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let mut state = init_token_id_finalization(
        artifact_store_roots,
        GemmaBpeOutput {
            piece_count: 2,
            iteration: 0,
            add_special_tokens: false,
            bpe_pieces_per_tile: 1,
        },
    )
    .expect("token id finalization should init");
    loop {
        let (done, next_state) =
            finalize_next_token_ids(state, &tokenizer).expect("token ids should advance");
        state = next_state;
        if done {
            break;
        }
    }
    let (artifact_store_roots, tokenization) =
        finalize_tokenize_prompt(state).expect("token ids should finalize");
    let token_ids_root =
        prompt_token_ids_root(&artifact_store_roots).expect("token ids root should be present");
    let token_ids = materialize_token_ids(token_ids_root, tokenization.token_count)
        .expect("token ids should materialize");

    assert_eq!(token_ids, vec![1, 3]);
    assert!(artifact_store_roots
        .artifact_entry_for_root(token_ids_root)
        .is_ok());
    let serialized = serde_json::to_string(&tokenization).expect("tokenization should serialize");
    assert!(!serialized.contains("token_ids_ref"));
}

#[test]
fn tokenize_prompt_applies_recursive_bpe_merges() {
    let tokenization =
        tokenize_prompt("ab", &test_tokenizer_source(), false).expect("prompt should tokenize");
    let token_ids_root = ArtifactIo::export_store_roots()
        .artifact_root_for_source_name("prompt-token-ids")
        .expect("token ids root should be present")
        .to_string();
    let token_ids = materialize_token_ids(&token_ids_root, tokenization.token_count)
        .expect("token ids should materialize");

    assert_eq!(token_ids, vec![3]);
}

#[test]
fn init_bpe_tokenize_prompt_returns_compact_ref_state() {
    let tokenizer = test_tokenizer_source();
    init_artifact_store();
    let state = init_bpe_tokenize_prompt(
        GemmaPreTokenizedText {
            segments: vec!["ab".to_string()],
            add_special_tokens: false,
        },
        &tokenizer,
        1,
        1,
    )
    .expect("BPE init should build ref state");

    let serialized = serde_json::to_string(&state).expect("state should serialize");
    assert!(!serialized.contains(r#""a""#));
    assert!(!serialized.contains(r#""b""#));
    assert!(!serialized.contains("pieces_ref"));
    assert_eq!(
        materialize_bpe_pieces(
            ArtifactIo::export_store_roots()
                .artifact_root_for_source_name("bpe-pieces-0")
                .expect("BPE pieces root should be present"),
            state.piece_count,
        )
        .expect("pieces should materialize"),
        vec!["a", "b"]
    );
}

#[test]
fn tokenize_prompt_uses_byte_fallback_for_unknown_chars() {
    let tokenization =
        tokenize_prompt("é", &test_tokenizer_source(), false).expect("prompt should tokenize");
    let token_ids_root = ArtifactIo::export_store_roots()
        .artifact_root_for_source_name("prompt-token-ids")
        .expect("token ids root should be present")
        .to_string();
    let token_ids = materialize_token_ids(&token_ids_root, tokenization.token_count)
        .expect("token ids should materialize");

    assert_eq!(token_ids, vec![10, 11]);
}

#[test]
fn tokenize_prompt_chunk_sizes_do_not_change_results() {
    let tokenizer = test_tokenizer_source();
    let tiny_chunks =
        tokenize_prompt_with_controls("aba", &tokenizer, false, 1, 1).expect("tiny chunks");
    let tiny_root = ArtifactIo::export_store_roots()
        .artifact_root_for_source_name("prompt-token-ids")
        .expect("tiny token ids root should be present")
        .to_string();
    let tiny_token_ids = materialize_token_ids(&tiny_root, tiny_chunks.token_count)
        .expect("tiny token ids should materialize");

    let larger_chunks =
        tokenize_prompt_with_controls("aba", &tokenizer, false, 8, 8).expect("larger chunks");
    let larger_root = ArtifactIo::export_store_roots()
        .artifact_root_for_source_name("prompt-token-ids")
        .expect("larger token ids root should be present")
        .to_string();
    let larger_token_ids = materialize_token_ids(&larger_root, larger_chunks.token_count)
        .expect("larger token ids should materialize");
    assert_eq!(tiny_token_ids, larger_token_ids);
    assert_eq!(tiny_token_ids, vec![12]);
}

#[test]
fn run_returns_root_backed_prompt_state() {
    let request = crate::shared::api::input::InferenceRequest {
        prompt_bytes: b"ab".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        execution_mode: crate::shared::api::input::InferenceExecutionMode::Deterministic,
        sampling: crate::shared::api::input::SamplingConfig::default(),
    };
    let model = ModelSpec {
        model_id: "gemma-4-test".to_string(),
        tokenizer_path: "tokenizer.json".into(),
        chat_template: "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        bos_token: None,
        eos_token: None,
        unk_token: None,
    };

    let tokenizer = test_tokenizer_source();
    let prepared_inputs = prepare_raster_prompt_input_roots(&request, &model, &tokenizer, 1, 1)
        .expect("raster prompt roots should build");
    let result = main(
        prepared_inputs.artifact_store_roots,
        prepared_inputs.input_roots,
        &tokenizer,
    )
    .expect("raster prompt refs should build");
    let token_ids = materialize_token_ids(
        &result.state.prompt_token_ids_root,
        result.state.prompt_token_count,
    )
    .expect("token ids should materialize for test assertions");
    let serialized = serde_json::to_string(&result.state).expect("state should serialize");

    assert_eq!(token_ids, vec![3]);
    assert_eq!(result.state.prompt_token_count, 1);
    assert!(!serialized.contains("prompt_token_ids\":["));
    assert!(!serialized.contains("prompt_token_ids_ref"));
    assert!(!serialized.contains("prompt_bytes_ref"));
    assert!(result
        .artifact_store_roots
        .artifact_entry_for_root(&result.state.prompt_token_ids_root)
        .is_ok());
}

#[test]
fn build_prompt_commitment_hashes_prompt_token_ids_only() {
    let digest = build_prompt_commitment(&[1, 2, 3]).expect("commitment should build");

    assert_eq!(
        digest,
        "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4"
    );
}

fn insert_bpe_piece_sequence_for_test(
    name: &str,
    pieces: Vec<String>,
) -> Result<RasterBpePieceSequenceRef> {
    let mut builder = raster_artifact_store::start_builder(
        RasterArtifactId::new(name).expect("artifact id"),
        raster_artifact_store::RasterArtifactMetadata::bpe_pieces(pieces.len()),
    )?;
    for (piece_idx, piece) in pieces.iter().enumerate() {
        raster_artifact_store::append_leaf(&mut builder, piece_idx, bpe_piece_leaf(piece))?;
    }
    RasterBpePieceSequenceRef::new(raster_artifact_store::finalize_builder(builder)?)
}

fn materialize_bpe_pieces(pieces_root: &str, piece_count: usize) -> Result<Vec<String>> {
    (0..piece_count)
        .map(|piece_idx| read_bpe_piece(pieces_root, piece_idx))
        .collect()
}

fn materialize_token_ids(token_ids_root: &str, token_count: usize) -> Result<Vec<u32>> {
    let token_ids_ref = RasterTokenIdSequenceRef::new(
        raster_artifact_store::artifact_ref_for_root(token_ids_root)?,
    )?;
    (0..token_count)
        .map(|token_idx| {
            raster_artifact_store::read_authenticated_leaf(token_ids_ref.artifact_ref(), token_idx)?
                .deserialize()
        })
        .collect()
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
                token: "aba".to_string(),
                id: 12,
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
                token: GemmaTokenizerSpec::byte_fallback_token(0xC3),
                id: 10,
            },
            GemmaVocabEntry {
                token: GemmaTokenizerSpec::byte_fallback_token(0xA9),
                id: 11,
            },
        ],
        vec![
            GemmaBpeMerge {
                left: "a".to_string(),
                right: "b".to_string(),
                merged: "ab".to_string(),
                rank: 0,
            },
            GemmaBpeMerge {
                left: "ab".to_string(),
                right: "a".to_string(),
                merged: "aba".to_string(),
                rank: 1,
            },
        ],
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
