use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

use super::{build_output_decode_commitment, detokenize_output_tokens};

#[test]
fn detokenize_output_tokens_decodes_generated_ids() {
    let tokenizer = test_tokenizer();
    let text = detokenize_output_tokens(&tokenizer, &[0, 1]).expect("generated ids should decode");
    assert_eq!(text, "hello world");
}

#[test]
fn build_output_decode_commitment_hashes_generated_token_ids_only() {
    let digest = build_output_decode_commitment(&[4, 5]).expect("commitment should build");
    assert_eq!(
        digest,
        "d4c7a98da55490b0a5a65cc5057db99aa708a436609b177748505342d569457b"
    );
}

fn test_tokenizer() -> tokenizers::Tokenizer {
    let vocab = [
        ("hello".to_string(), 0),
        ("world".to_string(), 1),
        ("<unk>".to_string(), 2),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("word level tokenizer");
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}
