use anyhow::{bail, Result};
use crate::shared::input::SamplingConfig;

pub fn validate_sampling_config(sampling: &SamplingConfig) -> Result<usize> {
    const DEFAULT_TEMPERATURE: f32 = 1.0;
    if let Some(temperature) = sampling.temperature {
        if (temperature - DEFAULT_TEMPERATURE).abs() > f32::EPSILON {
            bail!(
                "output decode only supports deterministic greedy decode; expected temperature {DEFAULT_TEMPERATURE}, got {temperature}"
            );
        }
    }
    if let Some(top_k) = sampling.top_k {
        bail!("output decode does not support top_k yet, got {top_k}");
    }
    if let Some(top_p) = sampling.top_p {
        bail!("output decode does not support top_p yet, got {top_p}");
    }

    Ok(sampling.max_new_tokens.unwrap_or(0))
}

#[allow(unused_imports)]
pub use crate::decode_select_token::tiles::{append_token, check_stop_condition, select_next_token};
#[allow(unused_imports)]
pub use crate::output_finalize::tiles::{build_output_decode_commitment, detokenize_output_tokens};

#[cfg(test)]
mod tests {
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::{
        append_token, build_output_decode_commitment, check_stop_condition,
        detokenize_output_tokens, select_next_token, validate_sampling_config,
    };
    use crate::shared::input::SamplingConfig;
    use crate::shared::output::OutputDecodeStopReason;

    #[test]
    fn select_next_token_returns_highest_logit_token_id() {
        let token_id = select_next_token(&[-2.0, 0.25, 4.0, 0.5]).expect("token selection");
        assert_eq!(token_id, 2);
    }

    #[test]
    fn select_next_token_breaks_equal_logits_by_lowest_token_id() {
        let token_id = select_next_token(&[1.0, 3.0, 3.0, 2.0]).expect("token selection");
        assert_eq!(token_id, 1);
    }

    #[test]
    fn select_next_token_rejects_empty_logits() {
        let error = select_next_token(&[]).expect_err("empty logits should fail");
        assert!(error.to_string().contains("at least one logit"));
    }

    #[test]
    fn append_token_returns_new_sequence() {
        let original = vec![1, 2];
        let appended = append_token(&original, 3);
        assert_eq!(original, vec![1, 2]);
        assert_eq!(appended, vec![1, 2, 3]);
    }

    #[test]
    fn check_stop_condition_finishes_at_max_new_tokens() {
        assert_eq!(
            check_stop_condition(2, 2),
            Some(OutputDecodeStopReason::MaxNewTokens)
        );
    }

    #[test]
    fn check_stop_condition_immediately_finishes_when_limit_is_zero() {
        assert_eq!(
            check_stop_condition(0, 0),
            Some(OutputDecodeStopReason::MaxNewTokens)
        );
    }

    #[test]
    fn validate_sampling_config_rejects_non_default_temperature() {
        let error = validate_sampling_config(&SamplingConfig {
            max_new_tokens: Some(4),
            temperature: Some(0.7),
            top_k: None,
            top_p: None,
        })
        .expect_err("non-default temperature should fail");

        assert!(error.to_string().contains("temperature"));
    }

    #[test]
    fn detokenize_output_tokens_decodes_generated_ids() {
        let tokenizer = test_tokenizer();
        let text =
            detokenize_output_tokens(&tokenizer, &[0, 1]).expect("generated ids should decode");
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
}
