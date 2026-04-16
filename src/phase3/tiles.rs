use anyhow::{bail, Result};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use super::types::Phase3StopReason;
use crate::phase1::SamplingConfig;

const DEFAULT_TEMPERATURE: f32 = 1.0;

pub fn select_next_token(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() {
        bail!("phase 3 requires at least one logit to select the next token");
    }

    let mut best_token = 0usize;
    for (token_id, logit) in logits.iter().enumerate().skip(1) {
        if logit.total_cmp(&logits[best_token]).is_gt() {
            best_token = token_id;
        }
    }

    Ok(best_token as u32)
}

pub fn append_token(token_ids: &[u32], next_token: u32) -> Vec<u32> {
    let mut appended = token_ids.to_vec();
    appended.push(next_token);
    appended
}

pub fn check_stop_condition(
    generated_token_count: usize,
    max_new_tokens: usize,
) -> Option<Phase3StopReason> {
    (generated_token_count >= max_new_tokens).then_some(Phase3StopReason::MaxNewTokens)
}

pub fn detokenize_output_tokens(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    if token_ids.is_empty() {
        return Ok(String::new());
    }

    tokenizer
        .decode(token_ids, true)
        .map_err(anyhow::Error::msg)
}

pub fn validate_sampling_config(sampling: &SamplingConfig) -> Result<usize> {
    if let Some(temperature) = sampling.temperature {
        if (temperature - DEFAULT_TEMPERATURE).abs() > f32::EPSILON {
            bail!(
                "phase 3 only supports deterministic greedy decode; expected temperature {DEFAULT_TEMPERATURE}, got {temperature}"
            );
        }
    }
    if let Some(top_k) = sampling.top_k {
        bail!("phase 3 does not support top_k yet, got {top_k}");
    }
    if let Some(top_p) = sampling.top_p {
        bail!("phase 3 does not support top_p yet, got {top_p}");
    }

    Ok(sampling.max_new_tokens.unwrap_or(0))
}

pub fn build_phase3_commitment(token_ids: &[u32]) -> Result<String> {
    let payload =
        serde_json::to_vec(token_ids).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let digest = Sha256::digest(payload);
    Ok(format!("{digest:x}"))
}

#[cfg(test)]
mod tests {
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::{
        append_token, build_phase3_commitment, check_stop_condition, detokenize_output_tokens,
        select_next_token, validate_sampling_config,
    };
    use crate::phase1::SamplingConfig;
    use crate::phase3::Phase3StopReason;

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
            Some(Phase3StopReason::MaxNewTokens)
        );
    }

    #[test]
    fn check_stop_condition_immediately_finishes_when_limit_is_zero() {
        assert_eq!(
            check_stop_condition(0, 0),
            Some(Phase3StopReason::MaxNewTokens)
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
    fn build_phase3_commitment_hashes_generated_token_ids_only() {
        let digest = build_phase3_commitment(&[4, 5]).expect("commitment should build");
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
