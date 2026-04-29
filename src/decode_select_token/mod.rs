use anyhow::Result;
use serde_json::json;

use crate::shared::output::DecodeState;
use crate::shared::{
    input::InferenceExecutionMode,
    raster_decode_select_token::AuthenticatedDecodeSelectLogitsSource,
};

pub mod raster_tiles;
pub mod tiles;

pub fn run(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
    execution_mode: InferenceExecutionMode,
) -> Result<Option<u32>> {
    if tiles::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens).is_some()
    {
        return Ok(None);
    }

    let next_token =
        tiles::select_next_token_internal(&decode_state.clone_internal_logits(), execution_mode)?;
    decode_state.full_token_ids = tiles::append_token(&decode_state.full_token_ids, next_token);
    decode_state.generated_token_ids =
        tiles::append_token(&decode_state.generated_token_ids, next_token);
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": crate::output_finalize::tiles::build_output_decode_commitment(&decode_state.generated_token_ids)?,
            "current_logits": decode_state.current_logits.clone(),
            "current_logits_sha256": crate::trace::sha256_hex(&decode_state.current_logits),
            "selected_next_token": next_token,
            "decode_position": decode_state.transformer_decode_state.position,
            "decode_token_count": decode_state.transformer_decode_state.token_count,
            "layer_caches": crate::trace::serialize_layer_caches(&decode_state.transformer_decode_state.layer_caches),
            "max_new_tokens": max_new_tokens,
        }),
    );
    Ok(Some(next_token))
}

pub fn run_raster(decode_state: &mut DecodeState, max_new_tokens: usize) -> Result<Option<u32>> {
    if raster_tiles::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens)
        .is_some()
    {
        return Ok(None);
    }

    let logits = decode_state.clone_internal_logits();
    let logits_source = AuthenticatedDecodeSelectLogitsSource::from_internal_logits(
        format!(
            "decode.select_token.position_{}",
            decode_state.transformer_decode_state.position
        ),
        &logits,
    )?;
    let output = raster_tiles::run(
        &decode_state.full_token_ids,
        &decode_state.generated_token_ids,
        max_new_tokens,
        &logits_source,
    )?
    .expect("stop condition should have returned earlier");

    decode_state.full_token_ids = output.full_token_ids;
    decode_state.generated_token_ids = output.generated_token_ids;
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": crate::output_finalize::tiles::build_output_decode_commitment(&decode_state.generated_token_ids)?,
            "current_logits": decode_state.current_logits.clone(),
            "current_logits_sha256": crate::trace::sha256_hex(&decode_state.current_logits),
            "det_current_logits_sha256": output.det_current_logits_sha256,
            "selected_next_token": output.next_token,
            "decode_position": decode_state.transformer_decode_state.position,
            "decode_token_count": decode_state.transformer_decode_state.token_count,
            "layer_caches": crate::trace::serialize_layer_caches(&decode_state.transformer_decode_state.layer_caches),
            "max_new_tokens": max_new_tokens,
        }),
    );
    Ok(Some(output.next_token))
}

#[cfg(test)]
mod tests {
    use super::run_raster;
    use crate::shared::{
        det_num::Act,
        output::DecodeState,
        transformer::{InternalLogits, TransformerDecodeState},
    };

    #[test]
    fn run_raster_appends_selected_token_to_decode_state() {
        let mut decode_state = decode_state_with_logits(vec![
            Act::from_bits(1),
            Act::from_bits(9),
            Act::from_bits(3),
        ]);

        let selected = run_raster(&mut decode_state, 1)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(selected, 1);
        assert_eq!(decode_state.full_token_ids, vec![7, 1]);
        assert_eq!(decode_state.generated_token_ids, vec![1]);
    }

    #[test]
    fn run_raster_uses_canonical_logits_over_public_f32_view() {
        let mut decode_state =
            decode_state_with_logits(vec![Act::from_bits(100), Act::from_bits(1)]);
        decode_state.current_logits = vec![0.0, 1000.0];

        let selected = run_raster(&mut decode_state, 1)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(selected, 0);
        assert_eq!(decode_state.generated_token_ids, vec![0]);
    }

    #[test]
    fn run_raster_rejects_f32_only_logits_without_mutating_decode_state() {
        let mut decode_state =
            DecodeState::new(vec![7], vec![0.0, 1.0], TransformerDecodeState::default());

        let error =
            run_raster(&mut decode_state, 1).expect_err("f32-only logits should fail in raster");

        assert!(error.to_string().contains("canonical deterministic logits"));
        assert_eq!(decode_state.full_token_ids, vec![7]);
        assert!(decode_state.generated_token_ids.is_empty());
    }

    #[test]
    fn run_raster_stops_without_requiring_canonical_logits() {
        let mut decode_state =
            DecodeState::new(vec![7], vec![0.0, 1.0], TransformerDecodeState::default());

        let selected = run_raster(&mut decode_state, 0).expect("stop should not inspect logits");

        assert_eq!(selected, None);
        assert_eq!(decode_state.full_token_ids, vec![7]);
        assert!(decode_state.generated_token_ids.is_empty());
    }

    fn decode_state_with_logits(det_logits: Vec<Act>) -> DecodeState {
        let internal = InternalLogits::from_det_values(det_logits);
        let mut decode_state = DecodeState::new(
            vec![7],
            internal.clone_f32(),
            TransformerDecodeState::default(),
        );
        decode_state.set_internal_logits(internal);
        decode_state
    }
}
