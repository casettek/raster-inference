use anyhow::{anyhow, bail, Result};

use crate::raster_authoring::prelude::{auth_read, call_recur_tile, call_tile, sequence, tile};
use crate::shared::{
    det_num::{argmax_first, Act},
    output::OutputDecodeStopReason,
    raster_decode_select_token::{
        AuthenticatedDecodeSelectLogitsSource, DecodeSelectLogitRequest,
        DecodeSelectLogitsMetadataRequest,
    },
};

const DEFAULT_DECODE_SELECT_LOGITS_PER_TILE: usize = 32;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectArgmaxState {
    next_token_idx: usize,
    logit_count: usize,
    best_token_id: u32,
    best_logit_bits: i32,
    logits_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectRasterOutput {
    pub next_token: u32,
    pub full_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub det_current_logits_sha256: String,
}

#[tile]
pub fn check_stop_condition(
    generated_token_count: usize,
    max_new_tokens: usize,
) -> Option<OutputDecodeStopReason> {
    (generated_token_count >= max_new_tokens).then_some(OutputDecodeStopReason::MaxNewTokens)
}

#[tile]
pub fn init_select_next_token(
    logits_source: &AuthenticatedDecodeSelectLogitsSource,
) -> Result<DecodeSelectArgmaxState> {
    let metadata = auth_read!(logits_source, DecodeSelectLogitsMetadataRequest)?;
    if metadata.logit_count == 0 {
        bail!("raster decode select token requires at least one canonical logit");
    }

    let best_logit_bits = auth_read!(logits_source, DecodeSelectLogitRequest { token_idx: 0 })?;
    Ok(DecodeSelectArgmaxState {
        next_token_idx: 1,
        logit_count: metadata.logit_count,
        best_token_id: 0,
        best_logit_bits,
        logits_per_tile: DEFAULT_DECODE_SELECT_LOGITS_PER_TILE,
    })
}

#[tile(kind = recursive)]
pub fn scan_next_token_logit(
    mut state: DecodeSelectArgmaxState,
    logits_source: &AuthenticatedDecodeSelectLogitsSource,
) -> Result<(bool, DecodeSelectArgmaxState)> {
    if state.next_token_idx >= state.logit_count {
        return Ok((true, state));
    }

    let end = state
        .next_token_idx
        .saturating_add(state.logits_per_tile)
        .min(state.logit_count);
    while state.next_token_idx < end {
        let candidate_bits = auth_read!(
            logits_source,
            DecodeSelectLogitRequest {
                token_idx: state.next_token_idx,
            },
        )?;
        if candidate_wins(state.best_logit_bits, candidate_bits) {
            state.best_token_id = u32::try_from(state.next_token_idx)
                .map_err(|_| anyhow!("raster decode selected token index exceeds u32"))?;
            state.best_logit_bits = candidate_bits;
        }
        state.next_token_idx += 1;
    }

    Ok((false, state))
}

fn candidate_wins(best_logit_bits: i32, candidate_bits: i32) -> bool {
    let candidates = [
        Act::from_bits(best_logit_bits),
        Act::from_bits(candidate_bits),
    ];
    argmax_first(&candidates) == 1
}

#[tile]
pub fn finalize_selected_token(state: DecodeSelectArgmaxState) -> Result<u32> {
    if state.logit_count == 0 {
        bail!("raster decode select token cannot finalize empty logits");
    }
    if state.next_token_idx != state.logit_count {
        bail!(
            "raster decode select token scanned {} logits, expected {}",
            state.next_token_idx,
            state.logit_count
        );
    }
    Ok(state.best_token_id)
}

#[tile]
pub fn append_token(token_ids: &[u32], next_token: u32) -> Vec<u32> {
    let mut appended = token_ids.to_vec();
    appended.push(next_token);
    appended
}

#[tile]
pub fn finalize_decode_select(
    next_token: u32,
    full_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    logits_source: &AuthenticatedDecodeSelectLogitsSource,
) -> Result<DecodeSelectRasterOutput> {
    let metadata = auth_read!(logits_source, DecodeSelectLogitsMetadataRequest)?;
    Ok(DecodeSelectRasterOutput {
        next_token,
        full_token_ids,
        generated_token_ids,
        det_current_logits_sha256: metadata.det_logits_sha256,
    })
}

#[sequence]
pub fn run(
    full_token_ids: &[u32],
    generated_token_ids: &[u32],
    max_new_tokens: usize,
    logits_source: &AuthenticatedDecodeSelectLogitsSource,
) -> Result<Option<DecodeSelectRasterOutput>> {
    if call_tile!(
        check_stop_condition,
        generated_token_ids.len(),
        max_new_tokens
    )
    .is_some()
    {
        return Ok(None);
    }

    let state = call_tile!(init_select_next_token, logits_source)?;
    let state = call_recur_tile!(scan_next_token_logit, state, logits_source)?;
    let next_token = call_tile!(finalize_selected_token, state)?;
    let full_token_ids = call_tile!(append_token, full_token_ids, next_token);
    let generated_token_ids = call_tile!(append_token, generated_token_ids, next_token);
    call_tile!(
        finalize_decode_select,
        next_token,
        full_token_ids,
        generated_token_ids,
        logits_source
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::{
        append_token, check_stop_condition, finalize_selected_token, run, scan_next_token_logit,
        DecodeSelectArgmaxState,
    };
    use crate::shared::{
        det_num::Act, input::InferenceExecutionMode, output::OutputDecodeStopReason,
        raster_decode_select_token::AuthenticatedDecodeSelectLogitsSource,
        transformer::InternalLogits,
    };

    #[test]
    fn run_selects_highest_canonical_logit() {
        let source = source(vec![
            Act::from_bits(-2),
            Act::from_bits(0),
            Act::from_bits(7),
            Act::from_bits(3),
        ]);

        let output = run(&[10], &[], 1, &source)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(output.next_token, 2);
        assert_eq!(output.full_token_ids, vec![10, 2]);
        assert_eq!(output.generated_token_ids, vec![2]);
    }

    #[test]
    fn run_breaks_equal_logits_by_lowest_token_id() {
        let source = source(vec![
            Act::from_bits(1),
            Act::from_bits(5),
            Act::from_bits(5),
        ]);

        let output = run(&[], &[], 1, &source)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(output.next_token, 1);
    }

    #[test]
    fn run_matches_native_deterministic_selection() {
        let det_logits = vec![
            Act::from_bits(-10),
            Act::from_bits(4),
            Act::from_bits(7),
            Act::from_bits(7),
        ];
        let internal = InternalLogits::from_det_values(det_logits.clone());
        let source = source(det_logits);

        let native = crate::decode_select_token::tiles::select_next_token_internal(
            &internal,
            InferenceExecutionMode::Deterministic,
        )
        .expect("native deterministic selection should run");
        let raster = run(&[10], &[], 1, &source)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(raster.next_token, native);
        assert_eq!(raster.full_token_ids, vec![10, native]);
        assert_eq!(raster.generated_token_ids, vec![native]);
    }

    #[test]
    fn run_selects_single_logit_token_zero() {
        let source = source(vec![Act::from_bits(11)]);

        let output = run(&[], &[], 1, &source)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(output.next_token, 0);
    }

    #[test]
    fn run_stops_without_selecting_at_max_new_tokens() {
        let source = source(vec![Act::from_bits(11)]);

        let output = run(&[1], &[2], 1, &source).expect("stop check should run");

        assert_eq!(output, None);
        assert_eq!(
            check_stop_condition(1, 1),
            Some(OutputDecodeStopReason::MaxNewTokens)
        );
    }

    #[test]
    fn finalize_selected_token_rejects_incomplete_scan() {
        let error = finalize_selected_token(DecodeSelectArgmaxState {
            next_token_idx: 1,
            logit_count: 2,
            best_token_id: 0,
            best_logit_bits: 0,
            logits_per_tile: 1,
        })
        .expect_err("incomplete scan should fail");

        assert!(error.to_string().contains("scanned 1 logits"));
    }

    #[test]
    fn scan_next_token_logit_bounds_work_per_recursive_step() {
        let source = source((0..40).map(Act::from_bits).collect());
        let state = DecodeSelectArgmaxState {
            next_token_idx: 1,
            logit_count: 40,
            best_token_id: 0,
            best_logit_bits: 0,
            logits_per_tile: 7,
        };

        let (done, state) =
            scan_next_token_logit(state, &source).expect("scan chunk should succeed");

        assert!(!done);
        assert_eq!(state.next_token_idx, 8);
        assert_eq!(state.best_token_id, 7);
        assert_eq!(state.best_logit_bits, 7);
    }

    #[test]
    fn append_token_returns_new_sequence() {
        let original = vec![1, 2];
        let appended = append_token(&original, 3);

        assert_eq!(original, vec![1, 2]);
        assert_eq!(appended, vec![1, 2, 3]);
    }

    fn source(det_logits: Vec<Act>) -> AuthenticatedDecodeSelectLogitsSource {
        let logits = InternalLogits::from_det_values(det_logits);
        AuthenticatedDecodeSelectLogitsSource::from_internal_logits("decode-test", &logits)
            .expect("source should build")
    }
}
