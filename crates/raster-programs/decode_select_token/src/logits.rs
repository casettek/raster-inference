//! Logit selection phase for `decode.select_token`.

use alloc::format;
use alloc::string::ToString;
use raster::prelude::*;

use crate::types::{
    chunk_count, field_selector, read_logit_bits, read_logit_selection, read_token_selection,
    shape_logit_count, DecodeSelectArgmaxState, DecodeSelectConfig, DecodeSelectCounts,
    DecodeSelectLogitSource, DecodeSelectSelectedState, DecodeSelectTokenSource,
};

#[tile]
pub fn init_decode_select_counts(
    logits: DecodeSelectLogitSource,
    full_token_ids: DecodeSelectTokenSource,
    generated_token_ids: DecodeSelectTokenSource,
    config: DecodeSelectConfig,
) -> Result<DecodeSelectCounts> {
    if config.logits_per_tile == 0 {
        return Err("raster decode select logits per tile must be greater than zero".to_string());
    }
    if config.token_ids_per_tile == 0 {
        return Err(
            "raster decode select token ids per tile must be greater than zero".to_string(),
        );
    }

    let row_count = read_logit_selection::<u32>(&logits, field_selector("row_count"), "row count");
    let width = read_logit_selection::<u32>(&logits, field_selector("width"), "width");
    let logit_count = shape_logit_count(row_count, width)?;
    let full_token_count =
        read_token_selection::<u32>(&full_token_ids, field_selector("token_count"), "full count");
    let generated_token_count = read_token_selection::<u32>(
        &generated_token_ids,
        field_selector("token_count"),
        "generated count",
    );

    Ok(DecodeSelectCounts {
        full_token_count,
        generated_token_count,
        logit_count,
        logits_per_tile: config.logits_per_tile,
        token_ids_per_tile: config.token_ids_per_tile,
    })
}

/// One chunk of `logits_per_tile` logits per recur iteration (the sim's
/// `scan_next_token_logit` chunk loop). The driver ordinal names the chunk;
/// the tile reads the chunk's logits from storage inside the body, so the
/// recur loop and its trace scale with `logit_count / logits_per_tile`.
#[tile(kind = recur)]
pub fn scan_one_logit_chunk(
    input: RecurInput<u32>,
    state: RecurState<DecodeSelectArgmaxState>,
    logits: DecodeSelectLogitSource,
    logit_count: u32,
    logits_per_tile: u32,
) -> RecurControl<RecurState<DecodeSelectArgmaxState>> {
    let chunk_idx = *input.value();
    let mut state = state;
    let expected_len = chunk_count(logit_count, logits_per_tile) as u64;
    let expected_idx = input.index() as u32;
    if input.len() != expected_len || chunk_idx != expected_idx {
        state.has_error = true;
        state.error = format!(
            "raster decode select logit loop driver ordinal {} was {chunk_idx} in list len {}, expected {expected_idx} in list len {expected_len}",
            input.index(),
            input.len()
        );
        return RecurControl::Break(state);
    }
    if !state.initialized {
        state.best_logit_bits = read_logit_bits(&logits, 0);
        state.next_token_idx = 1;
        state.initialized = true;
    }

    let end = chunk_idx
        .saturating_add(1)
        .saturating_mul(logits_per_tile)
        .min(logit_count);
    for token_idx in state.next_token_idx..end {
        let candidate_bits = read_logit_bits(&logits, token_idx);
        if candidate_bits > state.best_logit_bits {
            state.best_token_id = token_idx;
            state.best_logit_bits = candidate_bits;
        }
    }
    state.next_token_idx = state.next_token_idx.max(end);

    if state.next_token_idx >= logit_count {
        RecurControl::Break(state)
    } else {
        RecurControl::Continue(state)
    }
}

#[tile]
pub fn finalize_selected_token(
    argmax_state: DecodeSelectArgmaxState,
    logit_count: u32,
) -> Result<DecodeSelectSelectedState> {
    if logit_count == 0 {
        return Err("raster decode select token cannot finalize empty logits".to_string());
    }
    if argmax_state.has_error {
        return Err(argmax_state.error);
    }
    if argmax_state.next_token_idx != logit_count {
        return Err(format!(
            "raster decode select token scanned {} logits, expected {}",
            argmax_state.next_token_idx, logit_count
        ));
    }
    Ok(DecodeSelectSelectedState {
        next_token: argmax_state.best_token_id,
        logit_count,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::vec;

    use super::*;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("decode_select_logits_tests");
        run()
    }

    #[test]
    fn shape_accepts_row_or_column_vectors() {
        assert_eq!(shape_logit_count(3, 1), Ok(3));
        assert_eq!(shape_logit_count(1, 3), Ok(3));
    }

    #[test]
    fn chunk_scan_keeps_equal_logits_at_lowest_index() {
        let state = in_scope(|| {
            let logits_ref = raster::store_internal_value(&crate::types::DecodeSelectLogits {
                row_count: 3,
                width: 1,
                bits: vec![1, 5, 5],
            })
            .expect("store logits");
            scan_one_logit_chunk(
                RecurInput::new(0, 0, 1),
                RecurState::new(DecodeSelectArgmaxState::initial()),
                DecodeSelectLogitSource::internal(logits_ref),
                3,
                3,
            )
        });
        let state = match state {
            RecurControl::Break(state) | RecurControl::Continue(state) => state.into_inner(),
        };
        assert!(state.initialized);
        assert_eq!(state.next_token_idx, 3);
        assert_eq!(state.best_token_id, 1);
        assert_eq!(state.best_logit_bits, 5);
    }

    #[test]
    fn chunk_scan_covers_a_partial_final_chunk() {
        let state = in_scope(|| {
            let logits_ref = raster::store_internal_value(&crate::types::DecodeSelectLogits {
                row_count: 5,
                width: 1,
                bits: vec![0, 1, 2, 3, 9],
            })
            .expect("store logits");
            let source = DecodeSelectLogitSource::internal(logits_ref);
            let mut state = DecodeSelectArgmaxState::initial();
            for chunk_idx in 0..2u32 {
                let control = scan_one_logit_chunk(
                    RecurInput::new(chunk_idx, chunk_idx as u64, 2),
                    RecurState::new(state),
                    source.clone(),
                    5,
                    3,
                );
                state = match control {
                    RecurControl::Break(state) | RecurControl::Continue(state) => {
                        state.into_inner()
                    }
                };
            }
            state
        });
        assert_eq!(state.next_token_idx, 5);
        assert_eq!(state.best_token_id, 4);
        assert_eq!(state.best_logit_bits, 9);
    }

    #[test]
    fn out_of_order_chunk_ordinals_break_with_an_error() {
        let state = in_scope(|| {
            let logits_ref = raster::store_internal_value(&crate::types::DecodeSelectLogits {
                row_count: 3,
                width: 1,
                bits: vec![1, 2, 3],
            })
            .expect("store logits");
            scan_one_logit_chunk(
                RecurInput::new(2, 0, 3),
                RecurState::new(DecodeSelectArgmaxState::initial()),
                DecodeSelectLogitSource::internal(logits_ref),
                3,
                1,
            )
        });
        let state = match state {
            RecurControl::Break(state) | RecurControl::Continue(state) => state.into_inner(),
        };
        assert!(state.has_error);
        assert!(state.error.contains("logit loop driver ordinal 0"));
    }
}
