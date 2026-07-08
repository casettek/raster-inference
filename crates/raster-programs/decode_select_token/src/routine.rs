//! Routine sequence for `decode.select_token`.

use alloc::vec::Vec;
use raster::prelude::*;

use crate::logits::*;
use crate::token_ids::*;
use crate::types::{
    DecodeSelectArgmaxState, DecodeSelectConfig, DecodeSelectCopyState, DecodeSelectLogitSource,
    DecodeSelectLoopDrivers, DecodeSelectOutput, DecodeSelectTokenDraft, DecodeSelectTokenSource,
};

/// Canonical logits + token histories -> selected token and updated histories.
#[sequence]
pub fn select_decode_token(
    logits: DecodeSelectLogitSource,
    full_token_ids: DecodeSelectTokenSource,
    generated_token_ids: DecodeSelectTokenSource,
    loop_drivers: DecodeSelectLoopDrivers,
    config: DecodeSelectConfig,
) -> Result<DecodeSelectOutput> {
    let counts = call!(
        init_decode_select_counts,
        logits.clone(),
        full_token_ids.clone(),
        generated_token_ids.clone(),
        config
    )?;
    let logit_chunks = select!(Vec<u32>, loop_drivers.clone().logit_ordinals);
    let full_token_chunks = select!(Vec<u32>, loop_drivers.clone().full_token_ordinals);
    let generated_token_chunks = select!(Vec<u32>, loop_drivers.generated_token_ordinals);
    let logit_count = select!(u32, counts.clone().logit_count);
    let full_token_count = select!(u32, counts.clone().full_token_count);
    let generated_token_count = select!(u32, counts.clone().generated_token_count);
    let logits_per_tile = select!(u32, counts.clone().logits_per_tile);
    let token_ids_per_tile = select!(u32, counts.token_ids_per_tile);

    let argmax_state = call_recur!(
        tile = scan_one_logit_chunk,
        input = logit_chunks,
        state = DecodeSelectArgmaxState::initial(),
        args = (logits, logit_count.clone(), logits_per_tile)
    );
    let selected = call!(finalize_selected_token, argmax_state, logit_count)?;

    let full_copied = call_recur!(
        tile = copy_one_token_chunk,
        input = full_token_chunks,
        state = DecodeSelectCopyState::initial(),
        output = new!(DecodeSelectTokenDraft),
        args = (
            full_token_ids,
            full_token_count.clone(),
            token_ids_per_tile.clone()
        )
    );
    let generated_copied = call_recur!(
        tile = copy_one_token_chunk,
        input = generated_token_chunks,
        state = DecodeSelectCopyState::initial(),
        output = new!(DecodeSelectTokenDraft),
        args = (
            generated_token_ids,
            generated_token_count.clone(),
            token_ids_per_tile
        )
    );

    Ok(call!(
        append_selected_token,
        selected,
        full_copied,
        generated_copied,
        full_token_count,
        generated_token_count
    )?)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::vec;

    use raster::materialize_auth_result;

    use super::*;

    fn chunk_ordinals(item_count: usize, per_tile: u32) -> Vec<u32> {
        (0..crate::types::chunk_count(item_count as u32, per_tile)).collect()
    }

    fn expected_loop_drivers(
        logit_count: usize,
        full_token_count: usize,
        generated_token_count: usize,
        logits_per_tile: u32,
        token_ids_per_tile: u32,
    ) -> DecodeSelectLoopDrivers {
        DecodeSelectLoopDrivers {
            logit_ordinals: chunk_ordinals(logit_count, logits_per_tile),
            full_token_ordinals: chunk_ordinals(full_token_count, token_ids_per_tile),
            generated_token_ordinals: chunk_ordinals(generated_token_count, token_ids_per_tile),
        }
    }

    fn select_with_loop_drivers(
        logit_bits: Vec<i32>,
        full_token_ids: Vec<u32>,
        generated_token_ids: Vec<u32>,
        loop_drivers: DecodeSelectLoopDrivers,
        logits_per_tile: u32,
        token_ids_per_tile: u32,
    ) -> core::result::Result<DecodeSelectOutput, String> {
        let _guard = raster::__private::SequenceScopeGuard::enter("decode_select_routine_tests");
        materialize_auth_result::<DecodeSelectOutput, _>(
            __raster_sequence_auth_select_decode_token(
                internal!(
                    crate::types::DecodeSelectLogitSource,
                    raster::store_internal_value(&DecodeSelectLogitSource::internal(
                        raster::store_internal_value(&crate::types::DecodeSelectLogits {
                            row_count: logit_bits.len() as u32,
                            width: 1,
                            bits: logit_bits,
                        })
                        .expect("store logits")
                    ))
                    .expect("store logit source")
                ),
                internal!(
                    crate::types::DecodeSelectTokenSource,
                    raster::store_internal_value(&DecodeSelectTokenSource::internal(
                        raster::store_internal_value(&crate::types::DecodeSelectTokenIds {
                            token_count: full_token_ids.len() as u32,
                            token_ids: full_token_ids,
                        })
                        .expect("store full token ids")
                    ))
                    .expect("store full token source")
                ),
                internal!(
                    crate::types::DecodeSelectTokenSource,
                    raster::store_internal_value(&DecodeSelectTokenSource::internal(
                        raster::store_internal_value(&crate::types::DecodeSelectTokenIds {
                            token_count: generated_token_ids.len() as u32,
                            token_ids: generated_token_ids,
                        })
                        .expect("store generated token ids")
                    ))
                    .expect("store generated token source")
                ),
                internal!(
                    DecodeSelectLoopDrivers,
                    raster::store_internal_value(&loop_drivers).expect("store loop drivers")
                ),
                internal!(
                    DecodeSelectConfig,
                    raster::store_internal_value(&DecodeSelectConfig {
                        logits_per_tile,
                        token_ids_per_tile,
                    })
                    .expect("store config")
                ),
            ),
        )
    }

    fn select(
        logit_bits: Vec<i32>,
        full_token_ids: Vec<u32>,
        generated_token_ids: Vec<u32>,
        logits_per_tile: u32,
        token_ids_per_tile: u32,
    ) -> core::result::Result<DecodeSelectOutput, String> {
        let loop_drivers = expected_loop_drivers(
            logit_bits.len(),
            full_token_ids.len(),
            generated_token_ids.len(),
            logits_per_tile,
            token_ids_per_tile,
        );
        select_with_loop_drivers(
            logit_bits,
            full_token_ids,
            generated_token_ids,
            loop_drivers,
            logits_per_tile,
            token_ids_per_tile,
        )
    }

    #[test]
    fn selects_highest_logit_and_appends_token_ids() {
        let output = select(vec![-2, 0, 7, 3], vec![10], vec![], 2, 1).expect("select");
        assert_eq!(output.next_token, 2);
        assert_eq!(output.full_token_ids, vec![10, 2]);
        assert_eq!(output.generated_token_ids, vec![2]);
        assert_eq!(output.logit_count, 4);
    }

    #[test]
    fn equal_logits_choose_lowest_index() {
        let output = select(vec![1, 5, 5], vec![10], vec![], 1, 1).expect("select");
        assert_eq!(output.next_token, 1);
    }

    #[test]
    fn chunk_sizes_do_not_change_output() {
        let logits = (0..40).collect::<Vec<_>>();
        let single = select(logits.clone(), vec![1, 2], vec![2], 1, 1).expect("single");
        let multi = select(logits, vec![1, 2], vec![2], 11, 8).expect("multi");
        assert_eq!(single, multi);
    }

    #[test]
    fn zero_chunk_sizes_fail() {
        let error = select(vec![1, 2], vec![1], vec![], 0, 1).expect_err("zero logits");
        assert!(error.contains("greater than zero"));
        let error = select(vec![1, 2], vec![1], vec![], 1, 0).expect_err("zero tokens");
        assert!(error.contains("greater than zero"));
    }

    #[test]
    fn empty_logits_fail() {
        let error = select(vec![], vec![1], vec![], 1, 1).expect_err("empty logits");
        assert!(error.contains("at least one canonical logit"));
    }

    #[test]
    fn mismatched_loop_drivers_fail_before_selection() {
        let error = select_with_loop_drivers(
            vec![1, 2, 3],
            vec![7],
            vec![],
            DecodeSelectLoopDrivers {
                logit_ordinals: vec![0, 2],
                full_token_ordinals: vec![0],
                generated_token_ordinals: vec![],
            },
            1,
            1,
        )
        .expect_err("bad loop driver should fail");
        assert!(error.contains("logit loop driver ordinal 0"));
    }

    #[test]
    fn mismatched_token_loop_drivers_fail_at_output_boundary() {
        let error = select_with_loop_drivers(
            vec![1, 2, 3],
            vec![7],
            vec![],
            DecodeSelectLoopDrivers {
                logit_ordinals: vec![0, 1, 2],
                full_token_ordinals: vec![1],
                generated_token_ordinals: vec![],
            },
            1,
            1,
        )
        .expect_err("bad token driver should fail");
        assert!(error.contains("token loop driver ordinal 0"));
    }
}
