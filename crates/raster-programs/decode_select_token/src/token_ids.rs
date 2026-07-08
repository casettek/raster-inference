//! Token-id copy and append phase for `decode.select_token`.

use raster::prelude::*;

use crate::types::{
    chunk_count, read_token_id, DecodeSelectCopyState, DecodeSelectOutput,
    DecodeSelectSelectedState, DecodeSelectTokenDraft, DecodeSelectTokenDraftDraftExt,
    DecodeSelectTokenSource,
};

/// Copies one chunk of `token_ids_per_tile` token ids per recur iteration
/// (the sim's `copy_next_*_token_chunk`). The driver ordinal names the
/// chunk; the tile reads each id from storage inside the body and appends it
/// to the draft output.
#[tile(kind = recur)]
pub fn copy_one_token_chunk(
    input: RecurInput<u32>,
    state: RecurState<DecodeSelectCopyState>,
    output: RecurOutput<DecodeSelectTokenDraft>,
    input_token_ids: DecodeSelectTokenSource,
    token_count: u32,
    token_ids_per_tile: u32,
) -> RecurControl<(
    RecurState<DecodeSelectCopyState>,
    RecurOutput<DecodeSelectTokenDraft>,
)> {
    let chunk_idx = *input.value();
    let mut state = state;
    let mut output = output;
    let expected_len = chunk_count(token_count, token_ids_per_tile) as u64;
    let expected_idx = input.index() as u32;
    if input.len() != expected_len || chunk_idx != expected_idx {
        output.errors().push(alloc::format!(
            "raster decode select token loop driver ordinal {} was {chunk_idx} in list len {}, expected {expected_idx} in list len {expected_len}",
            input.index(),
            input.len()
        ));
        return RecurControl::Break((state, output));
    }
    let end = chunk_idx
        .saturating_add(1)
        .saturating_mul(token_ids_per_tile)
        .min(token_count);
    for token_idx in state.next_token_idx..end {
        output
            .token_ids()
            .push(read_token_id(&input_token_ids, token_idx));
    }
    state.next_token_idx = state.next_token_idx.max(end);
    RecurControl::Continue((state, output))
}

#[tile]
pub fn append_selected_token(
    selected: DecodeSelectSelectedState,
    full_tokens: DecodeSelectTokenDraft,
    generated_tokens: DecodeSelectTokenDraft,
    full_token_count: u32,
    generated_token_count: u32,
) -> Result<DecodeSelectOutput> {
    let DecodeSelectTokenDraft {
        token_ids: mut full_token_ids,
        errors: full_errors,
    } = full_tokens;
    let DecodeSelectTokenDraft {
        token_ids: mut generated_token_ids,
        errors: generated_errors,
    } = generated_tokens;
    if let Some(error) = full_errors.into_iter().next() {
        return Err(error);
    }
    if let Some(error) = generated_errors.into_iter().next() {
        return Err(error);
    }
    if full_token_ids.len() != full_token_count as usize {
        return Err(alloc::format!(
            "raster decode select full-token loop driver copied {} tokens, expected {full_token_count}",
            full_token_ids.len()
        ));
    }
    if generated_token_ids.len() != generated_token_count as usize {
        return Err(alloc::format!(
            "raster decode select generated-token loop driver copied {} tokens, expected {generated_token_count}",
            generated_token_ids.len()
        ));
    }
    full_token_ids.push(selected.next_token);
    generated_token_ids.push(selected.next_token);
    Ok(DecodeSelectOutput {
        next_token: selected.next_token,
        full_token_ids,
        generated_token_ids,
        logit_count: selected.logit_count,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn append_selected_token_appends_to_both_outputs() {
        let _guard = raster::__private::SequenceScopeGuard::enter("decode_select_token_tests");
        let output = append_selected_token(
            DecodeSelectSelectedState {
                next_token: 4,
                logit_count: 1,
            },
            DecodeSelectTokenDraft {
                token_ids: vec![7],
                errors: vec![],
            },
            DecodeSelectTokenDraft {
                token_ids: vec![],
                errors: vec![],
            },
            1,
            0,
        )
        .expect("append");
        assert_eq!(output.full_token_ids, vec![7, 4]);
        assert_eq!(output.generated_token_ids, vec![4]);
    }
}
