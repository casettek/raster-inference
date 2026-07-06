//! P1 — recur execution tiles (catalog C2/C7; grounds H2/G1).

use alloc::format;
use alloc::string::String;
use raster::prelude::*;

use crate::types::{ConvergeState, MaxState, SumState};

/// State-only recur over the whole list (no early exit).
#[tile(kind = recur)]
pub fn scan_max(input: RecurInput<u64>, state: RecurState<MaxState>) -> RecurState<MaxState> {
    let mut state = state;
    let value = *input.value();
    if value > state.max {
        state.max = value;
    }
    state
}

/// Break-early recur: stops as soon as the running sum reaches `limit`.
#[tile(kind = recur)]
pub fn sum_with_break(
    input: RecurInput<u64>,
    state: RecurState<SumState>,
    limit: u64,
) -> RecurControl<RecurState<SumState>> {
    let mut state = state;
    state.sum += *input.value();
    state.seen += 1;
    if state.sum >= limit {
        RecurControl::Break(state)
    } else {
        RecurControl::Continue(state)
    }
}

/// Gap-G1 shape: a condition-driven ("until done") loop expressed over a
/// bounded dummy-index list. The input item is only the iteration budget;
/// convergence breaks before the bound is exhausted.
#[tile(kind = recur)]
pub fn until_done_bounded(
    input: RecurInput<u64>,
    state: RecurState<ConvergeState>,
) -> RecurControl<RecurState<ConvergeState>> {
    let _bound_index = input.value();
    let mut state = state;
    if state.value <= 1 {
        return RecurControl::Break(state);
    }
    state.value /= 2;
    state.steps += 1;
    if state.value <= 1 {
        RecurControl::Break(state)
    } else {
        RecurControl::Continue(state)
    }
}

/// P1 recur-sequence support tile (catalog C4/C9/C10).
#[tile]
pub fn prefix_line(line: String, prefix: String) -> String {
    format!("{prefix}{line}")
}
