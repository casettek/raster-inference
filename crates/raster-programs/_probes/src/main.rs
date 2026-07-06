//! WS1 probe program: one `#[sequence] fn main()` composing all probe
//! sequences (P1–P5) in a single `cargo raster run`.
//!
//! Re-run recipe lives in docs/plans/ws1-dsl-translation-catalog.md §7.
//! Observations are recorded there per probe; `[output]` lines below are the
//! run-visible markers those observations cite.

use raster::prelude::*;
use raster::println;

// Glob imports: `call!` resolves hidden per-tile marker types generated next
// to each tile fn, so the whole defining module must be in scope.
use raster_ws1_probes::p1_recur::*;
use raster_ws1_probes::p2_abi::*;
use raster_ws1_probes::p3_external::*;
use raster_ws1_probes::p4_draft::*;
use raster_ws1_probes::p5_sequences::*;
use raster_ws1_probes::types::{Bundle, ConvergeState, MaxState, ProbeConfig, ProbeMode, SumState};

/// P1 recur sequence: per-item tile orchestration into a shared draft output.
/// Real recur sequences have no `RecurControl` — they always visit all items.
#[sequence(kind = recur)]
fn collect_prefixed(
    input: RecurSequenceInput<String>,
    output: RecurSequenceOutput<Bundle>,
    prefix: String,
) -> RecurSequenceOutput<Bundle> {
    let line = call!(prefix_line, input, prefix);
    call!(push_bundle_item, output, line)
}

/// P1 — recur execution (C2/C4/C7–C10; grounds H2/G1).
#[sequence]
fn probe_p1() -> u64 {
    // State-only recur over the full committed list.
    let items = external!(Vec<u64>, "p1_items");
    let max_state = call_recur!(
        tile = scan_max,
        input = items,
        state = MaxState { max: 0 },
        args = ()
    );
    let max = select!(u64, max_state.max);
    println!("p1 scan_max done");

    // Break-early recur: sum of [3,1,4,...] reaches 8 at the third item.
    let items_again = external!(Vec<u64>, "p1_items");
    let sum_state = call_recur!(
        tile = sum_with_break,
        input = items_again,
        state = SumState { sum: 0, seen: 0 },
        args = (8u64,)
    );
    let seen = select!(u64, sum_state.seen);
    println!("p1 sum_with_break done");

    // Gap-G1 shape: until-done loop over a bounded index list (bound = 8;
    // value 40 converges after 5 halvings, well before the bound).
    let bound = external!(Vec<u64>, "p1_bound");
    let converge_state = call_recur!(
        tile = until_done_bounded,
        input = bound,
        state = ConvergeState {
            value: 40,
            steps: 0
        },
        args = ()
    );
    let steps = select!(u64, converge_state.steps);
    println!("p1 until_done_bounded done");

    // Empty-list recur: finalizes with the initial state untouched.
    let empty = raster::store_internal_value(&Vec::<u64>::new()).expect("store empty list");
    let empty_state = call_recur!(
        tile = scan_max,
        input = internal!(Vec<u64>, empty),
        state = MaxState { max: 99 },
        args = ()
    );
    let empty_max = select!(u64, empty_state.max);
    println!("p1 empty-list recur done");

    // Recur sequence: all items visited, tiles orchestrated per item.
    let lines = external!(Vec<String>, "p1_lines");
    let seeded = call!(set_bundle_title, new!(Bundle), "p1".to_string());
    let bundle = call_recur_seq!(
        sequence = collect_prefixed,
        input = lines,
        output = seeded,
        args = ("line: ".to_string(),)
    );
    let first_line = select!(String, bundle.items[0]);
    let first_len = call!(item_len, first_line);
    println!("p1 recur sequence done");

    let a = call!(add_pair, max, seen);
    let b = call!(add_pair, steps, empty_max);
    let c = call!(add_pair, a, b);
    call!(add_pair, c, first_len)
}

/// P2 — multi-arg ABI + error contract (C1/C12/C23/C28).
#[sequence]
fn probe_p2() -> Result<String> {
    let sum = call!(add_pair, 40u64, 2u64);
    println!("p2 add_pair issued (2-arg tuple ABI)");

    let desc = call!(describe_triple, "trip".to_string(), vec![1u32, 2, 3], 2u32)?;
    println!("p2 describe_triple ok (3-arg tuple ABI)");

    let mode = call!(mode_name, ProbeMode::Careful { retries: 3 });
    println!("p2 mode_name issued (enum ABI)");

    let divisor = external!(u64, "p2_divisor");
    let quotient = call!(checked_div, 84u64, divisor)?;
    println!("p2 checked_div ok path done");

    let width = call!(item_len, desc);
    let mode_width = call!(item_len, mode);
    let total = call!(add_pair, sum, quotient);
    let total = call!(add_pair, total, width);
    let total = call!(add_pair, total, mode_width);
    let _ = total;
    Ok("p2-ok".to_string())
}

/// P2 Err path: the tile's user-terminal Err propagates out of the sequence
/// and is observed (without aborting the program) at the `call_seq!` boundary
/// in `main`. The tile execution and its Err outcome still land in the trace.
#[sequence]
fn probe_p2_err() -> Result<u64> {
    let quotient = call!(checked_div, 1u64, 0u64)?;
    Ok(quotient)
}

/// P3 — committed external binding + typed selection (C13/C14/C22).
#[sequence]
fn probe_p3() -> Result<String> {
    let label = select!(String, external!(ProbeConfig, "p3_config").label);
    let threshold = select!(u32, external!(ProbeConfig, "p3_config").thresholds[1]);
    let scale = select!(u32, external!(ProbeConfig, "p3_config").nested.scale);
    println!("p3 selections verified against manifest commitment");
    let combined = call!(combine_config, label, threshold, scale)?;
    Ok(combined)
}

/// P4 — draft + internal storage round-trip (C15–C17; grounds H1).
#[sequence]
fn probe_p4() -> u64 {
    let draft = new!(Bundle);
    let draft = call!(set_bundle_title, draft, "p4".to_string());
    let draft = call!(push_bundle_item, draft, "alpha".to_string());
    let draft = call!(push_bundle_item, draft, "beta".to_string());
    let bundle = finalize(draft);
    let first = select!(String, bundle.items[0]);
    let first_len = call!(item_len, first);
    println!("p4 draft round-trip done");

    let stored = raster::store_internal_value(&vec![7u64, 11, 13]).expect("store internal list");
    let second = select!(u64, internal!(Vec<u64>, stored)[1]);
    println!("p4 internal store/select done");

    call!(add_pair, first_len, second)
}

/// P5 — sequence-in-sequence composition (C3/C5/C6).
#[sequence]
fn inner_transform(x: u64) -> u64 {
    let doubled = call!(double, x);
    call!(inc, doubled)
}

#[sequence]
fn outer_pipeline(x: u64) -> u64 {
    let once = call_seq!(inner_transform, x);
    let twice = call_seq!(inner_transform, once);
    call!(add_pair, twice, 1u64)
}

#[sequence]
fn main() {
    let p1 = call_seq!(probe_p1);
    println!("p1 result ref: {:?}", p1);

    let p2 = call_seq!(probe_p2).expect("p2 should succeed");
    println!("p2 result ref: {:?}", p2);

    match call_seq!(probe_p2_err) {
        Err(message) => println!("p2 expected err surfaced: {message}"),
        Ok(_) => println!("p2 UNEXPECTED ok from probe_p2_err"),
    }

    let p3 = call_seq!(probe_p3).expect("p3 should succeed");
    println!("p3 result ref: {:?}", p3);

    let p4 = call_seq!(probe_p4);
    println!("p4 result ref: {:?}", p4);

    let p5 = call_seq!(outer_pipeline, 5u64);
    println!("p5 result ref: {:?}", p5);

    println!("all probes completed");
}
