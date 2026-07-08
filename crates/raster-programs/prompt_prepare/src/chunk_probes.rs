//! Step 0 probes for the storage refactor: verify the two load-bearing
//! mechanics before rewriting the phases (plan
//! `prompt.prepare storage refactor`).
//!
//! 1. `Vec<Vec<GemmaBpeMerge>>` works as a `Selectable` struct field
//!    (`select!` of the whole chunk list, one chunk, and one nested entry).
//! 2. A `#[tile(kind = recur)]` over a chunked list (`RecurInput<Vec<T>>`)
//!    with `RecurControl::Break` mid-list behaves as expected (the flat
//!    case is WS1 probe P1).
//!
//! Extended for the trace-slimming ref-based refactor (plan
//! `trace-slimming ref-based refactor`, Step 0):
//!
//! 3. A `#[sequence(kind = recur)]` over a chunked list
//!    (`RecurSequenceInput<Vec<T>>`) whose body passes **both** the opaque
//!    item handle and the threaded state into one plain tile, the tile's
//!    `AuthRef` return re-entering the state.
//! 4. That recur sequence nested **inside** another recur-sequence
//!    iteration (the shape of the BPE round loop around the scan).
//!
//! Extended for the storage-resident refactor (plan
//! `prompt.prepare storage-resident refactor`), probes P1–P4:
//!
//! P1. Round-boundary characterization: a recur sequence threading a
//!     `Vec<String>`-carrying state. Establishes (a) the body tile's
//!     returned state persists in internal storage via the tile-output
//!     store (`bind_infallible_call` → `store_execution_output_value`);
//!     (b) the driver's re-entry resolve (`From<AuthRef<T>> for
//!     `RecurSequenceState<T>` → `resolve_internal_value`) validates a
//!     coordinates lookup, commitment equality, and a recomputed integrity
//!     commitment; (c) the trace records the state **inline** (full
//!     postcard bytes) in every iteration's `RecurSequenceStart` record —
//!     O(iterations × state bytes) scaling. These findings are the
//!     pinned-rev justification for keeping the loop-carried pieces in
//!     `GemmaBpeLoopState` (invariant rule 4, deviation D6 re-founding).
//! P2. A fresh `RecurOutput` draft created (`output = new!(…)`) and
//!     finalized inside a recur-sequence body iteration; input list from a
//!     `select!` projection of a previous tile's internal output; the
//!     per-item plain tile returns `(RecurState<S>, RecurOutput<O>)`
//!     (cursor + draft through one tile); the finalized `AuthRef` consumed
//!     in the same body via `select!` and as a follow-up plain tile's
//!     selection-bound arg; the follow-up tile's return re-entering the
//!     outer threaded state. The exact shape of the rewritten apply loop.
//! P3. A `Vec` selection-bound arg on a plain tile called from a
//!     recur-sequence body traces as a binding, never inline; the tile
//!     materializes the full value at execution.
//! P4. Recur-*tile* drivers trace the input item, the state, and the args
//!     inline per iteration (`RecurTileIterationExec`) — gap G7's exact
//!     boundary; recur-tile loops carry scalars only.
//!
//! Test-only module: these tiles are evidence, not program surface.

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster::Selectable;
use raster_program_gemma_externals::types::GemmaBpeMerge;
use serde::{Deserialize, Serialize};

/// Chunked-table shape mirroring the revised `GemmaTokenizer` fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbeTable {
    pub chunks: Vec<Vec<GemmaBpeMerge>>,
}

/// Materialized outcome of the nested selections.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbeSelection {
    pub chunk_len: u32,
    pub entry_merge_index: u32,
    pub entry_merged: String,
}

/// Loop state of the chunked scan probe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeScanState {
    pub visited_chunks: u32,
    pub found: bool,
    pub found_merge_index: u32,
}

#[tile]
pub fn summarize_selection(chunk: Vec<GemmaBpeMerge>, entry: GemmaBpeMerge) -> ProbeSelection {
    ProbeSelection {
        chunk_len: chunk.len() as u32,
        entry_merge_index: entry.merge_index,
        entry_merged: entry.merged_token,
    }
}

/// Mechanic 1: nested-list selection through a `Selectable` field — the
/// whole chunk list, a single chunk (`[1]`), and a nested entry (`[1][0]`).
#[sequence]
pub fn probe_select_chunks(table: ProbeTable) -> ProbeSelection {
    let second_chunk = select!(Vec<GemmaBpeMerge>, table.clone().chunks[1]);
    let first_of_second = select!(GemmaBpeMerge, table.chunks[1][0]);
    call!(summarize_selection, second_chunk, first_of_second)
}

/// Mechanic 2: one chunk per recur iteration; `Break` as soon as a rule
/// matching `needle_left` is found (mid-list for the fixture).
#[tile(kind = recur)]
pub fn probe_scan_chunk(
    input: RecurInput<Vec<GemmaBpeMerge>>,
    state: RecurState<ProbeScanState>,
    needle_left: String,
) -> RecurControl<RecurState<ProbeScanState>> {
    let mut state = state;
    state.visited_chunks += 1;
    for merge in input.value() {
        if merge.left == needle_left {
            state.found = true;
            state.found_merge_index = merge.merge_index;
            return RecurControl::Break(state);
        }
    }
    RecurControl::Continue(state)
}

/// Drives the chunked recur over a `select!`-ed `Vec<Vec<GemmaBpeMerge>>`
/// field — the exact shape the refactored phases use.
#[sequence]
pub fn probe_chunked_recur(table: ProbeTable, needle_left: String) -> ProbeScanState {
    let chunks = select!(Vec<Vec<GemmaBpeMerge>>, table.chunks);
    call_recur!(
        tile = probe_scan_chunk,
        input = chunks,
        state = ProbeScanState {
            visited_chunks: 0,
            found: false,
            found_merge_index: 0,
        },
        args = (needle_left,)
    )
}

/// Nested-fixture shape for mechanic 4: the chunked table plus a bounded
/// round list, both selected out of one committed source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbeNestedFixture {
    pub chunks: Vec<Vec<GemmaBpeMerge>>,
    pub rounds: Vec<u32>,
}

/// Loop state threaded through the mechanic-4 outer round sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeRoundState {
    pub rounds_run: u32,
    pub total_visited_chunks: u32,
    pub found_rounds: u32,
    pub found_merge_index: u32,
}

/// Mechanic 3's per-chunk tile: the chunk arrives as a selection binding,
/// the tiny scan state inline; already-found scans no-op (recur sequences
/// cannot break early — gap G1).
#[tile]
pub fn probe_scan_one_chunk(
    state: ProbeScanState,
    chunk: Vec<GemmaBpeMerge>,
    needle_left: String,
) -> ProbeScanState {
    let mut state = state;
    if state.found {
        return state;
    }
    state.visited_chunks += 1;
    for merge in chunk {
        if merge.left == needle_left {
            state.found = true;
            state.found_merge_index = merge.merge_index;
            break;
        }
    }
    state
}

/// Mechanic 3: a recur *sequence* over the chunked list whose body passes
/// both the opaque item handle and the threaded state into one plain tile;
/// the tile's `AuthRef` return re-enters the state.
#[sequence(kind = recur)]
pub fn probe_scan_chunks_seq(
    input: RecurSequenceInput<Vec<GemmaBpeMerge>>,
    state: RecurSequenceState<ProbeScanState>,
    needle_left: String,
) -> RecurSequenceState<ProbeScanState> {
    call!(probe_scan_one_chunk, state, input, needle_left)
}

/// Drives mechanic 3 directly over a `select!`-ed chunk list.
#[sequence]
pub fn probe_chunked_recur_seq(table: ProbeTable, needle_left: String) -> ProbeScanState {
    let chunks = select!(Vec<Vec<GemmaBpeMerge>>, table.chunks);
    call_recur_seq!(
        sequence = probe_scan_chunks_seq,
        input = chunks,
        state = ProbeScanState {
            visited_chunks: 0,
            found: false,
            found_merge_index: 0,
        },
        args = (needle_left,)
    )
}

/// Folds one round's scan outcome into the outer round state.
#[tile]
pub fn probe_fold_round(state: ProbeRoundState, scan: ProbeScanState) -> ProbeRoundState {
    ProbeRoundState {
        rounds_run: state.rounds_run + 1,
        total_visited_chunks: state.total_visited_chunks + scan.visited_chunks,
        found_rounds: state.found_rounds + u32::from(scan.found),
        found_merge_index: if scan.found {
            scan.found_merge_index
        } else {
            state.found_merge_index
        },
    }
}

/// Mechanic 4: the mechanic-3 recur sequence nested inside another
/// recur-sequence iteration (the round-loop-around-scan shape).
#[sequence(kind = recur)]
pub fn probe_round_seq(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<ProbeRoundState>,
    chunks: Vec<Vec<GemmaBpeMerge>>,
    needle_left: String,
) -> RecurSequenceState<ProbeRoundState> {
    let _round_ordinal = &input;
    let scan = call_recur_seq!(
        sequence = probe_scan_chunks_seq,
        input = chunks,
        state = ProbeScanState {
            visited_chunks: 0,
            found: false,
            found_merge_index: 0,
        },
        args = (needle_left,)
    );
    call!(probe_fold_round, state, scan)
}

/// Drives mechanic 4: outer recur sequence over the round list, inner
/// recur sequence over the chunked table.
#[sequence]
pub fn probe_nested_recur_seq(fixture: ProbeNestedFixture, needle_left: String) -> ProbeRoundState {
    let chunks = select!(Vec<Vec<GemmaBpeMerge>>, fixture.clone().chunks);
    let rounds = select!(Vec<u32>, fixture.rounds);
    call_recur_seq!(
        sequence = probe_round_seq,
        input = rounds,
        state = ProbeRoundState {
            rounds_run: 0,
            total_visited_chunks: 0,
            found_rounds: 0,
            found_merge_index: 0,
        },
        args = (chunks, needle_left)
    )
}

// --- Storage-resident refactor probes (P1–P4) ---

/// P1 state: the minimal loop-carried collection shape (`GemmaBpeLoopState`
/// analog): a `Vec<String>` threaded through recur-sequence state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeRoundPieces {
    pub round: u32,
    pub pieces: Vec<String>,
}

/// P1 body tile: returns the next loop state. Its `call!` return is an
/// `AuthRef` backed by the tile-output internal store
/// (`bind_infallible_call` → `store_execution_output_value`); the sequence
/// re-enters threaded state through `From<AuthRef<T>> for
/// RecurSequenceState<T>` → `resolve_internal_value`.
#[tile]
pub fn probe_advance_round_pieces(state: ProbeRoundPieces, appended: String) -> ProbeRoundPieces {
    let mut state = state;
    let round = state.round;
    state.pieces.push(alloc::format!("{appended}{round}"));
    state.round += 1;
    state
}

/// P1 loop: one plain tile per iteration, the tile's `AuthRef` return
/// re-entering the threaded state (the exact `merge_bpe_round` boundary).
#[sequence(kind = recur)]
pub fn probe_round_boundary_seq(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<ProbeRoundPieces>,
    appended: String,
) -> RecurSequenceState<ProbeRoundPieces> {
    let _round_ordinal = &input;
    call!(probe_advance_round_pieces, state, appended)
}

/// P1 seed builder — recur loop seeds must be plain literals (A2), so the
/// drivers construct them in place; `width` scales the carried collection
/// for the size-characterization leg.
pub fn probe_p1_seed(piece_count: usize, piece_width: usize) -> ProbeRoundPieces {
    ProbeRoundPieces {
        round: 0,
        pieces: (0..piece_count)
            .map(|idx| alloc::format!("{idx:0>piece_width$}"))
            .collect(),
    }
}

/// P1 driver: a tiny seed, growing by one piece per round.
#[sequence]
pub fn probe_round_boundary(rounds: Vec<u32>, appended: String) -> ProbeRoundPieces {
    call_recur_seq!(
        sequence = probe_round_boundary_seq,
        input = rounds,
        state = probe_p1_seed(1, 1),
        args = (appended,)
    )
}

/// P1 driver, large-collection leg: 64 pieces of 32 bytes in the seed.
#[sequence]
pub fn probe_round_boundary_big(rounds: Vec<u32>, appended: String) -> ProbeRoundPieces {
    call_recur_seq!(
        sequence = probe_round_boundary_seq,
        input = rounds,
        state = probe_p1_seed(64, 32),
        args = (appended,)
    )
}

/// P1 storage leg: expose the body tile's output `InternalRef` so the test
/// can resolve it against internal storage and characterize what the
/// resolve validates.
#[sequence]
pub fn probe_tile_output_reference(seed: ProbeRoundPieces, appended: String) -> InternalRef {
    call!(probe_advance_round_pieces, seed, appended)
        .reference()
        .clone()
}

/// P2 select-root: the `open_round` analog — one Selectable output carrying
/// the round scalars and the round's pieces behind an internal ref, so the
/// pieces are consumed only through `select!` projections.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbeOpenedRound {
    pub round: u32,
    pub pieces: ProbePieces,
}

/// P2 pieces wrapper: `Selectable` for `select!` roots and the draft schema
/// for `RecurOutput<ProbePieces>` (the `BpePieces` analog).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Selectable)]
pub struct ProbePieces {
    pub pieces: Vec<String>,
}

/// Handle-only round state: the refactored `GemmaBpeLoopState` analog.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeRoundPieceHandle {
    pub round: u32,
    pub pieces_ref: InternalRef,
}

/// Publishes pieces as a tile output so the driver can thread only its
/// internal ref through recur state.
#[tile]
pub fn probe_publish_pieces(pieces: ProbePieces) -> ProbePieces {
    pieces
}

/// Rehydrates handle-carried pieces inside a plain tile and republishes the
/// value behind a selectable round output.
#[tile]
pub fn probe_open_handle_round(state: ProbeRoundPieceHandle) -> ProbeOpenedRound {
    let pieces = raster::resolve_internal_value::<ProbePieces>(state.pieces_ref)
        .unwrap_or_else(|error| panic!("failed to resolve probe pieces: {error}"))
        .value;
    ProbeOpenedRound {
        round: state.round,
        pieces,
    }
}

/// P2 cursor-only inner-loop state (the `GemmaBpeApplyCursor` analog).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeApplyCursor {
    pub skip_next: bool,
    pub emitted: u32,
}

/// P2 open tile: materializes the loop state once per round and republishes
/// the pieces behind a selectable internal ref.
#[tile]
pub fn probe_open_apply_round(state: ProbeRoundPieces) -> ProbeOpenedRound {
    ProbeOpenedRound {
        round: state.round,
        pieces: ProbePieces {
            pieces: state.pieces,
        },
    }
}

/// P2 per-piece tile: cursor state + the piece via the input handle + the
/// `RecurOutput` draft + decision scalars, returning both threads through
/// one tile (the `apply_one_piece` shape). Merges at index 0 for the
/// fixture: push `merged`, swallow the right-hand piece via `skip_next`,
/// copy everything else.
#[tile]
pub fn probe_apply_one_piece(
    state: ProbeApplyCursor,
    piece: String,
    output: Draft<ProbePieces>,
    merged: String,
) -> (RecurState<ProbeApplyCursor>, RecurOutput<ProbePieces>) {
    let mut state = state;
    let mut output = output;
    if state.skip_next {
        state.skip_next = false;
        return (RecurState::new(state), output);
    }
    if state.emitted == 0 {
        output.pieces().push(merged);
        state.skip_next = true;
    } else {
        output.pieces().push(piece);
    }
    state.emitted += 1;
    (RecurState::new(state), output)
}

/// P2 inner loop: a state+output recur sequence over a `select!` projection
/// of a previous tile's internal output; the draft is created fresh
/// (`output = new!(…)`) per outer iteration and finalized at inner loop end.
#[sequence(kind = recur)]
pub fn probe_apply_pieces_seq(
    input: RecurSequenceInput<String>,
    state: RecurSequenceState<ProbeApplyCursor>,
    output: RecurSequenceOutput<ProbePieces>,
    merged: String,
) -> (
    RecurSequenceState<ProbeApplyCursor>,
    RecurSequenceOutput<ProbePieces>,
) {
    let (cursor, output) = call!(probe_apply_one_piece, state, input, output, merged);
    let cursor: RecurSequenceState<ProbeApplyCursor> = cursor.into_inner().into();
    let output: RecurSequenceOutput<ProbePieces> = output.into();
    (cursor, output)
}

/// Handle-state finalizer: the next pieces are already finalized into
/// internal storage, so the outer state only keeps the new ref.
#[tile]
pub fn probe_finalize_handle_round(
    state: ProbeRoundPieceHandle,
    next_pieces_ref: InternalRef,
) -> ProbeRoundPieceHandle {
    ProbeRoundPieceHandle {
        round: state.round + 1,
        pieces_ref: next_pieces_ref,
    }
}

/// Handle-only outer round body: open ref → select pieces → draft apply →
/// carry the finalized draft ref into the next round.
#[sequence(kind = recur)]
pub fn probe_handle_round_seq(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<ProbeRoundPieceHandle>,
    merged: String,
) -> RecurSequenceState<ProbeRoundPieceHandle> {
    let _round_ordinal = &input;
    let opened = call!(probe_open_handle_round, state.clone());
    let items = select!(Vec<String>, opened.pieces.pieces);
    let applied = call_recur_seq!(
        sequence = probe_apply_pieces_seq,
        input = items,
        state = ProbeApplyCursor {
            skip_next: false,
            emitted: 0,
        },
        output = new!(ProbePieces),
        args = (merged,)
    );
    let next_ref = applied.reference().clone();
    call!(probe_finalize_handle_round, state, next_ref)
}

/// Driver for the handle-only storage-resident round-boundary probe.
#[sequence]
pub fn probe_handle_rounds(
    rounds: Vec<u32>,
    initial: ProbePieces,
    merged: String,
) -> ProbeRoundPieceHandle {
    let initial = call!(probe_publish_pieces, initial);
    call_recur_seq!(
        sequence = probe_handle_round_seq,
        input = rounds,
        state = ProbeRoundPieceHandle {
            round: 0,
            pieces_ref: initial.reference().clone(),
        },
        args = (merged,)
    )
}

/// P2 finalize tile: consumes the finalized draft whole (selection-bound
/// arg), plus a `select!` projection out of it, and returns the next outer
/// loop state (re-entering the threaded state).
#[tile]
pub fn probe_finalize_apply_round(
    applied: ProbePieces,
    first: String,
    round: u32,
) -> ProbeRoundPieces {
    let mut pieces = applied.pieces;
    pieces.push(first);
    ProbeRoundPieces {
        round: round + 1,
        pieces,
    }
}

/// P2 outer round body: open → select pieces → inner draft loop → consume
/// the finalized `AuthRef` via `select!` and as a plain tile's arg →
/// re-enter the threaded state through the finalize tile.
#[sequence(kind = recur)]
pub fn probe_apply_round_seq(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<ProbeRoundPieces>,
    merged: String,
) -> RecurSequenceState<ProbeRoundPieces> {
    let _round_ordinal = &input;
    let opened = call!(probe_open_apply_round, state);
    let items = select!(Vec<String>, opened.clone().pieces.pieces);
    let round_no = select!(u32, opened.round);
    let applied = call_recur_seq!(
        sequence = probe_apply_pieces_seq,
        input = items,
        state = ProbeApplyCursor {
            skip_next: false,
            emitted: 0,
        },
        output = new!(ProbePieces),
        args = (merged,)
    );
    let first = select!(String, applied.clone().pieces[0]);
    call!(probe_finalize_apply_round, applied, first, round_no)
}

/// P2 driver: outer recur sequence over a bounded round list, seeded from
/// a plain literal (A2) — pieces `[a, b, c]`, round 0.
#[sequence]
pub fn probe_apply_rounds(rounds: Vec<u32>, merged: String) -> ProbeRoundPieces {
    call_recur_seq!(
        sequence = probe_apply_round_seq,
        input = rounds,
        state = ProbeRoundPieces {
            round: 0,
            pieces: alloc::vec![String::from("a"), String::from("b"), String::from("c"),],
        },
        args = (merged,)
    )
}

/// P3 loop state: counts how many input items appear in the selection-bound
/// haystack arg (proving the tile materialized the full `Vec`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeVecArgState {
    pub seen: u32,
    pub hits: u32,
}

/// P3 plain tile with a `Vec` arg, called from a recur-sequence body.
#[tile]
pub fn probe_note_haystack(
    state: ProbeVecArgState,
    item: String,
    haystack: Vec<String>,
) -> ProbeVecArgState {
    let mut state = state;
    state.seen += 1;
    if haystack.contains(&item) {
        state.hits += 1;
    }
    state
}

/// P3 loop: the haystack rides `args` as an `AuthRef` and must trace as a
/// binding on every iteration and on the plain tile's own record.
#[sequence(kind = recur)]
pub fn probe_vec_arg_seq(
    input: RecurSequenceInput<String>,
    state: RecurSequenceState<ProbeVecArgState>,
    haystack: Vec<String>,
) -> RecurSequenceState<ProbeVecArgState> {
    call!(probe_note_haystack, state, input, haystack)
}

/// P3 driver: both the item list and the haystack arrive as internal
/// bindings (the test stores them and passes `internal!` refs).
#[sequence]
pub fn probe_vec_arg(items: Vec<String>, haystack: Vec<String>) -> ProbeVecArgState {
    call_recur_seq!(
        sequence = probe_vec_arg_seq,
        input = items,
        state = ProbeVecArgState { seen: 0, hits: 0 },
        args = (haystack,)
    )
}

/// P4 recur-tile state carrying a collection (deliberately — the probe
/// characterizes what the recur-tile driver traces inline per iteration).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeInlineState {
    pub log: Vec<String>,
}

/// P4 recur tile: chunked input item, collection-carrying state, and a
/// `Vec` arg — the driver materializes and traces all three inline per
/// iteration (`RecurTileIterationExec`, gap G7).
#[tile(kind = recur)]
pub fn probe_inline_recur(
    input: RecurInput<Vec<String>>,
    state: RecurState<ProbeInlineState>,
    extra: Vec<String>,
) -> RecurState<ProbeInlineState> {
    let mut state = state;
    let chunk = input.into_value();
    state
        .log
        .push(alloc::format!("{}+{}", chunk.join(""), extra.join("")));
    state
}

/// P4 driver: even though the chunk list and the extra arg arrive as
/// internal bindings, the recur-tile driver materializes and traces them
/// inline per iteration.
#[sequence]
pub fn probe_inline_recur_tile(chunks: Vec<Vec<String>>, extra: Vec<String>) -> ProbeInlineState {
    call_recur!(
        tile = probe_inline_recur,
        input = chunks,
        state = ProbeInlineState { log: Vec::new() },
        args = (extra,)
    )
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_chunk_probes");
        run()
    }

    fn merge(merge_index: u32, left: &str, right: &str) -> GemmaBpeMerge {
        GemmaBpeMerge {
            merge_index,
            left: left.to_string(),
            right: right.to_string(),
            merged_token: alloc::format!("{left}{right}"),
            has_token_id: true,
            token_id: merge_index,
        }
    }

    fn fixture() -> ProbeTable {
        ProbeTable {
            chunks: vec![
                vec![merge(0, "a", "b"), merge(1, "b", "a")],
                vec![merge(2, "c", "d"), merge(3, "d", "c"), merge(4, "e", "f")],
                vec![merge(5, "g", "h")],
            ],
        }
    }

    #[test]
    fn nested_chunk_lists_select_through_the_schema() {
        let selection = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeSelection, _>(
                __raster_sequence_auth_probe_select_chunks(internal!(ProbeTable, stored)),
            )
        });
        assert_eq!(
            selection,
            ProbeSelection {
                chunk_len: 3,
                entry_merge_index: 2,
                entry_merged: "cd".to_string(),
            }
        );
    }

    #[test]
    fn chunked_recur_input_breaks_mid_list() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur(
                    internal!(ProbeTable, stored),
                    "d".to_string(),
                ),
            )
        });
        assert!(state.found, "needle rule lives in the second chunk");
        assert_eq!(state.found_merge_index, 3);
        assert_eq!(
            state.visited_chunks, 2,
            "Break must stop the scan before the third chunk"
        );
    }

    #[test]
    fn chunked_recur_input_visits_every_chunk_without_a_match() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur(
                    internal!(ProbeTable, stored),
                    "z".to_string(),
                ),
            )
        });
        assert!(!state.found);
        assert_eq!(state.visited_chunks, 3);
    }

    #[test]
    fn recur_sequence_passes_chunk_handle_and_state_into_one_tile() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur_seq(
                    internal!(ProbeTable, stored),
                    "d".to_string(),
                ),
            )
        });
        assert!(state.found, "needle rule lives in the second chunk");
        assert_eq!(state.found_merge_index, 3);
        assert_eq!(
            state.visited_chunks, 2,
            "post-match chunks must no-op (recur sequences cannot break)"
        );
    }

    #[test]
    fn recur_sequence_without_a_match_scans_every_chunk() {
        let state = in_scope(|| {
            let stored = raster::store_internal_value(&fixture()).expect("store probe table");
            raster::materialize_auth_return::<ProbeScanState, _>(
                __raster_sequence_auth_probe_chunked_recur_seq(
                    internal!(ProbeTable, stored),
                    "z".to_string(),
                ),
            )
        });
        assert!(!state.found);
        assert_eq!(state.visited_chunks, 3);
    }

    use raster::core::draft::DraftReplayHandle;
    use raster::core::trace::FnInputValue;
    use serde::Deserialize;

    use crate::test_trace::{
        capture_trace_events, inline_bytes, recur_tile_iteration_records, sequence_start_records,
        tile_exec_records,
    };

    fn seed_pieces(round: u32, pieces: Vec<&str>) -> ProbeRoundPieces {
        ProbeRoundPieces {
            round,
            pieces: pieces.into_iter().map(ToString::to_string).collect(),
        }
    }

    fn probe_pieces(pieces: Vec<&str>) -> ProbePieces {
        ProbePieces {
            pieces: pieces.into_iter().map(ToString::to_string).collect(),
        }
    }

    fn bytes_contain(bytes: &[u8], marker: &str) -> bool {
        bytes
            .windows(marker.len())
            .any(|window| window == marker.as_bytes())
    }

    // --- P1: round-boundary characterization ---

    /// P1(c): the threaded state enters every iteration's
    /// `RecurSequenceStart` record as a full inline postcard value — not a
    /// binding — and its per-iteration byte size scales linearly with the
    /// carried collection.
    #[test]
    fn p1_state_traces_inline_per_iteration_and_scales_with_pieces() {
        let (final_state, events) = capture_trace_events(|| {
            in_scope(|| {
                let rounds =
                    raster::store_internal_value(&vec![0u32, 1, 2]).expect("store round list");
                raster::materialize_auth_return::<ProbeRoundPieces, _>(
                    __raster_sequence_auth_probe_round_boundary(
                        internal!(Vec<u32>, rounds),
                        "x".to_string(),
                    ),
                )
            })
        });
        assert_eq!(final_state.round, 3);
        assert_eq!(final_state.pieces.len(), 4);

        let records = sequence_start_records(&events, "probe_round_boundary_seq");
        assert_eq!(records.len(), 3, "one start record per iteration");
        let mut previous_len = 0usize;
        for (iteration, record) in records.iter().enumerate() {
            let input = record.input.as_ref().expect("iteration input trace");
            // values[0] = input marker, values[1] = threaded state,
            // values[2] = the `appended` arg.
            let state_bytes = inline_bytes(&input.values[1]);
            let state: ProbeRoundPieces =
                raster::core::postcard::from_bytes(&state_bytes).expect("state should decode");
            assert_eq!(state.round as usize, iteration);
            assert_eq!(
                state.pieces.len(),
                iteration + 1,
                "the full carried collection rides the record"
            );
            assert!(
                input.internal.get("state").is_none(),
                "state is inline, never an internal binding"
            );
            assert!(
                state_bytes.len() > previous_len,
                "state bytes grow with the carried pieces"
            );
            previous_len = state_bytes.len();
        }

        // Size scaling across runs: 64 seed pieces of 32 bytes each must
        // inflate the first iteration's state record by at least the
        // payload size.
        let (_, big_events) = capture_trace_events(|| {
            in_scope(|| {
                let rounds = raster::store_internal_value(&vec![0u32]).expect("store round list");
                raster::materialize_auth_return::<ProbeRoundPieces, _>(
                    __raster_sequence_auth_probe_round_boundary_big(
                        internal!(Vec<u32>, rounds),
                        "x".to_string(),
                    ),
                )
            })
        });
        let big_records = sequence_start_records(&big_events, "probe_round_boundary_seq");
        let big_bytes = inline_bytes(&big_records[0].input.as_ref().expect("input").values[1]);
        assert!(
            big_bytes.len() >= 64 * 32,
            "inline state bytes scale with the collection ({} < {})",
            big_bytes.len(),
            64 * 32
        );
    }

    /// P1(a)+(b): the body tile's returned state persists in internal
    /// storage at the tile's output coordinates, and the re-entry resolve
    /// validates the reference commitment (tamper rejected) before
    /// rematerializing the value.
    #[test]
    fn p1_tile_output_persists_in_internal_storage_and_resolve_validates() {
        in_scope(|| {
            let reference = raster::materialize_auth_return::<InternalRef, _>(
                __raster_sequence_auth_probe_tile_output_reference(
                    seed_pieces(4, vec!["a", "b"]),
                    "x".to_string(),
                ),
            );

            // (a) The stored tile output resolves from internal storage.
            let resolved = raster::resolve_internal_value::<ProbeRoundPieces>(reference.clone())
                .expect("tile output should persist in internal storage");
            assert_eq!(resolved.value.round, 5);
            assert_eq!(
                resolved.value.pieces,
                vec!["a".to_string(), "b".to_string(), "x4".to_string()]
            );

            // (b) The resolve validates the commitment: a tampered
            // reference is rejected, not silently rematerialized.
            let mut tampered = reference;
            tampered.commitment[0] ^= 0xFF;
            let error = raster::resolve_internal_value::<ProbeRoundPieces>(tampered)
                .expect_err("tampered commitment must fail the resolve");
            assert!(
                alloc::format!("{error}").contains("commitment mismatch"),
                "unexpected resolve error: {error}"
            );
        });
    }

    /// Handle-state feasibility leg: the outer recur state carries only an
    /// `InternalRef`; pieces rehydrate inside a plain tile and do not appear
    /// in the recur state trace.
    #[test]
    fn handle_state_threads_piece_refs_without_inline_piece_payloads() {
        const LEFT: &str = "HANDLE_LEFT_SENTINEL_0123456789";
        const RIGHT: &str = "HANDLE_RIGHT_SENTINEL_0123456789";
        const THIRD: &str = "HANDLE_THIRD_SENTINEL_0123456789";
        let merged = "HANDLE_MERGED".to_string();

        let (final_state, events) = capture_trace_events(|| {
            in_scope(|| {
                let rounds =
                    raster::store_internal_value(&vec![0u32, 1]).expect("store round list");
                let initial = raster::store_internal_value(&probe_pieces(vec![LEFT, RIGHT, THIRD]))
                    .expect("store initial pieces");
                raster::materialize_auth_return::<ProbeRoundPieceHandle, _>(
                    __raster_sequence_auth_probe_handle_rounds(
                        internal!(Vec<u32>, rounds),
                        internal!(ProbePieces, initial),
                        merged.clone(),
                    ),
                )
            })
        });

        assert_eq!(final_state.round, 2);
        let final_pieces =
            raster::resolve_internal_value::<ProbePieces>(final_state.pieces_ref.clone())
                .expect("final probe pieces should resolve")
                .value;
        assert_eq!(final_pieces.pieces, vec![merged]);

        let records = sequence_start_records(&events, "probe_handle_round_seq");
        assert_eq!(records.len(), 2, "one start record per iteration");
        for (iteration, record) in records.iter().enumerate() {
            let input = record.input.as_ref().expect("iteration input trace");
            let state_bytes = inline_bytes(&input.values[1]);
            let state: ProbeRoundPieceHandle =
                raster::core::postcard::from_bytes(&state_bytes).expect("state should decode");
            assert_eq!(state.round as usize, iteration);
            for marker in [LEFT, RIGHT, THIRD] {
                assert!(
                    !bytes_contain(&state_bytes, marker),
                    "handle-only state must not inline prompt piece marker {marker}"
                );
            }
        }

        let mut tampered = final_state.pieces_ref;
        tampered.commitment[0] ^= 0xFF;
        let error = raster::resolve_internal_value::<ProbePieces>(tampered)
            .expect_err("tampered piece ref must fail");
        assert!(
            alloc::format!("{error}").contains("commitment mismatch"),
            "unexpected resolve error: {error}"
        );
    }

    // --- P2: fresh RecurOutput draft inside a recur-sequence body ---

    /// P2 functional leg: per outer round, a fresh draft accumulates the
    /// applied pieces; the finalized `AuthRef` feeds a `select!` projection
    /// and a plain tile's selection-bound arg; the tile's return re-enters
    /// the outer threaded state.
    #[test]
    fn p2_draft_accumulates_and_finalizes_inside_a_body_iteration() {
        let final_state = in_scope(|| {
            let rounds = raster::store_internal_value(&vec![0u32, 1]).expect("store round list");
            raster::materialize_auth_return::<ProbeRoundPieces, _>(
                __raster_sequence_auth_probe_apply_rounds(
                    internal!(Vec<u32>, rounds),
                    "M".to_string(),
                ),
            )
        });
        // Round 1: [a,b,c] → apply (merge at 0) → [M,c] → finalize appends
        // select!-ed [0] → [M,c,M]. Round 2: [M,c,M] → [M,M] → [M,M,M].
        assert_eq!(final_state.round, 2);
        assert_eq!(
            final_state.pieces,
            vec!["M".to_string(), "M".to_string(), "M".to_string()]
        );
    }

    /// P2 trace leg: the draft rides iteration records as an inline replay
    /// handle (anchor + root, not payload); the input item and the
    /// finalized-draft arg ride as internal bindings.
    #[test]
    fn p2_draft_rides_the_trace_as_replay_handle_and_bindings() {
        let (_, events) = capture_trace_events(|| {
            in_scope(|| {
                let rounds = raster::store_internal_value(&vec![0u32]).expect("store round list");
                raster::materialize_auth_return::<ProbeRoundPieces, _>(
                    __raster_sequence_auth_probe_apply_rounds(
                        internal!(Vec<u32>, rounds),
                        "M".to_string(),
                    ),
                )
            })
        });

        let inner_records = sequence_start_records(&events, "probe_apply_pieces_seq");
        assert_eq!(inner_records.len(), 3, "one iteration per piece");
        for record in &inner_records {
            let input = record.input.as_ref().expect("iteration input trace");
            // values: [input marker, cursor state, output draft, merged].
            let handle_bytes = inline_bytes(&input.values[2]);
            let handle: DraftReplayHandle = raster::core::postcard::from_bytes(&handle_bytes)
                .expect("output draft should trace as a replay handle");
            assert_eq!(handle.schema_hash, ProbePieces::schema_hash());
            assert!(
                raster::core::postcard::from_bytes::<Draft<ProbePieces>>(&handle_bytes).is_err(),
                "trace bytes must not deserialize into a live draft"
            );
            assert!(
                input.internal.contains_key("input"),
                "the piece must reach the iteration as an internal binding"
            );
        }

        let finalize_records = tile_exec_records(&events, "probe_finalize_apply_round");
        assert_eq!(finalize_records.len(), 1);
        let finalize_input = finalize_records[0]
            .input
            .as_ref()
            .expect("finalize tile input trace");
        assert_eq!(
            finalize_input.values[0],
            FnInputValue::InternalBinding,
            "the finalized draft must reach the follow-up tile as a binding"
        );
        assert!(finalize_input.internal.contains_key("applied"));
        assert_eq!(
            finalize_input.values[1],
            FnInputValue::InternalBinding,
            "the select! projection out of the finalized draft is a binding"
        );
    }

    // --- P3: Vec selection-bound arg on a plain tile in a recur-sequence
    // body ---

    /// P3: the arg traces as a binding on the iteration record and the
    /// plain tile's own record, and the tile materializes the full value at
    /// execution (it can check membership against every element).
    #[test]
    fn p3_vec_arg_traces_as_binding_and_materializes_at_execution() {
        let (state, events) = capture_trace_events(|| {
            in_scope(|| {
                let items = raster::store_internal_value(&vec![
                    "x".to_string(),
                    "y".to_string(),
                    "z".to_string(),
                ])
                .expect("store item list");
                let haystack = raster::store_internal_value(&vec![
                    "y".to_string(),
                    "z".to_string(),
                    "w".to_string(),
                ])
                .expect("store haystack");
                raster::materialize_auth_return::<ProbeVecArgState, _>(
                    __raster_sequence_auth_probe_vec_arg(
                        internal!(Vec<String>, items),
                        internal!(Vec<String>, haystack),
                    ),
                )
            })
        });
        assert_eq!(state.seen, 3);
        assert_eq!(state.hits, 2, "the tile materialized the full haystack");

        let iteration_records = sequence_start_records(&events, "probe_vec_arg_seq");
        assert_eq!(iteration_records.len(), 3);
        for record in &iteration_records {
            let input = record.input.as_ref().expect("iteration input trace");
            // values: [input marker, state, haystack].
            assert_eq!(
                input.values[2],
                FnInputValue::InternalBinding,
                "the Vec arg must trace as a binding, never inline"
            );
            assert!(input.internal.contains_key("haystack"));
        }

        let tile_records = tile_exec_records(&events, "probe_note_haystack");
        assert_eq!(tile_records.len(), 3);
        for record in &tile_records {
            let input = record.input.as_ref().expect("tile input trace");
            // values: [state, item, haystack].
            assert_eq!(
                input.values[2],
                FnInputValue::InternalBinding,
                "the tile record keeps the Vec arg as a binding"
            );
        }
    }

    // --- P4: recur-tile per-iteration tracing scope (gap G7) ---

    /// P4: recur-*tile* drivers trace the input item, the state, and the
    /// args inline (full postcard payloads) in every
    /// `RecurTileIterationExec` record — the reason recur-tile loops carry
    /// scalars only.
    #[test]
    fn p4_recur_tile_iterations_trace_input_state_and_args_inline() {
        let chunks = vec![
            vec!["aa".to_string(), "bb".to_string()],
            vec!["cc".to_string()],
        ];
        let extra = vec!["E1".to_string(), "E2".to_string()];
        let (state, events) = capture_trace_events(|| {
            in_scope(|| {
                let chunk_source = raster::store_internal_value(&chunks).expect("store chunk list");
                let extra_source = raster::store_internal_value(&extra).expect("store extra");
                raster::materialize_auth_return::<ProbeInlineState, _>(
                    __raster_sequence_auth_probe_inline_recur_tile(
                        internal!(Vec<Vec<String>>, chunk_source),
                        internal!(Vec<String>, extra_source),
                    ),
                )
            })
        });
        assert_eq!(
            state.log,
            vec!["aabb+E1E2".to_string(), "cc+E1E2".to_string()]
        );

        let records = recur_tile_iteration_records(&events, "probe_inline_recur");
        assert_eq!(records.len(), 2, "one iteration record per chunk");
        for (iteration, record) in records.iter().enumerate() {
            let input = record.input.as_ref().expect("iteration input trace");
            // values: [input, state, extra] — every one inline.
            #[derive(Debug, Deserialize)]
            struct RecurInputMirror {
                value: Vec<String>,
                index: u64,
                len: u64,
            }
            let item_bytes = inline_bytes(&input.values[0]);
            let item: RecurInputMirror = raster::core::postcard::from_bytes(&item_bytes)
                .expect("recur input should trace inline with the full item");
            assert_eq!(item.value, chunks[iteration]);
            assert_eq!(item.index as usize, iteration);
            assert_eq!(item.len, 2);

            let state_bytes = inline_bytes(&input.values[1]);
            let state: ProbeInlineState = raster::core::postcard::from_bytes(&state_bytes)
                .expect("recur state should trace inline");
            assert_eq!(state.log.len(), iteration, "prior-iteration state, in full");

            let extra_bytes = inline_bytes(&input.values[2]);
            let traced_extra: Vec<String> = raster::core::postcard::from_bytes(&extra_bytes)
                .expect("recur args should trace inline");
            assert_eq!(traced_extra, extra, "args re-traced inline per iteration");
        }
    }

    #[test]
    fn recur_sequence_nests_inside_a_recur_sequence_iteration() {
        let nested_fixture = ProbeNestedFixture {
            chunks: fixture().chunks,
            rounds: vec![0, 1],
        };
        let state = in_scope(|| {
            let stored =
                raster::store_internal_value(&nested_fixture).expect("store nested fixture");
            raster::materialize_auth_return::<ProbeRoundState, _>(
                __raster_sequence_auth_probe_nested_recur_seq(
                    internal!(ProbeNestedFixture, stored),
                    "d".to_string(),
                ),
            )
        });
        assert_eq!(state.rounds_run, 2, "every round ordinal must execute");
        assert_eq!(
            state.found_rounds, 2,
            "each round's inner scan must find the rule"
        );
        assert_eq!(state.found_merge_index, 3);
        assert_eq!(
            state.total_visited_chunks, 4,
            "each round scans two chunks before the match no-ops the rest"
        );
    }
}
