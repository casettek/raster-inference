//! The BPE merge-round loop (sim `merge_bpe_tokenize_prompt`,
//! `tiles.rs:67-78`) — the gap-G1 canonical case.
//!
//! Sim shape: a pair-state recursive sequence looping *until* no merge
//! candidate remains. Real recur sequences are list-driven and cannot break
//! early, so the loop runs over a bounded round list (`initial_piece_count
//! − 1` ordinals — each merge removes one piece) and converged rounds
//! no-op (port-plan deviation D3).
//!
//! Storage-resident round body (deviations D15/D17): `open_round`
//! republishes the round's working pieces behind the selectable `BpePieces`
//! root; `build_pairs` derives the adjacent-pair list as its own internal
//! ref for the scan's selection-bound arg; the apply loop recurs over a
//! `select!` projection of the round's pieces, accumulating the next
//! round's pieces in a fresh `RecurOutput<BpePieces>` draft; and
//! `finalize_round` re-enters the threaded state (`From<AuthRef<T>> for
//! RecurSequenceState<T>`). The loop state's `pieces` is the program's
//! single loop-carried collection — storage-backed between iterations at
//! the pinned rev, trace form characterized by probe P1 (D6 re-founding).
//!
//! Value-flow idiom (port-plan constraint A2): the opaque threaded state
//! enters tiles as an ordinary argument (`RecurSequenceState<T>:
//! IntoAuthValue<T>`); inner recur loops seed from plain literals; the
//! final tile's `AuthRef` return re-enters the threaded state.
//!
//! Data placement (D13, unchanged): the model-scoped merge table stays an
//! `AuthRef` all the way into the scan recur *sequence*, whose per-chunk
//! tile receives each chunk as an external-selection binding — the chunk
//! bytes never ride the trace inline.

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaBpeMerge;

// Glob imports: `call!`/`call_recur_seq!` resolve hidden per-tile marker
// types generated next to each tile fn, so the whole defining module must
// be in scope (same convention as the WS1 probe crate).
use crate::bpe_apply::*;
use crate::bpe_scan::*;
use crate::types::{BpePieces, GemmaBpeApplyCursor, GemmaBpeLoopState, GemmaBpeScanState};

/// One BPE merge round: open (pieces behind the selectable root) →
/// priority-order merge scan → scalar apply decision → per-piece draft
/// rebuild → next loop state.
#[sequence(kind = recur)]
pub fn merge_bpe_round(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<GemmaBpeLoopState>,
    staged_pieces: BpePieces,
    merge_chunks: Vec<Vec<GemmaBpeMerge>>,
) -> RecurSequenceState<GemmaBpeLoopState> {
    // The input item is only the round budget ordinal (G1 bounded list).
    let _round_ordinal = &input;
    let opened = call!(open_round, state, staged_pieces);
    let round_pieces = select!(BpePieces, opened.clone().pieces);
    let skip = select!(bool, opened.clone().skip);
    let pairs = call!(build_pairs, round_pieces.clone());
    let scan = call_recur_seq!(
        sequence = scan_merge_chunks,
        input = merge_chunks,
        state = GemmaBpeScanState::initial(),
        args = (skip.clone(), pairs)
    );
    let decision = call!(finalize_bpe_merge_scan, scan, skip);
    let piece_items = select!(Vec<String>, round_pieces.pieces);
    let applied = call_recur_seq!(
        sequence = apply_round_pieces,
        input = piece_items,
        state = GemmaBpeApplyCursor::initial(),
        output = new!(BpePieces),
        args = (decision.clone(),)
    );
    // The tile's `AuthRef` return re-enters the threaded state through the
    // generated wrapper's `Into<RecurSequenceState<_>>` conversion.
    call!(finalize_round, applied, decision, opened)
}
