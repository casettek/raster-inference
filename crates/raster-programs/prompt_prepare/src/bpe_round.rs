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
//! RecurSequenceState<T>`). The loop state carries the current pieces'
//! internal-storage handle plus scalars; the prompt pieces themselves no
//! longer ride the inline recur state.
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

use alloc::vec::Vec;
use raster::prelude::*;

// Glob imports: `call!`/`call_recur_seq!` resolve hidden per-tile marker
// types generated next to each tile fn, so the whole defining module must
// be in scope (same convention as the WS1 probe crate).
use crate::bpe_apply::*;
use crate::bpe_scan::*;
use crate::budgets::*;
use crate::types::{
    BpePieces, GemmaBpeApplyCursor, GemmaBpeLoopState, GemmaBpeScanState, TokenizerTables,
};

/// One BPE merge round: open (pieces behind the selectable root) →
/// priority-order merge scan → scalar apply decision → per-piece draft
/// rebuild → next loop state.
#[sequence(kind = recur)]
pub fn merge_bpe_round(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<GemmaBpeLoopState>,
    tokenizer: TokenizerTables,
    merge_chunk_ordinals: Vec<u32>,
) -> RecurSequenceState<GemmaBpeLoopState> {
    // The input item is only the round budget ordinal (G1 bounded list).
    let _round_ordinal = &input;
    let opened = call!(open_round, state.clone());
    let scan = call_recur!(
        tile = scan_one_merge_chunk,
        input = merge_chunk_ordinals,
        state = GemmaBpeScanState::initial(),
        args = (tokenizer.clone(), opened.clone())
    );
    let decision = call!(finalize_bpe_merge_scan, scan, opened.clone());
    let apply_ordinals = call!(build_round_piece_ordinals, opened.clone());
    let piece_items = select!(Vec<u32>, apply_ordinals.ordinals);
    let applied = call_recur_seq!(
        sequence = apply_round_pieces,
        input = piece_items,
        state = GemmaBpeApplyCursor::initial(),
        output = new!(BpePieces),
        args = (decision.clone(), opened.clone())
    );
    let applied_ref = applied.reference().clone();
    // The tile's `AuthRef` return re-enters the threaded state through the
    // generated wrapper's `Into<RecurSequenceState<_>>` conversion.
    call!(finalize_round, decision, opened, state, applied_ref)
}
