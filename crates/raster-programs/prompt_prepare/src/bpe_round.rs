//! The BPE merge-round loop (sim `merge_bpe_tokenize_prompt`,
//! `tiles.rs:67-78`) — the gap-G1 canonical case.
//!
//! Sim shape: a pair-state recursive sequence looping *until* no merge
//! candidate remains. Real recur sequences are list-driven and cannot break
//! early, so the loop runs over a bounded round list (`initial_piece_count
//! − 1` ordinals — each merge removes one piece) and converged rounds
//! no-op (port-plan deviation D3).
//!
//! Value-flow idiom (port-plan constraint A2): the opaque threaded state
//! enters tiles as an ordinary argument (`RecurSequenceState<T>:
//! IntoAuthValue<T>`); inner recur loops seed from plain literals and take
//! their heavy read-only context through `args = (…)`; the final tile's
//! `AuthRef` return re-enters the threaded state
//! (`From<AuthRef<T>> for RecurSequenceState<T>`).

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster_program_gemma_externals::types::GemmaBpeMergeLookupEntry;

// Glob imports: `call!`/`call_recur!` resolve hidden per-tile marker types
// generated next to each tile fn, so the whole defining module must be in
// scope (same convention as the WS1 probe crate).
use crate::bpe_apply::*;
use crate::bpe_scan::*;
use crate::types::{BpeConfig, GemmaBpeApplyState, GemmaBpeLoopState, GemmaBpeScanState};

/// One BPE merge round: chunked pair scan → merge decision → chunked
/// pieces rebuild → next loop state.
#[sequence(kind = recur)]
pub fn merge_bpe_round(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<GemmaBpeLoopState>,
    initial_pieces: Vec<String>,
    config: BpeConfig,
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
    scan_chunks: Vec<u32>,
    apply_chunks: Vec<u32>,
) -> RecurSequenceState<GemmaBpeLoopState> {
    // The input item is only the round budget ordinal (G1 bounded list).
    let _round_ordinal = &input;
    let round = call!(init_bpe_merge_scan, state, initial_pieces);
    let scan = call_recur!(
        tile = scan_bpe_merge_candidates,
        input = scan_chunks,
        state = GemmaBpeScanState::initial(),
        args = (round.clone(), merge_lookup, config.clone())
    );
    let decision = call!(finalize_bpe_merge_scan, scan, round);
    let iteration = call!(init_bpe_merge_iteration, decision);
    let apply = call_recur!(
        tile = apply_bpe_merge_chunk_or_complete,
        input = apply_chunks,
        state = GemmaBpeApplyState::initial(),
        args = (iteration.clone(), config)
    );
    // The tile's `AuthRef` return re-enters the threaded state through the
    // generated wrapper's `Into<RecurSequenceState<_>>` conversion.
    call!(finalize_bpe_merge_iteration, apply, iteration)
}
