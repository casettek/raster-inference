//! BPE merge-round scan phase (sim `tiles.rs`: `init_bpe_merge_scan`,
//! `scan_bpe_merge_candidates`, `finalize_bpe_merge_scan`).
//!
//! Storage-read shape: the round opens through `open_round`, whose output
//! carries round scalars plus the current `BpePieces` internal ref. The scan
//! loop receives only merge-chunk ordinals and compact source descriptors;
//! each scan tile reads the selected tokenizer chunk and the needed prompt
//! pieces inside tile execution.
//!
//! The scan keeps the D11 orientation: a recur sequence over the
//! model-scoped `merge_chunks` (priority order preserved); the first rule
//! with an adjacent-pair occurrence wins (lowest `merge_index` globally,
//! leftmost pair). Recur sequences cannot break early (gap G1), so
//! post-winner and skipped chunks no-op, and a converged round pays one
//! full table pass.

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;
use raster::InternalRef;
use raster_program_gemma_externals::types::GemmaBpeMerge;

use crate::types::{
    field_index_selector, read_bpe_piece, read_tokenizer_selection, BpePieces,
    GemmaBpeApplyDecision, GemmaBpeLoopState, GemmaBpeOpenedRound, GemmaBpeScanCandidate,
    GemmaBpeScanState, TokenizerTables,
};

/// Leftmost occurrence of `(left, right)` in the round's adjacent-pair
/// list. Plain helper — only reachable from tile bodies.
pub(crate) fn find_leftmost_pair(
    pieces_ref: &InternalRef,
    piece_count: u32,
    left: &str,
    right: &str,
) -> Option<u32> {
    for pair_idx in 0..piece_count.saturating_sub(1) {
        let pair_left = read_bpe_piece(pieces_ref, pair_idx);
        let pair_right = read_bpe_piece(pieces_ref, pair_idx + 1);
        if pair_left == left && pair_right == right {
            return Some(pair_idx);
        }
    }
    None
}

/// Opens one BPE merge round: resolves the current pieces handle, flags
/// no-op rounds (loop complete or errored; recur sequences cannot break
/// early — gap G1), and republishes the pieces behind a selectable root so
/// the rest of the round consumes them only through authenticated reads. The
/// deferred error flattens to `(has_error, error)` scalars (no `Option` in
/// selectable shapes — G3); the message text rides through untouched (H4).
#[tile]
pub fn open_round(state: GemmaBpeLoopState) -> GemmaBpeOpenedRound {
    let (has_error, error) = match state.error {
        Some(message) => (true, message),
        None => (false, String::new()),
    };
    let skip = state.complete || has_error;
    if skip {
        return GemmaBpeOpenedRound {
            skip,
            round: state.round,
            piece_count: state.piece_count,
            has_error,
            error,
            pieces_ref: state.pieces_ref,
        };
    }

    let piece_count = if state.piece_count == 0 {
        raster::resolve_internal_value::<BpePieces>(state.pieces_ref.clone())
            .unwrap_or_else(|error| panic!("Failed to resolve BPE pieces for round: {error}"))
            .value
            .pieces
            .len() as u32
    } else {
        state.piece_count
    };
    GemmaBpeOpenedRound {
        skip,
        round: state.round,
        piece_count,
        has_error,
        error,
        pieces_ref: state.pieces_ref,
    }
}

/// One merge-table chunk per execution: checks its rules, in order, for an
/// occurrence in the round's adjacent pairs. The first hit is the global
/// winner — chunks preserve priority order — with the leftmost pair
/// occurrence; it sets `best` + `done` and every later chunk no-ops (as do
/// all chunks of a skipped round). Infallible per port-plan constraint A1.
#[tile(kind = recur)]
pub fn scan_one_merge_chunk(
    input: RecurInput<u32>,
    state: RecurState<GemmaBpeScanState>,
    tokenizer: TokenizerTables,
    opened: GemmaBpeOpenedRound,
) -> RecurControl<RecurState<GemmaBpeScanState>> {
    let mut state = state;
    if opened.skip || state.done {
        return RecurControl::Break(state);
    }
    let chunk_idx = *input.value();

    let chunk = read_tokenizer_selection::<Vec<GemmaBpeMerge>>(
        &tokenizer,
        field_index_selector("merge_chunks", chunk_idx),
        "merge chunk",
    );
    state.chunks_scanned += 1;
    for rule in chunk {
        if let Some(pair_idx) = find_leftmost_pair(
            &opened.pieces_ref,
            opened.piece_count,
            &rule.left,
            &rule.right,
        ) {
            state.best = Some(GemmaBpeScanCandidate {
                pair_idx,
                merge_index: rule.merge_index,
                merged: rule.merged_token,
            });
            state.done = true;
            return RecurControl::Break(state);
        }
    }
    RecurControl::Continue(state)
}

/// Closes the scan phase into the apply decision, scalars only (D15).
/// `selection: None` (the scan exhausted the table) and skipped rounds
/// collapse into `skip`; `finalize_round` carries the incoming pieces
/// forward for both.
#[tile]
pub fn finalize_bpe_merge_scan(
    scan: GemmaBpeScanState,
    opened: GemmaBpeOpenedRound,
) -> GemmaBpeApplyDecision {
    let selection = if opened.skip { None } else { scan.best };
    match selection {
        Some(candidate) => GemmaBpeApplyDecision {
            skip: false,
            merge_piece_idx: candidate.pair_idx,
            merged: candidate.merged,
        },
        None => GemmaBpeApplyDecision {
            skip: true,
            merge_piece_idx: 0,
            merged: String::new(),
        },
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use raster_program_gemma_externals::types::{
        GemmaDecoderMetadata, GemmaTokenizer, GemmaTokenizerMetadata,
    };

    use super::*;

    fn in_scope<T>(run: impl FnOnce() -> T) -> T {
        let _guard = raster::__private::SequenceScopeGuard::enter("prompt_prepare_scan_tests");
        run()
    }

    fn merge_rule(merge_index: u32, left: &str, right: &str) -> GemmaBpeMerge {
        GemmaBpeMerge {
            merge_index,
            left: left.to_string(),
            right: right.to_string(),
            merged_token: format!("{left}{right}"),
            has_token_id: true,
            token_id: merge_index,
        }
    }

    /// Priority-ordered rules split into `width`-wide chunks (the encoder's
    /// `chunk_table` shape).
    fn chunked(rules: Vec<GemmaBpeMerge>, width: usize) -> Vec<Vec<GemmaBpeMerge>> {
        let mut chunks: Vec<Vec<GemmaBpeMerge>> = Vec::new();
        for rule in rules {
            match chunks.last_mut() {
                Some(chunk) if chunk.len() < width => chunk.push(rule),
                _ => chunks.push(vec![rule]),
            }
        }
        chunks
    }

    fn tokenizer_with_merge_chunks(merge_chunks: Vec<Vec<GemmaBpeMerge>>) -> TokenizerTables {
        let tokenizer = GemmaTokenizer {
            metadata: GemmaTokenizerMetadata {
                space_replacement: "\u{2581}".to_string(),
                split_delimiter: " ".to_string(),
                split_behavior: "merged_with_previous".to_string(),
                invert: false,
                unk_token: "<unk>".to_string(),
                fuse_unk: false,
                byte_fallback: true,
                ignore_merges: false,
            },
            decoder: GemmaDecoderMetadata {
                space_replacement: "\u{2581}".to_string(),
                byte_fallback: true,
                fuse_decoder: false,
            },
            token_lookup_chunks: Vec::new(),
            tokens_by_id: Vec::new(),
            special_tokens: Vec::new(),
            merge_chunks,
        };
        let reference = raster::store_internal_value(&tokenizer).expect("store tokenizer");
        TokenizerTables::internal(reference)
    }

    /// Drives the per-chunk tile over every chunk — the recur sequence's
    /// full pass (no early break).
    fn run_scan(
        opened: &GemmaBpeOpenedRound,
        merge_chunks: &[Vec<GemmaBpeMerge>],
    ) -> GemmaBpeScanState {
        in_scope(|| {
            let tokenizer = tokenizer_with_merge_chunks(merge_chunks.to_vec());
            let mut state = GemmaBpeScanState::initial();
            for chunk_idx in 0..merge_chunks.len() as u32 {
                let input = RecurInput::new(chunk_idx, chunk_idx as u64, merge_chunks.len() as u64);
                let next = scan_one_merge_chunk(
                    input,
                    RecurState::new(state),
                    tokenizer.clone(),
                    opened.clone(),
                );
                match next {
                    RecurControl::Continue(next) => state = next.into_inner(),
                    RecurControl::Break(next) => {
                        state = next.into_inner();
                        break;
                    }
                }
            }
            state
        })
    }

    fn resolved_pieces(round: &GemmaBpeOpenedRound) -> Vec<String> {
        raster::resolve_internal_value::<BpePieces>(round.pieces_ref.clone())
            .expect("resolve round pieces")
            .value
            .pieces
    }

    /// The sim's selection rule, as specified by `tiles.rs:320-329`: lowest
    /// rank (merge_index) wins; ties keep the earlier pair occurrence.
    fn sim_reference_selection(
        pieces: &[String],
        rules: &[GemmaBpeMerge],
    ) -> Option<GemmaBpeScanCandidate> {
        let mut best: Option<GemmaBpeScanCandidate> = None;
        for pair_idx in 0..pieces.len().saturating_sub(1) {
            let left = &pieces[pair_idx];
            let right = &pieces[pair_idx + 1];
            let Some(rule) = rules
                .iter()
                .find(|rule| &rule.left == left && &rule.right == right)
            else {
                continue;
            };
            let better = match &best {
                Some(candidate) => rule.merge_index < candidate.merge_index,
                None => true,
            };
            if better {
                best = Some(GemmaBpeScanCandidate {
                    pair_idx: pair_idx as u32,
                    merge_index: rule.merge_index,
                    merged: rule.merged_token.clone(),
                });
            }
        }
        best
    }

    fn staged(pieces: Vec<&str>) -> BpePieces {
        BpePieces {
            pieces: pieces
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        }
    }

    fn state_for(pieces: Vec<&str>) -> GemmaBpeLoopState {
        let pieces = staged(pieces);
        let piece_count = pieces.pieces.len() as u32;
        let pieces_ref = raster::store_internal_value(&pieces).expect("store BPE pieces");
        let mut state = GemmaBpeLoopState::initial(pieces_ref);
        state.piece_count = piece_count;
        state
    }

    fn round_for(pieces: Vec<&str>) -> GemmaBpeOpenedRound {
        in_scope(|| open_round(state_for(pieces)))
    }

    #[test]
    fn round_opens_from_stored_piece_handle() {
        let round = round_for(vec!["a", "b"]);
        assert!(!round.skip);
        assert!(!round.has_error);
        assert_eq!(
            resolved_pieces(&round),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(round.piece_count, 2);
    }

    #[test]
    fn completed_loop_rounds_are_skipped() {
        let round = in_scope(|| {
            let mut state = state_for(vec!["ab"]);
            state.complete = true;
            state.piece_count = 1;
            open_round(state)
        });
        assert!(round.skip);
        assert_eq!(round.piece_count, 1);
    }

    #[test]
    fn deferred_errors_flatten_to_scalars_with_the_message_intact() {
        let round = in_scope(|| {
            let mut state = state_for(vec!["a", "b"]);
            state.piece_count = 2;
            state.error = Some("BPE merge apply finalized with 0 pieces, expected 1".to_string());
            open_round(state)
        });
        assert!(round.skip, "errored rounds must no-op");
        assert!(round.has_error);
        assert_eq!(
            round.error,
            "BPE merge apply finalized with 0 pieces, expected 1"
        );
    }

    #[test]
    fn skipped_rounds_no_op_every_chunk() {
        let round = in_scope(|| {
            let mut state = state_for(vec!["ab"]);
            state.complete = true;
            state.piece_count = 1;
            open_round(state)
        });
        let chunks = chunked(vec![merge_rule(0, "a", "b"), merge_rule(1, "b", "a")], 1);
        let scan = run_scan(&round, &chunks);
        assert_eq!(scan.chunks_scanned, 0, "skip must not scan any rules");
        assert!(scan.best.is_none());
        assert!(!scan.done);
    }

    #[test]
    fn priority_order_wins_over_pair_position() {
        // Rule 0 ("b","a") matches at pair 1; rule 1 ("a","b") matches at
        // pair 0. Priority (merge_index 0) must win even though its pair
        // occurs later in the pieces.
        in_scope(|| {
            let round = round_for(vec!["a", "b", "a"]);
            let rules = vec![merge_rule(0, "b", "a"), merge_rule(1, "a", "b")];
            let chunks = chunked(rules.clone(), 1);
            let scan = run_scan(&round, &chunks);
            let best = scan.best.clone().expect("candidate should be found");
            assert_eq!(best.merge_index, 0);
            assert_eq!(best.pair_idx, 1);
            assert_eq!(best.merged, "ba");
            assert!(scan.done, "the winning chunk must set done");
            assert_eq!(scan.chunks_scanned, 1, "chunks after the winner must no-op");
            assert_eq!(
                Some(best),
                sim_reference_selection(&resolved_pieces(&round), &rules),
                "priority scan must match the sim's min-rank selection"
            );
        });
    }

    #[test]
    fn repeated_pairs_keep_the_leftmost_occurrence() {
        // Sim tie case: the same rule matches at pairs 0 and 2; the earlier
        // occurrence wins.
        in_scope(|| {
            let round = round_for(vec!["a", "b", "a", "b"]);
            let rules = vec![merge_rule(0, "a", "b"), merge_rule(1, "b", "a")];
            let chunks = chunked(rules.clone(), 2);
            let scan = run_scan(&round, &chunks);
            let best = scan.best.expect("candidate should be found");
            assert_eq!(best.pair_idx, 0, "ties keep the earlier candidate");
            assert_eq!(best.merge_index, 0);
            assert_eq!(best.merged, "ab");
            assert_eq!(
                Some(best),
                sim_reference_selection(&resolved_pieces(&round), &rules),
                "leftmost-occurrence rule must match the sim's tie handling"
            );
        });
    }

    #[test]
    fn chunk_width_does_not_change_the_selection() {
        let pieces = vec!["a", "b", "a", "c", "a", "b"];
        let rules = vec![
            merge_rule(0, "c", "a"),
            merge_rule(1, "a", "b"),
            merge_rule(2, "b", "a"),
            merge_rule(3, "a", "c"),
        ];
        let reference = sim_reference_selection(
            &pieces.iter().map(ToString::to_string).collect::<Vec<_>>(),
            &rules,
        );
        in_scope(|| {
            for width in 1..=4 {
                let round = round_for(pieces.clone());
                let chunks = chunked(rules.clone(), width);
                let scan = run_scan(&round, &chunks);
                assert_eq!(
                    scan.best, reference,
                    "chunk width {width} must not change the winner"
                );
            }
        });
    }

    #[test]
    fn scan_without_candidates_converges_after_a_full_pass() {
        in_scope(|| {
            let round = round_for(vec!["x", "y"]);
            let chunks = chunked(vec![merge_rule(0, "a", "b"), merge_rule(1, "b", "a")], 1);
            let scan = run_scan(&round, &chunks);
            assert!(scan.best.is_none());
            assert!(!scan.done, "no winner: done stays clear");
            assert_eq!(
                scan.chunks_scanned, 2,
                "a converged round pays one full table pass"
            );
            let decision = finalize_bpe_merge_scan(scan, round);
            assert!(decision.skip, "no selection collapses into the skip flag");
            assert!(decision.merged.is_empty());
        });
    }

    #[test]
    fn skipped_rounds_suppress_the_selection() {
        // A stale candidate must not survive a skipped round (the deferred
        // error and completion cases both arrive as `skip`).
        let mut scan = GemmaBpeScanState::initial();
        scan.best = Some(GemmaBpeScanCandidate {
            pair_idx: 0,
            merge_index: 0,
            merged: "ab".to_string(),
        });
        scan.done = true;
        let decision = in_scope(|| {
            let skipped = GemmaBpeOpenedRound {
                skip: true,
                round: 0,
                piece_count: 0,
                has_error: false,
                error: String::new(),
                pieces_ref: raster::store_internal_value(&staged(vec![])).expect("store pieces"),
            };
            finalize_bpe_merge_scan(scan, skipped)
        });
        assert!(decision.skip);
        assert!(decision.merged.is_empty());
    }

    #[test]
    fn winning_candidates_become_the_apply_decision() {
        let mut scan = GemmaBpeScanState::initial();
        scan.best = Some(GemmaBpeScanCandidate {
            pair_idx: 2,
            merge_index: 7,
            merged: "ab".to_string(),
        });
        scan.done = true;
        let opened = in_scope(|| round_for(vec!["a", "b", "c", "d"]));
        let decision = in_scope(|| finalize_bpe_merge_scan(scan, opened));
        assert!(!decision.skip);
        assert_eq!(decision.merge_piece_idx, 2);
        assert_eq!(decision.merged, "ab");
    }
}
