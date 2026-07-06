# WS3 Port Plan — `prompt.prepare` → raster-core

Per-routine plan following `docs/plans/raster-core-routine-template.md` and the
WS3 agent prompt (Phase A). The sim path
(`src/routines/prompt_prepare/raster/`) is the logical specification; native
(`src/routines/prompt_prepare/native/`) is the output oracle. Companions: the
migration charter, `docs/plans/ws1-dsl-translation-catalog.md` (WS1),
`docs/plans/ws2-staging.md` (WS2).

## Routine Target

- **Routine id:** `prompt.prepare`
- **Example selector:** `--raster-core-at prompt.prepare:1` (the routine runs
  exactly once per inference; occurrence is always 1)
- **Occurrence semantics:** one occurrence per run, counted at the
  `sequence.rs` prompt.prepare decision point (today a
  `reject_if_selected_unsupported` call — the occurrence-counting call site is
  preserved by the new dispatch)
- **Unsupported sub-checkpoints:** none — the routine owns the single
  `prompt.prepare` checkpoint
- **WS1 catalog rows consumed:** C1, C3, C5, C2/C7 (recur tiles), C4/C9 (recur
  sequence; G1 canonical case), C11 (`select!`), C12 (multi-arg ABI, u32
  widths), C13 (tokenizer external + lookup idiom), C14 (root-string sources —
  dissolves), C15 (roots threading — dissolves), C16/C17 (builders/reads — see
  D6), C18 (host staging of pre-DSL values), C20 (integrity modes — no port),
  C22 (H3 reference params), C23 (H4 errors), C25 (zero-arg `main`), C28
  (enum-state branching), C33 (one module per file)

## Gate checks (WS3 prompt §"Before you start")

1. **Catalog readiness.** WS1 §5 marks `prompt.prepare` conditionally ready
   with all three conditions carrying verified shapes: the G1 bounded-loop
   restructuring is the approved catalog ruling (§6), the tokenizer committed
   external is built (WS2, `crates/raster-programs/gemma_externals/`), and the
   C15 roots-leg removal is mechanical. The two rows still `mapped-unverified`
   (C19, C21) are not consumed by this port: the adapter reproduces the
   *native* checkpoint formatter output exactly, so no raster-core commitment
   forms enter payloads (C19/WS5 untouched), and the C21 schema-design work is
   precisely what this plan's types section pins.
2. **Staging readiness.** WS2 §7 lists the Gemma tokenizer external as built
   (consumed by this routine); the remaining inputs are request-scoped
   postcard values needing no encoder. No deferred encoder work lands in
   Phase B.

**Prompt-vs-branch discrepancies (flagged per protocol, branch wins):**

- The WS3 prompt says the guard expects `pub fn run_raster_core(` in
  `{ROUTINE_DIR}/mod.rs`; the branch's guard test
  (`src/routines/mod.rs::raster_core_hosts_honor_the_migration_contract`)
  checks `{ROUTINE_DIR}/raster_core/mod.rs` via `RASTER_CORE_HOSTS` +
  `RASTER_CORE_MIGRATED`. The entrypoint lives in `raster_core/mod.rs`.
- Two authoring constraints not yet recorded in the WS1 catalog were
  confirmed from the pinned `raster` source while drafting this plan; they
  are folded into the design below and amended into the catalog in Phase B
  (see "Catalog amendments"):
  - **A1 — recur tiles and recur sequences are infallible.**
    `raster-macros/src/lib.rs` `ProtocolReturnKind` has no fallible recur
    variants and `validate_recur_tile_shape`/`validate_recur_sequence_shape`
    reject `Result` returns (compile-fail:
    `raster/crates/raster/tests/ui/recur_tile_invalid_return.rs`). Sim's
    fallible recur tiles defer in-loop guard errors through a state `error`
    field; the first fallible plain tile after the loop surfaces them as the
    terminal `Err` (H4 semantics preserved: deterministic message, committed
    outcome).
  - **A2 — recur initial state must be a plain value.** The generated recur
    drivers take `state: impl Into<RecurState<S>>` and only `From<T>` exists
    (`raster/crates/raster/src/input.rs:374`) — an `AuthRef<S>` produced by an
    init tile cannot seed a loop. Porting shape: loop-carried state seeds from
    a struct literal; state derived from staged inputs initializes on the
    first executed iteration inside a tile; heavy read-only context rides
    `args = (…)` (materialized once per recur site), which is the sim's
    trailing-context shape (C7). Conversely,
    `RecurSequenceState<T>: IntoAuthValue<T>` (`input.rs:1501`) makes the
    opaque sequence state a legal tile argument, and
    `From<AuthRef<T>> for RecurSequenceState<T>` (`input.rs:1516`) lets a
    tile's `AuthRef` return re-enter the threaded state — the two halves of
    the round-body idiom used below.

## Real-Raster Authoring Constraints (instantiated)

- Program crate `crates/raster-programs/prompt_prepare/`
  (`raster-program-prompt-prepare`): no_std + alloc lib, std-gated
  `#[sequence] fn main()` bin, `default-features = false` clean (CI-checked).
  Depends on `raster-program-gemma-externals` (schema types only) and
  `raster-program-support` (std, output file).
- One module per file; no tiles/sequences in `lib.rs` or inline `mod` blocks
  (C33).
- All cross-tile types owned serde with fixed-width integers (`u32`); tuples
  replaced by named `Selectable` structs (G3).
- Fallible plain tiles return `raster::exec::Result<T>`; recur tiles are
  infallible (A1).

## 1. Tile map

Sim source: `src/routines/prompt_prepare/raster/tiles.rs` (3 sequences, 11
tiles). Deviations are numbered D1–D9 (register below).

| # | Sim construct (tiles.rs) | Real construct (program crate) | Catalog rows | Deviations |
|---|---|---|---|---|
| S1 | `#[sequence] main(roots, input_roots, &tokenizer)` (:20) | `#[sequence] fn main()` (bin): binds `tokenizer`/`initial_pieces`/`bpe_config` externals, selects `token_lookup` + `merge_lookup`, `call_seq!(tokenize_prompt_pieces, …)`, materializes, writes `output.bin` | C25, C13, C11 | D1 (root-equality check dissolves), D2 (roots legs) |
| S1′ | — (inner body of sim `main`) | `#[sequence] tokenize_prompt_pieces(initial_pieces, config, token_lookup, merge_lookup) -> Result<PromptTokenization>`: budgets tile → outer recur seq → token-id init tile → token-id recur → fallible finalize | C3, C5 | — |
| S2 | `#[sequence(kind = recursive)] merge_bpe_tokenize_prompt` (:67) — until-done BPE round loop | `#[sequence(kind = recur)] merge_bpe_round(input: RecurSequenceInput<u32>, state: RecurSequenceState<GemmaBpeLoopState>, initial_pieces, config, merge_lookup, scan_chunks, apply_chunks)` over the bounded round list (`initial_piece_count − 1`); converged/errored rounds no-op | C4/C9, G1 | D3 (bounded list + no-op continuation), D4 (error-in-state, A1) |
| S3 | `#[sequence] tokenize_bpe_state` (:80) — test-only entry duplicating S1's body | not ported; program tests drive `tokenize_prompt_pieces` natively via `materialize_auth_return(__raster_sequence_auth_…)` | C25 note | D5 |
| T1 | `finalize_bpe_tokenize_prompt` (:121) — trivial `bpe_state.into_output()` | folded into `init_token_id_finalization` (T2') | C1 | D6a |
| T2 | `init_token_id_finalization` (:129) — starts token-ids builder | `init_token_id_finalization(loop_state: GemmaBpeLoopState, initial_pieces, config) -> GemmaTokenIdContext` — resolves final pieces (zero-round case falls back to `initial_pieces`), carries deferred round errors; builder-start dissolves (D6) | C1, C16 | D6 |
| T3 | `finalize_next_token_ids` (recur, :150) — chunked vocab lookups + `append_leaf` | `#[tile(kind = recur)] finalize_next_token_ids(input: RecurInput<u32>, state: RecurState<GemmaTokenIdState>, ctx: GemmaTokenIdContext, token_lookup, config)` — per chunk: binary search `token_lookup` per piece, accumulate ids in state; missing token → state error + `Break` | C2/C7, C13 | D4, D6, D7 (binary search replaces `auth_read`) |
| T4 | `finalize_tokenize_prompt` (:197) — completion check + builder finalize | `finalize_tokenize_prompt(state: GemmaTokenIdState, ctx: GemmaTokenIdContext) -> Result<PromptTokenization>` — surfaces deferred errors, checks cursor == piece_count | C1, C23 | D4 |
| T5 | `finalize_raster_prompt_preparation` (:226) — collects sim-store text roots into checkpoint state | not ported: host-side; the native checkpoint formatter (`format_native_prompt_as_raster_checkpoint_for_trace`) emits the payload | C15, C18, C19 | D8 |
| T6 | `init_bpe_merge_scan` (:258) | `init_bpe_merge_scan(state: GemmaBpeLoopState, initial_pieces, config) -> GemmaBpeRoundContext` — first-round init from `initial_pieces` (A2), skip flag when complete/errored | C1 | D2 |
| T7 | `scan_bpe_merge_candidates` (recur, :285) — chunked pair scan | `#[tile(kind = recur)] scan_bpe_merge_candidates(input: RecurInput<u32>, state: RecurState<GemmaBpeScanState>, round: GemmaBpeRoundContext, merge_lookup)` — chunk of pairs per iteration, binary search `merge_lookup`, keep lowest `merge_index` (ties: earlier candidate), `Break` past `pair_count` | C2/C7, C13 | D4, D7 |
| T8 | `finalize_bpe_merge_scan` (:341) — completion check + merged-token resolve | `finalize_bpe_merge_scan(scan: GemmaBpeScanState, round: GemmaBpeRoundContext) -> GemmaBpeMergeDecision` — merged token taken from the scan candidate (already carried by `merge_lookup` entries); errors deferred in the decision | C1, C13 | D4, D7b |
| T9 | `init_bpe_merge_iteration` (:386) — Complete/Applying + output builder start | `init_bpe_merge_iteration(decision: GemmaBpeMergeDecision, config) -> GemmaBpeIterationContext` — same Complete/Applying semantics via `complete` flag; range validation defers via error field; builder-start dissolves | C1, C28 | D4, D6 |
| T10 | `apply_bpe_merge_chunk_or_complete` (recur, :436) — chunked pieces rebuild via `append_leaf` | `#[tile(kind = recur)] apply_bpe_merge_chunk_or_complete(input: RecurInput<u32>, state: RecurState<GemmaBpeApplyState>, iteration: GemmaBpeIterationContext)` — rebuilds `output` pieces in state per chunk (merge point consumes two inputs), `Break` when complete/skip | C2/C7 | D4, D6 |
| T11 | `finalize_bpe_merge_iteration` (:519) — count check + builder finalize → next round state | `finalize_bpe_merge_iteration(apply: GemmaBpeApplyState, iteration: GemmaBpeIterationContext) -> GemmaBpeLoopState` — count check defers via error field; produces next round state (round+1, complete when no selection) | C1 | D4, D6 |
| — | (no sim analogue; G1/A2 obligation) | `build_chunk_budgets(initial_pieces, config) -> Result<ChunkBudgets>` — fallible plain tile deriving the bounded lists (`rounds`, `scan_chunks`, `apply_chunks`, `token_chunks`) from `initial_pieces.len()` and the chunk widths; validates non-zero chunk widths (hoists the sim's per-tile zero guards) | C1, C27, G1 | D9 |

**Round-body value flow (the A2 idiom):**

```text
merge_bpe_round body (state-only recur sequence):
  round    = call!(init_bpe_merge_scan, state, initial_pieces, config)      // opaque state → tile (IntoAuthValue)
  scan     = call_recur!(tile = scan_bpe_merge_candidates,
                         input = scan_chunks,                               // AuthRef arg from outer sequence
                         state = GemmaBpeScanState literal,                 // plain seed (A2)
                         args  = (round, merge_lookup))                     // heavy context, materialized once
  decision  = call!(finalize_bpe_merge_scan, scan, round)
  iteration = call!(init_bpe_merge_iteration, decision, config)
  apply     = call_recur!(tile = apply_bpe_merge_chunk_or_complete,
                          input = apply_chunks,
                          state = GemmaBpeApplyState literal,
                          args  = (iteration,))
  call!(finalize_bpe_merge_iteration, apply, iteration)                     // AuthRef → RecurSequenceState (From impl)
```

**G1 instantiation.** Outer loop bound: `initial_piece_count − 1` (each merge
removes one piece; empty list ⇒ zero rounds, P1-verified clean finalize).
Inner scan bound: `ceil((initial_piece_count − 1) / pairs_per_tile)` chunk
ordinals — an upper bound for every round since piece count only decreases;
`RecurControl::Break` truncates. Inner apply bound:
`ceil((initial_piece_count − 1) / pieces_per_tile)`. Token-id bound:
`ceil(initial_piece_count / pieces_per_tile)` (final count ≤ initial). All
four lists derived in-program by `build_chunk_budgets` (committed via the
trace, not staged — fewer inputs, bound provably derived from committed
inputs).

## 2. Types plan

Sim `types.rs` → program-crate `types.rs`, postcard-safe owned, fixed-width
(C12), `Selectable` where selected from (C11):

| Sim type | Real type | Notes |
|---|---|---|
| `RasterPromptPreparationResult`, `RasterPromptPreparedInputs`, `RasterPromptInputRoots` | dissolve | host/sim-store shapes (C14/C15); the program's boundary types are the staged inputs + `PromptTokenization` |
| `GemmaBpeState` (piece_count, add_special_tokens, iteration, per-tile widths) | `GemmaBpeLoopState { initialized: bool, complete: bool, round: u32, pieces: Vec<String>, error: Option<String> }` | pieces ride the loop state inline (D6); `add_special_tokens` dropped (D2b — inert in sim tiles); widths live in `BpeConfig` |
| `GemmaBpeTokenizeSequenceState` | `GemmaBpeLoopState` | roots leg deleted (C15) |
| `GemmaBpeScanState` / `GemmaBpeScanTileState` | `GemmaBpeScanState { next_pair_idx: u32, best: Option<GemmaBpeScanCandidate>, done: bool }` + `GemmaBpeRoundContext { skip: bool, round: u32, pieces: Vec<String>, pair_count: u32, error: Option<String> }` | loop cursor stays in recur state; pieces/config move to the context arg (A2) |
| `GemmaBpeScanCandidate { piece_idx, rank, merge_index }` | `GemmaBpeScanCandidate { pair_idx: u32, merge_index: u32, merged: String }` | `merge_index` is the rank (merges ordered by priority in the external); merged token captured at scan time (D7b) |
| `GemmaBpeMergeDecision` / `GemmaBpeMergeSelection` | `GemmaBpeMergeDecision { skip: bool, round: u32, pieces: Vec<String>, selection: Option<GemmaBpeScanCandidate>, error: Option<String> }` | selection folded (merged token already in candidate) |
| `GemmaBpeMergeIterationState` (enum Complete/Applying) + `GemmaBpeApplyState`/`GemmaBpeApplyTileState` | `GemmaBpeIterationContext { complete: bool, round: u32, pieces: Vec<String>, merge_piece_idx: u32, merged: String, error: Option<String> }` + `GemmaBpeApplyState { output: Vec<String>, input_cursor: u32, output_cursor: u32, done: bool }` | C28 enum branching becomes flag fields readable in tiles; apply output accumulates in state (D6) |
| `GemmaTokenIdFinalizeState`/`…TileState` | `GemmaTokenIdContext { pieces: Vec<String>, piece_count: u32, error: Option<String> }` + `GemmaTokenIdState { token_ids: Vec<u32>, next_piece_idx: u32, error: Option<String> }` | ids accumulate in state; builder dissolves (D6) |
| `RasterTokenizationResult { token_ids_root, token_count }` | `PromptTokenization { token_ids: Vec<u32>, token_count: u32 }` | the program output — values, not roots; host mirror struct decodes it (WS2 §9.6) |
| `TokenizePromptInput` | stays host-side (staging derivation) | — |
| — | `BpeConfig { bpe_pairs_per_tile: u32, bpe_pieces_per_tile: u32 }` (staged, `Selectable`) | C27: chunk widths staged by the host from `RasterSizingControls` (defaults 64), one obvious place for WS8 retune |
| — | `ChunkBudgets { rounds: Vec<u32>, scan_chunks: Vec<u32>, apply_chunks: Vec<u32>, token_chunks: Vec<u32> }` (`Selectable`) | budget tile output, fields selected as recur inputs |

Tokenizer-side types (`GemmaTokenizer`, `GemmaTokenIdEntry`,
`GemmaBpeMergeLookupEntry`, …) come from `raster-program-gemma-externals`
unchanged — no schema revision needed (WS2 §7 accepted-risk clause not
triggered).

## 3. I/O inventory

**Boundary ruling (spec-faithful):** the sim's pre-DSL host work stays
host-side. `prepare_raster_prompt_input_roots` (sim `utils.rs:105`) runs
`decode_prompt_bytes → build_gemma4_messages → render_prompt →
normalize_tokenize_prompt → split_tokenize_prompt → initial_bpe_pieces`
*before* sim `main`; the port keeps exactly that split — the adapter derives
the initial BPE pieces with the same shared utils (they are
`pub(in super::super)`, visible to `raster_core/`; their tokenizer reads are
in-memory `AuthRead` calls, no sim store involved) and stages them as a
committed input, mirroring the sim's pre-staged `bpe-pieces-0` artifact
(C18: value from native state ⇒ committed external).

**Call site:** `src/runtime/sequence.rs` prompt.prepare section (today
`policy.reject_if_selected_unsupported(RoutineId::PromptPrepare)`). Native
state in hand: `request` (prompt bytes, decoding policy, generation-prompt +
special-token flags, sampling), `model.model_spec()` (chat template,
tokenizer path), `model.raster_tokenizer()` (required — same capability rule
as the deterministic checkpoint), `raster_sizing_controls` (validated Some
when a detour is active).

**Values that must not be recomputed natively:** `prompt_token_ids` — they
come from ingestion only. (`prompt_text` is an *input* to the routine,
host-decoded on both legs; the checkpoint sha is derived from the ingested
ids via `build_prompt_commitment`.)

**Committed input set** (`input.json` + `input_manifest.json` via
`StagedInputs`):

| Logical name | Encoding | Content | Commitment |
|---|---|---|---|
| `tokenizer` | raster (mmap) | `GemmaTokenizer` external from the WS2 content-addressed cache (`gemma_externals` encoder, cache root `$TMPDIR/raster-inference-gemma-external-cache`, reused across runs; encoder invoked as a subprocess — `cargo run -p raster-program-gemma-externals --features encode --bin encode` — because `raster::write_raster_files` must not link into the main crate, charter invariant 6) | raster index root from `root_commitment.txt` |
| `initial_pieces` | postcard | `Vec<String>` — host-derived initial BPE pieces | sha256 of payload |
| `bpe_config` | postcard | `BpeConfig` from `RasterSizingControls` (tokenizer widths, defaults 64) | sha256 of payload |

**Outputs ingested:** `output.bin` = `postcard(Result<PromptTokenization,
String>)`; host mirror struct `PromptTokenization { token_ids: Vec<u32>,
token_count: u32 }` field-order-matched (WS2 §9.6). Plus the opaque
`commit.bin` fingerprint and the input-commitment map on
`RasterCoreRunResult` (WS3 exit criterion 3).

**Run directory:** `RasterCoreRunDir::create(RoutineId::PromptPrepare, 1)`;
kept on failure, deleted after successful ingestion.

## 4. Checkpoint plan

- **Committed checkpoint:** `prompt.prepare` (occurrence 1). On the native
  leg with `raster_tokenizer_enabled` (the parity-run configuration),
  `native::run_prompt_prepare` (a) calls
  `format_native_prompt_as_raster_checkpoint_for_trace` — the formatter that
  defines the payload environment and populates the sim store consumed by
  downstream checkpoints (`input.embedding`) — and (b) commits the value-form
  payload `{prompt_text, prompt_token_ids, prompt_token_ids_sha256,
  sampling}`.
- **Adapter rule (backend invariance made concrete):** `run_raster_core`
  returns a native-form `PromptPreparationState` materialized from ingestion
  (`prompt_text` host-decoded, `prompt_token_ids` from `output.bin`,
  `prompt_token_ids_sha256` via `build_prompt_commitment`). The *same*
  `native::run_prompt_prepare` code path then runs the formatter and commits
  the checkpoint from that state — the detour leg swaps only the
  `prompt_prepare::run(…)` call. Payload identity with full native holds by
  construction; no artifact roots, tile counts, run-dir paths, or backend
  labels are added or dropped.
- **WS5 boundary-checkpoint exception:** untouched. The value-form vs
  root-form question stays exactly where the native path put it; this port
  introduces no raster-core commitment forms into payloads (C19 not
  consumed).

**Decision-point wiring.** Replace
`policy.reject_if_selected_unsupported(RoutineId::PromptPrepare)` in
`sequence.rs` with `policy.mode_for(RoutineId::PromptPrepare)` (same
occurrence-counting call site):

- `StepMode::Native` → `native::run_prompt_prepare(…, raster_core_detour: false)`
  — byte-identical behavior;
- `StepMode::RasterCore` → same function with the detour flag on, which calls
  `prompt_prepare::raster_core::run_raster_core(request, model_spec,
  tokenizer_source, raster_sizing)` in place of `prompt_prepare::run`;
- `StepMode::Raster` (a *sim* spec selecting prompt.prepare) → the same
  "not implemented yet" error as today (`unimplemented_detour_error` words
  per backend — message unchanged).

**Error contract (H4/C23):** launch failures →
`RasterCoreError::Infrastructure`; `ingest` classifies the rest. The adapter
propagates the three classes as distinct `anyhow` errors (Display preserved);
a `Terminal` outcome is a committed, fault-provable result whose message must
be deterministic — all in-program error strings are format-stable (no paths,
no addresses).

## 5. Test plan

Sim `tests.rs` (14 tests) disposition:

| Sim test | Port |
|---|---|
| `decode_prompt_bytes_*` (2), `build_gemma4_messages_*`, `render_prompt_*`, `init_tokenize_prompt_*`, `build_prompt_commitment_*` | unchanged — they exercise native/host fns the port reuses verbatim |
| `tokenize_prompt_applies_recursive_bpe_merges` ("ab" → [3]) | program-crate test: drive `tokenize_prompt_pieces` natively (`materialize_auth_return(__raster_sequence_auth_…)`) with a mini token/merge table mirroring the sim fixture |
| `tokenize_prompt_uses_byte_fallback_for_unknown_chars` ("é" → [10, 11]) | split: host-side unit test asserting the staged initial-piece derivation for "é" (byte-fallback pieces), program-crate test tokenizing those pieces |
| `tokenize_prompt_chunk_sizes_do_not_change_results` | program-crate test: widths (1,1) vs (8,8) produce identical ids ("aba" → [12]) |
| `finalize_tokenize_prompt_returns_token_id_root` | program-crate test on the token-id loop path (ids [1, 3]); root assertions are sim-only |
| `init_bpe_tokenize_prompt_returns_compact_ref_state`, `run_returns_root_backed_prompt_state` | sim-only (assert sim-store roots/serialized-ref compactness — the mechanism C15 dissolves); their end-to-end intent is covered by the dev-run trace identity |

Added coverage:

- program-crate: missing-vocab piece → committed terminal `Err` (message
  parity with the sim's "missing from vocab" guard); zero chunk width →
  terminal `Err` from `build_chunk_budgets`; single-piece and empty-pieces
  edge cases (zero-round loop, A2 first-iteration init).
- host-side: adapter error mapping (missing `cargo-raster` →
  `Infrastructure`), initial-piece derivation determinism.
- **Dev-run verification (WS3 finish line):**
  `tests/raster_core_prompt_prepare_detour.rs`, CI-gated like the WS2 tests
  (skips loudly without `cargo-raster`; `REQUIRE_CARGO_RASTER=1` in CI): full
  native vs `--raster-core-at prompt.prepare:1` (via
  `RasterDetourSpec::parse_raster_core`) on tiny-gemma-dev, committed traces
  identical with divergence named by checkpoint id + occurrence, final
  outputs equal. Comparison helpers mirror
  `tests/raster_core_detour_parity.rs` so WS4 lifts the leg by appending
  `("prompt.prepare", 1)` to `ENABLED_ROUTINES`. **The parity-harness leg is
  not flipped in WS3.**

## 6. Commit sequence (Phase B)

1. `ws3(prompt.prepare): program-crate types and chunk-budget tile` — types
   module + `build_chunk_budgets` + unit tests beside the WS0 placeholder;
   no_std surface stays green.
2. `ws3(prompt.prepare): BPE merge-round tiles and recur sequence` — scan /
   decision / iteration / apply tiles, `merge_bpe_round`, program tests
   (includes the A1/A2 idioms' first in-tree exercise).
3. `ws3(prompt.prepare): token-id tiles, routine sequence, and program main`
   — token-id loop, `tokenize_prompt_pieces`, new `main` with the WS2 output
   idiom; placeholder tile removed; remaining program tests.
4. `ws3(prompt.prepare): run_raster_core host adapter and detour wiring` —
   staging (tokenizer cache + postcard inputs), runner, ingest, native-state
   materialization; `sequence.rs`/`native.rs` decision point; guard flip
   (`RASTER_CORE_MIGRATED += "prompt_prepare"`); host unit tests.
5. `ws3(prompt.prepare): dev-run trace-identity verification` — the CI-gated
   comparison test + final-output equality.
6. `ws3(prompt.prepare): port notes and status updates` — deviation register
   finalized here; WS1 catalog amendments (A1, A2 + C2/C4/C7/C9/C23 notes,
   amendment log); WS2 §10 note (no schema revision); charter tracker →
   `WS3`; WS4 handoff note.

## Deviation register

| # | Deviation | Reason |
|---|---|---|
| D1 | Sim `main`'s tokenizer root-equality guard (tiles.rs:26–33) not ported | C14: identity is enforced mechanically by the runtime's manifest commitment check (P3 tamper rejection); the root string no longer exists |
| D2 | All `RasterArtifactStoreRoots` legs deleted from signatures | C15 (verified): real internal storage commits reads/writes in the trace; no user-visible snapshot |
| D2b | `add_special_tokens` not threaded through program state | inert in sim tiles (never read after staging); consumed host-side during initial-piece derivation |
| D3 | Until-done BPE round loop → bounded round list (`initial_piece_count − 1`) with no-op continuation after convergence | G1 approved ruling; recur sequences cannot `Break` (C4) |
| D4 | Fallible recur tiles → infallible recur tiles with deferred `error: Option<String>` state fields, surfaced by the next fallible plain tile | A1: recur returns cannot be `Result` at the pinned rev; H4 preserved (terminal `Err` still committed, deterministic message) |
| D5 | `tokenize_bpe_state` sequence not ported | test-only entry; program tests drive the parameterized routine sequence natively (C25 idiom) |
| D6 | Builder/read ladder (`bpe-pieces-N` artifacts, token-ids builder) → pieces and ids carried inline in loop state/args | C15/C16 note: the sim's pair of (small state + store artifacts) collapses to state once roots dissolve; every value still rides the trace (state serialized per iteration, args materialized per site). `Draft` reserved for select-from-finalized needs, which this routine no longer has — its sole output is the materialized token-id vector. Trace-size cost noted for WS8 (`[MEASUREMENT-PENDING]`) |
| D6a | `finalize_bpe_tokenize_prompt` folded into `init_token_id_finalization` | the sim tile was a trivial state-to-output cast whose output type dissolves with D6 |
| D7 | `auth_read(tokenizer, …Request)` → in-tile binary search over selected `token_lookup` / `merge_lookup` | C13 idiom (a), the tokenizer-PoC shape; per-source choice assigned to this plan by the catalog |
| D7b | Merged token taken from the scan candidate instead of a separate `GemmaBpeMergedTokenRequest` read | the external's `merge_lookup` entry carries `merged_token`; a second dynamic-index read would need select-by-computed-index for data already commitment-checked in the same external |
| D8 | `finalize_raster_prompt_preparation` not ported | its output is the checkpoint payload, which is host-side by the backend-invariance rule (§4); the native formatter emits it on both legs |
| D9 | New `build_chunk_budgets` tile (no sim analogue) | G1 bound-list obligation + hoisted zero-width guards (sim's per-tile `bail!`s), kept fallible in one place |

## Measurements

- Tile sizing, per-iteration trace cost of inline piece state, cycle counts:
  `[MEASUREMENT-PENDING]` until WS8 profiling on the migrated path. Chunk
  widths stage from `RasterSizingControls` defaults (64) — one obvious knob
  for WS8 retuning.

## References

- `src/routines/prompt_prepare/raster/` — the spec
- `docs/plans/ws1-dsl-translation-catalog.md`, `docs/plans/ws2-staging.md`
- `docs/plans/2026-07-05-001-raster-core-migration-adr.md`
- `crates/raster-programs/_probes/` — verified construct examples
- `crates/raster-programs/gemma_externals/` — tokenizer schema + encoder
- `crates/raster-programs/_roundtrip/` — output-file idiom
- `raster-tokenizer` — authoring idiom only
