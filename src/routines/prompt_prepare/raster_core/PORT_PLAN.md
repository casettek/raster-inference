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

Tokenizer-side types (`GemmaTokenizer`, `GemmaTokenIdEntry`, …) come from
`raster-program-gemma-externals`. *(Phase A shipped the schema unrevised;
the storage refactor later triggered the WS2 §7 accepted-risk clause — see
"Storage refactor" below and deviations D10/D11.)*

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
| D6 | **Re-founded by the storage-resident refactor (2026-07-07; original rationale superseded by D15/D16).** The single surviving inline crossing is the loop-carried pieces in `GemmaBpeLoopState` — retained because the pinned rev's recur-sequence state threading is **storage-backed between iterations**: the round-finalize tile's returned state persists in internal storage via the tile-output store (`bind_infallible_call` → `store_execution_output_value`, `raster/src/lib.rs:393-409`) and re-enters the threaded state through an authenticated resolve (`From<AuthRef<T>> for RecurSequenceState<T>` → `resolve_internal_value`, which validates the coordinates lookup, stored-vs-reference commitment equality, and a recomputed integrity commitment over the stored bytes — `raster-runtime/src/internal_storage.rs:350-414`; tampered commitments are rejected). Measured trace form at the boundary (probe P1, `chunk_probes.rs`): the driver holds the resolved state in memory across the iteration boundary and each iteration's `RecurSequenceStart` records it as `FnInputValue::Inline` — full postcard bytes, no binding metadata (`raster-macros/src/lib.rs:550-559`) — so per-iteration record size scales linearly with the carried pieces (a 64×32-byte seed inflates the state record to ≥2,048 bytes; asserted by `p1_state_traces_inline_per_iteration_and_scales_with_pieces`). The same inline form covers the state's entry into the round's first tile (`open_round`), since `RecurSequenceState<T>: IntoAuthValue<T>` produces an inline auth value. This is a documented, justified exception on the trace-form side, not a waiver of storage residency; every other pieces crossing moved to drafts, selections, and bindings (D15/D16) |
| D6a | `finalize_bpe_tokenize_prompt` folded into `init_token_id_finalization` | the sim tile was a trivial state-to-output cast whose output type dissolves with D6 |
| D7 | `auth_read(tokenizer, …Request)` → in-tile binary search over selected `token_lookup` / `merge_lookup` | **superseded by D10/D11** (storage refactor): whole-table selection dragged the model-scoped tables through recur `args`, re-materializing ~262k entries per chunk iteration |
| D7b | Merged token taken from the scan candidate instead of a separate `GemmaBpeMergedTokenRequest` read | **superseded by D11** (the merged token now comes from the winning `GemmaBpeMerge` rule itself) |
| D8 | `finalize_raster_prompt_preparation` not ported | its output is the checkpoint payload, which is host-side by the backend-invariance rule (§4); the native formatter emits it on both legs |
| D9 | New `build_chunk_budgets` tile (no sim analogue) | G1 bound-list obligation + hoisted zero-width guards (sim's per-tile `bail!`s), kept fallible in one place. Storage refactor: only `rounds` and `apply_chunks` remain — the chunked model tables and the final pieces list are their own bounded recur inputs. **Storage-resident refactor (2026-07-07):** only `rounds` remains and the tile is infallible — the apply loop recurs over the round's own pieces, so the per-tile widths, `BpeConfig`, the `bpe_config` staged input, and the zero-width guards (with their sim-parity messages) are deleted end to end; the piece count comes from the new one-shot `count_pieces` authenticated read of the staged `BpePieces` external |
| D10 | **Chunked-external idiom (storage refactor, supersedes D7's data placement).** Model-scoped tables enter tiles *only* as pre-chunked committed-external recur input lists: `token_lookup_chunks` / `merge_chunks` (`Vec<Vec<Entry>>`, encode-time width 1024), one chunk (~30–50KB) per tile execution — never materialized tile `args`, never loop state | ZKVM sizing for real Gemma: the generated recur drivers materialize `args` inside the per-iteration closure, so any table in `args` re-enters the tile ABI/trace on every iteration. Recur *input* items are per-iteration selections instead. WS1 C13/C22 amended with this rule. **Loop mechanism superseded by D13** (recur-tile inputs still traced inline per iteration; chunk loops became recur sequences) — the chunked-external schema and the "never args/state" rule stand |
| D11 | **Priority-order scan (supersedes D7/D7b).** Per round, the scan recurs over `merge_chunks` in priority order; the first rule with an adjacent-pair occurrence in the round's pieces wins (lowest `merge_index` globally, leftmost pair) and breaks. The pair-keyed `merge_lookup` table is deleted from the external schema | equivalent to the sim's min-rank/earliest-pair selection (unit-proven against the sim fixture cases); dissolves the only consumer of `merge_lookup`, halving the external's derived data. A converged round pays one full table pass; the `complete` flag makes later rounds break on the first chunk |
| D12 | **Draft-accumulated token ids (amends D6).** Token-id resolution is a nested loop — outer recur *sequence* over the final pieces (its own natural bound), inner recur *tile* over `token_lookup_chunks` (`Break` on match or past the sorted position) — with ids/misses accumulated in a `TokenIdsBundle` draft output, not loop state | **superseded by D14** (trace-slimming refactor): the recur-*tile* inner loop still inlined each vocab chunk per iteration (G7), and the per-piece nesting paid `pieces × chunks` iterations. D6's original rationale stands; the mechanism moved |
| D13 | **Chunk loops are recur sequences (trace-slimming refactor, supersedes D10's recur-tile inputs).** Model-scoped chunk loops are `#[sequence(kind = recur)]` over the chunked table; each iteration passes the opaque item handle (`RecurSequenceInput<Vec<Entry>>`) plus the threaded state into one plain tile, so the chunk crosses the tile ABI as an external-selection binding (~100B commitment + selector) and materializes only at tile execution | recur-*tile* drivers trace each iteration's materialized input *and* re-materialized `args` inline (`RecurTileIterationExec` — gap G7): 86KB merge chunks / 34–50KB vocab chunks per iteration, 94% of the real-model trace. Plain-tile `AuthRef` args already trace as bindings; recur-sequence item handles are selection `AuthRef`s — combining the two is the fix. Cost: no `Break` in recur sequences (G1), so every round pays a full no-op pass after the winner (accepted; WS8 chunk-width retuning + the G7 upstream request are the mitigations) |
| D14 | **Inverted single vocab pass (supersedes D12).** One recur sequence over `token_lookup_chunks`; per chunk, `resolve_pieces_in_vocab_chunk` binary-searches every still-unresolved piece against the chunk, filling a per-piece slot vector (`GemmaTokenResolutionState { initialized, resolved: Vec<Option<u32>> }` — prompt-scoped, seeds from a literal per A2, sizes itself from the context on the first iteration; no-ops once all pieces resolve). The per-piece machinery (`resolve_piece_token_ids`, `open_piece_lookup`, `lookup_piece_token_id`, `record_piece_token_id`, `TokenIdsBundle`, `GemmaVocabLookupState`) is deleted; `GemmaTokenIdContext` is no longer `Selectable` (nothing selects its pieces) and its deferred error reverts to `Option<String>` | one full pass over the vocab chunks for the whole prompt (256 iterations) replaces `pieces × chunks` nested iterations, and every chunk crosses as a selection binding (D13 rule). `finalize_tokenize_prompt` surfaces the first unresolved slot with the sim's exact missing-vocab message via `ctx.pieces[i]` (H4/A1 preserved). **State mechanism superseded by D16** (append-only draft; single-pass orientation stands) |
| D15 | **Draft-accumulated round pieces (storage-resident refactor, supersedes the apply legs of D6/D9).** Each round's next pieces accumulate in a fresh `RecurOutput<BpePieces>` draft — the real-raster form of the sim's `bpe-pieces-{N+1}` builder — through a recur *sequence* over the round's own pieces (a `select!` projection of `open_round`'s output; not a recur tile — P4/G7). The per-iteration plain tile (`apply_one_piece`) threads cursor-only state `{skip_next, emitted}` plus the draft through a single `(RecurState<S>, RecurOutput<O>)` return; the piece arrives through the input handle, the decision scalars (`GemmaBpeApplyDecision { skip, merge_piece_idx, merged }`) as an internal binding. `open_round` republishes the round's pieces behind the selectable `BpePieces` root (the deferred error flattens to `(has_error, error)` scalars — `Option` has no `Selectable` schema, G3); `finalize_round` reconstructs the `Option<String>` convention, keeps the sim's range and count checks with their exact messages (H4), and carries the incoming pieces forward unchanged on skip/no-selection rounds — never the empty draft. The pieces-carrying `GemmaBpeRoundContext`/`GemmaBpeMergeDecision`/`GemmaBpeIterationContext`/`GemmaBpeApplyState` structs are deleted; `apply_chunks` and the per-tile widths die with them | the storage-residency invariant: prompt-derived collections persist in raster storage (staged external, tile-output store, finalized drafts) and cross the tile ABI only as authenticated reads — draft ops, input-handle selections, `select!` projections, selection-bound args — with the single P1-characterized loop-state exception (D6). Probe P2 pinned the exact driver surface (fresh draft created and finalized inside a recur-sequence body iteration over a `select!` projection input list; finalized ref consumed via `select!` and as a binding arg; re-entry into the outer threaded state) |
| D16 | **Append-only token-id matches (supersedes D14's state mechanism, keeps its single-pass orientation).** The vocab pass is an output-only recur sequence appending `TokenIdMatch { piece_idx, token_id }` entries into a `RecurOutput<TokenIdMatches>` draft per chunk — no threaded resolution state. `init_token_id_finalization` becomes the first fallible plain tile after the BPE loop (A1): it surfaces the deferred loop error and republishes the final pieces behind the selectable `BpePieces` root; the per-chunk tile takes the chunk via the input handle, the final pieces as a selection-bound arg (repeated authenticated reads accepted — inefficiency over cleverness), and the draft. The terminal `finalize_tokenize_prompt(matches, final_pieces)` materializes both at the program boundary (the only place ordered token ids materialize for the host), orders by `piece_idx`, errors on conflicting duplicate matches (defensive — the sorted vocab resolves each piece in exactly one chunk; new deterministic message `token-id finalization found conflicting ids for piece {idx}`) and on the first unresolved piece with the sim's exact missing-vocab message. `GemmaTokenIdContext` and `GemmaTokenResolutionState` dissolve; the structurally unreachable `token-id finalization stopped at piece …` completion check dies with them | no prompt-derived collection threads through recur state in the token phase; the matches persist in the draft (internal storage) and the ids materialize once. The deferred BPE-loop error now surfaces *before* the vocab pass instead of after it — identical committed `Err` string, program-internal trace order differs (behavioral gate unaffected) |
| D17 | **Whole-pairs authenticated read in the scan (permitted rule-2 crossing).** `build_pairs` derives the round's adjacent-pair list (`GemmaBpeAdjacentPairs`) as its own internal ref from the `select!`-ed round pieces; the per-chunk scan tile takes it as a selection-bound arg and materializes the full prompt-scoped pair set at execution, matching rules by leftmost occurrence in the pair list (D11's priority orientation stands). A separate tile rather than a fold into `open_round`: keeps the opened round free of derived data and each tile at one job | gap **G6** (computed-key selection) is the named unlock for the sim's keyed-lookup orientation — with it, the scan could select individual pairs by computed index instead of materializing the set; until then the whole-pairs binding read is the pinned-rev shape (an authenticated read, traced as a binding, prompt-bounded) |

## Measurements

- Tile sizing, per-iteration trace cost of inline piece state, cycle counts:
  `[MEASUREMENT-PENDING]` until WS8 profiling on the migrated path. Chunk
  widths stage from `RasterSizingControls` defaults (64) — one obvious knob
  for WS8 retuning.
- Post-refactor cost model (trace-slimming refactor, D13/D14 — see the
  dated section below): per tile execution = one table chunk (~30–50KB,
  materialized at execution from its selection binding) + KB-scale prompt
  data. Iteration counts on real Gemma: scan = rounds × 503 merge chunks
  (full pass — recur sequences cannot break, G1); token ids = one
  256-chunk vocab pass for the whole prompt (D14 inversion). Measured
  real-model numbers (prompt "Hello from Raster"): 35 rounds, 53,907
  trace items, max inline value 102B, program wall time ≈ 1 minute — see
  "Trace-slimming ref-based refactor (2026-07-06)".

## References

- `src/routines/prompt_prepare/raster/` — the spec
- `docs/plans/ws1-dsl-translation-catalog.md`, `docs/plans/ws2-staging.md`
- `docs/plans/2026-07-05-001-raster-core-migration-adr.md`
- `crates/raster-programs/_probes/` — verified construct examples
- `crates/raster-programs/gemma_externals/` — tokenizer schema + encoder
- `crates/raster-programs/_roundtrip/` — output-file idiom
- `raster-tokenizer` — authoring idiom only

---

## Port notes (Phase B outcome, 2026-07-06)

Landed as planned; the deviation register at Phase B close was D1–D9.
Catalog amendments A1/A2 were folded into WS1 rows C2/C4/C7 with a dated §9
entry. The tokenizer external schema shipped unrevised at this point (the
storage refactor later revised it to v2 — see "Storage refactor" below and
deviations D10–D12, which supersede D7/D7b and amend D6/D9; the tile-map
and types tables above describe the Phase B shape).

**Verification results:**

- Program crate: 26 unit tests green; builds under the pinned toolchain
  including the `--no-default-features` no_std surface (CI-checked, plus a
  `cargo test -p raster-program-prompt-prepare` CI step).
- Dev run (`tests/raster_core_prompt_prepare_detour.rs`, CI-gated with
  `REQUIRE_CARGO_RASTER=1`): full native vs raster-core detour at
  `prompt.prepare:1` on tiny-gemma-dev — committed checkpoint traces
  identical (11 committed checkpoints, `SHORT_PROMPT` with 2 decode steps),
  final inference states equal. Staged-input commitments (tokenizer raster
  root, `initial_pieces`/`bpe_config` sha256) are captured on the run result
  by `StagedInputs::write` → `ingest` (exit criterion 3).
- Full suite + parity gate + goldens green at every commit; goldens
  byte-identical to baseline (never regenerated).

**Things the next routine's port should know:**

1. **A1/A2 shape rules dominate the tile map.** Decide early which loop
   state seeds from a literal vs which context rides `args`; fallibility
   lives only in plain tiles, so plan one "surface deferred errors" tile per
   loop chain (here: `finalize_tokenize_prompt`).
2. **`call!` marker resolution needs glob imports** of every module that
   defines a called tile/sequence (`use crate::<mod>::*;`) — same
   convention as the probe crate; named imports fail to resolve the hidden
   `__RasterTileCallBinding_*` types.
3. **Reused bindings must be cloned.** `call_recur!`'s `args = (…)` moves
   its bindings; `round.clone()` / `iteration.clone()` where a context
   feeds both the loop and the follow-up tile.
4. **Tile fns need an active sequence scope even when driven natively** —
   unit tests wrap direct tile calls in
   `raster::__private::SequenceScopeGuard::enter(…)`.
5. **The unimplemented-detour reject test** in `tests/inference_sequence.rs`
   now skips migrated routines
   (`run_inference_rejects_raster_core_detours_for_unmigrated_routines`);
   each WS3 port adds its routine to that skip in the same change that wires
   its decision point.
6. **Program-crate run artifacts need a crate-local `.gitignore`**
   (`/target`) before the first `cargo raster run` — the root `.gitignore`
   only covers the workspace `target/`.
7. **Encoder-cache helper:** `output.finalize` consumes the same tokenizer
   external; lift `encode_tokenizer_external_cached` from this routine's
   `raster_core/mod.rs` into `src/runtime/raster_core/` when that port
   starts (WS2 §11 note).

---

## Storage refactor (2026-07-06)

Restructured after the Phase B landing (stage stays WS3): the original
shape selected the whole `token_lookup` (~262k entries) and `merge_lookup`
into recur-tile `args`, and the generated recur drivers materialize `args`
inside the per-iteration closure — every chunk iteration re-deserialized
the full table through the tile ABI/trace, violating the ZKVM-sizing goal
for real Gemma. Deviations D10–D12 (register above) supersede D7/D7b and
amend D6/D9. Behavior is unchanged; the dev-run trace-identity gate
(`tests/raster_core_prompt_prepare_detour.rs`) is the acceptance criterion
and stayed green.

**Data-placement rules (this routine's design contract):**

- **Model-scoped tables** (vocab, merges): committed external,
  **pre-chunked** at encode time (`Vec<Vec<Entry>>`, width 1024 — WS8
  retunes by re-encoding + cache-kind bump) — consumed *only* as recur
  input lists, one chunk (~30–50KB) per tile execution. Never in `args`,
  never in state.
- **Prompt-scoped data** (pieces, pairs, token ids): loop state, small
  args, or drafts — bounded by the prompt, not the model.
- **Cursors/flags/candidates**: tiny recur state structs.

**What moved:**

- `gemma_externals` schema v2: `token_lookup` → `token_lookup_chunks`,
  `merges` → `merge_chunks`, `merge_lookup` deleted (WS2 §7 accepted-risk
  clause triggered; determinism + tamper legs re-run green). Cache kind
  bumped `gemma-tokenizer` → `gemma-tokenizer-v2` in the encoder and the
  host adapter, so revised encodings never collide with stale entries.
- Scan phase: priority-order scan over `merge_chunks` (D11);
  `find_merge` binary search and the merge-lookup arg plumbing deleted;
  equivalence to the sim's min-rank/earliest-pair selection unit-proven
  (`bpe_scan.rs` tests, including tie and chunk-width-invariance cases).
- Token-id phase: nested loop with `TokenIdsBundle` draft output (D12);
  `GemmaTokenIdState` deleted; `GemmaTokenIdContext` became `Selectable`
  (its `pieces` list is the outer recur-sequence input), which forced the
  deferred error into `has_error: bool` + `error: String` — `Option` is
  not `Selectable`.
- `build_chunk_budgets` shrank to `rounds` + `apply_chunks` (D9 note); the
  apply phase kept its shape (prompt-scoped only; pieces still cross
  rounds once per round in loop state — O(prompt), flagged for WS8).
- Step-0 mechanics probes live as a test-only module
  (`chunk_probes.rs`): nested `Vec<Vec<T>>` selection (whole list, one
  chunk, nested entry) and chunked recur input with mid-list `Break`.
- Native-test staging idiom: recur input lists must be selectable
  external/*internal* sources — program tests store the chunked tables
  via `store_internal_value` + `internal!` under a scope guard (an inline
  `Vec` fails with "call_recur! requires a selectable external or
  internal list source").

**Catalog/G-register follow-through:** WS1 C13/C22 porting rules amended
("model-scoped data enters tiles only as chunked recur input lists"); gap
G6 filed as an *optimization* request (dynamic select-by-computed-index —
linear chunk scans are the always-available fallback, so G6 never
blocks).

*(The chunk-loop mechanism of this refactor was superseded on the same day
by the trace-slimming ref-based refactor below — D13/D14. The schema-v2
chunked external, the cache-kind bump, and the data-placement rules all
stand.)*

**WS4 handoff.** `prompt.prepare` is ready for the parity-leg flip: append
`("prompt.prepare", 1)` to `ENABLED_ROUTINES` in
`tests/raster_core_detour_parity.rs`. The dev-run comparison the harness
should absorb is `tests/raster_core_prompt_prepare_detour.rs` — it already
mirrors the harness's trace-capture and identity-assertion helpers
(divergence named by checkpoint id + occurrence) and adds final-output
equality; once the harness leg is on, the standalone test can be folded in
or retired at WS4's discretion.

---

## Trace-slimming ref-based refactor (2026-07-06)

Second restructuring pass after the storage refactor (stage stays WS3):
the storage refactor moved the model tables into recur *inputs*, but the
`#[tile(kind = recur)]` driver still materialized each chunk into
`RecurInput<T>` and traced it **inline per iteration**
(`RecurTileIterationExec` — gap G7), along with re-materialized args.
Deviations D13/D14 (register above) supersede D10's loop mechanism and
D12. Behavior is unchanged; the dev-run trace-identity gate
(`tests/raster_core_prompt_prepare_detour.rs`) is the acceptance
criterion and stayed green, as did the full suite and the no_std surface.

**Rule this refactor enforces:** data crosses the tile ABI as a storage
ref — a committed-external selection or an internal-storage binding —
and only cursors/flags/prompt-scoped values ride inline. Model chunks
reach per-chunk plain tiles as external-selection bindings via
recur-sequence item handles (D13); cross-tile contexts (round, decision,
iteration) are internal-storage-backed `AuthRef` tile outputs, traced as
`InternalBinding`.

**Step-0 probes** (extended `chunk_probes.rs`): (3) a recur sequence over
a chunked list whose body passes both the item handle and the threaded
state into one plain tile, the tile's `AuthRef` return re-entering the
state; (4) that recur sequence nested inside another recur-sequence
iteration (the round-loop shape). Both green before the phases were
rewritten.

**Measurements (real Gemma `~/models/gemma-4-E4B-it`, prompt "Hello from
Raster", 36 initial pieces → 35 rounds; 503 merge chunks / 256 vocab
chunks at encode width 1024; manual
`detour --raster-core-at prompt.prepare`, warm tokenizer cache):**

| | before (recur-tile chunks) | after (ref-based) |
|---|---|---|
| events | 1,646 | 53,907 |
| largest inline value in any event | 86KB merge chunk (`RecurTileIterationExec`, 94% of trace bytes) | **102B** (apply-phase cursors); `scan_one_merge_chunk` max inline = 18B state |
| model chunks in trace | inline per iteration | `ExternalBinding` only (commitment + selector) |
| cross-tile contexts | inline per iteration (re-materialized args) | `InternalBinding` only |
| raw `trace.ndjson` | 211MB | 1.93GB |

The raw-ndjson growth is expected and accepted: each trace item's
`input.data` carries the materialized input witness for that step's
commitment (the runtime hashes `input.data` per step, and the committed
fingerprint projects each item to ~1 bit), so raw file size is a
disk-side artifact of witness capture — not proof-size, and not data
crossing the ABI. The item count grew because recur sequences cannot
`Break` (G1): every round pays a full 503-chunk no-op pass after the
winner (17,605 scan iterations vs 1,297 break-truncated ones), while the
inverted vocab pass (D14) *shrank* its side from `pieces × chunks` to one
256-chunk pass. Program wall time ≈ 1 minute. Mitigations for the
full-pass cost stay as filed: WS8 chunk-width retuning and the G7
upstream request (selection-binding recur-tile inputs would also restore
`Break`).

**Catalog/G-register follow-through:** WS1 C13/C22 amendment refined —
chunked lists are consumed via **recur sequences** whose per-chunk plain
tiles receive selection refs; new gap **G7** (recur-tile drivers inline
per-iteration inputs/args in the trace) filed with the upstream request
"trace recur-tile inputs as selection bindings"; G6 unchanged. WS2
untouched (no schema/encoder change; cache kind stays
`gemma-tokenizer-v2`).

---

## Storage-resident refactor (2026-07-07)

Third restructuring pass (stage stays WS3; pinned rev unchanged — the
program adapts to it). The trace-slimming refactor moved model-scoped
chunks and cross-tile contexts to bindings, but prompt-derived collections
still accumulated in recur state (`GemmaBpeApplyState.output`,
`GemmaTokenResolutionState.resolved`) and rode context structs into
per-chunk/per-item tiles (`GemmaBpeRoundContext.pieces`,
`GemmaBpeMergeDecision.pieces`, `GemmaBpeIterationContext.pieces`,
`GemmaTokenIdContext.pieces`). This refactor ports the sim's ownership
model — pieces behind artifact roots, per-leaf reads, builder appends —
onto real raster storage. Deviations D15–D17 (register above) supersede
the apply/token legs of D6/D9/D14; D6 is re-founded on the probe-P1
findings. Behavior is unchanged; the dev-run trace-identity gate
(`tests/raster_core_prompt_prepare_detour.rs`) stayed green at every
commit, as did the full suite and the no_std surface.

**The invariant this refactor enforces:** prompt-derived data persists in
raster storage across every tile boundary and every recur iteration —
staged committed externals, and internal storage populated by tile outputs
and finalized `RecurOutput` drafts. Tiles materialize prompt data only
through authenticated reads: committed-external selections, input-handle
selections, `select!` projections, and selection-bound `AuthRef` args on
plain tiles called from recur-sequence bodies. No prompt-derived
collection accumulates in recur state (the P1-characterized loop-carried
pieces are the single documented exception — storage-backed between
iterations, inline in the per-round trace records); none rides context
structs or args into per-chunk/per-item tiles; recur-*tile* loops carry
scalars only (G7) — this program now contains none. Inefficiency is
accepted: full no-op passes under G1, per-item iteration, and repeated
authenticated reads of the same prompt-scoped value are all fine;
simplicity and verifiability win.

**Probe findings (Step 0, `chunk_probes.rs` P1–P4; all green before the
phases were rewritten, none contradicting the design):**

- **P1 — round-boundary characterization** (the D6 re-founding; findings
  quoted in the register): (a) a body tile's returned state persists in
  internal storage at the tile's output coordinates
  (`bind_infallible_call` → `store_execution_output_value`); resolving the
  returned `InternalRef` yields the stored state
  (`p1_tile_output_persists_in_internal_storage_and_resolve_validates`).
  (b) The re-entry resolve (`From<AuthRef<T>> for RecurSequenceState<T>`
  → `resolve_internal_value`) validates the coordinates lookup, the
  stored-vs-reference commitment, and a recomputed integrity commitment;
  a tampered commitment fails with `Internal store commitment mismatch at
  coordinates …`. (c) Each iteration's `RecurSequenceStart` records the
  threaded state as `FnInputValue::Inline` — full postcard, no binding
  metadata — with per-iteration size scaling linearly in the carried
  collection (64×32-byte seed ⇒ ≥2,048-byte state record); between
  iterations the driver holds the resolved value in memory.
- **P2 — fresh draft inside a body iteration:** a `RecurOutput` draft
  created (`output = new!(…)`) and finalized inside a recur-sequence body
  iteration works at the pinned rev, with the input list a `select!`
  projection of a previous tile's internal output, one plain tile
  threading cursor + draft via a `(RecurState<S>, RecurOutput<O>)`
  return, and the finalized `AuthRef` consumed in the same body via
  `select!` and as a follow-up tile's selection-bound arg before
  re-entering the outer threaded state. Draft payloads never ride the
  trace: iteration records carry a `DraftReplayHandle` (anchor + root)
  and the finalized value reaches consumers as `InternalBinding`.
- **P3 — selection-bound `Vec` args** on plain tiles in recur-sequence
  bodies trace as `InternalBinding` on both the iteration record and the
  tile record, and the tile materializes the full value at execution.
- **P4 — recur-tile tracing scope (G7's exact boundary):** recur-*tile*
  iteration records (`RecurTileIterationExec`) carry the input item, the
  state, and the args **inline** — full postcard payloads per iteration —
  which is why recur-tile loops carry scalars only and this program's
  last recur tile (the apply chunk loop) became a recur sequence.

**Tile-map delta** (rows follow the §1 table; unchanged rows omitted):

| # | Previous construct | Storage-resident construct | Catalog rows | Deviations |
|---|---|---|---|---|
| T-count | — (new) | `count_pieces(pieces: BpePieces) -> PieceCount` — one-shot authenticated read of the staged external; no staged count (an unchecked staged count is an integrity hole, a checked one is redundant) | C13, C11 | staging delta |
| T-budget | `build_chunk_budgets(initial_pieces, config) -> Result<ChunkBudgets>` | `build_chunk_budgets(piece_count: u32) -> ChunkBudgets { rounds }` — infallible; zero-width guards died with the widths | C1, G1 | D9 |
| S-round | `merge_bpe_round(input, state, initial_pieces: Vec<String>, config, merge_chunks, apply_chunks)` | `merge_bpe_round(input, state, staged_pieces: BpePieces, merge_chunks)` — body: open → select pieces/skip → pairs → scan → decision → per-piece draft apply → finalize | C4/C9, G1, A2 | D15 |
| T-open | `init_bpe_merge_scan(state, initial_pieces) -> GemmaBpeRoundContext` (pieces inline in the context) | `open_round(state, staged: BpePieces) -> GemmaBpeOpenedRound` — round scalars + pieces behind the selectable root; error flattened to `(has_error, error)` (G3) | C11, A2, G3 | D15 |
| T-pairs | — (scan windowed over inline pieces) | `build_pairs(pieces: BpePieces) -> GemmaBpeAdjacentPairs` — the round's pair list as its own internal ref | C11 | D17 |
| T-scan | `scan_one_merge_chunk(state, chunk, round /* pieces inline */)` | `scan_one_merge_chunk(state, chunk, skip: bool, pairs: GemmaBpeAdjacentPairs)` — whole-pairs selection-bound read (rule-2 permitted; G6 the named unlock); D11 priority orientation stands | C13/C22, D13 | D17 |
| T-decide | `finalize_bpe_merge_scan(scan, round) -> GemmaBpeMergeDecision` (pieces inline) | `finalize_bpe_merge_scan(scan, skip) -> GemmaBpeApplyDecision` — scalars only | C1 | D15 |
| T-apply | `apply_bpe_merge_chunk_or_complete` (recur *tile*; pieces accumulated in `GemmaBpeApplyState.output`) | `apply_one_piece(state: GemmaBpeApplyCursor, piece /* input handle */, output: Draft<BpePieces>, decision)` in recur sequence `apply_round_pieces` over the `select!`-ed round pieces; fresh draft per round, finalized at inner loop end | C4/C9, C16, G1, G7/P4 | D15 |
| T-close | `finalize_bpe_merge_iteration(apply, iteration)` | `finalize_round(applied: BpePieces, decision, opened) -> GemmaBpeLoopState` — range/count checks verbatim; skip/no-selection rounds carry the incoming pieces forward, never the empty draft | A1, C23/H4 | D15 |
| T-init-ids | `init_token_id_finalization(loop_state, initial_pieces) -> GemmaTokenIdContext` | `init_token_id_finalization(loop_state, staged: BpePieces) -> Result<BpePieces>` — fallible (A1 surfacing point); zero-round fallback to the staged pieces preserved | A1, C23 | D16 |
| T-resolve | `resolve_pieces_in_vocab_chunk(state, chunk, ctx)` + threaded slot vector | `resolve_pieces_in_vocab_chunk(chunk /* input handle */, final_pieces: BpePieces /* binding */, output: Draft<TokenIdMatches>)` — append-only, full pass per chunk | C13/C22, C16, D13 | D16 |
| T-final | `finalize_tokenize_prompt(resolution, ctx)` | `finalize_tokenize_prompt(matches: TokenIdMatches, final_pieces: BpePieces) -> Result<PromptTokenization>` — materializes both at the program boundary, orders by `piece_idx`, duplicate/out-of-range guards + the sim's exact missing-vocab message | C23/H4 | D16 |
| S-main | binds `bpe_config`; `initial_pieces` as bare `Vec<String>` | binds `external!(BpePieces, "initial_pieces")`; `bpe_config` binding deleted | C25 | staging delta |

**Types delta:** added `BpePieces { pieces }` (the selectable root and
draft schema for all prompt-scoped pieces), `TokenIdMatch` /
`TokenIdMatches` (append-only token-id draft), `PieceCount`,
`GemmaBpeOpenedRound`, `GemmaBpeAdjacentPair(s)`, `GemmaBpeApplyDecision`
(scalars), `GemmaBpeApplyCursor` (cursor-only). Changed:
`GemmaBpeLoopState` gains `piece_count` (mirrors sim `GemmaBpeState`) and
keeps `pieces` as the sole loop-carried collection; `ChunkBudgets` shrinks
to `{ rounds }`. Deleted: `BpeConfig`, `GemmaBpeRoundContext`,
`GemmaBpeMergeDecision`, `GemmaBpeIterationContext`, `GemmaBpeApplyState`,
`GemmaTokenIdContext`, `GemmaTokenResolutionState`. Error convention
unchanged (deferred `Option<String>` under A1; deterministic messages;
H4 parity where messages survive — the zero-width guard messages died
with their guards, and `token-id finalization stopped at piece …` became
structurally unreachable and was deleted with its test).

**Collections, where they live, and how they cross the ABI:**

| Collection | Lives in | Crosses the ABI as |
|---|---|---|
| staged initial pieces | committed external `BpePieces` | `ExternalBinding` args (`count_pieces`, `open_round`, `init_token_id_finalization`) |
| round pieces | internal storage (`open_round` output ref; finalized `RecurOutput<BpePieces>` drafts) | `select!` projections (skip scalar, apply input list), selection-bound args (`build_pairs`, `finalize_round`), draft push ops |
| loop-carried pieces (`GemmaBpeLoopState.pieces`) | tile-output internal store, authenticated-resolved at each boundary | inline state at the P1-characterized boundary — the single documented exception (D6) |
| adjacent pairs | internal storage (`build_pairs` output) | selection-bound arg on `scan_one_merge_chunk` (D17) |
| `merge_chunks` / `token_lookup_chunks` | committed external (unchanged) | recur-sequence input selection handles, one chunk per tile execution (D13) |
| round ordinals | internal storage (`build_chunk_budgets` output) | recur input selection handles |
| token-id matches | `RecurOutput<TokenIdMatches>` draft → internal storage | draft push ops; finalized ref as a binding to the finalizer |
| ordered token ids | materialized once in `finalize_tokenize_prompt` | terminal fallible tile output → `output.bin` |

**Staging and host-adapter delta:** `initial_pieces` stages as the
`BpePieces` root — postcard byte layout unchanged (a single-field postcard
struct is unframed; asserted host-side by
`staged_pieces_keep_the_bare_vec_byte_layout`), so the staged commitment
is identical. `StagedBpeConfig` and the `bpe_config` staged input are
deleted; `run_raster_core` drops its unused sizing-controls parameter
(`RasterSizingControls` itself is untouched — the frozen sim path consumes
it). WS2 §11 note appended.

**Structural guards** (`crates/raster-programs/prompt_prepare/src/guards.rs`,
test-only):

- Source-scan guard: the program's recur state types are pinned to
  `{GemmaBpeLoopState, GemmaBpeScanState, GemmaBpeApplyCursor}`;
  `GemmaBpeLoopState.pieces` is asserted to be the *only* collection field
  among them, and the per-item decision structs
  (`GemmaBpeApplyDecision`, `GemmaBpeScanCandidate`) are asserted to carry
  no pieces field and no collection.
- Trace-shape assertion: a native routine run over sentinel pieces
  verifies prompt-derived collections appear inline **only** in the
  P1-characterized records (`merge_bpe_round` iteration state,
  `open_round`'s state input) and that zero `RecurTileIterationExec`
  events exist; everywhere else the sentinels cross as bindings, draft
  ops, or the staged external. (Observed corollary: because the loop
  state seeds uninitialized per A2 and round 1 resolves the staged
  external, a one-round prompt never carries pieces inline at all.)
- Functional matrix: empty prompt, single piece / zero-round fallback,
  byte fallback, and the missing-piece error live in `routine.rs`;
  duplicate-conflict and out-of-range matches in `token_ids.rs`; no-merge
  multi-piece, repeated pieces, several rounds, and merge/vocab matches at
  chunk boundaries in `guards.rs`.

**Measurements (real Gemma `~/models/gemma-4-E4B-it`, prompt "Hello from
Raster", 36 initial pieces → 35 rounds, manual
`detour --raster-core-at prompt.prepare --max-new-tokens 1`, warm
tokenizer cache; trace analyzed by streaming `trace.ndjson`):**

| | trace-slimming baseline (2026-07-06) | storage-resident (2026-07-07) |
|---|---|---|
| trace events | 53,907 | 56,033 (18,616 `RecurSequenceStart`/`End` pairs, 18,725 `TileExec`, 72 recur-sequence sites) |
| `RecurTileIterationExec` events | present (apply chunk loop) | **0** — the program contains no recur tiles |
| largest inline value in any event's input | 102B (apply-phase cursors) | **99B** — the `merge_bpe_round` threaded-state record, i.e. the P1 boundary itself is now the largest inline crossing |
| input values traced as bindings | contexts as `InternalBinding`, chunks as `ExternalBinding` | 18,080 `ExternalBinding` + 109,528 `InternalBinding` input values (pieces, pairs, decisions, finalized drafts); 95,240 inline values (cursors, flags, markers, replay handles, the P1 state) |
| raw `trace.ndjson` | 1.93GB | 1.8GB |
| program wall time | ≈ 1 min | ≈ 107s (run-dir creation → last trace write; includes `cargo raster run` overhead) |

The iteration-count structure is unchanged in kind (G1 full passes:
35 rounds × 503 merge chunks = 17,605 scan iterations + one 256-chunk
vocab pass + 35 round iterations); the apply phase accounts for the
remaining 720 iterations — one per round piece instead of one per
width-64 chunk, the accepted per-item cost — and the draft/binding forms
replace the inline collections everywhere outside the P1 boundary.

**Acceptance record:** model tables remain chunked committed-external
recur-sequence inputs; initial pieces enter once as the staged `BpePieces`
external; each round's pieces are produced through
`RecurOutput<BpePieces>` and persist in internal storage; all pieces
consumption is through authenticated reads with the single P1 crossing
documented; no recur state accumulates a prompt-derived collection and no
context struct carries one into a loop; token resolution is append-only
with ids materialized once in the terminal finalizer; structural guards
in place; trace-identity gate, goldens, full suite, and the no_std surface
green at every commit.
