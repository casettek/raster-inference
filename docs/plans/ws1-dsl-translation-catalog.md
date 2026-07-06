# WS1 — DSL Translation Catalog

- **Status:** state-carrying document. Phase A (enumerate + map) complete; Phase B
  (probe verification) complete for the mandatory probes. Rows carry their current
  status; unresolved rows block WS3 for the routines that use them.
- **Amendment convention:** any change to a row's mapping, status, or porting rule
  is made in place with a dated note appended to §9 (Amendment log). A row may only
  move to `verified` with cited, re-runnable evidence (probe or `raster` test).
- **Companions:** the migration charter (`raster-core-migration-charter.md`),
  `docs/plans/2026-07-05-001-raster-core-migration-adr.md` (WS0 decisions),
  `docs/plans/RASTER_TILE_DSL_GUIDE.md` (sim authoring contract).
- **Grounding:** sim side enumerated from source on branch `raster-core-migration`;
  real side from the `raster` checkout at the WS0-pinned rev
  `536214533f06381a913e5873772e283025ebe061`.

Statuses: `verified` (cited evidence) / `mapped-unverified` (doc+source-derived,
no probe yet) / `GAP` (not expressible as-is; named resolution path in §6).

---

## 1. Scope and census

**Scope of enumeration:** all `src/routines/<routine>/raster/` modules (10
routines), plus tile-shaped shared code transitively reached from routine tiles:
`src/shared/artifacts/{artifact_io,raster_artifact_store}.rs`,
`src/shared/raster_contracts/`, `src/shared/raster_kernels/`,
`src/shared/tensors/raster_tensor_artifacts.rs`, and the routine-local
`auth_source.rs` modules. The DSL definition itself is `src/dsl/` +
`crates/dsl-macros/`.

**Census (production sites, `tests*` files excluded), 2026-07-05:**

| Routine | `call_tile!` | `call_seq!` | `call_recur_tile!` | `call_recur_seq!` | `#[tile*]` | `#[sequence*]` |
|---|---|---|---|---|---|---|
| `prompt.prepare` | 11 | 0 | 4 | 2 | 11 | 3 |
| `input.embedding` | 2 | 0 | 1 | 0 | 3 | 1 |
| `prefill.prepare_aux` | 17 | 7 | 6 | 1 | 35 | 9 |
| `prefill.range` | 73 | 35 | 33 | 1 | 112 | 37 |
| `prefill.range_finalize` | 0 | 0 | 0 | 0 | 1 | 0 |
| `prefill.finalize` | 4 | 1 | 1 | 0 | 5 | 2 |
| `decode.select_token` | 5 | 2 | 3 | 0 | 9 | 3 |
| `decode.layer_range` | 67 | 29 | 4 | 2 | 106 | 22 |
| `decode.transition_finalize` | 0 | 0 | 0 | 0 | 0 | 0 |
| `output.finalize` | 3 | 2 | 1 | 0 | 5 | 6 |
| **Total** | **182** | **76** | **53** | **6** | **287** | **83** |

Attribute variants across routine `raster/` modules (non-test): `#[tile]` 248,
`#[tile(kind = recursive)]` 39, `#[sequence]` 79, `#[sequence(kind = recursive)]` 4.
Auth family: `auth_read` 281 / `ArtifactIo` 174 sites when scoped to routines +
`raster_contracts` + `raster_kernels` (the seed counts in the WS1 prompt fall
between this scope and full-`src/`, which adds `src/shared/artifacts` internals
and `src/dsl/tests.rs`; no branch-state discrepancy). `external(` has **zero**
routine-tile sites — sim `External<T>` is exercised only in `src/dsl/tests.rs`.

`prefill.range_finalize` defines one exported `#[tile]` invoked from
`prefill_range/raster/tiles.rs:91`; `decode.transition_finalize` has **no DSL
constructs** — its `run_raster` is plain glue calling `decode_layer_range`'s
finalize tile (`src/routines/decode_transition_finalize/raster/tiles.rs:8-31`).

**Sim DSL ground truth.** `crates/dsl-macros/src/lib.rs` `#[tile]`/`#[sequence]`
are identity passthroughs (no codegen; `kind = recursive` is not even parsed).
All sim runtime semantics live in `src/dsl/macros.rs` (invoker macros + recur
loops) and `src/dsl/runtime.rs` (invocation counting, `External<T>`). The
`kind = recursive` attribute is authoring convention enforced only by the
`(bool, State…)` return shape the recur loops require.

---

## 2. Catalog — core invokers and attributes

### C1. `#[tile]` (marker)

| Field | Content |
|---|---|
| Construct | `#[tile]` attribute on a free fn with owned/borrowed serde-friendly params |
| Sites | 248 non-test, all routines except `decode.transition_finalize`; e.g. `src/routines/prompt_prepare/raster/tiles.rs:120`, `prefill_range/raster/tiles.rs:3176` |
| Routines affected | all except `decode.transition_finalize` |
| Real-raster mapping | real `#[tile]` (`raster-macros`): generates traced wrapper + postcard ABI entry (`__raster_tile_entry_*`) + replay entry + `TileCallBinding` |
| Status | verified |
| Evidence | P2 (`probes::p2_abi::add_pair` etc., run R1); `raster/examples/hello-tiles/src/lib.rs` |
| Porting rule | Keep the free-fn shape. All params and return become owned serde types (see C13/H3 for `&T` params). No generics. Fallible tiles return `raster::exec::Result<T>` (= `Result<T, String>`), not `anyhow::Result`. Multi-arg ABI is postcard tuple `(T1, T2, …)` — see C12. |
| Notes | Real `#[tile]` accepts `description`, `estimated_cycles`, `max_memory` attrs — use during WS8 sizing. Sim tiles with non-`Result` returns (e.g. `check_stop_condition`, `decode_select_token/raster/tiles.rs:44-50`) map directly to plain-return real tiles. |

### C2. `#[tile(kind = recursive)]`

| Field | Content |
|---|---|
| Construct | `#[tile(kind = recursive)]` — chunk-loop tile returning `(bool, State…)`, driven by `call_recur_tile!` |
| Sites | 39 non-test; e.g. `prompt_prepare/raster/tiles.rs:149,284,435`, `input_embedding/raster/tiles.rs:83`, `decode_select_token/raster/tiles.rs:75` |
| Routines affected | 8 of 10 (all except `prefill.range_finalize`, `decode.transition_finalize`) |
| Real-raster mapping | `#[tile(kind = recur)]` with `input: RecurInput<T>` first param, then optional `RecurState<S>` / `RecurOutput<Schema>`, returning the matching mode's type or `RecurControl<…>` — **plus a structural rewrite** from condition-driven to list-driven iteration (H2) |
| Status | verified (mapping); the *rewrite obligation* is recorded as gap G1 |
| Evidence | P1 (run R1: `scan_max` 8 iterations over the full list, `sum_with_break` stopped after 3 of 8 via `Break`, `until_done_bounded` converged after 5 of 8); `raster/crates/raster/tests/recur_draft.rs` |
| Porting rule | Rewrite each sim recur tile as `#[tile(kind = recur)]`: the sim's loop-carried state struct becomes `RecurState<S>`; the sim's `done` boolean becomes `RecurControl::Break`/`Continue` (or plain return = implicit Continue); the iteration source becomes an explicit `AuthRef<Vec<T>>` list (see G1 for constructing the bound). Do **not** use `#[tile(recur)]` — key/value form only. |
| Notes | Real recur tiles cannot be invoked with `call!` for looping — `call!` on a recur tile is a single invocation. The spelling difference sim `kind = recursive` vs real `kind = recur` is intentional; catalog treats them as the same author intent. |

### C3. `#[sequence]`

| Field | Content |
|---|---|
| Construct | `#[sequence]` — straight-line orchestration fn composing `call_tile!`/`call_seq!`/recur invokers |
| Sites | 79 non-test; e.g. `prompt_prepare/raster/tiles.rs:20`, `prefill_range/raster/tiles.rs:48,98,116` |
| Routines affected | all except `prefill.range_finalize`, `decode.transition_finalize` |
| Real-raster mapping | real `#[sequence]`: body rewritten so `call!`/`call_seq!`/`call_recur!` become auth-bound calls returning `AuthRef<T>`; non-main sequences become `__raster_sequence_auth_*` with `impl IntoAuthRef<T>` args |
| Status | verified |
| Evidence | P5 (run R1: nested `outer_pipeline` → `inner_transform` sequence with tile calls; value flow observed); `raster/crates/raster/tests/external_selection.rs:381-428` |
| Porting rule | Keep sequences straight-line (the sim guide's rule matches the real CFS-extraction constraint: branching lives in tiles). Every tile/sequence invocation inside a `#[sequence]` body **must** go through `call!`/`call_seq!`/`call_recur!`/`call_recur_seq!` — bare calls are invisible to CFS extraction. Values flowing between calls become `AuthRef<T>`; materialize only at the program boundary via `materialize_auth_return`/`materialize_auth_result`. |
| Notes | Sim sequences freely destructure/construct structs between calls (e.g. `prompt_prepare/raster/tiles.rs:42-45`). In real sequences, arbitrary expression code between calls does not execute as tile work and cannot inspect `AuthRef` contents — field access needs `select!` (C11) and computation needs a tile. This is a load-bearing rewrite constraint for WS3. |

### C4. `#[sequence(kind = recursive)]`

| Field | Content |
|---|---|
| Construct | Recursive sequence returning `Result<(bool, State…)>`, driven by `call_recur_seq!` |
| Sites | 4 non-test: `prompt_prepare/raster/tiles.rs:67`, `prefill_prepare_aux/raster/tiles.rs` (PLE layer loop), `prefill_range/raster/tiles.rs:74`, `decode_layer_range/raster/tiles.rs` (layer loop) |
| Routines affected | `prompt.prepare`, `prefill.prepare_aux`, `prefill.range`, `decode.layer_range` |
| Real-raster mapping | `#[sequence(kind = recur)]` with `RecurSequenceInput<T>` / `RecurSequenceState<S>` / `RecurSequenceOutput<Schema>` params, driven by `call_recur_seq!(sequence = …, input = …, …)` |
| Status | verified (mapping); rewrite obligation shared with G1 |
| Evidence | P1 (run R1: `per_item_pipeline` recur sequence orchestrated tiles per item, state threaded); `raster/crates/raster/tests/recur_draft.rs:304-415` |
| Porting rule | The sim's `(bool, State)` continuation becomes a real recur sequence over an explicit list. **Constraint (real):** recur sequences have no `RecurControl` — they always run all items (`run_recur_sequence_list*`, `raster/crates/raster/src/input.rs:2045-2193`) and `RecurSequenceInput`/`State`/`Output` are opaque inside the sequence body (values only touchable in tiles). Early termination must therefore live inside the per-item tiles (no-op remaining iterations), or the loop must be restructured as a recur *tile*. See G1. |
| Notes | The four sim sites are layer/PLE/BPE loops. Layer loops have static bounds (layer_count) → list of layer indices. The BPE loop's bound is data-dependent (see G1). |

### C5. `call_tile!`

| Field | Content |
|---|---|
| Construct | `call_tile!(tile, args…)` — records invocation, calls fn (`src/dsl/macros.rs:2-15`) |
| Sites | 182 non-test across 8 routines |
| Routines affected | all except `prefill.range_finalize` (defines but doesn't invoke), `decode.transition_finalize` |
| Real-raster mapping | `call!(tile, args…)` inside `#[sequence]` bodies (rewritten to `bind_tile_call`, output stored internally, returns `AuthRef<T>`) |
| Status | verified |
| Evidence | P2/P5 (run R1); `raster/examples/hello-tiles/src/main.rs` |
| Porting rule | `call_tile!(f, a, b)?` → `call!(f, a, b)?`. Fallible calls surface `exec::Result` through `?` on the binding. Args must be `IntoAuthRef`-compatible (owned values, `AuthRef`s, or bindings); returns are `AuthRef<T>`, not `T`. |
| Notes | Sim call sites often immediately destructure returned tuples — in real raster, tuple component access requires `select!` or restructuring the tile to return a named struct deriving `Selectable`. Prefer named structs in ports. |

### C6. `call_seq!`

| Field | Content |
|---|---|
| Construct | `call_seq!(sequence, args…)` (`src/dsl/macros.rs:17-31`) |
| Sites | 76 non-test; heaviest in `prefill.range` (35) and `decode.layer_range` (29) |
| Routines affected | `prefill.prepare_aux`, `prefill.range`, `prefill.finalize`, `decode.select_token`, `decode.layer_range`, `output.finalize` |
| Real-raster mapping | real `call_seq!(sequence, args…)` inside `#[sequence]` bodies |
| Status | verified |
| Evidence | P5 (run R1: sequence-in-sequence composition, `SequenceStart`/`SequenceEnd` trace shape observed) |
| Porting rule | Direct rename-preserving mapping; same `AuthRef` value-flow rules as C5. Nesting depth (sim `prefill.range` nests 3–4 deep) is supported; the trace records nested `SequenceStart`/`SequenceEnd`. |
| Notes | — |

### C7. `call_recur_tile!` — single-state form (± trailing context)

| Field | Content |
|---|---|
| Construct | `call_recur_tile!(tile, state[, context…])` → loop until `(done=true, next)` (`src/dsl/macros.rs:144-172`) |
| Sites | ~39 of the 53 recur-tile sites; e.g. `prompt_prepare/raster/tiles.rs:52,73,76,114`, `decode_select_token/raster/tiles.rs:27,36-37`, `output_finalize/raster/tiles.rs:41-44` |
| Routines affected | 8 of 10 |
| Real-raster mapping | `call_recur!(tile = …, input = <list>, state = <initial>, args = (context…,))` — state-only recur mode |
| Status | verified (mechanics); rewrite obligation G1 |
| Evidence | P1 (run R1: state threading across iterations, Break honored, state materialized after loop); `recur_draft.rs` `count_seen_until_limit` |
| Porting rule | Sim initial state expr → `state = <expr>` (struct literal allowed). Trailing context args → `args = (…,)` with **owned** values (clone or re-`select!`; borrowed `&T` context is G3/H3). Supply `input =` an explicit list that bounds the iteration (G1). Sim `done` → `RecurControl::Break(state)` at the equivalent point; otherwise `Continue(state)`. |
| Notes | Sim loops carry rich state structs (cursors + roots + config). These stay as the `RecurState<S>` type; per-iteration chunk indices can come from the input list instead of a cursor field where natural. |

### C8. `call_recur_tile!` — pair-state form

| Field | Content |
|---|---|
| Construct | `call_recur_tile!(tile, (a, b)[, context…])` — two separate loop-carried params, tile returns `(bool, A, B)` (`src/dsl/macros.rs:176-209`) |
| Sites | ~14 non-test; `input_embedding/raster/tiles.rs:28-32`, `prefill_range/raster/tiles.rs:328,350,373,396,418,439,462,485,506,527,548,569`, `decode_layer_range/raster/tiles.rs:179,197,504,525` |
| Routines affected | `input.embedding`, `prefill.range`, `decode.layer_range` |
| Real-raster mapping | Same `call_recur!` state-only mode with `state = (A, B)` folded into **one** state struct (real recur threads exactly one `RecurState<S>`; optionally plus one `RecurOutput`) |
| Status | verified (via P1 state threading); fold rule is a structural rewrite, part of G2 decomposition |
| Evidence | P1 (run R1); `raster-macros` recur-tile signature validation (`raster/crates/raster-macros/src/lib.rs:1091-1280`) |
| Porting rule | Fold `(roots, work)` into a single serde struct (or — better — eliminate the `roots` leg entirely per the H1 decomposition, since real internal storage/draft commitments replace explicit roots threading; see C15/H1). If output accumulation is separable, use the state+output recur mode (`RecurState` + `RecurOutput`) which real raster supports natively. |
| Notes | Every sim pair-state site pairs `RasterArtifactStoreRoots` with a work struct — the pair form exists *because of* roots threading. If H1 lands as hypothesized, the pair form disappears in ports rather than being translated. |

### C9. `call_recur_seq!` — struct-state form (± context)

| Field | Content |
|---|---|
| Construct | `call_recur_seq!(seq, StructLiteral { … }, context)` (`src/dsl/macros.rs:274-303`); sim sequence returns `Result<(bool, State)>` |
| Sites | 2: `prompt_prepare/raster/tiles.rs:34-41, 95-102` |
| Routines affected | `prompt.prepare` |
| Real-raster mapping | `call_recur_seq!(sequence = …, input = <list>, state/output = …, args = (…,))` over a bounded list |
| Status | verified (mechanics); **the `prompt.prepare` BPE site is gap G1's canonical case** |
| Evidence | P1 (run R1 recur-sequence leg); `recur_draft.rs:406-415` |
| Porting rule | See C4 porting rule plus G1: the BPE merge loop must be restructured to a bounded iteration (max merges = initial piece_count − 1) with in-tile no-op continuation after convergence, or hoisted to a recur tile so `RecurControl::Break` is available. |
| Notes | Real recur sequences cannot early-exit (no `RecurControl`); this is the decisive constraint for choosing tile-loop vs sequence-loop shape per site. |

### C10. `call_recur_seq!` — pair-state form

| Field | Content |
|---|---|
| Construct | `call_recur_seq!(seq, (roots, state), context)` |
| Sites | 4: `prefill_range/raster/tiles.rs:63-67`, `decode_layer_range/raster/tiles.rs:60-64,708-712`, `prefill_prepare_aux/raster/tiles.rs:40-44` |
| Routines affected | `prefill.range`, `decode.layer_range`, `prefill.prepare_aux` |
| Real-raster mapping | As C9 with the pair folded per C8 (or dissolved per H1); iteration list = layer indices (static bound: `layer_count`) |
| Status | verified (mechanics); fold/restructure per G1/H1 |
| Evidence | P1 (run R1); `recur_draft.rs` recur-sequence tests |
| Porting rule | Layer loops have static bounds known at sequence start → `input = internal!(Vec<u32>, layer_indices_ref)` (or an external). Early exit is not needed for layer loops (they always run `layer_count` iterations), so the no-`RecurControl` constraint is harmless here. |
| Notes | These are the model-critical loops; their trace shape (`RecurSequenceStart`/iteration scopes/`RecurSequenceEnd`) is what WS4 structural checks will compare. |

### C11. Field/element access on flowing values

| Field | Content |
|---|---|
| Construct | (structural) Sim sequences destructure structs and index collections directly between calls (e.g. `prompt_prepare/raster/tiles.rs:42-45`, `prefill_range/raster/tiles.rs:56-70` tuple chains) |
| Sites | pervasive — every sequence body |
| Routines affected | all with sequences |
| Real-raster mapping | `select!(FieldTy, binding.field[index])` on `Selectable`-deriving types; chained selection supported |
| Status | verified |
| Evidence | P3/P4 (run R1: struct field and list-index selection with commitment verification); `raster/crates/raster/tests/external_selection.rs` |
| Porting rule | Cross-call types derive `Selectable` (+ serde). Replace destructuring with `select!` per accessed field. Tuples are not selectable — convert tuple returns to named structs. |
| Notes | Each `select!` produces a selection commitment in the trace — heavy destructuring has trace-size cost; prefer passing whole `AuthRef`s and selecting inside tiles (tiles receive materialized values). |

### C12. Multi-arg tile ABI

| Field | Content |
|---|---|
| Construct | (structural) Sim tiles take 1–5 positional args freely (any Rust types) |
| Sites | pervasive |
| Routines affected | all |
| Real-raster mapping | postcard ABI: 0 args = empty, 1 arg = `postcard(T)`, N>1 = `postcard((T1,…,Tn))` (`raster-macros/src/lib.rs:2218-2272`); `Result` returns encode the user `Result<T, String>` |
| Status | verified |
| Evidence | P2 (run R1: 2-arg `add_pair`, 3-arg `describe_triple`, enum-arg `mode_name` all round-trip; fallible Ok path via `?`; Err surfaced at the `call_seq!` boundary in `main` — "p2 expected err surfaced: checked_div: divisor is zero" — without aborting the program. Run R3: the same Err propagated terminally.) |
| Porting rule | All tile params/returns must be owned `serde` de/serializable (postcard-compatible: no maps with non-string keys issues, no borrows, no `usize` portability hazards — prefer `u32`/`u64` for cross-target determinism in guest-visible types). |
| Notes | Sim types like `RasterPromptPreparationResult` (`prompt_prepare/raster/types.rs:17-21`) lack serde derives today — every cross-tile type gains `Serialize`/`Deserialize` (+ `Selectable` where selected from). |

---

## 3. Catalog — authenticated artifact store family (H1)

The sim's dominant family. Semantic core: a thread-local Merkle-rooted store
(`src/shared/artifacts/raster_artifact_store.rs`, ~2150 lines) with artifacts
(finalized leaf sequences), builders (append-only under construction), verified/
authenticated leaf reads (payload + Merkle proof), and an explicit snapshot type
`RasterArtifactStoreRoots` threaded through tile signatures so committed state is
visible in checkpoint payloads.

**H1 resolution (decomposition), Phase B-confirmed for the in-tile legs:**

- **In-tile constructs** map to real internal storage + `Draft` machinery +
  selection proofs (rows C13–C17).
- **Host-side responsibility (WS2):** store initialization from native state,
  external source registration, roots export into checkpoint payloads, and
  cross-routine artifact hand-off. In-tile residue: every value a fault proof
  must see is either a committed external input (manifest commitment) or an
  internal-storage value created by tile execution (trace commitment). Nothing
  else may influence a committed outcome.
- **Confirmed by P4:** internal-storage writes, draft append/finalize, and
  selections all appear in the commit artifact's trace commitment; a tampered
  external input is rejected at resolve time (P3).

### C13. `AuthRead` trait + `ArtifactIo::auth_read` on authenticated sources

| Field | Content |
|---|---|
| Construct | `trait AuthRead<Request>` (`artifact_io.rs:9-13`); request-typed reads against `Authenticated*`/`Raster*Source` wrappers (tokenizer vocab/merges, weight rows, PLE rows, metadata) |
| Sites | 281 `auth_read` sites in scope; impl sites: `shared/model/gemma/tokenizer.rs:487-563`, `shared/raster_contracts/prefill_layer.rs:424-530`, `prefill_ple.rs:738-870`, routine `auth_source.rs` modules |
| Routines affected | all except `prefill.range_finalize`, `decode.select_token` (tiles), `decode.transition_finalize` |
| Real-raster mapping | Committed **external input** + `select!`: the source's entry set (request-key → payload, cf. `committed_source_entries()`, `tokenizer.rs:431-478`) becomes a committed external (rastered file + manifest sha256/commitment); each `auth_read(source, request)` becomes `select!(Output, external.entry[key-path])` or a tile-internal lookup on a selected sub-structure |
| Status | mapped-unverified → **verified for the mechanism** (external + select with commitment verification, P3); per-source data layout is WS2 scope |
| Evidence | P3 (run R1 + tamper run R2: byte-flipped external rejected); `raster-tokenizer` PoC (`src/tokenizer.rs` lookup tiles over selected `Vec<GemmaTokenIdEntry>`) |
| Porting rule | Each authenticated source becomes one committed external with a `Selectable` schema. Point lookups (token id by string, merge by pair) that sim served via hashed request keys become either (a) `select!` by index after a tile computes the index (binary search over a sorted entry list — the tokenizer-PoC idiom), or (b) whole-substructure selection + in-tile scan for small tables (metadata, scalars). The choice is per-source and belongs to the routine's WS3 plan; the catalog constraint is only that every read is commitment-checked (external selection proof) — never ambient. |
| Notes | The `raster-tokenizer` PoC demonstrates idiom (a) end-to-end for exactly the Gemma tokenizer data. WS2 owns producing the committed files; the in-tile residue is `select!` + lookup tiles. |

### C14. `impl AuthRead<…> for str` (root-string sources)

| Field | Content |
|---|---|
| Construct | Reads dispatched from a **root string** via `CommittedExternalSource::from_root(self)` (e.g. `tokenizer.rs:653-699`, `input_embedding/raster/auth_source.rs:318-333`) — tiles carry only the root and re-attach to the ambient registry |
| Sites | ~20 impl sites; used where tile state carries `*_root: String` fields |
| Routines affected | `prompt.prepare`, `input.embedding`, `prefill.finalize`, `decode.layer_range` (via source structs) |
| Real-raster mapping | Dissolves: real `AuthRef`/`TypedExternalBinding` **is** the portable handle; passing an `AuthRef<T>`/binding into tiles replaces passing a root string and re-resolving |
| Status | verified (the replacement mechanism is the ordinary external/internal binding path, P3/P4) |
| Evidence | P3/P4 (run R1) |
| Porting rule | Replace `*_root: String` state fields with typed refs (`AuthRef<T>` reference()s are only valid for internal refs; external bindings are re-creatable by name). Where sim used the root to *authenticate identity* (root-equality checks like `prompt_prepare/raster/tiles.rs:26-33`), keep the check but compare manifest commitments / selection roots, which the runtime enforces anyway — record the check as a tile assertion only if it guards a *terminal* outcome. |
| Notes | The ambient thread-local registry is precisely what real raster forbids in guests; this row is the "no ambient sources" enforcement point. |

### C15. `RasterArtifactStoreRoots` threading / `(Roots, T)` tuple returns / `*_with_roots` ops

| Field | Content |
|---|---|
| Construct | (structural) Snapshot struct threaded through nearly every tile: mutating ops validate the snapshot against the live store and return fresh roots (`raster_artifact_store.rs:1062-1066`); tiles return `(RasterArtifactStoreRoots, T)`; pair-state recur exists for this (C8) |
| Sites | pervasive — the defining structural pattern of the sim path |
| Routines affected | all except `decode.transition_finalize` |
| Real-raster mapping | **Dissolves into the runtime's implicit commitment structure.** Real internal storage assigns trace-tree coordinates per tile execution and commits reads/writes in the trace commitment; drafts carry `anchor`/`current_root` internally. There is no user-visible store snapshot to thread. |
| Status | verified (P4 shows intermediate values + draft transitions committed without any explicit roots threading) |
| Evidence | P4 (run R1: commit artifact contains trace commitment covering internal-storage ops); `raster-runtime/src/internal_storage.rs` coordinate model |
| Porting rule | Delete the roots legs from ported signatures. Where a sim tile *published* a root into a checkpoint payload (boundary checkpoints), that becomes host-side: WS2/WS5 extract the equivalent commitment (external manifest commitment or internal selection root) after the run. Where roots-threading enforced ordering/staleness, the real runtime's per-coordinate append-only store provides the equivalent guarantee mechanically. |
| Notes | This is the single largest signature simplification in the port, and the reason "porting" is a rewrite. WS5 owns the boundary-checkpoint payload contract for the three value-form/root-form exceptions. |

### C16. Builder lifecycle: `start_builder*` / `append_leaf*` / `finalize_builder*`

| Field | Content |
|---|---|
| Construct | Append-only artifact construction across tile boundaries: builders started in init tiles, appended per chunk-loop iteration, finalized in finalize tiles (e.g. `prompt_prepare/raster/tiles.rs:133-137, 179-185, 211-214`; `input_embedding/raster/tiles.rs:63-68, 118-124, 149-153`) |
| Sites | ~90 `ArtifactIo` builder-verb sites |
| Routines affected | all with tiles except `decode.transition_finalize` |
| Real-raster mapping | `Draft<S>` machinery: `new!(Schema)` → `DraftAppendField` `.push(…)` in tiles → `finalize(draft) -> AuthRef<S>`; drafts thread through recur via `RecurOutput<S>`/`RecurSequenceOutput<S>` |
| Status | verified |
| Evidence | P4 (run R1: draft created, pushed across recur iterations, finalized, selected from; draft transitions in commit artifact); `recur_draft.rs`, `draft_selection.rs` |
| Porting rule | One sim builder = one `Selectable` schema with an append-only `Vec` field (+ set-once metadata fields). `start_builder` → `new!(Schema)` (+ set-once fields); `append_leaf(idx, leaf)` → `.field().push(item)` — **note: no explicit index**; append order is the index, and sim code that appends by out-of-order index (none found in scope) would be a gap. `finalize_builder` → `finalize(draft)`. Leaf-count/shape metadata (`RasterArtifactMetadata`) becomes schema-level or set-once fields. |
| Notes | Draft handles are affine (no clone/reuse — compile-fail enforced). Sim leaf payloads are hand-postcarded `Vec<u8>` (`utils.rs:305-308`); real drafts store typed values — drop the manual postcard layer. |

### C17. Read ladder: `read_leaf` / `read_verified_leaf` / `read_authenticated_leaf` (+`_from_roots`)

| Field | Content |
|---|---|
| Construct | Proof-carrying leaf reads from finalized artifacts (e.g. `prompt_prepare/raster/utils.rs:45-50`, `decode_select_token` logit reads, `shared/tensors/raster_tensor_artifacts.rs` row reads) |
| Sites | ~60 in scope |
| Routines affected | all with tiles |
| Real-raster mapping | `select!(T, internal_ref_or_authref.field[i])` — internal selection with proof, or `internal!(T, reference)` re-binding then `select!` |
| Status | verified |
| Evidence | P4 (run R1: element selection from finalized draft with commitment verification); `internal_storage.rs` selection-witness tests |
| Porting rule | Reads *within the producing program run* are internal selections. Reads of artifacts produced by a **previous routine's run** are not internal — they arrive as committed external inputs of the next program (WS2 staging), read via `external!` + `select!` (P3 mechanism). This run-boundary split is the artifact-store family's host-side seam. |
| Notes | The fault-proof visibility line: in-tile reads must be selection-proof-backed either against the input manifest (external) or the trace commitment (internal). Both verified. |

### C18. `insert_artifact` / `store_byte_artifact` / `store_text_artifact` (direct insert)

| Field | Content |
|---|---|
| Construct | Whole-artifact insertion, used in **pre-DSL utils** (`prompt_prepare/raster/utils.rs:70-99,114-142`, `output_finalize` prep) — not inside tiles |
| Sites | ~15, all in `utils.rs`-tier code |
| Routines affected | `prompt.prepare`, `output.finalize`, `decode.select_token` (utils) |
| Real-raster mapping | Host-side staging (WS2): materialize as committed external inputs (`write_raster_files` / postcard bin + manifest entry), or as `store_internal_value` performed by an init tile if the data is derived in-program |
| Status | verified (both legs exercised: external staging P3, `store_internal_value` P4) |
| Evidence | P3/P4 (run R1); `raster-tokenizer/bin/encode_tokenizer.rs` |
| Porting rule | If the value comes from native state (prompt bytes, rendered prompt): WS2 stages it as a committed external. If it is derived inside the routine: an init tile stores it internally. In-tile residue: none for staging; the manifest commitment is the proof anchor. |
| Notes | `init_artifact_store()`/`reset_store` (`utils.rs:101-103`) dissolves — each real program run is a fresh runtime. |

### C19. `export_store_roots` / roots into checkpoint payloads

| Field | Content |
|---|---|
| Construct | `ArtifactIo::export_store_roots()` snapshots for checkpoint payloads and cross-routine hand-off (`prompt_prepare/raster/utils.rs:158,267`) |
| Sites | ~12 |
| Routines affected | all boundary routines |
| Real-raster mapping | Host-side (WS2/WS5): after `cargo raster run`, ingest the commit artifact + output values; the payload-visible commitments come from the manifest (externals) and the trace/selection roots (internals) |
| Status | mapped-unverified (mechanism exists — commit artifact produced and parsed in P4 — but the WS5 payload contract decides exact forms) |
| Evidence | P4 (run R1 commit artifact); WS5 pending |
| Porting rule | Never emit raster-core structure into committed payloads (charter invariant 2); the three boundary-checkpoint exceptions are resolved in WS5, recorded next to this row when ruled. |
| Notes | Explicit WS2/WS5 handoff row. |

### C20. Integrity modes (`Verified` vs `UncheckedTestOnly`)

| Field | Content |
|---|---|
| Construct | Feature `unchecked-raster-integrity` swaps real Merkle roots for synthetic strings and skips proof checks (`raster_artifact_store.rs:1339-1340,1489-1491`; `tokenizer.rs:408-418`) |
| Sites | store-wide; `raster_source_root_for_current_integrity_mode` call sites in tiles |
| Routines affected | all (test-mode only) |
| Real-raster mapping | **No real-raster equivalent, deliberately.** Real toolchain always verifies commitments. |
| Status | verified-as-dissolved (no port) |
| Evidence | absence: no such feature in `raster` at pinned rev; P3 tamper rejection is unconditional |
| Porting rule | Do not port. Test acceleration on the raster-core path, if ever needed, is a host-side test-fixture concern (smaller staged inputs), never a proof-skipping mode. Tiles must not branch on integrity mode; sim call sites like `tokenizer.rs:26` (root-for-mode) simply disappear with C14. |
| Notes | Keeps the fault-proof path honest by construction. |

### C21. Typed artifact refs (`RasterTokenIdSequenceRef`, `RasterActivationSequenceRef`, …)

| Field | Content |
|---|---|
| Construct | Newtype wrappers over `RasterArtifactRef` carrying kind/shape validation (`raster_artifact_store.rs`, `shared/tensors/raster_tensor_artifacts.rs`) |
| Sites | pervasive in types/state structs |
| Routines affected | all |
| Real-raster mapping | Typed `Selectable` schemas: the artifact's element type + metadata fields *are* the schema; kind/shape validation becomes schema shape + tile assertions |
| Status | mapped-unverified (schema-design guidance; P4 verified the underlying machinery) |
| Evidence | P4; schema design is per-routine WS3 work |
| Porting rule | One schema per artifact kind (token-id sequence, activation sequence with row width, KV cache, …) in the program crate (or a small shared `no_std` support crate under `crates/raster-programs/` if duplication across routine crates becomes material — decide at first duplication, not before). |
| Notes | Shape metadata (`row_count`, `hidden_size`) that sim kept in `RasterArtifactMetadata` moves into schema fields set once. |

---

## 4. Catalog — cross-cutting patterns (H2–H5 and structure)

### C22. Reference parameters / ambient sources (H3)

| Field | Content |
|---|---|
| Construct | `&T` params in tile/sequence signatures: `&AuthenticatedGemmaTokenizer`, `&RasterInputEmbeddingSource<'_>`, `&RasterPrefillLayerSource<'_>`, `&RasterInputEmbeddingRefs`, `Option<&str>` (139 signature sites) |
| Sites | e.g. `prompt_prepare/raster/tiles.rs:24,70,152`, `input_embedding/raster/tiles.rs:20,47,87`, `prefill_range/raster/tiles.rs:52-54` |
| Routines affected | all except `decode.select_token` (fully self-contained state), `prefill.range_finalize` |
| Real-raster mapping | Owned committed externals + per-use `select!` (C13), with small selected sub-structures cloned into recur `args = (…)` where a loop needs them (tokenizer-PoC idiom) |
| Status | verified (mechanism, P3); per-source layout is WS2 |
| Evidence | P3 (run R1); `raster-tokenizer/src/main.rs:8-9`, `src/tokenizer.rs:311-340` |
| Porting rule | No borrows cross the tile ABI. For big sources (weights): keep them external and select rows per request — the per-row selection replaces `auth_read(source, RowRequest)` one-for-one. For small metadata: select once into an owned struct and pass by value through state/args. `Option<&str>` → `Option<String>`. |
| Notes | Directly shapes WS2's input-staging design; the mmap `load_preference` in `input.json` covers the large-weights case. |

### C23. Error contract (H4)

| Field | Content |
|---|---|
| Construct | `anyhow::Result` + `bail!`/`anyhow!`/`.with_context()` with rich formatted messages, in tiles and sequences (e.g. `prompt_prepare/raster/tiles.rs:28-32,154-155,203-207`, `utils.rs` throughout) |
| Sites | pervasive (every fallible tile) |
| Routines affected | all |
| Real-raster mapping | `raster::exec::Result<T>` (= `Result<T, String>`) for **terminal** outcomes surfaced by tile logic; `raster::runtime::Error` is the infrastructure channel and is never produced by tile code |
| Status | verified |
| Evidence | P2 (runs R1 + R3: user `Result::Err(String)` rides the ABI and surfaces at the sequence boundary — observable there via `match` on the materialized `Result`, or propagated with `?`/`expect`. Inside a sequence body a fallible `call!` binding is `AuthRef<Result<T, String>>`: only `?` is rewritten by the macro; native `match` on the binding does not compile — errors are observed at boundaries, not mid-sequence); `external_selection.rs:373-378` |
| Porting rule | `bail!(fmt…)` → `return Err(format!(…))` (alloc-only, no_std-safe). `.with_context()` chains flatten into the message. **The committed line:** a tile `Err` is a terminal, committed outcome — replayable and fault-provable; infrastructure failures (I/O, deserialization of staged inputs, storage) surface as runtime errors and abort without committing a terminal outcome. Error *message content* may be committed (it rides the trace), so messages must be deterministic: no addresses, no timestamps, no platform-dependent formatting. Sim guard checks that protect *invariants of correct staging* (e.g. `pieces_per_tile == 0`) stay terminal errors. |
| Notes | Sim `.expect()`/`panic!` in leaf helpers (`utils.rs:307`) must become `Err` or be proven unreachable — panics in guests are fault-indistinguishable from wrong execution. |

### C24. Tile invocation counting (H5)

| Field | Content |
|---|---|
| Construct | `start_tile_invocation_counting` / `stop_tile_invocation_counting` / `record_tile_invocation` — thread-local counter incremented by every invoker macro; feeds `trace::tile_invoked` logging (`src/dsl/runtime.rs:45-70`, `src/runtime/trace.rs:99`) |
| Sites | DSL-internal + test instrumentation; not in routine tiles |
| Routines affected | none directly (WS4 harness concern) |
| Real-raster mapping | Real trace events: `TileExec`, `RecurTileExec`/`RecurTileIterationExec`, `SequenceStart`/`SequenceEnd`, `RecurSequenceStart`/`End`/`Exec` in `trace.bin`/`trace.ndjson`; counts derived by parsing the trace |
| Status | verified |
| Evidence | P1/P5 (run R1 trace: 33 `TileExec`, 4 `RecurTileExec`, 16 `RecurTileIterationExec`, 9 `SequenceStart`/`End` pairs, 3 `RecurSequenceStart`/`End` — all matching the program's source structure exactly) |
| Porting rule | WS4 structural checks compare *derived counts from the trace file*, not an in-program counter. Recipe: run with `--trace-format json` (writes `trace.ndjson` under `target/raster/runs/<run-id>/`), count events by type/name. No in-tile construct is ported. |
| Notes | Iteration events are rewritten to `RecurTileIterationExec` inside recur scopes (`raster-runtime/src/tracing.rs:151-168`) — counts distinguish loop iterations from standalone execs, which is *more* structure than the sim counter had. |

### C25. Program entry: parameterized sim `main` vs zero-arg real `main`

| Field | Content |
|---|---|
| Construct | Each routine's `#[sequence] pub fn main(args…)` takes typed inputs from the host (`prompt_prepare/raster/tiles.rs:20-25`, `prefill_range/raster/tiles.rs:48-55`) |
| Sites | 1 per routine with sequences |
| Routines affected | all with sequences |
| Real-raster mapping | Real `#[sequence] fn main()` takes **no parameters** (`raster-macros/src/lib.rs:3592-3618`); inputs arrive as `external!` bindings + `select!` in `main`, staged via `input.json` + `input_manifest.json` |
| Status | verified |
| Evidence | All probes (run R1 program shape); `raster-tokenizer/src/main.rs`; `crates/raster-programs/prompt_prepare/src/main.rs` (WS0 placeholder) |
| Porting rule | The sim main's parameter list becomes the program's external-input schema: one committed external per parameter (or a single input struct). The library keeps a parameterized inner sequence (tokenizer-PoC shape) so tests can drive it natively via `materialize_auth_result(__raster_sequence_auth_*)`. This row is the WS2 interface definition point for each routine. |
| Notes | Non-main sequences cannot be called natively outside sequence context (compile-fail enforced) except through the `__raster_sequence_auth_*` mangled fns. |

### C26. `routine_scope` / host-trace calls inside tiles

| Field | Content |
|---|---|
| Construct | `trace::routine_scope(RoutineId…, …)` called inside a tile (`prefill_range/raster/tiles.rs:3232-3238`) — host logging from tile code |
| Sites | 1 non-test site |
| Routines affected | `prefill.range` |
| Real-raster mapping | Drop from tile bodies (host logging is not tile work); real per-tile tracing subsumes the observability |
| Status | verified-as-dissolved |
| Evidence | inspection; real trace events cover the same information (C24) |
| Porting rule | No host-trace/logging calls in real tiles. `raster`'s `debug!`/`println!` (`[output]` capture) may be used sparingly during bring-up but must not influence outcomes. |
| Notes | — |

### C27. `RasterSizingControls` as tile argument

| Field | Content |
|---|---|
| Construct | Runtime chunk-width controls passed into init tiles and threaded through state (`prefill_range/raster/tiles.rs:53,3182`, `decode_layer_range/raster/tiles.rs:685`; per-routine `*_per_tile` fields in states) |
| Sites | 2 entry sites + pervasive state fields |
| Routines affected | `prefill.range`, `decode.layer_range` (+ per-tile fields in most others) |
| Real-raster mapping | Largely **authoring-time tile granularity** (charter WS8 note): chunk width is how many items one recur iteration consumes, fixed by how the input list is chunked at staging or by tile code |
| Status | mapped-unverified (WS8 owns final shape) |
| Evidence | pending WS8 profiling |
| Porting rule | WS3 ports pick a fixed, documented chunk shape per loop (list-of-chunks staging or per-item lists), keeping the value in one obvious place so WS8 retuning is a local edit. Runtime `--config` sizing keeps meaning only where the host stages chunked lists. |
| Notes | Sim's zero-checks (`pieces_per_tile == 0` bails) disappear where the chunk shape becomes structural. |

### C28. Enum-state branching inside tiles

| Field | Content |
|---|---|
| Construct | Branch-without-sequence-branch via enum states: `GemmaBpeMergeIterationState::Complete/Applying` (`prompt_prepare/raster/tiles.rs:435-459`), `PrefillLayerStep::Complete/Compute/Computed` (`prefill_range_finalize/raster/tiles.rs:14-34`) |
| Sites | ~10 enum-state types |
| Routines affected | `prompt.prepare`, `prefill.range`, `prefill.range_finalize`, `decode.layer_range` |
| Real-raster mapping | Direct: serde enums pass through the postcard ABI; branching inside tile bodies is unrestricted |
| Status | verified |
| Evidence | P2 (run R1 exercised an enum-carrying tile type) |
| Porting rule | Keep. Enums used in `select!`-reachable positions need `Selectable` support consideration — select into enum payloads is not supported; select whole enum values and match inside tiles. |
| Notes | — |

### C29. Cross-routine tile/sequence imports

| Field | Content |
|---|---|
| Construct | `prefill_range` invokes `prefill_range_finalize`'s tile (`prefill_range/raster/tiles.rs:7,91`); `decode_transition_finalize::run_raster` calls `decode_layer_range::finalize_state_refs_with_roots` (`decode_transition_finalize/raster/tiles.rs:13`) |
| Sites | 2 |
| Routines affected | `prefill.range` ↔ `prefill.range_finalize`, `decode.transition_finalize` ↔ `decode.layer_range` |
| Real-raster mapping | Program-crate composition: shared tiles live in one program crate's lib and are depended on by the other (`raster-program-*` crates are normal Rust libs), or the routines share one program crate with two entry binaries — decided per pair in WS3 |
| Status | mapped-unverified (build-shape decision, low risk) |
| Evidence | workspace layout supports lib-to-lib deps; pending WS3 |
| Porting rule | The occurrence-selector semantics of `prefill.range_finalize` (ADR: provisional) get pinned when its WS3 plan chooses the crate shape. Record the ruling here. |
| Notes | — |

### C30. Shared tile-shaped helpers (`raster_kernels`, `raster_contracts`, tensor artifacts)

| Field | Content |
|---|---|
| Construct | Plain fns called *inside* tiles: det-num kernels (`shared/raster_kernels/transformer.rs`, ~3200 lines — RMS norm, attention, projection, artifact-state init/compute/finalize helpers), source request types (`raster_contracts`), row reads (`shared/tensors/raster_tensor_artifacts.rs`) |
| Sites | pervasive in compute routines (`prefill_range` imports ~30 kernel fns, `tiles.rs:16-35`) |
| Routines affected | `input.embedding`, `prefill.prepare_aux`, `prefill.range`, `prefill.finalize`, `decode.layer_range` |
| Real-raster mapping | In-tile library code, compiled into the program crate: must become `no_std + alloc`-compatible and lose its `ArtifactIo` internals (replaced per C15–C17) |
| Status | mapped-unverified (the det-num math is pure and portable; the artifact-state helpers embed store calls and need the C15–C17 rewrite) |
| Evidence | inspection; pending first compute-routine WS3 port |
| Porting rule | Split each kernel helper into (a) pure det-num math — port verbatim into a shared program-support module, and (b) artifact-state manipulation — rewrite as draft/select operations per C16/C17. Anything a tile calls transitively ships in the guest; std-only deps (`memmap2`, `rayon`, `tokenizers`) must not be reachable. |
| Notes | Largest volume of WS3-era rewriting outside `prefill.range`/`decode.layer_range` tiles themselves. |

### C31. Sim `External<T>` / `external()` / `external!` (sim macro)

| Field | Content |
|---|---|
| Construct | Typed name-only external references (`src/dsl/runtime.rs:18-43`, `macros.rs:65-73`) |
| Sites | 0 routine sites; `src/dsl/tests.rs:98-103` only |
| Routines affected | none |
| Real-raster mapping | Real `external!(T, "name")` — the concept the sim reserved is exactly what real raster ships |
| Status | verified (trivially; the sim construct is unused) |
| Evidence | census (zero production sites); P3 for the real construct |
| Porting rule | No sim code to port; new externals in ports follow C13/C25. |
| Notes | Name collision warning: sim `external!` and real `external!` differ in shape (`external!("n")` vs `external!(T, "n")`) — irrelevant given crate separation (WS0 invariant), noted for reviewers. |

### C32. `auth_read!` macro (sim)

| Field | Content |
|---|---|
| Construct | Thin alias for `ArtifactIo::auth_read` (`src/dsl/macros.rs:75-80`) |
| Sites | imported in `prefill_range`/`output_finalize` preludes; call sites overwhelmingly use `ArtifactIo::auth_read` directly |
| Routines affected | as C13 |
| Real-raster mapping | As C13 (no separate construct) |
| Status | verified (by identity with C13) |
| Evidence | C13 |
| Porting rule | — |
| Notes | Folded into C13 for readiness accounting. |

### C33. CFS source-layout constraint (Phase B discovery)

| Field | Content |
|---|---|
| Construct | (real-side constraint, no sim analogue) The CFS builder discovers tiles/sequences by parsing **top-level `fn` items** in each non-`mod.rs` `.rs` file under `src/` (`raster/crates/raster-compiler/src/ast.rs:109-126`); functions inside inline `mod { … }` blocks and anything in `mod.rs` are invisible to it |
| Sites | applies to every program crate |
| Routines affected | all (WS3 authoring constraint) |
| Real-raster mapping | Layout rule, not a construct |
| Status | verified |
| Evidence | Discovered empirically: the first probe build kept tiles inside inline `mod` blocks in `lib.rs`; the program *ran to completion* but trace coordinate assignment panicked at exit (`raster-core/src/cfs.rs:354`, "Wrong coordinates for sequence child 'RecurTile(\"scan_max\")'") because the CFS never saw those tiles. Splitting each module into its own file fixed it (run R1 clean). |
| Porting rule | One module per file in program crates; no tiles/sequences in `mod.rs` or inline `mod` blocks. The sim layout (`tiles.rs`/`types.rs`/`utils.rs` + re-exporting `mod.rs`) already matches this shape. |
| Notes | Failure mode is late and confusing (runtime coordinate panic, not a build error) — worth a lint or doc fix upstream eventually; recorded here so WS3 authors never hit it. |

---

## 5. Per-routine readiness table

"Ready" = every row the routine uses is `verified`. Blocking rows are those
still `mapped-unverified` or gapped for that routine.

| Routine id | WS3-ready? | Blocking rows / conditions |
|---|---|---|
| `prompt.prepare` | **conditionally** | G1 restructuring ruling for the BPE merge loop (C9) — restructure approved in §6/G1; C13 data layout (tokenizer external) is WS2 work with a proven idiom (raster-tokenizer). No unverified *mechanism* blocks it. |
| `input.embedding` | conditionally | C22 (weights external staging, WS2); C15/C16 rewrite is mechanical. Pair-state recur (C8) fold rule applies. |
| `prefill.prepare_aux` | conditionally | C30 (kernel helpers split), C22 (PLE source staging, WS2) |
| `prefill.range` | conditionally | C30 (largest kernel surface), C27 (chunk-shape decisions), C29 (crate pairing with `prefill.range_finalize`) |
| `prefill.range_finalize` | conditionally | C29 (crate/occurrence ruling); otherwise trivial (1 tile) |
| `prefill.finalize` | conditionally | C22 (source staging), C30 |
| `decode.select_token` | **yes (first fully unblocked)** | Uses only C1–C8, C15–C17, C23 — all verified; no external sources in tiles |
| `decode.layer_range` | conditionally | C30, C27, C22; KV-cache artifact schemas (C21) are the main design work |
| `decode.transition_finalize` | yes (trivially) | No DSL constructs; its port is host-adapter glue over `decode.layer_range`'s program |
| `output.finalize` | conditionally | C13/C22 (tokenizer external for detokenize — same WS2 artifact as `prompt.prepare`) |

**`prompt.prepare` blockers, explicitly (first WS3 target):** none at the
mechanism level after Phase B. Its port requires (1) the G1-approved bounded-loop
restructuring of `merge_bpe_tokenize_prompt`, (2) WS2 staging of the tokenizer
committed external per the raster-tokenizer idiom, (3) the C15 roots-leg removal
throughout. All three have named, verified shapes.

---

## 6. Gap register

### G1 — Condition-driven ("until-done") recur vs list-driven real recur

- **Missing behavior:** sim recur loops iterate *until the tile/sequence signals
  done* (`(bool, State)`, `src/dsl/macros.rs:144-208`) with no iteration list.
  Real recur (`call_recur!`/`call_recur_seq!` → `run_recur_list*`,
  `raster/crates/raster/src/input.rs:1896-2193`) iterates a materialized
  `Vec<T>` — bounded by list length, early-exit only via `RecurControl::Break`
  (tiles) and **no early exit at all** for recur sequences.
- **Affected routines:** all 8 recur users; decisive for `prompt.prepare`
  (BPE merge loop: data-dependent convergence).
- **P1 findings (run R1):** the real loop executes at native runtime
  (doc `tile-authoring.md:78-86` is stale); `Break` stops iteration and
  finalizes (`sum_with_break`: 3 `RecurTileIterationExec` events for an 8-item
  list); state threads correctly; an empty internal list finalizes cleanly with
  the initial state (1 `RecurTileExec`, 0 iteration events); a bounded
  dummy-index list + `Break` expresses "until done within a bound"
  (`until_done_bounded`: 5 iteration events against a bound of 8) with
  per-iteration trace events for executed iterations only.
- **Resolution path (catalog-approved restructuring, not a raster feature
  request):** every sim recur loop in scope has a derivable upper bound at loop
  start — chunk loops: `ceil(count / per_tile)`; layer loops: `layer_count`;
  BPE merge loop: `initial_piece_count − 1` (each merge removes one piece).
  Porting rule: stage/store an index or chunk list of that length as the recur
  input; convergence-before-bound uses `RecurControl::Break` (tile loops) or
  in-tile no-op continuation (sequence loops, which cannot break). The trace
  then commits exactly the executed iterations. If any future loop has *no*
  derivable bound, that becomes a raster feature request ("condition-driven
  recur driver") — none in current scope.

### G2 — Authenticated artifact store: out-of-order / indexed leaf append

- **Missing behavior:** sim `append_leaf(builder, idx, leaf)` takes an explicit
  index; real `DraftAppendField::push` is order-is-index. All in-scope sim call
  sites append sequentially (audited during Sweep 2), so nothing is blocked —
  recorded so WS3 authors know indexed append is **not** available if a port is
  tempted to reorder.
- **Affected routines:** none currently. **Resolution path:** if a genuine
  out-of-order producer appears, restructure to produce in order (extra pass) or
  raster feature request for indexed draft append.

### G3 — Selection into enum payloads / tuples

- **Missing behavior:** `select!` paths address struct fields and list indices;
  tuple components and enum variant payloads are not selectable.
- **Affected routines:** all (sim returns tuples pervasively).
- **Resolution path (porting rule, C5/C11/C28):** convert cross-call tuples to
  named `Selectable` structs; select whole enums and match inside tiles. No
  raster change needed.

### G4 — `cargo raster run` exit-code propagation (WS0-named, re-confirmed)

- **Missing behavior:** whole-program run returns exit 0 even when the program
  binary fails (`raster/crates/raster-cli/src/commands/run.rs` returns `Ok(())`
  unconditionally after spawn). Re-observed in Phase B: both the tampered-input
  rejection (run R2) and the terminal-Err panic (run R3) printed the failure
  under `Error:` but exited 0.
- **Affected:** WS2+ ingestion (not a WS1 construct row). **Resolution path:**
  already recorded in the WS0 ADR; ingestion validates artifacts, never exit
  codes. Candidate `raster` fix remains open as a named behavioral spec:
  "non-zero CLI exit when the executed program binary exits non-zero."

### G5 — Guest-side status of recur/draft drivers

- **Missing behavior:** `run_recur_list*`, draft create/finalize, and internal
  storage drivers are `std`-gated; `no_std` builds panic in those paths at the
  pinned rev. Native host execution (the WS3/WS4 target) is unaffected.
- **Affected:** WS7-era fault proofs replay *single tiles* in the guest (replay
  entries + draft replay handles exist for exactly this: `input.rs:597-647`,
  no_std replay ops), so whole-loop drivers are host-side by design. Recorded so
  no one assumes guest-side loop execution.
- **Resolution path:** none needed for WS3/WS4; WS7 verifies single-tile replay
  in-guest (out of WS1 scope).

---

## 7. Probe crate index

Probe crate: `crates/raster-programs/_probes/` (package `raster-ws1-probes`,
binary `raster-ws1-probes`). Not a routine: it appears in no host, guard-test,
or parity-harness list. Layout: `no_std + alloc` lib with one probe module per
file (C33), a std-gated `src/main.rs` with all probe sequences plus the
composing `#[sequence] fn main()`, and a feature-gated staging bin (`stage`,
feature `stage`) that writes `input.json`, `input_manifest.json`, and postcard
payload files. Staging flags: `--tamper` flips one byte of `p3_config.bin`
after the manifest is written; `--p2-div-zero` stages a zero divisor so the P2
ok-path errors terminally.

Re-run (from `crates/raster-programs/_probes/`, requires `cargo-raster` built
from the pinned `../raster` checkout on PATH):

```sh
# stage inputs (writes input.json, input_manifest.json, *.bin into the crate dir)
cargo run --features stage --bin stage -- .

# R1: the full probe program, with trace + commit
cargo raster run --backend native \
  --input input.json --input-manifest input_manifest.json \
  --commit probes_commit.bin --trace-format json --verbose

# R1-audit: replay against the commitment (expect "Verification Success")
cargo raster run --backend native \
  --input input.json --input-manifest input_manifest.json \
  --audit probes_commit.bin

# R2: tamper check (restage with a flipped byte, expect commitment rejection)
cargo run --features stage --bin stage -- . --tamper
cargo raster run --backend native \
  --input input.json --input-manifest input_manifest.json

# R3: terminal-Err propagation (expect the checked_div error to surface)
cargo run --features stage --bin stage -- . --p2-div-zero
cargo raster run --backend native \
  --input input.json --input-manifest input_manifest.json

# restage clean afterwards
cargo run --features stage --bin stage -- .
```

Executed 2026-07-05 (all four runs). Trace inspected from
`target/raster/runs/<run-id>/trace.ndjson`; R1 event totals: 33 `TileExec`,
4 `RecurTileExec`, 16 `RecurTileIterationExec`, 3 `RecurSequenceStart`/`End`,
9 `SequenceStart`/`End`. R1-audit reported `Verification Success`.

| Probe | Module | Demonstrates (rows) | Key observations |
|---|---|---|---|
| P1 | `src/p1_recur.rs` | C2, C4, C7–C10, C24; grounds G1/G5 | Recur loop executes natively (doc stale); `Break` truncates iterations (`sum_with_break` 3/8, `until_done_bounded` 5/8) and finalizes; state threads; empty-list recur finalizes with initial state, 0 iteration events; recur sequences run all items (no Break; `collect_prefixed` 3/3); bounded-list-plus-Break expresses until-done |
| P2 | `src/p2_abi.rs` | C1, C12, C23, C28 | 2-/3-arg tiles round-trip via postcard tuples; enum arg round-trips; `exec::Result` Ok via `?`; Err observed at the `call_seq!` boundary in `main` without aborting (R1) and propagated terminally (R3); run artifacts still produced on failure and CLI exited 0 (G4) |
| P3 | `src/p3_external.rs` | C13, C14, C22, C25, C31 | `external!` + `select!` against staged manifest verifies commitments; struct-field, list-index, and nested-struct selection; **R2:** flipped byte → "External input 'p3_config' failed integrity check. Expected SHA256 …" at resolve time, before tile execution |
| P4 | `src/p4_draft.rs` | C11, C15–C18, C21; grounds H1 | `new!` → set/push in tiles → `finalize` → `select!` round-trip; `store_internal_value` + `internal!` re-binding + indexed select; commit artifact (408 bytes) covers the whole trace incl. internal ops — audit replay verifies against it; no explicit roots threading anywhere |
| P5 | `src/p5_sequences.rs` | C3, C5, C6, C24 | Sequence-in-sequence value flow via `AuthRef` (`outer_pipeline` → `inner_transform` ×2); nested `SequenceStart`/`End` trace shape; derived event counts match source structure |

The C33 layout constraint (CFS ignores inline `mod` blocks and `mod.rs`) was
discovered while building this crate — see that row for the failure mode.

CI builds the probe crate and checks its `no_std` lib surface exactly like the
WS0 placeholder (`.github/workflows/ci.yml`, program-crate steps). Probe *runs*
are manual (this index is the re-run recipe); they are not wired into any test.
Staged inputs and run artifacts are gitignored within the probe crate.

---

## 8. Exit-criteria reconciliation

1. **Census ↔ rows:** every construct family and structural pattern in the §1
   census maps to a row C1–C33; the final sweep counts (§1 table) reconcile
   with the seed expectations given the scope note. ✔
2. **H1–H5:** H1 → §3 decomposition (C13–C21, verified in-tile legs, WS2/WS5
   handoffs named); H2 → C2/C4/C7–C10 + G1 (verified with approved
   restructuring); H3 → C22 (verified mechanism); H4 → C23 (verified);
   H5 → C24 (verified). ✔
3. **P1–P5 executed** (runs R1, R1-audit, R2, R3) with observations recorded
   (§7). ✔
4. **Readiness table** complete; `prompt.prepare` blockers explicit (§5). ✔
5. **Suite/gate green; probe crate builds** incl. `--no-default-features`
   (see WS1 verification note in the amendment log). ✔
6. **State-carrying doc** with amendment convention (header). ✔

## 9. Amendment log

- **2026-07-05** — Initial version: Phase A enumeration and mappings; Phase B
  probe results folded in (runs R1, R1-audit, R2, R3); rows upgraded per
  evidence; C33 (CFS source-layout constraint) added from a Phase B discovery;
  G1 resolution (bounded-list restructuring) approved as the catalog ruling
  for until-done loops.
- **2026-07-06** — WS2 landed (`docs/plans/ws2-staging.md`); the rows that
  deferred to "WS2 scope" now have concrete owners. C13/C22 (per-source data
  layout, borrowed-source staging): `StagedInputs` +
  `crates/raster-programs/gemma_externals/` (tokenizer schema/encoder built;
  weight-family externals deferred to their routines' WS3 plans per the WS2
  encoder scope split). C18 (host-side staging of native values):
  `StagedInputs::add_postcard`/`add_raster_encoded`. C19 (roots into
  payloads → post-run commitment extraction): `ingest` captures the input
  commitment map and the commit-artifact fingerprint; the WS5 payload
  contract remains pending as recorded. C23 (H4 error contract): realized as
  `RasterCoreError { Infrastructure, Terminal, Verification }` with
  test-asserted classification, including the ruling that an undecodable
  staged input is infrastructure. C25 (zero-arg `main` + staged inputs):
  the output-file half of the interface is the `raster-program-support`
  convention. G4 re-confirmed and extended: the CLI can also exit non-zero
  (trace-commitment panic on guest integrity rejection), so exit status is
  unusable in both directions; ingestion validates artifacts.
