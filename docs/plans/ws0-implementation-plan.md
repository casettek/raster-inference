# WS0 Implementation Plan — Workspace & Toolchain Enablement

Status-carrying execution plan for Workstream 0 of the raster-core migration.
Companion documents: the migration charter (`raster-core-migration-charter.md`)
and the WS0 agent prompt. This document records the Phase A grounding findings,
the resolved decisions with rationale, the ordered change list, the risk
register, and the baseline-capture results.

---

## 1. Baseline capture (work item 0)

Captured on branch `migration-ws0` (clean tree, HEAD `a209e5b "decoder views"`)
under `rustup run 1.91.1` (rustc 1.91.1, ed61e7d7e 2025-11-07) on
aarch64-apple-darwin, 2026-07-05:

| Command | Result |
|---|---|
| `cargo test` (debug, full suite) | green — all test binaries pass, 0 failures |
| `cargo test --release --test e2e_checkpoint_parity` | green — 4/4 legs pass (native-vs-raster, byte reproducibility, thread invariance, `prefill.range` detour smoke) |
| `cargo test --release --test golden_traces` | green — 5/5 goldens match |

Golden baseline hashes (SHA-256), for byte-identity verification throughout WS0:

```
a399f89e961338cd4ffbb3af20712472289fec4b848b55ce41251322e165901d  tests/goldens/detour-prefill-range.json
b9649843ef2c9ed1a0ef2a4b4a3961c2627615b8a0d68453e9f571c594205711  tests/goldens/native-det-full.json
d0b8fb9ffb487e4ffc9a24dc7b85594f702df08ba3fc254104d71abc2d7031ff  tests/goldens/native-det-terminal.json
90413edb2c45671362a563a9f02baf5710a7e2f24f01482d9084639915187637  tests/goldens/raster-full.json
9e1e3b5d2a28c9f6b8a99574a8731161cc4c5acd8c78d08156fee8b20c2ecca9  tests/goldens/raster-terminal.json
```

The gate is green under 1.91.1; no WS0 blocker.

## 2. Grounding findings (Phase A)

### raster-inference

- Single package (no `[workspace]`), edition 2021, no `rust-toolchain.toml`.
  One path dep: `crates/dsl-macros`. Key deps: serde 1 (derive),
  postcard 1.1.3 (alloc), clap 4.6 (derive), anyhow 1.
- **CLI drift vs the charter/prompt:** `src/main.rs` uses subcommands
  (`claim` / `detour` / `audit`). Detour targeting is `detour --at
  ROUTINE[:OCC]`. There are no top-level `--raster` / `--raster-at` /
  `--raster-*-per-tile` flags: full-raster mode is programmatic
  (`InferenceControls.raster`), and sizing controls come from `--config` TOML
  (`[ranges]`, `[tile_sizing]`) via `ExecutionTuning`
  (`src/runtime/roles/mod.rs`). Full-raster/detour mutual exclusion is
  enforced at runtime in `src/runtime/sequence.rs` (`run`).
- Detour machinery: `RasterDetourSpec::parse` (occurrence suffix, default
  `:1`) and `RasterDetourController` (`should_detour`,
  `ensure_matched_if_active`, `reject_if_selected_unsupported`) in
  `src/runtime/checkpoints.rs`; `ExecutionPolicy::mode_for` in
  `src/runtime/executors/mod.rs`; detour call sites in
  `src/runtime/executors/native.rs` and `src/runtime/sequence.rs`.
- Guard test: `src/routines/mod.rs` uses an `include_str!` const table of the
  10 routine hosts and asserts exactly one `pub fn run_raster(` per host
  (except `prefill_range_finalize`, which has no host entrypoint) plus a
  forbidden-wrapper-pattern list.
- Parity gate: 4 legs in `tests/e2e_checkpoint_parity.rs`; goldens in
  `tests/goldens/` (5 files) checked by `tests/golden_traces.rs`. CI
  (`.github/workflows/ci.yml`): 3 jobs (fmt+clippy, debug tests, release
  parity+goldens) on `dtolnay/rust-toolchain@stable` — no pin today.

### raster (real toolchain)

- 12-member workspace, `rust-toolchain.toml` pins **1.91.1**. HEAD
  `536214533f06381a913e5873772e283025ebe061`, no tags.
- `crates/raster` features: `default = ["std"]`, `alloc`, `profiling`.
  `exec::Result<T> = Result<T, String>`. Crate-root `#[macro_export]` macros
  (`call!`, `call_seq!`, `call_recur!`, `call_recur_seq!`, `external!`,
  `internal!`, `new!`, `println!`) collide with raster-inference's sim DSL
  crate-root macros — confirming that real tiles must live in separate
  program crates (charter invariant 6).
- Workspace dep versions: serde 1.0 (default-features = false, derive +
  alloc), postcard 1.0 (default-features = false, alloc) — semver-unifiable
  with raster-inference's serde 1 / postcard 1.1.3.
- `cargo raster run` (`raster-cli`, a `cargo-raster` subcommand): native
  backend only for whole-program runs; `--input`, `--input-manifest`,
  `--commit`; builds the project with `cargo build --release`, runs the
  binary, writes `target/raster/runs/<run_id>/trace.bin` (+ profile
  artifacts), and `--commit` writes a postcard `TraceCommitment`.

### raster-tokenizer (idiom reference only)

- `raster = { path = "../../raster/crates/raster", default-features = false }`,
  features `default = ["std"]`, `std = ["raster/std"]`, `encode` gating the
  staging bin. `cfg_attr(not(feature = "std"), no_std)` lib; alloc-only tile
  module; `#[sequence] fn main()` host binding externals via
  `select!`/`external!`; `bin/encode_tokenizer.rs` stages `input.json` +
  `input_manifest.json` (logical-name → path binding, and logical-name →
  sha256 commitment respectively).

## 3. Resolved decisions

1. **CLI mapping (user ruling, 2026-07-05).** The charter froze
   `--raster-core-at <routine-id[:occurrence]>` against a CLI that has since
   moved to subcommands. Ruling: literal charter compliance —
   `detour --raster-core-at ROUTINE[:OCC]` as a second targeting flag on the
   existing `detour` subcommand, `conflicts_with = "at"`. The charter's
   "`--raster` / `--raster-at`" correspond today to `InferenceControls.raster`
   (programmatic) / `detour --at`; mutual exclusion maps accordingly.
2. **Dependency pinning.** Path dependency
   `raster = { path = "../raster/crates/raster", default-features = false }`
   declared once in `[workspace.dependencies]`, plus a CI step that checks out
   `casettek/raster` at a pinned rev into the sibling path. The pin lives in
   exactly one place: the `RASTER_PIN` env var at the top of
   `.github/workflows/ci.yml`, initially
   `536214533f06381a913e5873772e283025ebe061`.
   *Bump procedure:* update `RASTER_PIN`, update the pin recorded in the
   migration ADR with a dated note, run the full suite + parity gate locally
   against the new rev, and land the bump as its own commit. Gate-green is the
   acceptance test for any bump.
   *Rationale:* matches the tokenizer-PoC idiom, zero-friction local iteration
   against the sibling checkout, avoids cargo git-auth friction in CI.
   Cargo.lock's non-pinning of path deps is mitigated by the CI pin + the
   toolchain pin.
3. **Workspace layout.** Root package stays at the repo root; root
   `Cargo.toml` gains `[workspace]` with `resolver = "2"` and members
   `crates/dsl-macros` and `crates/raster-programs/*`. No path churn for CI,
   docs, or the `include_str!` guard test.
4. **Controller extension.** `DetourBackend { Sim, RasterCore }` carried on
   `RasterDetourSpec` (existing `parse` yields `Sim`; `parse_raster_core`
   yields `RasterCore`). Occurrence matching and `ensure_matched_if_active`
   stay single-implementation in `RasterDetourController`. `StepMode` gains a
   `RasterCore` variant produced by `ExecutionPolicy::mode_for`; every WS0
   call site handling `RasterCore` bails with
   `"selective raster-core detour for {routine} is not implemented yet"`.
5. **Feature gating: always-compiled.** The main crate never depends on
   `raster` (invariant 6); only program crates under `crates/raster-programs/`
   do. serde/postcard versions verified unifiable (serde 1, postcard 1.x with
   alloc on both sides). No cargo feature added.
6. **Parity harness enumeration.** Explicit
   `const ENABLED_ROUTINES: &[(&str, usize)] = &[];` in
   `tests/raster_core_detour_parity.rs`; WS4 appends entries one routine at a
   time. No magic discovery.
7. **Sizing controls.** Nothing new: `--config` TOML is subcommand-agnostic
   and already threads `ExecutionTuning` into `InferenceControls`. It is
   accepted alongside `--raster-core-at` and consumed only from WS3+/WS8.

## 4. Ordered change list (commits; gate green after each)

1. **Baseline + this plan doc** (this commit).
2. **Toolchain pin.** `rust-toolchain.toml` (1.91.1, rustfmt + clippy);
   CI switches from `@stable` to the toolchain file.
3. **Workspace conversion.** `[workspace]` in root `Cargo.toml`;
   `crates/dsl-macros` as member; `[workspace.dependencies]` for `raster`;
   goldens byte-identical after the lockfile change.
4. **Placeholder program crate + CI raster checkout.**
   `crates/raster-programs/prompt_prepare/` (package
   `raster-program-prompt-prepare`): tokenizer-idiom shell. CI checks out
   `casettek/raster` at `$RASTER_PIN` into the sibling dir and builds the
   program crate including a `--no-default-features` check.
5. **CLI surface.** `--raster-core-at` on `DetourArgs`
   (`conflicts_with = "at"`); parse tests for both flag forms, occurrence
   suffixes, all 10 routine ids, unknown-id rejection, conflict rejection.
6. **Controller and plumbing.** `DetourBackend` on `RasterDetourSpec`;
   `StepMode::RasterCore`; every detour decision point bails cleanly for
   `RasterCore`; tests that each of the 10 routine ids errors with the exact
   unimplemented message; sim detour behavior untouched.
7. **Routine host contract.** `src/routines/<r>/raster_core/mod.rs` stubs;
   guard test extended with a `RASTER_CORE_MIGRATED` activation list —
   migrated hosts must expose exactly one `pub fn run_raster_core(`,
   unmigrated exactly zero; forbidden-wrapper-pattern list applied to the new
   namespace.
8. **Subprocess scaffolding.** `src/runtime/raster_core/` host module (no
   `raster` dep): run-directory conventions and a `cargo raster run`
   subprocess wrapper with a clear error when `cargo-raster` is absent.
9. **Parity harness scaffolding.** `tests/raster_core_detour_parity.rs`,
   hermetic tiny-gemma-dev setup, parametrized over `ENABLED_ROUTINES`
   (ships empty), divergence named by checkpoint id + occurrence.
10. **Docs.** Migration ADR in `docs/plans/` (frozen decisions, resolved
    decisions, CLI-drift ruling, pin-bump procedure, named gaps);
    `docs/plans/raster-core-routine-template.md` with a real-raster
    authoring-constraints section.

## 5. Risk register

- **Toolchain 1.89→1.91.1 changes traces** — ruled out by the baseline
  capture above (gate + goldens green under 1.91.1 before any change).
- **Workspace/lockfile feature unification perturbs main-crate deps** —
  resolver 2 + golden byte-identity re-check after the workspace commit.
- **`RasterDetourSpec` construction churn** from the new `backend` field —
  keep `parse` as the `Sim` constructor and add `parse_raster_core`, so
  existing call sites and tests stay valid.
- **raster-cli exit-code gap (named gap).** `cargo raster run` prints the
  child process's non-zero exit status but does not propagate it
  (`crates/raster-cli/src/commands/run.rs`, whole-program run path returns
  `Ok(())` after a failed child). WS2+ subprocess ingestion cannot rely on
  the CLI exit code alone to detect guest failure; it must validate the
  produced artifacts. Recorded here and in the migration ADR — not shimmed.
- **Charter drift** — the CLI-shape ruling is recorded and dated in the
  migration ADR per the charter's contradiction rule.

## 6. Exit criteria

1. Repo builds as a workspace under pinned Rust 1.91.1.
2. `--raster-core-at` parses for all 10 routine ids (both flag forms,
   occurrence suffixes) and errors cleanly as unimplemented; conflicts with
   `--at` rejected with tests.
3. Placeholder program crate compiles against real `raster` (including a
   `default-features = false` check).
4. Full existing test suite and parity gate green; committed traces
   byte-identical to the baseline hashes in §1.
5. Guard test extended and green; `tests/raster_core_detour_parity.rs`
   compiles with zero enabled routines.
6. ADR, routine template, and CI changes merged; the pin-bump procedure
   documented in the ADR.
