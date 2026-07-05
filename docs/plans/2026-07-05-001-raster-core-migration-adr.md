# ADR: Raster-Core Migration — WS0 Decisions

- **Status:** accepted (WS0 landed)
- **Date:** 2026-07-05
- **Companions:** the migration charter (`raster-core-migration-charter.md`,
  status-carrying), `docs/plans/ws0-implementation-plan.md` (grounding and
  baseline capture), `docs/plans/raster-core-routine-template.md` (per-routine
  plan template).

## Context

`raster-inference`'s verifiable execution path is authored against a
**simulated** raster layer (`src/dsl/` + `crates/dsl-macros` + per-routine
`raster/` modules). The migration adds a third execution path —
**raster_core** — in which each routine's tiles are authored on the real
`raster` toolchain, compiled by the real toolchain, executed against committed
inputs, and reconciled with native execution through the existing
checkpoint-parity machinery. WS0 lands all plumbing and zero behavior.

## Frozen decisions (from the charter; not revisited here)

1. **In-place migration.** No greenfield repo; goldens, parity gate, CI, role
   APIs, and checkpoint plumbing are the verification substrate.
2. **Path name `raster_core`.** Host adapters at
   `src/routines/<routine>/raster_core/`; migrated hosts expose exactly one
   `pub fn run_raster_core(` entrypoint.
3. **CLI selector `--raster-core-at <routine-id[:occurrence]>`,** mirroring
   the sim selector semantics; mutually exclusive with the sim selector. The
   full-path flag (`--raster-core`) is WS6 scope.
4. **Tile source location:** per-routine program crates under
   `crates/raster-programs/<routine>/`, shaped like the tokenizer PoC. Real
   raster is never imported by the main crate — the sim DSL's crate-root
   `#[macro_export]` macros (`call_seq!` et al.) collide with the real
   `raster` prelude's, and guest compilation requires a separate no_std lib
   regardless.
5. **Execution mechanism (WS0 era): subprocess-first.** Stage committed
   inputs to a run directory, invoke `cargo raster run`, ingest outputs and
   the commit artifact. In-process embedding via `raster-runtime` is a WS6
   consideration.
6. **Toolchain pinned to Rust 1.91.1** (matching the `raster` workspace) via
   `rust-toolchain.toml`.
7. **Sim path is the logical specification;** the tokenizer PoC is an
   authoring-idiom reference only. Nothing in WS0 modifies sim behavior.
8. **Zero behavior change until WS9.** Parity gate green at every commit;
   goldens never regenerated.

## Charter amendment: CLI shape (dated ruling)

The charter froze `--raster-core-at` against a flag-based CLI
(`--raster`/`--raster-at`). The CLI has since moved to subcommands: detour
targeting is `detour --at ROUTINE[:OCC]`, full-raster execution is
programmatic (`InferenceControls.raster`), and sizing controls come from the
`--config` TOML rather than `--raster-*-per-tile` flags.

**Ruling (user decision, 2026-07-05): literal charter compliance.** The
raster-core selector is a second targeting flag on the existing `detour`
subcommand:

```
raster-inference detour --raster-core-at <routine-id[:occurrence]> ...
```

- `--raster-core-at` and `--at` are mutually exclusive (clap
  `conflicts_with`); exactly one is required.
- Both flag forms work (`--raster-core-at prefill.range:2` and
  `--raster-core-at=prefill.range:2`), with the same occurrence-suffix
  grammar and defaults as `--at` (`:1` when omitted).
- The charter's "mutually exclusive with `--raster` and `--raster-at`" maps
  to: CLI conflict with `--at`, and the existing runtime rejection of
  full-raster (`InferenceControls.raster`) combined with any detour spec.

## Resolved WS0 decisions

### 1. Dependency pinning: path dependency + CI-pinned checkout

`raster` is consumed as a **path dependency** on a sibling checkout, declared
once in the workspace manifest:

```toml
[workspace.dependencies]
raster = { path = "../raster/crates/raster", default-features = false }
```

CI materializes the sibling checkout of `casettek/raster` at a pinned rev.
**The pin lives in exactly one place:** the `RASTER_PIN` env var at the top of
`.github/workflows/ci.yml`. Initial pin:

```
RASTER_PIN: 536214533f06381a913e5873772e283025ebe061
```

**Pin-bump procedure** (a deliberate act, per charter invariant "toolchain
drift"):

1. Update `RASTER_PIN` in `.github/workflows/ci.yml`.
2. Update your local `../raster` checkout to the same rev.
3. Run the full suite and the parity gate locally
   (`cargo test`, `cargo test --release --test e2e_checkpoint_parity`,
   `cargo test --release --test golden_traces`); gate-green is the acceptance
   test for the bump.
4. Land the bump as its own commit with a dated note appended to this ADR.

*Rationale:* matches the tokenizer-PoC idiom; zero-friction local iteration
against the sibling checkout; no cargo git-auth friction (the checkout step
uses the Actions token). Cargo.lock does not pin path-dep sources; the CI pin
plus the toolchain pin carry reproducibility.

*Rejected alternatives:* a git dependency pinned by rev (reproducible but
makes local iteration against uncommitted `raster` changes require `[patch]`
churn, and needs cargo-level git auth for a private repo); a `[patch]`-based
hybrid (two places to keep consistent, easy to land half-updated).

### 2. Workspace layout: root package stays at the repo root

The root `Cargo.toml` gained a `[workspace]` section (`resolver = "2"`) with
members `crates/dsl-macros` and `crates/raster-programs/*`. The
`raster-inference` package itself stays at the repo root, so CI commands,
docs paths, and the `include_str!`-based guard test in `src/routines/mod.rs`
are unaffected.

### 3. Controller extension: backend carried on the spec

`RasterDetourSpec` gained a `backend: DetourBackend { Sim, RasterCore }`
field. `RasterDetourSpec::parse` produces sim specs (unchanged call sites);
`RasterDetourSpec::parse_raster_core` produces raster-core specs. Occurrence
matching, `ensure_matched_if_active`, and `reject_if_selected_unsupported`
remain single implementations on `RasterDetourController`:

- `should_detour_sim(routine)` wraps the raw matcher: a matched sim spec
  detours as before; a matched raster-core spec errors
  `"selective raster-core detour for {routine} is not implemented yet"`
  (WS0 behavior for all 10 routines; WS3 replaces the error with dispatch to
  `run_raster_core` per routine).
- `StepMode` gained a `RasterCore` variant, produced by
  `ExecutionPolicy::mode_for` for matched raster-core specs (used at the
  `input.embedding` decision point in `runtime::sequence`).
- `prefill.range_finalize` has no sim detour decision point on the native
  path (a sim selector targeting it reports "target was not reached" at the
  end of the run — pre-existing behavior, preserved).
  `reject_if_selected_unsupported_raster_core` adds a raster-core-only
  decision point so the selector fails cleanly as unimplemented; **its
  occurrence semantics (once per prefill) are provisional until the routine's
  WS3 migration pins them.**

Sizing controls (`--config` TOML → `ExecutionTuning` → `InferenceControls`)
are accepted alongside `--raster-core-at` and validated exactly as for sim
detours; they are consumed by the real path starting WS3 (with the WS8 caveat
that real tile granularity is largely an authoring-time decision).

### 4. Feature gating: always-compiled

No cargo feature gates the raster-core plumbing or the program crates. The
main crate has no `raster` dependency (frozen decision 4), so the only build
surface the real toolchain adds is the program crates themselves. Checked for
conflicts: `serde` (raster: `1.0`, default-features = false, derive+alloc;
raster-inference: `1` with std) and `postcard` (raster: `1.0` alloc;
raster-inference: `1.1.3` alloc) unify within semver-compatible ranges;
workspace dependency resolution confirmed with goldens byte-identical after
conversion.

### 5. Parity harness enumeration: explicit const list

`tests/raster_core_detour_parity.rs` is parametrized by

```rust
const ENABLED_ROUTINES: &[(&str, usize)] = &[];
```

WS4 appends one `(routine-id, occurrence)` entry per migrated routine in the
same change that lands the migration. No discovery magic; a reviewer can see
exactly which legs the gate enforces. The identity assertion is strict —
committed checkpoints are backend-invariant — and any WS5 ruling that a
boundary checkpoint legitimately diverges in form is recorded next to the
routine's entry, never silently.

## Named gaps (recorded, not shimmed)

1. **`cargo raster run` does not propagate guest failure.** The whole-program
   run path in `raster/crates/raster-cli/src/commands/run.rs` prints a failed
   child process's exit status but returns `Ok(())`, so the CLI exits `0`.
   The WS0 subprocess wrapper (`src/runtime/raster_core/`) checks the exit
   status anyway, but WS2+ ingestion must validate the produced artifacts
   (outputs + commit file) rather than trusting the exit code. Missing
   behavior: non-zero CLI exit when the executed program binary fails.
2. **No release tags on `raster`.** Pinning is by commit SHA; acceptable, but
   tagged releases would make pin provenance easier to audit.
3. **Recur semantics (WS1 blocker, tracked in the charter).**
   `docs/tile-authoring.md` describes recursion as an annotation-level
   convention ("no runtime recursive execution loop"), while `raster::input`
   ships `run_recur_list*`/`RecurControl` drivers used by `call_recur!`
   inside `#[sequence]` bodies (std runtime only). WS1 must verify what
   actually executes before any routine that leans on the sim's
   `call_recur_tile!`/`call_recur_seq!` pair-state semantics is ported.

## WS0 deliverables (landed with this ADR)

- Workspace conversion; `rust-toolchain.toml` (1.91.1); CI pinned toolchain +
  pinned `raster` checkout; placeholder program crate
  `crates/raster-programs/prompt_prepare/` (built in CI including a
  `--no-default-features` no_std surface check).
- `detour --raster-core-at` parsing/validation/conflict tests; all 10 routine
  ids route to the clean unimplemented error via the controller.
- `raster_core/` host stub convention plus the per-routine guard-test
  activation list (`RASTER_CORE_MIGRATED` in `src/routines/mod.rs`).
- Run-directory conventions and `cargo raster` subprocess wrapper
  (`src/runtime/raster_core/`).
- Parity harness scaffolding (`tests/raster_core_detour_parity.rs`, zero
  routines enabled).
- Baseline capture and golden hashes in
  `docs/plans/ws0-implementation-plan.md`; committed traces byte-identical to
  the pre-WS0 baseline throughout.
