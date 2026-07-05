# Raster-Core Routine Plan Template

Use this template when creating a focused plan for migrating one routine to
the raster-core (real toolchain) path — WS3 (authoring) plus WS4 (parity) of
the migration charter. The shared setup keeps native deterministic CPU as the
outer driver and lets one selected routine occurrence
(`detour --raster-core-at <routine-id[:occurrence]>`) execute on the real
`raster` toolchain, then materialize back to native state.

The sim path (`src/routines/<routine>/raster/`) is the **logical
specification** for the routine's tile-shaped execution: porting means
re-expressing its logic under the real-raster authoring constraints below —
not copying code, and not redesigning the logic. The tokenizer PoC
(`raster-tokenizer`) is an authoring-idiom reference only.

## Routine Target

- Routine id:
- Example selector (`--raster-core-at` form):
- Occurrence semantics:
- Unsupported sub-checkpoints:
- WS1 catalog rows consumed (every sim construct this routine uses must have
  a confirmed real-raster mapping before authoring starts):

## Real-Raster Authoring Constraints

Everything in the routine's program crate
(`crates/raster-programs/<routine>/`) is bound by the real toolchain's tile
contract (`raster/docs/tile-authoring.md`); the WS0 placeholder crate
(`crates/raster-programs/prompt_prepare/`) shows the shell shape.

- **Crate shape:** `no_std`-compatible lib
  (`#![cfg_attr(not(feature = "std"), no_std)]`, `extern crate alloc`), tiles
  in an alloc-only module; `#[sequence] fn main()` host binary gated on the
  `std` feature; `raster = { workspace = true }` (path dep,
  `default-features = false`), `std = ["raster/std"]`. The crate must keep
  compiling under `--no-default-features` (CI-checked).
- **Tiles are free functions** with `#[tile]` (`kind = iter | recur`): no
  generics, no methods, simple identifier parameters.
- **Serde-compatible signatures only;** arguments and returns cross the
  postcard byte ABI: 1 argument encodes as `postcard(arg)`, N>1 arguments as
  the tuple `postcard((arg1, arg2, ...))`, returns as
  `postcard(return_value)` (or the ok value for `Result`).
- **Fallible tiles return `raster::exec::Result<T>`** — `Result<T, String>`.
  No `anyhow`, no custom error enums across the tile boundary.
- **No std-reachable shared code.** Sim tiles lean on std-heavy modules
  (`tokenizers`, `memmap2`, rayon-adjacent code); none of that is reachable
  from a program crate. The port is a rewrite against the spec, not a code
  move.
- **Sequences compose with `call!`/`call_seq!`** (real prelude); recursion
  uses `call_recur!`/`call_recur_seq!` inside `#[sequence]` bodies subject to
  the WS1 recur ruling. Bare function calls are not extracted into the CFS.
- **Macro hygiene:** the real prelude's crate-root macros collide with the
  sim DSL's; program crates import `raster::prelude::*`, the main crate never
  does.

## Input Staging (native → raster boundary)

- Before-execution native boundary:
- Native values available at that boundary:
- Values that must not be recomputed:
- Committed input set: `input.json` (logical input name → file binding:
  `path`, optional `index_path`, `load_preference: read | mmap`) +
  `input_manifest.json` (logical input name → `{type: sha256, encoding,
  commitment}`), per the tokenizer-PoC idiom:
- Staging code location (host-side, `src/routines/<routine>/raster_core/` or
  a staging bin in the program crate):
- Run-directory conventions: fresh unique directory per invocation via
  `runtime::raster_core::RasterCoreRunDir` (`input.json`,
  `input_manifest.json`, `commit.bin`); run directories are kept on failure
  as debugging evidence:

## Raster Execution

- Program crate and `#[sequence] fn main()` entry:
- Exact routine instance to execute:
- Invocation: `cargo raster run --backend native --input ...
  --input-manifest ... --commit ...` via
  `runtime::raster_core::CargoRasterRunner` (subprocess-first per the
  migration ADR; in-process execution is a WS6 consideration):
- Known gap: the CLI exits `0` even when the program binary fails — ingestion
  must validate produced artifacts, never trust the exit code alone:
- Checkpoint emission ownership:
- Tile telemetry behavior (verbose trace output only; never committed
  payloads):

## Commit-Artifact and Output Ingestion (raster → native boundary)

- Outputs to read back from the run:
- Commit artifact (`commit.bin`, postcard `TraceCommitment`) handling:
- Host `run_raster_core` materialization into native output structs:
- Canonical deterministic values that must be preserved:
- Failure mode if outputs cannot be materialized safely:

## Host Contract

- Exactly one `pub fn run_raster_core(` in
  `src/routines/<routine>/raster_core/mod.rs`; flip the routine into
  `RASTER_CORE_MIGRATED` in `src/routines/mod.rs` in the same change (the
  guard test enforces one entrypoint for migrated hosts, zero for
  unmigrated, and the forbidden-wrapper-pattern list).
- Decision-point wiring: replace the routine's WS0 unimplemented error with
  dispatch to `run_raster_core` at the same controller decision point:

## Trace Parity (WS4)

- Committed checkpoint payload that must match pure CPU (backend-invariant;
  no artifact roots, proof metadata, tile counts, or backend labels):
- Checkpoint name and occurrence:
- Boundary-checkpoint schema handling, if this routine is one of the three
  value-form/artifact-root-form exceptions (`prompt.prepare`,
  `input.embedding`, `prefill.prepare_aux`) — WS5 ruling reference:
- Enable the routine's leg: append `(routine-id, occurrence)` to
  `ENABLED_ROUTINES` in `tests/raster_core_detour_parity.rs` in the same
  change:

## Tests

- Pure CPU deterministic trace:
- CPU plus one raster-core detour trace (committed-trace identity, divergence
  named by checkpoint id + occurrence):
- Final output equality:
- Unsupported/error cases (missing `cargo-raster`, failed staging, failed
  ingestion):
- Sizing control coverage (note the WS8 caveat: real tile granularity is
  largely authoring-time; document which runtime knobs this routine actually
  consumes):
- Goldens are never regenerated; a divergence is a bug in the new path until
  proven otherwise.

## Measurements

- Tile sizing / cycle counts: `[MEASUREMENT-PENDING]` until WS8 profiling on
  the migrated path; never estimated inline.

## References To Follow

- `src/routines/<routine>/raster/` — the sim source (logical specification)
- `docs/plans/2026-07-05-001-raster-core-migration-adr.md` — WS0 decisions,
  named gaps, pin-bump procedure
- `docs/plans/selective-raster-routine-template.md` — the sim-era template
  this one extends
- `src/runtime/raster_core/` — run-directory + subprocess scaffolding
- `crates/raster-programs/prompt_prepare/` — program-crate shell shape
- `raster/docs/tile-authoring.md` — the tile contract
- `raster-tokenizer` — authoring idiom (crate shape, staging bin, invocation)
