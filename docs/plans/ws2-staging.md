# WS2 — Input Staging & Externals

- **Status:** state-carrying document. WS2 complete: the shared native→raster
  boundary — committed-input staging, subprocess execution, output/commit
  ingestion, and the error contract — landed with a CI-gated end-to-end round
  trip. WS3 routines consume the API described here; changes to conventions
  are made in place with a dated note appended to §10 (Amendment log).
- **Companions:** the migration charter (`raster-core-migration-charter.md`),
  `docs/plans/2026-07-05-001-raster-core-migration-adr.md` (WS0 decisions),
  `docs/plans/ws1-dsl-translation-catalog.md` (rows C13–C19, C22/C23, C25
  hand off to this document), `docs/plans/raster-core-routine-template.md`
  (per-routine plan template referencing these conventions).
- **Grounding:** `raster` at the WS0-pinned rev
  `536214533f06381a913e5873772e283025ebe061`.

---

## 1. What WS2 delivers

The reusable native→raster→native loop every WS3 routine's `run_raster_core`
host adapter is built from:

```
native state at the detour boundary
  → StagedInputs builder            (src/runtime/raster_core/staging.rs)
  → RasterCoreRunDir                (input.json + input_manifest.json + payloads)
  → CargoRasterRunner               (cargo raster run, ndjson trace, output env)
  → program crate binary            (#[sequence] fn main(), writes output.bin)
  → ingest::<T>                     (src/runtime/raster_core/ingest.rs)
  → RasterCoreRunResult<T>          (value + commitments + commit fingerprint)
```

Landed surface:

| Piece | Location |
|---|---|
| Staging builder (`StagedInputs`, commitment capture) | `src/runtime/raster_core/staging.rs` |
| Run-dir conventions (+ `output.bin`) | `src/runtime/raster_core/mod.rs` |
| Subprocess runner (ndjson trace, banner parsing, output env) | `src/runtime/raster_core/mod.rs` |
| Ingestion + error contract (`ingest`, `RasterCoreError`) | `src/runtime/raster_core/ingest.rs` |
| Program-side output helper | `crates/raster-programs/_support/` (`raster-program-support`) |
| Round-trip fixture program (not a routine) | `crates/raster-programs/_roundtrip/` |
| Gemma tokenizer committed external + encoder | `crates/raster-programs/gemma_externals/` |
| End-to-end tests (CI-gated) | `tests/raster_core_staging_roundtrip.rs`, `tests/raster_core_gemma_tokenizer_external.rs` |

## 2. Run-directory and file conventions

One fresh, uniquely named directory per routine invocation
(`RasterCoreRunDir::create(routine_id, occurrence)`), kept on failure as
debugging evidence, deleted by the caller only after successful ingestion:

| File | Producer | Content |
|---|---|---|
| `input.json` | host (`StagedInputs::write`) | logical name → `{path, [index_path], load_preference}` |
| `input_manifest.json` | host (`StagedInputs::write`) | logical name → `{type: sha256, encoding, commitment}` |
| `<name>.bin` | host | postcard payloads for postcard-encoded inputs |
| `commit.bin` | `cargo raster run --commit` | trace commitment (opaque to the host; WS7 decodes it) |
| `output.bin` | the program (via `raster-program-support`) | `postcard(Result<T, String>)` — the materialized outcome |

The CLI's own run artifacts (`trace.ndjson`) live under the program crate's
`target/raster/runs/<run-id>/`; the runner parses the stdout banner
(`Run artifacts dir:` / `Trace path:`) — the only discovery mechanism the CLI
offers — and resolves relative paths against the program crate directory.

## 3. The two input encodings

| | `postcard` | `raster` |
|---|---|---|
| For | request-scoped values (prompt bytes, config, chunk lists, prior-routine outputs) | model-scoped data (tokenizer tables, weights) |
| Encoded by | the main crate, at staging time (`add_postcard`) — postcard + sha2 are existing deps; no `raster` dependency (charter invariant 6) | a program crate's `encode`-feature bin, offline, once per model (`add_raster_encoded`) |
| Files | `<name>.bin` in the run dir | `.rastered` + `.rindex` in a content-addressed cache, referenced by absolute path |
| `load_preference` | `read` | `mmap` |
| Manifest commitment | SHA-256 of the payload file bytes | the raster index root commitment, reported by the encoder |

**Cache convention (raster-encoded externals):** entries live at
`<cache_root>/<kind>/<sha256(source)>/` with the encoded files plus
`root_commitment.txt`; re-encoding an already cached source reuses the entry.
Encoding is deterministic (same source → same commitment; test-asserted).
See `gemma_externals/src/encode.rs` for the reference implementation.

**Commitment capture:** `StagedInputs::write` returns the commitment map
(name → `{encoding, commitment}`); host adapters pass it into `ingest` so
every run result surfaces the commitments of the inputs it ran against
(WS3 exit criterion 3).

## 4. The output-file convention

`cargo raster run` has no result channel: values live only in the
raster-formatted trace (undecodable in the main crate without a `raster`
dependency), and `[output]` stdout capture is unframed text. WS2 closes the
gap with an explicit output file:

- **Host half:** `CargoRasterRunner::run` sets `RASTER_CORE_OUTPUT_PATH`
  (constant `runtime::raster_core::OUTPUT_PATH_ENV`) on the subprocess,
  pointing at the run directory's `output.bin`; the guest binary inherits it.
- **Program half:** the program's `#[sequence] fn main()` ends with
  (idiom, from the round-trip fixture):

  ```rust
  #[sequence]
  fn main() {
      let outcome = materialize_auth_result::<T, _>(call_seq!(routine_sequence));
      raster_program_support::write_program_output(&outcome);
  }
  ```

  `write_program_output` writes `postcard(Result<T, String>)` to the path
  (no-op when the env var is unset, e.g. manual `cargo raster run`).
  Materializing inside `main` is host-side std code; it does not disturb the
  trace or CFS extraction. A terminal `Err` is written like any other
  outcome — it is a committed result, not a broken run.
- The env-var contract between the two halves is asserted by the round-trip
  test (`OUTPUT_PATH_ENV` equality).

## 5. Error contract (WS1 catalog H4/C23, made concrete)

`RasterCoreError` (`src/runtime/raster_core/ingest.rs`) — three classes that
never collapse:

| Class | Meaning | Classified when | Test leg |
|---|---|---|---|
| `Infrastructure` | broken run; **no committed outcome exists**; fix the environment / retry | missing `cargo-raster`; staging IO failure; output/commit/trace artifact missing or undecodable without an integrity signature; **undecodable staged input** (corrupted `.rindex` — the H4 "deserialization of staged inputs" rule) | round-trip leg 4; tokenizer tamper leg B |
| `Terminal` | the program ran to completion and **committed an `Err(String)` outcome** — replayable, fault-provable | `output.bin` decodes to `Err(message)` | round-trip leg 2 (zero divisor) |
| `Verification` | committed-input integrity rejected, or artifacts mutually inconsistent | runtime integrity-check signature (`failed integrity check`) in the guest output with no `output.bin`; or an output value whose trace lacks `SequenceEnd` for `main` | round-trip leg 3 (tampered postcard payload); tokenizer tamper leg A (commitment mismatch) |

Host adapters map runner launch errors into `Infrastructure`; everything
else is classified by `ingest`.

## 6. Run validation: artifacts, never exit codes

The CLI's exit status is meaningless **in both directions** and is recorded
on `CargoRasterRunOutput::cli_success` for debugging only:

- It exits `0` when the guest program fails (ADR named gap G4).
- It exits non-zero for failures ingestion must still classify: the CLI
  panics building the trace commitment when a guest integrity rejection
  leaves the trace shorter than the verification window
  (`raster-prover/src/trace.rs`, "Trace length can't be less than
  verification window") — observed during WS2, recorded here so nobody
  reintroduces an exit-status check.

`ingest::<T>` validates instead: `output.bin` present and decodable as
`postcard(Result<T, String>)`; `commit.bin` present and non-empty (captured
as an opaque SHA-256 fingerprint — decoding is WS7 scope); the ndjson trace
contains `SequenceEnd` for `main`. The trace is always requested as
`--trace-format json` so the host parses it with plain `serde_json`; the
same trace carries the event counts WS4's structural checks derive (catalog
C24).

## 7. Encoder scope split

Committed-external encoders that **exist now**:

| External | Crate | Consuming routines |
|---|---|---|
| Gemma tokenizer (`GemmaTokenizer` schema **v2**: metadata, decoder metadata, **chunked** sorted token lookup (`token_lookup_chunks`) and priority-ordered merge table (`merge_chunks`) — `Vec<Vec<Entry>>`, encode-time width 1024 — dense id table, special tokens) | `crates/raster-programs/gemma_externals/` | `prompt.prepare` (WS3-consumed), `output.finalize` |

Schema provenance: the raster-tokenizer PoC shape, the idiom WS1 verified
for exactly this data (C13), revised to v2 by the `prompt.prepare` storage
refactor (2026-07-06, §11): model-scoped tables are pre-chunked so programs
consume them as recur input lists, and the pair-keyed `merge_lookup` was
deleted. **Accepted risk (WS2 ruling):** a consuming routine's WS3 plan may
revise the schema; a revision re-runs the determinism tests and the
tokenizer-external round trip, bumps the cache-kind segment (see §3 cache
convention — schema revisions must never collide with stale cache entries),
and appends a note to §11.

**Deferred to "build when WS3 reaches them"** (per the add-a-routine
checklist, §9): embedding table (`input.embedding`), PLE source
(`prefill.prepare_aux`), layer-weight sources (`prefill.range`,
`decode.layer_range`), prefill-finalize source (`prefill.finalize`), decode
transition source (`decode.transition_finalize`). These are `.detwgt`-derived
weight externals; their schemas are shaped by each routine's tile design
(catalog C21/C22) and belong to the routine's WS3 Phase A plan.

`decode.select_token` consumes no external source in tiles (catalog §5) —
its staging is postcard-only.

## 8. Cross-routine handoff

A previous routine's ingested outputs re-enter the next routine's run as
ordinary **postcard committed externals** through the same `StagedInputs`
builder (the catalog C17 run-boundary seam: reads of artifacts produced by a
previous routine's run are external inputs of the next program, never
internal reads). No additional machinery; the handoff is visible in both
manifests and both commitment maps.

## 9. Add-a-routine checklist (WS3 Phase B)

1. **Define the program's external-input schema** from the sim `main`'s
   parameter list (catalog C25): one logical input per parameter (or a
   single input struct), names `[a-z0-9_]+`.
2. **Postcard inputs:** stage at the call site with
   `StagedInputs::add_postcard` — no encoder to build.
3. **Raster-encoded inputs:** if the routine's external is in §7's deferred
   list, add a schema + encoder following `gemma_externals` (content-
   addressed cache, `root_commitment.txt`, deterministic; determinism +
   tamper tests mirroring `tests/raster_core_gemma_tokenizer_external.rs`).
4. **Program side:** depend on `raster-program-support` (std feature);
   end `#[sequence] fn main()` with the §4 idiom. One module per file
   (catalog C33).
5. **Host adapter (`run_raster_core`):** create `RasterCoreRunDir` → stage →
   `CargoRasterRunner::run` (map launch errors to
   `RasterCoreError::Infrastructure`) → `ingest::<T>` with the captured
   commitment map → format the checkpoint payload from the ingested value
   (backend-invariant; WS5 owns the boundary-checkpoint exceptions) →
   delete the run dir only after success.
6. **Output type:** owned serde types, fixed-width integers (catalog C12).
   If the main crate must decode the type without depending on the program
   crate, keep a field-order-matching mirror struct next to the host adapter
   (postcard layout contract), as the tokenizer-external test does.
7. Never branch on `cli_success` for run outcomes (§6).

## 10. Re-run recipe

Prerequisite (same as the WS1 probes): `cargo-raster` built from the pinned
sibling checkout on PATH —
`cargo install --path ../raster/crates/raster-cli --locked`.

```sh
# The full staging loop, one leg per error class:
cargo test --test raster_core_staging_roundtrip

# The tokenizer committed external (determinism, cache reuse, smoke run,
# both tamper modes), against assets/tiny-gemma-dev:
cargo test --test raster_core_gemma_tokenizer_external
```

Both tests skip loudly when `cargo-raster` is absent; CI installs it (cached
on `RASTER_PIN`) and runs them with `REQUIRE_CARGO_RASTER=1` so the skip can
never happen silently there (`.github/workflows/ci.yml`).

## 11. Amendment log

- **2026-07-07** — `prompt.prepare` staged-input set revised by the
  storage-resident refactor (PORT_PLAN deviations D15–D17). The
  `initial_pieces` postcard input is now the program crate's selectable
  root `BpePieces { pieces: Vec<String> }` rather than the bare
  `Vec<String>`; the postcard byte layout — and therefore the staged
  commitment — is unchanged (a single-field postcard struct is an
  unframed field sequence; asserted host-side by
  `staged_pieces_keep_the_bare_vec_byte_layout`). The host mirror is
  `StagedBpePieces` (WS2 §9.6 layout contract). The `bpe_config` staged
  input is **deleted**: the apply loop recurs over the round's own
  pieces, so no per-tile widths remain, and the piece count is derived
  in-program by a one-shot authenticated read of the staged external
  (a staged count would be either unchecked — an integrity hole — or
  redundant). The tokenizer external schema is untouched (cache kind
  stays `gemma-tokenizer-v2`).
- **2026-07-06** — Tokenizer external schema revised to **v2** by the
  `prompt.prepare` storage refactor — the first exercise of the §7
  accepted-risk clause. `token_lookup` → `token_lookup_chunks` and
  `merges` → `merge_chunks` (both `Vec<Vec<Entry>>`, encode-time chunk
  width 1024, order preserved); the pair-keyed `merge_lookup` deleted (its
  only consumer was the pre-refactor scan). Rationale: model-scoped tables
  enter tiles only as chunked recur input lists (WS1 C13/C22 amendment;
  PORT_PLAN deviations D10–D12). The cache-kind segment bumped
  `gemma-tokenizer` → `gemma-tokenizer-v2` in both the encoder
  (`encode::CACHE_KIND`) and the host adapter's
  `encode_tokenizer_external_cached`, so v2 encodings never collide with
  stale entries. Per the clause: determinism + tamper legs re-run green
  (`tests/raster_core_gemma_tokenizer_external.rs`, real toolchain), smoke
  program re-pointed through the chunked shape (nested `[0][0]`
  selection), and the `prompt.prepare` dev-run trace-identity gate stayed
  green.
- **2026-07-06** — WS3 `prompt.prepare` landed as the first consumer of this
  surface. The §7 accepted-risk clause was not triggered: the tokenizer
  schema shipped unrevised. One convention addition: the routine's host
  adapter resolves the tokenizer cache entry on demand — content-addressed
  hit check first (§3 cache convention), `cargo run`-subprocess encoder on a
  miss — with the cache root defaulting to
  `$TMPDIR/raster-inference-gemma-external-cache` and overridable via
  `RASTER_CORE_EXTERNAL_CACHE` (`src/routines/prompt_prepare/raster_core/`).
  When `output.finalize`'s WS3 port needs the same entry, lift the helper
  into `src/runtime/raster_core/` rather than duplicating it.
- **2026-07-06** — Initial version: staging builder, runner hardening,
  ingestion + `RasterCoreError`, `raster-program-support`, round-trip
  fixture + CI gating, Gemma tokenizer committed external. Discovery
  recorded in §6: the CLI also exits non-zero on guest integrity rejections
  (trace-commitment panic), so exit status is unusable in both directions.
  Encoder scope ruling (§7): tokenizer now, weight-family externals deferred
  to their routines' WS3 plans.
