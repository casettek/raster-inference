# Determinism parity gate

The parity gate is the mechanical enforcement of this repo's core security
property — **honest-party safety**: a committed checkpoint trace must be
exactly reproducible by any faithful replay, and native-deterministic and
raster (tile-level) execution of the same request must agree at every
committed checkpoint. It runs on every push and pull request via
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml), so a regression
surfaces on the diff that causes it.

Subsequent workstreams — the orchestration split of `inference.rs` /
`pipeline.rs` (Brief 02) and the thread-local/parallelism audit and
enforcement (Brief 01) — are executed **under this gate**: every refactoring
step must keep it green, and the golden traces Brief 02 captures are only
trustworthy because the self-reproducibility leg below certifies the baseline.

## What the gate guarantees

The end-to-end suite, [`tests/e2e_checkpoint_parity.rs`](../tests/e2e_checkpoint_parity.rs),
runs hermetically against the self-contained `assets/tiny-gemma-dev` bundle
(no network, no host-specific state) in the default **Verified** integrity
mode, and asserts four properties on the serialized checkpoint trace artifact
— the exact bytes a claimer would commit:

1. **Native vs raster checkpoint parity.** A full inference run in native
   deterministic mode and the identical request in full raster mode commit
   the same ordered sequence of checkpoint IDs/occurrences with identical
   SHA256 commitments at every checkpoint. Three boundary checkpoints
   (`prompt.prepare`, `input.embedding`, `prefill.prepare_aux`) legitimately
   commit different payload schemas (value form vs artifact-root form); those
   are compared field-wise via the run outcome state — the explicit mapping
   is documented in the test module docs.
2. **Native self-reproducibility.** The same native-deterministic request run
   twice produces **byte-identical** serialized trace artifacts. This catches
   scheduling sensitivity or unordered-map iteration leaking into output, and
   certifies golden-trace baselines.
3. **Thread-count invariance.** The committed trace artifact is byte-identical
   across rayon thread-pool sizes (1 thread, 4 threads, and the ambient
   pool). This is the test directly sensitive to side effects executed on
   rayon worker threads, which silently get fresh thread-local state.
4. **Detour-mode smoke parity.** A selective raster detour of one routine
   occurrence (`prefill.range:1`) leaves every checkpoint outside the
   detoured routine identical to the full-native run.

Failures are actionable: a divergence panic names the first divergent
checkpoint ID and its occurrence (e.g. `prefill.range:2`), not just a boolean
mismatch.

## How to run it

```bash
# Categories 2–4 (native-only legs; the full-raster leg is ignored in debug):
cargo test --test e2e_checkpoint_parity

# All four categories, including the full-raster Verified-mode leg:
cargo test --release --test e2e_checkpoint_parity
```

The full-raster leg performs Merkle proof work per tile and is feasible only
in optimized builds (~30 s in release; tens of minutes in debug), so it is
marked `#[cfg_attr(debug_assertions, ignore)]` and CI runs it in a dedicated
release job. It is bounded — short prompt, terminal checkpoint after the
first decode transition — but still crosses prompt prepare, input embedding,
the full prefill, one token selection, and one decode transition (asserted
explicitly). The self-reproducibility leg runs the full pipeline to
completion. The `unchecked-raster-integrity` feature is **not** used by the
gate; Verified mode is the security-relevant configuration.

## CI pipeline

Three jobs on every push/PR:

- **Format and lint** — `cargo fmt --check` (blocking) and `cargo clippy
  --all-targets` (report-only until the pre-existing lint debt is cleared;
  promoting clippy to `-D warnings` is a tracked follow-up).
- **Test suite (debug)** — `cargo test`, the full suite including the
  native-only parity legs.
- **Determinism parity gate (release)** —
  `cargo test --release --test e2e_checkpoint_parity`, all four categories.

CI works from a clean checkout and does not depend on the committed `target/`
directory (slated for removal as a separate cleanup item).
