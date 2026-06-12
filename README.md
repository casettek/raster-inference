# raster-inference

`raster-inference` is a super bare-bones, text-only Gemma 4 inference project.

The design goal is to keep the architecture extremely simple:

- tiles are plain side-effect-free functions
- sequences are explicit compositions of those functions
- inference is modeled as protocol `phase_id`s plus smaller checkpoint-to-checkpoint routines
- the CLI now runs the input-embedding routine first and then the transformer routines in serial

## Current Scope

This repo only targets:

- text-only Gemma 4 usage
- one Rust crate
- one function-first prompt-preparation path
- one serial input-to-transformer path
- one prompt-prefill Gemma 4 text path through all decoder layers
- one minimal output-decode greedy loop built on final-position logits
- one opt-in deterministic weight-path for comparing converted `model.detwgt` artifacts against the FP32 baseline
- no scheduler, server, cache manager, or framework abstraction

This repo does not yet include:

- scheduler or cache-manager abstractions beyond the explicit per-layer KV cache used for decode
- stochastic sampling logic (`temperature`, `top_k`, `top_p`, repetition penalties)
- multimodal support
- streaming or partial token deltas

## Phase IDs And Routines

The intended long-term shape uses two levels of naming:

1. Protocol `phase_id`s describe the coarse commitment layer:
   - `input_embedding`
   - `transformer_state_transition`
   - `output_decode`
2. `routine`s describe the fixed unit of work between two checkpoints.

Today the implemented routines map onto checkpoint families like this:

- `prompt_prepare` -> `prompt.prepare`
- `prefill_prepare_aux` -> `prefill.prepare_aux`
- `prefill_layer` -> `prefill.range`, `prefill.range_finalize`, and `prefill.layer_token.*`
- `prefill_finalize` -> `prefill.finalize`
- `select_output_token` -> `decode.select_token`
- `decode_transition` -> `decode.layer_token.*` and `decode.finalize`
- `finalize_output` -> `output.finalize`

The implementation now lives in routine-oriented modules plus shared kernels/contracts rather than phase directories.

Today the implemented serial path produces:

- SHA-256 of the prompt token IDs
- SHA-256 of the token embedding activations for those prompt token IDs
- SHA-256 of the final Gemma 4 prefill hidden states
- SHA-256 of the final-position logits
- deterministic greedy output text for up to `sampling.max_new_tokens`

## File Layout

- `src/runtime/checkpoints.rs`: protocol `phase_id` and `routine` taxonomy for checkpoint names, plus the raster detour spec/controller
- `src/runtime/trace.rs`: checkpoint emission, terminal-checkpoint tracking, and serialized trace artifact writing
- `src/runtime/sequence.rs`: the phase-sequencing skeleton — the single place that knows the canonical routine order
- `src/runtime/executors/`: the executor seam — `native.rs` (native deterministic/fp32 executor with selective raster detour hooks) and `raster.rs` (full root-backed raster tile executor)
- `src/runtime/roles/`: protocol role entry points — `claimer.rs` (`claimer::run`), `challenger.rs` (`challenger::audit`, replay/compare/detour), and `detour.rs` (`detour::run`, the shared single-routine raster detour)
- `src/runtime/inference.rs`: inference control/outcome types plus the deprecated legacy entry points (thin shims over `sequence::run`)
- `src/runtime/pipeline.rs`: composed prefill/decode bundle helpers for tests, benches, and golden capture (not on the production path)
- `src/routines/<routine>/{native,raster}/`: routine implementation details (tiles, types, utils, auth sources)
- `src/shared/api/`: request/outcome types and the challenger's audit report types
- `src/shared/model/gemma/`: all Gemma model-family code — transformer weight types, tokenizer, and weight/tokenizer loaders (see `docs/model-agnostic-layers.md`)
- `src/shared/`: shared contracts, artifacts, and transformer kernels
- `src/io.rs`: generic disk-loading helpers (chat templates, tokenizer files, safetensors/mmap infrastructure)
- `src/dsl/`: raster tile DSL machinery
- `src/lib.rs`: curated public API surface
- `src/main.rs`: clap-based protocol CLI (`claim` / `detour` / `audit`)
- `tests/goldens/`: golden checkpoint trace artifacts (byte-identity contract; see `tests/golden_traces.rs`)
- `assets/tiny-gemma-dev/`: checked-in hermetic test model bundle

## Philosophy

This project deliberately does not try to transplant the `mistral.rs` framework.
Instead, it uses `mistral.rs` as a reference for the specific Gemma 4 seams we
care about:

- chat templating
- tokenization
- later, text-only Gemma 4 forward execution

The point is to port only the needed techniques, one seam at a time, while
keeping the codebase small enough that each routine can be reasoned about in
isolation.

For the working porting method and the first-batch tile plan, see
`RASTER_INFERENCE_PORTING_GUIDE.md`.

## CLI

The CLI is a thin shell over the protocol role APIs (`claimer::run`, `challenger::audit`, `detour::run`) with one subcommand per protocol action. Every subcommand runs deterministic execution with checkpoint commitment on; the fp32 baseline is not reachable from the CLI. Run `raster-inference <subcommand> --help` for the full flag list.

### Common flags

- `--model <dir>`: model directory, resolved by convention — it must contain `tokenizer.json`, `chat_template.jinja`, and `model.detwgt`. The model id defaults to the directory name (`--model-id` overrides).
- The prompt is given as trailing arguments (joined with spaces) or via `--prompt-file <path>`.
- `--max-new-tokens <N>` (default 16) bounds generation. Temperature is a fixed protocol constant (`protocol::SAMPLING_TEMPERATURE`), not a flag.
- `--trace-dir <dir>`: where trace artifacts are written. Overrides the `RASTER_TRACE_DIR` env var; default `raster-traces/`.
- `--config <file>`: execution tuning TOML (see below).

Human-readable progress goes to stderr; stdout is a single machine-parseable JSON document.

### claim

Runs one inference as the claimer and emits the checkpoint trace artifact (the protocol object committed on-chain):

```bash
cargo run -- claim --model assets/tiny-gemma-dev "Hello from Raster"
```

stdout: `{ "trace_path": ..., "state": <InferenceState> }`.

For debugging, `--stop-at <checkpoint-id[:occurrence]>` pauses the run after the named checkpoint (e.g. `--stop-at prefill.finalize`, or `--stop-at prefill.range_finalize:2` for the second finalized prefill layer); stdout is then the `PausedInferenceState` JSON, with `terminal_checkpoint_id` plus the completed outputs gathered so far.

### detour

Re-runs an inference native-deterministically with exactly one selected raster routine occurrence swapped in, producing the raster detour trace artifact used by the dispute path:

```bash
cargo run -- detour --model assets/tiny-gemma-dev --at prefill.range:2 "Hello from Raster"
```

stdout: `{ "routine": ..., "occurrence": ..., "trace_path": ..., "state": <InferenceState> }`.

Add `--trace-tiles` to print verbose routine, progress, and individual tile execution logs to stderr.

### audit

Replays a claimed trace as the challenger: re-runs the request natively, compares committed checkpoints positionally against the claimed artifact, and on divergence automatically performs the raster detour at the divergent routine occurrence (when the routine supports detours):

```bash
cargo run -- audit --model assets/tiny-gemma-dev --claimed raster-traces/trace-1234.json "Hello from Raster"
```

stdout is either:

```json
{ "result": "no_divergence" }
```

or

```json
{
  "result": "divergence",
  "divergence": { "entry_index": 3, "checkpoint_id": "prefill.range", "occurrence": 1, "claimed_commitment": "...", "replayed_commitment": "..." },
  "detour": { "routine": "prefill.range", "occurrence": 1, "trace_path": "..." }
}
```

`detour` is `null` for structural divergences (id-sequence or length mismatch) and for routines without an implemented detour (e.g. `prompt.prepare`).

Exit codes are part of the scripting contract: `0` = no divergence, `2` = divergence found (report emitted), `1` = operational error. `--trace-tiles` is accepted here too.

### Execution tuning config

Raster tile sizing and range widths are not per-invocation flags; they live in a TOML file passed as `--config <path>`. Every key is optional — omitted keys (or omitting the file entirely) use the built-in library defaults:

```toml
[ranges]
prefill_token_range_width = 8
decode_layer_range_width = 4

[tile_sizing]
projection_rows_per_tile = 64
attention_kv_rows_per_tile = 128
sequence_rows_per_tile = 16
head_rows_per_tile = 4
tokenizer_bpe_pairs_per_tile = 1024
tokenizer_bpe_pieces_per_tile = 512
output_byte_flush_bytes_per_tile = 4096
```

Tuning is pure scheduling: committed checkpoint bytes are identical for every tuning, so traces produced with different configs still compare equal.

### Dev model bundle

For fast local runs, generate the tiny representative Gemma-style bundle:

```bash
cargo run --bin tiny-gemma-dev -- --output-dir assets/tiny-gemma-dev --force
```

The generated bundle includes `config.json`, `model.safetensors`, `model.detwgt`,
`tokenizer.json`, and `chat_template.jinja`. It keeps the deterministic `Wgt`
artifact format while shrinking the model to tiny PLE-enabled layers with a
small Gemma-compatible BPE tokenizer. (`model.safetensors` is the fp32 baseline,
used only by the library-level parity suites.)

### Deterministic path notes

The deterministic path is a **converted-weight parity path** with a canonical-state runtime core. It requires `.detwgt` provenance, loads canonical `Wgt` bytes from `model.detwgt`, keeps deterministic KV cache rows and layer activations in canonical `Act` form across deterministic prefill/decode boundaries, and converts config-derived scalars once into canonical `Act`/`Acc` carriers. The existing `f32` fields remain compatibility views for public API and JSON consumers.

Deterministic checkpoints may include optional `det_*_sha256` fields next to the compatibility hashes. Compatibility fields such as `activations_sha256`, `final_logits_sha256`, and serialized `layer_caches` still describe the public f32 views; `det_*` fields describe canonical fixed-point bytes and are omitted when deterministic internals are not present.

The `state` JSON in `claim`/`detour` output includes:

- `input_embedding.prompt_token_ids_sha256`: SHA-256 digest of the prompt token IDs
- `transformer_state_transition.activation_states[0].activations_sha256`: SHA-256 digest of the final Gemma 4 prefill hidden states
- `transformer_state_transition.prefill_logits.final_logits_sha256`: SHA-256 digest of the final-position logits
- `output_decode.generated_text`: detokenized text for the generated tokens only
- `output_decode.generated_token_ids_sha256`: SHA-256 digest of the generated token IDs
- `output_decode.stop_reason`: currently `max_new_tokens`

## Comparison Workflow

To validate a converted deterministic artifact against the current baseline (the fp32 path is no longer reachable from the CLI; use the library entry points, e.g. `sequence::run` with an fp32-mode request, as the parity suites do):

1. Run the prompt once against the original FP32 model path.
2. Run the same prompt again in deterministic mode against the converted `model.detwgt` directory.
3. Compare:
   - `output_decode.generated_text`
   - `output_decode.generated_token_ids_sha256`
   - `transformer_state_transition.prefill_logits.final_logits_sha256`

For small deterministic fixtures, exact agreement is the target. For real converted Gemma checkpoints, Phase 1 is meant to show whether the converted weight format preserves output quality closely enough before the repo switches to a full `det_num` arithmetic path.

The automated parity suite now also includes softcap-sensitive and cache-sensitive fixtures that keep unrelated behavior quiet so drift is attributable to deterministic routing, canonical KV reuse, activation carryover, or final logit softcapping. These fixtures are routing/parity checks, not quality verdicts. Real-model validation remains manual: run representative prompts through both the FP32 baseline and the deterministic path, then compare output quality rather than requiring exact token identity once deterministic-only seams are active.

## Determinism Parity Gate

CI enforces end-to-end checkpoint trace parity (native-deterministic vs raster, self-reproducibility, thread-count invariance, detour smoke parity) on every push and PR. See [docs/parity-gate.md](docs/parity-gate.md) for what the gate guarantees and how to run it locally.
