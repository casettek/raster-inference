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
- Raster DSL integration

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
- `prefill_layer` -> `prefill.layer` and `prefill.layer_token.*`
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

- `src/checkpoints.rs`: protocol `phase_id` and `routine` taxonomy for checkpoint names
- `src/io.rs`: thin disk-loading helpers for local assets
- `src/prompt_prepare/`, `src/prefill_prepare_aux/`, `src/prefill_layer/`, `src/prefill_finalize/`, `src/decode_select_token/`, `src/decode_transition/`, `src/output_finalize/`: routine implementation details
- `src/shared/`: shared contracts and transformer kernels
- `src/pipeline.rs`: consolidated composed inference helpers built from the routines
- `src/lib.rs`: public API and protocol-aligned aggregate state
- `src/main.rs`: tiny CLI for local smoke tests

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

## CLI Smoke Test

The current CLI expects local tokenizer and template artifacts plus a model path for either the default FP32 Gemma weights or the converted deterministic weight artifact:

```bash
cargo run -- \
  google/gemma-4-test \
  /path/to/tokenizer.json \
  /path/to/chat_template.jinja \
  /path/to/gemma-model \
  "Hello from Raster"
```

It runs the `input_embedding` prompt-preparation routine first, then immediately feeds the resulting prompt token IDs into the `transformer_state_transition` routines. For the default FP32 path it reads `model.language_model.embed_tokens.weight`, all Gemma text-layer weights, the final text norm, and the output projection path from the Gemma safetensors. It applies Gemma's embedding scale automatically, runs the full text prefill path, and produces final-position logits plus an explicit decode state with per-layer KV cache. The `output_decode` routine then performs deterministic greedy decode by selecting one token at a time, appending it to the explicit token sequence, and calling the incremental transformer decode-transition routine for the new token. This keeps the architecture simple while avoiding full-sequence replay on every generation step. `temperature`/`top_k`/`top_p` remain unsupported beyond accepting the current deterministic default configuration. The default model path can be:

- a Gemma model directory containing `model.safetensors.index.json`
- a Gemma model directory containing a single `model.safetensors` or `consolidated.safetensors`
- a direct path to a `.safetensors` file or `model.safetensors.index.json`

The deterministic comparison path is opt-in:

```bash
cargo run -- \
  --commit-checkpoints \
  --deterministic \
  google/gemma-4-test \
  /path/to/tokenizer.json \
  /path/to/chat_template.jinja \
  /path/to/converted-det-model \
  "Hello from Raster"
```

To stop after a specific checkpoint, pass `--terminal-checkpoint <checkpoint-id>`. For example, `--terminal-checkpoint prefill.finalize` stops after the prefill finalize checkpoint has been emitted.

When `--deterministic` is set, the model path must point to either:

- a directory containing `config.json` and `model.detwgt`
- a direct path to a `model.detwgt` file with a sibling `config.json`

This Phase 1 deterministic path is a **converted-weight parity path**, not the final end-to-end `det_num` runtime. It loads canonical `Wgt` bytes from `model.detwgt`, reconstructs runtime matrices, and then runs the same high-level inference routine structure so you can compare the converted artifact against the existing FP32 baseline.

The CLI prints the resulting `InferenceState` as formatted JSON with:

- `input_embedding.prompt_token_ids_sha256`: SHA-256 digest of the prompt token IDs
- `input_embedding.embedded_prompt_activations_sha256`: SHA-256 digest of the embedding activations for those prompt token IDs
- `transformer_state_transition.activation_states[0].activations_sha256`: SHA-256 digest of the final Gemma 4 prefill hidden states
- `transformer_state_transition.prefill_logits.final_logits_sha256`: SHA-256 digest of the final-position logits
- `output_decode.generated_text`: detokenized text for the generated tokens only
- `output_decode.generated_token_ids_sha256`: SHA-256 digest of the generated token IDs
- `output_decode.stop_reason`: currently `max_new_tokens`

If you stop at a terminal checkpoint before the normal end of inference, the CLI prints a `PausedInferenceState` instead, with `terminal_checkpoint_id` plus the completed outputs gathered so far.

Pass `--commit-checkpoints` to emit the checkpoint trace file at the end of the run. Leave it off to skip checkpoint commitment work entirely. Checkpoint hits and phase start/end logs still go to stderr even when checkpoint output is disabled.

```bash
cargo run -- \
  google/gemma-4-E4B-it \
  assets/gemma-4-E4B-it/tokenizer.json \
  assets/gemma-4-E4B-it/chat_template.jinja \
  /path/to/gemma-4-model \
  "Hello from Raster"
```

## Comparison Workflow

To validate a converted deterministic artifact against the current baseline:

1. Run the prompt once against the original FP32 model path.
2. Run the same prompt again with `--deterministic` against the converted `model.detwgt` directory.
3. Compare:
   - `output_decode.generated_text`
   - `output_decode.generated_token_ids_sha256`
   - `transformer_state_transition.prefill_logits.final_logits_sha256`

For small deterministic fixtures, exact agreement is the target. For real converted Gemma checkpoints, Phase 1 is meant to show whether the converted weight format preserves output quality closely enough before the repo switches to a full `det_num` arithmetic path.

The automated parity suite now also includes a softcap-sensitive fixture that keeps attention and
MLP behavior intentionally quiet so logits-tail drift is attributable to final logit softcapping.
That fixture is a routing/parity check, not a quality verdict. Real-model validation remains manual:
run representative prompts through both the FP32 baseline and the deterministic path, then compare
output quality rather than requiring exact token identity once deterministic-only seams are active.
