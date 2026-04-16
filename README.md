# raster-inference

`raster-inference` is a super bare-bones, text-only Gemma 4 inference project.

The design goal is to keep the architecture extremely simple:

- tiles are plain side-effect-free functions
- sequences are explicit compositions of those functions
- inference is modeled as three isolated phases
- the CLI now runs phase 1 and the first phase-2 embedding step in serial

## Current Scope

This repo only targets:

- text-only Gemma 4 usage
- one Rust crate
- one function-first phase-1 pipeline
- one serial phase-1-to-phase-2 path
- one prompt-prefill Gemma 4 text path through all decoder layers
- one minimal phase-3 greedy decode loop built on final-position logits
- no scheduler, server, cache manager, or framework abstraction

This repo does not yet include:

- scheduler or cache-manager abstractions beyond the explicit per-layer KV cache used for decode
- stochastic sampling logic (`temperature`, `top_k`, `top_p`, repetition penalties)
- multimodal support
- streaming or partial token deltas
- Raster DSL integration

## Phase Model

The intended long-term shape is:

1. Phase 1: prompt preparation
2. Phase 2: transformer state transition
3. Phase 3: logits-to-token decode

Today the implemented serial path produces:

- SHA-256 of the prompt token IDs
- SHA-256 of the token embedding activations for those prompt token IDs
- SHA-256 of the final Gemma 4 prefill hidden states
- SHA-256 of the final-position logits
- deterministic greedy output text for up to `sampling.max_new_tokens`

## File Layout

- `src/io.rs`: thin disk-loading helpers for local assets
- `src/phase1/`: phase-1 composition, types, and tiles
- `src/phase2/`: phase-2 composition, types, and tiles
- `src/lib.rs`: public API and future phase placeholders
- `src/main.rs`: tiny CLI for local smoke tests

## Philosophy

This project deliberately does not try to transplant the `mistral.rs` framework.
Instead, it uses `mistral.rs` as a reference for the specific Gemma 4 seams we
care about:

- chat templating
- tokenization
- later, text-only Gemma 4 forward execution

The point is to port only the needed techniques, one seam at a time, while
keeping the codebase small enough that each phase can be reasoned about in
isolation.

For the working porting method and the first-batch tile plan, see
`RASTER_INFERENCE_PORTING_GUIDE.md`.

## CLI Smoke Test

The current CLI expects local tokenizer and template artifacts plus a Gemma model path for the Gemma text weights:

```bash
cargo run -- \
  google/gemma-4-test \
  /path/to/tokenizer.json \
  /path/to/chat_template.jinja \
  /path/to/gemma-model \
  "Hello from phase one"
```

It runs phase 1 first, then immediately feeds the resulting prompt token IDs into phase 2. For phase 2 it reads `model.language_model.embed_tokens.weight`, all Gemma text-layer weights, the final text norm, and the output projection path from the Gemma safetensors. It applies Gemma's embedding scale automatically, runs the full text prefill path, and produces final-position logits plus an explicit decode state with per-layer KV cache. Phase 3 then performs deterministic greedy decode by selecting one token at a time, appending it to the explicit token sequence, and calling the incremental Phase 2 decode step for the new token. This keeps the architecture simple while avoiding full-sequence replay on every generation step. `temperature`/`top_k`/`top_p` remain unsupported beyond accepting the current deterministic default configuration. The model path can be:

- a Gemma model directory containing `model.safetensors.index.json`
- a Gemma model directory containing a single `model.safetensors` or `consolidated.safetensors`
- a direct path to a `.safetensors` file or `model.safetensors.index.json`

The CLI prints the resulting `InferenceState` as formatted JSON with both:

- `phase1`: SHA-256 digest of the prompt token IDs
- `phase2.token_embeddings`: SHA-256 digest of the embedding activations for those prompt token IDs
- `phase2.final_hidden_states`: SHA-256 digest of the final Gemma 4 prefill hidden states
- `phase2.prefill_logits.final_logits_sha256`: SHA-256 digest of the final-position logits
- `phase3.generated_text`: detokenized text for the generated tokens only
- `phase3.generated_token_ids_sha256`: SHA-256 digest of the generated token IDs
- `phase3.stop_reason`: currently `max_new_tokens`

To trace long Gemma runs tile-by-tile, set `RASTER_TRACE_TILES=1`. Trace logs go to stderr and include start/end timing for model loading, major phase-2 tiles, and each decoder layer.

```bash
cargo run -- \
  google/gemma-4-E4B-it \
  assets/gemma-4-E4B-it/tokenizer.json \
  assets/gemma-4-E4B-it/chat_template.jinja \
  /path/to/gemma-4-model \
  "Hello from phase one"
```
