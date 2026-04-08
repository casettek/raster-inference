# raster-inference

`raster-inference` is a super bare-bones, text-only Gemma 4 inference project.

The design goal is to keep the architecture extremely simple:

- tiles are plain side-effect-free functions
- sequences are explicit compositions of those functions
- inference is modeled as three isolated phases
- phase 1 is the only implemented phase right now

## Current Scope

This repo only targets:

- text-only Gemma 4 usage
- one Rust crate
- one function-first phase-1 pipeline
- one first phase-2 embedding tile
- no scheduler, server, cache manager, or framework abstraction

This repo does not yet include:

- the rest of phase 2 transformer execution beyond token embedding
- phase 3 logits-to-token decode
- multimodal support
- KV cache
- sampling logic
- Raster DSL integration

## Phase Model

The intended long-term shape is:

1. Phase 1: prompt preparation
2. Phase 2: transformer state transition
3. Phase 3: logits-to-token decode

Only phase 1 is implemented now. It stops at:

- prompt token IDs
- SHA-256 of the prompt token IDs

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

The current CLI expects local tokenizer and template artifacts:

```bash
cargo run -- \
  google/gemma-4-test \
  /path/to/tokenizer.json \
  /path/to/chat_template.jinja \
  "Hello from phase one"
```

It prints the resulting `Phase1State` as formatted JSON with the prompt token IDs and their SHA-256 digest.

```bash
cargo run -- \
  google/gemma-4-E4B-it \
  assets/gemma-4-E4B-it/tokenizer.json \
  assets/gemma-4-E4B-it/chat_template.jinja \
  "Hello from phase one"
```
