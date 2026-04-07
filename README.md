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
- no scheduler, server, cache manager, or framework abstraction

This repo does not yet include:

- phase 2 transformer execution
- phase 3 logits-to-token decode
- multimodal support
- KV cache
- sampling logic
- Raster DSL integration

## Phase Model

The intended long-term shape is:

1. Phase 1: canonical request preparation
2. Phase 2: transformer state transition
3. Phase 3: logits-to-token decode

Only phase 1 is implemented now. It stops at:

- canonicalized request
- rendered prompt string
- prompt token IDs
- deterministic phase-1 commitment

## File Layout

- `src/types.rs`: plain data structures
- `src/tiles.rs`: pure phase-1 tile functions
- `src/phase1.rs`: phase-1 composition and asset-loading edge helpers
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

It prints the resulting `Phase1State` as formatted JSON.
