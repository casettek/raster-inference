# Performance README

This document captures the current performance findings for `raster-inference`
and the recommended next workstreams. It is meant as a staging document before
writing separate implementation plans.

## Current Situation

The current runtime is correct enough to run real dense Gemma 4 inference, but
it is still very slow for both startup and token generation.

Observed behavior from traced runs:

- prompt prefill is very expensive
- each generated token is very expensive
- model loading is also expensive, especially around PLE weight loading

Representative timings from recent traced runs:

- `phase2.run_prefill_pass`: about 153s
- `phase2.run_text_layers_prefill`: about 143s
- each `phase2.decode_step`: about 66s
- each `phase2.run_text_layers_decode_step`: about 56s
- each `phase2.project_decode_hidden_to_logits`: about 9.6s

## Main Bottlenecks

### 1. Eager global PLE loading

`io.load_ple_global_weights` currently decodes very large vocab-sized PLE
tensors up front for all layers.

Why this is bad:

- high startup cost
- high time-to-first-token cost
- large memory materialization
- much of the data is not needed immediately

### 2. Weight decode and materialization

The current loader reads safetensors, decodes scalar values, and copies them
into owned `Vec<f32>` buffers (`MatrixF32`).

Why this is bad:

- repeated safetensors parsing work
- BF16/F16 to FP32 conversion overhead
- large heap allocations and copies
- startup cost scales with model size

### 3. Naive CPU linear algebra

The hottest math paths are plain scalar loops over `f32` buffers, especially:

- `linear_sequence`
- `linear_row`
- `project_to_logits`

Why this is bad:

- poor throughput for layer matmuls
- poor throughput for final vocab projection
- startup fixes alone will not solve tokens/sec

## Recommended Workstreams

These should become separate plans later.

### Workstream A: Make PLE loading lazy

Goal:

- remove the worst eager-load behavior without changing inference semantics

Direction:

- do not decode all global PLE weights at startup
- fetch only the token rows actually needed from
  `embed_tokens_per_layer.weight`
- load or cache each layer's `per_layer_model_projection.weight` slice on first
  use

Expected impact:

- much faster startup
- much faster time to first token
- lower peak memory pressure

### Workstream B: Add an offline FP32 runtime artifact

Goal:

- convert the known-good model once, offline, into a runtime-friendly format

Direction:

- produce a raster-native or safetensors-based FP32 artifact
- use it to avoid repeated BF16-to-FP32 decode during normal runs

Expected impact:

- faster startup on repeated runs
- simpler runtime loading path
- good fit for the current `f32`-based implementation

Important caveat:

- this improves loading much more than steady-state per-token compute

### Workstream C: Replace naive matmul paths with optimized kernels

Goal:

- improve actual inference throughput, not just model loading

Direction:

- replace the scalar CPU loops in `linear_sequence`, `linear_row`, and
  `project_to_logits`
- move to an optimized backend or kernel strategy

Expected impact:

- largest total runtime win
- biggest effect on prefill speed
- biggest effect on decode tokens/sec

### Workstream D: Optimize final logits projection separately

Goal:

- reduce the standalone cost of projecting hidden states to the full vocab

Direction:

- target `project_to_logits` as its own hotspot
- treat vocab projection as a first-class optimization target

Expected impact:

- directly reduces per-token decode latency
- valuable even after general matmul improvements

### Workstream E: Cache safetensors metadata and tensor lookup state

Goal:

- remove repeated safetensors container parsing and lookup overhead

Direction:

- cache deserialized safetensors state per mapped file
- avoid reparsing container metadata on each tensor access

Expected impact:

- modest but clean loading win
- especially useful once other loader issues are improved

## Priority Order

If the goal is "make this feel faster soon", the recommended order is:

1. lazy PLE loading
2. offline FP32 runtime artifact
3. optimized linear algebra kernels
4. dedicated logits projection optimization
5. safetensors metadata caching

## Practical Recommendation

Short-term best path:

- lazy PLE loading
- offline FP32 artifact for the working Gemma model

Longer-term best path:

- optimized compute backend for the hot linear algebra paths

## Non-Goals For Now

These are not the recommended first moves for performance:

- adding EOS/stopping logic
- moving directly to 6-bit support in the current runtime
- rewriting the runtime around BF16-native math first

Those may be useful later, but they are not the highest-leverage next steps for
the current codebase.
