# Raster zkVM State Refactor Roadmap

## Purpose

This document is source material for writing detailed implementation plans that make raster-authored inference practical to replay in a zkVM.

Phases A, B, and C made prefill layer execution much more granular by replacing coarse transformer-layer work with sequences and row/head-row recursive tiles. Those changes reduced the amount of arithmetic inside many individual tiles.

The remaining problem is different: many recursive tiles still carry large owned tensors in their serialized state. A tile may compute only one row, but if its input state includes the full prompt activation tensor, full Q/K/V heads, full donor cache, or output rows accumulated so far, one zkVM replay invocation may still be too large.

This roadmap describes the next refactor family: move from value-carrying recursive state to reference-carrying recursive state.

## Current Problem

The current bounded helpers often have this shape:

```text
state = full_input_tensor + full_output_so_far + cursor
recursive_tile(state) -> state
```

That is good for correctness and local parity testing, but risky for zkVM replay because every recursive tile invocation may need to deserialize, hash, copy, and return the full state.

The desired shape is:

```text
state = input_refs + output_ref_or_commitment + cursor + shape metadata
recursive_tile(state, authenticated_sources) -> small row result or compact state update
```

Each tile should read only the row or rows it needs, compute one semantic row/chunk, and update a compact commitment or output reference.

## Design Principle

A zkVM-feasible tile should bound both:

- **Compute work:** the arithmetic inside one tile invocation.
- **State materialization:** the serialized arguments and return value for one tile invocation.

Earlier phases mostly bounded compute. This roadmap is about bounding state.

## Important Constraints

- Raster behavior must remain deterministic-parity equivalent.
- Existing checkpoint commitments should remain stable unless a plan explicitly accepts a trace/checkpoint shape change.
- Tile and sequence invocation counting should continue to count every executed tile and sequence.
- Tiles must not call other tiles or sequences.
- Sequences should orchestrate by calling tiles, recursive tiles, recursive sequences, or other sequences.
- Large model data and intermediate tensors should be accessed through explicit authenticated references and typed row requests.
- Avoid adding a generic abstraction that cannot be tested incrementally against current helpers.

## Terms

### Value-Carrying State

Recursive state that owns the data it operates on.

Examples:

```text
input: RasterActivationSequence
queries: RasterAttentionHeadSequence
output_rows: Vec<RasterActivationRow>
donor_cache: RasterKvCache
```

### Reference-Carrying State

Recursive state that carries references, dimensions, cursors, and commitments instead of full tensors.

Example:

```text
input_ref: RasterTensorRef
output_ref: RasterTensorBuilderRef
next_row_idx: usize
row_count: usize
width: usize
```

### Intermediate Tensor Store

A logical store for raster intermediate tensors. It may start as an in-memory implementation, but the API should model the future zkVM contract: read typed rows through authenticated requests and verify shapes/commitments.

## Proposed Core Abstractions

The exact names can change during implementation, but plans should converge on these concepts.

```text
RasterTensorRef
  id/source name
  shape
  commitment
  kind

RasterTensorShape
  row_count
  width
  optional head_count
  optional sequence_len
  optional head_dim

RasterRowRequest
  tensor_ref
  row index

RasterHeadRowRequest
  tensor_ref
  head index
  token index

RasterKvRowRequest
  cache_ref
  key/value selector
  head index
  token index or window

RasterTensorBuilder
  expected shape
  rows written
  running commitment
```

The first implementation does not need to be perfect or final. It does need to make row-level access explicit and testable.

## Target Architecture

```text
sequence orchestration
  init state tile
    validates shape refs
    initializes compact cursor/commitment state

  recursive row tile
    reads one row or bounded row window through auth_read
    computes one semantic row
    appends row result or updates compact commitment

  finalize tile
    verifies row count and commitment
    returns a tensor ref or final native value when required by public output
```

The ideal tile invocation should not carry an entire activation sequence or attention head tensor as a function argument just to compute one row.

## Roadmap

Do not implement this as one giant change. Use the following sequence of plans.

### Plan 0: Design The Intermediate Tensor Contract

Goal: decide the reference and row-access model before rewriting call sites.

Questions to resolve:

- What is the minimal `RasterTensorRef` shape needed for activations, head tensors, KV caches, and partial outputs?
- Are intermediate tensors represented as one generic tensor shape, or as separate sequence/head/KV reference types?
- How are commitments computed and verified?
- Does a builder append concrete rows in memory for now, or only maintain commitments?
- How does the host provide authenticated row reads for intermediate outputs from prior tiles?
- What failure modes should be fail-closed?
- Which trace/checkpoint commitments must stay byte-for-byte stable?

Expected output:

- A design document or implementation plan that names the core types and request objects.
- A minimal test strategy for shape validation, row reads, commitment validation, and bad-index failures.
- A scope boundary that avoids rewriting all prefill operations in the first implementation PR.

### Plan 1: Add Reference-Backed Row Store Primitives

Goal: add the core abstraction without changing prefill behavior.

Likely files:

- `src/shared/raster_transformer_kernels.rs`
- new shared module if the abstraction grows too large
- focused tests near the abstraction

Implementation shape:

- Add reference types for sequence rows and head rows.
- Add authenticated read request types for rows.
- Add an in-memory implementation for tests and current raster execution.
- Add commitment helpers for tensor rows and full tensors.
- Add builder/finalizer logic that can verify expected row count and shape.

Tests:

- Reading a valid row returns the expected deterministic row.
- Out-of-range row/head indexes fail closed.
- Shape mismatches fail closed.
- Commitment mismatch fails closed if commitment validation is part of Plan 1.
- Builder finalization rejects missing rows, duplicate rows, wrong width, and wrong row count.

Done criteria:

- Existing raster behavior remains unchanged.
- New abstractions have enough tests to support one downstream conversion.

### Plan 2: Convert Attention State First

Goal: remove full Q/K/V/donor/output tensors from `RasterAttentionRowState`.

Why first:

- Attention state is one of the largest current recursive states.
- Attention already has clear row semantics: one `(query_head_idx, query_token_idx)` output row.
- The current implementation already has parity tests against `causal_attention_heads_with_cache`.

Current risk:

```text
RasterAttentionRowState
  queries: RasterAttentionHeadSequence
  keys: RasterAttentionHeadSequence
  values: RasterAttentionHeadSequence
  donor_cache: Option<RasterKvCache>
  output_heads: Vec<Vec<RasterActivationRow>>
```

Target:

```text
RasterAttentionRowState
  query_ref
  key_ref
  value_ref
  donor_cache_ref
  output_builder_ref or output commitment
  next_query_head_idx
  next_query_token_idx
  shape metadata
```

Implementation notes:

- Read one query row per tile.
- Read key/value rows for the visible causal or sliding window.
- Keep donor-cache semantics exactly the same.
- If one attention row still proves too expensive, split attention further into score, softmax/reduction, weighted-sum, and finalize-row states. Do not do that until measurement or obvious cost requires it.

Tests:

- Reference-backed attention matches current full helper for full attention.
- Sliding-window attention matches current behavior.
- Donor-cache attention matches current behavior.
- Grouped KV head mapping remains correct.
- Bad references, shapes, and row indexes fail closed.

### Plan 3: Convert Sequence Unary And Binary States

Goal: remove full activation sequences from row-wise sequence states.

Targets:

- RMSNorm sequence
- GELU sequence
- scale sequence
- add sequences
- multiply sequences

Current risk:

```text
RasterSequenceUnaryState
  input: RasterActivationSequence
  output_rows: Vec<RasterActivationRow>

RasterSequenceBinaryState
  lhs: RasterActivationSequence
  rhs: RasterActivationSequence
  output_rows: Vec<RasterActivationRow>
```

Target:

```text
unary_state
  input_ref
  output_builder_ref or commitment
  next_row_idx
  row_count
  width
  op metadata

binary_state
  lhs_ref
  rhs_ref
  output_builder_ref or commitment
  next_row_idx
  row_count
  width
  op metadata
```

Tests:

- Reference-backed RMSNorm/GELU/scale match full helpers.
- Reference-backed add/mul match full helpers.
- Missing norm weights, scalar, wrong width, wrong row count, and bad references fail closed.

### Plan 4: Convert Head, Layout, And KV States

Goal: remove full head tensors and KV caches from head/layout/cache recursive states.

Targets:

- head RMSNorm
- value RMSNorm
- RoPE
- reshape sequence to heads
- combine heads to sequence
- build KV cache

Current risk:

```text
RasterHeadUnaryState
  heads: RasterAttentionHeadSequence
  output_heads: Vec<Vec<RasterActivationRow>>

RasterReshapeHeadsState
  input: RasterActivationSequence
  output_heads: Vec<Vec<RasterActivationRow>>

RasterKvCacheBuildState
  keys: RasterAttentionHeadSequence
  values: RasterAttentionHeadSequence
  output_keys/output_values
```

Target:

```text
head_state
  head_tensor_ref
  output_ref/commitment
  next_head_idx
  next_token_idx
  shape metadata

layout_state
  input_ref/head_ref
  output_ref/commitment
  next row/head/token cursor

kv_cache_state
  key_ref
  value_ref
  output_cache_ref/commitment
  retained_start
  next_head_idx
  next_token_idx
```

Tests:

- Reference-backed head ops match current helpers.
- Reference-backed reshape/combine match current helpers.
- Reference-backed KV cache build matches current suffix retention behavior.
- Shape mismatch and bad row/head indexes fail closed.

### Plan 5: Convert Layer-Level State Where Practical

Goal: reduce the size of `PrefillLayerRasterState` and layer context movement.

Current risk:

```text
PrefillLayerRasterState
  current_activations: RasterActivationSequence
  layer_caches: Vec<RasterKvCache>
  per_layer_inputs: Vec<Option<RasterActivationSequence>>
  completed_layer_output_sha256s
```

Potential target:

```text
PrefillLayerRasterState
  current_activation_ref
  layer_cache_refs
  per_layer_input_refs
  completed commitments
  next_layer_idx
  layer_count
```

This plan is more sensitive because checkpoint outputs and public return values may require concrete tensors. It should happen only after the lower-level row-store abstraction is stable.

Tests:

- Existing raster prefill parity still passes.
- Existing terminal checkpoint behavior still works.
- Completed layer commitments remain stable.
- Donor-cache resolution still rejects non-prior donors.

### Plan 6: Measurement, Cleanup, And Optional Further Splits

Goal: use real tile counts and zkVM measurements to decide what remains too heavy.

Measure:

- tile invocation count by terminal checkpoint
- largest tile input/output serialized size
- largest tile cycle count in zkVM
- worst offenders by operation type

Likely follow-up if needed:

- Split attention row into score/softmax/weighted-sum phases.
- Add optional small chunks where one-row state overhead is too high.
- Reduce duplicate tensor materialization in test-only or host-only paths.

## Suggested Planning Prompts

Use these prompts to generate detailed implementation plans.

### Design Plan Prompt

```text
Create a design plan for the raster intermediate tensor reference contract.
Use RASTER_ZKVM_STATE_REFACTOR_ROADMAP.md as source material. Define the
minimal row-reference, shape, commitment, row request, and builder/finalizer
abstractions needed to replace value-carrying recursive states in raster
prefill. Do not rewrite prefill call sites in this plan. Include failure modes,
test scenarios, and compatibility requirements for existing checkpoint
commitments.
```

### Plan 1 Prompt

```text
Create an implementation plan for adding reference-backed raster row store
primitives. Use RASTER_ZKVM_STATE_REFACTOR_ROADMAP.md as source material.
Implement only the core reference, row request, shape, commitment, and in-memory
test source/builder primitives. Existing raster prefill behavior should remain
unchanged.
```

### Plan 2 Prompt

```text
Create an implementation plan for converting RasterAttentionRowState to use
reference-backed Q/K/V, donor cache, and output rows. Preserve full, sliding,
donor-cache, and grouped-KV attention behavior. Existing attention parity tests
must continue to pass.
```

### Plan 3 Prompt

```text
Create an implementation plan for converting raster sequence unary and binary
states to reference-backed row access. Cover RMSNorm, GELU, scale, add, and
multiply. Preserve deterministic arithmetic semantics and existing parity tests.
```

### Plan 4 Prompt

```text
Create an implementation plan for converting head-row, reshape/combine, and KV
cache build states to reference-backed row access. Preserve RoPE position
semantics, head ordering, combine ordering, and KV sliding-window suffix
retention.
```

### Plan 5 Prompt

```text
Create an implementation plan for reducing PrefillLayerRasterState and
PrefillLayerContext size using tensor/cache references where practical. Preserve
checkpoint commitments, terminal checkpoint behavior, donor-cache semantics, and
end-to-end raster prefill parity.
```

## Non-Goals

- Do not introduce a new public inference mode.
- Do not change native fp32 or non-raster deterministic execution.
- Do not change tokenizer, decode, or output finalize behavior unless a later plan explicitly scopes them in.
- Do not optimize by hiding large work in a plain helper called from a tile.
- Do not treat exact tile counts as stable public output.

## Success Criteria

The refactor family is successful when:

- no recursive tile carries full prompt/layer tensors merely to compute one row;
- tile serialized inputs and outputs are bounded by refs, cursors, row data, or small metadata;
- row reads are explicit and authenticated;
- existing raster checkpoint commitments remain stable or any intentional changes are documented;
- real zkVM measurements show no single prefill-layer tile invocation with impractical cycle or memory cost.
