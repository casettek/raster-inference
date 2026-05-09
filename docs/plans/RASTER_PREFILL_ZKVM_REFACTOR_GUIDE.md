# Raster Prefill zkVM Refactor Guide

## Purpose

This document is the source material for generating implementation plans and executing the remaining raster prefill refactor safely.

The goal is to make raster-authored prefill execution feasible to replay inside a zkVM by ensuring individual tiles are bounded, explicit, and small enough to prove. The current raster path has already moved projection weight reads toward a row-chunked shape, but the prefill layer execution still has coarse tile boundaries around work that is too large for realistic model sizes.

Use this document when generating plans, splitting PRs, reviewing implementation, or deciding how much work to take on at once.

## Current State

The raster inference path is selected by `InferenceControls.raster_tiles` in `src/lib.rs`. Native fp32 and native deterministic inference should remain unaffected by all work described here.

Already implemented foundations:

- `InferenceControls.raster_projection_rows_per_tile` controls raster projection row chunking.
- `--raster-projection-rows-per-tile` exposes that setting in the CLI.
- Raster prefill projections use chunked row reads through `RasterSequenceProjectionState`.
- Model-backed PLE projection row reads no longer materialize the full PLE projection matrix.
- Raster inference reports `raster_tile_invocations` in completed and paused outputs.
- Tile invocation counting happens at the raster authoring macro layer and is only enabled for raster inference.

Important existing files:

- `src/lib.rs`: raster control flow and inference summary output.
- `src/raster_authoring.rs`: tile/sequence call macros and tile invocation counting.
- `src/prefill_layer/raster_tiles.rs`: current coarse prefill layer raster implementation.
- `src/shared/raster_transformer_kernels.rs`: deterministic raster kernels and chunked projection helpers.
- `src/prefill_prepare_aux/raster_tiles.rs`: PLE prefill raster implementation.
- `src/prefill_finalize/raster_tiles.rs`: logits projection pattern using recursive bounded projection.
- `RASTER_TILE_AUTHORING_GUIDE.md`: general raster authoring rules.

## Problem Frame

`compute_next_prefill_layer` in `src/prefill_layer/raster_tiles.rs` is currently the heaviest tile. One invocation processes an entire transformer prefill layer over the whole prompt.

It currently:

- reads layer metadata
- resolves donor cache and optional PLE inputs
- runs the full attention block
- runs the full MLP block
- optionally runs PLE layer work
- builds/updates KV cache
- serializes and hashes layer checkpoint state
- advances the layer index

Inside that work, `run_basic_prefill_layer` performs many operations over the full prompt sequence. Some projection work is now chunked, but attention and elementwise sequence/head operations remain large.

If the zkVM treats `compute_next_prefill_layer` as a single tile execution, this is likely infeasible for real model sizes.

## Core Invariants

Every plan and implementation must preserve these invariants:

- Only raster inference changes. Native fp32 and non-raster deterministic paths stay behaviorally unchanged.
- Raster output parity with deterministic inference is the primary correctness requirement.
- Existing checkpoints and commitments should remain unchanged unless a plan explicitly says trace shape is intentionally changing.
- Tile inputs and outputs should be explicit, owned, serializable, and deterministic.
- Tiles should request narrow external data through `auth_read!`.
- Sequences should orchestrate. Tiles should do bounded deterministic work.
- Recursive state machines should handle loops whose trip count depends on prompt length, model shape, layer count, head count, or vocabulary size.
- Tile invocation count should continue to mean tile invocations only. Sequences are orchestration and should not count as tiles.

## Feasibility Rules For Tiles

A tile is a good zkVM unit when its work is bounded by a small configurable chunk or a single semantic row.

Good tile shapes:

- one projection row chunk
- one token row RMSNorm
- one head row RMSNorm
- one RoPE row for one head/token
- one attention output row for one query head and token
- one token row add/multiply/GELU
- one narrow metadata or scalar read

Risky tile shapes:

- a full transformer layer
- all heads and all prompt tokens for attention
- full-sequence RMSNorm or GELU for large prompts
- full prompt reshape/combine across all heads
- full KV cache construction for all heads/tokens
- full checkpoint serialization inside a tile if the payload is large

Prefer this bound:

```text
tile_work ~= chunk_size * row_width
```

Avoid this bound inside a single tile:

```text
tile_work ~= layers * prompt_tokens * hidden_width
tile_work ~= heads * prompt_tokens^2 * head_dim
tile_work ~= vocab_size * hidden_width
```

## Plan Generation Strategy

Do not create one giant plan for the full refactor. Generate several implementation plans and execute them one at a time.

Each plan should leave the raster path compiling, tested, and measurable. Do not combine broad structural refactors with risky math/state-machine changes unless the work is tiny.

Recommended plans:

1. Plan A: Make prefill layer execution orchestration-first.
2. Plan B: Chunk prefill attention.
3. Plan C: Chunk remaining sequence/head elementwise operations.
4. Plan D: Measurement, trace review, and cleanup.

Plan A should happen first. Plan B is the largest correctness risk. Plan C can be split further if Plan B exposes enough remaining heavy tiles. Plan D may be optional if existing global tile count is enough.

## Plan A: Make Prefill Layer Execution Orchestration-First

### Goal

Stop treating an entire prefill layer as one heavy tile.

`compute_next_prefill_layer` should no longer be the proof unit for the whole layer body. The layer loop should delegate to sequences and smaller tiles.

### Key Design Decision

Introduce fallible recursive sequence support if needed.

Today `compute_next_prefill_layer` is invoked through `call_recur_tile!`. If it becomes a recursive sequence returning `Result<(bool, State)>`, the raster authoring layer likely needs a `call_recur_seq!` macro that mirrors `call_recur_tile!`.

This macro should:

- repeatedly call a recursive sequence returning `Result<(bool, State)>`
- stop on `done`
- propagate errors
- not increment tile invocation count itself
- let tile calls inside the sequence continue to count normally

### Implementation Shape

Replace the current coarse layer tile with an orchestration sequence:

```text
run
  init_prefill_layer_state tile
  call_recur_seq compute_next_prefill_layer_sequence
  finalize_prefill_layer_state tile
```

`compute_next_prefill_layer_sequence` should:

- check whether all layers are complete
- call a small tile to read/validate layer metadata and prepare layer context
- call `run_prefill_layer_sequence`
- call a tile or small helper to update `PrefillLayerRasterState`
- emit checkpoint data in a bounded or clearly intentional place

`run_prefill_layer_sequence` should be straight-line orchestration:

```text
run_prefill_layer_sequence
  run_prefill_attention_block
  run_prefill_mlp_block
  run_prefill_ple_block if applicable
  apply optional layer scalar
```

### Files To Plan Around

- `src/raster_authoring.rs`
- `src/prefill_layer/raster_tiles.rs`
- `src/shared/raster_transformer_kernels.rs`

### Tests To Include

- `call_recur_seq!` runs until done and propagates errors.
- Tile invocation counting is unchanged for recursive sequences: sequence loops do not count, but nested tile calls do.
- Existing raster prefill layer parity tests still pass.
- Terminal checkpoint output still includes `raster_tile_invocations`.
- Non-raster deterministic pause tests still omit `raster_tile_invocations`.

### Done Criteria

- No tile named `compute_next_prefill_layer` performs the full layer body.
- The layer loop is visibly sequence orchestration.
- Projection chunking behavior remains unchanged.
- `cargo test` passes.

## Plan B: Chunk Prefill Attention

### Goal

Replace full attention over all heads/tokens inside a single call with bounded recursive attention tiles.

The original high-risk helper was `causal_attention_heads_with_cache` in `src/shared/raster_transformer_kernels.rs`. It loops over query heads and query tokens, and each `attention_output_row` scans the visible key/value rows. The raster path should keep that behavior decomposed into explicit recursive phases: score collection, softmax max scan, exponent sum, raw-weight build, residual correction, and value application.

### Key Design Decision

Attention should be computed by recursive state over query head and query token.

Start with one output row per tile. Add a configurable chunk size only after proving the one-row shape is correct and if tile counts become too high.

State should include:

- queries
- keys
- values
- optional donor cache or an owned/cache reference strategy compatible with replay
- attention window
- current query head index
- current query token index
- output heads built so far
- derived KV grouping metadata

Each recursive tile should process at most one bounded chunk for the current `(query_head_idx, query_token_idx)`: key rows during score collection, score rows during softmax, weight rows during residual correction, or value rows during weighted-sum application.

### Implementation Shape

```text
run_prefill_attention_rows
  init_attention_state tile
  project_next_attention_row recursive tile
  finalize_attention_state tile
```

`project_next_attention_row` should:

- map query head to KV head
- select the allowed key/value range
- collect score rows through bounded authenticated reads
- compute exact softmax through bounded recursive passes over committed score/weight refs
- apply corrected weights to value rows through bounded authenticated reads
- append output row to state
- advance token/head indices

### Files To Plan Around

- `src/prefill_layer/raster_tiles.rs`
- `src/shared/raster_transformer_kernels.rs`

### Tests To Include

- Full attention with a small two-token fixture matches existing `causal_attention_heads_with_cache`.
- Sliding-window attention matches existing behavior.
- Donor-cache attention matches existing behavior.
- Multi-head query with grouped KV heads maps query heads to KV heads correctly.
- Equal max logits split across chunks preserve first-index softmax residual correction.
- Recursive attention state serializes refs, builders, cursors, and scalar bits rather than materialized score or weight rows.
- Empty or malformed head/cache shapes fail closed with existing-style errors.
- Raster prefill layer parity tests still pass.

### Done Criteria

- `run_basic_prefill_layer` or its replacement no longer calls full `causal_attention_heads_with_cache` as one large operation.
- The largest attention tile is bounded to one configured row chunk for score collection, softmax, residual correction, or value application.
- Existing checkpoint commitments remain stable unless a plan explicitly accepts trace changes.

## Plan C: Chunk Remaining Sequence And Head Operations

### Goal

Remove remaining full-sequence or full-head operations from individual tiles.

After Plans A and B, projections and attention should be bounded. Remaining operations may still be heavy for long prompts or large hidden widths.

Candidates:

- `rms_norm_sequence`
- `rms_norm_heads`
- `value_rms_norm_heads`
- `gelu_sequence`
- `mul_sequences`
- `add_sequences`
- `scale_sequence`
- `apply_rope_to_heads`
- `reshape_sequence_heads`
- `combine_attention_heads`
- `build_raster_kv_cache`

### Implementation Shape

Create row/head-recursive variants where needed:

```text
run_sequence_rms_norm
  init state tile
  normalize_next_row recursive tile
  finalize tile

run_head_rms_norm
  init state tile
  normalize_next_head_row recursive tile
  finalize tile

run_rope_heads
  init state tile
  rotate_next_head_row recursive tile
  finalize tile
```

Do not chunk everything blindly. Prioritize operations that show up as heavy in tile counts, traces, or expected complexity.

### Files To Plan Around

- `src/prefill_layer/raster_tiles.rs`
- `src/shared/raster_transformer_kernels.rs`

### Tests To Include

- Each chunked helper matches its current full helper for small deterministic fixtures.
- Edge cases for empty sequences, mismatched widths, head count mismatch, and zero-width rows.
- End-to-end raster prefill layer parity.

### Done Criteria

- No tile still performs large prompt-length by hidden-width work unless it is explicitly accepted as feasible.
- Tests demonstrate parity between full helper and chunked helper.

## Plan D: Measurement, Trace Review, And Cleanup

### Goal

Use measurement to decide whether additional chunking is needed and make the final trace shape understandable.

### What To Measure

- `raster_tile_invocations` for terminal checkpoints:
  - `prompt.prepare`
  - `prefill.prepare_aux`
  - `prefill.layer`
  - `prefill.finalize`
  - `output.finalize`
- Whether the tile count moves as expected when `--raster-projection-rows-per-tile` changes.
- Whether a real model can reach later checkpoints without zkVM memory pressure.

### Possible Enhancements

Only if needed:

- Add optional phase-level tile counts.
- Add optional routine-level tile counts.
- Add per-heavy-sequence counts for attention/projection.

Avoid adding per-tile names or detailed event logs by default. The current requirement is a low-overhead count, not tracing every tile.

### Tests To Include

- Paused raster output includes `raster_tile_invocations`.
- Completed raster output includes `raster_tile_invocations`.
- Non-raster output omits `raster_tile_invocations`.
- Tile counter is reset after errors and pauses.

## Implementation Rules For Future Agents

When implementing any plan generated from this document:

1. Read `RASTER_TILE_AUTHORING_GUIDE.md` first.
2. Keep changes scoped to raster paths unless a shared helper must be added for raster use.
3. Do not change native fp32 or non-raster deterministic execution.
4. Keep sequence functions as orchestration.
5. Keep tiles bounded and semantic.
6. Use `auth_read!` for external data.
7. Keep persistent state owned and serializable.
8. Preserve deterministic arithmetic.
9. Run focused tests after each unit, then `cargo test`.
10. Check `raster_tile_invocations` after meaningful raster changes.

## How Much To Implement At Once

Implement one plan at a time.

Plan A can usually land as one PR because it is structural and should preserve math behavior.

Plan B should be treated carefully. Prefer sub-steps:

1. Build attention state and one-row attention tile behind helper tests.
2. Wire prefill attention to use it.
3. Add donor cache and sliding-window coverage.
4. Run full raster parity.

Plan C can be split by operation family:

- sequence row ops
- head row ops
- reshape/combine/cache ops

Do not implement Plan B and Plan C in the same pass unless Plan B turns out to be trivial. Attention is the riskiest change and should be isolated.

## Plan Template For Generated Plans

Each generated plan should include:

- problem frame tied to this document
- scope boundaries
- requirements trace
- implementation units with U-IDs
- files to modify
- tests to add
- invariants to preserve
- measurement expectations
- risks and mitigations

Minimum requirements for every plan:

- R1. Raster output remains deterministic-parity equivalent.
- R2. Native fp32 and non-raster deterministic paths are unchanged.
- R3. Each new tile has bounded work.
- R4. Existing terminal checkpoint behavior keeps working.
- R5. `raster_tile_invocations` remains present for paused/completed raster inference.

## Anti-Patterns

Avoid these:

- Moving loops from one helper into another helper without creating tile boundaries.
- Calling a sequence from inside a tile and assuming that makes the outer tile feasible.
- Adding hidden global configuration for chunk sizes.
- Using native floating point in replay-critical raster paths.
- Passing borrowed views in persistent recursive state.
- Materializing full model matrices in authenticated row reads.
- Combining attention chunking with broad elementwise chunking in one hard-to-review change.

## Suggested First Plan Prompt

Use this prompt to generate the first implementation plan:

```text
Create a plan for Plan A from RASTER_PREFILL_ZKVM_REFACTOR_GUIDE.md:
make raster prefill layer execution orchestration-first. The plan should convert
the current coarse compute_next_prefill_layer tile into recursive sequence
orchestration, introduce fallible recursive sequence support if needed, preserve
projection chunking, keep native and non-raster deterministic paths unchanged,
and include tests for parity, macro behavior, and tile invocation counting.
```

## Suggested Second Plan Prompt

Use this prompt after Plan A is complete:

```text
Create a plan for Plan B from RASTER_PREFILL_ZKVM_REFACTOR_GUIDE.md:
chunk raster prefill attention so each tile computes one query head/token row or
a small bounded chunk. Preserve deterministic parity, support sliding windows
and donor caches, keep checkpoints stable unless explicitly justified, and add
focused helper tests plus raster prefill layer parity tests.
```

## Final Success Criteria

The full refactor is successful when:

- no raster tile performs an entire transformer prefill layer
- projection tiles are row-chunked
- attention tiles are head/token-chunked
- remaining sequence/head operations are either chunked or explicitly measured as feasible
- raster inference can pause at every existing terminal checkpoint
- paused/completed raster outputs include `raster_tile_invocations`
- native fp32 and non-raster deterministic behavior remain unchanged
- `cargo test` passes
