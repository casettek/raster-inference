# Raster Intermediate Tensor References Design

## Purpose

This document defines the reference-backed intermediate tensor contract for raster-authored inference.

The current raster prefill path has bounded most arithmetic by using recursive row/head-row tiles, but many recursive states still carry full tensors. That makes each tile invocation expensive to replay in a zkVM because the tile state itself may contain full activation sequences, attention heads, donor caches, and output rows accumulated so far.

This design keeps the existing row-tile granularity but changes the state model from value-carrying to reference-carrying.

## Non-Goals

- Do not implement the reference store in this document.
- Do not convert prefill call sites in the first implementation plan.
- Do not change checkpoint JSON payload shape.
- Do not change public inference APIs or CLI flags.
- Do not change deterministic arithmetic semantics.
- Do not split attention rows into score/softmax/weighted-sum phases yet.

## Required Invariants

- Raster output must remain deterministic-parity equivalent.
- Existing CPU deterministic and raster checkpoint commitments must remain byte-compatible.
- Tile and sequence invocation counting remains unchanged.
- Tiles must not call tiles or sequences.
- Sequences remain orchestration only.
- All large intermediate reads must be explicit and typed.
- Bad shape, bad reference, bad row index, and bad commitment cases fail closed.

## Current Problem

Current recursive states commonly look like this:

```text
RasterAttentionRowState
  queries: RasterAttentionHeadSequence
  keys: RasterAttentionHeadSequence
  values: RasterAttentionHeadSequence
  donor_cache: Option<RasterKvCache>
  output_heads: Vec<Vec<RasterActivationRow>>
  cursor fields
```

This is row-bounded in compute but not row-bounded in state. A zkVM tile replay may need to deserialize and return the whole state for every row.

The desired shape is:

```text
RasterAttentionRowState
  query_ref
  key_ref
  value_ref
  donor_cache_ref
  output_builder_ref
  cursor fields
  shape metadata
```

The tile then reads only the row or bounded row window it needs.

## Core Decision Summary

- Use explicit typed reference structs, not hidden globals.
- Use `AuthRead<Request>` and `auth_read!` as the row-read mechanism.
- Use one shared shape model with typed wrappers for sequence, head, and KV use cases.
- Keep an in-memory intermediate store for current execution and tests.
- Store enough metadata in refs to validate shape before row reads.
- Store commitments in refs/builders so reads and finalization can be verified.
- Materialize tensors at public/checkpoint boundaries to preserve current commitments.
- Carry `RasterArtifactStoreRoots` as a separate routine boundary value, not inside routine-specific refs.
- Bind static model/tokenizer sources through committed roots or explicit static source refs instead of free-form source-name comparisons.

## Reference Taxonomy

### Shared Identity

Every intermediate reference should have a stable identifier.

```text
RasterTensorId
  source_name: String
```

The identifier should be opaque to tile code. Tile code should not infer shape or provenance from the string.

### Tensor Kind

Use a small enum to prevent accidentally reading a head tensor as a flat sequence.

```text
RasterTensorKind
  ActivationSequence
  AttentionHeads
  KvCacheKeys
  KvCacheValues
  PartialOutput
```

KV cache key/value can be represented either as separate tensor refs or as a `RasterKvCacheRef` containing two refs. The first implementation should prefer the explicit `RasterKvCacheRef` wrapper because it preserves key/value semantics and avoids mixing cache halves.

### Shape

Use a shared shape enum rather than a bag of optional fields.

```text
RasterTensorShape
  Sequence {
    row_count: usize
    width: usize
  }

  Heads {
    head_count: usize
    sequence_len: usize
    head_dim: usize
  }

  KvCache {
    head_count: usize
    current_len: usize
    head_dim: usize
  }
```

Rationale:

- `Sequence` maps to `RasterActivationSequence`.
- `Heads` maps to `RasterAttentionHeadSequence`.
- `KvCache` maps to `RasterKvCache`.
- This avoids optional fields that can create invalid combinations.

### References

Recommended public design shape:

```text
RasterTensorRef
  id: RasterTensorId
  kind: RasterTensorKind
  shape: RasterTensorShape
  det_commitment: String
```

Typed wrappers:

```text
RasterActivationSequenceRef(RasterTensorRef)
RasterAttentionHeadsRef(RasterTensorRef)

RasterKvCacheRef
  keys: RasterTensorRef
  values: RasterTensorRef
  shape: RasterTensorShape::KvCache
  det_commitment: String
```

The wrappers should enforce kind/shape compatibility at construction time.

### Routine Outputs

Each raster routine should return a proof-shaped output envelope:

```text
RasterRoutineOutput
  artifact_store_roots
  refs
```

The `refs` value should contain only compact artifact refs and small public metadata. It should not duplicate the artifact-store root snapshot. This keeps the one root chain explicit as prompt preparation, input embedding, prefill, decode select/transition, and output finalization append artifacts.

## Commitment Model

### Internal Deterministic Commitments

Intermediate refs should use deterministic fixed-point bytes as the canonical internal commitment source.

For activation/head rows, commit the same `Act` bit values currently represented by `RasterActivationRow`.

The internal commitment should be domain-separated from existing public commitments. A future implementation can define exact bytes, but the design should reserve tags like:

```text
raster-intermediate-sequence-v1
raster-intermediate-heads-v1
raster-intermediate-kv-cache-v1
raster-intermediate-row-v1
```

### Public Commitment Compatibility

Existing public commitments must not change:

- `build_activation_commitment(&[Vec<f32>])`
- `build_det_activation_commitment(&[Vec<Act>])`
- `build_det_vector_commitment(&[Act])`
- `build_det_kv_cache_commitment(&[LayerKvCache])`
- trace `sha256_hex(serde_json::to_vec(state))`

Reference-backed execution is internal. At checkpoint/public-output boundaries, refs must resolve to the same row-major f32 and deterministic `Act` values used today.

### Where Materialization Is Required

Materialization remains required at:

- `prefill.layer` checkpoint payloads
- terminal `prefill.layer_token.*` payloads
- `prefill.finalize`
- any public `ActivationSequence` output
- any public `LayerKvCache` output

The checkpoint code should not serialize refs unless a future versioned trace format explicitly chooses that.

## Authenticated Row Request API

The implementation should model intermediate tensors as authenticated sources.

### Sequence Rows

```text
RasterSequenceRowRequest
  tensor_ref: RasterActivationSequenceRef
  row_idx: usize
```

Output:

```text
RasterActivationRow
```

Validation:

- tensor kind is `ActivationSequence`
- shape is `Sequence`
- `row_idx < row_count`
- row width equals shape width
- row is covered by the tensor commitment, either directly or by source-level verification

### Head Rows

```text
RasterHeadRowRequest
  tensor_ref: RasterAttentionHeadsRef
  head_idx: usize
  token_idx: usize
```

Output:

```text
RasterActivationRow
```

Validation:

- tensor kind is `AttentionHeads`
- shape is `Heads`
- `head_idx < head_count`
- `token_idx < sequence_len`
- row width equals `head_dim`

### KV Cache Rows

```text
RasterKvRowKind
  Key
  Value

RasterKvRowRequest
  cache_ref: RasterKvCacheRef
  row_kind: RasterKvRowKind
  head_idx: usize
  token_idx: usize
```

Output:

```text
RasterActivationRow
```

Validation:

- key/value shape matches the `RasterKvCacheRef`
- `head_idx < head_count`
- `token_idx < current_len`
- row width equals `head_dim`

### Window Reads

Attention needs a visible K/V window. Do not introduce a separate large window payload unless measurement proves repeated row reads are too expensive.

Start with either:

```text
for token_idx in start..end:
  auth_read(RasterKvRowRequest { token_idx })
```

or a bounded request:

```text
RasterKvRowsWindowRequest
  cache_ref
  row_kind
  head_idx
  start
  len
```

If using the window form, the request must still be bounded by the current attention row's visible range and must validate `start + len <= current_len`.

## Output Builder Contract

The builder exists to avoid carrying `Vec<RasterActivationRow>` in recursive state.

Recommended shape:

```text
RasterTensorBuilderRef
  id: RasterTensorId
  kind: RasterTensorKind
  expected_shape: RasterTensorShape
  rows_written: usize
  running_commitment: String
```

The first implementation should keep materialized rows in an in-memory store behind the builder ref. The recursive state carries only the builder ref, not the rows.

### Append Requests

```text
RasterAppendSequenceRowRequest
  builder_ref
  row_idx
  row

RasterAppendHeadRowRequest
  builder_ref
  head_idx
  token_idx
  row

RasterAppendKvRowRequest
  builder_ref
  key_or_value
  head_idx
  token_idx
  row
```

Append validation:

- builder kind matches request kind
- row index is in bounds
- row width matches expected shape
- duplicate writes fail
- writes outside the declared shape fail
- builder commitment is updated deterministically

### Ordering

Prefer requiring the expected traversal order for the first implementation:

- sequence rows: row `0..row_count`
- heads: `(head_idx, token_idx)` in the same order existing state machines use
- KV cache: retained token rows in current suffix order

This makes the builder simpler and matches current recursive cursor semantics. If future plans need out-of-order writes, that should be a separate design change.

### Finalization

```text
RasterFinalizeTensorRequest
  builder_ref
```

Output:

```text
RasterTensorRef
```

Finalization validation:

- all expected rows are present
- final shape matches expected shape
- final commitment matches accumulated row bytes
- no duplicate or missing rows exist

For current execution, finalization may also support materialization:

```text
materialize_sequence(tensor_ref) -> RasterActivationSequence
materialize_heads(tensor_ref) -> RasterAttentionHeadSequence
materialize_kv_cache(cache_ref) -> RasterKvCache
```

These materializers are for compatibility with existing checkpoint/public outputs and tests.

## Source And Store Model

### Recommended First Implementation

Use an explicit in-memory store argument for row reads and appends.

Example conceptual shape:

```text
AuthenticatedRasterTensorStore
  tensors
  builders
```

It implements:

```text
AuthRead<RasterSequenceRowRequest>
AuthRead<RasterHeadRowRequest>
AuthRead<RasterKvRowRequest>
AuthRead<RasterAppend...Request>
AuthRead<RasterFinalizeTensorRequest>
```

Although append/finalize have side-effect-like semantics, keep the API explicit and typed. If `AuthRead` is semantically too read-only for append operations, introduce a sibling trait in a later implementation plan only after evaluating the local DSL model.

### Why Not Hidden Globals

Hidden global stores would make tile inputs misleading and complicate zkVM replay. The store/source handle must be visible in tile signatures or represented as an explicit `External<T>` handle.

### Why Not Refs In Checkpoints

Refs are internal. Existing checkpoints serialize materialized f32 tensors and deterministic commitments. Replacing those with refs would change `serde_json` checkpoint bytes and break trace compatibility.

## Failure Modes

Every implementation plan should include tests for:

- unknown tensor ref
- wrong tensor kind
- wrong shape kind
- row index out of bounds
- head index out of bounds
- token index out of bounds
- wrong row width
- duplicate builder append
- skipped row or missing row on finalization
- append after finalization
- commitment mismatch
- materialization of incomplete tensor
- KV key/value shape mismatch
- donor cache ref with wrong head count

## First Conversion Target: Attention

After primitives are implemented, convert attention first.

Current state:

```text
RasterAttentionRowState
  queries
  keys
  values
  donor_cache
  output_heads
  next_query_head_idx
  next_query_token_idx
```

Target state:

```text
RasterAttentionRowState
  query_ref
  key_ref
  value_ref
  donor_cache_ref
  output_builder_ref
  next_query_head_idx
  next_query_token_idx
  sequence_len
  kv_head_count
  kv_groups
  attention_window
```

One recursive attention tile should:

1. read one query head row;
2. read current K/V rows or donor K/V rows for the visible window;
3. compute one attention output row;
4. append one output row to the output builder;
5. advance the cursor.

The conversion is successful when attention parity tests still pass and tile state no longer owns Q/K/V/donor/output tensors.

## Migration Sequence

Use the following implementation sequence:

1. Add row-store primitives without changing prefill behavior.
2. Convert `RasterAttentionRowState`.
3. Convert sequence unary/binary states.
4. Convert head/layout/KV states.
5. Reduce `PrefillLayerRasterState` and `PrefillLayerContext` size.
6. Measure tile input/output bytes and zkVM cycle costs.

Do not combine these into one implementation PR.

## Test Strategy For The Primitive Implementation

The first implementation plan should include tests for:

- creating refs for activation sequences, attention heads, and KV caches;
- stable serialization of refs and shapes;
- reading sequence rows;
- reading head rows;
- reading KV rows;
- appending rows to sequence/head/KV builders;
- finalizing complete builders;
- rejecting incomplete builders;
- rejecting duplicate appends;
- rejecting wrong-width rows;
- verifying final commitments against materialized tensors;
- materializing refs back into current owned tensor types;
- proving materialized refs produce the same existing f32 and deterministic commitments.

## Compatibility Rules

Reference-backed intermediates must preserve the following:

- `RasterActivationRow` remains the canonical deterministic row payload.
- `Act` bit ordering remains unchanged.
- f32 checkpoint compatibility remains row-major and little-endian through existing helpers.
- deterministic activation commitments remain domain-tagged as today.
- deterministic KV cache commitments remain domain-tagged as today.
- checkpoint JSON payloads remain materialized unless trace versioning is explicitly introduced.

## Deferred Questions

These questions should be answered during or immediately before the primitive implementation plan:

- Does `AuthRead` remain the right trait for append/finalize operations, or do we need a separate authenticated write/build trait?
- Should commitment verification happen on every row read, finalization only, or both?
- Should builder refs expose `rows_written` in recursive state, or should state carry the cursor and builder track only commitments?
- Should tensor refs include both deterministic and f32 commitments, or deterministic only with f32 materialization computed at checkpoint boundaries?
- How should row-store lifetimes map onto eventual zkVM public/private inputs?

## Done Criteria

The design is complete when a follow-up implementation plan can add primitives without deciding:

- what ref types exist;
- what shape metadata is required;
- how rows are requested;
- how output rows are appended;
- how builders finalize;
- how existing commitments remain compatible;
- which state converts first.
