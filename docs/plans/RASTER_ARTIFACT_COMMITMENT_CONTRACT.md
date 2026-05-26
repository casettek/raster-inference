# Raster Artifact Commitment Contract

## Purpose

This document defines the target commitment model for raster tile replay. It is the foundation for migrating raster inference to ref-based dynamic artifacts without losing deterministic CPU parity.

The goal is not to make every value indirect. The goal is to make every growable or reusable data value available through a compact public reference and verified private reads, while keeping small public control state explicit.

## Core Statement Model

Each replayed tile should be understandable as a statement over:

```text
public inputs:
  tile identity
  input artifact refs
  output artifact refs or builder refs
  shape metadata
  small control values such as indexes, cursors, layer ids, enum choices, and counts

private witness:
  concrete rows, chunks, token ids, or scalar payloads needed by auth_read calls
  inclusion proofs or source proofs for each authenticated read
  intermediate arithmetic values needed to compute the output

public outputs:
  updated compact state
  output artifact refs or finalized commitments
  small public values that are intentionally part of the transcript
```

The public statement should not include full tensors, full token streams, full text, full KV caches, or accumulated output rows. Those values belong in authenticated artifacts.

## Routine Boundary Contract

Raster routines should hand state to the next routine as a root-chain envelope plus compact refs:

```text
input:
  artifact_store_roots
  input artifact refs
  static source roots or static source refs
  small controls such as counts, positions, sizing, and stop reasons

output:
  artifact_store_roots
  output artifact refs
  small public summaries
```

Routine-specific refs must not own a full `RasterArtifactStoreRoots` snapshot. The active root chain is carried separately by the routine input/output envelope, while refs carry artifact identity, shape/count metadata, and their own commitment root. This preserves proof-shaped handoffs and makes stale-root failures explicit at the next routine boundary.

Materialized transformer structs remain compatibility outputs for public APIs and checkpoint payloads. Raster-to-raster handoffs should prefer proof-shaped refs and materialize only through named adapter functions at those public/checkpoint boundaries.

## Value Categories

### Inline Public State

Keep small values inline when they are public, bounded, and useful for understanding the tile transition:

- row indexes
- token indexes
- layer indexes
- head indexes
- cursor positions
- row counts
- widths and shape metadata
- enum choices
- done flags
- small deterministic scalar controls

These values do not need a store entry by default. They are already part of the tile statement.

### Dynamic Authenticated Artifacts

Use artifact refs for values that can grow with input, model, or decode size:

- prompt bytes and rendered text
- BPE piece sequences
- prompt token ID sequences
- embedded prompt activation sequences
- PLE input sequences
- activation sequences
- attention head tensors
- KV caches
- logits
- generated token streams
- generated text chunks

These artifacts should have kind, identity, shape or length metadata, and a commitment.

### Static Authenticated Sources

Static model and tokenizer data should keep using typed authenticated reads:

- tokenizer metadata and token lookups
- embedding rows
- projection rows
- norm weights
- scalar metadata
- layer metadata
- cache and attention configuration

Static sources and dynamic artifacts should look similar at the tile boundary: both are compact references plus narrow typed requests.

## Unified Authenticated Selection

Dynamic artifacts and static external sources share one authenticated selection model. A read first resolves a compact source identity plus a narrow selector into a verified selected payload, then typed helpers decode that payload into routine-specific values.

```text
artifact source + leaf index
  -> verified selected payload
  -> typed artifact value

external source + request key
  -> verified selected payload
  -> typed static source value
```

This mirrors Raster Runtime's external selection shape at the API level: a source name/root plus selector produces selected bytes, proof material, and a typed value. Raster inference differs in its commitment scheme: both artifact leaves and external request responses are verified against Merkle roots, rather than a whole-file SHA256 commitment.

The low-level selected payload should carry:

- source kind
- source name
- commitment root
- selector metadata
- selected bytes
- Merkle leaf index and leaf payload
- Merkle proof

Typed helpers remain the normal tile-facing API. Tiles should not manually parse Merkle proofs or decode arbitrary bytes unless they are adapter/helper boundaries for authenticated reads.

## Commitment Rules

### Merkle Roots For Store Artifacts

Anything written to an authenticated store must be committed with a Merkle root. This applies even when the current implementation happens to read the value as a whole. Store consistency is more important than optimizing the commitment shape per artifact.

Use Merkle roots for all store-backed artifacts, including:

- token ID sequences
- text or byte chunks
- activation rows
- attention head rows
- KV rows
- logits rows
- generated token IDs

The Merkle leaf payload must be canonical and must include enough structure to avoid ambiguity. The root domain must identify the artifact kind and version.

### Plain Hashes Only Outside The Store

Plain hashes are acceptable only for values that are not written to an authenticated store. Use them for legacy checkpoint compatibility or small atomic transcript commitments that are never later addressed through `auth_read`.

Examples:

- a small private scalar commitment
- a fixed-size metadata blob
- a compact final summary value
- a legacy `*_sha256` compatibility field

If a value is stored, reused, or later read by a tile, it should be a Merkle artifact instead.

### Public Inline For Small Controls

Do not hash or store small public control values unless there is a privacy or reuse reason. A tile should carry clear public values like `row_idx`, `layer_idx`, `width`, and `next_token_idx` directly.

## Canonical Encoding Requirements

Every artifact commitment must define:

- artifact kind
- versioned domain string
- identity/source name rules
- shape metadata
- leaf ordering
- leaf payload encoding
- empty artifact behavior
- final root encoding
- builder/running-commitment behavior

Numeric values used for deterministic parity should commit to deterministic representations, not host-dependent floating point formatting. For deterministic activations and logits, prefer `Act` bit encodings over f32 views.

## Initial Artifact Kinds

### Token ID Sequence

Used for prompt tokens and generated tokens.

Required metadata:

- source ID
- token count
- token ID root

Leaf payload:

```text
u32 token_id as little-endian bytes
```

### Text Or Byte Sequence

Used for prompt bytes, rendered prompt text, normalized prompt text, generated output text, and pending byte flushes.

Required metadata:

- source ID
- byte length
- optional char count for UTF-8 text
- chunk count or byte count
- root

Leaf payload should be bytes or text chunks with explicit length handling.

### Activation Sequence

Used for embedded prompts, PLE outputs, layer outputs, final hidden states, and logits when represented as rows.

Required metadata:

- tensor ID
- row count
- width
- tensor kind
- root

Leaf payload:

```text
row width as u64 little-endian
each Act bit value as i32 little-endian
```

### Attention Heads

Used for Q/K/V heads and attention outputs.

Required metadata:

- tensor ID
- head count
- sequence length
- head dimension
- tensor kind
- root

Leaf identity should be deterministic over `(head_idx, token_idx)`.

### KV Cache

Used for prefill and decode caches.

Required metadata:

- cache ID or key/value tensor IDs
- head count
- current length
- head dimension
- key root
- value root
- combined cache root

Leaf identity should include key/value kind, head index, and token index.

## Builder Contract

Dynamic artifact writes should be ordered, deterministic, and fail closed:

- A builder is initialized with artifact kind and expected shape.
- Each append validates index, width, and ordering.
- Each append updates a compact running commitment.
- Finalization verifies that all expected leaves were written.
- Finalization returns the artifact ref.
- Duplicate, skipped, wrong-width, out-of-range, or wrong-kind writes fail.

The first implementation can keep concrete rows in an in-memory store for native execution and tests. The API should still model the future proof contract: a tile updates a compact commitment and later tiles authenticate reads from the committed artifact.

## Store Contract

A store is a logical artifact environment, not the source of truth for correctness.

Native execution may implement the store as maps of concrete values. ZKVM replay may implement the same contract as public refs plus private witness reads and proofs.

The correctness boundary is:

```text
artifact ref + typed request + private payload + proof -> verified response
```

Every artifact written to the store must therefore have a Merkle root, even if the native in-memory implementation also keeps the concrete value for convenience.

Code should not depend on map iteration order, pointer identity, implicit caches, or ambient process state.

## CPU Parity Contract

The deterministic CPU path should eventually be able to compute the same artifact commitments from its existing checkpoint values.

This does not require CPU inference to execute through the raster store. It requires a commitment adapter that:

- receives deterministic CPU values at checkpoint or routine boundaries
- converts them to the same canonical artifact payloads
- computes the same roots as raster
- records both legacy checkpoint hashes and artifact commitments during migration

Legacy f32 checkpoint hashes can remain as compatibility fields until trace/checkpoint shape is intentionally versioned.

## Public And Private Boundary

For ZKVM replay, artifact refs and small control state are public. Artifact contents can be private witness data.

This means:

- A tile can publicly commit to reading row 7 of activation root `R`.
- The private witness supplies row 7 and its inclusion proof.
- The tile verifies the proof before using the row.
- The tile computes its output and commits to the next artifact or builder state.

This is the reason large materialized values should not be tile inputs. Public tile inputs should be compact commitments and metadata.

## Decisions

These decisions guide the first implementation slice:

- New non-store transcript commitments should use Merkle roots for consistency unless they are legacy compatibility fields.
- New artifact domains should use explicit versioned names such as `raster-artifact-token-ids-merkle-v1`, `raster-artifact-activation-sequence-merkle-v1`, and `raster-artifact-kv-cache-merkle-v1`. Existing `raster-*` domains should remain stable until a migration intentionally replaces them.
- Empty generated token streams should be represented as zero-count artifacts with the domain's typed empty Merkle root.
- Empty KV caches should remain explicit empty cache slots until they are written to the store. Once store-backed, they should use zero-length key/value roots plus shape metadata that makes the empty state unambiguous.
- Builders should use a Merkle frontier so append updates are efficient and do not require recomputing roots over all prior leaves.
- Existing trace fields that affect checkpoint commitments must remain byte-for-byte stable during the transition unless the checkpoint format is intentionally versioned.

## Remaining Design Defaults

These defaults should be used unless implementation reveals a better reason to change them:

- Legacy `*_sha256` fields stay as compatibility fields. New artifact commitments should use `*_root` or another explicitly Merkle-named field so callers can distinguish old hashes from artifact roots.
- Logits should be represented as one scalar row per token ID for both prefill and decode. This keeps argmax and partial authenticated reads uniform. A vector-row representation can remain only as a compatibility adapter if an existing public output expects it.

## First Implementation Slice

The first implementation should be intentionally small:

1. Centralize Merkle artifact commitment helpers for token IDs and activation sequences.
2. Add known-vector tests for roots and proofs.
3. Add deterministic CPU adapter tests that compute the same roots from existing materialized deterministic values.
4. Avoid changing routine behavior in this slice unless the change is only a compatibility wrapper.

After that, migrate routine handoffs in dataflow order:

```text
prompt token refs
  -> input embedding activation refs
  -> prefill prepare aux refs
  -> prefill layer refs
  -> prefill finalize refs
  -> decode refs
  -> generated token/text refs
```

## Current Migration Status

The first handoff migration is intentionally a hard routine-boundary refactor:

- Raster prompt prepare now exits as prompt token refs plus the tokenizer store needed to serve authenticated token reads.
- Raster inference is expected to terminate at the `prompt.prepare` boundary for this slice.
- Input embedding has not yet been refactored to consume `RasterTokenIdSequenceRef`; that is the next routine boundary.
- No materializing compatibility adapter should be added to make raster prompt prepare look like the old materialized `PromptPreparationState`.

## Non-Goals

- Do not make every scalar a store object.
- Do not replace all current stores in one change.
- Do not remove legacy checkpoint hashes in the first implementation.
- Do not make CPU inference run through raster stores just to prove parity.
- Do not introduce a broad abstraction that cannot be tested against one artifact kind first.
- Do not write plain-hash artifacts into authenticated stores.

## Summary

The contract is:

```text
large data plane: artifact refs + Merkle-authenticated reads and writes
small control plane: explicit public state
private witness plane: concrete values and inclusion proofs
compatibility plane: adapters that compute equivalent commitments from deterministic CPU values
```

This gives raster tiles compact public statements for ZKVM replay while preserving a path to deterministic CPU parity.
