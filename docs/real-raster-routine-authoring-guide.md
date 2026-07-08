# Real Raster Routine Authoring Guide

Use this guide before planning or implementing a remaining routine on the real
`raster` toolchain. It captures the authoring model established by the new
`prompt_prepare` routine.

The simulator remains the logical specification. The real routine is not a
line-by-line port of the simulator. It is a re-expression of the same routine
under the real raster sequence, tile, storage, and trace constraints.

## The Mental Model

The simulated routines are artifact-store programs:

```text
state carries artifact roots and scalar cursors
tiles read leaves by root and index
tiles append leaves through builders
finalizers produce new artifact roots
```

Real raster routines should be storage-read programs:

```text
state carries raster refs and scalar cursors
tiles receive refs, selectors, ordinals, and small structs
tiles read storage inside the tile body
drafts build append-only outputs
finalizers carry InternalRef values forward
```

The goal is the same in both worlds: collections live in committed storage, and
state points at them. The difference is that real raster uses committed
externals, internal storage refs, selectors, and drafts rather than simulator
artifact-store helpers.

## Why Prompt Prepare Changed Shape

The old real `prompt_prepare` routine still let large values cross tile
boundaries:

- tokenizer chunks as `Vec<GemmaBpeMerge>` or `Vec<GemmaTokenIdEntry>`
- BPE pieces as `BpePieces` or `String`
- adjacent pairs as `GemmaBpeAdjacentPairs`

Those parameters are serialized into `FnInput.data`, so they made
`trace.ndjson` enormous even when the logical input was selected from storage.

The new `prompt_prepare` shape avoids that. Nonterminal tiles receive small
values only:

- `TokenizerTables` source descriptors
- `InternalRef` handles
- `u32` ordinals and counts
- scalar state and decisions

Tiles then perform authenticated reads inside the tile body using existing
raster storage APIs.

## Core Authoring Rules

### 1. Sequences Orchestrate, Tiles Compute

Sequence bodies must stay straight-line and declarative. They compose calls:

- `call!`
- `call_seq!`
- `call_recur!`
- `call_recur_seq!`
- `select!`

Do not hide computation in sequence bodies. Anything that inspects or transforms
data belongs in a tile.

### 2. Large Data Must Not Be A Tile Parameter

Do not put model-sized or prompt-sized collections in nonterminal tile
signatures.

Avoid signatures like:

```rust
chunk: Vec<GemmaBpeMerge>
chunk: Vec<GemmaTokenIdEntry>
pieces: BpePieces
piece: String
pairs: GemmaBpeAdjacentPairs
```

Prefer signatures like:

```rust
chunk_idx: u32
piece_idx: u32
tokenizer: TokenizerTables
opened: GemmaBpeOpenedRound
pieces_ref: InternalRef
```

The tile can then read the needed value from raster storage inside the body.

### 3. External Model Data Stays Behind Source Descriptors

Model-scoped tables should remain behind committed externals. In
`prompt_prepare`, the tokenizer is represented by `TokenizerTables`, a compact
descriptor that points at the committed `tokenizer` external in production and
at an internal test fixture in native tests.

Tiles read selected tokenizer chunks by constructing selectors at the point of
use.

Example pattern from `prompt_prepare`:

```text
scan_one_merge_chunk(chunk_idx, tokenizer, opened)
  -> read tokenizer.merge_chunks[chunk_idx]
  -> read BPE pieces by index
  -> update scalar scan state
```

### 4. Prompt Data Stays Behind Internal Refs

Prompt-derived collections should not ride through recur state or tile
parameters. Carry `InternalRef` plus small metadata instead.

`prompt_prepare` uses:

```text
GemmaBpeLoopState
  complete: bool
  round: u32
  piece_count: u32
  pieces_ref: InternalRef
  error: Option<String>
```

This mirrors the simulator's root-carrying state without copying the simulator
artifact-store machinery.

### 5. Loops Are Driven By Ordinals

Use explicit bounded lists of scalar ordinals as recur inputs:

- round ordinals
- merge chunk ordinals
- piece ordinals
- vocab chunk ordinals
- layer ordinals for layer loops

The ordinal is cheap to trace. The tile uses it to read the actual data.

### 6. Choose Recur Tile vs Recur Sequence Deliberately

Use `#[tile(kind = recur)]` with `call_recur!` when:

- the loop needs early `Break`
- every input, state, and arg is scalar/ref-sized
- no large payload will be traced inline per iteration

Use `#[sequence(kind = recur)]` with `call_recur_seq!` when:

- the body needs to orchestrate multiple tile calls
- the loop uses a `RecurOutput` draft
- running all items is acceptable

In `prompt_prepare`, the merge scan is a recur tile because it can break after
the winning merge is found and its inputs are now only refs and ordinals. The
apply loop remains a recur sequence because it threads a draft output.

### 7. Drafts Replace Simulator Builders

Simulator builders map to real raster drafts.

Use `RecurOutput<S>` / `Draft<S>` for append-only collections:

- next-round BPE pieces use `RecurOutput<BpePieces>`
- token-id matches use `RecurOutput<TokenIdMatches>`

Do not accumulate these collections in recur state.

### 8. Materialize Only At Boundaries

Materialization is acceptable at clear boundaries:

- staging initial committed inputs
- run-local raster-encoding of large request-scoped inputs that will be
  selected repeatedly by tiles
- publishing an initial tile output to get an internal ref
- validating a finalized draft
- producing the final host-visible output

Materialization is not acceptable as routine glue between every tile.

### 9. Trace Size Is A Correctness Signal

A real-raster routine can be behaviorally correct and still authored
incorrectly if the trace contains repeated large payloads.

After authoring a routine, inspect trace shape:

- Nonterminal tile `FnInput.data` should be scalar/ref-sized.
- Large model chunks should not appear in tile inputs.
- Large request-scoped collections should not use postcard external selection
  when tiles repeatedly read small parts of them; stage them as raster-encoded
  externals or internal refs so reads use indexed storage selection.
- Prompt strings/pieces should not appear in recur state or nonterminal tile
  inputs.
- Recur tile iterations are allowed only when their input/state/args are
  scalar/ref-sized.

For `prompt_prepare`, the real Gemma prompt trace dropped from over 1 GB to
about 16 MB after the storage-read rewrite and scalar recur scan.

## Prompt Prepare As The Reference

Use these files as the working example:

- `crates/raster-programs/prompt_prepare/src/main.rs`
- `crates/raster-programs/prompt_prepare/src/routine.rs`
- `crates/raster-programs/prompt_prepare/src/types.rs`
- `crates/raster-programs/prompt_prepare/src/bpe_round.rs`
- `crates/raster-programs/prompt_prepare/src/bpe_scan.rs`
- `crates/raster-programs/prompt_prepare/src/bpe_apply.rs`
- `crates/raster-programs/prompt_prepare/src/token_ids.rs`
- `crates/raster-programs/prompt_prepare/src/guards.rs`

The most important patterns to copy are:

- compact source descriptors instead of selected model-table values
- `InternalRef`-carried prompt collections
- raster-encoded externals for large repeatedly selected request/model
  collections
- ordinal-driven loops
- in-tile storage reads
- drafts for append-only outputs
- trace-shape guards that fail on eager large tile params

## Planning Checklist For The Next Routine

Before writing an implementation plan for another routine, answer these:

1. What are the simulator roots/builders/cursors?
2. Which collections are model-scoped, prompt/request-scoped, or output-scoped?
3. Which values are too large to appear in tile parameters?
4. What compact source descriptors, raster-encoded externals, and refs will
   replace those values?
5. What ordinal lists bound every recur loop?
6. Which loops need `Break`, and can they be scalar/ref-sized recur tiles?
7. Which append-only outputs should be drafts?
8. Where is final materialization actually required?
9. What trace-shape guard would catch accidental eager materialization?
10. What end-to-end trace size should be expected for a small real-model run?

## Common Anti-Patterns

Avoid these, even if they are easy to write:

- Selecting an entire model table into a sequence arg.
- Passing `Vec<T>` chunks directly to tiles when a ref plus index would do.
- Passing prompt pieces as `String` recur inputs.
- Building derived prompt collections just to pass them into the next tile.
- Threading collections through recur state.
- Using a recur tile while its args contain large selected values.
- Reading large repeatedly selected request collections through postcard
  externals one scalar at a time.
- Treating test-size trace success as proof that real-model trace shape is
  acceptable.

## Acceptance Criteria

A routine follows this guide when:

- behavior matches the simulator/native oracle;
- real-raster program tests pass;
- `--no-default-features` still compiles;
- detour parity passes;
- nonterminal tile signatures are small and serde-safe;
- large repeatedly selected externals are raster-encoded or otherwise indexed;
- trace guards reject large eager params and prompt/model payloads in
  `FnInput.data`;
- a real-model smoke trace is within the expected size budget for the routine.

