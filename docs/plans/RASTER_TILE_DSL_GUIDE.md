# Raster Tile Authoring Guide

## Purpose

Raster tiles are replayable implementations of existing routine tiles. They should produce the same observable result as their corresponding native path while using an DSL shape that can later be executed and checked in a zkVM in isolation.

The completed prompt-preparation conversion in `src/routines/prompt_prepare/raster/tiles.rs` is the illustrative example for this guide. It mirrors `src/routines/prompt_prepare/native/tiles.rs`, but replaces direct library calls and open-ended control flow with explicit tile boundaries, sequence composition, authenticated reads, and recursive state transitions.

---

## Core Contract

For every raster implementation:

- Match the native routine's external behavior first. Inputs, outputs, errors, commitments, and tie-breaking rules should stay aligned unless the native path depends on behavior that cannot be replayed deterministically.
- Keep all tile inputs and outputs explicit. A raster tile should be understandable from its arguments, return value, and authenticated reads.
- Treat sequences as strict orchestration. A `#[sequence]` function should be a straight-line composition of `call_tile!`, `call_seq!`, and recursive call helpers; it must not branch, loop, inspect `Option`/enum cases, or hide routine logic inline.
- Treat tiles as the replay units. A `#[tile]` function should perform one meaningful deterministic step with a narrow input and output contract.
- Use recursive tiles or recursive sequences for dynamic repetition. Do not put unbounded routine-level loops in a sequence when the number of iterations depends on data.
- Route external data access through `auth_read!`. Files, model weights, tokenizer tables, and other large or external sources must be represented as authenticated sources plus typed request objects.
- Use deterministic arithmetic for any arithmetic that can diverge between native execution and zkVM execution. Raster tiles must not use native floating-point operations or floating-point library methods for replay-critical arithmetic.
- Keep raster-specific shared code separate. If raster routines need shared helpers, sources, kernels, or state types, put them in dedicated `shared` files or modules such as `raster_transformer.rs`, `raster_transformer_kernels.rs`, or `shared/raster/...` rather than adding mode-conditionals to existing shared code.
- Preserve deterministic state. Intermediate state that crosses tile boundaries should use owned, serializable, deterministic data structures rather than borrowed views into external libraries.
- Fail closed when a feature is outside the supported replayable subset. It is better for a raster path to reject unsupported tokenizer/model behavior than to approximate it silently.

---

## Conversion Recipe

### 1. Start From The Native Routine

Use the existing native routine as the behavioral specification. Before changing structure, identify:

- the public routine entry point
- the final output type
- all intermediate helper functions
- all reads from files, tokenizers, model weights, caches, or global configuration
- loops whose trip count depends on prompt length, token count, layer count, vocab size, or model shape
- library calls whose semantics must be reproduced or narrowed
- arithmetic that already uses deterministic numeric helpers in the native deterministic path

Prompt preparation illustrates the pattern: `tiles.rs` defined the baseline of decoding bytes, building Gemma messages, rendering the chat template, tokenizing the rendered prompt, and hashing the prompt token IDs. The completed raster path keeps that same end-to-end result but expands tokenization into replayable steps.

When the native routine already has a deterministic arithmetic path, treat that path as the raster arithmetic specification. Do not fall back to the native `f32` implementation inside a raster tile just because it is simpler to call.

### 2. Choose Tile Boundaries Around Semantic Steps

Prefer tile boundaries that correspond to named, testable transformations:

- decode input bytes
- construct prompt messages
- render a template
- normalize tokenizer input
- split pre-tokenized text
- initialize BPE state
- apply one BPE merge step
- finalize token IDs
- build commitments
- construct the routine's final state

Avoid creating tiles that are only mechanical wrappers around individual lines unless the boundary is useful for authentication, recursion, or isolation.

### 3. Make The Sequence A Straight-Line Plan

A sequence should read like a small execution trace:

```rust
#[sequence]
pub fn run(...) -> Result<Output> {
    let a = call_tile!(first_step, ...)?;
    let b = call_tile!(second_step, a, ...)?;
    let c = call_seq!(nested_sequence, b, ...)?;
    call_tile!(finalize, c)
}
```

This style makes the future proof trace obvious: each line names a tile, passes explicit state, and either returns a final value or hands state to the next step. Keep all decisions inside tiles by returning explicit state variants such as "complete" or "continue"; the sequence should not inspect those variants itself.

### 4. Represent Dynamic Loops As Tail-Recursive State Machines

When a native implementation loops until data-dependent completion, split it into:

1. an initialization tile that builds explicit state
2. a `#[tile(kind = recursive)]` or `#[sequence(kind = recursive)]` step that performs one iteration
3. a finalization tile that converts state into the native output shape

Recursive tiles return a done flag plus the next state:

```rust
#[tile(kind = recursive)]
pub fn step(mut state: State, source: &AuthenticatedSource) -> Result<(bool, State)> {
    if done(&state) {
        return Ok((true, state));
    }

    state = advance_one_iteration(state, source)?;
    Ok((false, state))
}
```

Invoke fallible recursive tiles with `call_recur_tile!`:

```rust
let state = call_recur_tile!(step, state, source)?;
```

The completed prompt-preparation path uses this for BPE merging. The native tokenizer repeatedly chooses the best merge candidate until no merge remains; the raster version makes one logical merge iteration a recursive sequence over compact refs, cursors, and bounded scan/apply states. The no-merge branch is represented as tile state, so the sequence remains a straight-line chain of tile calls.

### 5. Replace File And Table Access With Authenticated Reads

Raster tiles should not open files, seek into assets, or call opaque library lookups directly. Instead:

1. define an authenticated source type
2. define small request types for each lookup shape
3. implement `AuthRead<Request>` for the source
4. call it inside tiles with `auth_read!(source, Request { ... })`

The prompt-preparation tokenizer illustrates the authenticated-read pattern:

- `AuthenticatedGemmaTokenizer` wraps the parsed tokenizer spec.
- `GemmaTokenizerMetadataRequest` reads normalization and split metadata.
- `GemmaTokenIdRequest` reads vocab IDs.
- `GemmaSpecialTokenAtRequest` checks for the longest special token at a byte offset.
- `GemmaBpeMergeRequest` reads a merge candidate by pair.
- `GemmaBpeMergedTokenRequest` resolves a merge index to the merged token.

Use request types that are as narrow as possible. A tile that only needs one weight row, token ID, layer scalar, or merge candidate should request only that value, not the entire source object.

### 6. Make State Owned And Replayable

Any value passed between tiles should be safe to serialize and replay:

- use owned `Vec`, `String`, structs, enums, and deterministic numeric types
- derive `Debug`, `Clone`, `Serialize`, `Deserialize`, `PartialEq`, and `Eq` when practical
- avoid borrowed references in persistent state structs
- avoid hidden handles into external libraries
- avoid relying on map iteration order unless the map is only used behind authenticated read methods with deterministic request semantics

In the completed prompt-preparation path, `TokenizePromptInput`, `GemmaNormalizedText`, `GemmaPreTokenizedText`, ref-backed `GemmaBpeState`, bounded BPE scan/apply states, and token-ID finalization state make each tokenizer phase explicit and replayable without serializing the scratch piece store through recursive state.

---

## Authoring Rules

### Tiles

- Mark replay units with `#[tile]`.
- Mark recursive replay units with `#[tile(kind = recursive)]`.
- Keep inputs and outputs explicit.
- Return `Result<T>` when the native behavior can fail.
- Use local helper functions only for pure deterministic substeps. If a helper hides meaningful state transitions or external reads, promote it to a tile.
- Do not perform filesystem reads, environment reads, network calls, random generation, clock reads, thread spawning, or global mutation inside a tile.

### Sequences

- Mark orchestration units with `#[sequence]`.
- Call tiles with `call_tile!`.
- Call nested sequences with `call_seq!`.
- Call recursive tiles with `call_recur_tile!`.
- Keep sequence bodies linear. Branching is not acceptable in sequences; all conditional behavior must live inside tiles or in recursive tile/sequence done flags.
- Do not use a sequence as a place to smuggle dynamic loops around the recursive DSL model.

### Authenticated Sources

- Use `AuthRead<Request>` for all external or large data reads.
- Keep request and response types deterministic and narrow.
- Include enough identity in the source to commit to the backing data, such as a tokenizer SHA-256 or model artifact identity.
- Validate source data before use where possible, then fail closed on unsupported shapes.
- Prefer many precise request types over one broad "give me everything" request.

### Dynamic Iteration

- Use recursion when iteration count depends on input data or model contents.
- Put all loop-carried values into an explicit state struct.
- Each recursive step should either return `(true, state)` unchanged or advance by exactly one logical iteration.
- Make progress obvious. If the state can fail to progress, add a guard or return an error.
- Keep context inputs separate from loop-carried state. For example, pass an authenticated source as context while the mutable state remains the recursive state value.

### Deterministic Arithmetic

- Use the deterministic numeric types and helpers defined by `DET_NUM_SPEC.md` for replay-critical arithmetic.
- Never use native floating-point operations, transcendental methods, comparison shortcuts, or reductions inside a raster tile when the result can differ across host CPU, compiler settings, or zkVM execution.
- If the corresponding normal routine already has a deterministic path, use the same deterministic arithmetic semantics in the raster tile.
- Keep narrowing, rounding, saturation, comparison, and argmax behavior routed through deterministic helpers rather than raw casts or ad hoc math.
- Fail closed if a needed arithmetic operation does not yet have a deterministic counterpart.

### Shared Raster Code

- Put reusable raster implementation code under `src/shared/`, but keep it physically separate from existing native or deterministic shared modules.
- Prefer names that make the raster boundary obvious, such as `raster_transformer.rs`, `raster_transformer_kernels.rs`, or a `shared/raster/` module tree.
- Do not turn existing shared modules into large `match execution_mode` or `if raster_tiles` blocks to support raster behavior.
- Only modify existing shared code when changing a truly shared contract. If the implementation differs because of DSL, authenticated reads, recursive state, or zkVM replay constraints, create raster-specific shared code instead.

---

## zkVM Isolation Principles

Design each raster tile as though it will be replayed with only:

- its function identity
- serialized arguments
- authenticated source references
- typed authenticated read responses
- its serialized return value

This implies:

- no ambient filesystem assumptions
- no dependence on host process state
- no dependence on CPU-specific floating-point behavior in deterministic paths
- no native floating-point arithmetic in raster tiles for replay-critical math
- no implicit caches whose contents are not part of the tile input or authenticated source
- no pointer identity, address ordering, or nondeterministic collection iteration in observable behavior

The current Rust macros are intentionally lightweight, but raster code should be written to the stricter future contract now.

---

## Testing Expectations

Every raster conversion should have focused tests at three levels:

- Tile tests for important local behavior and edge cases.
- Sequence tests for multi-step state flow, especially recursive completion.
- Parity tests comparing raster output with the native routine for representative inputs.

Prompt preparation illustrates this with `tests/gemma_tokenizer_parity.rs`, which compares raster tokenization with the Hugging Face tokenizer for the supported tokenizer subset. Future routine conversions should use the same style: choose small fixtures, run native and raster paths, and assert identical public outputs and commitments.

When a raster path intentionally supports only a deterministic subset, add tests that reject unsupported inputs with clear errors.

---

## Common Pitfalls

- Calling a library method inside a tile when the method reads hidden tables or carries hidden mutable state.
- Using `f32`/`f64` arithmetic inside a raster tile where deterministic arithmetic is required for zkVM parity.
- Keeping a data-dependent loop inside a sequence instead of moving it to a recursive tile.
- Passing an entire model, tokenizer, or file blob through tile state when the tile only needs a narrow lookup.
- Mixing raster-specific shared helpers into existing shared modules behind mode flags instead of creating dedicated raster shared files or modules.
- Depending on `HashMap` iteration order, filesystem ordering, current time, random seeds, or thread scheduling.
- Preserving native implementation structure when the native structure is not replayable. Preserve behavior, not incidental call shape.
- Adding compatibility shims for unsupported in-progress behavior. If the raster path is new and not shipped, make the contract strict and clear.

---

## Prompt Preparation As An Example

Prompt preparation is already converted. This section uses it as a concrete example for how to reason about future routine conversions, not as a record of pending work.

The native CPU path in `src/prompt_prepare/mod.rs` is intentionally simple:

1. Decode prompt bytes into prompt text.
2. Wrap the prompt text as a single Gemma user message.
3. Render the model chat template.
4. Tokenize the rendered prompt with `tokenizers::Tokenizer::encode_fast`.
5. Build the prompt-token commitment.

The completed raster path in `src/routines/prompt_prepare/raster/tiles.rs` preserves that semantic pipeline, but changes the execution and state shape. Instead of carrying raw values through an in-memory library call, it commits each meaningful large value as an artifact root and decomposes tokenization into bounded replayable work.

The raster prompt-preparation checkpoint carries:

- `prompt_bytes_root`
- `prompt_text_root`
- `rendered_prompt_root`
- `normalized_prompt_root`
- `prompt_token_ids_root`
- `prompt_token_count`

This is the shape future routines should aim for: compact public roots and counts at the boundary, with full data available only through authenticated reads.

### Example Conversion Pattern

Prompt preparation demonstrates these concrete moves that future routines should adapt:

- Simple deterministic helpers, such as byte decoding, message construction, template rendering, and final state construction, remain small direct tiles or pre-sequence preparation steps.
- Native tokenization is expanded into explicit stages: initialize tokenization input, normalize text, split text, initialize BPE pieces, recursively merge BPE pieces, finalize BPE output, and map final pieces to token IDs.
- Tokenizer metadata, vocab lookups, special-token lookups, and BPE merge lookups are read through typed authenticated requests rather than through an opaque tokenizer object inside a tile.
- BPE's data-dependent merge loop becomes a recursive sequence over `GemmaBpeState`, with the stop/apply decision represented by tile state rather than inline sequence branching.
- Each logical BPE merge iteration is split into bounded scan and apply phases.
- The scan phase reads adjacent piece pairs from a committed BPE piece artifact and chooses the best merge candidate by rank.
- The apply phase streams the old piece artifact into a new builder artifact, replacing exactly the selected pair with the merged piece.
- Token-ID finalization streams final BPE pieces into a committed token-ID artifact in bounded chunks.
- The top-level raster sequence returns the same routine boundary information as deterministic CPU checkpointing, but in root-backed form.

The most important structural lesson is that the raster path preserves behavior, not incidental native implementation shape. The CPU implementation can call a tokenizer library once because the host process can hold the whole tokenizer and whole prompt in memory. The raster implementation must expose the tokenizer tables as committed sources, expose dynamic prompt state as committed artifacts, and make each loop-carried transition replayable in isolation.

### Example Authenticated Input And Output Pattern

The prompt-preparation example uses two kinds of committed data:

- Static committed sources, such as the authenticated Gemma tokenizer source.
- Dynamic committed artifacts, such as prompt bytes, prompt text, normalized text, BPE pieces, and token IDs.

Future routines should make the same distinction. Model weights, tokenizer tables, and static metadata should be read through narrow authenticated source requests. Prompt-dependent or decode-dependent intermediates should be written as dynamic artifacts with explicit kind, domain, shape or length metadata, and canonical leaf encoding.

Every tile that reads from an artifact should verify the read before using the payload. Every tile that writes many values should append leaves incrementally through a builder and finalize only after the expected cursor or count has been reached.

### Example Bounded Work Pattern

The prompt-preparation example exposes two independent tile sizing controls:

- `bpe_pairs_per_tile` bounds how many adjacent BPE pairs one scan tile checks.
- `bpe_pieces_per_tile` bounds how many BPE pieces one apply or token-ID finalization tile processes.

The chunk sizes affect the number of tile invocations, not the result. Tests assert that tiny chunks and larger chunks produce the same token IDs. Future routines should follow the same rule: row, token, head, byte, or vocab chunk sizes may change proof granularity, but must not change commitments or public outputs.

### Native/Raster Equivalence Pattern

The deterministic CPU checkpoint is formatted into the same root-backed checkpoint shape as the raster path. Tests then compare that checkpoint against `prompt_prepare::run_raster`.

Use this pattern for every routine:

1. Keep a deterministic native path as the behavioral baseline.
2. Make the raster path produce the same public routine boundary.
3. When native deterministic execution reaches that boundary, format its state into the same raster checkpoint shape.
4. Add parity tests that compare native-formatted checkpoint state with raster-produced checkpoint state.
5. Add chunk-size invariance tests for every configurable tile bound.

### Routine Conversion Checklist

Use this checklist before converting another routine:

- Identify the native routine boundary, final state type, checkpoint payload, and commitment fields.
- List all large inputs, large outputs, static sources, and dynamic intermediates.
- Decide which values stay inline as small public controls and which become committed artifacts.
- Define artifact kind, domain, shape metadata, leaf ordering, and leaf payload encoding for every dynamic artifact.
- Define authenticated source request types for every static lookup, keeping requests as narrow as possible.
- Replace data-dependent loops with recursive tiles or recursive sequences that carry explicit cursors and compact refs.
- Split global-choice mutations into separate scan and apply phases when needed.
- Validate tile sizing controls and reject zero or unsupported settings.
- Keep sequence bodies linear and branch-free so the proof trace is readable.
- Ensure each recursive step makes obvious progress or returns a clear error.
- Add tests for local tile behavior, recursive completion, native/raster parity, and chunk-size invariance.
- Fail closed when native behavior depends on unsupported tokenizer, model, arithmetic, or layout features.

Future raster routines should follow the same discipline: start from the native deterministic result, expose every external dependency as an authenticated source, turn dynamic data into committed artifacts, turn data-dependent loops into recursive state machines, and keep the sequence as a readable proof trace.
