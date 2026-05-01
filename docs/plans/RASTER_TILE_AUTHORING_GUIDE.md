# Raster Tile Authoring Guide

## Purpose

Raster tiles are replayable implementations of existing routine tiles. They should produce the same observable result as their corresponding native path while using an authoring shape that can later be executed and checked in a zkVM in isolation.

The prompt-preparation conversion in `src/prompt_prepare/raster_tiles.rs` is the current reference implementation. It mirrors `src/prompt_prepare/tiles.rs`, but replaces direct library calls and open-ended control flow with explicit tile boundaries, sequence composition, authenticated reads, and recursive state transitions.

---

## Core Contract

For every raster implementation:

- Match the native routine's external behavior first. Inputs, outputs, errors, commitments, and tie-breaking rules should stay aligned unless the native path depends on behavior that cannot be replayed deterministically.
- Keep all tile inputs and outputs explicit. A raster tile should be understandable from its arguments, return value, and authenticated reads.
- Treat sequences as orchestration. A `#[sequence]` function should compose tiles with `call_tile!`, `call_seq!`, and recursive call helpers; it should not hide substantial routine logic inline.
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

For prompt preparation, `tiles.rs` defined the baseline: decode bytes, build Gemma messages, render the chat template, tokenize the rendered prompt, and hash the prompt token IDs. The raster path keeps that same end-to-end result but expands tokenization into replayable steps.

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

This style makes the future proof trace obvious: each line names a tile, passes explicit state, and either returns a final value or hands state to the next step.

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

Invoke fallible recursive tiles with `call_recur_tile_result!`:

```rust
let state = call_recur_tile_result!(step, state, source)?;
```

Prompt preparation uses this for BPE merging. The native tokenizer repeatedly chooses the best merge candidate until no merge remains; the raster version makes that one replayable merge step over `GemmaBpeState`.

### 5. Replace File And Table Access With Authenticated Reads

Raster tiles should not open files, seek into assets, or call opaque library lookups directly. Instead:

1. define an authenticated source type
2. define small request types for each lookup shape
3. implement `AuthRead<Request>` for the source
4. call it inside tiles with `auth_read!(source, Request { ... })`

The prompt-preparation tokenizer is the reference pattern:

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

For prompt preparation, `TokenizePromptInput`, `GemmaNormalizedText`, `GemmaPreTokenizedText`, `GemmaBpeState`, and `GemmaBpeOutput` make each tokenizer phase explicit and replayable.

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
- Call recursive tiles with `call_recur_tile!` or `call_recur_tile_result!`.
- Keep sequence bodies mostly linear. Branching is acceptable when it reflects routine semantics, but large branches should usually become separate tiles or sequences.
- Do not use a sequence as a place to smuggle dynamic loops around the recursive authoring model.

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
- Only modify existing shared code when changing a truly shared contract. If the implementation differs because of raster authoring, authenticated reads, recursive state, or zkVM replay constraints, create raster-specific shared code instead.

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

For prompt preparation, `tests/gemma_tokenizer_parity.rs` compares raster tokenization with the Hugging Face tokenizer for the supported tokenizer subset. Future routine conversions should use the same style: choose small fixtures, run native and raster paths, and assert identical public outputs and commitments.

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

## Prompt-Preparation Reference Map

The current prompt-preparation raster path demonstrates the intended pattern:

- simple native helpers like byte decoding, message construction, template rendering, and commitment hashing become direct `#[tile]` functions
- native tokenization is decomposed into initialization, normalization, splitting, BPE initialization, recursive BPE merging, and final token-ID lookup
- tokenizer metadata and lookup tables are accessed through `auth_read!`
- BPE's dynamic merge loop is represented as a tail-recursive tile over `GemmaBpeState`
- the top-level `run` sequence composes the routine into the same `PromptPreparationState` returned by the native path

Future raster routines should follow the same discipline: start from the native routine's result, expose every external dependency as an authenticated source, turn data-dependent loops into recursive state machines, and keep the sequence as a readable proof trace.
