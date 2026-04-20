# Raster Inference Porting Guide

This note explains how to build `raster-inference` in the minimal Raster-like style we want, while still using `mistral.rs` as a semantic reference where useful.

The immediate goal is not to copy `mistral.rs` exactly. The goal is to make inference work correctly for our supported path, while structuring the code so it already looks like a future Raster program:

- tiles are pure, side-effect-free functions
- sequences are explicit compositions of tiles and other sequences
- non-deterministic or host-only behavior stays outside the program shape
- we port semantics, not framework machinery

For now, this repo is intentionally narrow:

- text-only Gemma 4
- one Rust crate
- no Raster core dependency yet
- no scheduler/server/runtime abstractions
- only the pieces needed to reach inference-result parity

## Core principles

### 1. Port semantics, not architecture

When reading `mistral.rs`, do not try to reproduce its engine structure, request router, scheduler, cache manager, streaming layer, or general model framework.

Instead, extract only the semantic steps we actually need:

- how prompt bytes become the rendered prompt string
- how the rendered prompt becomes token IDs
- how token IDs become hidden states / logits / next tokens
- how decode state evolves

Everything else is guilty until proven necessary.

### 2. Keep the program model-specific and narrow

The current target is Gemma 4 text inference. If a user provides the wrong model, wrong tokenizer, or wrong template, the program may fail.

That is acceptable for now.

We do not need a generic capability matrix or a generalized validation layer just to preserve cleanliness. The supported path should be obvious from the code itself.

### 3. Treat templating as part of tokenization correctness

The user-facing input boundary is raw prompt bytes, not pre-tokenized IDs.

For chat models, accurate tokenization means:

1. decode prompt bytes into text
2. build the model-specific message structure
3. apply the model-specific chat template
4. tokenize the rendered prompt

Templating is therefore part of the tokenization seam, not an optional wrapper.

### 4. Remove ambient context

We do not want prompt rendering to depend on wall-clock time or other hidden state.

That means:

- no date stamp injection
- no ambient environment lookups
- no hidden filesystem/network access from tiles
- no randomness in decode logic

If some future model feature depends on dynamic context, that context must become an explicit input.

### 5. Sequences should be boring

Sequences should contain only explicit calls and data passing. No hidden helper logic, no ad hoc branching, no inline computation except wiring values forward.

Good sequence shape:

```rust
let decoded = decode_prompt_bytes(prompt_bytes);
let messages = build_gemma4_messages(decoded);
let rendered = apply_gemma4_template(messages, template_assets);
let tokens = tokenize_rendered_prompt(rendered, tokenizer_assets);
let truncated = truncate_prompt_tokens(tokens, limits, decode_config);
truncated
```

If a sequence wants to do real work, that work should become a tile.

### 6. Edge adapters are allowed

Loading assets from disk, CLI argument parsing, and other host conveniences should live in thin edge helpers outside the Raster-like program shape.

Inside the program shape, everything should look like pure computation over explicit inputs.

## The translation method

Whenever we tackle a new part of inference, use this method.

### Step 1. Find the seam in `mistral.rs`

Identify the smallest part of `mistral.rs` that owns the semantics we care about.

Examples:

- prompt preparation before tokenization
- text-model prefill
- one decode step
- stop condition handling
- detokenization

Ignore neighboring framework code unless it changes semantic behavior.

### Step 2. Separate semantic behavior from runtime machinery

For the chosen seam, classify each piece of behavior into one of two buckets.

Keep:

- anything that changes the produced prompt, tokens, logits, next token, or final text
- anything required for deterministic correctness
- model-specific formatting or tensor transforms

Discard or defer:

- scheduling
- batching
- streaming
- request validation that is not required for correctness
- generic abstractions for other model families
- telemetry/logging
- cache-management policies that are not part of the semantic boundary we are implementing

### Step 3. Write down the true input/output contract

Before writing code, define:

- the explicit inputs
- the explicit outputs
- what state is threaded through
- what part of the result must match the reference implementation

For this repo, prefer simple typed Rust structs first. The code should still be easy to map onto future byte-ABI tiles later.

### Step 4. Choose the tile boundary

Split the seam into a small set of tiles that each do one deterministic job.

A good tile:

- has one clear purpose
- takes only explicit inputs
- returns a concrete value or state struct
- is small enough to test directly

A bad tile:

- bundles unrelated concerns
- hides ambient state
- only exists to mimic a framework layer from `mistral.rs`

### Step 5. Keep the first version coarse enough to ship

We do not need the perfect final granularity immediately.

Prefer the smallest tile graph that still feels Raster-like and testable. If a future verifier/proof use case wants finer step boundaries, we can split tiles later.

Examples:

- prompt preparation can be several small tiles now
- a first transformer prefill pass can stay relatively coarse
- a decode loop may begin as one recursive step tile rather than dozens of per-op tiles

### Step 6. Test against reference behavior

Every new seam should have a small golden test harness against known-good behavior from `mistral.rs` or the exact tokenizer/template/model artifacts it uses.

Parity should be checked at the smallest meaningful boundary:

- rendered prompt string
- prompt token IDs
- prefill logits or selected hidden-state checkpoints
- next token choice
- final generated token sequence

Do not wait for end-to-end parity before validating a seam.

## First batch: prompt-preparation tiles

The first batch of work should focus on the prompt-preparation path for Gemma 4 text inference.

This is the right starting point because:

- it is the first semantic boundary in the pipeline
- it directly affects every downstream result
- it can be tested without implementing transformer execution
- it already maps naturally to pure tiles

### Goal

Given prompt bytes plus Gemma 4 tokenizer/template assets, produce the exact prompt token IDs that the Gemma 4 forward pass should consume.

### What we do not need here

- generic model validation
- prompt validation beyond what prevents nonsense failures
- date-based template context
- `mistral.rs` request-routing behavior
- multimodal handling
- streaming concerns

### Recommended tile batch

#### `decode_prompt_bytes`

Purpose:

- convert external prompt bytes into the text representation used by the program

Inputs:

- raw prompt bytes
- explicit text-decoding policy if needed

Outputs:

- prompt text

Notes:

- keep the decoding rule explicit and deterministic
- do not trim or normalize text unless the model path truly requires it

#### `build_gemma4_messages`

Purpose:

- construct the minimal message representation expected by the Gemma 4 template

Inputs:

- prompt text
- any explicit request options we support, such as `add_generation_prompt`

Outputs:

- model-specific message list / prompt-render context

Notes:

- keep this narrowly scoped to the supported Gemma 4 text path
- do not generalize for other model families yet

#### `apply_gemma4_template`

Purpose:

- render the exact chat prompt string the tokenizer should consume

Inputs:

- Gemma 4 messages
- template asset contents
- explicit template tokens such as BOS/EOS/UNK when needed
- explicit thinking flags only if we truly support them

Outputs:

- rendered prompt string

Notes:

- no date injection
- no hidden runtime values
- this tile is part of tokenization correctness

#### `tokenize_rendered_prompt`

Purpose:

- tokenize the rendered prompt string with the provided tokenizer assets

Inputs:

- rendered prompt string
- tokenizer
- `add_special_tokens` policy

Outputs:

- prompt token IDs

#### `truncate_prompt_tokens`

Purpose:

- enforce model context-window limits if needed

Inputs:

- prompt token IDs
- model limits
- decode config

Outputs:

- final prompt token IDs

Notes:

- this should only include the truncation policy we truly need for inference correctness
- keep it explicit rather than hiding it inside tokenization

#### `build_input_embedding_debug_commitments`

Purpose:

- optionally compute hashes that help us compare behavior while developing

Inputs:

- rendered prompt
- prompt token IDs

Outputs:

- debug hashes / commitments

Notes:

- these are for debugging and parity checks, not for pretending to be the final proof surface

### Recommended first sequence

The first explicit composition should look roughly like this:

- `seq_input_embedding_gemma4`

Responsibilities:

1. call `decode_prompt_bytes`
2. call `build_gemma4_messages`
3. call `apply_gemma4_template`
4. call `tokenize_rendered_prompt`
5. call `truncate_prompt_tokens`
6. optionally call `build_input_embedding_debug_commitments`
7. return the phase-1 state

The sequence should do no real logic beyond call ordering and value threading.

## How to approach future phases

The rest of inference should be approached the same way: one semantic seam at a time.

### Phase 2: transformer state transition

Primary goal:

- turn prompt token IDs or decode state into model hidden-state/logit results

Do not start by porting the whole model runner.

Instead, choose one narrow checkpoint such as:

- prompt prefill for one full input sequence
- one decode step with existing state
- one layer block over known activations

Likely early tile candidates:

- `embed_input_tokens`
- `run_prefill_pass`
- `extract_prefill_logits`
- `decode_step`
- `apply_stop_condition_inputs`

Later, if needed, we can split transformer execution more finely:

- per-layer tiles
- per-attention/MLP tiles
- explicit KV-cache update tiles

But the first pass should stay coarse enough to keep progress fast.

### Phase 3: logits to token/text

Primary goal:

- turn logits into a deterministic next token and eventually output text

For now, decode should be deterministic and simple:

- greedy argmax
- explicit tie-break rule
- no stochastic sampling

Likely tile candidates:

- `select_next_token`
- `append_token`
- `check_stop_condition`
- `detokenize_output_tokens`

### Recursive structure

Where looping is needed, prefer a Raster-shaped step interface.

For example:

- recursive tile: `decode_step(state) -> (done, next_state)`
- or recursive sequence: `seq_decode_loop(state) -> final_state`

The key rule is that the loop state must be explicit and returned each step. No hidden mutable globals.

## What to keep from `mistral.rs`, and what to ignore

### Usually keep

- model-specific prompt formatting behavior
- tokenization behavior
- tensor math that changes outputs
- deterministic truncation or decode behavior
- exact order of math when that affects parity

### Usually ignore

- generic request plumbing
- scheduler/batching logic
- cache allocation strategies that are purely performance-driven
- tracing/logging
- model-family abstraction layers
- runtime niceties that do not affect outputs

## Planning checklist for future AI agents

Before planning or implementing a new seam in `raster-inference`, answer these questions.

### Scope

- What exact part of inference are we implementing?
- What user-visible output or internal checkpoint must match the reference?
- What model path are we targeting right now?

### Inputs and outputs

- What are the explicit inputs?
- What values are derived versus externally supplied?
- What exact state must be threaded through the seam?

### Reference extraction

- Which `mistral.rs` files/functions own the semantics we need?
- Which nearby code is only framework/runtime machinery?
- Are there any hidden dynamic values that must be removed or made explicit?

### Tile design

- What is the smallest sensible set of pure functions for this seam?
- Which tiles are iterative versus recursive?
- Can the sequence be written as simple `let`-bound calls with no extra logic?

### Determinism

- Are we depending on time, randomness, environment, filesystem, or network?
- Are we introducing implicit normalization that changes outputs?
- Is every step reproducible from explicit inputs?

### Testing

- What golden vectors will prove this seam is correct?
- What should be compared against `mistral.rs` or the underlying assets?
- What intermediate checkpoints would make debugging easier?

## Implementation posture for this repo

When in doubt, prefer:

- narrower support
- fewer abstractions
- more explicit state
- model-specific code over premature generalization
- direct pure functions over framework-style layering

The success condition is not “this looks like `mistral.rs`.”

The success condition is:

- the supported inference path works correctly
- the code already resembles a Raster program
- future seams can be added by repeating the same planning method
