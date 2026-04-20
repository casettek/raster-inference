# Checkpoint Commitment Reference

This document inventories every checkpoint commitment emitted by `raster-inference` today and describes the raw data hashed at each point.

The goal is instrumentation parity: another inference library should be able to capture the same raw values at the same semantic points, hash them the same way, and produce checkpoint commitments compatible with this implementation.

## How checkpoint commitments work

All checkpoint commitments go through `trace_checkpoint()` in `src/trace.rs`.

```rust
pub fn trace_checkpoint<T: Serialize>(phase: &str, state: &T) {
    collector.checkpoints.push(json!({
        phase: sha256_hex(state),
    }));
}
```

Important consequences:

- Each checkpoint stores only `{ checkpoint_name: digest }` in the final trace bundle.
- The digest is `sha256_hex(state)`.
- `sha256_hex(state)` means: serialize `state` with `serde_json::to_vec`, then SHA-256 those exact bytes, then hex-encode the digest.
- For compatibility, the instrumented library should build the same logical payload object and hash its JSON bytes in the same way.

## Related helper commitment functions

Several checkpoint payloads include nested hash fields in addition to the top-level checkpoint digest.

### `trace::sha256_hex(value)`

Location: `src/trace.rs`

- Input: any `serde::Serialize` value
- Bytes hashed: `serde_json::to_vec(value)`
- Output: lowercase hex SHA-256 string

Used for:

- token id arrays
- logits arrays
- generated text
- some optional per-layer PLE inputs

### `build_phase2_commitment(activations)`

Location: `src/phase2/tiles.rs`

- Input: `&[Vec<f32>]`
- Bytes hashed: each `f32` in row-major order via `to_le_bytes()`
- Output: lowercase hex SHA-256 string

Used for:

- activation sequence commitments
- current prefill activations
- final hidden states

### `build_vector_commitment(values)`

Location: `src/phase2/tiles.rs`

- Input: `&[f32]`
- Bytes hashed: each `f32` via `to_le_bytes()`
- Output: lowercase hex SHA-256 string

Used for:

- single-token activation vectors
- decode input activation
- decode current activation
- prefill logits

### `build_phase3_commitment(token_ids)`

Location: `src/phase3/tiles.rs`

- Input: `&[u32]`
- Bytes hashed: `serde_json::to_vec(token_ids)`
- Output: lowercase hex SHA-256 string

Used for:

- generated token id sequence commitments during decode/output checkpoints

## Layer cache serialization

Several checkpoints include serialized KV cache state via `serialize_layer_caches()` in `src/trace.rs`.

The serialized shape is:

```text
[
  {
    "keys":   [head0_rows, head1_rows, ...],
    "values": [head0_rows, head1_rows, ...]
  },
  ...
]
```

Where:

- outer array = one entry per layer cache
- `keys`/`values` = one array per KV head
- each head contains an ordered sequence of cached token rows
- each row is a `Vec<f32>`

Any external instrumenter should capture the logical cache contents in the same order:

- layers in model order
- heads in head index order
- cached rows in token position order
- row values in dimension order

## What is not a checkpoint

`start_inference_trace()` and `finish_inference_trace()` are trace lifecycle calls, but they do not add entries to the checkpoint array.

- `start_inference_trace()` clears the collector
- `finish_inference_trace()` writes the collected checkpoint digests to disk
- `abort_inference_trace()` writes whatever had already been collected before failure

The summary payload passed to `finish_inference_trace()` is not itself a checkpoint commitment.

## Checkpoint inventory

The checkpoints below are listed in execution order.

### 1. `prompt.prepare`

Location: `src/lib.rs`

When to record:

- After phase 1 prompt processing
- After prompt token embeddings are available
- Before phase 2 prefill begins

Payload fields:

- `prompt_text: String`
- `prompt_token_ids: Vec<u32>`
- `prompt_token_ids_sha256: String`
- `embedded_prompt_activations: Vec<Vec<f32>>`
- `embedded_prompt_activations_sha256: String`
- `sampling: SamplingConfig`

Notes:

- `prompt_token_ids_sha256` is inherited from phase 1 state.
- `embedded_prompt_activations_sha256` is the activation commitment returned by the embedding step.

### 2. `prefill.prepare_aux`

Location: `src/phase2/mod.rs`

When to record:

- After optional prefill PLE inputs are computed
- Before entering the layer stack

Payload fields:

- `prompt_token_ids: &[u32]`
- `prompt_token_ids_sha256: String`
- `embedded_prompt_activations: Vec<Vec<f32>>`
- `embedded_prompt_activations_sha256: String`
- `per_layer_prefill_inputs: Option<Vec<Option<Vec<Vec<f32>>>>>`
- `per_layer_prefill_input_sha256s: Option<Vec<Option<String>>>`

Notes:

- `per_layer_prefill_inputs` is `None` if the model has no global PLE.
- Each element inside `per_layer_prefill_inputs` is either:
  - `None` for a layer without PLE, or
  - a sequence-shaped `Vec<Vec<f32>>` aligned to prompt token order
- Each element inside `per_layer_prefill_input_sha256s` is the JSON-hash of the corresponding optional per-layer input payload via `trace::sha256_hex`.

### 3. `prefill.layer`

Location: `src/phase2/tiles.rs`

Checkpoint family:

- emitted once per transformer layer during prefill

When to record:

- After one prefill layer finishes
- After the layer cache for that layer has been appended
- Before emitting token-level checkpoints for that layer

Payload fields:

- `next_layer_idx: usize`
- `current_activations: Vec<Vec<f32>>`
- `current_activations_sha256: String`
- `layer_caches: Vec<SerializableLayerKvCache>`
- `completed_layer_output_sha256s: Vec<String>`

Notes:

- `next_layer_idx` is `layer_idx + 1`.
- `current_activations` is the full post-layer activation sequence for the prompt.
- `current_activations_sha256` is `build_phase2_commitment(current_activations)`.
- `completed_layer_output_sha256s` contains one activation commitment per completed layer so far, in layer order.

### 4. `prefill.layer_token.layer_{layer_idx}.token_{token_idx}`

Location: `src/phase2/tiles.rs`

Checkpoint family:

- emitted once per token for each completed prefill layer

When to record:

- Immediately after the corresponding `prefill.layer` checkpoint
- One checkpoint per token row in the layer output sequence

Payload fields:

- `layer_idx: usize`
- `token_idx: usize`
- `token_count: usize`
- `token_activation: Vec<f32>`

Notes:

- `token_activation` is the single post-layer activation row for that token.
- These checkpoints are emitted after the full layer completes; they are not emitted from an actually token-by-token prefill execution path.

### 5. `prefill.finalize`

Location: `src/phase2/mod.rs`

When to record:

- After all prefill layers finish
- After final RMS norm and logits projection
- Before phase 2 returns its prefill result

Payload fields:

- `final_hidden_states: Vec<Vec<f32>>`
- `final_hidden_states_sha256: String`
- `prefill_logits: Vec<f32>`
- `prefill_logits_sha256: String`
- `decode_position: usize`
- `decode_token_count: usize`
- `layer_caches: Vec<SerializableLayerKvCache>`

Notes:

- `final_hidden_states` is the post-layer-stack activation sequence before final position selection is discarded.
- `prefill_logits` is the final prompt-position logits vector used to start decode.
- `prefill_logits_sha256` is `build_vector_commitment(prefill_logits)`.
- `decode_position` and `decode_token_count` are both `prompt_token_ids.len()` at this point.

### 6. `decode.select_token`

Location: `src/phase3/mod.rs`

When to record:

- At the start of each decode iteration
- After greedy token selection and append
- Before running the phase 2 decode step for that selected token

Payload fields:

- `full_token_ids: Vec<u32>`
- `full_token_ids_sha256: String`
- `generated_token_ids: Vec<u32>`
- `generated_token_ids_sha256: String`
- `current_logits: Vec<f32>`
- `current_logits_sha256: String`
- `selected_next_token: u32`
- `decode_position: usize`
- `decode_token_count: usize`
- `layer_caches: Vec<SerializableLayerKvCache>`
- `max_new_tokens: usize`

Notes:

- `full_token_ids` includes prompt + generated tokens so far, including the newly selected token.
- `generated_token_ids` includes generated tokens only, including the newly selected token.
- `full_token_ids_sha256` and `current_logits_sha256` use `trace::sha256_hex`.
- `generated_token_ids_sha256` uses `build_phase3_commitment`.

### 7. `decode.layer_token.layer_{layer_idx}.position_{position}`

Location: `src/phase2/tiles.rs`

Checkpoint family:

- emitted once per layer during each decode step

When to record:

- After one decode layer finishes for the current token/position
- After updating the completed prefix of layer caches
- Before moving to the next layer

Payload fields:

- `token_id: u32`
- `position: usize`
- `next_layer_idx: usize`
- `decode_input_activation: Vec<f32>`
- `decode_input_activation_sha256: String`
- `current_activation: Vec<f32>`
- `current_activation_sha256: String`
- `layer_caches: Vec<SerializableLayerKvCache>`
- `completed_layer_output_sha256s: Vec<String>`

Notes:

- `decode_input_activation` is the original embedded token activation that entered the decode layer stack, not the running `xs`.
- `decode_input_activation_sha256` is `build_vector_commitment(decode_input_activation)`.
- `current_activation` is the current post-layer activation after this layer.
- `current_activation_sha256` is `build_vector_commitment(current_activation)`.
- `layer_caches` is a stitched view:
  - updated caches for completed layers
  - untouched prior caches for remaining layers
- `completed_layer_output_sha256s` contains one vector commitment per completed decode layer so far.

### 8. `decode.finalize`

Location: `src/phase3/mod.rs`

When to record:

- After the phase 2 decode step completes for the selected token
- After new logits are available for the next decode iteration

Payload fields:

- `full_token_ids: Vec<u32>`
- `full_token_ids_sha256: String`
- `generated_token_ids: Vec<u32>`
- `generated_token_ids_sha256: String`
- `current_logits: Vec<f32>`
- `current_logits_sha256: String`
- `decode_position: usize`
- `decode_token_count: usize`
- `layer_caches: Vec<SerializableLayerKvCache>`

Notes:

- This is the post-decode-step state that seeds the next `decode.select_token`.
- Hash helpers match `decode.select_token`.

### 9. `output.finalize`

Location: `src/phase3/mod.rs`

When to record:

- When decode stops because the stop condition is satisfied
- After generated tokens are detokenized to text
- Before returning final phase 3 state

Payload fields:

- `full_token_ids: Vec<u32>`
- `full_token_ids_sha256: String`
- `generated_token_ids: Vec<u32>`
- `generated_token_ids_sha256: String`
- `generated_text: String`
- `generated_token_count: usize`
- `stop_reason: Phase3StopReason`

Notes:

- `generated_token_ids_sha256` uses `build_phase3_commitment`.
- `generated_text` is committed as raw string content inside the checkpoint payload.

## Instrumentation guidance

For another library to produce compatible checkpoint commitments, it should:

1. Record the same semantic checkpoint moments listed above.
2. Build payload objects with the same field names and logical values.
3. Preserve the same ordering semantics for sequence-shaped data:
   - tokens in prompt/decode order
   - layers in model order
   - heads in head index order
   - vector dimensions in native dimension order
4. Recompute nested helper digests using the same helper rules documented above.
5. Hash the final payload object as JSON bytes with SHA-256 to obtain the checkpoint digest.

## Source files

The current checkpoint implementation is spread across:

- `src/trace.rs`
- `src/lib.rs`
- `src/phase2/mod.rs`
- `src/phase2/tiles.rs`
- `src/phase3/mod.rs`
- `src/phase3/tiles.rs`
