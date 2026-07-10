# `input.embedding` Raster-Core Port Plan

`input.embedding` follows the real-raster authoring model from
`docs/real-raster-routine-authoring-guide.md`: collections stay in committed
storage, state carries descriptors and scalar cursors, and materialization
happens at the host-visible routine boundary.

## Inputs

- `prompt_token_ids`: run-local raster-encoded prompt token ids reconstructed
  from `PromptPreparationState` after verifying `prompt_token_ids_sha256`.
- `embedding`: pre-encoded raster Gemma input embedding rows (strict lookup
  against the `encode-externals` directory; cache kind
  `gemma-input-embedding-v3`). Rows are already-scaled canonical Act bits
  from the authenticated embedding source, hex-packed one `String` leaf per
  row (`pack_embedding_row_hex`): per-value leaves put every i32 in its own
  raster index node, which is unencodable and unloadable at real model
  scale (537M nodes for Gemma E4B), while one-leaf rows keep the index
  O(vocab). Tiles select a row leaf and decode it in-tile
  (`unpack_embedding_row_hex`); a malformed packed row is a committed `Err`
  outcome.
- `loop_drivers`: postcard token chunk ordinals.
- `config`: postcard chunk sizing and the roots/commitments the program echoes
  in its output.

## Storage Shape

- Model-scoped embedding rows stay behind the `EmbeddingSource` descriptor.
- Prompt-derived token ids stay behind the `PromptTokenSource` descriptor.
- Recur state is `InputEmbeddingCopyState { next_token_idx }`.
- The recur output draft accumulates activation rows; the final output
  materializes only at `output.bin` so the host can rebuild native
  `ActivationSequence`.

## Trace Guards

The program crate guards reject eager prompt/model collections in recur state
or nonterminal tile inputs. Sentinel trace tests assert token ids and embedding
rows cross the ABI as authenticated reads, not inline `FnInput.data` payloads.

## Checkpoint Semantics

Raster-core detours return native-form activations to the host. The normal
`input.embedding` checkpoint remains backend-invariant with the native run; the
existing artifact-root boundary exception remains limited to full-raster and
sim-raster paths.
