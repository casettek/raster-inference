# Selective Raster Detour Setup

Selective raster detour mode lets a deterministic CPU run nominate one routine occurrence for future raster execution. The setup intentionally keeps native deterministic CPU as the outer driver before and after the detour.

## Invariant

The committed checkpoint bundle represents logical deterministic inference state, not the execution backend. A correct CPU-plus-one-raster-detour run should match a pure CPU deterministic run in checkpoint names, ordering, occurrence counts, hashes, and final output.

Raster-specific details such as artifact roots, proof metadata, tile counts, or backend labels should stay out of committed checkpoint payloads. Use verbose trace output or separate telemetry for those details.

## Selector

The selector uses routine ids, not arbitrary checkpoint ids:

- `prompt.prepare`
- `input.embedding`
- `prefill.prepare_aux`
- `prefill.range`
- `prefill.range_finalize`
- `prefill.finalize`
- `decode.select_token`
- `decode.transition`
- `output.finalize`

Occurrence suffixes follow the existing `id:N` convention, for example `prefill.range_finalize:2`. Sub-checkpoints such as `prefill.layer_token.layer_0.token_0` are not routine ids and are not valid detour targets.

## Execution Shape

```text
native deterministic before routine
  -> selected routine boundary
  -> native state to raster refs
  -> one raster routine execution
  -> raster refs to native state
  -> native deterministic continues
```

The shared setup provides the selector, CLI/API control plumbing, raster sizing and integrity control propagation, and explicit unsupported errors. Each routine still needs its own plan and adapter implementation.

## Raster Controls

`--raster-at` is raster execution for one routine occurrence. Any raster execution parameters supplied with it must flow to that selected routine:

- projection rows per tile
- attention KV rows per tile
- sequence rows per tile
- head rows per tile
- prefill token range width
- tokenizer BPE chunk sizes
- output byte flush chunk size
- raster integrity mode

Future routine implementations should consume `RasterSizingControls` through the same path used by whole-run raster execution.

## Routine Plan Handoff

Use `docs/plans/selective-raster-routine-template.md` as the starter for each routine-specific plan. Each plan should identify the native boundary, raster input refs, authenticated source, materializer, checkpoint parity target, and trace equality tests for exactly one routine.

Useful adapter patterns:

- `format_native_prompt_as_raster_checkpoint_for_trace`
- `format_native_input_embedding_as_raster_checkpoint_for_trace`
- `format_native_prefill_prepare_aux_as_raster_checkpoint_for_trace`
- `materialize_prefill_layer_output_refs_from_roots_for_trace`
- `materialize_decode_state_from_raster_state_for_trace`
- `materialize_output_decode_state_for_api`
