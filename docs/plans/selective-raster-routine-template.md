# Selective Raster Routine Plan Template

Use this template when creating a focused plan for one routine in selective raster detour mode. The shared setup keeps native deterministic CPU as the outer driver and lets one selected routine occurrence switch to raster, then materialize back to native state.

## Routine Target

- Routine id:
- Example selector:
- Occurrence semantics:
- Unsupported sub-checkpoints:

## Native Boundary

- Before-execution native boundary:
- Native values available at that boundary:
- Values that must not be recomputed:
- State that must continue after the raster detour:

## Raster Inputs

- Native values to insert into the raster artifact store:
- Existing native-to-raster formatter to reuse or create:
- Authenticated raster source required:
- Raster sizing controls consumed:
- Integrity mode handling:

## Raster Execution

- Existing `run_raster` entrypoint:
- Exact routine instance to execute:
- Checkpoint emission ownership:
- Tile telemetry behavior:

## Native Materialization

- Existing raster-to-native materializer to reuse or create:
- Native output structs to reconstruct:
- Canonical deterministic values that must be preserved:
- Failure mode if raster output cannot be materialized safely:

## Trace Parity

- Committed checkpoint payload that must match pure CPU:
- Checkpoint name and occurrence:
- Values excluded from committed checkpoints:
- Optional telemetry outside the committed trace:

## Tests

- Pure CPU deterministic trace:
- CPU plus one raster detour trace:
- Final output equality:
- Unsupported/error cases:
- Raster sizing or integrity control coverage:

## References To Follow

- `src/routines/prompt_prepare/mod.rs`
- `src/routines/input_embedding/mod.rs`
- `src/routines/prefill_prepare_aux/mod.rs`
- `src/routines/prefill_layer/mod.rs`
- `src/routines/decode_transition/mod.rs`
- `src/routines/output_finalize/mod.rs`
