# Model-Agnostic Layers and Gemma Containment

## The rule

The following layers must remain **model-agnostic in naming and
serialization** — they define the protocol surface, not any one model
family:

- the checkpoint taxonomy (`src/runtime/checkpoints.rs`),
- the trace format and emission layer (`src/runtime/trace.rs`),
- the artifact commitment contract (`src/shared/artifacts/`),
- the `src/runtime/` orchestration layer (sequence skeleton, executors,
  roles),
- the role APIs and audit report types (`src/shared/api/`).

Model-family-specific code lives under `src/shared/model/<family>/`. Today
the only family is Gemma:

```
src/shared/model/gemma/
  mod.rs          Gemma4Prompt and module docs
  adapter.rs      GemmaModelBundle and runtime-facing source builders
  transformer.rs  Gemma4TransformerModel, Gemma4LayerWeights, Gemma4Ple*, …
  tokenizer.rs    GemmaTokenizerSpec, AuthenticatedGemmaTokenizer, BPE state
  io.rs           safetensors / det-wgt Gemma weight loaders, tokenizer-spec
                  parsing, embedding-row decoding
```

Generic counterparts stay outside the family module:

- `src/shared/model/runtime.rs` — `LoadedModel`, the narrow enum-backed
  runtime boundary used by sequence, executors, roles, and pipeline helpers,
- `src/shared/model/transformer.rs` — activation sequences, KV caches,
  det-num matrix storage, decode/prefill state types,
- `src/io.rs` — chat-template/tokenizer-file loading, generic
  safetensors/mmap infrastructure, dtype decoding helpers.

Adding a new model family (qwen3.6, MoE variants — planned, not near-term)
means adding `shared/model/<family>/`, **not** threading family types
through `runtime/`, `shared/api/`, or `shared/artifacts/`. Multi-model
*abstraction* (a model trait) is explicitly deferred; this rule is about
containment only. Runtime code may depend on `LoadedModel`; it should not
match on concrete model-family internals.

## Compatibility re-exports

To keep the containment refactor mechanical, two pre-containment import
paths still work via re-exports (the `shared::model::gemma_tokenizer` alias
and the crate-root flat re-exports have been removed):

- `shared::model::transformer::Gemma4*` → re-exported from
  `shared::model::gemma::transformer`,
- `io::load_*gemma*` / det-wgt loaders → re-exported from
  `shared::model::gemma::io`.

New code should import from the `shared::model::gemma::*` paths.

## Enforcement

`tests/model_family_containment.rs` scans the protected protocol/runtime
surfaces and fails if model-family names leak back into them. The protected
surface includes:

- `src/runtime/sequence.rs`,
- `src/runtime/inference/mod.rs`,
- `src/runtime/pipeline.rs`,
- `src/runtime/executors/`,
- `src/runtime/roles/`,
- `src/runtime/checkpoints.rs`,
- `src/runtime/trace.rs`,
- `src/shared/api/`,
- `src/shared/artifacts/`.

Test-only fixtures, `src/shared/model/gemma/`, and existing routine/auth-source
internals may still name Gemma. The next cleanup workstream can rename or wrap
authenticated routine sources; this containment layer only ensures runtime and
protocol surfaces do not carry Gemma types or constructors.
