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
  transformer.rs  Gemma4TransformerModel, Gemma4LayerWeights, Gemma4Ple*, …
  tokenizer.rs    GemmaTokenizerSpec, AuthenticatedGemmaTokenizer, BPE state
  io.rs           safetensors / det-wgt Gemma weight loaders, tokenizer-spec
                  parsing, embedding-row decoding
```

Generic counterparts stay outside the family module:

- `src/shared/model/transformer.rs` — activation sequences, KV caches,
  det-num matrix storage, decode/prefill state types,
- `src/io.rs` — chat-template/tokenizer-file loading, generic
  safetensors/mmap infrastructure, dtype decoding helpers.

Adding a new model family (qwen3.6, MoE variants — planned, not near-term)
means adding `shared/model/<family>/`, **not** threading family types
through `runtime/`, `shared/api/`, or `shared/artifacts/`. Multi-model
*abstraction* (a model trait) is explicitly deferred; this rule is about
containment only.

## Compatibility re-exports

To keep the containment refactor mechanical, the pre-containment import
paths still work via re-exports and are slated for removal once call sites
migrate:

- `shared::model::gemma_tokenizer` → alias of `shared::model::gemma::tokenizer`,
- `shared::model::transformer::Gemma4*` → re-exported from
  `shared::model::gemma::transformer`,
- `io::load_*gemma*` / det-wgt loaders → re-exported from
  `shared::model::gemma::io`.

New code should import from the `shared::model::gemma::*` paths.

## Audited exception list

`grep -rl Gemma src/runtime src/shared/api src/shared/artifacts` is expected
to return only the files below. Each reference is a *type-level use* of the
Gemma model/tokenizer types in entry-point signatures or control fields —
unavoidable until the deferred multi-model abstraction exists, because the
runtime executes exactly one concrete model family today:

| File | Reference | Justification |
| --- | --- | --- |
| `runtime/inference.rs` | `Gemma4TransformerModel` parameter; `InferenceControls.raster_tokenizer_source: Option<AuthenticatedGemmaTokenizer>` | legacy entry-point signature + tokenizer source control; abstraction deferred |
| `runtime/sequence.rs` | `Gemma4TransformerModel` parameter | skeleton passes the model through to executors |
| `runtime/executors/{native,raster}.rs` | `Gemma4TransformerModel`, `AuthenticatedGemma*Source` constructors | executors bind routine auth sources to the concrete model |
| `runtime/pipeline.rs` | `Gemma4TransformerModel` parameters; Gemma model fixtures in tests | test/bench bundle API over the concrete model |
| `runtime/roles/{claimer,challenger}.rs` | `Gemma4TransformerModel`, `AuthenticatedGemmaTokenizer` parameters | role entry points take the concrete model until abstraction lands |
| `runtime/det_goldens.rs` (test-only) | `Gemma4TransformerModel` fixtures | golden-capture test helper |

`runtime/checkpoints.rs`, `runtime/trace.rs`, all of `shared/api/`, and all
of `shared/artifacts/` are Gemma-free, including serialization: no
checkpoint id, trace field, or artifact domain string names a model family.
