//! End-to-end checkpoint trace parity gate.
//!
//! This suite enforces the protocol's honest-party safety property at the
//! trace-artifact level: a committed checkpoint trace must be exactly
//! reproducible by any faithful replay, and native-deterministic and raster
//! execution of the same request must agree at every committed checkpoint.
//!
//! Four test categories (see `docs/parity-gate.md`):
//!
//! 1. Native-deterministic vs full-raster checkpoint parity.
//! 2. Native self-reproducibility (byte-identical serialized artifacts).
//! 3. Thread-count invariance (rayon pool size must not affect the trace).
//! 4. Detour-mode smoke parity (selective raster detour of one routine).
//!
//! All tests run hermetically against `assets/tiny-gemma-dev` in the default
//! `Verified` integrity mode. Traces are captured through the same serialized
//! artifact file a claimer would commit (`RASTER_TRACE_DIR`), so byte-level
//! assertions cover the real on-disk format.
//!
//! The full-raster leg of category 1 is computationally feasible only in
//! optimized builds (Verified-mode Merkle proof work dominates), so that one
//! test is ignored under `debug_assertions` and runs via
//! `cargo test --release --test e2e_checkpoint_parity`.
//!
//! # Native vs raster checkpoint payload mapping (category 1)
//!
//! Native-deterministic and full-raster runs commit identical payloads — and
//! therefore identical SHA256 commitments — for every checkpoint except the
//! three boundary routines below, where the native payload carries value-form
//! fields and the raster payload carries artifact-root-form fields:
//!
//! | Checkpoint            | Native payload (value form)                  | Raster payload (root form)               | Cross-form assertion                                            |
//! |-----------------------|----------------------------------------------|-------------------------------------------|-----------------------------------------------------------------|
//! | `prompt.prepare`      | prompt text, token IDs, token-IDs sha256     | prompt/text/token-ID artifact roots       | outcome `PromptPreparationState` equality (text, IDs, sha256)   |
//! | `input.embedding`     | token IDs + det activation sha256            | adds nested raster artifact roots         | outcome `det_embedded_prompt_activations_sha256` equality       |
//! | `prefill.prepare_aux` | det activations + PLE value commitments      | nested embedding/PLE artifact roots       | transitive: all `prefill.range*` commitments must be identical, |
//! |                       |                                              |                                           | which fails if the prepared auxiliary inputs diverge            |
//!
//! The trace collector is process-global, so every test serializes through
//! [`suite_lock`].

// The parity/golden suites intentionally exercise the deprecated legacy
// entry points: they are what proves the shims stay equivalent.
#![allow(deprecated)]

use std::{
    collections::HashMap,
    env, fs,
    path::PathBuf,
    process,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, MutexGuard,
    },
};

use raster_inference::shared::artifacts::artifact_io::ArtifactIo;
use raster_inference::shared::artifacts::external_artifacts::reset_external_source_store;
use raster_inference::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use raster_inference::shared::model::transformer::Gemma4TransformerModel;
use raster_inference::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path, sequence, InferenceControls, InferenceExecutionMode,
    InferenceRequest, InferenceRunOutcome, ModelSpec, PausedInferenceState, RasterDetourSpec,
    SamplingConfig, TextDecodingPolicy,
};
use serde_json::Value;
use tokenizers::Tokenizer;

/// Prompts restricted to tokens of the tiny-gemma-dev tokenizer vocabulary so
/// the HuggingFace and raster tokenizer paths agree on the encoding. The
/// short prompt bounds the raster leg (prefill cost grows with token count);
/// the longer prompt exercises a wider native prefill.
const SHORT_PROMPT: &str = "hello";
const LONG_PROMPT: &str = "hello raster prompt";

/// Serializes all tests in this binary: the trace collector and the artifact
/// stores are process-wide state.
fn suite_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct Fixture {
    model_spec: ModelSpec,
    tokenizer: Tokenizer,
    transformer_model: Gemma4TransformerModel,
}

impl Fixture {
    fn load() -> Self {
        let model_dir = tiny_gemma_dir();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let chat_template = load_chat_template(model_dir.join("chat_template.jinja"))
            .expect("tiny-gemma-dev chat template should load");
        let tokenizer = load_tokenizer_from_path(&tokenizer_path)
            .expect("tiny-gemma-dev tokenizer should load");
        let transformer_model = load_transformer_state_model_from_det_num_wgt_path(&model_dir)
            .expect("tiny-gemma-dev deterministic weights should load");
        Self {
            model_spec: ModelSpec {
                model_id: "tiny-gemma-dev".to_string(),
                tokenizer_path,
                chat_template,
                bos_token: None,
                eos_token: None,
                unk_token: None,
            },
            tokenizer,
            transformer_model,
        }
    }

    fn raster_tokenizer_source(&self) -> AuthenticatedGemmaTokenizer {
        AuthenticatedGemmaTokenizer::new(
            load_gemma_tokenizer_spec_from_path(&self.model_spec.tokenizer_path)
                .expect("tiny-gemma-dev tokenizer spec should load"),
        )
    }
}

fn tiny_gemma_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/tiny-gemma-dev")
}

fn deterministic_request(prompt: &str, max_new_tokens: usize) -> InferenceRequest {
    InferenceRequest {
        prompt_bytes: prompt.as_bytes().to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: true,
        add_special_tokens: true,
        execution_mode: InferenceExecutionMode::Deterministic,
        sampling: SamplingConfig {
            max_new_tokens: Some(max_new_tokens),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    }
}

struct TraceRun {
    outcome: InferenceRunOutcome,
    /// Raw bytes of the serialized checkpoint trace artifact file — the exact
    /// bytes a claimer would commit.
    raw_artifact: Vec<u8>,
    /// Parsed artifact: an array of single-key `{checkpoint_id: sha256-hex}`
    /// objects in commit order.
    checkpoints: Value,
}

/// Runs one inference with fresh per-run stores and a fresh trace directory,
/// returning the serialized checkpoint trace artifact.
///
/// The caller must hold [`suite_lock`] for the duration of all runs it
/// intends to compare.
fn run_and_capture_trace(
    fixture: &Fixture,
    request: &InferenceRequest,
    controls: &InferenceControls,
) -> TraceRun {
    static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
    let run_id = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let trace_dir = env::temp_dir().join(format!(
        "raster-e2e-checkpoint-parity-{}-{run_id}",
        process::id()
    ));
    fs::create_dir_all(&trace_dir).expect("trace dir should be created");
    env::set_var("RASTER_TRACE_DIR", &trace_dir);

    ArtifactIo::reset_store();
    reset_external_source_store();

    let outcome = sequence::run(
        request,
        &fixture.model_spec,
        &fixture.tokenizer,
        &fixture.transformer_model,
        controls,
    )
    .expect("inference should complete");

    let mut trace_files = fs::read_dir(&trace_dir)
        .expect("trace dir should be readable")
        .map(|entry| entry.expect("trace dir entry should read").path())
        .collect::<Vec<_>>();
    assert_eq!(
        trace_files.len(),
        1,
        "expected exactly one trace artifact in {}, found {trace_files:?}",
        trace_dir.display()
    );
    let trace_path = trace_files.remove(0);
    let raw_artifact = fs::read(&trace_path).expect("trace artifact should be readable");
    let checkpoints: Value =
        serde_json::from_slice(&raw_artifact).expect("trace artifact should be valid JSON");
    assert!(
        checkpoints
            .as_array()
            .is_some_and(|array| !array.is_empty()),
        "trace artifact should be a non-empty checkpoint array"
    );
    fs::remove_dir_all(&trace_dir).expect("trace dir should be removable");

    TraceRun {
        outcome,
        raw_artifact,
        checkpoints,
    }
}

fn expect_completed(run: &TraceRun, description: &str) {
    assert!(
        matches!(run.outcome, InferenceRunOutcome::Completed(_)),
        "expected {description} to run to completion"
    );
}

fn expect_paused_at<'run>(
    run: &'run TraceRun,
    terminal_checkpoint_id: &str,
    description: &str,
) -> &'run PausedInferenceState {
    match &run.outcome {
        InferenceRunOutcome::Paused(paused) => {
            assert_eq!(
                paused.terminal_checkpoint_id, terminal_checkpoint_id,
                "{description} paused at an unexpected terminal checkpoint"
            );
            paused
        }
        _ => panic!("expected {description} to pause at `{terminal_checkpoint_id}`"),
    }
}

fn checkpoint_entry_name_and_commitment(entry: &Value, idx: usize) -> (&str, &str) {
    let object = entry
        .as_object()
        .unwrap_or_else(|| panic!("checkpoint entry {idx} should be an object"));
    assert_eq!(
        object.len(),
        1,
        "checkpoint entry {idx} should contain exactly one commitment"
    );
    let (checkpoint, commitment) = object
        .iter()
        .next()
        .unwrap_or_else(|| panic!("checkpoint entry {idx} should contain a commitment"));
    (
        checkpoint.as_str(),
        commitment
            .as_str()
            .unwrap_or_else(|| panic!("checkpoint entry {idx} commitment should be a string")),
    )
}

/// Describes one committed checkpoint as `id` plus its 1-based occurrence
/// among entries with the same id, e.g. `prefill.range:2`.
fn occurrence_label(occurrences: &mut HashMap<String, usize>, checkpoint: &str) -> String {
    let occurrence = occurrences
        .entry(checkpoint.to_string())
        .and_modify(|count| *count += 1)
        .or_insert(1);
    format!("{checkpoint}:{occurrence}")
}

/// Asserts the two checkpoint trace artifacts agree on the ordered sequence
/// of checkpoint IDs/occurrences and on every committed value. On divergence,
/// panics naming the first divergent checkpoint ID and occurrence.
///
/// `allow_divergence_at` lists `checkpoint-id:occurrence` labels whose
/// commitments legitimately differ between the two runs (schema-divergent
/// boundary checkpoints, or a detoured routine occurrence); every entry in
/// the list must be present in the traces, and every entry outside the list
/// must be identical.
fn assert_traces_match(
    left_label: &str,
    left: &Value,
    right_label: &str,
    right: &Value,
    allow_divergence_at: &[&str],
) {
    let left_entries = left
        .as_array()
        .expect("left checkpoint payload should be an array");
    let right_entries = right
        .as_array()
        .expect("right checkpoint payload should be an array");

    let mut occurrences = HashMap::new();
    let mut allowed_seen = 0;
    let common_len = left_entries.len().min(right_entries.len());
    for idx in 0..common_len {
        let (left_checkpoint, left_commitment) =
            checkpoint_entry_name_and_commitment(&left_entries[idx], idx);
        let (right_checkpoint, right_commitment) =
            checkpoint_entry_name_and_commitment(&right_entries[idx], idx);
        assert_eq!(
            left_checkpoint, right_checkpoint,
            "checkpoint ID sequence diverges at entry {idx}: \
             {left_label} committed `{left_checkpoint}` but {right_label} committed `{right_checkpoint}`"
        );
        let label = occurrence_label(&mut occurrences, left_checkpoint);
        if allow_divergence_at.contains(&label.as_str()) {
            allowed_seen += 1;
            continue;
        }
        assert_eq!(
            left_commitment, right_commitment,
            "first divergent checkpoint: `{label}` (entry {idx}): \
             {left_label} committed {left_commitment} but {right_label} committed {right_commitment}"
        );
    }
    if left_entries.len() != right_entries.len() {
        let (longer_label, longer) = if left_entries.len() > right_entries.len() {
            (left_label, left_entries)
        } else {
            (right_label, right_entries)
        };
        let (extra_checkpoint, _) =
            checkpoint_entry_name_and_commitment(&longer[common_len], common_len);
        panic!(
            "checkpoint counts diverge: {left_label} committed {} checkpoints but {right_label} \
             committed {}; first unmatched checkpoint is `{extra_checkpoint}` (entry {common_len}) in {longer_label}",
            left_entries.len(),
            right_entries.len(),
        );
    }
    assert_eq!(
        allowed_seen,
        allow_divergence_at.len(),
        "expected all schema-divergent/detoured checkpoints {allow_divergence_at:?} to be \
         present, but only {allowed_seen} matched"
    );
}

fn checkpoint_ids(trace: &Value) -> Vec<String> {
    trace
        .as_array()
        .expect("checkpoint payload should be an array")
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            checkpoint_entry_name_and_commitment(entry, idx)
                .0
                .to_string()
        })
        .collect()
}

/// Asserts the trace covers the minimum end-to-end span required by the
/// parity gate: prompt prepare, input embedding, full prefill, at least one
/// token selection, and at least one decode transition.
fn assert_minimum_span(label: &str, trace: &Value) {
    let ids = checkpoint_ids(trace);
    for required in [
        "prompt.prepare",
        "input.embedding",
        "prefill.prepare_aux",
        "prefill.finalize",
        "decode.select_token",
        "decode.transition_finalize",
    ] {
        assert!(
            ids.iter().any(|id| id == required),
            "{label} trace is missing required checkpoint `{required}`; committed IDs: {ids:?}"
        );
    }
}

/// §3.1 — full-native vs full-raster checkpoint parity, Verified integrity
/// mode. Both legs run the identical request bounded by a terminal checkpoint
/// after the first decode transition; the span still crosses prompt prepare,
/// input embedding, full prefill, one token selection, and one decode
/// transition (asserted explicitly).
///
/// Ignored in debug builds: the Verified-mode full-raster leg performs Merkle
/// proof work per tile and only completes in reasonable time with an
/// optimized build. CI runs it via
/// `cargo test --release --test e2e_checkpoint_parity`.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "full-raster Verified-mode leg requires an optimized build; run with \
              cargo test --release --test e2e_checkpoint_parity"
)]
fn native_and_raster_traces_agree_end_to_end() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let terminal = "decode.transition_finalize";

    let native = run_and_capture_trace(
        &fixture,
        &request,
        &InferenceControls {
            commit_checkpoints: true,
            terminal_checkpoint: Some(terminal.to_string()),
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    let native_state = expect_paused_at(&native, terminal, "native deterministic run");

    let raster = run_and_capture_trace(
        &fixture,
        &request,
        &InferenceControls {
            commit_checkpoints: true,
            terminal_checkpoint: Some(terminal.to_string()),
            raster: true,
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    let raster_state = expect_paused_at(&raster, terminal, "full raster run");

    assert_minimum_span("native", &native.checkpoints);
    assert_minimum_span("raster", &raster.checkpoints);

    // Every checkpoint commitment must be identical except the three
    // schema-divergent boundary checkpoints (see the module-level mapping
    // table), which are compared field-wise below.
    assert_traces_match(
        "native",
        &native.checkpoints,
        "raster",
        &raster.checkpoints,
        &[
            "prompt.prepare:1",
            "input.embedding:1",
            "prefill.prepare_aux:1",
        ],
    );

    // prompt.prepare: both paths materialize the same prompt preparation
    // values even though the raster checkpoint commits artifact roots.
    assert_eq!(
        native_state.input_embedding.prompt_preparation,
        raster_state.input_embedding.prompt_preparation,
        "native and raster runs disagree on prompt preparation \
         (prompt.prepare checkpoint inputs)"
    );

    // input.embedding: the canonical deterministic activation commitment must
    // agree across both payload forms.
    assert!(
        native_state
            .input_embedding
            .det_embedded_prompt_activations_sha256
            .is_some(),
        "native run should commit deterministic embedded prompt activations"
    );
    assert_eq!(
        native_state
            .input_embedding
            .det_embedded_prompt_activations_sha256,
        raster_state
            .input_embedding
            .det_embedded_prompt_activations_sha256,
        "native and raster runs disagree on the deterministic embedded prompt \
         activation commitment (input.embedding checkpoint inputs)"
    );

    // prefill.prepare_aux: agreement is enforced transitively — the
    // prefill.range/range_finalize/finalize commitments asserted above
    // consume the prepared auxiliary inputs and would diverge if they
    // differed.
}

/// §3.2 — native self-reproducibility: the same request run twice must
/// produce byte-identical serialized trace artifacts. This certifies the
/// serialized artifact as a trustworthy golden baseline. This leg runs the
/// full pipeline to completion (no terminal checkpoint).
#[test]
fn native_deterministic_trace_is_byte_reproducible() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(LONG_PROMPT, 8);
    let controls = InferenceControls {
        commit_checkpoints: true,
        raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
        ..Default::default()
    };

    let first = run_and_capture_trace(&fixture, &request, &controls);
    let second = run_and_capture_trace(&fixture, &request, &controls);
    expect_completed(&first, "first native deterministic run");
    expect_completed(&second, "second native deterministic run");

    if first.raw_artifact != second.raw_artifact {
        // Produce an actionable checkpoint-level diagnostic before failing on
        // the byte-level mismatch.
        assert_traces_match(
            "first run",
            &first.checkpoints,
            "second run",
            &second.checkpoints,
            &[],
        );
        panic!(
            "serialized trace artifacts differ at the byte level even though all checkpoint \
             commitments agree (artifact serialization is nondeterministic)"
        );
    }
}

/// §3.3 — thread-count invariance: the committed trace artifact must be
/// byte-identical across rayon pool sizes. This is the test directly
/// sensitive to side effects executed on rayon worker threads, which have
/// fresh thread-local state.
#[test]
fn native_trace_is_invariant_across_rayon_thread_counts() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(LONG_PROMPT, 4);
    let controls = InferenceControls {
        commit_checkpoints: true,
        raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
        ..Default::default()
    };

    let ambient = run_and_capture_trace(&fixture, &request, &controls);
    expect_completed(&ambient, "ambient-pool native run");

    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool should build");
        let run = pool.install(|| run_and_capture_trace(&fixture, &request, &controls));
        expect_completed(&run, "scoped-pool native run");
        if ambient.raw_artifact != run.raw_artifact {
            assert_traces_match(
                "ambient pool",
                &ambient.checkpoints,
                &format!("{threads}-thread pool"),
                &run.checkpoints,
                &[],
            );
            panic!(
                "trace artifacts differ at the byte level between the ambient pool and a \
                 {threads}-thread pool even though all checkpoint commitments agree"
            );
        }
    }
}

/// §3.4 — detour-mode smoke parity: a selective raster detour of the first
/// `prefill.range` occurrence must leave every checkpoint outside the
/// detoured routine identical to the full-native run, while the detoured
/// occurrence commits the raster-form payload.
#[test]
fn detour_of_first_prefill_range_preserves_all_other_checkpoints() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);

    let native = run_and_capture_trace(
        &fixture,
        &request,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    expect_completed(&native, "native deterministic run");

    let detour = run_and_capture_trace(
        &fixture,
        &request,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range").expect("detour spec should parse"),
            ),
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    expect_completed(&detour, "prefill.range detour run");

    assert_traces_match(
        "native",
        &native.checkpoints,
        "detour",
        &detour.checkpoints,
        &["prefill.range:1"],
    );
}
