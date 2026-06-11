//! Golden checkpoint-trace artifact gate for the orchestration refactor.
//!
//! Each test runs one fixed configuration on `assets/tiny-gemma-dev` with
//! checkpoint commitment enabled and asserts the serialized trace artifact is
//! **byte-identical** to a golden file checked in under `tests/goldens/`.
//! The goldens were captured before the orchestration-split refactor began
//! and are its contract: every refactor step must leave them untouched.
//!
//! Regenerate (only when an intentional, separately-reviewed format change
//! lands) with:
//!
//! ```text
//! UPDATE_GOLDENS=1 cargo test --test golden_traces
//! UPDATE_GOLDENS=1 cargo test --release --test golden_traces -- --ignored
//! ```
//!
//! Configuration matrix (invalid combinations excluded — raster and detour
//! require deterministic execution):
//!
//! | Golden                          | Mode | Path                       | Terminal checkpoint            |
//! |---------------------------------|------|----------------------------|--------------------------------|
//! | `native-det-full`               | det  | native                     | none (runs to completion)      |
//! | `native-det-terminal`           | det  | native                     | `prefill.finalize`             |
//! | `detour-prefill-range`          | det  | detour at `prefill.range`  | none                           |
//! | `raster-full`                   | det  | full raster                | none                           |
//! | `raster-terminal`               | det  | full raster                | `decode.transition_finalize`   |
//! | `fp32-native-full`              | fp32 | native                     | none                           |
//!
//! The two full-raster legs perform Verified-mode Merkle proof work per tile
//! and are feasible only in optimized builds; like the e2e parity gate they
//! are ignored under `debug_assertions` and run via
//! `cargo test --release --test golden_traces`.
//!
//! The fp32 golden is gated to macOS: fp32 checkpoint payloads include values
//! produced through platform libm transcendentals (softmax `exp`, …) whose
//! last-ulp behavior differs across platforms, so its bytes are only
//! reproducible on the platform that captured it. The deterministic and
//! raster goldens use the det_num integer numerics and are platform-exact.

// The parity/golden suites intentionally exercise the deprecated legacy
// entry points: they are what proves the shims stay equivalent.
#![allow(deprecated)]

use std::{
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
use raster_inference::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path, run_inference_with_controls,
    AuthenticatedGemmaTokenizer, Gemma4TransformerModel, InferenceControls, InferenceExecutionMode,
    InferenceRequest, InferenceRunOutcome, ModelSpec, RasterDetourSpec, SamplingConfig,
    TextDecodingPolicy,
};
use tokenizers::Tokenizer;

/// Prompts restricted to tokens of the tiny-gemma-dev tokenizer vocabulary so
/// the HuggingFace and raster tokenizer paths agree on the encoding.
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
        Self::load_with_model(
            load_transformer_state_model_from_det_num_wgt_path(&tiny_gemma_dir())
                .expect("tiny-gemma-dev deterministic weights should load"),
        )
    }

    /// Fp32 execution requires the safetensors-backed model, mirroring the
    /// CLI's fp32 model-loading path.
    fn load_fp32() -> Self {
        Self::load_with_model(
            load_transformer_state_model_from_gemma_model_path(&tiny_gemma_dir())
                .expect("tiny-gemma-dev safetensors weights should load"),
        )
    }

    fn load_with_model(transformer_model: Gemma4TransformerModel) -> Self {
        let model_dir = tiny_gemma_dir();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let chat_template = load_chat_template(model_dir.join("chat_template.jinja"))
            .expect("tiny-gemma-dev chat template should load");
        let tokenizer = load_tokenizer_from_path(&tokenizer_path)
            .expect("tiny-gemma-dev tokenizer should load");
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

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

fn request(prompt: &str, mode: InferenceExecutionMode, max_new_tokens: usize) -> InferenceRequest {
    InferenceRequest {
        prompt_bytes: prompt.as_bytes().to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: true,
        add_special_tokens: true,
        execution_mode: mode,
        sampling: SamplingConfig {
            max_new_tokens: Some(max_new_tokens),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    }
}

/// Runs one inference with fresh per-run stores and a fresh trace directory,
/// returning the outcome and the raw bytes of the serialized checkpoint trace
/// artifact — the exact bytes a claimer would commit.
fn run_and_capture_artifact(
    fixture: &Fixture,
    request: &InferenceRequest,
    controls: &InferenceControls,
) -> (InferenceRunOutcome, Vec<u8>) {
    static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
    let run_id = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let trace_dir =
        env::temp_dir().join(format!("raster-golden-traces-{}-{run_id}", process::id()));
    fs::create_dir_all(&trace_dir).expect("trace dir should be created");
    env::set_var("RASTER_TRACE_DIR", &trace_dir);

    ArtifactIo::reset_store();
    reset_external_source_store();

    let outcome = run_inference_with_controls(
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
    let raw_artifact = fs::read(trace_files.remove(0)).expect("trace artifact should be readable");
    fs::remove_dir_all(&trace_dir).expect("trace dir should be removable");

    (outcome, raw_artifact)
}

/// Compares the captured artifact against `tests/goldens/{name}.json`, or
/// rewrites the golden when `UPDATE_GOLDENS=1`.
fn assert_matches_golden(name: &str, raw_artifact: &[u8]) {
    let path = goldens_dir().join(format!("{name}.json"));
    if env::var_os("UPDATE_GOLDENS").is_some_and(|value| value == "1") {
        fs::create_dir_all(goldens_dir()).expect("goldens dir should be created");
        fs::write(&path, raw_artifact).expect("golden file should be writable");
        eprintln!("updated golden {}", path.display());
        return;
    }
    let golden = fs::read(&path).unwrap_or_else(|err| {
        panic!(
            "golden file {} is unreadable ({err}); capture it with \
             UPDATE_GOLDENS=1 cargo test --test golden_traces",
            path.display()
        )
    });
    assert!(
        golden == raw_artifact,
        "trace artifact for `{name}` is not byte-identical to {}; \
         this breaks the refactor's byte-identity contract. If the change is \
         an intentional, separately-reviewed format change, regenerate with \
         UPDATE_GOLDENS=1.\n--- golden ---\n{}\n--- actual ---\n{}",
        path.display(),
        String::from_utf8_lossy(&golden),
        String::from_utf8_lossy(raw_artifact),
    );
}

#[test]
fn golden_native_det_full() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let (outcome, artifact) = run_and_capture_artifact(
        &fixture,
        &request(LONG_PROMPT, InferenceExecutionMode::Deterministic, 2),
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    assert!(matches!(outcome, InferenceRunOutcome::Completed(_)));
    assert_matches_golden("native-det-full", &artifact);
}

#[test]
fn golden_native_det_terminal() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let (outcome, artifact) = run_and_capture_artifact(
        &fixture,
        &request(LONG_PROMPT, InferenceExecutionMode::Deterministic, 2),
        &InferenceControls {
            commit_checkpoints: true,
            terminal_checkpoint: Some("prefill.finalize".to_string()),
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    match outcome {
        InferenceRunOutcome::Paused(paused) => {
            assert_eq!(paused.terminal_checkpoint_id, "prefill.finalize")
        }
        _ => panic!("expected run to pause at prefill.finalize"),
    }
    assert_matches_golden("native-det-terminal", &artifact);
}

#[test]
fn golden_detour_prefill_range() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let (outcome, artifact) = run_and_capture_artifact(
        &fixture,
        &request(SHORT_PROMPT, InferenceExecutionMode::Deterministic, 2),
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range").expect("detour spec should parse"),
            ),
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    assert!(matches!(outcome, InferenceRunOutcome::Completed(_)));
    assert_matches_golden("detour-prefill-range", &artifact);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "full-raster Verified-mode leg requires an optimized build; run with \
              cargo test --release --test golden_traces"
)]
fn golden_raster_full() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let (outcome, artifact) = run_and_capture_artifact(
        &fixture,
        &request(SHORT_PROMPT, InferenceExecutionMode::Deterministic, 1),
        &InferenceControls {
            commit_checkpoints: true,
            raster: true,
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    assert!(matches!(outcome, InferenceRunOutcome::Completed(_)));
    assert_matches_golden("raster-full", &artifact);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "full-raster Verified-mode leg requires an optimized build; run with \
              cargo test --release --test golden_traces"
)]
fn golden_raster_terminal() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let (outcome, artifact) = run_and_capture_artifact(
        &fixture,
        &request(SHORT_PROMPT, InferenceExecutionMode::Deterministic, 2),
        &InferenceControls {
            commit_checkpoints: true,
            terminal_checkpoint: Some("decode.transition_finalize".to_string()),
            raster: true,
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    );
    match outcome {
        InferenceRunOutcome::Paused(paused) => {
            assert_eq!(paused.terminal_checkpoint_id, "decode.transition_finalize")
        }
        _ => panic!("expected run to pause at decode.transition_finalize"),
    }
    assert_matches_golden("raster-terminal", &artifact);
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "fp32 checkpoint payloads depend on platform libm; this golden was \
              captured on macOS (det goldens are platform-exact, fp32 is not)"
)]
fn golden_fp32_native_full() {
    let _guard = suite_lock();
    let fixture = Fixture::load_fp32();
    let (outcome, artifact) = run_and_capture_artifact(
        &fixture,
        &request(LONG_PROMPT, InferenceExecutionMode::Fp32, 2),
        &InferenceControls {
            commit_checkpoints: true,
            ..Default::default()
        },
    );
    assert!(matches!(outcome, InferenceRunOutcome::Completed(_)));
    assert_matches_golden("fp32-native-full", &artifact);
}
