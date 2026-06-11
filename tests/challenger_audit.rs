//! Challenger/claimer role tests.
//!
//! Validates the role layer against the protocol contract:
//!
//! 1. `claimer::run` produces a trace artifact byte-identical to the legacy
//!    native `--commit-checkpoints` path (the checked-in golden).
//! 2. `challenger::audit` against an honest claimer trace reports no
//!    divergence.
//! 3. Against tampered fixtures (one commitment altered in an honest trace),
//!    the challenger identifies exactly the tampered checkpoint id and
//!    occurrence and produces a raster detour trace byte-identical to the
//!    equivalent `--raster-at` run.
//! 4. Structural divergences (truncated trace, detour-unsupported routines)
//!    are reported without a detour artifact.
//!
//! The trace collector and artifact stores are process-global, so every test
//! serializes through `suite_lock`.

use std::{
    env, fs,
    path::PathBuf,
    process,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, MutexGuard,
    },
};

use raster_inference::runtime::roles::{challenger, claimer};
use raster_inference::shared::api::audit::AuditOutcome;
use raster_inference::shared::artifacts::artifact_io::ArtifactIo;
use raster_inference::shared::artifacts::external_artifacts::reset_external_source_store;
use raster_inference::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path, run_inference_with_controls,
    AuthenticatedGemmaTokenizer, Gemma4TransformerModel, InferenceControls, InferenceExecutionMode,
    InferenceRequest, InferenceRunOutcome, ModelSpec, RasterDetourSpec, SamplingConfig,
    TextDecodingPolicy,
};
use serde_json::Value;
use tokenizers::Tokenizer;

const SHORT_PROMPT: &str = "hello";

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

/// Points `RASTER_TRACE_DIR` at a fresh per-test temp dir and resets the
/// process-global artifact stores. Returns the trace dir.
fn fresh_run_env() -> PathBuf {
    static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
    let run_id = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let trace_dir = env::temp_dir().join(format!(
        "raster-challenger-audit-{}-{run_id}",
        process::id()
    ));
    fs::create_dir_all(&trace_dir).expect("trace dir should be created");
    env::set_var("RASTER_TRACE_DIR", &trace_dir);
    ArtifactIo::reset_store();
    reset_external_source_store();
    trace_dir
}

fn run_claimer(fixture: &Fixture, request: &InferenceRequest) -> Vec<u8> {
    fresh_run_env();
    let outcome = claimer::run(
        request,
        &fixture.model_spec,
        &fixture.tokenizer,
        &fixture.transformer_model,
        fixture.raster_tokenizer_source(),
    )
    .expect("claimer run should complete");
    fs::read(&outcome.trace_path).expect("claimer trace artifact should be readable")
}

fn run_audit(fixture: &Fixture, request: &InferenceRequest, claimed_trace: &[u8]) -> AuditOutcome {
    fresh_run_env();
    challenger::audit(
        request,
        &fixture.model_spec,
        &fixture.tokenizer,
        &fixture.transformer_model,
        fixture.raster_tokenizer_source(),
        claimed_trace,
    )
    .expect("challenger audit should complete")
}

/// Runs the legacy `--raster-at` equivalent (detour controls through the
/// legacy entry point) and returns the serialized trace artifact bytes.
fn run_reference_detour(fixture: &Fixture, request: &InferenceRequest, spec: &str) -> Vec<u8> {
    let trace_dir = fresh_run_env();
    let outcome = run_inference_with_controls(
        request,
        &fixture.model_spec,
        &fixture.tokenizer,
        &fixture.transformer_model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(RasterDetourSpec::parse(spec).expect("detour spec should parse")),
            raster_tokenizer_source: Some(fixture.raster_tokenizer_source()),
            ..Default::default()
        },
    )
    .expect("reference detour run should complete");
    assert!(
        matches!(outcome, InferenceRunOutcome::Completed(_)),
        "reference detour run should run to completion"
    );
    let mut trace_files = fs::read_dir(&trace_dir)
        .expect("trace dir should be readable")
        .map(|entry| entry.expect("trace dir entry should read").path())
        .collect::<Vec<_>>();
    assert_eq!(trace_files.len(), 1, "expected exactly one trace artifact");
    fs::read(trace_files.remove(0)).expect("reference detour trace should be readable")
}

/// Replaces the commitment of the `occurrence`-th entry named
/// `checkpoint_id` in a serialized trace artifact with a bogus value.
fn tamper_commitment(trace: &[u8], checkpoint_id: &str, occurrence: usize) -> Vec<u8> {
    let mut payload: Value =
        serde_json::from_slice(trace).expect("trace artifact should be valid JSON");
    let entries = payload
        .as_array_mut()
        .expect("trace artifact should be a JSON array");
    let mut seen = 0;
    for entry in entries.iter_mut() {
        let object = entry
            .as_object_mut()
            .expect("trace entry should be an object");
        if object.contains_key(checkpoint_id) {
            seen += 1;
            if seen == occurrence {
                object.insert(checkpoint_id.to_string(), Value::String("0".repeat(64)));
                return serde_json::to_vec_pretty(&payload)
                    .expect("tampered trace should serialize");
            }
        }
    }
    panic!("trace has no occurrence {occurrence} of `{checkpoint_id}`");
}

/// End-to-end tamper case: tampering `checkpoint_id:occurrence` in an honest
/// trace must be located exactly, and the resulting raster detour trace must
/// be byte-identical to the equivalent `--raster-at` run.
fn assert_tamper_located_and_detoured(
    fixture: &Fixture,
    request: &InferenceRequest,
    honest_trace: &[u8],
    checkpoint_id: &str,
    occurrence: usize,
) {
    let tampered = tamper_commitment(honest_trace, checkpoint_id, occurrence);
    let outcome = run_audit(fixture, request, &tampered);
    let AuditOutcome::Diverged { divergence, detour } = outcome else {
        panic!("tampering `{checkpoint_id}:{occurrence}` should diverge, got {outcome:?}");
    };
    assert_eq!(
        (divergence.checkpoint_id.as_str(), divergence.occurrence),
        (checkpoint_id, occurrence),
        "audit should locate exactly the tampered checkpoint"
    );
    assert!(
        divergence.claimed_checkpoint_id.is_none(),
        "a commitment tamper is not an id-sequence divergence"
    );
    let detour = detour.unwrap_or_else(|| {
        panic!("divergence at `{checkpoint_id}:{occurrence}` should produce a detour artifact")
    });
    let spec_label = if occurrence == 1 {
        checkpoint_id.to_string()
    } else {
        format!("{checkpoint_id}:{occurrence}")
    };
    assert_eq!(detour.spec.to_string(), spec_label);
    let detour_trace =
        fs::read(&detour.trace_path).expect("detour trace artifact should be readable");
    let reference_trace = run_reference_detour(fixture, request, &spec_label);
    assert_eq!(
        detour_trace, reference_trace,
        "challenger detour trace for `{spec_label}` should be byte-identical to the \
         equivalent --raster-at run"
    );
}

/// Acceptance: `claimer::run` reproduces the legacy native
/// `--commit-checkpoints` trace byte-for-byte (the refactor golden).
#[test]
fn claimer_trace_matches_legacy_golden() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    // Same request as the `native-det-full` golden configuration.
    let request = deterministic_request("hello raster prompt", 2);
    let claimer_trace = run_claimer(&fixture, &request);
    let golden = fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/native-det-full.json"),
    )
    .expect("native-det-full golden should be readable");
    assert_eq!(
        claimer_trace, golden,
        "claimer trace artifact should be byte-identical to the legacy native golden"
    );
}

#[test]
fn audit_of_honest_trace_reports_no_divergence() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    let outcome = run_audit(&fixture, &request, &honest);
    assert_eq!(outcome, AuditOutcome::NoDivergence);
}

#[test]
fn audit_locates_tampered_prefill_range_first_occurrence() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    assert_tamper_located_and_detoured(&fixture, &request, &honest, "prefill.range", 1);
}

/// Occurrence alignment: the checkpoint occurrence in the trace must map to
/// the same routine occurrence the detour controller counts.
#[test]
fn audit_locates_tampered_prefill_range_second_occurrence() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    assert_tamper_located_and_detoured(&fixture, &request, &honest, "prefill.range", 2);
}

#[test]
fn audit_locates_tampered_input_embedding() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    assert_tamper_located_and_detoured(&fixture, &request, &honest, "input.embedding", 1);
}

#[test]
fn audit_locates_tampered_prefill_finalize() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    assert_tamper_located_and_detoured(&fixture, &request, &honest, "prefill.finalize", 1);
}

#[test]
fn audit_locates_tampered_decode_select_token() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    assert_tamper_located_and_detoured(&fixture, &request, &honest, "decode.select_token", 1);
}

#[test]
fn audit_locates_tampered_decode_transition_finalize() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    assert_tamper_located_and_detoured(
        &fixture,
        &request,
        &honest,
        "decode.transition_finalize",
        1,
    );
}

#[test]
fn audit_locates_tampered_output_finalize() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    assert_tamper_located_and_detoured(&fixture, &request, &honest, "output.finalize", 1);
}

/// `prompt.prepare` has no implemented detour; the divergence must still be
/// located exactly but without a detour artifact.
#[test]
fn audit_reports_tampered_prompt_prepare_without_detour() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    let tampered = tamper_commitment(&honest, "prompt.prepare", 1);
    let outcome = run_audit(&fixture, &request, &tampered);
    let AuditOutcome::Diverged { divergence, detour } = outcome else {
        panic!("tampered prompt.prepare should diverge, got {outcome:?}");
    };
    assert_eq!(
        (divergence.checkpoint_id.as_str(), divergence.occurrence),
        ("prompt.prepare", 1)
    );
    assert!(
        detour.is_none(),
        "prompt.prepare does not support a raster detour"
    );
}

/// A truncated claimed trace is a structural (length) divergence: located at
/// the first unmatched entry, no detour.
#[test]
fn audit_reports_truncated_trace_as_structural_divergence() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);
    let honest = run_claimer(&fixture, &request);
    let mut payload: Value =
        serde_json::from_slice(&honest).expect("trace artifact should be valid JSON");
    let entries = payload
        .as_array_mut()
        .expect("trace artifact should be a JSON array");
    let removed = entries.pop().expect("trace should have entries");
    let removed_id = removed
        .as_object()
        .and_then(|object| object.keys().next().cloned())
        .expect("removed entry should have a checkpoint id");
    let truncated_len = entries.len();
    let truncated = serde_json::to_vec_pretty(&payload).expect("truncated trace should serialize");

    let outcome = run_audit(&fixture, &request, &truncated);
    let AuditOutcome::Diverged { divergence, detour } = outcome else {
        panic!("truncated trace should diverge, got {outcome:?}");
    };
    assert_eq!(divergence.entry_index, truncated_len);
    assert_eq!(divergence.checkpoint_id, removed_id);
    assert!(divergence.claimed_commitment.is_none());
    assert!(divergence.replayed_commitment.is_some());
    assert!(detour.is_none(), "length divergence has no detour");
}
