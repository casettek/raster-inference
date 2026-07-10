//! WS3 dev-run verification for the `input.embedding` raster-core port.
//!
//! Runs the same hermetic tiny-gemma-dev request natively and with
//! `--raster-core-at input.embedding:1`. The committed checkpoint traces must
//! be identical, and final inference outcomes must match apart from execution
//! count metadata.

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
use raster_inference::shared::model::gemma::adapter::GemmaModelBundle;
use raster_inference::shared::model::gemma::tokenizer::AuthenticatedGemmaTokenizer;
use raster_inference::shared::model::runtime::LoadedModel;
use raster_inference::shared::model::transformer::Gemma4TransformerModel;
use raster_inference::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path, sequence, warm_model_externals,
    InferenceControls, InferenceRequest, InferenceRunOutcome, ModelSpec, RasterDetourSpec,
    SamplingConfig, TextDecodingPolicy, EXTERNAL_CACHE_ENV,
};
use serde_json::Value;
use tokenizers::Tokenizer;

const DEV_RUN_SELECTOR: &str = "input.embedding:1";
const SHORT_PROMPT: &str = "hello";

fn require_or_skip() -> bool {
    let available = process::Command::new("cargo-raster")
        .arg("--version")
        .output()
        .is_ok();
    if available {
        return true;
    }
    if env::var_os("REQUIRE_CARGO_RASTER").is_some_and(|v| v == "1") {
        panic!(
            "REQUIRE_CARGO_RASTER=1 but cargo-raster is not on PATH; install it from the \
             pinned raster checkout (cargo install --path ../raster/crates/raster-cli)"
        );
    }
    eprintln!(
        "SKIPPED: raster_core_input_embedding_detour requires the cargo-raster CLI on PATH. \
         CI runs this test with REQUIRE_CARGO_RASTER=1."
    );
    false
}

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
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/tiny-gemma-dev");
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

    fn loaded_model(&self) -> LoadedModel {
        LoadedModel::Gemma(GemmaModelBundle::new(
            self.model_spec.clone(),
            self.tokenizer.clone(),
            self.transformer_model.clone(),
            Some(AuthenticatedGemmaTokenizer::new(
                load_gemma_tokenizer_spec_from_path(&self.model_spec.tokenizer_path)
                    .expect("tiny-gemma-dev tokenizer spec should load"),
            )),
        ))
    }
}

/// Points raster-core runs at a test-local pre-encoded externals directory
/// and warms it (mandatory pre-encode contract: model externals are strict
/// lookup-only at run time).
fn warm_externals(fixture: &Fixture) -> PathBuf {
    let externals_dir = env::temp_dir().join(format!(
        "raster-core-input-embedding-externals-{}",
        process::id()
    ));
    env::set_var(EXTERNAL_CACHE_ENV, &externals_dir);
    warm_model_externals(&fixture.loaded_model(), &externals_dir)
        .expect("model externals warm-up should succeed");
    externals_dir
}

fn deterministic_request(prompt: &str, max_new_tokens: usize) -> InferenceRequest {
    InferenceRequest {
        prompt_bytes: prompt.as_bytes().to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: true,
        add_special_tokens: true,
        sampling: SamplingConfig {
            max_new_tokens: Some(max_new_tokens),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    }
}

fn run_and_capture(
    fixture: &Fixture,
    request: &InferenceRequest,
    controls: &InferenceControls,
    description: &str,
) -> (Value, InferenceRunOutcome) {
    static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
    let run_id = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let trace_dir = env::temp_dir().join(format!(
        "raster-core-input-embedding-dev-run-{}-{run_id}",
        process::id()
    ));
    fs::create_dir_all(&trace_dir).expect("trace dir should be created");
    env::set_var("RASTER_TRACE_DIR", &trace_dir);

    ArtifactIo::reset_store();
    reset_external_source_store();

    let outcome = sequence::run(request, &fixture.loaded_model(), controls)
        .unwrap_or_else(|error| panic!("{description} should complete: {error:#}"));
    assert!(
        matches!(outcome, InferenceRunOutcome::Completed(_)),
        "expected {description} to run to completion"
    );

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

    let checkpoints: Value =
        serde_json::from_slice(&raw_artifact).expect("trace artifact should be valid JSON");
    assert!(
        checkpoints
            .as_array()
            .is_some_and(|array| !array.is_empty()),
        "trace artifact should be a non-empty checkpoint array"
    );
    (checkpoints, outcome)
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

fn assert_traces_identical(left_label: &str, left: &Value, right_label: &str, right: &Value) {
    let left_entries = left
        .as_array()
        .expect("left checkpoint payload should be an array");
    let right_entries = right
        .as_array()
        .expect("right checkpoint payload should be an array");

    let mut occurrences: HashMap<String, usize> = HashMap::new();
    let common_len = left_entries.len().min(right_entries.len());
    for idx in 0..common_len {
        let (left_checkpoint, left_commitment) =
            checkpoint_entry_name_and_commitment(&left_entries[idx], idx);
        let (right_checkpoint, right_commitment) =
            checkpoint_entry_name_and_commitment(&right_entries[idx], idx);
        assert_eq!(
            left_checkpoint, right_checkpoint,
            "checkpoint ID sequence diverges at entry {idx}: {left_label} committed \
             `{left_checkpoint}` but {right_label} committed `{right_checkpoint}`"
        );
        let occurrence = occurrences
            .entry(left_checkpoint.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        assert_eq!(
            left_commitment, right_commitment,
            "first divergent checkpoint: `{left_checkpoint}:{occurrence}` (entry {idx}): \
             {left_label} committed {left_commitment} but {right_label} committed \
             {right_commitment}"
        );
    }
    assert_eq!(
        left_entries.len(),
        right_entries.len(),
        "checkpoint counts diverge: {left_label} committed {} checkpoints but {right_label} \
         committed {}",
        left_entries.len(),
        right_entries.len(),
    );
}

#[test]
fn raster_core_input_embedding_detour_preserves_checkpoint_trace() {
    if !require_or_skip() {
        return;
    }
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let _externals_dir = warm_externals(&fixture);
    let request = deterministic_request(SHORT_PROMPT, 2);
    let (native_trace, native_outcome) = run_and_capture(
        &fixture,
        &request,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..Default::default()
        },
        "native deterministic run",
    );

    let spec = RasterDetourSpec::parse_raster_core(DEV_RUN_SELECTOR)
        .expect("dev-run selector should parse");
    let (detour_trace, detour_outcome) = run_and_capture(
        &fixture,
        &request,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(spec),
            raster_tokenizer_enabled: true,
            ..Default::default()
        },
        &format!("raster-core detour run at {DEV_RUN_SELECTOR}"),
    );

    assert_traces_identical(
        "native",
        &native_trace,
        &format!("raster-core detour at {DEV_RUN_SELECTOR}"),
        &detour_trace,
    );

    let (
        InferenceRunOutcome::Completed(mut native_state),
        InferenceRunOutcome::Completed(mut detour_state),
    ) = (native_outcome, detour_outcome)
    else {
        unreachable!("both runs were asserted completed");
    };
    native_state.raster_tile_invocations = None;
    detour_state.raster_tile_invocations = None;
    assert_eq!(
        native_state, detour_state,
        "native and raster-core detour final inference states must be equal"
    );
}

/// Runs one raster-core detour expected to fail before staging, returning
/// the rendered error chain.
fn run_detour_expect_error(fixture: &Fixture, description: &str) -> String {
    ArtifactIo::reset_store();
    reset_external_source_store();
    let spec = RasterDetourSpec::parse_raster_core(DEV_RUN_SELECTOR)
        .expect("dev-run selector should parse");
    let error = sequence::run(
        &deterministic_request(SHORT_PROMPT, 2),
        &fixture.loaded_model(),
        &InferenceControls {
            raster_detour: Some(spec),
            raster_tokenizer_enabled: true,
            ..Default::default()
        },
    )
    .expect_err(description);
    format!("{error:#}")
}

/// Strict-resolution contract: a raster-core detour with the externals
/// variable unset fails up front with the remediation guidance (no lazy
/// encode, no temp-dir fallback). Needs no toolchain — the failure happens
/// before anything is staged or run.
#[test]
fn raster_core_detour_without_externals_dir_fails_with_guidance() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let saved = env::var_os(EXTERNAL_CACHE_ENV);
    env::remove_var(EXTERNAL_CACHE_ENV);

    let message = run_detour_expect_error(
        &fixture,
        "raster-core detour without an externals directory must fail",
    );
    if let Some(saved) = saved {
        env::set_var(EXTERNAL_CACHE_ENV, saved);
    }
    assert!(
        message.contains(EXTERNAL_CACHE_ENV) && message.contains("--externals-dir"),
        "error should name the externals-directory knobs: {message}"
    );
    assert!(
        message.contains("encode-externals"),
        "error should carry the encode-externals remediation command: {message}"
    );
}

/// Strict-lookup contract: a raster-core detour against an empty externals
/// directory fails naming the missing kind, the directory, and the
/// encode-externals command.
#[test]
fn raster_core_detour_with_empty_externals_dir_names_the_missing_kind() {
    let _guard = suite_lock();
    let fixture = Fixture::load();
    let saved = env::var_os(EXTERNAL_CACHE_ENV);
    let empty_dir = env::temp_dir().join(format!(
        "raster-core-input-embedding-empty-externals-{}",
        process::id()
    ));
    fs::create_dir_all(&empty_dir).expect("empty externals dir should be created");
    env::set_var(EXTERNAL_CACHE_ENV, &empty_dir);

    let message = run_detour_expect_error(
        &fixture,
        "raster-core detour against an empty externals directory must fail",
    );
    match saved {
        Some(saved) => env::set_var(EXTERNAL_CACHE_ENV, saved),
        None => env::remove_var(EXTERNAL_CACHE_ENV),
    }
    fs::remove_dir_all(&empty_dir).ok();

    // The detour targets input.embedding (prompt.prepare runs natively),
    // so the miss is the embedding kind.
    assert!(
        message.contains("gemma-input-embedding-v3"),
        "error should name the missing external kind: {message}"
    );
    assert!(
        message.contains(&empty_dir.display().to_string()),
        "error should name the resolved externals directory: {message}"
    );
    assert!(
        message.contains("encode-externals"),
        "error should carry the encode-externals remediation command: {message}"
    );
}
