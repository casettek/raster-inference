//! Raster-core detour parity harness (WS4 scaffolding).
//!
//! Mirrors the detour-smoke leg of the main parity gate
//! (`tests/e2e_checkpoint_parity.rs`) for the raster-core (real toolchain)
//! backend: for every enabled `routine-id:occurrence`, a
//! `--raster-core-at`-equivalent run must commit a checkpoint trace identical
//! to the full-native run of the same request, with any divergence named by
//! checkpoint id + occurrence.
//!
//! **WS0 ships this harness with zero enabled routines.** WS4 flips routines
//! on one at a time by appending to [`ENABLED_ROUTINES`] in the same change
//! that lands the routine's WS3 migration. Enablement is deliberately an
//! explicit const list — no discovery magic — so a reviewer can see exactly
//! which legs the gate enforces.
//!
//! The identity assertion is strict (no allowed-divergence labels): committed
//! checkpoints are backend-invariant. If WS5 rules that a boundary
//! checkpoint's raster-core payload legitimately takes artifact-root form,
//! the affected label is added to the routine's entry alongside that ruling —
//! never silently.
//!
//! Hermetic like the main gate: runs against `assets/tiny-gemma-dev` in the
//! default Verified integrity mode, capturing the same serialized trace
//! artifact a claimer would commit. Existing goldens in `tests/goldens/` are
//! the baseline and are never regenerated.

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
    load_transformer_state_model_from_det_num_wgt_path, sequence, InferenceControls,
    InferenceRequest, InferenceRunOutcome, ModelSpec, RasterDetourSpec, SamplingConfig,
    TextDecodingPolicy,
};
use serde_json::Value;
use tokenizers::Tokenizer;

/// Routine occurrences with raster-core parity enabled, as
/// `(routine-id, occurrence)` — the same selector grammar as
/// `detour --raster-core-at <routine-id[:occurrence]>`.
///
/// WS4 appends one entry per migrated routine (e.g.
/// `("prompt.prepare", 1)`); WS0 ships the list empty.
const ENABLED_ROUTINES: &[(&str, usize)] = &[];

/// Prompt restricted to tokens of the tiny-gemma-dev tokenizer vocabulary
/// (same constraint as the main parity gate).
const SHORT_PROMPT: &str = "hello";

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

/// Runs one inference with fresh per-run stores and a fresh trace directory,
/// returning the parsed committed checkpoint trace (an array of single-key
/// `{checkpoint_id: sha256-hex}` objects in commit order).
///
/// The caller must hold [`suite_lock`] for the duration of all runs it
/// intends to compare.
fn run_and_capture_checkpoints(
    fixture: &Fixture,
    request: &InferenceRequest,
    controls: &InferenceControls,
    description: &str,
) -> Value {
    static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
    let run_id = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let trace_dir = env::temp_dir().join(format!(
        "raster-core-detour-parity-{}-{run_id}",
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
    checkpoints
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

/// Asserts the two committed checkpoint traces are identical — same ordered
/// checkpoint id sequence, same commitment at every entry. On divergence,
/// panics naming the first divergent checkpoint id and 1-based occurrence
/// (e.g. `prefill.range:2`).
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

/// For every enabled routine occurrence: a raster-core detour run of that
/// occurrence must commit a checkpoint trace identical to the full-native run
/// of the same request. Enabled legs are listed in [`ENABLED_ROUTINES`];
/// WS0 ships zero.
#[test]
fn raster_core_detours_preserve_the_native_checkpoint_trace() {
    let _guard = suite_lock();
    if ENABLED_ROUTINES.is_empty() {
        // WS0: no routine has a raster-core implementation yet. The harness
        // still compiles and runs so WS4 only has to append list entries.
        return;
    }
    let fixture = Fixture::load();
    let request = deterministic_request(SHORT_PROMPT, 2);

    let native = run_and_capture_checkpoints(
        &fixture,
        &request,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..Default::default()
        },
        "native deterministic run",
    );

    for (routine_id, occurrence) in ENABLED_ROUTINES {
        let selector = format!("{routine_id}:{occurrence}");
        let spec = RasterDetourSpec::parse_raster_core(&selector)
            .expect("enabled raster-core selector should parse");
        let detour = run_and_capture_checkpoints(
            &fixture,
            &request,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(spec),
                raster_tokenizer_enabled: true,
                ..Default::default()
            },
            &format!("raster-core detour run at {selector}"),
        );
        assert_traces_identical(
            "native",
            &native,
            &format!("raster-core detour at {selector}"),
            &detour,
        );
    }
}
