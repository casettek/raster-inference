//! Differential schedule-parity conformance gate (DET_NUM_SPEC "Parallelism
//! legality"): the full deterministic prefill + decode pipeline must produce
//! identical det checkpoint commitments under every execution schedule.
//!
//! Runs the reference model on a fixture covering full + sliding-window
//! attention, a donor (shared-KV) layer, PLE layers, and a 64-token decode,
//! under a 1-thread pool (the serial reference path — parallel drivers fall
//! back to the canonical serial schedule) and 2/4/8-thread pools, then
//! asserts every captured commitment is bit-identical across schedules.
//!
//! The MLP and logits matrices are sized above the GEMV parallel-split
//! threshold so the multi-thread schedules genuinely exercise the
//! output-row-parallel decode drivers.

use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use raster_inference::shared::numerics::det_num::{
    encode_det_wgt_artifact_with_widths, f32_to_wgt, DetWgtTensorSpec, DetWgtWidthPolicy,
};
use raster_inference::{
    decode_step_with_mode, input_embedding, load_transformer_state_model_from_det_num_wgt_path,
    run_prefill_pass_with_mode, Gemma4TransformerModel, InferenceExecutionMode,
    PromptPreparationState,
};

const PREFILL_TOKENS: usize = 64;
const DECODE_STEPS: usize = 64;

const HIDDEN: usize = 16;
const HEAD_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const LAYERS: usize = 4;
/// Above `DET_GEMV_MIN_PAR_ROWS` so MLP gate/up GEMVs split across threads.
const FF: usize = 160;
/// Above `DET_GEMV_MIN_PAR_ROWS` so the logits GEMV splits across threads.
const VOCAB: usize = 160;
const PLE_DIM: usize = 4;
const SLIDING_WINDOW: usize = 8;

/// Per-checkpoint det commitments captured from one full prefill + decode run.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct ScheduleCapture {
    embedding: Option<String>,
    prefill_final_hidden: Option<String>,
    prefill_logits: Option<String>,
    /// One `(activation, logits)` commitment pair per decode step.
    decode_steps: Vec<(Option<String>, Option<String>)>,
}

#[test]
fn det_checkpoint_commitments_are_identical_across_schedules() {
    let model = fixture_model();

    // 1-thread pool: parallel drivers take the serial reference path.
    let baseline = capture_with_threads(&model, 1);
    assert!(
        baseline.embedding.is_some()
            && baseline.prefill_final_hidden.is_some()
            && baseline.prefill_logits.is_some(),
        "deterministic run should produce det commitments"
    );
    assert_eq!(baseline.decode_steps.len(), DECODE_STEPS);
    for (activation, logits) in &baseline.decode_steps {
        assert!(
            activation.is_some() && logits.is_some(),
            "every decode step should produce det commitments"
        );
    }

    for threads in [2_usize, 4, 8] {
        let capture = capture_with_threads(&model, threads);
        assert_eq!(
            capture, baseline,
            "det checkpoint commitments diverged between the serial reference \
             schedule and the {threads}-thread schedule"
        );
    }
}

/// Environment variable that flips this test binary into "child capture"
/// mode for the backend-axis test: the child loads the model from the given
/// directory, captures commitments under whatever kernel backend
/// `RASTER_DET_KERNEL_BACKEND` selects for its process, and prints them as
/// framed JSON. Backend selection is cached once per process, so the scalar
/// leg must run in a separate process.
const CHILD_MODEL_DIR_ENV: &str = "SCHEDULE_PARITY_CHILD_MODEL_DIR";
const CAPTURE_BEGIN: &str = "SCHEDULE_PARITY_CAPTURE_BEGIN";
const CAPTURE_END: &str = "SCHEDULE_PARITY_CAPTURE_END";

/// Backend axis (force-scalar oracle vs auto-detected SIMD) and storage
/// width axis (detwgt v2 auto i16/i32 vs forced all-i32): every combination
/// must produce identical det checkpoint commitments.
#[test]
fn det_checkpoint_commitments_are_identical_across_backends_and_widths() {
    if let Ok(model_dir) = std::env::var(CHILD_MODEL_DIR_ENV) {
        let model = load_transformer_state_model_from_det_num_wgt_path(Path::new(&model_dir))
            .expect("child fixture should load");
        let capture = capture_with_threads(&model, 4);
        println!(
            "{CAPTURE_BEGIN}{}{CAPTURE_END}",
            serde_json::to_string(&capture).expect("capture should serialize")
        );
        return;
    }

    let (auto_dir, auto_model) = fixture_model_with_policy(DetWgtWidthPolicy::Auto);
    let (i32_dir, i32_model) = fixture_model_with_policy(DetWgtWidthPolicy::ForceI32);

    // The fixture must actually exercise the i16 path: auto-width storage
    // narrows the (sub-0.5 magnitude) matrix tensors, so its artifact is
    // strictly smaller than the forced all-i32 one.
    let auto_len = fs::metadata(auto_dir.join("model.detwgt")).unwrap().len();
    let i32_len = fs::metadata(i32_dir.join("model.detwgt")).unwrap().len();
    assert!(
        auto_len < i32_len,
        "auto-width fixture ({auto_len} bytes) should be smaller than forced-i32 ({i32_len} bytes)"
    );

    // Width axis, in-process (auto-detected backend).
    let auto_capture = capture_with_threads(&auto_model, 4);
    let i32_capture = capture_with_threads(&i32_model, 4);
    assert_eq!(
        auto_capture, i32_capture,
        "det checkpoint commitments diverged between i16/i32 auto-width and all-i32 storage"
    );

    // Backend axis, via subprocesses with the backend pinned per process.
    let scalar_capture = capture_in_subprocess(&auto_dir, "scalar");
    assert_eq!(
        scalar_capture, auto_capture,
        "det checkpoint commitments diverged between the force-scalar oracle and the \
         auto-detected kernel backend"
    );
}

fn capture_in_subprocess(model_dir: &Path, backend: &str) -> ScheduleCapture {
    let exe = std::env::current_exe().expect("test executable path should resolve");
    let output = std::process::Command::new(exe)
        .args([
            "--exact",
            "det_checkpoint_commitments_are_identical_across_backends_and_widths",
            "--nocapture",
        ])
        .env(CHILD_MODEL_DIR_ENV, model_dir)
        .env("RASTER_DET_KERNEL_BACKEND", backend)
        .output()
        .expect("child capture process should spawn");
    assert!(
        output.status.success(),
        "child capture process failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("child stdout should be UTF-8");
    let begin = stdout
        .find(CAPTURE_BEGIN)
        .expect("child output should contain the capture marker")
        + CAPTURE_BEGIN.len();
    let end = stdout[begin..]
        .find(CAPTURE_END)
        .expect("child output should terminate the capture marker")
        + begin;
    serde_json::from_str(&stdout[begin..end]).expect("child capture should deserialize")
}

fn capture_with_threads(model: &Gemma4TransformerModel, threads: usize) -> ScheduleCapture {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("scoped rayon pool should build");
    pool.install(|| capture_schedule(model))
}

fn capture_schedule(model: &Gemma4TransformerModel) -> ScheduleCapture {
    let prompt_token_ids = token_ids(PREFILL_TOKENS);
    let token_embeddings = input_embedding::run(
        &prompt_token_ids,
        model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("input embedding should succeed");
    let prefill = run_prefill_pass_with_mode(
        &prompt_preparation(&prompt_token_ids),
        model,
        &token_embeddings,
        InferenceExecutionMode::Deterministic,
    )
    .expect("prefill should succeed");

    let embedding = token_embeddings.det_activations_sha256.clone();
    let prefill_final_hidden = prefill.transformer_state.activation_states[0]
        .det_activations_sha256
        .clone();
    let prefill_logits = prefill
        .transformer_state
        .prefill_logits
        .det_final_logits_sha256
        .clone();

    // Fixed token feed so every schedule decodes the identical sequence; any
    // numeric divergence still surfaces in the per-step commitments.
    let mut decode_state = prefill.transformer_decode_state;
    let mut decode_steps = Vec::with_capacity(DECODE_STEPS);
    for step_idx in 0..DECODE_STEPS {
        let next_token = ((step_idx * 11 + 3) % VOCAB) as u32;
        let step = decode_step_with_mode(
            decode_state,
            next_token,
            model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("decode step should succeed");
        decode_steps.push((
            step.activation_state.det_activations_sha256.clone(),
            step.prefill_logits.det_final_logits_sha256.clone(),
        ));
        decode_state = step.transformer_decode_state;
    }

    ScheduleCapture {
        embedding,
        prefill_final_hidden,
        prefill_logits,
        decode_steps,
    }
}

fn token_ids(len: usize) -> Vec<u32> {
    (0..len).map(|idx| ((idx * 7 + 1) % VOCAB) as u32).collect()
}

fn prompt_preparation(prompt_token_ids: &[u32]) -> PromptPreparationState {
    PromptPreparationState {
        prompt_text: "schedule-parity".to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        prompt_token_ids_sha256: "schedule-parity".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Fixture model (donor + PLE + sliding/full attention)
// ---------------------------------------------------------------------------

struct FixtureRng(u64);

impl FixtureRng {
    fn next_weight(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bucket = ((self.0 >> 33) % 33) as i64 - 16;
        bucket as f32 / 256.0
    }

    fn weights(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_weight()).collect()
    }

    fn norm_weights(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| 1.0 + self.next_weight() / 4.0).collect()
    }
}

struct FixtureTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn fixture_model() -> Gemma4TransformerModel {
    fixture_model_with_policy(DetWgtWidthPolicy::Auto).1
}

fn fixture_model_with_policy(
    policy: DetWgtWidthPolicy,
) -> (std::path::PathBuf, Gemma4TransformerModel) {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should advance")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "raster-inference-schedule-parity-{unique}-{policy:?}"
    ));
    fs::create_dir_all(&dir).expect("fixture dir should create");

    let config = format!(
        r#"{{
  "text_config": {{
    "enable_moe_block": false,
    "head_dim": {HEAD_DIM},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN},
    "hidden_size_per_layer_input": {PLE_DIM},
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    "num_attention_heads": {NUM_HEADS},
    "num_hidden_layers": {LAYERS},
    "num_key_value_heads": {NUM_KV_HEADS},
    "num_kv_shared_layers": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": {SLIDING_WINDOW},
    "tie_word_embeddings": false,
    "vocab_size": {VOCAB},
    "vocab_size_per_layer_input": {VOCAB}
  }}
}}"#
    );
    fs::write(dir.join("config.json"), config).expect("config should write");

    let tensors = fixture_tensors();
    write_detwgt_file_with_policy(&dir.join("model.detwgt"), &tensors, policy);
    let model =
        load_transformer_state_model_from_det_num_wgt_path(&dir).expect("fixture should load");
    (dir, model)
}

fn fixture_tensors() -> Vec<FixtureTensor> {
    let mut rng = FixtureRng(0x00a5_c4ed_0a5c_4ed1);
    let mut tensors = Vec::new();
    let push =
        |tensors: &mut Vec<FixtureTensor>, name: String, shape: &[usize], values: Vec<f32>| {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            tensors.push(FixtureTensor {
                name,
                shape: shape.to_vec(),
                values,
            });
        };

    push(
        &mut tensors,
        "model.language_model.embed_tokens.weight".to_string(),
        &[VOCAB, HIDDEN],
        rng.weights(VOCAB * HIDDEN),
    );
    for layer_idx in 0..LAYERS {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_proj.weight"),
            &[NUM_HEADS * HEAD_DIM, HIDDEN],
            rng.weights(NUM_HEADS * HEAD_DIM * HIDDEN),
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN],
            rng.weights(NUM_KV_HEADS * HEAD_DIM * HIDDEN),
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.v_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN],
            rng.weights(NUM_KV_HEADS * HEAD_DIM * HIDDEN),
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN, NUM_HEADS * HEAD_DIM],
            rng.weights(HIDDEN * NUM_HEADS * HEAD_DIM),
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
            rng.norm_weights(HEAD_DIM),
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
            rng.norm_weights(HEAD_DIM),
        );
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            push(
                &mut tensors,
                format!("{prefix}.{norm}.weight"),
                &[HIDDEN],
                rng.norm_weights(HIDDEN),
            );
        }
        push(
            &mut tensors,
            format!("{prefix}.mlp.gate_proj.weight"),
            &[FF, HIDDEN],
            rng.weights(FF * HIDDEN),
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.up_proj.weight"),
            &[FF, HIDDEN],
            rng.weights(FF * HIDDEN),
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN, FF],
            rng.weights(HIDDEN * FF),
        );
        push(
            &mut tensors,
            format!("{prefix}.per_layer_input_gate.weight"),
            &[PLE_DIM, HIDDEN],
            rng.weights(PLE_DIM * HIDDEN),
        );
        push(
            &mut tensors,
            format!("{prefix}.per_layer_projection.weight"),
            &[HIDDEN, PLE_DIM],
            rng.weights(HIDDEN * PLE_DIM),
        );
        push(
            &mut tensors,
            format!("{prefix}.post_per_layer_input_norm.weight"),
            &[HIDDEN],
            rng.norm_weights(HIDDEN),
        );
    }
    push(
        &mut tensors,
        "model.language_model.embed_tokens_per_layer.weight".to_string(),
        &[VOCAB, LAYERS * PLE_DIM],
        rng.weights(VOCAB * LAYERS * PLE_DIM),
    );
    push(
        &mut tensors,
        "model.language_model.per_layer_model_projection.weight".to_string(),
        &[LAYERS * PLE_DIM, HIDDEN],
        rng.weights(LAYERS * PLE_DIM * HIDDEN),
    );
    push(
        &mut tensors,
        "model.language_model.per_layer_projection_norm.weight".to_string(),
        &[PLE_DIM],
        rng.norm_weights(PLE_DIM),
    );
    push(
        &mut tensors,
        "model.language_model.norm.weight".to_string(),
        &[HIDDEN],
        rng.norm_weights(HIDDEN),
    );
    push(
        &mut tensors,
        "model.language_model.lm_head.weight".to_string(),
        &[VOCAB, HIDDEN],
        rng.weights(VOCAB * HIDDEN),
    );
    tensors
}

fn write_detwgt_file_with_policy(
    path: &Path,
    tensors: &[FixtureTensor],
    policy: DetWgtWidthPolicy,
) {
    let specs = tensors
        .iter()
        .map(|tensor| DetWgtTensorSpec {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            wgt_bits: tensor
                .values
                .iter()
                .map(|value| f32_to_wgt(*value).to_bits())
                .collect(),
        })
        .collect::<Vec<_>>();
    let bytes =
        encode_det_wgt_artifact_with_widths(&specs, policy).expect("detwgt v2 should encode");
    fs::write(path, bytes).expect("detwgt should write");
}
