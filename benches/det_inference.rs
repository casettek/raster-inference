//! Deterministic-mode inference benchmarks.
//!
//! Measures prefill tokens/sec at seq lengths {128, 512, 2048} and decode
//! tokens/sec at context lengths {128, 1024}, single-threaded (rayon pinned to
//! one thread), on a synthetic detwgt fixture model.
//!
//! Set `RASTER_BENCH_MODEL_DIR` to point at a real detwgt model directory to
//! benchmark against it instead of the synthetic fixture.
//!
//! Backend / width axes:
//! - `RASTER_DET_KERNEL_BACKEND=scalar` forces the scalar reference kernels
//!   (default: auto-detected SIMD).
//! - `RASTER_BENCH_WGT_WIDTH=i32` forces the synthetic fixture to all-i32
//!   weight storage (default: detwgt v2 auto width, which stores the
//!   matrix tensors as i16).
//!
//! Run: `cargo bench --bench det_inference`
//! Record results in `benches/RESULTS.md`.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use raster_inference::shared::numerics::det_num::{
    encode_det_wgt_artifact_with_widths, f32_to_wgt, DetWgtTensorSpec, DetWgtWidthPolicy,
};
use raster_inference::routines::input_embedding;
use raster_inference::runtime::pipeline::{decode_step_with_mode, run_prefill_pass_with_mode};
use raster_inference::shared::model::transformer::{Gemma4TransformerModel, TransformerDecodeState};
use raster_inference::{
    load_transformer_state_model_from_det_num_wgt_path, InferenceExecutionMode,
    PromptPreparationState,
};

const PREFILL_SEQ_LENS: &[usize] = &[128, 512, 2048];
const DECODE_CONTEXT_LENS: &[usize] = &[128, 1024];

const HIDDEN: usize = 64;
const HEAD_DIM: usize = 16;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 1;
const LAYERS: usize = 2;
const FF: usize = 128;
const VOCAB: usize = 64;
const SLIDING_WINDOW: usize = 128;

fn bench_model() -> Gemma4TransformerModel {
    let model_dir = match std::env::var("RASTER_BENCH_MODEL_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => build_fixture_model_dir(),
    };
    load_transformer_state_model_from_det_num_wgt_path(&model_dir).expect("bench model should load")
}

fn token_ids(len: usize) -> Vec<u32> {
    (0..len).map(|idx| (idx % VOCAB) as u32).collect()
}

fn prompt_preparation(prompt_token_ids: &[u32]) -> PromptPreparationState {
    PromptPreparationState {
        prompt_text: "bench".to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        prompt_token_ids_sha256: "bench".to_string(),
    }
}

fn prefill_decode_state(
    model: &Gemma4TransformerModel,
    context_len: usize,
) -> TransformerDecodeState {
    let prompt_token_ids = token_ids(context_len);
    let token_embeddings = input_embedding::run(
        &prompt_token_ids,
        model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("bench embedding should succeed");
    run_prefill_pass_with_mode(
        &prompt_preparation(&prompt_token_ids),
        model,
        &token_embeddings,
        InferenceExecutionMode::Deterministic,
    )
    .expect("bench prefill should succeed")
    .transformer_decode_state
}

fn det_prefill(criterion: &mut Criterion) {
    let model = bench_model();
    let mut group = criterion.benchmark_group("det_prefill");
    group.sample_size(10);
    for &seq_len in PREFILL_SEQ_LENS {
        let prompt_token_ids = token_ids(seq_len);
        let token_embeddings = input_embedding::run(
            &prompt_token_ids,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("bench embedding should succeed");
        let preparation = prompt_preparation(&prompt_token_ids);
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_function(format!("seq_{seq_len}"), |bencher| {
            bencher.iter(|| {
                run_prefill_pass_with_mode(
                    &preparation,
                    &model,
                    &token_embeddings,
                    InferenceExecutionMode::Deterministic,
                )
                .expect("bench prefill should succeed")
            })
        });
    }
    group.finish();
}

fn det_decode(criterion: &mut Criterion) {
    let model = bench_model();
    let mut group = criterion.benchmark_group("det_decode");
    group.sample_size(20);
    for &context_len in DECODE_CONTEXT_LENS {
        let decode_state = prefill_decode_state(&model, context_len);
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("ctx_{context_len}"), |bencher| {
            bencher.iter_batched(
                || decode_state.clone(),
                |state| {
                    decode_step_with_mode(state, 1, &model, InferenceExecutionMode::Deterministic)
                        .expect("bench decode step should succeed")
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn configure() -> Criterion {
    // Single-threaded per the benchmark protocol.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = configure();
    targets = det_prefill, det_decode
}
criterion_main!(benches);

// ---------------------------------------------------------------------------
// Synthetic fixture model
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

fn build_fixture_model_dir() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should advance")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raster-inference-bench-{unique}"));
    fs::create_dir_all(&dir).expect("bench dir should create");

    let config = format!(
        r#"{{
  "text_config": {{
    "enable_moe_block": false,
    "head_dim": {HEAD_DIM},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN},
    "layer_types": ["sliding_attention", "full_attention"],
    "num_attention_heads": {NUM_HEADS},
    "num_hidden_layers": {LAYERS},
    "num_key_value_heads": {NUM_KV_HEADS},
    "rms_norm_eps": 0.000001,
    "sliding_window": {SLIDING_WINDOW},
    "tie_word_embeddings": false,
    "vocab_size": {VOCAB}
  }}
}}"#
    );
    fs::write(dir.join("config.json"), config).expect("config should write");

    let mut rng = FixtureRng(0xbe9c_be9c_be9c_be9c);
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
    }
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

    write_detwgt_file(&dir.join("model.detwgt"), &tensors);
    dir
}

fn write_detwgt_file(path: &Path, tensors: &[FixtureTensor]) {
    let policy = match std::env::var("RASTER_BENCH_WGT_WIDTH").as_deref() {
        Ok("i32") => DetWgtWidthPolicy::ForceI32,
        _ => DetWgtWidthPolicy::Auto,
    };
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
