//! Allocation regression gate for the deterministic hot paths.
//!
//! Uses a counting global allocator to verify the clone-elimination work:
//! - decode: allocations per decode step are independent of context length
//!   (the legacy path cloned every layer's KV cache per token, O(context)),
//!   and stay below a fixed per-step budget.
//! - prefill: allocations grow (sub-)linearly with sequence length (the
//!   legacy path materialized key/value windows per (head, query), O(seq²)).

use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use raster_inference::shared::numerics::det_num::{
    f32_to_wgt, wgt_to_le_bytes, DET_NUM_SPEC_VERSION, DET_WGT_ARTIFACT_FORMAT_VERSION,
    DET_WGT_ARTIFACT_MAGIC,
};
use raster_inference::{
    decode_step_with_mode, input_embedding, load_transformer_state_model_from_det_num_wgt_path,
    run_prefill_pass_with_mode, Gemma4TransformerModel, InferenceExecutionMode,
    PromptPreparationState, TransformerDecodeState,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocations() -> u64 {
    ALLOCATIONS.load(Ordering::Relaxed)
}

const HIDDEN: usize = 16;
const HEAD_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const LAYERS: usize = 2;
const FF: usize = 32;
const VOCAB: usize = 16;
const SLIDING_WINDOW: usize = 48;

#[test]
fn det_decode_allocations_are_context_independent_and_bounded() {
    // Pin rayon so allocation counts are stable.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let model = fixture_model();

    let per_step_small = decode_step_allocations(&model, 32);
    let per_step_large = decode_step_allocations(&model, 256);

    // The legacy path cloned every layer's KV cache per step, so allocations
    // grew linearly with context length. The flat-slab path appends in place:
    // per-step allocation counts must not grow with context.
    assert!(
        per_step_large <= per_step_small + per_step_small / 4 + 8,
        "decode allocations grew with context: ctx=32 -> {per_step_small}, ctx=256 -> {per_step_large}"
    );
    // Fixed per-step budget: scratch reuse keeps the per-layer loop alloc-free
    // apart from bounded bookkeeping (trace labels, state pushes, commitments).
    assert!(
        per_step_large < 600,
        "decode step allocations exceeded budget: {per_step_large}"
    );
}

#[test]
fn det_prefill_allocations_scale_subquadratically() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let model = fixture_model();

    let allocs_small = prefill_allocations(&model, 32);
    let allocs_large = prefill_allocations(&model, 64);

    // The legacy path materialized per-(head, query) key/value windows, an
    // O(seq²) allocation pattern: doubling the sequence would roughly
    // quadruple allocations. The flat path allocates per layer + per token
    // bookkeeping only, so doubling the sequence must stay near 2x.
    assert!(
        allocs_large < allocs_small * 3,
        "prefill allocations scale superlinearly: seq=32 -> {allocs_small}, seq=64 -> {allocs_large}"
    );
}

fn decode_step_allocations(model: &Gemma4TransformerModel, context_len: usize) -> u64 {
    let mut decode_state = prefill_decode_state(model, context_len);
    // Warm-up step (fills lazy weight caches and scratch capacities).
    let step = decode_step_with_mode(
        decode_state,
        1,
        model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("warm-up decode step should succeed");
    decode_state = step.transformer_decode_state;

    let before = allocations();
    let step = decode_step_with_mode(
        decode_state,
        2,
        model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("measured decode step should succeed");
    let after = allocations();
    drop(step);
    after - before
}

fn prefill_allocations(model: &Gemma4TransformerModel, seq_len: usize) -> u64 {
    // Warm-up (lazy weight materialization).
    run_prefill(model, seq_len);
    let before = allocations();
    run_prefill(model, seq_len);
    let after = allocations();
    after - before
}

fn run_prefill(model: &Gemma4TransformerModel, seq_len: usize) -> TransformerDecodeState {
    let prompt_token_ids = token_ids(seq_len);
    let token_embeddings = input_embedding::run(
        &prompt_token_ids,
        model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("embedding should succeed");
    run_prefill_pass_with_mode(
        &prompt_preparation(&prompt_token_ids),
        model,
        &token_embeddings,
        InferenceExecutionMode::Deterministic,
    )
    .expect("prefill should succeed")
    .transformer_decode_state
}

fn prefill_decode_state(
    model: &Gemma4TransformerModel,
    context_len: usize,
) -> TransformerDecodeState {
    run_prefill(model, context_len)
}

fn token_ids(len: usize) -> Vec<u32> {
    (0..len).map(|idx| (idx % VOCAB) as u32).collect()
}

fn prompt_preparation(prompt_token_ids: &[u32]) -> PromptPreparationState {
    PromptPreparationState {
        prompt_text: "alloc".to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        prompt_token_ids_sha256: "alloc".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Fixture model
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
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should advance")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raster-inference-alloc-{unique}"));
    fs::create_dir_all(&dir).expect("alloc fixture dir should create");

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

    let mut rng = FixtureRng(0xa110ca7e_a110ca7e);
    let mut tensors = Vec::new();
    let push = |tensors: &mut Vec<FixtureTensor>, name: String, shape: &[usize], values: Vec<f32>| {
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
    load_transformer_state_model_from_det_num_wgt_path(&dir).expect("alloc fixture should load")
}

fn write_detwgt_file(path: &Path, tensors: &[FixtureTensor]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

    for tensor in tensors {
        let name_bytes = tensor.name.as_bytes();
        let wgt_bits = tensor
            .values
            .iter()
            .map(|value| f32_to_wgt(*value))
            .collect::<Vec<_>>();
        let payload = wgt_bits
            .iter()
            .flat_map(|wgt| wgt_to_le_bytes(*wgt))
            .collect::<Vec<_>>();
        let row_len = tensor.shape.last().copied().unwrap_or(1).max(1);
        let max_row_mass = wgt_bits
            .chunks(row_len)
            .map(|row| {
                row.iter()
                    .map(|wgt| u64::from(wgt.to_bits().unsigned_abs()))
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0);
        let element_count = tensor.shape.iter().product::<usize>() as u64;

        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in &tensor.shape {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&element_count.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&max_row_mass.to_le_bytes());
        bytes.extend_from_slice(&payload);
    }

    fs::write(path, bytes).expect("detwgt should write");
}

