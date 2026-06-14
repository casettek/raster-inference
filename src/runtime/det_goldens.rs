//! Golden regression gate for canonical (`det_*`) commitments.
//!
//! Runs deterministic inference end-to-end on fixed fixtures covering full +
//! sliding-window attention, donor (shared-KV) layers, PLE layers, and a
//! 64-token decode, then asserts every commitment is bit-identical to the
//! checked-in goldens in `testdata/det_commitment_goldens.json`.
//!
//! Regenerate goldens (only when an intentional contract change lands) with:
//! `RASTER_BLESS_DET_GOLDENS=1 cargo test det_goldens`

use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{json, Value};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer};

use crate::load_transformer_state_model_from_det_num_wgt_path;
use crate::runtime::pipeline;
use crate::shared::api::input::{
    InferenceRequest, ModelSpec, PromptPreparationState, SamplingConfig, TextDecodingPolicy,
};
use crate::shared::model::gemma::adapter::GemmaModelBundle;
use crate::shared::model::runtime::LoadedModel;
use crate::shared::model::transformer::Gemma4TransformerModel;
use crate::shared::numerics::det_num::{encode_det_wgt_artifact, f32_to_wgt, DetWgtTensorSpec};
use crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment;

const DECODE_TOKENS: usize = 64;
const PROMPT: &str = "w0 w1 w2 w3 w4 w5";

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/det_commitment_goldens.json")
}

#[test]
fn det_commitments_match_goldens() {
    let mut fixtures = serde_json::Map::new();
    for (name, donor) in [("donor_ple", true), ("no_donor_ple", false)] {
        fixtures.insert(name.to_string(), run_fixture(name, donor));
    }
    let captured = Value::Object(fixtures);

    if std::env::var("RASTER_BLESS_DET_GOLDENS").is_ok() {
        fs::write(
            golden_path(),
            serde_json::to_vec_pretty(&captured).expect("goldens should serialize"),
        )
        .expect("golden file should be writable");
        eprintln!("blessed goldens at {}", golden_path().display());
        return;
    }

    let golden_bytes = fs::read(golden_path()).expect(
        "missing testdata/det_commitment_goldens.json; run with RASTER_BLESS_DET_GOLDENS=1 to create it",
    );
    let golden: Value = serde_json::from_slice(&golden_bytes).expect("goldens should parse");
    for (fixture_name, golden_fixture) in golden.as_object().expect("golden object") {
        let captured_fixture = &captured[fixture_name];
        for (mode_name, golden_mode) in golden_fixture.as_object().expect("fixture object") {
            assert_eq!(
                &captured_fixture[mode_name], golden_mode,
                "{fixture_name}/{mode_name} commitments diverged from goldens"
            );
        }
    }
}

fn run_fixture(name: &str, donor: bool) -> Value {
    let det_model = build_fixture_model(name, donor);
    let det = capture_det(&det_model);
    json!({ "det": det })
}

fn capture_det(model: &Gemma4TransformerModel) -> Value {
    let tokenizer = test_tokenizer();
    let model_spec = test_model_spec();
    let loaded_model = LoadedModel::Gemma(GemmaModelBundle::new(
        model_spec.clone(),
        tokenizer.clone(),
        model.clone(),
        None,
    ));
    let prompt_preparation =
        crate::routines::prompt_prepare::run(&test_request(), &model_spec, &tokenizer)
            .expect("prompt preparation should succeed");

    // Routine-level capture: embedding, prefill, per-step decode.
    let token_embeddings =
        crate::routines::input_embedding::run(&prompt_preparation.prompt_token_ids, model)
            .expect("input embedding should succeed");
    let prefill = pipeline::run_prefill_pass(
        &PromptPreparationState {
            prompt_text: prompt_preparation.prompt_text.clone(),
            prompt_token_ids: prompt_preparation.prompt_token_ids.clone(),
            prompt_token_ids_sha256: prompt_preparation.prompt_token_ids_sha256.clone(),
        },
        &loaded_model,
        &token_embeddings,
    )
    .expect("prefill should succeed");
    let final_hidden = &prefill.transformer_state.activation_states[0];
    let prefill_logits = &prefill.transformer_state.prefill_logits;

    let prefill_capture = json!({
        "embedding": token_embeddings.det_activations_sha256,
        "final_hidden": final_hidden.det_activations_sha256,
        "logits": prefill_logits.det_final_logits_sha256,
        "kv": build_det_kv_cache_commitment(&prefill.transformer_decode_state.layer_caches),
    });

    // Per-layer capture for the first decode step via the real layer-range routine.
    let first_token = crate::routines::decode_select_token::native::select_next_token_internal(
        &prefill_logits.clone_internal(),
    )
    .expect("token selection should succeed");
    let first_step_layers =
        capture_first_step_layers(model, prefill.transformer_decode_state.clone(), first_token);

    // Per-step decode capture (DECODE_TOKENS steps).
    let mut decode_steps = Vec::with_capacity(DECODE_TOKENS);
    let mut decode_state = prefill.transformer_decode_state.clone();
    let mut logits = prefill_logits.clone_internal();
    for _ in 0..DECODE_TOKENS {
        let next_token =
            crate::routines::decode_select_token::native::select_next_token_internal(&logits)
                .expect("token selection should succeed");
        let step = pipeline::decode_step(decode_state, next_token, &loaded_model)
            .expect("decode step should succeed");
        let step_capture = json!({
            "token": next_token,
            "activation": step.activation_state.det_activations_sha256,
            "logits": step.prefill_logits.det_final_logits_sha256,
            "kv": build_det_kv_cache_commitment(&step.transformer_decode_state.layer_caches),
        });
        decode_steps.push(step_capture);
        decode_state = step.transformer_decode_state;
        logits = step.prefill_logits.clone_internal();
    }

    // End-to-end capture through the public entry point.
    let outcome = crate::runtime::sequence::run(
        &test_request(),
        &loaded_model,
        &crate::InferenceControls::default(),
    )
    .expect("end-to-end inference should succeed");
    let crate::InferenceRunOutcome::Completed(state) = outcome else {
        panic!("end-to-end inference should complete");
    };

    json!({
        "prefill": prefill_capture,
        "first_step_layers": first_step_layers,
        "decode_steps": decode_steps,
        "end_to_end": {
            "generated_token_ids": state.output_decode.generated_token_ids,
            "generated_token_ids_sha256": state.output_decode.generated_token_ids_sha256,
        },
    })
}

fn capture_first_step_layers(
    model: &Gemma4TransformerModel,
    decode_state: crate::shared::model::transformer::TransformerDecodeState,
    next_token: u32,
) -> Value {
    let mut state = crate::routines::decode_layer_range::native::deterministic_tiles::init_state(
        decode_state,
        next_token,
        model,
    )
    .expect("decode layer range init should succeed");

    let mut layers = Vec::with_capacity(state.layer_count);
    while !state.is_complete() {
        let (next_state, _) =
            crate::routines::decode_layer_range::native::deterministic_tiles::run_range(
                state, model, 1,
            )
            .expect("decode layer range should succeed");
        state = next_state;
        let effective_caches = state.effective_layer_caches();
        let layer_capture = json!({
            "layer_output": state.completed_layer_output_det_sha256s.last().cloned(),
            "kv": build_det_kv_cache_commitment(&effective_caches),
        });
        layers.push(layer_capture);
    }
    Value::Array(layers)
}

fn test_request() -> InferenceRequest {
    InferenceRequest {
        prompt_bytes: PROMPT.as_bytes().to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(DECODE_TOKENS),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    }
}

fn test_model_spec() -> ModelSpec {
    ModelSpec {
        model_id: "gemma-4-det-golden".to_string(),
        tokenizer_path: "tokenizer.json".into(),
        chat_template: "{{ messages[0].content }}".to_string(),
        bos_token: None,
        eos_token: None,
        unk_token: Some("<unk>".to_string()),
    }
}

fn test_tokenizer() -> Tokenizer {
    let vocab = (0..8u32)
        .map(|idx| (format!("w{idx}"), idx))
        .chain([("<unk>".to_string(), 8)])
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("word level tokenizer");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

// ---------------------------------------------------------------------------
// Fixture construction
// ---------------------------------------------------------------------------

const HIDDEN: usize = 4;
const HEAD_DIM: usize = 2;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const LAYERS: usize = 4;
const FF: usize = 8;
const VOCAB: usize = 9;
const PLE_DIM: usize = 2;

fn build_fixture_model(name: &str, donor: bool) -> Gemma4TransformerModel {
    let det_dir = create_temp_dir(&format!("det-golden-{name}-det"));
    let config = fixture_config(donor);
    fs::write(det_dir.join("config.json"), &config).expect("config should write");

    let tensors = fixture_tensors();
    write_detwgt_file(&det_dir.join("model.detwgt"), &tensors);

    load_transformer_state_model_from_det_num_wgt_path(&det_dir).expect("det fixture should load")
}

fn fixture_config(donor: bool) -> String {
    let num_kv_shared_layers = if donor { 1 } else { 0 };
    format!(
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
    "num_kv_shared_layers": {num_kv_shared_layers},
    "rms_norm_eps": 0.000001,
    "sliding_window": 4,
    "tie_word_embeddings": false,
    "vocab_size": {VOCAB},
    "vocab_size_per_layer_input": {VOCAB}
  }}
}}"#
    )
}

/// Deterministic pseudo-random weights, exactly representable in both f32 and
/// Q16.16 (multiples of 1/64) so fixture bytes are stable across platforms.
struct FixtureRng(u64);

impl FixtureRng {
    fn next_weight(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bucket = ((self.0 >> 33) % 33) as i64 - 16;
        bucket as f32 / 64.0
    }

    fn next_norm_weight(&mut self) -> f32 {
        1.0 + self.next_weight() / 4.0
    }

    fn weights(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_weight()).collect()
    }

    fn norm_weights(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_norm_weight()).collect()
    }
}

struct FixtureTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn fixture_tensors() -> Vec<FixtureTensor> {
    let mut rng = FixtureRng(0x5eed_5eed_5eed_5eed);
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

fn create_temp_dir(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should advance")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raster-inference-{label}-{unique}"));
    fs::create_dir_all(&dir).expect("temp dir should create");
    dir
}

fn write_detwgt_file(path: &Path, tensors: &[FixtureTensor]) {
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
    let bytes = encode_det_wgt_artifact(&specs).expect("detwgt v2 should encode");
    fs::write(path, bytes).expect("detwgt should write");
}
