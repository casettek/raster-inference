//! Integration-style tests for the full inference sequence, exercising
//! `runtime::sequence::run` end to end through the public API.
//! Extracted from the former inline test module of `runtime::inference`.

use serde_json::json;
use tokenizers::Tokenizer;
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

use raster_inference::shared::model::gemma::adapter::GemmaModelBundle;
use raster_inference::shared::model::gemma::tokenizer::GemmaAddedToken;
use raster_inference::shared::model::gemma::tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaBpeMerge, GemmaTokenizerSpec, GemmaVocabEntry,
};
use raster_inference::shared::model::runtime::LoadedModel;
use raster_inference::shared::model::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4TransformerModel, GemmaEmbeddingTensorSource,
    MatrixF32,
};
use raster_inference::shared::numerics::det_num::{f32_to_acc, Act, Wgt};
use raster_inference::{
    InferenceControls, InferenceRequest, InferenceRunOutcome, InferenceState, ModelSpec,
    OutputDecodeStopReason, RasterDetourSpec, RoutineId, SamplingConfig, TextDecodingPolicy,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

fn run_inference_with_controls(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    controls: &InferenceControls,
) -> anyhow::Result<InferenceRunOutcome> {
    let raster_tokenizer =
        (controls.raster || controls.raster_tokenizer_enabled).then(test_gemma_tokenizer_source);
    run_inference_with_controls_and_raster_tokenizer(
        request,
        model,
        tokenizer,
        transformer_model,
        raster_tokenizer,
        controls,
    )
}

fn run_inference_with_controls_and_raster_tokenizer(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    raster_tokenizer: Option<AuthenticatedGemmaTokenizer>,
    controls: &InferenceControls,
) -> anyhow::Result<InferenceRunOutcome> {
    let loaded_model = LoadedModel::Gemma(GemmaModelBundle::new(
        model.clone(),
        tokenizer.clone(),
        transformer_model.clone(),
        raster_tokenizer,
    ));
    raster_inference::sequence::run(request, &loaded_model, controls)
}

fn run_inference(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
) -> anyhow::Result<InferenceState> {
    match run_inference_with_controls(
        request,
        model,
        tokenizer,
        transformer_model,
        &InferenceControls::default(),
    )? {
        InferenceRunOutcome::Completed(state) => Ok(state),
        InferenceRunOutcome::Paused(paused) => anyhow::bail!(
            "inference paused unexpectedly at checkpoint {}",
            paused.terminal_checkpoint_id
        ),
        InferenceRunOutcome::RasterPromptPrepared(state) => anyhow::bail!(
            "raster inference stopped at unsupported routine boundary {}",
            state.terminal_checkpoint_id
        ),
    }
}

fn expect_completed_state(outcome: InferenceRunOutcome, description: &str) -> InferenceState {
    let InferenceRunOutcome::Completed(state) = outcome else {
        panic!("expected {description} to complete");
    };
    state
}

fn assert_output_decode_matches(native: &InferenceState, detour: &InferenceState) {
    assert_eq!(
        native.output_decode.generated_token_ids,
        detour.output_decode.generated_token_ids
    );
    assert_eq!(
        native.output_decode.generated_token_ids_sha256,
        detour.output_decode.generated_token_ids_sha256
    );
    assert_eq!(
        native.output_decode.generated_text,
        detour.output_decode.generated_text
    );
    assert_eq!(
        raster_inference::trace::sha256_hex(&native.output_decode.generated_text),
        raster_inference::trace::sha256_hex(&detour.output_decode.generated_text)
    );
    assert_eq!(
        native.output_decode.generated_token_count,
        detour.output_decode.generated_token_count
    );
}

fn deterministic_prompt_request(max_new_tokens: usize) -> InferenceRequest {
    InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(max_new_tokens),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    }
}

fn test_model_spec() -> ModelSpec {
    ModelSpec {
        model_id: "gemma-4-test".to_string(),
        tokenizer_path: "tokenizer.json".into(),
        chat_template: "{{ messages[0].content }}".to_string(),
        bos_token: None,
        eos_token: None,
        unk_token: Some("<unk>".to_string()),
    }
}

fn test_tokenizer() -> tokenizers::Tokenizer {
    let vocab = [
        ("hello".to_string(), 0),
        ("prompt".to_string(), 1),
        ("<unk>".to_string(), 2),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("word level tokenizer");
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

fn test_gemma_tokenizer_source() -> AuthenticatedGemmaTokenizer {
    AuthenticatedGemmaTokenizer::new(test_gemma_tokenizer_spec())
}

fn test_native_matching_gemma_tokenizer_source() -> AuthenticatedGemmaTokenizer {
    AuthenticatedGemmaTokenizer::new(test_gemma_tokenizer_spec_with_output_token("hello"))
}

fn test_gemma_tokenizer_spec() -> GemmaTokenizerSpec {
    test_gemma_tokenizer_spec_with_output_token("raster-hello")
}

fn test_gemma_tokenizer_spec_with_output_token(output_token: &str) -> GemmaTokenizerSpec {
    GemmaTokenizerSpec::new(
        "digest".to_string(),
        vec![
            GemmaVocabEntry {
                token: output_token.to_string(),
                id: 0,
            },
            GemmaVocabEntry {
                token: "prompt".to_string(),
                id: 1,
            },
            GemmaVocabEntry {
                token: "<unk>".to_string(),
                id: 2,
            },
            GemmaVocabEntry {
                token: "p".to_string(),
                id: 3,
            },
            GemmaVocabEntry {
                token: "r".to_string(),
                id: 4,
            },
            GemmaVocabEntry {
                token: "o".to_string(),
                id: 5,
            },
            GemmaVocabEntry {
                token: "m".to_string(),
                id: 6,
            },
            GemmaVocabEntry {
                token: "t".to_string(),
                id: 7,
            },
            GemmaVocabEntry {
                token: "pr".to_string(),
                id: 8,
            },
            GemmaVocabEntry {
                token: "pro".to_string(),
                id: 9,
            },
            GemmaVocabEntry {
                token: "prom".to_string(),
                id: 10,
            },
            GemmaVocabEntry {
                token: "promp".to_string(),
                id: 11,
            },
        ],
        vec![
            GemmaBpeMerge {
                left: "p".to_string(),
                right: "r".to_string(),
                merged: "pr".to_string(),
                rank: 0,
            },
            GemmaBpeMerge {
                left: "pr".to_string(),
                right: "o".to_string(),
                merged: "pro".to_string(),
                rank: 1,
            },
            GemmaBpeMerge {
                left: "pro".to_string(),
                right: "m".to_string(),
                merged: "prom".to_string(),
                rank: 2,
            },
            GemmaBpeMerge {
                left: "prom".to_string(),
                right: "p".to_string(),
                merged: "promp".to_string(),
                rank: 3,
            },
            GemmaBpeMerge {
                left: "promp".to_string(),
                right: "t".to_string(),
                merged: "prompt".to_string(),
                rank: 4,
            },
        ],
        vec![GemmaAddedToken {
            id: 2,
            content: "<unk>".to_string(),
            special: true,
        }],
        "<unk>".to_string(),
        true,
        "▁".to_string(),
        " ".to_string(),
    )
    .expect("test Gemma tokenizer spec should build")
}

struct DeterministicModelFixture {
    model: Gemma4TransformerModel,
    weights_files: Vec<std::path::PathBuf>,
}

impl Drop for DeterministicModelFixture {
    fn drop(&mut self) {
        for weights_file in &self.weights_files {
            let _ = std::fs::remove_file(weights_file);
        }
    }
}

fn deterministic_no_ple_model_fixture() -> DeterministicModelFixture {
    let embedding_rows = vec![
        vec![Act::from_num(0.0); 4],
        vec![Act::from_num(0.0); 4],
        vec![Act::from_num(0.0); 4],
    ];
    let (weights_file, source) = write_det_embedding_weights(embedding_rows);
    let mut model = test_transformer_model();
    let (layer_weights_file, layer_sources) = write_det_layer_weights(vec![
        det_zero_matrix(4, 4),
        det_zero_matrix(2, 4),
        det_zero_matrix(2, 4),
        det_zero_matrix(4, 4),
        det_zero_matrix(8, 4),
        det_zero_matrix(8, 4),
        det_zero_matrix(4, 8),
    ]);
    let mut layer_sources = layer_sources.into_iter();
    model.embedding_source = Some(GemmaEmbeddingTensorSource::Deterministic {
        source,
        scale: 1.0,
        det_cache: Arc::new(Mutex::new(None::<Arc<DetNumMatrix>>)),
    });
    model.logits_projection = Gemma4LogitsProjection::UntiedLmHead {
        weight: zero_matrix(3, 4),
        det_weight: Some(Arc::new(DetNumMatrix {
            rows: 3,
            cols: 4,
            values: vec![Wgt::from_num(0.0).to_bits(); 12].into(),
        })),
    };
    model.rms_norm_eps_det = Some(f32_to_acc(model.rms_norm_eps));
    model.final_norm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
    let layer = &mut model.layers[0];
    layer.q_proj =
        Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("q source"));
    layer.k_proj =
        Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("k source"));
    layer.v_proj = Some(Gemma4LayerMatrixSource::from_det_num_source(
        layer_sources.next().expect("v source"),
    ));
    layer.o_proj =
        Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("o source"));
    layer.gate_proj =
        Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("gate source"));
    layer.up_proj =
        Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("up source"));
    layer.down_proj =
        Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("down source"));
    layer.rms_norm_eps_det = Some(f32_to_acc(layer.rms_norm_eps));
    layer.rope_base_det = Some(f32_to_acc(layer.rope_base));
    layer.q_norm_weight_det = Some(vec![Wgt::from_num(1.0); 2]);
    layer.k_norm_weight_det = Some(vec![Wgt::from_num(1.0); 2]);
    layer.input_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
    layer.post_attention_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
    layer.pre_feedforward_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
    layer.post_feedforward_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
    DeterministicModelFixture {
        model,
        weights_files: vec![weights_file, layer_weights_file],
    }
}

fn write_det_embedding_weights(
    rows: Vec<Vec<Act>>,
) -> (std::path::PathBuf, DetNumTensorSliceSource) {
    let unique_suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let unique_counter = next_fixture_counter();
    let path = std::env::temp_dir().join(format!(
        "raster-lib-det-embedding-{}-{unique_suffix}-{unique_counter}.detwgt",
        std::process::id(),
    ));
    let mut bytes = Vec::new();
    for row in &rows {
        for value in row {
            bytes.extend(value.to_bits().to_le_bytes());
        }
    }
    std::fs::write(&path, bytes).expect("det embedding fixture should write");
    let row_count = rows.len();
    let col_count = rows.first().map(Vec::len).unwrap_or(0);
    (
        path.clone(),
        DetNumTensorSliceSource {
            weights_path: path,
            total_rows: row_count,
            total_cols: col_count,
            data_offset: 0,
            element_width: raster_inference::shared::numerics::det_num::DetWgtElementWidth::I32,
            row_offset: 0,
            row_count,
            col_offset: 0,
            col_count,
        },
    )
}

fn write_det_layer_weights(
    matrices: Vec<Vec<Vec<Wgt>>>,
) -> (std::path::PathBuf, Vec<DetNumTensorSliceSource>) {
    let unique_suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let unique_counter = next_fixture_counter();
    let path = std::env::temp_dir().join(format!(
        "raster-lib-det-layer-{}-{unique_suffix}-{unique_counter}.detwgt",
        std::process::id(),
    ));
    let mut bytes = Vec::new();
    let mut sources = Vec::new();
    for matrix in matrices {
        let data_offset = bytes.len();
        for row in &matrix {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        sources.push(DetNumTensorSliceSource {
            weights_path: path.clone(),
            total_rows: matrix.len(),
            total_cols: matrix.first().map(Vec::len).unwrap_or(0),
            data_offset,
            element_width: raster_inference::shared::numerics::det_num::DetWgtElementWidth::I32,
            row_offset: 0,
            row_count: matrix.len(),
            col_offset: 0,
            col_count: matrix.first().map(Vec::len).unwrap_or(0),
        });
    }
    std::fs::write(&path, bytes).expect("det layer fixture should write");
    (path, sources)
}

fn det_zero_matrix(rows: usize, cols: usize) -> Vec<Vec<Wgt>> {
    vec![vec![Wgt::from_num(0.0); cols]; rows]
}

fn next_fixture_counter() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn test_transformer_model() -> Gemma4TransformerModel {
    Gemma4TransformerModel {
        embedding_source: None,
        layers: vec![Gemma4LayerWeights {
            attention_kind: Gemma4AttentionKind::Sliding,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: Some(2),
            cache_sliding_window: Some(2),
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
            rope_base: 10_000.0,
            rope_base_det: None,
            partial_rotary_dim: 2,
            rope_freq_base_dim: 2,
            kv_shared_layer_index: None,
            attention_k_eq_v: false,
            q_proj: zero_matrix(4, 4).into(),
            k_proj: zero_matrix(2, 4).into(),
            v_proj: Some(zero_matrix(2, 4).into()),
            o_proj: zero_matrix(4, 4).into(),
            q_norm_weight: vec![1.0, 1.0],
            q_norm_weight_det: None,
            k_norm_weight: vec![1.0, 1.0],
            k_norm_weight_det: None,
            input_layernorm_weight: vec![1.0; 4],
            input_layernorm_weight_det: None,
            post_attention_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight_det: None,
            pre_feedforward_layernorm_weight: vec![1.0; 4],
            pre_feedforward_layernorm_weight_det: None,
            post_feedforward_layernorm_weight: vec![1.0; 4],
            post_feedforward_layernorm_weight_det: None,
            gate_proj: zero_matrix(8, 4).into(),
            up_proj: zero_matrix(8, 4).into(),
            down_proj: zero_matrix(4, 8).into(),
            ple: None,
            layer_scalar: None,
            layer_scalar_det: None,
        }],
        ple_global: None,
        final_norm_weight: vec![1.0; 4],
        final_norm_weight_det: None,
        logits_projection: Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(3, 4),
            det_weight: None,
        },
        final_logit_softcapping: None,
        final_logit_softcapping_det: None,
        rms_norm_eps: 1e-6,
        rms_norm_eps_det: None,
    }
}

fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
    MatrixF32 {
        rows,
        cols,
        values: vec![0.0; rows * cols],
    }
}

#[test]
fn run_inference_generates_greedy_text_for_max_new_tokens() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let inference_state = run_inference(&request, &model, &tokenizer, &transformer_fixture.model)
        .expect("inference should succeed");

    assert_eq!(
        inference_state
            .input_embedding
            .prompt_preparation
            .prompt_token_ids,
        vec![1]
    );
    assert_eq!(
        inference_state.output_decode.generated_token_ids,
        vec![0, 0]
    );
    assert_eq!(inference_state.output_decode.generated_text, "hello hello");
    assert_eq!(inference_state.output_decode.generated_token_count, 2);
    assert_eq!(
        inference_state.output_decode.stop_reason,
        OutputDecodeStopReason::MaxNewTokens
    );
}

#[test]
fn run_inference_returns_empty_generation_when_max_new_tokens_is_zero() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let inference_state = run_inference(&request, &model, &tokenizer, &transformer_fixture.model)
        .expect("inference should succeed");

    assert!(inference_state.output_decode.generated_token_ids.is_empty());
    assert_eq!(inference_state.output_decode.generated_text, "");
    assert_eq!(inference_state.output_decode.generated_token_count, 0);
}

#[test]
fn run_inference_rejects_non_default_sampling_before_decode_loop() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(1),
            temperature: Some(1.0),
            top_k: Some(5),
            top_p: None,
        },
    };

    let error = run_inference(&request, &model, &tokenizer, &transformer_fixture.model)
        .expect_err("top_k should fail");
    assert!(error.to_string().contains("top_k"));
}

#[test]
fn run_inference_reports_unsupported_selected_raster_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(1),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prompt.prepare").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(4),
            ..InferenceControls::default()
        },
    )
    .expect_err("unimplemented detour should fail");

    let message = error.to_string();
    assert!(
        message.contains("selective raster detour for prompt.prepare is not implemented yet"),
        "{message}"
    );
}

#[test]
fn run_inference_reports_unmatched_selected_raster_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range:999").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect_err("unmatched detour should fail");

    assert!(error
        .to_string()
        .contains("selective raster detour target prefill.range:999 was not reached"));
}

#[test]
fn run_inference_reports_unmatched_decode_select_token_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("decode.select_token:2").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect_err("second decode select token detour should be unmatched");

    assert!(error
        .to_string()
        .contains("selective raster detour target decode.select_token:2 was not reached"));
}

#[test]
fn run_inference_validates_sequence_sizing_for_decode_select_token_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("decode.select_token").expect("detour should parse"),
            ),
            raster_sequence_rows_per_tile: Some(0),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect_err("zero sequence rows should fail");

    assert!(error
        .to_string()
        .contains("raster sequence rows per tile must be greater than zero"));
}

#[test]
fn run_inference_counts_prefill_layer_detour_occurrences() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let mut transformer_fixture = deterministic_no_ple_model_fixture();
    transformer_fixture
        .model
        .layers
        .push(transformer_fixture.model.layers[0].clone());
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls::default(),
    )
    .expect("native inference should complete");

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range:2").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect("second prefill layer detour should complete");

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "second prefill layer detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "prefill layer detour should count raster tiles"
    );
}

#[test]
fn run_inference_rejects_full_raster_with_raster_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prompt.prepare").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect_err("full raster and detour should conflict");

    assert!(error
        .to_string()
        .contains("--raster and selective raster detour cannot be used together"));
}

#[test]
fn run_inference_rejects_raster_core_detours_for_unmigrated_routines() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            // At least one decode iteration so every decode-loop decision
            // point (select_token, layer_range, transition_finalize,
            // output.finalize) is reached.
            max_new_tokens: Some(1),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    for routine_id in RoutineId::ALL {
        // Migrated routines (WS3) dispatch to their raster-core host
        // adapters instead of rejecting; their detour behavior is covered
        // by the per-routine dev-run verification tests.
        if matches!(
            routine_id,
            RoutineId::PromptPrepare | RoutineId::SelectOutputToken
        ) {
            continue;
        }
        raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse_raster_core(routine_id.as_str())
                        .expect("raster-core spec should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("raster-core detour should be rejected as unimplemented");

        let expected = format!(
            "selective raster-core detour for {} is not implemented yet",
            routine_id.as_str()
        );
        assert!(
            error.to_string().contains(&expected),
            "unexpected error for {}: {error:#}",
            routine_id.as_str()
        );
    }
}

#[test]
fn run_inference_reports_unreached_raster_core_detour_target() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse_raster_core("prefill.range:7")
                    .expect("raster-core spec should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect_err("unreached raster-core target should fail");

    assert!(error
        .to_string()
        .contains("selective raster-core detour target prefill.range:7 was not reached"));
}

#[test]
fn run_inference_validates_raster_sizing_for_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prompt.prepare").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(0),
            ..InferenceControls::default()
        },
    )
    .expect_err("zero raster sizing should fail for detour");

    assert!(error
        .to_string()
        .contains("raster projection rows per tile must be greater than zero"));
}

#[test]
fn run_inference_validates_attention_sizing_for_prefill_layer_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(0);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range").expect("detour should parse"),
            ),
            raster_attention_kv_rows_per_tile: Some(0),
            ..InferenceControls::default()
        },
    )
    .expect_err("zero attention KV rows should fail for prefill layer detour");

    assert!(error
        .to_string()
        .contains("raster attention KV rows per tile must be greater than zero"));
}

#[test]
fn run_inference_executes_input_embedding_raster_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls::default(),
    )
    .expect("native deterministic inference should complete");
    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("input embedding detour should complete");

    let InferenceRunOutcome::Completed(native) = native else {
        panic!("expected native inference to complete");
    };
    let InferenceRunOutcome::Completed(detour) = detour else {
        panic!("expected detour inference to complete");
    };
    assert_eq!(
        native.input_embedding.prompt_preparation.prompt_token_ids,
        detour.input_embedding.prompt_preparation.prompt_token_ids
    );
    assert_eq!(
        native.input_embedding.embedded_prompt_activations_sha256,
        detour.input_embedding.embedded_prompt_activations_sha256
    );
    assert_eq!(
        native
            .input_embedding
            .det_embedded_prompt_activations_sha256,
        detour
            .input_embedding
            .det_embedded_prompt_activations_sha256
    );
    assert_eq!(
        native.output_decode.generated_token_ids,
        detour.output_decode.generated_token_ids
    );
    assert_eq!(
        native.output_decode.generated_token_ids_sha256,
        detour.output_decode.generated_token_ids_sha256
    );
    assert_eq!(
        native.output_decode.generated_text,
        detour.output_decode.generated_text
    );
    assert_eq!(
        native.output_decode.generated_token_count,
        detour.output_decode.generated_token_count
    );
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "input embedding detour should count raster tiles"
    );
}

#[test]
fn run_inference_input_embedding_detour_can_pause_after_input_embedding() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            terminal_checkpoint: Some("input.embedding".to_string()),
            raster_detour: Some(
                RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("input embedding detour should pause");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "input.embedding");
            assert_eq!(
                state.input_embedding.prompt_preparation.prompt_token_ids,
                vec![1]
            );
            assert!(state
                .input_embedding
                .det_embedded_prompt_activations_sha256
                .is_some());
            assert!(state.transformer_state_transition.is_none());
            assert!(state.output_decode.is_none());
            assert!(
                state.raster_tile_invocations.unwrap_or(0) > 0,
                "input embedding detour should count raster tiles"
            );
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused detour inference")
        }
    }
}

#[test]
fn run_inference_prefill_layer_detour_can_pause_after_selected_layer() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let mut transformer_fixture = deterministic_no_ple_model_fixture();
    transformer_fixture
        .model
        .layers
        .push(transformer_fixture.model.layers[0].clone());
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            terminal_checkpoint: Some("prefill.range_finalize:2".to_string()),
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range:2").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("prefill layer detour should pause after selected layer");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prefill.range_finalize");
            assert!(state.transformer_state_transition.is_none());
            assert!(state.output_decode.is_none());
            assert!(
                state.raster_tile_invocations.unwrap_or(0) > 0,
                "prefill layer detour should count raster tiles"
            );
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused prefill layer detour inference")
        }
    }
}

#[test]
fn run_inference_executes_prefill_finalize_raster_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls::default(),
    )
    .expect("native deterministic inference should complete");

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("prefill finalize detour should complete");

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "prefill finalize detour inference");
    assert_output_decode_matches(&native, &detour);
    assert_eq!(
        native.transformer_state_transition.prefill_logits,
        detour.transformer_state_transition.prefill_logits
    );
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "prefill finalize detour should count raster tiles"
    );
}

#[test]
fn run_inference_prefill_finalize_detour_can_pause_after_finalize() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            terminal_checkpoint: Some("prefill.finalize".to_string()),
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("prefill finalize detour should pause after finalize");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
            assert!(state.transformer_state_transition.is_some());
            assert!(state.output_decode.is_none());
            assert!(
                state.raster_tile_invocations.unwrap_or(0) > 0,
                "prefill finalize detour should count raster tiles"
            );
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused prefill finalize detour inference")
        }
    }
}

#[test]
fn run_inference_reports_unmatched_second_prefill_finalize_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(0);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.finalize:2").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect_err("second prefill finalize detour should be unmatched");

    assert!(error
        .to_string()
        .contains("selective raster detour target prefill.finalize:2 was not reached"));
}

#[test]
fn run_inference_prefill_finalize_detour_uses_projection_tile_sizing() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(0);

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let single_row_chunks = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(1),
            ..InferenceControls::default()
        },
    )
    .expect("single-row projection chunks should complete");

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let multi_row_chunks = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("multi-row projection chunks should complete");

    let single_row_chunks = expect_completed_state(
        single_row_chunks,
        "single-row prefill finalize detour inference",
    );
    let multi_row_chunks = expect_completed_state(
        multi_row_chunks,
        "multi-row prefill finalize detour inference",
    );
    assert_output_decode_matches(&single_row_chunks, &multi_row_chunks);
    assert!(
        single_row_chunks.raster_tile_invocations.unwrap_or(0)
            > multi_row_chunks.raster_tile_invocations.unwrap_or(0),
        "smaller projection chunks should invoke more raster tiles"
    );
}

#[cfg(feature = "unchecked-raster-integrity")]
#[test]
fn run_inference_prefill_finalize_detour_runs_in_unchecked_integrity_mode() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::integrity_mode::with_raster_integrity_mode(
        raster_inference::RasterIntegrityMode::UncheckedTestOnly,
        || {
            raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
            let detour = run_inference_with_controls(
                &request,
                &model,
                &tokenizer,
                &transformer_fixture.model,
                &InferenceControls {
                    raster_detour: Some(
                        RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
                    ),
                    raster_projection_rows_per_tile: Some(2),
                    ..InferenceControls::default()
                },
            )
            .expect("unchecked prefill finalize detour should complete");

            let detour = expect_completed_state(detour, "unchecked prefill finalize detour");
            assert!(
                detour.raster_tile_invocations.unwrap_or(0) > 0,
                "unchecked prefill finalize detour should count raster tiles"
            );
        },
    );
}

#[test]
fn run_inference_executes_output_finalize_raster_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let tokenizer_source = test_native_matching_gemma_tokenizer_source();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls::default(),
    )
    .expect("native deterministic inference should complete");

    raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls_and_raster_tokenizer(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        Some(tokenizer_source),
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("output finalize detour should complete");

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "output finalize detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "output finalize detour should count raster tiles"
    );
}

#[test]
fn run_inference_output_finalize_detour_requires_authenticated_tokenizer_source() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect_err("output finalize detour requires tokenizer source");

    assert!(error
        .to_string()
        .contains("selective raster output.finalize detour requires raster tokenizer capability"));
}

#[test]
fn run_inference_reports_unmatched_second_output_finalize_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(0);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("output.finalize:2").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect_err("second output finalize detour should be unmatched");

    assert!(error
        .to_string()
        .contains("selective raster detour target output.finalize:2 was not reached"));
}

#[test]
fn run_inference_validates_output_byte_flush_sizing_for_output_finalize_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_output_byte_flush_bytes_per_tile: Some(0),
            ..InferenceControls::default()
        },
    )
    .expect_err("zero output byte flush bytes should fail for output finalize detour");

    assert!(error
        .to_string()
        .contains("raster output byte flush bytes per tile must be greater than zero"));
}

#[cfg(feature = "unchecked-raster-integrity")]
#[test]
fn run_inference_output_finalize_detour_runs_in_unchecked_integrity_mode() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::integrity_mode::with_raster_integrity_mode(
        raster_inference::RasterIntegrityMode::UncheckedTestOnly,
        || {
            raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
            let detour = run_inference_with_controls(
                &request,
                &model,
                &tokenizer,
                &transformer_fixture.model,
                &InferenceControls {
                    raster_detour: Some(
                        RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
                    ),
                    raster_tokenizer_enabled: true,
                    ..InferenceControls::default()
                },
            )
            .expect("unchecked output finalize detour should complete");

            let detour = expect_completed_state(detour, "unchecked output finalize detour");
            assert!(
                detour.raster_tile_invocations.unwrap_or(0) > 0,
                "unchecked output finalize detour should count raster tiles"
            );
        },
    );
}

#[test]
fn run_inference_input_embedding_detour_requires_prompt_artifact_roots() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect_err("input embedding detour requires prompt artifact roots");

    assert!(error
        .to_string()
        .contains("selective raster input.embedding detour requires raster prompt preparation"));
}

#[test]
fn run_inference_reports_unmatched_second_input_embedding_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(0);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("input.embedding:2").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect_err("second input embedding detour should be unmatched");

    assert!(error
        .to_string()
        .contains("selective raster detour target input.embedding:2 was not reached"));
}

#[test]
fn run_inference_prefill_prepare_aux_detour_requires_input_embedding_refs() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect_err("prefill prepare aux detour requires input embedding refs");

    assert!(error.to_string().contains(
        "selective raster prefill.prepare_aux detour requires input embedding raster refs"
    ));
}

#[test]
fn run_inference_reports_unmatched_second_prefill_prepare_aux_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(0);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.prepare_aux:2").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect_err("second prefill prepare aux detour should be unmatched");

    assert!(error
        .to_string()
        .contains("selective raster detour target prefill.prepare_aux:2 was not reached"));
}

#[test]
fn run_inference_validates_sequence_sizing_for_prefill_prepare_aux_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(0);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_sequence_rows_per_tile: Some(0),
            ..InferenceControls::default()
        },
    )
    .expect_err("zero sequence rows should fail for prepare aux detour");

    assert!(error
        .to_string()
        .contains("raster sequence rows per tile must be greater than zero"));
}

#[test]
fn run_inference_reports_unmatched_decode_transition_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("decode.layer_range:2").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect_err("second decode transition detour should be unmatched");

    assert!(error
        .to_string()
        .contains("selective raster detour target decode.layer_range:2 was not reached"));
}

#[test]
fn run_inference_validates_projection_sizing_for_decode_transition_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(0),
            ..InferenceControls::default()
        },
    )
    .expect_err("zero projection rows should fail for decode transition detour");

    assert!(error
        .to_string()
        .contains("raster projection rows per tile must be greater than zero"));
}

#[test]
fn run_inference_validates_attention_sizing_for_decode_transition_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
            ),
            raster_attention_kv_rows_per_tile: Some(0),
            ..InferenceControls::default()
        },
    )
    .expect_err("zero attention rows should fail for decode transition detour");

    assert!(error
        .to_string()
        .contains("raster attention KV rows per tile must be greater than zero"));
}

#[test]
fn run_inference_decode_transition_finalize_detour_matches_native() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls::default(),
    )
    .expect("native inference should complete");

    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("decode.transition_finalize").expect("detour should parse"),
            ),
            ..InferenceControls::default()
        },
    )
    .expect("decode transition finalize detour should complete");

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "decode transition finalize detour");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "finalize detour should execute raster tiles"
    );
}

#[cfg(feature = "unchecked-raster-integrity")]
#[test]
fn run_inference_decode_transition_detour_runs_in_unchecked_integrity_mode() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::integrity_mode::with_raster_integrity_mode(
        raster_inference::RasterIntegrityMode::UncheckedTestOnly,
        || {
            raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
            let detour = run_inference_with_controls(
                &request,
                &model,
                &tokenizer,
                &transformer_fixture.model,
                &InferenceControls {
                    raster_detour: Some(
                        RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
                    ),
                    raster_projection_rows_per_tile: Some(1),
                    raster_attention_kv_rows_per_tile: Some(1),
                    ..InferenceControls::default()
                },
            )
            .expect("unchecked decode transition detour should complete");

            let detour = expect_completed_state(detour, "unchecked decode transition detour");
            assert!(
                detour.raster_tile_invocations.unwrap_or(0) > 0,
                "unchecked decode transition detour should count raster tiles"
            );
        },
    );
}

#[cfg(feature = "unchecked-raster-integrity")]
#[test]
fn run_inference_decode_select_token_detour_runs_in_unchecked_integrity_mode() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    raster_inference::shared::artifacts::integrity_mode::with_raster_integrity_mode(
        raster_inference::RasterIntegrityMode::UncheckedTestOnly,
        || {
            raster_inference::shared::artifacts::artifact_io::ArtifactIo::reset_store();
            let detour = run_inference_with_controls(
                &request,
                &model,
                &tokenizer,
                &transformer_fixture.model,
                &InferenceControls {
                    raster_detour: Some(
                        RasterDetourSpec::parse("decode.select_token")
                            .expect("detour should parse"),
                    ),
                    raster_sequence_rows_per_tile: Some(1),
                    raster_tokenizer_enabled: true,
                    ..InferenceControls::default()
                },
            )
            .expect("unchecked decode select token detour should complete");

            let detour = expect_completed_state(detour, "unchecked decode select token detour");
            assert!(
                detour.raster_tile_invocations.unwrap_or(0) > 0,
                "unchecked decode select token detour should count raster tiles"
            );
        },
    );
}

#[test]
fn run_inference_with_controls_pauses_after_prompt_prepare_checkpoint() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("prompt.prepare".to_string()),
            raster: false,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("inference should pause");

    match paused {
        InferenceRunOutcome::RasterPromptPrepared(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prompt.prepare");
            assert_eq!(state.prompt_preparation.prompt_token_count, 1);
        }
        InferenceRunOutcome::Completed(_) => panic!("expected paused inference"),
        InferenceRunOutcome::Paused(_) => panic!("expected raster prompt prepared pause"),
    }
}

#[test]
fn deterministic_cpu_prompt_prepare_checkpoint_matches_raster_shape() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let tokenizer_source = test_gemma_tokenizer_source();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("prompt.prepare".to_string()),
            raster: false,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("deterministic inference should stop after prompt prepare");
    let expected = raster_inference::routines::prompt_prepare::run_raster(
        &request,
        &model,
        &tokenizer_source,
        InferenceControls::default()
            .raster_sizing_controls()
            .expect("default sizing"),
    )
    .expect("raster prompt prepare should run");

    match paused {
        InferenceRunOutcome::RasterPromptPrepared(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prompt.prepare");
            assert_eq!(state.prompt_preparation, expected.state);
            assert_eq!(state.sampling, request.sampling);
            assert_eq!(state.raster_tile_invocations, None);
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::Paused(_) => {
            panic!("expected deterministic CPU prompt boundary")
        }
    }
}

#[test]
fn run_inference_with_controls_raster_can_pause_after_input_embedding() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("input.embedding".to_string()),
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("raster inference should stop after input embedding");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "input.embedding");
            assert_eq!(
                state.input_embedding.prompt_preparation.prompt_token_ids,
                vec![1]
            );
            assert!(state
                .input_embedding
                .det_embedded_prompt_activations_sha256
                .is_some());
            assert!(state.transformer_state_transition.is_none());
            assert!(state.output_decode.is_none());
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused inference")
        }
    }
}

#[test]
fn run_inference_with_controls_raster_can_pause_after_prefill_prepare_aux() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("prefill.prepare_aux".to_string()),
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("raster inference should stop after prefill prepare aux");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prefill.prepare_aux");
            assert_eq!(
                state.input_embedding.prompt_preparation.prompt_token_ids,
                vec![1]
            );
            assert!(state.transformer_state_transition.is_none());
            assert!(state.output_decode.is_none());
            assert!(
                state.raster_tile_invocations.unwrap_or(0) > 0,
                "raster prefill checkpoint should include tile invocation count"
            );
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused raster inference")
        }
    }
}

#[test]
fn run_inference_with_controls_raster_can_pause_after_prefill_layer() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("prefill.range_finalize".to_string()),
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("raster inference should stop after prefill layer");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prefill.range_finalize");
            assert!(state.transformer_state_transition.is_none());
            assert!(state.output_decode.is_none());
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused raster inference")
        }
    }
}

#[test]
fn run_inference_with_controls_pauses_after_second_prefill_layer_checkpoint() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let mut transformer_fixture = deterministic_no_ple_model_fixture();
    let first_layer = transformer_fixture.model.layers[0].clone();
    transformer_fixture.model.layers.push(first_layer);
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(1),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("prefill.range_finalize:2".to_string()),
            raster: false,
            raster_detour: None,
            raster_tokenizer_enabled: false,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("inference should pause after the second prefill layer checkpoint");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prefill.range_finalize");
            assert!(state.transformer_state_transition.is_none());
            assert!(state.output_decode.is_none());
        }
        InferenceRunOutcome::Completed(_) => panic!("expected paused inference"),
        InferenceRunOutcome::RasterPromptPrepared(_) => panic!("expected paused inference"),
    }
}

#[test]
fn run_inference_with_controls_raster_can_pause_after_prefill_finalize() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("prefill.finalize".to_string()),
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("raster inference should stop after prefill finalize");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
            assert!(
                state.raster_tile_invocations.unwrap_or(0) > 0,
                "raster prefill should include tile invocation count"
            );
            assert!(state.transformer_state_transition.is_some());
            assert!(state.output_decode.is_none());
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused raster inference")
        }
    }
}

#[test]
fn run_inference_with_controls_raster_runs_decode_select_token() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let outcome = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: None,
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("raster inference should complete");

    match outcome {
        InferenceRunOutcome::Completed(state) => {
            assert_eq!(state.output_decode.generated_token_ids, vec![0, 0]);
            assert_eq!(
                state.output_decode.generated_text,
                "raster-helloraster-hello"
            );
            assert_eq!(state.output_decode.generated_token_count, 2);
            assert_eq!(state.output_decode.decode_transition_states.len(), 2);
            assert!(
                state.raster_tile_invocations.expect("tile count") > 0,
                "full raster inference should count raster tiles"
            );
        }
        InferenceRunOutcome::Paused(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected completed raster inference")
        }
    }
}

#[test]
fn run_inference_with_controls_raster_can_pause_after_decode_transition() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("decode.transition_finalize".to_string()),
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("raster inference should pause after decode transition finalize");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "decode.transition_finalize");
            let output_decode = state
                .output_decode
                .expect("partial output decode state should be present");
            assert_eq!(output_decode.generated_token_ids, vec![0]);
            assert_eq!(output_decode.generated_text, "raster-hello");
            assert_eq!(output_decode.generated_token_count, 1);
            assert_eq!(output_decode.decode_transition_states.len(), 1);
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused raster inference")
        }
    }
}

#[test]
fn run_inference_rejects_decode_layer_range_terminal_checkpoint() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            terminal_checkpoint: Some("decode.layer_range".to_string()),
            decode_layer_range_width: Some(1),
            ..InferenceControls::default()
        },
    )
    .expect_err("decode layer range terminal checkpoint should be rejected");

    assert!(error
        .to_string()
        .contains("terminal checkpoint decode.layer_range is not supported"));
}

#[test]
fn run_inference_with_controls_raster_can_pause_after_output_finalize() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(1),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("output.finalize".to_string()),
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("raster inference should pause after output finalize");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "output.finalize");
            assert!(state.transformer_state_transition.is_some());
            let output_decode = state.output_decode.expect("output phase should be present");
            assert_eq!(output_decode.generated_token_count, 1);
        }
        InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
            panic!("expected paused raster inference")
        }
    }
}

#[test]
fn run_inference_with_controls_raster_rejects_non_deterministic_model() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_model = test_transformer_model();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: None,
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect_err("raster inference should reject models without .detwgt weights");

    assert!(error.to_string().contains(".detwgt"));
}

#[test]
fn run_inference_with_controls_raster_rejects_zero_projection_rows_per_tile() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: None,
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(0),
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect_err("zero raster projection rows per tile should fail");

    assert!(error.to_string().contains("greater than zero"));
}

#[test]
fn run_inference_with_controls_raster_rejects_zero_attention_kv_rows_per_tile() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: None,
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: Some(0),
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect_err("zero raster attention KV rows per tile should fail");

    assert!(error.to_string().contains("greater than zero"));
}

#[test]
fn run_inference_with_controls_raster_rejects_zero_sequence_rows_per_tile() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: None,
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: Some(0),
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect_err("zero raster sequence rows per tile should fail");

    assert!(error.to_string().contains("greater than zero"));
}

#[test]
fn run_inference_with_controls_raster_rejects_zero_head_rows_per_tile() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(0),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let error = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: None,
            raster: true,
            raster_detour: None,
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: None,
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: Some(0),
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect_err("zero raster head rows per tile should fail");

    assert!(error.to_string().contains("greater than zero"));
}

#[test]
fn raster_sizing_controls_reject_zero_output_tokenizer_chunks() {
    let pair_error = InferenceControls {
        raster_tokenizer_bpe_pairs_per_tile: Some(0),
        ..InferenceControls::default()
    }
    .raster_sizing_controls()
    .expect_err("zero tokenizer pair chunk should fail");
    assert!(pair_error
        .to_string()
        .contains("BPE pairs per tile must be greater than zero"));

    let piece_error = InferenceControls {
        raster_tokenizer_bpe_pieces_per_tile: Some(0),
        ..InferenceControls::default()
    }
    .raster_sizing_controls()
    .expect_err("zero tokenizer piece chunk should fail");
    assert!(piece_error
        .to_string()
        .contains("BPE pieces per tile must be greater than zero"));

    let output_error = InferenceControls {
        raster_output_byte_flush_bytes_per_tile: Some(0),
        ..InferenceControls::default()
    }
    .raster_sizing_controls()
    .expect_err("zero output byte flush chunk should fail");
    assert!(output_error
        .to_string()
        .contains("output byte flush bytes per tile must be greater than zero"));
}

#[test]
fn run_inference_with_controls_pauses_after_prefill_finalize_checkpoint() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(2),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let paused = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: false,
            terminal_checkpoint: Some("prefill.finalize".to_string()),
            raster: false,
            raster_detour: None,
            raster_tokenizer_enabled: false,
            raster_projection_rows_per_tile: Some(0),
            raster_attention_kv_rows_per_tile: None,
            raster_sequence_rows_per_tile: None,
            raster_head_rows_per_tile: None,
            prefill_token_range_width: None,
            decode_layer_range_width: None,
            raster_tokenizer_bpe_pairs_per_tile: None,
            raster_tokenizer_bpe_pieces_per_tile: None,
            raster_output_byte_flush_bytes_per_tile: None,
        },
    )
    .expect("inference should pause");

    match paused {
        InferenceRunOutcome::Paused(state) => {
            assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
            assert_eq!(state.raster_tile_invocations, None);
            assert_eq!(
                state
                    .transformer_state_transition
                    .expect("transformer phase should be present")
                    .activation_states
                    .len(),
                1
            );
            assert!(state.output_decode.is_none());
        }
        InferenceRunOutcome::Completed(_) => panic!("expected paused inference"),
        InferenceRunOutcome::RasterPromptPrepared(_) => panic!("expected paused inference"),
    }
}

#[test]
fn inference_state_serializes_with_protocol_phase_keys() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = InferenceRequest {
        prompt_bytes: b"prompt".to_vec(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: false,
        add_special_tokens: false,
        sampling: SamplingConfig {
            max_new_tokens: Some(1),
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
        },
    };

    let inference_state = run_inference(&request, &model, &tokenizer, &transformer_fixture.model)
        .expect("inference should succeed");
    let serialized = serde_json::to_value(&inference_state).expect("serialize inference state");
    let object = serialized
        .as_object()
        .expect("serialized inference state should be an object");

    assert!(object.contains_key("input_embedding"));
    assert!(object.contains_key("transformer_state_transition"));
    assert!(object.contains_key("output_decode"));
}

#[test]
fn inference_state_deserializes_protocol_taxonomy_keys() {
    let serialized_shape = json!({
        "input_embedding": {
            "prompt_text": "prompt",
            "prompt_token_ids": [1],
            "prompt_token_ids_sha256": "prompt-digest",
            "embedded_prompt_activations_sha256": "embed-digest"
        },
        "transformer_state_transition": {
            "activation_states": [
                {
                    "activations_sha256": "hidden-digest"
                }
            ],
            "prefill_logits": {
                "final_logits_sha256": "logits-digest"
            }
        },
        "output_decode": {
            "generated_token_ids": [0],
            "generated_token_ids_sha256": "generated-digest",
            "generated_text": "hello"
        }
    });

    let inference_state: InferenceState =
        serde_json::from_value(serialized_shape).expect("serialized shape should deserialize");

    assert_eq!(
        inference_state
            .input_embedding
            .prompt_preparation
            .prompt_token_ids,
        vec![1]
    );
    assert_eq!(
        inference_state
            .input_embedding
            .embedded_prompt_activations_sha256
            .as_deref(),
        Some("embed-digest")
    );
    assert_eq!(
        inference_state
            .transformer_state_transition
            .prefill_logits
            .final_logits_sha256
            .as_deref(),
        Some("logits-digest")
    );
    assert_eq!(inference_state.output_decode.generated_text, "hello");
}
