//! Tests that need crate-private access (trace checkpoint payloads and
//! routine-internal entry points); the public-API sequence tests live in
//! `tests/inference_sequence.rs`.

use serde_json::json;
use tokenizers::Tokenizer;
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

use crate::routines::decode_select_token::run as run_decode_select_token;
use crate::routines::decode_transition_finalize::trace_checkpoint as finalize_decode_transition;
use crate::routines::output_finalize::run as run_output_finalize;
use crate::routines::prefill_finalize::raster::auth_source::AuthenticatedDecoderPrefillFinalizeSource;
use crate::routines::prefill_finalize::run as run_prefill_finalize;
use crate::routines::prefill_prepare_aux::run as run_prefill_prepare_aux;
use crate::routines::prompt_prepare::run as run_prompt_prepare;
use crate::runtime::pipeline::decode_step;
use crate::shared::model::gemma::adapter::GemmaModelBundle;
use crate::shared::model::gemma::tokenizer::GemmaAddedToken;
use crate::shared::model::gemma::tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaBpeMerge, GemmaTokenizerSpec, GemmaVocabEntry,
};
use crate::shared::model::runtime::LoadedModel;
use crate::shared::model::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, Gemma4LayerMatrixSource, GemmaEmbeddingTensorSource,
    InternalActivationSequence, InternalLogits,
};
use crate::shared::model::transformer::{
    Gemma4AttentionKind, Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4PleGlobalWeights,
    Gemma4PleLayerWeights, Gemma4TransformerModel, MatrixF32,
};
use crate::shared::numerics::det_num::{f32_to_acc, Act, Wgt};
use crate::shared::raster_contracts::prefill_layer::AuthenticatedDecoderPrefillLayerSource;
use crate::shared::raster_kernels::transformer::RasterActivationSequence;
use crate::shared::tensors::raster_tensor_artifacts::insert_activation_sequence_artifact_ref;
use crate::{
    DecodeState, InferenceControls, InferenceRequest, InferenceRunOutcome, ModelSpec,
    OutputDecodeStopReason, RasterDetourSpec, SamplingConfig, TextDecodingPolicy,
};
use std::{
    env, fs, process,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
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
    crate::runtime::sequence::run(request, &loaded_model, controls)
}

fn trace_test_lock() -> &'static Mutex<()> {
    crate::trace::test_trace_lock()
}

fn checkpoint_commitments(payload: &serde_json::Value, checkpoint: &str) -> Vec<String> {
    payload
        .as_array()
        .expect("checkpoint payload should be an array")
        .iter()
        .filter_map(|entry| entry.get(checkpoint)?.as_str().map(ToString::to_string))
        .collect()
}

fn checkpoint_commitments_with_prefix(
    payload: &serde_json::Value,
    checkpoint_prefix: &str,
) -> Vec<String> {
    payload
        .as_array()
        .expect("checkpoint payload should be an array")
        .iter()
        .filter_map(|entry| {
            let object = entry.as_object()?;
            object.iter().find_map(|(checkpoint, commitment)| {
                checkpoint
                    .starts_with(checkpoint_prefix)
                    .then(|| commitment.as_str().map(ToString::to_string))
                    .flatten()
            })
        })
        .collect()
}

fn checkpoint_name_count(payload: &serde_json::Value, checkpoint: &str) -> usize {
    payload
        .as_array()
        .expect("checkpoint payload should be an array")
        .iter()
        .filter(|entry| entry.get(checkpoint).is_some())
        .count()
}

fn checkpoint_entry_name_and_commitment(entry: &serde_json::Value) -> (&str, &str) {
    let object = entry
        .as_object()
        .expect("checkpoint entry should be an object");
    let (checkpoint, commitment) = object
        .iter()
        .next()
        .expect("checkpoint entry should contain a commitment");
    (
        checkpoint.as_str(),
        commitment
            .as_str()
            .expect("checkpoint commitment should be a string"),
    )
}

fn assert_checkpoint_payloads_match_except_detour(
    native_payload: &serde_json::Value,
    detour_payload: &serde_json::Value,
    detour_spec: &str,
) {
    let detour_spec = RasterDetourSpec::parse(detour_spec).expect("detour spec should parse");
    let native_entries = native_payload
        .as_array()
        .expect("native checkpoint payload should be an array");
    let detour_entries = detour_payload
        .as_array()
        .expect("detour checkpoint payload should be an array");
    assert_eq!(
        native_entries.len(),
        detour_entries.len(),
        "checkpoint payloads should have the same shape"
    );

    let mut selected_seen = 0;
    for (idx, (native_entry, detour_entry)) in native_entries.iter().zip(detour_entries).enumerate()
    {
        let (native_checkpoint, native_commitment) =
            checkpoint_entry_name_and_commitment(native_entry);
        let (detour_checkpoint, detour_commitment) =
            checkpoint_entry_name_and_commitment(detour_entry);
        assert_eq!(
            native_checkpoint, detour_checkpoint,
            "checkpoint name mismatch at entry {idx}"
        );

        if native_checkpoint == detour_spec.routine_id().as_str() {
            selected_seen += 1;
            if selected_seen == detour_spec.occurrence() {
                continue;
            }
        }

        assert_eq!(
            native_commitment, detour_commitment,
            "non-detoured checkpoint {native_checkpoint} differed at entry {idx}"
        );
    }

    assert!(
        selected_seen >= detour_spec.occurrence(),
        "selected checkpoint {} was not present",
        detour_spec
    );
}

struct TraceDirGuard {
    previous: Option<std::ffi::OsString>,
}

impl TraceDirGuard {
    fn new(test_name: &str) -> Self {
        let previous = env::var_os("RASTER_TRACE_DIR");
        let trace_dir =
            env::temp_dir().join(format!("raster-inference-{test_name}-{}", process::id()));
        fs::create_dir_all(&trace_dir).expect("trace dir should be created");
        env::set_var("RASTER_TRACE_DIR", trace_dir);
        Self { previous }
    }
}

impl Drop for TraceDirGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            env::set_var("RASTER_TRACE_DIR", previous);
        } else {
            env::remove_var("RASTER_TRACE_DIR");
        }
    }
}

fn expect_completed_state(
    outcome: InferenceRunOutcome,
    description: &str,
) -> super::InferenceState {
    let InferenceRunOutcome::Completed(state) = outcome else {
        panic!("expected {description} to complete");
    };
    state
}

fn assert_output_decode_matches(native: &super::InferenceState, detour: &super::InferenceState) {
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
        crate::trace::sha256_hex(&native.output_decode.generated_text),
        crate::trace::sha256_hex(&detour.output_decode.generated_text)
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

fn assert_decode_select_detour_matches_native(
    test_name: &str,
    max_new_tokens: usize,
    detour_spec: &str,
    expected_decode_select_checkpoints: usize,
) {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new(test_name);
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(max_new_tokens);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(RasterDetourSpec::parse(detour_spec).expect("detour should parse")),
            raster_tokenizer_enabled: true,
            raster_sequence_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("decode select token detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(&native_payload, &detour_payload, detour_spec);
    assert_eq!(
        checkpoint_commitments(&native_payload, "decode.select_token").len(),
        expected_decode_select_checkpoints
    );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "decode select token detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
}

fn assert_decode_transition_detour_matches_native(
    test_name: &str,
    max_new_tokens: usize,
    detour_spec: &str,
    expected_decode_transitions: usize,
) {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new(test_name);
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(max_new_tokens);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(RasterDetourSpec::parse(detour_spec).expect("detour should parse")),
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("decode transition detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(&native_payload, &detour_payload, detour_spec);
    assert_eq!(
        checkpoint_commitments_with_prefix(&native_payload, "decode.layer_token.").len(),
        0
    );
    assert_eq!(
        checkpoint_commitments_with_prefix(&detour_payload, "decode.layer_token.").len(),
        0
    );
    assert_eq!(
        checkpoint_name_count(&native_payload, "decode.layer_range"),
        expected_decode_transitions
    );
    assert_eq!(
        checkpoint_name_count(&detour_payload, "decode.layer_range"),
        expected_decode_transitions
    );
    assert_eq!(
        checkpoint_name_count(&native_payload, "decode.transition_finalize"),
        expected_decode_transitions
    );
    assert_eq!(
        checkpoint_name_count(&detour_payload, "decode.transition_finalize"),
        expected_decode_transitions
    );
    assert!(
            checkpoint_name_count(&native_payload, "decode.finalize") == 0
                && checkpoint_name_count(&detour_payload, "decode.finalize") == 0,
            "decode.transition_finalize is the decode finalization checkpoint; decode.finalize should not be committed"
        );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "decode transition detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
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

fn deterministic_ple_model_fixture() -> DeterministicModelFixture {
    let mut fixture = deterministic_no_ple_model_fixture();
    let (ple_layer_weights_file, ple_layer_sources) =
        write_det_layer_weights(vec![det_zero_matrix(2, 4), det_zero_matrix(4, 2)]);
    let mut ple_layer_sources = ple_layer_sources.into_iter();
    fixture.model.layers[0].ple = Some(Gemma4PleLayerWeights {
        input_gate: Gemma4LayerMatrixSource::from_det_num_source(
            ple_layer_sources
                .next()
                .expect("PLE input gate layer source"),
        ),
        layer_projection: Gemma4LayerMatrixSource::from_det_num_source(
            ple_layer_sources
                .next()
                .expect("PLE layer projection source"),
        ),
        post_input_norm_weight: vec![1.0; 4],
        post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); 4]),
    });

    let (ple_global_weights_file, ple_global_sources) =
        write_det_layer_weights(vec![det_zero_matrix(3, 2), det_zero_matrix(2, 4)]);
    let mut ple_global_sources = ple_global_sources.into_iter();
    fixture.model.ple_global = Some(Gemma4PleGlobalWeights::from_det_num_sources_with_canonical(
        vec![ple_global_sources.next().expect("PLE token embeddings")],
        vec![ple_global_sources.next().expect("PLE model projection")],
        vec![1.0; 2],
        vec![Wgt::from_num(1.0); 2],
        1.0,
        Act::from_num(1.0),
        1.0,
        Act::from_num(1.0),
        1.0,
        Act::from_num(1.0),
    ));
    fixture.weights_files.push(ple_layer_weights_file);
    fixture.weights_files.push(ple_global_weights_file);
    fixture
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
            element_width: crate::shared::numerics::det_num::DetWgtElementWidth::I32,
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
            element_width: crate::shared::numerics::det_num::DetWgtElementWidth::I32,
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
fn deterministic_cpu_trace_matches_input_embedding_detour_trace() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("input-embedding-detour-trace");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_sequence_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("input embedding detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(
        &native_payload,
        &detour_payload,
        "input.embedding",
    );
    assert_eq!(
        checkpoint_commitments(&native_payload, "input.embedding").len(),
        1
    );
    assert_eq!(
        checkpoint_commitments(&detour_payload, "input.embedding").len(),
        1
    );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "prefill prepare aux detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
}

#[test]
fn deterministic_cpu_trace_matches_prefill_prepare_aux_detour_trace() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("prefill-prepare-aux-detour-trace");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: Some(2),
            raster_sequence_rows_per_tile: Some(2),
            raster_head_rows_per_tile: Some(2),
            raster_tokenizer_bpe_pairs_per_tile: Some(2),
            raster_tokenizer_bpe_pieces_per_tile: Some(2),
            raster_output_byte_flush_bytes_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("prefill prepare aux detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(
        &native_payload,
        &detour_payload,
        "prefill.prepare_aux",
    );
    assert_eq!(
        checkpoint_commitments(&native_payload, "prefill.prepare_aux").len(),
        1
    );
    assert_eq!(
        checkpoint_commitments(&detour_payload, "prefill.prepare_aux").len(),
        1
    );

    let InferenceRunOutcome::Completed(native) = native else {
        panic!("expected native inference to complete");
    };
    let InferenceRunOutcome::Completed(detour) = detour else {
        panic!("expected detour inference to complete");
    };
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
        crate::trace::sha256_hex(&native.output_decode.generated_text),
        crate::trace::sha256_hex(&detour.output_decode.generated_text)
    );
    assert_eq!(
        native.output_decode.generated_token_count,
        detour.output_decode.generated_token_count
    );
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
}

#[test]
fn deterministic_cpu_trace_matches_no_ple_prefill_prepare_aux_detour_trace() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("no-ple-prefill-prepare-aux-detour-trace");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("no-PLE prefill prepare aux detour should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(
        &native_payload,
        &detour_payload,
        "prefill.prepare_aux",
    );
    assert_eq!(
        checkpoint_commitments(&native_payload, "prefill.prepare_aux").len(),
        1
    );
    assert_eq!(
        checkpoint_commitments(&detour_payload, "prefill.prepare_aux").len(),
        1
    );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "no-PLE prefill prepare aux detour inference");
    assert_output_decode_matches(&native, &detour);
}

#[test]
fn deterministic_cpu_trace_matches_prefill_layer_detour_trace() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("prefill-layer-detour-trace");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let mut transformer_fixture = deterministic_no_ple_model_fixture();
    transformer_fixture
        .model
        .layers
        .push(transformer_fixture.model.layers[0].clone());
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range:2").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: Some(2),
            raster_sequence_rows_per_tile: Some(2),
            raster_head_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("prefill layer detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(
        &native_payload,
        &detour_payload,
        "prefill.range_finalize:2",
    );
    assert_eq!(
        checkpoint_commitments(&native_payload, "prefill.range_finalize").len(),
        2
    );
    assert_eq!(
        checkpoint_commitments(&detour_payload, "prefill.range_finalize").len(),
        2
    );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "prefill layer detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
}

#[test]
fn deterministic_cpu_trace_matches_ple_prefill_layer_detour_trace() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("ple-prefill-layer-detour-trace");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.range").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_attention_kv_rows_per_tile: Some(2),
            raster_sequence_rows_per_tile: Some(2),
            raster_head_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("PLE prefill layer detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(
        &native_payload,
        &detour_payload,
        "prefill.range_finalize",
    );
    assert_eq!(
        checkpoint_commitments(&native_payload, "prefill.range_finalize").len(),
        1
    );
    assert_eq!(
        checkpoint_commitments(&detour_payload, "prefill.range_finalize").len(),
        1
    );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "PLE prefill layer detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
}

#[test]
fn deterministic_cpu_trace_matches_prefill_finalize_detour_trace() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("prefill-finalize-detour-trace");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("prefill finalize detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(
        &native_payload,
        &detour_payload,
        "prefill.finalize",
    );
    assert_eq!(
        checkpoint_commitments(&native_payload, "prefill.finalize").len(),
        1
    );
    assert_eq!(
        checkpoint_commitments(&detour_payload, "prefill.finalize").len(),
        1
    );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "prefill finalize detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
}

#[test]
fn deterministic_cpu_trace_matches_output_finalize_detour_trace() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("output-finalize-detour-trace");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let tokenizer_source = test_native_matching_gemma_tokenizer_source();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls_and_raster_tokenizer(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        Some(tokenizer_source.clone()),
        &InferenceControls {
            commit_checkpoints: true,
            raster_tokenizer_enabled: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls_and_raster_tokenizer(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        Some(tokenizer_source),
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_output_byte_flush_bytes_per_tile: Some(1),
            ..InferenceControls::default()
        },
    )
    .expect("output finalize detour inference should complete");
    let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_checkpoint_payloads_match_except_detour(
        &native_payload,
        &detour_payload,
        "output.finalize",
    );
    assert_eq!(
        checkpoint_commitments(&native_payload, "output.finalize").len(),
        1
    );
    assert_eq!(
        checkpoint_commitments(&detour_payload, "output.finalize").len(),
        1
    );

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "output finalize detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "detour should expose raster tile telemetry outside committed checkpoints"
    );
}

#[test]
fn deterministic_cpu_trace_matches_decode_select_token_detour_trace() {
    assert_decode_select_detour_matches_native(
        "decode-select-token-detour-trace",
        2,
        "decode.select_token",
        2,
    );
}

#[test]
fn deterministic_cpu_trace_matches_second_decode_select_token_detour_trace() {
    assert_decode_select_detour_matches_native(
        "second-decode-select-token-detour-trace",
        2,
        "decode.select_token:2",
        2,
    );
}

#[test]
fn deterministic_cpu_trace_matches_decode_transition_detour_trace() {
    assert_decode_transition_detour_matches_native(
        "decode-transition-detour-trace",
        2,
        "decode.layer_range",
        2,
    );
}

#[test]
fn deterministic_cpu_trace_matches_second_decode_transition_detour_trace() {
    assert_decode_transition_detour_matches_native(
        "second-decode-transition-detour-trace",
        2,
        "decode.layer_range:2",
        2,
    );
}

#[test]
fn deterministic_cpu_decode_transition_non_detoured_checkpoints_match_raster_detour() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("decode-transition-checkpoint-commitment");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let deterministic_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            raster_detour: Some(
                RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
            ),
            raster_projection_rows_per_tile: Some(3),
            raster_attention_kv_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("decode transition raster detour inference should complete");
    let raster_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_eq!(
        checkpoint_commitments_with_prefix(&deterministic_payload, "decode.layer_token.").len(),
        0
    );
    assert_checkpoint_payloads_match_except_detour(
        &deterministic_payload,
        &raster_payload,
        "decode.layer_range",
    );
}

#[test]
fn deterministic_cpu_decode_layer_range_width_splits_checkpoints() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let _trace_dir = TraceDirGuard::new("decode-layer-range-width");
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let mut transformer_fixture = deterministic_no_ple_model_fixture();
    transformer_fixture
        .model
        .layers
        .push(transformer_fixture.model.layers[0].clone());
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            commit_checkpoints: true,
            decode_layer_range_width: Some(1),
            ..InferenceControls::default()
        },
    )
    .expect("native deterministic inference should complete");
    let payload = crate::trace::take_completed_checkpoint_payload_for_tests();

    assert_eq!(checkpoint_name_count(&payload, "decode.layer_range"), 2);
    assert_eq!(
        checkpoint_name_count(&payload, "decode.transition_finalize"),
        1
    );
}

#[test]
fn deterministic_cpu_prefill_layer_checkpoint_commitment_matches_raster() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let input_rows = vec![
        vec![
            Act::from_num(1.0),
            Act::from_num(0.0),
            Act::from_num(0.0),
            Act::from_num(0.0),
        ],
        vec![
            Act::from_num(0.0),
            Act::from_num(1.0),
            Act::from_num(0.0),
            Act::from_num(0.0),
        ],
    ];

    let deterministic_payload = crate::trace::with_checkpointing_enabled(true, || {
        crate::trace::start_inference_trace(&json!({ "test": "deterministic-prefill-layer" }));
        crate::routines::prefill_range::run_internal_with_detour(
            InternalActivationSequence::from_det_values(input_rows.clone()),
            &transformer_fixture.model,
            None,
            None,
            None,
            1,
        )
        .expect("deterministic prefill layer should run");
        crate::trace::checkpoint_payload_for_tests()
    });

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let input_ref = insert_activation_sequence_artifact_ref(
        "test.prefill.layer.input_embedding",
        RasterActivationSequence::from_acts(input_rows),
    )
    .expect("input embedding activation ref");
    let input_embedding_roots =
        crate::shared::artifacts::artifact_io::ArtifactIo::export_store_roots();
    let input_embedding_refs = crate::routines::input_embedding::raster::RasterInputEmbeddingRefs {
        source_id: "embedding-fixture".to_string(),
        embedding_source_root: "embedding-root".to_string(),
        prompt_token_ids_root: "token-root".to_string(),
        prompt_token_count: input_ref.row_count(),
        embedded_prompt_activations_ref: input_ref,
    };
    let layer_source = AuthenticatedDecoderPrefillLayerSource::from_model(
        "prefill-layer",
        &transformer_fixture.model,
    )
    .expect("prefill layer source");
    let raster_payload = crate::trace::with_checkpointing_enabled(true, || {
        crate::trace::start_inference_trace(&json!({ "test": "raster-prefill-layer" }));
        crate::routines::prefill_range::run_raster(
            input_embedding_roots.clone(),
            &input_embedding_refs,
            &layer_source,
            None,
            InferenceControls {
                prefill_token_range_width: Some(1),
                ..InferenceControls::default()
            }
            .raster_sizing_controls()
            .expect("default sizing"),
        )
        .expect("raster prefill layer should run");
        crate::trace::checkpoint_payload_for_tests()
    });

    assert_eq!(
        checkpoint_commitments(&deterministic_payload, "prefill.range_finalize"),
        checkpoint_commitments(&raster_payload, "prefill.range_finalize")
    );
    assert_eq!(
        checkpoint_commitments(&deterministic_payload, "prefill.range").len(),
        2
    );
    assert_eq!(
        checkpoint_commitments(&deterministic_payload, "prefill.range"),
        checkpoint_commitments(&raster_payload, "prefill.range")
    );
}

#[test]
fn deterministic_cpu_prefill_finalize_checkpoint_commitment_matches_raster() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let token_ids = vec![1, 2];
    let input_rows = vec![
        vec![
            Act::from_num(1.0),
            Act::from_num(0.0),
            Act::from_num(0.0),
            Act::from_num(0.0),
        ],
        vec![
            Act::from_num(0.0),
            Act::from_num(1.0),
            Act::from_num(0.0),
            Act::from_num(0.0),
        ],
    ];

    let deterministic_payload = crate::trace::with_checkpointing_enabled(true, || {
        crate::trace::start_inference_trace(&json!({ "test": "deterministic-prefill-finalize" }));
        let (final_hidden_states, layer_caches) = crate::routines::prefill_range::run_internal(
            InternalActivationSequence::from_det_values(input_rows.clone()),
            &transformer_fixture.model,
            None,
        )
        .expect("deterministic prefill layer should run");
        crate::routines::prefill_finalize::run(
            &token_ids,
            &transformer_fixture.model,
            final_hidden_states,
            layer_caches,
        )
        .expect("deterministic prefill finalize should run");
        crate::trace::checkpoint_payload_for_tests()
    });

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let input_ref = insert_activation_sequence_artifact_ref(
        "test.prefill.finalize.input_embedding",
        RasterActivationSequence::from_acts(input_rows),
    )
    .expect("input embedding activation ref");
    let input_embedding_roots =
        crate::shared::artifacts::artifact_io::ArtifactIo::export_store_roots();
    let input_embedding_refs = crate::routines::input_embedding::raster::RasterInputEmbeddingRefs {
        source_id: "embedding-fixture".to_string(),
        embedding_source_root: "embedding-root".to_string(),
        prompt_token_ids_root: "token-root".to_string(),
        prompt_token_count: input_ref.row_count(),
        embedded_prompt_activations_ref: input_ref,
    };
    let layer_source = AuthenticatedDecoderPrefillLayerSource::from_model(
        "prefill-finalize-layer",
        &transformer_fixture.model,
    )
    .expect("prefill layer source");
    let finalize_source = AuthenticatedDecoderPrefillFinalizeSource::from_model(
        "prefill-finalize",
        &transformer_fixture.model,
    )
    .expect("prefill finalize source");
    let raster_payload = crate::trace::with_checkpointing_enabled(true, || {
        crate::trace::start_inference_trace(&json!({ "test": "raster-prefill-finalize" }));
        let (layer_roots, layer_refs) = crate::routines::prefill_range::run_raster(
            input_embedding_roots.clone(),
            &input_embedding_refs,
            &layer_source,
            None,
            InferenceControls::default()
                .raster_sizing_controls()
                .expect("default sizing"),
        )
        .expect("raster prefill layer should run");
        crate::routines::prefill_finalize::materialize_raster_input_roots_for_api(
            layer_roots,
            token_ids.len(),
            &finalize_source,
            layer_refs.final_hidden_states_ref,
            layer_refs.layer_caches,
            InferenceControls::default()
                .raster_sizing_controls()
                .expect("default sizing")
                .projection_rows_per_tile,
        )
        .expect("raster prefill finalize should run");
        crate::trace::checkpoint_payload_for_tests()
    });

    assert_eq!(
        checkpoint_commitments(&deterministic_payload, "prefill.finalize"),
        checkpoint_commitments(&raster_payload, "prefill.finalize")
    );
}

#[test]
fn deterministic_cpu_decode_select_checkpoint_commitment_matches_raster() {
    let _trace_guard = trace_test_lock().lock().expect("trace test lock");
    let det_logits = vec![Act::from_bits(2), Act::from_bits(5), Act::from_bits(3)];
    let internal_logits = InternalLogits::from_det_values(det_logits);

    let deterministic_payload = crate::trace::with_checkpointing_enabled(true, || {
        crate::trace::start_inference_trace(&json!({ "test": "deterministic-decode-select" }));
        let mut decode_state = DecodeState::new(
            vec![7],
            internal_logits.clone_f32(),
            crate::shared::model::transformer::TransformerDecodeState::default(),
        );
        decode_state.set_internal_logits(internal_logits.clone());
        run_decode_select_token(&mut decode_state, 1)
            .expect("deterministic decode select should run");
        crate::trace::checkpoint_payload_for_tests()
    });

    let raster_payload = crate::trace::with_checkpointing_enabled(true, || {
        crate::trace::start_inference_trace(&json!({ "test": "raster-decode-select" }));
        let mut decode_state = DecodeState::new(
            vec![7],
            internal_logits.clone_f32(),
            crate::shared::model::transformer::TransformerDecodeState::default(),
        );
        decode_state.set_internal_logits(internal_logits.clone());
        crate::routines::decode_select_token::materialize_run_raster_for_api(&mut decode_state, 1)
            .expect("raster decode select should run");
        crate::trace::checkpoint_payload_for_tests()
    });

    assert_eq!(
        checkpoint_commitments(&deterministic_payload, "decode.select_token"),
        checkpoint_commitments(&raster_payload, "decode.select_token")
    );
}

#[test]
fn routine_exports_support_manual_inference_orchestration() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_no_ple_model_fixture();
    let transformer_model = &transformer_fixture.model;
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

    let prompt_preparation =
        run_prompt_prepare(&request, &model, &tokenizer).expect("prompt prepare");
    let token_embeddings = crate::routines::input_embedding::run(
        &prompt_preparation.prompt_token_ids,
        transformer_model,
    )
    .expect("embed tokens");
    let ple_inputs = run_prefill_prepare_aux(
        &prompt_preparation.prompt_token_ids,
        transformer_model,
        &token_embeddings,
    )
    .expect("prefill prepare aux");
    let (final_hidden_states, layer_caches) = crate::routines::prefill_range::run_internal(
        token_embeddings.clone_internal(),
        transformer_model,
        ple_inputs.as_ref(),
    )
    .expect("prefill layer");
    let prefill = run_prefill_finalize(
        &prompt_preparation.prompt_token_ids,
        transformer_model,
        final_hidden_states,
        layer_caches,
    )
    .expect("prefill finalize");

    let mut decode_state = DecodeState::new(
        prompt_preparation.prompt_token_ids.clone(),
        prefill.transformer_state.prefill_logits.logits.clone(),
        prefill.transformer_decode_state.clone(),
    );
    decode_state.set_internal_logits(prefill.transformer_state.prefill_logits.clone_internal());
    let next_token = run_decode_select_token(&mut decode_state, 1).expect("decode select token");
    let next_token = next_token.expect("should select a token");
    let loaded_model = LoadedModel::Gemma(GemmaModelBundle::new(
        test_model_spec(),
        tokenizer.clone(),
        transformer_model.clone(),
        None,
    ));
    let decode_transition = decode_step(
        std::mem::take(&mut decode_state.transformer_decode_state),
        next_token,
        &loaded_model,
    )
    .expect("decode transition");
    decode_state.set_internal_logits(decode_transition.prefill_logits.clone_internal());
    decode_state.transformer_decode_state = decode_transition.transformer_decode_state;
    finalize_decode_transition(&decode_state).expect("decode finalize trace");

    let output = run_output_finalize(decode_state, &tokenizer).expect("output finalize");
    assert_eq!(output.generated_token_ids, vec![0]);
    assert_eq!(output.generated_text, "hello");
    assert_eq!(output.stop_reason, OutputDecodeStopReason::MaxNewTokens);
}

#[test]
fn run_inference_executes_prefill_prepare_aux_raster_detour() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_ple_model_fixture();
    let request = deterministic_prompt_request(1);

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls::default(),
    )
    .expect("native deterministic inference should complete");

    crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
    let detour = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls {
            raster_detour: Some(
                RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
            ),
            raster_tokenizer_enabled: true,
            raster_projection_rows_per_tile: Some(2),
            raster_sequence_rows_per_tile: Some(2),
            ..InferenceControls::default()
        },
    )
    .expect("prefill prepare aux detour should complete");

    let native = expect_completed_state(native, "native inference");
    let detour = expect_completed_state(detour, "prefill prepare aux detour inference");
    assert_output_decode_matches(&native, &detour);
    assert!(
        detour.raster_tile_invocations.unwrap_or(0) > 0,
        "prefill prepare aux detour should count raster tiles"
    );
}

#[test]
fn run_inference_with_controls_raster_uses_ple_ref_bridge() {
    let tokenizer = test_tokenizer();
    let model = test_model_spec();
    let transformer_fixture = deterministic_ple_model_fixture();
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

    let native = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_fixture.model,
        &InferenceControls::default(),
    )
    .expect("native deterministic inference should complete");
    let raster = run_inference_with_controls(
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

    let InferenceRunOutcome::Completed(native) = native else {
        panic!("expected native inference to complete");
    };
    let InferenceRunOutcome::Completed(raster) = raster else {
        panic!("expected completed raster inference");
    };
    assert_eq!(
        raster.output_decode.generated_token_ids,
        native.output_decode.generated_token_ids
    );
    assert!(raster.raster_tile_invocations.expect("tile count") > 0);
}
