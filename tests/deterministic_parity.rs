use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use raster_inference::{
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path, run_inference, InferenceExecutionMode,
    InferenceRequest, ModelSpec, SamplingConfig, TextDecodingPolicy,
};
use raster_inference::shared::det_num::{
    f32_to_wgt, wgt_to_le_bytes, DET_NUM_SPEC_VERSION, DET_WGT_ARTIFACT_FORMAT_VERSION,
    DET_WGT_ARTIFACT_MAGIC,
};
use raster_inference::Gemma4LogitsProjection;
use safetensors::tensor::{serialize_to_file, TensorView};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer};

#[test]
fn deterministic_loader_reconstructs_tied_embedding_projection() {
    let model_dir = create_temp_dir("det-loader-tied");
    write_config(
        &model_dir,
        r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 2,
    "tie_word_embeddings": true,
    "vocab_size": 3
  }
}"#,
    );
    write_detwgt_file(
        &model_dir.join("model.detwgt"),
        &[
            tensor("model.language_model.embed_tokens.weight", &[3, 4], &[0.0, 0.5, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0]),
            tensor("model.language_model.layers.0.self_attn.q_proj.weight", &[4, 4], &[0.0; 16]),
            tensor("model.language_model.layers.0.self_attn.k_proj.weight", &[2, 4], &[0.0; 8]),
            tensor("model.language_model.layers.0.self_attn.v_proj.weight", &[2, 4], &[0.0; 8]),
            tensor("model.language_model.layers.0.self_attn.o_proj.weight", &[4, 4], &[0.0; 16]),
            tensor("model.language_model.layers.0.self_attn.q_norm.weight", &[2], &[1.0, 1.0]),
            tensor("model.language_model.layers.0.self_attn.k_norm.weight", &[2], &[1.0, 1.0]),
            tensor("model.language_model.layers.0.input_layernorm.weight", &[4], &[1.0; 4]),
            tensor("model.language_model.layers.0.post_attention_layernorm.weight", &[4], &[1.0; 4]),
            tensor("model.language_model.layers.0.pre_feedforward_layernorm.weight", &[4], &[1.0; 4]),
            tensor("model.language_model.layers.0.post_feedforward_layernorm.weight", &[4], &[1.0; 4]),
            tensor("model.language_model.layers.0.mlp.gate_proj.weight", &[8, 4], &[0.0; 32]),
            tensor("model.language_model.layers.0.mlp.up_proj.weight", &[8, 4], &[0.0; 32]),
            tensor("model.language_model.layers.0.mlp.down_proj.weight", &[4, 8], &[0.0; 32]),
            tensor("model.language_model.norm.weight", &[4], &[1.0; 4]),
        ],
    );

    let model = load_transformer_state_model_from_det_num_wgt_path(&model_dir).unwrap();
    assert!(model.embedding_table.is_none());
    assert!(model.embedding_source.is_some());
    assert_eq!(model.embedding_source.as_ref().unwrap().scale(), 2.0);
    match model.logits_projection {
        Gemma4LogitsProjection::TiedEmbedding(ref matrix) => {
            assert_eq!(matrix.rows, 3);
            assert_eq!(matrix.cols, 4);
        }
        Gemma4LogitsProjection::UntiedLmHead(_) => panic!("expected tied embedding projection"),
    }
}

#[test]
fn deterministic_loader_rejects_unsupported_artifact_version() {
    let model_dir = create_temp_dir("det-loader-bad-version");
    write_config(
        &model_dir,
        r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 2,
    "tie_word_embeddings": true,
    "vocab_size": 3
  }
}"#,
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&(DET_WGT_ARTIFACT_FORMAT_VERSION + 1).to_le_bytes());
    bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    fs::write(model_dir.join("model.detwgt"), bytes).unwrap();

    let error =
        load_transformer_state_model_from_det_num_wgt_path(&model_dir).expect_err("bad version");
    assert!(error.to_string().contains("unsupported deterministic artifact format version"));
}

#[test]
fn deterministic_mode_matches_fp32_path_on_representable_fixture() {
    let fp32_dir = create_temp_dir("det-parity-fp32");
    let det_dir = create_temp_dir("det-parity-det");
    write_config(
        &fp32_dir,
        r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 2,
    "tie_word_embeddings": false,
    "vocab_size": 3
  }
}"#,
    );
    fs::copy(fp32_dir.join("config.json"), det_dir.join("config.json")).unwrap();

    let tensors = vec![
        tensor("model.language_model.embed_tokens.weight", &[3, 4], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        tensor("model.language_model.layers.0.self_attn.q_proj.weight", &[4, 4], &[0.0; 16]),
        tensor("model.language_model.layers.0.self_attn.k_proj.weight", &[2, 4], &[0.0; 8]),
        tensor("model.language_model.layers.0.self_attn.v_proj.weight", &[2, 4], &[0.0; 8]),
        tensor("model.language_model.layers.0.self_attn.o_proj.weight", &[4, 4], &[0.0; 16]),
        tensor("model.language_model.layers.0.self_attn.q_norm.weight", &[2], &[1.0, 1.0]),
        tensor("model.language_model.layers.0.self_attn.k_norm.weight", &[2], &[1.0, 1.0]),
        tensor("model.language_model.layers.0.input_layernorm.weight", &[4], &[1.0; 4]),
        tensor("model.language_model.layers.0.post_attention_layernorm.weight", &[4], &[1.0; 4]),
        tensor("model.language_model.layers.0.pre_feedforward_layernorm.weight", &[4], &[1.0; 4]),
        tensor("model.language_model.layers.0.post_feedforward_layernorm.weight", &[4], &[1.0; 4]),
        tensor("model.language_model.layers.0.mlp.gate_proj.weight", &[8, 4], &[0.0; 32]),
        tensor("model.language_model.layers.0.mlp.up_proj.weight", &[8, 4], &[0.0; 32]),
        tensor("model.language_model.layers.0.mlp.down_proj.weight", &[4, 8], &[0.0; 32]),
        tensor("model.language_model.norm.weight", &[4], &[1.0; 4]),
        tensor("model.language_model.lm_head.weight", &[3, 4], &[0.0; 12]),
    ];
    write_fp32_model_file(&fp32_dir.join("model.safetensors"), &tensors);
    write_detwgt_file(&det_dir.join("model.detwgt"), &tensors);

    let model_spec = ModelSpec {
        model_id: "gemma-4-test".to_string(),
        tokenizer_path: "tokenizer.json".into(),
        chat_template: "{{ messages[0].content }}".to_string(),
        bos_token: None,
        eos_token: None,
        unk_token: Some("<unk>".to_string()),
    };
    let tokenizer = test_tokenizer();
    let fp32_model = load_transformer_state_model_from_gemma_model_path(&fp32_dir).unwrap();
    let det_model = load_transformer_state_model_from_det_num_wgt_path(&det_dir).unwrap();

    let fp32_state = run_inference(
        &InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        },
        &model_spec,
        &tokenizer,
        &fp32_model,
    )
    .unwrap();
    let det_state = run_inference(
        &InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        },
        &model_spec,
        &tokenizer,
        &det_model,
    )
    .unwrap();

    assert_eq!(det_state.output_decode.generated_token_ids, fp32_state.output_decode.generated_token_ids);
    assert_eq!(det_state.output_decode.generated_text, fp32_state.output_decode.generated_text);
    assert_eq!(
        det_state.transformer_state_transition.prefill_logits.final_logits_sha256,
        fp32_state.transformer_state_transition.prefill_logits.final_logits_sha256
    );
    assert_eq!(
        det_state.output_decode.generated_token_ids_sha256,
        fp32_state.output_decode.generated_token_ids_sha256
    );
}

fn create_temp_dir(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raster-inference-{label}-{unique}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_config(dir: &Path, config: &str) {
    fs::write(dir.join("config.json"), config).unwrap();
}

fn tensor(name: &str, shape: &[usize], values: &[f32]) -> FixtureTensor {
    FixtureTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        values: values.to_vec(),
    }
}

fn write_fp32_model_file(path: &Path, tensors: &[FixtureTensor]) {
    let mut byte_storage = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        byte_storage.push(
            tensor
                .values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
    }
    let mut metadata = BTreeMap::new();
    for (tensor, bytes) in tensors.iter().zip(byte_storage.iter()) {
        metadata.insert(
            tensor.name.clone(),
            TensorView::new(safetensors::Dtype::F32, tensor.shape.clone(), bytes).unwrap(),
        );
    }
    serialize_to_file(&metadata, &None, path).unwrap();
}

fn write_detwgt_file(path: &Path, tensors: &[FixtureTensor]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

    for tensor in tensors {
        let name_bytes = tensor.name.as_bytes();
        let payload = tensor
            .values
            .iter()
            .flat_map(|value| wgt_to_le_bytes(f32_to_wgt(*value)))
            .collect::<Vec<_>>();
        let element_count = tensor.shape.iter().product::<usize>() as u64;

        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in &tensor.shape {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&element_count.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
    }

    fs::write(path, bytes).unwrap();
}

fn test_tokenizer() -> Tokenizer {
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
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

struct FixtureTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}
