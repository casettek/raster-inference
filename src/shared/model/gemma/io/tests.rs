use super::{
    decode_embedding_rows_for_token_ids_from_det_num, decode_matrix_row_from_source,
    decode_matrix_slice_from_source, decode_single_scalar, decode_vector,
    load_transformer_state_model_from_det_num_wgt_path, parse_gemma_tokenizer_spec_bytes,
    parse_safetensors_metadata,
};
use crate::io::decode_matrix_slice;
use crate::shared::model::transformer::DetNumTensorSliceSource;
use crate::shared::model::transformer::Gemma4LogitsProjection;
use crate::shared::numerics::det_num::{
    f32_to_wgt, wgt_to_le_bytes, Act, DetWgtElementWidth, DetWgtTensorSpec, DET_NUM_SPEC_VERSION,
    DET_WGT_ARTIFACT_FORMAT_VERSION, DET_WGT_ARTIFACT_MAGIC,
};
use memmap2::Mmap;
use safetensors::tensor::{serialize_to_file, TensorView};
use std::{
    collections::BTreeMap,
    fs,
    fs::File,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

struct FixtureTensor {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[test]
fn parse_gemma_tokenizer_spec_accepts_supported_subset() {
    let spec = parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
        .expect("supported tokenizer should parse");

    assert_eq!(spec.token_id("<unk>"), Some(0));
    assert_eq!(spec.token_id("ab"), Some(3));
    assert_eq!(spec.merges.len(), 1);
    assert_eq!(spec.merges[0].left, "a");
    assert_eq!(spec.merges[0].right, "b");
    assert!(spec.byte_fallback);
}

#[test]
fn parse_gemma_tokenizer_spec_rejects_unsupported_model_type() {
    let json = minimal_gemma_tokenizer_json().replace("\"type\":\"BPE\"", "\"type\":\"WordLevel\"");
    let error = parse_gemma_tokenizer_spec_bytes(json.as_bytes())
        .expect_err("unsupported tokenizer model should fail");

    assert!(error
        .to_string()
        .contains("unsupported Gemma tokenizer model"));
}

#[test]
fn decode_matrix_slice_copies_f32_rows_without_scalar_loop() {
    let bytes = f32_to_bytes(&[
        1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
    ]);
    let tensor = TensorView::new(safetensors::Dtype::F32, vec![3, 4], &bytes).unwrap();

    let matrix = decode_matrix_slice(&tensor, 1, 2, 1, 2).unwrap();

    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.cols, 2);
    assert_eq!(matrix.values, vec![20.0, 30.0, 200.0, 300.0]);
}

#[test]
fn decode_vector_and_scalar_copy_f32_values_directly() {
    let vector_bytes = f32_to_bytes(&[1.25, -2.5, 3.75]);
    let vector = TensorView::new(safetensors::Dtype::F32, vec![3], &vector_bytes).unwrap();
    let scalar_bytes = f32_to_bytes(&[9.5]);
    let scalar = TensorView::new(safetensors::Dtype::F32, vec![1], &scalar_bytes).unwrap();

    assert_eq!(decode_vector(&vector).unwrap(), vec![1.25, -2.5, 3.75]);
    assert_eq!(decode_single_scalar(&scalar).unwrap(), 9.5);
}

#[test]
fn det_matrix_loader_borrows_aligned_mmap_payloads_and_copies_misaligned() {
    let model_dir = create_test_model_dir("det-matrix-mmap-view");
    let weights_path = model_dir.join("weights.detwgt");
    let payload_values: Vec<i32> = vec![1, -2, 3, 4, -5, 6];
    let mut bytes = vec![0u8; 8]; // aligned payload offset
    for value in &payload_values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&weights_path, &bytes).unwrap();
    let file = File::open(&weights_path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { Mmap::map(&file) }.unwrap());

    let aligned_source = DetNumTensorSliceSource {
        weights_path: weights_path.clone(),
        total_rows: 2,
        total_cols: 3,
        data_offset: 8,
        element_width: DetWgtElementWidth::I32,
        row_offset: 0,
        row_count: 2,
        col_offset: 0,
        col_count: 3,
    };
    let aligned = super::decode_det_num_matrix_from_source_shared(&aligned_source, &mmap)
        .expect("aligned matrix should load");
    assert!(aligned.values.is_mmap_backed());
    assert_eq!(aligned.values.to_widened_vec(), payload_values);

    // Misaligned payload start (detwgt v1 packs payloads after
    // variable-length names): falls back to an owned copy.
    let mut misaligned_bytes = vec![0u8; 6];
    for value in &payload_values {
        misaligned_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let misaligned_path = model_dir.join("misaligned.detwgt");
    fs::write(&misaligned_path, &misaligned_bytes).unwrap();
    let misaligned_file = File::open(&misaligned_path).unwrap();
    let misaligned_mmap = std::sync::Arc::new(unsafe { Mmap::map(&misaligned_file) }.unwrap());
    let misaligned_source = DetNumTensorSliceSource {
        weights_path: misaligned_path,
        total_rows: 2,
        total_cols: 3,
        data_offset: 6,
        element_width: DetWgtElementWidth::I32,
        row_offset: 0,
        row_count: 2,
        col_offset: 0,
        col_count: 3,
    };
    let copied =
        super::decode_det_num_matrix_from_source_shared(&misaligned_source, &misaligned_mmap)
            .expect("misaligned matrix should load via copy");
    assert!(!copied.values.is_mmap_backed());
    assert_eq!(copied.values.to_widened_vec(), payload_values);

    // Column slices are not contiguous in the file: copy fallback.
    let sliced_source = DetNumTensorSliceSource {
        weights_path,
        total_rows: 2,
        total_cols: 3,
        data_offset: 8,
        element_width: DetWgtElementWidth::I32,
        row_offset: 0,
        row_count: 2,
        col_offset: 1,
        col_count: 2,
    };
    let sliced = super::decode_det_num_matrix_from_source_shared(&sliced_source, &mmap)
        .expect("column slice should load via copy");
    assert!(!sliced.values.is_mmap_backed());
    assert_eq!(sliced.values.to_widened_vec(), vec![-2, 3, -5, 6]);
}

#[test]
fn decode_det_num_embedding_rows_preserves_raw_act_bits() {
    let model_dir = create_test_model_dir("det-embedding-raw-act");
    let weights_path = model_dir.join("embedding.detwgt");
    let canonical = Act::from_bits((1 << 24) + 1);
    fs::write(&weights_path, canonical.to_bits().to_le_bytes()).unwrap();
    let file = File::open(&weights_path).unwrap();
    let mmap = unsafe { Mmap::map(&file) }.unwrap();
    let source = DetNumTensorSliceSource {
        weights_path,
        total_rows: 1,
        total_cols: 1,
        data_offset: 0,
        element_width: DetWgtElementWidth::I32,
        row_offset: 0,
        row_count: 1,
        col_offset: 0,
        col_count: 1,
    };

    let decoded =
        decode_embedding_rows_for_token_ids_from_det_num(&source, &[0], 1, 1.0, &mmap).unwrap();

    assert_eq!(decoded.det_values().unwrap()[0][0], canonical);
    // Single-track deterministic embedding: no f32 mirror is materialized.
    assert!(decoded.as_f32_slice().is_empty());
}

#[test]
fn decode_lazy_f32_ple_rows_and_slices() {
    let model_dir = create_test_model_dir("f32-lazy-slice");
    let tensor_name = "model.language_model.embed_tokens_per_layer.weight";
    let values = [
        1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
    ];
    write_model_file(
        &model_dir,
        &[FixtureTensor {
            name: tensor_name.to_string(),
            shape: vec![3, 4],
            bytes: f32_to_bytes(&values),
        }],
    );
    let model_path = model_dir.join("model.safetensors");
    let file = File::open(&model_path).unwrap();
    let mmap = unsafe { Mmap::map(&file) }.unwrap();
    let metadata = parse_safetensors_metadata(&mmap, &model_path).unwrap();
    let tensor = metadata.get(tensor_name).unwrap();
    let source = crate::shared::model::transformer::GemmaTensorSliceSource {
        weights_path: model_path,
        dtype: tensor.dtype,
        total_rows: tensor.shape[0],
        total_cols: tensor.shape[1],
        data_offset: tensor.data_offset,
        row_offset: 1,
        row_count: 2,
        col_offset: 1,
        col_count: 2,
    };

    let row = decode_matrix_row_from_source(&source, 1, &mmap).unwrap();
    let matrix = decode_matrix_slice_from_source(&source, &mmap).unwrap();

    assert_eq!(row, vec![200.0, 300.0]);
    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.cols, 2);
    assert_eq!(matrix.values, vec![20.0, 30.0, 200.0, 300.0]);
}

#[test]
fn resolve_layer_weights_only_attaches_raw_det_weights_for_det_attention_and_mlp_projections() {
    let det_dir = create_test_model_dir("det-mlp-proj");
    let config = r#"{
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
}"#;
    write_config(&det_dir, config);

    let q_proj_values = [
        0.5, -0.25, 0.125, 0.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25, -0.125, 0.0,
        0.0,
    ];
    let k_proj_values = [0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.125, 0.0];
    let v_proj_values = [1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0];
    let o_proj_values = [
        -0.5, 0.25, -0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25, -0.125,
        0.0,
    ];
    let lm_head_values = [
        0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0,
    ];
    let gate_proj_values = [
        0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25, -0.125,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let up_proj_values = [
        0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75,
        0.0, 0.5, 0.25, -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let down_proj_values = [
        0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25, -0.125,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    write_detwgt_file(
        &det_dir,
        &[
            det_matrix_tensor(
                "model.language_model.embed_tokens.weight",
                &[3, 4],
                &[0.0; 12],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                &[4, 4],
                &q_proj_values,
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.k_proj.weight",
                &[2, 4],
                &k_proj_values,
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.v_proj.weight",
                &[2, 4],
                &v_proj_values,
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.o_proj.weight",
                &[4, 4],
                &o_proj_values,
            ),
            det_vector_tensor(
                "model.language_model.layers.0.self_attn.q_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.self_attn.k_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.input_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.post_attention_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                &[8, 4],
                &gate_proj_values,
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.mlp.up_proj.weight",
                &[8, 4],
                &up_proj_values,
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.mlp.down_proj.weight",
                &[4, 8],
                &down_proj_values,
            ),
            det_vector_tensor("model.language_model.norm.weight", &[4], &[1.0; 4]),
            det_matrix_tensor(
                "model.language_model.lm_head.weight",
                &[3, 4],
                &lm_head_values,
            ),
        ],
    );

    let det_model = load_transformer_state_model_from_det_num_wgt_path(&det_dir).unwrap();
    let det_layer = super::resolve_layer_weights(&det_model.layers[0]).unwrap();
    let det_lm_head = match &det_model.logits_projection {
        Gemma4LogitsProjection::UntiedLmHead { weight, det_weight } => {
            assert_eq!(weight.rows, 3);
            assert_eq!(weight.cols, 4);
            det_weight
                .as_ref()
                .expect("deterministic model should retain raw lm_head weights")
        }
        other => panic!("expected untied lm head, got {other:?}"),
    };

    let det_q_proj = det_layer
        .q_proj_det
        .expect("deterministic model should retain raw q_proj weights");
    let det_k_proj = det_layer
        .k_proj_det
        .expect("deterministic model should retain raw k_proj weights");
    let det_v_proj = det_layer
        .v_proj_det
        .expect("deterministic model should retain raw v_proj weights");
    let det_o_proj = det_layer
        .o_proj_det
        .expect("deterministic model should retain raw o_proj weights");
    let det_gate_proj = det_layer
        .gate_proj_det
        .expect("deterministic model should retain raw gate_proj weights");
    let det_up_proj = det_layer
        .up_proj_det
        .expect("deterministic model should retain raw up_proj weights");
    let det_down_proj = det_layer
        .down_proj_det
        .expect("deterministic model should retain raw down_proj weights");
    assert_eq!(det_q_proj.rows, 4);
    assert_eq!(det_q_proj.cols, 4);
    assert_eq!(
        det_q_proj.values.wgt_bits(0),
        f32_to_wgt(q_proj_values[0]).to_bits()
    );
    assert_eq!(
        det_q_proj.values.wgt_bits(1),
        f32_to_wgt(q_proj_values[1]).to_bits()
    );
    assert_eq!(det_k_proj.rows, 2);
    assert_eq!(det_k_proj.cols, 4);
    assert_eq!(
        det_k_proj.values.wgt_bits(0),
        f32_to_wgt(k_proj_values[0]).to_bits()
    );
    assert_eq!(
        det_k_proj.values.wgt_bits(1),
        f32_to_wgt(k_proj_values[1]).to_bits()
    );
    assert_eq!(det_v_proj.rows, 2);
    assert_eq!(det_v_proj.cols, 4);
    assert_eq!(
        det_v_proj.values.wgt_bits(0),
        f32_to_wgt(v_proj_values[0]).to_bits()
    );
    assert_eq!(
        det_v_proj.values.wgt_bits(1),
        f32_to_wgt(v_proj_values[1]).to_bits()
    );
    assert_eq!(det_o_proj.rows, 4);
    assert_eq!(det_o_proj.cols, 4);
    assert_eq!(
        det_o_proj.values.wgt_bits(0),
        f32_to_wgt(o_proj_values[0]).to_bits()
    );
    assert_eq!(
        det_o_proj.values.wgt_bits(1),
        f32_to_wgt(o_proj_values[1]).to_bits()
    );
    assert_eq!(det_lm_head.rows, 3);
    assert_eq!(det_lm_head.cols, 4);
    assert_eq!(
        det_lm_head.values.wgt_bits(0),
        f32_to_wgt(lm_head_values[0]).to_bits()
    );
    assert_eq!(
        det_lm_head.values.wgt_bits(1),
        f32_to_wgt(lm_head_values[1]).to_bits()
    );
    assert_eq!(det_gate_proj.rows, 8);
    assert_eq!(det_gate_proj.cols, 4);
    assert_eq!(
        det_gate_proj.values.wgt_bits(0),
        f32_to_wgt(gate_proj_values[0]).to_bits()
    );
    assert_eq!(
        det_gate_proj.values.wgt_bits(1),
        f32_to_wgt(gate_proj_values[1]).to_bits()
    );
    assert_eq!(det_up_proj.rows, 8);
    assert_eq!(det_up_proj.cols, 4);
    assert_eq!(
        det_up_proj.values.wgt_bits(0),
        f32_to_wgt(up_proj_values[0]).to_bits()
    );
    assert_eq!(
        det_up_proj.values.wgt_bits(1),
        f32_to_wgt(up_proj_values[1]).to_bits()
    );
    assert_eq!(det_down_proj.rows, 4);
    assert_eq!(det_down_proj.cols, 8);
    assert_eq!(
        det_down_proj.values.wgt_bits(0),
        f32_to_wgt(down_proj_values[0]).to_bits()
    );
    assert_eq!(
        det_down_proj.values.wgt_bits(1),
        f32_to_wgt(down_proj_values[1]).to_bits()
    );
}

#[test]
fn resolve_layer_weights_and_ple_globals_attach_raw_det_ple_weights() {
    let det_dir = create_test_model_dir("det-ple-proj");
    let config = r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "hidden_size_per_layer_input": 2,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 2,
    "tie_word_embeddings": false,
    "vocab_size": 3,
    "vocab_size_per_layer_input": 3
  }
}"#;
    write_config(&det_dir, config);

    let input_gate_values = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let layer_projection_values = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let global_projection_values = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let mut det_tensors = vec![det_matrix_tensor(
        "model.language_model.embed_tokens.weight",
        &[3, 4],
        &[0.0; 12],
    )];
    det_tensors.extend([
        det_matrix_tensor(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            &[4, 4],
            &[0.0; 16],
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.self_attn.k_proj.weight",
            &[2, 4],
            &[0.0; 8],
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.self_attn.v_proj.weight",
            &[2, 4],
            &[0.0; 8],
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.self_attn.o_proj.weight",
            &[4, 4],
            &[0.0; 16],
        ),
        det_vector_tensor(
            "model.language_model.layers.0.self_attn.q_norm.weight",
            &[2],
            &[1.0, 1.0],
        ),
        det_vector_tensor(
            "model.language_model.layers.0.self_attn.k_norm.weight",
            &[2],
            &[1.0, 1.0],
        ),
        det_vector_tensor(
            "model.language_model.layers.0.input_layernorm.weight",
            &[4],
            &[1.0; 4],
        ),
        det_vector_tensor(
            "model.language_model.layers.0.post_attention_layernorm.weight",
            &[4],
            &[1.0; 4],
        ),
        det_vector_tensor(
            "model.language_model.layers.0.pre_feedforward_layernorm.weight",
            &[4],
            &[1.0; 4],
        ),
        det_vector_tensor(
            "model.language_model.layers.0.post_feedforward_layernorm.weight",
            &[4],
            &[1.0; 4],
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.mlp.gate_proj.weight",
            &[8, 4],
            &[0.0; 32],
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.mlp.up_proj.weight",
            &[8, 4],
            &[0.0; 32],
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.mlp.down_proj.weight",
            &[4, 8],
            &[0.0; 32],
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.per_layer_input_gate.weight",
            &[2, 4],
            &input_gate_values,
        ),
        det_matrix_tensor(
            "model.language_model.layers.0.per_layer_projection.weight",
            &[4, 2],
            &layer_projection_values,
        ),
        det_vector_tensor(
            "model.language_model.layers.0.post_per_layer_input_norm.weight",
            &[4],
            &[1.0; 4],
        ),
        det_matrix_tensor(
            "model.language_model.embed_tokens_per_layer.weight",
            &[3, 2],
            &[0.0; 6],
        ),
        det_matrix_tensor(
            "model.language_model.per_layer_model_projection.weight",
            &[2, 4],
            &global_projection_values,
        ),
        det_vector_tensor(
            "model.language_model.per_layer_projection_norm.weight",
            &[2],
            &[1.0, 1.0],
        ),
        det_vector_tensor("model.language_model.norm.weight", &[4], &[1.0; 4]),
        det_matrix_tensor("model.language_model.lm_head.weight", &[3, 4], &[0.0; 12]),
    ]);
    write_detwgt_file(&det_dir, &det_tensors);

    let det_model = load_transformer_state_model_from_det_num_wgt_path(&det_dir).unwrap();
    let det_layer = super::resolve_layer_weights(&det_model.layers[0]).unwrap();
    let det_ple = det_layer.ple.expect("det PLE should resolve");
    let det_global_projection = super::materialize_det_num_ple_model_projection(
        det_model
            .ple_global
            .as_ref()
            .expect("det PLE globals should load"),
        0,
    )
    .unwrap()
    .expect("det PLE model projection should materialize");

    assert_eq!(det_ple.input_gate_det.as_ref().unwrap().rows, 2);
    assert_eq!(det_ple.input_gate_det.as_ref().unwrap().cols, 4);
    assert_eq!(
        det_ple.input_gate_det.as_ref().unwrap().values.wgt_bits(0),
        f32_to_wgt(input_gate_values[0]).to_bits()
    );
    assert_eq!(det_ple.layer_projection_det.as_ref().unwrap().rows, 4);
    assert_eq!(det_ple.layer_projection_det.as_ref().unwrap().cols, 2);
    assert_eq!(
        det_ple
            .layer_projection_det
            .as_ref()
            .unwrap()
            .values
            .wgt_bits(0),
        f32_to_wgt(layer_projection_values[0]).to_bits()
    );
    assert_eq!(det_global_projection.rows, 2);
    assert_eq!(det_global_projection.cols, 4);
    assert_eq!(
        det_global_projection.values.wgt_bits(0),
        f32_to_wgt(global_projection_values[0]).to_bits()
    );
}

fn create_test_model_dir(suffix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raster-inference-{suffix}-{unique}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_config(dir: &Path, config: &str) {
    fs::write(dir.join("config.json"), config).unwrap();
}

fn write_model_file(dir: &Path, tensors: &[FixtureTensor]) {
    let mut metadata = BTreeMap::new();
    for tensor in tensors {
        metadata.insert(
            tensor.name.clone(),
            TensorView::new(safetensors::Dtype::F32, tensor.shape.clone(), &tensor.bytes).unwrap(),
        );
    }
    serialize_to_file(&metadata, &None, &dir.join("model.safetensors")).unwrap();
}

#[test]
fn det_artifact_loader_rejects_v0_format_version() {
    let dir = create_test_model_dir("det-v0-format");
    let path = dir.join("model.detwgt");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    let error = match super::DetNumTensorReader::load_artifact(&path) {
        Ok(_) => panic!("v0 format version should fail closed"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("unsupported deterministic artifact format version"));
}

#[test]
fn det_artifact_loader_rejects_v0_spec_version() {
    let dir = create_test_model_dir("det-v0-spec");
    let path = dir.join("model.detwgt");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    let error = match super::DetNumTensorReader::load_artifact(&path) {
        Ok(_) => panic!("v0 spec version should fail closed"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("unsupported deterministic artifact det_num spec version"));
}

fn single_tensor_detwgt_bytes(name: &str, shape: &[u64], max_row_mass: u64) -> Vec<u8> {
    use crate::shared::numerics::det_num::artifact::{
        file_header_bytes, padding_for_offset, tensor_header_bytes,
    };
    let shape_usize = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
    let element_count: u64 = shape.iter().product();
    let mut bytes = file_header_bytes(1);
    bytes.extend_from_slice(
        &tensor_header_bytes(name, &shape_usize, DetWgtElementWidth::I32, max_row_mass).unwrap(),
    );
    bytes.resize(bytes.len() + padding_for_offset(bytes.len() as u64), 0);
    bytes.extend(std::iter::repeat(0u8).take((element_count * 4) as usize));
    bytes
}

#[test]
fn det_artifact_loader_rejects_matrix_row_mass_violating_bound() {
    let dir = create_test_model_dir("det-row-mass-violation");
    let path = dir.join("model.detwgt");
    fs::write(
        &path,
        single_tensor_detwgt_bytes("tensor", &[1, 2], super::DET_WGT_ROW_MASS_LIMIT),
    )
    .unwrap();

    let error = match super::DetNumTensorReader::load_artifact(&path) {
        Ok(_) => panic!("matrix row mass at the limit should fail closed"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("violates the conversion-time overflow bound"));
}

#[test]
fn det_artifact_loader_accepts_rank1_tensor_with_unbounded_mass() {
    let dir = create_test_model_dir("det-row-mass-rank1");
    let path = dir.join("model.detwgt");
    // Rank-1 tensors are elementwise operands (e.g. RMSNorm gains); the
    // MAC overflow bound does not apply to their recorded mass.
    fs::write(
        &path,
        single_tensor_detwgt_bytes("norm.weight", &[2], super::DET_WGT_ROW_MASS_LIMIT * 2),
    )
    .unwrap();

    super::DetNumTensorReader::load_artifact(&path)
        .map(|_| ())
        .expect("rank-1 tensor with large recorded mass should load");
}

fn write_detwgt_file(dir: &Path, tensors: &[FixtureTensor]) {
    let specs = tensors
        .iter()
        .map(|tensor| DetWgtTensorSpec {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            wgt_bits: tensor
                .bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        })
        .collect::<Vec<_>>();
    let bytes = crate::shared::numerics::det_num::encode_det_wgt_artifact(&specs).unwrap();
    fs::write(dir.join("model.detwgt"), bytes).unwrap();
}

fn det_matrix_tensor(name: &str, shape: &[usize], values: &[f32]) -> FixtureTensor {
    FixtureTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        bytes: det_wgt_bytes(values),
    }
}

fn det_vector_tensor(name: &str, shape: &[usize], values: &[f32]) -> FixtureTensor {
    FixtureTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        bytes: det_wgt_bytes(values),
    }
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn det_wgt_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| wgt_to_le_bytes(f32_to_wgt(*value)))
        .collect()
}

fn minimal_gemma_tokenizer_json() -> String {
    serde_json::json!({
        "version": "1.0",
        "added_tokens": [
            {
                "id": 4,
                "content": "<bos>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ],
        "normalizer": {
            "type": "Replace",
            "pattern": { "String": " " },
            "content": "▁"
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": { "String": " " },
            "behavior": "MergedWithPrevious",
            "invert": false
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [],
            "pair": [],
            "special_tokens": {}
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": { "String": "▁" },
                    "content": " "
                },
                {
                    "type": "ByteFallback"
                },
                {
                    "type": "Fuse"
                }
            ]
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": {
                "<unk>": 0,
                "a": 1,
                "b": 2,
                "ab": 3,
                "<bos>": 4
            },
            "merges": [
                ["a", "b"]
            ]
        }
    })
    .to_string()
}
