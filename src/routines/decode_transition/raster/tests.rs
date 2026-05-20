use super::{
    append_decode_kv_cache_ref, compute_next_decode_layer, finalize_decode_layer_state,
    init_decode_logits_projection, init_decode_transition_state, init_decode_transition_store,
    main, materialize_decode_layer_cache_from_store, prepare_next_decode_layer_context,
    register_decode_layer_cache, run, DecodeLayerCacheSlot, RasterDecodeTransitionInputRoots,
};
use crate::decode_transition::raster::auth_source::AuthenticatedGemmaDecodeTransitionSource;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    token_id_leaf, RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots,
    RasterSelectedTokenRef, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, TransformerDecodeState,
};
use crate::shared::numerics::det_num::{Acc, Act, Wgt};
use crate::shared::raster_kernels::transformer::{
    RasterActivationRow, RasterAttentionHeadSequence, RasterKvCache,
};
use crate::shared::tensors::raster_row_store::RasterTensorId;
use crate::RasterSizingControls;
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[test]
fn raster_decode_transition_matches_deterministic_no_ple() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = decode_state_with_cache(1);

    let raster =
        run(decode_state.clone(), 1, &source, raster_sizing(1)).expect("raster decode should run");
    let deterministic = crate::decode_transition::run_with_mode(
        decode_state,
        1,
        &model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("deterministic decode should run");

    assert_eq!(raster.activation_state, deterministic.activation_state);
    assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
    assert_eq!(
        raster.transformer_decode_state,
        deterministic.transformer_decode_state
    );
}

#[test]
fn root_backed_main_reads_selected_token_ref() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = decode_state_with_cache(1);
    let (roots, selected_token_ref) =
        selected_token_ref("decode.transition.selected", 1).expect("selected token ref");

    let raster = main(
        RasterDecodeTransitionInputRoots {
            artifact_store_roots: roots,
            transformer_decode_state: decode_state.clone(),
            selected_token_ref,
            decode_transition_source_root: source.static_source_root(),
            output_source_prefix: "decode.transition.root-backed".to_string(),
            raster_sizing: raster_sizing(1),
        },
        &source,
    )
    .expect("root-backed raster decode should run")
    .transition_result;
    let deterministic = crate::decode_transition::run_with_mode(
        decode_state,
        1,
        &model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("deterministic decode should run");

    assert_eq!(raster.activation_state, deterministic.activation_state);
    assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
    assert_eq!(
        raster.transformer_decode_state,
        deterministic.transformer_decode_state
    );
}

#[test]
fn root_backed_main_rejects_missing_selected_token_root() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let (_roots, selected_token_ref) =
        selected_token_ref("decode.transition.missing.selected", 1).expect("selected token ref");

    let error = main(
        RasterDecodeTransitionInputRoots {
            artifact_store_roots: RasterArtifactStoreRoots::default(),
            transformer_decode_state: decode_state_with_cache(1),
            selected_token_ref,
            decode_transition_source_root: source.static_source_root(),
            output_source_prefix: "decode.transition.missing".to_string(),
            raster_sizing: raster_sizing(1),
        },
        &source,
    )
    .expect_err("missing selected-token root should fail");

    assert!(error.to_string().contains("not present"));
}

#[test]
fn root_backed_main_threads_roots_without_local_tensor_store() {
    let source = include_str!("tiles.rs");
    let main_start = source
        .find("pub fn main(\n    input_roots: RasterDecodeTransitionInputRoots,")
        .expect("decode transition main should exist");
    let after_main = &source[main_start..];
    let next_sequence = after_main
        .find("\n#[sequence]\nfn run_basic_decode_layer")
        .expect("legacy helper should follow root-backed main block");
    let main_and_root_helpers = &after_main[..next_sequence];

    assert!(main_and_root_helpers.contains("compute_next_decode_layer_with_roots"));
    assert!(
        !main_and_root_helpers.contains("init_decode_transition_store"),
        "proof-shaped decode transition main must thread artifact roots instead of creating a tensor store"
    );
    assert!(
        !main_and_root_helpers.contains("AuthenticatedRasterTensorStore::new()"),
        "proof-shaped decode transition main must not allocate a local tensor store"
    );
}

#[test]
fn raster_decode_transition_matches_across_projection_and_attention_chunks() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = decode_state_with_cache(3);
    let deterministic = crate::decode_transition::run_with_mode(
        decode_state.clone(),
        1,
        &model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("deterministic decode should run");

    for sizing in [
        raster_sizing_with_attention(1, 1),
        raster_sizing_with_attention(2, 1),
        raster_sizing_with_attention(8, 2),
    ] {
        let raster =
            run(decode_state.clone(), 1, &source, sizing).expect("raster decode should run");

        assert_eq!(raster.activation_state, deterministic.activation_state);
        assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
        assert_eq!(
            raster.transformer_decode_state,
            deterministic.transformer_decode_state
        );
    }
}

#[test]
fn raster_decode_rejects_zero_sizing_controls() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = decode_state_with_cache(1);

    let projection_error = run(
        decode_state.clone(),
        1,
        &source,
        raster_sizing_with_attention(0, 1),
    )
    .expect_err("zero projection chunk should fail");
    assert!(projection_error
        .to_string()
        .contains("projection rows per tile"));

    let attention_error = run(decode_state, 1, &source, raster_sizing_with_attention(1, 0))
        .expect_err("zero attention chunk should fail");
    assert!(attention_error
        .to_string()
        .contains("attention KV rows per tile"));
}

#[test]
fn decode_recursive_state_serializes_refs_and_builders() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = decode_state_with_cache(2);
    let mut store = init_decode_transition_store();
    let state =
        init_decode_transition_state(&mut store, decode_state, 1, &source, raster_sizing(1))
            .expect("state should initialize");
    let encoded = serde_json::to_string(&state).expect("state should serialize");

    assert!(encoded.contains("artifact_store_roots"));
    assert!(encoded.contains("decode_input_ref"));
    assert!(encoded.contains("current_activation_ref"));
    assert!(encoded.contains("original_layer_caches"));
    assert!(encoded.contains("Ref"));
    assert!(!encoded.contains("decode_input_activation"));
    assert!(!encoded.contains("current_activation\":["));
    assert!(!encoded.contains("\"keys\":[[["));
    assert!(!encoded.contains("\"values\":[[["));

    let normalized = RasterActivationRow::from_acts(vec![Act::from_num(0.0), Act::from_num(0.0)]);
    let logits_state =
        init_decode_logits_projection(&mut store, normalized, &source, 1).expect("logits init");
    let encoded = serde_json::to_string(&logits_state).expect("logits state should serialize");

    assert!(encoded.contains("output_builder_ref"));
    assert!(!encoded.contains("logit_bits"));
}

#[test]
fn decode_layer_context_serializes_cache_refs_without_materialized_rows() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = decode_state_with_cache(2);
    let mut store = init_decode_transition_store();
    let state =
        init_decode_transition_state(&mut store, decode_state, 1, &source, raster_sizing(1))
            .expect("state should initialize");

    let context =
        prepare_next_decode_layer_context(&state, &source).expect("context should prepare");
    let encoded = serde_json::to_string(&context).expect("context should serialize");

    assert!(encoded.contains("cache_slot"));
    assert!(encoded.contains("Ref"));
    assert!(!encoded.contains("\"keys\":[[["));
    assert!(!encoded.contains("\"values\":[[["));
    assert!(!encoded.contains("act_bits"));
    assert_eq!(store.materialization_counts().1, 0);
}

#[test]
fn decode_layer_replay_avoids_kv_materialization_until_public_boundary() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = decode_state_with_cache(2);
    let mut store = init_decode_transition_store();
    let state =
        init_decode_transition_state(&mut store, decode_state, 1, &source, raster_sizing(1))
            .expect("state should initialize");
    assert_eq!(store.materialization_counts().1, 0);

    let (_done, state) =
        compute_next_decode_layer(state, &source, &mut store).expect("layer should compute");
    assert_eq!(
        store.materialization_counts().1,
        0,
        "normal decode layer replay should not materialize full KV caches"
    );

    finalize_decode_layer_state(&store, state).expect("public layer output should finalize");
    assert!(
        store.materialization_counts().1 > 0,
        "public/cache compatibility boundary should materialize KV caches"
    );
}

#[test]
fn decode_ref_cache_append_handles_empty_and_sliding_windows() {
    let mut store = init_decode_transition_store();
    let key_ref = store
        .insert_attention_heads(
            RasterTensorId::new("test.empty.key").expect("key id"),
            heads_from_rows(&[&[7]]),
        )
        .expect("key ref");
    let value_ref = store
        .insert_attention_heads(
            RasterTensorId::new("test.empty.value").expect("value id"),
            heads_from_rows(&[&[8]]),
        )
        .expect("value ref");
    let empty_slot = DecodeLayerCacheSlot::Empty { num_kv_heads: 1 };

    let appended =
        append_decode_kv_cache_ref(&mut store, empty_slot, key_ref, value_ref, 0, None, 1)
            .expect("empty append should build ref");
    let appended = materialize_decode_layer_cache_from_store(&store, &appended).expect("cache");
    assert_eq!(cache_key_bits(&appended), vec![vec![7]]);
    assert_eq!(cache_value_bits(&appended), vec![vec![8]]);

    let old_cache = RasterKvCache::from_heads(
        vec![rows_from_bits(&[1, 2, 3])],
        vec![rows_from_bits(&[11, 12, 13])],
    )
    .expect("old cache");
    let old_slot =
        register_decode_layer_cache(&mut store, "test.old", 0, old_cache).expect("old cache slot");
    let key_ref = store
        .insert_attention_heads(
            RasterTensorId::new("test.sliding.key").expect("key id"),
            heads_from_rows(&[&[4]]),
        )
        .expect("key ref");
    let value_ref = store
        .insert_attention_heads(
            RasterTensorId::new("test.sliding.value").expect("value id"),
            heads_from_rows(&[&[14]]),
        )
        .expect("value ref");

    let appended =
        append_decode_kv_cache_ref(&mut store, old_slot, key_ref, value_ref, 1, Some(2), 1)
            .expect("sliding append should build ref");
    let appended = materialize_decode_layer_cache_from_store(&store, &appended).expect("cache");
    assert_eq!(cache_key_bits(&appended), vec![vec![3, 4]]);
    assert_eq!(cache_value_bits(&appended), vec![vec![13, 14]]);
}

#[test]
fn decode_ref_cache_append_rejects_invalid_shapes() {
    let mut store = init_decode_transition_store();
    let key_ref = store
        .insert_attention_heads(
            RasterTensorId::new("test.bad.key").expect("key id"),
            heads_from_rows(&[&[1]]),
        )
        .expect("key ref");
    let value_ref = store
        .insert_attention_heads(
            RasterTensorId::new("test.bad.value").expect("value id"),
            heads_from_rows(&[&[2]]),
        )
        .expect("value ref");

    let error = append_decode_kv_cache_ref(
        &mut store,
        DecodeLayerCacheSlot::Empty { num_kv_heads: 2 },
        key_ref,
        value_ref,
        0,
        None,
        1,
    )
    .expect_err("head mismatch should fail");

    assert!(error.to_string().contains("empty cache head count"));
}

#[test]
fn raster_decode_transition_matches_sliding_cache_window() {
    let (_path, model) = no_ple_model(true);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");

    for cache_len in [0, 1, 2, 3] {
        let decode_state = decode_state_with_cache(cache_len);
        let raster = run(decode_state.clone(), 1, &source, raster_sizing(1))
            .expect("raster decode should run");
        let deterministic = crate::decode_transition::run_with_mode(
            decode_state,
            1,
            &model,
            InferenceExecutionMode::Deterministic,
        )
        .expect("deterministic decode should run");

        assert_eq!(
            raster.transformer_decode_state.layer_caches[0].current_len(),
            1,
            "sliding cache should retain one row for cache length {cache_len}"
        );
        assert_eq!(
            raster.transformer_decode_state, deterministic.transformer_decode_state,
            "sliding decode parity failed for cache length {cache_len}"
        );
    }
}

#[test]
fn raster_decode_transition_matches_deterministic_with_attention_k_eq_v() {
    let (_path, mut model) = no_ple_model(false);
    model.layers[0].v_proj = None;
    model.layers[0].attention_k_eq_v = true;
    assert_raster_matches_deterministic(&model, decode_state_with_cache(2));
}

#[test]
fn raster_decode_transition_matches_deterministic_with_layer_scalar() {
    let (_path, mut model) = no_ple_model(false);
    model.layers[0].layer_scalar = Some(0.5);
    model.layers[0].layer_scalar_det = Some(Act::from_num(0.5));
    assert_raster_matches_deterministic(&model, decode_state_with_cache(2));
}

#[test]
fn raster_decode_transition_matches_deterministic_with_ple() {
    let (_paths, model) = ple_model();
    assert_raster_matches_deterministic(&model, decode_state_with_cache(2));
}

#[test]
fn raster_decode_transition_matches_deterministic_with_donor_cache() {
    let (_path, mut model) = no_ple_model(false);
    let mut donor_layer = model.layers[0].clone();
    donor_layer.kv_shared_layer_index = Some(0);
    model.layers.push(donor_layer);
    assert_raster_matches_deterministic(&model, decode_state_with_layer_count(2, 2));
}

#[test]
fn raster_decode_rejects_f32_only_cache() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let decode_state = TransformerDecodeState {
        layer_caches: vec![LayerKvCache::from_f32_heads(
            vec![VecDeque::from([vec![0.0, 0.0]])],
            vec![VecDeque::from([vec![0.0, 0.0]])],
        )],
        position: 1,
        token_count: 1,
    };

    let error = run(decode_state, 1, &source, raster_sizing(1)).expect_err("f32 cache should fail");

    assert!(error.to_string().contains("canonical layer cache keys"));
}

#[test]
fn decode_source_rejects_non_deterministic_model() {
    let (_path, mut model) = no_ple_model(false);
    model.provenance = Gemma4ModelProvenance::Fp32;

    let error = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect_err("fp32 model should fail");

    assert!(error.to_string().contains(".detwgt artifact"));
}

fn assert_raster_matches_deterministic(
    model: &Gemma4TransformerModel,
    decode_state: TransformerDecodeState,
) {
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", model)
        .expect("source should build");
    let raster = run(
        decode_state.clone(),
        1,
        &source,
        raster_sizing_with_attention(2, 1),
    )
    .expect("raster decode should run");
    let deterministic = crate::decode_transition::run_with_mode(
        decode_state,
        1,
        model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("deterministic decode should run");

    assert_eq!(raster.activation_state, deterministic.activation_state);
    assert_eq!(raster.prefill_logits, deterministic.prefill_logits);
    assert_eq!(
        raster.transformer_decode_state,
        deterministic.transformer_decode_state
    );
}

fn selected_token_ref(
    source_name: &str,
    token_id: u32,
) -> Result<(RasterArtifactStoreRoots, RasterSelectedTokenRef)> {
    ArtifactIo::reset_store();
    let roots = ArtifactIo::export_store_roots();
    let (roots, artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        &roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::token_ids(1),
        vec![token_id_leaf(token_id)],
    )?;
    let selected_token_ref =
        RasterSelectedTokenRef::new(RasterTokenIdSequenceRef::new(artifact_ref)?)?;
    Ok((roots, selected_token_ref))
}

fn no_ple_model(sliding: bool) -> (PathBuf, Gemma4TransformerModel) {
    let hidden_width = 2;
    let matrices = vec![
        vec![
            vec![Wgt::from_num(0.0), Wgt::from_num(0.0)],
            vec![Wgt::from_num(1.0), Wgt::from_num(-0.5)],
        ],
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
    ];
    let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
    let mut sources = sources.into_iter();
    let embedding_source = sources.next().expect("embedding source");
    let layer = Gemma4LayerWeights {
        attention_kind: if sliding {
            Gemma4AttentionKind::Sliding
        } else {
            Gemma4AttentionKind::Full
        },
        hidden_size: hidden_width,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden_width,
        sliding_window: sliding.then_some(1),
        cache_sliding_window: sliding.then_some(1),
        rms_norm_eps: 0.0,
        rms_norm_eps_det: Some(Acc::from_num(0.0)),
        rope_base: 10_000.0,
        rope_base_det: None,
        partial_rotary_dim: 0,
        rope_freq_base_dim: 2,
        kv_shared_layer_index: None,
        attention_k_eq_v: false,
        q_proj: det_matrix(sources.next().expect("q source")),
        k_proj: det_matrix(sources.next().expect("k source")),
        v_proj: Some(det_matrix(sources.next().expect("v source"))),
        o_proj: det_matrix(sources.next().expect("o source")),
        q_norm_weight: vec![1.0; hidden_width],
        q_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        k_norm_weight: vec![1.0; hidden_width],
        k_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        input_layernorm_weight: vec![1.0; hidden_width],
        input_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        post_attention_layernorm_weight: vec![1.0; hidden_width],
        post_attention_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        pre_feedforward_layernorm_weight: vec![1.0; hidden_width],
        pre_feedforward_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        post_feedforward_layernorm_weight: vec![1.0; hidden_width],
        post_feedforward_layernorm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        gate_proj: det_matrix(sources.next().expect("gate source")),
        up_proj: det_matrix(sources.next().expect("up source")),
        down_proj: det_matrix(sources.next().expect("down source")),
        ple: None,
        layer_scalar: None,
        layer_scalar_det: None,
    };

    (
        path,
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::DetNumWgt,
            embedding_table: None,
            embedding_source: Some(GemmaEmbeddingTensorSource::Deterministic {
                source: embedding_source,
                scale: 1.0,
                det_cache: Arc::new(Mutex::new(None)),
            }),
            layers: vec![layer],
            ple_global: None,
            final_norm_weight: vec![1.0; hidden_width],
            final_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: MatrixF32 {
                    rows: 2,
                    cols: hidden_width,
                    values: vec![1.0, 0.0, 0.0, 1.0],
                },
                det_weight: Some(Arc::new(det_num_matrix(identity_matrix()))),
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 0.0,
            rms_norm_eps_det: Some(Acc::from_num(0.0)),
        },
    )
}

fn ple_model() -> (Vec<PathBuf>, Gemma4TransformerModel) {
    let (base_path, mut model) = no_ple_model(false);
    let hidden_width = 2;
    let (ple_layer_path, ple_layer_sources) =
        write_det_matrices(vec![identity_matrix(), identity_matrix()])
            .expect("fixture PLE layer weights should write");
    let mut ple_layer_sources = ple_layer_sources.into_iter();
    model.layers[0].ple = Some(crate::Gemma4PleLayerWeights {
        input_gate: det_matrix(ple_layer_sources.next().expect("PLE input gate")),
        layer_projection: det_matrix(ple_layer_sources.next().expect("PLE layer projection")),
        post_input_norm_weight: vec![1.0; hidden_width],
        post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
    });

    let (ple_global_path, ple_global_sources) =
        write_det_matrices(vec![identity_matrix(), identity_matrix()])
            .expect("fixture PLE global weights should write");
    let mut ple_global_sources = ple_global_sources.into_iter();
    model.ple_global = Some(crate::Gemma4PleGlobalWeights::from_det_num_sources(
        vec![ple_global_sources.next().expect("PLE token embeddings")],
        vec![ple_global_sources.next().expect("PLE model projection")],
        vec![1.0; hidden_width],
        1.0,
        1.0,
        1.0,
    ));

    (vec![base_path, ple_layer_path, ple_global_path], model)
}

fn raster_sizing(projection_rows_per_tile: usize) -> RasterSizingControls {
    raster_sizing_with_attention(projection_rows_per_tile, 1)
}

fn raster_sizing_with_attention(
    projection_rows_per_tile: usize,
    attention_kv_rows_per_tile: usize,
) -> RasterSizingControls {
    RasterSizingControls {
        projection_rows_per_tile,
        attention_kv_rows_per_tile,
        sequence_rows_per_tile: 1,
        head_rows_per_tile: 1,
        tokenizer_bpe_pairs_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE,
        tokenizer_bpe_pieces_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE,
        output_byte_flush_bytes_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    }
}

fn decode_state_with_cache(cache_len: usize) -> TransformerDecodeState {
    decode_state_with_layer_count(1, cache_len)
}

fn decode_state_with_layer_count(layer_count: usize, cache_len: usize) -> TransformerDecodeState {
    let key_rows = (0..cache_len)
        .map(|_| vec![Act::from_num(0.0), Act::from_num(0.0)])
        .collect::<VecDeque<_>>();
    let value_rows = key_rows.clone();
    TransformerDecodeState {
        layer_caches: (0..layer_count)
            .map(|_| LayerKvCache::from_det_heads(vec![key_rows.clone()], vec![value_rows.clone()]))
            .collect(),
        position: cache_len,
        token_count: cache_len,
    }
}

fn heads_from_rows(heads: &[&[i32]]) -> RasterAttentionHeadSequence {
    RasterAttentionHeadSequence::from_heads(
        heads
            .iter()
            .map(|rows| rows_from_bits(rows))
            .collect::<Vec<_>>(),
    )
}

fn rows_from_bits(bits: &[i32]) -> Vec<RasterActivationRow> {
    bits.iter()
        .map(|bits| RasterActivationRow::from_acts(vec![Act::from_bits(*bits)]))
        .collect()
}

fn cache_key_bits(cache: &RasterKvCache) -> Vec<Vec<i32>> {
    cache
        .keys()
        .iter()
        .map(|head| head.iter().map(first_act_bits).collect())
        .collect()
}

fn cache_value_bits(cache: &RasterKvCache) -> Vec<Vec<i32>> {
    cache
        .values()
        .iter()
        .map(|head| head.iter().map(first_act_bits).collect())
        .collect()
}

fn first_act_bits(row: &RasterActivationRow) -> i32 {
    row.acts()
        .first()
        .expect("test row should have one value")
        .to_bits()
}

fn identity_matrix() -> Vec<Vec<Wgt>> {
    vec![
        vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
        vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
    ]
}

fn zero_matrix(width: usize) -> Vec<Vec<Wgt>> {
    vec![vec![Wgt::from_num(0.0); width]; width]
}

fn det_matrix(source: DetNumTensorSliceSource) -> Gemma4LayerMatrixSource {
    Gemma4LayerMatrixSource::from_det_num_source(source)
}

fn det_num_matrix(rows: Vec<Vec<Wgt>>) -> DetNumMatrix {
    DetNumMatrix {
        rows: rows.len(),
        cols: rows.first().map(Vec::len).unwrap_or(0),
        values: rows
            .into_iter()
            .flat_map(|row| row.into_iter().map(|value| value.to_bits()))
            .collect(),
    }
}

fn write_det_matrices(
    matrices: Vec<Vec<Vec<Wgt>>>,
) -> Result<(PathBuf, Vec<DetNumTensorSliceSource>)> {
    let unique_suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "raster-decode-transition-{}-{}-{}.detwgt",
        std::process::id(),
        unique_suffix,
        crate::trace::sha256_hex(&format!("{:?}", matrices))
    ));
    let mut bytes = Vec::new();
    let mut sources = Vec::new();

    for rows in matrices {
        let data_offset = bytes.len();
        for row in &rows {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        sources.push(det_source(&path, rows.len(), rows[0].len(), data_offset));
    }

    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
    Ok((path, sources))
}

fn det_source(
    path: &Path,
    rows: usize,
    cols: usize,
    data_offset: usize,
) -> DetNumTensorSliceSource {
    DetNumTensorSliceSource {
        weights_path: path.to_path_buf(),
        total_rows: rows,
        total_cols: cols,
        data_offset,
        row_offset: 0,
        row_count: rows,
        col_offset: 0,
        col_count: cols,
    }
}
