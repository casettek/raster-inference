use super::{
    main, main_state_refs, materialize_decode_layer_caches_from_roots,
    raster_cache_from_layer_cache, register_decode_layer_cache_with_roots, run,
    RasterDecodeTransitionInputRefs, RasterDecodeTransitionInputRoots,
};
use crate::decode_transition::raster::auth_source::{
    AuthenticatedGemmaDecodeTransitionSource, RasterDecodeTransitionSource,
};
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::external_artifacts::reset_external_source_store;
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
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterSequenceRowRequest,
};
use crate::RasterSizingControls;
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn raster_source(
    source: &AuthenticatedGemmaDecodeTransitionSource,
) -> RasterDecodeTransitionSource<'_> {
    RasterDecodeTransitionSource::for_current_integrity_mode(source)
        .expect("source should prepare for raster reads")
}

#[test]
fn decode_transition_sequences_are_branch_free_orchestration() {
    let source = include_str!("tiles.rs");
    let violations = branch_free_sequence_violations(source);

    assert!(
        violations.is_empty(),
        "decode_transition sequences must be straight-line tile/sequence calls: {violations:?}"
    );
}

#[test]
fn decode_transition_tiles_do_not_call_tiles_or_sequences() {
    let source = include_str!("tiles.rs");
    let violations = tile_call_violations(source);

    assert!(
        violations.is_empty(),
        "decode_transition tiles must not invoke tile/sequence calls: {violations:?}"
    );
}

#[test]
fn decode_transition_tiles_do_not_directly_call_authored_functions() {
    let source = include_str!("tiles.rs");
    let violations = tile_authored_call_violations(source);

    assert!(
        violations.is_empty(),
        "decode_transition tiles must not directly invoke authored tile/sequence functions: {violations:?}"
    );
}

#[test]
fn decode_transition_tiles_rs_contains_only_authored_functions() {
    let source = include_str!("tiles.rs");
    let violations = unauthored_function_violations(source);

    assert!(
        violations.is_empty(),
        "decode_transition tiles.rs should keep helper code in utils.rs, not plain functions: {violations:?}"
    );
}

#[test]
fn raster_decode_transition_matches_deterministic_no_ple() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let committed_source = raster_source(&source);
    let decode_state = decode_state_with_cache(1);

    let raster = run(decode_state.clone(), 1, &committed_source, raster_sizing(1))
        .expect("raster decode should run");
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
    let committed_source = raster_source(&source);
    let decode_state = decode_state_with_cache(1);
    let (roots, selected_token_ref) =
        selected_token_ref("decode.transition.selected", 1).expect("selected token ref");

    let raster = main(
        RasterDecodeTransitionInputRoots {
            artifact_store_roots: roots,
            transformer_decode_state: decode_state.clone(),
            selected_token_ref,
            decode_transition_source_root: committed_source.root().to_string(),
            output_source_prefix: "decode.transition.root-backed".to_string(),
            raster_sizing: raster_sizing(1),
        },
        &committed_source,
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
fn root_backed_main_rejects_mismatched_committed_source_root() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let committed_source = raster_source(&source);
    let (roots, selected_token_ref) =
        selected_token_ref("decode.transition.wrong-source.selected", 1)
            .expect("selected token ref");

    let error = main(
        RasterDecodeTransitionInputRoots {
            artifact_store_roots: roots,
            transformer_decode_state: decode_state_with_cache(1),
            selected_token_ref,
            decode_transition_source_root: "wrong-decode-source-root".to_string(),
            output_source_prefix: "decode.transition.wrong-source".to_string(),
            raster_sizing: raster_sizing(1),
        },
        &committed_source,
    )
    .expect_err("mismatched committed decode source root should fail");

    assert!(error
        .to_string()
        .contains("does not match input source root"));
}

#[test]
fn root_backed_state_main_returns_refs_without_materialized_transition_result() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let committed_source = raster_source(&source);
    let decode_state = decode_state_with_cache(1);
    let (mut roots, selected_token_ref) =
        selected_token_ref("decode.transition.state.selected", 1).expect("selected token ref");
    let mut layer_caches = Vec::with_capacity(decode_state.layer_caches.len());
    for (layer_idx, cache) in decode_state.layer_caches.iter().enumerate() {
        let raster_cache = raster_cache_from_layer_cache(cache).expect("cache should convert");
        let (next_roots, cache_slot) = register_decode_layer_cache_with_roots(
            &roots,
            "decode.transition.state.original.cache",
            layer_idx,
            raster_cache,
        )
        .expect("cache should register");
        roots = next_roots;
        layer_caches.push(cache_slot);
    }

    let output = main_state_refs(
        RasterDecodeTransitionInputRefs {
            artifact_store_roots: roots,
            position: decode_state.position,
            token_count: decode_state.token_count,
            layer_caches,
            selected_token_ref,
            decode_transition_source_root: committed_source.root().to_string(),
            output_source_prefix: "decode.transition.state".to_string(),
            raster_sizing: raster_sizing(1),
        },
        &committed_source,
    )
    .expect("state-backed raster decode should run");

    assert_eq!(output.position, decode_state.position + 1);
    assert_eq!(output.token_count, decode_state.token_count + 1);
    assert_eq!(output.layer_caches.len(), decode_state.layer_caches.len());
    let (row_count, width) = output
        .logits_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()
        .expect("logits shape");
    assert_eq!(row_count, 1);
    assert!(width > 0);

    let deterministic = crate::decode_transition::run_with_mode(
        decode_state,
        1,
        &model,
        InferenceExecutionMode::Deterministic,
    )
    .expect("deterministic decode should run");
    let final_hidden_row = read_sequence_row_from_roots(
        &output.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: output.final_hidden_state_ref.clone(),
            row_idx: 0,
        },
    )
    .expect("final hidden ref should materialize");
    assert_eq!(
        vec![final_hidden_row.acts()],
        deterministic
            .activation_state
            .clone_internal()
            .det_values()
            .expect("deterministic activation values")
    );
    let logits_row = read_sequence_row_from_roots(
        &output.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: output.logits_ref.clone(),
            row_idx: 0,
        },
    )
    .expect("logits ref should materialize");
    assert_eq!(
        logits_row.acts(),
        deterministic
            .prefill_logits
            .clone_internal()
            .det_values()
            .expect("deterministic logits")
    );
    assert_eq!(
        materialize_decode_layer_caches_from_roots(
            &output.artifact_store_roots,
            &output.layer_caches
        )
        .expect("cache refs should materialize"),
        deterministic.transformer_decode_state.layer_caches
    );
}

#[test]
fn root_backed_state_main_rejects_mismatched_committed_source_root() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let committed_source = raster_source(&source);
    let decode_state = decode_state_with_cache(1);
    let (mut roots, selected_token_ref) =
        selected_token_ref("decode.transition.state.wrong-source.selected", 1)
            .expect("selected token ref");
    let mut layer_caches = Vec::with_capacity(decode_state.layer_caches.len());
    for (layer_idx, cache) in decode_state.layer_caches.iter().enumerate() {
        let raster_cache = raster_cache_from_layer_cache(cache).expect("cache should convert");
        let (next_roots, cache_slot) = register_decode_layer_cache_with_roots(
            &roots,
            "decode.transition.state.wrong-source.original.cache",
            layer_idx,
            raster_cache,
        )
        .expect("cache should register");
        roots = next_roots;
        layer_caches.push(cache_slot);
    }

    let error = main_state_refs(
        RasterDecodeTransitionInputRefs {
            artifact_store_roots: roots,
            position: decode_state.position,
            token_count: decode_state.token_count,
            layer_caches,
            selected_token_ref,
            decode_transition_source_root: "wrong-decode-source-root".to_string(),
            output_source_prefix: "decode.transition.state.wrong-source".to_string(),
            raster_sizing: raster_sizing(1),
        },
        &committed_source,
    )
    .expect_err("mismatched committed decode source root should fail");

    assert!(error
        .to_string()
        .contains("does not match input source root"));
}

#[test]
fn root_backed_main_rejects_missing_selected_token_root() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let committed_source = raster_source(&source);
    let (_roots, selected_token_ref) =
        selected_token_ref("decode.transition.missing.selected", 1).expect("selected token ref");

    let error = main(
        RasterDecodeTransitionInputRoots {
            artifact_store_roots: RasterArtifactStoreRoots::default(),
            transformer_decode_state: decode_state_with_cache(1),
            selected_token_ref,
            decode_transition_source_root: committed_source.root().to_string(),
            output_source_prefix: "decode.transition.missing".to_string(),
            raster_sizing: raster_sizing(1),
        },
        &committed_source,
    )
    .expect_err("missing selected-token root should fail");

    assert!(error.to_string().contains("not present"));
}

#[test]
fn raster_decode_transition_matches_across_projection_and_attention_chunks() {
    let (_path, model) = no_ple_model(false);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let committed_source = raster_source(&source);
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
        let raster = run(decode_state.clone(), 1, &committed_source, sizing)
            .expect("raster decode should run");

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
    let committed_source = raster_source(&source);
    let decode_state = decode_state_with_cache(1);

    let projection_error = run(
        decode_state.clone(),
        1,
        &committed_source,
        raster_sizing_with_attention(0, 1),
    )
    .expect_err("zero projection chunk should fail");
    assert!(projection_error
        .to_string()
        .contains("projection rows per tile"));

    let attention_error = run(
        decode_state,
        1,
        &committed_source,
        raster_sizing_with_attention(1, 0),
    )
    .expect_err("zero attention chunk should fail");
    assert!(attention_error
        .to_string()
        .contains("attention KV rows per tile"));
}

#[test]
fn raster_decode_transition_matches_sliding_cache_window() {
    let (_path, model) = no_ple_model(true);
    let source = AuthenticatedGemmaDecodeTransitionSource::from_model("decode", &model)
        .expect("source should build");
    let committed_source = raster_source(&source);

    for cache_len in [0, 1, 2, 3] {
        let decode_state = decode_state_with_cache(cache_len);
        let raster = run(decode_state.clone(), 1, &committed_source, raster_sizing(1))
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
    let committed_source = raster_source(&source);
    let decode_state = TransformerDecodeState {
        layer_caches: vec![LayerKvCache::from_f32_heads(
            vec![VecDeque::from([vec![0.0, 0.0]])],
            vec![VecDeque::from([vec![0.0, 0.0]])],
        )],
        position: 1,
        token_count: 1,
    };

    let error = run(decode_state, 1, &committed_source, raster_sizing(1))
        .expect_err("f32 cache should fail");

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
    let committed_source = raster_source(&source);
    let raster = run(
        decode_state.clone(),
        1,
        &committed_source,
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
    reset_external_source_store();
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

fn raster_sequence_bodies(source: &str) -> Vec<(String, String)> {
    attributed_function_bodies(source, "#[sequence")
}

fn raster_tile_bodies(source: &str) -> Vec<(String, String)> {
    attributed_function_bodies(source, "#[tile")
}

fn attributed_function_bodies(source: &str, attribute: &str) -> Vec<(String, String)> {
    let mut bodies = Vec::new();
    let mut search_start = 0;

    while let Some(attribute_offset) = source[search_start..].find(attribute) {
        let attribute_start = search_start + attribute_offset;
        let fn_start = attribute_start
            + source[attribute_start..]
                .find("fn ")
                .expect("attribute should be followed by a function")
            + "fn ".len();
        let name_end = fn_start
            + source[fn_start..]
                .find('(')
                .expect("authored function should have a parameter list");
        let function_name = source[fn_start..name_end].trim().to_string();
        let body_start = name_end
            + source[name_end..]
                .find('{')
                .expect("authored function should have a body");
        let mut depth = 0usize;
        let mut body_end = body_start;

        for (offset, character) in source[body_start..].char_indices() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        body_end = body_start + offset;
                        break;
                    }
                }
                _ => {}
            }
        }

        bodies.push((function_name, source[body_start + 1..body_end].to_string()));
        search_start = body_end + 1;
    }

    bodies
}

fn tile_call_violations(source: &str) -> Vec<String> {
    let tile_bodies = raster_tile_bodies(source);
    assert!(
        !tile_bodies.is_empty(),
        "source should have tile bodies to validate"
    );
    let forbidden_fragments = [
        "call_tile!",
        "call_seq!",
        "call_recur_tile!",
        "call_recur_seq!",
    ];
    let mut violations = Vec::new();

    for (tile_name, tile_body) in tile_bodies {
        for forbidden_fragment in forbidden_fragments {
            if tile_body.contains(forbidden_fragment) {
                violations.push(format!("`{tile_name}` contains `{forbidden_fragment}`"));
            }
        }
    }

    violations
}

fn tile_authored_call_violations(source: &str) -> Vec<String> {
    let authored_names = attributed_function_bodies(source, "#[")
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    let tile_bodies = raster_tile_bodies(source);
    let mut violations = Vec::new();

    for (tile_name, tile_body) in tile_bodies {
        for authored_name in &authored_names {
            if authored_name == &tile_name {
                continue;
            }
            let call_fragment = format!("{authored_name}(");
            if tile_body.contains(&call_fragment) {
                violations.push(format!("`{tile_name}` directly calls `{authored_name}`"));
            }
        }
    }

    violations
}

fn unauthored_function_violations(source: &str) -> Vec<String> {
    let lines = source.lines().collect::<Vec<_>>();
    let mut violations = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if !(trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("pub(in ")
            || trimmed.starts_with("pub(crate) fn "))
        {
            continue;
        }

        let previous = index
            .checked_sub(1)
            .and_then(|previous_index| lines.get(previous_index))
            .map(|line| line.trim_start())
            .unwrap_or_default();
        if !(previous.starts_with("#[sequence") || previous.starts_with("#[tile")) {
            let function_name = trimmed
                .split_once("fn ")
                .and_then(|(_, rest)| rest.split_once('('))
                .map(|(name, _)| name.trim())
                .unwrap_or(trimmed);
            violations.push(format!("`{function_name}` is not immediately authored"));
        }
    }

    violations
}

fn branch_free_sequence_violations(source: &str) -> Vec<String> {
    let sequence_bodies = raster_sequence_bodies(source);
    assert!(
        !sequence_bodies.is_empty(),
        "source should have sequence bodies to validate"
    );
    let control_flow_tokens = ["if", "return", "match", "for", "while", "loop"];
    let inline_logic_fragments = [
        "ok_or_else",
        "RasterArtifactId::new",
        "RasterTensorId::new",
        "::new",
        "format!(",
        "Some(",
        ".clone()",
        ".",
        "Act::from_bits",
        "Acc::from_bits",
        ".collect()",
    ];
    let mut violations = Vec::new();

    for (sequence_name, sequence_body) in sequence_bodies {
        let tokenized = sequence_body
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '_' {
                    character
                } else {
                    ' '
                }
            })
            .collect::<String>();

        for token in tokenized.split_whitespace() {
            if control_flow_tokens.contains(&token) {
                violations.push(format!("`{sequence_name}` contains `{token}`"));
            }
        }
        for forbidden_fragment in inline_logic_fragments {
            if sequence_body.contains(forbidden_fragment) {
                violations.push(format!("`{sequence_name}` contains `{forbidden_fragment}`"));
            }
        }
    }

    violations
}
