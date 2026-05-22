use super::*;
use crate::shared::api::input::InferenceExecutionMode;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactId;
use crate::shared::model::transformer::{
    ActivationSequence, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
    Gemma4PleLayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
    InternalActivationSequence, MatrixF32,
};
use crate::shared::numerics::det_num::act_to_f32;
use crate::shared::numerics::det_num::{f32_to_acc, Act, Wgt};
use crate::shared::raster_contracts::prefill_ple::{
    AuthenticatedGemmaPleSource, GemmaPleLayerConfig, GemmaPleScalars,
};
use crate::RasterSizingControls;
use anyhow::{Context, Result};
use std::path::PathBuf;

#[test]
fn ref_backed_prepare_aux_helpers_do_not_hide_completion_loops() {
    let source = include_str!("tiles.rs");
    let forbidden = concat!("while !", "state.is_complete()");

    assert!(
        !source.contains(forbidden),
        "ref-backed prepare-aux helpers must expose dynamic loops through recursive tile calls"
    );
}

#[test]
fn scaled_token_embedding_rows_advance_one_recursive_step_at_a_time() {
    let fixture = PleFixture::single_layer().expect("fixture should build");
    reset_artifact_store();
    let token_ids_ref = store_prefill_token_ids_artifact(&[0, 1]).expect("token ids ref");
    let (mut artifact_store_roots, state) = init_scaled_token_embedding_sequence_ref(
        ArtifactIo::export_store_roots(),
        token_ids_ref.id().source_name(),
        token_ids_ref.token_count(),
        0,
        Act::from_num(1.0),
        2,
    )
    .expect("init token embedding state");
    let encoded = serde_json::to_string(&state).expect("serialize token embedding state");
    assert!(encoded.contains("token_ids_source_name"));
    assert!(!encoded.contains("[0,1]"));
    assert!(!encoded.contains(token_ids_ref.root()));
    assert_eq!(state.next_token_idx, 0);

    let (done, next_roots, state) =
        append_next_scaled_token_embedding_row(artifact_store_roots, state, &fixture.source)
            .expect("append first row");
    artifact_store_roots = next_roots;
    assert!(!done);
    assert_eq!(state.next_token_idx, 1);

    let (done, next_roots, state) =
        append_next_scaled_token_embedding_row(artifact_store_roots, state, &fixture.source)
            .expect("append second row");
    artifact_store_roots = next_roots;
    assert!(!done);
    assert_eq!(state.next_token_idx, 2);

    let (done, artifact_store_roots, state) =
        append_next_scaled_token_embedding_row(artifact_store_roots, state, &fixture.source)
            .expect("observe completion");
    assert!(done);

    let (_artifact_store_roots, sequence_ref) =
        finalize_scaled_token_embedding_sequence_ref(artifact_store_roots, state)
            .expect("finalize embedded sequence");
    assert_eq!(sequence_ref.row_count(), 2);
    assert_eq!(sequence_ref.width(), 2);
}

#[test]
fn no_ple_globals_return_none_without_requiring_deterministic_inputs() {
    let source = AuthenticatedGemmaPleSource::no_ple(
        "no-ple",
        vec![GemmaPleLayerConfig {
            has_ple: false,
            hidden_width: 2,
        }],
    )
    .expect("source should build");
    let input = ActivationSequence::from_values(vec![vec![1.0, 2.0]], "digest".to_string());

    let output = run_materialized(&[0], &input, &source, 1).expect("raster PLE should run");

    assert!(output.is_none());
}

#[test]
fn prefill_prepare_aux_roots_can_be_built_from_input_embedding_refs() {
    reset_artifact_store();
    let token_ids_ref = store_prefill_token_ids_artifact(&[0, 1]).expect("token ids ref");
    let activations_ref = insert_activation_sequence(
        RasterArtifactId::new("input.embedding.test.embedded").expect("artifact id"),
        crate::shared::raster_kernels::transformer::RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ]),
    )
    .expect("activation ref");
    let input_embedding_roots = ArtifactIo::export_store_roots();
    let input_embedding_refs = crate::input_embedding::raster::RasterInputEmbeddingRefs {
        source_id: "embedding-fixture".to_string(),
        embedding_source_root: "embedding-source-root".to_string(),
        prompt_token_ids_root: token_ids_ref.root().to_string(),
        prompt_token_count: token_ids_ref.token_count(),
        embedded_prompt_activations_ref: activations_ref.clone(),
    };
    let ple_source = AuthenticatedGemmaPleSource::from_canonical_parts(
        "ple-fixture",
        vec![GemmaPleLayerConfig {
            has_ple: true,
            hidden_width: 2,
        }],
        vec![vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ]],
        vec![vec![
            vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
            vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
        ]],
        vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
        test_scalars(),
    )
    .expect("PLE source");

    let (_artifact_store_roots, roots) =
        prepare_raster_prefill_ple_input_roots_from_embedding_refs(
            input_embedding_roots,
            &input_embedding_refs,
            &ple_source,
            raster_sizing_with_projection_rows(1),
        )
        .expect("input roots");

    assert_eq!(
        roots.token_ids_source_name,
        token_ids_ref.id().source_name()
    );
    assert_eq!(roots.token_count, token_ids_ref.token_count());
    assert_eq!(roots.input_activations_ref, Some(activations_ref));
    assert!(roots.has_ple_global);
}

#[test]
fn no_ple_globals_match_native_none_output() {
    let source = AuthenticatedGemmaPleSource::no_ple(
        "no-ple",
        vec![GemmaPleLayerConfig {
            has_ple: false,
            hidden_width: 2,
        }],
    )
    .expect("source should build");
    let input = ActivationSequence::from_values(vec![vec![1.0, 2.0]], "digest".to_string());
    let native_model = no_ple_model(2);

    let raster = run_materialized(&[0], &input, &source, 1).expect("raster PLE should run");
    let native = crate::prefill_prepare_aux::run(
        &[0],
        &native_model,
        &input,
        InferenceExecutionMode::Deterministic,
    )
    .expect("native PLE should run");

    assert_eq!(raster, native);
    assert!(raster.is_none());
}

#[test]
fn single_token_single_ple_layer_matches_native_prefill_ple_computation() {
    let fixture = PleFixture::new(
        vec![true],
        vec![vec![
            vec![Act::from_num(0.25), Act::from_num(-0.5)],
            vec![Act::from_num(1.0), Act::from_num(0.5)],
        ]],
        vec![vec![
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
        ]],
    )
    .expect("fixture should build");
    let token_ids = [0];
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

    let (raster, native) = compare_raster_and_native(&fixture, &token_ids, &input);

    assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
    assert_eq!(
        raster
            .clone_layer_internal(0)
            .expect("raster layer")
            .det_values(),
        native
            .clone_layer_internal(0)
            .expect("native layer")
            .det_values()
    );
}

#[test]
fn multiple_prompt_tokens_single_ple_layer_matches_native_prefill_ple_computation() {
    let fixture = PleFixture::new(
        vec![true],
        vec![vec![
            vec![Act::from_num(0.25), Act::from_num(-0.5)],
            vec![Act::from_num(1.0), Act::from_num(0.5)],
        ]],
        vec![vec![
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
        ]],
    )
    .expect("fixture should build");
    let token_ids = [0, 1];
    let input = activation_sequence(vec![
        vec![Act::from_num(1.0), Act::from_num(0.5)],
        vec![Act::from_num(-1.0), Act::from_num(2.0)],
    ]);

    let (raster, native) = compare_raster_and_native(&fixture, &token_ids, &input);

    assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
    assert_eq!(
        raster
            .clone_layer_internal(0)
            .expect("raster layer")
            .det_values(),
        native
            .clone_layer_internal(0)
            .expect("native layer")
            .det_values()
    );
}

#[test]
fn chunked_prefill_ple_projection_matches_native_prefill_ple_computation() {
    let fixture = PleFixture::new(
        vec![true],
        vec![vec![
            vec![Act::from_num(0.25), Act::from_num(-0.5)],
            vec![Act::from_num(1.0), Act::from_num(0.5)],
        ]],
        vec![vec![
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
        ]],
    )
    .expect("fixture should build");
    let token_ids = [0, 1];
    let input = activation_sequence(vec![
        vec![Act::from_num(1.0), Act::from_num(0.5)],
        vec![Act::from_num(-1.0), Act::from_num(2.0)],
    ]);

    let raster = run_materialized(&token_ids, &input, &fixture.source, 2)
        .expect("raster PLE should run")
        .expect("raster PLE inputs");
    let native = native_prefill_ple_inputs(&fixture, &token_ids, &input);

    assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
}

#[test]
fn chunked_prefill_ple_sequence_rows_do_not_change_output() {
    let fixture = PleFixture::new(
        vec![true],
        vec![vec![
            vec![Act::from_num(0.25), Act::from_num(-0.5)],
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-0.25), Act::from_num(0.75)],
        ]],
        vec![vec![
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
        ]],
    )
    .expect("fixture should build");
    let token_ids = [0, 1, 2];
    let input = activation_sequence(vec![
        vec![Act::from_num(1.0), Act::from_num(0.5)],
        vec![Act::from_num(-1.0), Act::from_num(2.0)],
        vec![Act::from_num(0.25), Act::from_num(-0.75)],
    ]);
    let one_row = run_materialized_with_sizing(
        &token_ids,
        &input,
        &fixture.source,
        raster_sizing_with_projection_rows(1),
    )
    .expect("one-row raster PLE should run")
    .expect("one-row PLE inputs");
    let mut two_rows_sizing = raster_sizing_with_projection_rows(1);
    two_rows_sizing.sequence_rows_per_tile = 2;
    let two_rows =
        run_materialized_with_sizing(&token_ids, &input, &fixture.source, two_rows_sizing)
            .expect("two-row raster PLE should run")
            .expect("two-row PLE inputs");

    assert_eq!(one_row.per_layer_inputs, two_rows.per_layer_inputs);
}

#[test]
fn prefill_ple_state_serializes_refs_not_activation_rows() {
    let fixture = PleFixture::single_layer().expect("fixture should build");
    let token_ids = [0, 1];
    let input = activation_sequence(vec![
        vec![Act::from_num(1.0), Act::from_num(0.5)],
        vec![Act::from_num(-1.0), Act::from_num(2.0)],
    ]);
    let (artifact_store_roots, input_roots) = prepare_raster_prefill_ple_input_roots(
        &token_ids,
        &input,
        &fixture.source,
        raster_sizing_with_projection_rows(1),
    )
    .expect("prepare input roots");
    let (artifact_store_roots, state) =
        init_prefill_ple_state(artifact_store_roots, input_roots).expect("init state");

    let encoded = serde_json::to_string(&state).expect("serialize initial state");
    assert!(encoded.contains("token_ids_source_name"));
    assert!(encoded.contains("input_activations_ref"));
    assert!(encoded.contains("per_layer_inputs"));
    assert!(!encoded.contains("[0,1]"));
    assert!(!encoded.contains("act_bits"));

    let (_complete, artifact_store_roots, state) =
        compute_next_prefill_ple_layer_sequence(artifact_store_roots, state, &fixture.source)
            .expect("compute layer");
    let encoded = serde_json::to_string(&state).expect("serialize computed state");
    assert!(encoded.contains("prefill.prepare_aux.input.initial"));
    assert!(encoded.contains("prefill.prepare_aux.per_layer_input.0"));
    assert!(!encoded.contains("act_bits"));

    let (artifact_store_roots, refs) =
        finalize_prefill_ple_input_refs(artifact_store_roots, state).expect("finalize");
    let refs = crate::prefill_prepare_aux::prefill_ple_input_refs_from_manifest(
        artifact_store_roots,
        refs.as_deref(),
    )
    .expect("read manifest")
    .expect("PLE refs");
    let finalized =
        crate::prefill_prepare_aux::materialize_prefill_ple_input_refs_for_trace(Some(&refs))
            .expect("materialize refs")
            .expect("PLE inputs");
    let native = native_prefill_ple_inputs(&fixture, &token_ids, &input);
    assert_eq!(finalized.per_layer_inputs, native.per_layer_inputs);
}

#[test]
fn prefill_ple_ref_manifest_serializes_refs_not_activation_rows() {
    let fixture = PleFixture::single_layer().expect("fixture should build");
    let token_ids = [0, 1];
    let input = activation_sequence(vec![
        vec![Act::from_num(1.0), Act::from_num(0.5)],
        vec![Act::from_num(-1.0), Act::from_num(2.0)],
    ]);
    let (artifact_store_roots, manifest_root) = run(
        &token_ids,
        &input,
        &fixture.source,
        raster_sizing_with_projection_rows(1),
    )
    .expect("raster PLE refs should run");
    let refs = crate::prefill_prepare_aux::prefill_ple_input_refs_from_manifest(
        artifact_store_roots,
        manifest_root.as_deref(),
    )
    .expect("read manifest")
    .expect("PLE refs");

    let encoded = serde_json::to_string(&refs).expect("serialize refs");
    assert!(encoded.contains("prefill.prepare_aux.per_layer_input.0"));
    assert!(encoded.contains("per_layer_inputs"));
    assert!(!encoded.contains("act_bits"));

    let materialized =
        crate::prefill_prepare_aux::materialize_prefill_ple_input_refs_for_trace(Some(&refs))
            .expect("materialize refs")
            .expect("PLE inputs");
    let native = native_prefill_ple_inputs(&fixture, &token_ids, &input);
    assert_eq!(materialized.per_layer_inputs, native.per_layer_inputs);
}

#[test]
fn missing_input_ref_fails_closed() {
    let fixture = PleFixture::single_layer().expect("fixture should build");
    let token_ids = [0];
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);
    let (artifact_store_roots, input_roots) = prepare_raster_prefill_ple_input_roots(
        &token_ids,
        &input,
        &fixture.source,
        raster_sizing_with_projection_rows(1),
    )
    .expect("prepare input roots");
    let (artifact_store_roots, mut state) =
        init_prefill_ple_state(artifact_store_roots, input_roots).expect("init state");
    state.input_activations_ref = None;

    let error =
        compute_next_prefill_ple_layer_sequence(artifact_store_roots, state, &fixture.source)
            .expect_err("missing input ref should fail");

    assert!(error
        .to_string()
        .contains("raster PLE state is missing input activation ref"));
}

#[test]
fn mixed_ple_and_non_ple_layers_match_native_layout() {
    let fixture = PleFixture::new(
        vec![true, false],
        vec![
            vec![
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
                vec![Act::from_num(1.0), Act::from_num(0.5)],
            ],
            vec![
                vec![Act::from_num(0.0), Act::from_num(0.0)],
                vec![Act::from_num(0.0), Act::from_num(0.0)],
            ],
        ],
        vec![
            vec![
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ],
            vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
                vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
            ],
        ],
    )
    .expect("fixture should build");
    let token_ids = [0];
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

    let (raster, native) = compare_raster_and_native(&fixture, &token_ids, &input);

    assert_eq!(raster.per_layer_inputs, native.per_layer_inputs);
    assert_eq!(raster.per_layer_inputs.len(), 2);
    assert!(raster.per_layer_inputs[0].is_some());
    assert!(raster.per_layer_inputs[1].is_none());
    assert!(raster
        .clone_layer_internal(0)
        .expect("layer 0")
        .det_values()
        .is_some());
    assert!(raster.clone_layer_internal(1).is_none());
    assert!(native.clone_layer_internal(1).is_none());
}

#[test]
fn empty_activation_sequence_fails_like_native_ple_computation() {
    let fixture = PleFixture::single_layer().expect("fixture should build");
    let input = activation_sequence(Vec::new());

    let error = run_materialized(&[], &input, &fixture.source, 1)
        .expect_err("empty activations should fail");

    assert!(error
        .to_string()
        .contains("requires at least one activation row"));
}

#[test]
fn token_and_activation_count_mismatch_fails_clearly() {
    let fixture = PleFixture::single_layer().expect("fixture should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

    let error =
        run_materialized(&[0, 1], &input, &fixture.source, 1).expect_err("mismatch should fail");

    assert!(error
        .to_string()
        .contains("token ids and activations to have matching lengths"));
}

#[test]
fn source_construction_reports_token_embedding_layer_mismatch() {
    let error = AuthenticatedGemmaPleSource::from_canonical_parts(
        "bad-token-layers",
        vec![
            GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 2,
            },
            GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 2,
            },
        ],
        vec![vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]],
        vec![identity_projection(), identity_projection()],
        vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
        test_scalars(),
    )
    .expect_err("token embedding layer mismatch should fail");

    assert!(error
        .to_string()
        .contains("token embedding slice count mismatch"));
}

#[test]
fn source_construction_reports_model_projection_layer_mismatch() {
    let error = AuthenticatedGemmaPleSource::from_canonical_parts(
        "bad-projection-layers",
        vec![
            GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 2,
            },
            GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 2,
            },
        ],
        vec![
            vec![vec![Act::from_num(1.0), Act::from_num(0.0)]],
            vec![vec![Act::from_num(0.0), Act::from_num(1.0)]],
        ],
        vec![identity_projection()],
        vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
        test_scalars(),
    )
    .expect_err("projection layer mismatch should fail");

    assert!(error
        .to_string()
        .contains("model projection slice count mismatch"));
}

#[test]
fn unsupported_fp32_only_ple_source_fails_closed() {
    let ple_global = Gemma4PleGlobalWeights::from_materialized(
        vec![MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![1.0, 0.0, 0.0, 1.0],
        }],
        vec![MatrixF32 {
            rows: 2,
            cols: 2,
            values: vec![1.0, 0.0, 0.0, 1.0],
        }],
        vec![1.0, 1.0],
        1.0,
        1.0,
        1.0,
    );

    let error = AuthenticatedGemmaPleSource::from_ple_global(
        "fp32-ple",
        Gemma4ModelProvenance::DetNumWgt,
        vec![GemmaPleLayerConfig {
            has_ple: true,
            hidden_width: 2,
        }],
        Some(ple_global),
        Some(f32_to_acc(0.0)),
    )
    .expect_err("fp32-only PLE source should fail");

    assert!(error.to_string().contains(".detwgt token embedding source"));
}

struct PleFixture {
    source: AuthenticatedGemmaPleSource,
    native_ple_global: Gemma4PleGlobalWeights,
    layers: Vec<Gemma4LayerWeights>,
    _weights_file: PathBuf,
}

impl Drop for PleFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self._weights_file);
    }
}

impl PleFixture {
    fn single_layer() -> Result<Self> {
        Self::new(
            vec![true],
            vec![vec![
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
                vec![Act::from_num(1.0), Act::from_num(0.5)],
            ]],
            vec![vec![
                vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
                vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            ]],
        )
    }

    fn new(
        has_ple_layers: Vec<bool>,
        token_embeddings: Vec<Vec<Vec<Act>>>,
        model_projections: Vec<Vec<Vec<Wgt>>>,
    ) -> Result<Self> {
        let hidden_width = model_projections
            .first()
            .and_then(|layer| layer.first())
            .map(Vec::len)
            .context("fixture requires at least one projection row")?;
        let ple_width = token_embeddings
            .first()
            .and_then(|layer| layer.first())
            .map(Vec::len)
            .context("fixture requires at least one token embedding row")?;
        let layer_configs = has_ple_layers
            .iter()
            .copied()
            .map(|has_ple| GemmaPleLayerConfig {
                has_ple,
                hidden_width,
            })
            .collect::<Vec<_>>();
        let source = AuthenticatedGemmaPleSource::from_canonical_parts(
            "ple-raster-fixture",
            layer_configs,
            token_embeddings.clone(),
            model_projections.clone(),
            vec![Wgt::from_num(1.0); ple_width],
            test_scalars(),
        )?;
        let (weights_file, token_sources, projection_sources) =
            write_det_weights(token_embeddings.clone(), model_projections.clone())?;
        let native_ple_global = Gemma4PleGlobalWeights::from_det_num_sources_with_canonical(
            token_sources,
            projection_sources,
            vec![1.0; ple_width],
            vec![Wgt::from_num(1.0); ple_width],
            1.0,
            Act::from_num(1.0),
            1.0,
            Act::from_num(1.0),
            1.0,
            Act::from_num(1.0),
        );
        let layers = has_ple_layers
            .into_iter()
            .map(|has_ple| test_layer(hidden_width, has_ple))
            .collect();

        Ok(Self {
            source,
            native_ple_global,
            layers,
            _weights_file: weights_file,
        })
    }
}

fn compare_raster_and_native(
    fixture: &PleFixture,
    token_ids: &[u32],
    input: &ActivationSequence,
) -> (Gemma4PrefillPleInputs, Gemma4PrefillPleInputs) {
    let raster = run_materialized(token_ids, input, &fixture.source, 1)
        .expect("raster PLE should run")
        .expect("raster PLE inputs");
    let native = native_prefill_ple_inputs(fixture, token_ids, input);
    (raster, native)
}

fn run_materialized(
    token_ids: &[u32],
    input: &ActivationSequence,
    source: &AuthenticatedGemmaPleSource,
    projection_rows_per_tile: usize,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    run_materialized_with_sizing(
        token_ids,
        input,
        source,
        raster_sizing_with_projection_rows(projection_rows_per_tile),
    )
}

fn run_materialized_with_sizing(
    token_ids: &[u32],
    input: &ActivationSequence,
    source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let (artifact_store_roots, manifest_root) = run(token_ids, input, source, raster_sizing)?;
    let refs = crate::prefill_prepare_aux::prefill_ple_input_refs_from_manifest(
        artifact_store_roots,
        manifest_root.as_deref(),
    )?;
    crate::prefill_prepare_aux::materialize_prefill_ple_input_refs_for_trace(refs.as_ref())
}

fn native_prefill_ple_inputs(
    fixture: &PleFixture,
    token_ids: &[u32],
    input: &ActivationSequence,
) -> Gemma4PrefillPleInputs {
    crate::prefill_prepare_aux::native::compute_prefill_ple_inputs_internal(
        token_ids,
        input.clone_internal(),
        &fixture.layers,
        &fixture.native_ple_global,
        0.0,
        Some(f32_to_acc(0.0)),
        InferenceExecutionMode::Deterministic,
    )
    .expect("native PLE should run")
}

fn raster_sizing_with_projection_rows(projection_rows_per_tile: usize) -> RasterSizingControls {
    RasterSizingControls {
        projection_rows_per_tile,
        attention_kv_rows_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE,
        sequence_rows_per_tile: crate::InferenceControls::DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE,
        head_rows_per_tile: crate::InferenceControls::DEFAULT_RASTER_HEAD_ROWS_PER_TILE,
        tokenizer_bpe_pairs_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE,
        tokenizer_bpe_pieces_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE,
        output_byte_flush_bytes_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    }
}

fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
    let values = rows
        .iter()
        .map(|row| row.iter().copied().map(act_to_f32).collect())
        .collect::<Vec<Vec<f32>>>();
    ActivationSequence::from_internal(
        InternalActivationSequence::from_det_values(rows),
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&values),
    )
}

fn test_scalars() -> GemmaPleScalars {
    GemmaPleScalars {
        embedding_scale: Act::from_num(1.0),
        projection_scalar: Act::from_num(1.0),
        input_scale: Act::from_num(1.0),
        rms_norm_eps: f32_to_acc(0.0),
    }
}

fn identity_projection() -> Vec<Vec<Wgt>> {
    vec![
        vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
        vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
    ]
}

fn write_det_weights(
    token_embeddings: Vec<Vec<Vec<Act>>>,
    model_projections: Vec<Vec<Vec<Wgt>>>,
) -> Result<(
    PathBuf,
    Vec<DetNumTensorSliceSource>,
    Vec<DetNumTensorSliceSource>,
)> {
    let unique_suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique_counter = FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "raster-prefill-ple-{}-{}-{}-{}.detwgt",
        std::process::id(),
        unique_suffix,
        unique_counter,
        crate::trace::sha256_hex(&format!("{:?}{:?}", token_embeddings, model_projections))
    ));
    let mut bytes = Vec::new();
    let mut token_sources = Vec::new();
    let mut projection_sources = Vec::new();

    for matrix in &token_embeddings {
        let data_offset = bytes.len();
        for row in matrix {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        token_sources.push(det_source(
            &path,
            matrix.len(),
            matrix[0].len(),
            data_offset,
        ));
    }

    for matrix in &model_projections {
        let data_offset = bytes.len();
        for row in matrix {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        projection_sources.push(det_source(
            &path,
            matrix.len(),
            matrix[0].len(),
            data_offset,
        ));
    }

    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
    Ok((path, token_sources, projection_sources))
}

fn det_source(
    path: &std::path::Path,
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

fn test_layer(hidden_width: usize, has_ple: bool) -> Gemma4LayerWeights {
    Gemma4LayerWeights {
        attention_kind: Gemma4AttentionKind::Full,
        hidden_size: hidden_width,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden_width,
        sliding_window: None,
        cache_sliding_window: None,
        rms_norm_eps: 0.0,
        rms_norm_eps_det: Some(f32_to_acc(0.0)),
        rope_base: 10_000.0,
        rope_base_det: None,
        partial_rotary_dim: 0,
        rope_freq_base_dim: 2,
        kv_shared_layer_index: None,
        attention_k_eq_v: false,
        q_proj: zero_layer_matrix(hidden_width),
        k_proj: zero_layer_matrix(hidden_width),
        v_proj: Some(zero_layer_matrix(hidden_width)),
        o_proj: zero_layer_matrix(hidden_width),
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
        gate_proj: zero_layer_matrix(hidden_width),
        up_proj: zero_layer_matrix(hidden_width),
        down_proj: zero_layer_matrix(hidden_width),
        ple: has_ple.then(|| Gemma4PleLayerWeights {
            input_gate: zero_layer_matrix(hidden_width),
            layer_projection: zero_layer_matrix(hidden_width),
            post_input_norm_weight: vec![1.0; hidden_width],
            post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        }),
        layer_scalar: None,
        layer_scalar_det: None,
    }
}

fn no_ple_model(hidden_width: usize) -> Gemma4TransformerModel {
    Gemma4TransformerModel {
        provenance: Gemma4ModelProvenance::DetNumWgt,
        embedding_table: None,
        embedding_source: None,
        layers: vec![test_layer(hidden_width, false)],
        ple_global: None,
        final_norm_weight: vec![1.0; hidden_width],
        final_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        logits_projection: Gemma4LogitsProjection::UntiedLmHead {
            weight: MatrixF32 {
                rows: 1,
                cols: hidden_width,
                values: vec![0.0; hidden_width],
            },
            det_weight: None,
        },
        final_logit_softcapping: None,
        final_logit_softcapping_det: None,
        rms_norm_eps: 0.0,
        rms_norm_eps_det: Some(f32_to_acc(0.0)),
    }
}

fn zero_layer_matrix(width: usize) -> Gemma4LayerMatrixSource {
    Gemma4LayerMatrixSource::from(MatrixF32 {
        rows: width,
        cols: width,
        values: vec![0.0; width * width],
    })
}
