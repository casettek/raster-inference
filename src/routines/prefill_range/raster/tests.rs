use crate::dsl::prelude::auth_read;
use crate::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::prefill_range::native::deterministic_tiles;
use crate::prefill_range::{
    materialize_prefill_layer_output_refs_from_roots_for_trace, run_raster,
};
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
use crate::shared::model::transformer::{
    ActivationSequence, DetNumTensorSliceSource, Gemma4AttentionKind, Gemma4LayerMatrixSource,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleLayerWeights,
    Gemma4PrefillPleInputs, Gemma4TransformerModel, InternalActivationSequence, MatrixF32,
};
use crate::shared::numerics::det_num::{Acc, Act, Wgt};
use crate::shared::raster_contracts::prefill_layer::{
    AuthenticatedGemmaPrefillLayerSource, GemmaPrefillLayerSourceMetadataRequest,
};
use crate::shared::raster_contracts::prefill_ple::store_prefill_ple_input_manifest_with_roots;
use crate::shared::raster_kernels::transformer::RasterActivationSequence;
use crate::shared::tensors::raster_tensor_artifacts::insert_activation_sequence_artifact_ref;
use crate::RasterSizingControls;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

fn raster_sizing(projection_rows_per_tile: usize) -> RasterSizingControls {
    RasterSizingControls {
        projection_rows_per_tile,
        attention_kv_rows_per_tile: usize::MAX,
        sequence_rows_per_tile: 1,
        head_rows_per_tile: 1,
        prefill_token_range_width: crate::InferenceControls::DEFAULT_PREFILL_TOKEN_RANGE_WIDTH,
        decode_layer_range_width: crate::InferenceControls::DEFAULT_DECODE_LAYER_RANGE_WIDTH,
        tokenizer_bpe_pairs_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE,
        tokenizer_bpe_pieces_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE,
        output_byte_flush_bytes_per_tile:
            crate::InferenceControls::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE,
    }
}

#[test]
fn ref_backed_prefill_helpers_do_not_hide_completion_loops() {
    let source = include_str!("tiles.rs");
    let forbidden = concat!("while !", "state.is_complete()");

    assert!(
        !source.contains(forbidden),
        "ref-backed prefill helpers must expose dynamic loops through recursive tile calls"
    );
}

#[test]
fn prefill_layer_sequences_are_branch_free_orchestration() {
    let source = include_str!("tiles.rs");
    let violations = branch_free_sequence_violations(source);

    assert!(
        violations.is_empty(),
        "prefill_layer sequences must be straight-line tile/sequence calls: {violations:?}"
    );
}

#[test]
fn prefill_layer_tiles_do_not_call_tiles_or_sequences() {
    let source = include_str!("tiles.rs");
    let violations = tile_call_violations(source);

    assert!(
        violations.is_empty(),
        "prefill_layer tiles must not invoke tile/sequence calls: {violations:?}"
    );
}

#[test]
fn prefill_layer_tiles_do_not_directly_call_authored_functions() {
    let source = include_str!("tiles.rs");
    let violations = tile_authored_call_violations(source);

    assert!(
        violations.is_empty(),
        "prefill_layer tiles must not directly invoke authored tile/sequence functions: {violations:?}"
    );
}

#[test]
fn prefill_layer_tiles_rs_contains_only_authored_functions() {
    let source = include_str!("tiles.rs");
    let violations = unauthored_function_violations(source);

    assert!(
        violations.is_empty(),
        "every function in prefill_layer tiles.rs must be immediately authored as a tile or sequence: {violations:?}"
    );
}

#[test]
fn branch_free_sequence_guard_rejects_inline_branches() {
    let source = r#"
#[sequence]
fn assigned_branch() -> Result<()> {
    let next = if true { call_tile!(a)? } else { call_tile!(b)? };
    call_tile!(finish, next)
}

#[sequence]
fn nested_match() -> Result<()> {
    let state = call_tile!(start)?;
    let next = match state { State::Done => state };
    call_tile!(finish, next)
}
"#;
    let violations = branch_free_sequence_violations(source);

    assert!(
        violations.iter().any(|violation| violation.contains("if")),
        "guard should reject assigned `if` branches: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("match")),
        "guard should reject assigned `match` branches: {violations:?}"
    );
}

#[test]
fn authored_function_guard_rejects_plain_functions() {
    let source = r#"
#[tile]
fn authored_tile() -> Result<()> {
    Ok(())
}

fn plain_helper() -> Result<()> {
    Ok(())
}
"#;
    let violations = unauthored_function_violations(source);

    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("plain_helper")),
        "guard should reject plain helper functions: {violations:?}"
    );
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
                .expect("sequence attribute should be followed by a function")
            + "fn ".len();
        let name_end = fn_start
            + source[fn_start..]
                .find('(')
                .expect("sequence function should have a parameter list");
        let sequence_name = source[fn_start..name_end].trim().to_string();
        let body_start = name_end
            + source[name_end..]
                .find('{')
                .expect("sequence function should have a body");
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

        bodies.push((sequence_name, source[body_start + 1..body_end].to_string()));
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

#[test]
fn authored_prefill_layer_surfaces_do_not_accept_materialized_stage_inputs() {
    let source = include_str!("tiles.rs");
    let materialized_ple_type = concat!("Gemma4", "PrefillPleInputs");
    let materialized_activation_arg = concat!("&", "ActivationSequence");
    let lines = source.lines().collect::<Vec<_>>();
    let mut index = 0;

    while index < lines.len() {
        let marker = lines[index].trim();
        if marker == "#[tile]" || marker == "#[sequence]" {
            let mut signature = String::new();
            index += 1;
            while index < lines.len() {
                signature.push_str(lines[index]);
                signature.push('\n');
                if lines[index].contains('{') {
                    break;
                }
                index += 1;
            }
            assert!(
                !signature.contains(materialized_ple_type),
                "authored raster tile/sequence must not accept materialized PLE inputs: {signature}"
            );
            assert!(
                !signature.contains(materialized_activation_arg),
                "authored raster tile/sequence must not accept materialized activation inputs: {signature}"
            );
        }
        index += 1;
    }

    let materialized_compat_fn = concat!("pub fn ", "run_materialized_compat(");
    let compatibility_comment = concat!("Compatibility adapter", " for dev/tests");
    assert!(!source.contains(materialized_compat_fn));
    assert!(!source.contains(compatibility_comment));
}

#[test]
fn prefill_layer_main_threads_roots_without_local_tensor_store() {
    let source = include_str!("tiles.rs");
    let main_start = source
        .find("pub fn main(\n    artifact_store_roots: RasterArtifactStoreRoots,")
        .expect("prefill layer main should exist");
    let after_main = &source[main_start..];
    let next_tile = after_main
        .find("\n#[tile]\npub fn init_prefill_sequence_projection")
        .expect("next authored helper marks end of main");
    let main_body = &after_main[..next_tile];

    assert!(main_body.contains("compute_next_prefill_layer_sequence_with_roots"));
    assert!(
        !main_body.contains(&format!(
            "{}::new()",
            concat!("Authenticated", "RasterTensorStore")
        )),
        "proof-shaped prefill main must thread artifact roots instead of creating a tensor store"
    );
}

#[test]
fn single_layer_no_ple_matches_deterministic_prefill_layer() {
    let (_path, model) = no_ple_model();

    assert_raster_matches_deterministic(
        &model,
        vec![vec![Act::from_num(1.0), Act::from_num(-0.5)]],
    );
}

#[test]
fn sliding_attention_matches_deterministic_prefill_layer() {
    let (_path, mut model) = no_ple_model();
    model.layers[0].attention_kind = Gemma4AttentionKind::Sliding;
    model.layers[0].sliding_window = Some(1);
    model.layers[0].cache_sliding_window = Some(1);

    let raster = assert_raster_matches_deterministic(
        &model,
        vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.5), Act::from_num(-0.5)],
            vec![Act::from_num(-1.0), Act::from_num(1.0)],
        ],
    );
    assert_eq!(raster.1[0].current_len(), 1);
}

#[test]
fn attention_k_equals_v_matches_deterministic_prefill_layer() {
    let (_path, mut model) = no_ple_model();
    model.layers[0].v_proj = None;
    model.layers[0].attention_k_eq_v = true;

    assert_raster_matches_deterministic(
        &model,
        vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ],
    );
}

#[test]
fn donor_kv_sharing_matches_deterministic_prefill_layer() {
    let (_path, mut model) = no_ple_model();
    let mut donor_layer = model.layers[0].clone();
    donor_layer.kv_shared_layer_index = Some(0);
    model.layers.push(donor_layer);

    let raster = assert_raster_matches_deterministic(
        &model,
        vec![
            vec![Act::from_num(1.0), Act::from_num(0.0)],
            vec![Act::from_num(0.0), Act::from_num(1.0)],
        ],
    );
    assert_eq!(raster.1.len(), 2);
    assert_eq!(raster.1[1].current_len(), 0);
}

#[test]
fn zero_length_self_cache_materializes_as_empty_slot() {
    let (_path, mut model) = no_ple_model();
    model.layers[0].attention_kind = Gemma4AttentionKind::Sliding;
    model.layers[0].sliding_window = Some(1);
    model.layers[0].cache_sliding_window = Some(0);
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
    let raster = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
        .expect("zero-length self cache should succeed");

    assert_eq!(raster.1.len(), 1);
    assert_eq!(raster.1[0].current_len(), 0);
}

#[test]
fn empty_donor_cache_fails_closed() {
    let (_path, mut model) = no_ple_model();
    let mut shared_layer = model.layers[0].clone();
    shared_layer.kv_shared_layer_index = Some(0);
    let mut chained_layer = model.layers[0].clone();
    chained_layer.kv_shared_layer_index = Some(1);
    model.layers.push(shared_layer);
    model.layers.push(chained_layer);
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

    let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
        .expect_err("empty donor cache should fail");

    assert!(error.to_string().contains("donor cache is empty"));
}

#[test]
fn multi_head_sliding_attention_matches_deterministic_prefill_layer() {
    let (_path, model) = multi_head_sliding_model();

    let raster = assert_raster_matches_deterministic(
        &model,
        vec![
            vec![
                Act::from_num(1.0),
                Act::from_num(0.0),
                Act::from_num(-0.5),
                Act::from_num(0.25),
            ],
            vec![
                Act::from_num(0.25),
                Act::from_num(0.75),
                Act::from_num(0.5),
                Act::from_num(-0.25),
            ],
            vec![
                Act::from_num(-1.0),
                Act::from_num(1.0),
                Act::from_num(0.0),
                Act::from_num(0.5),
            ],
        ],
    );
    assert_eq!(raster.1[0].current_len(), 2);
}

#[test]
fn non_prior_donor_cache_fails_closed() {
    let (_path, mut model) = no_ple_model();
    model.layers[0].kv_shared_layer_index = Some(0);
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

    let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
        .expect_err("self donor should fail");

    assert!(error
        .to_string()
        .contains("cannot share KV with non-prior donor"));
}

#[test]
fn nonzero_mlp_and_layer_scalar_match_deterministic_prefill_layer() {
    let (_path, model) = nonzero_model(false, true);

    assert_raster_matches_deterministic(
        &model,
        vec![
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
            vec![Act::from_num(0.25), Act::from_num(0.75)],
        ],
    );
}

#[test]
fn chunked_projection_matches_deterministic_prefill_layer() {
    let (_path, model) = nonzero_model(false, true);
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let rows = vec![
        vec![Act::from_num(1.0), Act::from_num(-0.5)],
        vec![Act::from_num(0.25), Act::from_num(0.75)],
    ];
    let input_internal = InternalActivationSequence::from_det_values(rows);
    let input = activation_sequence_from_internal(input_internal.clone());

    let raster = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(2))
        .expect("raster prefill layer should run");
    let deterministic = deterministic_tiles::run_internal(input_internal, &model, None)
        .expect("deterministic prefill layer should run");

    assert_eq!(
        raster.0.clone_internal().det_values().map(<[Vec<Act>]>::to_vec),
        deterministic.0.clone_internal().det_values().map(<[Vec<Act>]>::to_vec)
    );
    assert_eq!(
        raster.0.det_activations_sha256,
        deterministic.0.det_activations_sha256
    );
    assert_eq!(raster.1, deterministic.1);
}

#[test]
fn ple_layer_with_matching_input_matches_deterministic_prefill_layer() {
    let (_path, model) = nonzero_model(true, false);
    let ple_inputs = ple_inputs(vec![
        vec![Act::from_num(0.5), Act::from_num(-0.25)],
        vec![Act::from_num(1.0), Act::from_num(0.25)],
    ]);

    assert_raster_matches_deterministic_with_ple(
        &model,
        vec![
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
            vec![Act::from_num(0.25), Act::from_num(0.75)],
        ],
        Some(&ple_inputs),
    );
}

#[test]
fn ple_input_width_can_differ_from_hidden_size() {
    let (_path, model) = ple_width_differs_from_hidden_model();
    let ple_inputs = ple_inputs(vec![
        vec![Act::from_num(0.5), Act::from_num(-0.25)],
        vec![Act::from_num(1.0), Act::from_num(0.25)],
    ]);

    assert_raster_matches_deterministic_with_ple(
        &model,
        vec![
            vec![
                Act::from_num(1.0),
                Act::from_num(-0.5),
                Act::from_num(0.25),
                Act::from_num(0.75),
            ],
            vec![
                Act::from_num(0.25),
                Act::from_num(0.75),
                Act::from_num(-0.5),
                Act::from_num(1.0),
            ],
        ],
        Some(&ple_inputs),
    );
}

#[test]
fn ple_layer_missing_input_fails_closed() {
    let (_path, model) = nonzero_model(true, false);
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

    let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
        .expect_err("missing PLE input should fail");

    assert!(error
        .to_string()
        .contains("requires PLE inputs but none were provided"));
}

#[test]
fn ple_input_on_non_ple_layer_fails_closed() {
    let (_path, model) = no_ple_model();
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
    let ple_inputs = ple_inputs(vec![vec![Act::from_num(0.5), Act::from_num(0.25)]]);

    let error =
        run_roots_path_with_optional_ple(&input, &source, Some(&ple_inputs), raster_sizing(1))
            .expect_err("PLE input should fail");

    assert!(error
        .to_string()
        .contains("received PLE inputs without PLE weights"));
}

#[test]
fn ple_input_shape_mismatch_fails_closed() {
    let (_path, model) = nonzero_model(true, false);
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
    let ple_inputs = ple_inputs(vec![vec![
        Act::from_num(0.5),
        Act::from_num(0.25),
        Act::from_num(0.125),
    ]]);

    let error =
        run_roots_path_with_optional_ple(&input, &source, Some(&ple_inputs), raster_sizing(1))
            .expect_err("PLE width should fail");

    assert!(error
        .to_string()
        .contains("transformer layer PLE input width 3, expected 2"));
}

#[test]
fn zero_layer_source_fails_closed() {
    let model = Gemma4TransformerModel {
        provenance: Gemma4ModelProvenance::DetNumWgt,
        embedding_table: None,
        embedding_source: None,
        layers: Vec::new(),
        ple_global: None,
        final_norm_weight: vec![1.0, 1.0],
        final_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(1.0)]),
        logits_projection: Gemma4LogitsProjection::UntiedLmHead {
            weight: MatrixF32 {
                rows: 1,
                cols: 2,
                values: vec![0.0, 0.0],
            },
            det_weight: None,
        },
        final_logit_softcapping: None,
        final_logit_softcapping_det: None,
        rms_norm_eps: 0.0,
        rms_norm_eps_det: Some(Acc::from_num(0.0)),
    };
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("empty", &model)
        .expect("source should build");
    let input = activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

    let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
        .expect_err("zero layers should fail");

    assert!(error
        .to_string()
        .contains("transformer prefill requires at least one layer"));
}

#[test]
fn empty_activation_sequence_fails_closed() {
    let (_path, model) = no_ple_model();
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = activation_sequence(Vec::new());

    let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
        .expect_err("empty input should fail");

    assert!(error
        .to_string()
        .contains("requires at least one activation row"));
}

#[test]
fn non_deterministic_activation_input_fails_closed() {
    let (_path, model) = no_ple_model();
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", &model)
        .expect("source should build");
    let input = ActivationSequence::from_values(
        vec![vec![1.0, 0.0]],
        crate::shared::numerics::transformer_kernels::build_activation_commitment(&[vec![
            1.0, 0.0,
        ]]),
    );

    let error = run_roots_path_with_optional_ple(&input, &source, None, raster_sizing(1))
        .expect_err("f32-only input should fail");

    assert!(error.to_string().contains("requires canonical activations"));
}

fn assert_raster_matches_deterministic(
    model: &Gemma4TransformerModel,
    rows: Vec<Vec<Act>>,
) -> (
    ActivationSequence,
    Vec<crate::shared::model::transformer::LayerKvCache>,
) {
    assert_raster_matches_deterministic_with_ple(model, rows, None)
}

fn assert_raster_matches_deterministic_with_ple(
    model: &Gemma4TransformerModel,
    rows: Vec<Vec<Act>>,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> (
    ActivationSequence,
    Vec<crate::shared::model::transformer::LayerKvCache>,
) {
    let source = AuthenticatedGemmaPrefillLayerSource::from_model("prefill-layer", model)
        .expect("source should build");
    let input_internal = InternalActivationSequence::from_det_values(rows);
    let input = activation_sequence_from_internal(input_internal.clone());

    let raster = run_roots_path_with_optional_ple(&input, &source, ple_inputs, raster_sizing(1))
        .expect("raster prefill layer should run");
    let deterministic = deterministic_tiles::run_internal(input_internal, model, ple_inputs)
        .expect("deterministic prefill layer should run");

    assert_eq!(
        raster.0.clone_internal().det_values().map(<[Vec<Act>]>::to_vec),
        deterministic.0.clone_internal().det_values().map(<[Vec<Act>]>::to_vec)
    );
    assert_eq!(
        raster.0.det_activations_sha256,
        deterministic.0.det_activations_sha256
    );
    assert_eq!(raster.1, deterministic.1);
    raster
}

fn run_roots_path_with_optional_ple(
    input: &ActivationSequence,
    source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
    raster_sizing: RasterSizingControls,
) -> Result<(
    ActivationSequence,
    Vec<crate::shared::model::transformer::LayerKvCache>,
)> {
    ArtifactIo::reset_store();
    let input_internal = input.clone_internal();
    let input_rows = input_internal
        .det_values()
        .ok_or_else(|| anyhow::anyhow!("raster prefill layer requires canonical activations"))?;
    let input_ref = insert_activation_sequence_artifact_ref(
        "prefill.layer.test.input_embedding",
        RasterActivationSequence::from_acts(input_rows.to_vec()),
    )?;
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let input_embedding_refs = RasterInputEmbeddingRefs {
        source_id: "embedding-fixture".to_string(),
        embedding_source_root: "embedding-root".to_string(),
        prompt_token_ids_root: "token-root".to_string(),
        prompt_token_count: input_ref.row_count(),
        embedded_prompt_activations_ref: input_ref,
    };
    let (artifact_store_roots, ple_input_manifest_root) =
        store_materialized_ple_inputs_with_roots(artifact_store_roots, source, ple_inputs)?;
    let (artifact_store_roots, refs) = run_raster(
        artifact_store_roots,
        &input_embedding_refs,
        source,
        ple_input_manifest_root.as_deref(),
        raster_sizing,
    )?;
    materialize_prefill_layer_output_refs_from_roots_for_trace(&artifact_store_roots, &refs)
}

fn store_materialized_ple_inputs_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    source: &AuthenticatedGemmaPrefillLayerSource,
    ple_inputs: Option<&Gemma4PrefillPleInputs>,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    let Some(ple_inputs) = ple_inputs else {
        return Ok((artifact_store_roots, None));
    };
    let metadata = auth_read!(source, GemmaPrefillLayerSourceMetadataRequest)?;
    let mut token_count = None;
    let mut per_layer_inputs = Vec::with_capacity(metadata.layer_count);

    for layer_idx in 0..metadata.layer_count {
        let input = ple_inputs.clone_layer_internal(layer_idx);
        let Some(input) = input else {
            per_layer_inputs.push(None);
            continue;
        };
        let rows = input
            .det_values()
            .ok_or_else(|| anyhow::anyhow!("raster PLE inputs require canonical activations"))?;
        match token_count {
            Some(expected) if expected != rows.len() => {
                anyhow::bail!(
                    "materialized PLE input layer {layer_idx} contains {} tokens, expected {expected}",
                    rows.len()
                );
            }
            None => token_count = Some(rows.len()),
            _ => {}
        }
        per_layer_inputs.push(Some(insert_activation_sequence_artifact_ref(
            &format!("prefill.layer.test.ple.{layer_idx}"),
            RasterActivationSequence::from_acts(rows.to_vec()),
        )?));
    }

    if per_layer_inputs.iter().all(Option::is_none) {
        return Ok((ArtifactIo::export_store_roots(), None));
    }

    let artifact_store_roots = ArtifactIo::export_store_roots();
    let (artifact_store_roots, manifest_root) = store_prefill_ple_input_manifest_with_roots(
        &artifact_store_roots,
        metadata.source_id,
        metadata.layer_count,
        token_count
            .ok_or_else(|| anyhow::anyhow!("materialized PLE inputs contained no layer rows"))?,
        &per_layer_inputs,
    )?;
    Ok((artifact_store_roots, Some(manifest_root)))
}

fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
    activation_sequence_from_internal(InternalActivationSequence::from_det_values_only(rows))
}

fn activation_sequence_from_internal(
    input_internal: InternalActivationSequence,
) -> ActivationSequence {
    // Single-track deterministic fixture: canonical commitment only.
    let det_activations_sha256 = Some(
        crate::shared::numerics::transformer_kernels::build_det_activation_commitment(
            input_internal.det_values().expect("det input"),
        ),
    );
    ActivationSequence::from_det_internal(input_internal, det_activations_sha256)
}

fn ple_inputs(rows: Vec<Vec<Act>>) -> Gemma4PrefillPleInputs {
    Gemma4PrefillPleInputs::from_internal(vec![Some(InternalActivationSequence::from_det_values(
        rows,
    ))])
}

fn no_ple_model() -> (PathBuf, Gemma4TransformerModel) {
    let hidden_width = 2;
    let matrices = vec![
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
    let layer = Gemma4LayerWeights {
        attention_kind: Gemma4AttentionKind::Full,
        hidden_size: hidden_width,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden_width,
        sliding_window: None,
        cache_sliding_window: None,
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
            embedding_source: None,
            layers: vec![layer],
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
            rms_norm_eps_det: Some(Acc::from_num(0.0)),
        },
    )
}

fn nonzero_model(has_ple: bool, has_layer_scalar: bool) -> (PathBuf, Gemma4TransformerModel) {
    let hidden_width = 2;
    let matrices = vec![
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        zero_matrix(hidden_width),
        identity_matrix(),
        identity_matrix(),
        identity_matrix(),
        identity_matrix(),
        identity_matrix(),
    ];
    let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
    let mut sources = sources.into_iter();
    let q_proj = det_matrix(sources.next().expect("q source"));
    let k_proj = det_matrix(sources.next().expect("k source"));
    let v_proj = det_matrix(sources.next().expect("v source"));
    let o_proj = det_matrix(sources.next().expect("o source"));
    let gate_proj = det_matrix(sources.next().expect("gate source"));
    let up_proj = det_matrix(sources.next().expect("up source"));
    let down_proj = det_matrix(sources.next().expect("down source"));
    let ple_input_gate = det_matrix(sources.next().expect("PLE input gate source"));
    let ple_layer_projection = det_matrix(sources.next().expect("PLE projection source"));

    let layer = Gemma4LayerWeights {
        attention_kind: Gemma4AttentionKind::Full,
        hidden_size: hidden_width,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden_width,
        sliding_window: None,
        cache_sliding_window: None,
        rms_norm_eps: 0.001,
        rms_norm_eps_det: Some(Acc::from_num(0.001)),
        rope_base: 10_000.0,
        rope_base_det: None,
        partial_rotary_dim: 0,
        rope_freq_base_dim: 2,
        kv_shared_layer_index: None,
        attention_k_eq_v: false,
        q_proj,
        k_proj,
        v_proj: Some(v_proj),
        o_proj,
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
        gate_proj,
        up_proj,
        down_proj,
        ple: has_ple.then(|| Gemma4PleLayerWeights {
            input_gate: ple_input_gate,
            layer_projection: ple_layer_projection,
            post_input_norm_weight: vec![1.0; hidden_width],
            post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        }),
        layer_scalar: has_layer_scalar.then_some(0.5),
        layer_scalar_det: has_layer_scalar.then_some(Act::from_num(0.5)),
    };

    (
        path,
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::DetNumWgt,
            embedding_table: None,
            embedding_source: None,
            layers: vec![layer],
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
            rms_norm_eps: 0.001,
            rms_norm_eps_det: Some(Acc::from_num(0.001)),
        },
    )
}

fn ple_width_differs_from_hidden_model() -> (PathBuf, Gemma4TransformerModel) {
    let hidden_width = 4;
    let ple_width = 2;
    let matrices = vec![
        zero_matrix_rect(hidden_width, hidden_width),
        zero_matrix_rect(hidden_width, hidden_width),
        zero_matrix_rect(hidden_width, hidden_width),
        zero_matrix_rect(hidden_width, hidden_width),
        zero_matrix_rect(hidden_width, hidden_width),
        zero_matrix_rect(hidden_width, hidden_width),
        zero_matrix_rect(hidden_width, hidden_width),
        zero_matrix_rect(ple_width, hidden_width),
        zero_matrix_rect(hidden_width, ple_width),
    ];
    let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
    let mut sources = sources.into_iter();
    let layer = Gemma4LayerWeights {
        attention_kind: Gemma4AttentionKind::Full,
        hidden_size: hidden_width,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden_width,
        sliding_window: None,
        cache_sliding_window: None,
        rms_norm_eps: 0.001,
        rms_norm_eps_det: Some(Acc::from_num(0.001)),
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
        ple: Some(Gemma4PleLayerWeights {
            input_gate: det_matrix(sources.next().expect("PLE input gate source")),
            layer_projection: det_matrix(sources.next().expect("PLE projection source")),
            post_input_norm_weight: vec![1.0; hidden_width],
            post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); hidden_width]),
        }),
        layer_scalar: None,
        layer_scalar_det: None,
    };

    (
        path,
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::DetNumWgt,
            embedding_table: None,
            embedding_source: None,
            layers: vec![layer],
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
            rms_norm_eps: 0.001,
            rms_norm_eps_det: Some(Acc::from_num(0.001)),
        },
    )
}

fn multi_head_sliding_model() -> (PathBuf, Gemma4TransformerModel) {
    let hidden_width = 4;
    let matrices = vec![
        zero_matrix_rect(4, 4),
        zero_matrix_rect(2, 4),
        zero_matrix_rect(2, 4),
        zero_matrix_rect(4, 4),
        zero_matrix_rect(8, 4),
        zero_matrix_rect(8, 4),
        zero_matrix_rect(4, 8),
    ];
    let (path, sources) = write_det_matrices(matrices).expect("fixture weights should write");
    let mut sources = sources.into_iter();
    let layer = Gemma4LayerWeights {
        attention_kind: Gemma4AttentionKind::Sliding,
        hidden_size: hidden_width,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        sliding_window: Some(2),
        cache_sliding_window: Some(2),
        rms_norm_eps: 0.0,
        rms_norm_eps_det: Some(Acc::from_num(0.0)),
        rope_base: 10_000.0,
        rope_base_det: Some(Acc::from_num(10_000.0)),
        partial_rotary_dim: 0,
        rope_freq_base_dim: 2,
        kv_shared_layer_index: None,
        attention_k_eq_v: false,
        q_proj: det_matrix(sources.next().expect("q source")),
        k_proj: det_matrix(sources.next().expect("k source")),
        v_proj: Some(det_matrix(sources.next().expect("v source"))),
        o_proj: det_matrix(sources.next().expect("o source")),
        q_norm_weight: vec![1.0; 2],
        q_norm_weight_det: Some(vec![Wgt::from_num(1.0); 2]),
        k_norm_weight: vec![1.0; 2],
        k_norm_weight_det: Some(vec![Wgt::from_num(1.0); 2]),
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
            embedding_source: None,
            layers: vec![layer],
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
            rms_norm_eps_det: Some(Acc::from_num(0.0)),
        },
    )
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

fn zero_matrix_rect(rows: usize, cols: usize) -> Vec<Vec<Wgt>> {
    vec![vec![Wgt::from_num(0.0); cols]; rows]
}

fn det_matrix(source: DetNumTensorSliceSource) -> Gemma4LayerMatrixSource {
    Gemma4LayerMatrixSource::from_det_num_source(source)
}

fn write_det_matrices(
    matrices: Vec<Vec<Vec<Wgt>>>,
) -> Result<(PathBuf, Vec<DetNumTensorSliceSource>)> {
    let unique_suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "raster-prefill-layer-state-{}-{}-{}.detwgt",
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
        element_width: crate::shared::numerics::det_num::DetWgtElementWidth::I32,
        row_offset: 0,
        row_count: rows,
        col_offset: 0,
        col_count: cols,
    }
}
