use anyhow::{anyhow, bail, Result};

use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile_result, call_tile, sequence, tile,
};
use crate::shared::det_num::{softcap_act, Act};
use crate::shared::raster_prefill_finalize::{
    AuthenticatedGemmaPrefillFinalizeSource, GemmaPrefillFinalizeMetadataRequest,
    GemmaPrefillFinalizeNormWeightsRequest, GemmaPrefillFinalizeProjectionRowRequest,
    GemmaPrefillFinalizeScalarsRequest,
};
use crate::shared::raster_row_store::{
    AuthenticatedRasterTensorStore, RasterProjectionOutputBuilderRef, RasterTensorId,
};
use crate::shared::raster_transformer_kernels::{
    project_row_with_weights, rms_norm_sequence, validate_projection_rows_per_tile,
    RasterActivationRow, RasterActivationSequence,
};
use crate::shared::transformer::{
    ActivationSequence, InternalLogits, LayerKvCache, PrefillLogits, TransformerPrefillResult,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillFinalizeRasterState {
    normalized_final_position: RasterActivationRow,
    next_logit_idx: usize,
    logit_count: usize,
    output_builder_ref: RasterProjectionOutputBuilderRef,
    softcap_bits: Option<i32>,
    projection_rows_per_tile: usize,
}

#[tile]
pub fn init_prefill_finalize_store() -> AuthenticatedRasterTensorStore {
    AuthenticatedRasterTensorStore::new()
}

#[tile]
pub fn select_final_position(
    final_hidden_states: &ActivationSequence,
) -> Result<RasterActivationRow> {
    // Public compatibility boundary: prefill.layer has already produced the
    // materialized final hidden state returned by the native API. zkVM-target
    // layer substeps should consume refs before this point.
    let internal = final_hidden_states.clone_internal();
    let det_rows = internal.det_values().ok_or_else(|| {
        anyhow!("deterministic raster prefill finalize requires canonical final hidden activations")
    })?;
    let final_row = det_rows.last().ok_or_else(|| {
        anyhow!("transformer final-position selection requires at least one activation row")
    })?;
    Ok(RasterActivationRow::from_acts(final_row.clone()))
}

#[tile]
pub fn normalize_final_position(
    final_position: RasterActivationRow,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
) -> Result<RasterActivationRow> {
    let norm_weights = auth_read!(finalize_source, GemmaPrefillFinalizeNormWeightsRequest)?;
    let scalars = auth_read!(finalize_source, GemmaPrefillFinalizeScalarsRequest)?;
    let normalized = rms_norm_sequence(
        &RasterActivationSequence::from_rows(vec![final_position]),
        Some(&norm_weights),
        Some(scalars.rms_norm_eps),
    )?;
    normalized
        .into_rows()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("deterministic final RMSNorm returned no rows"))
}

#[tile]
pub fn init_prefill_finalize_projection(
    store: &mut AuthenticatedRasterTensorStore,
    normalized_final_position: RasterActivationRow,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    projection_rows_per_tile: usize,
) -> Result<PrefillFinalizeRasterState> {
    validate_projection_rows_per_tile(projection_rows_per_tile)?;
    let metadata = auth_read!(finalize_source, GemmaPrefillFinalizeMetadataRequest)?;
    if metadata.projection_rows == 0 {
        bail!("deterministic logits projection requires at least one projection row");
    }
    if normalized_final_position.width() != metadata.hidden_width {
        bail!(
            "deterministic final logits projection input has width {}, expected {}",
            normalized_final_position.width(),
            metadata.hidden_width
        );
    }
    if metadata.projection_cols != metadata.hidden_width {
        bail!(
            "deterministic final logits projection metadata width mismatch: {} vs {}",
            metadata.projection_cols,
            metadata.hidden_width
        );
    }

    let scalars = auth_read!(finalize_source, GemmaPrefillFinalizeScalarsRequest)?;
    let output_builder_ref = store.start_projection_output_builder(
        RasterTensorId::new("prefill.finalize.logits")?,
        1,
        metadata.projection_rows,
    )?;
    Ok(PrefillFinalizeRasterState {
        normalized_final_position,
        next_logit_idx: 0,
        logit_count: metadata.projection_rows,
        output_builder_ref,
        softcap_bits: scalars.final_logit_softcapping.map(Act::to_bits),
        projection_rows_per_tile,
    })
}

#[tile(kind = recursive)]
pub fn project_next_prefill_logit(
    mut state: PrefillFinalizeRasterState,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    store: &mut AuthenticatedRasterTensorStore,
) -> Result<(bool, PrefillFinalizeRasterState)> {
    if state.next_logit_idx >= state.logit_count {
        return Ok((true, state));
    }

    let end = state
        .next_logit_idx
        .saturating_add(state.projection_rows_per_tile)
        .min(state.logit_count);
    let start_logit_idx = state.next_logit_idx;
    let mut logit_bits = Vec::with_capacity(end - start_logit_idx);
    while state.next_logit_idx < end {
        let projection_row = auth_read!(
            finalize_source,
            GemmaPrefillFinalizeProjectionRowRequest {
                row_idx: state.next_logit_idx,
            },
        )?;
        let mut logit =
            project_row_with_weights(&state.normalized_final_position, &projection_row)?;
        if let Some(softcap_bits) = state.softcap_bits {
            logit = softcap_act(logit, Act::from_bits(softcap_bits));
        }
        logit_bits.push(logit.to_bits());
        state.next_logit_idx += 1;
    }
    store.append_projection_output_chunk(
        &mut state.output_builder_ref,
        0,
        start_logit_idx,
        &logit_bits,
    )?;
    Ok((false, state))
}

#[tile]
pub fn finalize_prefill_result(
    store: &mut AuthenticatedRasterTensorStore,
    state: PrefillFinalizeRasterState,
    prompt_token_count: usize,
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
) -> Result<TransformerPrefillResult> {
    // Public compatibility boundary. The logits and layer caches are shaped
    // like the existing native `TransformerPrefillResult`; replay-critical
    // projection state above carries only the output builder ref/cursors.
    if state.next_logit_idx != state.logit_count {
        bail!(
            "raster prefill finalize completed {} logits, expected {}",
            state.next_logit_idx,
            state.logit_count
        );
    }
    let logits_ref = store.finalize_projection_output_builder(state.output_builder_ref)?;
    let logits_sequence = store.materialize_sequence(&logits_ref)?;
    let det_logits = logits_sequence
        .into_rows()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("raster prefill finalize produced no logits row"))?
        .acts();
    let internal_logits = InternalLogits::from_det_values(det_logits.clone());
    let final_logits_sha256 =
        crate::shared::transformer_kernels::build_vector_commitment(internal_logits.as_f32_slice());
    let mut prefill_logits = PrefillLogits::from_internal(internal_logits, final_logits_sha256);
    prefill_logits.det_final_logits_sha256 =
        Some(crate::shared::transformer_kernels::build_det_vector_commitment(&det_logits));

    super::build_prefill_result(
        prompt_token_count,
        final_hidden_states,
        layer_caches,
        prefill_logits,
    )
}

#[sequence]
pub fn run(
    prompt_token_ids: &[u32],
    final_hidden_states: ActivationSequence,
    layer_caches: Vec<LayerKvCache>,
    finalize_source: &AuthenticatedGemmaPrefillFinalizeSource,
    projection_rows_per_tile: usize,
) -> Result<TransformerPrefillResult> {
    crate::trace::trace_event("prefill.select_final_position");
    let final_position = call_tile!(select_final_position, &final_hidden_states)?;
    crate::trace::trace_event("prefill.project_to_logits");
    let normalized = call_tile!(normalize_final_position, final_position, finalize_source)?;
    let mut store = call_tile!(init_prefill_finalize_store);
    let state = call_tile!(
        init_prefill_finalize_projection,
        &mut store,
        normalized,
        finalize_source,
        projection_rows_per_tile
    )?;
    let state = call_recur_tile_result!(
        project_next_prefill_logit,
        state,
        finalize_source,
        &mut store
    )?;
    call_tile!(
        finalize_prefill_result,
        &mut store,
        state,
        prompt_token_ids.len(),
        final_hidden_states,
        layer_caches
    )
}

#[cfg(test)]
mod tests {
    use super::{init_prefill_finalize_projection, init_prefill_finalize_store, run};
    use crate::shared::det_num::{Acc, Act, Wgt};
    use crate::shared::input::InferenceExecutionMode;
    use crate::shared::raster_prefill_finalize::AuthenticatedGemmaPrefillFinalizeSource;
    use crate::shared::raster_transformer_kernels::RasterActivationRow;
    use crate::shared::transformer::{
        ActivationSequence, DetNumMatrix, DetNumTensorSliceSource, Gemma4LogitsProjection,
        Gemma4ModelProvenance, Gemma4TransformerModel, GemmaEmbeddingTensorSource,
        InternalActivationSequence, LayerKvCache, MatrixF32,
    };
    use anyhow::{Context, Result};
    use std::collections::VecDeque;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    #[test]
    fn raster_finalize_matches_native_deterministic_for_untied_projection() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states = activation_sequence(vec![
            vec![Act::from_num(0.25), Act::from_num(0.5)],
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
        ]);

        let raster = run(&[3, 4], final_hidden_states.clone(), vec![], &source, 1)
            .expect("raster finalize should run");
        let native = crate::prefill_finalize::run(
            &[3, 4],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .expect("native finalize should run");

        assert_eq!(
            raster.transformer_state.prefill_logits.logits,
            native.transformer_state.prefill_logits.logits
        );
        assert_eq!(
            raster
                .transformer_state
                .prefill_logits
                .det_final_logits_sha256,
            native
                .transformer_state
                .prefill_logits
                .det_final_logits_sha256
        );
        assert_eq!(raster.transformer_decode_state.position, 2);
        assert_eq!(raster.transformer_decode_state.token_count, 2);
    }

    #[test]
    fn finalize_projection_state_serializes_builder_not_logits() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_position =
            RasterActivationRow::from_acts(vec![Act::from_num(1.0), Act::from_num(0.0)]);
        let mut store = init_prefill_finalize_store();

        let state = init_prefill_finalize_projection(&mut store, final_position, &source, 1)
            .expect("init finalize projection");
        let encoded = serde_json::to_string(&state).expect("serialize state");

        assert!(encoded.contains("output_builder_ref"));
        assert!(!encoded.contains("logit_bits"));
    }

    #[test]
    fn raster_finalize_matches_native_deterministic_for_tied_projection() {
        let (_path, model) = tied_model().expect("tied fixture should build");
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("tied", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(0.5), Act::from_num(-0.25)]]);

        let raster = run(&[9], final_hidden_states.clone(), vec![], &source, 1)
            .expect("raster finalize should run");
        let native = crate::prefill_finalize::run(
            &[9],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .expect("native finalize should run");

        assert_eq!(
            raster.transformer_state.prefill_logits.logits,
            native.transformer_state.prefill_logits.logits
        );
        assert_eq!(raster.transformer_decode_state.position, 1);
        assert_eq!(raster.transformer_decode_state.token_count, 1);
    }

    #[test]
    fn raster_finalize_matches_native_deterministic_with_softcap() {
        let model = untied_model(true);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("softcap", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let raster = run(&[1], final_hidden_states.clone(), vec![], &source, 2)
            .expect("raster finalize should run");
        let native = crate::prefill_finalize::run(
            &[1],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .expect("native finalize should run");

        assert_eq!(
            raster.transformer_state.prefill_logits.logits,
            native.transformer_state.prefill_logits.logits
        );
    }

    #[test]
    fn raster_finalize_rejects_f32_only_final_hidden_states() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            ActivationSequence::from_values(vec![vec![1.0, 0.0]], "f32-only".to_string());

        let error = run(&[1], final_hidden_states, vec![], &source, 1)
            .expect_err("f32-only input should fail");

        assert!(error
            .to_string()
            .contains("canonical final hidden activations"));
    }

    #[test]
    fn raster_finalize_rejects_zero_projection_rows_per_tile() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);

        let error = run(&[1], final_hidden_states, vec![], &source, 0)
            .expect_err("zero rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn raster_finalize_uses_internal_det_row_not_public_f32_view() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let mut final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        final_hidden_states.activations = vec![vec![0.0, 1.0]];

        let raster = run(&[1], final_hidden_states.clone(), vec![], &source, 1)
            .expect("raster finalize should run");
        let native = crate::prefill_finalize::run(
            &[1],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .expect("native finalize should run");

        assert_eq!(
            raster.transformer_state.prefill_logits.logits,
            native.transformer_state.prefill_logits.logits
        );
    }

    #[test]
    fn raster_finalize_preserves_layer_caches_in_decode_state() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let layer_cache = LayerKvCache::from_det_heads(
            vec![VecDeque::from(vec![vec![Act::from_num(0.25)]])],
            vec![VecDeque::from(vec![vec![Act::from_num(-0.25)]])],
        );

        let raster = run(
            &[1, 2, 3],
            final_hidden_states,
            vec![layer_cache.clone()],
            &source,
            2,
        )
        .expect("raster finalize should run");

        assert_eq!(raster.transformer_decode_state.position, 3);
        assert_eq!(raster.transformer_decode_state.token_count, 3);
        assert_eq!(
            raster.transformer_decode_state.layer_caches,
            vec![layer_cache]
        );
    }

    fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
        let internal = InternalActivationSequence::from_det_values(rows.clone());
        let mut sequence = ActivationSequence::from_internal(
            internal,
            crate::shared::transformer_kernels::build_activation_commitment(
                &rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .copied()
                            .map(crate::shared::det_num::act_to_f32)
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>(),
            ),
        );
        sequence.det_activations_sha256 =
            Some(crate::shared::transformer_kernels::build_det_activation_commitment(&rows));
        sequence
    }

    fn untied_model(with_softcap: bool) -> Gemma4TransformerModel {
        let mut model = base_model(
            Gemma4LogitsProjection::UntiedLmHead {
                weight: matrix_f32(2, 2),
                det_weight: Some(Arc::new(DetNumMatrix {
                    rows: 2,
                    cols: 2,
                    values: vec![
                        Wgt::from_num(1.0).to_bits(),
                        Wgt::from_num(0.0).to_bits(),
                        Wgt::from_num(0.0).to_bits(),
                        Wgt::from_num(1.0).to_bits(),
                    ],
                })),
            },
            None,
        );
        if with_softcap {
            model.final_logit_softcapping = Some(0.5);
            model.final_logit_softcapping_det = Some(Act::from_num(0.5));
        }
        model
    }

    fn tied_model() -> Result<(PathBuf, Gemma4TransformerModel)> {
        let (path, source) = write_det_matrix(vec![
            vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
            vec![Wgt::from_num(0.0), Wgt::from_num(1.0)],
        ])?;
        let embedding_source = GemmaEmbeddingTensorSource::Deterministic {
            source,
            scale: 1.0,
            det_cache: Arc::new(Mutex::new(None)),
        };
        Ok((
            path,
            base_model(
                Gemma4LogitsProjection::TiedEmbedding(matrix_f32(2, 2)),
                Some(embedding_source),
            ),
        ))
    }

    fn base_model(
        logits_projection: Gemma4LogitsProjection,
        embedding_source: Option<GemmaEmbeddingTensorSource>,
    ) -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::DetNumWgt,
            embedding_table: None,
            embedding_source,
            layers: vec![],
            ple_global: None,
            final_norm_weight: vec![1.0, 1.0],
            final_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(1.0)]),
            logits_projection,
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 0.001,
            rms_norm_eps_det: Some(Acc::from_num(0.001)),
        }
    }

    fn matrix_f32(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }

    fn write_det_matrix(rows: Vec<Vec<Wgt>>) -> Result<(PathBuf, DetNumTensorSliceSource)> {
        let path = std::env::temp_dir().join(format!(
            "raster-prefill-finalize-tiles-{}-{}.detwgt",
            std::process::id(),
            crate::trace::sha256_hex(&format!("{:?}", rows))
        ));
        let mut bytes = Vec::new();
        for row in &rows {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        std::fs::write(&path, bytes)
            .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
        let source = det_source(&path, rows.len(), rows[0].len(), 0);
        Ok((path, source))
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
}
