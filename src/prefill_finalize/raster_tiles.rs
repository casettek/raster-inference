use anyhow::{anyhow, bail, Result};

use super::raster_utils::build_prefill_result_from_root_refs;
use crate::prefill_finalize::authenticated_source::{
    GemmaPrefillFinalizeMetadataRequest, GemmaPrefillFinalizeNormWeightsRequest,
    GemmaPrefillFinalizeProjectionRowRequest, GemmaPrefillFinalizeScalarsRequest,
};
use crate::raster_authoring::prelude::{
    auth_read, call_recur_tile, call_seq, call_tile, sequence, tile,
};
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactId, RasterArtifactStoreRoots, RasterRoutineOutput,
};
use crate::shared::model::transformer::TransformerPrefillResult;
use crate::shared::numerics::det_num::{softcap_act, Act};
use crate::shared::raster_kernels::transformer::{
    project_row_with_weights, rms_norm_sequence, validate_projection_rows_per_tile,
    RasterActivationRow, RasterActivationSequence,
};
use crate::shared::tensors::raster_row_store::{
    append_sequence_row_by_source_name_with_roots,
    finalize_sequence_builder_by_source_name_with_roots, read_sequence_row_from_roots,
    start_sequence_builder_with_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
    RasterTensorId,
};

pub const NORMALIZED_FINAL_POSITION_ARTIFACT_NAME: &str =
    "prefill.finalize.normalized_final_position";
pub const PREFILL_LOGITS_ARTIFACT_NAME: &str = "prefill.finalize.logits";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillFinalizeInputRoots {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub prompt_token_count: usize,
    pub finalize_source_root: String,
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot>,
    pub projection_rows_per_tile: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillFinalizeRefs {
    pub source_id: String,
    pub finalize_source_root: String,
    pub prompt_token_count: usize,
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot>,
    pub normalized_final_position_ref: RasterActivationSequenceRef,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
}

pub type RasterPrefillFinalizeOutput = RasterRoutineOutput<RasterPrefillFinalizeRefs>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PrefillFinalizeRasterState {
    artifact_store_roots: RasterArtifactStoreRoots,
    source_id: String,
    finalize_source_root: String,
    prompt_token_count: usize,
    final_hidden_states_ref: RasterActivationSequenceRef,
    normalized_final_position_ref: Option<RasterActivationSequenceRef>,
    next_logit_idx: usize,
    logit_count: usize,
    hidden_width: usize,
    softcap_bits: Option<i32>,
    projection_rows_per_tile: usize,
}

impl PrefillFinalizeRasterState {
    fn is_complete(&self) -> bool {
        self.next_logit_idx >= self.logit_count
    }
}

#[tile]
pub fn init_prefill_finalize_state(
    mut artifact_store_roots: RasterArtifactStoreRoots,
    prompt_token_count: usize,
    finalize_source_root: String,
    final_hidden_states_ref: RasterActivationSequenceRef,
    layer_caches: &[crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot],
    projection_rows_per_tile: usize,
) -> Result<PrefillFinalizeRasterState> {
    validate_projection_rows_per_tile(projection_rows_per_tile)?;
    if prompt_token_count == 0 {
        bail!("raster prefill finalize requires at least one prompt token");
    }
    let (row_count, hidden_width) = final_hidden_states_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    if row_count == 0 {
        bail!("transformer final-position selection requires at least one activation row");
    }
    ensure_artifact_root_present(
        &artifact_store_roots,
        final_hidden_states_ref.tensor_ref().det_commitment(),
    )?;
    validate_layer_cache_roots(&artifact_store_roots, layer_caches)?;

    let metadata = auth_read!(
        finalize_source_root.as_str(),
        GemmaPrefillFinalizeMetadataRequest
    )?;
    if metadata.projection_rows == 0 {
        bail!("deterministic logits projection requires at least one projection row");
    }
    if hidden_width != metadata.hidden_width {
        bail!(
            "deterministic final logits projection input has width {}, expected {}",
            hidden_width,
            metadata.hidden_width,
        );
    }
    if metadata.projection_cols != metadata.hidden_width {
        bail!(
            "deterministic final logits projection metadata width mismatch: {} vs {}",
            metadata.projection_cols,
            metadata.hidden_width
        );
    }
    let scalars = auth_read!(
        finalize_source_root.as_str(),
        GemmaPrefillFinalizeScalarsRequest
    )?;
    artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(NORMALIZED_FINAL_POSITION_ARTIFACT_NAME)?,
        1,
        hidden_width,
    )?;
    artifact_store_roots = start_sequence_builder_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(PREFILL_LOGITS_ARTIFACT_NAME)?,
        metadata.projection_rows,
        1,
    )?;

    Ok(PrefillFinalizeRasterState {
        artifact_store_roots,
        source_id: metadata.source_id,
        finalize_source_root,
        prompt_token_count,
        final_hidden_states_ref,
        normalized_final_position_ref: None,
        next_logit_idx: 0,
        logit_count: metadata.projection_rows,
        hidden_width,
        softcap_bits: scalars.final_logit_softcapping.map(Act::to_bits),
        projection_rows_per_tile,
    })
}

#[tile]
pub fn normalize_final_position_to_artifact(
    mut state: PrefillFinalizeRasterState,
) -> Result<PrefillFinalizeRasterState> {
    let (row_count, _) = state
        .final_hidden_states_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    let final_position = read_sequence_row_from_roots(
        &state.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: state.final_hidden_states_ref.clone(),
            row_idx: row_count - 1,
        },
    )?;
    let norm_weights = auth_read!(
        state.finalize_source_root.as_str(),
        GemmaPrefillFinalizeNormWeightsRequest
    )?;
    let scalars = auth_read!(
        state.finalize_source_root.as_str(),
        GemmaPrefillFinalizeScalarsRequest
    )?;
    let normalized = rms_norm_sequence(
        &RasterActivationSequence::from_rows(vec![final_position]),
        Some(&norm_weights),
        Some(scalars.rms_norm_eps),
    )?
    .into_rows()
    .into_iter()
    .next()
    .ok_or_else(|| anyhow!("deterministic final RMSNorm returned no rows"))?;
    if normalized.width() != state.hidden_width {
        bail!(
            "deterministic final RMSNorm produced width {}, expected {}",
            normalized.width(),
            state.hidden_width
        );
    }
    state.artifact_store_roots = append_sequence_row_by_source_name_with_roots(
        &state.artifact_store_roots,
        NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
        0,
        normalized,
    )?;
    let (artifact_store_roots, normalized_final_position_ref) =
        finalize_sequence_builder_by_source_name_with_roots(
            &state.artifact_store_roots,
            NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
            RasterTensorId::new(NORMALIZED_FINAL_POSITION_ARTIFACT_NAME)?,
        )?;
    state.artifact_store_roots = artifact_store_roots;
    state.normalized_final_position_ref = Some(normalized_final_position_ref);
    Ok(state)
}

#[tile(kind = recursive)]
pub fn project_next_prefill_logit_chunk(
    mut state: PrefillFinalizeRasterState,
) -> Result<(bool, PrefillFinalizeRasterState)> {
    if state.is_complete() {
        return Ok((true, state));
    }
    let normalized_final_position_ref = state
        .normalized_final_position_ref
        .clone()
        .ok_or_else(|| anyhow!("raster prefill finalize projection missing normalized row ref"))?;
    let normalized_final_position = read_sequence_row_from_roots(
        &state.artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: normalized_final_position_ref,
            row_idx: 0,
        },
    )?;
    if normalized_final_position.width() != state.hidden_width {
        bail!(
            "deterministic final logits projection input has width {}, expected {}",
            normalized_final_position.width(),
            state.hidden_width
        );
    }

    let end = state
        .next_logit_idx
        .saturating_add(state.projection_rows_per_tile)
        .min(state.logit_count);
    while state.next_logit_idx < end {
        let projection_row = auth_read!(
            state.finalize_source_root.as_str(),
            GemmaPrefillFinalizeProjectionRowRequest {
                row_idx: state.next_logit_idx,
            },
        )?;
        let mut logit = project_row_with_weights(&normalized_final_position, &projection_row)?;
        if let Some(softcap_bits) = state.softcap_bits {
            logit = softcap_act(logit, Act::from_bits(softcap_bits));
        }
        state.artifact_store_roots = append_sequence_row_by_source_name_with_roots(
            &state.artifact_store_roots,
            PREFILL_LOGITS_ARTIFACT_NAME,
            state.next_logit_idx,
            RasterActivationRow::from_acts(vec![logit]),
        )?;
        state.next_logit_idx += 1;
    }
    Ok((state.is_complete(), state))
}

#[tile]
pub fn finalize_prefill_finalize_refs(
    state: PrefillFinalizeRasterState,
    layer_caches: Vec<crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot>,
) -> Result<(RasterArtifactStoreRoots, RasterPrefillFinalizeRefs)> {
    let normalized_final_position_ref = state
        .normalized_final_position_ref
        .clone()
        .ok_or_else(|| anyhow!("raster prefill finalize missing normalized row ref"))?;
    if state.next_logit_idx != state.logit_count {
        bail!(
            "raster prefill finalize completed {} logits, expected {}",
            state.next_logit_idx,
            state.logit_count
        );
    }
    let (artifact_store_roots, logits_ref) = finalize_sequence_builder_by_source_name_with_roots(
        &state.artifact_store_roots,
        PREFILL_LOGITS_ARTIFACT_NAME,
        RasterTensorId::new(PREFILL_LOGITS_ARTIFACT_NAME)?,
    )?;
    Ok((
        artifact_store_roots,
        RasterPrefillFinalizeRefs {
            source_id: state.source_id,
            finalize_source_root: state.finalize_source_root,
            prompt_token_count: state.prompt_token_count,
            final_hidden_states_ref: state.final_hidden_states_ref,
            layer_caches,
            normalized_final_position_ref,
            logits_ref,
            logit_count: state.logit_count,
        },
    ))
}

#[sequence]
pub fn main_refs(
    input_roots: RasterPrefillFinalizeInputRoots,
) -> Result<RasterPrefillFinalizeOutput> {
    crate::trace::trace_event("prefill.select_final_position");
    let state = call_tile!(
        init_prefill_finalize_state,
        input_roots.artifact_store_roots,
        input_roots.prompt_token_count,
        input_roots.finalize_source_root,
        input_roots.final_hidden_states_ref,
        &input_roots.layer_caches,
        input_roots.projection_rows_per_tile
    )?;
    let state = call_tile!(normalize_final_position_to_artifact, state)?;
    crate::trace::trace_event("prefill.project_to_logits");
    let state = call_recur_tile!(project_next_prefill_logit_chunk, state)?;
    let (artifact_store_roots, refs) = call_tile!(
        finalize_prefill_finalize_refs,
        state,
        input_roots.layer_caches
    )?;
    Ok(RasterPrefillFinalizeOutput::new(artifact_store_roots, refs))
}

#[sequence]
pub fn main(input_roots: RasterPrefillFinalizeInputRoots) -> Result<TransformerPrefillResult> {
    let output = call_seq!(main_refs, input_roots)?;
    call_tile!(
        build_prefill_result_from_refs,
        output.artifact_store_roots,
        output.refs
    )
}

#[tile]
pub fn build_prefill_result_from_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    refs: RasterPrefillFinalizeRefs,
) -> Result<TransformerPrefillResult> {
    build_prefill_result_from_root_refs(&artifact_store_roots, &refs)
}

fn validate_layer_cache_roots(
    roots: &RasterArtifactStoreRoots,
    layer_caches: &[crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot],
) -> Result<()> {
    for cache in layer_caches {
        if let crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot::Ref(cache_ref) = cache {
            ensure_artifact_root_present(roots, cache_ref.keys().det_commitment())?;
            ensure_artifact_root_present(roots, cache_ref.values().det_commitment())?;
        }
    }
    Ok(())
}

fn ensure_artifact_root_present(roots: &RasterArtifactStoreRoots, root: &str) -> Result<()> {
    if roots.artifacts.iter().any(|entry| entry.root() == root) {
        return Ok(());
    }
    bail!("raster artifact root {root} is not present in the store roots snapshot")
}

#[cfg(test)]
mod tests {
    use super::{
        finalize_prefill_finalize_refs, init_prefill_finalize_state, main,
        normalize_final_position_to_artifact, project_next_prefill_logit_chunk,
        RasterPrefillFinalizeInputRoots, NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
    };
    use crate::prefill_finalize::authenticated_source::AuthenticatedGemmaPrefillFinalizeSource;
    use crate::prefill_layer::raster_tiles::PrefillLayerCacheSlot;
    use crate::shared::api::input::InferenceExecutionMode;
    use crate::shared::artifacts::artifact_io::ArtifactIo;
    use crate::shared::artifacts::raster_artifact_store::RasterArtifactStoreRoots;
    use crate::shared::model::transformer::{
        ActivationSequence, DetNumMatrix, DetNumTensorSliceSource, Gemma4LogitsProjection,
        Gemma4ModelProvenance, Gemma4TransformerModel, GemmaEmbeddingTensorSource,
        InternalActivationSequence, LayerKvCache, MatrixF32,
    };
    use crate::shared::numerics::det_num::{Acc, Act, Wgt};
    use crate::shared::raster_kernels::transformer::{
        RasterActivationRow, RasterActivationSequence, RasterKvCache,
    };
    use crate::shared::tensors::raster_row_store::{
        AuthenticatedRasterTensorStore, RasterTensorId,
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
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let raster = run_ref_backed_finalize(2, final_hidden_states_ref, vec![], &source, 1)
            .expect("root-backed raster finalize should run");
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
    fn ref_backed_finalize_matches_native_deterministic_for_untied_projection() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states = activation_sequence(vec![
            vec![Act::from_num(0.25), Act::from_num(0.5)],
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
        ]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let raster = run_ref_backed_finalize(2, final_hidden_states_ref, vec![], &source, 1)
            .expect("ref-backed raster finalize should run");
        let native = crate::prefill_finalize::run(
            &[3, 4],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .expect("native finalize should run");

        assert_eq!(
            raster.transformer_state.prefill_logits,
            native.transformer_state.prefill_logits
        );
        assert_eq!(raster.transformer_decode_state.position, 2);
        assert_eq!(raster.transformer_decode_state.token_count, 2);
    }

    #[test]
    fn finalize_projection_state_serializes_builder_not_logits() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let source_ref = source.committed_source_ref().expect("source should commit");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);
        let roots = ArtifactIo::export_store_roots();

        let state = init_prefill_finalize_state(
            roots,
            1,
            source_ref.root().to_string(),
            final_hidden_states_ref,
            &[],
            1,
        )
        .expect("init finalize projection");
        let encoded = serde_json::to_string(&state).expect("serialize state");

        assert!(encoded.contains("artifact_store_roots"));
        assert!(!encoded.contains("normalized_final_position\":"));
        assert!(!encoded.contains("logit_bits"));
        assert!(!encoded.contains("layer_caches"));
        assert!(!encoded.contains("final_hidden_states\":["));
        assert!(!encoded.contains("prefill_logits"));
        assert!(!encoded.contains("token_ids"));
    }

    #[test]
    fn in_progress_projection_state_serializes_refs_not_payloads() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("in-progress", &model)
            .expect("source should build");
        let source_ref = source.committed_source_ref().expect("source should commit");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);
        let state = init_prefill_finalize_state(
            ArtifactIo::export_store_roots(),
            1,
            source_ref.root().to_string(),
            final_hidden_states_ref,
            &[],
            1,
        )
        .expect("init finalize state");
        let state =
            normalize_final_position_to_artifact(state).expect("normalization should write a ref");
        let (done, state) =
            project_next_prefill_logit_chunk(state).expect("first projection chunk should run");
        assert!(!done);

        let encoded = serde_json::to_string(&state).expect("serialize state");
        assert!(encoded.contains("normalized_final_position_ref"));
        assert!(!encoded.contains("normalized_final_position\":"));
        assert!(!encoded.contains("logit_bits"));
        assert!(!encoded.contains("current_row_bits"));
        assert!(!encoded.contains("layer_caches"));
        assert!(!encoded.contains("final_hidden_states\":["));
        assert!(!encoded.contains("prefill_logits"));
        assert!(!encoded.contains("token_ids"));
    }

    #[test]
    fn raster_finalize_matches_native_deterministic_for_tied_projection() {
        let (_path, model) = tied_model().expect("tied fixture should build");
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("tied", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(0.5), Act::from_num(-0.25)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let raster = run_ref_backed_finalize(1, final_hidden_states_ref, vec![], &source, 1)
            .expect("root-backed raster finalize should run");
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
    fn ref_backed_finalize_matches_native_deterministic_for_tied_projection() {
        let (_path, model) = tied_model().expect("tied fixture should build");
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("tied", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(0.5), Act::from_num(-0.25)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let raster = run_ref_backed_finalize(1, final_hidden_states_ref, vec![], &source, 1)
            .expect("ref-backed raster finalize should run");
        let native = crate::prefill_finalize::run(
            &[9],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .expect("native finalize should run");

        assert_eq!(
            raster.transformer_state.prefill_logits,
            native.transformer_state.prefill_logits
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
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let raster = run_ref_backed_finalize(1, final_hidden_states_ref, vec![], &source, 2)
            .expect("root-backed raster finalize should run");
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
    fn raster_finalize_projection_chunk_sizes_do_not_change_result() {
        let model = untied_model(true);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("chunks", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);
        ArtifactIo::reset_store();
        let mut single_store = AuthenticatedRasterTensorStore::new();
        let single_ref =
            insert_activation_ref(&mut single_store, "finalize.hidden", &final_hidden_states);
        let single = run_ref_backed_finalize(1, single_ref, vec![], &source, 1)
            .expect("single-row chunks should run");

        ArtifactIo::reset_store();
        let mut multi_store = AuthenticatedRasterTensorStore::new();
        let multi_ref =
            insert_activation_ref(&mut multi_store, "finalize.hidden", &final_hidden_states);
        let multi = run_ref_backed_finalize(1, multi_ref, vec![], &source, 2)
            .expect("multi-row chunks should run");

        assert_eq!(
            single.transformer_state.prefill_logits,
            multi.transformer_state.prefill_logits
        );
    }

    #[test]
    fn ref_backed_finalize_matches_native_deterministic_with_softcap() {
        let model = untied_model(true);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("softcap", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let raster = run_ref_backed_finalize(1, final_hidden_states_ref, vec![], &source, 2)
            .expect("ref-backed raster finalize should run");
        let native = crate::prefill_finalize::run(
            &[1],
            &model,
            final_hidden_states,
            vec![],
            InferenceExecutionMode::Deterministic,
        )
        .expect("native finalize should run");

        assert_eq!(
            raster.transformer_state.prefill_logits,
            native.transformer_state.prefill_logits
        );
    }

    #[test]
    fn raster_finalize_rejects_zero_projection_rows_per_tile() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let error = run_ref_backed_finalize(1, final_hidden_states_ref, vec![], &source, 0)
            .expect_err("zero rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn ref_backed_finalize_fails_closed_for_missing_sequence_ref() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut source_store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut source_store, "finalize.hidden", &final_hidden_states);

        let error = run_ref_backed_finalize_with_roots(
            RasterArtifactStoreRoots::default(),
            1,
            final_hidden_states_ref,
            vec![],
            &source,
            1,
        )
        .expect_err("missing sequence ref should fail");

        assert!(error.to_string().contains("not present"));
    }

    #[test]
    fn ref_backed_finalize_reports_width_mismatch() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states = activation_sequence(vec![vec![
            Act::from_num(1.0),
            Act::from_num(0.0),
            Act::from_num(0.5),
        ]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let error = run_ref_backed_finalize(1, final_hidden_states_ref, vec![], &source, 1)
            .expect_err("width mismatch should fail");

        assert!(error.to_string().contains("width"));
    }

    #[test]
    fn ref_backed_finalize_fails_closed_for_bad_source_root() {
        let model = untied_model(false);
        let _source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let error = main(RasterPrefillFinalizeInputRoots {
            artifact_store_roots: ArtifactIo::export_store_roots(),
            prompt_token_count: 1,
            finalize_source_root: "missing-finalize-source-root".to_string(),
            final_hidden_states_ref,
            layer_caches: vec![],
            projection_rows_per_tile: 1,
        })
        .expect_err("bad source root should fail");

        assert!(error.to_string().contains("not registered"));
    }

    #[test]
    fn ref_backed_finalize_fails_closed_for_missing_cache_roots() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);
        let stale_roots = ArtifactIo::export_store_roots();
        let cache = RasterKvCache::from_heads(
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                0.25,
            )])]],
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                -0.25,
            )])]],
        )
        .expect("cache");
        let cache_ref = store
            .insert_kv_cache(
                RasterTensorId::new("finalize.cache.keys").expect("keys id"),
                RasterTensorId::new("finalize.cache.values").expect("values id"),
                cache,
            )
            .expect("cache ref");

        let error = run_ref_backed_finalize_with_roots(
            stale_roots,
            1,
            final_hidden_states_ref,
            vec![PrefillLayerCacheSlot::Ref(cache_ref)],
            &source,
            1,
        )
        .expect_err("missing cache roots should fail");

        assert!(error.to_string().contains("not present"));
    }

    #[test]
    fn prefill_finalize_roots_aware_mutation_rejects_stale_builder_root() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("stale", &model)
            .expect("source should build");
        let source_ref = source.committed_source_ref().expect("source should commit");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);
        let state = init_prefill_finalize_state(
            ArtifactIo::export_store_roots(),
            1,
            source_ref.root().to_string(),
            final_hidden_states_ref,
            &[],
            1,
        )
        .expect("init should build roots-backed state");
        ArtifactIo::append_leaf_by_builder_source_name_with_roots(
            &state.artifact_store_roots,
            NORMALIZED_FINAL_POSITION_ARTIFACT_NAME,
            0,
            crate::shared::artifacts::raster_artifact_store::activation_row_leaf(
                &RasterActivationRow::from_acts(vec![Act::from_num(0.0), Act::from_num(0.0)]),
            ),
        )
        .expect("external append should stale the state roots");

        let error = normalize_final_position_to_artifact(state)
            .expect_err("stale builder root should fail");

        assert!(error.to_string().contains("snapshot"));
    }

    #[test]
    fn finalize_refs_rejects_overadvanced_projection_cursor() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("overadvanced", &model)
            .expect("source should build");
        let source_ref = source.committed_source_ref().expect("source should commit");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);
        let state = init_prefill_finalize_state(
            ArtifactIo::export_store_roots(),
            1,
            source_ref.root().to_string(),
            final_hidden_states_ref,
            &[],
            1,
        )
        .expect("init should build state");
        let mut state =
            normalize_final_position_to_artifact(state).expect("normalization should finish");
        state.next_logit_idx = state.logit_count + 1;

        let error = finalize_prefill_finalize_refs(state, vec![])
            .expect_err("overadvanced cursor should fail closed");

        assert!(error.to_string().contains("completed"));
    }

    #[test]
    fn raster_finalize_uses_internal_det_row_not_public_f32_view() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let mut final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        final_hidden_states.activations = vec![vec![0.0, 1.0]];
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);

        let raster = run_ref_backed_finalize(1, final_hidden_states_ref, vec![], &source, 1)
            .expect("root-backed raster finalize should run");
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
        let raster_cache = RasterKvCache::from_heads(
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                0.25,
            )])]],
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                -0.25,
            )])]],
        )
        .expect("cache");
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);
        let cache_ref = store
            .insert_kv_cache(
                RasterTensorId::new("finalize.cache.keys").expect("keys id"),
                RasterTensorId::new("finalize.cache.values").expect("values id"),
                raster_cache,
            )
            .expect("cache ref");

        let raster = run_ref_backed_finalize(
            3,
            final_hidden_states_ref,
            vec![PrefillLayerCacheSlot::Ref(cache_ref)],
            &source,
            2,
        )
        .expect("root-backed raster finalize should run");

        assert_eq!(raster.transformer_decode_state.position, 3);
        assert_eq!(raster.transformer_decode_state.token_count, 3);
        assert_eq!(
            raster.transformer_decode_state.layer_caches,
            vec![layer_cache]
        );
    }

    #[test]
    fn ref_backed_finalize_materializes_public_activation_and_layer_cache() {
        let model = untied_model(false);
        let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
            .expect("source should build");
        let final_hidden_states =
            activation_sequence(vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]);
        let layer_cache = RasterKvCache::from_heads(
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                0.25,
            )])]],
            vec![vec![RasterActivationRow::from_acts(vec![Act::from_num(
                -0.25,
            )])]],
        )
        .expect("cache");
        let expected_layer_cache = LayerKvCache::from_det_heads(
            vec![VecDeque::from(vec![vec![Act::from_num(0.25)]])],
            vec![VecDeque::from(vec![vec![Act::from_num(-0.25)]])],
        );
        ArtifactIo::reset_store();
        let mut store = AuthenticatedRasterTensorStore::new();
        let final_hidden_states_ref =
            insert_activation_ref(&mut store, "finalize.hidden", &final_hidden_states);
        let cache_ref = store
            .insert_kv_cache(
                RasterTensorId::new("finalize.cache.keys").expect("keys id"),
                RasterTensorId::new("finalize.cache.values").expect("values id"),
                layer_cache,
            )
            .expect("cache ref");

        let raster = run_ref_backed_finalize(
            1,
            final_hidden_states_ref,
            vec![PrefillLayerCacheSlot::Ref(cache_ref)],
            &source,
            1,
        )
        .expect("ref-backed raster finalize should run");

        assert_eq!(
            raster.transformer_state.activation_states,
            vec![final_hidden_states]
        );
        assert_eq!(
            raster.transformer_decode_state.layer_caches,
            vec![expected_layer_cache]
        );
    }

    #[test]
    fn proof_shaped_finalize_signatures_stay_ref_backed() {
        let source = include_str!("raster_tiles.rs");

        assert_signature_omits(
            source,
            "pub fn main",
            &[
                ": &ActivationSequence",
                ": ActivationSequence",
                "Vec<LayerKvCache",
                ": LayerKvCache",
            ],
        );
        assert_signature_omits(
            source,
            "pub fn init_prefill_finalize_state",
            &[
                ": &ActivationSequence",
                ": ActivationSequence",
                "Vec<LayerKvCache",
                ": LayerKvCache",
            ],
        );
        assert_signature_omits(
            source,
            "pub fn finalize_prefill_finalize_refs",
            &[
                ": &ActivationSequence",
                ": ActivationSequence",
                "Vec<LayerKvCache",
                ": LayerKvCache",
            ],
        );
    }

    fn run_ref_backed_finalize(
        prompt_token_count: usize,
        final_hidden_states_ref: crate::shared::tensors::raster_row_store::RasterActivationSequenceRef,
        layer_caches: Vec<PrefillLayerCacheSlot>,
        source: &AuthenticatedGemmaPrefillFinalizeSource,
        projection_rows_per_tile: usize,
    ) -> Result<crate::shared::model::transformer::TransformerPrefillResult> {
        run_ref_backed_finalize_with_roots(
            ArtifactIo::export_store_roots(),
            prompt_token_count,
            final_hidden_states_ref,
            layer_caches,
            source,
            projection_rows_per_tile,
        )
    }

    fn run_ref_backed_finalize_with_roots(
        artifact_store_roots: RasterArtifactStoreRoots,
        prompt_token_count: usize,
        final_hidden_states_ref: crate::shared::tensors::raster_row_store::RasterActivationSequenceRef,
        layer_caches: Vec<PrefillLayerCacheSlot>,
        source: &AuthenticatedGemmaPrefillFinalizeSource,
        projection_rows_per_tile: usize,
    ) -> Result<crate::shared::model::transformer::TransformerPrefillResult> {
        let source_ref = source.committed_source_ref()?;
        main(RasterPrefillFinalizeInputRoots {
            artifact_store_roots,
            prompt_token_count,
            finalize_source_root: source_ref.root().to_string(),
            final_hidden_states_ref,
            layer_caches,
            projection_rows_per_tile,
        })
    }

    fn activation_sequence(rows: Vec<Vec<Act>>) -> ActivationSequence {
        let internal = InternalActivationSequence::from_det_values(rows.clone());
        let mut sequence = ActivationSequence::from_internal(
            internal,
            crate::shared::numerics::transformer_kernels::build_activation_commitment(
                &rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .copied()
                            .map(crate::shared::numerics::det_num::act_to_f32)
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>(),
            ),
        );
        sequence.det_activations_sha256 = Some(
            crate::shared::numerics::transformer_kernels::build_det_activation_commitment(&rows),
        );
        sequence
    }

    fn insert_activation_ref(
        store: &mut AuthenticatedRasterTensorStore,
        id: &str,
        sequence: &ActivationSequence,
    ) -> crate::shared::tensors::raster_row_store::RasterActivationSequenceRef {
        let det_rows = sequence
            .clone_internal()
            .det_values()
            .expect("det activation sequence")
            .to_vec();
        store
            .insert_activation_sequence(
                RasterTensorId::new(id).expect("tensor id"),
                RasterActivationSequence::from_acts(det_rows),
            )
            .expect("activation ref")
    }

    fn assert_signature_omits(source: &str, needle: &str, forbidden: &[&str]) {
        let start = source.find(needle).expect("signature start");
        let rest = &source[start..];
        let end = rest.find(") ->").expect("signature end");
        let signature = &rest[..end];

        for forbidden_type in forbidden {
            assert!(
                !signature.contains(forbidden_type),
                "{needle} signature should not contain {forbidden_type}: {signature}"
            );
        }
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
