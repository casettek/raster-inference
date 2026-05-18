use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::shared::artifact_io::ArtifactIo;
use crate::shared::input::InferenceExecutionMode;
use crate::shared::output::DecodeState;
use crate::shared::raster_artifact_store::{
    activation_row_leaf, read_token_id_from_ref_roots, token_id_leaf,
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::raster_row_store::{
    activation_sequence_ref_from_artifact, RasterActivationSequenceRef, RasterTensorId,
};
use crate::shared::raster_transformer_kernels::RasterActivationRow;

pub mod raster_tiles;
pub mod tiles;

pub fn run(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
    execution_mode: InferenceExecutionMode,
) -> Result<Option<u32>> {
    if tiles::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens).is_some()
    {
        return Ok(None);
    }

    let logits = decode_state.clone_internal_logits();
    let next_token = tiles::select_next_token_internal(&logits, execution_mode)?;
    decode_state.full_token_ids = tiles::append_token(&decode_state.full_token_ids, next_token);
    decode_state.generated_token_ids =
        tiles::append_token(&decode_state.generated_token_ids, next_token);
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &decode_select_checkpoint_state(decode_state, next_token, max_new_tokens)?,
    );
    Ok(Some(next_token))
}

pub fn run_raster(decode_state: &mut DecodeState, max_new_tokens: usize) -> Result<Option<u32>> {
    Ok(run_raster_refs(decode_state, max_new_tokens)?.map(|output| output.next_token))
}

pub fn run_raster_refs(
    decode_state: &mut DecodeState,
    max_new_tokens: usize,
) -> Result<Option<raster_tiles::RasterDecodeSelectOutputRefs>> {
    if raster_tiles::check_stop_condition(decode_state.generated_token_ids.len(), max_new_tokens)
        .is_some()
    {
        return Ok(None);
    }

    let input_roots = prepare_raster_decode_select_input_roots(decode_state, max_new_tokens)?;
    let output =
        raster_tiles::main(input_roots)?.expect("stop condition should have returned earlier");

    decode_state.full_token_ids =
        materialize_token_ids(&output.artifact_store_roots, &output.full_token_ids_ref)?;
    decode_state.generated_token_ids = materialize_token_ids(
        &output.artifact_store_roots,
        &output.generated_token_ids_ref,
    )?;
    crate::trace::trace_checkpoint(
        "decode.select_token",
        &decode_select_checkpoint_state(decode_state, output.next_token, max_new_tokens)?,
    );
    Ok(Some(output))
}

pub fn run_raster_with_roots(
    input_roots: raster_tiles::RasterDecodeSelectInputRoots,
) -> Result<Option<raster_tiles::RasterDecodeSelectOutputRefs>> {
    raster_tiles::main(input_roots)
}

fn prepare_raster_decode_select_input_roots(
    decode_state: &DecodeState,
    max_new_tokens: usize,
) -> Result<raster_tiles::RasterDecodeSelectInputRoots> {
    ArtifactIo::reset_store();
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let position = decode_state.transformer_decode_state.position;
    let generated_count = decode_state.generated_token_ids.len();
    let source_prefix = format!("decode.select_token.position_{position}.step_{generated_count}");
    let (artifact_store_roots, full_token_ids_root) = insert_token_ids_artifact_with_roots(
        artifact_store_roots,
        format!("{source_prefix}.input.full_token_ids"),
        &decode_state.full_token_ids,
    )?;
    let (artifact_store_roots, generated_token_ids_root) = insert_token_ids_artifact_with_roots(
        artifact_store_roots,
        format!("{source_prefix}.input.generated_token_ids"),
        &decode_state.generated_token_ids,
    )?;
    let (artifact_store_roots, logits_ref) = insert_logits_artifact_with_roots(
        artifact_store_roots,
        format!("{source_prefix}.input.logits"),
        decode_state,
    )?;

    Ok(raster_tiles::RasterDecodeSelectInputRoots {
        artifact_store_roots,
        logits_ref,
        full_token_ids_root,
        full_token_count: decode_state.full_token_ids.len(),
        generated_token_ids_root,
        generated_token_count: decode_state.generated_token_ids.len(),
        max_new_tokens,
        logits_per_tile: raster_tiles::DEFAULT_DECODE_SELECT_LOGITS_PER_TILE,
        token_ids_per_tile: raster_tiles::DEFAULT_DECODE_SELECT_TOKEN_IDS_PER_TILE,
        output_full_token_ids_source_name: format!("{source_prefix}.output.full_token_ids"),
        output_generated_token_ids_source_name: format!(
            "{source_prefix}.output.generated_token_ids"
        ),
        output_selected_token_source_name: format!("{source_prefix}.output.selected_token"),
    })
}

fn insert_token_ids_artifact_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    source_name: String,
    token_ids: &[u32],
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    if token_ids.is_empty() {
        return Ok((artifact_store_roots, None));
    }
    let leaves = token_ids.iter().copied().map(token_id_leaf).collect();
    let (artifact_store_roots, token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(source_name)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?;
    Ok((artifact_store_roots, Some(token_ids_ref.root().to_string())))
}

fn insert_logits_artifact_with_roots(
    artifact_store_roots: RasterArtifactStoreRoots,
    source_name: String,
    decode_state: &DecodeState,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let logits = decode_state.clone_internal_logits();
    let det_logits = logits.det_values().ok_or_else(|| {
        anyhow!("raster decode select token requires canonical deterministic logits")
    })?;
    if det_logits.is_empty() {
        anyhow::bail!("raster decode select token requires at least one canonical logit");
    }

    let leaves = det_logits
        .iter()
        .map(|logit| activation_row_leaf(&RasterActivationRow::from_acts(vec![*logit])))
        .collect::<Vec<_>>();
    let (artifact_store_roots, logits_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new(source_name.clone())?,
        RasterArtifactMetadata::activation_rows(det_logits.len(), 1)?,
        leaves,
    )?;
    let logits_ref = activation_sequence_ref_from_artifact(
        RasterTensorId::new(source_name)?,
        RasterActivationSequenceArtifactRef::new(logits_artifact_ref)?,
    )?;
    Ok((artifact_store_roots, logits_ref))
}

fn materialize_token_ids(
    artifact_store_roots: &RasterArtifactStoreRoots,
    token_ids_ref: &RasterTokenIdSequenceRef,
) -> Result<Vec<u32>> {
    (0..token_ids_ref.token_count())
        .map(|token_idx| {
            read_token_id_from_ref_roots(artifact_store_roots, token_ids_ref, token_idx)
        })
        .collect()
}

fn decode_select_checkpoint_state(
    decode_state: &DecodeState,
    selected_next_token: u32,
    max_new_tokens: usize,
) -> Result<Value> {
    Ok(json!({
        "full_token_ids": decode_state.full_token_ids.clone(),
        "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
        "generated_token_ids": decode_state.generated_token_ids.clone(),
        "generated_token_ids_sha256": crate::output_finalize::tiles::build_output_decode_commitment(&decode_state.generated_token_ids)?,
        "current_logits": decode_state.current_logits.clone(),
        "current_logits_sha256": crate::trace::sha256_hex(&decode_state.current_logits),
        "det_current_logits_sha256": current_det_logits_commitment(decode_state),
        "selected_next_token": selected_next_token,
        "decode_position": decode_state.transformer_decode_state.position,
        "decode_token_count": decode_state.transformer_decode_state.token_count,
        "layer_caches": crate::trace::serialize_layer_caches(&decode_state.transformer_decode_state.layer_caches),
        "max_new_tokens": max_new_tokens,
    }))
}

fn current_det_logits_commitment(decode_state: &DecodeState) -> Option<String> {
    let logits = decode_state.clone_internal_logits();
    logits
        .det_values()
        .map(crate::shared::transformer_kernels::build_det_vector_commitment)
}

#[cfg(test)]
mod tests {
    use super::{decode_select_checkpoint_state, run_raster};
    use crate::shared::{
        det_num::Act,
        output::DecodeState,
        transformer::{InternalLogits, TransformerDecodeState},
    };

    #[test]
    fn run_raster_appends_selected_token_to_decode_state() {
        let mut decode_state = decode_state_with_logits(vec![
            Act::from_bits(1),
            Act::from_bits(9),
            Act::from_bits(3),
        ]);

        let selected = run_raster(&mut decode_state, 1)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(selected, 1);
        assert_eq!(decode_state.full_token_ids, vec![7, 1]);
        assert_eq!(decode_state.generated_token_ids, vec![1]);
    }

    #[test]
    fn run_raster_uses_canonical_logits_over_public_f32_view() {
        let mut decode_state =
            decode_state_with_logits(vec![Act::from_bits(100), Act::from_bits(1)]);
        decode_state.current_logits = vec![0.0, 1000.0];

        let selected = run_raster(&mut decode_state, 1)
            .expect("raster select should run")
            .expect("should select token");

        assert_eq!(selected, 0);
        assert_eq!(decode_state.generated_token_ids, vec![0]);
    }

    #[test]
    fn run_raster_rejects_f32_only_logits_without_mutating_decode_state() {
        let mut decode_state =
            DecodeState::new(vec![7], vec![0.0, 1.0], TransformerDecodeState::default());

        let error =
            run_raster(&mut decode_state, 1).expect_err("f32-only logits should fail in raster");

        assert!(error.to_string().contains("canonical deterministic logits"));
        assert_eq!(decode_state.full_token_ids, vec![7]);
        assert!(decode_state.generated_token_ids.is_empty());
    }

    #[test]
    fn checkpoint_uses_canonical_deterministic_logits_commitment() {
        let internal =
            InternalLogits::from_det_values(vec![Act::from_bits(100), Act::from_bits(1)]);
        let mut decode_state = DecodeState::new(
            vec![7],
            internal.clone_f32(),
            TransformerDecodeState::default(),
        );
        decode_state.set_internal_logits(internal.clone());
        decode_state.current_logits = vec![0.0, 1000.0];

        let payload = decode_select_checkpoint_state(&decode_state, 0, 1).expect("checkpoint");
        let expected_det_commitment =
            crate::shared::transformer_kernels::build_det_vector_commitment(
                internal.det_values().expect("canonical logits"),
            );
        let expected_public_commitment = crate::trace::sha256_hex(&decode_state.current_logits);

        assert_eq!(
            payload
                .get("det_current_logits_sha256")
                .and_then(|value| value.as_str()),
            Some(expected_det_commitment.as_str())
        );
        assert_eq!(
            payload
                .get("current_logits_sha256")
                .and_then(|value| value.as_str()),
            Some(expected_public_commitment.as_str())
        );
    }

    #[test]
    fn run_raster_stops_without_requiring_canonical_logits() {
        let mut decode_state =
            DecodeState::new(vec![7], vec![0.0, 1.0], TransformerDecodeState::default());

        let selected = run_raster(&mut decode_state, 0).expect("stop should not inspect logits");

        assert_eq!(selected, None);
        assert_eq!(decode_state.full_token_ids, vec![7]);
        assert!(decode_state.generated_token_ids.is_empty());
    }

    fn decode_state_with_logits(det_logits: Vec<Act>) -> DecodeState {
        let internal = InternalLogits::from_det_values(det_logits);
        let mut decode_state = DecodeState::new(
            vec![7],
            internal.clone_f32(),
            TransformerDecodeState::default(),
        );
        decode_state.set_internal_logits(internal);
        decode_state
    }
}
