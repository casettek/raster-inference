use anyhow::{anyhow, bail, Result};

use crate::shared::artifacts::raster_artifact_store::{
    read_token_id_from_ref_roots, RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::numerics::det_num::{argmax_first, Act};
use crate::shared::tensors::raster_tensor_artifacts::{
    read_sequence_row_from_roots, RasterActivationSequenceRef, RasterSequenceRowRequest,
};

pub(in super::super) fn read_logit_bits(
    artifact_store_roots: &RasterArtifactStoreRoots,
    logits_ref: &RasterActivationSequenceRef,
    token_idx: usize,
) -> Result<i32> {
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    let (row_idx, col_idx) = match (row_count, width) {
        (_, 1) => (token_idx, 0),
        (1, _) => (0, token_idx),
        _ => bail!("raster decode select logits shape {row_count}x{width} must be Nx1 or 1xN"),
    };
    let row = read_sequence_row_from_roots(
        artifact_store_roots,
        RasterSequenceRowRequest {
            tensor_ref: logits_ref.clone(),
            row_idx,
        },
    )?;
    row.act_bits()
        .get(col_idx)
        .copied()
        .ok_or_else(|| anyhow!("raster decode select logit {token_idx} is missing"))
}

pub(in super::super) fn decode_select_logit_count(
    logits_ref: &RasterActivationSequenceRef,
) -> Result<usize> {
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    match (row_count, width) {
        (0, _) | (_, 0) => {
            bail!("raster decode select token requires at least one canonical logit")
        }
        (rows, 1) => Ok(rows),
        (1, cols) => Ok(cols),
        _ => bail!("raster decode select logits shape {row_count}x{width} must be Nx1 or 1xN"),
    }
}

pub(in super::super) fn candidate_wins(best_logit_bits: i32, candidate_bits: i32) -> bool {
    let candidates = [
        Act::from_bits(best_logit_bits),
        Act::from_bits(candidate_bits),
    ];
    argmax_first(&candidates) == 1
}

pub(in super::super) fn validate_token_input(
    artifact_store_roots: &RasterArtifactStoreRoots,
    token_ids_ref: Option<&RasterTokenIdSequenceRef>,
    token_count: usize,
    label: &str,
) -> Result<()> {
    match (token_ids_ref, token_count) {
        (Some(token_ids_ref), count) => {
            if token_ids_ref.token_count() != count {
                bail!(
                    "raster decode select {label} token count mismatch: ref has {}, expected {count}",
                    token_ids_ref.token_count()
                );
            }
            if count > 0 {
                read_token_id_from_ref_roots(artifact_store_roots, token_ids_ref, 0)?;
            }
            Ok(())
        }
        (None, 0) => Ok(()),
        (None, _) => bail!("raster decode select {label} token ids root is missing"),
    }
}
