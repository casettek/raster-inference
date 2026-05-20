use anyhow::{anyhow, bail, Result};

use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::merkle::merkle_root;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, decode_activation_row_leaf, decode_token_id_leaf, token_id_leaf,
    RasterActivationSequenceArtifactRef, RasterArtifactBuilderRef, RasterArtifactId,
    RasterArtifactMetadata, RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
    TOKEN_ID_ARTIFACT_DOMAIN,
};
use crate::shared::model::transformer::{ActivationSequence, InternalActivationSequence};
use crate::shared::raster_kernels::transformer::{RasterActivationRow, RasterActivationSequence};
use crate::shared::tensors::raster_row_store::AuthenticatedRasterTensorStore;

pub(in super::super) fn reset_artifact_store() {
    ArtifactIo::reset_store();
}

pub(in super::super) fn tensor_store_snapshot() -> AuthenticatedRasterTensorStore {
    AuthenticatedRasterTensorStore::new()
}

pub(in super::super) fn insert_activation_sequence(
    id: RasterArtifactId,
    sequence: RasterActivationSequence,
) -> Result<RasterActivationSequenceArtifactRef> {
    let width = sequence.width()?;
    let leaves = sequence
        .rows()
        .iter()
        .map(activation_row_leaf)
        .collect::<Vec<_>>();
    let artifact_ref = ArtifactIo::insert_artifact(
        id,
        RasterArtifactMetadata::activation_rows(sequence.len(), width)?,
        leaves,
    )?;
    RasterActivationSequenceArtifactRef::new(artifact_ref)
}

pub(in super::super) fn insert_activation_sequence_with_roots(
    roots: &RasterArtifactStoreRoots,
    id: RasterArtifactId,
    sequence: RasterActivationSequence,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let width = sequence.width()?;
    let leaves = sequence
        .rows()
        .iter()
        .map(activation_row_leaf)
        .collect::<Vec<_>>();
    let (roots, artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        roots,
        id,
        RasterArtifactMetadata::activation_rows(sequence.len(), width)?,
        leaves,
    )?;
    Ok((
        roots,
        RasterActivationSequenceArtifactRef::new(artifact_ref)?,
    ))
}

pub(in super::super) fn start_sequence_builder_with_roots(
    roots: &RasterArtifactStoreRoots,
    id: RasterArtifactId,
    row_count: usize,
    width: usize,
) -> Result<(RasterArtifactStoreRoots, RasterArtifactBuilderRef)> {
    ArtifactIo::start_builder_with_roots(
        roots,
        id,
        RasterArtifactMetadata::activation_rows(row_count, width)?,
    )
}

pub(in super::super) fn append_sequence_row_by_builder_root_with_roots(
    roots: &RasterArtifactStoreRoots,
    builder_root: &str,
    row_idx: usize,
    row: RasterActivationRow,
) -> Result<(RasterArtifactStoreRoots, String)> {
    ArtifactIo::append_leaf_by_builder_root_with_roots(
        roots,
        builder_root,
        row_idx,
        activation_row_leaf(&row),
    )
}

pub(in super::super) fn finalize_sequence_builder_by_root_with_roots(
    roots: &RasterArtifactStoreRoots,
    builder_root: &str,
) -> Result<(
    RasterArtifactStoreRoots,
    RasterActivationSequenceArtifactRef,
)> {
    let (roots, artifact_ref) =
        ArtifactIo::finalize_builder_by_root_with_roots(roots, builder_root)?;
    Ok((
        roots,
        RasterActivationSequenceArtifactRef::new(artifact_ref)?,
    ))
}

pub(in super::super) fn materialize_sequence(
    artifact_ref: &RasterActivationSequenceArtifactRef,
) -> Result<RasterActivationSequence> {
    let rows = (0..artifact_ref.row_count())
        .map(|row_idx| read_activation_row_from_ref(artifact_ref, row_idx))
        .collect::<Result<Vec<_>>>()?;
    Ok(RasterActivationSequence::from_rows(rows))
}

pub(in super::super) fn read_activation_row_from_ref(
    activation_ref: &RasterActivationSequenceArtifactRef,
    row_idx: usize,
) -> Result<RasterActivationRow> {
    if row_idx >= activation_ref.row_count() {
        bail!(
            "activation row index {row_idx} is out of range for {} rows",
            activation_ref.row_count()
        );
    }
    let read = ArtifactIo::read_leaf(activation_ref.artifact_ref(), row_idx)?;
    ArtifactIo::verify_artifact_read(activation_ref.artifact_ref(), &read)?;
    let row = decode_activation_row_leaf(read.payload())?;
    if row.width() != activation_ref.width() {
        bail!(
            "activation artifact row {row_idx} has width {}, expected {}",
            row.width(),
            activation_ref.width()
        );
    }
    Ok(row)
}

pub(in super::super) fn store_prefill_token_ids_artifact(
    token_ids: &[u32],
) -> Result<RasterTokenIdSequenceRef> {
    let leaves = token_ids
        .iter()
        .copied()
        .map(token_id_leaf)
        .collect::<Vec<_>>();
    let token_ids_artifact_root = merkle_root(TOKEN_ID_ARTIFACT_DOMAIN.as_bytes(), &leaves);
    if let Ok(artifact_ref) = ArtifactIo::artifact_ref_for_root(&token_ids_artifact_root) {
        return RasterTokenIdSequenceRef::new(artifact_ref);
    }

    let artifact_ref = ArtifactIo::insert_artifact(
        prefill_token_ids_artifact_id(token_ids)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?;
    RasterTokenIdSequenceRef::new(artifact_ref)
}

pub(in super::super) fn store_prefill_token_ids_artifact_with_roots(
    roots: &RasterArtifactStoreRoots,
    token_ids: &[u32],
) -> Result<(RasterArtifactStoreRoots, RasterTokenIdSequenceRef)> {
    let leaves = token_ids
        .iter()
        .copied()
        .map(token_id_leaf)
        .collect::<Vec<_>>();
    let token_ids_artifact_root = merkle_root(TOKEN_ID_ARTIFACT_DOMAIN.as_bytes(), &leaves);
    if let Ok(artifact_ref) = ArtifactIo::artifact_ref_for_root(&token_ids_artifact_root) {
        roots.artifact_entry_for_root(&token_ids_artifact_root)?;
        return Ok((roots.clone(), RasterTokenIdSequenceRef::new(artifact_ref)?));
    }

    let (roots, artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        roots,
        prefill_token_ids_artifact_id(token_ids)?,
        RasterArtifactMetadata::token_ids(token_ids.len()),
        leaves,
    )?;
    Ok((roots, RasterTokenIdSequenceRef::new(artifact_ref)?))
}

fn prefill_token_ids_artifact_id(token_ids: &[u32]) -> Result<RasterArtifactId> {
    RasterArtifactId::new(format!(
        "prefill.prepare_aux.token_ids.{}",
        crate::trace::sha256_hex(&token_ids)
    ))
}

pub(in super::super) fn read_prefill_token_id(
    roots: &RasterArtifactStoreRoots,
    token_ids_artifact_root: &str,
    token_count: usize,
    token_idx: usize,
) -> Result<u32> {
    roots.artifact_entry_for_root(token_ids_artifact_root)?;
    let token_ids_ref =
        RasterTokenIdSequenceRef::new(ArtifactIo::artifact_ref_for_root(token_ids_artifact_root)?)?;
    if token_ids_ref.token_count() != token_count {
        bail!(
            "PLE token-id artifact has {} tokens, expected {token_count}",
            token_ids_ref.token_count()
        );
    }
    if token_idx >= token_count {
        bail!("PLE token index {token_idx} is out of range for {token_count} tokens");
    }
    let read = ArtifactIo::read_leaf(token_ids_ref.artifact_ref(), token_idx)?;
    ArtifactIo::verify_artifact_read(token_ids_ref.artifact_ref(), &read)?;
    decode_token_id_leaf(read.payload())
}

pub(in super::super) fn raster_activation_sequence_from_embedding(
    input_activations: &ActivationSequence,
) -> Result<RasterActivationSequence> {
    raster_activation_sequence_from_internal(&input_activations.clone_internal())
}

pub(in super::super) fn raster_activation_sequence_from_internal(
    input_activations: &InternalActivationSequence,
) -> Result<RasterActivationSequence> {
    let det_rows = input_activations.det_values().ok_or_else(|| {
        anyhow!("deterministic raster activation sequence requires canonical activations")
    })?;
    Ok(RasterActivationSequence::from_acts(det_rows.to_vec()))
}

pub(in super::super) fn internal_sequence_from_raster(
    sequence: RasterActivationSequence,
) -> InternalActivationSequence {
    InternalActivationSequence::from_det_values(
        sequence
            .into_rows()
            .into_iter()
            .map(|row| row.acts())
            .collect(),
    )
}
