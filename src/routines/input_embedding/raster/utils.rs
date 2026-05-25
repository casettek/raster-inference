use anyhow::{anyhow, bail, Result};

use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, RasterActivationSequenceArtifactRef, RasterArtifactBuilderRef,
    RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::model::transformer::{ActivationSequence, InternalActivationSequence};
use crate::shared::raster_kernels::transformer::{RasterActivationRow, RasterActivationSequence};

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

pub(in super::super) fn start_activation_sequence_builder_with_roots(
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

pub(in super::super) fn append_activation_row_by_builder_root_with_roots(
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

pub(in super::super) fn finalize_activation_sequence_builder_by_root_with_roots(
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

pub(in super::super) fn read_prompt_token_id(
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
            "input embedding token-id artifact has {} tokens, expected {token_count}",
            token_ids_ref.token_count()
        );
    }
    if token_idx >= token_count {
        bail!("input embedding token index {token_idx} is out of range for {token_count} tokens");
    }
    ArtifactIo::read_verified_leaf_from_roots(roots, token_ids_ref.artifact_ref(), token_idx)?
        .deserialize()
}

pub(in super::super) fn materialize_sequence(
    artifact_ref: &RasterActivationSequenceArtifactRef,
) -> Result<RasterActivationSequence> {
    let rows = (0..artifact_ref.row_count())
        .map(|row_idx| read_activation_row_from_ref(artifact_ref, row_idx))
        .collect::<Result<Vec<_>>>()?;
    Ok(RasterActivationSequence::from_rows(rows))
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

pub(in super::super) fn raster_activation_sequence_from_embedding(
    input_activations: &ActivationSequence,
) -> Result<RasterActivationSequence> {
    let internal = input_activations.clone_internal();
    let det_rows = internal.det_values().ok_or_else(|| {
        anyhow!("deterministic raster input embedding checkpoint requires canonical embedded prompt activations")
    })?;
    Ok(RasterActivationSequence::from_acts(det_rows.to_vec()))
}

fn read_activation_row_from_ref(
    activation_ref: &RasterActivationSequenceArtifactRef,
    row_idx: usize,
) -> Result<RasterActivationRow> {
    if row_idx >= activation_ref.row_count() {
        bail!(
            "activation row index {row_idx} is out of range for {} rows",
            activation_ref.row_count()
        );
    }
    let row: RasterActivationRow =
        ArtifactIo::read_verified_leaf(activation_ref.artifact_ref(), row_idx)?.deserialize()?;
    if row.width() != activation_ref.width() {
        bail!(
            "activation artifact row {row_idx} has width {}, expected {}",
            row.width(),
            activation_ref.width()
        );
    }
    Ok(row)
}
