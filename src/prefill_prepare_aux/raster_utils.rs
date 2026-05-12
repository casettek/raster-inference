use anyhow::{anyhow, bail, Result};

use crate::shared::artifact_io::ArtifactIo;
use crate::shared::merkle::merkle_root;
use crate::shared::raster_artifact_store::{
    activation_row_leaf, decode_activation_row_leaf, RasterActivationSequenceArtifactRef,
    RasterArtifactBuilderRef, RasterArtifactId, RasterArtifactMetadata, RasterTokenIdSequenceRef,
    TOKEN_ID_ARTIFACT_DOMAIN,
};
use crate::shared::raster_row_store::AuthenticatedRasterTensorStore;
use crate::shared::raster_transformer_kernels::{RasterActivationRow, RasterActivationSequence};
use crate::shared::transformer::{ActivationSequence, InternalActivationSequence};

pub(super) fn reset_artifact_store() {
    ArtifactIo::reset_store();
}

pub(super) fn tensor_store_snapshot() -> AuthenticatedRasterTensorStore {
    AuthenticatedRasterTensorStore::new()
}

pub(super) fn insert_activation_sequence(
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

pub(super) fn start_sequence_builder(
    id: RasterArtifactId,
    row_count: usize,
    width: usize,
) -> Result<RasterArtifactBuilderRef> {
    ArtifactIo::start_builder(
        id,
        RasterArtifactMetadata::activation_rows(row_count, width)?,
    )
}

pub(super) fn append_sequence_row_by_builder_root(
    builder_root: &str,
    row_idx: usize,
    row: RasterActivationRow,
) -> Result<String> {
    ArtifactIo::append_leaf_by_builder_root(builder_root, row_idx, activation_row_leaf(&row))
}

pub(super) fn finalize_sequence_builder_by_root(
    builder_root: &str,
) -> Result<RasterActivationSequenceArtifactRef> {
    RasterActivationSequenceArtifactRef::new(ArtifactIo::finalize_builder_by_root(builder_root)?)
}

pub(super) fn materialize_sequence(
    artifact_ref: &RasterActivationSequenceArtifactRef,
) -> Result<RasterActivationSequence> {
    let rows = (0..artifact_ref.row_count())
        .map(|row_idx| read_activation_row_from_ref(artifact_ref, row_idx))
        .collect::<Result<Vec<_>>>()?;
    Ok(RasterActivationSequence::from_rows(rows))
}

pub(super) fn read_activation_row_from_ref(
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

pub(super) fn store_prefill_token_ids_artifact(
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

fn prefill_token_ids_artifact_id(token_ids: &[u32]) -> Result<RasterArtifactId> {
    RasterArtifactId::new(format!(
        "prefill.prepare_aux.token_ids.{}",
        crate::trace::sha256_hex(&token_ids)
    ))
}

pub(super) fn read_prefill_token_id(
    token_ids_artifact_root: &str,
    token_count: usize,
    token_idx: usize,
) -> Result<u32> {
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

fn token_id_leaf(token_id: u32) -> Vec<u8> {
    token_id.to_le_bytes().to_vec()
}

fn decode_token_id_leaf(payload: &[u8]) -> Result<u32> {
    if payload.len() != 4 {
        bail!("token-id leaf payload must be exactly four bytes");
    }
    Ok(u32::from_le_bytes(
        payload.try_into().expect("payload length checked above"),
    ))
}

pub(super) fn raster_activation_sequence_from_embedding(
    input_activations: &ActivationSequence,
) -> Result<RasterActivationSequence> {
    let internal = input_activations.clone_internal();
    let det_rows = internal.det_values().ok_or_else(|| {
        anyhow!("deterministic raster PLE input requires canonical embedded prompt activations")
    })?;
    Ok(RasterActivationSequence::from_acts(det_rows.to_vec()))
}

pub(super) fn internal_sequence_from_raster(
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
