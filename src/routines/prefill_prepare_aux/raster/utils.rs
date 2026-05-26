use anyhow::{anyhow, bail, Result};

use super::types::*;
use crate::dsl::prelude::auth_read;
use crate::input_embedding::raster::RasterInputEmbeddingRefs;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::merkle::merkle_root;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, token_id_leaf, RasterActivationSequenceArtifactRef,
    RasterArtifactBuilderRef, RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots,
    RasterTokenIdSequenceRef, TOKEN_ID_ARTIFACT_DOMAIN,
};
use crate::shared::model::transformer::{ActivationSequence, InternalActivationSequence};
use crate::shared::raster_contracts::prefill_ple::{
    AuthenticatedGemmaPleSource, GemmaPleLayerMetadataRequest, GemmaPleMetadataRequest,
};
use crate::shared::raster_kernels::transformer::{
    validate_projection_rows_per_tile, validate_sequence_rows_per_tile,
};
use crate::shared::raster_kernels::transformer::{RasterActivationRow, RasterActivationSequence};
use crate::RasterSizingControls;

pub(in super::super) fn reset_artifact_store() {
    ArtifactIo::reset_store();
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
    let row: RasterActivationRow =
        ArtifactIo::read_authenticated_leaf(activation_ref.artifact_ref(), row_idx)?
            .deserialize()?;
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
    ArtifactIo::read_authenticated_leaf_from_roots(roots, token_ids_ref.artifact_ref(), token_idx)?
        .deserialize()
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

pub(in super::super) fn artifact_source_name_for_root(
    artifact_store_roots: &RasterArtifactStoreRoots,
    root: &str,
) -> Result<String> {
    Ok(artifact_store_roots
        .artifact_entry_for_root(root)?
        .id()
        .source_name()
        .to_string())
}

pub(in super::super) fn token_ids_root<'a>(
    artifact_store_roots: &'a RasterArtifactStoreRoots,
    source_name: &str,
) -> Result<&'a str> {
    artifact_store_roots.artifact_root_for_source_name(source_name)
}

pub fn prepare_raster_prefill_ple_input_roots(
    token_ids: &[u32],
    input_activations: &ActivationSequence,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, RasterPrefillPleInputRoots)> {
    reset_artifact_store();
    let artifact_store_roots = ArtifactIo::export_store_roots();
    let ple_committed_source = ple_source.committed_source()?;
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    validate_sequence_rows_per_tile(raster_sizing.sequence_rows_per_tile)?;
    let (artifact_store_roots, token_ids_ref) =
        store_prefill_token_ids_artifact_with_roots(&artifact_store_roots, token_ids)?;
    let token_count = token_ids_ref.token_count();
    let token_ids_source_name = token_ids_ref.id().source_name().to_string();
    let metadata = auth_read!(&ple_committed_source, GemmaPleMetadataRequest)?;
    if !metadata.has_ple_global {
        return Ok((
            artifact_store_roots,
            RasterPrefillPleInputRoots {
                source_id: metadata.source_id,
                ple_source_root: ple_committed_source.root().to_string(),
                token_ids_source_name,
                token_count: token_ids_ref.token_count(),
                input_activations_ref: None,
                layer_count: metadata.layer_count,
                has_ple_global: false,
                raster_sizing,
            },
        ));
    }

    if metadata.layer_count == 0 {
        bail!("transformer PLE computation requires at least one layer");
    }
    if metadata.token_embedding_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            metadata.token_embedding_layer_count,
            metadata.layer_count
        );
    }
    if metadata.model_projection_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            metadata.model_projection_layer_count,
            metadata.layer_count
        );
    }

    let input_activations = raster_activation_sequence_from_embedding(input_activations)?;
    if input_activations.is_empty() {
        bail!("transformer PLE computation requires at least one activation row");
    }
    if token_count != input_activations.len() {
        bail!(
            "transformer PLE computation requires token ids and activations to have matching lengths"
        );
    }

    let first_layer = auth_read!(
        &ple_committed_source,
        GemmaPleLayerMetadataRequest { layer_idx: 0 }
    )?;
    let activation_width = input_activations.width()?;
    if activation_width != first_layer.hidden_width {
        bail!(
            "input activations row 0 has width {}, expected {}",
            activation_width,
            first_layer.hidden_width
        );
    }
    let (artifact_store_roots, input_activations_ref) = insert_activation_sequence_with_roots(
        &artifact_store_roots,
        RasterArtifactId::new("prefill.prepare_aux.input.initial")?,
        input_activations,
    )?;

    Ok((
        artifact_store_roots,
        RasterPrefillPleInputRoots {
            source_id: metadata.source_id,
            ple_source_root: ple_committed_source.root().to_string(),
            token_ids_source_name,
            token_count: token_ids_ref.token_count(),
            input_activations_ref: Some(input_activations_ref),
            layer_count: metadata.layer_count,
            has_ple_global: true,
            raster_sizing,
        },
    ))
}

pub fn prepare_raster_prefill_ple_input_roots_from_embedding_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, RasterPrefillPleInputRoots)> {
    let ple_committed_source = ple_source.committed_source()?;
    validate_projection_rows_per_tile(raster_sizing.projection_rows_per_tile)?;
    validate_sequence_rows_per_tile(raster_sizing.sequence_rows_per_tile)?;
    let token_ids_source_name = artifact_source_name_for_root(
        &artifact_store_roots,
        &input_embedding_refs.prompt_token_ids_root,
    )?;
    let metadata = auth_read!(&ple_committed_source, GemmaPleMetadataRequest)?;
    if !metadata.has_ple_global {
        return Ok((
            artifact_store_roots,
            RasterPrefillPleInputRoots {
                source_id: metadata.source_id,
                ple_source_root: ple_committed_source.root().to_string(),
                token_ids_source_name,
                token_count: input_embedding_refs.prompt_token_count,
                input_activations_ref: None,
                layer_count: metadata.layer_count,
                has_ple_global: false,
                raster_sizing,
            },
        ));
    }

    if metadata.layer_count == 0 {
        bail!("transformer PLE computation requires at least one layer");
    }
    if metadata.token_embedding_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            metadata.token_embedding_layer_count,
            metadata.layer_count
        );
    }
    if metadata.model_projection_layer_count != metadata.layer_count {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            metadata.model_projection_layer_count,
            metadata.layer_count
        );
    }
    if input_embedding_refs.prompt_token_count
        != input_embedding_refs
            .embedded_prompt_activations_ref
            .row_count()
    {
        bail!("transformer PLE computation requires token ids and activations to have matching lengths");
    }

    let first_layer = auth_read!(
        &ple_committed_source,
        GemmaPleLayerMetadataRequest { layer_idx: 0 }
    )?;
    if input_embedding_refs.embedded_prompt_activations_ref.width() != first_layer.hidden_width {
        bail!(
            "input activations row 0 has width {}, expected {}",
            input_embedding_refs.embedded_prompt_activations_ref.width(),
            first_layer.hidden_width
        );
    }
    artifact_store_roots
        .artifact_entry_for_root(input_embedding_refs.embedded_prompt_activations_ref.root())?;

    Ok((
        artifact_store_roots,
        RasterPrefillPleInputRoots {
            source_id: metadata.source_id,
            ple_source_root: ple_committed_source.root().to_string(),
            token_ids_source_name,
            token_count: input_embedding_refs.prompt_token_count,
            input_activations_ref: Some(
                input_embedding_refs.embedded_prompt_activations_ref.clone(),
            ),
            layer_count: metadata.layer_count,
            has_ple_global: true,
            raster_sizing,
        },
    ))
}

pub fn run(
    token_ids: &[u32],
    input_activations: &ActivationSequence,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<(RasterArtifactStoreRoots, Option<String>)> {
    let ple_committed_source = ple_source.committed_source()?;
    let (artifact_store_roots, input_roots) = prepare_raster_prefill_ple_input_roots(
        token_ids,
        input_activations,
        ple_source,
        raster_sizing,
    )?;
    super::tiles::main(artifact_store_roots, input_roots, &ple_committed_source)
}

pub fn run_with_input_embedding_refs(
    artifact_store_roots: RasterArtifactStoreRoots,
    input_embedding_refs: &RasterInputEmbeddingRefs,
    ple_source: &AuthenticatedGemmaPleSource,
    raster_sizing: RasterSizingControls,
) -> Result<RasterPrefillPleOutput> {
    let ple_committed_source = ple_source.committed_source()?;
    let (artifact_store_roots, input_roots) =
        prepare_raster_prefill_ple_input_roots_from_embedding_refs(
            artifact_store_roots,
            input_embedding_refs,
            ple_source,
            raster_sizing,
        )?;
    let (artifact_store_roots, refs) =
        super::tiles::main(artifact_store_roots, input_roots, &ple_committed_source)?;
    Ok(RasterPrefillPleOutput::new(artifact_store_roots, refs))
}
