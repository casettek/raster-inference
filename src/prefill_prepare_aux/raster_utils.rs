use std::cell::RefCell;

use anyhow::{anyhow, bail, Result};

use crate::shared::artifact_io::ArtifactIo;
use crate::shared::det_num::Act;
use crate::shared::merkle::merkle_root;
use crate::shared::raster_artifact_store::{
    RasterArtifactId, RasterArtifactMetadata, RasterTokenIdSequenceRef, TOKEN_ID_ARTIFACT_DOMAIN,
};
use crate::shared::raster_row_store::{
    AuthenticatedRasterTensorStore, RasterActivationSequenceRef, RasterTensorBuilderRef,
    RasterTensorId,
};
use crate::shared::raster_transformer_kernels::{
    append_projection_chunk_to_state, compute_next_sequence_binary_row,
    compute_next_sequence_unary_row, finalize_sequence_binary_row_state_ref,
    finalize_sequence_projection_state, finalize_sequence_projection_state_ref,
    finalize_sequence_unary_row_state_ref, init_sequence_add_row_state_from_refs,
    init_sequence_projection_state, init_sequence_projection_state_from_ref,
    init_sequence_rms_norm_row_state_from_ref, init_sequence_scale_row_state_from_ref,
    RasterActivationRow, RasterActivationSequence, RasterSequenceBinaryState,
    RasterSequenceProjectionState, RasterSequenceUnaryState,
};
use crate::shared::transformer::{ActivationSequence, InternalActivationSequence};

thread_local! {
    static PREFILL_PLE_TENSOR_STORE: RefCell<AuthenticatedRasterTensorStore> =
        RefCell::new(AuthenticatedRasterTensorStore::new());
}

pub(super) fn reset_tensor_store() {
    PREFILL_PLE_TENSOR_STORE.with(|store_ref| {
        *store_ref.borrow_mut() = AuthenticatedRasterTensorStore::new();
    });
}

pub(super) fn tensor_store_snapshot() -> AuthenticatedRasterTensorStore {
    PREFILL_PLE_TENSOR_STORE.with(|store_ref| store_ref.borrow().clone())
}

fn with_tensor_store<T>(
    f: impl FnOnce(&mut AuthenticatedRasterTensorStore) -> Result<T>,
) -> Result<T> {
    PREFILL_PLE_TENSOR_STORE.with(|store_ref| {
        let mut store = store_ref.borrow_mut();
        f(&mut store)
    })
}

fn read_tensor_store<T>(f: impl FnOnce(&AuthenticatedRasterTensorStore) -> Result<T>) -> Result<T> {
    PREFILL_PLE_TENSOR_STORE.with(|store_ref| {
        let store = store_ref.borrow();
        f(&store)
    })
}

pub(super) fn insert_activation_sequence(
    id: RasterTensorId,
    sequence: RasterActivationSequence,
) -> Result<RasterActivationSequenceRef> {
    with_tensor_store(|store| store.insert_activation_sequence(id, sequence))
}

pub(super) fn start_sequence_builder(
    id: RasterTensorId,
    row_count: usize,
    width: usize,
) -> Result<RasterTensorBuilderRef> {
    with_tensor_store(|store| store.start_sequence_builder(id, row_count, width))
}

pub(super) fn append_sequence_row(
    builder_ref: &mut RasterTensorBuilderRef,
    row_idx: usize,
    row: RasterActivationRow,
) -> Result<()> {
    with_tensor_store(|store| store.append_sequence_row(builder_ref, row_idx, row))
}

pub(super) fn finalize_sequence_builder(
    builder_ref: RasterTensorBuilderRef,
) -> Result<RasterActivationSequenceRef> {
    with_tensor_store(|store| store.finalize_sequence_builder(builder_ref))
}

pub(super) fn materialize_sequence(
    tensor_ref: &RasterActivationSequenceRef,
) -> Result<RasterActivationSequence> {
    read_tensor_store(|store| store.materialize_sequence(tensor_ref))
}

pub(super) fn init_projection_state(
    input: &RasterActivationSequence,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    with_tensor_store(|store| {
        init_sequence_projection_state(store, input, projection_rows, projection_rows_per_tile)
    })
}

pub(super) fn init_projection_state_from_ref(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    projection_rows: usize,
    projection_rows_per_tile: usize,
) -> Result<RasterSequenceProjectionState> {
    with_tensor_store(|store| {
        init_sequence_projection_state_from_ref(
            store,
            input_ref,
            output_id,
            projection_rows,
            projection_rows_per_tile,
        )
    })
}

pub(super) fn append_projection_chunk(
    state: &mut RasterSequenceProjectionState,
    rows: &[Vec<crate::shared::det_num::Wgt>],
) -> Result<()> {
    with_tensor_store(|store| append_projection_chunk_to_state(state, store, rows))
}

pub(super) fn finalize_projection_state(
    state: RasterSequenceProjectionState,
) -> Result<RasterActivationSequence> {
    with_tensor_store(|store| finalize_sequence_projection_state(state, store))
}

pub(super) fn finalize_projection_state_ref(
    state: RasterSequenceProjectionState,
) -> Result<RasterActivationSequenceRef> {
    with_tensor_store(|store| finalize_sequence_projection_state_ref(state, store))
}

pub(super) fn init_scale_row_state_from_ref(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    scalar: Option<Act>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    with_tensor_store(|store| {
        init_sequence_scale_row_state_from_ref(store, input_ref, output_id, scalar, rows_per_tile)
    })
}

pub(super) fn init_rms_norm_row_state_from_ref(
    input_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    norm_weights: Option<&[crate::shared::det_num::Wgt]>,
    eps: Option<crate::shared::det_num::Acc>,
    rows_per_tile: usize,
) -> Result<RasterSequenceUnaryState> {
    with_tensor_store(|store| {
        init_sequence_rms_norm_row_state_from_ref(
            store,
            input_ref,
            output_id,
            norm_weights,
            eps,
            rows_per_tile,
        )
    })
}

pub(super) fn compute_next_unary_row(
    state: RasterSequenceUnaryState,
) -> Result<(bool, RasterSequenceUnaryState)> {
    with_tensor_store(|store| compute_next_sequence_unary_row(state, store))
}

pub(super) fn finalize_unary_row_state_ref(
    state: RasterSequenceUnaryState,
) -> Result<RasterActivationSequenceRef> {
    with_tensor_store(|store| finalize_sequence_unary_row_state_ref(state, store))
}

pub(super) fn init_add_row_state_from_refs(
    lhs_ref: RasterActivationSequenceRef,
    rhs_ref: RasterActivationSequenceRef,
    output_id: RasterTensorId,
    rows_per_tile: usize,
) -> Result<RasterSequenceBinaryState> {
    with_tensor_store(|store| {
        init_sequence_add_row_state_from_refs(store, lhs_ref, rhs_ref, output_id, rows_per_tile)
    })
}

pub(super) fn compute_next_binary_row(
    state: RasterSequenceBinaryState,
) -> Result<(bool, RasterSequenceBinaryState)> {
    with_tensor_store(|store| compute_next_sequence_binary_row(state, store))
}

pub(super) fn finalize_binary_row_state_ref(
    state: RasterSequenceBinaryState,
) -> Result<RasterActivationSequenceRef> {
    with_tensor_store(|store| finalize_sequence_binary_row_state_ref(state, store))
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
