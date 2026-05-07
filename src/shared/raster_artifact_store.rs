use std::{cell::RefCell, collections::HashMap};

use anyhow::{anyhow, bail, Context, Result};

use crate::shared::merkle::{merkle_proof, merkle_root, verify_merkle_proof, MerkleProof};
use crate::shared::raster_transformer_kernels::RasterActivationRow;

pub const TOKEN_ID_ARTIFACT_KIND: &str = "token_ids";
pub const BPE_PIECE_ARTIFACT_KIND: &str = "bpe_pieces";
pub const ACTIVATION_ROW_ARTIFACT_KIND: &str = "activation_rows";

pub const TOKEN_ID_ARTIFACT_DOMAIN: &str = "raster-artifact-token-ids-merkle-v1";
pub const BPE_PIECE_ARTIFACT_DOMAIN: &str = "raster-artifact-bpe-pieces-merkle-v1";
pub const ACTIVATION_ROW_ARTIFACT_DOMAIN: &str = "raster-artifact-activation-sequence-merkle-v1";

const SHAPE_WIDTH: &str = "width";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterArtifactId {
    source_name: String,
}

impl RasterArtifactId {
    pub fn new(source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        if source_name.is_empty() {
            bail!("raster artifact id requires a non-empty source name");
        }
        Ok(Self { source_name })
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterArtifactShapeEntry {
    name: String,
    value: usize,
}

impl RasterArtifactShapeEntry {
    pub fn new(name: impl Into<String>, value: usize) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            bail!("raster artifact shape entry requires a non-empty name");
        }
        Ok(Self { name, value })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> usize {
        self.value
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterArtifactMetadata {
    kind: String,
    domain: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    leaf_count: Option<usize>,
    shape: Vec<RasterArtifactShapeEntry>,
}

impl RasterArtifactMetadata {
    pub fn new(
        kind: impl Into<String>,
        domain: impl Into<String>,
        leaf_count: usize,
        shape: Vec<RasterArtifactShapeEntry>,
    ) -> Result<Self> {
        let kind = kind.into();
        if kind.is_empty() {
            bail!("raster artifact metadata requires a non-empty kind");
        }
        let domain = domain.into();
        if domain.is_empty() {
            bail!("raster artifact metadata requires a non-empty domain");
        }
        Ok(Self {
            kind,
            domain,
            leaf_count: Some(leaf_count),
            shape,
        })
    }

    pub fn open(
        kind: impl Into<String>,
        domain: impl Into<String>,
        shape: Vec<RasterArtifactShapeEntry>,
    ) -> Result<Self> {
        let kind = kind.into();
        if kind.is_empty() {
            bail!("raster artifact metadata requires a non-empty kind");
        }
        let domain = domain.into();
        if domain.is_empty() {
            bail!("raster artifact metadata requires a non-empty domain");
        }
        Ok(Self {
            kind,
            domain,
            leaf_count: None,
            shape,
        })
    }

    pub fn token_ids(token_count: usize) -> Self {
        Self::new(
            TOKEN_ID_ARTIFACT_KIND,
            TOKEN_ID_ARTIFACT_DOMAIN,
            token_count,
            Vec::new(),
        )
        .expect("static token-id artifact metadata should be valid")
    }

    pub fn open_token_ids() -> Self {
        Self::open(TOKEN_ID_ARTIFACT_KIND, TOKEN_ID_ARTIFACT_DOMAIN, Vec::new())
            .expect("static token-id artifact metadata should be valid")
    }

    pub fn bpe_pieces(piece_count: usize) -> Self {
        Self::new(
            BPE_PIECE_ARTIFACT_KIND,
            BPE_PIECE_ARTIFACT_DOMAIN,
            piece_count,
            Vec::new(),
        )
        .expect("static BPE piece artifact metadata should be valid")
    }

    pub fn open_bpe_pieces() -> Self {
        Self::open(
            BPE_PIECE_ARTIFACT_KIND,
            BPE_PIECE_ARTIFACT_DOMAIN,
            Vec::new(),
        )
        .expect("static BPE piece artifact metadata should be valid")
    }

    pub fn activation_rows(row_count: usize, width: usize) -> Result<Self> {
        if row_count == 0 {
            bail!("raster activation-row artifact requires at least one row");
        }
        if width == 0 {
            bail!("raster activation-row artifact requires non-zero width");
        }
        Ok(Self::new(
            ACTIVATION_ROW_ARTIFACT_KIND,
            ACTIVATION_ROW_ARTIFACT_DOMAIN,
            row_count,
            vec![RasterArtifactShapeEntry::new(SHAPE_WIDTH, width)?],
        )?)
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn domain(&self) -> &str {
        &self.domain
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_count
            .expect("raster artifact metadata leaf count should be known")
    }

    pub fn leaf_count_opt(&self) -> Option<usize> {
        self.leaf_count
    }

    pub fn with_leaf_count(mut self, leaf_count: usize) -> Self {
        self.leaf_count = Some(leaf_count);
        self
    }

    pub fn shape_value(&self, name: &str) -> Option<usize> {
        self.shape
            .iter()
            .find(|entry| entry.name() == name)
            .map(RasterArtifactShapeEntry::value)
    }

    fn domain_bytes(&self) -> &[u8] {
        self.domain.as_bytes()
    }

    fn ensure_kind(&self, expected: &str) -> Result<()> {
        if self.kind != expected {
            bail!("raster artifact kind mismatch: {} vs {expected}", self.kind);
        }
        Ok(())
    }

    fn ensure_finalized(&self) -> Result<()> {
        if self.leaf_count.is_none() {
            bail!("raster artifact metadata is missing finalized leaf count");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterArtifactRef {
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
    root: String,
}

impl RasterArtifactRef {
    pub fn id(&self) -> &RasterArtifactId {
        &self.id
    }

    pub fn metadata(&self) -> &RasterArtifactMetadata {
        &self.metadata
    }

    pub fn kind(&self) -> &str {
        self.metadata.kind()
    }

    pub fn root(&self) -> &str {
        &self.root
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterArtifactBuilderRef {
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
    leaves_written: usize,
    running_root: String,
}

impl RasterArtifactBuilderRef {
    pub fn id(&self) -> &RasterArtifactId {
        &self.id
    }

    pub fn metadata(&self) -> &RasterArtifactMetadata {
        &self.metadata
    }

    pub fn leaves_written(&self) -> usize {
        self.leaves_written
    }

    pub fn running_root(&self) -> &str {
        &self.running_root
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterArtifactRead {
    leaf_idx: usize,
    payload: Vec<u8>,
    proof: MerkleProof,
}

impl RasterArtifactRead {
    pub fn leaf_idx(&self) -> usize {
        self.leaf_idx
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn proof(&self) -> &MerkleProof {
        &self.proof
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterBpePieceSequenceRef {
    artifact_ref: RasterArtifactRef,
}

impl RasterBpePieceSequenceRef {
    pub fn new(artifact_ref: RasterArtifactRef) -> Result<Self> {
        artifact_ref
            .metadata()
            .ensure_kind(BPE_PIECE_ARTIFACT_KIND)?;
        artifact_ref.metadata().ensure_finalized()?;
        Ok(Self { artifact_ref })
    }

    pub fn artifact_ref(&self) -> &RasterArtifactRef {
        &self.artifact_ref
    }

    pub fn id(&self) -> &RasterArtifactId {
        self.artifact_ref.id()
    }

    pub fn piece_count(&self) -> usize {
        self.artifact_ref.metadata().leaf_count()
    }

    pub fn root(&self) -> &str {
        self.artifact_ref.root()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterTokenIdSequenceRef {
    artifact_ref: RasterArtifactRef,
}

impl RasterTokenIdSequenceRef {
    pub fn new(artifact_ref: RasterArtifactRef) -> Result<Self> {
        artifact_ref
            .metadata()
            .ensure_kind(TOKEN_ID_ARTIFACT_KIND)?;
        artifact_ref.metadata().ensure_finalized()?;
        Ok(Self { artifact_ref })
    }

    pub fn artifact_ref(&self) -> &RasterArtifactRef {
        &self.artifact_ref
    }

    pub fn id(&self) -> &RasterArtifactId {
        self.artifact_ref.id()
    }

    pub fn token_count(&self) -> usize {
        self.artifact_ref.metadata().leaf_count()
    }

    pub fn root(&self) -> &str {
        self.artifact_ref.root()
    }
}

#[derive(Debug, Clone)]
struct StoredArtifact {
    metadata: RasterArtifactMetadata,
    leaves: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct ArtifactBuilderState {
    metadata: RasterArtifactMetadata,
    leaves: Vec<Vec<u8>>,
}

#[derive(Debug, Default, Clone)]
pub struct RasterArtifactStore {
    artifacts: HashMap<RasterArtifactId, StoredArtifact>,
    builders: HashMap<RasterArtifactId, ArtifactBuilderState>,
}

thread_local! {
    static ARTIFACT_STORE: RefCell<RasterArtifactStore> =
        RefCell::new(RasterArtifactStore::new());
}

impl RasterArtifactStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start_builder(
        &mut self,
        id: RasterArtifactId,
        metadata: RasterArtifactMetadata,
    ) -> Result<RasterArtifactBuilderRef> {
        self.ensure_id_available(&id)?;
        let state = ArtifactBuilderState {
            metadata,
            leaves: Vec::new(),
        };
        let builder_ref = artifact_builder_ref(id.clone(), &state);
        self.builders.insert(id, state);
        Ok(builder_ref)
    }

    pub fn insert_artifact(
        &mut self,
        id: RasterArtifactId,
        metadata: RasterArtifactMetadata,
        leaves: Vec<Vec<u8>>,
    ) -> Result<RasterArtifactRef> {
        self.ensure_id_available(&id)?;
        let metadata = match metadata.leaf_count_opt() {
            Some(expected) if leaves.len() != expected => {
                bail!(
                    "raster artifact insert received {} leaves, expected {}",
                    leaves.len(),
                    expected
                );
            }
            Some(_) => metadata,
            None => metadata.with_leaf_count(leaves.len()),
        };
        let state = ArtifactBuilderState { metadata, leaves };
        let artifact_ref = artifact_ref(id.clone(), &state);
        self.artifacts.insert(
            id,
            StoredArtifact {
                metadata: state.metadata,
                leaves: state.leaves,
            },
        );
        Ok(artifact_ref)
    }

    pub fn start_activation_sequence_builder(
        &mut self,
        id: RasterArtifactId,
        row_count: usize,
        width: usize,
    ) -> Result<RasterArtifactBuilderRef> {
        self.start_builder(
            id,
            RasterArtifactMetadata::activation_rows(row_count, width)?,
        )
    }

    pub fn append_leaf(
        &mut self,
        builder_ref: &mut RasterArtifactBuilderRef,
        leaf_idx: usize,
        payload: Vec<u8>,
    ) -> Result<()> {
        let state = self.builder_mut(builder_ref)?;
        if leaf_idx != state.leaves.len() {
            bail!(
                "raster artifact builder expected leaf {}, received {leaf_idx}",
                state.leaves.len()
            );
        }
        if let Some(expected_leaf_count) = state.metadata.leaf_count_opt() {
            if leaf_idx >= expected_leaf_count {
                bail!(
                    "raster artifact builder leaf {leaf_idx} is out of range for {} leaves",
                    expected_leaf_count
                );
            }
        }
        state.leaves.push(payload);
        update_builder_ref(builder_ref, state);
        Ok(())
    }

    pub fn append_activation_row(
        &mut self,
        builder_ref: &mut RasterArtifactBuilderRef,
        row_idx: usize,
        row: &RasterActivationRow,
    ) -> Result<()> {
        builder_ref
            .metadata()
            .ensure_kind(ACTIVATION_ROW_ARTIFACT_KIND)?;
        let width = builder_ref
            .metadata()
            .shape_value(SHAPE_WIDTH)
            .ok_or_else(|| anyhow!("raster activation artifact metadata missing width"))?;
        if row.width() != width {
            bail!(
                "raster activation artifact row {row_idx} has width {}, expected {width}",
                row.width()
            );
        }
        self.append_leaf(builder_ref, row_idx, activation_row_leaf(row))
    }

    pub fn finalize_builder(
        &mut self,
        builder_ref: RasterArtifactBuilderRef,
    ) -> Result<RasterArtifactRef> {
        let state = self.builders.get(builder_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster artifact builder {} is not registered",
                builder_ref.id().source_name()
            )
        })?;
        ensure_builder_ref_matches(&builder_ref, state)?;
        if let Some(expected_leaf_count) = state.metadata.leaf_count_opt() {
            if state.leaves.len() != expected_leaf_count {
                bail!(
                    "raster artifact builder finalized with {} leaves, expected {}",
                    state.leaves.len(),
                    expected_leaf_count
                );
            }
        }
        let mut state = self
            .builders
            .remove(builder_ref.id())
            .expect("builder existence checked before removal");
        if state.metadata.leaf_count_opt().is_none() {
            state.metadata = state.metadata.with_leaf_count(state.leaves.len());
        }
        let artifact_ref = artifact_ref(builder_ref.id.clone(), &state);
        self.artifacts.insert(
            builder_ref.id,
            StoredArtifact {
                metadata: state.metadata,
                leaves: state.leaves,
            },
        );
        Ok(artifact_ref)
    }

    pub fn read_leaf(
        &self,
        artifact_ref: &RasterArtifactRef,
        leaf_idx: usize,
    ) -> Result<RasterArtifactRead> {
        let artifact = self.artifact(artifact_ref)?;
        if leaf_idx >= artifact.metadata.leaf_count() {
            bail!(
                "raster artifact leaf {leaf_idx} is out of range for {} leaves",
                artifact.metadata.leaf_count()
            );
        }
        let payload = artifact
            .leaves
            .get(leaf_idx)
            .ok_or_else(|| anyhow!("raster artifact leaf {leaf_idx} is missing"))?
            .clone();
        let proof = merkle_proof(artifact.metadata.domain_bytes(), &artifact.leaves, leaf_idx)?;
        Ok(RasterArtifactRead {
            leaf_idx,
            payload,
            proof,
        })
    }

    fn ensure_id_available(&self, id: &RasterArtifactId) -> Result<()> {
        if self.artifacts.contains_key(id) || self.builders.contains_key(id) {
            bail!(
                "raster artifact id {} is already registered",
                id.source_name()
            );
        }
        Ok(())
    }

    fn builder_mut(
        &mut self,
        builder_ref: &RasterArtifactBuilderRef,
    ) -> Result<&mut ArtifactBuilderState> {
        let state = self.builders.get_mut(builder_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster artifact builder {} is not registered",
                builder_ref.id().source_name()
            )
        })?;
        ensure_builder_ref_matches(builder_ref, state)?;
        Ok(state)
    }

    fn artifact(&self, artifact_ref: &RasterArtifactRef) -> Result<&StoredArtifact> {
        let artifact = self.artifacts.get(artifact_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster artifact ref {} is not registered",
                artifact_ref.id().source_name()
            )
        })?;
        if artifact.metadata != artifact_ref.metadata {
            bail!("raster artifact ref metadata mismatch");
        }
        let actual_root = artifact_root(&artifact.metadata, &artifact.leaves);
        if actual_root != artifact_ref.root {
            bail!(
                "raster artifact root mismatch: {} vs {}",
                actual_root,
                artifact_ref.root
            );
        }
        Ok(artifact)
    }
}

pub fn reset_artifact_store() {
    ARTIFACT_STORE.with(|store_ref| {
        *store_ref.borrow_mut() = RasterArtifactStore::new();
    });
}

pub fn with_artifact_store<T>(f: impl FnOnce(&mut RasterArtifactStore) -> Result<T>) -> Result<T> {
    ARTIFACT_STORE.with(|store_ref| {
        let mut store = store_ref.borrow_mut();
        f(&mut store)
    })
}

pub fn read_artifact_store<T>(f: impl FnOnce(&RasterArtifactStore) -> Result<T>) -> Result<T> {
    ARTIFACT_STORE.with(|store_ref| {
        let store = store_ref.borrow();
        f(&store)
    })
}

pub fn artifact_store_snapshot() -> RasterArtifactStore {
    ARTIFACT_STORE.with(|store_ref| store_ref.borrow().clone())
}

pub fn start_builder(
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
) -> Result<RasterArtifactBuilderRef> {
    with_artifact_store(|store| store.start_builder(id, metadata))
}

pub fn insert_artifact(
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
    leaves: Vec<Vec<u8>>,
) -> Result<RasterArtifactRef> {
    with_artifact_store(|store| store.insert_artifact(id, metadata, leaves))
}

pub fn append_leaf(
    builder_ref: &mut RasterArtifactBuilderRef,
    leaf_idx: usize,
    payload: Vec<u8>,
) -> Result<()> {
    with_artifact_store(|store| store.append_leaf(builder_ref, leaf_idx, payload))
}

pub fn finalize_builder(builder_ref: RasterArtifactBuilderRef) -> Result<RasterArtifactRef> {
    with_artifact_store(|store| store.finalize_builder(builder_ref))
}

pub fn read_leaf(artifact_ref: &RasterArtifactRef, leaf_idx: usize) -> Result<RasterArtifactRead> {
    read_artifact_store(|store| store.read_leaf(artifact_ref, leaf_idx))
}

pub fn verify_artifact_read(
    artifact_ref: &RasterArtifactRef,
    read: &RasterArtifactRead,
) -> Result<()> {
    if read.leaf_idx >= artifact_ref.metadata.leaf_count() {
        bail!(
            "raster artifact read leaf {} is out of range for {} leaves",
            read.leaf_idx,
            artifact_ref.metadata.leaf_count()
        );
    }
    verify_merkle_proof(
        artifact_ref.metadata.domain_bytes(),
        artifact_ref.root(),
        read.leaf_idx,
        &read.payload,
        &read.proof,
    )
}

pub fn token_id_leaf(token_id: u32) -> Vec<u8> {
    token_id.to_le_bytes().to_vec()
}

pub fn bpe_piece_leaf(piece: &str) -> Vec<u8> {
    let bytes = piece.as_bytes();
    let mut payload = Vec::with_capacity(8 + bytes.len());
    payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(bytes);
    payload
}

pub fn activation_row_leaf(row: &RasterActivationRow) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + row.width() * std::mem::size_of::<i32>());
    payload.extend_from_slice(&(row.width() as u64).to_le_bytes());
    for bits in row.act_bits() {
        payload.extend_from_slice(&bits.to_le_bytes());
    }
    payload
}

pub fn decode_token_id_leaf(payload: &[u8]) -> Result<u32> {
    let bytes: [u8; 4] = payload
        .try_into()
        .context("token-id leaf payload must be exactly four bytes")?;
    Ok(u32::from_le_bytes(bytes))
}

pub fn decode_bpe_piece_leaf(payload: &[u8]) -> Result<String> {
    if payload.len() < 8 {
        bail!("BPE piece leaf payload is too short");
    }
    let len = u64::from_le_bytes(
        payload[0..8]
            .try_into()
            .expect("slice length checked above"),
    ) as usize;
    let bytes = &payload[8..];
    if bytes.len() != len {
        bail!("BPE piece leaf length mismatch: {} vs {len}", bytes.len());
    }
    String::from_utf8(bytes.to_vec()).context("BPE piece leaf is not valid UTF-8")
}

fn artifact_ref(id: RasterArtifactId, state: &ArtifactBuilderState) -> RasterArtifactRef {
    RasterArtifactRef {
        id,
        metadata: state.metadata.clone(),
        root: artifact_root(&state.metadata, &state.leaves),
    }
}

fn artifact_builder_ref(
    id: RasterArtifactId,
    state: &ArtifactBuilderState,
) -> RasterArtifactBuilderRef {
    RasterArtifactBuilderRef {
        id,
        metadata: state.metadata.clone(),
        leaves_written: state.leaves.len(),
        running_root: artifact_root(&state.metadata, &state.leaves),
    }
}

fn update_builder_ref(builder_ref: &mut RasterArtifactBuilderRef, state: &ArtifactBuilderState) {
    *builder_ref = artifact_builder_ref(builder_ref.id.clone(), state);
}

fn artifact_root(metadata: &RasterArtifactMetadata, leaves: &[Vec<u8>]) -> String {
    merkle_root(metadata.domain_bytes(), leaves)
}

fn ensure_builder_ref_matches(
    builder_ref: &RasterArtifactBuilderRef,
    state: &ArtifactBuilderState,
) -> Result<()> {
    let expected = artifact_builder_ref(builder_ref.id.clone(), state);
    if expected != *builder_ref {
        bail!(
            "raster artifact builder {} metadata mismatch",
            builder_ref.id().source_name()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::det_num::Act;

    fn artifact_id(name: &str) -> RasterArtifactId {
        RasterArtifactId::new(name).expect("artifact id")
    }

    fn start_token_builder(
        store: &mut RasterArtifactStore,
        name: &str,
        token_count: usize,
    ) -> RasterArtifactBuilderRef {
        store
            .start_builder(
                artifact_id(name),
                RasterArtifactMetadata::token_ids(token_count),
            )
            .expect("token builder")
    }

    fn finalize_token_builder(
        store: &mut RasterArtifactStore,
        builder: RasterArtifactBuilderRef,
    ) -> RasterTokenIdSequenceRef {
        RasterTokenIdSequenceRef::new(store.finalize_builder(builder).expect("finalize"))
            .expect("token ref")
    }

    fn append_token_id(
        store: &mut RasterArtifactStore,
        builder: &mut RasterArtifactBuilderRef,
        token_idx: usize,
        token_id: u32,
    ) -> Result<()> {
        store.append_leaf(builder, token_idx, token_id_leaf(token_id))
    }

    fn read_token_id(
        store: &RasterArtifactStore,
        token_ref: &RasterTokenIdSequenceRef,
        token_idx: usize,
    ) -> Result<u32> {
        let read = store.read_leaf(token_ref.artifact_ref(), token_idx)?;
        verify_artifact_read(token_ref.artifact_ref(), &read)?;
        decode_token_id_leaf(read.payload())
    }

    fn materialize_token_ids(
        store: &RasterArtifactStore,
        token_ref: &RasterTokenIdSequenceRef,
    ) -> Result<Vec<u32>> {
        (0..token_ref.token_count())
            .map(|token_idx| read_token_id(store, token_ref, token_idx))
            .collect()
    }

    fn insert_bpe_pieces(
        store: &mut RasterArtifactStore,
        name: &str,
        pieces: Vec<String>,
    ) -> RasterBpePieceSequenceRef {
        let leaves = pieces
            .iter()
            .map(|piece| bpe_piece_leaf(piece))
            .collect::<Vec<_>>();
        RasterBpePieceSequenceRef::new(
            store
                .insert_artifact(
                    artifact_id(name),
                    RasterArtifactMetadata::bpe_pieces(pieces.len()),
                    leaves,
                )
                .expect("insert pieces"),
        )
        .expect("piece ref")
    }

    fn read_bpe_piece(
        store: &RasterArtifactStore,
        pieces_ref: &RasterBpePieceSequenceRef,
        piece_idx: usize,
    ) -> Result<String> {
        let read = store.read_leaf(pieces_ref.artifact_ref(), piece_idx)?;
        verify_artifact_read(pieces_ref.artifact_ref(), &read)?;
        decode_bpe_piece_leaf(read.payload())
    }

    fn materialize_bpe_pieces(
        store: &RasterArtifactStore,
        pieces_ref: &RasterBpePieceSequenceRef,
    ) -> Result<Vec<String>> {
        (0..pieces_ref.piece_count())
            .map(|piece_idx| read_bpe_piece(store, pieces_ref, piece_idx))
            .collect()
    }

    #[test]
    fn writes_reads_and_verifies_generic_byte_artifact() {
        let mut store = RasterArtifactStore::new();
        let metadata =
            RasterArtifactMetadata::new("test_bytes", "raster-test-bytes-v1", 2, Vec::new())
                .expect("metadata");
        let mut builder = store
            .start_builder(artifact_id("bytes"), metadata)
            .expect("builder");

        store
            .append_leaf(&mut builder, 0, b"a".to_vec())
            .expect("first leaf");
        store
            .append_leaf(&mut builder, 1, b"b".to_vec())
            .expect("second leaf");
        let artifact_ref = store.finalize_builder(builder).expect("finalize");

        assert_eq!(artifact_ref.kind(), "test_bytes");
        assert_eq!(artifact_ref.metadata().leaf_count(), 2);
        let read = store.read_leaf(&artifact_ref, 1).expect("read");
        assert_eq!(read.payload(), b"b");
        verify_artifact_read(&artifact_ref, &read).expect("read should verify");
    }

    #[test]
    fn open_builder_finalizes_with_actual_leaf_count() {
        let mut store = RasterArtifactStore::new();
        let metadata =
            RasterArtifactMetadata::open("test_stream", "raster-test-stream-v1", Vec::new())
                .expect("metadata");
        let mut builder = store
            .start_builder(artifact_id("stream"), metadata)
            .expect("builder");

        store
            .append_leaf(&mut builder, 0, b"a".to_vec())
            .expect("first leaf");
        store
            .append_leaf(&mut builder, 1, b"b".to_vec())
            .expect("second leaf");
        let artifact_ref = store.finalize_builder(builder).expect("finalize");

        assert_eq!(artifact_ref.metadata().leaf_count(), 2);
        assert_eq!(
            store.read_leaf(&artifact_ref, 1).expect("read").payload(),
            b"b"
        );
    }

    #[test]
    fn writes_reads_and_verifies_token_id_artifact() {
        let mut store = RasterArtifactStore::new();
        let mut builder = start_token_builder(&mut store, "tokens", 2);

        append_token_id(&mut store, &mut builder, 0, 17).expect("first token");
        append_token_id(&mut store, &mut builder, 1, 23).expect("second token");
        let token_ref = finalize_token_builder(&mut store, builder);

        assert_eq!(token_ref.artifact_ref().kind(), TOKEN_ID_ARTIFACT_KIND);
        assert_eq!(token_ref.token_count(), 2);
        assert_eq!(read_token_id(&store, &token_ref, 1).expect("read"), 23);
        assert_eq!(
            materialize_token_ids(&store, &token_ref).expect("materialize"),
            vec![17, 23]
        );
    }

    #[test]
    fn writes_reads_and_verifies_bpe_piece_artifact() {
        let mut store = RasterArtifactStore::new();
        let pieces_ref = insert_bpe_pieces(
            &mut store,
            "pieces",
            vec!["a".to_string(), "é".to_string(), "<0xC3>".to_string()],
        );

        assert_eq!(pieces_ref.artifact_ref().kind(), BPE_PIECE_ARTIFACT_KIND);
        assert_eq!(pieces_ref.piece_count(), 3);
        assert_eq!(read_bpe_piece(&store, &pieces_ref, 1).expect("piece"), "é");
        let left = read_bpe_piece(&store, &pieces_ref, 1).expect("left");
        let right = read_bpe_piece(&store, &pieces_ref, 2).expect("right");
        assert_eq!((left, right), ("é".to_string(), "<0xC3>".to_string()));
        assert_eq!(
            materialize_bpe_pieces(&store, &pieces_ref).expect("pieces"),
            vec!["a", "é", "<0xC3>"]
        );
    }

    #[test]
    fn writes_reads_and_verifies_activation_row_artifact() {
        let mut store = RasterArtifactStore::new();
        let mut builder = store
            .start_activation_sequence_builder(artifact_id("activations"), 1, 2)
            .expect("builder");
        let row = RasterActivationRow::from_acts(vec![Act::from_num(1.0), Act::from_num(-0.5)]);

        store
            .append_activation_row(&mut builder, 0, &row)
            .expect("activation row");
        let activation_ref = store.finalize_builder(builder).expect("finalize");

        assert_eq!(activation_ref.kind(), ACTIVATION_ROW_ARTIFACT_KIND);
        let read = store.read_leaf(&activation_ref, 0).expect("read");
        assert_eq!(read.payload(), activation_row_leaf(&row));
        verify_artifact_read(&activation_ref, &read).expect("read should verify");
    }

    #[test]
    fn empty_artifact_finalizes_to_empty_root_and_rejects_reads() {
        let mut store = RasterArtifactStore::new();
        let builder = start_token_builder(&mut store, "empty", 0);
        let token_ref = finalize_token_builder(&mut store, builder);

        assert_eq!(
            token_ref.root(),
            merkle_root(TOKEN_ID_ARTIFACT_DOMAIN.as_bytes(), &[])
        );
        assert!(store.read_leaf(token_ref.artifact_ref(), 0).is_err());
    }

    #[test]
    fn store_rejects_bad_refs_indexes_roots_and_domains() {
        let mut store = RasterArtifactStore::new();
        let mut builder = start_token_builder(&mut store, "tokens", 1);
        append_token_id(&mut store, &mut builder, 0, 17).expect("token");
        let token_ref = finalize_token_builder(&mut store, builder);

        let mut wrong_id_ref = token_ref.artifact_ref().clone();
        wrong_id_ref.id = artifact_id("missing");
        assert!(store.read_leaf(&wrong_id_ref, 0).is_err());

        let mut wrong_kind_ref = token_ref.artifact_ref().clone();
        wrong_kind_ref.metadata =
            RasterArtifactMetadata::activation_rows(1, 1).expect("activation metadata");
        assert!(store.read_leaf(&wrong_kind_ref, 0).is_err());

        let mut tampered_root_ref = token_ref.artifact_ref().clone();
        tampered_root_ref.root = "not-the-root".to_string();
        assert!(store.read_leaf(&tampered_root_ref, 0).is_err());

        assert!(store.read_leaf(token_ref.artifact_ref(), 1).is_err());
    }

    #[test]
    fn builders_reject_invalid_write_and_finalize_sequences() {
        let mut store = RasterArtifactStore::new();
        let mut builder = start_token_builder(&mut store, "tokens", 1);

        assert!(append_token_id(&mut store, &mut builder, 1, 17).is_err());
        append_token_id(&mut store, &mut builder, 0, 17).expect("token");
        assert!(append_token_id(&mut store, &mut builder, 0, 17).is_err());
        assert!(append_token_id(&mut store, &mut builder, 1, 23).is_err());

        let mut tampered = builder.clone();
        tampered.running_root = "tampered".to_string();
        assert!(store.finalize_builder(tampered).is_err());

        store.finalize_builder(builder).expect("finalize");

        let incomplete = start_token_builder(&mut store, "incomplete", 2);
        assert!(store.finalize_builder(incomplete).is_err());
    }

    #[test]
    fn bpe_piece_reads_fail_closed_for_bad_payloads_and_pairs() {
        let mut store = RasterArtifactStore::new();
        let pieces_ref = insert_bpe_pieces(&mut store, "pieces", vec!["a".to_string()]);
        assert!(read_bpe_piece(&store, &pieces_ref, 1).is_err());

        let bad_utf8_ref = store
            .insert_artifact(
                artifact_id("bad-utf8"),
                RasterArtifactMetadata::bpe_pieces(1),
                vec![{
                    let mut payload = Vec::new();
                    payload.extend_from_slice(&1_u64.to_le_bytes());
                    payload.push(0xff);
                    payload
                }],
            )
            .expect("bad utf8 artifact");
        let bad_utf8_ref = RasterBpePieceSequenceRef::new(bad_utf8_ref).expect("typed ref");
        assert!(read_bpe_piece(&store, &bad_utf8_ref, 0).is_err());

        let bad_length_ref = store
            .insert_artifact(
                artifact_id("bad-length"),
                RasterArtifactMetadata::bpe_pieces(1),
                vec![{
                    let mut payload = Vec::new();
                    payload.extend_from_slice(&2_u64.to_le_bytes());
                    payload.push(b'a');
                    payload
                }],
            )
            .expect("bad length artifact");
        let bad_length_ref = RasterBpePieceSequenceRef::new(bad_length_ref).expect("typed ref");
        assert!(read_bpe_piece(&store, &bad_length_ref, 0).is_err());
    }

    #[test]
    fn token_proof_does_not_verify_against_activation_domain() {
        let mut store = RasterArtifactStore::new();
        let mut token_builder = start_token_builder(&mut store, "tokens", 1);
        append_token_id(&mut store, &mut token_builder, 0, 17).expect("token");
        let token_ref = finalize_token_builder(&mut store, token_builder);
        let token_read = store
            .read_leaf(token_ref.artifact_ref(), 0)
            .expect("token read");

        let mut activation_builder = store
            .start_activation_sequence_builder(artifact_id("activations"), 1, 1)
            .expect("activation builder");
        let row = RasterActivationRow::from_acts(vec![Act::from_num(1.0)]);
        store
            .append_activation_row(&mut activation_builder, 0, &row)
            .expect("activation");
        let activation_ref = store
            .finalize_builder(activation_builder)
            .expect("activation ref");

        assert!(verify_artifact_read(&activation_ref, &token_read).is_err());
    }
}
