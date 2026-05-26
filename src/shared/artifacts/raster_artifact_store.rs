use std::{cell::RefCell, collections::HashMap};

use anyhow::{anyhow, bail, Result};
use serde::{de::DeserializeOwned, Serialize};

use crate::shared::artifacts::authenticated_selection::{
    AuthenticatedSelector, VerifiedSelectedPayload,
};
use crate::shared::artifacts::integrity_mode::raster_integrity_is_unchecked;
use crate::shared::artifacts::merkle::{
    merkle_proof, merkle_root, verify_merkle_proof, MerkleProof,
};
use crate::shared::raster_kernels::transformer::RasterActivationRow;

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub struct RasterArtifactStoreRoots {
    pub artifacts: Vec<RasterArtifactRootEntry>,
    pub builders: Vec<RasterArtifactBuilderRootEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterRoutineOutput<T> {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub refs: T,
}

impl<T> RasterRoutineOutput<T> {
    pub fn new(artifact_store_roots: RasterArtifactStoreRoots, refs: T) -> Self {
        Self {
            artifact_store_roots,
            refs,
        }
    }

    pub fn into_parts(self) -> (RasterArtifactStoreRoots, T) {
        (self.artifact_store_roots, self.refs)
    }
}

impl RasterArtifactStoreRoots {
    pub fn artifact_root_for_source_name(&self, source_name: &str) -> Result<&str> {
        Ok(self.artifact_entry_for_source_name(source_name)?.root())
    }

    pub fn artifact_entry_for_root(&self, root: &str) -> Result<&RasterArtifactRootEntry> {
        let mut matches = self.artifacts.iter().filter(|entry| entry.root() == root);
        let Some(entry) = matches.next() else {
            bail!("raster artifact root {root} is not present in the store roots snapshot");
        };
        if matches.next().is_some() {
            bail!("raster artifact root {root} matches multiple snapshot artifacts");
        }
        Ok(entry)
    }

    pub fn artifact_entry_for_source_name(
        &self,
        source_name: &str,
    ) -> Result<&RasterArtifactRootEntry> {
        let mut matches = self
            .artifacts
            .iter()
            .filter(|entry| entry.id().source_name() == source_name);
        let Some(entry) = matches.next() else {
            bail!("raster artifact id {source_name} is not present in the store roots snapshot");
        };
        if matches.next().is_some() {
            bail!("raster artifact id {source_name} matches multiple snapshot artifacts");
        }
        Ok(entry)
    }

    pub fn builder_root_for_source_name(&self, source_name: &str) -> Result<&str> {
        Ok(self
            .builder_entry_for_source_name(source_name)?
            .running_root())
    }

    pub fn builder_entry_for_root(&self, root: &str) -> Result<&RasterArtifactBuilderRootEntry> {
        let mut matches = self
            .builders
            .iter()
            .filter(|entry| entry.running_root() == root);
        let Some(entry) = matches.next() else {
            bail!("raster artifact builder root {root} is not present in the store roots snapshot");
        };
        if matches.next().is_some() {
            bail!("raster artifact builder root {root} matches multiple snapshot builders");
        }
        Ok(entry)
    }

    pub fn builder_entry_for_source_name(
        &self,
        source_name: &str,
    ) -> Result<&RasterArtifactBuilderRootEntry> {
        let mut matches = self
            .builders
            .iter()
            .filter(|entry| entry.id().source_name() == source_name);
        let Some(entry) = matches.next() else {
            bail!(
                "raster artifact builder id {source_name} is not present in the store roots snapshot"
            );
        };
        if matches.next().is_some() {
            bail!("raster artifact builder id {source_name} matches multiple snapshot builders");
        }
        Ok(entry)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterArtifactRootEntry {
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
    root: String,
}

impl RasterArtifactRootEntry {
    pub fn id(&self) -> &RasterArtifactId {
        &self.id
    }

    pub fn metadata(&self) -> &RasterArtifactMetadata {
        &self.metadata
    }

    pub fn root(&self) -> &str {
        &self.root
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterArtifactBuilderRootEntry {
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
    leaves_written: usize,
    running_root: String,
}

impl RasterArtifactBuilderRootEntry {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedArtifactRead {
    commitment: String,
    leaf_idx: usize,
    payload: Vec<u8>,
}

impl VerifiedArtifactRead {
    pub fn from_selected_artifact(selected: VerifiedSelectedPayload) -> Result<Self> {
        let AuthenticatedSelector::ArtifactLeaf { leaf_idx } = selected.selector() else {
            bail!("verified selected payload is not a raster artifact leaf");
        };
        Ok(Self {
            commitment: selected.commitment().to_string(),
            leaf_idx: *leaf_idx,
            payload: selected.bytes().to_vec(),
        })
    }

    pub fn commitment(&self) -> &str {
        &self.commitment
    }

    pub fn leaf_idx(&self) -> usize {
        self.leaf_idx
    }

    pub fn bytes(&self) -> &[u8] {
        &self.payload
    }

    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        postcard::from_bytes(self.bytes()).map_err(|error| {
            anyhow!(
                "failed to deserialize raster artifact leaf from postcard bytes: {}",
                error
            )
        })
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterSelectedTokenRef {
    token_ids_ref: RasterTokenIdSequenceRef,
}

impl RasterSelectedTokenRef {
    pub fn new(token_ids_ref: RasterTokenIdSequenceRef) -> Result<Self> {
        if token_ids_ref.token_count() != 1 {
            bail!(
                "raster selected-token ref requires exactly one token, got {}",
                token_ids_ref.token_count()
            );
        }
        Ok(Self { token_ids_ref })
    }

    pub fn token_ids_ref(&self) -> &RasterTokenIdSequenceRef {
        &self.token_ids_ref
    }

    pub fn id(&self) -> &RasterArtifactId {
        self.token_ids_ref.id()
    }

    pub fn source_name(&self) -> &str {
        self.token_ids_ref.id().source_name()
    }

    pub fn root(&self) -> &str {
        self.token_ids_ref.root()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterActivationSequenceArtifactRef {
    artifact_ref: RasterArtifactRef,
}

impl RasterActivationSequenceArtifactRef {
    pub fn new(artifact_ref: RasterArtifactRef) -> Result<Self> {
        artifact_ref
            .metadata()
            .ensure_kind(ACTIVATION_ROW_ARTIFACT_KIND)?;
        artifact_ref.metadata().ensure_finalized()?;
        artifact_ref
            .metadata()
            .shape_value(SHAPE_WIDTH)
            .ok_or_else(|| anyhow!("raster activation artifact metadata missing width"))?;
        Ok(Self { artifact_ref })
    }

    pub fn artifact_ref(&self) -> &RasterArtifactRef {
        &self.artifact_ref
    }

    pub fn id(&self) -> &RasterArtifactId {
        self.artifact_ref.id()
    }

    pub fn row_count(&self) -> usize {
        self.artifact_ref.metadata().leaf_count()
    }

    pub fn width(&self) -> usize {
        self.artifact_ref
            .metadata()
            .shape_value(SHAPE_WIDTH)
            .expect("activation sequence artifact refs validate width metadata")
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

    pub fn roots_snapshot(&self) -> RasterArtifactStoreRoots {
        let mut artifacts = self
            .artifacts
            .iter()
            .map(|(id, artifact)| RasterArtifactRootEntry {
                id: id.clone(),
                metadata: artifact.metadata.clone(),
                root: artifact_root_for_id(id, &artifact.metadata, &artifact.leaves),
            })
            .collect::<Vec<_>>();
        artifacts.sort_by(|left, right| left.id.source_name().cmp(right.id.source_name()));

        let mut builders = self
            .builders
            .iter()
            .map(|(id, builder)| RasterArtifactBuilderRootEntry {
                id: id.clone(),
                metadata: builder.metadata.clone(),
                leaves_written: builder.leaves.len(),
                running_root: builder_root_for_id(id, &builder.metadata, &builder.leaves),
            })
            .collect::<Vec<_>>();
        builders.sort_by(|left, right| left.id.source_name().cmp(right.id.source_name()));

        RasterArtifactStoreRoots {
            artifacts,
            builders,
        }
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

    pub fn append_leaf_by_builder_root(
        &mut self,
        builder_root: &str,
        leaf_idx: usize,
        payload: Vec<u8>,
    ) -> Result<String> {
        let mut builder_ref = self.builder_ref_for_root(builder_root)?;
        self.append_leaf(&mut builder_ref, leaf_idx, payload)?;
        Ok(builder_ref.running_root().to_string())
    }

    pub fn append_leaf_by_builder_root_with_roots(
        &mut self,
        roots: &RasterArtifactStoreRoots,
        builder_root: &str,
        leaf_idx: usize,
        payload: Vec<u8>,
    ) -> Result<(RasterArtifactStoreRoots, String)> {
        self.ensure_builder_root_in_snapshot(roots, builder_root)?;
        let running_root = self.append_leaf_by_builder_root(builder_root, leaf_idx, payload)?;
        Ok((self.roots_snapshot(), running_root))
    }

    pub fn append_leaf_by_builder_source_name_with_roots(
        &mut self,
        roots: &RasterArtifactStoreRoots,
        source_name: &str,
        leaf_idx: usize,
        payload: Vec<u8>,
    ) -> Result<(RasterArtifactStoreRoots, String)> {
        let entry = roots.builder_entry_for_source_name(source_name)?;
        let mut builder_ref = self.builder_ref_for_source_name(source_name)?;
        if entry.id() != builder_ref.id()
            || entry.metadata() != builder_ref.metadata()
            || entry.leaves_written() != builder_ref.leaves_written()
            || entry.running_root() != builder_ref.running_root()
        {
            bail!("raster artifact builder {source_name} snapshot metadata mismatch");
        }
        self.append_leaf(&mut builder_ref, leaf_idx, payload)?;
        Ok((
            self.roots_snapshot(),
            builder_ref.running_root().to_string(),
        ))
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

    pub fn finalize_builder_by_root(&mut self, builder_root: &str) -> Result<RasterArtifactRef> {
        let builder_ref = self.builder_ref_for_root(builder_root)?;
        self.finalize_builder(builder_ref)
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

    pub fn start_builder_with_roots(
        &mut self,
        roots: &RasterArtifactStoreRoots,
        id: RasterArtifactId,
        metadata: RasterArtifactMetadata,
    ) -> Result<(RasterArtifactStoreRoots, RasterArtifactBuilderRef)> {
        self.ensure_roots_snapshot_matches(roots)?;
        let builder_ref = self.start_builder(id, metadata)?;
        Ok((self.roots_snapshot(), builder_ref))
    }

    pub fn insert_artifact_with_roots(
        &mut self,
        roots: &RasterArtifactStoreRoots,
        id: RasterArtifactId,
        metadata: RasterArtifactMetadata,
        leaves: Vec<Vec<u8>>,
    ) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
        self.ensure_roots_snapshot_matches(roots)?;
        let artifact_ref = self.insert_artifact(id, metadata, leaves)?;
        Ok((self.roots_snapshot(), artifact_ref))
    }

    pub fn finalize_builder_with_roots(
        &mut self,
        roots: &RasterArtifactStoreRoots,
        builder_ref: RasterArtifactBuilderRef,
    ) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
        self.ensure_builder_root_in_snapshot(roots, builder_ref.running_root())?;
        let artifact_ref = self.finalize_builder(builder_ref)?;
        Ok((self.roots_snapshot(), artifact_ref))
    }

    pub fn finalize_builder_by_root_with_roots(
        &mut self,
        roots: &RasterArtifactStoreRoots,
        builder_root: &str,
    ) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
        self.ensure_builder_root_in_snapshot(roots, builder_root)?;
        let artifact_ref = self.finalize_builder_by_root(builder_root)?;
        Ok((self.roots_snapshot(), artifact_ref))
    }

    pub fn finalize_builder_by_source_name_with_roots(
        &mut self,
        roots: &RasterArtifactStoreRoots,
        source_name: &str,
    ) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
        let entry = roots.builder_entry_for_source_name(source_name)?;
        let builder_ref = self.builder_ref_for_source_name(source_name)?;
        if entry.id() != builder_ref.id()
            || entry.metadata() != builder_ref.metadata()
            || entry.leaves_written() != builder_ref.leaves_written()
            || entry.running_root() != builder_ref.running_root()
        {
            bail!("raster artifact builder {source_name} snapshot metadata mismatch");
        }
        let artifact_ref = self.finalize_builder(builder_ref)?;
        Ok((self.roots_snapshot(), artifact_ref))
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
        let proof = artifact_read_proof(&artifact.metadata, &artifact.leaves, leaf_idx)?;
        Ok(RasterArtifactRead {
            leaf_idx,
            payload,
            proof,
        })
    }

    pub fn read_verified_leaf(
        &self,
        artifact_ref: &RasterArtifactRef,
        leaf_idx: usize,
    ) -> Result<VerifiedArtifactRead> {
        VerifiedArtifactRead::from_selected_artifact(
            self.read_authenticated_leaf(artifact_ref, leaf_idx)?,
        )
    }

    pub fn read_authenticated_leaf(
        &self,
        artifact_ref: &RasterArtifactRef,
        leaf_idx: usize,
    ) -> Result<VerifiedSelectedPayload> {
        let read = self.read_leaf(artifact_ref, leaf_idx)?;
        verify_artifact_read(artifact_ref, &read)?;
        Ok(VerifiedSelectedPayload::artifact_leaf(
            artifact_ref.id().source_name(),
            artifact_ref.root(),
            read.leaf_idx,
            read.payload,
            read.proof,
        ))
    }

    pub fn artifact_ref_for_root(&self, root: &str) -> Result<RasterArtifactRef> {
        let mut matches = self.artifacts.iter().filter_map(|(id, artifact)| {
            let actual_root = artifact_root_for_id(id, &artifact.metadata, &artifact.leaves);
            (actual_root == root).then_some((id, artifact))
        });
        let Some((id, artifact)) = matches.next() else {
            bail!("raster artifact root {root} is not registered");
        };
        if matches.next().is_some() {
            bail!("raster artifact root {root} matches multiple registered artifacts");
        }
        Ok(RasterArtifactRef {
            id: id.clone(),
            metadata: artifact.metadata.clone(),
            root: root.to_string(),
        })
    }

    pub fn artifact_ref_for_root_any(&self, root: &str) -> Result<RasterArtifactRef> {
        let Some((id, artifact)) = self.artifacts.iter().find(|(id, artifact)| {
            artifact_root_for_id(id, &artifact.metadata, &artifact.leaves) == root
        }) else {
            bail!("raster artifact root {root} is not registered");
        };
        Ok(RasterArtifactRef {
            id: id.clone(),
            metadata: artifact.metadata.clone(),
            root: root.to_string(),
        })
    }

    fn builder_ref_for_root(&self, root: &str) -> Result<RasterArtifactBuilderRef> {
        let mut matches = self.builders.iter().filter_map(|(id, state)| {
            let running_root = builder_root_for_id(id, &state.metadata, &state.leaves);
            (running_root == root).then_some((id, state, running_root))
        });
        let Some((id, state, running_root)) = matches.next() else {
            bail!("raster artifact builder root {root} is not registered");
        };
        if matches.next().is_some() {
            bail!("raster artifact builder root {root} matches multiple registered builders");
        }
        Ok(RasterArtifactBuilderRef {
            id: id.clone(),
            metadata: state.metadata.clone(),
            leaves_written: state.leaves.len(),
            running_root,
        })
    }

    fn builder_ref_for_source_name(&self, source_name: &str) -> Result<RasterArtifactBuilderRef> {
        let id = RasterArtifactId::new(source_name)?;
        let state = self
            .builders
            .get(&id)
            .ok_or_else(|| anyhow!("raster artifact builder {source_name} is not registered"))?;
        Ok(artifact_builder_ref(id, state))
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

    fn ensure_builder_root_in_snapshot(
        &self,
        roots: &RasterArtifactStoreRoots,
        builder_root: &str,
    ) -> Result<()> {
        let entry = roots.builder_entry_for_root(builder_root)?;
        let builder_ref = self.builder_ref_for_root(builder_root)?;
        if entry.id() != builder_ref.id()
            || entry.metadata() != builder_ref.metadata()
            || entry.leaves_written() != builder_ref.leaves_written()
        {
            bail!("raster artifact builder root {builder_root} snapshot metadata mismatch");
        }
        Ok(())
    }

    fn ensure_roots_snapshot_matches(&self, roots: &RasterArtifactStoreRoots) -> Result<()> {
        if self.roots_snapshot() != *roots {
            bail!("raster artifact store roots snapshot does not match the current store state");
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
        let actual_root =
            artifact_root_for_id(artifact_ref.id(), &artifact.metadata, &artifact.leaves);
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

pub fn artifact_store_roots_snapshot() -> RasterArtifactStoreRoots {
    ARTIFACT_STORE.with(|store_ref| store_ref.borrow().roots_snapshot())
}

pub fn start_builder(
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
) -> Result<RasterArtifactBuilderRef> {
    with_artifact_store(|store| store.start_builder(id, metadata))
}

pub fn start_builder_with_roots(
    roots: &RasterArtifactStoreRoots,
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
) -> Result<(RasterArtifactStoreRoots, RasterArtifactBuilderRef)> {
    with_artifact_store(|store| store.start_builder_with_roots(roots, id, metadata))
}

pub fn insert_artifact(
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
    leaves: Vec<Vec<u8>>,
) -> Result<RasterArtifactRef> {
    with_artifact_store(|store| store.insert_artifact(id, metadata, leaves))
}

pub fn insert_artifact_with_roots(
    roots: &RasterArtifactStoreRoots,
    id: RasterArtifactId,
    metadata: RasterArtifactMetadata,
    leaves: Vec<Vec<u8>>,
) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
    with_artifact_store(|store| store.insert_artifact_with_roots(roots, id, metadata, leaves))
}

pub fn append_leaf(
    builder_ref: &mut RasterArtifactBuilderRef,
    leaf_idx: usize,
    payload: Vec<u8>,
) -> Result<()> {
    with_artifact_store(|store| store.append_leaf(builder_ref, leaf_idx, payload))
}

pub fn append_leaf_by_builder_root(
    builder_root: &str,
    leaf_idx: usize,
    payload: Vec<u8>,
) -> Result<String> {
    with_artifact_store(|store| store.append_leaf_by_builder_root(builder_root, leaf_idx, payload))
}

pub fn append_leaf_by_builder_root_with_roots(
    roots: &RasterArtifactStoreRoots,
    builder_root: &str,
    leaf_idx: usize,
    payload: Vec<u8>,
) -> Result<(RasterArtifactStoreRoots, String)> {
    with_artifact_store(|store| {
        store.append_leaf_by_builder_root_with_roots(roots, builder_root, leaf_idx, payload)
    })
}

pub fn append_leaf_by_builder_source_name_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: &str,
    leaf_idx: usize,
    payload: Vec<u8>,
) -> Result<(RasterArtifactStoreRoots, String)> {
    with_artifact_store(|store| {
        store.append_leaf_by_builder_source_name_with_roots(roots, source_name, leaf_idx, payload)
    })
}

pub fn finalize_builder(builder_ref: RasterArtifactBuilderRef) -> Result<RasterArtifactRef> {
    with_artifact_store(|store| store.finalize_builder(builder_ref))
}

pub fn finalize_builder_with_roots(
    roots: &RasterArtifactStoreRoots,
    builder_ref: RasterArtifactBuilderRef,
) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
    with_artifact_store(|store| store.finalize_builder_with_roots(roots, builder_ref))
}

pub fn finalize_builder_by_root(builder_root: &str) -> Result<RasterArtifactRef> {
    with_artifact_store(|store| store.finalize_builder_by_root(builder_root))
}

pub fn finalize_builder_by_root_with_roots(
    roots: &RasterArtifactStoreRoots,
    builder_root: &str,
) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
    with_artifact_store(|store| store.finalize_builder_by_root_with_roots(roots, builder_root))
}

pub fn finalize_builder_by_source_name_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: &str,
) -> Result<(RasterArtifactStoreRoots, RasterArtifactRef)> {
    with_artifact_store(|store| {
        store.finalize_builder_by_source_name_with_roots(roots, source_name)
    })
}

pub fn read_leaf(artifact_ref: &RasterArtifactRef, leaf_idx: usize) -> Result<RasterArtifactRead> {
    read_artifact_store(|store| store.read_leaf(artifact_ref, leaf_idx))
}

pub fn read_verified_leaf(
    artifact_ref: &RasterArtifactRef,
    leaf_idx: usize,
) -> Result<VerifiedArtifactRead> {
    read_artifact_store(|store| store.read_verified_leaf(artifact_ref, leaf_idx))
}

pub fn read_authenticated_leaf(
    artifact_ref: &RasterArtifactRef,
    leaf_idx: usize,
) -> Result<VerifiedSelectedPayload> {
    read_artifact_store(|store| store.read_authenticated_leaf(artifact_ref, leaf_idx))
}

pub fn read_verified_leaf_from_roots(
    roots: &RasterArtifactStoreRoots,
    artifact_ref: &RasterArtifactRef,
    leaf_idx: usize,
) -> Result<VerifiedArtifactRead> {
    VerifiedArtifactRead::from_selected_artifact(read_authenticated_leaf_from_roots(
        roots,
        artifact_ref,
        leaf_idx,
    )?)
}

pub fn read_authenticated_leaf_from_roots(
    roots: &RasterArtifactStoreRoots,
    artifact_ref: &RasterArtifactRef,
    leaf_idx: usize,
) -> Result<VerifiedSelectedPayload> {
    let entry = roots.artifact_entry_for_source_name(artifact_ref.id().source_name())?;
    if entry.root() != artifact_ref.root() {
        bail!(
            "raster artifact root mismatch for {}: snapshot has {}, ref has {}",
            artifact_ref.id().source_name(),
            entry.root(),
            artifact_ref.root()
        );
    }
    read_authenticated_leaf(artifact_ref, leaf_idx)
}

pub fn read_verified_leaf_by_root_from_roots(
    roots: &RasterArtifactStoreRoots,
    artifact_root: &str,
    leaf_idx: usize,
) -> Result<VerifiedArtifactRead> {
    VerifiedArtifactRead::from_selected_artifact(read_authenticated_leaf_by_root_from_roots(
        roots,
        artifact_root,
        leaf_idx,
    )?)
}

pub fn read_authenticated_leaf_by_root_from_roots(
    roots: &RasterArtifactStoreRoots,
    artifact_root: &str,
    leaf_idx: usize,
) -> Result<VerifiedSelectedPayload> {
    roots.artifact_entry_for_root(artifact_root)?;
    let artifact_ref = artifact_ref_for_root_any(artifact_root)?;
    read_authenticated_leaf(&artifact_ref, leaf_idx)
}

pub fn read_authenticated_leaf_by_present_root_from_roots(
    roots: &RasterArtifactStoreRoots,
    artifact_root: &str,
    leaf_idx: usize,
) -> Result<VerifiedSelectedPayload> {
    if !roots
        .artifacts
        .iter()
        .any(|entry| entry.root() == artifact_root)
    {
        bail!("raster artifact root {artifact_root} is not present in the store roots snapshot");
    }
    let artifact_ref = artifact_ref_for_root_any(artifact_root)?;
    read_authenticated_leaf(&artifact_ref, leaf_idx)
}

pub fn artifact_ref_for_root(root: &str) -> Result<RasterArtifactRef> {
    read_artifact_store(|store| store.artifact_ref_for_root(root))
}

pub fn artifact_ref_for_root_any(root: &str) -> Result<RasterArtifactRef> {
    read_artifact_store(|store| store.artifact_ref_for_root_any(root))
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
    if raster_integrity_is_unchecked() {
        return Ok(());
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
    postcard_leaf(&token_id, "token-id")
}

pub fn decode_token_id_leaf(payload: &[u8]) -> Result<u32> {
    decode_postcard_leaf(payload, "token-id")
}

pub fn token_ids_ref_for_root(
    roots: &RasterArtifactStoreRoots,
    token_ids_root: &str,
    token_count: usize,
) -> Result<RasterTokenIdSequenceRef> {
    roots.artifact_entry_for_root(token_ids_root)?;
    let token_ids_ref = RasterTokenIdSequenceRef::new(artifact_ref_for_root(token_ids_root)?)?;
    if token_ids_ref.token_count() != token_count {
        bail!(
            "token-id artifact {token_ids_root} contains {} tokens, expected {token_count}",
            token_ids_ref.token_count()
        );
    }
    Ok(token_ids_ref)
}

pub fn read_token_id_from_roots(
    roots: &RasterArtifactStoreRoots,
    token_ids_root: &str,
    token_count: usize,
    token_idx: usize,
) -> Result<u32> {
    if token_idx >= token_count {
        bail!("token-id index {token_idx} is out of range for {token_count} tokens");
    }
    token_ids_ref_for_root(roots, token_ids_root, token_count)?;
    read_authenticated_leaf_by_root_from_roots(roots, token_ids_root, token_idx)?.deserialize()
}

pub fn read_token_id_from_ref_roots(
    roots: &RasterArtifactStoreRoots,
    token_ids_ref: &RasterTokenIdSequenceRef,
    token_idx: usize,
) -> Result<u32> {
    if token_idx >= token_ids_ref.token_count() {
        bail!(
            "token-id index {token_idx} is out of range for {} tokens",
            token_ids_ref.token_count()
        );
    }
    let entry = roots.artifact_entry_for_source_name(token_ids_ref.id().source_name())?;
    if entry.root() != token_ids_ref.root() {
        bail!(
            "token-id artifact root mismatch for {}: snapshot has {}, ref has {}",
            token_ids_ref.id().source_name(),
            entry.root(),
            token_ids_ref.root()
        );
    }
    read_authenticated_leaf_from_roots(roots, token_ids_ref.artifact_ref(), token_idx)?
        .deserialize()
}

pub fn read_selected_token_from_roots(
    roots: &RasterArtifactStoreRoots,
    selected_token_ref: &RasterSelectedTokenRef,
) -> Result<u32> {
    read_token_id_from_ref_roots(roots, selected_token_ref.token_ids_ref(), 0)
}

pub fn activation_row_leaf(row: &RasterActivationRow) -> Vec<u8> {
    postcard_leaf(row, "activation row")
}

pub fn decode_activation_row_leaf(payload: &[u8]) -> Result<RasterActivationRow> {
    decode_postcard_leaf(payload, "activation row")
}

fn postcard_leaf<T: Serialize>(value: &T, label: &str) -> Vec<u8> {
    postcard::to_allocvec(value)
        .unwrap_or_else(|error| panic!("failed to serialize raster {label} leaf: {error}"))
}

fn decode_postcard_leaf<T: DeserializeOwned>(payload: &[u8], label: &str) -> Result<T> {
    postcard::from_bytes(payload)
        .map_err(|error| anyhow!("failed to deserialize raster {label} leaf: {error}"))
}

fn artifact_ref(id: RasterArtifactId, state: &ArtifactBuilderState) -> RasterArtifactRef {
    RasterArtifactRef {
        root: artifact_root_for_id(&id, &state.metadata, &state.leaves),
        id,
        metadata: state.metadata.clone(),
    }
}

fn artifact_builder_ref(
    id: RasterArtifactId,
    state: &ArtifactBuilderState,
) -> RasterArtifactBuilderRef {
    RasterArtifactBuilderRef {
        running_root: builder_root_for_id(&id, &state.metadata, &state.leaves),
        id,
        metadata: state.metadata.clone(),
        leaves_written: state.leaves.len(),
    }
}

fn update_builder_ref(builder_ref: &mut RasterArtifactBuilderRef, state: &ArtifactBuilderState) {
    *builder_ref = artifact_builder_ref(builder_ref.id.clone(), state);
}

fn artifact_root(metadata: &RasterArtifactMetadata, leaves: &[Vec<u8>]) -> String {
    merkle_root(metadata.domain_bytes(), leaves)
}

fn artifact_root_for_id(
    id: &RasterArtifactId,
    metadata: &RasterArtifactMetadata,
    leaves: &[Vec<u8>],
) -> String {
    if raster_integrity_is_unchecked() {
        unchecked_root("artifact", id)
    } else {
        artifact_root(metadata, leaves)
    }
}

fn builder_root_for_id(
    id: &RasterArtifactId,
    metadata: &RasterArtifactMetadata,
    leaves: &[Vec<u8>],
) -> String {
    if raster_integrity_is_unchecked() {
        unchecked_root("builder", id)
    } else {
        artifact_root(metadata, leaves)
    }
}

fn unchecked_root(kind: &str, id: &RasterArtifactId) -> String {
    format!("raster-unchecked-test:{kind}:{}", id.source_name())
}

fn artifact_read_proof(
    metadata: &RasterArtifactMetadata,
    leaves: &[Vec<u8>],
    leaf_idx: usize,
) -> Result<MerkleProof> {
    if raster_integrity_is_unchecked() {
        MerkleProof::new(leaves.len(), Vec::new())
    } else {
        merkle_proof(metadata.domain_bytes(), leaves, leaf_idx)
    }
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
    #[cfg(feature = "unchecked-raster-integrity")]
    use crate::shared::artifacts::integrity_mode::{
        with_raster_integrity_mode, RasterIntegrityMode,
    };
    use crate::shared::numerics::det_num::Act;

    fn artifact_id(name: &str) -> RasterArtifactId {
        RasterArtifactId::new(name).expect("artifact id")
    }

    fn bpe_piece_leaf(piece: &str) -> Vec<u8> {
        postcard_leaf(&piece, "BPE piece")
    }

    fn decode_bpe_piece_leaf(payload: &[u8]) -> Result<String> {
        decode_postcard_leaf(payload, "BPE piece")
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
        store
            .read_verified_leaf(token_ref.artifact_ref(), token_idx)?
            .deserialize()
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
        let read = store.read_verified_leaf(pieces_ref.artifact_ref(), piece_idx)?;
        decode_bpe_piece_leaf(read.bytes())
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
        let read = store
            .read_verified_leaf(&artifact_ref, 1)
            .expect("read should verify");
        assert_eq!(read.bytes(), b"b");
        assert_eq!(read.commitment(), artifact_ref.root());
    }

    #[test]
    fn authenticated_artifact_leaf_exposes_selector_bytes_and_proof() {
        let mut store = RasterArtifactStore::new();
        let artifact_ref = store
            .insert_artifact(
                artifact_id("authenticated-bytes"),
                RasterArtifactMetadata::new("test_bytes", "raster-test-bytes-v1", 2, Vec::new())
                    .expect("metadata"),
                vec![b"a".to_vec(), b"b".to_vec()],
            )
            .expect("insert artifact");

        let selected = store
            .read_authenticated_leaf(&artifact_ref, 1)
            .expect("authenticated read");

        assert_eq!(
            selected.source_kind(),
            &crate::shared::artifacts::authenticated_selection::AuthenticatedSourceKind::RasterArtifact
        );
        assert_eq!(
            selected.selector(),
            &crate::shared::artifacts::authenticated_selection::AuthenticatedSelector::ArtifactLeaf {
                leaf_idx: 1
            }
        );
        assert_eq!(selected.source_name(), "authenticated-bytes");
        assert_eq!(selected.commitment(), artifact_ref.root());
        assert_eq!(selected.bytes(), b"b");
        assert_eq!(selected.merkle_leaf_payload(), b"b");
        assert_eq!(selected.proof().leaf_count(), 2);
    }

    #[test]
    fn authenticated_artifact_leaf_from_roots_rejects_root_mismatch() {
        reset_artifact_store();
        let artifact_ref = insert_artifact(
            artifact_id("root-mismatch"),
            RasterArtifactMetadata::token_ids(1),
            vec![token_id_leaf(1)],
        )
        .expect("insert artifact");
        let roots = artifact_store_roots_snapshot();
        let mut bad_ref = artifact_ref.clone();
        bad_ref.root = "wrong-root".to_string();

        let error = read_authenticated_leaf_from_roots(&roots, &bad_ref, 0)
            .expect_err("root mismatch should fail");

        assert!(error.to_string().contains("root mismatch"));
    }

    #[test]
    fn authenticated_artifact_leaf_by_root_rejects_out_of_range_leaf() {
        reset_artifact_store();
        let artifact_ref = insert_artifact(
            artifact_id("root-out-of-range"),
            RasterArtifactMetadata::token_ids(1),
            vec![token_id_leaf(1)],
        )
        .expect("insert artifact");
        let roots = artifact_store_roots_snapshot();

        let error = read_authenticated_leaf_by_root_from_roots(&roots, artifact_ref.root(), 1)
            .expect_err("out of range read should fail");

        assert!(error.to_string().contains("out of range"));
    }

    #[test]
    fn authenticated_artifact_leaf_by_root_rejects_duplicate_snapshot_roots() {
        reset_artifact_store();
        let artifact_ref = insert_artifact(
            artifact_id("duplicate-root"),
            RasterArtifactMetadata::token_ids(1),
            vec![token_id_leaf(1)],
        )
        .expect("insert artifact");
        let mut roots = artifact_store_roots_snapshot();
        roots.artifacts.push(roots.artifacts[0].clone());

        let error = read_authenticated_leaf_by_root_from_roots(&roots, artifact_ref.root(), 0)
            .expect_err("duplicate root should fail");

        assert!(error
            .to_string()
            .contains("matches multiple snapshot artifacts"));
    }

    #[test]
    fn verified_artifact_read_deserializes_postcard_payload() {
        let mut store = RasterArtifactStore::new();
        let artifact_ref = store
            .insert_artifact(
                artifact_id("postcard-token"),
                RasterArtifactMetadata::token_ids(1),
                vec![token_id_leaf(99)],
            )
            .expect("insert token artifact");

        let read = store
            .read_verified_leaf(&artifact_ref, 0)
            .expect("verified read");

        assert_eq!(read.commitment(), artifact_ref.root());
        assert_eq!(read.leaf_idx(), 0);
        assert_eq!(read.deserialize::<u32>().expect("token decode"), 99);
    }

    #[test]
    fn resolves_artifact_ref_by_root() {
        let mut store = RasterArtifactStore::new();
        let mut builder = start_token_builder(&mut store, "tokens", 1);
        append_token_id(&mut store, &mut builder, 0, 17).expect("token");
        let token_ref = finalize_token_builder(&mut store, builder);

        let resolved = store
            .artifact_ref_for_root(token_ref.root())
            .expect("root should resolve");

        assert_eq!(resolved, *token_ref.artifact_ref());
    }

    #[test]
    fn appends_and_finalizes_builder_by_root() {
        let mut store = RasterArtifactStore::new();
        let builder = start_token_builder(&mut store, "tokens", 2);
        let builder_root = builder.running_root().to_string();

        let builder_root = store
            .append_leaf_by_builder_root(&builder_root, 0, token_id_leaf(17))
            .expect("first token append should update root");
        let builder_root = store
            .append_leaf_by_builder_root(&builder_root, 1, token_id_leaf(23))
            .expect("second token append should update root");
        let token_ref = RasterTokenIdSequenceRef::new(
            store
                .finalize_builder_by_root(&builder_root)
                .expect("builder root should finalize"),
        )
        .expect("typed token ref");

        assert_eq!(
            materialize_token_ids(&store, &token_ref).expect("token ids"),
            vec![17, 23]
        );
    }

    #[test]
    fn roots_snapshot_tracks_builder_updates_and_finalization() {
        let mut store = RasterArtifactStore::new();
        let roots = store.roots_snapshot();
        let (roots, builder) = store
            .start_builder_with_roots(
                &roots,
                artifact_id("tokens"),
                RasterArtifactMetadata::token_ids(2),
            )
            .expect("builder should start with roots");
        assert_eq!(roots.artifacts.len(), 0);
        assert_eq!(roots.builders.len(), 1);
        assert_eq!(roots.builders[0].running_root(), builder.running_root());

        let (roots, builder_root) = store
            .append_leaf_by_builder_root_with_roots(
                &roots,
                builder.running_root(),
                0,
                token_id_leaf(17),
            )
            .expect("first append should update roots");
        assert_eq!(roots.builders.len(), 1);
        assert_eq!(roots.builders[0].running_root(), builder_root);
        assert_eq!(roots.builders[0].leaves_written(), 1);

        let (roots, builder_root) = store
            .append_leaf_by_builder_root_with_roots(&roots, &builder_root, 1, token_id_leaf(23))
            .expect("second append should update roots");
        let (roots, token_ref) = store
            .finalize_builder_by_root_with_roots(&roots, &builder_root)
            .expect("finalization should update roots");

        assert_eq!(roots.builders.len(), 0);
        assert_eq!(roots.artifacts.len(), 1);
        assert_eq!(roots.artifacts[0].root(), token_ref.root());
        assert!(roots.artifact_entry_for_root(token_ref.root()).is_ok());
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn unchecked_mode_uses_synthetic_roots_without_rehashing_appends() {
        with_raster_integrity_mode(RasterIntegrityMode::UncheckedTestOnly, || {
            let mut store = RasterArtifactStore::new();
            let roots = store.roots_snapshot();
            let (roots, builder) = store
                .start_builder_with_roots(
                    &roots,
                    artifact_id("tokens"),
                    RasterArtifactMetadata::token_ids(2),
                )
                .expect("builder should start");
            let builder_root = builder.running_root().to_string();
            assert_eq!(builder_root, "raster-unchecked-test:builder:tokens");

            let (roots, next_builder_root) = store
                .append_leaf_by_builder_root_with_roots(&roots, &builder_root, 0, token_id_leaf(17))
                .expect("first token append should succeed");
            assert_eq!(next_builder_root, builder_root);
            assert_eq!(roots.builders[0].running_root(), builder_root);
            assert_eq!(roots.builders[0].leaves_written(), 1);

            let (roots, next_builder_root) = store
                .append_leaf_by_builder_root_with_roots(&roots, &builder_root, 1, token_id_leaf(23))
                .expect("second token append should succeed");
            assert_eq!(next_builder_root, builder_root);

            let (roots, token_ref) = store
                .finalize_builder_by_root_with_roots(&roots, &builder_root)
                .expect("builder should finalize");
            assert_eq!(token_ref.root(), "raster-unchecked-test:artifact:tokens");
            assert_eq!(roots.artifacts[0].root(), token_ref.root());
            let token_ref = RasterTokenIdSequenceRef::new(token_ref).expect("token ref");

            assert_eq!(
                materialize_token_ids(&store, &token_ref).expect("token ids"),
                vec![17, 23]
            );
        });
    }

    #[test]
    fn roots_aware_append_rejects_stale_builder_snapshot() {
        let mut store = RasterArtifactStore::new();
        let roots = store.roots_snapshot();
        let (roots, builder) = store
            .start_builder_with_roots(
                &roots,
                artifact_id("tokens"),
                RasterArtifactMetadata::token_ids(2),
            )
            .expect("builder should start with roots");
        let stale_roots = roots.clone();

        let (roots, builder_root) = store
            .append_leaf_by_builder_root_with_roots(
                &roots,
                builder.running_root(),
                0,
                token_id_leaf(17),
            )
            .expect("first append should update roots");
        let error = store
            .append_leaf_by_builder_root_with_roots(
                &stale_roots,
                &builder_root,
                1,
                token_id_leaf(23),
            )
            .expect_err("stale snapshot should be rejected");

        assert!(error.to_string().contains("snapshot"));
        assert_eq!(roots.builders[0].leaves_written(), 1);
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
    fn roots_aware_token_id_read_fails_closed() {
        reset_artifact_store();
        let roots = artifact_store_roots_snapshot();
        let (roots, builder) = start_builder_with_roots(
            &roots,
            artifact_id("roots.tokens"),
            RasterArtifactMetadata::token_ids(1),
        )
        .expect("builder should start");
        let (roots, builder_root) = append_leaf_by_builder_root_with_roots(
            &roots,
            builder.running_root(),
            0,
            token_id_leaf(42),
        )
        .expect("token should append");
        let (roots, token_ref) = finalize_builder_by_root_with_roots(&roots, &builder_root)
            .expect("token builder should finalize");

        assert_eq!(
            read_token_id_from_roots(&roots, token_ref.root(), 1, 0).expect("token should read"),
            42
        );
        assert!(read_token_id_from_roots(
            &RasterArtifactStoreRoots::default(),
            token_ref.root(),
            1,
            0
        )
        .expect_err("missing root should fail")
        .to_string()
        .contains("not present"));
        assert!(read_token_id_from_roots(&roots, token_ref.root(), 1, 1)
            .expect_err("out-of-range token should fail")
            .to_string()
            .contains("out of range"));
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
        let read = store
            .read_verified_leaf(&activation_ref, 0)
            .expect("read should verify");
        assert_eq!(read.bytes(), activation_row_leaf(&row));
        assert_eq!(
            read.deserialize::<RasterActivationRow>()
                .expect("activation row decode"),
            row
        );
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

        let bad_postcard_ref = store
            .insert_artifact(
                artifact_id("bad-postcard"),
                RasterArtifactMetadata::bpe_pieces(1),
                vec![vec![0xff]],
            )
            .expect("bad postcard artifact");
        let bad_postcard_ref = RasterBpePieceSequenceRef::new(bad_postcard_ref).expect("typed ref");
        assert!(read_bpe_piece(&store, &bad_postcard_ref, 0).is_err());

        let truncated_ref = store
            .insert_artifact(
                artifact_id("truncated-postcard"),
                RasterArtifactMetadata::bpe_pieces(1),
                vec![vec![2, b'a']],
            )
            .expect("truncated postcard artifact");
        let truncated_ref = RasterBpePieceSequenceRef::new(truncated_ref).expect("typed ref");
        assert!(read_bpe_piece(&store, &truncated_ref, 0).is_err());
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
