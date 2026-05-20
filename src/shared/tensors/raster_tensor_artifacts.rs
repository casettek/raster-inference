use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::merkle::merkle_root;
use crate::shared::artifacts::raster_artifact_store::{
    activation_row_leaf, decode_activation_row_leaf, RasterActivationSequenceArtifactRef,
    RasterArtifactId, RasterArtifactMetadata, RasterArtifactStoreRoots,
    ACTIVATION_ROW_ARTIFACT_DOMAIN,
};
use crate::shared::raster_kernels::transformer::{
    RasterActivationRow, RasterActivationSequence, RasterAttentionHeadSequence, RasterKvCache,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterTensorId {
    source_name: String,
}

impl RasterTensorId {
    pub fn new(source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        if source_name.is_empty() {
            bail!("raster tensor id requires a non-empty source name");
        }
        Ok(Self { source_name })
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum RasterTensorKind {
    ActivationSequence,
    AttentionHeads,
    KvCacheKeys,
    KvCacheValues,
    PartialOutput,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum RasterTensorShape {
    Sequence {
        row_count: usize,
        width: usize,
    },
    Heads {
        head_count: usize,
        sequence_len: usize,
        head_dim: usize,
    },
    KvCache {
        head_count: usize,
        current_len: usize,
        head_dim: usize,
    },
}

impl RasterTensorShape {
    pub fn sequence(row_count: usize, width: usize) -> Result<Self> {
        if row_count == 0 {
            bail!("raster sequence tensor requires at least one row");
        }
        if width == 0 {
            bail!("raster sequence tensor requires non-zero width");
        }
        Ok(Self::Sequence { row_count, width })
    }

    pub fn heads(head_count: usize, sequence_len: usize, head_dim: usize) -> Result<Self> {
        if head_count == 0 {
            bail!("raster heads tensor requires at least one head");
        }
        if sequence_len == 0 {
            bail!("raster heads tensor requires at least one token row");
        }
        if head_dim == 0 {
            bail!("raster heads tensor requires non-zero head dimension");
        }
        Ok(Self::Heads {
            head_count,
            sequence_len,
            head_dim,
        })
    }

    pub fn kv_cache(head_count: usize, current_len: usize, head_dim: usize) -> Result<Self> {
        if head_count == 0 {
            bail!("raster KV cache tensor requires at least one head");
        }
        if current_len == 0 {
            bail!("raster KV cache tensor requires at least one row");
        }
        if head_dim == 0 {
            bail!("raster KV cache tensor requires non-zero head dimension");
        }
        Ok(Self::KvCache {
            head_count,
            current_len,
            head_dim,
        })
    }

    pub fn heads_metadata(&self) -> Result<(usize, usize, usize)> {
        match *self {
            Self::Heads {
                head_count,
                sequence_len,
                head_dim,
            } => Ok((head_count, sequence_len, head_dim)),
            _ => bail!("raster tensor shape is not attention heads"),
        }
    }

    pub fn sequence_metadata(&self) -> Result<(usize, usize)> {
        match *self {
            Self::Sequence { row_count, width } => Ok((row_count, width)),
            _ => bail!("raster tensor shape is not an activation sequence"),
        }
    }

    pub fn kv_cache_metadata(&self) -> Result<(usize, usize, usize)> {
        match *self {
            Self::KvCache {
                head_count,
                current_len,
                head_dim,
            } => Ok((head_count, current_len, head_dim)),
            _ => bail!("raster tensor shape is not KV cache"),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterTensorRef {
    id: RasterTensorId,
    kind: RasterTensorKind,
    shape: RasterTensorShape,
    det_commitment: String,
}

impl RasterTensorRef {
    pub fn new(
        id: RasterTensorId,
        kind: RasterTensorKind,
        shape: RasterTensorShape,
        det_commitment: impl Into<String>,
    ) -> Result<Self> {
        ensure_kind_matches_shape(kind, &shape)?;
        let det_commitment = det_commitment.into();
        if det_commitment.is_empty() {
            bail!("raster tensor ref requires a non-empty deterministic commitment");
        }
        Ok(Self {
            id,
            kind,
            shape,
            det_commitment,
        })
    }

    pub fn id(&self) -> &RasterTensorId {
        &self.id
    }

    pub fn kind(&self) -> RasterTensorKind {
        self.kind
    }

    pub fn shape(&self) -> &RasterTensorShape {
        &self.shape
    }

    pub fn det_commitment(&self) -> &str {
        &self.det_commitment
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterActivationSequenceRef(RasterTensorRef);

impl RasterActivationSequenceRef {
    pub fn new(reference: RasterTensorRef) -> Result<Self> {
        if reference.kind != RasterTensorKind::ActivationSequence {
            bail!(
                "raster activation sequence ref received {:?} tensor",
                reference.kind
            );
        }
        if !matches!(reference.shape, RasterTensorShape::Sequence { .. }) {
            bail!("raster activation sequence ref requires sequence shape");
        }
        Ok(Self(reference))
    }

    pub fn tensor_ref(&self) -> &RasterTensorRef {
        &self.0
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterAttentionHeadsRef(RasterTensorRef);

impl RasterAttentionHeadsRef {
    pub fn new(reference: RasterTensorRef) -> Result<Self> {
        if reference.kind != RasterTensorKind::AttentionHeads {
            bail!(
                "raster attention heads ref received {:?} tensor",
                reference.kind
            );
        }
        if !matches!(reference.shape, RasterTensorShape::Heads { .. }) {
            bail!("raster attention heads ref requires heads shape");
        }
        Ok(Self(reference))
    }

    pub fn tensor_ref(&self) -> &RasterTensorRef {
        &self.0
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterKvCacheRef {
    keys: RasterTensorRef,
    values: RasterTensorRef,
    shape: RasterTensorShape,
    det_commitment: String,
}

impl RasterKvCacheRef {
    pub fn new(
        keys: RasterTensorRef,
        values: RasterTensorRef,
        det_commitment: impl Into<String>,
    ) -> Result<Self> {
        if keys.kind != RasterTensorKind::KvCacheKeys {
            bail!("raster KV cache keys ref received {:?} tensor", keys.kind);
        }
        if values.kind != RasterTensorKind::KvCacheValues {
            bail!(
                "raster KV cache values ref received {:?} tensor",
                values.kind
            );
        }
        if !matches!(keys.shape, RasterTensorShape::KvCache { .. }) {
            bail!("raster KV cache keys ref requires KV cache shape");
        }
        if keys.shape != values.shape {
            bail!("raster KV cache key/value shape mismatch");
        }
        let det_commitment = det_commitment.into();
        if det_commitment.is_empty() {
            bail!("raster KV cache ref requires a non-empty deterministic commitment");
        }
        Ok(Self {
            shape: keys.shape.clone(),
            keys,
            values,
            det_commitment,
        })
    }

    pub fn keys(&self) -> &RasterTensorRef {
        &self.keys
    }

    pub fn values(&self) -> &RasterTensorRef {
        &self.values
    }

    pub fn shape(&self) -> &RasterTensorShape {
        &self.shape
    }

    pub fn det_commitment(&self) -> &str {
        &self.det_commitment
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterSequenceRowRequest {
    pub tensor_ref: RasterActivationSequenceRef,
    pub row_idx: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterHeadRowRequest {
    pub tensor_ref: RasterAttentionHeadsRef,
    pub head_idx: usize,
    pub token_idx: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum RasterKvRowKind {
    Key,
    Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterKvRowRequest {
    pub cache_ref: RasterKvCacheRef,
    pub row_kind: RasterKvRowKind,
    pub head_idx: usize,
    pub token_idx: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterTensorBuilderRef {
    id: RasterTensorId,
    kind: RasterTensorKind,
    expected_shape: RasterTensorShape,
    rows_written: usize,
    running_commitment: String,
}

impl RasterTensorBuilderRef {
    pub fn id(&self) -> &RasterTensorId {
        &self.id
    }

    pub fn kind(&self) -> RasterTensorKind {
        self.kind
    }

    pub fn expected_shape(&self) -> &RasterTensorShape {
        &self.expected_shape
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    pub fn running_commitment(&self) -> &str {
        &self.running_commitment
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterProjectionOutputBuilderRef {
    builder_ref: RasterTensorBuilderRef,
    token_count: usize,
    projection_rows: usize,
    next_token_idx: usize,
    next_projection_row_idx: usize,
}

impl RasterProjectionOutputBuilderRef {
    pub fn next_token_idx(&self) -> usize {
        self.next_token_idx
    }

    pub fn next_projection_row_idx(&self) -> usize {
        self.next_projection_row_idx
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn projection_rows(&self) -> usize {
        self.projection_rows
    }

    pub fn builder_ref(&self) -> &RasterTensorBuilderRef {
        &self.builder_ref
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterKvCacheBuilderRef {
    keys: RasterTensorBuilderRef,
    values: RasterTensorBuilderRef,
    expected_shape: RasterTensorShape,
}

impl RasterKvCacheBuilderRef {
    pub fn keys(&self) -> &RasterTensorBuilderRef {
        &self.keys
    }

    pub fn keys_mut(&mut self) -> &mut RasterTensorBuilderRef {
        &mut self.keys
    }

    pub fn values(&self) -> &RasterTensorBuilderRef {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut RasterTensorBuilderRef {
        &mut self.values
    }

    pub fn expected_shape(&self) -> &RasterTensorShape {
        &self.expected_shape
    }
}

pub fn activation_sequence_ref_from_artifact(
    id: RasterTensorId,
    artifact_ref: RasterActivationSequenceArtifactRef,
) -> Result<RasterActivationSequenceRef> {
    let shape = RasterTensorShape::sequence(artifact_ref.row_count(), artifact_ref.width())?;
    RasterActivationSequenceRef::new(RasterTensorRef::new(
        id,
        RasterTensorKind::ActivationSequence,
        shape,
        artifact_ref.root(),
    )?)
}

pub fn attention_heads_ref_from_artifact(
    id: RasterTensorId,
    artifact_ref: RasterActivationSequenceArtifactRef,
    head_count: usize,
    sequence_len: usize,
    head_dim: usize,
) -> Result<RasterAttentionHeadsRef> {
    if artifact_ref.row_count() != head_count * sequence_len {
        bail!(
            "raster heads artifact has {} rows, expected {}",
            artifact_ref.row_count(),
            head_count * sequence_len
        );
    }
    if artifact_ref.width() != head_dim {
        bail!(
            "raster heads artifact width {}, expected {head_dim}",
            artifact_ref.width()
        );
    }
    RasterAttentionHeadsRef::new(RasterTensorRef::new(
        id,
        RasterTensorKind::AttentionHeads,
        RasterTensorShape::heads(head_count, sequence_len, head_dim)?,
        artifact_ref.root(),
    )?)
}

pub fn kv_cache_ref_from_artifacts(
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    keys_artifact_ref: RasterActivationSequenceArtifactRef,
    values_artifact_ref: RasterActivationSequenceArtifactRef,
    head_count: usize,
    current_len: usize,
    head_dim: usize,
) -> Result<RasterKvCacheRef> {
    let expected_rows = head_count * current_len;
    if keys_artifact_ref.row_count() != expected_rows {
        bail!(
            "raster KV keys artifact has {} rows, expected {expected_rows}",
            keys_artifact_ref.row_count()
        );
    }
    if values_artifact_ref.row_count() != expected_rows {
        bail!(
            "raster KV values artifact has {} rows, expected {expected_rows}",
            values_artifact_ref.row_count()
        );
    }
    if keys_artifact_ref.width() != head_dim {
        bail!(
            "raster KV keys artifact width {}, expected {head_dim}",
            keys_artifact_ref.width()
        );
    }
    if values_artifact_ref.width() != head_dim {
        bail!(
            "raster KV values artifact width {}, expected {head_dim}",
            values_artifact_ref.width()
        );
    }

    let shape = RasterTensorShape::kv_cache(head_count, current_len, head_dim)?;
    let keys_ref = RasterTensorRef::new(
        keys_id,
        RasterTensorKind::KvCacheKeys,
        shape.clone(),
        keys_artifact_ref.root(),
    )?;
    let values_ref = RasterTensorRef::new(
        values_id,
        RasterTensorKind::KvCacheValues,
        shape,
        values_artifact_ref.root(),
    )?;
    RasterKvCacheRef::new(
        keys_ref,
        values_ref,
        build_intermediate_kv_cache_roots_commitment(
            keys_artifact_ref.root(),
            values_artifact_ref.root(),
        ),
    )
}

pub fn start_sequence_builder_with_roots(
    roots: &RasterArtifactStoreRoots,
    id: RasterArtifactId,
    row_count: usize,
    width: usize,
) -> Result<RasterArtifactStoreRoots> {
    let (roots, _builder) = ArtifactIo::start_builder_with_roots(
        roots,
        id,
        RasterArtifactMetadata::activation_rows(row_count, width)?,
    )?;
    Ok(roots)
}

pub fn append_sequence_row_by_source_name_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: &str,
    row_idx: usize,
    row: RasterActivationRow,
) -> Result<RasterArtifactStoreRoots> {
    let (roots, _builder_root) = ArtifactIo::append_leaf_by_builder_source_name_with_roots(
        roots,
        source_name,
        row_idx,
        activation_row_leaf(&row),
    )?;
    Ok(roots)
}

pub fn finalize_sequence_builder_by_source_name_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: &str,
    id: RasterTensorId,
) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
    let (roots, artifact_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(roots, source_name)?;
    let sequence_ref = activation_sequence_ref_from_artifact(
        id,
        RasterActivationSequenceArtifactRef::new(artifact_ref)?,
    )?;
    Ok((roots, sequence_ref))
}

pub fn finalize_heads_builder_by_source_name_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: &str,
    id: RasterTensorId,
    head_count: usize,
    sequence_len: usize,
    head_dim: usize,
) -> Result<(RasterArtifactStoreRoots, RasterAttentionHeadsRef)> {
    let (roots, artifact_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(roots, source_name)?;
    let heads_ref = attention_heads_ref_from_artifact(
        id,
        RasterActivationSequenceArtifactRef::new(artifact_ref)?,
        head_count,
        sequence_len,
        head_dim,
    )?;
    Ok((roots, heads_ref))
}

pub fn finalize_kv_cache_builders_by_source_name_with_roots(
    roots: &RasterArtifactStoreRoots,
    keys_source_name: &str,
    values_source_name: &str,
    keys_id: RasterTensorId,
    values_id: RasterTensorId,
    head_count: usize,
    current_len: usize,
    head_dim: usize,
) -> Result<(RasterArtifactStoreRoots, RasterKvCacheRef)> {
    let (roots, keys_artifact_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(roots, keys_source_name)?;
    let (roots, values_artifact_ref) =
        ArtifactIo::finalize_builder_by_source_name_with_roots(&roots, values_source_name)?;
    let cache_ref = kv_cache_ref_from_artifacts(
        keys_id,
        values_id,
        RasterActivationSequenceArtifactRef::new(keys_artifact_ref)?,
        RasterActivationSequenceArtifactRef::new(values_artifact_ref)?,
        head_count,
        current_len,
        head_dim,
    )?;
    Ok((roots, cache_ref))
}

pub fn append_head_row_by_source_name_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_name: &str,
    head_idx: usize,
    token_idx: usize,
    sequence_len: usize,
    row: RasterActivationRow,
) -> Result<RasterArtifactStoreRoots> {
    append_sequence_row_by_source_name_with_roots(
        roots,
        source_name,
        head_idx * sequence_len + token_idx,
        row,
    )
}

pub fn read_sequence_row_from_roots(
    roots: &RasterArtifactStoreRoots,
    request: RasterSequenceRowRequest,
) -> Result<RasterActivationRow> {
    let (row_count, width) = request
        .tensor_ref
        .tensor_ref()
        .shape()
        .sequence_metadata()?;
    if request.row_idx >= row_count {
        bail!(
            "sequence row {} is out of range for {} rows",
            request.row_idx,
            row_count
        );
    }
    ensure_artifact_root_present(roots, request.tensor_ref.tensor_ref().det_commitment())?;
    read_activation_artifact_row_by_tensor_ref(
        request.tensor_ref.tensor_ref(),
        request.row_idx,
        width,
    )
}

pub fn read_head_row_from_roots(
    roots: &RasterArtifactStoreRoots,
    request: RasterHeadRowRequest,
) -> Result<RasterActivationRow> {
    let (head_count, sequence_len, head_dim) =
        request.tensor_ref.tensor_ref().shape().heads_metadata()?;
    if request.head_idx >= head_count {
        bail!(
            "head row request head {} is out of range for {} heads",
            request.head_idx,
            head_count
        );
    }
    if request.token_idx >= sequence_len {
        bail!(
            "head row request token {} is out of range for {} rows",
            request.token_idx,
            sequence_len
        );
    }
    ensure_artifact_root_present(roots, request.tensor_ref.tensor_ref().det_commitment())?;
    read_activation_artifact_row_by_tensor_ref(
        request.tensor_ref.tensor_ref(),
        request.head_idx * sequence_len + request.token_idx,
        head_dim,
    )
}

pub fn read_kv_row_from_roots(
    roots: &RasterArtifactStoreRoots,
    request: RasterKvRowRequest,
) -> Result<RasterActivationRow> {
    let (head_count, current_len, head_dim) = request.cache_ref.shape().kv_cache_metadata()?;
    if request.head_idx >= head_count {
        bail!(
            "KV row request head {} is out of range for {} heads",
            request.head_idx,
            head_count
        );
    }
    if request.token_idx >= current_len {
        bail!(
            "KV row request token {} is out of range for {} rows",
            request.token_idx,
            current_len
        );
    }
    let tensor_ref = match request.row_kind {
        RasterKvRowKind::Key => request.cache_ref.keys(),
        RasterKvRowKind::Value => request.cache_ref.values(),
    };
    ensure_artifact_root_present(roots, tensor_ref.det_commitment())?;
    read_activation_artifact_row_by_tensor_ref(
        tensor_ref,
        request.head_idx * current_len + request.token_idx,
        head_dim,
    )
}

fn ensure_artifact_root_present(roots: &RasterArtifactStoreRoots, root: &str) -> Result<()> {
    if roots.artifacts.iter().any(|entry| entry.root() == root) {
        return Ok(());
    }
    bail!("raster artifact root {root} is not present in the store roots snapshot")
}

pub fn build_intermediate_sequence_commitment(sequence: &RasterActivationSequence) -> String {
    build_rows_commitment(b"raster-intermediate-sequence-v1", sequence.rows())
}

pub fn build_intermediate_heads_commitment(heads: &RasterAttentionHeadSequence) -> String {
    build_nested_rows_commitment(b"raster-intermediate-heads-v1", heads.heads())
}

pub fn build_intermediate_kv_cache_commitment(cache: &RasterKvCache) -> String {
    build_intermediate_kv_cache_rows_commitment(cache.keys(), cache.values())
}

pub fn build_intermediate_kv_cache_roots_commitment(keys_root: &str, values_root: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-intermediate-kv-cache-roots-v1");
    hasher.update((keys_root.len() as u64).to_le_bytes());
    hasher.update(keys_root.as_bytes());
    hasher.update((values_root.len() as u64).to_le_bytes());
    hasher.update(values_root.as_bytes());
    hex_digest(hasher.finalize())
}

fn artifact_ref_for_tensor_ref(
    tensor_ref: &RasterTensorRef,
) -> Result<RasterActivationSequenceArtifactRef> {
    RasterActivationSequenceArtifactRef::new(ArtifactIo::artifact_ref_for_root_any(
        tensor_ref.det_commitment(),
    )?)
}

fn read_activation_artifact_row_by_tensor_ref(
    tensor_ref: &RasterTensorRef,
    row_idx: usize,
    expected_width: usize,
) -> Result<RasterActivationRow> {
    let artifact_ref = artifact_ref_for_tensor_ref(tensor_ref)?;
    let row = read_activation_artifact_row(&artifact_ref, row_idx)?;
    if row.width() != expected_width {
        bail!(
            "activation artifact row {row_idx} has width {}, expected {expected_width}",
            row.width()
        );
    }
    Ok(row)
}

fn build_intermediate_kv_cache_rows_commitment(
    keys: &[Vec<RasterActivationRow>],
    values: &[Vec<RasterActivationRow>],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-intermediate-kv-cache-v1");
    update_nested_rows(&mut hasher, keys);
    update_nested_rows(&mut hasher, values);
    hex_digest(hasher.finalize())
}

fn ensure_kind_matches_shape(kind: RasterTensorKind, shape: &RasterTensorShape) -> Result<()> {
    match (kind, shape) {
        (RasterTensorKind::ActivationSequence, RasterTensorShape::Sequence { .. })
        | (RasterTensorKind::AttentionHeads, RasterTensorShape::Heads { .. })
        | (RasterTensorKind::KvCacheKeys, RasterTensorShape::KvCache { .. })
        | (RasterTensorKind::KvCacheValues, RasterTensorShape::KvCache { .. }) => Ok(()),
        (RasterTensorKind::PartialOutput, _) => Ok(()),
        _ => bail!("raster tensor kind {kind:?} is incompatible with shape {shape:?}"),
    }
}

pub fn insert_activation_sequence_artifact_ref(
    source_name: &str,
    sequence: RasterActivationSequence,
) -> Result<RasterActivationSequenceArtifactRef> {
    let width = sequence.width()?;
    let leaves = sequence
        .rows()
        .iter()
        .map(activation_row_leaf)
        .collect::<Vec<_>>();
    let root = merkle_root(ACTIVATION_ROW_ARTIFACT_DOMAIN.as_bytes(), &leaves);
    if let Ok(artifact_ref) = ArtifactIo::artifact_ref_for_root_any(&root) {
        return RasterActivationSequenceArtifactRef::new(artifact_ref);
    }
    let artifact_ref = ArtifactIo::insert_artifact(
        RasterArtifactId::new(format!("{source_name}.{root}"))?,
        RasterArtifactMetadata::activation_rows(sequence.len(), width)?,
        leaves,
    )?;
    RasterActivationSequenceArtifactRef::new(artifact_ref)
}

fn read_activation_artifact_row(
    artifact_ref: &RasterActivationSequenceArtifactRef,
    row_idx: usize,
) -> Result<RasterActivationRow> {
    if row_idx >= artifact_ref.row_count() {
        bail!(
            "activation artifact row index {row_idx} is out of range for {} rows",
            artifact_ref.row_count()
        );
    }
    let read = ArtifactIo::read_leaf(artifact_ref.artifact_ref(), row_idx)?;
    ArtifactIo::verify_artifact_read(artifact_ref.artifact_ref(), &read)?;
    let row = decode_activation_row_leaf(read.payload())?;
    if row.width() != artifact_ref.width() {
        bail!(
            "activation artifact row {row_idx} has width {}, expected {}",
            row.width(),
            artifact_ref.width()
        );
    }
    Ok(row)
}

fn build_rows_commitment(domain: &[u8], rows: &[RasterActivationRow]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    update_rows(&mut hasher, rows);
    hex_digest(hasher.finalize())
}

fn build_nested_rows_commitment(domain: &[u8], rows: &[Vec<RasterActivationRow>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    update_nested_rows(&mut hasher, rows);
    hex_digest(hasher.finalize())
}

fn update_nested_rows(hasher: &mut Sha256, rows: &[Vec<RasterActivationRow>]) {
    hasher.update((rows.len() as u64).to_le_bytes());
    for head in rows {
        update_rows(hasher, head);
    }
}

fn update_rows(hasher: &mut Sha256, rows: &[RasterActivationRow]) {
    hasher.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        hasher.update((row.width() as u64).to_le_bytes());
        for bits in row.act_bits() {
            hasher.update(bits.to_le_bytes());
        }
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
