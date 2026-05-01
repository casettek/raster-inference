use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use sha2::{Digest, Sha256};

use crate::raster_authoring::AuthRead;
use crate::shared::raster_transformer_kernels::{
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

    fn row_count(&self) -> usize {
        match *self {
            Self::Sequence { row_count, .. } => row_count,
            Self::Heads {
                head_count,
                sequence_len,
                ..
            } => head_count * sequence_len,
            Self::KvCache {
                head_count,
                current_len,
                ..
            } => head_count * current_len,
        }
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

#[derive(Debug, Clone)]
enum StoredTensor {
    Sequence(RasterActivationSequence),
    Heads(RasterAttentionHeadSequence),
    KvKeys(Vec<Vec<RasterActivationRow>>),
    KvValues(Vec<Vec<RasterActivationRow>>),
}

#[derive(Debug, Clone)]
enum BuilderRows {
    Sequence(Vec<RasterActivationRow>),
    Heads(Vec<Vec<RasterActivationRow>>),
    Kv(Vec<Vec<RasterActivationRow>>),
}

#[derive(Debug, Clone)]
struct BuilderState {
    kind: RasterTensorKind,
    expected_shape: RasterTensorShape,
    rows_written: usize,
    rows: BuilderRows,
    finalized: bool,
}

#[derive(Debug, Default, Clone)]
pub struct AuthenticatedRasterTensorStore {
    tensors: HashMap<RasterTensorId, StoredTensor>,
    builders: HashMap<RasterTensorId, BuilderState>,
}

impl AuthenticatedRasterTensorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_activation_sequence(
        &mut self,
        id: RasterTensorId,
        sequence: RasterActivationSequence,
    ) -> Result<RasterActivationSequenceRef> {
        let shape = sequence_shape(&sequence)?;
        let commitment = build_intermediate_sequence_commitment(&sequence);
        let reference = RasterTensorRef::new(
            id.clone(),
            RasterTensorKind::ActivationSequence,
            shape,
            commitment,
        )?;
        self.insert_tensor(id, StoredTensor::Sequence(sequence))?;
        RasterActivationSequenceRef::new(reference)
    }

    pub fn insert_attention_heads(
        &mut self,
        id: RasterTensorId,
        heads: RasterAttentionHeadSequence,
    ) -> Result<RasterAttentionHeadsRef> {
        let shape = heads_shape(&heads)?;
        let commitment = build_intermediate_heads_commitment(&heads);
        let reference = RasterTensorRef::new(
            id.clone(),
            RasterTensorKind::AttentionHeads,
            shape,
            commitment,
        )?;
        self.insert_tensor(id, StoredTensor::Heads(heads))?;
        RasterAttentionHeadsRef::new(reference)
    }

    pub fn insert_kv_cache(
        &mut self,
        keys_id: RasterTensorId,
        values_id: RasterTensorId,
        cache: RasterKvCache,
    ) -> Result<RasterKvCacheRef> {
        let shape = kv_cache_shape(&cache)?;
        let keys_ref = RasterTensorRef::new(
            keys_id.clone(),
            RasterTensorKind::KvCacheKeys,
            shape.clone(),
            build_nested_rows_commitment(b"raster-intermediate-kv-keys-v1", cache.keys()),
        )?;
        let values_ref = RasterTensorRef::new(
            values_id.clone(),
            RasterTensorKind::KvCacheValues,
            shape,
            build_nested_rows_commitment(b"raster-intermediate-kv-values-v1", cache.values()),
        )?;
        let cache_ref = RasterKvCacheRef::new(
            keys_ref.clone(),
            values_ref.clone(),
            build_intermediate_kv_cache_commitment(&cache),
        )?;
        self.insert_tensor(keys_id, StoredTensor::KvKeys(cache.keys().to_vec()))?;
        self.insert_tensor(values_id, StoredTensor::KvValues(cache.values().to_vec()))?;
        Ok(cache_ref)
    }

    pub fn start_sequence_builder(
        &mut self,
        id: RasterTensorId,
        row_count: usize,
        width: usize,
    ) -> Result<RasterTensorBuilderRef> {
        let shape = RasterTensorShape::sequence(row_count, width)?;
        self.start_builder(id, RasterTensorKind::ActivationSequence, shape)
    }

    pub fn start_heads_builder(
        &mut self,
        id: RasterTensorId,
        head_count: usize,
        sequence_len: usize,
        head_dim: usize,
    ) -> Result<RasterTensorBuilderRef> {
        let shape = RasterTensorShape::heads(head_count, sequence_len, head_dim)?;
        self.start_builder(id, RasterTensorKind::AttentionHeads, shape)
    }

    pub fn start_kv_cache_builder(
        &mut self,
        keys_id: RasterTensorId,
        values_id: RasterTensorId,
        head_count: usize,
        current_len: usize,
        head_dim: usize,
    ) -> Result<RasterKvCacheBuilderRef> {
        let expected_shape = RasterTensorShape::kv_cache(head_count, current_len, head_dim)?;
        Ok(RasterKvCacheBuilderRef {
            keys: self.start_builder(
                keys_id,
                RasterTensorKind::KvCacheKeys,
                expected_shape.clone(),
            )?,
            values: self.start_builder(
                values_id,
                RasterTensorKind::KvCacheValues,
                expected_shape.clone(),
            )?,
            expected_shape,
        })
    }

    pub fn append_sequence_row(
        &mut self,
        builder_ref: &mut RasterTensorBuilderRef,
        row_idx: usize,
        row: RasterActivationRow,
    ) -> Result<()> {
        let builder = self.builder_mut(builder_ref)?;
        let RasterTensorShape::Sequence { width, .. } = builder.expected_shape else {
            bail!("sequence append requires sequence builder");
        };
        if row.width() != width {
            bail!(
                "sequence builder row {row_idx} has width {}, expected {width}",
                row.width()
            );
        }
        append_flat_row(builder, builder_ref, row_idx, row)
    }

    pub fn append_head_row(
        &mut self,
        builder_ref: &mut RasterTensorBuilderRef,
        head_idx: usize,
        token_idx: usize,
        row: RasterActivationRow,
    ) -> Result<()> {
        let builder = self.builder_mut(builder_ref)?;
        let RasterTensorShape::Heads {
            head_count,
            sequence_len,
            head_dim,
        } = builder.expected_shape
        else {
            bail!("head append requires heads builder");
        };
        if head_idx >= head_count {
            bail!("head builder head {head_idx} is out of range for {head_count} heads");
        }
        if token_idx >= sequence_len {
            bail!("head builder token {token_idx} is out of range for {sequence_len} rows");
        }
        if row.width() != head_dim {
            bail!(
                "head builder row ({head_idx}, {token_idx}) has width {}, expected {head_dim}",
                row.width()
            );
        }
        append_nested_row(builder, builder_ref, token_idx, head_idx, row)
    }

    pub fn append_kv_row(
        &mut self,
        builder_ref: &mut RasterTensorBuilderRef,
        row_kind: RasterKvRowKind,
        head_idx: usize,
        token_idx: usize,
        row: RasterActivationRow,
    ) -> Result<()> {
        let builder = self.builder_mut(builder_ref)?;
        match (row_kind, builder.kind) {
            (RasterKvRowKind::Key, RasterTensorKind::KvCacheKeys)
            | (RasterKvRowKind::Value, RasterTensorKind::KvCacheValues) => {}
            _ => bail!(
                "KV append {:?} is incompatible with {:?} builder",
                row_kind,
                builder.kind
            ),
        }
        let RasterTensorShape::KvCache {
            head_count,
            current_len,
            head_dim,
        } = builder.expected_shape
        else {
            bail!("KV append requires KV cache builder");
        };
        if head_idx >= head_count {
            bail!("KV builder head {head_idx} is out of range for {head_count} heads");
        }
        if token_idx >= current_len {
            bail!("KV builder token {token_idx} is out of range for {current_len} rows");
        }
        if row.width() != head_dim {
            bail!(
                "KV builder row ({head_idx}, {token_idx}) has width {}, expected {head_dim}",
                row.width()
            );
        }
        append_nested_row(builder, builder_ref, token_idx, head_idx, row)
    }

    pub fn finalize_sequence_builder(
        &mut self,
        builder_ref: RasterTensorBuilderRef,
    ) -> Result<RasterActivationSequenceRef> {
        let (shape, rows) = self.take_builder_rows(&builder_ref)?;
        let RasterTensorShape::Sequence { row_count, .. } = shape else {
            bail!("sequence finalization requires sequence builder");
        };
        let BuilderRows::Sequence(rows) = rows else {
            bail!("sequence finalization received non-sequence rows");
        };
        if rows.len() != row_count {
            bail!(
                "sequence builder finalized with {} rows, expected {row_count}",
                rows.len()
            );
        }
        self.insert_activation_sequence(builder_ref.id, RasterActivationSequence::from_rows(rows))
    }

    pub fn finalize_heads_builder(
        &mut self,
        builder_ref: RasterTensorBuilderRef,
    ) -> Result<RasterAttentionHeadsRef> {
        let (shape, rows) = self.take_builder_rows(&builder_ref)?;
        let RasterTensorShape::Heads {
            head_count,
            sequence_len,
            ..
        } = shape
        else {
            bail!("heads finalization requires heads builder");
        };
        let BuilderRows::Heads(rows) = rows else {
            bail!("heads finalization received non-head rows");
        };
        if rows.len() != head_count {
            bail!(
                "heads builder finalized with {} heads, expected {head_count}",
                rows.len()
            );
        }
        if let Some((head_idx, head)) = rows
            .iter()
            .enumerate()
            .find(|(_, head)| head.len() != sequence_len)
        {
            bail!(
                "heads builder finalized head {head_idx} with {} rows, expected {sequence_len}",
                head.len()
            );
        }
        self.insert_attention_heads(
            builder_ref.id,
            RasterAttentionHeadSequence::from_heads(rows),
        )
    }

    pub fn finalize_kv_cache_builder(
        &mut self,
        builder_ref: RasterKvCacheBuilderRef,
    ) -> Result<RasterKvCacheRef> {
        if builder_ref.keys.expected_shape != builder_ref.expected_shape
            || builder_ref.values.expected_shape != builder_ref.expected_shape
        {
            bail!("KV cache builder key/value shape mismatch");
        }
        let keys = self.take_kv_builder_rows(&builder_ref.keys, RasterKvRowKind::Key)?;
        let values = self.take_kv_builder_rows(&builder_ref.values, RasterKvRowKind::Value)?;
        let cache = RasterKvCache::from_heads(keys.clone(), values.clone())?;
        self.insert_kv_cache(builder_ref.keys.id, builder_ref.values.id, cache)
    }

    pub fn materialize_sequence(
        &self,
        tensor_ref: &RasterActivationSequenceRef,
    ) -> Result<RasterActivationSequence> {
        let sequence = match self.tensor(tensor_ref.tensor_ref())? {
            StoredTensor::Sequence(sequence) => sequence.clone(),
            _ => bail!("raster activation sequence ref points to non-sequence tensor"),
        };
        ensure_commitment(
            tensor_ref.tensor_ref().det_commitment(),
            &build_intermediate_sequence_commitment(&sequence),
            "activation sequence",
        )?;
        Ok(sequence)
    }

    pub fn materialize_heads(
        &self,
        tensor_ref: &RasterAttentionHeadsRef,
    ) -> Result<RasterAttentionHeadSequence> {
        let heads = match self.tensor(tensor_ref.tensor_ref())? {
            StoredTensor::Heads(heads) => heads.clone(),
            _ => bail!("raster attention heads ref points to non-head tensor"),
        };
        ensure_commitment(
            tensor_ref.tensor_ref().det_commitment(),
            &build_intermediate_heads_commitment(&heads),
            "attention heads",
        )?;
        Ok(heads)
    }

    pub fn materialize_kv_cache(&self, cache_ref: &RasterKvCacheRef) -> Result<RasterKvCache> {
        let keys = match self.tensor(cache_ref.keys())? {
            StoredTensor::KvKeys(keys) => keys.clone(),
            _ => bail!("raster KV cache keys ref points to non-key tensor"),
        };
        let values = match self.tensor(cache_ref.values())? {
            StoredTensor::KvValues(values) => values.clone(),
            _ => bail!("raster KV cache values ref points to non-value tensor"),
        };
        let cache = RasterKvCache::from_heads(keys, values)?;
        ensure_commitment(
            cache_ref.det_commitment(),
            &build_intermediate_kv_cache_commitment(&cache),
            "KV cache",
        )?;
        Ok(cache)
    }

    fn insert_tensor(&mut self, id: RasterTensorId, tensor: StoredTensor) -> Result<()> {
        if self.tensors.contains_key(&id) || self.builders.contains_key(&id) {
            bail!(
                "raster tensor id {} is already registered",
                id.source_name()
            );
        }
        self.tensors.insert(id, tensor);
        Ok(())
    }

    fn start_builder(
        &mut self,
        id: RasterTensorId,
        kind: RasterTensorKind,
        expected_shape: RasterTensorShape,
    ) -> Result<RasterTensorBuilderRef> {
        if self.tensors.contains_key(&id) || self.builders.contains_key(&id) {
            bail!(
                "raster tensor id {} is already registered",
                id.source_name()
            );
        }
        ensure_kind_matches_shape(kind, &expected_shape)?;
        let rows = match expected_shape {
            RasterTensorShape::Sequence { row_count, .. } => {
                BuilderRows::Sequence(Vec::with_capacity(row_count))
            }
            RasterTensorShape::Heads { head_count, .. } => {
                BuilderRows::Heads(vec![Vec::new(); head_count])
            }
            RasterTensorShape::KvCache { head_count, .. } => {
                BuilderRows::Kv(vec![Vec::new(); head_count])
            }
        };
        let builder = BuilderState {
            kind,
            expected_shape: expected_shape.clone(),
            rows_written: 0,
            rows,
            finalized: false,
        };
        let builder_ref = RasterTensorBuilderRef {
            id: id.clone(),
            kind,
            expected_shape,
            rows_written: 0,
            running_commitment: builder_running_commitment(&builder),
        };
        self.builders.insert(id, builder);
        Ok(builder_ref)
    }

    fn tensor(&self, tensor_ref: &RasterTensorRef) -> Result<&StoredTensor> {
        let tensor = self.tensors.get(tensor_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster tensor ref {} is not registered",
                tensor_ref.id().source_name()
            )
        })?;
        validate_stored_tensor(tensor_ref, tensor)?;
        Ok(tensor)
    }

    fn builder_mut(&mut self, builder_ref: &RasterTensorBuilderRef) -> Result<&mut BuilderState> {
        let builder = self.builders.get_mut(builder_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster tensor builder {} is not registered",
                builder_ref.id().source_name()
            )
        })?;
        if builder.finalized {
            bail!(
                "raster tensor builder {} is already finalized",
                builder_ref.id().source_name()
            );
        }
        if builder.kind != builder_ref.kind || builder.expected_shape != builder_ref.expected_shape
        {
            bail!(
                "raster tensor builder {} metadata mismatch",
                builder_ref.id().source_name()
            );
        }
        Ok(builder)
    }

    fn take_builder_rows(
        &mut self,
        builder_ref: &RasterTensorBuilderRef,
    ) -> Result<(RasterTensorShape, BuilderRows)> {
        let builder = self.builders.get(builder_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster tensor builder {} is not registered",
                builder_ref.id().source_name()
            )
        })?;
        if builder.rows_written != builder.expected_shape.row_count() {
            bail!(
                "raster tensor builder {} finalized with {} rows, expected {}",
                builder_ref.id().source_name(),
                builder.rows_written,
                builder.expected_shape.row_count()
            );
        }
        let builder = self
            .builders
            .remove(builder_ref.id())
            .expect("builder existence checked before removal");
        Ok((builder.expected_shape, builder.rows))
    }

    fn take_kv_builder_rows(
        &mut self,
        builder_ref: &RasterTensorBuilderRef,
        kind: RasterKvRowKind,
    ) -> Result<Vec<Vec<RasterActivationRow>>> {
        let (shape, rows) = self.take_builder_rows(builder_ref)?;
        if !matches!(shape, RasterTensorShape::KvCache { .. }) {
            bail!("KV cache finalization requires KV cache shape");
        }
        match (kind, rows) {
            (RasterKvRowKind::Key, BuilderRows::Kv(rows))
            | (RasterKvRowKind::Value, BuilderRows::Kv(rows)) => Ok(rows),
            _ => bail!("KV cache finalization received incompatible rows"),
        }
    }
}

impl AuthRead<RasterSequenceRowRequest> for AuthenticatedRasterTensorStore {
    type Output = RasterActivationRow;

    fn auth_read(&self, request: RasterSequenceRowRequest) -> Result<Self::Output> {
        let sequence = self.materialize_sequence(&request.tensor_ref)?;
        let RasterTensorShape::Sequence { row_count, width } =
            request.tensor_ref.tensor_ref().shape()
        else {
            bail!("sequence row request requires sequence shape");
        };
        if request.row_idx >= *row_count {
            bail!(
                "sequence row {} is out of range for {} rows",
                request.row_idx,
                row_count
            );
        }
        let row = sequence
            .rows()
            .get(request.row_idx)
            .ok_or_else(|| anyhow!("sequence row {} is missing", request.row_idx))?;
        if row.width() != *width {
            bail!(
                "sequence row {} has width {}, expected {}",
                request.row_idx,
                row.width(),
                width
            );
        }
        Ok(row.clone())
    }
}

impl AuthRead<RasterHeadRowRequest> for AuthenticatedRasterTensorStore {
    type Output = RasterActivationRow;

    fn auth_read(&self, request: RasterHeadRowRequest) -> Result<Self::Output> {
        let heads = self.materialize_heads(&request.tensor_ref)?;
        let RasterTensorShape::Heads {
            head_count,
            sequence_len,
            head_dim,
        } = request.tensor_ref.tensor_ref().shape()
        else {
            bail!("head row request requires heads shape");
        };
        if request.head_idx >= *head_count {
            bail!(
                "head row request head {} is out of range for {} heads",
                request.head_idx,
                head_count
            );
        }
        if request.token_idx >= *sequence_len {
            bail!(
                "head row request token {} is out of range for {} rows",
                request.token_idx,
                sequence_len
            );
        }
        let row = heads
            .heads()
            .get(request.head_idx)
            .and_then(|head| head.get(request.token_idx))
            .ok_or_else(|| {
                anyhow!(
                    "head row ({}, {}) is missing",
                    request.head_idx,
                    request.token_idx
                )
            })?;
        if row.width() != *head_dim {
            bail!(
                "head row ({}, {}) has width {}, expected {}",
                request.head_idx,
                request.token_idx,
                row.width(),
                head_dim
            );
        }
        Ok(row.clone())
    }
}

impl AuthRead<RasterKvRowRequest> for AuthenticatedRasterTensorStore {
    type Output = RasterActivationRow;

    fn auth_read(&self, request: RasterKvRowRequest) -> Result<Self::Output> {
        let cache = self.materialize_kv_cache(&request.cache_ref)?;
        let RasterTensorShape::KvCache {
            head_count,
            current_len,
            head_dim,
        } = request.cache_ref.shape()
        else {
            bail!("KV row request requires KV cache shape");
        };
        if request.head_idx >= *head_count {
            bail!(
                "KV row request head {} is out of range for {} heads",
                request.head_idx,
                head_count
            );
        }
        if request.token_idx >= *current_len {
            bail!(
                "KV row request token {} is out of range for {} rows",
                request.token_idx,
                current_len
            );
        }
        let rows = match request.row_kind {
            RasterKvRowKind::Key => cache.keys(),
            RasterKvRowKind::Value => cache.values(),
        };
        let row = rows
            .get(request.head_idx)
            .and_then(|head| head.get(request.token_idx))
            .ok_or_else(|| {
                anyhow!(
                    "KV row {:?} ({}, {}) is missing",
                    request.row_kind,
                    request.head_idx,
                    request.token_idx
                )
            })?;
        if row.width() != *head_dim {
            bail!(
                "KV row {:?} ({}, {}) has width {}, expected {}",
                request.row_kind,
                request.head_idx,
                request.token_idx,
                row.width(),
                head_dim
            );
        }
        Ok(row.clone())
    }
}

pub fn build_intermediate_sequence_commitment(sequence: &RasterActivationSequence) -> String {
    build_rows_commitment(b"raster-intermediate-sequence-v1", sequence.rows())
}

pub fn build_intermediate_heads_commitment(heads: &RasterAttentionHeadSequence) -> String {
    build_nested_rows_commitment(b"raster-intermediate-heads-v1", heads.heads())
}

pub fn build_intermediate_kv_cache_commitment(cache: &RasterKvCache) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-intermediate-kv-cache-v1");
    update_nested_rows(&mut hasher, cache.keys());
    update_nested_rows(&mut hasher, cache.values());
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

fn sequence_shape(sequence: &RasterActivationSequence) -> Result<RasterTensorShape> {
    RasterTensorShape::sequence(sequence.len(), sequence.width()?)
}

fn heads_shape(heads: &RasterAttentionHeadSequence) -> Result<RasterTensorShape> {
    RasterTensorShape::heads(
        heads.head_count(),
        heads.sequence_len()?,
        heads.head_width()?,
    )
}

fn kv_cache_shape(cache: &RasterKvCache) -> Result<RasterTensorShape> {
    if cache.current_len() == 0 {
        bail!("raster row store does not yet support empty KV cache refs");
    }
    let head_dim = cache
        .keys()
        .iter()
        .chain(cache.values().iter())
        .find_map(|head| head.first().map(RasterActivationRow::width))
        .ok_or_else(|| anyhow!("KV cache rows are missing"))?;
    RasterTensorShape::kv_cache(cache.head_count(), cache.current_len(), head_dim)
}

fn validate_stored_tensor(tensor_ref: &RasterTensorRef, tensor: &StoredTensor) -> Result<()> {
    match (tensor_ref.kind, tensor) {
        (RasterTensorKind::ActivationSequence, StoredTensor::Sequence(sequence)) => {
            let shape = sequence_shape(sequence)?;
            if &shape != tensor_ref.shape() {
                bail!("stored activation sequence shape mismatch");
            }
            ensure_commitment(
                tensor_ref.det_commitment(),
                &build_intermediate_sequence_commitment(sequence),
                "activation sequence",
            )
        }
        (RasterTensorKind::AttentionHeads, StoredTensor::Heads(heads)) => {
            let shape = heads_shape(heads)?;
            if &shape != tensor_ref.shape() {
                bail!("stored attention heads shape mismatch");
            }
            ensure_commitment(
                tensor_ref.det_commitment(),
                &build_intermediate_heads_commitment(heads),
                "attention heads",
            )
        }
        (RasterTensorKind::KvCacheKeys, StoredTensor::KvKeys(rows))
        | (RasterTensorKind::KvCacheValues, StoredTensor::KvValues(rows)) => {
            validate_kv_rows_shape(rows, tensor_ref.shape())?;
            ensure_commitment(
                tensor_ref.det_commitment(),
                &build_nested_rows_commitment(
                    match tensor_ref.kind {
                        RasterTensorKind::KvCacheKeys => b"raster-intermediate-kv-keys-v1",
                        RasterTensorKind::KvCacheValues => b"raster-intermediate-kv-values-v1",
                        _ => unreachable!("matched above"),
                    },
                    rows,
                ),
                "KV cache rows",
            )
        }
        _ => bail!("stored tensor kind mismatch"),
    }
}

fn validate_kv_rows_shape(
    rows: &[Vec<RasterActivationRow>],
    shape: &RasterTensorShape,
) -> Result<()> {
    let RasterTensorShape::KvCache {
        head_count,
        current_len,
        head_dim,
    } = *shape
    else {
        bail!("KV rows require KV cache shape");
    };
    if rows.len() != head_count {
        bail!("KV row head count mismatch: {} vs {head_count}", rows.len());
    }
    for (head_idx, head) in rows.iter().enumerate() {
        if head.len() != current_len {
            bail!(
                "KV row head {head_idx} has {} rows, expected {current_len}",
                head.len()
            );
        }
        if let Some((row_idx, row)) = head
            .iter()
            .enumerate()
            .find(|(_, row)| row.width() != head_dim)
        {
            bail!(
                "KV row head {head_idx} row {row_idx} has width {}, expected {head_dim}",
                row.width()
            );
        }
    }
    Ok(())
}

fn append_flat_row(
    builder: &mut BuilderState,
    builder_ref: &mut RasterTensorBuilderRef,
    row_idx: usize,
    row: RasterActivationRow,
) -> Result<()> {
    if row_idx != builder.rows_written {
        bail!(
            "raster tensor builder expected row {}, received {row_idx}",
            builder.rows_written
        );
    }
    let BuilderRows::Sequence(rows) = &mut builder.rows else {
        bail!("flat append requires sequence builder rows");
    };
    if row_idx != rows.len() {
        bail!("sequence builder duplicate or skipped row {row_idx}");
    }
    rows.push(row);
    update_builder_after_append(builder, builder_ref);
    Ok(())
}

fn append_nested_row(
    builder: &mut BuilderState,
    builder_ref: &mut RasterTensorBuilderRef,
    token_idx: usize,
    head_idx: usize,
    row: RasterActivationRow,
) -> Result<()> {
    let rows = match &mut builder.rows {
        BuilderRows::Heads(rows) | BuilderRows::Kv(rows) => rows,
        BuilderRows::Sequence(_) => bail!("nested append requires nested builder rows"),
    };
    let head = rows
        .get_mut(head_idx)
        .ok_or_else(|| anyhow!("builder head {head_idx} is out of range"))?;
    if token_idx != head.len() {
        bail!(
            "builder head {head_idx} expected token {}, received {token_idx}",
            head.len()
        );
    }
    head.push(row);
    update_builder_after_append(builder, builder_ref);
    Ok(())
}

fn update_builder_after_append(
    builder: &mut BuilderState,
    builder_ref: &mut RasterTensorBuilderRef,
) {
    builder.rows_written += 1;
    builder_ref.rows_written = builder.rows_written;
    builder_ref.running_commitment = builder_running_commitment(builder);
}

fn builder_running_commitment(builder: &BuilderState) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-intermediate-builder-v1");
    hasher.update(format!("{:?}", builder.kind).as_bytes());
    hasher.update((builder.rows_written as u64).to_le_bytes());
    match &builder.rows {
        BuilderRows::Sequence(rows) => update_rows(&mut hasher, rows),
        BuilderRows::Heads(rows) | BuilderRows::Kv(rows) => update_nested_rows(&mut hasher, rows),
    }
    hex_digest(hasher.finalize())
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

fn ensure_commitment(expected: &str, actual: &str, label: &str) -> Result<()> {
    if expected != actual {
        bail!("{label} commitment mismatch: {actual} vs {expected}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster_authoring::auth_read;
    use crate::shared::det_num::Act;
    use crate::shared::transformer_kernels::build_det_activation_commitment;

    fn tensor_id(name: &str) -> RasterTensorId {
        RasterTensorId::new(name).expect("tensor id")
    }

    fn sequence_fixture() -> RasterActivationSequence {
        RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(-0.5)],
            vec![Act::from_num(0.25), Act::from_num(0.75)],
        ])
    }

    fn heads_fixture() -> RasterAttentionHeadSequence {
        RasterAttentionHeadSequence::from_acts(vec![
            vec![
                vec![Act::from_num(1.0), Act::from_num(0.0)],
                vec![Act::from_num(0.0), Act::from_num(1.0)],
            ],
            vec![
                vec![Act::from_num(0.5), Act::from_num(-0.5)],
                vec![Act::from_num(0.25), Act::from_num(0.75)],
            ],
        ])
    }

    fn kv_fixture() -> RasterKvCache {
        RasterKvCache::from_heads(
            vec![
                vec![
                    RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
                    RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
                ],
                vec![
                    RasterActivationRow::from_acts(vec![Act::from_num(3.0)]),
                    RasterActivationRow::from_acts(vec![Act::from_num(4.0)]),
                ],
            ],
            vec![
                vec![
                    RasterActivationRow::from_acts(vec![Act::from_num(5.0)]),
                    RasterActivationRow::from_acts(vec![Act::from_num(6.0)]),
                ],
                vec![
                    RasterActivationRow::from_acts(vec![Act::from_num(7.0)]),
                    RasterActivationRow::from_acts(vec![Act::from_num(8.0)]),
                ],
            ],
        )
        .expect("kv cache")
    }

    #[test]
    fn typed_refs_reject_incompatible_shapes_and_kinds() {
        let seq_shape = RasterTensorShape::sequence(2, 2).expect("shape");
        let heads_shape = RasterTensorShape::heads(2, 2, 1).expect("shape");

        let seq_ref = RasterTensorRef::new(
            tensor_id("seq"),
            RasterTensorKind::ActivationSequence,
            seq_shape.clone(),
            "abc",
        )
        .expect("seq ref");
        assert!(RasterActivationSequenceRef::new(seq_ref.clone()).is_ok());
        assert!(RasterAttentionHeadsRef::new(seq_ref).is_err());

        let heads_ref = RasterTensorRef::new(
            tensor_id("heads"),
            RasterTensorKind::AttentionHeads,
            heads_shape,
            "abc",
        )
        .expect("heads ref");
        assert!(RasterAttentionHeadsRef::new(heads_ref).is_ok());

        assert!(RasterTensorRef::new(
            tensor_id("bad"),
            RasterTensorKind::ActivationSequence,
            RasterTensorShape::heads(1, 1, 1).expect("shape"),
            "abc",
        )
        .is_err());
        assert!(RasterTensorShape::sequence(0, 1).is_err());
        assert!(RasterTensorShape::heads(1, 0, 1).is_err());
        assert!(RasterTensorShape::kv_cache(1, 1, 0).is_err());
    }

    #[test]
    fn refs_have_stable_serialization() {
        let reference = RasterTensorRef::new(
            tensor_id("seq"),
            RasterTensorKind::ActivationSequence,
            RasterTensorShape::sequence(2, 2).expect("shape"),
            "commitment",
        )
        .expect("ref");

        let serialized = serde_json::to_string(&reference).expect("serialize");

        assert_eq!(
            serialized,
            r#"{"id":{"source_name":"seq"},"kind":"ActivationSequence","shape":{"Sequence":{"row_count":2,"width":2}},"det_commitment":"commitment"}"#
        );
    }

    #[test]
    fn row_reads_return_expected_rows_and_fail_closed() {
        let mut store = AuthenticatedRasterTensorStore::new();
        let seq_ref = store
            .insert_activation_sequence(tensor_id("seq"), sequence_fixture())
            .expect("insert seq");
        let heads_ref = store
            .insert_attention_heads(tensor_id("heads"), heads_fixture())
            .expect("insert heads");
        let kv_ref = store
            .insert_kv_cache(tensor_id("keys"), tensor_id("values"), kv_fixture())
            .expect("insert kv");

        let seq_row = auth_read(
            &store,
            RasterSequenceRowRequest {
                tensor_ref: seq_ref.clone(),
                row_idx: 1,
            },
        )
        .expect("seq row");
        assert_eq!(
            seq_row.act_bits(),
            &[Act::from_num(0.25).to_bits(), Act::from_num(0.75).to_bits()]
        );

        let head_row = auth_read(
            &store,
            RasterHeadRowRequest {
                tensor_ref: heads_ref.clone(),
                head_idx: 1,
                token_idx: 0,
            },
        )
        .expect("head row");
        assert_eq!(
            head_row.act_bits(),
            &[Act::from_num(0.5).to_bits(), Act::from_num(-0.5).to_bits()]
        );

        let kv_row = auth_read(
            &store,
            RasterKvRowRequest {
                cache_ref: kv_ref,
                row_kind: RasterKvRowKind::Value,
                head_idx: 1,
                token_idx: 1,
            },
        )
        .expect("kv row");
        assert_eq!(kv_row.act_bits(), &[Act::from_num(8.0).to_bits()]);

        assert!(auth_read(
            &store,
            RasterSequenceRowRequest {
                tensor_ref: seq_ref,
                row_idx: 2,
            },
        )
        .is_err());
        assert!(auth_read(
            &store,
            RasterHeadRowRequest {
                tensor_ref: heads_ref,
                head_idx: 2,
                token_idx: 0,
            },
        )
        .is_err());
    }

    #[test]
    fn builders_finalize_and_materialize_tensors() {
        let mut store = AuthenticatedRasterTensorStore::new();
        let mut builder = store
            .start_sequence_builder(tensor_id("seq"), 2, 2)
            .expect("builder");
        store
            .append_sequence_row(
                &mut builder,
                0,
                RasterActivationRow::from_acts(vec![Act::from_num(1.0), Act::from_num(2.0)]),
            )
            .expect("append");
        store
            .append_sequence_row(
                &mut builder,
                1,
                RasterActivationRow::from_acts(vec![Act::from_num(3.0), Act::from_num(4.0)]),
            )
            .expect("append");
        let seq_ref = store.finalize_sequence_builder(builder).expect("finalize");
        assert_eq!(store.materialize_sequence(&seq_ref).expect("seq").len(), 2);

        let mut heads_builder = store
            .start_heads_builder(tensor_id("heads"), 1, 2, 1)
            .expect("builder");
        store
            .append_head_row(
                &mut heads_builder,
                0,
                0,
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
            )
            .expect("append");
        store
            .append_head_row(
                &mut heads_builder,
                0,
                1,
                RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
            )
            .expect("append");
        let heads_ref = store
            .finalize_heads_builder(heads_builder)
            .expect("finalize");
        assert_eq!(
            store
                .materialize_heads(&heads_ref)
                .expect("heads")
                .sequence_len()
                .expect("len"),
            2
        );

        let mut kv_builder = store
            .start_kv_cache_builder(tensor_id("keys"), tensor_id("values"), 1, 1, 1)
            .expect("builder");
        assert_eq!(
            kv_builder.expected_shape(),
            &RasterTensorShape::kv_cache(1, 1, 1).expect("shape")
        );
        store
            .append_kv_row(
                &mut kv_builder.keys,
                RasterKvRowKind::Key,
                0,
                0,
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
            )
            .expect("append key");
        store
            .append_kv_row(
                &mut kv_builder.values,
                RasterKvRowKind::Value,
                0,
                0,
                RasterActivationRow::from_acts(vec![Act::from_num(2.0)]),
            )
            .expect("append value");
        let kv_ref = store
            .finalize_kv_cache_builder(kv_builder)
            .expect("finalize");
        assert_eq!(
            store
                .materialize_kv_cache(&kv_ref)
                .expect("kv")
                .current_len(),
            1
        );
    }

    #[test]
    fn builders_fail_closed_for_invalid_writes_and_finalization() {
        let mut store = AuthenticatedRasterTensorStore::new();
        let mut builder = store
            .start_sequence_builder(tensor_id("seq"), 2, 1)
            .expect("builder");

        assert!(store
            .append_sequence_row(
                &mut builder,
                1,
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)])
            )
            .is_err());
        assert!(store.finalize_sequence_builder(builder.clone()).is_err());
        store
            .append_sequence_row(
                &mut builder,
                0,
                RasterActivationRow::from_acts(vec![Act::from_num(1.0)]),
            )
            .expect("append");
        assert!(store
            .append_sequence_row(
                &mut builder,
                1,
                RasterActivationRow::from_acts(vec![Act::from_num(1.0), Act::from_num(2.0)])
            )
            .is_err());

        let mut heads_builder = store
            .start_heads_builder(tensor_id("heads"), 1, 1, 1)
            .expect("builder");
        assert!(store
            .append_head_row(
                &mut heads_builder,
                0,
                0,
                RasterActivationRow::from_acts(vec![Act::from_num(1.0), Act::from_num(2.0)])
            )
            .is_err());
    }

    #[test]
    fn materializers_detect_commitment_mismatch() {
        let mut store = AuthenticatedRasterTensorStore::new();
        let seq_ref = store
            .insert_activation_sequence(tensor_id("seq"), sequence_fixture())
            .expect("insert seq");
        store.tensors.insert(
            tensor_id("seq"),
            StoredTensor::Sequence(RasterActivationSequence::from_acts(vec![
                vec![Act::from_num(9.0), Act::from_num(9.0)],
                vec![Act::from_num(8.0), Act::from_num(8.0)],
            ])),
        );

        let error = store
            .materialize_sequence(&seq_ref)
            .expect_err("corrupt store should fail");

        assert!(error.to_string().contains("commitment mismatch"));
    }

    #[test]
    fn materialized_refs_match_existing_det_commitments() {
        let mut store = AuthenticatedRasterTensorStore::new();
        let sequence = sequence_fixture();
        let seq_ref = store
            .insert_activation_sequence(tensor_id("seq"), sequence.clone())
            .expect("insert seq");
        let materialized = store.materialize_sequence(&seq_ref).expect("materialize");

        assert_eq!(
            build_det_activation_commitment(
                &sequence
                    .rows()
                    .iter()
                    .map(RasterActivationRow::acts)
                    .collect::<Vec<_>>()
            ),
            build_det_activation_commitment(
                &materialized
                    .rows()
                    .iter()
                    .map(RasterActivationRow::acts)
                    .collect::<Vec<_>>()
            )
        );
    }
}
