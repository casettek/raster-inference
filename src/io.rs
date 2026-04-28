use std::{
    collections::{HashMap, HashSet},
    fs,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use sha2::Digest;
use tokenizers::Tokenizer;

use crate::shared::det_num::{
    f32_to_acc, f32_to_act, scale_act, Act, Wgt, DET_NUM_SPEC_VERSION,
    DET_WGT_ARTIFACT_FORMAT_VERSION, DET_WGT_ARTIFACT_MAGIC,
};
use crate::shared::gemma_tokenizer::{
    GemmaAddedToken, GemmaBpeMerge, GemmaTokenizerSpec, GemmaVocabEntry,
};
use crate::shared::input::InferenceExecutionMode;
use crate::shared::transformer::{
    ActivationSequence, DetNumMatrix, DetNumTensorSliceSource, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerMatrixSource, Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4PleGlobalWeights,
    Gemma4PleLayerWeights, Gemma4PleMatrixSource, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource, GemmaTensorSliceSource, InternalActivationRow,
    InternalActivationSequence, MatrixF32, ResolvedGemma4LayerWeights,
    ResolvedGemma4PleLayerWeights,
};
// use crate::trace::{trace_event, trace_scope};

const GEMMA_EMBED_TENSOR_NAMES: &[&str] = &[
    "model.language_model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "embed_tokens.weight",
];

#[derive(Clone, serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(serde::Deserialize)]
struct GemmaConfigFile {
    text_config: GemmaTextConfigFile,
}

#[derive(serde::Deserialize)]
struct GemmaTextConfigFile {
    enable_moe_block: bool,
    final_logit_softcapping: Option<f32>,
    global_head_dim: Option<usize>,
    head_dim: usize,
    hidden_activation: String,
    hidden_size: usize,
    hidden_size_per_layer_input: Option<usize>,
    layer_types: Vec<String>,
    num_global_key_value_heads: Option<usize>,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_parameters: Option<Gemma4RopeParametersFile>,
    sliding_window: usize,
    tie_word_embeddings: Option<bool>,
    vocab_size: usize,
    vocab_size_per_layer_input: Option<usize>,
    attention_k_eq_v: Option<bool>,
    num_kv_shared_layers: Option<usize>,
    use_bidirectional_attention: Option<String>,
}

#[derive(serde::Deserialize)]
struct Gemma4RopeParametersFile {
    full_attention: Option<Gemma4RopeLayerParamsFile>,
    sliding_attention: Option<Gemma4RopeLayerParamsFile>,
    rope_theta: Option<f32>,
}

#[derive(serde::Deserialize)]
struct Gemma4RopeLayerParamsFile {
    partial_rotary_factor: Option<f32>,
    rope_theta: Option<f32>,
}

impl GemmaTextConfigFile {
    fn attention_kind_for_layer(&self, layer_idx: usize) -> Result<Gemma4AttentionKind> {
        let layer_type = self
            .layer_types
            .get(layer_idx)
            .ok_or_else(|| anyhow!("missing Gemma layer type for layer {layer_idx}"))?;
        if layer_type == "sliding_attention" {
            Ok(Gemma4AttentionKind::Sliding)
        } else {
            Ok(Gemma4AttentionKind::Full)
        }
    }

    fn global_head_dim(&self) -> usize {
        self.global_head_dim.unwrap_or(self.head_dim)
    }

    fn attention_k_eq_v(&self) -> bool {
        self.attention_k_eq_v.unwrap_or(false)
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(true)
    }

    fn full_attention_partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.full_attention.as_ref())
            .and_then(|params| params.partial_rotary_factor)
            .unwrap_or(0.25)
    }

    fn rope_local_base_freq(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.sliding_attention.as_ref())
            .and_then(|params| params.rope_theta)
            .unwrap_or(10_000.0)
    }

    fn rope_full_base_freq(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.full_attention.as_ref())
            .and_then(|params| params.rope_theta)
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|params| params.rope_theta)
            })
            .unwrap_or(1_000_000.0)
    }

    fn num_kv_shared_layers(&self) -> usize {
        self.num_kv_shared_layers.unwrap_or(0)
    }

    fn effective_sliding_window(&self) -> usize {
        if self.use_bidirectional_attention.as_deref() == Some("all") {
            (self.sliding_window / 2) + 1
        } else {
            self.sliding_window
        }
    }
}

fn first_kv_shared_layer_idx(config: &GemmaTextConfigFile) -> usize {
    config
        .num_hidden_layers
        .saturating_sub(config.num_kv_shared_layers())
}

fn kv_shared_layer_index(config: &GemmaTextConfigFile, layer_idx: usize) -> Result<Option<usize>> {
    let first_shared_layer_idx = first_kv_shared_layer_idx(config);
    if config.num_kv_shared_layers() == 0 || layer_idx < first_shared_layer_idx {
        return Ok(None);
    }

    let attention_type = config
        .layer_types
        .get(layer_idx)
        .ok_or_else(|| anyhow!("missing Gemma layer type for layer {layer_idx}"))?;
    config.layer_types[..first_shared_layer_idx]
        .iter()
        .rposition(|layer_type| layer_type == attention_type)
        .map(Some)
        .ok_or_else(|| {
            anyhow!(
                "Gemma layer {layer_idx} is configured to share KV without a prior `{attention_type}` donor layer"
            )
        })
}

enum GemmaModelSource {
    Single {
        root_dir: PathBuf,
        weights_path: PathBuf,
    },
    Indexed {
        root_dir: PathBuf,
        index: SafetensorsIndex,
    },
}

impl GemmaModelSource {
    fn root_dir(&self) -> &Path {
        match self {
            Self::Single { root_dir, .. } | Self::Indexed { root_dir, .. } => root_dir.as_path(),
        }
    }
}

struct DetNumModelSource {
    root_dir: PathBuf,
    weights_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DetNumTensorMetadata {
    shape: Vec<usize>,
    data_offset: usize,
    data_len: usize,
}

struct DetNumTensorReader {
    weights_path: PathBuf,
    mmap: std::sync::Arc<Mmap>,
    tensors: HashMap<String, DetNumTensorMetadata>,
}

impl DetNumTensorReader {
    fn load_artifact(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open deterministic artifact {}", path.display()))?;
        let mmap = std::sync::Arc::new(unsafe { Mmap::map(&file) }.with_context(|| {
            format!("failed to mmap deterministic artifact {}", path.display())
        })?);
        let bytes = mmap.as_ref();
        let mut cursor = 0usize;

        fn take_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
            let end = cursor
                .checked_add(len)
                .ok_or_else(|| anyhow!("artifact byte range overflowed"))?;
            let slice = bytes
                .get(*cursor..end)
                .ok_or_else(|| anyhow!("artifact ended unexpectedly"))?;
            *cursor = end;
            Ok(slice)
        }

        fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
            Ok(u32::from_le_bytes(
                take_bytes(bytes, cursor, 4)?
                    .try_into()
                    .expect("u32 byte width should match"),
            ))
        }

        fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
            Ok(u64::from_le_bytes(
                take_bytes(bytes, cursor, 8)?
                    .try_into()
                    .expect("u64 byte width should match"),
            ))
        }

        let magic = take_bytes(&bytes, &mut cursor, DET_WGT_ARTIFACT_MAGIC.len())?;
        if magic != DET_WGT_ARTIFACT_MAGIC {
            bail!(
                "unsupported deterministic artifact magic: expected {:?}, got {:?}",
                DET_WGT_ARTIFACT_MAGIC,
                magic
            );
        }

        let format_version = read_u32(&bytes, &mut cursor)?;
        if format_version != DET_WGT_ARTIFACT_FORMAT_VERSION {
            bail!(
                "unsupported deterministic artifact format version {format_version}; expected {}",
                DET_WGT_ARTIFACT_FORMAT_VERSION
            );
        }

        let spec_version = read_u32(&bytes, &mut cursor)?;
        if spec_version != DET_NUM_SPEC_VERSION {
            bail!(
                "unsupported deterministic artifact det_num spec version {spec_version}; expected {}",
                DET_NUM_SPEC_VERSION
            );
        }

        let tensor_count = usize::try_from(read_u64(&bytes, &mut cursor)?)
            .map_err(|_| anyhow!("tensor count does not fit into usize"))?;
        let mut tensors = HashMap::with_capacity(tensor_count);

        for _ in 0..tensor_count {
            let name_len = usize::try_from(read_u32(&bytes, &mut cursor)?)
                .map_err(|_| anyhow!("tensor name length does not fit into usize"))?;
            let name_bytes = take_bytes(&bytes, &mut cursor, name_len)?;
            let name = String::from_utf8(name_bytes.to_vec())
                .map_err(|_| anyhow!("tensor name is not valid UTF-8"))?;
            let rank = usize::try_from(read_u32(&bytes, &mut cursor)?)
                .map_err(|_| anyhow!("tensor rank does not fit into usize"))?;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(
                    usize::try_from(read_u64(&bytes, &mut cursor)?)
                        .map_err(|_| anyhow!("tensor dimension does not fit into usize"))?,
                );
            }
            let element_count = usize::try_from(read_u64(&bytes, &mut cursor)?)
                .map_err(|_| anyhow!("tensor element count does not fit into usize"))?;
            let expected_element_count = shape
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| anyhow!("tensor shape overflowed"))?;
            if element_count != expected_element_count {
                bail!(
                    "tensor `{name}` element count mismatch: header {element_count} vs shape product {expected_element_count}"
                );
            }

            let payload_len = usize::try_from(read_u64(&bytes, &mut cursor)?)
                .map_err(|_| anyhow!("tensor payload length does not fit into usize"))?;
            let expected_payload_len = element_count
                .checked_mul(4)
                .ok_or_else(|| anyhow!("tensor payload byte count overflowed"))?;
            if payload_len != expected_payload_len {
                bail!(
                    "tensor `{name}` payload length mismatch: expected {expected_payload_len}, got {payload_len}"
                );
            }

            let payload_start = cursor;
            let _payload_bytes = take_bytes(bytes, &mut cursor, payload_len)?;
            if tensors
                .insert(
                    name.clone(),
                    DetNumTensorMetadata {
                        shape,
                        data_offset: payload_start,
                        data_len: payload_len,
                    },
                )
                .is_some()
            {
                bail!("deterministic artifact contains duplicate tensor `{name}`");
            }
        }

        if cursor != bytes.len() {
            bail!(
                "deterministic artifact has {} trailing bytes after tensor payloads",
                bytes.len() - cursor
            );
        }

        Ok(Self {
            weights_path: path.to_path_buf(),
            mmap,
            tensors,
        })
    }

    fn load_first_available_matrix_source(
        &self,
        names: &[&str],
    ) -> Result<(String, DetNumTensorSliceSource)> {
        for name in names {
            if let Some(tensor) = self.tensors.get(*name) {
                if tensor.shape.len() != 2 {
                    bail!(
                        "tensor `{name}` has rank {}, expected 2 for a matrix",
                        tensor.shape.len()
                    );
                }
                return Ok((
                    (*name).to_string(),
                    self.resolve_matrix_source(name, 0, tensor.shape[0], 0, tensor.shape[1])?,
                ));
            }
        }
        bail!("failed to find any supported embedding tensor");
    }

    fn load_matrix(&self, tensor_name: &str) -> Result<MatrixF32> {
        let tensor = self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!("failed to load tensor `{tensor_name}` from deterministic artifact")
        })?;
        self.tensor_to_matrix(tensor_name, tensor)
    }

    fn resolve_full_matrix_source(&self, tensor_name: &str) -> Result<DetNumTensorSliceSource> {
        let tensor = self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!("failed to load tensor `{tensor_name}` from deterministic artifact")
        })?;
        if tensor.shape.len() != 2 {
            bail!(
                "tensor `{tensor_name}` has rank {}, expected 2 for a matrix",
                tensor.shape.len()
            );
        }
        self.resolve_matrix_source(tensor_name, 0, tensor.shape[0], 0, tensor.shape[1])
    }

    fn load_vector(&self, tensor_name: &str) -> Result<Vec<f32>> {
        Ok(self
            .load_vector_wgt(tensor_name)?
            .into_iter()
            .map(|value| det_wgt_to_f32(value.to_bits()))
            .collect())
    }

    fn load_vector_wgt(&self, tensor_name: &str) -> Result<Vec<Wgt>> {
        let tensor = self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!("failed to load tensor `{tensor_name}` from deterministic artifact")
        })?;
        if tensor.shape.len() != 1 {
            bail!(
                "tensor `{tensor_name}` has rank {}, expected 1 for a vector",
                tensor.shape.len()
            );
        }
        let payload = self.payload_slice(tensor_name, tensor)?;
        Ok(payload
            .chunks_exact(4)
            .map(|chunk| {
                Wgt::from_bits(i32::from_le_bytes(
                    chunk.try_into().expect("i32 byte width should match"),
                ))
            })
            .collect())
    }

    fn load_optional_scalar(&self, tensor_name: &str) -> Result<Option<f32>> {
        Ok(self
            .load_optional_scalar_wgt(tensor_name)?
            .map(|value| det_wgt_to_f32(value.to_bits())))
    }

    fn load_optional_scalar_wgt(&self, tensor_name: &str) -> Result<Option<Wgt>> {
        let Some(tensor) = self.tensors.get(tensor_name) else {
            return Ok(None);
        };
        if tensor.shape != [1] {
            bail!(
                "tensor `{tensor_name}` has shape {:?}, expected [1] for a scalar",
                tensor.shape
            );
        }
        let payload = self.payload_slice(tensor_name, tensor)?;
        let bytes = payload
            .get(..4)
            .ok_or_else(|| anyhow!("tensor `{tensor_name}` is missing its scalar payload"))?;
        Ok(Some(Wgt::from_bits(i32::from_le_bytes(
            bytes.try_into().expect("i32 byte width should match"),
        ))))
    }

    fn tensor_to_matrix(
        &self,
        tensor_name: &str,
        tensor: &DetNumTensorMetadata,
    ) -> Result<MatrixF32> {
        if tensor.shape.len() != 2 {
            bail!(
                "tensor `{tensor_name}` has rank {}, expected 2 for a matrix",
                tensor.shape.len()
            );
        }
        let source =
            self.resolve_matrix_source(tensor_name, 0, tensor.shape[0], 0, tensor.shape[1])?;
        decode_matrix_slice_from_det_num_source(&source, self.mmap.as_ref())
    }

    fn resolve_matrix_source(
        &self,
        tensor_name: &str,
        row_offset: usize,
        row_count: usize,
        col_offset: usize,
        col_count: usize,
    ) -> Result<DetNumTensorSliceSource> {
        let tensor = self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!("failed to load tensor `{tensor_name}` from deterministic artifact")
        })?;
        if tensor.shape.len() != 2 {
            bail!(
                "tensor `{tensor_name}` has rank {}, expected 2 for a matrix",
                tensor.shape.len()
            );
        }
        let total_rows = tensor.shape[0];
        let total_cols = tensor.shape[1];
        let row_end = row_offset
            .checked_add(row_count)
            .ok_or_else(|| anyhow!("tensor row slice overflowed"))?;
        let col_end = col_offset
            .checked_add(col_count)
            .ok_or_else(|| anyhow!("tensor column slice overflowed"))?;
        if row_end > total_rows || col_end > total_cols {
            bail!(
                "tensor `{tensor_name}` slice [{row_offset}..{row_end}, {col_offset}..{col_end}] is out of bounds for {}x{} tensor",
                total_rows,
                total_cols
            );
        }
        Ok(DetNumTensorSliceSource {
            weights_path: self.artifact_path(tensor_name)?,
            total_rows,
            total_cols,
            data_offset: tensor.data_offset,
            row_offset,
            row_count,
            col_offset,
            col_count,
        })
    }

    fn payload_slice<'a>(
        &'a self,
        tensor_name: &str,
        tensor: &DetNumTensorMetadata,
    ) -> Result<&'a [u8]> {
        let end = tensor
            .data_offset
            .checked_add(tensor.data_len)
            .ok_or_else(|| anyhow!("tensor `{tensor_name}` byte range overflowed"))?;
        self.mmap
            .get(tensor.data_offset..end)
            .ok_or_else(|| anyhow!("tensor `{tensor_name}` byte range is out of bounds"))
    }

    fn artifact_path(&self, tensor_name: &str) -> Result<PathBuf> {
        let _ = self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!("failed to locate tensor `{tensor_name}` in deterministic artifact")
        })?;
        Ok(self.weights_path.clone())
    }
}

#[derive(Clone)]
struct CachedTensorMetadata {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offset: usize,
    data_len: usize,
}

struct CachedTensorFile {
    mmap: Mmap,
    tensors: HashMap<String, CachedTensorMetadata>,
}

impl CachedTensorFile {
    fn tensor<'a>(&'a self, tensor_name: &str, path: &Path) -> Result<CachedTensorView<'a>> {
        let metadata = self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!(
                "failed to load tensor {tensor_name} from {}",
                path.display()
            )
        })?;
        let data_end = metadata
            .data_offset
            .checked_add(metadata.data_len)
            .ok_or_else(|| anyhow!("tensor byte range overflowed for {}", path.display()))?;
        let data = self
            .mmap
            .get(metadata.data_offset..data_end)
            .ok_or_else(|| anyhow!("tensor byte range is out of bounds for {}", path.display()))?;
        Ok(CachedTensorView { metadata, data })
    }

    fn tensor_metadata<'a>(
        &'a self,
        tensor_name: &str,
        path: &Path,
    ) -> Result<&'a CachedTensorMetadata> {
        self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!(
                "failed to load tensor {tensor_name} from {}",
                path.display()
            )
        })
    }
}

struct CachedTensorView<'a> {
    metadata: &'a CachedTensorMetadata,
    data: &'a [u8],
}

trait TensorBytes {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> Dtype;
    fn data(&self) -> &[u8];
}

impl TensorBytes for CachedTensorView<'_> {
    fn shape(&self) -> &[usize] {
        &self.metadata.shape
    }

    fn dtype(&self) -> Dtype {
        self.metadata.dtype
    }

    fn data(&self) -> &[u8] {
        self.data
    }
}

impl TensorBytes for TensorView<'_> {
    fn shape(&self) -> &[usize] {
        TensorView::shape(self)
    }

    fn dtype(&self) -> Dtype {
        TensorView::dtype(self)
    }

    fn data(&self) -> &[u8] {
        TensorView::data(self)
    }
}

struct GemmaTensorReader {
    source: GemmaModelSource,
    cached_files: HashMap<PathBuf, CachedTensorFile>,
}

impl GemmaTensorReader {
    fn new(source: GemmaModelSource) -> Self {
        Self {
            source,
            cached_files: HashMap::new(),
        }
    }

    fn load_first_available_tensor_metadata(
        &mut self,
        tensor_names: &[&str],
    ) -> Result<(String, usize, usize)> {
        for tensor_name in tensor_names {
            if let Ok(tensor) = self.load_tensor(tensor_name) {
                let shape = tensor.shape();
                if shape.len() != 2 {
                    bail!("expected rank-2 tensor for {tensor_name}, got shape {shape:?}");
                }
                return Ok(((*tensor_name).to_string(), shape[0], shape[1]));
            }
        }

        bail!("failed to locate any of the requested tensors: {tensor_names:?}")
    }

    fn load_matrix(&mut self, tensor_name: &str) -> Result<MatrixF32> {
        let tensor = self.load_tensor(tensor_name)?;
        decode_matrix(&tensor)
    }

    fn load_vector(&mut self, tensor_name: &str) -> Result<Vec<f32>> {
        let tensor = self.load_tensor(tensor_name)?;
        decode_vector(&tensor)
    }

    fn resolve_matrix_source(&mut self, tensor_name: &str) -> Result<Gemma4LayerMatrixSource> {
        let path = self.path_for_tensor(tensor_name)?;
        let cached_file = self.cached_file_for_path(&path)?;
        let metadata = cached_file.tensor_metadata(tensor_name, &path)?;
        if metadata.shape.len() != 2 {
            bail!(
                "expected rank-2 tensor for {tensor_name}, got shape {:?}",
                metadata.shape
            );
        }
        let rows = metadata.shape[0];
        let cols = metadata.shape[1];
        Ok(Gemma4LayerMatrixSource::from_source(
            self.resolve_matrix_slice_source(tensor_name, 0, rows, 0, cols)?,
        ))
    }

    fn resolve_optional_matrix_source(
        &mut self,
        tensor_name: &str,
    ) -> Result<Option<Gemma4LayerMatrixSource>> {
        match self.resolve_matrix_source(tensor_name) {
            Ok(source) => Ok(Some(source)),
            Err(_) => Ok(None),
        }
    }

    fn load_optional_scalar(&mut self, tensor_name: &str) -> Result<Option<f32>> {
        match self.load_tensor(tensor_name) {
            Ok(tensor) => Ok(Some(decode_single_scalar(&tensor)?)),
            Err(_) => Ok(None),
        }
    }

    fn load_tensor(&mut self, tensor_name: &str) -> Result<CachedTensorView<'_>> {
        let path = self.path_for_tensor(tensor_name)?;
        let cached_file = self.cached_file_for_path(&path)?;
        cached_file.tensor(tensor_name, &path)
    }

    fn resolve_matrix_slice_source(
        &mut self,
        tensor_name: &str,
        row_offset: usize,
        row_count: usize,
        col_offset: usize,
        col_count: usize,
    ) -> Result<GemmaTensorSliceSource> {
        let path = self.path_for_tensor(tensor_name)?;
        let cached_file = self.cached_file_for_path(&path)?;
        let metadata = cached_file.tensor_metadata(tensor_name, &path)?;
        if metadata.shape.len() != 2 {
            bail!(
                "expected rank-2 tensor for {tensor_name}, got shape {:?}",
                metadata.shape
            );
        }
        let total_rows = metadata.shape[0];
        let total_cols = metadata.shape[1];
        if row_offset + row_count > total_rows || col_offset + col_count > total_cols {
            bail!(
                "matrix slice [{row_offset}..{}, {col_offset}..{}] is out of bounds for shape {:?}",
                row_offset + row_count,
                col_offset + col_count,
                metadata.shape
            );
        }

        Ok(GemmaTensorSliceSource {
            weights_path: path,
            dtype: metadata.dtype,
            total_rows,
            total_cols,
            data_offset: metadata.data_offset,
            row_offset,
            row_count,
            col_offset,
            col_count,
        })
    }

    fn path_for_tensor(&self, tensor_name: &str) -> Result<PathBuf> {
        match &self.source {
            GemmaModelSource::Single { weights_path, .. } => Ok(weights_path.clone()),
            GemmaModelSource::Indexed { root_dir, index } => {
                let shard = index.weight_map.get(tensor_name).ok_or_else(|| {
                    anyhow!("failed to locate tensor {tensor_name} in model.safetensors.index.json")
                })?;
                Ok(root_dir.join(shard))
            }
        }
    }

    fn cached_file_for_path(&mut self, path: &Path) -> Result<&CachedTensorFile> {
        if !self.cached_files.contains_key(path) {
            let file = File::open(path)
                .with_context(|| format!("failed to open safetensors file {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("failed to mmap safetensors file {}", path.display()))?;
            let tensors = parse_safetensors_metadata(&mmap, path)?;
            self.cached_files
                .insert(path.to_path_buf(), CachedTensorFile { mmap, tensors });
        }

        self.cached_files.get(path).ok_or_else(|| {
            anyhow!(
                "failed to cache safetensors metadata for {}",
                path.display()
            )
        })
    }
}

#[derive(serde::Deserialize)]
struct SafetensorsHeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn parse_safetensors_metadata(
    mmap: &Mmap,
    path: &Path,
) -> Result<HashMap<String, CachedTensorMetadata>> {
    if mmap.len() < 8 {
        bail!("safetensors file {} is missing a header", path.display());
    }
    let header_len = u64::from_le_bytes(
        mmap[..8]
            .try_into()
            .expect("safetensors header prefix should contain 8 bytes"),
    ) as usize;
    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        anyhow!(
            "safetensors header length overflowed for {}",
            path.display()
        )
    })?;
    let header_bytes = mmap
        .get(8..header_end)
        .ok_or_else(|| anyhow!("safetensors header is out of bounds for {}", path.display()))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header_bytes)
        .with_context(|| {
        format!("failed to parse safetensors header from {}", path.display())
    })?;
    let data_section_offset = header_end;
    let mut tensors = HashMap::new();
    for (tensor_name, raw_entry) in header {
        if tensor_name == "__metadata__" {
            continue;
        }
        let entry: SafetensorsHeaderTensor =
            serde_json::from_value(raw_entry).with_context(|| {
                format!(
                    "failed to parse safetensors tensor header for {tensor_name} in {}",
                    path.display()
                )
            })?;
        let dtype = parse_safetensors_dtype(&entry.dtype)?;
        let data_start = data_section_offset
            .checked_add(entry.data_offsets[0])
            .ok_or_else(|| anyhow!("tensor data offset overflowed for {}", path.display()))?;
        let data_end = data_section_offset
            .checked_add(entry.data_offsets[1])
            .ok_or_else(|| anyhow!("tensor data offset overflowed for {}", path.display()))?;
        if data_end < data_start || data_end > mmap.len() {
            bail!(
                "tensor {tensor_name} byte range [{data_start}..{data_end}] is out of bounds for {}",
                path.display()
            );
        }
        tensors.insert(
            tensor_name,
            CachedTensorMetadata {
                dtype,
                shape: entry.shape,
                data_offset: data_start,
                data_len: data_end - data_start,
            },
        );
    }
    Ok(tensors)
}

fn parse_safetensors_dtype(dtype: &str) -> Result<Dtype> {
    match dtype {
        "F16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::BF16),
        "F32" => Ok(Dtype::F32),
        "F64" => Ok(Dtype::F64),
        _ => bail!("unsupported safetensors dtype {dtype}"),
    }
}

pub fn load_chat_template<P: AsRef<Path>>(path: P) -> Result<String> {
    fs::read_to_string(path.as_ref()).with_context(|| {
        format!(
            "failed to read chat template from {}",
            path.as_ref().display()
        )
    })
}

pub fn load_tokenizer_from_path<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    let raw = fs::read(path.as_ref())
        .with_context(|| format!("failed to read tokenizer from {}", path.as_ref().display()))?;

    Tokenizer::from_bytes(raw.as_slice()).map_err(anyhow::Error::msg)
}

#[derive(serde::Deserialize)]
struct GemmaTokenizerJson {
    #[serde(default)]
    added_tokens: Vec<GemmaTokenizerAddedTokenJson>,
    normalizer: GemmaTokenizerNormalizerJson,
    pre_tokenizer: GemmaTokenizerPreTokenizerJson,
    post_processor: GemmaTokenizerPostProcessorJson,
    model: GemmaTokenizerModelJson,
}

#[derive(serde::Deserialize)]
struct GemmaTokenizerAddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(serde::Deserialize)]
struct GemmaTokenizerStringPatternJson {
    #[serde(rename = "String")]
    string: String,
}

#[derive(serde::Deserialize)]
struct GemmaTokenizerNormalizerJson {
    #[serde(rename = "type")]
    kind: String,
    pattern: GemmaTokenizerStringPatternJson,
    content: String,
}

#[derive(serde::Deserialize)]
struct GemmaTokenizerPreTokenizerJson {
    #[serde(rename = "type")]
    kind: String,
    pattern: GemmaTokenizerStringPatternJson,
    behavior: String,
    #[serde(default)]
    invert: bool,
}

#[derive(serde::Deserialize)]
struct GemmaTokenizerPostProcessorJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    special_tokens: HashMap<String, serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct GemmaTokenizerModelJson {
    #[serde(rename = "type")]
    kind: String,
    unk_token: String,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<[String; 2]>,
}

pub fn load_gemma_tokenizer_spec_from_path<P: AsRef<Path>>(path: P) -> Result<GemmaTokenizerSpec> {
    let raw = fs::read(path.as_ref()).with_context(|| {
        format!(
            "failed to read Gemma tokenizer spec from {}",
            path.as_ref().display()
        )
    })?;

    parse_gemma_tokenizer_spec_bytes(&raw)
}

pub fn parse_gemma_tokenizer_spec_bytes(raw: &[u8]) -> Result<GemmaTokenizerSpec> {
    let tokenizer: GemmaTokenizerJson = serde_json::from_slice(raw)
        .context("failed to parse Gemma tokenizer JSON into supported spec")?;

    validate_gemma_tokenizer_json(&tokenizer)?;

    let tokenizer_sha256 = format!("{:x}", sha2::Sha256::digest(raw));
    let vocab = tokenizer
        .model
        .vocab
        .into_iter()
        .map(|(token, id)| GemmaVocabEntry { token, id })
        .collect::<Vec<_>>();
    let merges = tokenizer
        .model
        .merges
        .into_iter()
        .enumerate()
        .map(|(rank, [left, right])| GemmaBpeMerge {
            merged: format!("{left}{right}"),
            left,
            right,
            rank,
        })
        .collect::<Vec<_>>();
    let added_tokens = tokenizer
        .added_tokens
        .into_iter()
        .map(|token| GemmaAddedToken {
            id: token.id,
            content: token.content,
            special: token.special,
        })
        .collect::<Vec<_>>();

    GemmaTokenizerSpec::new(
        tokenizer_sha256,
        vocab,
        merges,
        added_tokens,
        tokenizer.model.unk_token,
        tokenizer.model.byte_fallback,
        tokenizer.normalizer.content,
        tokenizer.pre_tokenizer.pattern.string,
    )
}

fn validate_gemma_tokenizer_json(tokenizer: &GemmaTokenizerJson) -> Result<()> {
    if tokenizer.normalizer.kind != "Replace" {
        bail!(
            "unsupported Gemma tokenizer normalizer {}",
            tokenizer.normalizer.kind
        );
    }
    if tokenizer.normalizer.pattern.string != " " || tokenizer.normalizer.content != "▁" {
        bail!("unsupported Gemma tokenizer normalizer replacement");
    }
    if tokenizer.pre_tokenizer.kind != "Split" {
        bail!(
            "unsupported Gemma tokenizer pre-tokenizer {}",
            tokenizer.pre_tokenizer.kind
        );
    }
    if tokenizer.pre_tokenizer.pattern.string != " "
        || tokenizer.pre_tokenizer.behavior != "MergedWithPrevious"
        || tokenizer.pre_tokenizer.invert
    {
        bail!("unsupported Gemma tokenizer pre-tokenizer split behavior");
    }
    if tokenizer.post_processor.kind != "TemplateProcessing" {
        bail!(
            "unsupported Gemma tokenizer post-processor {}",
            tokenizer.post_processor.kind
        );
    }
    if !tokenizer.post_processor.special_tokens.is_empty() {
        bail!("unsupported Gemma tokenizer post-processor special tokens");
    }
    if tokenizer.model.kind != "BPE" {
        bail!("unsupported Gemma tokenizer model {}", tokenizer.model.kind);
    }
    if !tokenizer.model.byte_fallback {
        bail!("Gemma tokenizer spec requires byte_fallback = true");
    }
    if tokenizer.model.ignore_merges {
        bail!("Gemma tokenizer spec does not support ignore_merges = true");
    }
    if tokenizer.model.vocab.is_empty() {
        bail!("Gemma tokenizer spec requires a non-empty vocab");
    }

    Ok(())
}

pub fn load_embedding_table_from_path<P: AsRef<Path>>(path: P) -> Result<EmbeddingTable> {
    let raw = fs::read_to_string(path.as_ref()).with_context(|| {
        format!(
            "failed to read embedding table from {}",
            path.as_ref().display()
        )
    })?;

    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse embedding table JSON from {}",
            path.as_ref().display()
        )
    })
}

pub fn load_embedding_table_from_gemma_model_path<P: AsRef<Path>>(
    path: P,
) -> Result<EmbeddingTable> {
    let mut model = load_transformer_state_model_from_gemma_model_path(path)?;
    let source = model
        .embedding_source
        .take()
        .ok_or_else(|| anyhow!("transformer state model is missing an embedding source"))?;
    load_full_embedding_table_from_source(&source)
}

pub fn load_transformer_state_model_from_gemma_model_path<P: AsRef<Path>>(
    path: P,
) -> Result<Gemma4TransformerModel> {
    // let _trace = trace_scope("io.load_transformer_state_model_from_gemma_model_path");
    let source = resolve_gemma_model_source(path.as_ref())?;
    let config = load_gemma_text_config(source.root_dir().join("config.json"))?;

    if config.enable_moe_block {
        bail!("transformer state model currently only supports Gemma 4 dense layers, not MoE checkpoints");
    }
    if config.hidden_activation != "gelu_pytorch_tanh" {
        bail!(
            "transformer state model currently only supports gelu_pytorch_tanh, got {}",
            config.hidden_activation
        );
    }
    if config.num_hidden_layers != config.layer_types.len() {
        bail!(
            "Gemma config layer count mismatch: num_hidden_layers={} vs layer_types={}",
            config.num_hidden_layers,
            config.layer_types.len()
        );
    }

    let mut reader = GemmaTensorReader::new(source);
    let (embedding_tensor_name, _vocab_size, hidden_size) =
        reader.load_first_available_tensor_metadata(GEMMA_EMBED_TENSOR_NAMES)?;
    let first_shared_layer_idx = first_kv_shared_layer_idx(&config);
    let mut kv_donor_layers = HashSet::new();
    if first_shared_layer_idx < config.num_hidden_layers {
        for shared_layer_idx in first_shared_layer_idx..config.num_hidden_layers {
            if let Some(donor_layer_idx) = kv_shared_layer_index(&config, shared_layer_idx)? {
                kv_donor_layers.insert(donor_layer_idx);
            }
        }
    }
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        // trace_event(format!("io.load_gemma4_layer_weights layer={layer_idx}"));
        layers.push(load_gemma4_layer_weights(
            &mut reader,
            &config,
            layer_idx,
            kv_donor_layers.contains(&layer_idx),
        )?);
    }
    let ple_global = load_ple_global_weights(&mut reader, &config)?;
    // trace_event("io.load_final_norm_weight");
    let final_norm_weight = reader.load_vector("model.language_model.norm.weight")?;
    // trace_event("io.load_logits_projection");
    let logits_projection = if config.tie_word_embeddings() {
        Gemma4LogitsProjection::TiedEmbedding(reader.load_matrix(&embedding_tensor_name)?)
    } else {
        Gemma4LogitsProjection::UntiedLmHead {
            weight: reader.load_matrix("model.language_model.lm_head.weight")?,
            det_weight: None,
        }
    };
    let embedding_source = build_embedding_source(&reader, &embedding_tensor_name, hidden_size);

    Ok(Gemma4TransformerModel {
        provenance: crate::shared::transformer::Gemma4ModelProvenance::Fp32,
        embedding_table: None,
        embedding_source: Some(embedding_source),
        layers,
        ple_global,
        final_norm_weight,
        final_norm_weight_det: None,
        logits_projection,
        final_logit_softcapping: config.final_logit_softcapping,
        final_logit_softcapping_det: None,
        rms_norm_eps: config.rms_norm_eps,
        rms_norm_eps_det: None,
    })
}

pub fn load_transformer_state_model_from_det_num_wgt_path<P: AsRef<Path>>(
    path: P,
) -> Result<Gemma4TransformerModel> {
    let source = resolve_det_num_model_source(path.as_ref())?;
    let config = load_gemma_text_config(source.root_dir.join("config.json"))?;

    if config.enable_moe_block {
        bail!(
            "transformer state model currently only supports Gemma 4 dense layers, not MoE checkpoints"
        );
    }
    if config.hidden_activation != "gelu_pytorch_tanh" {
        bail!(
            "transformer state model currently only supports gelu_pytorch_tanh, got {}",
            config.hidden_activation
        );
    }
    if config.num_hidden_layers != config.layer_types.len() {
        bail!(
            "Gemma config layer count mismatch: num_hidden_layers={} vs layer_types={}",
            config.num_hidden_layers,
            config.layer_types.len()
        );
    }

    let reader = DetNumTensorReader::load_artifact(&source.weights_path)?;
    let (embedding_tensor_name, embedding_source) =
        build_det_num_embedding_source(&reader, GEMMA_EMBED_TENSOR_NAMES)?;
    let first_shared_layer_idx = first_kv_shared_layer_idx(&config);
    let mut kv_donor_layers = HashSet::new();
    if first_shared_layer_idx < config.num_hidden_layers {
        for shared_layer_idx in first_shared_layer_idx..config.num_hidden_layers {
            if let Some(donor_layer_idx) = kv_shared_layer_index(&config, shared_layer_idx)? {
                kv_donor_layers.insert(donor_layer_idx);
            }
        }
    }
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        layers.push(load_det_num_gemma4_layer_weights(
            &reader,
            &config,
            layer_idx,
            kv_donor_layers.contains(&layer_idx),
        )?);
    }
    let ple_global = load_det_num_ple_global_weights(&reader, &config)?;
    let final_norm_weight_det = reader.load_vector_wgt("model.language_model.norm.weight")?;
    let final_norm_weight = final_norm_weight_det
        .iter()
        .map(|value| det_wgt_to_f32(value.to_bits()))
        .collect();
    let logits_projection = if config.tie_word_embeddings() {
        Gemma4LogitsProjection::TiedEmbedding(reader.load_matrix(&embedding_tensor_name)?)
    } else {
        Gemma4LogitsProjection::UntiedLmHead {
            weight: reader.load_matrix("model.language_model.lm_head.weight")?,
            det_weight: Some(std::sync::Arc::new(decode_det_num_matrix_from_source(
                &reader.resolve_full_matrix_source("model.language_model.lm_head.weight")?,
                reader.mmap.as_ref(),
            )?)),
        }
    };

    Ok(Gemma4TransformerModel {
        provenance: crate::shared::transformer::Gemma4ModelProvenance::DetNumWgt,
        embedding_table: None,
        embedding_source: Some(embedding_source),
        layers,
        ple_global,
        final_norm_weight,
        final_norm_weight_det: Some(final_norm_weight_det),
        logits_projection,
        final_logit_softcapping: config.final_logit_softcapping,
        final_logit_softcapping_det: config.final_logit_softcapping.map(f32_to_act),
        rms_norm_eps: config.rms_norm_eps,
        rms_norm_eps_det: Some(f32_to_acc(config.rms_norm_eps)),
    })
}

fn det_wgt_to_f32(value: i32) -> f32 {
    value as f32 / 65_536.0
}

fn det_wgt_vec_to_f32(values: &[Wgt]) -> Vec<f32> {
    values
        .iter()
        .map(|value| det_wgt_to_f32(value.to_bits()))
        .collect()
}

fn det_wgt_to_act(value: Wgt) -> Act {
    Act::from_bits(value.to_bits())
}

fn load_ple_global_weights(
    reader: &mut GemmaTensorReader,
    config: &GemmaTextConfigFile,
) -> Result<Option<Gemma4PleGlobalWeights>> {
    // let _trace = trace_scope("io.load_ple_global_weights");
    let hidden_size = config.hidden_size;
    let ple_dim = config.hidden_size_per_layer_input.unwrap_or(0);
    if ple_dim == 0 {
        return Ok(None);
    }

    let ple_vocab_size = config
        .vocab_size_per_layer_input
        .unwrap_or(config.vocab_size);
    let mut token_embeddings = Vec::with_capacity(config.num_hidden_layers);
    let mut model_projections = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        token_embeddings.push(reader.resolve_matrix_slice_source(
            "model.language_model.embed_tokens_per_layer.weight",
            0,
            ple_vocab_size,
            layer_idx * ple_dim,
            ple_dim,
        )?);
        model_projections.push(reader.resolve_matrix_slice_source(
            "model.language_model.per_layer_model_projection.weight",
            layer_idx * ple_dim,
            ple_dim,
            0,
            hidden_size,
        )?);
    }

    Ok(Some(Gemma4PleGlobalWeights::from_sources(
        token_embeddings,
        model_projections,
        reader.load_vector("model.language_model.per_layer_projection_norm.weight")?,
        (ple_dim as f32).sqrt(),
        (hidden_size as f32).powf(-0.5),
        2f32.powf(-0.5),
    )))
}

fn load_det_num_ple_global_weights(
    reader: &DetNumTensorReader,
    config: &GemmaTextConfigFile,
) -> Result<Option<Gemma4PleGlobalWeights>> {
    let hidden_size = config.hidden_size;
    let ple_dim = config.hidden_size_per_layer_input.unwrap_or(0);
    if ple_dim == 0 {
        return Ok(None);
    }

    let ple_vocab_size = config
        .vocab_size_per_layer_input
        .unwrap_or(config.vocab_size);
    let mut token_embeddings = Vec::with_capacity(config.num_hidden_layers);
    let mut model_projections = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        token_embeddings.push(reader.resolve_matrix_source(
            "model.language_model.embed_tokens_per_layer.weight",
            0,
            ple_vocab_size,
            layer_idx * ple_dim,
            ple_dim,
        )?);
        model_projections.push(reader.resolve_matrix_source(
            "model.language_model.per_layer_model_projection.weight",
            layer_idx * ple_dim,
            ple_dim,
            0,
            hidden_size,
        )?);
    }

    let projection_norm_weight_det =
        reader.load_vector_wgt("model.language_model.per_layer_projection_norm.weight")?;
    Ok(Some(
        Gemma4PleGlobalWeights::from_det_num_sources_with_canonical(
            token_embeddings,
            model_projections,
            projection_norm_weight_det
                .iter()
                .map(|value| det_wgt_to_f32(value.to_bits()))
                .collect(),
            projection_norm_weight_det,
            (ple_dim as f32).sqrt(),
            f32_to_act((ple_dim as f32).sqrt()),
            (hidden_size as f32).powf(-0.5),
            f32_to_act((hidden_size as f32).powf(-0.5)),
            2f32.powf(-0.5),
            f32_to_act(2f32.powf(-0.5)),
        ),
    ))
}

pub(crate) fn load_ple_token_embedding_row_internal(
    ple_global: &Gemma4PleGlobalWeights,
    layer_idx: usize,
    token_id: u32,
) -> Result<InternalActivationRow> {
    let row_idx = usize::try_from(token_id).expect("u32 should fit into usize");
    let cache_key = (layer_idx, row_idx);
    let source = ple_global.token_embeddings.get(layer_idx).ok_or_else(|| {
        anyhow!("transformer PLE token embedding slice count mismatch at layer {layer_idx}")
    })?;
    if !matches!(source, Gemma4PleMatrixSource::DetNumLazy(_)) {
        if let Some(cached_row) = ple_global
            .token_row_cache
            .lock()
            .map_err(|_| anyhow!("PLE token row cache is poisoned"))?
            .get(&cache_key)
            .cloned()
        {
            return Ok(InternalActivationRow::from_values(cached_row));
        }
    }
    let row = match source {
        Gemma4PleMatrixSource::Materialized(matrix) => {
            InternalActivationRow::from_values(matrix_row(matrix, row_idx)?)
        }
        Gemma4PleMatrixSource::Lazy(source) => {
            let mmap = ple_mmap_for_path(ple_global, &source.weights_path)?;
            InternalActivationRow::from_values(decode_matrix_row_from_source(
                source,
                row_idx,
                mmap.as_ref(),
            )?)
        }
        Gemma4PleMatrixSource::DetNumLazy(source) => {
            let mmap = ple_mmap_for_path(ple_global, &source.weights_path)?;
            InternalActivationRow::from_det_values(decode_matrix_row_acts_from_det_num_source(
                source,
                row_idx,
                mmap.as_ref(),
            )?)
        }
    };
    ple_global
        .token_row_cache
        .lock()
        .map_err(|_| anyhow!("PLE token row cache is poisoned"))?
        .insert(cache_key, row.clone_f32());
    Ok(row)
}

pub(crate) fn load_ple_model_projection(
    ple_global: &Gemma4PleGlobalWeights,
    layer_idx: usize,
) -> Result<MatrixF32> {
    if let Some(cached_matrix) = ple_global
        .model_projection_cache
        .lock()
        .map_err(|_| anyhow!("PLE model projection cache is poisoned"))?
        .get(&layer_idx)
        .cloned()
    {
        return Ok(cached_matrix);
    }

    let source = ple_global.model_projections.get(layer_idx).ok_or_else(|| {
        anyhow!("transformer PLE model projection slice count mismatch at layer {layer_idx}")
    })?;
    let matrix = match source {
        Gemma4PleMatrixSource::Materialized(matrix) => matrix.clone(),
        Gemma4PleMatrixSource::Lazy(source) => {
            let mmap = ple_mmap_for_path(ple_global, &source.weights_path)?;
            decode_matrix_slice_from_source(source, mmap.as_ref())?
        }
        Gemma4PleMatrixSource::DetNumLazy(source) => {
            let mmap = ple_mmap_for_path(ple_global, &source.weights_path)?;
            decode_matrix_slice_from_det_num_source(source, mmap.as_ref())?
        }
    };
    ple_global
        .model_projection_cache
        .lock()
        .map_err(|_| anyhow!("PLE model projection cache is poisoned"))?
        .insert(layer_idx, matrix.clone());
    Ok(matrix)
}

pub(crate) fn materialize_det_num_ple_model_projection(
    ple_global: &Gemma4PleGlobalWeights,
    layer_idx: usize,
) -> Result<Option<std::sync::Arc<DetNumMatrix>>> {
    if let Some(cached_matrix) = ple_global
        .model_projection_det_cache
        .lock()
        .map_err(|_| anyhow!("PLE model projection det cache is poisoned"))?
        .get(&layer_idx)
        .cloned()
    {
        return Ok(Some(cached_matrix));
    }

    let Some(source) = ple_global.model_projections.get(layer_idx) else {
        bail!("transformer PLE model projection slice count mismatch at layer {layer_idx}");
    };
    let Gemma4PleMatrixSource::DetNumLazy(source) = source else {
        return Ok(None);
    };
    let mmap = ple_mmap_for_path(ple_global, &source.weights_path)?;
    let matrix = std::sync::Arc::new(decode_det_num_matrix_from_source(source, mmap.as_ref())?);
    ple_global
        .model_projection_det_cache
        .lock()
        .map_err(|_| anyhow!("PLE model projection det cache is poisoned"))?
        .insert(layer_idx, matrix.clone());
    Ok(Some(matrix))
}

pub(crate) fn resolve_layer_weights(
    layer: &Gemma4LayerWeights,
) -> Result<ResolvedGemma4LayerWeights> {
    Ok(ResolvedGemma4LayerWeights {
        attention_kind: layer.attention_kind,
        hidden_size: layer.hidden_size,
        num_heads: layer.num_heads,
        num_kv_heads: layer.num_kv_heads,
        head_dim: layer.head_dim,
        sliding_window: layer.sliding_window,
        cache_sliding_window: layer.cache_sliding_window,
        rms_norm_eps: layer.rms_norm_eps,
        rms_norm_eps_det: layer.rms_norm_eps_det,
        rope_base: layer.rope_base,
        rope_base_det: layer.rope_base_det,
        partial_rotary_dim: layer.partial_rotary_dim,
        rope_freq_base_dim: layer.rope_freq_base_dim,
        kv_shared_layer_index: layer.kv_shared_layer_index,
        attention_k_eq_v: layer.attention_k_eq_v,
        q_proj: materialize_layer_matrix_source(&layer.q_proj)?,
        k_proj: materialize_layer_matrix_source(&layer.k_proj)?,
        v_proj: layer
            .v_proj
            .as_ref()
            .map(materialize_layer_matrix_source)
            .transpose()?,
        o_proj: materialize_layer_matrix_source(&layer.o_proj)?,
        q_proj_det: materialize_det_num_layer_matrix_source(&layer.q_proj)?,
        k_proj_det: materialize_det_num_layer_matrix_source(&layer.k_proj)?,
        v_proj_det: layer
            .v_proj
            .as_ref()
            .map(materialize_det_num_layer_matrix_source)
            .transpose()?
            .flatten(),
        o_proj_det: materialize_det_num_layer_matrix_source(&layer.o_proj)?,
        q_norm_weight: layer.q_norm_weight.clone(),
        q_norm_weight_det: layer.q_norm_weight_det.clone(),
        k_norm_weight: layer.k_norm_weight.clone(),
        k_norm_weight_det: layer.k_norm_weight_det.clone(),
        input_layernorm_weight: layer.input_layernorm_weight.clone(),
        input_layernorm_weight_det: layer.input_layernorm_weight_det.clone(),
        post_attention_layernorm_weight: layer.post_attention_layernorm_weight.clone(),
        post_attention_layernorm_weight_det: layer.post_attention_layernorm_weight_det.clone(),
        pre_feedforward_layernorm_weight: layer.pre_feedforward_layernorm_weight.clone(),
        pre_feedforward_layernorm_weight_det: layer.pre_feedforward_layernorm_weight_det.clone(),
        post_feedforward_layernorm_weight: layer.post_feedforward_layernorm_weight.clone(),
        post_feedforward_layernorm_weight_det: layer.post_feedforward_layernorm_weight_det.clone(),
        gate_proj: materialize_layer_matrix_source(&layer.gate_proj)?,
        up_proj: materialize_layer_matrix_source(&layer.up_proj)?,
        down_proj: materialize_layer_matrix_source(&layer.down_proj)?,
        gate_proj_det: materialize_det_num_layer_matrix_source(&layer.gate_proj)?,
        up_proj_det: materialize_det_num_layer_matrix_source(&layer.up_proj)?,
        down_proj_det: materialize_det_num_layer_matrix_source(&layer.down_proj)?,
        ple: layer
            .ple
            .as_ref()
            .map(resolve_ple_layer_weights)
            .transpose()?,
        layer_scalar: layer.layer_scalar,
        layer_scalar_det: layer.layer_scalar_det,
    })
}

fn resolve_ple_layer_weights(ple: &Gemma4PleLayerWeights) -> Result<ResolvedGemma4PleLayerWeights> {
    Ok(ResolvedGemma4PleLayerWeights {
        input_gate: materialize_layer_matrix_source(&ple.input_gate)?,
        layer_projection: materialize_layer_matrix_source(&ple.layer_projection)?,
        input_gate_det: materialize_det_num_layer_matrix_source(&ple.input_gate)?,
        layer_projection_det: materialize_det_num_layer_matrix_source(&ple.layer_projection)?,
        post_input_norm_weight: ple.post_input_norm_weight.clone(),
        post_input_norm_weight_det: ple.post_input_norm_weight_det.clone(),
    })
}

fn materialize_layer_matrix_source(
    source: &Gemma4LayerMatrixSource,
) -> Result<std::sync::Arc<MatrixF32>> {
    match source {
        Gemma4LayerMatrixSource::Materialized(matrix) => Ok(matrix.clone()),
        Gemma4LayerMatrixSource::Lazy { source, cache } => {
            if let Some(matrix) = cache
                .lock()
                .map_err(|_| anyhow!("layer matrix cache is poisoned"))?
                .clone()
            {
                return Ok(matrix);
            }
            let file = File::open(&source.weights_path).with_context(|| {
                format!(
                    "failed to open safetensors file {}",
                    source.weights_path.display()
                )
            })?;
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
                format!(
                    "failed to mmap safetensors file {}",
                    source.weights_path.display()
                )
            })?;
            let matrix = std::sync::Arc::new(decode_matrix_slice_from_source(source, &mmap)?);
            *cache
                .lock()
                .map_err(|_| anyhow!("layer matrix cache is poisoned"))? = Some(matrix.clone());
            Ok(matrix)
        }
        Gemma4LayerMatrixSource::DetNumLazy { source, cache, .. } => {
            if let Some(matrix) = cache
                .lock()
                .map_err(|_| anyhow!("layer matrix cache is poisoned"))?
                .clone()
            {
                return Ok(matrix);
            }
            let file = File::open(&source.weights_path).with_context(|| {
                format!(
                    "failed to open deterministic artifact {}",
                    source.weights_path.display()
                )
            })?;
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
                format!(
                    "failed to mmap deterministic artifact {}",
                    source.weights_path.display()
                )
            })?;
            let matrix =
                std::sync::Arc::new(decode_matrix_slice_from_det_num_source(source, &mmap)?);
            *cache
                .lock()
                .map_err(|_| anyhow!("layer matrix cache is poisoned"))? = Some(matrix.clone());
            Ok(matrix)
        }
    }
}

fn materialize_det_num_layer_matrix_source(
    source: &Gemma4LayerMatrixSource,
) -> Result<Option<std::sync::Arc<DetNumMatrix>>> {
    let Gemma4LayerMatrixSource::DetNumLazy {
        source, det_cache, ..
    } = source
    else {
        return Ok(None);
    };

    if let Some(matrix) = det_cache
        .lock()
        .map_err(|_| anyhow!("deterministic layer matrix cache is poisoned"))?
        .clone()
    {
        return Ok(Some(matrix));
    }

    let file = File::open(&source.weights_path).with_context(|| {
        format!(
            "failed to open deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
        format!(
            "failed to mmap deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let matrix = std::sync::Arc::new(decode_det_num_matrix_from_source(source, &mmap)?);
    *det_cache
        .lock()
        .map_err(|_| anyhow!("deterministic layer matrix cache is poisoned"))? =
        Some(matrix.clone());
    Ok(Some(matrix))
}

fn ple_mmap_for_path(
    ple_global: &Gemma4PleGlobalWeights,
    path: &Path,
) -> Result<std::sync::Arc<Mmap>> {
    if let Some(mmap) = ple_global
        .mmap_cache
        .lock()
        .map_err(|_| anyhow!("PLE mmap cache is poisoned"))?
        .get(path)
        .cloned()
    {
        return Ok(mmap);
    }

    let file = File::open(path)
        .with_context(|| format!("failed to open safetensors file {}", path.display()))?;
    let mmap = std::sync::Arc::new(
        unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap safetensors file {}", path.display()))?,
    );
    ple_global
        .mmap_cache
        .lock()
        .map_err(|_| anyhow!("PLE mmap cache is poisoned"))?
        .insert(path.to_path_buf(), mmap.clone());
    Ok(mmap)
}

fn decode_matrix_row_from_source(
    source: &GemmaTensorSliceSource,
    row_idx: usize,
    mmap: &Mmap,
) -> Result<Vec<f32>> {
    if row_idx >= source.row_count {
        bail!(
            "matrix row {row_idx} is out of bounds for slice with {} rows",
            source.row_count
        );
    }
    let bytes_per_scalar = bytes_per_scalar(source.dtype)?;
    let row_bytes = source
        .total_cols
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let global_row_idx = source.row_offset + row_idx;
    let start = source
        .data_offset
        .checked_add(global_row_idx * row_bytes)
        .and_then(|offset| offset.checked_add(source.col_offset * bytes_per_scalar))
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let end = start
        .checked_add(source.col_count * bytes_per_scalar)
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let encoded_row = mmap
        .get(start..end)
        .ok_or_else(|| anyhow!("matrix slice byte range is out of bounds"))?;
    if source.dtype == Dtype::F32 {
        return decode_f32_bytes_to_vec(encoded_row);
    }
    let mut row = Vec::with_capacity(source.col_count);
    for encoded_value in encoded_row.chunks_exact(bytes_per_scalar) {
        row.push(decode_scalar(encoded_value, source.dtype)?);
    }
    Ok(row)
}

fn decode_matrix_slice_from_source(
    source: &GemmaTensorSliceSource,
    mmap: &Mmap,
) -> Result<MatrixF32> {
    let bytes_per_scalar = bytes_per_scalar(source.dtype)?;
    let row_bytes = source
        .total_cols
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let mut values = Vec::with_capacity(source.row_count * source.col_count);
    for row_idx in 0..source.row_count {
        let global_row_idx = source.row_offset + row_idx;
        let start = source
            .data_offset
            .checked_add(global_row_idx * row_bytes)
            .and_then(|offset| offset.checked_add(source.col_offset * bytes_per_scalar))
            .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
        let end = start
            .checked_add(source.col_count * bytes_per_scalar)
            .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
        let encoded_row = mmap
            .get(start..end)
            .ok_or_else(|| anyhow!("matrix slice byte range is out of bounds"))?;
        if source.dtype == Dtype::F32 {
            let write_start = values.len();
            values.resize(write_start + source.col_count, 0.0);
            copy_f32_bytes_into_slice(encoded_row, &mut values[write_start..])?;
        } else {
            for encoded_value in encoded_row.chunks_exact(bytes_per_scalar) {
                values.push(decode_scalar(encoded_value, source.dtype)?);
            }
        }
    }
    Ok(MatrixF32 {
        rows: source.row_count,
        cols: source.col_count,
        values,
    })
}

fn decode_matrix_row_acts_from_det_num_source(
    source: &DetNumTensorSliceSource,
    row_idx: usize,
    mmap: &Mmap,
) -> Result<Vec<Act>> {
    if row_idx >= source.row_count {
        bail!(
            "matrix row {row_idx} is out of bounds for slice with {} rows",
            source.row_count
        );
    }
    let row_bytes = source
        .total_cols
        .checked_mul(4)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let global_row_idx = source.row_offset + row_idx;
    let start = source
        .data_offset
        .checked_add(global_row_idx * row_bytes)
        .and_then(|offset| offset.checked_add(source.col_offset * 4))
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let end = start
        .checked_add(source.col_count * 4)
        .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
    let encoded_row = mmap
        .get(start..end)
        .ok_or_else(|| anyhow!("matrix slice byte range is out of bounds"))?;
    let mut row = Vec::with_capacity(source.col_count);
    for encoded_value in encoded_row.chunks_exact(4) {
        row.push(Act::from_bits(i32::from_le_bytes(
            encoded_value
                .try_into()
                .expect("i32 byte width should match"),
        )));
    }
    Ok(row)
}

fn decode_matrix_slice_from_det_num_source(
    source: &DetNumTensorSliceSource,
    mmap: &Mmap,
) -> Result<MatrixF32> {
    let row_bytes = source
        .total_cols
        .checked_mul(4)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let mut values = Vec::with_capacity(source.row_count * source.col_count);
    for row_idx in 0..source.row_count {
        let global_row_idx = source.row_offset + row_idx;
        let start = source
            .data_offset
            .checked_add(global_row_idx * row_bytes)
            .and_then(|offset| offset.checked_add(source.col_offset * 4))
            .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
        let end = start
            .checked_add(source.col_count * 4)
            .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
        let encoded_row = mmap
            .get(start..end)
            .ok_or_else(|| anyhow!("matrix slice byte range is out of bounds"))?;
        for encoded_value in encoded_row.chunks_exact(4) {
            values.push(det_wgt_to_f32(i32::from_le_bytes(
                encoded_value
                    .try_into()
                    .expect("i32 byte width should match"),
            )));
        }
    }
    Ok(MatrixF32 {
        rows: source.row_count,
        cols: source.col_count,
        values,
    })
}

fn decode_det_num_matrix_from_source(
    source: &DetNumTensorSliceSource,
    mmap: &Mmap,
) -> Result<DetNumMatrix> {
    let row_bytes = source
        .total_cols
        .checked_mul(4)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let mut values = Vec::with_capacity(source.row_count * source.col_count);
    for row_idx in 0..source.row_count {
        let global_row_idx = source.row_offset + row_idx;
        let start = source
            .data_offset
            .checked_add(global_row_idx * row_bytes)
            .and_then(|offset| offset.checked_add(source.col_offset * 4))
            .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
        let end = start
            .checked_add(source.col_count * 4)
            .ok_or_else(|| anyhow!("matrix slice byte range overflowed"))?;
        let encoded_row = mmap
            .get(start..end)
            .ok_or_else(|| anyhow!("matrix slice byte range is out of bounds"))?;
        for encoded_value in encoded_row.chunks_exact(4) {
            values.push(i32::from_le_bytes(
                encoded_value
                    .try_into()
                    .expect("i32 byte width should match"),
            ));
        }
    }
    Ok(DetNumMatrix {
        rows: source.row_count,
        cols: source.col_count,
        values,
    })
}

fn matrix_row(matrix: &MatrixF32, row_idx: usize) -> Result<Vec<f32>> {
    if row_idx >= matrix.rows {
        bail!(
            "matrix row {row_idx} is out of bounds for matrix with {} rows",
            matrix.rows
        );
    }
    let start = row_idx
        .checked_mul(matrix.cols)
        .ok_or_else(|| anyhow!("matrix row start overflowed"))?;
    let end = start
        .checked_add(matrix.cols)
        .ok_or_else(|| anyhow!("matrix row end overflowed"))?;
    Ok(matrix.values[start..end].to_vec())
}

fn load_gemma4_layer_weights(
    reader: &mut GemmaTensorReader,
    config: &GemmaTextConfigFile,
    layer_idx: usize,
    is_kv_donor: bool,
) -> Result<Gemma4LayerWeights> {
    // let _trace = trace_scope(format!("io.load_gemma4_layer_weights layer={layer_idx}"));
    let layer_prefix = format!("model.language_model.layers.{layer_idx}");
    let attention_kind = config.attention_kind_for_layer(layer_idx)?;
    let is_sliding = attention_kind == Gemma4AttentionKind::Sliding;
    let head_dim = if is_sliding {
        config.head_dim
    } else {
        config.global_head_dim()
    };
    let num_kv_heads = if is_sliding {
        config.num_key_value_heads
    } else if config.attention_k_eq_v() {
        config
            .num_global_key_value_heads
            .unwrap_or(config.num_key_value_heads)
    } else {
        config.num_key_value_heads
    };
    let partial_rotary_dim = if is_sliding {
        head_dim
    } else {
        ((head_dim as f32) * config.full_attention_partial_rotary_factor()) as usize
    };
    let kv_shared_layer_index = kv_shared_layer_index(config, layer_idx)?;
    let effective_sliding_window = config.effective_sliding_window();
    let ple = if config.hidden_size_per_layer_input.unwrap_or(0) > 0 {
        Some(Gemma4PleLayerWeights {
            input_gate: reader
                .resolve_matrix_source(&format!("{layer_prefix}.per_layer_input_gate.weight"))?,
            layer_projection: reader
                .resolve_matrix_source(&format!("{layer_prefix}.per_layer_projection.weight"))?,
            post_input_norm_weight: reader
                .load_vector(&format!("{layer_prefix}.post_per_layer_input_norm.weight"))?,
            post_input_norm_weight_det: None,
        })
    } else {
        None
    };

    Ok(Gemma4LayerWeights {
        attention_kind,
        hidden_size: config.hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads,
        head_dim,
        sliding_window: is_sliding.then_some(effective_sliding_window),
        cache_sliding_window: if is_sliding && !is_kv_donor {
            Some(effective_sliding_window)
        } else {
            None
        },
        rms_norm_eps: config.rms_norm_eps,
        rms_norm_eps_det: None,
        rope_base: if is_sliding {
            config.rope_local_base_freq()
        } else {
            config.rope_full_base_freq()
        },
        rope_base_det: None,
        partial_rotary_dim,
        rope_freq_base_dim: head_dim,
        kv_shared_layer_index,
        attention_k_eq_v: !is_sliding && config.attention_k_eq_v(),
        q_proj: reader.resolve_matrix_source(&format!("{layer_prefix}.self_attn.q_proj.weight"))?,
        k_proj: reader.resolve_matrix_source(&format!("{layer_prefix}.self_attn.k_proj.weight"))?,
        v_proj: if !is_sliding && config.attention_k_eq_v() {
            reader.resolve_optional_matrix_source(&format!(
                "{layer_prefix}.self_attn.v_proj.weight"
            ))?
        } else {
            Some(reader.resolve_matrix_source(&format!("{layer_prefix}.self_attn.v_proj.weight"))?)
        },
        o_proj: reader.resolve_matrix_source(&format!("{layer_prefix}.self_attn.o_proj.weight"))?,
        q_norm_weight: reader.load_vector(&format!("{layer_prefix}.self_attn.q_norm.weight"))?,
        q_norm_weight_det: None,
        k_norm_weight: reader.load_vector(&format!("{layer_prefix}.self_attn.k_norm.weight"))?,
        k_norm_weight_det: None,
        input_layernorm_weight: reader
            .load_vector(&format!("{layer_prefix}.input_layernorm.weight"))?,
        input_layernorm_weight_det: None,
        post_attention_layernorm_weight: reader
            .load_vector(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
        post_attention_layernorm_weight_det: None,
        pre_feedforward_layernorm_weight: reader
            .load_vector(&format!("{layer_prefix}.pre_feedforward_layernorm.weight"))?,
        pre_feedforward_layernorm_weight_det: None,
        post_feedforward_layernorm_weight: reader
            .load_vector(&format!("{layer_prefix}.post_feedforward_layernorm.weight"))?,
        post_feedforward_layernorm_weight_det: None,
        gate_proj: reader.resolve_matrix_source(&format!("{layer_prefix}.mlp.gate_proj.weight"))?,
        up_proj: reader.resolve_matrix_source(&format!("{layer_prefix}.mlp.up_proj.weight"))?,
        down_proj: reader.resolve_matrix_source(&format!("{layer_prefix}.mlp.down_proj.weight"))?,
        ple,
        layer_scalar: reader.load_optional_scalar(&format!("{layer_prefix}.layer_scalar"))?,
        layer_scalar_det: None,
    })
}

fn load_det_num_gemma4_layer_weights(
    reader: &DetNumTensorReader,
    config: &GemmaTextConfigFile,
    layer_idx: usize,
    is_kv_donor: bool,
) -> Result<Gemma4LayerWeights> {
    let layer_prefix = format!("model.language_model.layers.{layer_idx}");
    let attention_kind = config.attention_kind_for_layer(layer_idx)?;
    let is_sliding = attention_kind == Gemma4AttentionKind::Sliding;
    let head_dim = if is_sliding {
        config.head_dim
    } else {
        config.global_head_dim()
    };
    let num_kv_heads = if is_sliding {
        config.num_key_value_heads
    } else if config.attention_k_eq_v() {
        config
            .num_global_key_value_heads
            .unwrap_or(config.num_key_value_heads)
    } else {
        config.num_key_value_heads
    };
    let partial_rotary_dim = if is_sliding {
        head_dim
    } else {
        ((head_dim as f32) * config.full_attention_partial_rotary_factor()) as usize
    };
    let kv_shared_layer_index = kv_shared_layer_index(config, layer_idx)?;
    let effective_sliding_window = config.effective_sliding_window();
    let ple = if config.hidden_size_per_layer_input.unwrap_or(0) > 0 {
        let post_input_norm_weight_det =
            reader.load_vector_wgt(&format!("{layer_prefix}.post_per_layer_input_norm.weight"))?;
        Some(Gemma4PleLayerWeights {
            input_gate: Gemma4LayerMatrixSource::from_det_num_source(
                reader.resolve_full_matrix_source(&format!(
                    "{layer_prefix}.per_layer_input_gate.weight"
                ))?,
            ),
            layer_projection: Gemma4LayerMatrixSource::from_det_num_source(
                reader.resolve_full_matrix_source(&format!(
                    "{layer_prefix}.per_layer_projection.weight"
                ))?,
            ),
            post_input_norm_weight: det_wgt_vec_to_f32(&post_input_norm_weight_det),
            post_input_norm_weight_det: Some(post_input_norm_weight_det),
        })
    } else {
        None
    };
    let rope_base = if is_sliding {
        config.rope_local_base_freq()
    } else {
        config.rope_full_base_freq()
    };
    let q_norm_weight_det =
        reader.load_vector_wgt(&format!("{layer_prefix}.self_attn.q_norm.weight"))?;
    let k_norm_weight_det =
        reader.load_vector_wgt(&format!("{layer_prefix}.self_attn.k_norm.weight"))?;
    let input_layernorm_weight_det =
        reader.load_vector_wgt(&format!("{layer_prefix}.input_layernorm.weight"))?;
    let post_attention_layernorm_weight_det =
        reader.load_vector_wgt(&format!("{layer_prefix}.post_attention_layernorm.weight"))?;
    let pre_feedforward_layernorm_weight_det =
        reader.load_vector_wgt(&format!("{layer_prefix}.pre_feedforward_layernorm.weight"))?;
    let post_feedforward_layernorm_weight_det =
        reader.load_vector_wgt(&format!("{layer_prefix}.post_feedforward_layernorm.weight"))?;
    let layer_scalar_det =
        reader.load_optional_scalar_wgt(&format!("{layer_prefix}.layer_scalar"))?;

    Ok(Gemma4LayerWeights {
        attention_kind,
        hidden_size: config.hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads,
        head_dim,
        sliding_window: is_sliding.then_some(effective_sliding_window),
        cache_sliding_window: if is_sliding && !is_kv_donor {
            Some(effective_sliding_window)
        } else {
            None
        },
        rms_norm_eps: config.rms_norm_eps,
        rms_norm_eps_det: Some(f32_to_acc(config.rms_norm_eps)),
        rope_base,
        rope_base_det: Some(f32_to_acc(rope_base)),
        partial_rotary_dim,
        rope_freq_base_dim: head_dim,
        kv_shared_layer_index,
        attention_k_eq_v: !is_sliding && config.attention_k_eq_v(),
        q_proj: Gemma4LayerMatrixSource::from_det_num_source(
            reader
                .resolve_full_matrix_source(&format!("{layer_prefix}.self_attn.q_proj.weight"))?,
        ),
        k_proj: Gemma4LayerMatrixSource::from_det_num_source(
            reader
                .resolve_full_matrix_source(&format!("{layer_prefix}.self_attn.k_proj.weight"))?,
        ),
        v_proj: if !is_sliding && config.attention_k_eq_v() {
            if reader
                .tensors
                .contains_key(&format!("{layer_prefix}.self_attn.v_proj.weight"))
            {
                Some(Gemma4LayerMatrixSource::from_det_num_source(
                    reader.resolve_full_matrix_source(&format!(
                        "{layer_prefix}.self_attn.v_proj.weight"
                    ))?,
                ))
            } else {
                None
            }
        } else {
            Some(Gemma4LayerMatrixSource::from_det_num_source(
                reader.resolve_full_matrix_source(&format!(
                    "{layer_prefix}.self_attn.v_proj.weight"
                ))?,
            ))
        },
        o_proj: Gemma4LayerMatrixSource::from_det_num_source(
            reader
                .resolve_full_matrix_source(&format!("{layer_prefix}.self_attn.o_proj.weight"))?,
        ),
        q_norm_weight: det_wgt_vec_to_f32(&q_norm_weight_det),
        q_norm_weight_det: Some(q_norm_weight_det),
        k_norm_weight: det_wgt_vec_to_f32(&k_norm_weight_det),
        k_norm_weight_det: Some(k_norm_weight_det),
        input_layernorm_weight: det_wgt_vec_to_f32(&input_layernorm_weight_det),
        input_layernorm_weight_det: Some(input_layernorm_weight_det),
        post_attention_layernorm_weight: det_wgt_vec_to_f32(&post_attention_layernorm_weight_det),
        post_attention_layernorm_weight_det: Some(post_attention_layernorm_weight_det),
        pre_feedforward_layernorm_weight: det_wgt_vec_to_f32(&pre_feedforward_layernorm_weight_det),
        pre_feedforward_layernorm_weight_det: Some(pre_feedforward_layernorm_weight_det),
        post_feedforward_layernorm_weight: det_wgt_vec_to_f32(
            &post_feedforward_layernorm_weight_det,
        ),
        post_feedforward_layernorm_weight_det: Some(post_feedforward_layernorm_weight_det),
        gate_proj: Gemma4LayerMatrixSource::from_det_num_source(
            reader.resolve_full_matrix_source(&format!("{layer_prefix}.mlp.gate_proj.weight"))?,
        ),
        up_proj: Gemma4LayerMatrixSource::from_det_num_source(
            reader.resolve_full_matrix_source(&format!("{layer_prefix}.mlp.up_proj.weight"))?,
        ),
        down_proj: Gemma4LayerMatrixSource::from_det_num_source(
            reader.resolve_full_matrix_source(&format!("{layer_prefix}.mlp.down_proj.weight"))?,
        ),
        ple,
        layer_scalar: layer_scalar_det.map(|value| det_wgt_to_f32(value.to_bits())),
        layer_scalar_det: layer_scalar_det.map(det_wgt_to_act),
    })
}

pub fn embed_input_tokens_from_gemma_source(
    token_ids: &[u32],
    source: &GemmaEmbeddingTensorSource,
) -> Result<ActivationSequence> {
    embed_input_tokens_from_gemma_source_with_mode(token_ids, source, InferenceExecutionMode::Fp32)
}

pub fn embed_input_tokens_from_gemma_source_with_mode(
    token_ids: &[u32],
    source: &GemmaEmbeddingTensorSource,
    execution_mode: InferenceExecutionMode,
) -> Result<ActivationSequence> {
    // let _trace = trace_scope("io.embed_input_tokens_from_gemma_source");
    if token_ids.is_empty() {
        bail!("transformer embedding requires at least one token id");
    }

    let internal = match source {
        GemmaEmbeddingTensorSource::Deterministic { source, scale, .. } => {
            let file = File::open(&source.weights_path).with_context(|| {
                format!(
                    "failed to open deterministic artifact {}",
                    source.weights_path.display()
                )
            })?;
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
                format!(
                    "failed to mmap deterministic artifact {}",
                    source.weights_path.display()
                )
            })?;
            decode_embedding_rows_for_token_ids_from_det_num(
                source,
                token_ids,
                source.col_count,
                *scale,
                &mmap,
                execution_mode,
            )?
        }
        _ if execution_mode == InferenceExecutionMode::Deterministic => {
            bail!("deterministic embedding requires a .detwgt embedding source")
        }
        _ => with_embedding_tensor(source, |tensor| {
            decode_embedding_rows_for_token_ids(
                tensor,
                token_ids,
                source.hidden_size(),
                source.scale(),
                execution_mode,
            )
        })?,
    };
    let activations = internal.clone_f32();
    let activations_sha256 = build_activation_commitment(&activations);
    let det_activations_sha256 = internal
        .det_values()
        .map(crate::shared::transformer_kernels::build_det_activation_commitment);

    let mut activation_sequence = ActivationSequence::from_internal(internal, activations_sha256);
    activation_sequence.det_activations_sha256 = det_activations_sha256;
    Ok(activation_sequence)
}

fn build_det_num_embedding_source(
    reader: &DetNumTensorReader,
    tensor_names: &[&str],
) -> Result<(String, GemmaEmbeddingTensorSource)> {
    let (tensor_name, source) = reader.load_first_available_matrix_source(tensor_names)?;
    let scale = (source.col_count as f32).sqrt();
    Ok((
        tensor_name,
        GemmaEmbeddingTensorSource::Deterministic {
            source,
            scale,
            det_cache: std::sync::Arc::new(std::sync::Mutex::new(None)),
        },
    ))
}

fn build_embedding_source(
    reader: &GemmaTensorReader,
    tensor_name: &str,
    hidden_size: usize,
) -> GemmaEmbeddingTensorSource {
    let scale = (hidden_size as f32).sqrt();
    match &reader.source {
        GemmaModelSource::Single { weights_path, .. } => GemmaEmbeddingTensorSource::Single {
            weights_path: weights_path.clone(),
            tensor_name: tensor_name.to_string(),
            hidden_size,
            scale,
        },
        GemmaModelSource::Indexed { root_dir, index } => GemmaEmbeddingTensorSource::Indexed {
            root_dir: root_dir.clone(),
            weight_map: index.weight_map.clone(),
            tensor_name: tensor_name.to_string(),
            hidden_size,
            scale,
        },
    }
}

fn resolve_gemma_model_source(path: &Path) -> Result<GemmaModelSource> {
    if path.is_dir() {
        let index_path = path.join("model.safetensors.index.json");
        if index_path.is_file() {
            return Ok(GemmaModelSource::Indexed {
                root_dir: path.to_path_buf(),
                index: load_safetensors_index(&index_path)?,
            });
        }

        for filename in ["model.safetensors", "consolidated.safetensors"] {
            let candidate = path.join(filename);
            if candidate.is_file() {
                return Ok(GemmaModelSource::Single {
                    root_dir: path.to_path_buf(),
                    weights_path: candidate,
                });
            }
        }
    }

    if path.extension().is_some_and(|ext| ext == "json") {
        return Ok(GemmaModelSource::Indexed {
            root_dir: path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            index: load_safetensors_index(path)?,
        });
    }

    if path.extension().is_some_and(|ext| ext == "safetensors") {
        return Ok(GemmaModelSource::Single {
            root_dir: path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            weights_path: path.to_path_buf(),
        });
    }

    bail!(
        "unsupported model path {}: expected a model directory, .safetensors file, or model.safetensors.index.json",
        path.display()
    )
}

fn resolve_det_num_model_source(path: &Path) -> Result<DetNumModelSource> {
    if path.is_dir() {
        let weights_path = path.join("model.detwgt");
        if !weights_path.is_file() {
            bail!(
                "deterministic model directory {} is missing model.detwgt",
                path.display()
            );
        }
        let config_path = path.join("config.json");
        if !config_path.is_file() {
            bail!(
                "deterministic model directory {} is missing config.json",
                path.display()
            );
        }
        return Ok(DetNumModelSource {
            root_dir: path.to_path_buf(),
            weights_path,
        });
    }

    if path.extension().is_some_and(|ext| ext == "detwgt") {
        let root_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        if !root_dir.join("config.json").is_file() {
            bail!(
                "could not locate config.json next to deterministic artifact {}",
                path.display()
            );
        }
        return Ok(DetNumModelSource {
            root_dir,
            weights_path: path.to_path_buf(),
        });
    }

    bail!(
        "unsupported deterministic model path {}: expected a directory containing model.detwgt or a direct .detwgt file",
        path.display()
    )
}

fn load_safetensors_index(path: &Path) -> Result<SafetensorsIndex> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read index file {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse index JSON from {}", path.display()))
}

fn load_gemma_text_config(path: PathBuf) -> Result<GemmaTextConfigFile> {
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read Gemma config from {}", path.display()))?;
    let config: GemmaConfigFile = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse Gemma config from {}", path.display()))?;
    Ok(config.text_config)
}

fn load_full_embedding_table_from_source(
    source: &GemmaEmbeddingTensorSource,
) -> Result<EmbeddingTable> {
    let matrix = match source {
        GemmaEmbeddingTensorSource::Deterministic { source, .. } => {
            let file = File::open(&source.weights_path).with_context(|| {
                format!(
                    "failed to open deterministic artifact {}",
                    source.weights_path.display()
                )
            })?;
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
                format!(
                    "failed to mmap deterministic artifact {}",
                    source.weights_path.display()
                )
            })?;
            decode_matrix_slice_from_det_num_source(source, &mmap)?
        }
        _ => with_embedding_tensor(source, |tensor| decode_matrix(tensor))?,
    };

    let mut rows = Vec::with_capacity(matrix.rows);
    for row_idx in 0..matrix.rows {
        rows.push(matrix.values[row_idx * matrix.cols..(row_idx + 1) * matrix.cols].to_vec());
    }

    Ok(EmbeddingTable {
        rows,
        scale: source.scale(),
    })
}

pub(crate) fn materialize_det_num_embedding_matrix(
    source: &GemmaEmbeddingTensorSource,
) -> Result<Option<std::sync::Arc<DetNumMatrix>>> {
    let (source, det_cache) = match source {
        GemmaEmbeddingTensorSource::Deterministic {
            source, det_cache, ..
        } => (source, det_cache),
        _ => return Ok(None),
    };

    if let Some(matrix) = det_cache
        .lock()
        .map_err(|_| anyhow!("deterministic embedding cache lock poisoned"))?
        .clone()
    {
        return Ok(Some(matrix));
    }

    let file = File::open(&source.weights_path).with_context(|| {
        format!(
            "failed to open deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
        format!(
            "failed to mmap deterministic artifact {}",
            source.weights_path.display()
        )
    })?;
    let matrix = std::sync::Arc::new(decode_det_num_matrix_from_source(source, &mmap)?);
    *det_cache
        .lock()
        .map_err(|_| anyhow!("deterministic embedding cache lock poisoned"))? =
        Some(matrix.clone());
    Ok(Some(matrix))
}

fn with_embedding_tensor<T>(
    source: &GemmaEmbeddingTensorSource,
    f: impl FnOnce(&TensorView<'_>) -> Result<T>,
) -> Result<T> {
    let path = match source {
        GemmaEmbeddingTensorSource::Single { weights_path, .. } => weights_path.clone(),
        GemmaEmbeddingTensorSource::Indexed {
            root_dir,
            weight_map,
            tensor_name,
            ..
        } => root_dir.join(
            weight_map
                .get(tensor_name)
                .ok_or_else(|| anyhow!("failed to locate tensor {tensor_name} in weight map"))?,
        ),
        GemmaEmbeddingTensorSource::Deterministic { .. } => {
            bail!("deterministic embedding sources are not backed by safetensors tensors")
        }
    };

    let file = File::open(&path)
        .with_context(|| format!("failed to open safetensors file {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap safetensors file {}", path.display()))?;
    let safetensors = SafeTensors::deserialize(mmap.as_ref())
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to deserialize safetensors file {}", path.display()))?;

    let tensor_name = match source {
        GemmaEmbeddingTensorSource::Single { tensor_name, .. }
        | GemmaEmbeddingTensorSource::Indexed { tensor_name, .. } => tensor_name.as_str(),
        GemmaEmbeddingTensorSource::Deterministic { .. } => {
            bail!("deterministic embedding sources are not backed by safetensors tensors")
        }
    };

    let tensor = safetensors
        .tensor(tensor_name)
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to load tensor {tensor_name} from {}",
                path.display()
            )
        })?;

    f(&tensor)
}

fn decode_embedding_rows_for_token_ids(
    tensor: &impl TensorBytes,
    token_ids: &[u32],
    hidden_size: usize,
    scale: f32,
    execution_mode: InferenceExecutionMode,
) -> Result<InternalActivationSequence> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        bail!("expected rank-2 embedding tensor, got shape {shape:?}");
    }
    if shape[1] != hidden_size {
        bail!(
            "embedding tensor width mismatch: expected {hidden_size}, got {}",
            shape[1]
        );
    }

    let bytes_per_scalar = bytes_per_scalar(tensor.dtype())?;
    let row_bytes = hidden_size
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow!("embedding row byte size overflowed"))?;

    let mut activations = Vec::with_capacity(token_ids.len());
    let mut acts = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        if row_idx >= shape[0] {
            bail!("token id {token_id} is out of bounds for embedding tensor");
        }
        let encoded_row = &tensor.data()[row_idx * row_bytes..(row_idx + 1) * row_bytes];
        let row = if tensor.dtype() == Dtype::F32 {
            decode_f32_bytes_to_vec(encoded_row)?
        } else {
            let mut row = Vec::with_capacity(hidden_size);
            for encoded_value in encoded_row.chunks_exact(bytes_per_scalar) {
                row.push(decode_scalar(encoded_value, tensor.dtype())?);
            }
            row
        };
        match execution_mode {
            InferenceExecutionMode::Fp32 => {
                activations.push(row.into_iter().map(|value| value * scale).collect());
            }
            InferenceExecutionMode::Deterministic => {
                acts.push(scale_act_row(
                    row.into_iter().map(f32_to_act).collect(),
                    scale,
                ));
            }
        }
    }

    Ok(match execution_mode {
        InferenceExecutionMode::Fp32 => InternalActivationSequence::from_values(activations),
        InferenceExecutionMode::Deterministic => InternalActivationSequence::from_det_values(acts),
    })
}

fn decode_embedding_rows_for_token_ids_from_det_num(
    source: &DetNumTensorSliceSource,
    token_ids: &[u32],
    hidden_size: usize,
    scale: f32,
    mmap: &Mmap,
    execution_mode: InferenceExecutionMode,
) -> Result<InternalActivationSequence> {
    if source.row_offset != 0
        || source.row_count != source.total_rows
        || source.col_offset != 0
        || source.col_count != source.total_cols
    {
        bail!("deterministic embedding source must reference the full embedding matrix");
    }
    if source.total_cols != hidden_size {
        bail!(
            "embedding tensor width mismatch: expected {hidden_size}, got {}",
            source.total_cols
        );
    }

    let row_bytes = hidden_size
        .checked_mul(4)
        .ok_or_else(|| anyhow!("embedding row byte size overflowed"))?;
    let mut activations = Vec::with_capacity(token_ids.len());
    let mut acts = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        if row_idx >= source.total_rows {
            bail!("token id {token_id} is out of bounds for embedding tensor");
        }
        let start = source
            .data_offset
            .checked_add(row_idx * row_bytes)
            .ok_or_else(|| anyhow!("embedding row byte range overflowed"))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| anyhow!("embedding row byte range overflowed"))?;
        let encoded_row = mmap
            .get(start..end)
            .ok_or_else(|| anyhow!("embedding row byte range is out of bounds"))?;
        match execution_mode {
            InferenceExecutionMode::Fp32 => {
                let mut row = Vec::with_capacity(hidden_size);
                for encoded_value in encoded_row.chunks_exact(4) {
                    row.push(det_wgt_to_f32(i32::from_le_bytes(
                        encoded_value
                            .try_into()
                            .expect("i32 byte width should match"),
                    )));
                }
                activations.push(row.into_iter().map(|value| value * scale).collect());
            }
            InferenceExecutionMode::Deterministic => {
                let mut row = Vec::with_capacity(hidden_size);
                for encoded_value in encoded_row.chunks_exact(4) {
                    row.push(Act::from_bits(i32::from_le_bytes(
                        encoded_value
                            .try_into()
                            .expect("i32 byte width should match"),
                    )));
                }
                acts.push(scale_act_row(row, scale));
            }
        }
    }

    Ok(match execution_mode {
        InferenceExecutionMode::Fp32 => InternalActivationSequence::from_values(activations),
        InferenceExecutionMode::Deterministic => InternalActivationSequence::from_det_values(acts),
    })
}

fn scale_act_row(row: Vec<Act>, scale: f32) -> Vec<Act> {
    let scale = f32_to_act(scale);
    row.into_iter()
        .map(|value| scale_act(value, scale))
        .collect()
}

fn build_activation_commitment(activations: &[Vec<f32>]) -> String {
    let mut hasher = sha2::Sha256::new();
    for row in activations {
        for value in row {
            hasher.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn decode_matrix(tensor: &impl TensorBytes) -> Result<MatrixF32> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        bail!("expected rank-2 tensor, got shape {shape:?}");
    }

    decode_matrix_slice(tensor, 0, shape[0], 0, shape[1])
}

fn decode_matrix_slice(
    tensor: &impl TensorBytes,
    row_offset: usize,
    row_count: usize,
    col_offset: usize,
    col_count: usize,
) -> Result<MatrixF32> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        bail!("expected rank-2 tensor, got shape {shape:?}");
    }
    let total_rows = shape[0];
    let total_cols = shape[1];
    if row_offset + row_count > total_rows || col_offset + col_count > total_cols {
        bail!(
            "matrix slice [{row_offset}..{}, {col_offset}..{}] is out of bounds for shape {shape:?}",
            row_offset + row_count,
            col_offset + col_count
        );
    }

    let bytes_per_scalar = bytes_per_scalar(tensor.dtype())?;
    let row_bytes = total_cols
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow!("matrix row byte size overflowed"))?;
    let expected_bytes = total_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| anyhow!("matrix byte size overflowed"))?;
    if tensor.data().len() != expected_bytes {
        bail!(
            "matrix tensor byte length mismatch: expected {expected_bytes}, got {}",
            tensor.data().len()
        );
    }

    let mut values = Vec::with_capacity(row_count * col_count);
    for row_idx in row_offset..row_offset + row_count {
        let encoded_row = &tensor.data()[row_idx * row_bytes..(row_idx + 1) * row_bytes];
        let row_values = &encoded_row
            [col_offset * bytes_per_scalar..(col_offset + col_count) * bytes_per_scalar];
        if tensor.dtype() == Dtype::F32 {
            let write_start = values.len();
            values.resize(write_start + col_count, 0.0);
            copy_f32_bytes_into_slice(row_values, &mut values[write_start..])?;
        } else {
            for encoded_value in row_values.chunks_exact(bytes_per_scalar) {
                values.push(decode_scalar(encoded_value, tensor.dtype())?);
            }
        }
    }

    Ok(MatrixF32 {
        rows: row_count,
        cols: col_count,
        values,
    })
}

fn decode_vector(tensor: &impl TensorBytes) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 1 {
        bail!("expected rank-1 tensor, got shape {shape:?}");
    }

    let bytes_per_scalar = bytes_per_scalar(tensor.dtype())?;
    let expected_bytes = shape[0]
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow!("vector byte size overflowed"))?;
    if tensor.data().len() != expected_bytes {
        bail!(
            "vector tensor byte length mismatch: expected {expected_bytes}, got {}",
            tensor.data().len()
        );
    }

    if tensor.dtype() == Dtype::F32 {
        return decode_f32_bytes_to_vec(tensor.data());
    }

    let mut values = Vec::with_capacity(shape[0]);
    for encoded_value in tensor.data().chunks_exact(bytes_per_scalar) {
        values.push(decode_scalar(encoded_value, tensor.dtype())?);
    }
    Ok(values)
}

fn decode_single_scalar(tensor: &impl TensorBytes) -> Result<f32> {
    let shape = tensor.shape();
    if shape != [1] {
        bail!("expected single-scalar tensor shaped [1], got {shape:?}");
    }

    if tensor.dtype() == Dtype::F32 {
        let mut value = [0.0];
        copy_f32_bytes_into_slice(tensor.data(), &mut value)?;
        return Ok(value[0]);
    }

    decode_scalar(tensor.data(), tensor.dtype())
}

fn bytes_per_scalar(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F16 | Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        Dtype::F64 => Ok(8),
        _ => bail!("unsupported tensor dtype {dtype:?}"),
    }
}

fn decode_f32_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    let len = bytes.len() / std::mem::size_of::<f32>();
    let mut values = vec![0.0; len];
    copy_f32_bytes_into_slice(bytes, &mut values)?;
    Ok(values)
}

fn copy_f32_bytes_into_slice(bytes: &[u8], target: &mut [f32]) -> Result<()> {
    let expected_bytes = target
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("F32 byte size overflowed"))?;
    if bytes.len() != expected_bytes {
        bail!(
            "F32 byte length mismatch: expected {expected_bytes}, got {}",
            bytes.len()
        );
    }

    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            target.as_mut_ptr().cast::<u8>(),
            expected_bytes,
        );
    }
    #[cfg(target_endian = "big")]
    for value in target.iter_mut() {
        *value = f32::from_bits(value.to_bits().swap_bytes());
    }
    Ok(())
}

fn decode_scalar(bytes: &[u8], dtype: Dtype) -> Result<f32> {
    match dtype {
        Dtype::F16 => Ok(f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()),
        Dtype::BF16 => Ok(bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()),
        Dtype::F32 => Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        Dtype::F64 => Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f32),
        _ => bail!("unsupported tensor dtype {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_embedding_rows_for_token_ids, decode_embedding_rows_for_token_ids_from_det_num,
        decode_matrix_row_from_source, decode_matrix_slice, decode_matrix_slice_from_source,
        decode_single_scalar, decode_vector, load_ple_model_projection,
        load_ple_token_embedding_row_internal, load_transformer_state_model_from_det_num_wgt_path,
        load_transformer_state_model_from_gemma_model_path, parse_gemma_tokenizer_spec_bytes,
        parse_safetensors_metadata,
    };
    use crate::shared::det_num::{
        f32_to_act, f32_to_wgt, wgt_to_le_bytes, Act, DET_NUM_SPEC_VERSION,
        DET_WGT_ARTIFACT_FORMAT_VERSION, DET_WGT_ARTIFACT_MAGIC,
    };
    use crate::shared::input::InferenceExecutionMode;
    use crate::shared::transformer::DetNumTensorSliceSource;
    use crate::{Gemma4AttentionKind, Gemma4LogitsProjection};
    use memmap2::Mmap;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use std::{
        collections::BTreeMap,
        fs,
        fs::File,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    struct FixtureTensor {
        name: String,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    #[test]
    fn parse_gemma_tokenizer_spec_accepts_supported_subset() {
        let spec = parse_gemma_tokenizer_spec_bytes(minimal_gemma_tokenizer_json().as_bytes())
            .expect("supported tokenizer should parse");

        assert_eq!(spec.token_id("<unk>"), Some(0));
        assert_eq!(spec.token_id("ab"), Some(3));
        assert_eq!(spec.merges.len(), 1);
        assert_eq!(spec.merges[0].left, "a");
        assert_eq!(spec.merges[0].right, "b");
        assert!(spec.byte_fallback);
    }

    #[test]
    fn parse_gemma_tokenizer_spec_rejects_unsupported_model_type() {
        let json =
            minimal_gemma_tokenizer_json().replace("\"type\":\"BPE\"", "\"type\":\"WordLevel\"");
        let error = parse_gemma_tokenizer_spec_bytes(json.as_bytes())
            .expect_err("unsupported tokenizer model should fail");

        assert!(error
            .to_string()
            .contains("unsupported Gemma tokenizer model"));
    }

    #[test]
    fn decode_matrix_slice_copies_f32_rows_without_scalar_loop() {
        let bytes = f32_to_bytes(&[
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
        ]);
        let tensor = TensorView::new(safetensors::Dtype::F32, vec![3, 4], &bytes).unwrap();

        let matrix = decode_matrix_slice(&tensor, 1, 2, 1, 2).unwrap();

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.values, vec![20.0, 30.0, 200.0, 300.0]);
    }

    #[test]
    fn decode_vector_and_scalar_copy_f32_values_directly() {
        let vector_bytes = f32_to_bytes(&[1.25, -2.5, 3.75]);
        let vector = TensorView::new(safetensors::Dtype::F32, vec![3], &vector_bytes).unwrap();
        let scalar_bytes = f32_to_bytes(&[9.5]);
        let scalar = TensorView::new(safetensors::Dtype::F32, vec![1], &scalar_bytes).unwrap();

        assert_eq!(decode_vector(&vector).unwrap(), vec![1.25, -2.5, 3.75]);
        assert_eq!(decode_single_scalar(&scalar).unwrap(), 9.5);
    }

    #[test]
    fn decode_embedding_rows_for_token_ids_applies_scale_after_f32_copy() {
        let bytes = f32_to_bytes(&[1.0, 2.0, -3.0, 4.0, 5.0, -6.0]);
        let tensor = TensorView::new(safetensors::Dtype::F32, vec![3, 2], &bytes).unwrap();

        let rows = decode_embedding_rows_for_token_ids(
            &tensor,
            &[2, 0],
            2,
            0.5,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();

        assert_eq!(rows.clone_f32(), vec![vec![2.5, -3.0], vec![0.5, 1.0]]);
    }

    #[test]
    fn decode_embedding_rows_for_token_ids_requantizes_scale_on_deterministic_path() {
        let bytes = f32_to_bytes(&[1.0 / 65_536.0]);
        let tensor = TensorView::new(safetensors::Dtype::F32, vec![1, 1], &bytes).unwrap();

        let fp32 = decode_embedding_rows_for_token_ids(
            &tensor,
            &[0],
            1,
            0.5,
            InferenceExecutionMode::Fp32,
        )
        .unwrap();
        let det = decode_embedding_rows_for_token_ids(
            &tensor,
            &[0],
            1,
            0.5,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(det.clone_f32(), vec![vec![0.0]]);
        assert!(fp32.as_f32_slice()[0][0] > det.as_f32_slice()[0][0]);
    }

    #[test]
    fn decode_det_num_embedding_rows_preserves_raw_act_bits() {
        let model_dir = create_test_model_dir("det-embedding-raw-act");
        let weights_path = model_dir.join("embedding.detwgt");
        let canonical = Act::from_bits((1 << 24) + 1);
        fs::write(&weights_path, canonical.to_bits().to_le_bytes()).unwrap();
        let file = File::open(&weights_path).unwrap();
        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let source = DetNumTensorSliceSource {
            weights_path,
            total_rows: 1,
            total_cols: 1,
            data_offset: 0,
            row_offset: 0,
            row_count: 1,
            col_offset: 0,
            col_count: 1,
        };

        let decoded = decode_embedding_rows_for_token_ids_from_det_num(
            &source,
            &[0],
            1,
            1.0,
            &mmap,
            InferenceExecutionMode::Deterministic,
        )
        .unwrap();

        assert_eq!(decoded.det_values().unwrap()[0][0], canonical);
        assert_ne!(f32_to_act(decoded.as_f32_slice()[0][0]), canonical);
    }

    #[test]
    fn decode_lazy_f32_ple_rows_and_slices() {
        let model_dir = create_test_model_dir("f32-lazy-slice");
        let tensor_name = "model.language_model.embed_tokens_per_layer.weight";
        let values = [
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
        ];
        write_model_file(
            &model_dir,
            &[FixtureTensor {
                name: tensor_name.to_string(),
                shape: vec![3, 4],
                bytes: f32_to_bytes(&values),
            }],
        );
        let model_path = model_dir.join("model.safetensors");
        let file = File::open(&model_path).unwrap();
        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let metadata = parse_safetensors_metadata(&mmap, &model_path).unwrap();
        let tensor = metadata.get(tensor_name).unwrap();
        let source = crate::shared::transformer::GemmaTensorSliceSource {
            weights_path: model_path,
            dtype: tensor.dtype,
            total_rows: tensor.shape[0],
            total_cols: tensor.shape[1],
            data_offset: tensor.data_offset,
            row_offset: 1,
            row_count: 2,
            col_offset: 1,
            col_count: 2,
        };

        let row = decode_matrix_row_from_source(&source, 1, &mmap).unwrap();
        let matrix = decode_matrix_slice_from_source(&source, &mmap).unwrap();

        assert_eq!(row, vec![200.0, 300.0]);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.values, vec![20.0, 30.0, 200.0, 300.0]);
    }

    #[test]
    fn load_transformer_model_loads_arbitrary_layers_and_untied_lm_head() {
        let model_dir = create_test_model_dir("untied");
        write_config(
            &model_dir,
            r#"{
  "text_config": {
    "enable_moe_block": false,
    "final_logit_softcapping": 7.5,
    "global_head_dim": 2,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention", "full_attention"],
    "num_global_key_value_heads": 1,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "rope_parameters": {
      "full_attention": { "partial_rotary_factor": 1.0, "rope_theta": 12345.0 },
      "sliding_attention": { "rope_theta": 10000.0 }
    },
    "sliding_window": 5,
    "tie_word_embeddings": false,
    "vocab_size": 3,
    "attention_k_eq_v": true
  }
}"#,
        );
        let mut tensors = vec![matrix_tensor(
            "model.language_model.embed_tokens.weight",
            &[3, 4],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.extend(layer_tensors(1, 2.0, false));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        tensors.push(matrix_tensor(
            "model.language_model.lm_head.weight",
            &[3, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ));
        tensors.push(scalar_tensor(
            "model.language_model.layers.1.layer_scalar",
            0.5,
        ));
        write_model_file(&model_dir, &tensors);

        let model = load_transformer_state_model_from_gemma_model_path(&model_dir).unwrap();
        let first_layer = crate::io::resolve_layer_weights(&model.layers[0]).unwrap();
        let second_layer = crate::io::resolve_layer_weights(&model.layers[1]).unwrap();

        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].attention_kind, Gemma4AttentionKind::Sliding);
        assert_eq!(model.layers[1].attention_kind, Gemma4AttentionKind::Full);
        assert_eq!(model.layers[0].rope_freq_base_dim, 2);
        assert_eq!(model.layers[1].rope_freq_base_dim, 2);
        assert_eq!(first_layer.q_proj.values[0], 1.0);
        assert_eq!(second_layer.q_proj.values[0], 2.0);
        assert!(second_layer.v_proj.is_none());
        assert_eq!(model.layers[1].layer_scalar, Some(0.5));
        assert_eq!(model.final_norm_weight, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(model.final_logit_softcapping, Some(7.5));
        match &model.logits_projection {
            Gemma4LogitsProjection::UntiedLmHead { weight, det_weight } => {
                assert_eq!(weight.rows, 3);
                assert_eq!(weight.cols, 4);
                assert!(det_weight.is_none());
            }
            other => panic!("expected untied lm head, got {other:?}"),
        }
    }

    #[test]
    fn resolve_layer_weights_only_attaches_raw_det_weights_for_det_attention_and_mlp_projections() {
        let fp32_dir = create_test_model_dir("fp32-mlp-proj");
        let det_dir = create_test_model_dir("det-mlp-proj");
        let config = r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 2,
    "tie_word_embeddings": false,
    "vocab_size": 3
  }
}"#;
        write_config(&fp32_dir, config);
        write_config(&det_dir, config);

        let q_proj_values = [
            0.5, -0.25, 0.125, 0.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25, -0.125,
            0.0, 0.0,
        ];
        let k_proj_values = [0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.125, 0.0];
        let v_proj_values = [1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0];
        let o_proj_values = [
            -0.5, 0.25, -0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25,
            -0.125, 0.0,
        ];
        let lm_head_values = [
            0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0,
        ];
        let gate_proj_values = [
            0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25,
            -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        let up_proj_values = [
            0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5,
            -0.75, 0.0, 0.5, 0.25, -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        let down_proj_values = [
            0.5, -0.25, 0.125, 0.0, 1.0, -1.0, 0.75, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5, 0.25,
            -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        write_model_file(
            &fp32_dir,
            &[
                matrix_tensor(
                    "model.language_model.embed_tokens.weight",
                    &[3, 4],
                    &[0.0; 12],
                ),
                matrix_tensor(
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    &[4, 4],
                    &q_proj_values,
                ),
                matrix_tensor(
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    &[2, 4],
                    &k_proj_values,
                ),
                matrix_tensor(
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    &[2, 4],
                    &v_proj_values,
                ),
                matrix_tensor(
                    "model.language_model.layers.0.self_attn.o_proj.weight",
                    &[4, 4],
                    &o_proj_values,
                ),
                vector_tensor(
                    "model.language_model.layers.0.self_attn.q_norm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
                vector_tensor(
                    "model.language_model.layers.0.self_attn.k_norm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
                vector_tensor(
                    "model.language_model.layers.0.input_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                vector_tensor(
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                vector_tensor(
                    "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                vector_tensor(
                    "model.language_model.layers.0.post_feedforward_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                matrix_tensor(
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    &[8, 4],
                    &gate_proj_values,
                ),
                matrix_tensor(
                    "model.language_model.layers.0.mlp.up_proj.weight",
                    &[8, 4],
                    &up_proj_values,
                ),
                matrix_tensor(
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    &[4, 8],
                    &down_proj_values,
                ),
                vector_tensor("model.language_model.norm.weight", &[4], &[1.0; 4]),
                matrix_tensor(
                    "model.language_model.lm_head.weight",
                    &[3, 4],
                    &lm_head_values,
                ),
            ],
        );
        write_detwgt_file(
            &det_dir,
            &[
                det_matrix_tensor(
                    "model.language_model.embed_tokens.weight",
                    &[3, 4],
                    &[0.0; 12],
                ),
                det_matrix_tensor(
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    &[4, 4],
                    &q_proj_values,
                ),
                det_matrix_tensor(
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    &[2, 4],
                    &k_proj_values,
                ),
                det_matrix_tensor(
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    &[2, 4],
                    &v_proj_values,
                ),
                det_matrix_tensor(
                    "model.language_model.layers.0.self_attn.o_proj.weight",
                    &[4, 4],
                    &o_proj_values,
                ),
                det_vector_tensor(
                    "model.language_model.layers.0.self_attn.q_norm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
                det_vector_tensor(
                    "model.language_model.layers.0.self_attn.k_norm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
                det_vector_tensor(
                    "model.language_model.layers.0.input_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                det_vector_tensor(
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                det_vector_tensor(
                    "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                det_vector_tensor(
                    "model.language_model.layers.0.post_feedforward_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                det_matrix_tensor(
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    &[8, 4],
                    &gate_proj_values,
                ),
                det_matrix_tensor(
                    "model.language_model.layers.0.mlp.up_proj.weight",
                    &[8, 4],
                    &up_proj_values,
                ),
                det_matrix_tensor(
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    &[4, 8],
                    &down_proj_values,
                ),
                det_vector_tensor("model.language_model.norm.weight", &[4], &[1.0; 4]),
                det_matrix_tensor(
                    "model.language_model.lm_head.weight",
                    &[3, 4],
                    &lm_head_values,
                ),
            ],
        );

        let fp32_model = load_transformer_state_model_from_gemma_model_path(&fp32_dir).unwrap();
        let det_model = load_transformer_state_model_from_det_num_wgt_path(&det_dir).unwrap();
        let fp32_layer = super::resolve_layer_weights(&fp32_model.layers[0]).unwrap();
        let det_layer = super::resolve_layer_weights(&det_model.layers[0]).unwrap();
        let fp32_lm_head = match &fp32_model.logits_projection {
            Gemma4LogitsProjection::UntiedLmHead { weight, det_weight } => {
                assert!(det_weight.is_none());
                weight
            }
            other => panic!("expected untied lm head, got {other:?}"),
        };
        let det_lm_head = match &det_model.logits_projection {
            Gemma4LogitsProjection::UntiedLmHead { weight, det_weight } => {
                assert_eq!(weight.rows, 3);
                assert_eq!(weight.cols, 4);
                det_weight
                    .as_ref()
                    .expect("deterministic model should retain raw lm_head weights")
            }
            other => panic!("expected untied lm head, got {other:?}"),
        };

        assert!(fp32_layer.q_proj_det.is_none());
        assert!(fp32_layer.k_proj_det.is_none());
        assert!(fp32_layer.v_proj_det.is_none());
        assert!(fp32_layer.o_proj_det.is_none());
        assert!(fp32_layer.gate_proj_det.is_none());
        assert!(fp32_layer.up_proj_det.is_none());
        assert!(fp32_layer.down_proj_det.is_none());
        let det_q_proj = det_layer
            .q_proj_det
            .expect("deterministic model should retain raw q_proj weights");
        let det_k_proj = det_layer
            .k_proj_det
            .expect("deterministic model should retain raw k_proj weights");
        let det_v_proj = det_layer
            .v_proj_det
            .expect("deterministic model should retain raw v_proj weights");
        let det_o_proj = det_layer
            .o_proj_det
            .expect("deterministic model should retain raw o_proj weights");
        let det_gate_proj = det_layer
            .gate_proj_det
            .expect("deterministic model should retain raw gate_proj weights");
        let det_up_proj = det_layer
            .up_proj_det
            .expect("deterministic model should retain raw up_proj weights");
        let det_down_proj = det_layer
            .down_proj_det
            .expect("deterministic model should retain raw down_proj weights");
        assert_eq!(det_q_proj.rows, 4);
        assert_eq!(det_q_proj.cols, 4);
        assert_eq!(det_q_proj.values[0], f32_to_wgt(q_proj_values[0]).to_bits());
        assert_eq!(det_q_proj.values[1], f32_to_wgt(q_proj_values[1]).to_bits());
        assert_eq!(det_k_proj.rows, 2);
        assert_eq!(det_k_proj.cols, 4);
        assert_eq!(det_k_proj.values[0], f32_to_wgt(k_proj_values[0]).to_bits());
        assert_eq!(det_k_proj.values[1], f32_to_wgt(k_proj_values[1]).to_bits());
        assert_eq!(det_v_proj.rows, 2);
        assert_eq!(det_v_proj.cols, 4);
        assert_eq!(det_v_proj.values[0], f32_to_wgt(v_proj_values[0]).to_bits());
        assert_eq!(det_v_proj.values[1], f32_to_wgt(v_proj_values[1]).to_bits());
        assert_eq!(det_o_proj.rows, 4);
        assert_eq!(det_o_proj.cols, 4);
        assert_eq!(det_o_proj.values[0], f32_to_wgt(o_proj_values[0]).to_bits());
        assert_eq!(det_o_proj.values[1], f32_to_wgt(o_proj_values[1]).to_bits());
        assert_eq!(fp32_lm_head.values[0], lm_head_values[0]);
        assert_eq!(det_lm_head.rows, 3);
        assert_eq!(det_lm_head.cols, 4);
        assert_eq!(
            det_lm_head.values[0],
            f32_to_wgt(lm_head_values[0]).to_bits()
        );
        assert_eq!(
            det_lm_head.values[1],
            f32_to_wgt(lm_head_values[1]).to_bits()
        );
        assert_eq!(det_gate_proj.rows, 8);
        assert_eq!(det_gate_proj.cols, 4);
        assert_eq!(
            det_gate_proj.values[0],
            f32_to_wgt(gate_proj_values[0]).to_bits()
        );
        assert_eq!(
            det_gate_proj.values[1],
            f32_to_wgt(gate_proj_values[1]).to_bits()
        );
        assert_eq!(det_up_proj.rows, 8);
        assert_eq!(det_up_proj.cols, 4);
        assert_eq!(
            det_up_proj.values[0],
            f32_to_wgt(up_proj_values[0]).to_bits()
        );
        assert_eq!(
            det_up_proj.values[1],
            f32_to_wgt(up_proj_values[1]).to_bits()
        );
        assert_eq!(det_down_proj.rows, 4);
        assert_eq!(det_down_proj.cols, 8);
        assert_eq!(
            det_down_proj.values[0],
            f32_to_wgt(down_proj_values[0]).to_bits()
        );
        assert_eq!(
            det_down_proj.values[1],
            f32_to_wgt(down_proj_values[1]).to_bits()
        );
    }

    #[test]
    fn resolve_layer_weights_and_ple_globals_attach_raw_det_ple_weights() {
        let fp32_dir = create_test_model_dir("fp32-ple-proj");
        let det_dir = create_test_model_dir("det-ple-proj");
        let config = r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "hidden_size_per_layer_input": 2,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 2,
    "tie_word_embeddings": false,
    "vocab_size": 3,
    "vocab_size_per_layer_input": 3
  }
}"#;
        write_config(&fp32_dir, config);
        write_config(&det_dir, config);

        let input_gate_values = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let layer_projection_values = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let global_projection_values = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let mut fp32_tensors = vec![matrix_tensor(
            "model.language_model.embed_tokens.weight",
            &[3, 4],
            &[0.0; 12],
        )];
        fp32_tensors.extend(layer_tensors(0, 0.0, true));
        fp32_tensors.push(matrix_tensor(
            "model.language_model.layers.0.per_layer_input_gate.weight",
            &[2, 4],
            &input_gate_values,
        ));
        fp32_tensors.push(matrix_tensor(
            "model.language_model.layers.0.per_layer_projection.weight",
            &[4, 2],
            &layer_projection_values,
        ));
        fp32_tensors.push(matrix_tensor(
            "model.language_model.embed_tokens_per_layer.weight",
            &[3, 2],
            &[0.0; 6],
        ));
        fp32_tensors.push(matrix_tensor(
            "model.language_model.per_layer_model_projection.weight",
            &[2, 4],
            &global_projection_values,
        ));
        fp32_tensors.push(vector_tensor(
            "model.language_model.layers.0.post_per_layer_input_norm.weight",
            &[4],
            &[1.0; 4],
        ));
        fp32_tensors.push(vector_tensor(
            "model.language_model.per_layer_projection_norm.weight",
            &[2],
            &[1.0, 1.0],
        ));
        fp32_tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0; 4],
        ));
        fp32_tensors.push(matrix_tensor(
            "model.language_model.lm_head.weight",
            &[3, 4],
            &[0.0; 12],
        ));
        write_model_file(&fp32_dir, &fp32_tensors);

        let mut det_tensors = vec![det_matrix_tensor(
            "model.language_model.embed_tokens.weight",
            &[3, 4],
            &[0.0; 12],
        )];
        det_tensors.extend([
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                &[4, 4],
                &[0.0; 16],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.k_proj.weight",
                &[2, 4],
                &[0.0; 8],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.v_proj.weight",
                &[2, 4],
                &[0.0; 8],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.self_attn.o_proj.weight",
                &[4, 4],
                &[0.0; 16],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.self_attn.q_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.self_attn.k_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.input_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.post_attention_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_vector_tensor(
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                &[8, 4],
                &[0.0; 32],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.mlp.up_proj.weight",
                &[8, 4],
                &[0.0; 32],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.mlp.down_proj.weight",
                &[4, 8],
                &[0.0; 32],
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.per_layer_input_gate.weight",
                &[2, 4],
                &input_gate_values,
            ),
            det_matrix_tensor(
                "model.language_model.layers.0.per_layer_projection.weight",
                &[4, 2],
                &layer_projection_values,
            ),
            det_vector_tensor(
                "model.language_model.layers.0.post_per_layer_input_norm.weight",
                &[4],
                &[1.0; 4],
            ),
            det_matrix_tensor(
                "model.language_model.embed_tokens_per_layer.weight",
                &[3, 2],
                &[0.0; 6],
            ),
            det_matrix_tensor(
                "model.language_model.per_layer_model_projection.weight",
                &[2, 4],
                &global_projection_values,
            ),
            det_vector_tensor(
                "model.language_model.per_layer_projection_norm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            det_vector_tensor("model.language_model.norm.weight", &[4], &[1.0; 4]),
            det_matrix_tensor("model.language_model.lm_head.weight", &[3, 4], &[0.0; 12]),
        ]);
        write_detwgt_file(&det_dir, &det_tensors);

        let fp32_model = load_transformer_state_model_from_gemma_model_path(&fp32_dir).unwrap();
        let det_model = load_transformer_state_model_from_det_num_wgt_path(&det_dir).unwrap();
        let fp32_layer = super::resolve_layer_weights(&fp32_model.layers[0]).unwrap();
        let det_layer = super::resolve_layer_weights(&det_model.layers[0]).unwrap();
        let fp32_ple = fp32_layer.ple.expect("fp32 PLE should resolve");
        let det_ple = det_layer.ple.expect("det PLE should resolve");
        let det_global_projection = super::materialize_det_num_ple_model_projection(
            det_model
                .ple_global
                .as_ref()
                .expect("det PLE globals should load"),
            0,
        )
        .unwrap()
        .expect("det PLE model projection should materialize");

        assert!(fp32_ple.input_gate_det.is_none());
        assert!(fp32_ple.layer_projection_det.is_none());
        assert_eq!(det_ple.input_gate_det.as_ref().unwrap().rows, 2);
        assert_eq!(det_ple.input_gate_det.as_ref().unwrap().cols, 4);
        assert_eq!(
            det_ple.input_gate_det.as_ref().unwrap().values[0],
            f32_to_wgt(input_gate_values[0]).to_bits()
        );
        assert_eq!(det_ple.layer_projection_det.as_ref().unwrap().rows, 4);
        assert_eq!(det_ple.layer_projection_det.as_ref().unwrap().cols, 2);
        assert_eq!(
            det_ple.layer_projection_det.as_ref().unwrap().values[0],
            f32_to_wgt(layer_projection_values[0]).to_bits()
        );
        assert_eq!(det_global_projection.rows, 2);
        assert_eq!(det_global_projection.cols, 4);
        assert_eq!(
            det_global_projection.values[0],
            f32_to_wgt(global_projection_values[0]).to_bits()
        );
    }

    #[test]
    fn load_transformer_model_reuses_embeddings_for_tied_projection() {
        let model_dir = create_test_model_dir("tied");
        write_config(
            &model_dir,
            r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 5,
    "tie_word_embeddings": true,
    "vocab_size": 3
  }
}"#,
        );
        let mut tensors = vec![matrix_tensor(
            "model.language_model.embed_tokens.weight",
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        write_model_file(&model_dir, &tensors);

        let model = load_transformer_state_model_from_gemma_model_path(&model_dir).unwrap();

        match &model.logits_projection {
            Gemma4LogitsProjection::TiedEmbedding(weight) => {
                assert_eq!(weight.rows, 3);
                assert_eq!(weight.cols, 4);
                assert_eq!(weight.values[0], 1.0);
                assert_eq!(weight.values[11], 12.0);
            }
            other => panic!("expected tied embedding projection, got {other:?}"),
        }
    }

    #[test]
    fn load_transformer_model_builds_lazy_ple_sources() {
        let model_dir = create_test_model_dir("lazy-ple");
        write_config(
            &model_dir,
            r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "hidden_size_per_layer_input": 2,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 5,
    "tie_word_embeddings": true,
    "vocab_size": 3,
    "vocab_size_per_layer_input": 3
  }
}"#,
        );
        let mut tensors = vec![matrix_tensor(
            "model.language_model.embed_tokens.weight",
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.extend(ple_layer_tensors(0, 1.0));
        tensors.push(matrix_tensor(
            "model.language_model.embed_tokens_per_layer.weight",
            &[3, 2],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ));
        tensors.push(matrix_tensor(
            "model.language_model.per_layer_model_projection.weight",
            &[2, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ));
        tensors.push(vector_tensor(
            "model.language_model.per_layer_projection_norm.weight",
            &[2],
            &[1.0, 1.0],
        ));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        write_model_file(&model_dir, &tensors);

        let model = load_transformer_state_model_from_gemma_model_path(&model_dir).unwrap();
        let ple_global = model.ple_global.as_ref().expect("PLE globals should load");

        assert_eq!(ple_global.token_embedding_layer_count(), 1);
        assert_eq!(ple_global.model_projection_layer_count(), 1);
        assert_eq!(ple_global.projection_norm_weight, vec![1.0, 1.0]);
        assert_eq!(ple_global.embedding_scale, 2f32.sqrt());
        assert_eq!(
            load_ple_token_embedding_row_internal(ple_global, 0, 1)
                .unwrap()
                .clone_f32(),
            vec![0.3, 0.4]
        );
        let projection = load_ple_model_projection(ple_global, 0).unwrap();
        assert_eq!(projection.rows, 2);
        assert_eq!(projection.cols, 4);
        assert_eq!(
            projection.values,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn load_transformer_model_parses_effective_window_and_kv_sharing_metadata() {
        let model_dir = create_test_model_dir("shared-kv");
        write_config(
            &model_dir,
            r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention", "sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "num_kv_shared_layers": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 5,
    "tie_word_embeddings": true,
    "use_bidirectional_attention": "all",
    "vocab_size": 3
  }
}"#,
        );
        let mut tensors = vec![matrix_tensor(
            "model.language_model.embed_tokens.weight",
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.extend(layer_tensors(1, 2.0, true));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        write_model_file(&model_dir, &tensors);

        let model = load_transformer_state_model_from_gemma_model_path(&model_dir).unwrap();

        assert_eq!(model.layers[0].sliding_window, Some(3));
        assert_eq!(model.layers[0].cache_sliding_window, None);
        assert_eq!(model.layers[0].kv_shared_layer_index, None);
        assert_eq!(model.layers[1].sliding_window, Some(3));
        assert_eq!(model.layers[1].kv_shared_layer_index, Some(0));
    }

    #[test]
    fn load_transformer_model_rejects_kv_sharing_without_a_donor_layer() {
        let model_dir = create_test_model_dir("invalid-shared-kv");
        write_config(
            &model_dir,
            r#"{
  "text_config": {
    "enable_moe_block": false,
    "head_dim": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 4,
    "layer_types": ["sliding_attention"],
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "num_kv_shared_layers": 1,
    "rms_norm_eps": 0.000001,
    "sliding_window": 5,
    "tie_word_embeddings": true,
    "vocab_size": 3
  }
}"#,
        );
        let mut tensors = vec![matrix_tensor(
            "model.language_model.embed_tokens.weight",
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        write_model_file(&model_dir, &tensors);

        let error = load_transformer_state_model_from_gemma_model_path(&model_dir)
            .expect_err("missing donor should fail");

        assert!(error.to_string().contains("share KV without a prior"));
    }

    fn create_test_model_dir(suffix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("raster-inference-{suffix}-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, config: &str) {
        fs::write(dir.join("config.json"), config).unwrap();
    }

    fn write_model_file(dir: &Path, tensors: &[FixtureTensor]) {
        let mut metadata = BTreeMap::new();
        for tensor in tensors {
            metadata.insert(
                tensor.name.clone(),
                TensorView::new(safetensors::Dtype::F32, tensor.shape.clone(), &tensor.bytes)
                    .unwrap(),
            );
        }
        serialize_to_file(&metadata, &None, &dir.join("model.safetensors")).unwrap();
    }

    fn write_detwgt_file(dir: &Path, tensors: &[FixtureTensor]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(DET_WGT_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&DET_NUM_SPEC_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

        for tensor in tensors {
            let name_bytes = tensor.name.as_bytes();
            let element_count = tensor.shape.iter().product::<usize>() as u64;
            bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(name_bytes);
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dim in &tensor.shape {
                bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            bytes.extend_from_slice(&element_count.to_le_bytes());
            bytes.extend_from_slice(&(tensor.bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&tensor.bytes);
        }

        fs::write(dir.join("model.detwgt"), bytes).unwrap();
    }

    fn layer_tensors(layer_idx: usize, base: f32, include_v_proj: bool) -> Vec<FixtureTensor> {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        let mut tensors = vec![
            matrix_tensor(
                &format!("{prefix}.self_attn.q_proj.weight"),
                &[4, 4],
                &[base; 16],
            ),
            matrix_tensor(
                &format!("{prefix}.self_attn.k_proj.weight"),
                &[2, 4],
                &[base; 8],
            ),
            matrix_tensor(
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[4, 4],
                &[base; 16],
            ),
            vector_tensor(
                &format!("{prefix}.self_attn.q_norm.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            vector_tensor(
                &format!("{prefix}.self_attn.k_norm.weight"),
                &[2],
                &[1.0, 1.0],
            ),
            vector_tensor(&format!("{prefix}.input_layernorm.weight"), &[4], &[1.0; 4]),
            vector_tensor(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[4],
                &[1.0; 4],
            ),
            vector_tensor(
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
                &[4],
                &[1.0; 4],
            ),
            vector_tensor(
                &format!("{prefix}.post_feedforward_layernorm.weight"),
                &[4],
                &[1.0; 4],
            ),
            matrix_tensor(
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[8, 4],
                &[0.0; 32],
            ),
            matrix_tensor(&format!("{prefix}.mlp.up_proj.weight"), &[8, 4], &[0.0; 32]),
            matrix_tensor(
                &format!("{prefix}.mlp.down_proj.weight"),
                &[4, 8],
                &[0.0; 32],
            ),
        ];
        if include_v_proj {
            tensors.push(matrix_tensor(
                &format!("{prefix}.self_attn.v_proj.weight"),
                &[2, 4],
                &[base; 8],
            ));
        }
        tensors
    }

    fn ple_layer_tensors(layer_idx: usize, base: f32) -> Vec<FixtureTensor> {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        vec![
            matrix_tensor(
                &format!("{prefix}.per_layer_input_gate.weight"),
                &[2, 4],
                &[base; 8],
            ),
            matrix_tensor(
                &format!("{prefix}.per_layer_projection.weight"),
                &[4, 2],
                &[base; 8],
            ),
            vector_tensor(
                &format!("{prefix}.post_per_layer_input_norm.weight"),
                &[4],
                &[1.0, 1.0, 1.0, 1.0],
            ),
        ]
    }

    fn matrix_tensor(name: &str, shape: &[usize], values: &[f32]) -> FixtureTensor {
        FixtureTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            bytes: f32_to_bytes(values),
        }
    }

    fn vector_tensor(name: &str, shape: &[usize], values: &[f32]) -> FixtureTensor {
        FixtureTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            bytes: f32_to_bytes(values),
        }
    }

    fn det_matrix_tensor(name: &str, shape: &[usize], values: &[f32]) -> FixtureTensor {
        FixtureTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            bytes: det_wgt_bytes(values),
        }
    }

    fn det_vector_tensor(name: &str, shape: &[usize], values: &[f32]) -> FixtureTensor {
        FixtureTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            bytes: det_wgt_bytes(values),
        }
    }

    fn scalar_tensor(name: &str, value: f32) -> FixtureTensor {
        FixtureTensor {
            name: name.to_string(),
            shape: vec![1],
            bytes: value.to_le_bytes().to_vec(),
        }
    }

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn det_wgt_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| wgt_to_le_bytes(f32_to_wgt(*value)))
            .collect()
    }

    fn minimal_gemma_tokenizer_json() -> String {
        serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {
                    "id": 4,
                    "content": "<bos>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": {
                "type": "Replace",
                "pattern": { "String": " " },
                "content": "▁"
            },
            "pre_tokenizer": {
                "type": "Split",
                "pattern": { "String": " " },
                "behavior": "MergedWithPrevious",
                "invert": false
            },
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [],
                "pair": [],
                "special_tokens": {}
            },
            "decoder": {
                "type": "Sequence",
                "decoders": []
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<unk>",
                "fuse_unk": true,
                "byte_fallback": true,
                "ignore_merges": false,
                "vocab": {
                    "<unk>": 0,
                    "a": 1,
                    "b": 2,
                    "ab": 3,
                    "<bos>": 4
                },
                "merges": [
                    ["a", "b"]
                ]
            }
        })
        .to_string()
    }
}
