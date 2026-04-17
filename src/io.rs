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

use crate::phase2::{
    ActivationSequence, EmbeddingTable, Gemma4AttentionKind, Gemma4LayerWeights,
    Gemma4LogitsProjection, Gemma4Phase2Model, Gemma4PleGlobalWeights, Gemma4PleLayerWeights,
    GemmaEmbeddingTensorSource, MatrixF32,
};
use crate::phase2::types::{Gemma4PleMatrixSource, GemmaTensorSliceSource};
use crate::trace::{trace_event, trace_scope};

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
            anyhow!("failed to load tensor {tensor_name} from {}", path.display())
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

    fn tensor_metadata<'a>(&'a self, tensor_name: &str, path: &Path) -> Result<&'a CachedTensorMetadata> {
        self.tensors.get(tensor_name).ok_or_else(|| {
            anyhow!("failed to load tensor {tensor_name} from {}", path.display())
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

    fn load_optional_matrix(&mut self, tensor_name: &str) -> Result<Option<MatrixF32>> {
        match self.load_tensor(tensor_name) {
            Ok(tensor) => Ok(Some(decode_matrix(&tensor)?)),
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

        self.cached_files
            .get(path)
            .ok_or_else(|| anyhow!("failed to cache safetensors metadata for {}", path.display()))
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
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| anyhow!("safetensors header length overflowed for {}", path.display()))?;
    let header_bytes = mmap
        .get(8..header_end)
        .ok_or_else(|| anyhow!("safetensors header is out of bounds for {}", path.display()))?;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(header_bytes)
            .with_context(|| format!("failed to parse safetensors header from {}", path.display()))?;
    let data_section_offset = header_end;
    let mut tensors = HashMap::new();
    for (tensor_name, raw_entry) in header {
        if tensor_name == "__metadata__" {
            continue;
        }
        let entry: SafetensorsHeaderTensor = serde_json::from_value(raw_entry).with_context(|| {
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

pub fn load_embedding_table_from_gemma_model_path<P: AsRef<Path>>(path: P) -> Result<EmbeddingTable> {
    let mut model = load_phase2_model_from_gemma_model_path(path)?;
    let source = model
        .embedding_source
        .take()
        .ok_or_else(|| anyhow!("phase 2 model is missing an embedding source"))?;
    load_full_embedding_table_from_source(&source)
}

pub fn load_phase2_model_from_gemma_model_path<P: AsRef<Path>>(path: P) -> Result<Gemma4Phase2Model> {
    let _trace = trace_scope("io.load_phase2_model_from_gemma_model_path");
    let source = resolve_gemma_model_source(path.as_ref())?;
    let config = load_gemma_text_config(source.root_dir().join("config.json"))?;

    if config.enable_moe_block {
        bail!("phase 2 currently only supports Gemma 4 dense layers, not MoE checkpoints");
    }
    if config.hidden_activation != "gelu_pytorch_tanh" {
        bail!(
            "phase 2 currently only supports gelu_pytorch_tanh, got {}",
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
        trace_event(format!("io.load_gemma4_layer_weights layer={layer_idx}"));
        layers.push(load_gemma4_layer_weights(
            &mut reader,
            &config,
            layer_idx,
            kv_donor_layers.contains(&layer_idx),
        )?);
    }
    let ple_global = load_ple_global_weights(&mut reader, &config)?;
    trace_event("io.load_final_norm_weight");
    let final_norm_weight = reader.load_vector("model.language_model.norm.weight")?;
    trace_event("io.load_logits_projection");
    let logits_projection = if config.tie_word_embeddings() {
        Gemma4LogitsProjection::TiedEmbedding(reader.load_matrix(&embedding_tensor_name)?)
    } else {
        Gemma4LogitsProjection::UntiedLmHead(
            reader.load_matrix("model.language_model.lm_head.weight")?,
        )
    };
    let embedding_source = build_embedding_source(&reader, &embedding_tensor_name, hidden_size);

    Ok(Gemma4Phase2Model {
        embedding_table: None,
        embedding_source: Some(embedding_source),
        layers,
        ple_global,
        final_norm_weight,
        logits_projection,
        final_logit_softcapping: config.final_logit_softcapping,
        rms_norm_eps: config.rms_norm_eps,
    })
}

fn load_ple_global_weights(
    reader: &mut GemmaTensorReader,
    config: &GemmaTextConfigFile,
) -> Result<Option<Gemma4PleGlobalWeights>> {
    let _trace = trace_scope("io.load_ple_global_weights");
    let hidden_size = config.hidden_size;
    let ple_dim = config.hidden_size_per_layer_input.unwrap_or(0);
    if ple_dim == 0 {
        return Ok(None);
    }

    let ple_vocab_size = config.vocab_size_per_layer_input.unwrap_or(config.vocab_size);
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

pub(crate) fn load_ple_token_embedding_row(
    ple_global: &Gemma4PleGlobalWeights,
    layer_idx: usize,
    token_id: u32,
) -> Result<Vec<f32>> {
    let row_idx = usize::try_from(token_id).expect("u32 should fit into usize");
    let cache_key = (layer_idx, row_idx);
    if let Some(cached_row) = ple_global
        .token_row_cache
        .lock()
        .map_err(|_| anyhow!("PLE token row cache is poisoned"))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(cached_row);
    }

    let source = ple_global
        .token_embeddings
        .get(layer_idx)
        .ok_or_else(|| anyhow!("phase 2 PLE token embedding slice count mismatch at layer {layer_idx}"))?;
    let row = match source {
        Gemma4PleMatrixSource::Materialized(matrix) => matrix_row(matrix, row_idx)?,
        Gemma4PleMatrixSource::Lazy(source) => {
            let mmap = ple_mmap_for_path(ple_global, &source.weights_path)?;
            decode_matrix_row_from_source(source, row_idx, mmap.as_ref())?
        }
    };
    ple_global
        .token_row_cache
        .lock()
        .map_err(|_| anyhow!("PLE token row cache is poisoned"))?
        .insert(cache_key, row.clone());
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
        anyhow!("phase 2 PLE model projection slice count mismatch at layer {layer_idx}")
    })?;
    let matrix = match source {
        Gemma4PleMatrixSource::Materialized(matrix) => matrix.clone(),
        Gemma4PleMatrixSource::Lazy(source) => {
            let mmap = ple_mmap_for_path(ple_global, &source.weights_path)?;
            decode_matrix_slice_from_source(source, mmap.as_ref())?
        }
    };
    ple_global
        .model_projection_cache
        .lock()
        .map_err(|_| anyhow!("PLE model projection cache is poisoned"))?
        .insert(layer_idx, matrix.clone());
    Ok(matrix)
}

fn ple_mmap_for_path(ple_global: &Gemma4PleGlobalWeights, path: &Path) -> Result<std::sync::Arc<Mmap>> {
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

fn decode_matrix_slice_from_source(source: &GemmaTensorSliceSource, mmap: &Mmap) -> Result<MatrixF32> {
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
    let _trace = trace_scope(format!("io.load_gemma4_layer_weights layer={layer_idx}"));
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
            input_gate: reader.load_matrix(&format!("{layer_prefix}.per_layer_input_gate.weight"))?,
            layer_projection: reader.load_matrix(&format!("{layer_prefix}.per_layer_projection.weight"))?,
            post_input_norm_weight: reader
                .load_vector(&format!("{layer_prefix}.post_per_layer_input_norm.weight"))?,
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
        rope_base: if is_sliding {
            config.rope_local_base_freq()
        } else {
            config.rope_full_base_freq()
        },
        partial_rotary_dim,
        rope_freq_base_dim: head_dim,
        kv_shared_layer_index,
        attention_k_eq_v: !is_sliding && config.attention_k_eq_v(),
        q_proj: reader.load_matrix(&format!("{layer_prefix}.self_attn.q_proj.weight"))?,
        k_proj: reader.load_matrix(&format!("{layer_prefix}.self_attn.k_proj.weight"))?,
        v_proj: if !is_sliding && config.attention_k_eq_v() {
            reader.load_optional_matrix(&format!("{layer_prefix}.self_attn.v_proj.weight"))?
        } else {
            Some(reader.load_matrix(&format!("{layer_prefix}.self_attn.v_proj.weight"))?)
        },
        o_proj: reader.load_matrix(&format!("{layer_prefix}.self_attn.o_proj.weight"))?,
        q_norm_weight: reader.load_vector(&format!("{layer_prefix}.self_attn.q_norm.weight"))?,
        k_norm_weight: reader.load_vector(&format!("{layer_prefix}.self_attn.k_norm.weight"))?,
        input_layernorm_weight: reader.load_vector(&format!("{layer_prefix}.input_layernorm.weight"))?,
        post_attention_layernorm_weight: reader
            .load_vector(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
        pre_feedforward_layernorm_weight: reader
            .load_vector(&format!("{layer_prefix}.pre_feedforward_layernorm.weight"))?,
        post_feedforward_layernorm_weight: reader
            .load_vector(&format!("{layer_prefix}.post_feedforward_layernorm.weight"))?,
        gate_proj: reader.load_matrix(&format!("{layer_prefix}.mlp.gate_proj.weight"))?,
        up_proj: reader.load_matrix(&format!("{layer_prefix}.mlp.up_proj.weight"))?,
        down_proj: reader.load_matrix(&format!("{layer_prefix}.mlp.down_proj.weight"))?,
        ple,
        layer_scalar: reader.load_optional_scalar(&format!("{layer_prefix}.layer_scalar"))?,
    })
}

pub fn embed_input_tokens_from_gemma_source(
    token_ids: &[u32],
    source: &GemmaEmbeddingTensorSource,
) -> Result<ActivationSequence> {
    let _trace = trace_scope("io.embed_input_tokens_from_gemma_source");
    if token_ids.is_empty() {
        bail!("phase 2 embedding requires at least one token id");
    }

    let activations = with_embedding_tensor(source, |tensor| {
        decode_embedding_rows_for_token_ids(tensor, token_ids, source.hidden_size(), source.scale())
    })?;
    let activations_sha256 = build_activation_commitment(&activations);

    Ok(ActivationSequence {
        activations,
        activations_sha256,
    })
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

fn load_full_embedding_table_from_source(source: &GemmaEmbeddingTensorSource) -> Result<EmbeddingTable> {
    let matrix = with_embedding_tensor(source, |tensor| decode_matrix(tensor))?;

    let mut rows = Vec::with_capacity(matrix.rows);
    for row_idx in 0..matrix.rows {
        rows.push(matrix.values[row_idx * matrix.cols..(row_idx + 1) * matrix.cols].to_vec());
    }

    Ok(EmbeddingTable {
        rows,
        scale: source.scale(),
    })
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
    };

    let tensor = safetensors
        .tensor(tensor_name)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to load tensor {tensor_name} from {}", path.display()))?;

    f(&tensor)
}

fn decode_embedding_rows_for_token_ids(
    tensor: &impl TensorBytes,
    token_ids: &[u32],
    hidden_size: usize,
    scale: f32,
) -> Result<Vec<Vec<f32>>> {
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
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        if row_idx >= shape[0] {
            bail!("token id {token_id} is out of bounds for embedding tensor");
        }
        let encoded_row = &tensor.data()[row_idx * row_bytes..(row_idx + 1) * row_bytes];
        let mut row = if tensor.dtype() == Dtype::F32 {
            decode_f32_bytes_to_vec(encoded_row)?
        } else {
            let mut row = Vec::with_capacity(hidden_size);
            for encoded_value in encoded_row.chunks_exact(bytes_per_scalar) {
                row.push(decode_scalar(encoded_value, tensor.dtype())?);
            }
            row
        };
        for value in &mut row {
            *value *= scale;
        }
        activations.push(row);
    }

    Ok(activations)
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
        let row_values = &encoded_row[col_offset * bytes_per_scalar..(col_offset + col_count) * bytes_per_scalar];
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
        decode_embedding_rows_for_token_ids, decode_matrix_row_from_source, decode_matrix_slice,
        decode_matrix_slice_from_source, decode_single_scalar, decode_vector,
        load_phase2_model_from_gemma_model_path, load_ple_model_projection, load_ple_token_embedding_row,
        parse_safetensors_metadata,
    };
    use crate::phase2::{Gemma4AttentionKind, Gemma4LogitsProjection};
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
    fn decode_matrix_slice_copies_f32_rows_without_scalar_loop() {
        let bytes = f32_to_bytes(&[
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
        ]);
        let tensor = TensorView::new(
            safetensors::Dtype::F32,
            vec![3, 4],
            &bytes,
        )
        .unwrap();

        let matrix = decode_matrix_slice(&tensor, 1, 2, 1, 2).unwrap();

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.values, vec![20.0, 30.0, 200.0, 300.0]);
    }

    #[test]
    fn decode_vector_and_scalar_copy_f32_values_directly() {
        let vector_bytes = f32_to_bytes(&[1.25, -2.5, 3.75]);
        let vector = TensorView::new(
            safetensors::Dtype::F32,
            vec![3],
            &vector_bytes,
        )
        .unwrap();
        let scalar_bytes = f32_to_bytes(&[9.5]);
        let scalar = TensorView::new(safetensors::Dtype::F32, vec![1], &scalar_bytes).unwrap();

        assert_eq!(decode_vector(&vector).unwrap(), vec![1.25, -2.5, 3.75]);
        assert_eq!(decode_single_scalar(&scalar).unwrap(), 9.5);
    }

    #[test]
    fn decode_embedding_rows_for_token_ids_applies_scale_after_f32_copy() {
        let bytes = f32_to_bytes(&[1.0, 2.0, -3.0, 4.0, 5.0, -6.0]);
        let tensor = TensorView::new(
            safetensors::Dtype::F32,
            vec![3, 2],
            &bytes,
        )
        .unwrap();

        let rows = decode_embedding_rows_for_token_ids(&tensor, &[2, 0], 2, 0.5).unwrap();

        assert_eq!(rows, vec![vec![2.5, -3.0], vec![0.5, 1.0]]);
    }

    #[test]
    fn decode_lazy_f32_ple_rows_and_slices() {
        let model_dir = create_test_model_dir("f32-lazy-slice");
        let tensor_name = "model.language_model.embed_tokens_per_layer.weight";
        let values = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0];
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
        let source = crate::phase2::types::GemmaTensorSliceSource {
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
    fn load_phase2_model_loads_arbitrary_layers_and_untied_lm_head() {
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
        tensors.push(scalar_tensor("model.language_model.layers.1.layer_scalar", 0.5));
        write_model_file(&model_dir, &tensors);

        let model = load_phase2_model_from_gemma_model_path(&model_dir).unwrap();

        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].attention_kind, Gemma4AttentionKind::Sliding);
        assert_eq!(model.layers[1].attention_kind, Gemma4AttentionKind::Full);
        assert_eq!(model.layers[0].rope_freq_base_dim, 2);
        assert_eq!(model.layers[1].rope_freq_base_dim, 2);
        assert_eq!(model.layers[0].q_proj.values[0], 1.0);
        assert_eq!(model.layers[1].q_proj.values[0], 2.0);
        assert!(model.layers[1].v_proj.is_none());
        assert_eq!(model.layers[1].layer_scalar, Some(0.5));
        assert_eq!(model.final_norm_weight, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(model.final_logit_softcapping, Some(7.5));
        match &model.logits_projection {
            Gemma4LogitsProjection::UntiedLmHead(weight) => {
                assert_eq!(weight.rows, 3);
                assert_eq!(weight.cols, 4);
            }
            other => panic!("expected untied lm head, got {other:?}"),
        }
    }

    #[test]
    fn load_phase2_model_reuses_embeddings_for_tied_projection() {
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
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        write_model_file(&model_dir, &tensors);

        let model = load_phase2_model_from_gemma_model_path(&model_dir).unwrap();

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
    fn load_phase2_model_builds_lazy_ple_sources() {
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
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
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

        let model = load_phase2_model_from_gemma_model_path(&model_dir).unwrap();
        let ple_global = model.ple_global.as_ref().expect("PLE globals should load");

        assert_eq!(ple_global.token_embedding_layer_count(), 1);
        assert_eq!(ple_global.model_projection_layer_count(), 1);
        assert_eq!(ple_global.projection_norm_weight, vec![1.0, 1.0]);
        assert_eq!(ple_global.embedding_scale, 2f32.sqrt());
        assert_eq!(load_ple_token_embedding_row(ple_global, 0, 1).unwrap(), vec![0.3, 0.4]);
        let projection = load_ple_model_projection(ple_global, 0).unwrap();
        assert_eq!(projection.rows, 2);
        assert_eq!(projection.cols, 4);
        assert_eq!(projection.values, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn load_phase2_model_parses_effective_window_and_kv_sharing_metadata() {
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
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.extend(layer_tensors(1, 2.0, true));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        write_model_file(&model_dir, &tensors);

        let model = load_phase2_model_from_gemma_model_path(&model_dir).unwrap();

        assert_eq!(model.layers[0].sliding_window, Some(3));
        assert_eq!(model.layers[0].cache_sliding_window, None);
        assert_eq!(model.layers[0].kv_shared_layer_index, None);
        assert_eq!(model.layers[1].sliding_window, Some(3));
        assert_eq!(model.layers[1].kv_shared_layer_index, Some(0));
    }

    #[test]
    fn load_phase2_model_rejects_kv_sharing_without_a_donor_layer() {
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
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )];
        tensors.extend(layer_tensors(0, 1.0, true));
        tensors.push(vector_tensor(
            "model.language_model.norm.weight",
            &[4],
            &[1.0, 1.0, 1.0, 1.0],
        ));
        write_model_file(&model_dir, &tensors);

        let error = load_phase2_model_from_gemma_model_path(&model_dir).expect_err("missing donor should fail");

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

    fn layer_tensors(layer_idx: usize, base: f32, include_v_proj: bool) -> Vec<FixtureTensor> {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        let mut tensors = vec![
            matrix_tensor(&format!("{prefix}.self_attn.q_proj.weight"), &[4, 4], &[base; 16]),
            matrix_tensor(&format!("{prefix}.self_attn.k_proj.weight"), &[2, 4], &[base; 8]),
            matrix_tensor(&format!("{prefix}.self_attn.o_proj.weight"), &[4, 4], &[base; 16]),
            vector_tensor(&format!("{prefix}.self_attn.q_norm.weight"), &[2], &[1.0, 1.0]),
            vector_tensor(&format!("{prefix}.self_attn.k_norm.weight"), &[2], &[1.0, 1.0]),
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
            matrix_tensor(&format!("{prefix}.mlp.gate_proj.weight"), &[8, 4], &[0.0; 32]),
            matrix_tensor(&format!("{prefix}.mlp.up_proj.weight"), &[8, 4], &[0.0; 32]),
            matrix_tensor(&format!("{prefix}.mlp.down_proj.weight"), &[4, 8], &[0.0; 32]),
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

    fn scalar_tensor(name: &str, value: f32) -> FixtureTensor {
        FixtureTensor {
            name: name.to_string(),
            shape: vec![1],
            bytes: value.to_le_bytes().to_vec(),
        }
    }

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }
}
