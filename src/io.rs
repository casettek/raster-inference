use std::{
    collections::HashMap,
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
    ActivationSequence, EmbeddingTable, Gemma4Layer0Weights, Gemma4Phase2Model,
    Gemma4PleLayerWeights, GemmaEmbeddingTensorSource, MatrixF32,
};

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
    head_dim: usize,
    hidden_activation: String,
    hidden_size: usize,
    hidden_size_per_layer_input: Option<usize>,
    layer_types: Vec<String>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    sliding_window: usize,
    vocab_size: usize,
    vocab_size_per_layer_input: Option<usize>,
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

struct GemmaTensorReader {
    source: GemmaModelSource,
    mmaps: HashMap<PathBuf, Mmap>,
}

impl GemmaTensorReader {
    fn new(source: GemmaModelSource) -> Self {
        Self {
            source,
            mmaps: HashMap::new(),
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

    fn load_matrix_slice(
        &mut self,
        tensor_name: &str,
        row_offset: usize,
        row_count: usize,
        col_offset: usize,
        col_count: usize,
    ) -> Result<MatrixF32> {
        let tensor = self.load_tensor(tensor_name)?;
        decode_matrix_slice(&tensor, row_offset, row_count, col_offset, col_count)
    }

    fn load_vector(&mut self, tensor_name: &str) -> Result<Vec<f32>> {
        let tensor = self.load_tensor(tensor_name)?;
        decode_vector(&tensor)
    }

    fn load_optional_scalar(&mut self, tensor_name: &str) -> Result<Option<f32>> {
        match self.load_tensor(tensor_name) {
            Ok(tensor) => Ok(Some(decode_single_scalar(&tensor)?)),
            Err(_) => Ok(None),
        }
    }

    fn load_tensor(&mut self, tensor_name: &str) -> Result<TensorView<'_>> {
        let path = self.path_for_tensor(tensor_name)?;
        let mmap = self.mmap_for_path(&path)?;
        let safetensors = SafeTensors::deserialize(mmap.as_ref())
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to deserialize safetensors file {}", path.display()))?;

        safetensors
            .tensor(tensor_name)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load tensor {tensor_name} from {}", path.display()))
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

    fn mmap_for_path(&mut self, path: &Path) -> Result<&Mmap> {
        if !self.mmaps.contains_key(path) {
            let file = File::open(path)
                .with_context(|| format!("failed to open safetensors file {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("failed to mmap safetensors file {}", path.display()))?;
            self.mmaps.insert(path.to_path_buf(), mmap);
        }

        self.mmaps
            .get(path)
            .ok_or_else(|| anyhow!("failed to cache mmap for {}", path.display()))
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
    if config.layer_types.first().map(String::as_str) != Some("sliding_attention") {
        bail!("phase 2 currently expects Gemma layer 0 to use sliding attention");
    }

    let mut reader = GemmaTensorReader::new(source);
    let (embedding_tensor_name, _vocab_size, hidden_size) =
        reader.load_first_available_tensor_metadata(GEMMA_EMBED_TENSOR_NAMES)?;
    let layer0 = load_layer0_weights(&mut reader, &config)?;
    let embedding_source = build_embedding_source(&reader, &embedding_tensor_name, hidden_size);

    Ok(Gemma4Phase2Model {
        embedding_table: None,
        embedding_source: Some(embedding_source),
        layer0,
    })
}

fn load_layer0_weights(
    reader: &mut GemmaTensorReader,
    config: &GemmaTextConfigFile,
) -> Result<Gemma4Layer0Weights> {
    let layer0_prefix = "model.language_model.layers.0";
    let hidden_size = config.hidden_size;
    let ple_dim = config.hidden_size_per_layer_input.unwrap_or(0);
    let ple_vocab_size = config.vocab_size_per_layer_input.unwrap_or(config.vocab_size);
    let ple = if ple_dim > 0 {
        Some(Gemma4PleLayerWeights {
            token_embedding: reader.load_matrix_slice(
                "model.language_model.embed_tokens_per_layer.weight",
                0,
                ple_vocab_size,
                0,
                ple_dim,
            )?,
            model_projection: reader.load_matrix_slice(
                "model.language_model.per_layer_model_projection.weight",
                0,
                ple_dim,
                0,
                hidden_size,
            )?,
            projection_norm_weight: reader.load_vector("model.language_model.per_layer_projection_norm.weight")?,
            input_gate: reader.load_matrix(&format!("{layer0_prefix}.per_layer_input_gate.weight"))?,
            layer_projection: reader.load_matrix(&format!("{layer0_prefix}.per_layer_projection.weight"))?,
            post_input_norm_weight: reader.load_vector(&format!(
                "{layer0_prefix}.post_per_layer_input_norm.weight"
            ))?,
            embedding_scale: (ple_dim as f32).sqrt(),
            projection_scalar: (hidden_size as f32).powf(-0.5),
            input_scale: 2f32.powf(-0.5),
        })
    } else {
        None
    };

    Ok(Gemma4Layer0Weights {
        hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        sliding_window: config.sliding_window,
        rms_norm_eps: config.rms_norm_eps,
        q_proj: reader.load_matrix(&format!("{layer0_prefix}.self_attn.q_proj.weight"))?,
        k_proj: reader.load_matrix(&format!("{layer0_prefix}.self_attn.k_proj.weight"))?,
        v_proj: reader.load_matrix(&format!("{layer0_prefix}.self_attn.v_proj.weight"))?,
        o_proj: reader.load_matrix(&format!("{layer0_prefix}.self_attn.o_proj.weight"))?,
        q_norm_weight: reader.load_vector(&format!("{layer0_prefix}.self_attn.q_norm.weight"))?,
        k_norm_weight: reader.load_vector(&format!("{layer0_prefix}.self_attn.k_norm.weight"))?,
        input_layernorm_weight: reader.load_vector(&format!("{layer0_prefix}.input_layernorm.weight"))?,
        post_attention_layernorm_weight: reader
            .load_vector(&format!("{layer0_prefix}.post_attention_layernorm.weight"))?,
        pre_feedforward_layernorm_weight: reader
            .load_vector(&format!("{layer0_prefix}.pre_feedforward_layernorm.weight"))?,
        post_feedforward_layernorm_weight: reader
            .load_vector(&format!("{layer0_prefix}.post_feedforward_layernorm.weight"))?,
        gate_proj: reader.load_matrix(&format!("{layer0_prefix}.mlp.gate_proj.weight"))?,
        up_proj: reader.load_matrix(&format!("{layer0_prefix}.mlp.up_proj.weight"))?,
        down_proj: reader.load_matrix(&format!("{layer0_prefix}.mlp.down_proj.weight"))?,
        ple,
        layer_scalar: reader.load_optional_scalar(&format!("{layer0_prefix}.layer_scalar"))?,
    })
}

pub fn embed_input_tokens_from_gemma_source(
    token_ids: &[u32],
    source: &GemmaEmbeddingTensorSource,
) -> Result<ActivationSequence> {
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
    let matrix = with_embedding_tensor(source, decode_matrix)?;

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
    tensor: &TensorView<'_>,
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
        let mut row = Vec::with_capacity(hidden_size);
        for encoded_value in encoded_row.chunks_exact(bytes_per_scalar) {
            row.push(decode_scalar(encoded_value, tensor.dtype())? * scale);
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

fn decode_matrix(tensor: &TensorView<'_>) -> Result<MatrixF32> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        bail!("expected rank-2 tensor, got shape {shape:?}");
    }

    decode_matrix_slice(tensor, 0, shape[0], 0, shape[1])
}

fn decode_matrix_slice(
    tensor: &TensorView<'_>,
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
        for encoded_value in row_values.chunks_exact(bytes_per_scalar) {
            values.push(decode_scalar(encoded_value, tensor.dtype())?);
        }
    }

    Ok(MatrixF32 {
        rows: row_count,
        cols: col_count,
        values,
    })
}

fn decode_vector(tensor: &TensorView<'_>) -> Result<Vec<f32>> {
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

    let mut values = Vec::with_capacity(shape[0]);
    for encoded_value in tensor.data().chunks_exact(bytes_per_scalar) {
        values.push(decode_scalar(encoded_value, tensor.dtype())?);
    }
    Ok(values)
}

fn decode_single_scalar(tensor: &TensorView<'_>) -> Result<f32> {
    let shape = tensor.shape();
    if shape != [1] {
        bail!("expected single-scalar tensor shaped [1], got {shape:?}");
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
