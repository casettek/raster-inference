use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use tokenizers::Tokenizer;

use crate::phase2::EmbeddingTable;

const GEMMA_EMBED_TENSOR_NAMES: &[&str] = &[
    "model.language_model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "embed_tokens.weight",
];

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
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
    let path = path.as_ref();
    let (weights_path, tensor_name) = resolve_gemma_embedding_tensor_path(path)?;
    load_embedding_table_from_safetensors_path(&weights_path, tensor_name)
}

fn resolve_gemma_embedding_tensor_path(path: &Path) -> Result<(PathBuf, &'static str)> {
    if path.is_dir() {
        let index_path = path.join("model.safetensors.index.json");
        if index_path.is_file() {
            return resolve_gemma_embedding_tensor_from_index(&index_path);
        }

        for filename in ["model.safetensors", "consolidated.safetensors"] {
            let candidate = path.join(filename);
            if candidate.is_file() {
                return resolve_gemma_embedding_tensor_in_file(&candidate);
            }
        }

        let mut safetensors_files = fs::read_dir(path)
            .with_context(|| format!("failed to read model directory {}", path.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|entry_path| entry_path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        safetensors_files.sort();

        if safetensors_files.len() == 1 {
            return resolve_gemma_embedding_tensor_in_file(&safetensors_files[0]);
        }

        bail!(
            "failed to locate Gemma embedding weights in {}: expected model.safetensors.index.json, model.safetensors, consolidated.safetensors, or a single .safetensors shard",
            path.display()
        );
    }

    if path.extension().is_some_and(|ext| ext == "json") {
        return resolve_gemma_embedding_tensor_from_index(path);
    }

    if path.extension().is_some_and(|ext| ext == "safetensors") {
        return resolve_gemma_embedding_tensor_in_file(path);
    }

    bail!(
        "unsupported model path {}: expected a model directory, .safetensors file, or model.safetensors.index.json",
        path.display()
    )
}

fn resolve_gemma_embedding_tensor_from_index(index_path: &Path) -> Result<(PathBuf, &'static str)> {
    let raw = fs::read_to_string(index_path)
        .with_context(|| format!("failed to read index file {}", index_path.display()))?;
    let index: SafetensorsIndex = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse index JSON from {}", index_path.display()))?;

    let tensor_name = GEMMA_EMBED_TENSOR_NAMES
        .iter()
        .find(|candidate| index.weight_map.contains_key(**candidate))
        .copied()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "failed to locate Gemma embedding tensor in {}",
                index_path.display()
            )
        })?;

    let shard = index
        .weight_map
        .get(tensor_name)
        .expect("tensor_name should exist in weight map");
    let weights_path = index_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(shard);

    Ok((weights_path, tensor_name))
}

fn resolve_gemma_embedding_tensor_in_file(weights_path: &Path) -> Result<(PathBuf, &'static str)> {
    let raw = fs::read(weights_path)
        .with_context(|| format!("failed to read safetensors file {}", weights_path.display()))?;
    let safetensors = SafeTensors::deserialize(&raw)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to deserialize safetensors file {}", weights_path.display()))?;

    let tensor_name = GEMMA_EMBED_TENSOR_NAMES
        .iter()
        .find(|candidate| safetensors.tensor(candidate).is_ok())
        .copied()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "failed to locate Gemma embedding tensor in {}",
                weights_path.display()
            )
        })?;

    Ok((weights_path.to_path_buf(), tensor_name))
}

fn load_embedding_table_from_safetensors_path(weights_path: &Path, tensor_name: &str) -> Result<EmbeddingTable> {
    let raw = fs::read(weights_path)
        .with_context(|| format!("failed to read safetensors file {}", weights_path.display()))?;
    let safetensors = SafeTensors::deserialize(&raw)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to deserialize safetensors file {}", weights_path.display()))?;
    let tensor = safetensors.tensor(tensor_name).map_err(anyhow::Error::msg)?;

    let shape = tensor.shape();
    if shape.len() != 2 {
        bail!(
            "expected Gemma embedding tensor {tensor_name} to be rank 2, got shape {shape:?}"
        );
    }

    let vocab_size = shape[0];
    let hidden_size = shape[1];
    if vocab_size == 0 || hidden_size == 0 {
        bail!(
            "expected Gemma embedding tensor {tensor_name} to have non-zero shape, got {shape:?}"
        );
    }

    let rows = decode_embedding_rows(tensor.data(), tensor.dtype(), vocab_size, hidden_size)?;

    Ok(EmbeddingTable {
        rows,
        scale: (hidden_size as f32).sqrt(),
    })
}

fn decode_embedding_rows(
    data: &[u8],
    dtype: Dtype,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let bytes_per_scalar = bytes_per_scalar(dtype)?;
    let values = vocab_size
        .checked_mul(hidden_size)
        .ok_or_else(|| anyhow::anyhow!("embedding tensor shape is too large"))?;
    let expected_bytes = values
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow::anyhow!("embedding tensor byte size overflowed"))?;

    if data.len() != expected_bytes {
        bail!(
            "embedding tensor byte length mismatch: expected {expected_bytes}, got {}",
            data.len()
        );
    }

    let mut rows = Vec::with_capacity(vocab_size);
    let row_bytes = hidden_size
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow::anyhow!("embedding row byte size overflowed"))?;

    for encoded_row in data.chunks_exact(row_bytes) {
        let mut row = Vec::with_capacity(hidden_size);
        for encoded_value in encoded_row.chunks_exact(bytes_per_scalar) {
            row.push(decode_scalar(encoded_value, dtype)?);
        }
        rows.push(row);
    }

    Ok(rows)
}

fn bytes_per_scalar(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F16 | Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        Dtype::F64 => Ok(8),
        _ => bail!("unsupported embedding tensor dtype {dtype:?}"),
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
        _ => bail!("unsupported embedding tensor dtype {dtype:?}"),
    }
}
