//! Generic disk-loading helpers: chat templates, tokenizer files, embedding
//! tables, and model-agnostic safetensors/mmap infrastructure.
//!
//! Model-family-specific loaders (Gemma safetensors/det-wgt weight loaders,
//! Gemma tokenizer spec parsing) live in `shared/model/gemma/io.rs`; see
//! `docs/model-agnostic-layers.md`.

use std::{collections::HashMap, fs, path::Path};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{tensor::TensorView, Dtype};
use tokenizers::Tokenizer;

use crate::shared::model::transformer::{EmbeddingTable, MatrixF32};

/// Compatibility re-exports for the pre-containment paths; new code should
/// import Gemma loaders from `shared::model::gemma::io`.
pub use crate::shared::model::gemma::io::{
    embed_input_tokens_from_gemma_source, load_gemma_tokenizer_spec_from_path,
    load_transformer_state_model_from_det_num_wgt_path, parse_gemma_tokenizer_spec_bytes,
};
pub(crate) use crate::shared::model::gemma::io::{
    load_ple_model_projection, load_ple_token_embedding_row_internal,
    materialize_det_num_embedding_matrix, materialize_det_num_layer_matrix_source,
    materialize_det_num_ple_model_projection, resolve_layer_weights,
};

#[derive(Clone)]
pub(crate) struct CachedTensorMetadata {
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offset: usize,
    pub(crate) data_len: usize,
}

pub(crate) struct CachedTensorFile {
    pub(crate) mmap: Mmap,
    pub(crate) tensors: HashMap<String, CachedTensorMetadata>,
}

impl CachedTensorFile {
    pub(crate) fn tensor<'a>(
        &'a self,
        tensor_name: &str,
        path: &Path,
    ) -> Result<CachedTensorView<'a>> {
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

    pub(crate) fn tensor_metadata<'a>(
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

pub(crate) struct CachedTensorView<'a> {
    metadata: &'a CachedTensorMetadata,
    data: &'a [u8],
}

pub(crate) trait TensorBytes {
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

#[derive(serde::Deserialize)]
struct SafetensorsHeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(crate) fn parse_safetensors_metadata(
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

pub(crate) fn parse_safetensors_dtype(dtype: &str) -> Result<Dtype> {
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

pub(crate) fn decode_matrix(tensor: &impl TensorBytes) -> Result<MatrixF32> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        bail!("expected rank-2 tensor, got shape {shape:?}");
    }

    decode_matrix_slice(tensor, 0, shape[0], 0, shape[1])
}

pub(crate) fn decode_matrix_slice(
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

pub(crate) fn decode_vector(tensor: &impl TensorBytes) -> Result<Vec<f32>> {
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

pub(crate) fn decode_single_scalar(tensor: &impl TensorBytes) -> Result<f32> {
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

pub(crate) fn bytes_per_scalar(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F16 | Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        Dtype::F64 => Ok(8),
        _ => bail!("unsupported tensor dtype {dtype:?}"),
    }
}

pub(crate) fn decode_f32_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    let len = bytes.len() / std::mem::size_of::<f32>();
    let mut values = vec![0.0; len];
    copy_f32_bytes_into_slice(bytes, &mut values)?;
    Ok(values)
}

pub(crate) fn copy_f32_bytes_into_slice(bytes: &[u8], target: &mut [f32]) -> Result<()> {
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

pub(crate) fn decode_scalar(bytes: &[u8], dtype: Dtype) -> Result<f32> {
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
