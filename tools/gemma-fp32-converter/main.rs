use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process,
};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype, SafeTensors,
};

const CONFIG_FILENAME: &str = "config.json";
const OUTPUT_WEIGHTS_FILENAME: &str = "model.safetensors";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = CliArgs::parse(env::args().skip(1))?;
    let summary = convert_model_to_fp32_artifact(&args)?;
    println!(
        "Wrote {} FP32 tensors to {}",
        summary.tensor_count,
        summary.output_weights_path.display()
    );
    println!("Copied config to {}", summary.output_config_path.display());
    println!(
        "Converted {} -> {}",
        human_bytes(summary.input_tensor_bytes),
        human_bytes(summary.output_tensor_bytes)
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct CliArgs {
    input: PathBuf,
    output_dir: PathBuf,
    config: Option<PathBuf>,
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut input = None;
        let mut output_dir = None;
        let mut config = None;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => {
                    input = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --input"))?,
                    ));
                }
                "--output-dir" => {
                    output_dir = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --output-dir"))?,
                    ));
                }
                "--config" => {
                    config = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --config"))?,
                    ));
                }
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                other => bail!("unrecognized argument `{other}`"),
            }
        }

        Ok(Self {
            input: input.ok_or_else(|| anyhow!("missing required --input argument"))?,
            output_dir: output_dir
                .ok_or_else(|| anyhow!("missing required --output-dir argument"))?,
            config,
        })
    }
}

#[derive(Debug, Clone)]
struct ResolvedSource {
    weights_path: PathBuf,
    config_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ConversionSummary {
    tensor_count: usize,
    input_tensor_bytes: usize,
    output_tensor_bytes: usize,
    output_weights_path: PathBuf,
    output_config_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ConvertedTensor {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn print_usage() {
    eprintln!(
        "Usage: gemma-fp32-converter --input <model-dir-or-safetensors> --output-dir <artifact-dir> [--config <config.json>]"
    );
    eprintln!("If --input is a model directory, the converter looks for `config.json` and `model.safetensors`.");
    eprintln!("If --input is a `.safetensors` file, the converter uses a sibling `config.json` unless --config is provided.");
}

fn convert_model_to_fp32_artifact(args: &CliArgs) -> Result<ConversionSummary> {
    let source = resolve_source(&args.input, args.config.as_deref())?;
    prepare_output_dir(&args.output_dir)?;

    let raw = fs::read(&source.weights_path).with_context(|| {
        format!(
            "failed to read source safetensors file {}",
            source.weights_path.display()
        )
    })?;
    let safetensors = SafeTensors::deserialize(raw.as_slice())
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to deserialize source safetensors file {}",
                source.weights_path.display()
            )
        })?;

    let mut converted_tensors = Vec::with_capacity(safetensors.names().len());
    let mut input_tensor_bytes = 0usize;
    let mut output_tensor_bytes = 0usize;

    for tensor_name in safetensors.names() {
        let tensor = safetensors
            .tensor(tensor_name)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load tensor `{tensor_name}`"))?;
        input_tensor_bytes += tensor.data().len();
        let bytes = convert_tensor_bytes_to_f32(&tensor)
            .with_context(|| format!("failed to convert tensor `{tensor_name}` to F32"))?;
        output_tensor_bytes += bytes.len();
        converted_tensors.push(ConvertedTensor {
            name: tensor_name.to_string(),
            shape: tensor.shape().to_vec(),
            bytes,
        });
    }

    let output_weights_path = args.output_dir.join(OUTPUT_WEIGHTS_FILENAME);
    let mut metadata = BTreeMap::new();
    for tensor in &converted_tensors {
        metadata.insert(
            tensor.name.clone(),
            TensorView::new(Dtype::F32, tensor.shape.clone(), &tensor.bytes).with_context(
                || {
                    format!(
                        "failed to build serialized tensor view for `{}`",
                        tensor.name
                    )
                },
            )?,
        );
    }
    serialize_to_file(&metadata, &None, &output_weights_path).with_context(|| {
        format!(
            "failed to write converted safetensors file {}",
            output_weights_path.display()
        )
    })?;

    let output_config_path = args.output_dir.join(CONFIG_FILENAME);
    fs::copy(&source.config_path, &output_config_path).with_context(|| {
        format!(
            "failed to copy config from {} to {}",
            source.config_path.display(),
            output_config_path.display()
        )
    })?;

    Ok(ConversionSummary {
        tensor_count: converted_tensors.len(),
        input_tensor_bytes,
        output_tensor_bytes,
        output_weights_path,
        output_config_path,
    })
}

fn resolve_source(input: &Path, config_override: Option<&Path>) -> Result<ResolvedSource> {
    if input.is_dir() {
        if config_override.is_some() {
            bail!("--config is only supported when --input points to a `.safetensors` file");
        }

        let config_path = input.join(CONFIG_FILENAME);
        if !config_path.is_file() {
            bail!("model directory {} is missing config.json", input.display());
        }

        let weights_path = ["model.safetensors", "consolidated.safetensors"]
            .into_iter()
            .map(|filename| input.join(filename))
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| {
                anyhow!(
                    "model directory {} is missing `model.safetensors` or `consolidated.safetensors`",
                    input.display()
                )
            })?;

        return Ok(ResolvedSource {
            weights_path,
            config_path,
        });
    }

    if input
        .extension()
        .is_some_and(|extension| extension == "safetensors")
    {
        let config_path = match config_override {
            Some(path) => path.to_path_buf(),
            None => input
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(CONFIG_FILENAME),
        };
        if !config_path.is_file() {
            bail!(
                "could not locate config.json for {}; pass --config explicitly",
                input.display()
            );
        }
        return Ok(ResolvedSource {
            weights_path: input.to_path_buf(),
            config_path,
        });
    }

    bail!(
        "unsupported input path {}; expected a model directory or `.safetensors` file",
        input.display()
    )
}

fn prepare_output_dir(output_dir: &Path) -> Result<()> {
    if output_dir.exists() {
        if !output_dir.is_dir() {
            bail!(
                "output path {} exists but is not a directory",
                output_dir.display()
            );
        }
        if fs::read_dir(output_dir)
            .with_context(|| format!("failed to read output directory {}", output_dir.display()))?
            .next()
            .is_some()
        {
            bail!(
                "output directory {} must not already contain files",
                output_dir.display()
            );
        }
    } else {
        fs::create_dir_all(output_dir).with_context(|| {
            format!("failed to create output directory {}", output_dir.display())
        })?;
    }
    Ok(())
}

fn convert_tensor_bytes_to_f32(tensor: &TensorView<'_>) -> Result<Vec<u8>> {
    let bytes_per_scalar = bytes_per_scalar(tensor.dtype())?;
    let element_count = tensor
        .shape()
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| anyhow!("tensor shape overflowed"))?;
    let expected_bytes = element_count
        .checked_mul(bytes_per_scalar)
        .ok_or_else(|| anyhow!("tensor byte count overflowed"))?;
    if tensor.data().len() != expected_bytes {
        bail!(
            "tensor byte length mismatch: expected {expected_bytes}, got {}",
            tensor.data().len()
        );
    }

    let mut converted = Vec::with_capacity(element_count * 4);
    for encoded_value in tensor.data().chunks_exact(bytes_per_scalar) {
        converted.extend_from_slice(&decode_scalar(encoded_value, tensor.dtype())?.to_le_bytes());
    }
    Ok(converted)
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

fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0usize;
    while value >= 1024.0 && unit_idx < UNITS.len() - 1 {
        value /= 1024.0;
        unit_idx += 1;
    }
    format!("{value:.2} {}", UNITS[unit_idx])
}

#[cfg(test)]
mod tests {
    use super::{
        convert_model_to_fp32_artifact, CliArgs, CONFIG_FILENAME, OUTPUT_WEIGHTS_FILENAME,
    };
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype, SafeTensors,
    };
    use std::{
        collections::BTreeMap,
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    #[derive(Debug)]
    struct FixtureTensor {
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    #[test]
    fn converts_model_directory_to_fp32_artifact() {
        let input_dir = create_temp_dir("source");
        let output_dir = create_temp_dir("output");
        fs::remove_dir_all(&output_dir).unwrap();

        fs::write(
            input_dir.join(CONFIG_FILENAME),
            "{\"text_config\":{\"hidden_size\":4}}",
        )
        .unwrap();
        write_model_file(
            &input_dir,
            &[
                FixtureTensor::bf16(
                    "model.language_model.embed_tokens.weight",
                    &[2, 2],
                    &[1.5, -2.0, 3.25, 4.5],
                ),
                FixtureTensor::f16("model.language_model.norm.weight", &[2], &[0.5, 1.0]),
            ],
        );

        let summary = convert_model_to_fp32_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir.clone(),
            config: None,
        })
        .unwrap();

        assert_eq!(summary.tensor_count, 2);
        assert!(summary.output_weights_path.is_file());
        assert!(summary.output_config_path.is_file());
        assert_eq!(
            fs::read_to_string(output_dir.join(CONFIG_FILENAME)).unwrap(),
            "{\"text_config\":{\"hidden_size\":4}}"
        );

        let output_raw = fs::read(output_dir.join(OUTPUT_WEIGHTS_FILENAME)).unwrap();
        let output = SafeTensors::deserialize(&output_raw).unwrap();
        let embedding = output
            .tensor("model.language_model.embed_tokens.weight")
            .unwrap();
        assert_eq!(embedding.dtype(), Dtype::F32);
        assert_eq!(
            decode_f32_values(embedding.data()),
            vec![1.5, -2.0, 3.25, 4.5]
        );
        let norm = output.tensor("model.language_model.norm.weight").unwrap();
        assert_eq!(norm.dtype(), Dtype::F32);
        assert_eq!(decode_f32_values(norm.data()), vec![0.5, 1.0]);
    }

    #[test]
    fn supports_direct_safetensors_input_with_sibling_config() {
        let input_dir = create_temp_dir("direct-file");
        let output_dir = create_temp_dir("direct-file-output");
        fs::remove_dir_all(&output_dir).unwrap();

        fs::write(input_dir.join(CONFIG_FILENAME), "{}").unwrap();
        write_model_file(
            &input_dir,
            &[FixtureTensor::f32(
                "model.language_model.norm.weight",
                &[1],
                &[2.5],
            )],
        );

        let summary = convert_model_to_fp32_artifact(&CliArgs {
            input: input_dir.join(OUTPUT_WEIGHTS_FILENAME),
            output_dir: output_dir.clone(),
            config: None,
        })
        .unwrap();

        assert_eq!(summary.tensor_count, 1);
        let output_raw = fs::read(output_dir.join(OUTPUT_WEIGHTS_FILENAME)).unwrap();
        let output = SafeTensors::deserialize(&output_raw).unwrap();
        let norm = output.tensor("model.language_model.norm.weight").unwrap();
        assert_eq!(norm.dtype(), Dtype::F32);
        assert_eq!(decode_f32_values(norm.data()), vec![2.5]);
    }

    fn create_temp_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("gemma-fp32-converter-{label}-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_model_file(dir: &Path, tensors: &[FixtureTensor]) {
        let mut metadata = BTreeMap::new();
        for tensor in tensors {
            metadata.insert(
                tensor.name.clone(),
                TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes).unwrap(),
            );
        }
        serialize_to_file(&metadata, &None, &dir.join(OUTPUT_WEIGHTS_FILENAME)).unwrap();
    }

    fn decode_f32_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    impl FixtureTensor {
        fn bf16(name: &str, shape: &[usize], values: &[f32]) -> Self {
            Self {
                name: name.to_string(),
                dtype: Dtype::BF16,
                shape: shape.to_vec(),
                bytes: values
                    .iter()
                    .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
                    .collect(),
            }
        }

        fn f16(name: &str, shape: &[usize], values: &[f32]) -> Self {
            Self {
                name: name.to_string(),
                dtype: Dtype::F16,
                shape: shape.to_vec(),
                bytes: values
                    .iter()
                    .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
                    .collect(),
            }
        }

        fn f32(name: &str, shape: &[usize], values: &[f32]) -> Self {
            Self {
                name: name.to_string(),
                dtype: Dtype::F32,
                shape: shape.to_vec(),
                bytes: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            }
        }
    }
}
