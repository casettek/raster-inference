use std::{
    env, fs,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process,
};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use raster_inference::shared::det_num::{f32_to_wgt, wgt_to_le_bytes};
use safetensors::{Dtype, SafeTensors};

const CONFIG_FILENAME: &str = "config.json";
const OUTPUT_WEIGHTS_FILENAME: &str = "model.detwgt";
const ARTIFACT_MAGIC: &[u8; 8] = b"DNWGTV0\0";
const ARTIFACT_FORMAT_VERSION: u32 = 0;
const DET_NUM_SPEC_VERSION: u32 = 0;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = CliArgs::parse(env::args().skip(1))?;
    let summary = convert_model_to_det_num_wgt_artifact(&args)?;
    println!(
        "Wrote {} canonical Wgt tensors to {}",
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConvertedTensor {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn print_usage() {
    eprintln!(
        "Usage: gemma-det-num-wgt-converter --input <model-dir-or-safetensors> --output-dir <artifact-dir> [--config <config.json>]"
    );
    eprintln!("If --input is a model directory, the converter looks for `config.json` and `model.safetensors`.");
    eprintln!("If --input is a `.safetensors` file, the converter uses a sibling `config.json` unless --config is provided.");
}

fn convert_model_to_det_num_wgt_artifact(args: &CliArgs) -> Result<ConversionSummary> {
    let source = resolve_source(&args.input, args.config.as_deref())?;
    prepare_output_dir(&args.output_dir)?;

    let file = File::open(&source.weights_path).with_context(|| {
        format!(
            "failed to open source safetensors file {}",
            source.weights_path.display()
        )
    })?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
        format!(
            "failed to mmap source safetensors file {}",
            source.weights_path.display()
        )
    })?;
    let safetensors = SafeTensors::deserialize(mmap.as_ref())
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to deserialize source safetensors file {}",
                source.weights_path.display()
            )
        })?;

    let mut tensor_names = safetensors.names();
    tensor_names.sort_unstable();

    let mut input_tensor_bytes = 0usize;
    let mut output_tensor_bytes = 0usize;
    let output_weights_path = args.output_dir.join(OUTPUT_WEIGHTS_FILENAME);
    let output_file = File::create(&output_weights_path).with_context(|| {
        format!(
            "failed to create det_num Wgt artifact {}",
            output_weights_path.display()
        )
    })?;
    let mut writer = BufWriter::new(output_file);
    write_artifact_header(&mut writer, tensor_names.len() as u64)?;

    for tensor_name in tensor_names {
        let tensor = safetensors
            .tensor(tensor_name)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load tensor `{tensor_name}`"))?;
        input_tensor_bytes += tensor.data().len();
        let converted_tensor = ConvertedTensor {
            name: tensor_name.to_string(),
            shape: tensor.shape().to_vec(),
            bytes: Vec::new(),
        };
        output_tensor_bytes += payload_len_for_tensor(&tensor)?;
        write_tensor_header(
            &mut writer,
            &converted_tensor,
            payload_len_for_tensor(&tensor)?,
        )?;
        write_tensor_payload_as_wgt(&mut writer, &tensor)
            .with_context(|| format!("failed to convert tensor `{tensor_name}` to Wgt"))?;
    }
    writer.flush().with_context(|| {
        format!(
            "failed to flush det_num Wgt artifact {}",
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
        tensor_count: tensor_names_len(&safetensors),
        input_tensor_bytes,
        output_tensor_bytes,
        output_weights_path,
        output_config_path,
    })
}

fn tensor_names_len(safetensors: &SafeTensors<'_>) -> usize {
    safetensors.names().len()
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

fn payload_len_for_tensor(tensor: &safetensors::tensor::TensorView<'_>) -> Result<usize> {
    let element_count = tensor
        .shape()
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| anyhow!("tensor shape overflowed"))?;
    element_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("tensor payload byte count overflowed"))
}

fn write_tensor_payload_as_wgt(
    writer: &mut impl Write,
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Result<()> {
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

    let mut chunk_buffer = Vec::with_capacity(16_384 * 4);
    for encoded_value in tensor.data().chunks_exact(bytes_per_scalar) {
        let decoded = decode_scalar(encoded_value, tensor.dtype())?;
        if !decoded.is_finite() {
            bail!("non-finite source values are not supported");
        }
        chunk_buffer.extend_from_slice(&wgt_to_le_bytes(f32_to_wgt(decoded)));
        if chunk_buffer.len() >= 16_384 * 4 {
            writer.write_all(&chunk_buffer)?;
            chunk_buffer.clear();
        }
    }
    if !chunk_buffer.is_empty() {
        writer.write_all(&chunk_buffer)?;
    }
    Ok(())
}

fn write_artifact_header(writer: &mut impl Write, tensor_count: u64) -> Result<()> {
    writer.write_all(ARTIFACT_MAGIC)?;
    writer.write_all(&ARTIFACT_FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&DET_NUM_SPEC_VERSION.to_le_bytes())?;
    writer.write_all(&tensor_count.to_le_bytes())?;
    Ok(())
}

fn write_tensor_header(
    writer: &mut impl Write,
    tensor: &ConvertedTensor,
    payload_len: usize,
) -> Result<()> {
    let name_bytes = tensor.name.as_bytes();
    let name_len = u32::try_from(name_bytes.len()).map_err(|_| anyhow!("tensor name too long"))?;
    let rank = u32::try_from(tensor.shape.len()).map_err(|_| anyhow!("tensor rank too large"))?;
    let element_count = tensor
        .shape
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim as u64))
        .ok_or_else(|| anyhow!("tensor shape overflowed"))?;
    let payload_len =
        u64::try_from(payload_len).map_err(|_| anyhow!("tensor payload too large"))?;

    writer.write_all(&name_len.to_le_bytes())?;
    writer.write_all(name_bytes)?;
    writer.write_all(&rank.to_le_bytes())?;
    for dim in &tensor.shape {
        writer.write_all(&(*dim as u64).to_le_bytes())?;
    }
    writer.write_all(&element_count.to_le_bytes())?;
    writer.write_all(&payload_len.to_le_bytes())?;
    Ok(())
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
        convert_model_to_det_num_wgt_artifact, CliArgs, CONFIG_FILENAME, OUTPUT_WEIGHTS_FILENAME,
    };
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype,
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

    #[derive(Debug, PartialEq, Eq)]
    struct ParsedTensor {
        name: String,
        shape: Vec<u64>,
        payload: Vec<i32>,
    }

    #[test]
    fn converts_model_directory_to_det_num_wgt_artifact() {
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

        let summary = convert_model_to_det_num_wgt_artifact(&CliArgs {
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

        let tensors = parse_artifact(&fs::read(output_dir.join(OUTPUT_WEIGHTS_FILENAME)).unwrap());
        assert_eq!(
            tensors,
            vec![
                ParsedTensor {
                    name: "model.language_model.embed_tokens.weight".to_string(),
                    shape: vec![2, 2],
                    payload: vec![98_304, -131_072, 212_992, 294_912],
                },
                ParsedTensor {
                    name: "model.language_model.norm.weight".to_string(),
                    shape: vec![2],
                    payload: vec![32_768, 65_536],
                }
            ]
        );
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
                &[3],
                &[0.0, 1.5 / 65_536.0, 2.5 / 65_536.0],
            )],
        );

        let summary = convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.join("model.safetensors"),
            output_dir: output_dir.clone(),
            config: None,
        })
        .unwrap();

        assert_eq!(summary.tensor_count, 1);
        let tensors = parse_artifact(&fs::read(output_dir.join(OUTPUT_WEIGHTS_FILENAME)).unwrap());
        assert_eq!(
            tensors[0].payload,
            vec![0, 2, 2],
            "ties should round to even"
        );
    }

    #[test]
    fn output_is_stable_and_saturating() {
        let input_dir = create_temp_dir("stable");
        let output_dir_a = create_temp_dir("stable-output-a");
        let output_dir_b = create_temp_dir("stable-output-b");
        fs::remove_dir_all(&output_dir_a).unwrap();
        fs::remove_dir_all(&output_dir_b).unwrap();

        fs::write(input_dir.join(CONFIG_FILENAME), "{}").unwrap();
        write_model_file(
            &input_dir,
            &[FixtureTensor::f32(
                "tensor",
                &[4],
                &[40_000.0, -40_000.0, -3.5 / 65_536.0, 3.5 / 65_536.0],
            )],
        );

        convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir_a.clone(),
            config: None,
        })
        .unwrap();
        convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir_b.clone(),
            config: None,
        })
        .unwrap();

        let bytes_a = fs::read(output_dir_a.join(OUTPUT_WEIGHTS_FILENAME)).unwrap();
        let bytes_b = fs::read(output_dir_b.join(OUTPUT_WEIGHTS_FILENAME)).unwrap();
        assert_eq!(bytes_a, bytes_b);

        let tensors = parse_artifact(&bytes_a);
        assert_eq!(tensors[0].payload, vec![i32::MAX, i32::MIN, -4, 4]);
    }

    fn parse_artifact(bytes: &[u8]) -> Vec<ParsedTensor> {
        fn read_u32(bytes: &[u8], cursor: &mut usize) -> u32 {
            let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
            *cursor += 4;
            value
        }

        fn read_u64(bytes: &[u8], cursor: &mut usize) -> u64 {
            let value = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap());
            *cursor += 8;
            value
        }

        let mut cursor = 0usize;
        assert_eq!(&bytes[cursor..cursor + 8], b"DNWGTV0\0");
        cursor += 8;
        assert_eq!(read_u32(bytes, &mut cursor), 0);
        assert_eq!(read_u32(bytes, &mut cursor), 0);
        let tensor_count = read_u64(bytes, &mut cursor) as usize;

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name_len = read_u32(bytes, &mut cursor) as usize;
            let name = String::from_utf8(bytes[cursor..cursor + name_len].to_vec()).unwrap();
            cursor += name_len;
            let rank = read_u32(bytes, &mut cursor) as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(read_u64(bytes, &mut cursor));
            }
            let element_count = read_u64(bytes, &mut cursor) as usize;
            let payload_len = read_u64(bytes, &mut cursor) as usize;
            assert_eq!(payload_len, element_count * 4);
            let payload = bytes[cursor..cursor + payload_len]
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            cursor += payload_len;
            tensors.push(ParsedTensor {
                name,
                shape,
                payload,
            });
        }

        assert_eq!(cursor, bytes.len());
        tensors
    }

    fn create_temp_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("gemma-det-num-wgt-converter-{label}-{unique}"));
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
        serialize_to_file(&metadata, &None, &dir.join("model.safetensors")).unwrap();
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
