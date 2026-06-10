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
use raster_inference::shared::numerics::det_num::{
    f32_to_wgt, wgt_to_le_bytes, DET_NUM_SPEC_VERSION, DET_WGT_ARTIFACT_FORMAT_VERSION,
    DET_WGT_ARTIFACT_MAGIC, DET_WGT_ROW_MASS_LIMIT,
};
use safetensors::{Dtype, SafeTensors};

const CONFIG_FILENAME: &str = "config.json";
const OUTPUT_WEIGHTS_FILENAME: &str = "model.detwgt";

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
    if args.report_bounds {
        print_bounds_report(&summary.tensor_bounds);
    }
    Ok(())
}

fn print_bounds_report(tensor_bounds: &[TensorBounds]) {
    println!("Overflow-bound report (row mass limit = {DET_WGT_ROW_MASS_LIMIT}):");
    for bounds in tensor_bounds {
        let margin = if bounds.max_row_mass == 0 {
            "inf".to_string()
        } else {
            format!(
                "{:.2}",
                DET_WGT_ROW_MASS_LIMIT as f64 / bounds.max_row_mass as f64
            )
        };
        let note = if bounds.mac_bound {
            ""
        } else {
            " (elementwise; bound not enforced)"
        };
        println!(
            "  {} max_row_mass={} margin={}x{note}",
            bounds.name, bounds.max_row_mass, margin
        );
    }
}

#[derive(Debug, Clone)]
struct CliArgs {
    input: PathBuf,
    output_dir: PathBuf,
    config: Option<PathBuf>,
    report_bounds: bool,
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut input = None;
        let mut output_dir = None;
        let mut config = None;
        let mut report_bounds = false;

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
                "--report-bounds" => {
                    report_bounds = true;
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
            report_bounds,
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
    tensor_bounds: Vec<TensorBounds>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorBounds {
    name: String,
    max_row_mass: u64,
    mac_bound: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConvertedTensor {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn print_usage() {
    eprintln!(
        "Usage: gemma-det-num-wgt-converter --input <model-dir-or-safetensors> --output-dir <artifact-dir> [--config <config.json>] [--report-bounds]"
    );
    eprintln!("If --input is a model directory, the converter looks for `config.json` and `model.safetensors`.");
    eprintln!("If --input is a `.safetensors` file, the converter uses a sibling `config.json` unless --config is provided.");
    eprintln!("`--report-bounds` prints the per-tensor max row mass and overflow-bound margin after conversion.");
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

    let mut tensor_bounds = Vec::with_capacity(tensor_names.len());
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
        let max_row_mass = compute_max_row_mass(&tensor).with_context(|| {
            format!("tensor `{tensor_name}` failed the conversion-time overflow bound check")
        })?;
        tensor_bounds.push(TensorBounds {
            name: tensor_name.to_string(),
            max_row_mass,
            mac_bound: row_mass_bound_applies(tensor.shape()),
        });
        output_tensor_bytes += payload_len_for_tensor(&tensor)?;
        write_tensor_header(
            &mut writer,
            &converted_tensor,
            payload_len_for_tensor(&tensor)?,
            max_row_mass,
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
        tensor_bounds,
    })
}

/// Returns whether the conversion-time overflow bound applies to a tensor shape.
///
/// The bound is defined for weight tensors used in MAC reductions, whose
/// last-dimension rows are dot-product reduction rows: tensors of rank >= 2.
/// Rank-1 tensors (norm gains, scalars) are elementwise operands that never
/// enter a MAC reduction; their mass is recorded for audit but not enforced.
fn row_mass_bound_applies(shape: &[usize]) -> bool {
    shape.len() >= 2
}

/// Computes the per-tensor maximum row mass (`max_r sum_i |wgt_bits[r][i]|`) over
/// last-dimension rows. For MAC-bound tensors (rank >= 2) it enforces the
/// normative conversion-time overflow bound: every row mass must be strictly
/// below `DET_WGT_ROW_MASS_LIMIT` (2^31), failing closed on the first violating
/// row. For other tensors the mass is computed but not enforced.
fn compute_max_row_mass(tensor: &safetensors::tensor::TensorView<'_>) -> Result<u64> {
    let bytes_per_scalar = bytes_per_scalar(tensor.dtype())?;
    let row_len = tensor.shape().last().copied().unwrap_or(1).max(1);
    let enforce_bound = row_mass_bound_applies(tensor.shape());

    let mut max_row_mass = 0u64;
    let mut row_mass = 0u64;
    let mut row_idx = 0usize;
    let mut col_idx = 0usize;
    for encoded_value in tensor.data().chunks_exact(bytes_per_scalar) {
        let decoded = decode_scalar(encoded_value, tensor.dtype())?;
        if !decoded.is_finite() {
            bail!("non-finite source values are not supported");
        }
        row_mass += u64::from(f32_to_wgt(decoded).to_bits().unsigned_abs());
        col_idx += 1;
        if col_idx == row_len {
            if enforce_bound && row_mass >= DET_WGT_ROW_MASS_LIMIT {
                bail!(
                    "row {row_idx} has row mass {row_mass}, violating the overflow bound (must be < {DET_WGT_ROW_MASS_LIMIT})"
                );
            }
            max_row_mass = max_row_mass.max(row_mass);
            row_mass = 0;
            col_idx = 0;
            row_idx += 1;
        }
    }
    Ok(max_row_mass)
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
    writer.write_all(DET_WGT_ARTIFACT_MAGIC)?;
    writer.write_all(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&DET_NUM_SPEC_VERSION.to_le_bytes())?;
    writer.write_all(&tensor_count.to_le_bytes())?;
    Ok(())
}

fn write_tensor_header(
    writer: &mut impl Write,
    tensor: &ConvertedTensor,
    payload_len: usize,
    max_row_mass: u64,
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
    writer.write_all(&max_row_mass.to_le_bytes())?;
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
    use raster_inference::load_transformer_state_model_from_det_num_wgt_path;

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
        max_row_mass: u64,
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
            report_bounds: false,
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
                    max_row_mass: 507_904,
                    payload: vec![98_304, -131_072, 212_992, 294_912],
                },
                ParsedTensor {
                    name: "model.language_model.norm.weight".to_string(),
                    shape: vec![2],
                    max_row_mass: 98_304,
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
            report_bounds: false,
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
        // Row 0 saturates to i32::MAX, whose row mass of 2^31 - 1 sits exactly at
        // the overflow bound minus one and must be accepted.
        write_model_file(
            &input_dir,
            &[FixtureTensor::f32(
                "tensor",
                &[2, 2],
                &[40_000.0, 0.0, -3.5 / 65_536.0, 3.5 / 65_536.0],
            )],
        );

        convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir_a.clone(),
            config: None,
            report_bounds: false,
        })
        .unwrap();
        convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir_b.clone(),
            config: None,
            report_bounds: false,
        })
        .unwrap();

        let bytes_a = fs::read(output_dir_a.join(OUTPUT_WEIGHTS_FILENAME)).unwrap();
        let bytes_b = fs::read(output_dir_b.join(OUTPUT_WEIGHTS_FILENAME)).unwrap();
        assert_eq!(bytes_a, bytes_b);

        let tensors = parse_artifact(&bytes_a);
        assert_eq!(tensors[0].payload, vec![i32::MAX, 0, -4, 4]);
        assert_eq!(tensors[0].max_row_mass, u64::from(i32::MAX.unsigned_abs()));
    }

    #[test]
    fn rejects_tensor_violating_row_mass_bound() {
        let input_dir = create_temp_dir("bound-violation");
        let output_dir = create_temp_dir("bound-violation-output");
        fs::remove_dir_all(&output_dir).unwrap();

        fs::write(input_dir.join(CONFIG_FILENAME), "{}").unwrap();
        // -40000.0 saturates to i32::MIN, whose magnitude (2^31) alone reaches
        // the row mass limit, so the converter must fail closed.
        write_model_file(
            &input_dir,
            &[FixtureTensor::f32(
                "tensor",
                &[2, 2],
                &[-40_000.0, 0.0, 0.0, 0.0],
            )],
        );

        let error = convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir.clone(),
            config: None,
            report_bounds: false,
        })
        .expect_err("row mass bound violation should fail conversion");
        assert!(
            format!("{error:#}").contains("violating the overflow bound"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn accepts_rank1_norm_tensor_exceeding_mac_bound_mass() {
        let input_dir = create_temp_dir("rank1-norm");
        let output_dir = create_temp_dir("rank1-norm-output");
        fs::remove_dir_all(&output_dir).unwrap();

        fs::write(input_dir.join(CONFIG_FILENAME), "{}").unwrap();
        // Norm gain vectors are elementwise operands, never MAC reduction rows;
        // a whole-vector mass beyond 2^31 (here 2 * 40000 * 2^16) must convert.
        write_model_file(
            &input_dir,
            &[FixtureTensor::f32(
                "model.norm.weight",
                &[2],
                &[20_000.0, 20_000.0],
            )],
        );

        let summary = convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir.clone(),
            config: None,
            report_bounds: false,
        })
        .expect("rank-1 tensor with large mass should convert");

        assert_eq!(summary.tensor_bounds.len(), 1);
        assert!(!summary.tensor_bounds[0].mac_bound);
        assert_eq!(
            summary.tensor_bounds[0].max_row_mass,
            2 * 20_000 * 65_536_u64
        );

        let tensors = parse_artifact(&fs::read(output_dir.join(OUTPUT_WEIGHTS_FILENAME)).unwrap());
        assert_eq!(tensors[0].max_row_mass, 2 * 20_000 * 65_536_u64);
    }

    #[test]
    fn converted_artifact_loads_via_deterministic_model_loader() {
        let input_dir = create_temp_dir("loader-roundtrip");
        let output_dir = create_temp_dir("loader-roundtrip-output");
        fs::remove_dir_all(&output_dir).unwrap();

        fs::write(
            input_dir.join(CONFIG_FILENAME),
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
    "sliding_window": 2,
    "tie_word_embeddings": false,
    "vocab_size": 3
  }
}"#,
        )
        .unwrap();
        write_model_file(
            &input_dir,
            &[
                FixtureTensor::f32(
                    "model.language_model.embed_tokens.weight",
                    &[3, 4],
                    &[0.0; 12],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    &[4, 4],
                    &[0.0; 16],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    &[2, 4],
                    &[0.0; 8],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    &[2, 4],
                    &[0.0; 8],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.self_attn.o_proj.weight",
                    &[4, 4],
                    &[0.0; 16],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.self_attn.q_norm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.self_attn.k_norm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.input_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.post_feedforward_layernorm.weight",
                    &[4],
                    &[1.0; 4],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    &[8, 4],
                    &[0.0; 32],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.mlp.up_proj.weight",
                    &[8, 4],
                    &[0.0; 32],
                ),
                FixtureTensor::f32(
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    &[4, 8],
                    &[0.0; 32],
                ),
                FixtureTensor::f32("model.language_model.norm.weight", &[4], &[1.0; 4]),
                FixtureTensor::f32("model.language_model.lm_head.weight", &[3, 4], &[0.0; 12]),
            ],
        );

        convert_model_to_det_num_wgt_artifact(&CliArgs {
            input: input_dir.clone(),
            output_dir: output_dir.clone(),
            config: None,
            report_bounds: false,
        })
        .unwrap();

        let model = load_transformer_state_model_from_det_num_wgt_path(&output_dir).unwrap();
        assert!(model.embedding_source.is_some());
        assert_eq!(model.layers.len(), 1);
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
        assert_eq!(
            read_u32(bytes, &mut cursor),
            super::DET_WGT_ARTIFACT_FORMAT_VERSION
        );
        assert_eq!(read_u32(bytes, &mut cursor), super::DET_NUM_SPEC_VERSION);
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
            let max_row_mass = read_u64(bytes, &mut cursor);
            let payload: Vec<i32> = bytes[cursor..cursor + payload_len]
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            cursor += payload_len;
            let row_len = shape.last().copied().unwrap_or(1).max(1) as usize;
            let recomputed_max_row_mass = payload
                .chunks(row_len)
                .map(|row| row.iter().map(|bits| bits.unsigned_abs() as u64).sum())
                .max()
                .unwrap_or(0u64);
            assert_eq!(
                max_row_mass, recomputed_max_row_mass,
                "tensor `{name}` max row mass should match independent recomputation"
            );
            tensors.push(ParsedTensor {
                name,
                shape,
                max_row_mass,
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
