use std::{
    collections::BTreeMap,
    env, fs,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process,
};

use anyhow::{anyhow, bail, Context, Result};
use raster_inference::shared::numerics::det_num::{
    f32_to_wgt, wgt_to_le_bytes, DET_NUM_SPEC_VERSION, DET_WGT_ARTIFACT_FORMAT_VERSION,
    DET_WGT_ARTIFACT_MAGIC,
};
use safetensors::tensor::{serialize_to_file, TensorView};
use serde_json::json;

const DEFAULT_OUTPUT_DIR: &str = "assets/tiny-gemma-dev";
const CONFIG_FILENAME: &str = "config.json";
const FP32_WEIGHTS_FILENAME: &str = "model.safetensors";
const DET_WEIGHTS_FILENAME: &str = "model.detwgt";
const TOKENIZER_FILENAME: &str = "tokenizer.json";
const CHAT_TEMPLATE_FILENAME: &str = "chat_template.jinja";

const LAYER_COUNT: usize = 4;
const HIDDEN_SIZE: usize = 4;
const HEAD_DIM: usize = 2;
const NUM_ATTENTION_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const MLP_WIDTH: usize = 8;
const PLE_DIM: usize = 2;
const BASE_VOCAB_SIZE: usize = 24;
const BYTE_FALLBACK_VOCAB_SIZE: usize = 256;
const VOCAB_SIZE: usize = BASE_VOCAB_SIZE + BYTE_FALLBACK_VOCAB_SIZE;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = CliArgs::parse(env::args().skip(1))?;
    let summary = write_tiny_gemma_dev_bundle(&args)?;

    println!(
        "Wrote tiny Gemma dev bundle with {} tensors to {}",
        summary.tensor_count,
        summary.output_dir.display()
    );
    println!("FP32 weights: {}", summary.fp32_weights_path.display());
    println!(
        "Deterministic weights: {}",
        summary.det_weights_path.display()
    );
    println!("Tokenizer: {}", summary.tokenizer_path.display());
    Ok(())
}

#[derive(Debug, Clone)]
struct CliArgs {
    output_dir: PathBuf,
    force: bool,
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut output_dir = None;
        let mut force = false;
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--output-dir" => {
                    output_dir = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --output-dir"))?,
                    ));
                }
                "--force" => force = true,
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                other => bail!("unrecognized argument `{other}`"),
            }
        }

        Ok(Self {
            output_dir: output_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR)),
            force,
        })
    }
}

#[derive(Debug, Clone)]
struct GenerationSummary {
    output_dir: PathBuf,
    tensor_count: usize,
    fp32_weights_path: PathBuf,
    det_weights_path: PathBuf,
    tokenizer_path: PathBuf,
}

#[derive(Debug, Clone)]
struct FixtureTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn print_usage() {
    eprintln!("Usage: tiny-gemma-dev [--output-dir <dir>] [--force]");
    eprintln!("Defaults to writing {DEFAULT_OUTPUT_DIR}.");
    eprintln!("Use --force to overwrite known bundle files in an existing directory.");
}

fn write_tiny_gemma_dev_bundle(args: &CliArgs) -> Result<GenerationSummary> {
    prepare_output_dir(&args.output_dir, args.force)?;

    let tensors = tiny_model_tensors();
    let config_path = args.output_dir.join(CONFIG_FILENAME);
    let fp32_weights_path = args.output_dir.join(FP32_WEIGHTS_FILENAME);
    let det_weights_path = args.output_dir.join(DET_WEIGHTS_FILENAME);
    let tokenizer_path = args.output_dir.join(TOKENIZER_FILENAME);
    let chat_template_path = args.output_dir.join(CHAT_TEMPLATE_FILENAME);

    fs::write(&config_path, tiny_config_json())
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    write_fp32_model_file(&fp32_weights_path, &tensors)?;
    write_detwgt_file(&det_weights_path, &tensors)?;
    fs::write(&tokenizer_path, tiny_gemma_tokenizer_json())
        .with_context(|| format!("failed to write {}", tokenizer_path.display()))?;
    fs::write(
        &chat_template_path,
        "{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    .with_context(|| format!("failed to write {}", chat_template_path.display()))?;

    Ok(GenerationSummary {
        output_dir: args.output_dir.clone(),
        tensor_count: tensors.len(),
        fp32_weights_path,
        det_weights_path,
        tokenizer_path,
    })
}

fn prepare_output_dir(output_dir: &Path, force: bool) -> Result<()> {
    if output_dir.exists() {
        if !output_dir.is_dir() {
            bail!(
                "output path {} exists but is not a directory",
                output_dir.display()
            );
        }
        let mut entries = fs::read_dir(output_dir)
            .with_context(|| format!("failed to read output directory {}", output_dir.display()))?;
        if entries.next().transpose()?.is_some() && !force {
            bail!(
                "output directory {} is not empty; pass --force to overwrite known bundle files",
                output_dir.display()
            );
        }
    } else {
        fs::create_dir_all(output_dir).with_context(|| {
            format!("failed to create output directory {}", output_dir.display())
        })?;
    }

    if force {
        for filename in [
            CONFIG_FILENAME,
            FP32_WEIGHTS_FILENAME,
            DET_WEIGHTS_FILENAME,
            TOKENIZER_FILENAME,
            CHAT_TEMPLATE_FILENAME,
        ] {
            let path = output_dir.join(filename);
            if path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
            }
        }
    }

    Ok(())
}

fn tiny_config_json() -> String {
    serde_json::to_string_pretty(&json!({
        "text_config": {
            "enable_moe_block": false,
            "final_logit_softcapping": 7.5,
            "global_head_dim": HEAD_DIM,
            "head_dim": HEAD_DIM,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": HIDDEN_SIZE,
            "hidden_size_per_layer_input": PLE_DIM,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_global_key_value_heads": NUM_KV_HEADS,
            "num_hidden_layers": LAYER_COUNT,
            "num_key_value_heads": NUM_KV_HEADS,
            "num_kv_shared_layers": 2,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 12345.0
                },
                "sliding_attention": {
                    "rope_theta": 10000.0
                }
            },
            "sliding_window": 2,
            "tie_word_embeddings": false,
            "vocab_size": VOCAB_SIZE,
            "vocab_size_per_layer_input": VOCAB_SIZE,
            "attention_k_eq_v": true
        }
    }))
    .expect("static tiny config should serialize")
}

fn tiny_gemma_tokenizer_json() -> String {
    serde_json::to_string_pretty(&json!({
        "version": "1.0",
        "added_tokens": [
            {
                "id": 1,
                "content": "<bos>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            },
            {
                "id": 2,
                "content": "<eos>",
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
            "single": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                }
            ],
            "pair": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                },
                {
                    "Sequence": {
                        "id": "B",
                        "type_id": 1
                    }
                }
            ],
            "special_tokens": {}
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": { "String": "▁" },
                    "content": " "
                },
                {
                    "type": "ByteFallback"
                },
                {
                    "type": "Fuse"
                }
            ]
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": tiny_tokenizer_vocab(),
            "merges": []
        }
    }))
    .expect("static tiny tokenizer should serialize")
}

fn tiny_tokenizer_vocab() -> BTreeMap<String, u32> {
    let mut vocab = [
        ("<unk>", 0),
        ("<bos>", 1),
        ("<eos>", 2),
        ("▁", 3),
        ("p", 4),
        ("r", 5),
        ("o", 6),
        ("m", 7),
        ("t", 8),
        ("h", 9),
        ("e", 10),
        ("l", 11),
        ("w", 12),
        ("d", 13),
        ("a", 14),
        ("s", 15),
        ("i", 16),
        ("n", 17),
        ("g", 18),
        ("u", 19),
        ("f", 20),
        ("prompt", 21),
        ("hello", 22),
        ("raster", 23),
    ]
    .into_iter()
    .map(|(token, id)| (token.to_string(), id))
    .collect::<BTreeMap<_, _>>();

    for byte in 0..BYTE_FALLBACK_VOCAB_SIZE {
        vocab.insert(format!("<0x{byte:02X}>"), (BASE_VOCAB_SIZE + byte) as u32);
    }

    vocab
}

fn tiny_model_tensors() -> Vec<FixtureTensor> {
    let mut tensors = vec![
        tensor(
            "model.language_model.embed_tokens.weight",
            &[VOCAB_SIZE, HIDDEN_SIZE],
            pattern_values(VOCAB_SIZE * HIDDEN_SIZE, 1),
        ),
        tensor(
            "model.language_model.embed_tokens_per_layer.weight",
            &[VOCAB_SIZE, LAYER_COUNT * PLE_DIM],
            pattern_values(VOCAB_SIZE * LAYER_COUNT * PLE_DIM, 2),
        ),
        tensor(
            "model.language_model.per_layer_model_projection.weight",
            &[LAYER_COUNT * PLE_DIM, HIDDEN_SIZE],
            pattern_values(LAYER_COUNT * PLE_DIM * HIDDEN_SIZE, 3),
        ),
        tensor(
            "model.language_model.per_layer_projection_norm.weight",
            &[PLE_DIM],
            vec![1.0; PLE_DIM],
        ),
    ];

    for layer_idx in 0..LAYER_COUNT {
        tensors.extend(layer_tensors(layer_idx));
    }

    tensors.push(tensor(
        "model.language_model.norm.weight",
        &[HIDDEN_SIZE],
        vec![1.0; HIDDEN_SIZE],
    ));
    tensors.push(tensor(
        "model.language_model.lm_head.weight",
        &[VOCAB_SIZE, HIDDEN_SIZE],
        pattern_values(VOCAB_SIZE * HIDDEN_SIZE, 80),
    ));

    tensors
}

fn layer_tensors(layer_idx: usize) -> Vec<FixtureTensor> {
    let prefix = format!("model.language_model.layers.{layer_idx}");
    let is_sliding = layer_idx % 2 == 0;
    let seed = (layer_idx as u32 + 1) * 10;

    let mut tensors = vec![
        tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE],
            pattern_values(NUM_ATTENTION_HEADS * HEAD_DIM * HIDDEN_SIZE, seed),
        ),
        tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            pattern_values(NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE, seed + 1),
        ),
        tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM],
            pattern_values(HIDDEN_SIZE * NUM_ATTENTION_HEADS * HEAD_DIM, seed + 3),
        ),
        tensor(
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
            vec![1.0; HEAD_DIM],
        ),
        tensor(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
            vec![1.0; HEAD_DIM],
        ),
        tensor(
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN_SIZE],
            vec![1.0; HIDDEN_SIZE],
        ),
        tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN_SIZE],
            vec![1.0; HIDDEN_SIZE],
        ),
        tensor(
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            &[HIDDEN_SIZE],
            vec![1.0; HIDDEN_SIZE],
        ),
        tensor(
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            &[HIDDEN_SIZE],
            vec![1.0; HIDDEN_SIZE],
        ),
        tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[MLP_WIDTH, HIDDEN_SIZE],
            pattern_values(MLP_WIDTH * HIDDEN_SIZE, seed + 4),
        ),
        tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[MLP_WIDTH, HIDDEN_SIZE],
            pattern_values(MLP_WIDTH * HIDDEN_SIZE, seed + 5),
        ),
        tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN_SIZE, MLP_WIDTH],
            pattern_values(HIDDEN_SIZE * MLP_WIDTH, seed + 6),
        ),
        tensor(
            &format!("{prefix}.per_layer_input_gate.weight"),
            &[PLE_DIM, HIDDEN_SIZE],
            pattern_values(PLE_DIM * HIDDEN_SIZE, seed + 7),
        ),
        tensor(
            &format!("{prefix}.per_layer_projection.weight"),
            &[HIDDEN_SIZE, PLE_DIM],
            pattern_values(HIDDEN_SIZE * PLE_DIM, seed + 8),
        ),
        tensor(
            &format!("{prefix}.post_per_layer_input_norm.weight"),
            &[HIDDEN_SIZE],
            vec![1.0; HIDDEN_SIZE],
        ),
        tensor(&format!("{prefix}.layer_scalar"), &[1], vec![0.5]),
    ];

    if is_sliding {
        tensors.push(tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            pattern_values(NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE, seed + 2),
        ));
    }

    tensors
}

fn tensor(name: &str, shape: &[usize], values: Vec<f32>) -> FixtureTensor {
    let element_count = shape.iter().product::<usize>();
    assert_eq!(
        element_count,
        values.len(),
        "tensor {name} has {} values for shape {:?}",
        values.len(),
        shape
    );
    FixtureTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        values,
    }
}

fn pattern_values(len: usize, seed: u32) -> Vec<f32> {
    const VALUES: [f32; 8] = [-0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625];
    (0..len)
        .map(|idx| VALUES[(idx + seed as usize) % VALUES.len()])
        .collect()
}

fn write_fp32_model_file(path: &Path, tensors: &[FixtureTensor]) -> Result<()> {
    let byte_storage = tensors
        .iter()
        .map(|tensor| f32_to_bytes(&tensor.values))
        .collect::<Vec<_>>();
    let mut metadata = BTreeMap::new();
    for (tensor, bytes) in tensors.iter().zip(byte_storage.iter()) {
        metadata.insert(
            tensor.name.clone(),
            TensorView::new(safetensors::Dtype::F32, tensor.shape.clone(), bytes)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("failed to build tensor view for `{}`", tensor.name))?,
        );
    }
    serialize_to_file(&metadata, &None, path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn write_detwgt_file(path: &Path, tensors: &[FixtureTensor]) -> Result<()> {
    let output_file = File::create(path)
        .with_context(|| format!("failed to create deterministic artifact {}", path.display()))?;
    let mut writer = BufWriter::new(output_file);
    writer.write_all(DET_WGT_ARTIFACT_MAGIC)?;
    writer.write_all(&DET_WGT_ARTIFACT_FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&DET_NUM_SPEC_VERSION.to_le_bytes())?;
    writer.write_all(&(tensors.len() as u64).to_le_bytes())?;

    for tensor in tensors {
        let payload = tensor
            .values
            .iter()
            .flat_map(|value| wgt_to_le_bytes(f32_to_wgt(*value)))
            .collect::<Vec<_>>();
        let name_bytes = tensor.name.as_bytes();
        let element_count = tensor.shape.iter().product::<usize>() as u64;

        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;
        writer.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
        for dim in &tensor.shape {
            writer.write_all(&(*dim as u64).to_le_bytes())?;
        }
        writer.write_all(&element_count.to_le_bytes())?;
        writer.write_all(&(payload.len() as u64).to_le_bytes())?;
        writer.write_all(&payload)?;
    }

    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
