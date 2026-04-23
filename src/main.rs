use std::{env, path::PathBuf, process};

use raster_inference::{
    load_chat_template, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path, run_inference, InferenceExecutionMode,
    InferenceRequest, ModelSpec, SamplingConfig, TextDecodingPolicy,
};

const CLI_MAX_NEW_TOKENS: usize = 3;
const CLI_TEMPERATURE: f32 = 1.0;

fn print_usage() {
    eprintln!(
        "Usage: raster-inference [--deterministic] <model-id> <tokenizer.json> <chat-template.jinja> <model-path> <prompt...>"
    );
    eprintln!(
        "Set RASTER_TRACE_TILES=1 for full tile timing logs plus checkpoint traces, or RASTER_TRACE_TILES=0 for logs only with no checkpoint hashing or trace files."
    );
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let cli_args = CliArgs::parse(env::args().skip(1))?;
    if cli_args.prompt.is_empty() {
        print_usage();
        process::exit(1);
    }
    let chat_template = load_chat_template(&cli_args.template_path)?;
    let tokenizer = load_tokenizer_from_path(&cli_args.tokenizer_path)?;
    let transformer_model = match cli_args.execution_mode {
        InferenceExecutionMode::Fp32 => {
            load_transformer_state_model_from_gemma_model_path(&cli_args.model_path)?
        }
        InferenceExecutionMode::Deterministic => {
            load_transformer_state_model_from_det_num_wgt_path(&cli_args.model_path)?
        }
    };

    let model = ModelSpec {
        model_id: cli_args.model_id,
        tokenizer_path: cli_args.tokenizer_path,
        chat_template,
        bos_token: None,
        eos_token: None,
        unk_token: None,
    };

    let request = InferenceRequest {
        prompt_bytes: cli_args.prompt.into_bytes(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: true,
        add_special_tokens: true,
        execution_mode: cli_args.execution_mode,
        sampling: SamplingConfig {
            max_new_tokens: Some(CLI_MAX_NEW_TOKENS),
            temperature: Some(CLI_TEMPERATURE),
            ..SamplingConfig::default()
        },
    };

    let inference_state = run_inference(&request, &model, &tokenizer, &transformer_model)?;
    println!("{}", serde_json::to_string_pretty(&inference_state)?);

    Ok(())
}

struct CliArgs {
    execution_mode: InferenceExecutionMode,
    model_id: String,
    tokenizer_path: PathBuf,
    template_path: PathBuf,
    model_path: PathBuf,
    prompt: String,
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> anyhow::Result<Self> {
        let mut execution_mode = InferenceExecutionMode::Fp32;
        let mut positional_args = Vec::new();
        for arg in args {
            match arg.as_str() {
                "--deterministic" => execution_mode = InferenceExecutionMode::Deterministic,
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                _ => positional_args.push(arg),
            }
        }

        if positional_args.len() < 5 {
            anyhow::bail!("expected at least 5 positional arguments");
        }

        Ok(Self {
            execution_mode,
            model_id: positional_args[0].clone(),
            tokenizer_path: PathBuf::from(&positional_args[1]),
            template_path: PathBuf::from(&positional_args[2]),
            model_path: PathBuf::from(&positional_args[3]),
            prompt: positional_args[4..].join(" "),
        })
    }
}
