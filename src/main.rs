use std::{env, path::PathBuf, process};

use raster_inference::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path, run_inference_with_controls,
    AuthenticatedGemmaTokenizer, InferenceControls, InferenceExecutionMode, InferenceRequest,
    InferenceRunOutcome, ModelSpec, SamplingConfig, TextDecodingPolicy,
};

const CLI_MAX_NEW_TOKENS: usize = 3;
const CLI_TEMPERATURE: f32 = 1.0;

fn print_usage() {
    eprintln!(
        "Usage: raster-inference [--deterministic] [--raster] [--commit-checkpoints] [--terminal-checkpoint <checkpoint-id>] <model-id> <tokenizer.json> <chat-template.jinja> <model-path> <prompt...>"
    );
    eprintln!(
        "Pass --commit-checkpoints to emit the checkpoint trace file at the end of the run. Pass --terminal-checkpoint to stop after a named checkpoint such as prefill.finalize. Pass --raster to use raster-authored tiles and stop after prefill.prepare_aux while that path is being built out."
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
    let raster_tokenizer_source = if cli_args.raster_tiles {
        Some(AuthenticatedGemmaTokenizer::new(
            load_gemma_tokenizer_spec_from_path(&cli_args.tokenizer_path)?,
        ))
    } else {
        None
    };
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

    let inference_outcome = run_inference_with_controls(
        &request,
        &model,
        &tokenizer,
        &transformer_model,
        &InferenceControls {
            commit_checkpoints: cli_args.commit_checkpoints,
            terminal_checkpoint: cli_args.terminal_checkpoint,
            raster_tiles: cli_args.raster_tiles,
            raster_tokenizer_source,
        },
    )?;
    match inference_outcome {
        InferenceRunOutcome::Completed(inference_state) => {
            println!("{}", serde_json::to_string_pretty(&inference_state)?);
        }
        InferenceRunOutcome::Paused(paused_state) => {
            println!("{}", serde_json::to_string_pretty(&paused_state)?);
        }
    }

    Ok(())
}

struct CliArgs {
    commit_checkpoints: bool,
    execution_mode: InferenceExecutionMode,
    raster_tiles: bool,
    terminal_checkpoint: Option<String>,
    model_id: String,
    tokenizer_path: PathBuf,
    template_path: PathBuf,
    model_path: PathBuf,
    prompt: String,
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> anyhow::Result<Self> {
        let mut commit_checkpoints = false;
        let mut execution_mode = InferenceExecutionMode::Fp32;
        let mut raster_tiles = false;
        let mut terminal_checkpoint = None;
        let mut positional_args = Vec::new();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--commit-checkpoints" => commit_checkpoints = true,
                "--deterministic" => execution_mode = InferenceExecutionMode::Deterministic,
                "--raster" => raster_tiles = true,
                "--terminal-checkpoint" => {
                    let checkpoint_id = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a checkpoint id after {arg}"))?;
                    terminal_checkpoint = Some(checkpoint_id);
                }
                _ if arg.starts_with("--terminal-checkpoint=") => {
                    let checkpoint_id = arg
                        .split_once('=')
                        .map(|(_, value)| value)
                        .expect("split_once should succeed for --terminal-checkpoint=value");
                    terminal_checkpoint = Some(checkpoint_id.to_string());
                }
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

        if raster_tiles {
            execution_mode = InferenceExecutionMode::Deterministic;
        }

        Ok(Self {
            commit_checkpoints,
            execution_mode,
            raster_tiles,
            terminal_checkpoint,
            model_id: positional_args[0].clone(),
            tokenizer_path: PathBuf::from(&positional_args[1]),
            template_path: PathBuf::from(&positional_args[2]),
            model_path: PathBuf::from(&positional_args[3]),
            prompt: positional_args[4..].join(" "),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::CliArgs;
    use raster_inference::InferenceExecutionMode;

    #[test]
    fn parse_terminal_checkpoint_flag() {
        let args = CliArgs::parse([
            "--terminal-checkpoint".to_string(),
            "prefill.finalize".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(
            args.terminal_checkpoint.as_deref(),
            Some("prefill.finalize")
        );
        assert_eq!(args.execution_mode, InferenceExecutionMode::Fp32);
        assert!(!args.raster_tiles);
    }

    #[test]
    fn parse_terminal_checkpoint_equals_and_deterministic_flag() {
        let args = CliArgs::parse([
            "--deterministic".to_string(),
            "--terminal-checkpoint=output.finalize".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.terminal_checkpoint.as_deref(), Some("output.finalize"));
        assert_eq!(args.execution_mode, InferenceExecutionMode::Deterministic);
    }

    #[test]
    fn parse_raster_flag() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert!(args.raster_tiles);
        assert_eq!(args.execution_mode, InferenceExecutionMode::Deterministic);
    }

    #[test]
    fn parse_commit_checkpoints_flag() {
        let args = CliArgs::parse([
            "--commit-checkpoints".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert!(args.commit_checkpoints);
    }
}
