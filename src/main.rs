use std::{env, path::PathBuf, process};

use raster_inference::{
    load_chat_template, load_phase2_model_from_gemma_model_path, load_tokenizer_from_path,
    run_inference, InferenceRequest, ModelSpec, SamplingConfig, TextDecodingPolicy,
};

const CLI_MAX_NEW_TOKENS: usize = 3;
const CLI_TEMPERATURE: f32 = 1.0;

fn print_usage() {
    eprintln!(
        "Usage: raster-inference <model-id> <tokenizer.json> <chat-template.jinja> <gemma-model-path> <prompt...>"
    );
    eprintln!("Set RASTER_TRACE_TILES=1 to print tile-level timing logs to stderr.");
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 6 {
        print_usage();
        process::exit(1);
    }

    let model_id = args[1].clone();
    let tokenizer_path = PathBuf::from(&args[2]);
    let template_path = PathBuf::from(&args[3]);
    let gemma_model_path = PathBuf::from(&args[4]);
    let prompt = args[5..].join(" ");

    let chat_template = load_chat_template(&template_path)?;
    let tokenizer = load_tokenizer_from_path(&tokenizer_path)?;
    let phase2_model = load_phase2_model_from_gemma_model_path(&gemma_model_path)?;

    let model = ModelSpec {
        model_id,
        tokenizer_path,
        chat_template,
        bos_token: None,
        eos_token: None,
        unk_token: None,
    };

    let request = InferenceRequest {
        prompt_bytes: prompt.into_bytes(),
        text_decoding_policy: TextDecodingPolicy::Utf8,
        add_generation_prompt: true,
        add_special_tokens: true,
        sampling: SamplingConfig {
            max_new_tokens: Some(CLI_MAX_NEW_TOKENS),
            temperature: Some(CLI_TEMPERATURE),
            ..SamplingConfig::default()
        },
    };

    let inference_state = run_inference(&request, &model, &tokenizer, &phase2_model)?;
    println!("{}", serde_json::to_string_pretty(&inference_state)?);

    Ok(())
}
