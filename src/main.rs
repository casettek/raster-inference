use std::{env, path::PathBuf, process};

use raster_inference::{
    load_chat_template, load_tokenizer_from_path, run_phase1, InferenceRequest, MessageRole,
    ModelSpec, SamplingConfig, TextMessage,
};

fn print_usage() {
    eprintln!(
        "Usage: raster-inference <model-id> <tokenizer.json> <chat-template.jinja> <prompt...>"
    );
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 5 {
        print_usage();
        process::exit(1);
    }

    let model_id = args[1].clone();
    let tokenizer_path = PathBuf::from(&args[2]);
    let template_path = PathBuf::from(&args[3]);
    let prompt = args[4..].join(" ");

    let chat_template = load_chat_template(&template_path)?;
    let tokenizer = load_tokenizer_from_path(&tokenizer_path)?;

    let model = ModelSpec {
        model_id,
        tokenizer_path,
        chat_template,
        bos_token: None,
        eos_token: None,
        unk_token: None,
    };

    let request = InferenceRequest {
        messages: vec![TextMessage {
            role: MessageRole::User,
            content: prompt,
        }],
        add_generation_prompt: true,
        add_special_tokens: true,
        sampling: SamplingConfig::default(),
    };

    let phase1_state = run_phase1(&request, &model, &tokenizer)?;
    println!("{}", serde_json::to_string_pretty(&phase1_state)?);

    Ok(())
}
