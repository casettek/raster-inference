use std::{env, path::PathBuf, process};

use raster_inference::shared::artifacts::integrity_mode::with_raster_integrity_mode;
use raster_inference::{
    load_chat_template, load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path, sequence, trace,
    AuthenticatedGemmaTokenizer, InferenceControls, InferenceExecutionMode, InferenceRequest,
    InferenceRunOutcome, ModelSpec, RasterDetourSpec, RasterIntegrityMode, SamplingConfig,
    TextDecodingPolicy,
};

const CLI_MAX_NEW_TOKENS: usize = 3;
const CLI_TEMPERATURE: f32 = 1.0;

fn print_usage() {
    eprintln!(
        "Usage: raster-inference [--deterministic] [--raster] [--raster-at <routine-id[:occurrence]>] [--raster-unchecked-test-mode] [--raster-trace-tiles] [--prefill-token-range-width <tokens>] [--decode-layer-range-width <layers>] [--raster-projection-rows-per-tile <rows>] [--raster-attention-kv-rows-per-tile <rows>] [--raster-sequence-rows-per-tile <rows>] [--raster-head-rows-per-tile <rows>] [--raster-tokenizer-bpe-pairs-per-tile <pairs>] [--raster-tokenizer-bpe-pieces-per-tile <pieces>] [--raster-output-byte-flush-bytes-per-tile <bytes>] [--commit-checkpoints] [--terminal-checkpoint <checkpoint-id[:occurrence]>] <model-id> <tokenizer.json> <chat-template.jinja> <model-path> <prompt...>"
    );
    eprintln!(
        "Pass --commit-checkpoints to emit the checkpoint trace file at the end of the run. Pass --terminal-checkpoint to stop after a named checkpoint such as prefill.finalize, or prefill.range_finalize:2 for the second finalized prefill layer. Pass --raster to use the single root-backed raster tile inference path. Pass --raster-at to run native deterministic CPU with one selected raster routine occurrence. Pass --raster-unchecked-test-mode to use synthetic raster handles and skip Merkle proof work in test builds. Pass --raster-trace-tiles to print verbose routine, progress, and individual tile execution logs. Pass --prefill-token-range-width to bound prompt tokens covered by each prefill.range routine. Pass --decode-layer-range-width to bound transformer layers covered by each decode.layer_range routine. Pass --raster-projection-rows-per-tile to bound raster projection row chunks. Pass --raster-attention-kv-rows-per-tile to bound visible key/value rows read by each raster attention tile. Pass --raster-sequence-rows-per-tile and --raster-head-rows-per-tile to batch independent row ops. Pass --raster-tokenizer-bpe-pairs-per-tile and --raster-tokenizer-bpe-pieces-per-tile to bound tokenizer BPE scan and apply chunks. Pass --raster-output-byte-flush-bytes-per-tile to bound byte-fallback output flush chunks."
    );
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    configure_rayon_thread_pool()?;
    let cli_args = CliArgs::parse(env::args().skip(1))?;
    if cli_args.prompt.is_empty() {
        print_usage();
        process::exit(1);
    }
    let chat_template = load_chat_template(&cli_args.template_path)?;
    let tokenizer = load_tokenizer_from_path(&cli_args.tokenizer_path)?;
    let raster_tokenizer_source =
        if cli_args.raster || cli_args.execution_mode == InferenceExecutionMode::Deterministic {
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

    let raster_integrity_mode = cli_args.raster_integrity_mode;
    let controls = InferenceControls {
        commit_checkpoints: cli_args.commit_checkpoints,
        terminal_checkpoint: cli_args.terminal_checkpoint,
        raster: cli_args.raster,
        raster_detour: cli_args.raster_detour,
        raster_tokenizer_source,
        raster_projection_rows_per_tile: cli_args.raster_projection_rows_per_tile,
        raster_attention_kv_rows_per_tile: cli_args.raster_attention_kv_rows_per_tile,
        raster_sequence_rows_per_tile: cli_args.raster_sequence_rows_per_tile,
        raster_head_rows_per_tile: cli_args.raster_head_rows_per_tile,
        prefill_token_range_width: cli_args.prefill_token_range_width,
        decode_layer_range_width: cli_args.decode_layer_range_width,
        raster_tokenizer_bpe_pairs_per_tile: cli_args.raster_tokenizer_bpe_pairs_per_tile,
        raster_tokenizer_bpe_pieces_per_tile: cli_args.raster_tokenizer_bpe_pieces_per_tile,
        raster_output_byte_flush_bytes_per_tile: cli_args.raster_output_byte_flush_bytes_per_tile,
    };
    let run_inference = || {
        with_raster_integrity_mode(raster_integrity_mode, || {
            sequence::run(&request, &model, &tokenizer, &transformer_model, &controls)
        })
    };
    let inference_outcome = if cli_args.raster_trace_tiles {
        trace::with_trace_logging_enabled(true, run_inference)
    } else {
        run_inference()
    }?;
    match inference_outcome {
        InferenceRunOutcome::Completed(inference_state) => {
            println!("{}", serde_json::to_string_pretty(&inference_state)?);
        }
        InferenceRunOutcome::Paused(paused_state) => {
            println!("{}", serde_json::to_string_pretty(&paused_state)?);
        }
        InferenceRunOutcome::RasterPromptPrepared(prompt_state) => {
            println!("{}", serde_json::to_string_pretty(&prompt_state)?);
        }
    }

    Ok(())
}

#[derive(Debug)]
struct CliArgs {
    commit_checkpoints: bool,
    execution_mode: InferenceExecutionMode,
    raster: bool,
    raster_detour: Option<RasterDetourSpec>,
    raster_integrity_mode: RasterIntegrityMode,
    raster_trace_tiles: bool,
    raster_projection_rows_per_tile: Option<usize>,
    raster_attention_kv_rows_per_tile: Option<usize>,
    raster_sequence_rows_per_tile: Option<usize>,
    raster_head_rows_per_tile: Option<usize>,
    prefill_token_range_width: Option<usize>,
    decode_layer_range_width: Option<usize>,
    raster_tokenizer_bpe_pairs_per_tile: Option<usize>,
    raster_tokenizer_bpe_pieces_per_tile: Option<usize>,
    raster_output_byte_flush_bytes_per_tile: Option<usize>,
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
        let mut raster = false;
        let mut raster_detour = None;
        let mut raster_integrity_mode = RasterIntegrityMode::Verified;
        let mut raster_decode_only = false;
        let mut raster_trace_tiles = false;
        let mut raster_projection_rows_per_tile = None;
        let mut raster_attention_kv_rows_per_tile = None;
        let mut raster_sequence_rows_per_tile = None;
        let mut raster_head_rows_per_tile = None;
        let mut prefill_token_range_width = None;
        let mut decode_layer_range_width = None;
        let mut raster_tokenizer_bpe_pairs_per_tile = None;
        let mut raster_tokenizer_bpe_pieces_per_tile = None;
        let mut raster_output_byte_flush_bytes_per_tile = None;
        let mut terminal_checkpoint = None;
        let mut positional_args = Vec::new();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--commit-checkpoints" => commit_checkpoints = true,
                "--deterministic" => execution_mode = InferenceExecutionMode::Deterministic,
                "--raster" => raster = true,
                "--raster-at" => {
                    let detour = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a routine id after {arg}"))?;
                    raster_detour = Some(RasterDetourSpec::parse(&detour)?);
                }
                _ if arg.starts_with("--raster-at=") => {
                    let detour = arg
                        .split_once('=')
                        .map(|(_, value)| value)
                        .expect("split_once should succeed for --raster-at=value");
                    raster_detour = Some(RasterDetourSpec::parse(detour)?);
                }
                "--raster-unchecked-test-mode" => {
                    raster_integrity_mode = parse_raster_unchecked_test_mode_flag()?
                }
                "--raster-decode-only" => raster_decode_only = true,
                "--raster-trace-tiles" => raster_trace_tiles = true,
                "--prefill-token-range-width" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a token count after {arg}"))?;
                    prefill_token_range_width = Some(parse_prefill_token_range_width(&value)?);
                }
                _ if arg.starts_with("--prefill-token-range-width=") => {
                    let value = arg
                        .split_once('=')
                        .map(|(_, value)| value)
                        .expect("split_once should succeed for --prefill-token-range-width=value");
                    prefill_token_range_width = Some(parse_prefill_token_range_width(value)?);
                }
                "--decode-layer-range-width" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a layer count after {arg}"))?;
                    decode_layer_range_width = Some(parse_decode_layer_range_width(&value)?);
                }
                _ if arg.starts_with("--decode-layer-range-width=") => {
                    let value = arg
                        .split_once('=')
                        .map(|(_, value)| value)
                        .expect("split_once should succeed for --decode-layer-range-width=value");
                    decode_layer_range_width = Some(parse_decode_layer_range_width(value)?);
                }
                "--raster-projection-rows-per-tile" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a row count after {arg}"))?;
                    raster_projection_rows_per_tile =
                        Some(parse_raster_projection_rows_per_tile(&value)?);
                }
                _ if arg.starts_with("--raster-projection-rows-per-tile=") => {
                    let value = arg.split_once('=').map(|(_, value)| value).expect(
                        "split_once should succeed for --raster-projection-rows-per-tile=value",
                    );
                    raster_projection_rows_per_tile =
                        Some(parse_raster_projection_rows_per_tile(value)?);
                }
                "--raster-attention-kv-rows-per-tile" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a row count after {arg}"))?;
                    raster_attention_kv_rows_per_tile =
                        Some(parse_raster_attention_kv_rows_per_tile(&value)?);
                }
                _ if arg.starts_with("--raster-attention-kv-rows-per-tile=") => {
                    let value = arg.split_once('=').map(|(_, value)| value).expect(
                        "split_once should succeed for --raster-attention-kv-rows-per-tile=value",
                    );
                    raster_attention_kv_rows_per_tile =
                        Some(parse_raster_attention_kv_rows_per_tile(value)?);
                }
                "--raster-sequence-rows-per-tile" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a row count after {arg}"))?;
                    raster_sequence_rows_per_tile =
                        Some(parse_raster_sequence_rows_per_tile(&value)?);
                }
                _ if arg.starts_with("--raster-sequence-rows-per-tile=") => {
                    let value = arg.split_once('=').map(|(_, value)| value).expect(
                        "split_once should succeed for --raster-sequence-rows-per-tile=value",
                    );
                    raster_sequence_rows_per_tile =
                        Some(parse_raster_sequence_rows_per_tile(value)?);
                }
                "--raster-head-rows-per-tile" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a row count after {arg}"))?;
                    raster_head_rows_per_tile = Some(parse_raster_head_rows_per_tile(&value)?);
                }
                _ if arg.starts_with("--raster-head-rows-per-tile=") => {
                    let value = arg
                        .split_once('=')
                        .map(|(_, value)| value)
                        .expect("split_once should succeed for --raster-head-rows-per-tile=value");
                    raster_head_rows_per_tile = Some(parse_raster_head_rows_per_tile(value)?);
                }
                "--raster-tokenizer-bpe-pairs-per-tile" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a pair count after {arg}"))?;
                    raster_tokenizer_bpe_pairs_per_tile =
                        Some(parse_raster_tokenizer_bpe_pairs_per_tile(&value)?);
                }
                _ if arg.starts_with("--raster-tokenizer-bpe-pairs-per-tile=") => {
                    let value = arg.split_once('=').map(|(_, value)| value).expect(
                        "split_once should succeed for --raster-tokenizer-bpe-pairs-per-tile=value",
                    );
                    raster_tokenizer_bpe_pairs_per_tile =
                        Some(parse_raster_tokenizer_bpe_pairs_per_tile(value)?);
                }
                "--raster-tokenizer-bpe-pieces-per-tile" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a piece count after {arg}"))?;
                    raster_tokenizer_bpe_pieces_per_tile =
                        Some(parse_raster_tokenizer_bpe_pieces_per_tile(&value)?);
                }
                _ if arg.starts_with("--raster-tokenizer-bpe-pieces-per-tile=") => {
                    let value = arg
                        .split_once('=')
                        .map(|(_, value)| value)
                        .expect("split_once should succeed for --raster-tokenizer-bpe-pieces-per-tile=value");
                    raster_tokenizer_bpe_pieces_per_tile =
                        Some(parse_raster_tokenizer_bpe_pieces_per_tile(value)?);
                }
                "--raster-output-byte-flush-bytes-per-tile" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("expected a byte count after {arg}"))?;
                    raster_output_byte_flush_bytes_per_tile =
                        Some(parse_raster_output_byte_flush_bytes_per_tile(&value)?);
                }
                _ if arg.starts_with("--raster-output-byte-flush-bytes-per-tile=") => {
                    let value = arg
                        .split_once('=')
                        .map(|(_, value)| value)
                        .expect("split_once should succeed for --raster-output-byte-flush-bytes-per-tile=value");
                    raster_output_byte_flush_bytes_per_tile =
                        Some(parse_raster_output_byte_flush_bytes_per_tile(value)?);
                }
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
        if raster_decode_only {
            anyhow::bail!("--raster-decode-only has been removed; use --raster");
        }
        let raster_execution_requested = raster || raster_detour.is_some();
        if raster && raster_detour.is_some() {
            anyhow::bail!("--raster and --raster-at cannot be used together");
        }
        if raster_integrity_mode.is_unchecked_test_only() && !raster_execution_requested {
            anyhow::bail!("--raster-unchecked-test-mode requires --raster or --raster-at");
        }
        if raster_projection_rows_per_tile.is_some() && !raster_execution_requested {
            anyhow::bail!("--raster-projection-rows-per-tile requires --raster or --raster-at");
        }
        if raster_attention_kv_rows_per_tile.is_some() && !raster_execution_requested {
            anyhow::bail!("--raster-attention-kv-rows-per-tile requires --raster or --raster-at");
        }
        if raster_sequence_rows_per_tile.is_some() && !raster_execution_requested {
            anyhow::bail!("--raster-sequence-rows-per-tile requires --raster or --raster-at");
        }
        if raster_head_rows_per_tile.is_some() && !raster_execution_requested {
            anyhow::bail!("--raster-head-rows-per-tile requires --raster or --raster-at");
        }
        if raster_tokenizer_bpe_pairs_per_tile.is_some() && !raster_execution_requested {
            anyhow::bail!("--raster-tokenizer-bpe-pairs-per-tile requires --raster or --raster-at");
        }
        if raster_tokenizer_bpe_pieces_per_tile.is_some() && !raster_execution_requested {
            anyhow::bail!(
                "--raster-tokenizer-bpe-pieces-per-tile requires --raster or --raster-at"
            );
        }
        if raster_output_byte_flush_bytes_per_tile.is_some() && !raster_execution_requested {
            anyhow::bail!(
                "--raster-output-byte-flush-bytes-per-tile requires --raster or --raster-at"
            );
        }

        if raster_execution_requested {
            execution_mode = InferenceExecutionMode::Deterministic;
        }

        Ok(Self {
            commit_checkpoints,
            execution_mode,
            raster,
            raster_detour,
            raster_integrity_mode,
            raster_trace_tiles,
            raster_projection_rows_per_tile,
            raster_attention_kv_rows_per_tile,
            raster_sequence_rows_per_tile,
            raster_head_rows_per_tile,
            prefill_token_range_width,
            decode_layer_range_width,
            raster_tokenizer_bpe_pairs_per_tile,
            raster_tokenizer_bpe_pieces_per_tile,
            raster_output_byte_flush_bytes_per_tile,
            terminal_checkpoint,
            model_id: positional_args[0].clone(),
            tokenizer_path: PathBuf::from(&positional_args[1]),
            template_path: PathBuf::from(&positional_args[2]),
            model_path: PathBuf::from(&positional_args[3]),
            prompt: positional_args[4..].join(" "),
        })
    }
}

/// Builds the global rayon pool with `RASTER_NUM_THREADS` worker threads when
/// the variable is set; otherwise rayon's default pool (one thread per
/// logical core) is used. Parallelism is pure scheduling — committed bytes
/// are identical for every thread count. `RASTER_PARALLELISM=off` forces the
/// serial reference kernel paths regardless of pool size.
fn configure_rayon_thread_pool() -> anyhow::Result<()> {
    let Ok(value) = env::var("RASTER_NUM_THREADS") else {
        return Ok(());
    };
    let threads = value
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("RASTER_NUM_THREADS must be a positive integer"))?;
    if threads == 0 {
        anyhow::bail!("RASTER_NUM_THREADS must be greater than zero");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|error| anyhow::anyhow!("failed to build rayon thread pool: {error}"))
}

fn parse_raster_unchecked_test_mode_flag() -> anyhow::Result<RasterIntegrityMode> {
    #[cfg(feature = "unchecked-raster-integrity")]
    {
        Ok(RasterIntegrityMode::UncheckedTestOnly)
    }

    #[cfg(not(feature = "unchecked-raster-integrity"))]
    {
        anyhow::bail!(
            "--raster-unchecked-test-mode requires building with the unchecked-raster-integrity feature"
        )
    }
}

fn parse_raster_projection_rows_per_tile(value: &str) -> anyhow::Result<usize> {
    let rows = value.parse::<usize>().map_err(|_| {
        anyhow::anyhow!("raster projection rows per tile must be a positive integer")
    })?;
    if rows == 0 {
        anyhow::bail!("raster projection rows per tile must be greater than zero");
    }
    Ok(rows)
}

fn parse_raster_attention_kv_rows_per_tile(value: &str) -> anyhow::Result<usize> {
    let rows = value.parse::<usize>().map_err(|_| {
        anyhow::anyhow!("raster attention KV rows per tile must be a positive integer")
    })?;
    if rows == 0 {
        anyhow::bail!("raster attention KV rows per tile must be greater than zero");
    }
    Ok(rows)
}

fn parse_raster_sequence_rows_per_tile(value: &str) -> anyhow::Result<usize> {
    let rows = value
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("raster sequence rows per tile must be a positive integer"))?;
    if rows == 0 {
        anyhow::bail!("raster sequence rows per tile must be greater than zero");
    }
    Ok(rows)
}

fn parse_raster_head_rows_per_tile(value: &str) -> anyhow::Result<usize> {
    let rows = value
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("raster head rows per tile must be a positive integer"))?;
    if rows == 0 {
        anyhow::bail!("raster head rows per tile must be greater than zero");
    }
    Ok(rows)
}

fn parse_prefill_token_range_width(value: &str) -> anyhow::Result<usize> {
    let tokens = value
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("prefill token range width must be a positive integer"))?;
    if tokens == 0 {
        anyhow::bail!("prefill token range width must be greater than zero");
    }
    Ok(tokens)
}

fn parse_decode_layer_range_width(value: &str) -> anyhow::Result<usize> {
    let layers = value
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("decode layer range width must be a positive integer"))?;
    if layers == 0 {
        anyhow::bail!("decode layer range width must be greater than zero");
    }
    Ok(layers)
}

fn parse_raster_tokenizer_bpe_pairs_per_tile(value: &str) -> anyhow::Result<usize> {
    let pairs = value.parse::<usize>().map_err(|_| {
        anyhow::anyhow!("raster tokenizer BPE pairs per tile must be a positive integer")
    })?;
    if pairs == 0 {
        anyhow::bail!("raster tokenizer BPE pairs per tile must be greater than zero");
    }
    Ok(pairs)
}

fn parse_raster_tokenizer_bpe_pieces_per_tile(value: &str) -> anyhow::Result<usize> {
    let pieces = value.parse::<usize>().map_err(|_| {
        anyhow::anyhow!("raster tokenizer BPE pieces per tile must be a positive integer")
    })?;
    if pieces == 0 {
        anyhow::bail!("raster tokenizer BPE pieces per tile must be greater than zero");
    }
    Ok(pieces)
}

fn parse_raster_output_byte_flush_bytes_per_tile(value: &str) -> anyhow::Result<usize> {
    let bytes = value.parse::<usize>().map_err(|_| {
        anyhow::anyhow!("raster output byte flush bytes per tile must be a positive integer")
    })?;
    if bytes == 0 {
        anyhow::bail!("raster output byte flush bytes per tile must be greater than zero");
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::CliArgs;
    #[cfg(feature = "unchecked-raster-integrity")]
    use raster_inference::RasterIntegrityMode;
    use raster_inference::{InferenceExecutionMode, RoutineId};

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
        assert!(!args.raster);
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
    fn parse_terminal_checkpoint_occurrence_suffix() {
        let args = CliArgs::parse([
            "--terminal-checkpoint".to_string(),
            "prefill.range_finalize:2".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(
            args.terminal_checkpoint.as_deref(),
            Some("prefill.range_finalize:2")
        );
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

        assert!(args.raster);
        assert_eq!(args.execution_mode, InferenceExecutionMode::Deterministic);
        assert!(!args.raster_trace_tiles);
        assert_eq!(args.raster_projection_rows_per_tile, None);
        assert_eq!(args.raster_attention_kv_rows_per_tile, None);
        assert_eq!(args.raster_output_byte_flush_bytes_per_tile, None);
    }

    #[test]
    fn parse_raster_at_flag() {
        let args = CliArgs::parse([
            "--raster-at".to_string(),
            "prefill.range:2".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        let detour = args.raster_detour.expect("detour should be parsed");
        assert_eq!(detour.routine_id(), RoutineId::PrefillRange);
        assert_eq!(detour.occurrence(), 2);
        assert_eq!(args.execution_mode, InferenceExecutionMode::Deterministic);
        assert!(!args.raster);
    }

    #[test]
    fn parse_raster_at_equals_flag() {
        let args = CliArgs::parse([
            "--raster-at=input.embedding".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        let detour = args.raster_detour.expect("detour should be parsed");
        assert_eq!(detour.routine_id(), RoutineId::InputEmbedding);
        assert_eq!(detour.occurrence(), 1);
        assert_eq!(args.execution_mode, InferenceExecutionMode::Deterministic);
    }

    #[test]
    fn parse_raster_at_rejects_full_raster() {
        let error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-at=prefill.range".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("full raster and raster-at should conflict");

        assert!(error
            .to_string()
            .contains("--raster and --raster-at cannot be used together"));
    }

    #[test]
    fn parse_raster_at_requires_value() {
        let error = CliArgs::parse([
            "--raster-at".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("missing detour value should fail");

        assert!(error.to_string().contains("unknown routine id `model`"));
    }

    #[test]
    fn parse_raster_at_rejects_unknown_routine() {
        let error = CliArgs::parse([
            "--raster-at=not.a.routine".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("unknown detour target should fail");

        assert!(error.to_string().contains("unknown routine id"));
        assert!(error.to_string().contains("prefill.range"));
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn parse_raster_unchecked_test_mode_flag() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "--raster-unchecked-test-mode".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(
            args.raster_integrity_mode,
            RasterIntegrityMode::UncheckedTestOnly
        );
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn parse_raster_unchecked_test_mode_with_raster_at() {
        let args = CliArgs::parse([
            "--raster-at=prefill.range".to_string(),
            "--raster-unchecked-test-mode".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(
            args.raster_integrity_mode,
            RasterIntegrityMode::UncheckedTestOnly
        );
    }

    #[test]
    fn parse_raster_unchecked_test_mode_requires_raster_or_feature() {
        let error = CliArgs::parse([
            "--raster-unchecked-test-mode".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("unchecked test mode should not parse without raster");

        #[cfg(feature = "unchecked-raster-integrity")]
        assert!(error
            .to_string()
            .contains("--raster-unchecked-test-mode requires --raster or --raster-at"));

        #[cfg(not(feature = "unchecked-raster-integrity"))]
        assert!(error
            .to_string()
            .contains("requires building with the unchecked-raster-integrity feature"));
    }

    #[test]
    fn parse_raster_decode_only_flag_is_removed() {
        let error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-decode-only".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("raster decode-only should be removed");

        assert!(error
            .to_string()
            .contains("--raster-decode-only has been removed"));
    }

    #[test]
    fn parse_raster_decode_only_without_raster_is_removed() {
        let error = CliArgs::parse([
            "--raster-decode-only".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("raster decode-only should be removed");

        assert!(error
            .to_string()
            .contains("--raster-decode-only has been removed"));
    }

    #[test]
    fn parse_raster_trace_tiles_flag() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "--raster-trace-tiles".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert!(args.raster_trace_tiles);
    }

    #[test]
    fn parse_prefill_token_range_width_flag() {
        let args = CliArgs::parse([
            "--deterministic".to_string(),
            "--prefill-token-range-width".to_string(),
            "100".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.prefill_token_range_width, Some(100));

        let args = CliArgs::parse([
            "--prefill-token-range-width=64".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.prefill_token_range_width, Some(64));
    }

    #[test]
    fn parse_prefill_token_range_width_rejects_zero() {
        let error = CliArgs::parse([
            "--prefill-token-range-width=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero range width should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_decode_layer_range_width_flag() {
        let args = CliArgs::parse([
            "--deterministic".to_string(),
            "--decode-layer-range-width".to_string(),
            "4".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.decode_layer_range_width, Some(4));

        let args = CliArgs::parse([
            "--decode-layer-range-width=8".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.decode_layer_range_width, Some(8));
    }

    #[test]
    fn parse_decode_layer_range_width_rejects_zero() {
        let error = CliArgs::parse([
            "--decode-layer-range-width=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero range width should fail");

        assert!(error
            .to_string()
            .contains("decode layer range width must be greater than zero"));
    }

    #[test]
    fn parse_raster_tokenizer_bpe_chunk_flags() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "--raster-tokenizer-bpe-pairs-per-tile=2".to_string(),
            "--raster-tokenizer-bpe-pieces-per-tile".to_string(),
            "3".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.raster_tokenizer_bpe_pairs_per_tile, Some(2));
        assert_eq!(args.raster_tokenizer_bpe_pieces_per_tile, Some(3));
    }

    #[test]
    fn parse_raster_tokenizer_bpe_chunk_flags_reject_zero() {
        let pair_error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-tokenizer-bpe-pairs-per-tile=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero pairs per tile should fail");
        assert!(pair_error.to_string().contains("greater than zero"));

        let piece_error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-tokenizer-bpe-pieces-per-tile=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero pieces per tile should fail");
        assert!(piece_error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_raster_tokenizer_bpe_chunk_flags_require_raster() {
        let pair_error = CliArgs::parse([
            "--raster-tokenizer-bpe-pairs-per-tile=2".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("BPE pair chunk flag should require raster");
        assert!(pair_error
            .to_string()
            .contains("--raster-tokenizer-bpe-pairs-per-tile requires --raster or --raster-at"));

        let piece_error = CliArgs::parse([
            "--raster-tokenizer-bpe-pieces-per-tile=3".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("BPE piece chunk flag should require raster");
        assert!(piece_error
            .to_string()
            .contains("--raster-tokenizer-bpe-pieces-per-tile requires --raster or --raster-at"));
    }

    #[test]
    fn parse_raster_output_byte_flush_chunk_flag() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "--raster-output-byte-flush-bytes-per-tile=1048576".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(
            args.raster_output_byte_flush_bytes_per_tile,
            Some(1_048_576)
        );
    }

    #[test]
    fn parse_raster_output_byte_flush_chunk_flag_rejects_zero() {
        let error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-output-byte-flush-bytes-per-tile=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero byte flush chunk size should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_raster_output_byte_flush_chunk_flag_requires_raster() {
        let error = CliArgs::parse([
            "--raster-output-byte-flush-bytes-per-tile=1048576".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("output byte flush chunk flag should require raster");

        assert!(error.to_string().contains(
            "--raster-output-byte-flush-bytes-per-tile requires --raster or --raster-at"
        ));
    }

    #[test]
    fn parse_raster_projection_rows_per_tile_flag() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "--raster-projection-rows-per-tile".to_string(),
            "4".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.raster_projection_rows_per_tile, Some(4));
    }

    #[test]
    fn parse_raster_sizing_flags_with_raster_at() {
        let args = CliArgs::parse([
            "--raster-at=prefill.range:2".to_string(),
            "--raster-projection-rows-per-tile=4".to_string(),
            "--raster-attention-kv-rows-per-tile=8".to_string(),
            "--raster-sequence-rows-per-tile=3".to_string(),
            "--raster-head-rows-per-tile=2".to_string(),
            "--raster-tokenizer-bpe-pairs-per-tile=5".to_string(),
            "--raster-tokenizer-bpe-pieces-per-tile=6".to_string(),
            "--raster-output-byte-flush-bytes-per-tile=1048576".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.raster_projection_rows_per_tile, Some(4));
        assert_eq!(args.raster_attention_kv_rows_per_tile, Some(8));
        assert_eq!(args.raster_sequence_rows_per_tile, Some(3));
        assert_eq!(args.raster_head_rows_per_tile, Some(2));
        assert_eq!(args.raster_tokenizer_bpe_pairs_per_tile, Some(5));
        assert_eq!(args.raster_tokenizer_bpe_pieces_per_tile, Some(6));
        assert_eq!(
            args.raster_output_byte_flush_bytes_per_tile,
            Some(1_048_576)
        );
        assert_eq!(args.execution_mode, InferenceExecutionMode::Deterministic);
    }

    #[test]
    fn parse_raster_attention_kv_rows_per_tile_flag() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "--raster-attention-kv-rows-per-tile=8".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.raster_attention_kv_rows_per_tile, Some(8));
    }

    #[test]
    fn parse_raster_attention_kv_rows_per_tile_rejects_zero() {
        let error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-attention-kv-rows-per-tile=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_raster_attention_kv_rows_per_tile_requires_raster() {
        let error = CliArgs::parse([
            "--raster-attention-kv-rows-per-tile=8".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("raster attention flag should require raster");

        assert!(error
            .to_string()
            .contains("--raster-attention-kv-rows-per-tile requires --raster or --raster-at"));
    }

    #[test]
    fn parse_raster_sequence_and_head_rows_per_tile_flags() {
        let args = CliArgs::parse([
            "--raster".to_string(),
            "--raster-sequence-rows-per-tile=3".to_string(),
            "--raster-head-rows-per-tile".to_string(),
            "4".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect("cli args should parse");

        assert_eq!(args.raster_sequence_rows_per_tile, Some(3));
        assert_eq!(args.raster_head_rows_per_tile, Some(4));
    }

    #[test]
    fn parse_raster_sequence_and_head_rows_per_tile_reject_zero() {
        let sequence_error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-sequence-rows-per-tile=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero sequence rows per tile should fail");
        assert!(sequence_error.to_string().contains("greater than zero"));

        let head_error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-head-rows-per-tile=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero head rows per tile should fail");
        assert!(head_error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_raster_sequence_and_head_rows_per_tile_require_raster() {
        let sequence_error = CliArgs::parse([
            "--raster-sequence-rows-per-tile=3".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("sequence rows flag should require raster");
        assert!(sequence_error
            .to_string()
            .contains("--raster-sequence-rows-per-tile requires --raster or --raster-at"));

        let head_error = CliArgs::parse([
            "--raster-head-rows-per-tile=4".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("head rows flag should require raster");
        assert!(head_error
            .to_string()
            .contains("--raster-head-rows-per-tile requires --raster or --raster-at"));
    }

    #[test]
    fn parse_raster_projection_rows_per_tile_rejects_zero() {
        let error = CliArgs::parse([
            "--raster".to_string(),
            "--raster-projection-rows-per-tile=0".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("zero rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_raster_projection_rows_per_tile_requires_raster() {
        let error = CliArgs::parse([
            "--raster-projection-rows-per-tile=4".to_string(),
            "model".to_string(),
            "tokenizer.json".to_string(),
            "chat_template.jinja".to_string(),
            "model-path".to_string(),
            "hello".to_string(),
        ])
        .expect_err("raster rows per tile should require raster mode");

        assert!(error
            .to_string()
            .contains("requires --raster or --raster-at"));
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
