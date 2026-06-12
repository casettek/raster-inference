use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::{env, fs};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Deserialize;
use tokenizers::Tokenizer;

use raster_inference::shared::model::gemma::transformer::Gemma4TransformerModel;
use raster_inference::{
    challenger, claimer, detour, load_chat_template, load_gemma_tokenizer_spec_from_path,
    load_tokenizer_from_path, load_transformer_state_model_from_det_num_wgt_path, protocol, trace,
    AuditOutcome, AuthenticatedGemmaTokenizer, ClaimerOptions, ClaimerRunOutcome, ExecutionTuning,
    InferenceExecutionMode, InferenceRequest, ModelSpec, RasterDetourSpec, SamplingConfig,
    TextDecodingPolicy,
};

/// Exit code when `audit` finds a divergence (`0` = no divergence,
/// `1` = operational error). Part of the scripting contract.
const EXIT_CODE_DIVERGENCE: u8 = 2;

#[derive(Debug, Parser)]
#[command(
    name = "raster-inference",
    version,
    about = "Verifiable deterministic inference for the Raster fraud-proof protocol"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run an inference and produce the checkpoint trace artifact
    Claim(ClaimArgs),
    /// Re-run with exactly one raster routine occurrence swapped in,
    /// producing the raster detour trace artifact
    Detour(DetourArgs),
    /// Replay a claimed trace, report the first divergence, and detour at it
    Audit(AuditArgs),
}

#[derive(Debug, Args)]
struct CommonArgs {
    /// Model directory containing tokenizer.json, chat_template.jinja and
    /// model.detwgt
    #[arg(long, value_name = "DIR")]
    model: PathBuf,

    /// Model id (defaults to the model directory name)
    #[arg(long, value_name = "ID")]
    model_id: Option<String>,

    /// Execution tuning TOML file (tile sizing and range widths)
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Maximum number of new tokens to generate
    #[arg(long, value_name = "N", default_value_t = 16)]
    max_new_tokens: usize,

    /// Directory for trace artifacts (overrides RASTER_TRACE_DIR)
    #[arg(long, value_name = "DIR")]
    trace_dir: Option<PathBuf>,

    /// Read the prompt from a file instead of trailing arguments
    #[arg(long, value_name = "FILE", conflicts_with = "prompt")]
    prompt_file: Option<PathBuf>,

    /// Prompt text
    #[arg(value_name = "PROMPT")]
    prompt: Vec<String>,
}

#[derive(Debug, Args)]
struct ClaimArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Pause the run after the named checkpoint, e.g. prefill.finalize or
    /// prefill.range_finalize:2 (debug; output is the paused-state JSON)
    #[arg(long, value_name = "CHECKPOINT[:OCC]")]
    stop_at: Option<String>,
}

#[derive(Debug, Args)]
struct DetourArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Routine occurrence to execute at raster (tile) level, e.g.
    /// prefill.range:2
    #[arg(long, value_name = "ROUTINE[:OCC]")]
    at: String,

    /// Print verbose routine and tile execution logs to stderr
    #[arg(long)]
    trace_tiles: bool,
}

#[derive(Debug, Args)]
struct AuditArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Path to the claimed checkpoint trace artifact to audit
    #[arg(long, value_name = "FILE")]
    claimed: PathBuf,

    /// Print verbose routine and tile execution logs to stderr
    #[arg(long)]
    trace_tiles: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    configure_rayon_thread_pool()?;
    match cli.command {
        Command::Claim(args) => run_claim(args),
        Command::Detour(args) => run_detour(args),
        Command::Audit(args) => run_audit(args),
    }
}

fn run_claim(args: ClaimArgs) -> Result<ExitCode> {
    let ctx = RunContext::prepare(&args.common)?;
    eprintln!("claim: model {}", ctx.model.model_id);
    let options = ClaimerOptions {
        terminal_checkpoint: args.stop_at,
        tuning: ctx.tuning.clone(),
    };
    let outcome = claimer::run(
        &ctx.request,
        &ctx.model,
        &ctx.tokenizer,
        &ctx.transformer_model,
        ctx.raster_tokenizer_source.clone(),
        &options,
    )?;
    match outcome {
        ClaimerRunOutcome::Completed(outcome) => {
            eprintln!("claim: trace artifact {}", outcome.trace_path.display());
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "trace_path": outcome.trace_path,
                    "state": outcome.state,
                }))?
            );
        }
        ClaimerRunOutcome::Paused(paused) => {
            eprintln!(
                "claim: paused at checkpoint {}",
                paused.terminal_checkpoint_id
            );
            println!("{}", serde_json::to_string_pretty(&paused)?);
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn run_detour(args: DetourArgs) -> Result<ExitCode> {
    let spec = RasterDetourSpec::parse(&args.at)?;
    let ctx = RunContext::prepare(&args.common)?;
    eprintln!(
        "detour: model {}, raster routine {}:{}",
        ctx.model.model_id,
        spec.routine_id(),
        spec.occurrence()
    );
    let run = || {
        detour::run(
            &ctx.request,
            &ctx.model,
            &ctx.tokenizer,
            &ctx.transformer_model,
            ctx.raster_tokenizer_source.clone(),
            spec,
            &ctx.tuning,
        )
    };
    let outcome = if args.trace_tiles {
        trace::with_trace_logging_enabled(true, run)
    } else {
        run()
    }?;
    eprintln!(
        "detour: trace artifact {}",
        outcome.artifact.trace_path.display()
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "routine": outcome.artifact.spec.routine_id().to_string(),
            "occurrence": outcome.artifact.spec.occurrence(),
            "trace_path": outcome.artifact.trace_path,
            "state": outcome.state,
        }))?
    );
    Ok(ExitCode::SUCCESS)
}

fn run_audit(args: AuditArgs) -> Result<ExitCode> {
    let claimed = fs::read(&args.claimed).with_context(|| {
        format!(
            "failed to read claimed trace artifact {}",
            args.claimed.display()
        )
    })?;
    let ctx = RunContext::prepare(&args.common)?;
    eprintln!(
        "audit: model {}, claimed trace {}",
        ctx.model.model_id,
        args.claimed.display()
    );
    let run = || {
        challenger::audit(
            &ctx.request,
            &ctx.model,
            &ctx.tokenizer,
            &ctx.transformer_model,
            ctx.raster_tokenizer_source.clone(),
            &claimed,
            &ctx.tuning,
        )
    };
    let outcome = if args.trace_tiles {
        trace::with_trace_logging_enabled(true, run)
    } else {
        run()
    }?;
    match outcome {
        AuditOutcome::NoDivergence => {
            eprintln!("audit: no divergence");
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "result": "no_divergence",
                }))?
            );
            Ok(ExitCode::SUCCESS)
        }
        AuditOutcome::Diverged { divergence, detour } => {
            eprintln!(
                "audit: divergence at checkpoint {}:{} (entry {})",
                divergence.checkpoint_id, divergence.occurrence, divergence.entry_index
            );
            let detour_json = detour.map(|artifact| {
                eprintln!(
                    "audit: detour trace artifact {}",
                    artifact.trace_path.display()
                );
                serde_json::json!({
                    "routine": artifact.spec.routine_id().to_string(),
                    "occurrence": artifact.spec.occurrence(),
                    "trace_path": artifact.trace_path,
                })
            });
            if detour_json.is_none() {
                eprintln!("audit: divergent routine is not detourable; no detour artifact");
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "result": "divergence",
                    "divergence": divergence,
                    "detour": detour_json,
                }))?
            );
            Ok(ExitCode::from(EXIT_CODE_DIVERGENCE))
        }
    }
}

/// Everything a role entry point needs, loaded once per invocation.
struct RunContext {
    request: InferenceRequest,
    model: ModelSpec,
    tokenizer: Tokenizer,
    transformer_model: Gemma4TransformerModel,
    raster_tokenizer_source: AuthenticatedGemmaTokenizer,
    tuning: ExecutionTuning,
}

impl RunContext {
    fn prepare(common: &CommonArgs) -> Result<Self> {
        if let Some(trace_dir) = &common.trace_dir {
            env::set_var("RASTER_TRACE_DIR", trace_dir);
        }
        let assets = resolve_model_dir(&common.model, common.model_id.clone())?;
        let tuning = load_tuning(common.config.as_deref())?;
        let prompt = resolve_prompt(common)?;

        let chat_template = load_chat_template(&assets.template_path)?;
        let tokenizer = load_tokenizer_from_path(&assets.tokenizer_path)?;
        let raster_tokenizer_source = AuthenticatedGemmaTokenizer::new(
            load_gemma_tokenizer_spec_from_path(&assets.tokenizer_path)?,
        );
        let transformer_model =
            load_transformer_state_model_from_det_num_wgt_path(&assets.weights_path)?;

        let model = ModelSpec {
            model_id: assets.model_id,
            tokenizer_path: assets.tokenizer_path,
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
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(common.max_new_tokens),
                temperature: Some(protocol::SAMPLING_TEMPERATURE),
                top_k: None,
                top_p: None,
            },
        };
        Ok(Self {
            request,
            model,
            tokenizer,
            transformer_model,
            raster_tokenizer_source,
            tuning,
        })
    }
}

/// Asset paths resolved from a model directory by convention.
#[derive(Debug, PartialEq, Eq)]
struct ModelAssets {
    model_id: String,
    tokenizer_path: PathBuf,
    template_path: PathBuf,
    weights_path: PathBuf,
}

fn resolve_model_dir(dir: &Path, model_id: Option<String>) -> Result<ModelAssets> {
    if !dir.is_dir() {
        anyhow::bail!("model directory {} does not exist", dir.display());
    }
    let model_id = match model_id {
        Some(id) => id,
        None => dir
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .with_context(|| {
                format!(
                    "cannot derive a model id from {}; pass --model-id",
                    dir.display()
                )
            })?,
    };
    let asset = |name: &str| -> Result<PathBuf> {
        let path = dir.join(name);
        if !path.is_file() {
            anyhow::bail!("model directory {} is missing {name}", dir.display());
        }
        Ok(path)
    };
    Ok(ModelAssets {
        model_id,
        tokenizer_path: asset("tokenizer.json")?,
        template_path: asset("chat_template.jinja")?,
        weights_path: asset("model.detwgt")?,
    })
}

fn resolve_prompt(common: &CommonArgs) -> Result<String> {
    if let Some(path) = &common.prompt_file {
        return fs::read_to_string(path)
            .map(|text| text.trim_end_matches('\n').to_string())
            .with_context(|| format!("failed to read prompt file {}", path.display()));
    }
    if common.prompt.is_empty() {
        anyhow::bail!("a prompt is required: pass trailing arguments or --prompt-file");
    }
    Ok(common.prompt.join(" "))
}

// ---------------------------------------------------------------------------
// Execution tuning config file
// ---------------------------------------------------------------------------

/// `--config` TOML shape. Every key is optional; omitted keys use the
/// library defaults.
#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TuningConfig {
    #[serde(default)]
    ranges: RangesConfig,
    #[serde(default)]
    tile_sizing: TileSizingConfig,
}

#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RangesConfig {
    prefill_token_range_width: Option<usize>,
    decode_layer_range_width: Option<usize>,
}

#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TileSizingConfig {
    projection_rows_per_tile: Option<usize>,
    attention_kv_rows_per_tile: Option<usize>,
    sequence_rows_per_tile: Option<usize>,
    head_rows_per_tile: Option<usize>,
    tokenizer_bpe_pairs_per_tile: Option<usize>,
    tokenizer_bpe_pieces_per_tile: Option<usize>,
    output_byte_flush_bytes_per_tile: Option<usize>,
}

impl TuningConfig {
    fn into_tuning(self) -> ExecutionTuning {
        ExecutionTuning {
            prefill_token_range_width: self.ranges.prefill_token_range_width,
            decode_layer_range_width: self.ranges.decode_layer_range_width,
            projection_rows_per_tile: self.tile_sizing.projection_rows_per_tile,
            attention_kv_rows_per_tile: self.tile_sizing.attention_kv_rows_per_tile,
            sequence_rows_per_tile: self.tile_sizing.sequence_rows_per_tile,
            head_rows_per_tile: self.tile_sizing.head_rows_per_tile,
            tokenizer_bpe_pairs_per_tile: self.tile_sizing.tokenizer_bpe_pairs_per_tile,
            tokenizer_bpe_pieces_per_tile: self.tile_sizing.tokenizer_bpe_pieces_per_tile,
            output_byte_flush_bytes_per_tile: self.tile_sizing.output_byte_flush_bytes_per_tile,
        }
    }
}

fn load_tuning(path: Option<&Path>) -> Result<ExecutionTuning> {
    let Some(path) = path else {
        return Ok(ExecutionTuning::default());
    };
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    parse_tuning(&text).with_context(|| format!("invalid config file {}", path.display()))
}

fn parse_tuning(text: &str) -> Result<ExecutionTuning> {
    let config: TuningConfig = toml::from_str(text)?;
    Ok(config.into_tuning())
}

/// Builds the global rayon pool with `RASTER_NUM_THREADS` worker threads when
/// the variable is set; otherwise rayon's default pool (one thread per
/// logical core) is used. Parallelism is pure scheduling — committed bytes
/// are identical for every thread count. `RASTER_PARALLELISM=off` forces the
/// serial reference kernel paths regardless of pool size.
fn configure_rayon_thread_pool() -> Result<()> {
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

#[cfg(test)]
mod tests {
    use clap::Parser;
    use raster_inference::ExecutionTuning;

    use super::{parse_tuning, resolve_prompt, Cli, Command};

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("cli args should parse")
    }

    #[test]
    fn claim_parses_stop_at_and_prompt() {
        let cli = parse(&[
            "raster-inference",
            "claim",
            "--model",
            "assets/tiny-gemma-dev",
            "--stop-at",
            "prefill.range_finalize:2",
            "--max-new-tokens",
            "5",
            "hello",
            "world",
        ]);
        let Command::Claim(args) = cli.command else {
            panic!("expected claim subcommand");
        };
        assert_eq!(args.stop_at.as_deref(), Some("prefill.range_finalize:2"));
        assert_eq!(args.common.max_new_tokens, 5);
        assert_eq!(
            resolve_prompt(&args.common).expect("prompt should resolve"),
            "hello world"
        );
    }

    #[test]
    fn claim_defaults_max_new_tokens() {
        let cli = parse(&[
            "raster-inference",
            "claim",
            "--model",
            "assets/tiny-gemma-dev",
            "hi",
        ]);
        let Command::Claim(args) = cli.command else {
            panic!("expected claim subcommand");
        };
        assert_eq!(args.common.max_new_tokens, 16);
        assert!(args.stop_at.is_none());
    }

    #[test]
    fn claim_requires_a_prompt() {
        let cli = parse(&[
            "raster-inference",
            "claim",
            "--model",
            "assets/tiny-gemma-dev",
        ]);
        let Command::Claim(args) = cli.command else {
            panic!("expected claim subcommand");
        };
        let error = resolve_prompt(&args.common).expect_err("missing prompt should fail");
        assert!(error.to_string().contains("a prompt is required"));
    }

    #[test]
    fn prompt_file_conflicts_with_positional_prompt() {
        let error = Cli::try_parse_from([
            "raster-inference",
            "claim",
            "--model",
            "assets/tiny-gemma-dev",
            "--prompt-file",
            "prompt.txt",
            "hello",
        ])
        .expect_err("prompt file and positional prompt should conflict");
        assert!(error.to_string().contains("cannot be used with"));
    }

    #[test]
    fn detour_requires_at() {
        let error = Cli::try_parse_from([
            "raster-inference",
            "detour",
            "--model",
            "assets/tiny-gemma-dev",
            "hello",
        ])
        .expect_err("detour without --at should fail");
        assert!(error.to_string().contains("--at"));
    }

    #[test]
    fn detour_parses_at_and_trace_tiles() {
        let cli = parse(&[
            "raster-inference",
            "detour",
            "--model",
            "assets/tiny-gemma-dev",
            "--at",
            "prefill.range:2",
            "--trace-tiles",
            "hello",
        ]);
        let Command::Detour(args) = cli.command else {
            panic!("expected detour subcommand");
        };
        assert_eq!(args.at, "prefill.range:2");
        assert!(args.trace_tiles);
    }

    #[test]
    fn audit_requires_claimed() {
        let error = Cli::try_parse_from([
            "raster-inference",
            "audit",
            "--model",
            "assets/tiny-gemma-dev",
            "hello",
        ])
        .expect_err("audit without --claimed should fail");
        assert!(error.to_string().contains("--claimed"));
    }

    #[test]
    fn claim_rejects_trace_tiles() {
        Cli::try_parse_from([
            "raster-inference",
            "claim",
            "--model",
            "assets/tiny-gemma-dev",
            "--trace-tiles",
            "hello",
        ])
        .expect_err("claim should not accept --trace-tiles");
    }

    #[test]
    fn tuning_config_parses_full_shape() {
        let tuning = parse_tuning(
            r#"
            [ranges]
            prefill_token_range_width = 8
            decode_layer_range_width = 4

            [tile_sizing]
            projection_rows_per_tile = 64
            attention_kv_rows_per_tile = 128
            sequence_rows_per_tile = 16
            head_rows_per_tile = 4
            tokenizer_bpe_pairs_per_tile = 1024
            tokenizer_bpe_pieces_per_tile = 512
            output_byte_flush_bytes_per_tile = 4096
            "#,
        )
        .expect("config should parse");

        assert_eq!(
            tuning,
            ExecutionTuning {
                prefill_token_range_width: Some(8),
                decode_layer_range_width: Some(4),
                projection_rows_per_tile: Some(64),
                attention_kv_rows_per_tile: Some(128),
                sequence_rows_per_tile: Some(16),
                head_rows_per_tile: Some(4),
                tokenizer_bpe_pairs_per_tile: Some(1024),
                tokenizer_bpe_pieces_per_tile: Some(512),
                output_byte_flush_bytes_per_tile: Some(4096),
            }
        );
    }

    #[test]
    fn tuning_config_allows_partial_and_empty_files() {
        let tuning = parse_tuning("").expect("empty config should parse");
        assert_eq!(tuning, ExecutionTuning::default());

        let tuning = parse_tuning("[ranges]\nprefill_token_range_width = 8\n")
            .expect("partial config should parse");
        assert_eq!(tuning.prefill_token_range_width, Some(8));
        assert_eq!(tuning.projection_rows_per_tile, None);
    }

    #[test]
    fn tuning_config_rejects_unknown_keys() {
        let error = parse_tuning("[tile_sizing]\nnot_a_knob = 1\n")
            .expect_err("unknown config key should fail");
        assert!(error.to_string().contains("not_a_knob"));
    }

    #[test]
    fn resolve_model_dir_derives_id_and_checks_assets() {
        let dir = std::env::temp_dir().join(format!("raster-cli-test-{}", std::process::id()));
        let model_dir = dir.join("my-model");
        std::fs::create_dir_all(&model_dir).expect("temp model dir should be creatable");
        for name in ["tokenizer.json", "chat_template.jinja"] {
            std::fs::write(model_dir.join(name), b"{}").expect("asset should be writable");
        }

        let error = super::resolve_model_dir(&model_dir, None)
            .expect_err("missing weights should fail resolution");
        assert!(error.to_string().contains("model.detwgt"));

        std::fs::write(model_dir.join("model.detwgt"), b"").expect("asset should be writable");
        let assets =
            super::resolve_model_dir(&model_dir, None).expect("complete dir should resolve");
        assert_eq!(assets.model_id, "my-model");
        assert_eq!(assets.weights_path, model_dir.join("model.detwgt"));

        let assets = super::resolve_model_dir(&model_dir, Some("override".to_string()))
            .expect("complete dir should resolve");
        assert_eq!(assets.model_id, "override");

        std::fs::remove_dir_all(&dir).ok();
    }
}
