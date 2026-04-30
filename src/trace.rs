use std::{
    cell::Cell,
    cell::RefCell,
    env, fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::checkpoints::PhaseId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TraceMode {
    Off,
    Verbose,
}

pub struct TraceSpan {
    label: String,
    start: Instant,
    enabled: bool,
}

impl TraceSpan {
    pub fn new(label: impl Into<String>) -> Self {
        let label = label.into();
        let enabled = trace_logging_enabled();
        if enabled {
            emit("start", &label, None);
        }
        Self {
            label,
            start: Instant::now(),
            enabled,
        }
    }
}

impl Drop for TraceSpan {
    fn drop(&mut self) {
        if self.enabled {
            emit("end", &self.label, Some(self.start.elapsed()));
        }
    }
}

pub fn trace_scope(label: impl Into<String>) -> TraceSpan {
    TraceSpan::new(label)
}

pub fn trace_event(label: impl AsRef<str>) {
    if trace_logging_enabled() {
        emit("event", label.as_ref(), None);
    }
}

pub fn phase_started(phase_id: PhaseId) {
    emit_phase("start", phase_id);
}

pub fn phase_finished(phase_id: PhaseId) {
    emit_phase("end", phase_id);
}

pub fn phase_paused(phase_id: PhaseId) {
    emit_phase("pause", phase_id);
}

pub fn raster_tile_invocations_finished(total: u64) {
    let elapsed = process_start().elapsed().as_secs_f64();
    eprintln!("[raster-tiles +{elapsed:>8.3}s] total {total}");
}

pub fn with_checkpointing_enabled<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    CHECKPOINTING_ENABLED.with(|checkpointing_enabled| {
        let previous = checkpointing_enabled.replace(enabled);
        let reset = ResetCheckpointingFlag(previous);
        let result = f();
        drop(reset);
        result
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalCheckpointSpec {
    checkpoint_id: String,
    occurrence: usize,
}

impl TerminalCheckpointSpec {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        let (checkpoint_id, occurrence) = match value.rsplit_once(':') {
            Some((checkpoint_id, occurrence)) => {
                if checkpoint_id.is_empty() {
                    anyhow::bail!("terminal checkpoint id must not be empty");
                }
                let occurrence = occurrence.parse::<usize>().map_err(|_| {
                    anyhow::anyhow!("terminal checkpoint occurrence must be a positive integer")
                })?;
                (checkpoint_id.to_string(), occurrence)
            }
            None => (value.to_string(), 1),
        };

        if checkpoint_id.is_empty() {
            anyhow::bail!("terminal checkpoint id must not be empty");
        }
        if occurrence == 0 {
            anyhow::bail!("terminal checkpoint occurrence must be greater than zero");
        }

        Ok(Self {
            checkpoint_id,
            occurrence,
        })
    }

    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }

    pub fn occurrence(&self) -> usize {
        self.occurrence
    }
}

#[derive(Clone, Debug)]
struct TerminalCheckpointState {
    spec: TerminalCheckpointSpec,
    seen: usize,
    reached: bool,
}

pub fn with_terminal_checkpoint<T>(
    spec: Option<TerminalCheckpointSpec>,
    f: impl FnOnce() -> T,
) -> T {
    TERMINAL_CHECKPOINT.with(|terminal_checkpoint| {
        let previous = terminal_checkpoint.replace(spec.map(|spec| TerminalCheckpointState {
            spec,
            seen: 0,
            reached: false,
        }));
        let reset = ResetTerminalCheckpoint(previous);
        let result = f();
        drop(reset);
        result
    })
}

pub fn reached_terminal_checkpoint_id() -> Option<String> {
    TERMINAL_CHECKPOINT.with(|terminal_checkpoint| {
        terminal_checkpoint
            .borrow()
            .as_ref()
            .filter(|state| state.reached)
            .map(|state| state.spec.checkpoint_id.clone())
    })
}

#[derive(Default)]
struct TraceCollector {
    checkpoints: Vec<Value>,
    completed_trace_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SerializableLayerKvCache {
    pub keys: Vec<Vec<Vec<f32>>>,
    pub values: Vec<Vec<Vec<f32>>>,
}

pub fn start_inference_trace<T: Serialize>(run_metadata: &T) {
    if !trace_checkpointing_enabled() {
        return;
    }

    let _ = run_metadata;
    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector.checkpoints.clear();
    collector.completed_trace_path = None;
}

pub fn trace_checkpoint<T: Serialize>(checkpoint_name: &str, state: &T) -> bool {
    emit_checkpoint(checkpoint_name);
    let reached_terminal_checkpoint = mark_terminal_checkpoint(checkpoint_name);
    if !trace_checkpointing_enabled() || !should_commit_checkpoint(checkpoint_name) {
        return reached_terminal_checkpoint;
    }

    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector.checkpoints.push(json!({
        checkpoint_name: sha256_hex(state),
    }));
    reached_terminal_checkpoint
}

pub fn finish_inference_trace<T: Serialize>(summary: &T) {
    if !trace_checkpointing_enabled() {
        return;
    }

    let _ = summary;
    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    let payload = Value::Array(std::mem::take(&mut collector.checkpoints));
    collector.completed_trace_path = write_checkpoint_bundle(&payload).ok();
    emit_checkpoint_bundle(&payload, collector.completed_trace_path.as_deref());
}

pub fn abort_inference_trace(error: &anyhow::Error) {
    if !trace_checkpointing_enabled() {
        return;
    }

    let _ = error;
    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    let payload = Value::Array(std::mem::take(&mut collector.checkpoints));
    collector.completed_trace_path = write_checkpoint_bundle(&payload).ok();
    emit_checkpoint_bundle(&payload, collector.completed_trace_path.as_deref());
}

pub fn serialize_layer_caches(
    layer_caches: &[crate::shared::transformer::LayerKvCache],
) -> Vec<SerializableLayerKvCache> {
    // Trace/checkpoint payloads intentionally keep the existing f32 cache shape.
    // Deterministic mode may carry canonical rows internally, but public trace
    // commitments stay anchored to this compatibility view until versioned.
    layer_caches
        .iter()
        .map(|cache| SerializableLayerKvCache {
            keys: cache
                .keys
                .iter()
                .map(|head| head.iter().cloned().collect())
                .collect(),
            values: cache
                .values
                .iter()
                .map(|head| head.iter().cloned().collect())
                .collect(),
        })
        .collect()
}

pub fn sha256_hex<T: Serialize>(value: &T) -> String {
    match serde_json::to_vec(value) {
        Ok(payload) => format!("{:x}", Sha256::digest(payload)),
        Err(error) => format!("serialization-error:{error}"),
    }
}

fn trace_mode() -> TraceMode {
    match env::var("RASTER_TRACE_TILES").as_deref() {
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") => TraceMode::Verbose,
        _ => TraceMode::Off,
    }
}

fn trace_logging_enabled() -> bool {
    matches!(trace_mode(), TraceMode::Verbose)
}

fn trace_checkpointing_enabled() -> bool {
    CHECKPOINTING_ENABLED.with(Cell::get)
}

fn should_commit_checkpoint(checkpoint_name: &str) -> bool {
    !checkpoint_name.starts_with("prefill.layer_token.")
}

fn mark_terminal_checkpoint(checkpoint_name: &str) -> bool {
    TERMINAL_CHECKPOINT.with(|terminal_checkpoint| {
        let mut terminal_checkpoint = terminal_checkpoint.borrow_mut();
        let Some(state) = terminal_checkpoint.as_mut() else {
            return false;
        };
        if state.reached || state.spec.checkpoint_id != checkpoint_name {
            return state.reached;
        }

        state.seen += 1;
        if state.seen == state.spec.occurrence {
            state.reached = true;
        }
        state.reached
    })
}

fn emit(kind: &str, label: &str, duration: Option<Duration>) {
    let elapsed = process_start().elapsed().as_secs_f64();
    match duration {
        Some(duration) => {
            eprintln!(
                "[raster-trace +{elapsed:>8.3}s] {kind:<5} {label} ({:.3}s)",
                duration.as_secs_f64()
            );
        }
        None => {
            eprintln!("[raster-trace +{elapsed:>8.3}s] {kind:<5} {label}");
        }
    }
}

fn emit_phase(kind: &str, phase_id: PhaseId) {
    let elapsed = process_start().elapsed().as_secs_f64();
    eprintln!(
        "[raster-phase +{elapsed:>8.3}s] {kind:<5} {}",
        phase_id.as_str()
    );
}

fn emit_checkpoint(checkpoint_name: &str) {
    let elapsed = process_start().elapsed().as_secs_f64();
    eprintln!("[raster-checkpoint +{elapsed:>8.3}s] hit   {checkpoint_name}");
}

fn emit_checkpoint_bundle(payload: &Value, saved_path: Option<&std::path::Path>) {
    let elapsed = process_start().elapsed().as_secs_f64();
    match serde_json::to_string_pretty(payload) {
        Ok(serialized) => {
            if let Some(saved_path) = saved_path {
                eprintln!(
                    "[raster-trace +{elapsed:>8.3}s] checkpoints saved={}\n{serialized}",
                    saved_path.display()
                );
            } else {
                eprintln!("[raster-trace +{elapsed:>8.3}s] checkpoints\n{serialized}");
            }
        }
        Err(error) => {
            eprintln!(
                "[raster-trace +{elapsed:>8.3}s] checkpoints <serialization failed: {error}>"
            );
        }
    }
}

fn write_checkpoint_bundle(payload: &Value) -> anyhow::Result<PathBuf> {
    let trace_dir = trace_output_directory();
    fs::create_dir_all(&trace_dir)?;
    let trace_path = trace_dir.join(format!(
        "trace-{}-{}.json",
        process_id(),
        unix_timestamp_ms()?
    ));
    fs::write(&trace_path, serde_json::to_vec_pretty(payload)?)?;
    Ok(trace_path)
}

fn trace_output_directory() -> PathBuf {
    env::var("RASTER_TRACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("raster-traces"))
}

fn unix_timestamp_ms() -> anyhow::Result<u128> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis())
}

fn process_id() -> u32 {
    std::process::id()
}

fn trace_collector() -> &'static Mutex<TraceCollector> {
    static COLLECTOR: OnceLock<Mutex<TraceCollector>> = OnceLock::new();
    COLLECTOR.get_or_init(|| Mutex::new(TraceCollector::default()))
}

fn process_start() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
}

struct ResetCheckpointingFlag(bool);

impl Drop for ResetCheckpointingFlag {
    fn drop(&mut self) {
        CHECKPOINTING_ENABLED.with(|checkpointing_enabled| checkpointing_enabled.set(self.0));
    }
}

struct ResetTerminalCheckpoint(Option<TerminalCheckpointState>);

impl Drop for ResetTerminalCheckpoint {
    fn drop(&mut self) {
        TERMINAL_CHECKPOINT.with(|terminal_checkpoint| {
            terminal_checkpoint.replace(self.0.take());
        });
    }
}

thread_local! {
    static CHECKPOINTING_ENABLED: Cell<bool> = const { Cell::new(false) };
    static TERMINAL_CHECKPOINT: RefCell<Option<TerminalCheckpointState>> = const { RefCell::new(None) };
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{should_commit_checkpoint, TerminalCheckpointSpec};

    #[test]
    fn checkpoint_commitments_skip_prefill_layer_token_entries() {
        assert!(!should_commit_checkpoint(
            "prefill.layer_token.layer_0.token_0"
        ));
        assert!(should_commit_checkpoint("prefill.layer"));
        assert!(should_commit_checkpoint(
            "decode.layer_token.layer_0.position_0"
        ));
    }

    #[test]
    fn terminal_checkpoint_spec_defaults_to_first_occurrence() {
        let spec =
            TerminalCheckpointSpec::parse("prefill.finalize").expect("checkpoint should parse");

        assert_eq!(spec.checkpoint_id(), "prefill.finalize");
        assert_eq!(spec.occurrence(), 1);
    }

    #[test]
    fn terminal_checkpoint_spec_accepts_occurrence_suffix() {
        let spec =
            TerminalCheckpointSpec::parse("prefill.layer:2").expect("checkpoint should parse");

        assert_eq!(spec.checkpoint_id(), "prefill.layer");
        assert_eq!(spec.occurrence(), 2);
    }

    #[test]
    fn terminal_checkpoint_spec_rejects_zero_occurrence() {
        let error = TerminalCheckpointSpec::parse("prefill.layer:0").expect_err("zero should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn terminal_checkpoint_tracking_reaches_requested_occurrence() {
        let spec =
            TerminalCheckpointSpec::parse("prefill.layer:2").expect("checkpoint should parse");

        super::with_terminal_checkpoint(Some(spec), || {
            assert!(!super::trace_checkpoint(
                "prefill.layer",
                &json!({ "index": 0 })
            ));
            assert_eq!(super::reached_terminal_checkpoint_id(), None);
            assert!(super::trace_checkpoint(
                "prefill.layer",
                &json!({ "index": 1 })
            ));
            assert_eq!(
                super::reached_terminal_checkpoint_id().as_deref(),
                Some("prefill.layer")
            );
        });
    }
}
