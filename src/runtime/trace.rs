use std::{
    cell::Cell,
    cell::RefCell,
    collections::HashMap,
    env, fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::checkpoints::{PhaseId, RoutineId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TraceMode {
    Off,
    Verbose,
}

pub struct TraceSpan;

pub struct RoutineSpan {
    label: String,
    start: Instant,
    enabled: bool,
}

pub struct NativeSpan;

impl TraceSpan {
    pub fn new(_label: impl Into<String>) -> Self {
        Self
    }
}

impl RoutineSpan {
    pub fn new(routine_id: RoutineId, details: impl Into<String>) -> Self {
        let enabled = trace_logging_enabled();
        let label = next_routine_label(routine_id);
        let details = details.into();
        if enabled {
            ACTIVE_ROUTINES
                .with(|active_routines| active_routines.borrow_mut().push(label.clone()));
            emit_routine("start", &label, &details, None);
        }
        Self {
            label,
            start: Instant::now(),
            enabled,
        }
    }
}

impl Drop for RoutineSpan {
    fn drop(&mut self) {
        if self.enabled {
            emit_routine("complete", &self.label, "", Some(self.start.elapsed()));
            ACTIVE_ROUTINES.with(|active_routines| {
                let mut active_routines = active_routines.borrow_mut();
                match active_routines
                    .iter()
                    .rposition(|label| label == &self.label)
                {
                    Some(index) => {
                        active_routines.remove(index);
                    }
                    None => {}
                }
            });
        }
    }
}

impl NativeSpan {
    pub fn new(_label: impl Into<String>) -> Self {
        Self
    }
}

pub fn trace_scope(label: impl Into<String>) -> TraceSpan {
    TraceSpan::new(label)
}

pub fn routine_scope(routine_id: RoutineId, details: impl Into<String>) -> RoutineSpan {
    RoutineSpan::new(routine_id, details)
}

pub fn native_scope(label: impl Into<String>) -> NativeSpan {
    NativeSpan::new(label)
}

pub fn trace_event(_label: impl AsRef<str>) {}

pub fn trace_native(_label: impl AsRef<str>) {}

pub fn tile_invoked(invocation_kind: &str, name: &str, ordinal: u64) {
    if !trace_logging_enabled() {
        return;
    }

    let tile_label = if invocation_kind == "tile" {
        name.to_string()
    } else {
        format!("{invocation_kind} {name}")
    };
    let routine = active_routine_label();
    let label = match routine {
        Some(routine) => {
            format!("{tile_label} count={ordinal} routine={routine}")
        }
        None => format!("{tile_label} count={ordinal}"),
    };
    emit("raster-tile", &label, None);
}

pub fn with_trace_logging_enabled<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    TRACE_LOGGING_OVERRIDE.with(|trace_logging_override| {
        let previous = trace_logging_override.replace(Some(enabled));
        let previous_routine_counts = ROUTINE_OCCURRENCES
            .with(|routine_occurrences| routine_occurrences.replace(HashMap::new()));
        let previous_active_routines =
            ACTIVE_ROUTINES.with(|active_routines| active_routines.replace(Vec::new()));
        let reset = ResetTraceLoggingOverride(previous);
        let reset_routine_context = ResetRoutineTraceContext {
            routine_occurrences: previous_routine_counts,
            active_routines: previous_active_routines,
        };
        let result = f();
        drop(reset_routine_context);
        drop(reset);
        result
    })
}

pub fn phase_started(_phase_id: PhaseId) {}

pub fn phase_finished(_phase_id: PhaseId) {}

pub fn phase_paused(_phase_id: PhaseId) {}

pub fn raster_tile_invocations_finished(total: u64) {
    emit("tiles", &format!("total {total}"), None);
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
    #[cfg(test)]
    completed_checkpoints: Option<Value>,
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
    clear_completed_checkpoints_for_tests(&mut collector);
    collector.completed_trace_path = None;
}

pub fn trace_checkpoint<T: Serialize>(checkpoint_name: &str, state: &T) -> bool {
    trace_checkpoint_lazy(checkpoint_name, || state)
}

pub fn trace_checkpoint_lazy<T: Serialize>(
    checkpoint_name: &str,
    build_state: impl FnOnce() -> T,
) -> bool {
    let should_commit = should_commit_checkpoint(checkpoint_name);
    let observed_by_terminal_checkpoint = terminal_checkpoint_observes(checkpoint_name);
    let reached_terminal_checkpoint = mark_terminal_checkpoint(checkpoint_name);
    if should_commit || trace_logging_enabled() || observed_by_terminal_checkpoint {
        emit_checkpoint(checkpoint_name);
    }
    if !trace_checkpointing_enabled() || !should_commit {
        return reached_terminal_checkpoint;
    }

    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    let state = build_state();
    collector.checkpoints.push(json!({
        checkpoint_name: sha256_hex(&state),
    }));
    reached_terminal_checkpoint
}

pub fn trace_checkpoint_lazy_result<T: Serialize>(
    checkpoint_name: &str,
    build_state: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<bool> {
    let should_commit = should_commit_checkpoint(checkpoint_name);
    let observed_by_terminal_checkpoint = terminal_checkpoint_observes(checkpoint_name);
    let reached_terminal_checkpoint = mark_terminal_checkpoint(checkpoint_name);
    if should_commit || trace_logging_enabled() || observed_by_terminal_checkpoint {
        emit_checkpoint(checkpoint_name);
    }
    if !trace_checkpointing_enabled() || !should_commit {
        return Ok(reached_terminal_checkpoint);
    }

    let state = build_state()?;
    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector.checkpoints.push(json!({
        checkpoint_name: sha256_hex(&state),
    }));
    Ok(reached_terminal_checkpoint)
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
    store_completed_checkpoints_for_tests(&mut collector, &payload);
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
    store_completed_checkpoints_for_tests(&mut collector, &payload);
    collector.completed_trace_path = write_checkpoint_bundle(&payload).ok();
    emit_checkpoint_bundle(&payload, collector.completed_trace_path.as_deref());
}

#[cfg(test)]
pub(crate) fn checkpoint_payload_for_tests() -> Value {
    let checkpoints = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned")
        .checkpoints
        .clone();
    Value::Array(checkpoints)
}

#[cfg(test)]
pub(crate) fn test_trace_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
pub(crate) fn take_completed_checkpoint_payload_for_tests() -> Value {
    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector
        .completed_checkpoints
        .take()
        .unwrap_or_else(|| Value::Array(Vec::new()))
}

#[cfg(test)]
fn clear_completed_checkpoints_for_tests(collector: &mut TraceCollector) {
    collector.completed_checkpoints = None;
}

#[cfg(not(test))]
fn clear_completed_checkpoints_for_tests(_collector: &mut TraceCollector) {}

#[cfg(test)]
fn store_completed_checkpoints_for_tests(collector: &mut TraceCollector, payload: &Value) {
    collector.completed_checkpoints = Some(payload.clone());
}

#[cfg(not(test))]
fn store_completed_checkpoints_for_tests(_collector: &mut TraceCollector, _payload: &Value) {}

pub fn serialize_layer_caches(
    layer_caches: &[crate::shared::model::transformer::LayerKvCache],
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
    if let Some(enabled) = TRACE_LOGGING_OVERRIDE.with(Cell::get) {
        return if enabled {
            TraceMode::Verbose
        } else {
            TraceMode::Off
        };
    }
    TraceMode::Off
}

fn trace_logging_enabled() -> bool {
    matches!(trace_mode(), TraceMode::Verbose)
}

fn trace_checkpointing_enabled() -> bool {
    CHECKPOINTING_ENABLED.with(Cell::get)
}

fn should_commit_checkpoint(checkpoint_name: &str) -> bool {
    !checkpoint_name.starts_with("prefill.layer_token.")
        && !checkpoint_name.starts_with("decode.layer_token.")
}

fn terminal_checkpoint_observes(checkpoint_name: &str) -> bool {
    TERMINAL_CHECKPOINT.with(|terminal_checkpoint| {
        terminal_checkpoint
            .borrow()
            .as_ref()
            .is_some_and(|state| state.spec.checkpoint_id == checkpoint_name)
    })
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

fn next_routine_label(routine_id: RoutineId) -> String {
    ROUTINE_OCCURRENCES.with(|routine_occurrences| {
        let mut routine_occurrences = routine_occurrences.borrow_mut();
        let occurrence = routine_occurrences
            .entry(routine_id.as_str())
            .and_modify(|occurrence| *occurrence += 1)
            .or_insert(1);
        format_occurrence_label(routine_id.as_str(), *occurrence)
    })
}

fn format_occurrence_label(label: &str, occurrence: usize) -> String {
    if occurrence == 1 {
        label.to_string()
    } else {
        format!("{label}:{occurrence}")
    }
}

fn active_routine_label() -> Option<String> {
    ACTIVE_ROUTINES.with(|active_routines| active_routines.borrow().last().cloned())
}

fn emit(kind: &str, label: &str, duration: Option<Duration>) {
    let elapsed = process_start().elapsed().as_secs_f64();
    match duration {
        Some(duration) => {
            eprintln!(
                "[{kind:<16} {elapsed:>8.3}s] {label} ({:.3}s)",
                duration.as_secs_f64()
            );
        }
        None => {
            eprintln!("[{kind:<16} {elapsed:>8.3}s] {label}");
        }
    }
}

fn emit_routine(kind: &str, label: &str, details: &str, duration: Option<Duration>) {
    let label = if details.is_empty() {
        label.to_string()
    } else {
        format!("{label} {details}")
    };
    emit(&format!("routine-{kind}"), &label, duration);
}

fn emit_checkpoint(checkpoint_name: &str) {
    emit("checkpoint", checkpoint_name, None);
}

fn emit_checkpoint_bundle(payload: &Value, saved_path: Option<&std::path::Path>) {
    let elapsed = process_start().elapsed().as_secs_f64();
    match serde_json::to_string_pretty(payload) {
        Ok(serialized) => {
            if let Some(saved_path) = saved_path {
                eprintln!(
                    "[{:<16} {elapsed:>8.3}s] saved={}\n{serialized}",
                    "checkpoints",
                    saved_path.display()
                );
            } else {
                eprintln!("[{:<16} {elapsed:>8.3}s]\n{serialized}", "checkpoints");
            }
        }
        Err(error) => {
            eprintln!(
                "[{:<16} {elapsed:>8.3}s] <serialization failed: {error}>",
                "checkpoints"
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

struct ResetTraceLoggingOverride(Option<bool>);

impl Drop for ResetTraceLoggingOverride {
    fn drop(&mut self) {
        TRACE_LOGGING_OVERRIDE.with(|trace_logging_override| {
            trace_logging_override.set(self.0.take());
        });
    }
}

struct ResetRoutineTraceContext {
    routine_occurrences: HashMap<&'static str, usize>,
    active_routines: Vec<String>,
}

impl Drop for ResetRoutineTraceContext {
    fn drop(&mut self) {
        ROUTINE_OCCURRENCES.with(|routine_occurrences| {
            routine_occurrences.replace(std::mem::take(&mut self.routine_occurrences));
        });
        ACTIVE_ROUTINES.with(|active_routines| {
            active_routines.replace(std::mem::take(&mut self.active_routines));
        });
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
    static TRACE_LOGGING_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static TERMINAL_CHECKPOINT: RefCell<Option<TerminalCheckpointState>> = const { RefCell::new(None) };
    static ROUTINE_OCCURRENCES: RefCell<HashMap<&'static str, usize>> = RefCell::new(HashMap::new());
    static ACTIVE_ROUTINES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{should_commit_checkpoint, TerminalCheckpointSpec};
    use crate::runtime::checkpoints::RoutineId;

    #[test]
    fn checkpoint_commitments_skip_layer_token_entries() {
        assert!(!should_commit_checkpoint(
            "prefill.layer_token.layer_0.token_0"
        ));
        assert!(should_commit_checkpoint("prefill.layer"));
        assert!(!should_commit_checkpoint(
            "decode.layer_token.layer_0.position_0"
        ));
        assert!(should_commit_checkpoint("decode.transition"));
    }

    #[test]
    fn lazy_checkpoint_does_not_build_state_when_checkpointing_disabled() {
        let mut built = false;

        super::with_checkpointing_enabled(false, || {
            let reached = super::trace_checkpoint_lazy_result("prefill.prepare_aux", || {
                built = true;
                Ok(json!({ "materialized": true }))
            })
            .expect("lazy checkpoint should succeed");

            assert!(!reached);
        });

        assert!(!built);
    }

    #[test]
    fn lazy_checkpoint_builds_state_when_checkpointing_enabled() {
        let _trace_guard = super::test_trace_lock()
            .lock()
            .expect("trace test lock should not be poisoned");
        let mut built = false;

        super::with_checkpointing_enabled(true, || {
            let reached = super::trace_checkpoint_lazy_result("prefill.prepare_aux", || {
                built = true;
                Ok(json!({ "materialized": true }))
            })
            .expect("lazy checkpoint should succeed");

            assert!(!reached);
        });

        assert!(built);
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

    #[test]
    fn routine_occurrence_labels_match_terminal_checkpoint_suffixes() {
        super::with_trace_logging_enabled(true, || {
            assert_eq!(
                super::next_routine_label(RoutineId::PrefillLayer),
                "prefill.layer"
            );
            assert_eq!(
                super::next_routine_label(RoutineId::PrefillLayer),
                "prefill.layer:2"
            );
            assert_eq!(
                super::next_routine_label(RoutineId::PrefillLayer),
                "prefill.layer:3"
            );
        });
    }

    #[test]
    fn trace_logging_defaults_to_cli_override_only() {
        assert_eq!(super::trace_mode(), super::TraceMode::Off);
        super::with_trace_logging_enabled(true, || {
            assert_eq!(super::trace_mode(), super::TraceMode::Verbose);
        });
        assert_eq!(super::trace_mode(), super::TraceMode::Off);
    }
}
