use std::{
    env,
    fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub struct TraceSpan {
    label: String,
    start: Instant,
    enabled: bool,
}

impl TraceSpan {
    pub fn new(label: impl Into<String>) -> Self {
        let label = label.into();
        let enabled = tracing_enabled();
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
    if tracing_enabled() {
        emit("event", label.as_ref(), None);
    }
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
    if !tracing_enabled() {
        return;
    }

    let _ = run_metadata;
    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector.checkpoints.clear();
    collector.completed_trace_path = None;
}

pub fn trace_checkpoint<T: Serialize>(phase: &str, state: &T) {
    if !tracing_enabled() {
        return;
    }

    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector.checkpoints.push(json!({
        phase: sha256_hex(state),
    }));
}

pub fn finish_inference_trace<T: Serialize>(summary: &T) {
    if !tracing_enabled() {
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
    if !tracing_enabled() {
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
    layer_caches: &[crate::phase2::LayerKvCache],
) -> Vec<SerializableLayerKvCache> {
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

fn tracing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var("RASTER_TRACE_TILES")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
            .unwrap_or(false)
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
            eprintln!("[raster-trace +{elapsed:>8.3}s] checkpoints <serialization failed: {error}>");
        }
    }
}

fn write_checkpoint_bundle(payload: &Value) -> anyhow::Result<PathBuf> {
    let trace_dir = trace_output_directory();
    fs::create_dir_all(&trace_dir)?;
    let trace_path = trace_dir.join(format!("trace-{}-{}.json", process_id(), unix_timestamp_ms()?));
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
