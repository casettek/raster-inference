use std::{
    env,
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
    run_metadata: Option<Value>,
    checkpoints: Vec<Value>,
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

    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector.run_metadata = Some(serialize_trace_value(run_metadata));
    collector.checkpoints.clear();
}

pub fn trace_checkpoint<T: Serialize>(phase: &str, state: &T) {
    if !tracing_enabled() {
        return;
    }

    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    collector.checkpoints.push(json!({
        "phase": phase,
        "state_commitment_sha256": sha256_hex(state),
    }));
}

pub fn finish_inference_trace<T: Serialize>(summary: &T) {
    if !tracing_enabled() {
        return;
    }

    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    let payload = json!({
        "run": collector.run_metadata.take(),
        "summary": serialize_trace_value(summary),
        "checkpoints": std::mem::take(&mut collector.checkpoints),
    });
    emit_checkpoint_bundle(&payload);
}

pub fn abort_inference_trace(error: &anyhow::Error) {
    if !tracing_enabled() {
        return;
    }

    let mut collector = trace_collector()
        .lock()
        .expect("trace collector mutex should not be poisoned");
    let payload = json!({
        "run": collector.run_metadata.take(),
        "error": error.to_string(),
        "checkpoints": std::mem::take(&mut collector.checkpoints),
    });
    emit_checkpoint_bundle(&payload);
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

fn emit_checkpoint_bundle(payload: &Value) {
    let elapsed = process_start().elapsed().as_secs_f64();
    match serde_json::to_string_pretty(payload) {
        Ok(serialized) => {
            eprintln!("[raster-trace +{elapsed:>8.3}s] checkpoints\n{serialized}");
        }
        Err(error) => {
            eprintln!("[raster-trace +{elapsed:>8.3}s] checkpoints <serialization failed: {error}>");
        }
    }
}

fn serialize_trace_value<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or_else(|error| json!({
        "serialization_error": error.to_string(),
    }))
}

fn trace_collector() -> &'static Mutex<TraceCollector> {
    static COLLECTOR: OnceLock<Mutex<TraceCollector>> = OnceLock::new();
    COLLECTOR.get_or_init(|| Mutex::new(TraceCollector::default()))
}

fn process_start() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
}
