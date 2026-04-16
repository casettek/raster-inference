use std::{
    env,
    sync::OnceLock,
    time::{Duration, Instant},
};

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

fn process_start() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
}
