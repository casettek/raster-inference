//! Shared trace-capture harness for the probe and guard tests (the
//! `raster` repo's own `recur_draft.rs` capture pattern).
//!
//! The runtime accepts exactly one global publisher per process, so every
//! trace-asserting test in this crate must share this module's capturing
//! publisher; capture is scoped per test by a lock, an active flag, and the
//! capturing thread id.
//!
//! Test-only module: never program surface.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};
use std::thread::ThreadId;
use std::vec::Vec;

use raster::core::trace::{FnInputValue, TraceEvent};
use raster::prelude::FnCallRecord;
use raster_runtime::Publisher;

static TRACE_CAPTURE_LOCK: Mutex<()> = Mutex::new(());
static TRACE_INIT: Once = Once::new();
static TRACE_EVENTS: Mutex<Vec<TraceEvent>> = Mutex::new(Vec::new());
static TRACE_CAPTURE_ACTIVE: AtomicBool = AtomicBool::new(false);
static TRACE_CAPTURE_THREAD: Mutex<Option<ThreadId>> = Mutex::new(None);

struct CapturePublisher;

impl Publisher for CapturePublisher {
    fn publish(&self, event: TraceEvent) {
        let current_thread = std::thread::current().id();
        let capture_thread = TRACE_CAPTURE_THREAD.lock().unwrap().to_owned();
        if TRACE_CAPTURE_ACTIVE.load(Ordering::SeqCst) && capture_thread == Some(current_thread) {
            TRACE_EVENTS.lock().unwrap().push(event);
        }
    }

    fn finish(&self) {}
}

/// Runs `f` with trace capture on the current thread and returns its result
/// alongside every trace event it published.
pub fn capture_trace_events<F, T>(f: F) -> (T, Vec<TraceEvent>)
where
    F: FnOnce() -> T,
{
    let _guard = TRACE_CAPTURE_LOCK.lock().unwrap();
    TRACE_INIT.call_once(|| raster::init_with(CapturePublisher));
    TRACE_EVENTS.lock().unwrap().clear();
    *TRACE_CAPTURE_THREAD.lock().unwrap() = Some(std::thread::current().id());
    TRACE_CAPTURE_ACTIVE.store(true, Ordering::SeqCst);

    let result = f();
    let events = TRACE_EVENTS.lock().unwrap().clone();
    TRACE_CAPTURE_ACTIVE.store(false, Ordering::SeqCst);
    *TRACE_CAPTURE_THREAD.lock().unwrap() = None;
    (result, events)
}

/// Per-iteration `RecurSequenceStart` records for one recur sequence.
pub fn sequence_start_records(events: &[TraceEvent], fn_name: &str) -> Vec<FnCallRecord> {
    events
        .iter()
        .filter_map(|event| match event {
            TraceEvent::RecurSequenceStart(record) if record.fn_name == fn_name => {
                Some(record.clone())
            }
            _ => None,
        })
        .collect()
}

/// `TileExec` records for one plain tile.
pub fn tile_exec_records(events: &[TraceEvent], fn_name: &str) -> Vec<FnCallRecord> {
    events
        .iter()
        .filter_map(|event| match event {
            TraceEvent::TileExec(record) if record.fn_name == fn_name => Some(record.clone()),
            _ => None,
        })
        .collect()
}

/// `RecurTileIterationExec` records for one recur tile.
pub fn recur_tile_iteration_records(events: &[TraceEvent], fn_name: &str) -> Vec<FnCallRecord> {
    events
        .iter()
        .filter_map(|event| match event {
            TraceEvent::RecurTileIterationExec(record) if record.fn_name == fn_name => {
                Some(record.clone())
            }
            _ => None,
        })
        .collect()
}

/// Unwraps an inline trace value's bytes; panics on bindings.
pub fn inline_bytes(value: &FnInputValue) -> Vec<u8> {
    match value {
        FnInputValue::Inline(bytes) => bytes.clone(),
        other => panic!("expected inline trace value, found {other:?}"),
    }
}
