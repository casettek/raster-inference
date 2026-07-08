//! Shared trace-capture harness for decode-select program tests.

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

pub fn event_record(event: &TraceEvent) -> Option<FnCallRecord> {
    match event {
        TraceEvent::SequenceStart(record)
        | TraceEvent::SequenceEnd(record)
        | TraceEvent::RecurSequenceStart(record)
        | TraceEvent::RecurSequenceEnd(record)
        | TraceEvent::TileExec(record)
        | TraceEvent::RecurTileIterationExec(record)
        | TraceEvent::RecurTileExec(record)
        | TraceEvent::RecurSequenceExec(record) => Some(record.clone()),
    }
}

pub fn inline_bytes(value: &FnInputValue) -> Option<&[u8]> {
    match value {
        FnInputValue::Inline(bytes) => Some(bytes),
        _ => None,
    }
}
