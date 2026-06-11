//! Executor seam: per-routine native vs raster execution beneath the
//! phase-sequencing skeleton (`runtime::sequence`).
//!
//! The two things the protocol compares — native and raster execution of the
//! same routine — live in sibling modules behind this seam so neither path is
//! constructed by interleaved flag plumbing:
//!
//! - [`native`]: the native deterministic/fp32 executor, including the
//!   selective raster detour hooks (the detour is the native executor
//!   swapping in exactly one raster routine occurrence, governed by
//!   `RasterDetourController`).
//! - [`raster`]: the full root-backed raster tile executor.

pub(crate) mod native;
pub(crate) mod raster;

use anyhow::Result;

use crate::runtime::checkpoints::{RasterDetourController, RasterDetourSpec, RoutineId};
use crate::runtime::inference::InferenceControls;

/// Which executor a routine step is dispatched to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepMode {
    Native,
    Raster,
}

/// Per-(routine, occurrence) executor selection.
///
/// Three policies exist: full-native, full-raster, and single-detour (native
/// everywhere except one selected routine occurrence). The detour policy is
/// sourced from [`RasterDetourController`] — the same occurrence-counting
/// semantics as `--raster-at` — rather than reimplemented; full-raster and
/// full-native short-circuit without consulting the controller, which is
/// inactive for those policies (`--raster` and `--raster-at` are mutually
/// exclusive).
pub(crate) struct ExecutionPolicy {
    raster: bool,
    controller: RasterDetourController,
}

impl ExecutionPolicy {
    pub fn from_controls(controls: &InferenceControls) -> Self {
        Self {
            raster: controls.raster,
            controller: RasterDetourController::new(controls.raster_detour),
        }
    }

    pub fn is_full_raster(&self) -> bool {
        self.raster
    }

    pub fn is_detour_active(&self) -> bool {
        self.controller.is_active()
    }

    pub fn selected_detour_spec(&self) -> Option<RasterDetourSpec> {
        self.controller.selected_spec()
    }

    /// Decides the executor for the next occurrence of `routine`. Counts
    /// detour occurrences, so it must be called exactly once per routine
    /// occurrence on the native path (order-sensitive).
    pub fn mode_for(&mut self, routine: RoutineId) -> StepMode {
        if self.raster {
            return StepMode::Raster;
        }
        if self.controller.should_detour(routine) {
            StepMode::Raster
        } else {
            StepMode::Native
        }
    }

    /// Counts an occurrence of `routine` and errors if the detour selected
    /// it (used for routines whose detour is not implemented on the current
    /// path).
    pub fn reject_if_selected_unsupported(&mut self, routine: RoutineId) -> Result<()> {
        self.controller.reject_if_selected_unsupported(routine)
    }

    pub fn ensure_matched_if_active(&self) -> Result<()> {
        self.controller.ensure_matched_if_active()
    }

    /// The underlying detour controller, for routines that perform their own
    /// per-chunk detour decisions (`prefill.range`, the decode loop).
    pub fn detour_controller_mut(&mut self) -> &mut RasterDetourController {
        &mut self.controller
    }
}
