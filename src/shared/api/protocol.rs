//! Protocol constants.
//!
//! Values fixed by the Raster fraud-proof protocol. They are not
//! configurable: every claimer and challenger must use identical values or
//! committed checkpoints will not reproduce.

/// The protocol's fixed sampling temperature. Inference is deterministic;
/// temperature is a static protocol parameter, not a tuning knob.
pub const SAMPLING_TEMPERATURE: f32 = 1.0;
