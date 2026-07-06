//! Shared fixture types (also used by the encoder bin and the main-crate
//! round-trip test).

use alloc::string::String;
use raster::Selectable;
use serde::{Deserialize, Serialize};

/// The raster-encoded committed external: model-scoped configuration the
/// encoder bin pre-encodes to `.rastered`/`.rindex` (mmap load preference).
#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct FixtureConfig {
    pub label: String,
    pub scale: u64,
}
