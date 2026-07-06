//! Shared probe types (also used by the staging bin).

use alloc::string::String;
use alloc::vec::Vec;
use raster::Selectable;
use serde::{Deserialize, Serialize};

/// P1 state-only recur state (catalog C7).
#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct MaxState {
    pub max: u64,
}

/// P1 break-early recur state (catalog C7, `RecurControl::Break`).
#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct SumState {
    pub sum: u64,
    pub seen: u64,
}

/// P1 "until-done within a bound" recur state (gap G1 restructuring).
#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct ConvergeState {
    pub value: u64,
    pub steps: u64,
}

/// P1/P4 draft schema (catalog C16: sim builder -> draft append).
#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct Bundle {
    pub title: String,
    pub items: Vec<String>,
}

/// P3 committed external input (catalog C13/C22).
#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct ProbeConfig {
    pub label: String,
    pub thresholds: Vec<u32>,
    pub nested: NestedCfg,
}

#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct NestedCfg {
    pub scale: u32,
}

/// P2 enum-carrying ABI type (catalog C28).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProbeMode {
    Fast,
    Careful { retries: u32 },
}
