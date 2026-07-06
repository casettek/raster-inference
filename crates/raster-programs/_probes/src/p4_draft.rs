//! P4 — internal storage / draft round-trip tiles (catalog C15–C17; grounds H1).

use alloc::string::String;
use raster::prelude::*;

use crate::types::{Bundle, BundleDraftExt};

/// Set-once draft field (sim builder metadata analogue).
#[tile]
pub fn set_bundle_title(output: Draft<Bundle>, title: String) -> Draft<Bundle> {
    let mut output = output;
    output.title().set(title);
    output
}

/// Append-only draft field (sim `append_leaf` analogue; order is index).
#[tile]
pub fn push_bundle_item(output: Draft<Bundle>, item: String) -> Draft<Bundle> {
    let mut output = output;
    output.items().push(item);
    output
}

#[tile]
pub fn item_len(item: String) -> u64 {
    item.len() as u64
}
