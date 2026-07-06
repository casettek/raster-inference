//! Stages the probe program's committed inputs.
//!
//! Writes `input.json`, `input_manifest.json`, and the postcard payload files
//! into the given directory (default: current directory, expected to be the
//! probe crate root so `cargo raster run` finds them).
//!
//! Usage:
//!   cargo run --features stage --bin stage -- [OUT_DIR] [--tamper] [--p2-div-zero]
//!
//! `--tamper` flips one byte of `p3_config.bin` *after* the manifest is
//! written, so the run must fail commitment verification (probe P3, run R2).
//! `--p2-div-zero` stages `p2_divisor` as 0 so `checked_div` on the ok path
//! errors terminally (probe P2 propagated-Err variant, run R3).

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use raster::core::postcard;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use raster_ws1_probes::types::{NestedCfg, ProbeConfig};

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

struct Staged {
    input: serde_json::Map<String, serde_json::Value>,
    manifest: serde_json::Map<String, serde_json::Value>,
    out_dir: PathBuf,
}

impl Staged {
    fn add<T: Serialize>(
        &mut self,
        name: &str,
        file: &str,
        value: &T,
    ) -> Result<(), Box<dyn Error>> {
        let bytes = postcard::to_allocvec(value)?;
        fs::write(self.out_dir.join(file), &bytes)?;
        self.input.insert(
            name.to_string(),
            json!({ "path": file, "load_preference": "read" }),
        );
        self.manifest.insert(
            name.to_string(),
            json!({
                "type": "sha256",
                "encoding": "postcard",
                "commitment": sha256_hex(&bytes)
            }),
        );
        Ok(())
    }
}

fn flip_last_byte(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut bytes = fs::read(path)?;
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flags: BTreeMap<&str, bool> = [
        ("--tamper", args.iter().any(|a| a == "--tamper")),
        ("--p2-div-zero", args.iter().any(|a| a == "--p2-div-zero")),
    ]
    .into_iter()
    .collect();
    let out_dir = PathBuf::from(
        args.iter()
            .find(|a| !a.starts_with("--"))
            .cloned()
            .unwrap_or_else(|| ".".to_string()),
    );
    fs::create_dir_all(&out_dir)?;

    let mut staged = Staged {
        input: serde_json::Map::new(),
        manifest: serde_json::Map::new(),
        out_dir: out_dir.clone(),
    };

    // P1 inputs.
    staged.add("p1_items", "p1_items.bin", &vec![3u64, 1, 4, 1, 5, 9, 2, 6])?;
    staged.add(
        "p1_lines",
        "p1_lines.bin",
        &vec!["one".to_string(), "two".to_string(), "three".to_string()],
    )?;
    // Dummy index list = iteration bound for the until-done probe.
    staged.add("p1_bound", "p1_bound.bin", &(0u64..8).collect::<Vec<_>>())?;

    // P2 input.
    let divisor: u64 = if flags["--p2-div-zero"] { 0 } else { 3 };
    staged.add("p2_divisor", "p2_divisor.bin", &divisor)?;

    // P3 input.
    let config = ProbeConfig {
        label: "cfg".to_string(),
        thresholds: vec![10, 20, 30],
        nested: NestedCfg { scale: 4 },
    };
    staged.add("p3_config", "p3_config.bin", &config)?;

    fs::write(
        out_dir.join("input.json"),
        serde_json::to_vec_pretty(&serde_json::Value::Object(staged.input))?,
    )?;
    fs::write(
        out_dir.join("input_manifest.json"),
        serde_json::to_vec_pretty(&serde_json::Value::Object(staged.manifest))?,
    )?;

    if flags["--tamper"] {
        flip_last_byte(&out_dir.join("p3_config.bin"))?;
        println!("tampered: flipped last byte of p3_config.bin after manifest write");
    }

    println!("staged probe inputs in {}", out_dir.display());
    Ok(())
}
