//! Offline encoder for the round-trip fixture's raster-encoded external.
//!
//! Usage:
//!   cargo run -p raster-program-roundtrip --features encode --bin encode -- \
//!     <OUT_DIR> [scale] [label]
//!
//! Writes `config.rastered` + `config.rindex` into OUT_DIR and prints the
//! raster index root commitment on a stable, machine-parseable line
//! (`root_commitment: <hex>`), the value staged into `input_manifest.json`
//! by the host. Encoding is deterministic: the same config value yields the
//! same commitment (asserted by the round-trip test).

use std::path::PathBuf;

use raster_program_roundtrip::types::FixtureConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = PathBuf::from(args.first().cloned().unwrap_or_else(|| ".".to_string()));
    let scale: u64 = args
        .get(1)
        .map(|raw| raw.parse().expect("scale must be a u64"))
        .unwrap_or(3);
    let label = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "ws2-roundtrip".to_string());

    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let data_path = out_dir.join("config.rastered");
    let index_path = out_dir.join("config.rindex");

    let config = FixtureConfig { label, scale };
    let commitment = raster::write_raster_files(&config, &data_path, &index_path)
        .expect("raster-encode fixture config");

    println!("wrote: {}", data_path.display());
    println!("wrote: {}", index_path.display());
    println!("root_commitment: {commitment}");
}
