//! Run-local encoder for `decode.select_token` hot request inputs.
//!
//! Usage:
//!   cargo run -p raster-program-decode-select-token --features encode --bin encode -- \
//!     <SOURCE_JSON> <OUT_DIR>
//!
//! Writes `logits`, `full_token_ids`, and `generated_token_ids` as
//! `.rastered` + `.rindex` pairs and prints machine-readable paths and root
//! commitments for host staging.

use std::error::Error;
use std::path::{Path, PathBuf};

use raster_program_decode_select_token::types::{
    DecodeSelectEncodedInputs, DecodeSelectLogits, DecodeSelectTokenIds,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("decode_select_token encoder failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let source_path = args
        .first()
        .map(PathBuf::from)
        .ok_or("usage: encode <SOURCE_JSON> <OUT_DIR>")?;
    let out_dir = args
        .get(1)
        .map(PathBuf::from)
        .ok_or("usage: encode <SOURCE_JSON> <OUT_DIR>")?;
    std::fs::create_dir_all(&out_dir)?;

    let source_bytes = std::fs::read(&source_path)?;
    let inputs: DecodeSelectEncodedInputs = serde_json::from_slice(&source_bytes)?;

    encode_logit_input("logits", &inputs.logits, &out_dir)?;
    encode_token_input("full_token_ids", &inputs.full_token_ids, &out_dir)?;
    encode_token_input("generated_token_ids", &inputs.generated_token_ids, &out_dir)?;

    Ok(())
}

fn encode_logit_input(
    name: &str,
    value: &DecodeSelectLogits,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let data_path = out_dir.join(format!("{name}.rastered"));
    let index_path = out_dir.join(format!("{name}.rindex"));
    let root_commitment = raster::write_raster_files(value, &data_path, &index_path)
        .map_err(|error| format!("failed to raster-encode {name}: {error}"))?;
    print_entry(name, &data_path, &index_path, &root_commitment);
    Ok(())
}

fn encode_token_input(
    name: &str,
    value: &DecodeSelectTokenIds,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let data_path = out_dir.join(format!("{name}.rastered"));
    let index_path = out_dir.join(format!("{name}.rindex"));
    let root_commitment = raster::write_raster_files(value, &data_path, &index_path)
        .map_err(|error| format!("failed to raster-encode {name}: {error}"))?;
    print_entry(name, &data_path, &index_path, &root_commitment);
    Ok(())
}

fn print_entry(name: &str, data_path: &Path, index_path: &Path, root_commitment: &str) {
    println!("{name}_data_path: {}", data_path.display());
    println!("{name}_index_path: {}", index_path.display());
    println!("{name}_root_commitment: {root_commitment}");
}
