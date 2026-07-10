//! Encoder for `input.embedding` committed raster inputs.
//!
//! Usage:
//!   cargo run -p raster-program-input-embedding --features encode --bin encode -- \
//!     <SOURCE_JSON> <OUT_DIR>
//!
//! The source JSON may contain either or both inputs. Each present input is
//! written as `.rastered` + `.rindex` and reported with machine-readable
//! paths/root commitments for host staging.

use std::error::Error;
use std::path::{Path, PathBuf};

use raster_program_input_embedding::types::InputEmbeddingEncodedInputs;
use serde::Serialize;

fn main() {
    if let Err(error) = run() {
        eprintln!("input_embedding encoder failed: {error}");
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

    // Stream-parse: the embedding source JSON is gigabytes for a real
    // model, so avoid holding the raw bytes alongside the parsed table.
    let source_file = std::fs::File::open(&source_path)?;
    let inputs: InputEmbeddingEncodedInputs =
        serde_json::from_reader(std::io::BufReader::new(source_file))?;

    if let Some(prompt_token_ids) = inputs.prompt_token_ids.as_ref() {
        encode_input("prompt_token_ids", prompt_token_ids, &out_dir)?;
    }
    if let Some(embedding) = inputs.embedding.as_ref() {
        encode_input("embedding", embedding, &out_dir)?;
    }

    Ok(())
}

fn encode_input<T: Serialize>(name: &str, value: &T, out_dir: &Path) -> Result<(), Box<dyn Error>> {
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
