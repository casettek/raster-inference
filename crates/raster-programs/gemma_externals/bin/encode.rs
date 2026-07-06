//! Offline encoder bin for the Gemma tokenizer committed external.
//!
//! Usage:
//!   cargo run -p raster-program-gemma-externals --features encode \
//!     --bin encode -- <TOKENIZER_JSON> <CACHE_ROOT>
//!
//! Encodes into the content-addressed cache
//! (`<CACHE_ROOT>/gemma-tokenizer/<sha256(tokenizer.json)>/`) and prints
//! machine-parseable lines:
//!
//!   data_path: <path>
//!   index_path: <path>
//!   root_commitment: <hex>

use std::path::PathBuf;

use raster_program_gemma_externals::encode::encode_tokenizer_to_cache;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [tokenizer_json, cache_root] = args.as_slice() else {
        eprintln!("usage: encode <TOKENIZER_JSON> <CACHE_ROOT>");
        std::process::exit(2);
    };

    let encoded =
        encode_tokenizer_to_cache(&PathBuf::from(tokenizer_json), &PathBuf::from(cache_root))
            .expect("encode gemma tokenizer external");

    println!("data_path: {}", encoded.data_path.display());
    println!("index_path: {}", encoded.index_path.display());
    println!("root_commitment: {}", encoded.root_commitment);
}
