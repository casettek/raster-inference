//! WS2 Gemma tokenizer committed external: encode → stage → run → ingest
//! against the hermetic tiny-gemma-dev tokenizer, plus the encoder
//! determinism and tamper legs.
//!
//! Requires the `cargo-raster` CLI on PATH (skips loudly otherwise; CI sets
//! `REQUIRE_CARGO_RASTER=1`). Re-run recipe: docs/plans/ws2-staging.md.

use std::path::{Path, PathBuf};
use std::process::Command;

use raster_inference::runtime::checkpoints::RoutineId;
use raster_inference::runtime::raster_core::ingest::{ingest, RasterCoreError};
use raster_inference::runtime::raster_core::staging::StagedInputs;
use raster_inference::runtime::raster_core::{CargoRasterRunner, RasterCoreRunDir};

/// Field-order mirror of `raster_program_gemma_externals::smoke::
/// TokenizerSmoke` (postcard layout contract; the main crate must not
/// depend on program crates that carry the real `raster` dependency).
#[derive(Debug, serde::Deserialize, PartialEq, Eq)]
struct TokenizerSmoke {
    unk_token: String,
    first_token: String,
    first_token_id: u32,
    special_token_count: u32,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn program_crate_dir() -> PathBuf {
    workspace_root().join("crates/raster-programs/gemma_externals")
}

fn tokenizer_json_path() -> PathBuf {
    workspace_root().join("assets/tiny-gemma-dev/tokenizer.json")
}

fn require_or_skip() -> bool {
    let available = Command::new("cargo-raster")
        .arg("--version")
        .output()
        .is_ok();
    if available {
        return true;
    }
    if std::env::var_os("REQUIRE_CARGO_RASTER").is_some_and(|v| v == "1") {
        panic!(
            "REQUIRE_CARGO_RASTER=1 but cargo-raster is not on PATH; install it from the \
             pinned raster checkout (cargo install --path ../raster/crates/raster-cli)"
        );
    }
    eprintln!(
        "SKIPPED: raster_core_gemma_tokenizer_external requires the cargo-raster CLI on PATH. \
         CI runs this test with REQUIRE_CARGO_RASTER=1."
    );
    false
}

struct EncodedEntry {
    data_path: PathBuf,
    index_path: PathBuf,
    root_commitment: String,
}

/// Runs the encoder bin against the given cache root, parsing its
/// machine-readable output lines.
fn encode_tokenizer(cache_root: &Path) -> EncodedEntry {
    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "-p",
            "raster-program-gemma-externals",
            "--features",
            "encode",
            "--bin",
            "encode",
            "--",
        ])
        .arg(tokenizer_json_path())
        .arg(cache_root)
        .current_dir(workspace_root())
        .output()
        .expect("encoder bin should launch");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "encoder bin failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let field = |prefix: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("encoder printed no '{prefix}' line\nstdout:\n{stdout}"))
            .trim()
            .to_string()
    };
    EncodedEntry {
        data_path: PathBuf::from(field("data_path: ")),
        index_path: PathBuf::from(field("index_path: ")),
        root_commitment: field("root_commitment: "),
    }
}

fn stage_tokenizer(entry: &EncodedEntry, commitment: &str) -> RasterCoreRunDir {
    let run_dir = RasterCoreRunDir::create(RoutineId::PromptPrepare, 1).expect("run dir");
    let mut staged = StagedInputs::new();
    staged
        .add_raster_encoded("tokenizer", &entry.data_path, &entry.index_path, commitment)
        .expect("stage tokenizer external");
    staged.write(&run_dir).expect("write staged inputs");
    run_dir
}

/// Expected smoke values derived from the source `tokenizer.json` itself —
/// the test never hardcodes tiny-gemma-dev specifics.
fn expected_smoke() -> TokenizerSmoke {
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(tokenizer_json_path()).expect("tokenizer.json should be readable"),
    )
    .expect("tokenizer.json should parse");
    let vocab = raw["model"]["vocab"]
        .as_object()
        .expect("vocab should be an object");
    let (first_token, first_id) = vocab
        .iter()
        .min_by(|(a, _), (b, _)| a.cmp(b))
        .expect("vocab should be non-empty");
    let special_token_count = raw["added_tokens"]
        .as_array()
        .expect("added_tokens should be an array")
        .iter()
        .filter(|token| token["special"].as_bool() == Some(true))
        .count() as u32;
    TokenizerSmoke {
        unk_token: raw["model"]["unk_token"]
            .as_str()
            .expect("unk_token should be a string")
            .to_string(),
        first_token: first_token.clone(),
        first_token_id: first_id.as_u64().expect("token id should be a u64") as u32,
        special_token_count,
    }
}

#[test]
fn tokenizer_external_encodes_deterministically_and_round_trips() {
    if !require_or_skip() {
        return;
    }
    let temp_root = std::env::temp_dir().join(format!(
        "ws2-gemma-tokenizer-external-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_root).expect("temp root");
    let runner = CargoRasterRunner::default();
    let program_dir = program_crate_dir();

    // Determinism: two cache roots, identical commitments; re-encoding into
    // a warm cache reuses the entry (same commitment reported).
    let entry = encode_tokenizer(&temp_root.join("cache-a"));
    let entry_again = encode_tokenizer(&temp_root.join("cache-b"));
    assert_eq!(
        entry.root_commitment, entry_again.root_commitment,
        "raster encoding of the same tokenizer.json must be deterministic"
    );
    let entry_cached = encode_tokenizer(&temp_root.join("cache-a"));
    assert_eq!(entry_cached.root_commitment, entry.root_commitment);
    assert!(
        entry
            .data_path
            .parent()
            .is_some_and(|dir| dir.file_name().is_some_and(|name| name.len() == 64)),
        "cache entries are addressed by the source sha256: {}",
        entry.data_path.display()
    );

    // Happy path: the smoke program selects through the schema and the
    // ingested values match the source tokenizer.json.
    let run_dir = stage_tokenizer(&entry, &entry.root_commitment);
    let run_output = runner
        .run(&program_dir, &run_dir)
        .expect("cargo raster run should complete");
    let result = ingest::<TokenizerSmoke>(&run_dir, &run_output, Default::default())
        .unwrap_or_else(|error| {
            panic!(
                "tokenizer smoke ingest failed: {error}\nstdout:\n{}\nstderr:\n{}",
                run_output.stdout, run_output.stderr
            )
        });
    assert_eq!(result.value, expected_smoke());
    std::fs::remove_dir_all(run_dir.root()).ok();

    // Tamper leg A — commitment mismatch: intact encoded files staged
    // against an altered manifest commitment. The runtime's integrity check
    // rejects the external at resolve time → Verification.
    let mut flipped_commitment = entry.root_commitment.clone();
    let last_char = flipped_commitment.pop().expect("commitment is non-empty");
    flipped_commitment.push(if last_char == '0' { '1' } else { '0' });
    let run_dir = stage_tokenizer(&entry, &flipped_commitment);
    let run_output = runner
        .run(&program_dir, &run_dir)
        .expect("runner launch should succeed regardless of guest outcome");
    let error = ingest::<TokenizerSmoke>(&run_dir, &run_output, Default::default())
        .expect_err("commitment mismatch must classify as an error");
    match &error {
        RasterCoreError::Verification { detail } => {
            assert!(
                detail.contains("integrity"),
                "unexpected verification detail: {detail}"
            );
        }
        other => panic!(
            "expected Verification, got {other}\nstdout:\n{}\nstderr:\n{}",
            run_output.stdout, run_output.stderr
        ),
    }
    std::fs::remove_dir_all(run_dir.root()).ok();

    // Tamper leg B — corrupt staged file: a byte-flipped index no longer
    // decodes, which is a broken run per the H4 rule ("deserialization of
    // staged inputs" is infrastructure, not a committed outcome).
    let tampered_dir = temp_root.join("tampered");
    std::fs::create_dir_all(&tampered_dir).expect("tamper dir");
    let tampered = EncodedEntry {
        data_path: tampered_dir.join("tokenizer.rastered"),
        index_path: tampered_dir.join("tokenizer.rindex"),
        root_commitment: entry.root_commitment.clone(),
    };
    std::fs::copy(&entry.data_path, &tampered.data_path).expect("copy data");
    let mut index_bytes = std::fs::read(&entry.index_path).expect("read index");
    let last = index_bytes.len() - 1;
    index_bytes[last] ^= 0xff;
    std::fs::write(&tampered.index_path, index_bytes).expect("write tampered index");

    let run_dir = stage_tokenizer(&tampered, &entry.root_commitment);
    let run_output = runner
        .run(&program_dir, &run_dir)
        .expect("runner launch should succeed regardless of guest outcome");
    let error = ingest::<TokenizerSmoke>(&run_dir, &run_output, Default::default())
        .expect_err("undecodable index must classify as an error");
    assert!(
        matches!(error, RasterCoreError::Infrastructure(_)),
        "expected Infrastructure for an undecodable staged input, got {error}"
    );
    std::fs::remove_dir_all(run_dir.root()).ok();

    std::fs::remove_dir_all(&temp_root).ok();
}
