//! Structural enforcement of the DET_NUM_SPEC "Parallelism legality" rules.
//!
//! The zkVM guest profile executes the serial reference schedule: canonical
//! kernel sources (`det_num`), raster tile kernels, and raster routine
//! modules must remain free of parallelism constructs. Parallel drivers live
//! in native-only modules (`det_kernels`, `transformer_kernels`) that invoke
//! the canonical semantics.
//!
//! This test denies `rayon` and `std::thread` usage in the guest-profile
//! source trees so the "reduction-axis parallelism" class of bugs is
//! structurally prevented rather than reviewed away. The same trees must
//! also stay free of SIMD constructs (`std::arch`, `target_feature`): SIMD
//! schedules live in the native-only `det_simd` module, and the guest
//! profile always executes the canonical scalar reference.

use std::fs;
use std::path::{Path, PathBuf};

/// Substrings that indicate a parallelism or SIMD construct. Matches anywhere
/// in the source (including comments) to keep the rule simple and
/// conservative.
const FORBIDDEN_TOKENS: &[&str] = &[
    "rayon",
    "par_iter",
    "par_chunks",
    "par_bridge",
    "std::thread",
    "thread::spawn",
    "std::arch",
    "core::arch",
    "target_feature",
    "portable_simd",
];

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn collect_rust_sources(dir: &Path, sources: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry should read").path();
        if path.is_dir() {
            collect_rust_sources(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

fn serial_only_directories() -> Vec<PathBuf> {
    let root = manifest_dir();
    let mut directories = vec![
        root.join("src/shared/numerics/det_num"),
        root.join("src/shared/raster_kernels"),
    ];
    let routines = root.join("src/routines");
    for entry in fs::read_dir(&routines).expect("routines directory should read") {
        let path = entry.expect("dir entry should read").path();
        let raster = path.join("raster");
        if raster.is_dir() {
            directories.push(raster);
        }
    }
    directories
}

#[test]
fn guest_profile_sources_contain_no_parallelism_or_simd_constructs() {
    let mut violations = Vec::new();
    for directory in serial_only_directories() {
        assert!(
            directory.is_dir(),
            "serial-only directory is missing: {}",
            directory.display()
        );
        let mut sources = Vec::new();
        collect_rust_sources(&directory, &mut sources);
        assert!(
            !sources.is_empty(),
            "serial-only directory has no Rust sources: {}",
            directory.display()
        );
        for source in sources {
            let contents = fs::read_to_string(&source)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", source.display()));
            for (line_idx, line) in contents.lines().enumerate() {
                for token in FORBIDDEN_TOKENS {
                    if line.contains(token) {
                        violations.push(format!(
                            "{}:{}: forbidden parallelism/SIMD token `{token}`: {}",
                            source.display(),
                            line_idx + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "parallelism/SIMD constructs found in serial-only (guest-profile) sources;\n\
         parallel and SIMD drivers must live in native-only modules (see \
         DET_NUM_SPEC \"Parallelism legality\"):\n{}",
        violations.join("\n")
    );
}
