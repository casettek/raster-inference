use std::fs;
use std::path::{Path, PathBuf};

const FAMILY_NEEDLES: &[&str] = &[
    "Gemma",
    "gemma",
    "AuthenticatedGemma",
    "Gemma4TransformerModel",
    "shared::model::transformer::Gemma",
];

#[test]
fn protocol_and_runtime_surfaces_do_not_name_model_families() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();

    for path in [
        "src/runtime/sequence.rs",
        "src/runtime/inference/mod.rs",
        "src/runtime/pipeline.rs",
        "src/runtime/checkpoints.rs",
        "src/runtime/trace.rs",
        "src/runtime/executors",
        "src/runtime/roles",
        "src/shared/api",
        "src/shared/artifacts",
    ] {
        collect_rust_files(&root.join(path), &mut files);
    }

    let mut violations = Vec::new();
    for file in files {
        let relative = file
            .strip_prefix(&root)
            .expect("file should be under crate root");
        let text = fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", relative.display()));
        for needle in FAMILY_NEEDLES {
            if text.contains(needle) {
                violations.push(format!("{} contains `{needle}`", relative.display()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "model-family names leaked into protected surfaces:\n{}",
        violations.join("\n")
    );
}

fn collect_rust_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path.to_path_buf());
        }
        return;
    }

    let entries = fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to read directory {}: {error}", path.display()));
    for entry in entries {
        let entry = entry.expect("directory entry should be readable");
        collect_rust_files(&entry.path(), files);
    }
}
