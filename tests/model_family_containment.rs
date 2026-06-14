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

#[test]
fn authenticated_source_construction_stays_at_model_family_boundary() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();

    collect_rust_files(&root.join("src"), &mut files);

    let mut violations = Vec::new();
    for file in files {
        let relative = file
            .strip_prefix(&root)
            .expect("file should be under crate root");
        if is_model_family_module(relative) || is_test_or_fixture_file(relative) {
            continue;
        }

        let text = fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", relative.display()));
        let production_text = strip_cfg_test_modules(&text);
        for (line_idx, line) in production_text.lines().enumerate() {
            if line.contains("AuthenticatedDecoder") && line.contains("::from_model(") {
                violations.push(format!(
                    "{}:{} constructs an authenticated source from a model",
                    relative.display(),
                    line_idx + 1
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "authenticated source construction must stay under shared/model/<family>:\n{}",
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

fn is_model_family_module(path: &Path) -> bool {
    path.starts_with(Path::new("src/shared/model/gemma"))
        || path.starts_with(Path::new("src/shared/model/qwen"))
}

fn is_test_or_fixture_file(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "tests")
        || path
            .file_name()
            .is_some_and(|file_name| file_name == "tests.rs")
}

fn strip_cfg_test_modules(text: &str) -> String {
    let mut output = String::new();
    let mut pending_cfg_test = false;
    let mut skipping_cfg_test_module = false;
    let mut brace_depth = 0isize;

    for line in text.lines() {
        let trimmed = line.trim_start();

        if skipping_cfg_test_module {
            brace_depth += count_char(line, '{') - count_char(line, '}');
            if brace_depth <= 0 {
                skipping_cfg_test_module = false;
                brace_depth = 0;
            }
            continue;
        }

        if trimmed.starts_with("#[cfg(test)]") {
            pending_cfg_test = true;
            continue;
        }

        if pending_cfg_test && trimmed.starts_with("mod tests") && trimmed.contains('{') {
            skipping_cfg_test_module = true;
            brace_depth = count_char(line, '{') - count_char(line, '}');
            if brace_depth <= 0 {
                skipping_cfg_test_module = false;
                brace_depth = 0;
            }
            pending_cfg_test = false;
            continue;
        }

        if pending_cfg_test && !trimmed.starts_with("#[") && !trimmed.is_empty() {
            pending_cfg_test = false;
        }

        output.push_str(line);
        output.push('\n');
    }

    output
}

fn count_char(line: &str, target: char) -> isize {
    line.chars()
        .filter(|character| *character == target)
        .count() as isize
}
