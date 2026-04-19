//! Smoke test: drive the `tidyup` binary with `--help` / `config` / `rollback --list`
//! against a scratch `$TIDYUP_DATA_DIR` so the local environment isn't touched.
//!
//! Full end-to-end scan/migrate tests require the ONNX model which lives outside
//! the repo — those are covered by `tidyup-app`'s service integration tests
//! against a deterministic stub embedding backend.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use tempfile::TempDir;

fn bin() -> std::path::PathBuf {
    // `CARGO_BIN_EXE_<name>` is injected at *compile time* for integration
    // tests living in a binary crate — hence `env!` rather than `var_os`.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_tidyup"))
}

#[test]
fn help_subcommand_lists_all_commands() {
    let out = Command::new(bin())
        .args(["--help"])
        .output()
        .expect("binary runs");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    for cmd in ["migrate", "scan", "rollback", "config"] {
        assert!(stdout.contains(cmd), "help missing `{cmd}`:\n{stdout}");
    }
}

#[test]
fn rollback_list_on_empty_db_reports_no_runs() {
    let data = TempDir::new().unwrap();
    let out = Command::new(bin())
        .args(["rollback", "--list"])
        .env("TIDYUP_DATA_DIR", data.path())
        .output()
        .expect("binary runs");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("No recorded runs"), "stdout={stdout}");
}

#[test]
fn missing_model_surfaces_installation_instructions() {
    let data = TempDir::new().unwrap();
    let model_cache = TempDir::new().unwrap();
    let source = TempDir::new().unwrap();
    // Scan requires the embedding model — with TIDYUP_MODEL_CACHE pointing at
    // an empty dir, the binary should fail fast with the installer hint.
    let out = Command::new(bin())
        .args([
            "--yes",
            "scan",
            source.path().to_str().unwrap(),
            "--dry-run",
        ])
        .env("TIDYUP_DATA_DIR", data.path())
        .env("TIDYUP_MODEL_CACHE", model_cache.path())
        .output()
        .expect("binary runs");
    assert!(
        !out.status.success(),
        "scan should fail without model; stdout={}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("Missing embedding model"),
        "expected installer instructions in stderr, got: {stderr}"
    );
    assert!(
        stderr.contains("bge-small-en-v1.5"),
        "expected model name in stderr, got: {stderr}"
    );
}

#[test]
fn config_subcommand_prints_parsed_defaults() {
    let data = TempDir::new().unwrap();
    let out = Command::new(bin())
        .args(["config"])
        .env("TIDYUP_DATA_DIR", data.path())
        .output()
        .expect("binary runs");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("[inference]"), "stdout={stdout}");
    assert!(stdout.contains("llm_fallback = false"), "stdout={stdout}");
    assert!(
        !stdout.contains("remote-"),
        "default config should not hint at any remote backend: {stdout}"
    );
}
