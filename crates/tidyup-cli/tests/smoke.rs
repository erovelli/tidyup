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
fn boolish_env_vars_activate_the_tier3_gates() {
    // End-to-end guard for the WP-3 env-activation wiring: `TIDYUP_LLM_FALLBACK=1`
    // and `TIDYUP_REMOTE=1` are read (boolishly) in dispatch. Setting BOTH must
    // trip the mutual-exclusion error, which fires in dispatch *before* any model
    // load — so this needs no ONNX model and proves both env vars activated.
    // If the env wiring were reverted (env ignored), neither would activate and
    // the run would instead fail later on the missing model, not here.
    let data = TempDir::new().unwrap();
    let source = TempDir::new().unwrap();
    let out = Command::new(bin())
        .args(["scan", source.path().to_str().unwrap(), "--dry-run"])
        .env("TIDYUP_DATA_DIR", data.path())
        .env("TIDYUP_LLM_FALLBACK", "1")
        .env("TIDYUP_REMOTE", "1")
        .output()
        .expect("binary runs");
    assert!(!out.status.success(), "conflicting activation must fail");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("mutually exclusive"),
        "both env vars must activate (mutual-exclusion error expected), got: {stderr}"
    );
}

#[test]
fn falsey_env_var_does_not_activate() {
    // `TIDYUP_LLM_FALLBACK=0` must NOT activate — with a falsey value the run
    // proceeds past dispatch to the model check (default build is LLM-silent).
    let data = TempDir::new().unwrap();
    let model_cache = TempDir::new().unwrap();
    let source = TempDir::new().unwrap();
    let out = Command::new(bin())
        .args(["scan", source.path().to_str().unwrap(), "--dry-run"])
        .env("TIDYUP_DATA_DIR", data.path())
        .env("TIDYUP_MODEL_CACHE", model_cache.path())
        .env("TIDYUP_LLM_FALLBACK", "0")
        .env("TIDYUP_REMOTE", "0")
        .output()
        .expect("binary runs");
    let stderr = String::from_utf8(out.stderr).unwrap();
    // Not a mutual-exclusion or activation error — it fails later, on the model.
    assert!(
        !stderr.contains("mutually exclusive"),
        "falsey env must not activate anything: {stderr}"
    );
    assert!(
        stderr.contains("Missing embedding model"),
        "falsey env should let the run proceed to the model check: {stderr}"
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
