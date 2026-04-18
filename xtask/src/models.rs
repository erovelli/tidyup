//! `cargo xtask download-models` — fetch the default embedding model.
//!
//! This is the sanctioned installer for developers and packagers. The
//! default tidyup CLI binary is network-silent by design (see `CLAUDE.md`)
//! and never links an HTTP client. `xtask` is a dev-only workspace member,
//! so `reqwest` here is not present in release artifacts.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

struct Artifact {
    filename: &'static str,
    url: &'static str,
}

const MODEL_DIR: &str = "bge-small-en-v1.5";

const ARTIFACTS: &[Artifact] = &[
    Artifact {
        filename: "model.onnx",
        url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
    },
    Artifact {
        filename: "tokenizer.json",
        url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json",
    },
];

#[allow(unreachable_pub)]
pub fn download(force: bool) -> Result<()> {
    let target_dir = resolve_cache_dir()?.join(MODEL_DIR);
    std::fs::create_dir_all(&target_dir)
        .with_context(|| format!("create cache dir {}", target_dir.display()))?;

    println!("Downloading into {}", target_dir.display());
    for artifact in ARTIFACTS {
        let dest = target_dir.join(artifact.filename);
        if dest.exists() && !force {
            println!(
                "  · {} already present (use --force to overwrite)",
                artifact.filename
            );
            continue;
        }
        fetch(&dest, artifact)?;
        let hash = blake3_file(&dest)?;
        println!("  ✓ {} blake3={}", artifact.filename, hash);
    }
    Ok(())
}

fn resolve_cache_dir() -> Result<PathBuf> {
    if let Ok(overridden) = std::env::var("TIDYUP_MODEL_CACHE") {
        return Ok(PathBuf::from(overridden));
    }
    dirs::cache_dir()
        .map(|d| d.join("tidyup").join("models"))
        .ok_or_else(|| {
            anyhow::anyhow!("platform cache dir unavailable; set TIDYUP_MODEL_CACHE to pick one",)
        })
}

fn fetch(dest: &Path, artifact: &Artifact) -> Result<()> {
    let mut response =
        reqwest::blocking::get(artifact.url).with_context(|| format!("GET {}", artifact.url))?;
    if !response.status().is_success() {
        bail!("GET {} returned {}", artifact.url, response.status());
    }
    let total = response.content_length().unwrap_or(0);
    let bar = make_bar(total, artifact.filename);

    let tmp = dest.with_extension("partial");
    {
        let mut writer = BufWriter::new(
            File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?,
        );
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = response.read(&mut buf).context("stream read")?;
            if n == 0 {
                break;
            }
            writer
                .write_all(&buf[..n])
                .with_context(|| format!("write {}", tmp.display()))?;
            bar.inc(n as u64);
        }
        writer.flush().ok();
    }
    bar.finish_and_clear();

    std::fs::rename(&tmp, dest)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
    Ok(())
}

fn make_bar(total: u64, name: &str) -> ProgressBar {
    let bar = ProgressBar::new(total);
    #[allow(clippy::literal_string_with_formatting_args)]
    let template = "  {msg:>18} {bar:40.cyan/blue} {bytes:>10}/{total_bytes:>10} {elapsed_precise}";
    if let Ok(style) = ProgressStyle::with_template(template) {
        bar.set_style(style);
    }
    bar.set_message(name.to_string());
    bar
}

fn blake3_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}
