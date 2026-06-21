//! `cargo xtask download-models` + `verify-models` — fetch and integrity-check
//! the model bundles.
//!
//! This is the sanctioned installer for developers and packagers. The default
//! tidyup CLI binary is network-silent by design (see `CLAUDE.md`) and never
//! links an HTTP client. `xtask` is a dev-only workspace member, so `reqwest`
//! here is not present in release artifacts.
//!
//! Bundle definitions live in `tidyup_embeddings_ort::install` (the same
//! `BundleSpec` / `ArtifactSpec` the runtime verifier uses), so the downloader
//! and the binary can never disagree on filenames, URLs, or pinned checksums.
//! After each download the file is verified: a pinned BLAKE3 is **enforced**
//! (the corrupt file is deleted and the run fails), and an unpinned artifact
//! reports its digest + size so a maintainer can pin it.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use tidyup_embeddings_ort::install::{
    artifact_digest, verify_artifact, ArtifactSpec, BundleSpec, CLAP_BUNDLE, DEFAULT_BUNDLE,
    SIGLIP_BUNDLE,
};
use tidyup_embeddings_ort::{verify_clap_model, verify_default_model, verify_siglip_model};

#[allow(unreachable_pub)]
pub fn download(force: bool, siglip: bool, clap: bool) -> Result<()> {
    let cache = resolve_cache_dir()?;
    let mut bundles: Vec<&BundleSpec> = vec![&DEFAULT_BUNDLE];
    if siglip {
        bundles.push(&SIGLIP_BUNDLE);
    }
    if clap {
        bundles.push(&CLAP_BUNDLE);
    }

    let mut unpinned = 0usize;
    for bundle in bundles {
        unpinned += download_bundle(&cache, bundle, force)?;
    }

    if unpinned > 0 {
        println!();
        println!(
            "{unpinned} artifact(s) are unpinned (no checksum enforced). To pin them, copy the\n\
             printed blake3/size into the matching ArtifactSpec in\n\
             crates/tidyup-embeddings-ort/src/install.rs — ideally after switching the URL from\n\
             `resolve/main/` to an immutable `resolve/<commit-sha>/` revision so the pin is stable."
        );
    }
    Ok(())
}

/// Outcome of integrity-checking an artifact on disk.
enum Verdict {
    /// The spec pins a checksum and the file matched it.
    Pinned,
    /// The spec does not pin a checksum; here is what the file hashes to.
    Unpinned { blake3: String, size: u64 },
}

/// Download one bundle. Returns the count of unpinned artifacts seen, so the
/// caller can print a single pin-me hint at the end.
fn download_bundle(cache: &Path, bundle: &BundleSpec, force: bool) -> Result<usize> {
    let target_dir = cache.join(bundle.dir);
    std::fs::create_dir_all(&target_dir)
        .with_context(|| format!("create cache dir {}", target_dir.display()))?;

    println!(
        "[{}] Downloading into {}",
        bundle.name,
        target_dir.display()
    );

    let mut unpinned = 0usize;
    for spec in bundle.artifacts {
        let dest = target_dir.join(spec.filename);
        if dest.exists() && !force {
            // Skip the download, but still verify what's already on disk.
            match report_or_verify(&dest, spec)? {
                Verdict::Pinned => {
                    println!("  · {} already present, checksum OK", spec.filename);
                }
                Verdict::Unpinned { blake3, size } => {
                    println!(
                        "  · {} already present — blake3={blake3} size={size} (unpinned)",
                        spec.filename
                    );
                    unpinned += 1;
                }
            }
            continue;
        }
        fetch(&dest, spec)?;
        match report_or_verify(&dest, spec)? {
            Verdict::Pinned => {
                println!("  ✓ {} verified against pinned checksum", spec.filename);
            }
            Verdict::Unpinned { blake3, size } => {
                println!(
                    "  ✓ {} downloaded — blake3={blake3} size={size} (unpinned)",
                    spec.filename
                );
                unpinned += 1;
            }
        }
    }
    Ok(unpinned)
}

/// Verify a file against its spec. A pinned-checksum mismatch deletes the
/// corrupt file and returns an error; an unpinned artifact reports its digest
/// so a maintainer can pin it.
fn report_or_verify(dest: &Path, spec: &ArtifactSpec) -> Result<Verdict> {
    if spec.blake3_hex.is_empty() {
        let (blake3, size) = artifact_digest(dest)?;
        Ok(Verdict::Unpinned { blake3, size })
    } else if let Err(e) = verify_artifact(dest, spec) {
        let _ = std::fs::remove_file(dest);
        Err(e).with_context(|| {
            format!(
                "integrity check failed for {}; removed the corrupt download",
                spec.filename
            )
        })
    } else {
        Ok(Verdict::Pinned)
    }
}

/// A bundle name paired with its verification entry point.
type VerifyTarget = (&'static str, fn() -> Result<PathBuf>);

/// `cargo xtask verify-models` — verify already-installed bundles against their
/// specs and report per-bundle pass/fail.
#[allow(unreachable_pub)]
pub fn verify(siglip: bool, clap: bool) -> Result<()> {
    let mut targets: Vec<VerifyTarget> = vec![("bge-small-en-v1.5", verify_default_model)];
    if siglip {
        targets.push(("siglip-base-patch16-224", verify_siglip_model));
    }
    if clap {
        targets.push(("clap-htsat-unfused", verify_clap_model));
    }

    let mut failures = 0usize;
    for (name, verify_fn) in targets {
        match verify_fn() {
            Ok(dir) => println!("  ✓ {name}: OK ({})", dir.display()),
            Err(e) => {
                failures += 1;
                println!("  ✗ {name}: {e:#}");
            }
        }
    }
    if failures > 0 {
        bail!("{failures} bundle(s) failed verification");
    }
    println!("All requested model bundles verified.");
    Ok(())
}

fn resolve_cache_dir() -> Result<PathBuf> {
    if let Ok(overridden) = std::env::var("TIDYUP_MODEL_CACHE") {
        return Ok(PathBuf::from(overridden));
    }
    dirs::cache_dir()
        .map(|d| d.join("tidyup").join("models"))
        .ok_or_else(|| {
            anyhow::anyhow!("platform cache dir unavailable; set TIDYUP_MODEL_CACHE to pick one")
        })
}

fn fetch(dest: &Path, spec: &ArtifactSpec) -> Result<()> {
    let mut response =
        reqwest::blocking::get(spec.url).with_context(|| format!("GET {}", spec.url))?;
    if !response.status().is_success() {
        bail!("GET {} returned {}", spec.url, response.status());
    }
    let total = response.content_length().unwrap_or(0);
    let bar = make_bar(total, spec.filename);

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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn spec(blake3_hex: &'static str) -> ArtifactSpec {
        ArtifactSpec {
            filename: "blob.bin",
            url: "https://huggingface.co/example/resolve/main/blob.bin",
            size_bytes: 0,
            blake3_hex,
        }
    }

    #[test]
    fn unpinned_spec_reports_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        std::fs::write(&path, b"hello tidyup").unwrap();
        match report_or_verify(&path, &spec("")).unwrap() {
            Verdict::Unpinned { blake3, size } => {
                assert_eq!(size, 12);
                assert_eq!(blake3, blake3::hash(b"hello tidyup").to_hex().to_string());
            }
            Verdict::Pinned => panic!("empty checksum must be reported as unpinned"),
        }
        assert!(path.exists(), "unpinned check must not delete the file");
    }

    #[test]
    fn pinned_matching_checksum_passes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        std::fs::write(&path, b"hello tidyup").unwrap();
        let hex = blake3::hash(b"hello tidyup").to_hex().to_string();
        // `spec` takes a &'static str; leak the computed hex for the test only.
        let pinned = Box::leak(hex.into_boxed_str());
        assert!(matches!(
            report_or_verify(&path, &spec(pinned)).unwrap(),
            Verdict::Pinned
        ));
        assert!(path.exists());
    }

    #[test]
    fn pinned_mismatch_deletes_file_and_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        std::fs::write(&path, b"hello tidyup").unwrap();
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(report_or_verify(&path, &spec(wrong)).is_err());
        assert!(!path.exists(), "a corrupt download must be removed");
    }
}
