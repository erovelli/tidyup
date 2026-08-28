//! Local screenshot OCR backed by Apple's Vision framework.
//!
//! The tiny Swift helper is compiled at Rust build time and embedded into the
//! desktop binary. At runtime it is materialised under the process temp
//! directory and invoked only for an image being extracted. No network, cloud
//! API, or user account is involved.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use anyhow::{anyhow, Context, Result};

const HELPER_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tidyup-ocr"));
static HELPER_PATH: OnceLock<Result<PathBuf, String>> = OnceLock::new();

fn helper_path() -> Result<&'static Path> {
    let resolved = HELPER_PATH.get_or_init(|| {
        let path = std::env::temp_dir().join(format!(
            "tidyup-ocr-{}-{}",
            env!("CARGO_PKG_VERSION"),
            std::process::id()
        ));
        let install = (|| -> Result<PathBuf> {
            fs::write(&path, HELPER_BYTES)
                .with_context(|| format!("writing embedded OCR helper to {}", path.display()))?;
            let mut permissions = fs::metadata(&path)?.permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&path, permissions)?;
            Ok(path)
        })();
        install.map_err(|error| error.to_string())
    });

    match resolved {
        Ok(path) => Ok(path.as_path()),
        Err(message) => Err(anyhow!(message.clone())),
    }
}

#[allow(clippy::redundant_pub_crate)]
pub(super) fn recognize(path: &Path) -> Result<String> {
    let output = Command::new(helper_path()?)
        .arg(path)
        .output()
        .with_context(|| format!("running local OCR for {}", path.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "local OCR failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let text = String::from_utf8(output.stdout).context("OCR output was not UTF-8")?;
    Ok(text.trim().to_string())
}
