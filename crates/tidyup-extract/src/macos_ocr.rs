//! Local screenshot OCR backed by Apple's Vision framework.
//!
//! The tiny Swift helper is compiled at Rust build time and embedded into the
//! desktop binary. At runtime it is materialised under the process temp
//! directory and invoked only for an image being extracted. No network, cloud
//! API, or user account is involved.

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use anyhow::{anyhow, Context, Result};

const HELPER_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tidyup-ocr"));
static HELPER_FILE: OnceLock<Result<tempfile::NamedTempFile, String>> = OnceLock::new();

fn helper_path() -> Result<&'static Path> {
    let resolved = HELPER_FILE.get_or_init(|| {
        let install = (|| -> Result<tempfile::NamedTempFile> {
            let mut file = tempfile::Builder::new()
                .prefix("tidyup-ocr-")
                .tempfile()
                .context("creating private temporary OCR helper")?;
            file.write_all(HELPER_BYTES)
                .context("writing embedded OCR helper")?;
            file.flush().context("flushing embedded OCR helper")?;
            let mut permissions = file.as_file().metadata()?.permissions();
            permissions.set_mode(0o700);
            file.as_file().set_permissions(permissions)?;
            Ok(file)
        })();
        install.map_err(|error| error.to_string())
    });

    match resolved {
        Ok(file) => Ok(file.path()),
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
