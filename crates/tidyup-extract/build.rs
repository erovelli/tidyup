use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/macos_ocr.swift");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var_os("CARGO_FEATURE_IMAGE").is_none()
    {
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("cargo did not set OUT_DIR")?);
    let output = out_dir.join("tidyup-ocr");
    let module_cache = out_dir.join("swift-module-cache");
    let arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    let target = format!("{arch}-apple-macosx14.0");
    let status = Command::new("xcrun")
        .args(["swiftc", "-O", "-target"])
        .arg(target)
        .arg("-module-cache-path")
        .arg(module_cache)
        .args(["src/macos_ocr.swift", "-o"])
        .arg(&output)
        .status()?;

    if !status.success() {
        return Err("compiling the local macOS OCR helper failed".into());
    }
    Ok(())
}
