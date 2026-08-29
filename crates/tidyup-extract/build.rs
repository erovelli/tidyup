use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/macos_ocr.swift");
    println!("cargo:rustc-check-cfg=cfg(macos_vision_ocr)");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var_os("CARGO_FEATURE_IMAGE").is_none()
    {
        return;
    }

    if let Err(error) = compile_ocr_helper() {
        println!(
            "cargo:warning=macOS Vision OCR disabled: {error}. Install an Xcode toolchain with Swift 5 and the macOS 14 SDK to enable it."
        );
    }
}

fn compile_ocr_helper() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("cargo did not set OUT_DIR")?);
    let output = out_dir.join("tidyup-ocr");
    let module_cache = out_dir.join("swift-module-cache");
    let arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    let target = format!("{arch}-apple-macosx14.0");
    let output_result = Command::new("xcrun")
        .args(["swiftc", "-O", "-target"])
        .arg(target)
        .arg("-module-cache-path")
        .arg(module_cache)
        .args(["src/macos_ocr.swift", "-o"])
        .arg(&output)
        .output();

    let command_output = output_result.map_err(|error| format!("could not run xcrun: {error}"))?;

    if !command_output.status.success() {
        let stderr = String::from_utf8_lossy(&command_output.stderr);
        return Err(format!("Swift helper compilation failed: {}", stderr.trim()).into());
    }
    println!("cargo:rustc-cfg=macos_vision_ocr");
    Ok(())
}
