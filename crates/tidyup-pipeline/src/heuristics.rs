//! Tier 1 — extension + MIME + marker-file classifier.
//!
//! The cheapest stage of the classification cascade (~1 ms per file). Files with
//! unambiguous categories by filename, extension, or MIME type never descend to
//! Tier 2 embedding similarity. The mapping is deterministic and curated — it
//! does not learn, it does not call a model, it does not touch the network.
//!
//! # Output shape
//!
//! A hit returns a [`HeuristicMatch`] whose `taxonomy_path` is a string matching
//! one of the leaves in the default taxonomy — scan mode uses it directly,
//! migration mode uses it as a prior on top of embedding similarity.
//!
//! # Calibration
//!
//! Confidence is deliberately conservative. Marker filenames (`.gitignore`,
//! `Cargo.toml`, `package.json`) score 0.95 — practically certain. MIME-obvious
//! media types score 0.85–0.90. Extension-only fallthroughs score 0.70–0.80
//! so the pipeline's `heuristic_threshold` (default 0.60 in
//! [`ClassifierConfig`](tidyup_domain::ClassifierConfig)) admits them while
//! still surfacing ambiguous cases to Tier 2.

use std::path::Path;

/// A Tier-1 match into the default taxonomy.
#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicMatch {
    /// Taxonomy path with trailing slash, e.g. `"Code/Config/"`.
    pub taxonomy_path: &'static str,
    /// Raw confidence in `[0.0, 1.0]`. No calibration in v0.1.
    pub confidence: f32,
    /// Human-readable rule that fired, for proposal reasoning and audit logs.
    pub reason: &'static str,
}

/// Classify a file by filename, extension, and optional MIME type.
///
/// Returns `None` when no rule fires — the caller (scan or migration pipeline)
/// should route to Tier 2 embedding similarity. MIME should be supplied via
/// `tidyup_extract::mime::detect` or a cached value on the `FileIndex` so the
/// file is sniffed at most once per pipeline pass.
#[must_use]
pub fn classify(path: &Path, mime: Option<&str>) -> Option<HeuristicMatch> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if name.is_empty() {
        return None;
    }

    if let Some(hit) = classify_by_filename(name) {
        return Some(hit);
    }

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    if let Some(hit) = classify_by_extension(&ext, name) {
        return Some(hit);
    }

    if let Some(mime) = mime {
        if let Some(hit) = classify_by_mime(mime) {
            return Some(hit);
        }
    }

    None
}

/// Exact-filename rules. Marker files score high — their semantics are
/// unambiguous regardless of where they sit on disk.
fn classify_by_filename(name: &str) -> Option<HeuristicMatch> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        ".gitignore"
        | ".gitattributes"
        | ".editorconfig"
        | ".prettierrc"
        | ".eslintrc"
        | ".eslintrc.json"
        | ".eslintrc.js"
        | ".stylelintrc"
        | ".babelrc"
        | ".npmrc"
        | ".nvmrc"
        | ".rubocop.yml"
        | ".dockerignore"
        | "dockerfile"
        | "makefile"
        | ".env"
        | ".env.local"
        | ".env.example"
        | "pre-commit-config.yaml"
        | ".pre-commit-config.yaml" => Some(HeuristicMatch {
            taxonomy_path: "Code/Config/",
            confidence: 0.95,
            reason: "marker config filename",
        }),
        "readme" | "readme.md" | "readme.txt" | "readme.rst" | "changelog" | "changelog.md"
        | "contributing.md" | "license" | "license.md" | "license.txt" | "license-mit"
        | "license-apache" | "authors" | "notice" | "code_of_conduct.md" | "security.md" => {
            Some(HeuristicMatch {
                taxonomy_path: "Documents/Manuals/",
                confidence: 0.85,
                reason: "project readme or license filename",
            })
        }
        _ => None,
    }
}

/// Extension-keyed rules. The bulk of Tier 1's carry. Screenshot detection
/// leans on the filename prefix because macOS and Windows screenshot files
/// share extensions with arbitrary photos.
#[allow(clippy::too_many_lines)]
fn classify_by_extension(ext: &str, name: &str) -> Option<HeuristicMatch> {
    match ext {
        // -- Source code -----------------------------------------------------
        "rs" | "go" | "py" | "rb" | "js" | "mjs" | "cjs" | "ts" | "tsx" | "jsx" | "java" | "kt"
        | "scala" | "swift" | "c" | "h" | "cpp" | "cc" | "cxx" | "hpp" | "cs" | "php" | "pl"
        | "lua" | "r" | "dart" | "zig" | "nim" | "ex" | "exs" | "erl" | "hs" | "clj" | "cljs"
        | "elm" | "fs" | "ml" | "sh" | "bash" | "zsh" | "fish" | "ps1" | "vue" | "svelte" => {
            Some(HeuristicMatch {
                taxonomy_path: "Code/",
                confidence: 0.80,
                reason: "source-code extension",
            })
        }

        // -- Config files ----------------------------------------------------
        "toml" | "yaml" | "yml" | "ini" | "cfg" | "conf" | "properties" | "env" => {
            Some(HeuristicMatch {
                taxonomy_path: "Code/Config/",
                confidence: 0.80,
                reason: "config-file extension",
            })
        }
        "json" => Some(HeuristicMatch {
            taxonomy_path: "Code/Config/",
            confidence: 0.70,
            reason: "json often used as config",
        }),

        // -- Archives --------------------------------------------------------
        "zip" | "tar" | "gz" | "tgz" | "bz2" | "xz" | "7z" | "rar" | "lz" | "zst" => {
            Some(HeuristicMatch {
                taxonomy_path: "Archives/",
                confidence: 0.90,
                reason: "archive extension",
            })
        }

        // -- Installers / disk images ---------------------------------------
        "dmg" | "iso" | "img" | "vmdk" | "vdi" | "qcow2" => Some(HeuristicMatch {
            taxonomy_path: "Software/Disk Images/",
            confidence: 0.90,
            reason: "disk-image extension",
        }),
        "exe" | "msi" | "pkg" | "deb" | "rpm" | "appimage" | "snap" | "flatpakref" => {
            Some(HeuristicMatch {
                taxonomy_path: "Software/Installers/",
                confidence: 0.90,
                reason: "installer extension",
            })
        }

        // -- Fonts -----------------------------------------------------------
        "ttf" | "otf" | "woff" | "woff2" | "eot" => Some(HeuristicMatch {
            taxonomy_path: "Fonts/",
            confidence: 0.95,
            reason: "font extension",
        }),

        // -- 3D / CAD --------------------------------------------------------
        "stl" | "obj" | "fbx" | "gltf" | "glb" | "dae" | "3ds" | "ply" | "step" | "stp"
        | "iges" | "igs" => Some(HeuristicMatch {
            taxonomy_path: "3D Models/",
            confidence: 0.90,
            reason: "3D-model extension",
        }),

        // -- Maps / GIS ------------------------------------------------------
        "kml" | "kmz" | "gpx" | "geojson" | "shp" => Some(HeuristicMatch {
            taxonomy_path: "Maps/",
            confidence: 0.90,
            reason: "geospatial extension",
        }),

        // -- Books -----------------------------------------------------------
        "epub" | "mobi" | "azw" | "azw3" | "fb2" => Some(HeuristicMatch {
            taxonomy_path: "Books/",
            confidence: 0.90,
            reason: "ebook extension",
        }),

        // -- Spreadsheets ----------------------------------------------------
        "csv" | "tsv" | "xlsx" | "xlsm" | "xls" | "ods" | "numbers" => Some(HeuristicMatch {
            taxonomy_path: "Spreadsheets/",
            confidence: 0.85,
            reason: "spreadsheet / tabular extension",
        }),

        // -- Databases -------------------------------------------------------
        "db" | "sqlite" | "sqlite3" | "mdb" | "accdb" => Some(HeuristicMatch {
            taxonomy_path: "Databases/",
            confidence: 0.90,
            reason: "database extension",
        }),

        // -- Presentations ---------------------------------------------------
        "pptx" | "ppt" | "key" | "odp" => Some(HeuristicMatch {
            taxonomy_path: "Presentations/",
            confidence: 0.85,
            reason: "presentation extension",
        }),

        // -- Images (screenshots split by filename prefix) ------------------
        "png" | "jpg" | "jpeg" | "heic" | "heif" | "webp" | "gif" | "bmp" | "tiff" | "tif" => {
            if is_screenshot_filename(name) {
                Some(HeuristicMatch {
                    taxonomy_path: "Screenshots/",
                    confidence: 0.85,
                    reason: "screenshot filename pattern",
                })
            } else {
                Some(HeuristicMatch {
                    taxonomy_path: "Photos/",
                    confidence: 0.75,
                    reason: "image extension",
                })
            }
        }
        "raw" | "cr2" | "cr3" | "nef" | "arw" | "orf" | "dng" | "rw2" | "raf" => {
            Some(HeuristicMatch {
                taxonomy_path: "Photos/",
                confidence: 0.90,
                reason: "raw-camera extension",
            })
        }

        // -- Videos ----------------------------------------------------------
        "mp4" | "mov" | "mkv" | "avi" | "wmv" | "flv" | "webm" | "m4v" | "mpg" | "mpeg" => {
            Some(HeuristicMatch {
                taxonomy_path: "Videos/",
                confidence: 0.80,
                reason: "video extension",
            })
        }

        // -- Audio -----------------------------------------------------------
        "mp3" | "flac" | "m4a" | "wav" | "ogg" | "opus" | "aiff" | "aif" | "ape" | "wma"
        | "alac" => Some(HeuristicMatch {
            taxonomy_path: "Music/",
            confidence: 0.75,
            reason: "audio extension",
        }),

        _ => None,
    }
}

/// MIME-keyed rules. Only fires when filename + extension both came up empty.
fn classify_by_mime(mime: &str) -> Option<HeuristicMatch> {
    let primary = mime.split('/').next().unwrap_or_default();
    match primary {
        "image" => Some(HeuristicMatch {
            taxonomy_path: "Photos/",
            confidence: 0.75,
            reason: "image MIME type",
        }),
        "video" => Some(HeuristicMatch {
            taxonomy_path: "Videos/",
            confidence: 0.75,
            reason: "video MIME type",
        }),
        "audio" => Some(HeuristicMatch {
            taxonomy_path: "Music/",
            confidence: 0.70,
            reason: "audio MIME type",
        }),
        _ => None,
    }
}

/// Heuristic for screenshot filenames: the platform defaults plus a few common
/// manual prefixes. Not exhaustive — anything this misses falls through to
/// `Photos/` with lower confidence, which is the right default.
fn is_screenshot_filename(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("screenshot")
        || lower.starts_with("screen shot")
        || lower.starts_with("scrn_")
        || lower.starts_with("screen_capture")
        || lower.starts_with("capture_")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn unknown_extension_returns_none() {
        assert!(classify(&PathBuf::from("/tmp/weird.xyz"), None).is_none());
    }

    #[test]
    fn empty_filename_returns_none() {
        assert!(classify(&PathBuf::from("/"), None).is_none());
    }

    #[test]
    fn gitignore_hits_config() {
        let m = classify(&PathBuf::from("/repo/.gitignore"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Code/Config/");
        assert!(m.confidence >= 0.90);
    }

    #[test]
    fn readme_hits_documents() {
        let m = classify(&PathBuf::from("/repo/README.md"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Documents/Manuals/");
    }

    #[test]
    fn rust_source_hits_code() {
        let m = classify(&PathBuf::from("/tmp/main.rs"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Code/");
    }

    #[test]
    fn dmg_hits_disk_images() {
        let m = classify(&PathBuf::from("/tmp/installer.dmg"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Software/Disk Images/");
    }

    #[test]
    fn font_hits_fonts() {
        let m = classify(&PathBuf::from("/tmp/Inter.ttf"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Fonts/");
    }

    #[test]
    fn screenshot_filename_diverts_from_photos() {
        let m = classify(&PathBuf::from("/tmp/Screenshot 2024-02-03.png"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Screenshots/");
    }

    #[test]
    fn plain_photo_hits_photos() {
        let m = classify(&PathBuf::from("/tmp/IMG_1234.HEIC"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Photos/");
    }

    #[test]
    fn raw_camera_extension_hits_photos() {
        let m = classify(&PathBuf::from("/tmp/DSC_0481.NEF"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Photos/");
    }

    #[test]
    fn raw_camera_extension_ignores_screenshot_prefix() {
        // Raw extensions are never screenshots regardless of filename.
        let m = classify(&PathBuf::from("/tmp/screenshot_0481.NEF"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Photos/");
    }

    #[test]
    fn mime_fallback_fires_when_extension_missing() {
        let m = classify(&PathBuf::from("/tmp/opaque"), Some("audio/mpeg")).unwrap();
        assert_eq!(m.taxonomy_path, "Music/");
    }

    #[test]
    fn mime_unused_when_extension_matches() {
        let m = classify(&PathBuf::from("/tmp/main.rs"), Some("text/x-rust")).unwrap();
        assert_eq!(m.taxonomy_path, "Code/");
    }

    #[test]
    fn case_insensitive_extension() {
        let m = classify(&PathBuf::from("/tmp/track.FLAC"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Music/");
    }

    #[test]
    fn dockerfile_without_extension_hits_config() {
        let m = classify(&PathBuf::from("/repo/Dockerfile"), None).unwrap();
        assert_eq!(m.taxonomy_path, "Code/Config/");
    }

    #[test]
    fn confidence_is_bounded() {
        for path in [
            "/tmp/main.rs",
            "/tmp/archive.zip",
            "/repo/.gitignore",
            "/tmp/IMG_1.png",
        ] {
            let m = classify(&PathBuf::from(path), None).unwrap();
            assert!(
                (0.0..=1.0).contains(&m.confidence),
                "confidence out of range for {path}: {}",
                m.confidence,
            );
        }
    }
}
