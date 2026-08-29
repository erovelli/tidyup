# tidyup

> A local-first file organizer that never phones home.

**tidyup** watches a directory, understands what's in your files using small language models running _entirely on your machine_, and proposes a tidier structure. You review every change before anything moves. Nothing is uploaded. Nothing is logged. No account. No cloud. No telemetry.

Your files stay where they belong — with you.

---

## Status: pre-alpha — end-to-end, not yet v0.1

> **Warning — tidyup is still under active construction.**
>
> `tidyup migrate`, `tidyup scan`, and `tidyup rollback` run end-to-end on
> the default binary: proposals are generated, reviewed, shelved, moved, and
> reversible. First-run fails fast with installer instructions when the
> embedding model is missing. The `tidyup-desktop` Dioxus UI is wired on top
> of the same `ServiceContext` — the plug-and-play seam promised by the
> architecture: CLI and UI differ only in how they report progress and
> gather review decisions. Phase 7 added optional cross-modal embeddings for
> images (SigLIP) and audio (CLAP) in scan mode; both load when their ONNX
> bundles are present. The optional LLM fallback is wired through the
> same pipeline seam — opt in with `--features llm-fallback` (or
> `--features remote`) plus the matching config + flag activation, and
> low-confidence embedding verdicts get a second-opinion re-rank. Interactive
> bundle review now works in **both the CLI and the desktop UI** — each detected
> bundle gets an atomic approve/reject instead of silently staying pending.
> Migration mode now profiles target folders with a text content centroid (from
> each folder's documents), plus image/audio centroids when the SigLIP/CLAP
> bundles are installed, so files route on their own contents instead of
> folder-name text. The desktop UI has a **Settings → LLM fallback toggle** that
> surfaces the same three-gate activation as the CLI (build the UI with
> `--features llm-fallback`, set `[inference] llm_fallback = true`, then flip the
> per-session switch). That said: confidence thresholds aren't calibrated against
> a real corpus by default. Don't point tidyup at files you care about.
>
> Everything below describes the target design. Check the
> [roadmap](#roadmap) for what actually works today.

---

## Contents

- [Why this exists](#why-this-exists)
- [What it does](#what-it-does)
- [Privacy guarantees](#privacy-guarantees)
- [Installing](#installing)
- [Building](#building)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Environment variables](#environment-variables)
- [Where tidyup stores things](#where-tidyup-stores-things)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why this exists

Most "smart" file organizers are thin wrappers around someone else's API. You hand them your tax returns, your medical bills, your draft manuscripts, your half-written love letters — and trust that a privacy policy somewhere protects you.

tidyup takes a different stance:

- **Deterministic, offline classification.** The default binary classifies by embedding similarity via `bge-small-en-v1.5` on ONNX Runtime — entirely local, fully reproducible, no LLM generation. Classification is bit-for-bit stable across runs.
- **LLMs are an optional escape hatch, not a default.** The default binary contains no LLM inference — `mistralrs`/`candle`/`hf-hub` are not loaded, not linked, not reachable. A local LLM fallback is available as a compile-time + runtime opt-in (`--features llm-fallback`) for sparse learned hierarchies and pathological extraction. A truly empty migration target has no candidates for any backend to rank: bundles are preserved at its root at zero confidence for review, and loose files are reported unclassified.
- **No network calls by default.** The default binary has no HTTP client — not loaded, not linked, not reachable. Airplane-mode your machine and it still works. A remote backend is available as a symmetric compile-time + runtime opt-in (`--features remote`) for power users.
- **Groupings stay grouped.** A coding project, photo burst, or music album moves as an atomic bundle or not at all — tidyup will never fragment one.
- **Content-aware renames, but never silent.** When a filename clearly disagrees with the contents, tidyup proposes a rename from extractive metadata, local OCR text, or keyphrases. Every source must clear the same two tunable confidence/mismatch thresholds, and renames always require explicit approval. Colliding names inside a semantic collection are suffixed deterministically and duplicate manual edits are rejected inline.
- **No telemetry.** No analytics, no crash reporting, no "anonymous usage data."
- **Human-in-the-loop, always.** Every rename and move is a _proposal_. Nothing touches your filesystem until you approve it; `--yes` is explicit approval for high-confidence move-only proposals and never includes renames.
- **Every change is reversible.** Originals are copied to a backup shelf before any move. Restore anything within 30 days (configurable). Rollback is edit-safe: before deleting a moved file it verifies the shelf copy is intact (content hash) and that the destination still matches what was moved — a file you've edited since apply is reported as a **conflict** and left in place rather than overwritten, and a new file occupying the original location is never clobbered.

This is a portfolio project and a personal tool. It is also a statement: useful AI does not require surrendering your data.

---

## What it does

- **Hashes file contents with BLAKE3** and logs every proposal, move, and backup to a local SQLite database, so each run is reviewable and reversible. (Content-addressed dedup — classifying identical contents once no matter how many copies exist — is planned, not yet wired: each loose file is currently classified independently.)
- **Detects logical groupings first.** Coding projects, photo bursts, music albums, Jupyter notebook sets, document series — tidyup recognizes these as bundles via structural markers (`.git/`, `Cargo.toml`, `package.json`, consistent EXIF timestamps, matching ID3 album tags, etc.) and moves them as atomic units. A coding project is never shredded; either the whole tree relocates or nothing does.
- **Classifies each loose file semantically.** The default path embeds filename/context and extracted content with `bge-small-en-v1.5`, then ranks the fixed scan taxonomy or learned migration profiles by cosine similarity. There is no extension/keyword destination router. An optional local LLM fallback (1–10s, `--features llm-fallback`) can rerank uncertain embedding results; its `suggested_name` is ignored. Default builds exclude the LLM entirely and surface low-confidence files for review.
- **Proposes a destination folder** — with a plain-English reason.
- **Proposes a rename** when filename and contents disagree — using two tunable signals (classification confidence × filename-content mismatch). Renames never auto-apply: even `--yes` auto-rejects them, so approving a rename requires an interactive review run.
- **Shows you a diff-style review UI** — approve or reject per file or per bundle.
- **Backs up originals before moving** — restore anything, anytime.

Two modes:

- **Scan mode** — organize a messy directory against a built-in taxonomy, or your own via `scan --taxonomy <file.toml>`.
- **Migration mode** — sort a source directory into an _existing_ target hierarchy whose structure tidyup learns: it profiles each target folder (name + content centroid + organizational type: semantic / date-based / project-based / status-based) and routes new files to where they semantically fit.

---

## Privacy guarantees

| Guarantee             | How it's enforced                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| No network inference (default) | Default binary is built without `tidyup-inference-remote`. No HTTP client, no `reqwest`/`hyper`/`rustls` linked. Verified in CI by `cargo tree -p tidyup-cli -e normal` on the default feature set. |
| No LLM inference (default) | Default binary is built without `tidyup-inference-mistralrs`. No `mistralrs`/`candle`/`hf-hub`/heavy tokenizer tree linked. Classification is deterministic embedding similarity only. `--features llm-fallback` additionally links `reqwest` transitively via `hf-hub` (needed for one-shot model download); there is no classifier-time phone-home. |
| No telemetry          | No analytics/telemetry crate in any `Cargo.toml`; `cargo xtask check-privacy` keeps the default CLI/UI graphs free of network & LLM deps. |
| No background uploads | App has no cloud sync feature, by design.                                                             |
| Local-only storage    | SQLite DB + config + backups all under platform data dir.                                             |
| Reversible by default | Every move is preceded by a copy to the backup shelf.                                                 |
| Atomic bundle moves   | Coding projects, photo albums, and other groupings move as a single unit or not at all.                |
| Extractive renames only | Rename proposals come from embedded metadata or keyword-template fill — never fabricated. Structurally incapable of generating a name without evidence. |

**Remote inference and LLM fallback are symmetric power-user opt-ins**, not defaults. Each requires the same **three-gate** opt-in: (a) compile with the feature flag (`--features remote` or `--features llm-fallback`), (b) configure (`[inference.remote]` section or `[inference] llm_fallback = true`), and (c) pass the per-invocation flag (`--remote` / `TIDYUP_REMOTE=1` or `--llm-fallback` / `TIDYUP_LLM_FALLBACK=1`). The CLI rejects activation without the matching cargo feature. The two flags are mutually exclusive. First-run and onboarding never recommend either; the tool is designed to be excellent offline with embedding-based classification as the spine.

If you find a privacy claim here that doesn't match the code, that's a bug — please open an issue.

---

## Installing

Tagged releases (`vX.Y.Z`) publish prebuilt `tidyup` CLI binaries for **Linux**
(x86_64), **macOS** (x86_64 + Apple Silicon), and **Windows** (x86_64) to
[GitHub Releases](https://github.com/erovelli/tidyup/releases) — each a
`.tar.gz` / `.zip` archive (binary + `LICENSE` + `README`) with a matching
`.sha256`. The published binary is the **default build**: network-silent and
LLM-silent, byte-for-byte the feature set of `cargo build --release -p
tidyup-cli`. Verify the checksum, extract, and put `tidyup` on your `PATH`. The
~35 MB embedding model is fetched separately (see [Building](#building)) — the
binary has no network code path and cannot download anything itself.

> The desktop UI (`tidyup-desktop`) is built from source for now; signed app
> bundles are tracked in the roadmap.

---

## Building

```bash
# Default release build — fully offline, embedding-only classification.
# No LLM inference, no HTTP client compiled in.
cargo build --release -p tidyup-cli
cargo run   --release -p tidyup-cli -- --help

# Power user: include the optional local LLM fallback backend.
# Enables mistralrs/candle/hf-hub deps. Still requires runtime config + --llm-fallback flag.
cargo build --release -p tidyup-cli --features llm-fallback

# With hardware acceleration for the optional LLM fallback
cargo build --release -p tidyup-cli --features llm-metal   # macOS
cargo build --release -p tidyup-cli --features llm-cuda    # NVIDIA

# Power user: include the optional remote inference backend.
# Enables HTTP client deps (reqwest, rustls). Still requires runtime config + --remote flag.
cargo build --release -p tidyup-cli --features remote
```

Rust pinned to 1.95 via `rust-toolchain.toml`. The default-binary embedding model (~35 MB) is fetched out-of-band by `cargo xtask download-models` — the release binary itself has no HTTP client and cannot download anything. Packagers are expected to bundle the model alongside the binary; developers run the xtask once. Model cache: `dirs::cache_dir()/tidyup/models/` (overridable via `TIDYUP_MODEL_CACHE`).

**Multimodal model bundles (optional, Phase 7).** Specialized image and audio
classification needs the SigLIP and CLAP ONNX bundles — neither ships by
default because each adds several hundred MB to the on-disk install (SigLIP ~370 MB, CLAP ~600 MB).

```bash
# Just the SigLIP image encoder (~370 MB).
cargo xtask download-models --siglip
# Just the CLAP audio encoder (~600 MB).
cargo xtask download-models --clap
# Both at once.
cargo xtask download-models --multimodal
```

The CLI and UI binaries detect the bundles at startup and enable the
modality-specific embedding path automatically when present. Absent bundles
are not an error — image/audio files fall back to the general text-embedding
path using their filename, local path context, and extracted EXIF/ID3 text.

**macOS screenshot OCR.** The default image feature can compile a tiny local
Apple Vision helper when `xcrun`, Swift 5, and the macOS 14 SDK are available.
If that toolchain is missing or incompatible, the build emits a warning and
continues without OCR. At runtime OCR is limited to plausible screenshot names
under `[extraction] ocr_max_bytes`; ordinary photos never launch the helper.

**Integrity.** `download-models` verifies each file after fetching against the
same `BundleSpec`/`ArtifactSpec` the binary checks at load time (one source of
truth in `tidyup-embeddings-ort::install`, so downloader and runtime can't
drift). A pinned BLAKE3 checksum is enforced — a corrupt download is deleted and
the run fails — and unpinned artifacts print their digest + size so a maintainer
can pin them (ideally against an immutable `resolve/<commit-sha>/` revision).
`cargo xtask verify-models [--siglip --clap | --multimodal]` re-checks an
existing install on demand.

---

## Quickstart

Nothing moves without your approval — every run proposes changes you review first. Start with a dry run:

```bash
# See what tidyup would do to a messy folder against the built-in taxonomy — no changes made.
tidyup scan ~/Downloads --dry-run

# Drop --dry-run to generate proposals and review them interactively (approve / reject).
tidyup scan ~/Downloads

# Sort a source folder into an existing target hierarchy whose structure tidyup learns.
tidyup migrate ~/Inbox ~/Documents --dry-run

# Advisory watch: re-scan and report what it *would* propose on each change. Never moves anything.
tidyup watch ~/Downloads

# Inspect state + recent runs; list and reverse a run (the run id prints in each run summary).
tidyup status
tidyup rollback --list
tidyup rollback <run-id>

# Expire shelved backups past the retention window; print the resolved config + its path.
tidyup prune --days 30
tidyup config
```

Flags: `--yes`, `--json`, `--llm-fallback`, and `--remote` are **global** (accepted in any position); `--dry-run` (propose only) is a per-command flag on `scan` and `migrate`. `--yes` auto-approves *moves* above the confidence threshold but never renames — it auto-rejects them, so approving a rename needs an interactive run. `--json` emits machine-readable events for scripting; `--llm-fallback` / `--remote` activate optional reranking backends (see [Privacy guarantees](#privacy-guarantees)). The loop is always **dry-run → review → apply → reversible**.

---

## Configuration

Config is layered: built-in defaults → a TOML file → a few environment overrides. `tidyup config` prints the resolved config and the file path. The file lives at the platform config path — `~/.config/tidyup/config.toml` (Linux), `~/Library/Application Support/tidyup/config.toml` (macOS), `%APPDATA%\tidyup\config.toml` (Windows) — or wherever `TIDYUP_CONFIG_PATH` points. Every section is optional; omitted keys fall back to the defaults shown here, and unknown keys are rejected at load so typos surface immediately.

```toml
[storage]
# data_dir = "/custom/path"            # default: platform data dir (see "Where tidyup stores things")
backup_retention_days = 30             # shelved originals older than this are eligible for `prune`

[classifier]
tiers = ["embeddings"]                 # compatibility field; only semantic embeddings are currently used
min_confidence = 0.75                  # --yes auto-approve threshold for moves; does not change classification/review thresholds

[inference]
backends = ["embeddings-ort"]          # reserved: parsed for forward-compat but not yet consulted; the
                                       # context builder selects a fallback via llm_fallback / [inference.remote]
llm_fallback = false                   # gate (b) for LLM reranking; still needs the feature + flag

# [inference.remote]                    # only consulted under --features remote + --remote / TIDYUP_REMOTE=1
# endpoint = "https://api.openai.com/v1"
# api_key_env = "OPENAI_API_KEY"        # name of the env var holding the key (the key is never written to disk)
# model = "gpt-4o-mini"

[inference.embedding]
model_id = "bge-small-en-v1.5"         # default on-device embedding model

[extraction]
ocr_enabled = true                     # local Apple Vision OCR for plausible screenshots on supported macOS builds
ocr_max_bytes = 20971520               # 20 MiB cap for a whole image passed to Vision

[rename]
min_classification_confidence = 0.85   # both thresholds must clear before a rename is proposed
min_mismatch_score = 0.60              # 1.0 - cosine(embed(filename), content_embedding)

[bundle_detection]
enabled = true
extra_markers = []                     # extra directory-bundle marker filenames, e.g. ["deno.json", "flake.nix"]
soft_bundle_enabled = true             # metadata clusters: EXIF photo bursts, ID3 albums, filename series
```

`classifier.tiers` is retained so old config files continue to parse; the current classifier always uses semantic embeddings and ignores unknown/removed tier ids. The OCR controls are live in both CLI and desktop service construction. The `[bundle_detection]` fields are parsed but not yet wired into the pipeline.

---

## Environment variables

| Variable | Effect |
|---|---|
| `TIDYUP_CONFIG_PATH` | Override the config-file location. |
| `TIDYUP_DATA_DIR` | Override the data root (SQLite DB + backup shelf). |
| `TIDYUP_MODEL_CACHE` | Override the model-cache directory (ONNX bundles + taxonomy cache). |
| `TIDYUP_LLM_FALLBACK=1` | Per-invocation activation of optional LLM reranking (same as `--llm-fallback`). In the desktop UI it pre-arms the session toggle only when the cargo feature and config gate also permit it. Never persisted. |
| `TIDYUP_REMOTE=1` | Per-invocation activation of the remote backend (same as `--remote`). Never persisted. |
| `RUST_LOG` | Standard `tracing`/`EnvFilter` log verbosity, e.g. `RUST_LOG=tidyup=debug` (defaults to `info`). |

The two activation gates are evaluated per run and never written back into the config file — config describes *available* capabilities; the flag/env decides whether to use them this run.

---

## Where tidyup stores things

| What | Location |
|---|---|
| Config file | `~/.config/tidyup/config.toml` (Linux) · `~/Library/Application Support/tidyup/config.toml` (macOS) · `%APPDATA%\tidyup\config.toml` (Windows) — or `TIDYUP_CONFIG_PATH` |
| Data root | `<platform data dir>/tidyup/` — or `TIDYUP_DATA_DIR` |
| ↳ SQLite index | `<data root>/tidyup.db` |
| ↳ Backup shelf | `<data root>/backup/` (originals, restorable via `rollback`) |
| Model cache | `<platform cache dir>/tidyup/models/` — or `TIDYUP_MODEL_CACHE` |

The model cache sits under the OS *cache* dir (`~/.cache/tidyup/models/` on Linux, `~/Library/Caches/tidyup/models/` on macOS, `%LOCALAPPDATA%\tidyup\models\` on Windows), separate from the data root — so clearing caches never touches your index or backups.

---

## Troubleshooting

- **"Missing embedding model" on first run.** The default binary ships without the model. Fetch it with `cargo xtask download-models` (or place the `bge-small-en-v1.5` files under the model cache), then re-run. The binary has no network path and never downloads anything itself.
- **Confirm the model is installed.** `tidyup status` reports embedding-model presence (`--json` for scripts).
- **"platform cache directory unavailable".** Set `TIDYUP_MODEL_CACHE` to an explicit directory.
- **Image/audio files need stronger content classification.** Install the optional SigLIP/CLAP bundles with `cargo xtask download-models --multimodal`. Without them, media uses the general text-embedding path over filenames and extracted metadata; weak results are held for review.
- **Screenshot OCR is unavailable on macOS.** Install Xcode command-line tools with Swift 5 and the macOS 14 SDK, then rebuild. OCR is optional: a missing/incompatible toolchain produces a build warning, not a failure. It can also be disabled with `[extraction] ocr_enabled = false`.

---

## Roadmap

tidyup is being built in phases. Each phase lands an independently compilable slice of the hexagonal architecture. The CLI binary exists from Phase 0, but doesn't become end-to-end runnable until Phase 5.

| Phase | Scope                                                                                       | Status         |
| ----- | ------------------------------------------------------------------------------------------- | -------------- |
| 0     | Workspace scaffold, port traits, CI, lints, `deny.toml`, `xtask`                            | [x] Complete   |
| 1     | Domain types, SQLite storage, BLAKE3 indexer, layered config, `BundleProposal` aggregate    | [x] Complete   |
| 2     | Content extractors: router + MIME detection, plain text, PDF, Excel, image, audio           | [x] Complete   |
| 3     | Inference: `bge-small-en-v1.5` via ONNX Runtime (default); optional LLM + remote backends   | [x] Complete   |
| 4     | Pipeline: semantic routing, bundle detection, scan + migration classifiers, rename cascade | [x] Complete   |
| 5     | CLI wiring, apply + rollback, first-run model check, end-to-end flows                       | [x] Complete   |
| 6     | Dioxus desktop UI (dashboard, review, runs, settings) on the same service seam              | [x] Complete   |
| 7     | Multimodal encoders (SigLIP image / CLAP audio) wired into scan and migration | [x] Complete   |
| 8+    | Video keyframe encoder, code signing, package-manager distribution (Homebrew/winget), UI app bundles | [ ] Backlog    |

**What currently works:**

- `cargo build --release -p tidyup-cli` produces a fully functional default binary: `tidyup migrate`, `tidyup scan`, `tidyup watch`, `tidyup rollback`, `tidyup prune`, `tidyup status`, `tidyup config`
- `tidyup prune [--days N]` expires shelved backups past the retention window (default `[storage] backup_retention_days`, 30) and removes them from the shelf; `tidyup status` summarizes the data dir, embedding-model presence, retention setting, and recent runs (both support `--json`)
- `tidyup watch <dir>` is an **advisory** filesystem watch: it re-scans (dry-run) on each debounced change and reports what it *would* propose, loading the model once and reusing it across rescans. It never moves anything — run `tidyup scan` to apply (editor/temp churn is filtered so a single save doesn't storm rescans)
- `cargo xtask ci` is green: privacy check + `fmt` + `clippy --all-features -D warnings` + workspace tests
- `cargo xtask check-privacy` asserts the default dep graph contains no `reqwest`/`hyper`/`rustls`/`mistralrs`/`candle-core`/`hf-hub`
- SQLite storage: `FileIndex`, `ChangeLog`, `BackupStore`, `RunLog` with bundle-atomic shelving
- Layered TOML config with platform-aware paths
- `tidyup-extract`: MIME detection + router + `PlainTextExtractor` + `PdfExtractor` + `ExcelExtractor` + `ImageExtractor` (dimensions, EXIF, and bounded/configurable macOS Vision OCR for plausible screenshots) + `AudioExtractor` (ID3/Vorbis tags), each behind its own cargo feature. Missing Swift/Xcode support degrades to no OCR instead of failing the build.
- `tidyup-embeddings-ort`: `bge-small-en-v1.5` ONNX classifier, taxonomy cache (BLAKE3-invalidated), custom-taxonomy loader (`scan --taxonomy file.toml`, validated `[[entry]]` tables), model-install verifier
- `tidyup-inference-mistralrs` (opt-in `--features llm-fallback`): `TextBackend` + lazy `VisionBackend` via `mistralrs`; Metal/CUDA pass-through features
- `tidyup-inference-remote` (opt-in `--features remote`): `TextBackend` over OpenAI-compatible endpoints. Anthropic and Ollama endpoint variants exist in the crate but are not yet selectable from CLI config — only the OpenAI-compatible path is wired today
- `tidyup-pipeline`: semantic embedding routing, directory bundle detection (Cargo/npm/pyproject/Gradle/Xcode/.git/Jupyter), and atomic **content-cluster detection** (photo bursts by EXIF time, music albums by ID3 album tag, document series by filename family, and cross-format semantic collections by descriptive entity stem plus optional overlapping screenshot text). Numeric/date-only and generic screenshot prefixes cannot anchor semantic collections; filename tokenization and screenshot extraction are each performed once per clustering pass. The pipeline also provides target-tree name+centroid profiles, inline n-gram YAKE keyphrase extraction (language-aware stopwords: EN/ES/FR/DE), the extractive rename cascade, optional LLM reranking, and scan/migration classifiers.
- **Optional LLM fallback (off-by-default)**: when embedding routing lands in the review zone, a `TextBackend` can re-classify the content; `summary + category + tags` is re-embedded and re-ranked against the same candidates, and adopted only if its score improves. Activation is triple-gated by cargo feature, config, and invocation flag. The LLM's `suggested_name` is ignored.
- `tidyup-app`: `ScanService`, `MigrationService`, and `RollbackService` driving the pipeline end-to-end — shelve → move → mark applied → per-run rollback via the `RunLog`. Bundles are decided through the `ReviewHandler::review_bundles` seam — `--yes` auto-applies move-only bundles at or above `0.50` raw semantic cosine, while bundles containing member renames require explicit review — and move all-or-nothing. Frontend-edited semantic labels/filenames are reconciled against the original immutable member ids, source paths, hashes, and destination parent before any shelf or move operation.
- First-run model check: scan/migrate surface `cargo xtask download-models` (or a manual placement hint) when the embedding bundle is missing, without linking an HTTP client
- `tidyup-ui`: Dioxus 0.7 desktop binary (`cargo run -p tidyup-ui --bin tidyup-desktop`) with Dashboard / Review / Runs / Settings pages, signal-backed `ProgressReporter` and oneshot-channel `ReviewHandler`. Same `ServiceContext` construction, extractors, and embedding model as the CLI — the only difference is the frontend port impls. **Bundle review is interactive**: the Review page surfaces each detected bundle for atomic approve/reject through the same `ReviewHandler::review_bundles` seam the CLI uses (a second review pass after loose proposals). Styled per `DESIGN.md` ("The Verdant Archive")
- **Phase 7 multimodal embeddings (optional, off-by-default)**: SigLIP-base for cross-modal image classification and CLAP-htsat-unfused for audio. Both load only when their model bundles exist. Without them, image/audio metadata and names use the general text-embedding route and uncertain results are reviewed. Fetch them with `cargo xtask download-models --multimodal`. Scan ranks against per-modality taxonomies; migration ranks against per-folder centroids in isolated latent spaces.
- **Model-integrity verification**: `cargo xtask download-models` and the runtime loader share one `BundleSpec` source of truth (`tidyup-embeddings-ort::install`); downloads are checksum-verified (pinned BLAKE3 enforced and a corrupt file deleted; unpinned digests reported so they can be pinned), and `cargo xtask verify-models` checks an install against those specs on demand
- **Classification eval harness** (`cargo xtask eval`): a labeled golden corpus plus accuracy, per-label precision/recall/F1, coverage, and confusions (`--json`). Classification entries require the `bge-small-en-v1.5` bundle; without it they are reported as deferred. `cargo xtask eval --calibrate` fits a Platt calibrator and reports Expected Calibration Error. The harness stays out of model-free `cargo xtask ci`.
- **Held-out routing eval** (`cargo xtask eval-routing <corpus>`): the *falsifiable* test of the core premise "route by contents, not filename." It treats an already-organized directory as ground truth (folder = label), holds out files, routes them with the real embedding backend (the migration centroid-cosine rule), and reports top-1/top-3 with bootstrap 95% CIs against three baselines — **filename-embedding** (the one it must beat), most-frequent, and extension — plus the content−filename delta and a PASS/FAIL verdict. The split/metrics/baselines are unit-tested deterministically (a stub backend proves the instrument without the model). The **`model-eval` nightly lane** provisions libonnxruntime + the bundle and runs both `eval` and `eval-routing` on real 20-Newsgroups data, gated so a broken premise fails the lane — the only CI lane that exercises the real model path

**What does not yet work:**

- ~~Migration-mode multimodal.~~ **Now shipped:** the migration profiler builds per-folder centroids in three latent spaces — text `content_centroid` (from each folder's documents, always), plus image/audio centroids when the SigLIP/CLAP bundles are installed — and routes each source file against the centroid in its own space, falling back to folder-name embeddings when a folder lacks the matching centroid.
- Video keyframe encoder. Video files have no dedicated content encoder; they can only use extractable text/name context and otherwise remain low-confidence or unclassified. Pure-Rust frame extraction is gated on the `ffmpeg-next` FFI vs metadata-only decision.
- Calibrated confidence **by default**. The calibration mechanism now exists — Platt scaling via `Calibration` (default `Identity`), fit with `cargo xtask eval --calibrate`, measured by Expected Calibration Error — but the shipped default is still raw weighted-cosine. Enabling a fitted default needs the embedding model plus a held-out corpus larger than the current fixture set.
- ~~UI LLM fallback toggle.~~ **Now shipped:** Settings surfaces a per-session toggle under the same three-gate model. `TIDYUP_LLM_FALLBACK=1` pre-arms it only when the feature and config gates are also active; config loading happens asynchronously after root construction. The UI deliberately does not expose remote inference, and `cargo xtask check-privacy` asserts the default UI graph is LLM-silent.
- ~~Prebuilt binary releases.~~ **Now shipped:** `release.yml` builds the default `tidyup` CLI for Linux/macOS(x2)/Windows on a `vX.Y.Z` tag and uploads checksummed archives to GitHub Releases (default features only — the published binary stays network- and LLM-silent). Still to come: **signed** binaries and Homebrew/winget package-manager distribution.

The invariants the finished tool will uphold — human-in-the-loop review, reversible moves, bundle atomicity, no-network-by-default, extractive-only renames — are now enforced at the code path, not just the design.

---

## Contributing

This is primarily a personal project, but issues, discussions, and PRs are welcome. If you're considering a substantial change, please open an issue first so we can talk about fit.

See `CONTRIBUTING.md` for setup and style guidelines.

---

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).

Apache-2.0 is a permissive license: you can use, modify, and redistribute tidyup — including in commercial and closed-source projects — provided you preserve the copyright notice (and the `NOTICE` file, if present). It also includes an explicit patent grant from contributors, protecting you and downstream users.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in tidyup shall be licensed as above, without any additional terms or conditions.
