<p align="center">
  <img src="assets/brand/logo.svg" alt="tidyup" width="440">
</p>

> A local-first file organizer that never phones home.

**tidyup** watches a directory, understands what's in your files using compact embedding models running _entirely on your machine_, and proposes a tidier structure. You review every change before anything moves. Nothing is uploaded. Operational state is recorded only in the local SQLite database so runs can be reviewed and rolled back. No account. No cloud. No telemetry.

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
> gather review decisions. Optional cross-modal embeddings add local image
> (SigLIP) and audio (CLAP) understanding in scan and migration mode; both load when their ONNX
> bundles are present. The optional LLM fallback is wired through the
> same pipeline seam — opt in with `--features llm-fallback` (or
> `--features remote`) plus the matching config + flag activation, and
> low-confidence embedding verdicts get a second-opinion re-rank. Interactive
> bundle review now works in **both the CLI and the desktop UI** — each detected
> bundle gets an atomic approve/reject instead of silently staying pending.
> Migration mode gives every target folder modality-specific prototypes derived
> from its label and path, then adds content centroids when files already live
> there. Empty named folders therefore remain useful candidates while populated
> hierarchies adapt to the user's actual contents. The desktop UI has a
> **Settings → LLM fallback toggle** that
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

- **Deterministic, offline classification.** The default binary classifies by embedding similarity via `bge-small-en-v1.5` on ONNX Runtime — entirely local and without LLM generation. With the same verified model, preprocessing version, runtime, and hardware path, the scoring flow is repeatable and auditable.
- **LLMs are an optional escape hatch, not a default.** The default binary contains no LLM inference — `mistralrs`/`candle`/`hf-hub` are not loaded, not linked, not reachable. A local LLM fallback is available as a compile-time + runtime opt-in (`--features llm-fallback`) for sparse learned hierarchies and pathological extraction. A truly empty migration target has no candidates for any backend to rank: bundles are preserved at its root at zero confidence for review, and loose files are reported unclassified.
- **No network calls by default.** The default binary has no HTTP client — not loaded, not linked, not reachable. Airplane-mode your machine and it still works. A remote backend is available as a symmetric compile-time + runtime opt-in (`--features remote`) for power users.
- **Groupings stay grouped.** A coding project, photo burst, or music album moves as an atomic bundle or not at all — tidyup will never fragment one.
- **Content-aware renames, but never silent.** When a filename clearly disagrees with the contents, tidyup proposes a grounded rename from exact metadata, local OCR text, keyphrases, or scored non-textual concepts retrieved by a local contrastive model. Textual sources clear the two configurable confidence/mismatch gates; visual concepts additionally clear grounding and ambiguity gates. Renames always require explicit approval. Colliding names inside a semantic collection are suffixed deterministically and duplicate manual edits are rejected inline.
- **No telemetry.** No analytics, no crash reporting, no "anonymous usage data."
- **Human-in-the-loop, always.** Every rename and move is a _proposal_. Nothing touches your filesystem until you approve it; `--yes` is explicit approval only for high-confidence loose moves and recognized opaque structural bundles. Soft/file-set bundles and every rename remain in review.
- **Every change is reversible while its shelf copy is retained.** Originals are copied to a backup shelf before any move. The default retention window is 30 days and `prune` removes expired shelves. Rollback is edit-safe: before deleting a moved file it verifies the shelf copy is intact (content hash) and that the destination still matches what was moved — a file you've edited since apply is reported as a **conflict** and left in place rather than overwritten, and a new file occupying the original location is never clobbered.

This is a portfolio project and a personal tool. It is also a statement: useful AI does not require surrendering your data.

---

## What it does

- **Indexes identities with BLAKE3** and logs proposals, moves, backups, runtime capabilities, and per-file processing outcomes to a local SQLite database. Read/hash failures remain visible with a stage and reason instead of disappearing from counts. Stage transitions are transaction-batched. Image/audio semantic artifacts are cached as compact `f32` blobs by content hash + model/preprocessing version across runs and expire through `tidyup prune`; full text-classification fan-out for duplicate documents remains planned.
- **Detects logical groupings first without hiding nested files.** The scanner descends through ordinary directories, so loose descendants are classified and projects nested at any depth are detected at their own marker roots. Coding projects, photo bursts, music albums, Jupyter notebook sets, document series, and conservative visual-neighbor collections move as atomic units. A coding project is never shredded; EXIF bursts outrank visual-neighbor grouping; a visual collection requires at least three members, all-pairs embedding cohesion, and a grounded semantic label.
- **Classifies each loose file semantically.** The default path embeds filename/context and extracted content with `bge-small-en-v1.5`, then ranks the fixed scan taxonomy or learned migration profiles by cosine similarity. There is no extension/keyword destination router. An optional local LLM fallback (1–10s, `--features llm-fallback`) can rerank uncertain embedding results; its `suggested_name` is ignored. Default builds exclude the LLM entirely and surface low-confidence files for review.
- **Proposes a destination folder** — with a plain-English reason.
- **Proposes a grounded rename** when filename and contents disagree. Text uses metadata/OCR/keyphrases; non-textual images can retrieve scored visible concepts such as `cat` + `beach` without an LLM. Destination, grounding, ambiguity, and mismatch gates must pass. Renames never auto-apply.
- **Shows you a diff-style review UI** — approve or reject per file or per bundle.
- **Backs up originals before moving** — restore while the shelf copy remains inside its retention window.

Two modes:

- **Scan mode** — organize a messy directory against a built-in taxonomy, or your own via `scan --taxonomy <file.toml>`.
- **Migration mode** — sort a source directory into an _existing_ target hierarchy whose structure tidyup learns. Modality-specific text prototypes let even an empty named folder such as `Photos/Cats` attract matching content; populated folders add content centroids as learned usage evidence.

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
| Grounded renames only | Rename terms come from exact metadata/text or scored concepts retrieved in a local contrastive-model space. No free-form LLM/VLM filename generation. |

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

**Multimodal model bundles (optional).** Specialized image and audio
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
modality-specific embedding path automatically when present. SigLIP enables
grounded visual renames and complete-link visual collections in addition to
placement. Absent bundles are not an error—image/audio files fall back to the
general text-embedding path using their filename, local path context, and
extracted EXIF/ID3 text.

Measure the warm image path on representative hardware with a real local
fixture. The command reads, hashes, embeds, and concept-ranks the image,
reports cold model load separately, and fails when warm p95 exceeds one second:

```bash
cargo xtask bench-semantic path/to/representative.png
```

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

# Expire shelved backups and semantic cache entries; print the resolved config + its path.
tidyup prune --days 30
tidyup config
```

Flags: `--yes`, `--json`, `--llm-fallback`, and `--remote` are **global** (accepted in any position); `--dry-run` (propose only) is a per-command flag on `scan` and `migrate`. A dry-run reports successful dispositions as “would apply” while leaving source/destination files, shelves, and applied/rejected states untouched; run provenance and the proposed plan are still recorded locally for diagnostics. `--yes` auto-approves loose move-only proposals above `[classifier] min_confidence` and recognized opaque structural bundles above a separate internal `0.50` raw-cosine floor. Soft/file-set collections remain explicit-review only until action-specific confidence is calibrated; any rename also requires review. `--json` emits machine-readable events for scripting; `--llm-fallback` / `--remote` activate optional reranking backends (see [Privacy guarantees](#privacy-guarantees)). The loop is always **dry-run → review → apply → reversible**.

---

## Configuration

Config is layered: built-in defaults → a TOML file → a few environment overrides. `tidyup config` prints the resolved config and the file path. The file lives at the platform config path — `~/.config/tidyup/config.toml` (Linux), `~/Library/Application Support/tidyup/config.toml` (macOS), `%APPDATA%\tidyup\config.toml` (Windows) — or wherever `TIDYUP_CONFIG_PATH` points. Every section is optional; omitted keys fall back to the defaults shown here, and unknown keys are rejected at load so typos surface immediately.

```toml
[storage]
# data_dir = "/custom/path"            # default: platform data dir (see "Where tidyup stores things")
backup_retention_days = 30             # shelved originals older than this are eligible for `prune`

[classifier]
tiers = ["embeddings"]                 # compatibility field; only semantic embeddings are currently used
min_confidence = 0.75                  # --yes threshold for loose move-only proposals; renames remain review-only

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
min_grounded_mismatch = 0.60           # fraction of selected visual concepts absent from filename tokens
min_grounding_confidence = 0.30        # raw contrastive score floor for non-textual concepts
min_grounding_gap = 0.02               # selected concept must separate from the next candidate

[bundle_detection]
enabled = true
extra_markers = []                     # extra directory-bundle marker filenames, e.g. ["deno.json", "flake.nix"]
soft_bundle_enabled = true             # metadata clusters: EXIF photo bursts, ID3 albums, filename series
```

`classifier.tiers` is retained so old config files continue to parse; the current classifier always uses semantic embeddings and ignores unknown/removed values. Opaque recognized structural bundles use a separate internal raw-cosine auto-approval floor of `0.50`; file-set collections and every rename remain review-only. The OCR controls are live in both CLI and desktop service construction. The `[bundle_detection]` fields are parsed but not yet wired into the pipeline.

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

The activation gates are evaluated per run and never written back into the config file. Config permits an optional backend, the flag/env requests it for this invocation, and the persisted capability manifest records what was actually loaded or unavailable.

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
| 1     | Domain types, SQLite storage, port-neutral BLAKE3 identity indexer, layered config, `BundleProposal` aggregate | [x] Complete   |
| 2     | Content extractors: router + MIME detection, plain text, PDF, Excel, image, audio           | [x] Complete   |
| 3     | Inference: `bge-small-en-v1.5` via ONNX Runtime (default); optional LLM + remote backends   | [x] Complete   |
| 4     | Pipeline: semantic routing, bundle detection, scan + migration classifiers, rename cascade | [x] Complete   |
| 5     | CLI wiring, apply + rollback, first-run model check, end-to-end flows                       | [x] Complete   |
| 6     | Dioxus desktop UI (dashboard, review, runs, settings) on the same service seam              | [x] Complete   |
| 7     | Multimodal encoders (SigLIP image / CLAP audio) wired into scan and migration | [x] Complete   |
| 8     | General semantic spine: dynamic modality folder prototypes, grounded visual naming/clustering, versioned artifact cache, durable run accounting, capability manifests, latency harness | [~] Implemented; instrumentation and calibration remain |
| 9+    | Video keyframe encoder, text-result dedup, code signing, package-manager distribution, UI app bundles | [ ] Backlog    |

**What currently works:**

- `cargo build --release -p tidyup-cli` produces a fully functional default binary: `tidyup migrate`, `tidyup scan`, `tidyup watch`, `tidyup rollback`, `tidyup prune`, `tidyup status`, `tidyup config`
- `tidyup prune [--days N]` expires shelved backups and reconstructible semantic-cache artifacts past the retention window (default `[storage] backup_retention_days`, 30); `tidyup status` summarizes the data dir, embedding-model presence, retention setting, and recent runs (both support `--json`)
- `tidyup watch <dir>` is an **advisory** filesystem watch: it re-scans (dry-run) on each debounced change and reports what it *would* propose, loading the model once and reusing it across rescans. It never moves anything — run `tidyup scan` to apply (editor/temp churn is filtered so a single save doesn't storm rescans)
- `cargo xtask ci` is green: privacy check + formatting + a layered clippy matrix + workspace/default and selected feature tests. Accelerator features are intentionally excluded from the portable local matrix.
- `cargo xtask check-privacy` asserts the default dep graph contains no `reqwest`/`hyper`/`rustls`/`mistralrs`/`candle-core`/`hf-hub`
- SQLite storage: `FileIndex`, `ChangeLog`, `BackupStore`, `RunLog` with bundle-atomic shelving
- Durable run provenance: every regular file the walker can enumerate receives a source/target-profile processing state (and a failure reason when applicable), while each run stores the exact loaded/disabled model, extractor, OCR, and optional LLM capabilities. Identity indexing runs before semantic work, and its `FileId` and BLAKE3 are reused by the proposals it produces, so a file is hashed once per run and the run's processing records and change proposals join on a stable id rather than on a path a move is about to change
- `tidyup scan` reorganizes **in place** and only proposes what is out of place: a file already sitting at its classified destination is recorded as classified with no move proposed, and reported separately as "already in place". Runs also report when a directory exceeded the clustering work cap, since those images are classified individually and never offered for collection grouping
- Layered TOML config with platform-aware paths
- `tidyup-extract`: MIME detection + router + `PlainTextExtractor` + `PdfExtractor` + `ImageExtractor` (dimensions, EXIF, and bounded/configurable macOS Vision OCR for plausible screenshots) + `AudioExtractor` (ID3/Vorbis tags) in the shipped CLI/UI stack. An `ExcelExtractor` for Excel/ODS is available behind the crate's non-default `excel` feature, but the current frontend context builders do not register it. Missing Swift/Xcode support degrades to no OCR instead of failing the build.
- `tidyup-embeddings-ort`: `bge-small-en-v1.5` ONNX classifier, taxonomy cache (BLAKE3-invalidated), custom-taxonomy loader (`scan --taxonomy file.toml`, validated `[[entry]]` tables), model-install verifier
- `tidyup-inference-mistralrs` (opt-in `--features llm-fallback`): `TextBackend` + lazy `VisionBackend` via `mistralrs`; Metal/CUDA pass-through features
- `tidyup-inference-remote` (opt-in `--features remote`): `TextBackend` over OpenAI-compatible endpoints. Anthropic and Ollama endpoint variants exist in the crate but are not yet selectable from CLI config — only the OpenAI-compatible path is wired today
- `tidyup-pipeline`: recursive loose-file discovery plus marker-root bundle detection (Cargo/npm/pyproject/Gradle/Xcode/.git/Jupyter), semantic embedding routing, and atomic **content-cluster detection** (photo bursts by EXIF time, music albums by ID3 album tag, document series by filename family, descriptive cross-format collections, and complete-link visual-neighbor collections). Unmarked directories stay transparent; nested marker roots remain opaque. Numeric/date-only and generic screenshot prefixes cannot anchor semantic collections. A shared per-run cache prevents duplicate extraction/model calls across clustering, placement, and naming. The pipeline also provides dynamic target-folder prototypes plus content centroids, inline n-gram YAKE keyphrase extraction, grounded multimodal rename retrieval, optional LLM reranking, and scan/migration classifiers.
- **Optional LLM fallback (off-by-default)**: when embedding routing lands in the review zone, a `TextBackend` can re-classify the content; `summary + category + tags` is re-embedded and re-ranked against the same candidates, and adopted only if its score improves. Activation is triple-gated by cargo feature, config, and invocation flag. The LLM's `suggested_name` is ignored.
- `tidyup-app`: `ScanService`, `MigrationService`, and `RollbackService` driving the pipeline end-to-end — shelve → write-ahead `Applied` journal mark → move → per-run rollback via the `RunLog`. Interactive frontends receive the complete plan through `ReviewHandler::review_all`; `--yes` separately auto-applies loose move-only proposals above `[classifier] min_confidence` and recognized opaque structural bundles above their internal `0.50` raw-cosine floor. Soft/file-set collections and all renames remain explicit-review only. All bundle moves remain all-or-nothing. Frontend-edited semantic labels/filenames are reconciled against the original immutable member ids, source paths, hashes, and destination parent before any shelf or move operation.
- First-run model check: scan/migrate surface `cargo xtask download-models` (or a manual placement hint) when the embedding bundle is missing, without linking an HTTP client
- `tidyup-ui`: Dioxus 0.7 desktop binary (`cargo run --release -p tidyup-ui --bin tidyup-desktop` — the stylesheet is compiled into the binary, so no `dx` CLI or asset bundling step is needed; `--release` matters because debug-profile inference is unusably slow) with Dashboard / Review / Runs / Settings pages, signal-backed `ProgressReporter` and oneshot-channel `ReviewHandler`. Settings opens with an **About** card reporting the version, the git revision the binary was built from (with a `-dirty` marker for uncommitted trees), whether the embedding model is actually present on disk, and whether the optional LLM reranker was compiled in — the four facts a pre-alpha bug report needs. It renders even when config loading fails. Dashboard Scan and Migrate default to **Preview only (dry run)**, reporting what would apply without changing files, shelves, or proposal state; execution requires explicitly turning preview off. Raw routing evidence is displayed as a similarity score, not a percentage, until a fitted calibrator compatible with the run’s capability manifest ships. It uses the same services, extractor registry, and embedding models as the CLI. **Review is a complete-plan surface** through `ReviewHandler::review_all`: loose changes and atomic bundles are shown together; semantic collection labels/member basenames can be edited, while duplicate sibling names are rejected inline and immutable bundle identity is revalidated by the executor. Styled per `DESIGN.md` ("The Verdant Archive") with the Manrope/Inter pairing self-hosted — Latin-subset variable faces are embedded in the binary and injected as `data:` URLs, so the design renders without the webview reaching a font CDN. It launches as **Tidyup** with the project mark as its window icon (the `tidyup-desktop` binary name stays a developer-facing handle that disambiguates it from the `tidyup` CLI)
- **Multimodal embeddings (optional, off-by-default)**: SigLIP-base for cross-modal image semantics and CLAP-htsat-unfused for audio. Both load only when their model bundles exist. Scan ranks per-modality taxonomies; migration combines matching-space folder-label prototypes with optional content centroids, so empty named folders remain candidates. SigLIP also grounds visible concepts used for rename suggestions and conservative visual collections. Embeddings persist in SQLite under exact content/model/preprocessing/latent-space keys.
- **Semantic latency harness** (`cargo xtask bench-semantic <image>`): reports cold model load separately from warm read+hash+SigLIP+concept-ranking+grounded-name p50/p95/worst latency and enforces a configurable one-second p95 gate.
- **Model-integrity verification**: `cargo xtask download-models` and the runtime loader share one `BundleSpec` source of truth (`tidyup-embeddings-ort::install`); downloads are checksum-verified (pinned BLAKE3 enforced and a corrupt file deleted; unpinned digests reported so they can be pinned), and `cargo xtask verify-models` checks an install against those specs on demand
- **Classification eval harness** (`cargo xtask eval`): a labeled golden corpus plus accuracy, per-label precision/recall/F1, coverage, and confusions (`--json`). Classification entries require the `bge-small-en-v1.5` bundle; without it they are reported as deferred. `cargo xtask eval --calibrate` fits a Platt calibrator and reports Expected Calibration Error. The harness stays out of model-free `cargo xtask ci`.
- **Held-out routing eval** (`cargo xtask eval-routing <corpus>`): the *falsifiable* test of the core premise "route by contents, not filename." It treats an already-organized directory as ground truth (folder = label), holds out files, routes them with the real embedding backend (the migration centroid-cosine rule), and reports top-1/top-3 with bootstrap 95% CIs against three baselines — **filename-embedding** (the one it must beat), most-frequent, and extension — plus the content−filename delta and a PASS/FAIL verdict. The split/metrics/baselines are unit-tested deterministically (a stub backend proves the instrument without the model). The **`model-eval` nightly lane** provisions libonnxruntime + the bundle and runs both `eval` and `eval-routing` on real 20-Newsgroups data, gated so a broken premise fails the lane — the only CI lane that exercises the real model path

**What does not yet work:**

- **Calibrated confidence by default.** Platt scaling and the fitting command exist, but the shipped `Calibration::Identity` reports raw cosine evidence. Routing, rename, clustering, and auto-approval still need separate held-out calibrators before percentages can be interpreted as probabilities.
- **Production per-stage latency enforcement.** The semantic image benchmark measures a representative warm path, and the processing ledger records lifecycle outcomes, but production runs do not yet persist per-stage spans or enforce cancellable one-second per-file deadlines.
- **Incremental semantic indexing.** Image/audio embeddings persist by exact content/model/preprocessing/latent-space key, but text extraction/OCR evidence and final decisions are not yet fanned out by content hash. Target profiles are rebuilt in full on every migration; `ScanDiff` is not wired into incremental invalidation.
- **Dedicated video semantics.** Video files have no keyframe/audio-content encoder; they use available name/context evidence and otherwise remain low-confidence or unclassified. Decoder selection remains unresolved.
- **Excel/ODS in the shipped frontend stack.** The extractor implementation and tests exist behind `tidyup-extract/excel`, but CLI/UI context construction does not currently register it.
- **Signed/package-manager distribution.** Tagged releases publish checksummed default CLI archives, but signing, notarized desktop bundles, Homebrew, and winget remain backlog items.

The invariants the finished tool will uphold — human-in-the-loop review, reversible moves, bundle atomicity, no-network-by-default, and grounded non-generative renames — are now enforced at the code path, not just the design.

---

## Contributing

This is primarily a personal project, but issues, discussions, and PRs are welcome. If you're considering a substantial change, please open an issue first so we can talk about fit.

See `CONTRIBUTING.md` for setup and style guidelines.

---

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).

Apache-2.0 is a permissive license: you can use, modify, and redistribute tidyup — including in commercial and closed-source projects — provided you preserve the copyright notice (and the `NOTICE` file, if present). It also includes an explicit patent grant from contributors, protecting you and downstream users.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in tidyup shall be licensed as above, without any additional terms or conditions.
