# Classification

Design doc for how tidyup decides where a file belongs, whether it should be renamed, and whether it is part of a larger group. Complements `ARCHITECTURE.md` (crate graph, port traits, privacy model) by fleshing out what sits *behind* the `Classifier` port.

## Contents

- [The question this doc answers](#the-question-this-doc-answers)
- [Decision: three-tier cascade](#decision-three-tier-cascade-embeddings-at-the-spine)
- [The unified-pattern property](#the-unified-pattern-property)
- [v0.1 scope: text by default, image/audio opt-in](#v01-scope-text-by-default-imageaudio-opt-in)
- [Tier 2: how embedding classification works](#tier-2-how-embedding-classification-works)
- [Confidence: raw cosine by default, calibration available](#confidence-raw-cosine-by-default-calibration-available)
- [Bundle detection](#bundle-detection)
- [Rename strategy: extractive cascade](#rename-strategy-extractive-cascade)
- [Why embeddings over a local LLM on the default path](#why-embeddings-over-a-local-llm-on-the-default-path)
- [What embeddings give up — honestly](#what-embeddings-give-up--honestly)
- [Tier 3: the LLM-fallback escape hatch](#tier-3-the-llm-fallback-escape-hatch)
- [Architectural implications](#architectural-implications)
- [Open questions](#open-questions)
- [References in repo](#references-in-repo)

## The question this doc answers

A messy source directory contains text documents, images, audio, video, and mixed bundles. For each file, the pipeline must produce:

1. A **target folder** under the preexisting destination hierarchy.
2. A **confidence score** that drives the rename gate and the `--yes` auto-approve threshold.
3. An optional **rename suggestion** when filename and content disagree.
4. For soft bundles (photo bursts, albums, document series), a **grouping decision** that treats related files as a single atomic move.

All of this must run **locally, deterministically, reversibly, and fast enough for interactive review** — per the four product promises in `CLAUDE.md`. What sits behind the `Classifier` port to deliver this with one unified pattern?

## Decision: three-tier cascade, embeddings at the spine

tidyup classifies via a three-tier cascade, each tier cheaper than the last, short-circuiting on confidence. The default-binary spine is non-LLM: deterministic heuristics plus embedding similarity carry the classification load. The LLM is an optional escape hatch, never a default.

1. **Tier 1 — Heuristics (~1 ms).** Extension, MIME, marker-file matching, simple keyword rules. Handles the obvious cases (`.gitignore`, `Cargo.toml`, `*.env`, MIME-obvious media types) for free. Files with unambiguous category by extension never reach Tier 2.
2. **Tier 2 — Embedding similarity (~50 ms).** Content embedded to a vector; scored against target folder profiles. The UES spine. Default Tier 2.
3. **Tier 3 — Local LLM fallback (optional, opt-in, ~1–10 s).** Feature-gated under `--features llm-fallback`, off by default. When compiled in and enabled per-invocation, files that fall below Tier 2 confidence thresholds can be routed to a local LLM for a second opinion. Default builds exclude this tier entirely.

**Non-LLM AI on the default path.** The default binary ships Tier 1 and Tier 2 only. Classification is deterministic, auditable, bit-reproducible, and runs in ~50 ms per file on CPU. Tier 3 sits symmetrically with remote inference under the same **three-gate** pattern (compile-time feature + runtime config + per-invocation flag).

## The unified-pattern property

"Unified" is a *contract* property, not a *model* property. Behind the seam, multiple encoders and optionally a local LLM may run; at the seam, one shape:

```rust
trait Classifier {
    fn classify(&self, file: &ExtractedContent) -> ClassificationResult;
}

struct ClassificationResult {
    target: FolderId,
    confidence: f32,          // raw weighted-cosine score in v0.1 (see Confidence)
    reasoning: String,        // templated, auditable
    rename_suggestion: Option<RenameProposal>,
}
```

One input, one output, same for every modality and every operation. That is the unification.

## v0.1 scope: text by default, image/audio opt-in

v0.1 ships text Tier 2 via `bge-small-en-v1.5` by default. Phase 7 added
optional cross-modal Tier 2 for images (`SigLIP-base-patch16-224`) and audio
(`CLAP-htsat-unfused`); both are off by default and only activate when their
ONNX bundles are present in the platform model cache. Cross-modal Tier 2 now
applies in **both** scan and migration mode: scan ranks against per-modality
taxonomies, migration ranks against per-folder image/audio centroids built by
the profiler (see "Migration-mode multimodal centroids" below). Video still
routes through Tier 1 heuristics — keyframe extraction is gated on the
`ffmpeg-next` FFI decision.

The scan-mode text taxonomy is the built-in `default_taxonomy()` unless the user
supplies their own with `scan --taxonomy <file.toml>` — an array of validated
`[[entry]]` tables (`path` must end with `/`, `description`, optional
`temporal`), loaded by `load_taxonomy_file` and embedded in place of the default.
The disk embedding cache is keyed by description hash, so a custom taxonomy
simply misses the default cache and embeds fresh; image/audio taxonomies are
unaffected (they remain the per-modality defaults).

A custom taxonomy file looks like this:

```toml
# custom-taxonomy.toml — used via: tidyup scan <root> --taxonomy custom-taxonomy.toml
[[entry]]
path = "Finance/Taxes/"          # required; must end with "/"
description = "tax returns, W-2 and 1099 forms, IRS correspondence, deductions"
temporal = true                   # optional; bucket by year/date when placing

[[entry]]
path = "Projects/Writing/"
description = "essays, drafts, manuscripts, blog posts"

[[entry]]
path = "Reference/Manuals/"
description = "product manuals, datasheets, user guides, specifications"
```

It overrides only the text taxonomy; the image and audio taxonomies stay at their per-modality defaults.

| Modality | Default handling | Phase 7 (opt-in) | Post-Phase-7 |
|---|---|---|---|
| Text (pdf / docx / md / source / ipynb) | Tier 2 via `bge-small-en-v1.5` | — | — |
| Image (jpg / png / heic / raw) | Tier 1 heuristics (extension + EXIF) | Tier 2 via `SigLIP-base` (cross-modal) when bundle installed | — |
| Audio (mp3 / flac / m4a / wav) | Tier 1 heuristics (extension + ID3 via `lofty`) | Tier 2 via `CLAP-htsat-unfused` (cross-modal) when bundle installed | — |
| Video (mp4 / mov / mkv) | Tier 1 heuristics only (extension + container metadata) | — | SigLIP(keyframe) + CLAP(audio), gated on pure-Rust video decode vs `ffmpeg-next` FFI decision |

Adding a modality is a new encoder behind a port trait — currently
[`ImageEmbeddingBackend`](crates/tidyup-core/src/inference.rs) and
[`AudioEmbeddingBackend`](crates/tidyup-core/src/inference.rs) — not a
service-layer refactor. Each modality's backend is held as
`Option<Arc<dyn …>>` on `ServiceContext`; the pipeline routes by
[`FileModality`](crates/tidyup-core/src/inference.rs) and short-circuits to
the text Tier 2 path when the modality backend is absent.

### Cross-modal latent-space isolation

Image and audio backends produce vectors in **modality-specific** latent
spaces — SigLIP's image embeddings are not comparable to bge-small text
embeddings, nor to CLAP's audio embeddings. The pipeline keeps each
modality's candidate list separate (`ImageContext.candidates`,
`AudioContext.candidates`) so a misconfigured caller cannot compute a
cross-space cosine. Each modality has its own natural-language taxonomy
authored as captions ("a photograph of a person", "a podcast episode") rather
than the keyword soup that works best for `bge-small`.

## Tier 2: how embedding classification works

Text extracted via `tidyup-extract` → embedded with `bge-small-en-v1.5` (384-dim, ~35 MB Q8 ONNX, via `ort`) → scored against each candidate folder's profile:

```
score(v, folder) = w_cent · cos(v, folder.content_centroid)
                 + w_name · cos(v, folder.name_embedding)
                 + w_meta · metadata_match(file, folder)
```

Default weights (from `ClassifierConfig::ScoreWeights` in `tidyup-domain`): `w_cent = 0.55`, `w_name = 0.25`, `w_meta = 0.10`. Tunable per-config. (`ScoreWeights` also carries a `w_hier = 0.10` weight for a path-hierarchy prior, but that term is **reserved and currently inert** — `hierarchy_adjustment` is fixed at `0.0` pending an implementation — so it contributes nothing today.)

**Centroid-absent fallback.** The profiler populates `content_centroid` from a bounded sample of each folder's documents (see "Migration-mode multimodal centroids" below). A folder with no extractable text documents — an empty/cold target folder, or one holding only images/audio — keeps `content_centroid = None`. When the centroid is missing, the pipeline redistributes `w_cent` onto `w_name` (so the effective `w_name` is `0.80` and the centroid term is `0`), keeping the composite on the same `[0, 1]` scale rather than shrinking it. For a cold target, folder-name embedding similarity is what carries placements until the folder accumulates documents.

Decision per file:

1. Compute `score` against every candidate folder.
2. Top score above `embedding_threshold` (default 0.35) AND gap-to-second above `ambiguity_gap` (default 0.05) ⇒ confident proposal, exit cascade.
3. Below either threshold ⇒ surface to review. If `--features llm-fallback` is compiled in and enabled per-invocation, route to Tier 3 instead of surfacing directly to review.

### Migration-mode centroids (text + cross-modal)

Migration profiles carry up to three centroids, each built from a bounded
sample (`CENTROID_SAMPLE_CAP` = 24) of a folder's *direct* files and each in its
own latent space:

- **`content_centroid` (text).** The profiler extracts the bodies of the
  folder's text documents (everything that isn't image/audio/video) and embeds
  them with the same `bge-small` backend as `name_embedding`. This is the
  `w_cent = 0.55` term — the dominant signal — so a folder full of tax PDFs
  attracts tax-like source files by *content*, not just folder name. A folder
  with no text documents keeps `content_centroid = None` (see fallback above).
- **`image_centroid` (SigLIP) / `audio_centroid` (CLAP).** When the cross-modal
  bundles are installed, the profiler also samples the folder's image / audio
  files and averages + L2-normalizes them into the matching centroid. Built only
  when the corresponding backend is loaded; the default install leaves both
  `None`.

Source files then route by modality:

- **Text** files score against `content_centroid` + `name_embedding` via the
  composite above.
- **Image / audio** files are embedded with SigLIP / CLAP and ranked by cosine
  **only** against folders that carry an `image_centroid` / `audio_centroid`,
  gated by the same `embedding_threshold` / `ambiguity_gap`. The verdict
  reasoning records `tier2 image-centroid: …` / `tier2 audio-centroid: …`.

**Latent-space isolation.** An image embedding is never compared against
`name_embedding`, `content_centroid`, or `audio_centroid` — they live in
disjoint spaces. When no folder has a centroid in the file's modality (or the
file can't be read/embedded), the cascade falls through to the text Tier 2 /
Tier 1 path rather than fabricating a cross-space match.

**Renames stay on the text path.** As in scan mode, cross-modal placement does
not generate a rename — image/audio renames remain extractive (EXIF / ID3
metadata → keyword fill → adapt → keep).

## Confidence: raw cosine by default, calibration available

By default tidyup reports `confidence` as the **raw weighted-cosine score**, not a calibrated probability. Thresholds (`embedding_threshold`, `min_classification_confidence`, `min_mismatch_score`) are tunable defaults chosen empirically from development use, documented in the config as raw-score values, and exposed in proposal reasoning so users can calibrate intuition by observation.

The calibration **mechanism** now exists: `ClassifierConfig.calibration` (a `tidyup_domain::Calibration`, default `Identity`) applies optional Platt scaling `sigmoid(a·raw + b)` to reported confidence, and `cargo xtask eval --calibrate` fits `(a, b)` over the golden corpus and reports Expected Calibration Error before/after (`tidyup_pipeline::calibration`). But the **shipped default stays `Identity` (uncalibrated)**: a trustworthy fitted parameter set needs the embedding model plus a held-out labelled corpus larger than the current fixture set. Until that ships, claiming "calibrated 85%" would be marketing, not engineering — the tooling to earn it is in place; the corpus isn't yet.

### Validating the premise (held-out routing eval)

Calibration is downstream of a more basic question: *does content-embedding routing actually beat the cheap baselines it claims to replace?* `cargo xtask eval-routing <corpus>` (`xtask/src/routing_eval.rs`) is the falsifiable experiment. It treats an **already-organized directory as ground truth** (folder = label — 20 Newsgroups, BBC-News-by-category, etc. drop straight in), holds out files per label, builds each folder's content centroid from the train split, and routes the held-out files with the **real** embedding backend and the migration centroid-cosine rule. It reports top-1/top-3 with bootstrap 95% CIs against three baselines — **filename-embedding** (the one tidyup must beat to justify reading contents at all), most-frequent (chance floor), and extension — plus the content−filename delta; `--fail-under <margin>` turns that delta into a gate.

Honest bounds: public corpora are cleanly separable, so they are an **upper bound** ("if it can't sort 20NG it can't sort a real Downloads folder" — necessary, not sufficient); the real test is a consented personal corpus, and reported accuracy is capped by inter-annotator agreement on the labels. The split, metrics, and baselines are unit-tested deterministically against a stub backend, so the *instrument* is proven without a model; the **`model-eval` nightly lane** runs it (and `eval`) against the real model + corpus — the only CI lane that touches the embedding path.

Proposal reasoning strings surface raw sub-scores:

```
centroid match 0.92 to Research/Papers; gap to Archive/2023 = 0.08; name-embedding match 0.71. filename 'DSC_0481.jpg' vs content cos-sim 0.03 → rename suggestion from EXIF subject.
```

## Bundle detection

Bundles are identified in the walk phase, before per-file scoring.

**Hard bundles** (deterministic marker detection — not AI):

| Kind | Marker |
|---|---|
| `GitRepository` | `.git/` |
| `NodeProject` | `package.json` |
| `RustCrate` | `Cargo.toml` |
| `PythonProject` | `pyproject.toml` / `setup.py` / `setup.cfg` |
| `XcodeProject` | `*.xcodeproj` |
| `AndroidStudioProject` | `settings.gradle` / `build.gradle` |
| `JupyterNotebookSet` | `.ipynb` neighbours |

**Soft bundles (file-set bundles) — v0.1 (metadata/filename path):**

Loose sibling files clustered by content metadata in `pipeline::clustering`. They have no shared directory, so they move as atomic file-sets (`BundleKind::moves_as_file_set()`), member by member.

| Kind | Signal |
|---|---|
| `PhotoBurst` | EXIF capture timestamps within a window (`burst_window_secs`, default 60s; `min_burst` default 3) |
| `MusicAlbum` | shared ID3 `album` tag (`min_album` default 3) |
| `DocumentSeries` | numeric/date filename family grouped by a shared stem (`invoice-2024-01`, …; `min_series` default 3) |

HDBSCAN (density-based clustering) is the *planned* embedding-verification step — a textbook non-LLM AI technique, deterministic under a fixed `min_cluster_size`: clusters too small are rejected, and clusters whose embedding spread exceeds a threshold are rejected as accidental neighbours. **It is not wired yet.** Today all three soft-bundle kinds detect on metadata/filename signals only — `PhotoBurst` by EXIF capture-time window, `MusicAlbum` by shared ID3 `album` tag, `DocumentSeries` by filename family (all in `pipeline::clustering`) — without embedding verification, acceptable for the common cases. The SigLIP/CLAP encoders the image/audio embedding path needs have since landed (Phase 7), so the remaining work is the verification pass itself, not the encoders.

Once a bundle is identified, its subtree is **opaque** to per-file classification, and the bundle is *placed* as a unit rather than scored per file. Scan mode routes it to a default taxonomy folder chosen by `BundleKind` at a fixed 0.90 confidence; migration mode places it by embedding the bundle-kind label plus the bundle's leaf directory name against the target folders' `name_embedding`s (also fixed 0.90), falling back to the kind default when the profile cache is empty. The result is a `BundleProposal` (not a per-file `ClassificationResult`); there is no pooled embedding of member contents. Bundle members never receive rename proposals (per the rename policy in `CLAUDE.md`).

## Rename strategy: extractive cascade

Rename proposals come from an extractive cascade. Each step is strictly higher-signal than the one below; the first that fires produces the proposal.

1. **Embedded metadata.** PDF `/Title`, DOCX `core.xml` title, ID3 `TIT2`, EXIF `ImageDescription`, Office core properties. If present and non-trivially different from the current filename, this is the rename.
2. **Keyword-template fill.** Extract top-k keyphrases from content — n-grams up to 3 words (inlined YAKE — see below). The target folder's siblings are analysed for a naming pattern via regex inference. Top keyphrases fill the `<topic>` slot, flattened into a word-deduplicated stem (`"tax return"` + `"tax form"` → `tax_return_form`); dates come from EXIF or file mtime.
3. **No signal → no rename.** Keep the filename; just move.

The rename policy (`CLAUDE.md`) says renames never auto-apply, bundle members never rename, and two signals — classification confidence and filename-content mismatch — must clear configured thresholds. This cascade is **structurally incapable of fabricating a rename without extractive evidence**.

Filename-content mismatch: `1.0 - cos(embed(filename_as_text), content_embedding)`. `taxes_2023.pdf` with tax-return content scores low (name matches content); `DSC_0481.jpg` of a wedding scores high.

**Keyword extraction crate.** YAKE is the algorithm, now n-gram-aware: candidates are phrases up to 3 words built from runs of consecutive content tokens (never spanning a stopword/numeric/punctuation boundary) and scored by the YAKE keyphrase rule `∏ S(t) / (TF · (1 + ∑ S(t)))`. Stopwords are **language-aware**: a lightweight, dependency-free detector picks the document's language by stopword overlap (English, Spanish, French, German — English is the conservative default and only loses to a clearly-dominant language), so renames for non-English content don't fill with `le`/`la`/`der`/`el`. This is independent of classification, which still embeds with English `bge-small`. Available crates (`keyword-extraction-rs` at ~20k DL/mo) sit below the 100k-DL/mo dependency threshold in `CLAUDE.md`. v0.1 inlines the YAKE logic (a few hundred lines, in `yake.rs`) rather than depending on the below-threshold crate. Re-evaluate if a mainstream pure-Rust option matures.

## Why embeddings over a local LLM on the default path

Every constraint in `README.md`, `CLAUDE.md`, and `ARCHITECTURE.md` scores embedding-default higher, except generative rename — and generative rename is the one operation the spec already treats as conservative and threshold-gated.

| Constraint | Default LLM (Qwen3 + SmolVLM) | Default embeddings (bge-small) |
|---|---|---|
| Local-first, network-silent default | ✓ | ✓ |
| First-run download size | ~800 MB | ~35 MB |
| Minimize external deps | `mistralrs` → `candle` → `hf-hub` → heavy tokenizer tree | only `ort` |
| Deterministic / auditable | ✗ — sampling, temperature, version drift | ✓ — cosine math is bit-reproducible |
| Interactive latency | 1–10 s/file on CPU | ~50 ms/file |
| Review-first, never auto-apply rename | OK | OK (and less hallucinatory — fewer noisy rejects) |
| Atomic bundles, reversible moves | Classifier choice irrelevant | Same |
| Pure-Rust-preferred | `mistralrs`/`candle` (Rust, but deep tree) | `ort` — FFI cost already accepted |

Both modes treat the LLM as an optional Tier 3 escape hatch rather than a default — the cost of CPU inference (25–50s/file) makes embedding-default the right baseline, with the human review step as the final safety net for low-confidence Tier 2 verdicts. Tier 3 sits behind the three-gate activation in both scan and migration mode; same code path, same threshold logic, same review fallback when the LLM still produces a low-confidence verdict.

## What embeddings give up — honestly

1. **Generative naming.** No coining of novel descriptive names like "Q3 board meeting notes" without extractive evidence. Consistent with the rename policy, but less ambitious than a multimodal LLM at freeform rename synthesis.
2. **Natural-language reasoning strings.** Templated, not lyrical — strictly better for post-hoc threshold tuning, worse for explanation readability.
3. **Cold-start into empty target hierarchies.** An LLM can classify into an empty folder via pure semantic reasoning; embeddings need at least a `name_embedding` or one sibling file for a centroid. **This is a real regression** for the "migrate into a freshly-sketched hierarchy" UX. Mitigations: seed empty folders from name embeddings only and accept weaker confidence; detect the empty-target case and surface all proposals to review; document `--features llm-fallback` as the recommended remedy for this specific workflow.
4. **Long-tail esoterica.** Niche vocabulary unfamiliar to bge-small yields weak signal. Mitigation: weak confidence routes to review.
5. **Multilingual coverage.** `bge-small-en-v1.5` is English-only, so *classification* of non-English documents is still weak — a multilingual embedding model (`bge-m3`, `multilingual-e5`) is the real fix but is larger (~500 MB+) and remains a roadmap item gated on demand vs. binary-size. What *does* work today: **keyword extraction is language-aware** (EN/ES/FR/DE stopword detection), so extractive renames for non-English content stay clean even though the classifier embeds in English.

None of these break a spec invariant. They shift judgment to the human review step, which the spec already frames as the safety net. (3) is the sharpest — call it out in `--help` output and docs.

## Tier 3: the LLM-fallback escape hatch

`tidyup-inference-mistralrs` is retained but feature-gated, symmetric with `tidyup-inference-remote`:

- Default builds **exclude** `tidyup-inference-mistralrs` from the dependency graph entirely (no `mistralrs`, no `candle`, no heavy tokenizer tree). Privacy check: `cargo tree -p tidyup-cli | grep -E 'mistralrs|candle'` returns nothing.
- `cargo build --features llm-fallback` includes the crate.
- Runtime activation requires explicit config (`[inference] llm_fallback = true`) **and** a per-invocation flag (`--llm-fallback` / `TIDYUP_LLM_FALLBACK=1`). The CLI rejects activation without the matching cargo feature compiled in.
- Never recommended in first-run UX or default docs.
- `cargo-deny` ban on `mistralrs`/`candle` outside the `llm-fallback` feature (same pattern as the existing `remote` rule).

### What Tier 3 actually does (current implementation)

When Tier 2 lands in the **review zone** (`needs_review = true` — below `embedding_threshold` or inside `ambiguity_gap`) and a `TextBackend` is wired in, the pipeline:

1. Calls `text_backend.classify_text(content, filename)` — the LLM emits a `ContentClassification { category, tags, summary, suggested_name }`.
2. Builds a query string from `category + tags + summary` (the `suggested_name` is **deliberately dropped** — renames stay extractive per the rename policy).
3. Re-embeds the query via the same `EmbeddingBackend` Tier 2 used.
4. Re-ranks the same candidate list (scan: `ScanCandidate[]`; migration: `FolderProfile[]`) under the same scoring rules.
5. Adopts the LLM-reranked top **only if** it scores higher than the Tier 2 top. Otherwise the Tier 2 verdict stands.

The cost (1–10 s of inference) is paid only on hard cases — Tier 2 hits that already cleared their thresholds skip Tier 3 entirely. The verdict's `reasoning` field records `tier3 llm-rerank: …` so post-hoc auditing can tell which tier resolved each file. In migration mode the result also carries `Tier::Llm` in `ClassificationResult.resolved_at`.

`tidyup-inference-remote` plugs into the same seam: it implements `TextBackend`, so `--remote` swaps the local mistralrs engine for a remote OpenAI-compatible / Anthropic / Ollama endpoint without any pipeline changes.

## Architectural implications

- **`tidyup-inference-mistralrs` is feature-gated.** `--features llm-fallback`. Not in the default crate graph.
- **`tidyup-embeddings-ort` carries the default classifier.** Hosts `bge-small-en-v1.5` with room to add modality-specific encoders post-v0.1.
- **`tidyup-pipeline` hosts Tier 1 + Tier 2.** Plus soft-bundle clustering (metadata/filename-only today in `pipeline::clustering`; HDBSCAN-over-embeddings is the planned upgrade), the extractive rename cascade, and (when the feature is on) the Tier 3 call-through.
- **Marker bundle detection stays in `pipeline::bundle`; soft-bundle clustering lives in `pipeline::clustering`.** Marker detection unchanged; soft-bundle clustering is metadata/filename-only in v0.1 (no embedding step yet).
- **`ClassifierConfig.calibration` defaults to `Identity` (raw cosine).** The Platt-scaling mechanism + fitting tool (`cargo xtask eval --calibrate`) exist; the shipped default stays uncalibrated until a corpus-fit parameter set lands.

## Open questions

- **Held-out corpus for calibration (v0.2).** Ship a synthetic fixture corpus? Calibrate on first run against a labelled sample? Defer until there's real-world feedback to mine (with user opt-in)?
- **Multilingual support.** When to swap to `bge-m3` or `multilingual-e5`? Gate on binary-size impact vs observed demand.
- **Image-side rename gating.** Phase 7 image classification produces a folder choice but no rename proposal. The rename cascade still runs against text Tier 2 (EXIF metadata → keyword fill → keep). A future enhancement: cross-modal mismatch gate using SigLIP text + image embeddings of filename and content.
- **Video keyframe extraction.** Still pending the `ffmpeg-next` FFI vs metadata-only decision.
- **Cold-start UX mitigation.** Does tidyup detect "target hierarchy has no files yet" and advise the user about `--features llm-fallback`, or silently surface all proposals to review with low confidence? A `--bootstrap` mode that seeds profiles from folder-name embeddings only?
- **Inline-YAKE maintenance.** A few hundred lines of keyword extraction inline is cheap but adds a small maintenance item. Acceptable until a mainstream crate crosses the DL threshold.

## References in repo

- `ARCHITECTURE.md` — crate graph, port traits, privacy model, bundle atomicity.
- `CLAUDE.md` — coding standards, invariants, dependency policy, rename policy, safety model.
- `README.md` — roadmap entries, feature flag structure, phase-by-phase ship plan.
- `crates/tidyup-domain/src/migration.rs` — `ClassifierConfig`, `ScoreWeights`, `FolderProfile`, `OrganizationType`.
