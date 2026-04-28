# Classification

Design doc for how tidyup decides where a file belongs, whether it should be renamed, and whether it is part of a larger group. Complements `ARCHITECTURE.md` (crate graph, port traits, privacy model) by fleshing out what sits *behind* the `Classifier` port.

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
ONNX bundles are present in the platform model cache. Video still routes
through Tier 1 heuristics — keyframe extraction is gated on the `ffmpeg-next`
FFI decision.

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
                 + w_hier · hierarchy_prior(file.path, folder.path)
```

Default weights (from `ClassifierConfig::ScoreWeights` in `tidyup-domain`): `w_cent = 0.55`, `w_name = 0.25`, `w_meta = 0.10`, `w_hier = 0.10`. Tunable per-config.

**Centroid-absent fallback (v0.1).** Target folders start with `content_centroid = None` on first scan — centroids are populated only once a folder has enough member embeddings to average meaningfully. When a profile's centroid is missing, the pipeline redistributes `w_cent` onto `w_name` (so the effective `w_name` is `0.80` and the centroid term is `0`), keeping the composite on the same `[0, 1]` scale rather than shrinking it. Folder names are the only semantic signal for a cold target; name-embedding similarity is what carries those placements.

Decision per file:

1. Compute `score` against every candidate folder.
2. Top score above `embedding_threshold` (default 0.35) AND gap-to-second above `ambiguity_gap` (default 0.05) ⇒ confident proposal, exit cascade.
3. Below either threshold ⇒ surface to review. If `--features llm-fallback` is compiled in and enabled per-invocation, route to Tier 3 instead of surfacing directly to review.

## Confidence in v0.1: raw cosine with documented thresholds

v0.1 reports `confidence` as the **raw weighted-cosine score**, not a calibrated probability. Thresholds (`embedding_threshold`, `min_classification_confidence`, `min_mismatch_score`) are tunable defaults chosen empirically from development use, documented in the config as raw-score values, and exposed in proposal reasoning so users can calibrate intuition by observation.

Calibration to well-behaved probabilities — Platt scaling or isotonic regression against a held-out labelled corpus — is a **v0.2 story**. It requires a corpus that doesn't yet exist. Claiming "calibrated 85%" before that corpus exists would be marketing, not engineering.

Proposal reasoning strings surface raw sub-scores:

```
centroid match 0.92 to Research/Papers; gap to Archive/2023 = 0.08; name-embedding match 0.71; hierarchy prior +0.05. filename 'DSC_0481.jpg' vs content cos-sim 0.03 → rename suggestion from EXIF subject.
```

## Bundle detection

Bundles are identified in the walk phase, before per-file scoring.

**Hard bundles** (deterministic marker detection — not AI):

| Kind | Marker |
|---|---|
| `GitRepository` | `.git/` |
| `NodeProject` | `package.json` + `node_modules/` (never descended) |
| `RustCrate` | `Cargo.toml` (climbs to `[workspace]` root) |
| `PythonProject` | `pyproject.toml` / `setup.py` / `__init__.py` |
| `XcodeProject` | `*.xcodeproj` |
| `AndroidStudioProject` | `settings.gradle` / `build.gradle` |
| `JupyterNotebookSet` | `.ipynb` neighbours |
| `DocumentSeries` | filename regex family (`invoice_*.pdf`, `meeting_YYYY-MM-DD_*`) |

**Soft bundles — v0.1 (text-only path):**

| Kind | Signal |
|---|---|
| `DocumentSeries` (augmented) | filename family + HDBSCAN cluster over text embeddings of candidate members |

**Soft bundles — post-v0.1** (require image/audio encoders):

| Kind | Signal | Requires |
|---|---|---|
| `PhotoBurst` | EXIF timestamps within N seconds + same camera + HDBSCAN over image embeddings | SigLIP Tier 2 |
| `MusicAlbum` | consistent ID3 `album` tag + HDBSCAN over audio embeddings | CLAP Tier 2 |

HDBSCAN (density-based clustering) is a textbook non-LLM AI technique, deterministic under fixed `min_cluster_size`. Clusters too small are rejected; clusters whose embedding spread exceeds a threshold are rejected as accidental neighbours. For v0.1, `PhotoBurst` and `MusicAlbum` fall back to metadata-only detection (timestamp/tag proximity) without embedding verification — acceptable for the common cases, tightenable once the encoders land.

Once a bundle is identified, its subtree is **opaque** to per-file classification. Only the bundle root receives a `ClassificationResult`, using the pooled embedding of its text members (v0.1). Bundle members never receive rename proposals (per the rename policy in `CLAUDE.md`).

## Rename strategy: extractive cascade

Rename proposals come from an extractive cascade. Each step is strictly higher-signal than the one below; the first that fires produces the proposal.

1. **Embedded metadata.** PDF `/Title`, DOCX `core.xml` title, ID3 `TIT2`, EXIF `ImageDescription`, Office core properties. If present and non-trivially different from the current filename, this is the rename.
2. **Keyword-template fill.** Extract top-k terms from content (inlined YAKE — see below). The target folder's siblings are analysed for a naming pattern via regex inference. Top keywords fill the `<topic>` slot; dates come from EXIF or file mtime.
3. **Nearest-neighbor adaptation.** Nearest content-neighbor in the target folder by cosine distance. Adopt its naming template. Substitute content-specific tokens from the current file.
4. **No signal → no rename.** Keep the filename; just move.

The rename policy (`CLAUDE.md`) says renames never auto-apply, bundle members never rename, and two signals — classification confidence and filename-content mismatch — must clear configured thresholds. This cascade is **structurally incapable of fabricating a rename without extractive evidence**.

Filename-content mismatch: `1.0 - cos(embed(filename_as_text), content_embedding)`. `taxes_2023.pdf` with tax-return content scores low (name matches content); `DSC_0481.jpg` of a wedding scores high.

**Keyword extraction crate.** YAKE is the algorithm. Available crates (`keyword-extraction-rs` at ~20k DL/mo) sit below the 100k-DL/mo dependency threshold in `CLAUDE.md`. v0.1 inlines ~150 LoC of YAKE rather than depending on the below-threshold crate. Re-evaluate if a mainstream pure-Rust option matures.

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
5. **Multilingual coverage.** `bge-small-en-v1.5` is English-only. Non-English documents cluster together unhelpfully. Multilingual variants (`bge-m3`, `multilingual-e5`) exist but are larger (~500 MB+). v0.1 is English-only; multilingual support is a roadmap item.

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
- **`tidyup-pipeline` hosts Tier 1 + Tier 2.** Plus HDBSCAN soft-bundle clustering, the extractive rename cascade, and (when the feature is on) the Tier 3 call-through.
- **Bundle detection stays in `pipeline::bundle`.** Marker detection unchanged; soft-bundle clustering uses text embeddings in v0.1.
- **`ClassifierConfig` gains no calibration parameters in v0.1.** Raw-cosine thresholds only. Calibration config lands in v0.2.

## Open questions

- **Held-out corpus for calibration (v0.2).** Ship a synthetic fixture corpus? Calibrate on first run against a labelled sample? Defer until there's real-world feedback to mine (with user opt-in)?
- **Multilingual support.** When to swap to `bge-m3` or `multilingual-e5`? Gate on binary-size impact vs observed demand.
- **Migration-mode multimodal.** Phase 7 wires SigLIP/CLAP into scan mode only. Migration-mode profiling builds folder centroids in the bge-small text space; image/audio files still route through that text-space match in migration mode. Adding `image_name_embedding` / `audio_name_embedding` to `FolderProfile` (and recomputing from existing folder contents) would make migration-mode multimodal symmetric with scan-mode.
- **Image-side rename gating.** Phase 7 image classification produces a folder choice but no rename proposal. The rename cascade still runs against text Tier 2 (EXIF metadata → keyword fill → adapt → keep). A future enhancement: cross-modal mismatch gate using SigLIP text + image embeddings of filename and content.
- **Video keyframe extraction.** Still pending the `ffmpeg-next` FFI vs metadata-only decision.
- **Cold-start UX mitigation.** Does tidyup detect "target hierarchy has no files yet" and advise the user about `--features llm-fallback`, or silently surface all proposals to review with low confidence? A `--bootstrap` mode that seeds profiles from folder-name embeddings only?
- **Inline-YAKE maintenance.** ~150 LoC of keyword extraction inline is cheap but adds a small maintenance item. Acceptable until a mainstream crate crosses the DL threshold.

## References in repo

- `ARCHITECTURE.md` — crate graph, port traits, privacy model, bundle atomicity.
- `CLAUDE.md` — coding standards, invariants, dependency policy, rename policy, safety model.
- `README.md` — roadmap entries, feature flag structure, phase-by-phase ship plan.
- `crates/tidyup-domain/src/migration.rs` — `ClassifierConfig`, `ScoreWeights`, `FolderProfile`, `OrganizationType`.
