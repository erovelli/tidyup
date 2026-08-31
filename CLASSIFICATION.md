# Classification

Design doc for how tidyup decides where a file belongs, whether it should be renamed, and whether it is part of a larger group. Complements `ARCHITECTURE.md` (crate graph, port traits, privacy model) by fleshing out what sits *behind* the `Classifier` port.

## Contents

- [The question this doc answers](#the-question-this-doc-answers)
- [Decision: semantic embeddings with optional reranking](#decision-semantic-embeddings-with-optional-reranking)
- [The unified-pattern property](#the-unified-pattern-property)
- [v0.1 scope: text by default, image/audio opt-in](#v01-scope-text-by-default-imageaudio-opt-in)
- [How embedding classification works](#how-embedding-classification-works)
- [Confidence: raw cosine by default, calibration available](#confidence-raw-cosine-by-default-calibration-available)
- [Bundle detection](#bundle-detection)
- [Rename strategy: grounded evidence cascade](#rename-strategy-grounded-evidence-cascade)
- [Why embeddings over a local LLM on the default path](#why-embeddings-over-a-local-llm-on-the-default-path)
- [What embeddings give up — honestly](#what-embeddings-give-up--honestly)
- [The optional LLM-fallback escape hatch](#the-optional-llm-fallback-escape-hatch)
- [Architectural implications](#architectural-implications)
- [Open questions](#open-questions)
- [References in repo](#references-in-repo)

## The question this doc answers

A messy source directory contains text documents, images, audio, video, and mixed bundles. For each file, the pipeline must produce:

1. A **target folder** under the preexisting destination hierarchy.
2. A **reported confidence score** for review and loose-proposal `--yes` selection, plus the raw semantic evidence retained for the rename gate.
3. An optional **rename suggestion** when filename and content disagree.
4. For soft bundles (photo bursts, albums, document series), a **grouping decision** that treats related files as a single atomic move.

All of this must run **locally, deterministically, reversibly, and fast enough for interactive review** — per the four product promises in `CLAUDE.md`. What sits behind the `Classifier` port to deliver this with one unified pattern?

## Decision: semantic embeddings with optional reranking

tidyup routes destinations by semantic similarity. Extension and keyword heuristics do not choose folders; structural marker detection is reserved for preserving load-bearing bundles. The default path is one deterministic embedding stage, with an optional LLM reranker for uncertain results.

1. **Semantic embedding routing.** In scan mode, separately embedded filename/path/MIME context and extracted content are weighted by `ScoreWeights` and ranked against a fixed taxonomy. In migration mode, extracted content is ranked against learned target-folder name and content-centroid embeddings. Missing content redistributes weight to the live signal instead of deflating confidence.
2. **Local LLM reranking (optional, opt-in).** Feature-gated under `--features llm-fallback`, off by default. When compiled and triple-gated per invocation, uncertain embedding results can receive a second opinion. The result is re-embedded and must beat the original score; it never generates filenames.

**Non-LLM AI on the default path.** The default binary uses embeddings only. The scoring flow is deterministic and auditable for a fixed verified model/preprocessing/runtime stack. The product target is at most one second per warm file, but the current production path does not yet persist per-stage spans or enforce a cancellable deadline. `cargo xtask bench-semantic` measures one representative warm SigLIP path; it is not evidence that every modality and machine meets the SLA. Optional local/remote rerankers sit behind the same **three-gate** pattern (compile-time feature + runtime config + per-invocation flag).

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

v0.1 ships text embeddings via `bge-small-en-v1.5` by default. Optional
cross-modal embeddings for images (`SigLIP-base-patch16-224`) and audio
(`CLAP-htsat-unfused`); both are off by default and only activate when their
ONNX bundles are present in the platform model cache. Cross-modal routing now
applies in **both** scan and migration mode: scan ranks against per-modality
taxonomies, migration ranks against modality-specific folder-label prototypes
and optional image/audio content centroids built by the profiler. Video has no
dedicated content encoder; it uses any available text/name context and
otherwise remains low-confidence or unclassified. Keyframe extraction is
gated on the `ffmpeg-next` FFI decision.

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

The examples below describe semantic modalities, not destination-routing rules. MIME sniffing identifies modality where possible; an extension lookup is only a parser/decoder fallback and never maps a file directly to a folder.

| Modality (examples) | Default handling | Optional specialized encoder | Future |
|---|---|---|---|
| Text (plain text / Markdown / source / notebook; PDF with the `pdf` extractor feature) | `bge-small-en-v1.5` over filename/context + extracted text | — | — |
| Image (jpg / png / heic / raw) | general text embeddings over filename/context + EXIF/OCR text | SigLIP cross-modal when installed | — |
| Audio (mp3 / flac / m4a / wav) | general text embeddings over filename/context + ID3 text | CLAP cross-modal when installed | — |
| Video (mp4 / mov / mkv) | name/context text only; weak results review/unclassified | — | SigLIP(keyframe) + CLAP(audio), pending decoder decision |

Adding a modality is a new encoder behind a port trait — currently
[`ImageEmbeddingBackend`](crates/tidyup-core/src/inference.rs) and
[`AudioEmbeddingBackend`](crates/tidyup-core/src/inference.rs) — not a
service-layer refactor. Each modality's backend is held as
`Option<Arc<dyn …>>` on `ServiceContext`; the pipeline routes by
[`FileModality`](crates/tidyup-core/src/inference.rs). When the matching
backend **is** loaded (the bundle is on disk), an image/audio file is classified cross-modally first — in scan
mode against the image/audio taxonomy, in migration mode against the folders'
`image_centroid` / `audio_centroid`. When the backend is **absent** (the default install), or the file is
too large to read, or no folder carries a centroid, routing short-circuits to
the general text-embedding path over filename/context and extracted metadata;
uncertain results are held for review rather than extension-routed.

Before semantic work, each run persists a versioned capability manifest describing the
backends and extractors actually wired: text/image/audio model ids and dimensions, optional
LLM/VLM activation and network requirement, registered extractors, and OCR
availability/configuration. `Available`, `Unavailable`, and `Disabled` are distinct states.
Missing modalities are therefore explicit evidence provenance in review and history, not
something inferred from the resulting proposals.

### Cross-modal latent-space isolation

Image and audio backends produce vectors in **modality-specific** latent
spaces — SigLIP's image embeddings are not comparable to bge-small text
embeddings, nor to CLAP's audio embeddings. The pipeline keeps each
modality's candidate list separate (`ImageContext.candidates`,
`AudioContext.candidates`) so a misconfigured caller cannot compute a
cross-space cosine. Each modality has its own natural-language taxonomy
authored as captions ("a photograph of a person", "a podcast episode") rather
than the keyword soup that works best for `bge-small`.

## How embedding classification works

Text extracted via `tidyup-extract` → embedded with `bge-small-en-v1.5` (384-dim, ~35 MB Q8 ONNX, via `ort`) → scored against each candidate folder's profile:

```
score(v, folder) = w_cent · cos(v, folder.content_centroid)
                 + w_name · cos(v, folder.name_embedding)
```

Default weights (from `ClassifierConfig::ScoreWeights` in `tidyup-domain`): `w_cent = 0.6875`, `w_name = 0.3125`. They preserve the former 55:25 ratio but now sum to `1.0`. The removed metadata and hierarchy fields were permanently zero after generalized semantic routing replaced extension compatibility; deleting those dead terms is a deliberate breaking change to the domain scoring contract and prevents confidence from being capped at `0.80`.

**Centroid-absent fallback.** The profiler populates `content_centroid` from a bounded sample of each folder's documents (see "Migration-mode multimodal centroids" below). A folder with no extractable text documents — an empty/cold target folder, or one holding only images/audio — keeps `content_centroid = None`. When the centroid is missing, the pipeline redistributes `w_cent` onto `w_name` (so the effective `w_name` is `1.0` and the centroid term is `0`), keeping the composite on the same `[0, 1]` scale rather than shrinking it. For a cold target, folder-name embedding similarity is what carries placements until the folder accumulates documents.

Decision per file:

1. Compute `score` against every candidate folder.
2. Top score above `embedding_threshold` (default 0.35) AND gap-to-second above `ambiguity_gap` (default 0.05) ⇒ confident proposal.
3. Below either threshold ⇒ surface to review. If the optional LLM fallback is compiled and triple-gated for the invocation, rerank before surfacing.

### Migration-mode prototypes and centroids (text + cross-modal)

Migration profiles carry modality-specific text prototypes plus up to three
content centroids. Centroids are built from a bounded sample
(`CENTROID_SAMPLE_CAP` = 24) of a folder's *direct* files; every signal remains
inside its own latent space:

- **`content_centroid` (text).** The profiler extracts the bodies of the
  folder's text documents (everything that isn't image/audio/video) and embeds
  them with the same `bge-small` backend as `name_embedding`. This is the
  `w_cent = 0.6875` term — the dominant signal — so a folder full of tax PDFs
  attracts tax-like source files by *content*, not just folder name. A folder
  with no text documents keeps `content_centroid = None` (see fallback above).
- **`image_name_embedding` (SigLIP text tower) / `audio_name_embedding` (CLAP
  text tower).** Every folder path and label is phrased as a natural-language
  modality description and embedded by the matching text tower. These
  prototypes exist even for empty named folders.
- **`image_centroid` (SigLIP) / `audio_centroid` (CLAP).** When the cross-modal
  bundles are installed, the profiler also samples the folder's image / audio
  files and averages + L2-normalizes them into the matching centroid.

Source files then route by modality:

- **Text** files score against `content_centroid` + `name_embedding` via the
  composite above.
- **Image / audio** files are embedded with SigLIP / CLAP and ranked against
  their matching folder-name prototype plus optional content centroid. Missing
  signals redistribute their weight, so an empty `Photos/Cats` folder remains
  a valid image destination. The current persisted reasoning strings retain the legacy
  `tier2 image-profile: …` / `tier2 audio-profile: …` labels for compatibility; these mean
  the deterministic embedding stage, not a surviving three-tier router.

**Latent-space isolation.** An image embedding is never compared against the
BGE `name_embedding`, `content_centroid`, or an audio signal. It compares only
with SigLIP text prototypes and SigLIP image centroids. Audio follows the same
rule in CLAP space. When no matching-space profile exists (or the file cannot
be read/embedded), routing falls through to the general text path.

The same cached image embedding also ranks a bounded visual concept asset plus
dynamic target-folder labels. Strong concepts may ground a semantic rename or
label a visually coherent collection; concepts never map directly to a
destination.

### Semantic artifact reuse

Extraction and modality embeddings are shared within a run by
`SemanticRunCache`, so clustering, placement, and naming never launch separate
inference for the same path/model. Image/audio vectors are also persisted by
`FileIndex` under `(BLAKE3 content hash, model id, preprocessing version,
latent space)` as compact little-endian `f32` blobs, with read compatibility
for legacy JSON rows. An exact cache hit skips model inference on later runs;
`tidyup prune` removes entries older than its retention window. A missing,
stale, malformed, or unavailable cache entry degrades to fresh local inference.
Full persistent text-extraction/result caching is not wired yet.

Identity indexing also writes a run-scoped processing ledger before classification. Every regular
file the walker can enumerate moves from `Discovered` to `Indexed` or `Failed`; source planning
then records `Classified`, `Unclassified`, or terminal identity-only `Indexed`, while migration
target-profile files retain a separate role. Records are committed in one transaction per stage,
not one transaction per file transition. A failure keeps its stage and reason even when hashing
failed before a stable `FileId` existed. If the operating system prevents directory enumeration
entirely, only a walk error path exposed by `WalkDir` can be recorded; unknown children cannot be
named individually.

## Confidence: raw cosine by default, calibration available

By default tidyup reports `confidence` as the **raw weighted-cosine score**, not a calibrated probability. Pipeline routing uses model-specific defaults: BGE text/general routing uses `text_min_similarity = 0.35` and `text_ambiguity_gap = 0.05`; SigLIP image routing uses `image_min_similarity = 0.04` and `image_ambiguity_gap = 0.01`; aggregate bundles use `bundle_min_similarity = 0.35` and `bundle_ambiguity_gap = 0.03`. These values are user-tunable under `[classifier]` because raw scores from different embedding spaces are not comparable. Rename thresholds are user-tunable under `[rename]`, and loose move-only `--yes` approval uses `[classifier] min_confidence = 0.75`. Recognized opaque structural bundles use a separate internal raw-cosine floor of `0.50`. File-set collections and every rename remain review-only. Proposal reasoning exposes raw sub-scores for later calibration.

The calibration **mechanism** now exists: pipeline `ClassifierConfig.calibration` (a `tidyup_domain::Calibration`, default `Identity`) applies optional Platt scaling `sigmoid(a·raw + b)` to reported proposal confidence, and `cargo xtask eval --calibrate` fits `(a, b)` over the golden corpus and reports Expected Calibration Error before/after (`tidyup_pipeline::calibration`). It is not currently a TOML setting. Rename eligibility deliberately uses the original deterministic embedding score, never an optional LLM-reranked score. The **shipped default stays `Identity` (uncalibrated)**: trustworthy parameters require larger, held-out, action-specific corpora. Accordingly, the desktop UI labels today’s values as `similarity 0.81` and keeps its high/medium/low tiers as relative guidance; it never shows a raw score as a percentage. A percentage is reserved for a fitted calibrator whose model and preprocessing identity is compatible with the run manifest.

### Validating the premise (held-out routing eval)

Calibration is downstream of a more basic question: *does content-embedding routing actually beat the cheap baselines it claims to replace?* `cargo xtask eval-routing <corpus>` (`xtask/src/routing_eval.rs`) is the falsifiable experiment. It treats an **already-organized directory as ground truth** (folder = label — 20 Newsgroups, BBC-News-by-category, etc. drop straight in), holds out files per label, builds each folder's content centroid from the train split, and routes the held-out files with the **real** embedding backend and the migration centroid-cosine rule. It reports top-1/top-3 with bootstrap 95% CIs against three baselines — **filename-embedding** (the one tidyup must beat to justify reading contents at all), most-frequent (chance floor), and extension — plus the content−filename delta; `--fail-under <margin>` turns that delta into a gate.

Honest bounds: public corpora are cleanly separable, so they are an **upper bound** ("if it can't sort 20NG it can't sort a real Downloads folder" — necessary, not sufficient); the real test is a consented personal corpus, and reported accuracy is capped by inter-annotator agreement on the labels. The split, metrics, and baselines are unit-tested deterministically against a stub backend, so the *instrument* is proven without a model; the **`model-eval` nightly lane** runs it (and `eval`) against the real model + corpus — the only CI lane that touches the embedding path.

Proposal reasoning strings surface raw sub-scores:

```
centroid match 0.92 to Research/Papers; gap to Archive/2023 = 0.08; name-embedding match 0.71. filename 'DSC_0481.jpg' vs content cos-sim 0.03 → rename suggestion from EXIF subject.
```

## Bundle detection

Bundles are identified before loose-file placement: recognized marker-root directory bundles during a recursive walk, then file-set bundles during cached sibling clustering. Unmarked directories are transparent, so ordinary descendants remain classifiable and nested marker roots are still detected.

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

**Soft bundles (file-set bundles):**

Loose sibling files clustered by content metadata in `pipeline::clustering`. They have no shared directory, so they move as atomic file-sets (`BundleKind::moves_as_file_set()`), member by member.

| Kind | Signal |
|---|---|
| `PhotoBurst` | EXIF capture timestamps within a window (internal clustering defaults: 60 seconds, minimum 3) |
| `MusicAlbum` | shared ID3 `album` tag (internal minimum 3) |
| `DocumentSeries` | sequenced filename family grouped by a shared stem (`invoice-01`, `invoice-02`, …; internal minimum 3) |
| `SemanticCollection` | cross-format siblings sharing descriptive entity evidence, or at least three visually similar sibling images passing a conservative complete-link similarity floor and grounded-label gate |

Semantic-collection matching is deliberately conservative because the result is atomic. Filename tokens are computed once per file, then compared pairwise. Purely numeric/date-like shared prefixes and generic screenshot prefixes (`Screenshot`, `Screen Shot`, with space, dash, or underscore separators) cannot anchor a collection. Expensive extraction is cached per run. Explicit EXIF capture-time bursts are removed before visual-neighbor clustering, preserving `PhotoBurst` semantics and preventing grounded member renames from being added merely because burst frames look alike. Visual clustering then uses deterministic complete-link grouping: a candidate must clear the cosine floor against every current member, preventing an ambiguous bridge image from chaining unrelated groups together. A visual group is emitted only when its centroid also produces a grounded concept label.

Image-embedding clustering is wired without a new ML dependency: the bounded sibling set is small enough for deterministic complete-link comparison, which makes the all-pairs cohesion rule explicit and auditable. Photo-burst, music-album, and document-series signals remain supported as metadata/structure evidence. General audio/text neighbor clustering and corpus-calibrated thresholds remain follow-up work.

Scan reorganizes in place, so a loose file whose classified destination is the directory it already occupies is recorded as classified with no proposal emitted. The interesting output of a housekeeping pass is what sits in the wrong place; a no-op move would add review noise and, because the destination ledger refuses an approved target that already exists, would abort the plan over a file that needed nothing done to it. Migration has no equivalent case — every source file moves into a disjoint target tree.

Only **recognized marker-root directory bundles** are opaque to per-file classification; unmarked directories are traversed recursively and their ordinary files stay loose. Nested markers therefore create bundles at their own roots rather than being swallowed by an ancestor. Directory bundles route from aggregate semantic evidence (kind, root label, and a bounded sample of member names). Their confidence is the raw cosine against taxonomy candidates in scan mode or target-folder name embeddings in migration mode. Semantic collections classify their loose members and report the weaker of collection-routing cosine and mean member confidence. Scan's filename/path/MIME channel uses `ClassifierConfig.weights.name` and its extracted-body channel uses `weights.centroid`, matching migration's configured composite rather than maintaining a second inline weighting scale. Until action-specific calibration exists, `--yes` applies the configured raw-cosine threshold only to recognized opaque structural bundles. Semantic collections, photo bursts, music albums, and document series remain explicit-review only, regardless of score; any member rename is also held for review.

## Rename strategy: grounded evidence cascade

Rename proposals come from grounded evidence. Exact textual evidence remains preferred; non-textual content can use bounded semantic retrieval.

1. **Embedded metadata.** The current extractors expose audio artist/title tags, EXIF `ImageDescription`, EXIF make/model fallback, and a generic top-level `title` when an extractor supplies one. If present and non-trivially different from the current filename, this is the rename. The current PDF extractor provides body text/scanned status rather than PDF title metadata, and there is no DOCX extractor yet.
2. **Local OCR + keyword composition.** On supported macOS builds, bounded Vision OCR feeds the same YAKE keyphrase path as other extracted text. There are no corpus-specific screenshot-role templates. `RenameSource::Ocr` records provenance; the image destination must clear its SigLIP routing threshold and the filename/content mismatch must clear `[rename] min_ocr_mismatch_score`. This keeps OCR-backed screenshot names on an independently tunable path instead of applying BGE's text-routing scale to SigLIP output.
3. **Keyword composition.** Extract top-k keyphrases from content—n-grams up to three words—and compose a deterministic, word-deduplicated stem (`"tax return"` + `"tax form"` → `tax_return_form`).
4. **Grounded multimodal concepts.** A contrastive model ranks the file against a versioned data asset of visible concepts, augmented in migration by dynamic target-folder labels. At most two non-redundant concepts form a deterministic name such as `cat_on_beach.png`. Destination confidence, concept confidence/separation, and `[rename] min_grounded_mismatch` must all pass. Grounded mismatch is the fraction of selected concept labels absent from literal filename tokens; it deliberately does not reuse text's continuous cosine threshold.
5. **No signal → no rename.** Keep the filename; just move.

An extractor-supplied exact title, such as a Markdown level-one heading, is treated as direct document evidence and may produce a review-only rename without relying on embedding mismatch. Platform metadata (`.DS_Store`, `.localized`, `Thumbs.db`, and `desktop.ini`) and hidden paths are excluded by the default `[discovery]` policy before indexing; users can opt hidden content back in explicitly.

The rename policy (`CLAUDE.md`) says renames never auto-apply. Structural bundle members never rename; formerly loose members in a semantic collection may carry grounded renames, but the whole collection must then be reviewed explicitly. The default path has no free-form captioner: every proposed term has inspectable metadata/text or a scored contrastive-model concept.

Before an atomic collection is constructed, proposed member basenames are checked case-insensitively. Unchanged move-only names reserve their basenames first; repeated grounded rename candidates receive deterministic `_2`, `_3`, … suffixes so one collision cannot invalidate the entire collection. A move-only collision is rejected before review rather than receiving an ungrounded fabricated rename. File-set discovery is directory-local today; future cross-directory clustering must preserve relative placement or provide grounded names before relaxing that constraint. The desktop editor enforces the same sibling-uniqueness rule inline before review is submitted.

Filename-content mismatch: `1.0 - cos(embed(filename_as_text), content_embedding)`. `taxes_2023.pdf` with tax-return content scores low (name matches content); `DSC_0481.jpg` of a wedding scores high.

**Keyword extraction crate.** YAKE is the algorithm, now n-gram-aware: candidates are phrases up to 3 words built from runs of consecutive content tokens (never spanning a stopword/numeric/punctuation boundary) and scored by the YAKE keyphrase rule `∏ S(t) / (TF · (1 + ∑ S(t)))`. Stopwords are **language-aware**: a lightweight, dependency-free detector picks the document's language by stopword overlap (English, Spanish, French, German — English is the conservative default and only loses to a clearly-dominant language), so renames for non-English content don't fill with `le`/`la`/`der`/`el`. This is independent of classification, which still embeds with English `bge-small`. Available crates (`keyword-extraction-rs` at ~20k DL/mo) sit below the 100k-DL/mo dependency threshold in `CLAUDE.md`. v0.1 inlines the YAKE logic (a few hundred lines, in `yake.rs`) rather than depending on the below-threshold crate. Re-evaluate if a mainstream pure-Rust option matures.

## Why embeddings over a local LLM on the default path

Every constraint in `README.md`, `CLAUDE.md`, and `ARCHITECTURE.md` scores embedding-default higher, except generative rename — and generative rename is the one operation the spec already treats as conservative and threshold-gated.

| Constraint | Default LLM (Qwen3 + SmolVLM) | Default embeddings (bge-small) |
|---|---|---|
| Local-first, network-silent default | ✓ | ✓ |
| First-run download size | ~800 MB | ~35 MB |
| Minimize external deps | `mistralrs` → `candle` → `hf-hub` → heavy tokenizer tree | only `ort` |
| Deterministic / auditable | ✗ — sampling, temperature, version drift | ✓ — fixed scoring flow with versioned model/preprocessing identity |
| Interactive latency | hardware/model dependent; commonly seconds on CPU | designed for a ≤1 s warm path; only the image benchmark is instrumented today |
| Review-first, never auto-apply rename | OK | OK (and less hallucinatory — fewer noisy rejects) |
| Atomic bundles, reversible moves | Classifier choice irrelevant | Same |
| Pure-Rust-preferred | `mistralrs`/`candle` (Rust, but deep tree) | `ort` — FFI cost already accepted |

Both modes treat the LLM as an optional reranker rather than a default. Its latency is model- and hardware-dependent and can exceed the one-second core semantic budget, so it is reserved for explicitly enabled hard cases; human review remains the final safety net. The reranker sits behind three-gate activation in both modes with the same review fallback when it remains uncertain.

## What embeddings give up — honestly

1. **Open-ended captioning.** Grounded retrieval can name known concepts such as `cat_on_beach`, but it cannot freely describe an unseen long-tail scene the way a multimodal generator can. Dynamic hierarchy labels expand the vocabulary; weak or novel evidence remains `Keep`.
2. **Natural-language reasoning strings.** Templated, not lyrical — strictly better for post-hoc threshold tuning, worse for explanation readability.
3. **Cold-start into empty target hierarchies.** With no leaf folders there is no honest destination candidate for either embeddings or an LLM to rank. Loose files remain explicitly `unclassified`. Atomic directory bundles and semantic collections are preserved in the plan at the target root with confidence `0.0`, unchanged member names, and mandatory review. This deterministic fallback is category-neutral: it prevents files disappearing from the plan without fabricating a taxonomy from bundle kind or extension.

   A **populated** hierarchy that nothing clears is treated the same way. When no folder clears both `embedding_threshold` and `ambiguity_gap` for a semantic collection, migration abstains to the target root at confidence `0.0` with every member marked for review, rather than filing the collection under the least-wrong folder. Unanimous member agreement is a shortcut only when at least one member cleared its own placement gate; agreement among members that all failed it is agreement without confidence, and abstains too.
4. **Long-tail esoterica.** Niche vocabulary unfamiliar to bge-small yields weak signal. Mitigation: weak confidence routes to review.
5. **Multilingual coverage.** `bge-small-en-v1.5` is English-only, so *classification* of non-English documents is still weak — a multilingual embedding model (`bge-m3`, `multilingual-e5`) is the real fix but is larger (~500 MB+) and remains a roadmap item gated on demand vs. binary-size. What *does* work today: **keyword extraction is language-aware** (EN/ES/FR/DE stopword detection), so text-grounded renames for non-English content stay clean even though the classifier embeds in English.

None of these break a spec invariant. They shift judgment to the human review step, which the spec already frames as the safety net. (3) is the sharpest — call it out in `--help` output and docs.

## The optional LLM-fallback escape hatch

`tidyup-inference-mistralrs` is retained but feature-gated, symmetric with `tidyup-inference-remote`:

- Default builds **exclude** `tidyup-inference-mistralrs` from the dependency graph entirely (no `mistralrs`, no `candle`, no heavy tokenizer tree). Privacy check: `cargo tree -p tidyup-cli | grep -E 'mistralrs|candle'` returns nothing.
- `cargo build --features llm-fallback` includes the crate.
- Runtime activation requires explicit config (`[inference] llm_fallback = true`) **and** a per-invocation flag (`--llm-fallback` / `TIDYUP_LLM_FALLBACK=1`). The CLI rejects activation without the matching cargo feature compiled in.
- Never recommended in first-run UX or default docs.
- `cargo xtask check-privacy` asserts `mistralrs`/`candle-core`/`hf-hub` are absent from the default `tidyup-cli`/`tidyup-ui` dep graph (the same check that guards the `reqwest`/`hyper`/`rustls` network surface). `cargo-deny` covers licenses/advisories/sources — it does **not** enforce these feature-gated crate bans (`deny.toml [bans].deny` is empty).

### What the fallback actually does (current implementation)

When embedding routing lands in the **review zone** (`needs_review = true` — below `embedding_threshold` or inside `ambiguity_gap`) and a `TextBackend` is wired in, the pipeline:

1. Calls `text_backend.classify_text(content, filename)` — the LLM emits a `ContentClassification { category, tags, summary, suggested_name }`.
2. Builds a query string from `category + tags + summary` (the `suggested_name` is **deliberately dropped** — LLMs never author filenames under the grounded rename policy).
3. Re-embeds the query via the same `EmbeddingBackend` used for the original result.
4. Re-ranks the same candidate list (scan: `ScanCandidate[]`; migration: `FolderProfile[]`) under the same scoring rules.
5. Adopts the LLM-reranked top **only if** it scores higher than the original embedding top. Otherwise the original verdict stands.

The hardware/model-dependent inference cost is paid only on hard cases; confident embedding hits skip the fallback entirely. The verdict's legacy `reasoning` label records `tier3 llm-rerank: …` so post-hoc auditing can identify fallback use. In migration mode the result also carries the legacy `Tier::Llm` discriminator in `ClassificationResult.resolved_at`; both names are persisted compatibility vocabulary, not a destination-routing tier cascade.

`tidyup-inference-remote` plugs into the same seam: it implements `TextBackend`, so `--remote` swaps the local mistralrs engine for a remote endpoint without any pipeline changes. The crate ships OpenAI-compatible, Anthropic, and Ollama `RemoteEndpoint` adapters, but **only the OpenAI-compatible path is selectable from CLI config today** (`RemoteBackendConfig` has no provider discriminator); the Anthropic/Ollama variants exist but aren't yet wired from config. An Ollama server is still reachable via its OpenAI-compatible `/v1` endpoint.

## Architectural implications

- **`tidyup-inference-mistralrs` is feature-gated.** `--features llm-fallback`. Not in the default crate graph.
- **`tidyup-embeddings-ort` carries the default classifier.** Hosts `bge-small-en-v1.5` with room to add modality-specific encoders post-v0.1.
- **`tidyup-pipeline` hosts semantic embedding routing.** It also owns conservative file-set clustering, the rename cascade, and the optional LLM rerank call-through. Structural bundle markers preserve codebases but never choose destinations for loose files.
- **Marker bundle detection stays in `pipeline::bundle`; soft-bundle clustering lives in `pipeline::clustering`.** Structural markers only protect opaque bundles. Loose-file grouping combines metadata, descriptive filename/text evidence, and conservative complete-link visual similarity without using file type to choose a destination.
- **`ClassifierConfig.calibration` defaults to `Identity` (raw cosine).** The Platt-scaling mechanism + fitting tool (`cargo xtask eval --calibrate`) exist; the shipped default stays uncalibrated until a corpus-fit parameter set lands.

## Open questions

- **Held-out corpus for calibration (v0.2).** Ship a synthetic fixture corpus? Calibrate on first run against a labelled sample? Defer until there's real-world feedback to mine (with user opt-in)?
- **Multilingual support.** When to swap to `bge-m3` or `multilingual-e5`? Gate on binary-size impact vs observed demand.
- **Grounded visual vocabulary coverage.** The bundled concept asset is intentionally bounded, and migration augments it with target-folder labels. Which additional general concepts improve long-tail coverage without reducing score separation must be driven by held-out corpus results, not corpus-specific control flow.
- **Video keyframe extraction.** Still pending the `ffmpeg-next` FFI vs metadata-only decision.
- **Cold-start loose-file UX.** Bundles are now preserved safely at the target root, but loose files still have no honest category when the target exposes zero leaves. Should a future `--bootstrap` mode accept a user-authored taxonomy, or is an explicit unclassified list the right permanent behavior?
- **Inline-YAKE maintenance.** A few hundred lines of keyword extraction inline is cheap but adds a small maintenance item. Acceptable until a mainstream crate crosses the DL threshold.

## References in repo

- `ARCHITECTURE.md` — crate graph, port traits, privacy model, bundle atomicity.
- `CLAUDE.md` — coding standards, invariants, dependency policy, rename policy, safety model.
- `README.md` — roadmap entries, feature flag structure, phase-by-phase ship plan.
- `crates/tidyup-domain/src/migration.rs` — `ClassifierConfig`, `ScoreWeights`, `FolderProfile`, `OrganizationType`.
