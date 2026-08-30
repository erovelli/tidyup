# tidyup-pipeline

Classification pipelines for [tidyup](https://github.com/erovelli/tidyup):

- **scan mode** — semantic embeddings with optional LLM refinement against a fixed or user-supplied taxonomy.
- **migration mode** — target-aware routing through dynamic folder-label prototypes plus learned content centroids.
- **shared semantic artifacts** — one cached image/audio embedding feeds placement, conservative complete-link clustering, and grounded rename retrieval; persisted vectors use compact blobs and expire through the storage prune path.
- **grounded naming** — exact metadata/text remains preferred; non-text images retrieve scored concepts from a versioned data asset and dynamic hierarchy labels. No LLM authors filenames.
- **identity indexing** — a port-neutral streaming BLAKE3 pass persists every enumerable regular file before semantic work and records run-scoped discovery/indexing failures.
- **calibration mechanics** — raw-score identity calibration is the shipped default; Platt fitting/ECE helpers exist, but no fitted action-specific parameters ship yet.

Structural markers are used only to preserve opaque bundles such as codebases.
Unmarked directories are transparent to the recursive scanner, so their files
remain loose and markers nested at any depth establish bundles at their own
roots. Loose siblings may form atomic photo bursts, music
albums, sequenced document series, descriptive cross-format collections, or
conservative complete-link visual collections. Numeric/date-only and generic
screenshot prefixes cannot anchor a semantic collection. EXIF bursts are
resolved before visual-neighbor grouping. File extensions and
keywords do not route loose files to destinations; MIME/extension knowledge is
limited to choosing a compatible extractor or modality backend.

Text routing combines name/path/MIME evidence with extracted content using the
same normalized `ScoreWeights` in scan and migration. Image/audio routing stays
inside SigLIP/CLAP latent spaces and uses modality-specific taxonomies or
dynamic target-folder prototypes. Missing signals redistribute weight instead
of depressing the attainable score. Grounded collection-member names are
case-insensitively uniquified before review. Move-only basename collisions are
held rather than assigned ungrounded names.

Built atop the port traits from `tidyup-core`. Has no knowledge of which concrete backends are wired in.
