# LinkedIn Project — tidyup

Draft copy for the "Add project" form. Plain text; LinkedIn does not render
markdown. Asterisks/bullets are shown as ASCII so they survive copy-paste.

---

## Project name (≤ 255 chars)

```
tidyup — Local-first AI File Organizer (Rust, on-device ONNX embeddings)
```

71 / 255 characters. Leads with the project name, the value prop
("local-first AI"), and the two highest-signal tech keywords for recruiter
search ("Rust", "ONNX").

### Alternates

- `tidyup — Content-Aware File Organizer in Rust (on-device ML, multimodal)`
- `tidyup: Privacy-First AI File Organizer (Rust, ONNX Runtime, BLAKE3)`
- `tidyup — On-Device AI File Organizer with Hexagonal Architecture (Rust)`

---

## Description (≤ 2,000 chars)

```
tidyup is an open-source, CLI-first tool that auto-organizes file systems by
analyzing content — not just filenames. Built in Rust, it runs entirely
on-device: no cloud, no telemetry, no LLM by default.

WHAT IT DOES
Recursively scans a source directory, classifies each file by its semantic
content using on-device embedding models (bge-small-en-v1.5 via ONNX
Runtime), and proposes rename-and-move operations into a target hierarchy.
Nothing moves without explicit review. Every operation is reversible.

HIGHLIGHTS
- Three-tier classification cascade: heuristic (~1ms) -> on-device
  embeddings (~50ms) -> optional LLM fallback. Each tier short-circuits at
  its confidence threshold so most files never touch the expensive path.
- Multimodal: SigLIP for images, CLAP for audio, BGE for text — each in
  disjoint latent spaces with per-modality natural-language taxonomies.
- Privacy by construction: the default binary links zero HTTP clients and
  zero LLM dependencies. `cargo tree` is the proof. Network code only
  enters under explicit feature flags, gated three ways (cargo feature +
  config + per-invocation CLI flag).
- Bundle atomicity: coding projects, photo bursts, and music albums move
  as single units. Same-volume uses a single atomic rename(2); cross-volume
  uses copy-verify-delete with full rollback on any failure.
- Reversibility: every move is backed up; originals are shelved, never
  deleted.

ARCHITECTURE
Hexagonal layout enforced by the crate graph: zero-dep `domain` -> port
traits in `core` -> swappable impl crates (storage-sqlite, embeddings-ort,
inference-*) -> `pipeline` -> `app` -> thin frontend adapters (`cli`,
`ui`). Plug-and-play seams for frontends and inference backends — adding
either is "implement two traits", with no service-layer refactor.

TECH
Rust 1.90, ONNX Runtime, SQLite, BLAKE3, Tokio, Dioxus (UI), optional
mistralrs + candle (LLM fallback). CI runs clippy::pedantic +
clippy::nursery with warnings-as-errors, plus cargo-deny license checks
and a feature-matrix build.
```

~1,950 / 2,000 characters.

---

## Top 5 skills

Picked for LinkedIn recruiter-search volume + accurate fit:

1. **Rust (Programming Language)** — primary stack, hottest systems-language
   keyword.
2. **Machine Learning** — embeddings, contrastive retrieval, model
   selection.
3. **Software Architecture** — hexagonal layering, port/adapter seams,
   crate-graph-enforced dependency direction.
4. **Artificial Intelligence (AI)** — broader-search complement to ML;
   recruiters often filter on this term specifically.
5. **Open Source Software** — signals collaborative-development experience
   and is a searchable filter.

### Swap candidates (if you want to dial in a specific role)

- For **infra / systems** roles: replace "Open Source Software" with
  **Systems Programming**.
- For **applied ML / NLP** roles: replace "Artificial Intelligence (AI)"
  with **Natural Language Processing (NLP)**.
- For **DX / tooling** roles: replace "Open Source Software" with
  **Command-Line Interface (CLI)**.

---

## Media (optional, recommended)

Up to 50 attachments. Strongest options, in priority order:

1. **GitHub repo link** — `https://github.com/erovelli/tidyup` (primary CTA;
   LinkedIn will render a card with description).
2. **Architecture diagram** — screenshot/export of the crate layering from
   `ARCHITECTURE.md` (clearly shows hexagonal design at a glance).
3. **CLI demo screenshot/GIF** — `tidyup migrate <src> <tgt> --dry-run`
   output showing classification proposals + confidence scores.
4. **UI screenshot** — the Dioxus desktop review surface, if it's in a
   demoable state.
5. **README.md link** — supplemental; the repo card already covers this.

If only one slot is used, make it the GitHub repo link.
