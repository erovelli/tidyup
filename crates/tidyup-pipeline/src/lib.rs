//! Classification pipelines built atop `tidyup-core` ports.
//!
//! - [`scan`]      — tiered cascade (heuristics -> embeddings -> LLM) against a taxonomy.
//! - [`migration`] — target-aware: learns folder profiles from an existing hierarchy.
//! - [`profiler`]  — `FolderProfile` construction + centroid caching.
//! - [`scanner`]   — target tree walk + `OrganizationType` detection.
//! - [`heuristics`] — Tier 1 MIME + keyword + date patterns.
//! - [`naming`]    — filename sanitization + rendering from proposals.
//! - [`yake`]      — inlined keyword extraction for the rename cascade.

pub mod heuristics;
pub mod migration;
pub mod naming;
pub mod profiler;
pub mod scan;
pub mod scanner;
pub mod yake;
