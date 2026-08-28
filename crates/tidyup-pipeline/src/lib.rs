//! Classification pipelines built atop `tidyup-core` ports.
//!
//! - [`scan`]      — semantic embeddings with optional LLM refinement against a taxonomy.
//! - [`migration`] — target-aware: learns folder profiles from an existing hierarchy.
//! - [`profiler`]  — `FolderProfile` construction + centroid caching.
//! - [`scanner`]   — target tree walk + `OrganizationType` detection.
//! - [`naming`]    — filename sanitization + rendering from proposals.
//! - [`yake`]      — inlined keyword extraction for the rename cascade.

pub mod calibration;
pub mod clustering;
pub mod hashing;
pub mod migration;
pub mod naming;
pub mod profiler;
pub mod scan;
pub mod scanner;
pub mod text_util;
pub mod yake;
