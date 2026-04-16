//! `mistralrs`-backed [`TextBackend`](tidyup_core::inference::TextBackend) and
//! [`VisionBackend`](tidyup_core::inference::VisionBackend).
//! Runtime accelerator detection.
//!
//! Registers itself with id `"mistralrs"` — `tidyup-config` references this id
//! in `inference.backends` to opt in without recompiling.

// TODO: MistralRsText, MistralRsVision, register_with(&mut registry).
