// Proper-noun model names (SigLIP, CLAP) appear throughout. Avoid the
// doc_markdown lint noise.
#![allow(clippy::doc_markdown)]

//! Default scan-mode image taxonomy — natural-language category descriptions
//! used to classify images via SigLIP cross-modal cosine similarity.
//!
//! # Why a separate taxonomy
//!
//! SigLIP's text tower expects natural-language captions, not keyword lists.
//! The default text taxonomy in [`crate::taxonomy::default_taxonomy`] is
//! tuned for `bge-small`-style sentence-embedding similarity — terse keyword
//! soups work best there. SigLIP needs phrasings like "a photograph of a
//! person", "a screenshot of a desktop", "a scanned document".
//!
//! # Folder mapping
//!
//! Each entry routes into a leaf of the same default folder hierarchy used by
//! the text taxonomy (`Photos/`, `Screenshots/`, etc.) so the resulting
//! `ChangeProposal` is shape-compatible with the text-classified ones.

use crate::taxonomy::TaxonomyEntry;

/// Image-modality taxonomy. Descriptions are natural-language captions — they
/// compare meaningfully against image embeddings in SigLIP's joint space.
#[must_use]
pub fn default_image_taxonomy() -> Vec<TaxonomyEntry> {
    vec![
        TaxonomyEntry {
            path: "Photos/",
            description: "a photograph of a person, place, or scene from a camera",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Photos/Portraits/",
            description: "a portrait photograph of one or more people",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Photos/Landscapes/",
            description: "a landscape photograph of natural scenery, mountains, beach, or forest",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Photos/Food/",
            description: "a photograph of food, a meal, or a dish on a plate",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Photos/Pets/",
            description: "a photograph of a pet animal, dog, cat, bird, or fish",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Screenshots/",
            description: "a screenshot of a computer or phone screen showing a user interface",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Documents/Scans/",
            description: "a scanned document, page of text, receipt, or printed paper",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Creative/Design/",
            description:
                "a graphic design, illustration, logo, icon, or vector art rendered digitally",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Creative/Memes/",
            description: "a meme image with overlaid text or a funny caption",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Maps/",
            description: "a map, geographic image, satellite image, or topographic chart",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Charts/",
            description: "a chart, graph, plot, or data visualization with axes and labels",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Diagrams/",
            description: "a diagram, schematic, flowchart, or technical illustration",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Photos/Events/",
            description: "a photograph from an event, party, wedding, concert, or social gathering",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Photos/Travel/",
            description:
                "a vacation or travel photograph from a tourist destination or foreign country",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Photos/Family/",
            description: "a candid family photograph of relatives or children",
            temporal: true,
        },
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn image_taxonomy_has_entries() {
        assert!(default_image_taxonomy().len() >= 10);
    }

    #[test]
    fn image_taxonomy_paths_end_with_slash() {
        for entry in default_image_taxonomy() {
            assert!(
                entry.path.ends_with('/'),
                "path missing trailing slash: {}",
                entry.path,
            );
        }
    }

    #[test]
    fn image_taxonomy_descriptions_are_sentences() {
        for entry in default_image_taxonomy() {
            // Sanity-check: SigLIP wants natural-language captions, not
            // keyword lists. A heuristic threshold is "starts with `a `".
            let lower = entry.description.to_ascii_lowercase();
            assert!(
                lower.starts_with("a ") || lower.starts_with("an "),
                "image-taxonomy description should be a caption: {}",
                entry.description,
            );
        }
    }
}
