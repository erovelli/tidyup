// Proper-noun model names (CLAP, SigLIP) appear throughout. Avoid
// doc_markdown lint noise.
#![allow(clippy::doc_markdown)]

//! Default scan-mode audio taxonomy — natural-language category descriptions
//! used to classify audio via CLAP cross-modal cosine similarity.
//!
//! See [`crate::image_taxonomy`] for the rationale: cross-modal contrastive
//! encoders (CLAP for audio, SigLIP for images) need natural-language
//! captions, not keyword lists.

use crate::taxonomy::TaxonomyEntry;

/// Audio-modality taxonomy. Descriptions are natural-language captions in
/// CLAP's joint audio-text space.
#[must_use]
pub fn default_audio_taxonomy() -> Vec<TaxonomyEntry> {
    vec![
        TaxonomyEntry {
            path: "Music/",
            description: "music or a song with instruments and vocals",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Music/Instrumental/",
            description: "instrumental music with no singing or vocals",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Music/Classical/",
            description: "classical music with orchestra, piano, or string instruments",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Podcasts/",
            description: "a podcast episode or talk-radio show with one or more speakers",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Memos/",
            description: "a personal voice memo or short voice recording",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Audiobooks/",
            description: "an audiobook narration with a single voice reading a long passage",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Lectures/",
            description: "a lecture, lesson, or educational talk",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Calls/",
            description: "a recorded phone call, meeting, or conversation",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Effects/",
            description: "a sound effect, sample, or short non-musical audio clip",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Ambient/",
            description: "ambient sound, environmental noise, or background atmosphere",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Ringtones/",
            description: "a ringtone or notification sound",
            temporal: false,
        },
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn audio_taxonomy_has_entries() {
        assert!(default_audio_taxonomy().len() >= 5);
    }

    #[test]
    fn audio_taxonomy_paths_end_with_slash() {
        for entry in default_audio_taxonomy() {
            assert!(
                entry.path.ends_with('/'),
                "path missing trailing slash: {}",
                entry.path,
            );
        }
    }
}
