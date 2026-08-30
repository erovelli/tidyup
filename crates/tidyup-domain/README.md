# tidyup-domain

Pure domain types shared across the [tidyup](https://github.com/erovelli/tidyup) workspace. Zero I/O, zero async, and no dependencies on other tidyup crates. Forms the stable contract between every other crate.

The current contract includes atomic `BundleProposal`s (including editable-but-atomic `SemanticCollection`s), calibrated-or-raw classifier configuration, `FileProcessingRecord` lifecycle accounting, versioned `CapabilityManifest` run provenance, semantic-artifact identities, and hash-bearing reversible change records. Changes here are deliberate compatibility decisions because every other workspace layer consumes these shapes.
