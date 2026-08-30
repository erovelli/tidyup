# tidyup-extract

Content extractors for [tidyup](https://github.com/erovelli/tidyup). Default crate features provide bounded plain-text reads, bounded PDF body extraction, and image dimensions/EXIF. `audio` adds ID3/Vorbis metadata; non-default `excel` adds bounded Excel/ODS cell extraction. Each implementation is feature-gated so custom binaries can choose their parser surface. The shipped CLI/UI enable text, PDF, image, and audio; they do not currently register `ExcelExtractor`.

On macOS image builds, a local Apple Vision helper is compiled only when a compatible `xcrun`/Swift/macOS 14 SDK is available. Failure degrades to no OCR. Runtime OCR is configurable, capped at 20 MiB by default, and restricted to plausible screenshot names; ordinary photos still receive dimensions/EXIF extraction without launching OCR. PDF/Excel whole-document parsers defer inputs above 128 MiB rather than risking unbounded memory.

Implements the `ContentExtractor` port from `tidyup-core`; each extractor exposes a stable capability identity for run manifests.
