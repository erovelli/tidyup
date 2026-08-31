# Third-party licences

tidyup is licensed under Apache-2.0 (see `LICENSE`). This file records
third-party material that is **vendored into the repository** as data rather
than pulled in as a cargo dependency.

Cargo dependencies are governed separately by `deny.toml` and verified in CI by
`cargo xtask deny`. `cargo-deny` only inspects the crate graph, so it does not
and cannot see the assets listed here — which is exactly why they are tracked
manually.

---

## Fonts

Both faces are embedded in the `tidyup-desktop` binary and injected as `data:`
URLs at startup (`crates/tidyup-ui/src/main.rs`). The copies in the repository
are Latin subsets of the upstream variable fonts, clamped to the 400–700 weight
range; see `assets/brand/fonts/README.md` for the exact subsetting command.

### Inter

- **Copyright** 2020 The Inter Project Authors (https://github.com/rsms/inter)
- **Licence** SIL Open Font License, Version 1.1
- **Full text** `assets/brand/fonts/OFL-Inter.txt`
- **Vendored as** `assets/brand/fonts/Inter-Variable-latin.woff2`

### Manrope

- **Copyright** 2018 The Manrope Project Authors (https://github.com/sharanda/manrope)
- **Licence** SIL Open Font License, Version 1.1
- **Full text** `assets/brand/fonts/OFL-Manrope.txt`
- **Vendored as** `assets/brand/fonts/Manrope-Variable-latin.woff2`

### A note on OFL-1.1 and `deny.toml`

OFL-1.1 is not on the `deny.toml` allowlist (Apache-2.0 / MIT / BSD / ISC /
Unicode / Zlib / CC0 / MPL). That is deliberate and the allowlist is left
unchanged: it governs cargo dependencies, and widening it to accommodate two
font files would also relax the policy for every future crate. Fonts are
tracked here instead.

Two OFL obligations are worth stating explicitly, since both are easy to breach
by accident:

1. **The licence must ship with the font.** Keeping `OFL-*.txt` next to the
   `.woff2` files satisfies this. Don't prune them as "unused files".
2. **The subsets are Modified Versions.** Removing glyphs/axes and converting
   the upstream fonts to WOFF2 are modifications under OFL-1.1. The vendored
   Inter and Manrope headers declare no Reserved Font Names, so their family
   names may be retained. Recheck that fact before updating either upstream
   font; any Reserved Font Name in a future source must not be used for its
   modified subset.
