# Self-hosted webfonts

`DESIGN.md` §3 specifies a Manrope (display) / Inter (body) pairing. These are
the two faces the desktop UI embeds, so the design system actually renders
instead of falling back to the system sans stack.

## Why self-hosted and not a CDN

A `@import url(fonts.googleapis.com/…)` would make the webview reach the network
on every launch. The desktop UI is not covered by the CLI's network-silence
invariant (`cargo xtask check-privacy` asserts that for `tidyup-cli` and only
LLM-silence for `tidyup-ui`), but an application whose first README line is "a
local-first file organizer that never phones home" should not open a socket to
render its own text. The faces are embedded in the binary and injected as
`data:` URLs at startup — see `crates/tidyup-ui/src/main.rs`.

## What is here

| File | Size | Axes |
| ---- | ---- | ---- |
| `Inter-Variable-latin.woff2` | ~28 KB | `wght` 400–700 (`opsz` pinned to 14) |
| `Manrope-Variable-latin.woff2` | ~21 KB | `wght` 400–700 |

Variable rather than eight static files: `theme.css` uses exactly four weights
(400, 500, 600, 700), which two variable faces cover in two HTTP-less blobs.
The `wght` axis is clamped to 400–700 and the glyph set is subset to Latin plus
the arrows the UI actually draws (`RunRow` renders `source → target`), which is
what keeps both files under 30 KB.

## Regenerating

`subset.py` is the exact script used, kept for reproducibility. It needs
`fonttools[woff]` and `brotli`. Point it at the upstream variable TTFs:

```bash
pip install 'fonttools[woff]' brotli
curl -LO 'https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf'
curl -LO 'https://raw.githubusercontent.com/google/fonts/main/ofl/manrope/Manrope%5Bwght%5D.ttf'
mv 'Inter[opsz,wght].ttf' Inter-var.ttf && mv 'Manrope[wght].ttf' Manrope-var.ttf
python3 subset.py
```

If you widen `theme.css` beyond the 400–700 range, widen the clamp in
`subset.py` to match — a weight outside the axis range is synthesised by the
renderer and looks noticeably worse than a real master.

## Licence

Both families are under the **SIL Open Font License 1.1** — see `OFL-Inter.txt`
and `OFL-Manrope.txt`, reproduced verbatim as the licence requires.

Note that OFL-1.1 is *not* on the `deny.toml` allowlist (Apache-2.0 / MIT / BSD
/ ISC / Unicode / Zlib / CC0 / MPL). That policy governs **cargo dependencies**,
and `cargo-deny` never sees these files because they are vendored assets rather
than crates — so CI is unaffected either way. The allowlist is left alone on
purpose: widening it would relax the rule for every future *crate* too. Font
licensing is tracked here and in `THIRD-PARTY-LICENSES.md` instead.

OFL reserves font names on modified versions. The subset keeps the original
family names (a subset is not a modification under the licence's definition), so
do not rename the families in `subset.py` output.
