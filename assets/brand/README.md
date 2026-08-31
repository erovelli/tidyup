# Brand assets

Single source of truth for tidyup's visual identity. Both the desktop UI and the
project README read from here, so there is exactly one copy of the mark.

> **These are placeholders.** Every file in this directory is a stand-in that
> follows the `DESIGN.md` palette so the wiring can be verified end to end. They
> are meant to be replaced wholesale, not iterated on.

## Files

| File | Purpose |
| ---- | ------- |
| `mark.svg` | Source of truth. Square mark, 256×256 viewBox. |
| `logo.svg` | Horizontal lockup (mark + wordmark) for the README header. |
| `icon-256.png` | **Embedded into the desktop binary** as the window icon (`crates/tidyup-ui/src/main.rs`). |
| `icon-512.png` | HiDPI window icon; freedesktop `hicolor` install size. |
| `icon-1024.png` | Rasterization source for `icon.icns`. |
| `icon.ico` | Windows bundle icon (16/32/48/256 multi-resolution). |
| `icon.icns` | macOS `.app` bundle icon. |

## Constraints the mark has to satisfy

1. **Legible at 16px.** It appears in Linux taskbars and Windows title bars at
   that size. Check the 16×16 frame inside `icon.ico` before calling it done.
2. **Survives dark OS chrome.** macOS ignores `Config::with_icon` entirely and
   reads the icon from the `.app` bundle, but on Linux and Windows the mark is
   composited against whatever the user's panel colour is. The current
   placeholder solves this with a filled `primary` container shape; a mark drawn
   as bare `primary` strokes on transparency would vanish on a dark panel.
3. **Sits on `--surface` (`#f9f9f9`) in-app.** That is also the window
   background colour set at launch, so the two should not fight.

## Regenerating the rasters

There is deliberately no build-time rasterization step — the icons are checked
in so that a plain `cargo build` needs no image toolchain. After editing
`mark.svg`, regenerate with any SVG rasterizer, e.g.:

```bash
cd assets/brand
for s in 256 512 1024; do rsvg-convert -w $s -h $s mark.svg -o icon-$s.png; done
magick icon-1024.png -define icon:auto-resize=256,48,32,16 icon.ico
png2icns icon.icns icon-1024.png        # or `iconutil -c icns` from an .iconset on macOS
```

Keep the filenames stable: `icon-256.png` is referenced by `include_bytes!` and
the rest by `Dioxus.toml` and `packaging/tidyup.desktop`.

## Fonts

`fonts/` holds the two self-hosted webfaces the desktop UI embeds. See
`fonts/README.md` for the subsetting command and licence terms.
