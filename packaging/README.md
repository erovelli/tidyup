# Packaging

Metadata for distributing the desktop UI. Signing and notarization remain
backlog items (see the roadmap in `README.md`); what is here is the manifest
layer those steps will build on.

## Linux — freedesktop entry

`tidyup.desktop` gives the desktop UI a launcher entry instead of leaving users
with a bare binary. To install for the current user after
`cargo build --release -p tidyup-ui`:

```bash
install -Dm755 target/release/tidyup-desktop ~/.local/bin/tidyup-desktop
install -Dm644 packaging/tidyup.desktop      ~/.local/share/applications/tidyup.desktop
for s in 256 512; do
  install -Dm644 "assets/brand/icon-$s.png" \
    "$HOME/.local/share/icons/hicolor/${s}x${s}/apps/tidyup.png"
done
update-desktop-database ~/.local/share/applications 2>/dev/null || true
gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor 2>/dev/null || true
```

`StartupWMClass` in the entry has to match the `WM_CLASS` the toolkit actually
sets, which GTK derives from the binary name rather than from anything in this
file. Confirm it against a running build rather than assuming:

```bash
xwininfo -root -tree | grep -i tidyup
# 0x200003 "Tidyup": ("tidyup-desktop" "Tidyup-desktop")  1280x860+0+0
#                     ^^^^^^^^^^^^^^^^ this is StartupWMClass
```

If the binary is ever renamed, this value changes with it, and a stale value
shows up as a generic icon on the running window rather than as an error.

## macOS and Windows — bundle metadata

`Dioxus.toml` at the workspace root carries the bundle identifier, category,
descriptions, and icon set for `dx bundle`. Two things worth knowing:

- The `[application] name` there feeds `dioxus_cli_config::app_title()`, which
  is populated **only** under the `dx` CLI. The window title is also set in
  `crates/tidyup-ui/src/main.rs` because a plain `cargo run` never sees it. Both
  are load-bearing; neither is redundant.
- macOS ignores the window icon set at runtime and reads `icon.icns` from the
  built `.app` bundle, so the bundle path is the only way the mark appears in
  the Dock there.

## Release scope

`.github/workflows/release.yml` publishes the default **CLI** binary only. The
desktop UI is still built from source; packaging it means signing (macOS
notarization, Windows Authenticode), which is tracked as a roadmap item rather
than wired here.
