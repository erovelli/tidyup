# Design System Document: The Verdant Archive

## 1. Overview & Creative North Star
**Creative North Star: The Digital Arboretum**
This design system moves away from the sterile, industrial nature of traditional file management. Instead, it treats digital organization as a mindful, restorative practice. We are building a "Digital Arboretum"—a space that feels abundant yet orderly, airy yet structured.

To break the "template" look common in utility software, this system utilizes **intentional asymmetry** and **tonal layering**. We avoid rigid, boxy grids in favor of organic groupings and generous whitespace. The interface should feel less like a "tool" and more like an editorial spread in a high-end architectural magazine: clean, purposeful, and premium.

---


> **Loading the stylesheet.** `assets/theme.css` is compiled into the binary via
> `include_str!` and rendered through `document::Style`. It is deliberately not
> referenced with `asset!`, which resolves to a hashed URL that only exists once
> the `dx` CLI has bundled assets — a plain `cargo run`/`cargo build` binary got
> an unresolvable href and fell back to the webview's default stylesheet, so the
> whole app rendered as unstyled serif HTML with a non-animating spinner.

## 2. Colors & Surface Philosophy
The palette is a sophisticated "Spring" spectrum, balanced to provide high legibility without the harshness of pure white or high-contrast blacks.

### The "No-Line" Rule
**Borders are strictly prohibited for sectioning.** To define boundaries between the navigation tree, the file grid, and the inspector panel, designers must use background shifts. 
*   **Main Canvas:** `surface` (#f9f9f9)
*   **Sidebar/Navigation:** `surface-container-low` (#f2f4f4)
*   **Active/Focus Areas:** `surface-container-highest` (#dfe3e4)

*Carve-out:* the rule bans lines for **sectioning**. A solid outline used purely as an **interaction-state** affordance — e.g. the 2px `primary` outline on a selected review-tree row — is permitted, since it marks focus/selection rather than dividing layout.

### Surface Hierarchy & Nesting
Treat the UI as physical layers of fine paper. 
*   **Nesting:** A file card (`surface-container-lowest`) should sit atop a gallery view (`surface-container-low`). The subtle shift in hex value provides all the "edge" required.
*   **Glass & Gradient Rule:** For floating menus or "Spring" modals, use a backdrop-blur (12px–20px) with `surface` at 80% opacity. 
*   **Signature Gradients:** For primary CTAs (e.g., "Upload" or "New Folder"), use a linear gradient from `primary` (#48664c) to `primary_dim` (#3c5a41) at a 135° angle to add depth and "soul."

### Confidence Tiers (Functional Color)
*   **High Confidence:** `primary_container` (#c8ebca) / `on_primary_container` text.
*   **Medium Confidence:** `tertiary_container` (#fbf195) / `on_tertiary_container` text.
*   **Low Confidence:** `error_container` (#fd795a) / `on_error_container` text. *Note: We use a soft peach/orange tone to signal caution without the alarmism of standard red.*

---

## 3. Typography
We employ a dual-font strategy to balance editorial authority with functional clarity.

*   **Display & Headlines (Manrope):** Chosen for its modern, geometric construction. Use `display-lg` and `headline-md` for folder names or "Empty State" messaging to create an authoritative, premium feel.
*   **Body & Labels (Inter):** The workhorse. Inter provides exceptional legibility at small sizes (`body-sm` or `label-md`) required for file metadata and breadcrumbs.

> **Implementation note (not yet wired):** the build does not bundle these webfonts — `theme.css` names `Manrope`/`Inter` but ships no `@font-face` rule or font assets, so the desktop UI currently falls back to the system sans-serif stack (`-apple-system`, `Segoe UI`, …). The Manrope/Inter pairing is the intended design; self-hosting the two families (to keep the desktop build network-silent) is the step that makes it render.

**Visual Hierarchy Tip:** Use `on_surface_variant` (#5b6061) for secondary metadata (date modified, file size) to ensure the primary filename (`on_surface`) remains the focal point.

---

## 4. Elevation & Depth
In this system, elevation is a product of light and shadow, not lines.

*   **The Layering Principle:** Avoid shadows on static elements. Achieve "lift" through color: Place a `surface-container-lowest` card on a `surface-container` background.
*   **Ambient Shadows:** For "floating" elements like context menus or dragged files, use: 
    *   `box-shadow: 0 12px 32px -4px rgba(47, 51, 52, 0.06);`
    *   This uses a tinted version of `on_surface` at a very low opacity to mimic natural morning light.
*   **The "Ghost Border":** For accessibility in high-glare environments, use a 1px border of `outline_variant` (#afb3b3) at **15% opacity**.

---

## 4b. Brand Mark & Window Identity

The mark lives in `assets/brand/` at the workspace root — one canonical copy read
by the desktop binary, the README, and the packaging manifests alike. The files
currently checked in are **placeholders** following this palette; see
`assets/brand/README.md` for the regeneration commands.

*   **Application display name is `Tidyup`.** The `tidyup-desktop` binary name is a developer-facing handle that disambiguates it from the `tidyup` CLI at the `cargo run` level; users should never see it. The window title, the freedesktop entry, and the bundle manifests all say `Tidyup`.
*   **Set the title in code, not the manifest.** dioxus-desktop falls back to the literal `"Dioxus App"` whenever `dioxus_cli_config::app_title()` is unset, and it is always unset under the plain `cargo run` that `README.md` documents — `Dioxus.toml` only feeds that value under the `dx` CLI. Same for the icon: with `window_icon` unset, dioxus-desktop substitutes *its own* logo.
*   **The mark must survive foreign chrome.** In-app it sits on `surface` (#f9f9f9), but on Linux and Windows the OS composites it against a panel colour we do not control. Draw it as a filled container shape rather than bare `primary` strokes on transparency, which vanish on a dark panel. macOS ignores window icons entirely and reads the `.icns` from the app bundle.
*   **Legibility floor is 16px.** That is the size Linux taskbars and Windows title bars use. Detail that does not resolve there is decoration, not identity.
*   **Window geometry:** 1280×860 default, 960×640 minimum. The floor is a layout constraint, not taste — the review diff draws connectors in the gutter between two columns and they overlap below that width.
*   **Paint the host surface.** The window background is set to `surface` at launch so the webview does not flash white before the stylesheet paints.

---

## 5. Components

### Tree Views & Navigation
*   **Structure:** No vertical guide lines. Use 24px of horizontal indentation per level.
*   **Selection (sidebar nav):** Active nav items use a soft capsule shape with `secondary_container` (#bee9ff) and 12px (`md`) rounded corners (`.nav-item.active`).
*   **Selection & decisions (Review / Proposed-Structure tree):** Rows use 6px (`sm`) corners. The *selected* row is marked with a 2px `primary` outline (not a fill); an *approved* row fills with `primary_container` (#c8ebca) and a *rejected* row with `error_container`. This is the app's primary tree surface, so it — not the nav rule above — governs file/folder selection.

### Action Buttons
*   **Primary:** Gradient fill (`primary` to `primary_dim`), `xl` (1.5rem) roundedness. High-end, pill-shaped.
*   **Secondary:** Ghost style. No background, `on_surface` text. On hover, transition to `surface_container_high`.
*   **Tertiary:** `surface_container_lowest` with a "Ghost Border."
*   **Destructive / Danger:** Solid `error_container` (#fd795a) fill with `on_error_container` text, same pill geometry as the others. Reserved for irreversible-feeling actions (e.g. the **Rollback** button on the Runs page). Keep the soft peach tone consistent with the Confidence-Tier `error_container` usage — destructive, not panic-inducing.

### Cards & File Items
*   **Rule:** Forbid the use of divider lines between list items. Use 8px of vertical spacing (`md`) to separate rows. 
*   **Visuals:** File icons should use soft `secondary` (#0f6784) and `primary` (#48664c) tones rather than harsh multi-color sets.
*   **Inline validation:** Invalid editable filenames keep the editor in place, use a 2px `error_container` interaction outline, and show a compact `error_container` message directly below the field. Validation must happen before submit; do not defer a duplicate-name error to an execute-time toast.

### Complete Organization Plan
*   **One decision surface:** Loose proposals and atomic bundles belong in the same review page. Do not force users through disconnected review screens when `ReviewHandler::review_all` can present the complete plan together.
*   **Diff semantics:** Always keep both current storage and proposed destination visible. Distinguish move-only and rename-and-move changes, and show every member rename before an atomic collection can be approved.
*   **Atomic collections:** Approve/reject controls operate on the collection, never individual members. A semantic collection may expose editable label and member basenames; structural bundles remain immutable and binary.
*   **Honest counts:** “Files indexed” comes from indexing progress/report state. Proposal, collection, unclassified, and failure counts are separate concepts and must not be substituted for it.

### Confidence Indicators (Chips)
*   Small, pill-shaped (`full` roundedness) containers. Use the Tier colors defined in Section 2.
*   Typography: `label-sm` in Semi-Bold to ensure the status is readable against the soft background.

### Power-User Gates (Settings)
*   **Purpose:** Surface opt-in capabilities (for example, the optional LLM-reranker toggle) without ever *recommending* them. These live on the Settings page as their own card, never on the primary scan/migrate flow.
*   **Honest disabled states:** When a gate is unavailable (feature not compiled, or config bool unset), render the control **disabled** with a plain-language status line explaining exactly which gate is missing and how to satisfy it — never hide the control silently, and never let it appear actionable when it isn't. The control only becomes enabled when *all* upstream gates are satisfied.
*   **State, not persistence:** A per-session toggle reflects runtime activation only; it does not write config. Use the Primary button style when active, Secondary when inactive, so the current state reads at a glance.

---

## 6. Do’s and Don’ts

### Do
*   **Do** embrace the "Abundance of Space." If you think there is enough padding, add 8px more.
*   **Do** use `md` (12px) corners for large containers and `lg` (16px) for hero elements.
*   **Do** use subtle transitions (200ms ease-out) for all hover states to maintain the "mindful" aesthetic.

### Don't
*   **Don't** use 100% black (#000000) for text. Use `on_surface` (#2f3334).
*   **Don't** use standard 1px borders to separate the sidebar from the main content. Use a background color shift to `surface-container-low`.
*   **Don't** clutter the view. If a file has 10 metadata points, show only the 2 most important; hide the rest in an inspector panel.
