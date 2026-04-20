# Design System Document: The Verdant Archive

## 1. Overview & Creative North Star
**Creative North Star: The Digital Arboretum**
This design system moves away from the sterile, industrial nature of traditional file management. Instead, it treats digital organization as a mindful, restorative practice. We are building a "Digital Arboretum"—a space that feels abundant yet orderly, airy yet structured.

To break the "template" look common in utility software, this system utilizes **intentional asymmetry** and **tonal layering**. We avoid rigid, boxy grids in favor of organic groupings and generous whitespace. The interface should feel less like a "tool" and more like an editorial spread in a high-end architectural magazine: clean, purposeful, and premium.

---

## 2. Colors & Surface Philosophy
The palette is a sophisticated "Spring" spectrum, balanced to provide high legibility without the harshness of pure white or high-contrast blacks.

### The "No-Line" Rule
**Borders are strictly prohibited for sectioning.** To define boundaries between the navigation tree, the file grid, and the inspector panel, designers must use background shifts. 
*   **Main Canvas:** `surface` (#f9f9f9)
*   **Sidebar/Navigation:** `surface-container-low` (#f2f4f4)
*   **Active/Focus Areas:** `surface-container-highest` (#dfe3e4)

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

## 5. Components

### Tree Views & Navigation
*   **Structure:** No vertical guide lines. Use 24px of horizontal indentation per level.
*   **Selection:** Active states use a soft capsule shape with `secondary_container` (#bee9ff) and 12px (`md`) rounded corners.

### Action Buttons
*   **Primary:** Gradient fill (`primary` to `primary_dim`), `xl` (1.5rem) roundedness. High-end, pill-shaped.
*   **Secondary:** Ghost style. No background, `on_surface` text. On hover, transition to `surface_container_high`.
*   **Tertiary:** `surface_container_lowest` with a "Ghost Border."

### Cards & File Items
*   **Rule:** Forbid the use of divider lines between list items. Use 8px of vertical spacing (`md`) to separate rows. 
*   **Visuals:** File icons should use soft `secondary` (#0f6784) and `primary` (#48664c) tones rather than harsh multi-color sets.

### Confidence Indicators (Chips)
*   Small, pill-shaped (`full` roundedness) containers. Use the Tier colors defined in Section 2.
*   Typography: `label-sm` in Semi-Bold to ensure the status is readable against the soft background.

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
