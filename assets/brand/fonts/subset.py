from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools import subset

# Standard Google-Fonts "latin" subset, plus the arrows the UI actually renders
# (RunRow prints " → " between source and target roots) and the box-drawing-free
# typographic punctuation theme.css relies on.
UNICODES = (
    "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,"
    "U+2000-206F,U+2074,U+20AC,U+2122,U+2190-2193,U+2212,U+2215,U+FEFF,U+FFFD"
)

def build(src, dst, pins):
    font = TTFont(src)
    font = instancer.instantiateVariableFont(font, pins, inplace=True, updateFontNames=True)
    # fontTools' lazy `gvar` dict raises KeyError for glyphs that carry no delta
    # set (e.g. ZWJ). Materialize it with empty deltas so the subsetter's
    # dict-comprehension over the retained glyph set cannot miss a key.
    if "gvar" in font:
        gvar = font["gvar"]
        gvar.variations = {g: gvar.variations.get(g, []) for g in font.getGlyphOrder()}

    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "liga", "calt", "ccmp", "locl", "mark", "mkmk", "rlig"]
    opts.desubroutinize = False
    opts.drop_tables += ["DSIG"]
    opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]  # keep licence + family names (OFL)
    opts.notdef_outline = True
    sub = subset.Subsetter(options=opts)
    sub.populate(unicodes=subset.parse_unicodes(UNICODES))
    sub.subset(font)
    font.flavor = "woff2"
    font.save(dst)
    print(dst, "axes:", [(a.axisTag, a.minValue, a.maxValue) for a in font["fvar"].axes] if "fvar" in font else "static")

# The (min, default, max) triples pin the DEFAULT instance to 400 as well as
# clamping the range. Manrope's upstream default is 200 (ExtraLight); without an
# explicit default the subset's name table reports the family as "Manrope
# ExtraLight" and any consumer that matches on the internal family name — rather
# than on the @font-face declaration — gets the wrong face.
#
# theme.css uses weights 400 / 500 / 600 / 700 only — clamp the wght axis to that
# span so the delta sets carry no data the UI can never request. Inter's optical
# size axis is pinned to its text default (14); the UI has no display-size usage
# that would benefit from the display end of that axis.
build("Inter-var.ttf",   "Inter-Variable-latin.woff2",   {"opsz": 14, "wght": (400, 400, 700)})
build("Manrope-var.ttf", "Manrope-Variable-latin.woff2", {"wght": (400, 400, 700)})
