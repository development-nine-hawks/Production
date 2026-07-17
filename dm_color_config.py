"""
Shared DataMatrix module-color configuration.

Single source of truth for both label-generation paths (the raster/OpenCV
path in cdp_engine.draw_auth_block_opencv and the vector/ReportLab PDF path
in app.draw_dm) so neither can drift out of sync with the other.

Per the DM generation/decode audit: decode (zxingcpp/pylibdmtx) and the
Step2b+/RANSAC homography correction operate on luminance and on coordinate
geometry already resolved upstream, never on the actual module color — so
changing dark-module colors here only affects rendering, not decodability,
as long as every dark color stays on the correct side of the fixed 128
grayscale threshold used downstream (app.py, cdp_engine.py generate_cropped_dm
and _scale_dm_bw).
"""

import colorsys
import hashlib

# "bw"      -> pure black/white, byte-identical to pre-colorization behavior.
# "colored" -> light modules stay pure white; each dark module gets its own
#              color from dark_color_for_module (per-cell, deterministic).
DM_COLOR_MODE = "colored"

# RGB tuples (0-255). DM_DARK_COLOR is kept only as the last-resort fallback
# used by dark_color_for_module if hue resampling can't clear the luminance
# ceiling. DM_LIGHT_COLOR is no longer used for rendering — light modules
# are always rendered as plain white — but stays defined/validated in case
# something downstream still references it.
DM_DARK_COLOR = (25, 25, 112)     # midnight blue
DM_LIGHT_COLOR = (255, 250, 205)  # lemon chiffon

# dark_color_for_module tuning: fixed lightness/saturation for the HSL color
# it generates — only hue varies per (row, col), so cells read as distinct
# hues rather than distinct shades of near-black.
_DARK_LIGHTNESS = 0.22
_DARK_SATURATION = 0.85
_DARK_LUMINANCE_CEILING = 90
_MAX_RESAMPLE_ATTEMPTS = 8


def dm_luminance(rgb) -> float:
    """Rec.709 relative luminance for an (R, G, B) 0-255 tuple."""
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hash_hue(row: int, col: int, seed, attempt: int) -> float:
    """Deterministic pseudo-random hue in [0, 1) from (row, col, seed, attempt)."""
    key = f"{row}:{col}:{seed}:{attempt}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    val = int.from_bytes(digest[:4], "big")
    return (val % 360) / 360.0


def dark_color_for_module(row: int, col: int, seed=None) -> tuple:
    """Deterministic, visually-distinct dark color for one DM module cell.

    Hue is derived from a hash of (row, col, seed), so the same label
    renders identically every time (no per-run randomness). Lightness is
    fixed low and saturation fixed high, so cells read as distinct hues
    rather than shades of near-black. If a sampled hue's luminance doesn't
    clear _DARK_LUMINANCE_CEILING (checked via dm_luminance, same check
    applied to DM_DARK_COLOR above), lightness is nudged down and the hue
    is resampled — this can never erode the downstream 128 decode
    threshold margin.
    """
    lightness = _DARK_LIGHTNESS
    for attempt in range(_MAX_RESAMPLE_ATTEMPTS):
        hue = _hash_hue(row, col, seed, attempt)
        r, g, b = colorsys.hls_to_rgb(hue, lightness, _DARK_SATURATION)
        rgb = (round(r * 255), round(g * 255), round(b * 255))
        if dm_luminance(rgb) < _DARK_LUMINANCE_CEILING:
            return rgb
        lightness *= 0.8
    return DM_DARK_COLOR  # deterministic, already-validated fallback


_dark_lum = dm_luminance(DM_DARK_COLOR)
_light_lum = dm_luminance(DM_LIGHT_COLOR)
assert _dark_lum < 90, (
    f"DM_DARK_COLOR luminance {_dark_lum:.1f} too high — leaves too little "
    f"margin below the downstream 128 decode/render threshold"
)
assert _light_lum > 180, (
    f"DM_LIGHT_COLOR luminance {_light_lum:.1f} too low — leaves too little "
    f"margin above the downstream 128 decode/render threshold"
)
