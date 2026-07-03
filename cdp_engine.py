"""
PhoneCDP Core Engine — Pattern Generation & Verification

Self-contained module with all CDP logic. No external PhoneCDP imports.
"""

import numpy as np
import cv2
from datetime import datetime
import os
from config import BORDER_CELL_SIZE
import math
import logging
_crop_logger = logging.getLogger("batch_runner")

import zxingcpp

# ── CDP Discrimination Flags ─────────────────────────────────────────────
# Toggle each flag independently to test contribution of each change.
# True = change active, False = original behaviour restored for that change.
CDP_FLAG_MOIRE_FINE_BANDS    = False  # Point 1: add ultra-fine moire bands
CDP_FLAG_MOIRE_NORM_DIVISOR  = False  # Point 2: lower normalization divisor
CDP_FLAG_CORR_FINE_BLOCKS    = False  # Point 3: add 4px/8px block sizes
CDP_FLAG_CORR_COARSE_CAP     = False  # Point 4: cap coarse block scores
# ─────────────────────────────────────────────────────────────────────────

# ==========================================================================
# CONSTANTS
# ==========================================================================


PATTERN_SIZE = (512, 512)

# distance can be changed independently without inflating/deflating ring radii.
BLOCK_SIZE = 16

# M1 fiducial marker constants — ported from cdp_engine_main.py unchanged.
# Ring geometry is derived from FIDUCIAL_MARKER_SIZE // 2 so placement distance
# (FIDUCIAL_MARKER_OFFSET) can be changed without rescaling the ring radii.
FIDUCIAL_MARKER_SIZE   = 64
FIDUCIAL_MARKER_OFFSET = 34
CENTER_MARKER = True
_MHF         = FIDUCIAL_MARKER_SIZE // 2                   # 32 — ring geometry base
_RING_BG_R   = _MHF - 1                                    # 31 — background circle radius
_RING_OUTER  = int(round(20 * _MHF / 24))                  # 27 — outer ring (all non-BR)
_RING_MID    = int(round(13 * _MHF / 24))                  # 17 — BL mid ring
_RING_MID2   = int(round(10 * _MHF / 24))                  # 13 — TR inner ring
_RING_INNER  = int(round( 6 * _MHF / 24))                  #  8 — BL inner ring
_RING_THICK  = int(round( 3 * _MHF / 24))                  #  4 — draw thickness

# Label-aware detection: aspect ratios within this range are treated as
# "near-square" (likely the CDP pattern itself). Anything outside is
# treated as a potential label card wrapping the pattern.
LABEL_ASPECT_RANGE = (0.75, 1.33)
# When markers can't be found in a label, crop the bottom fraction to
# search for the pattern geometrically.
LABEL_PATTERN_BOTTOM_FRACTION = 0.50

# Verification weights
# Moire is the most discriminative test — counterfeits lose grating fidelity.
# Correlation (PRNG block) can score high on quality counterfeits, so weight reduced.
WEIGHT_MOIRE = 0.65
WEIGHT_COLOR = 0.10
WEIGHT_CORRELATION = 0.10
WEIGHT_GRADIENT = 0.15

# Thresholds — calibrate after testing genuine vs counterfeit prints
THRESHOLD_AUTHENTIC = 0.65
THRESHOLD_SUSPICIOUS = 0.45


# ==========================================================================
# PATTERN GENERATION
# ==========================================================================

def generate_frequency_modulated_grating(width, height, base_freq, mod_freq, mod_depth):
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    freq_modulation = base_freq * (1 + mod_depth * np.sin(2 * np.pi * mod_freq * X / width))
    phase = np.cumsum(freq_modulation, axis=1) * 2 * np.pi / width
    grating = np.sin(phase)
    return ((grating + 1) / 2 * 255).astype(np.uint8)


def generate_prng_macro_pattern(width, height, seed, block_size=16):
    rng = np.random.RandomState(seed)
    blocks_x = math.ceil(width / block_size)
    blocks_y = math.ceil(height / block_size)
    block_values = rng.randint(0, 256, size=(blocks_y, blocks_x)).astype(np.uint8)
    pattern = np.repeat(np.repeat(block_values, block_size, axis=0), block_size, axis=1)
    return pattern[:height, :width]


def add_rgb_perturbations(pattern_gray, seed=42, intensity=25, block_size=16):
    rng = np.random.RandomState(seed + 1000)
    h, w = pattern_gray.shape
    pattern_rgb = cv2.cvtColor(pattern_gray, cv2.COLOR_GRAY2RGB)
    blocks_x = (w + block_size - 1) // block_size
    blocks_y = (h + block_size - 1) // block_size
    for i in range(blocks_y):
        for j in range(blocks_x):
            y1 = i * block_size
            x1 = j * block_size
            y2 = min(y1 + block_size, h)
            x2 = min(x1 + block_size, w)
            for ch in range(3):
                shift = rng.randint(-intensity, intensity)
                pattern_rgb[y1:y2, x1:x2, ch] = np.clip(
                    pattern_rgb[y1:y2, x1:x2, ch].astype(np.int16) + shift, 0, 255
                ).astype(np.uint8)
    return pattern_rgb



def draw_dm_border(canvas, cell_size):
    h, w = canvas.shape[:2]
    is_rgb = len(canvas.shape) == 3
    black = (0, 0, 0) if is_rgb else 0
    white = (255, 255, 255) if is_rgb else 255
    
    cells_x = w // cell_size
    cells_y = h // cell_size

    # Draw top alternating
    for i in range(cells_x):
        color = black if i % 2 == 0 else white
        x1, y1 = i * cell_size, 0
        x2, y2 = (i + 1) * cell_size, cell_size
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

    # Draw right alternating
    for i in range(cells_y):
        color = black if i % 2 == 0 else white
        x1, y1 = w - cell_size, i * cell_size
        x2, y2 = w, (i + 1) * cell_size
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

    # Draw left solid
    cv2.rectangle(canvas, (0, 0), (cell_size, h), black, -1)

    # Draw bottom solid
    cv2.rectangle(canvas, (0, h - cell_size), (w, h), black, -1)
    
    return canvas


# ── Corner fiducial markers (white disc + black ring) ──────────────────────
# Four high-contrast markers at the pattern corners give 4 point correspondences
# for a perspective homography — enough to rectify camera warp, which the 2-point
# DM-center similarity transform cannot. Drawn in BOTH generate_pattern and
# regenerate_reference so they appear identically in capture and reference; their
# regions are neutralized before scoring (see _blank_fiducials) so they never
# bias the discrimination tests.
FIDUCIAL_INSET  = 40   # marker center offset from each edge (in 512x512 space)
FIDUCIAL_RING_R = 22   # black outer-disc radius
FIDUCIAL_DISC_R = 13   # white inner-disc radius (the detection target)


def _fiducial_centers(w, h):
    """Canonical marker centers, order TL, TR, BR, BL (clockwise)."""
    s = min(w, h) / 512.0
    off = FIDUCIAL_INSET * s
    return [(off, off), (w - off, off), (w - off, h - off), (off, h - off)]


def add_fiducial_markers(pattern, marker_color=0):
    """Draw 4 corner fiducials (white disc on a black ring) in-place; order TL,TR,BR,BL.

    `marker_color` is accepted for backward compatibility but ignored — the
    markers are always black-ring/white-disc for maximum, color-independent
    detectability against the busy pattern.
    """
    h, w = pattern.shape[:2]
    is_rgb = len(pattern.shape) == 3
    black = (0, 0, 0) if is_rgb else 0
    white = (255, 255, 255) if is_rgb else 255
    s = min(w, h) / 512.0
    ring_r = max(3, int(round(FIDUCIAL_RING_R * s)))
    disc_r = max(2, int(round(FIDUCIAL_DISC_R * s)))
    for cx, cy in _fiducial_centers(w, h):
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(pattern, c, ring_r, black, -1)
        cv2.circle(pattern, c, disc_r, white, -1)
    return pattern


def _blank_fiducials(img):
    """Return a copy with the 4 marker regions flattened to the image median, so
    the fiducials contribute no signal to the moire/correlation tests (keeps
    scoring identical for labels printed with or without markers)."""
    out = img.copy()
    h, w = out.shape[:2]
    s = min(w, h) / 512.0
    r = int(round(FIDUCIAL_RING_R * s)) + 2
    if out.ndim == 2:
        fill = int(np.median(out))
    else:
        fill = tuple(int(np.median(out[:, :, c])) for c in range(out.shape[2]))
    for cx, cy in _fiducial_centers(w, h):
        cv2.circle(out, (int(round(cx)), int(round(cy))), r, fill, -1)
    return out


def fine_align_via_fiducials(crop_512, debug=None):
    """Fine-alignment correction using the 4 corner fiducial markers.

    Runs AFTER Step 2b's coarse DM-corner homography and BEFORE Step 5
    scoring.  Locates each physical ring+disc fiducial in the coarse 512×512
    crop by template matching, derives a median translation offset, and
    applies it so the PRNG pattern lands on the correct block-grid phase.

    Policy (see design doc):
      >= 3 of 4 fiducials cleared the score threshold → apply correction
       < 3                                            → return crop unchanged

    Args:
        crop_512 : uint8 ndarray (H, W) or (H, W, 3) — the coarse crop.
        debug    : optional dict; filled with per-fiducial results if provided.

    Returns:
        (corrected_crop, applied: bool)
    """
    SEARCH_HALF = 30                     # ± px around each canonical centre
    TMPL_R      = FIDUCIAL_RING_R + 4   # template half-side → (2*TMPL_R+1)² px
    SCORE_THR   = 0.40                  # min TM_CCOEFF_NORMED to accept
    MIN_DETECT  = 3                     # need at least this many to apply

    h, w = crop_512.shape[:2]
    gray = (cv2.cvtColor(crop_512, cv2.COLOR_BGR2GRAY)
            if crop_512.ndim == 3 else crop_512)

    # Build synthetic ring+disc template matching add_fiducial_markers() exactly
    T = 2 * TMPL_R + 1
    tmpl = np.full((T, T), 128, dtype=np.uint8)
    cv2.circle(tmpl, (TMPL_R, TMPL_R), FIDUCIAL_RING_R, 0,   -1)  # black ring
    cv2.circle(tmpl, (TMPL_R, TMPL_R), FIDUCIAL_DISC_R, 255, -1)  # white disc

    canon_centers = _fiducial_centers(w, h)
    slot_labels   = ['TL', 'TR', 'BR', 'BL']

    accepted = []   # dicts for fiducials that cleared the threshold
    rejected = []   # dicts for those that didn't

    for lbl, (cx, cy) in zip(slot_labels, canon_centers):
        cx_i, cy_i = int(round(cx)), int(round(cy))

        # Extract padded search region (clamped to image boundaries)
        sx0 = max(0, cx_i - SEARCH_HALF - TMPL_R)
        sy0 = max(0, cy_i - SEARCH_HALF - TMPL_R)
        sx1 = min(w, cx_i + SEARCH_HALF + TMPL_R + 1)
        sy1 = min(h, cy_i + SEARCH_HALF + TMPL_R + 1)
        patch = gray[sy0:sy1, sx0:sx1]

        if patch.shape[0] < T or patch.shape[1] < T:
            rejected.append({'label': lbl, 'score': 0.0, 'reason': 'patch_too_small'})
            continue

        res = cv2.matchTemplate(patch.astype(np.float32),
                                tmpl.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        max_val = float(max_val)

        # Template centre in crop coordinates
        det_cx = sx0 + max_loc[0] + TMPL_R
        det_cy = sy0 + max_loc[1] + TMPL_R

        entry = {
            'label':    lbl,
            'canon':    (cx_i, cy_i),
            'detected': (det_cx, det_cy),
            'offset':   (cx_i - det_cx, cy_i - det_cy),  # correction direction
            'score':    max_val,
        }
        if max_val >= SCORE_THR:
            accepted.append(entry)
        else:
            entry['reason'] = 'below_threshold'
            rejected.append(entry)

    # Compute median correction from accepted detections
    applied = False
    corrected = crop_512
    dx = dy = 0.0

    if len(accepted) >= MIN_DETECT:
        dx = float(np.median([e['offset'][0] for e in accepted]))
        dy = float(np.median([e['offset'][1] for e in accepted]))
        M  = np.float32([[1, 0, dx], [0, 1, dy]])
        corrected = cv2.warpAffine(crop_512, M, (w, h),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REFLECT_101)
        applied = True
        print(f"[FINE_ALIGN] correction applied  dx={dx:+.1f} dy={dy:+.1f}  "
              f"({len(accepted)}/4 fiducials)")
    else:
        print(f"[FINE_ALIGN] correction SKIPPED  only {len(accepted)}/4 fiducials "
              f"detected (need {MIN_DETECT})")

    if debug is not None:
        debug.update(
            fa_accepted=accepted,
            fa_rejected=rejected,
            fa_dx=dx, fa_dy=dy,
            fa_applied=applied,
            fa_n_detected=len(accepted),
        )

    return corrected, applied


def _sample_mean(gray, cx, cy, radii, n=24):
    """Mean intensity sampled on circles of the given radii around (cx,cy)."""
    h, w = gray.shape[:2]
    vals = []
    for rr in radii:
        for k in range(n):
            ang = 2.0 * np.pi * k / n
            x = int(round(cx + rr * np.cos(ang)))
            y = int(round(cy + rr * np.sin(ang)))
            if 0 <= x < w and 0 <= y < h:
                vals.append(float(gray[y, x]))
    return float(np.mean(vals)) if vals else 0.0


def detect_fiducials(crop_gray, search_frac=0.28):
    """Locate the 4 corner fiducials in a roughly-aligned crop.

    For each corner, finds white inner-disc candidates in a search window, keeps
    only those encircled by a dark ring (the marker signature — rejects the
    pattern's own bright pixels), and takes the one nearest the expected corner.
    Returns a float32 (4,2) array of centroids in order TL,TR,BR,BL — or None if
    any corner's marker can't be found (e.g. legacy labels without fiducials).
    """
    if crop_gray.ndim == 3:
        crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)
    h, w = crop_gray.shape[:2]
    s = min(w, h) / 512.0
    r_disc = FIDUCIAL_DISC_R * s
    ring_thick = max(4.0, (FIDUCIAL_RING_R - FIDUCIAL_DISC_R) * s)
    disc_area = np.pi * r_disc ** 2
    a_lo, a_hi = disc_area * 0.12, disc_area * 6.0
    win = max(8, int(min(h, w) * search_frac))
    pts = []
    for ex, ey in _fiducial_centers(w, h):
        x0, y0 = max(0, int(ex - win)), max(0, int(ey - win))
        x1, y1 = min(w, int(ex + win)), min(h, int(ey + win))
        sub = crop_gray[y0:y1, x0:x1]
        if sub.size == 0:
            return None
        _, th = cv2.threshold(sub, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, -1e18
        for c in cnts:
            a = cv2.contourArea(c)
            if a < a_lo or a > a_hi:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            gx, gy = x0 + M["m10"] / M["m00"], y0 + M["m01"] / M["m00"]
            # Score by marker signature: a bright disc inside a dark ring. The
            # ring annulus is taken just outside THIS blob's own radius so it
            # survives perspective foreshortening. Pick the strongest match
            # (not merely the closest), which robustly rejects pattern texture.
            r_eq = max(r_disc * 0.5, (a / np.pi) ** 0.5)
            center_val = _sample_mean(crop_gray, gx, gy, (0.0, r_eq * 0.45), n=12)
            ring_val = _sample_mean(crop_gray, gx, gy,
                                    (r_eq + 1.0 + 0.35 * ring_thick,
                                     r_eq + 1.0 + 0.75 * ring_thick))
            if center_val < 170 or ring_val > 125:
                continue
            dist = ((gx - ex) ** 2 + (gy - ey) ** 2) ** 0.5
            score = (center_val - ring_val) - 0.10 * dist
            if score > best_score:
                best_score, best = score, (gx, gy)
        if best is None:
            return None
        pts.append(best)
    return np.float32(pts)


def rectify_by_markers(crop_bgr):
    """Perspective-correct a roughly-aligned crop using its 4 corner fiducials.

    Returns (rectified_bgr, True) on success, or (crop_bgr, False) when the 4
    markers can't be found — so callers degrade gracefully to the unrectified
    crop on legacy labels.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    h, w = gray.shape[:2]
    src = detect_fiducials(gray)
    if src is None:
        return crop_bgr, False
    dst = np.float32(_fiducial_centers(w, h))
    try:
        H = cv2.getPerspectiveTransform(src, dst)
        rectified = cv2.warpPerspective(crop_bgr, H, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
    except cv2.error:
        return crop_bgr, False
    return rectified, True


def _find_marker_near(gray, cx, cy, win, r_disc, ring_thick, debug_cands=None):
    """Find the fiducial nearest to (cx,cy) within a +/-win window, full-image
    resolution. Scores candidates by the disc+ring signature and returns the
    strongest match's centroid (gx,gy) or None. Records all candidates into
    debug_cands if provided."""
    h, w = gray.shape[:2]
    x0, y0 = max(0, int(cx - win)), max(0, int(cy - win))
    x1, y1 = min(w, int(cx + win)), min(h, int(cy + win))
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    disc_area = np.pi * r_disc ** 2
    a_lo, a_hi = disc_area * 0.12, disc_area * 6.0
    _, th = cv2.threshold(sub, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1e18
    for c in cnts:
        a = cv2.contourArea(c)
        if a < a_lo or a > a_hi:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        gx, gy = x0 + M["m10"] / M["m00"], y0 + M["m01"] / M["m00"]
        r_eq = max(r_disc * 0.5, (a / np.pi) ** 0.5)
        cval = _sample_mean(gray, gx, gy, (0.0, r_eq * 0.45), n=12)
        rval = _sample_mean(gray, gx, gy,
                            (r_eq + 1.0 + 0.35 * ring_thick, r_eq + 1.0 + 0.75 * ring_thick))
        passed = (cval >= 170 and rval <= 125)
        if debug_cands is not None:
            debug_cands.append((float(gx), float(gy), float(a), float(cval), float(rval), bool(passed)))
        if passed:
            score = (cval - rval) - 0.10 * ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5
            if score > best_score:
                best_score, best = score, (gx, gy)
    return best


def align_crop_from_quad(image, quad, debug=None):
    """ALIGN-THEN-CROP: given the DM-derived pattern quad (4 corners in full-image
    coords, oriented TL,TR,BR,BL to match the reference), predict where the 4
    fiducials are in the FULL image, detect them precisely, then apply ONE
    homography that maps the full photo straight to the canonical PATTERN_SIZE
    crop — alignment and crop in a single resampling.

    Returns (crop_bgr, True) on success, or (None, False) when the 4 markers
    can't be located (legacy labels) so the caller can fall back to the rough crop.
    When `debug` is a dict it is filled with the predicted/searched/detected
    geometry for visualization.
    """
    out_w, out_h = PATTERN_SIZE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    quad = np.float32(quad)

    # Predict marker locations: map canonical pattern square -> observed quad.
    canon_sq = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    try:
        H_pred = cv2.getPerspectiveTransform(canon_sq, quad)
    except cv2.error:
        if debug is not None:
            debug["stage"] = "bad_quad"
        return None, False
    marks_canon = np.float32(_fiducial_centers(out_w, out_h)).reshape(-1, 1, 2)
    pred = cv2.perspectiveTransform(marks_canon, H_pred).reshape(-1, 2)

    # Marker scale + search window derived from the pattern's size in the photo.
    pat_side = float(np.linalg.norm(quad[1] - quad[0]))
    s = max(pat_side, 1.0) / 512.0
    r_disc = FIDUCIAL_DISC_R * s
    ring_thick = max(4.0, (FIDUCIAL_RING_R - FIDUCIAL_DISC_R) * s)
    win = max(12, int(0.16 * pat_side))

    detected, dbg_search = [], []
    for (px, py) in pred:
        cands = []
        m = _find_marker_near(gray, px, py, win, r_disc, ring_thick, cands)
        dbg_search.append((float(px), float(py), int(win), cands, m))
        detected.append(m)

    if debug is not None:
        debug.update(quad=quad, pred=pred, search=dbg_search,
                     n_found=sum(1 for m in detected if m is not None))

    if any(m is None for m in detected):
        if debug is not None:
            debug["stage"] = "markers_not_found"
        return None, False

    src = np.float32(detected)
    dst = np.float32(_fiducial_centers(out_w, out_h))
    try:
        H = cv2.getPerspectiveTransform(src, dst)
        crop = cv2.warpPerspective(image, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    except cv2.error:
        if debug is not None:
            debug["stage"] = "homography_failed"
        return None, False

    if debug is not None:
        debug.update(stage="ok", detected=src, H=H)
    return crop, True



def align_crop_by_dm_corners(image, raw_results, debug=None):
    """ALIGN-THEN-CROP using the two DataMatrix codes' exact corners.

    Accepts the already-decoded raw_results list from extract_seed_from_image()
    (each item has .data bytes and .corners_cv np.float32 4×2 in y-down coords)
    so no second decode pass is needed.  Their 8 corners → one homography that
    maps the full photo straight to the canonical PATTERN_SIZE crop (alignment +
    crop fused, one resampling).

    Returns (crop_bgr, True) on success, or (None, False) so the caller can fall
    back to the rough crop.
    """
    out_w, out_h = PATTERN_SIZE

    if not raw_results or len(raw_results) < 2:
        print(f"[STEP2B] FAIL decode<2: raw_results has "
              f"{len(raw_results) if raw_results else 0} entries")
        if debug is not None:
            debug["stage"] = "decode<2"
        return None, False

    # Build (text, corners_cv) list from already-decoded results.
    dms_by_text = {}
    for r in raw_results:
        try:
            text = r.data.decode("utf-8").strip()
        except Exception:
            text = ""
        if text and text not in dms_by_text:
            dms_by_text[text] = (text, r.corners_cv)
    dms = list(dms_by_text.values())
    if len(dms) < 2:
        print(f"[STEP2B] FAIL decode<2: only {len(dms)} unique DM text(s) after dedup: "
              f"{list(dms_by_text.keys())}")
        if debug is not None:
            debug["stage"] = "decode<2"
        return None, False
    print(f"[STEP2B] {len(dms)} DM(s) available: {[d[0] for d in dms]}")

    # Classify Top (share_a) vs Right (share_b) DM by content (rotation-invariant).
    b32 = re.compile(r'^[A-Z2-7]{4}$')
    top = right = None
    for i, j in itertools.combinations(range(len(dms)), 2):
        ti, tj = dms[i][0], dms[j][0]
        if not (b32.match(ti) and b32.match(tj)):
            continue
        if recombine_seed_from_dm(ti, tj) is not None:
            top, right = dms[i], dms[j]; break
        if recombine_seed_from_dm(tj, ti) is not None:
            top, right = dms[j], dms[i]; break
    if top is None:
        texts   = [d[0] for d in dms]
        b32_ok  = [t for t in texts if b32.match(t)]
        b32_bad = [t for t in texts if not b32.match(t)]
        print(f"[STEP2B] FAIL classify: no valid share_a/share_b pair. "
              f"b32-valid={b32_ok} b32-invalid={b32_bad}")
        if debug is not None:
            debug["stage"] = "classify_fail"
        return None, False
    print(f"[STEP2B] classified: top={top[0]!r} "
          f"centroid=({top[1].mean(0)[0]:.1f},{top[1].mean(0)[1]:.1f})  "
          f"right={right[0]!r} "
          f"centroid=({right[1].mean(0)[0]:.1f},{right[1].mean(0)[1]:.1f})")

    # Global orientation (canonical +x/+y in image space) from the DM centroids.
    tc, rc = top[1].mean(0), right[1].mean(0)
    vec_layout = np.array([_RIGHT_DM_CX - _TOP_DM_CX, _RIGHT_DM_CY - _TOP_DM_CY], float)
    vec_img = (rc - tc).astype(float)
    cdst, odst = np.linalg.norm(vec_layout), np.linalg.norm(vec_img)
    if odst < 1e-6:
        print(f"[STEP2B] FAIL centroid-coincide: DM centroids are identical "
              f"(dist={odst:.2e})")
        if debug is not None:
            debug["stage"] = "centroid_coincide"
        return None, False
    angle_deg = math.degrees(math.atan2(float(vec_img[1]), float(vec_img[0])))
    print(f"[STEP2B] orientation: centroid-vec=({vec_img[0]:.1f},{vec_img[1]:.1f}) "
          f"dist={odst:.1f}px angle={angle_deg:.1f}°")
    vln, vin = vec_layout / cdst, vec_img / odst
    cos_t = float(vln @ vin); sin_t = float(vln[0]*vin[1] - vln[1]*vin[0])
    ur = np.array([cos_t, sin_t]); ud = np.array([-sin_t, cos_t])

    def _order(q):
        """Sort 4 corners into TL/TR/BR/BL using orientation axes ur/ud.
        Returns (ordered_float32, None) on success, or (None, reason_str) on
        failure — same safety style as _sd_quad() elsewhere."""
        ctr = q.mean(0); r = (q - ctr) @ ur; d = (q - ctr) @ ud
        s, df = r + d, r - d
        ordered = np.float32([q[np.argmin(s)], q[np.argmax(df)],
                               q[np.argmax(s)], q[np.argmin(df)]])
        # Reject if any two slots received the same input corner (degenerate assignment)
        unique_pts = len({(round(float(p[0])), round(float(p[1]))) for p in ordered})
        if unique_pts < 4:
            return None, f"duplicate-slot ({unique_pts} unique of 4)"
        # Reject near-zero-area quads (corners collapsed onto a line)
        x, y = ordered[:, 0], ordered[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        if area < 100.0:
            return None, f"area={area:.1f}px² < 100"
        # Reject non-convex (bowtie) quads — cross products must all have the same sign
        signs = []
        for i in range(4):
            e1 = ordered[(i + 1) % 4] - ordered[i]
            e2 = ordered[(i + 2) % 4] - ordered[(i + 1) % 4]
            signs.append(float(e1[0] * e2[1] - e1[1] * e2[0]))
        if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
            return None, f"non-convex (cross-products={[round(v, 1) for v in signs]})"
        return ordered, None

    top_quad,   top_reason   = _order(top[1])
    right_quad, right_reason = _order(right[1])
    if top_quad is None:
        print(f"[STEP2B] FAIL _order top DM ({top[0]!r}): {top_reason}")
        if debug is not None:
            debug["stage"] = "order_degenerate"
        return None, False
    if right_quad is None:
        print(f"[STEP2B] FAIL _order right DM ({right[0]!r}): {right_reason}")
        if debug is not None:
            debug["stage"] = "order_degenerate"
        return None, False

    def _quad_area(q):
        x, y = q[:, 0], q[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    print(f"[STEP2B] _order OK: top area={_quad_area(top_quad):.0f}px²  "
          f"right area={_quad_area(right_quad):.0f}px²")

    # ── [STEP2B-LOCAL] Diagnostic: does the global axis assign corners the same  ──
    # ── way as each DM's own local orientation?  Pure logging — no logic change. ──
    def _corner_indices_under(q, ur_ax, ud_ax):
        """Return [TL_idx, TR_idx, BR_idx, BL_idx] into q for the given axes."""
        ctr = q.mean(0)
        r_ax = (q - ctr) @ ur_ax
        d_ax = (q - ctr) @ ud_ax
        s_ax, df_ax = r_ax + d_ax, r_ax - d_ax
        return [int(np.argmin(s_ax)), int(np.argmax(df_ax)),
                int(np.argmax(s_ax)), int(np.argmin(df_ax))]

    def _local_axes_minrect(q):
        """Local orientation axes from cv2.minAreaRect on q (4 pts)."""
        rect = cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))
        angle_rad = math.radians(rect[2])
        ur_l = np.array([math.cos(angle_rad), math.sin(angle_rad)], float)
        if ur_l[0] < 0:
            ur_l = -ur_l          # keep pointing into +x half-plane
        ud_l = np.array([-ur_l[1], ur_l[0]], float)
        # Signed angle from global ur to local ur_l (positive = CCW in image)
        dot  = float(np.clip(ur_l @ ur, -1.0, 1.0))
        diff = math.degrees(math.acos(dot))
        if float(ur_l[0]*ur[1] - ur_l[1]*ur[0]) < 0:
            diff = -diff
        return ur_l, ud_l, diff

    _SLOT = ["TL", "TR", "BR", "BL"]
    for _dm_label, _dm_corners, _dm_text in [
            ("top",   top[1],   top[0]),
            ("right", right[1], right[0])]:
        _ur_l, _ud_l, _ang_diff = _local_axes_minrect(_dm_corners)
        _g_idx = _corner_indices_under(_dm_corners, ur,    ud)
        _l_idx = _corner_indices_under(_dm_corners, _ur_l, _ud_l)
        _bad   = [(s, _g_idx[s], _l_idx[s])
                  for s in range(4) if _g_idx[s] != _l_idx[s]]
        if _bad:
            print(f"[STEP2B-LOCAL] {_dm_label} ({_dm_text!r}): "
                  f"MISMATCH {len(_bad)}/4 slots  "
                  f"local_vs_global={_ang_diff:+.1f}°")
            for _s, _gi, _li in _bad:
                _gp = _dm_corners[_gi]; _lp = _dm_corners[_li]
                print(f"[STEP2B-LOCAL]   {_SLOT[_s]}: "
                      f"global→#{_gi}({_gp[0]:.0f},{_gp[1]:.0f})  "
                      f"local→#{_li}({_lp[0]:.0f},{_lp[1]:.0f})  "
                      f"Δ={np.linalg.norm(_gp - _lp):.0f}px")
        else:
            print(f"[STEP2B-LOCAL] {_dm_label} ({_dm_text!r}): "
                  f"agree all 4 slots  local_vs_global={_ang_diff:+.1f}°")
        _go = [_dm_corners[i] for i in _g_idx]
        _lo = [_dm_corners[i] for i in _l_idx]
        print("[STEP2B-LOCAL]   global: " +
              "  ".join(f"{_SLOT[s]}=({_go[s][0]:.0f},{_go[s][1]:.0f})"
                        for s in range(4)))
        print("[STEP2B-LOCAL]   local:  " +
              "  ".join(f"{_SLOT[s]}=({_lo[s][0]:.0f},{_lo[s][1]:.0f})"
                        for s in range(4)))
    # ── end [STEP2B-LOCAL] ────────────────────────────────────────────────────────

    # Canonical DM corners expressed in the PATTERN frame ([0,PATTERN_SIZE]).
    def _to_pat(X, Y):
        return [(X - _pat_x) / _pat_w * out_w, (Y - _pat_y) / _pat_h * out_h]

    def _rect4(x, y, w, h):
        return [_to_pat(x, y), _to_pat(x + w, y), _to_pat(x + w, y + h), _to_pat(x, y + h)]

    src = np.float32(list(top_quad) + list(right_quad))
    dst = np.float32(_rect4(_top_dm_x, _top_dm_y, _top_dm_w, _top_dm_h) +
                     _rect4(_right_x, _right_y, _right_w, _right_h))
    print(f"[STEP2B] findHomography: 8 src→dst point pairs (top 4 + right 4)")
    try:
        Hh, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if Hh is None:
            print(f"[STEP2B] FAIL homography_none: findHomography returned None "
                  f"for 8 correspondences")
            if debug is not None:
                debug["stage"] = "homography_none"
            return None, False
        inliers = int(mask.sum()) if mask is not None else -1
        print(f"[STEP2B] homography solved: {inliers}/8 RANSAC inliers  "
              f"det(H)={np.linalg.det(Hh):.4f}")
        crop = cv2.warpPerspective(image, Hh, (out_w, out_h), flags=cv2.INTER_LINEAR)
    except cv2.error as _e:
        print(f"[STEP2B] FAIL homography_failed: cv2.error — {_e}")
        if debug is not None:
            debug["stage"] = "homography_failed"
        return None, False

    print(f"[STEP2B] SUCCESS: warpPerspective → {out_w}×{out_h} crop")
    if debug is not None:
        debug.update(stage="ok", top_quad=top_quad, right_quad=right_quad, H=Hh)
    return crop, True


def _draw_alignment_overlay(image, dbg, disp_w=1100):
    """Annotated full-photo view for the UI: the two detected DM quads (green=Top,
    blue=Right) with corner dots, plus the derived pattern region (red) that gets
    cropped. Downscaled for serving. Always returns an image (even on failure, so
    the user can see what was/wasn't detected)."""
    ov = image.copy()
    H_img, W_img = ov.shape[:2]
    t = max(2, W_img // 600)
    R = max(5, W_img // 200)
    stage = dbg.get("stage")
    if dbg.get("top_quad") is not None and dbg.get("right_quad") is not None:
        for q, col in ((dbg["top_quad"], (0, 255, 0)), (dbg["right_quad"], (255, 0, 0))):
            cv2.polylines(ov, [np.int32(q)], True, col, t)
            for pt, lab in zip(q, ("TL", "TR", "BR", "BL")):
                cv2.circle(ov, (int(pt[0]), int(pt[1])), R, col, -1)
                cv2.putText(ov, lab, (int(pt[0]) + R, int(pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W_img / 2600), col, t)
        Hh = dbg.get("H")
        if Hh is not None:
            try:
                inv = np.linalg.inv(Hh)
                pw, ph = PATTERN_SIZE
                canon = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
                pq = cv2.perspectiveTransform(canon, inv).reshape(-1, 2)
                cv2.polylines(ov, [np.int32(pq)], True, (0, 0, 255), t + 1)
                cx, cy = pq.mean(0)
                cv2.putText(ov, "PATTERN", (int(cx) - 60, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W_img / 2400), (0, 0, 255), t)
            except Exception:
                pass
    label = "DM-corner align OK" if stage == "ok" else f"align FAILED ({stage}) -> rough crop"
    cv2.putText(ov, label, (20, max(40, R * 4)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.9, W_img / 2000), (0, 255, 255), t + 1)
    if W_img > disp_w:
        sc = disp_w / float(W_img)
        ov = cv2.resize(ov, (disp_w, int(H_img * sc)))
    return ov


def generate_pattern(output_dir, seed=None, serial_number="SN-0001", pattern_size=512, block_size=16):
    if seed is None:
        seed = int(np.random.randint(0, 2**31))
    
    w = h = pattern_size

    grating_rng = np.random.RandomState(seed=seed + 2000)
    base_freq = 8 + grating_rng.random() * 6
    mod_freq = 1.5 + grating_rng.random() * 2.5
    mod_depth = 0.2 + grating_rng.random() * 0.3

    grating = generate_frequency_modulated_grating(w, h, base_freq, mod_freq, mod_depth)
    prng = generate_prng_macro_pattern(w, h, seed, block_size=block_size)
    combined = cv2.addWeighted(grating, 0.5, prng, 0.5, 0)
    pattern_rgb = add_rgb_perturbations(combined, seed=seed, intensity=25, block_size=block_size)
    
    add_fiducial_markers(pattern_rgb, marker_color=0)
    canvas = pattern_rgb

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phone_cdp_{timestamp}_{seed}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return {"seed": seed, "filename": filename, "filepath": filepath,
            "serial_number": serial_number, "pattern_size": pattern_size,
            "block_size": block_size}


# ==========================================================================
# VERIFICATION — STEP 0: Pattern Detection
# ==========================================================================

def _find_pattern_contour(gray, min_area_pct=0.05):
    h, w = gray.shape
    image_mean = float(gray.mean())
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Raise the threshold so lighter pattern pixels at the edges are included.
    # Otsu adapts to the whole image and tends to cut too low, shrinking the
    # detected region inward. Shifting up ~20 recovers those edge pixels.
    _, binary = cv2.threshold(blurred, min(int(otsu_val) + 20, 220), 255,
                              cv2.THRESH_BINARY_INV)

    # Try two morphology strategies:
    # 1. Close only (original) — works for standalone patterns
    # 2. Erode then close — breaks thin connections (card borders, text
    #    strokes) that merge separate dark regions into one huge contour
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_passes = [
        cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel),
        cv2.morphologyEx(cv2.erode(binary, erode_kernel), cv2.MORPH_CLOSE, close_kernel),
    ]

    def _score_contours(morphed_img):
        contours, _ = cv2.findContours(morphed_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best, best_sc = None, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w) * min_area_pct:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-10)
            if not (0.5 < aspect < 2.0):
                continue
            if area > (h * w) * 0.85:
                continue
            region = gray[by:by + bh, bx:bx + bw]
            region_mean = float(region.mean())
            region_std = float(region.std())
            if region_mean > image_mean * 0.95:
                continue
            darkness = max(0, image_mean - region_mean)
            squareness_bonus = (1.0 - abs(1.0 - aspect)) * 20
            score = darkness + region_std + squareness_bonus
            if score > best_sc:
                best_sc = score
                best = cnt
        return best

    # Pass 1: close only (works for standalone patterns)
    best_contour = _score_contours(morph_passes[0])
    if best_contour is not None:
        return best_contour

    # Pass 2: erode+close (breaks thin connections in labels).
    # Only run when the image is significantly larger than the expected
    # pattern — for a standalone pattern-sized image, erode would break
    # the pattern into fragments and return a wrong sub-region.
    max_pat = max(PATTERN_SIZE)
    if min(h, w) > max_pat * 1.15:
        best_contour = _score_contours(morph_passes[1])
    else:
        best_contour = None
    if best_contour is not None:
        # The erode shrinks the contour by ~2-3px, losing pattern edge pixels.
        # Dilate back by a LARGER kernel to slightly overshoot — a generous crop
        # (with a few background pixels) is far better than a tight crop that
        # misses pattern edge pixels, because alignment can handle extra pixels
        # but cannot recover missing ones.
        bx, by, bw, bh = cv2.boundingRect(best_contour)
        restore_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        restored = cv2.dilate(morph_passes[1], restore_kernel)
        contours_r, _ = cv2.findContours(restored, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_r:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            # Match the contour that overlaps with the original detection
            if abs(cx - bx) < 10 and abs(cy - by) < 10:
                return cnt
        return best_contour  # fallback to eroded contour if no match

    return None


def _crop_from_contour(image, contour):
    h, w = image.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(contour)

    # Try to fit the contour as a true quadrilateral via polygon approximation.
    # Use convexHull first so approxPolyDP sees the outermost extent of the
    # binary region, then refine with Canny edges for pixel-precise corners.
    hull = cv2.convexHull(contour)
    hull_peri = cv2.arcLength(hull, True)

    def _sd_quad(pts4):
        """Assign TL/TR/BR/BL via sum/diff, then verify all 4 are distinct.

        sum/diff is the standard approach and works correctly for axis-aligned
        and mildly-rotated rectangles.  It is degenerate only at exactly ±45°
        where two corners share the same x+y (or y−x), causing the same point
        to fill two slots.  The uniqueness check catches this and returns None
        so the caller can fall through to a more robust method.
        """
        s = pts4.sum(axis=1)
        d = pts4[:, 1] - pts4[:, 0]
        tl = pts4[np.argmin(s)]
        br = pts4[np.argmax(s)]
        tr = pts4[np.argmin(d)]
        bl = pts4[np.argmax(d)]
        corners = [tl, tr, br, bl]
        # Reject if any two slots received the same point (degenerate at ~45°)
        if len({(round(p[0]), round(p[1])) for p in corners}) < 4:
            return None
        return np.float32(corners)

    src_pts = None
    for eps_frac in [0.02, 0.04, 0.06, 0.08]:
        approx = cv2.approxPolyDP(hull, eps_frac * hull_peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            candidate = _sd_quad(pts)
            if candidate is None:
                continue   # degenerate at ~45°, try next eps or fall through
            # Reject degenerate quads where one side is far shorter than the others.
            # When the top corner of a rotated diamond is lost in thresholding, the
            # contour has a flat cut top and approxPolyDP places a false vertex along
            # that flat edge — giving 4 corners but with one near-zero side.  Catching
            # this here lets the parallelogram fallback below infer the real corner.
            sides = [np.linalg.norm(candidate[i] - candidate[(i + 1) % 4]) for i in range(4)]
            if max(sides) > 0 and min(sides) / max(sides) > 0.30:
                src_pts = candidate
                break

    # Parallelogram rule fallback: when one corner (e.g. the top tip of a rotated
    # diamond) is lost in thresholding, approxPolyDP returns only 3 corners.
    # For a flat printed pattern the 4 corners ARE a parallelogram, so the missing
    # 4th corner = A + C - B  where B is the "diagonal opposite" middle point.
    # We try each of the 3 points as B; the valid candidate is the one that produces
    # a convex hull of exactly 4 points (all distinct, no self-intersection).
    if src_pts is None:
        for eps_frac in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
            approx = cv2.approxPolyDP(hull, eps_frac * hull_peri, True)
            if len(approx) == 3:
                pts3 = approx.reshape(3, 2).astype(np.float32)
                for mid_idx in range(3):
                    other = [pts3[i] for i in range(3) if i != mid_idx]
                    p4 = other[0] + other[1] - pts3[mid_idx]
                    # Inferred point must lie within a 20% margin of the image frame
                    if not (-w * 0.2 <= p4[0] <= w * 1.2 and -h * 0.2 <= p4[1] <= h * 1.2):
                        continue
                    all4 = np.vstack([pts3, p4[np.newaxis]])
                    # Require a proper convex quad (all 4 corners on the hull)
                    hull4 = cv2.convexHull(all4.reshape(-1, 1, 2).astype(np.int32))
                    if len(hull4) != 4:
                        continue
                    candidate = _sd_quad(all4)
                    if candidate is None:
                        continue   # degenerate; try next mid_idx
                    src_pts = candidate
                    break
                if src_pts is not None:
                    break

    if src_pts is not None:
        # Compute output size from the actual quad sides (not a fixed rectangle)
        out_w = int(max(np.linalg.norm(src_pts[1] - src_pts[0]),   # top edge
                        np.linalg.norm(src_pts[2] - src_pts[3])))  # bottom edge
        out_h = int(max(np.linalg.norm(src_pts[3] - src_pts[0]),   # left edge
                        np.linalg.norm(src_pts[2] - src_pts[1])))  # right edge
        out_size = max(out_w, out_h, 64)
        dst_pts = np.float32([
            [0, 0], [out_size - 1, 0],
            [out_size - 1, out_size - 1], [0, out_size - 1]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped = cv2.warpPerspective(image, M, (out_size, out_size),
                                      borderValue=(255, 255, 255))
        return cropped, (bx, by, bw, bh), src_pts

    # Fallback: minAreaRect (handles cases where approxPolyDP can't find 4 corners)
    rect = cv2.minAreaRect(contour)
    box = np.float32(cv2.boxPoints(rect))
    _, (rect_w, rect_h), angle = rect
    if rect_w < rect_h:
        rect_w, rect_h = rect_h, rect_w
    effective_angle = abs(angle) % 90
    if effective_angle > 45:
        effective_angle = 90 - effective_angle
    if effective_angle < 5:
        pad_x, pad_y = int(bw * 0.02), int(bh * 0.02)
        bx = max(0, bx - pad_x)
        by = max(0, by - pad_y)
        bw = min(w - bx, bw + 2 * pad_x)
        bh = min(h - by, bh + 2 * pad_y)
        quad = np.float32([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]])
        return image[by:by + bh, bx:bx + bw], (bx, by, bw, bh), quad
    sorted_pts = sorted(box, key=lambda p: p[1])
    top_pts = sorted(sorted_pts[:2], key=lambda p: p[0])
    bot_pts = sorted(sorted_pts[2:], key=lambda p: p[0])
    src_pts = np.float32([top_pts[0], top_pts[1], bot_pts[1], bot_pts[0]])
    out_size = max(int(rect_w), int(rect_h))
    dst_pts = np.float32([
        [0, 0], [out_size - 1, 0],
        [out_size - 1, out_size - 1], [0, out_size - 1]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (out_size, out_size),
                                  borderValue=(255, 255, 255))
    return cropped, (bx, by, bw, bh), src_pts



def _extract_pattern_from_label(label_image):
    """Extract the CDP pattern from within a label using contour hierarchy.

    Uses RETR_TREE to find nested contours. The 3px alignment border around
    the QR creates a near-perfect-square child contour inside the label.
    This works regardless of label orientation (any rotation).

    Returns (cropped_image, (rel_x, rel_y, rel_w, rel_h)) where the bbox
    is relative to label_image coordinates.
    """
    h, w = label_image.shape[:2]
    gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY) \
        if len(label_image.shape) == 3 else label_image

    # Threshold and find contour hierarchy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(blurred, min(int(otsu_val) + 20, 220), 255,
                              cv2.THRESH_BINARY_INV)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Find the most-square nested contour (the QR alignment border).
    # Must be: a child contour (has parent), near-square (0.85-1.18 aspect),
    # and reasonably sized (> 0.5% of image area).
    best_contour = None
    best_squareness = float('inf')
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        pct = area / (h * w)
        if pct < 0.005 or pct > 0.50:
            continue
        parent = hierarchy[0][i][3]
        if parent < 0:  # must be a child, not top-level
            continue
        cbx, cby, cbw, cbh = cv2.boundingRect(cnt)
        aspect = cbw / (cbh + 1e-10)
        if not (0.85 <= aspect <= 1.18):
            continue
        squareness_err = abs(1.0 - aspect)
        if squareness_err < best_squareness:
            best_squareness = squareness_err
            best_contour = cnt

    if best_contour is not None:
        cbx, cby, cbw, cbh = cv2.boundingRect(best_contour)
        # Use _crop_from_contour for perspective correction -- the phone
        # photo may be tilted and the perspective warp corrects this.
        cropped, bbox, quad = _crop_from_contour(label_image, best_contour)
        return cropped, (cbx, cby, cbw, cbh)

    # Fallback: center square
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return label_image[y1:y1 + side, x1:x1 + side], (x1, y1, side, side)




def _marker_crop(*args, **kwargs):
    return None

def _dm_rect_to_corners_cv(rect, image_height):
    """Return the 4 corners of a pylibdmtx rect in OpenCV (y-down) pixel coordinates.

    pylibdmtx uses a y-up convention; height is negative when the symbol was
    detected in a downward direction (rotated / portrait DMs).  This helper
    handles both signs in a single, branchless conversion that mirrors the
    same logic used in the dm_results_raw reprojection path (lines ~543-550).

    Parameters
    ----------
    rect         : object with .left, .top, .width, .height  (pylibdmtx convention)
    image_height : pixel height of the image the rect lives in (y-up reference)

    Returns
    -------
    np.float32 array of shape (4, 2) — TL, TR, BR, BL in OpenCV y-down coords
    """
    left    = rect.left
    w       = abs(rect.width)
    h_signed = rect.height
    y_top_cv    = image_height - (rect.top + max(0, h_signed))
    y_bottom_cv = image_height - (rect.top + min(0, h_signed))
    return np.float32([
        [left,     y_top_cv],
        [left + w, y_top_cv],
        [left + w, y_bottom_cv],
        [left,     y_bottom_cv],
    ])

def detect_center_marker(gray, expected_center, search_radius, expected_marker_px):
    """
    Find the white circle with black border center marker near expected_center.
    Uses HoughCircles — same approach as corner marker detection.
    Returns (cx, cy) in full image coordinates, or None if not found.
    """
    ih, iw = gray.shape
    ex, ey = int(expected_center[0]), int(expected_center[1])

    # Crop search region
    x1 = max(0, ex - search_radius)
    y1 = max(0, ey - search_radius)
    x2 = min(iw, ex + search_radius)
    y2 = min(ih, ey + search_radius)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        _crop_logger.info("[CENTER] not_found — search region empty")
        return None

    blurred = cv2.GaussianBlur(crop, (9, 9), 2)

    # Expected radius in crop pixels
    min_r = max(3, int(expected_marker_px * 0.4))
    max_r = max(min_r + 2, int(expected_marker_px * 1.6))

    circles = None
    for param2 in [30, 20, 15]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5,
            minDist=min_r * 2,
            param1=100, param2=param2,
            minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            break

    if circles is None:
        _crop_logger.info(
            f"[CENTER] not_found — HoughCircles found nothing "
            f"expected=({ex},{ey}) expected_marker_px={expected_marker_px:.1f} "
            f"min_r={min_r} max_r={max_r}")
        return None

    # Pick circle closest to expected center
    best = None
    best_dist = float('inf')
    for cx, cy, _ in np.round(circles[0]).astype(int):
        full_cx = cx + x1
        full_cy = cy + y1
        dist = np.linalg.norm(
            np.array([full_cx, full_cy]) - np.array([ex, ey]))
        if dist < best_dist:
            best_dist = dist
            best = (full_cx, full_cy)

    _crop_logger.info(
        f"[CENTER] detected at ({best[0]},{best[1]}) "
        f"dist_from_expected={best_dist:.1f} "
        f"expected=({ex},{ey}) "
        f"expected_marker_px={expected_marker_px:.1f}")
    return best


def _ncc_thumbnail(crop_bgr, reference_gray, thumb=128):
    """
    Quick NCC check at thumbnail resolution.
    Returns float in [-1, 1]. Higher = better match to reference.
    Returns -1.0 on any failure.
    """
    try:
        if crop_bgr is None or crop_bgr.size == 0:
            return -1.0
        gray = (cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                if len(crop_bgr.shape) == 3 else crop_bgr)
        cap_t = cv2.resize(gray, (thumb, thumb)).astype(np.float32)
        ref_t = cv2.resize(reference_gray, (thumb, thumb)).astype(np.float32)
        cf = cap_t.flatten() - cap_t.mean()
        rf = ref_t.flatten() - ref_t.mean()
        ncc = float(np.dot(cf, rf) /
                    (np.linalg.norm(cf) * np.linalg.norm(rf) + 1e-10))
        return ncc
    except Exception:
        return -1.0


def detect_and_crop_pattern(image, dm_results_raw=None, dm_shrink_used=1, seed=None, debug_image_name: str = None, block_size=BLOCK_SIZE):
    """
    Detect and crop the CDP pattern directly from DM geometry in the original image.

    Stage A (label-card contour detection and M_label flattening) has been removed.
    DM positions from dm_results_raw are used directly in full-image pixel
    coordinates, eliminating the intermediate 400x600 canvas step that was the
    confirmed root cause of out-of-bounds crop coordinates.

    Classification uses payload content (share_a printed on Top DM, share_b on
    Right DM) rather than aspect ratio, giving angle-independent identification.

    Parameters
    ----------
    image           : BGR image (full resolution capture).
    dm_results_raw  : List of pylibdmtx decode results from extract_seed_from_image.
                      Required. If None or fewer than 2 results, returns None.
    dm_shrink_used  : The shrink factor used to produce dm_results_raw.
                      Rects are multiplied by this to restore full-image pixel
                      coordinates.
    """
    if dm_results_raw is None or len(dm_results_raw) < 2:
        _crop_logger.info("[CROP] dm_results_raw missing or insufficient — cannot crop.")
        return None, (0, 0, 0, 0), False, None

    img_h = image.shape[0]
    _crop_logger.info(f"[CROP] image shape: {image.shape}")

    scale = dm_shrink_used

    # Step 1 — Recover DM corners in full-image OpenCV (y-down) pixel coordinates
    dm_data = []  # list of (corners_cv: np.float32 (4,2), text: str)
    for res in dm_results_raw:
        r = res.rect
        w_full = abs(r.width)
        h_full = abs(r.height)

        # When width is negative, left is the right edge — correct it
        left_full = r.left
        if r.width < 0:
            left_full = left_full - w_full

        # When height is negative, top is the bottom edge — correct it
        top_full = r.top
        if r.height < 0:
            top_full = top_full - h_full

        y_top_cv    = img_h - top_full - h_full
        y_bottom_cv = img_h - top_full
        _crop_logger.info(
            f"[CROP] DM raw rect: left={r.left}, top={r.top}, width={r.width}, height={r.height} | "
            f"corrected: left_full={left_full:.2f}, top_full={top_full:.2f}, "
            f"w_full={w_full:.2f}, h_full={h_full:.2f} | "
            f"y_top_cv={y_top_cv:.2f}, y_bottom_cv={y_bottom_cv:.2f} | "
            f"img_h={img_h}, scale={scale}"
        )

        corners = np.float32([
            [left_full,          y_top_cv   ],
            [left_full + w_full, y_top_cv   ],
            [left_full + w_full, y_bottom_cv],
            [left_full,          y_bottom_cv],
        ])

        try:
            text = res.data.decode('utf-8').strip()
        except Exception:
            text = ''

        dm_data.append((corners, text))

    # Step 2 — Classify Top DM vs Right DM by payload content.
    # Convention (enforced at label generation in app.py):
    #   share_a → Top DM,  share_b → Right DM
    # recombine_seed_from_dm(a, b) returns non-None exactly when a=share_a, b=share_b.
    top_corners = right_corners = None
    classified_by = 'content'

    b32_re = re.compile(r'^[A-Z2-7]{4}$')
    for i, j in itertools.combinations(range(len(dm_data)), 2):
        t_i, t_j = dm_data[i][1], dm_data[j][1]
        if not (b32_re.match(t_i) and b32_re.match(t_j)):
            continue
        if recombine_seed_from_dm(t_i, t_j) is not None:
            top_corners, right_corners = dm_data[i][0], dm_data[j][0]
            break
        if recombine_seed_from_dm(t_j, t_i) is not None:
            top_corners, right_corners = dm_data[j][0], dm_data[i][0]
            break

    if top_corners is None or right_corners is None:
        _crop_logger.info("[CROP] Content classification failed — falling back to y-position sort.")
        classified_by = 'y-sort'
        sorted_dm = sorted(
            dm_data,
            key=lambda d: (d[0][:, 0].max() - d[0][:, 0].min()) * (d[0][:, 1].max() - d[0][:, 1].min()),
            reverse=True,
        )
        c0, c1 = sorted_dm[0][0], sorted_dm[1][0]
        if c0[:, 1].mean() < c1[:, 1].mean():  # smaller OpenCV y = higher on image = Top
            top_corners, right_corners = c0, c1
        else:
            top_corners, right_corners = c1, c0

    # Step 3 — DM bounding box dimensions (for scale) + orientation from diagonal.
    #
    # Scale anchor: the DM long side is 48 modules = 480 canonical units, which equals
    # the pattern side (also 48 modules).  Use it directly so the crop is exactly the
    # right number of pixels — avoiding the quiet-zone ambiguity in the diagonal length.
    #
    # Orientation: the Top→Right diagonal is always 45° below-right in the canonical
    # layout.  Rotating the observed diagonal unit vector by ±45° gives the canonical
    # layout-right (e_r) and layout-down (e_d) directions in image coordinates.
    top_dm_w  = float(top_corners[:, 0].max() - top_corners[:, 0].min())
    top_dm_h  = float(top_corners[:, 1].max() - top_corners[:, 1].min())
    right_dm_w = float(right_corners[:, 0].max() - right_corners[:, 0].min())
    right_dm_h = float(right_corners[:, 1].max() - right_corners[:, 1].min())
    dm_long = max(top_dm_w, top_dm_h, right_dm_w, right_dm_h)   # long side of DM bbox

    _crop_logger.info(
        f"[CROP] class={classified_by} "
        f"top_dm={top_dm_w:.0f}x{top_dm_h:.0f} "
        f"right_dm={right_dm_w:.0f}x{right_dm_h:.0f} "
        f"dm_long={dm_long:.0f}"
    )

    top_dm_center   = top_corners.mean(axis=0)
    right_dm_center = right_corners.mean(axis=0)

    # Layout ratios — derived dynamically from calculate_auth_block_layout()
    # at module load time (see _R_TOP_X etc. globals above).
    R_TOP_X           = _R_TOP_X
    R_TOP_Y           = _R_TOP_Y
    R_RIGHT_X         = _R_RIGHT_X
    R_RIGHT_Y         = _R_RIGHT_Y
    CANONICAL_DM_DIST = _CANONICAL_DM_DIST
    PATTERN_PX        = _PATTERN_PX

    # Get reference for NCC scoring if seed available
    ref_gray = None
    if seed is not None:
        try:
            _, ref_gray = regenerate_reference(seed, block_size=block_size)
        except Exception:
            ref_gray = None

    ih, iw = image.shape[:2]
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3 else image.copy())

    NCC_MIN = 0.05

    # ── LAYER 3: DM geometry (last resort) ─────────────────────────────────
    _crop_logger.info("[CROP] Layer 3 — DM geometry fallback")
    try:
        # The canonical Top→Right vector in layout space
        vec_dm_layout = np.array([_RIGHT_DM_CX - _TOP_DM_CX,
                                   _RIGHT_DM_CY - _TOP_DM_CY], dtype=np.float64)

        # The observed Top→Right vector in image space
        vec_dm_image = (right_dm_center - top_dm_center).astype(np.float64)
        observed_dm_dist = float(np.linalg.norm(vec_dm_image))
        canonical_dm_dist = float(np.linalg.norm(vec_dm_layout))

        if observed_dm_dist < 10:
            _crop_logger.info("[CROP] Layer 3 failed — DM centers too close")
            return None, (0, 0, 0, 0), False, None

        l3_scale = observed_dm_dist / canonical_dm_dist

        # Build rotation matrix from canonical→image space
        vec_dm_layout_norm = vec_dm_layout / canonical_dm_dist
        vec_dm_image_norm  = vec_dm_image  / observed_dm_dist

        cos_t = float(np.dot(vec_dm_layout_norm, vec_dm_image_norm))
        sin_t = float(vec_dm_layout_norm[0] * vec_dm_image_norm[1]
                      - vec_dm_layout_norm[1] * vec_dm_image_norm[0])

        unit_right = np.array([cos_t,  sin_t], dtype=np.float32)
        unit_down  = np.array([-sin_t, cos_t], dtype=np.float32)
        _crop_logger.info(
            f"[CROP] Layer 3 rotation debug: cos_t={cos_t:.4f}, sin_t={sin_t:.4f}, "
            f"unit_right={unit_right}, unit_down={unit_down} | "
            f"top_dm_center={top_dm_center}, right_dm_center={right_dm_center} | "
            f"_TOP_DM_CX={_TOP_DM_CX:.2f}, _TOP_DM_CY={_TOP_DM_CY:.2f} | "
            f"_pat_x={_pat_x:.2f}, _pat_y={_pat_y:.2f}, _pat_w={_pat_w:.2f}, _pat_h={_pat_h:.2f}"
        )

        # Compute all 4 pattern corners by transforming canonical offsets
        # from top_dm_center through the rotation+scale
        def _tx(rx, ry):
            return (top_dm_center
                    + (rx * l3_scale) * unit_right
                    + (ry * l3_scale) * unit_down).astype(np.float32)

        TL = _tx(_pat_x           - _TOP_DM_CX, _pat_y           - _TOP_DM_CY)
        TR = _tx(_pat_x + _pat_w  - _TOP_DM_CX, _pat_y           - _TOP_DM_CY)
        BR = _tx(_pat_x + _pat_w  - _TOP_DM_CX, _pat_y + _pat_h  - _TOP_DM_CY)
        BL = _tx(_pat_x           - _TOP_DM_CX, _pat_y + _pat_h  - _TOP_DM_CY)

        pattern_center = ((TL + BR) / 2).astype(np.float32)

        corners3 = np.array([TL, TR, BR, BL])
        _crop_logger.info(
            f"[CROP] Layer 3 computed corners: TL={TL}, TR={TR}, BR={BR}, BL={BL} "
            f"| image_bounds=({iw}, {ih}) | l3_scale={l3_scale:.3f} | "
            f"observed_dm_dist={observed_dm_dist:.1f} canonical_dm_dist={canonical_dm_dist:.1f}"
        )
        if (np.any(corners3[:, 0] < -iw * 0.1) or
                np.any(corners3[:, 0] >  iw * 1.1) or
                np.any(corners3[:, 1] < -ih * 0.1) or
                np.any(corners3[:, 1] >  ih * 1.1)):
            _crop_logger.info("[CROP] Layer 3 corners out of bounds")
            return None, (0, 0, 0, 0), False, None

        src_pts3 = np.float32([TL, TR, BR, BL])
        out_size  = max(int(PATTERN_PX * l3_scale), 64)
        dst_pts3  = np.float32([
            [0, 0], [out_size - 1, 0],
            [out_size - 1, out_size - 1], [0, out_size - 1]
        ])
        M3 = cv2.getPerspectiveTransform(src_pts3, dst_pts3)
        crop3 = cv2.warpPerspective(image, M3, (out_size, out_size))

        # GEOMETRY-ONLY: TL/TR/BR/BL above are already correctly oriented via
        # unit_right/unit_down computed from the DM rotation, so no rotation
        # search is needed. ref_gray (if available) is kept only as a fast
        # diagnostic NCC check, not a search loop.
        best_ncc = _ncc_thumbnail(crop3, ref_gray) if ref_gray is not None else -1.0
        _crop_logger.info(f"[CROP] Layer 3 geometry-only crop ncc={best_ncc:.4f}")
        target_w, target_h = PATTERN_SIZE
        cur_h, cur_w = crop3.shape[:2]
        interp3 = cv2.INTER_LANCZOS4 if (cur_w < target_w or cur_h < target_h) else cv2.INTER_AREA
        crop3 = cv2.resize(crop3, (target_w, target_h), interpolation=interp3)

        xs = corners3[:, 0]; ys = corners3[:, 1]
        bbox3 = (int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min()), int(ys.max() - ys.min()))

        ncc3 = _ncc_thumbnail(crop3, ref_gray) if ref_gray is not None else -1.0
        _crop_logger.info(
            f"[CROP] Layer 3 ncc={ncc3:.4f} "
            f"pattern_center=({pattern_center[0]:.0f},{pattern_center[1]:.0f}) "
            f"scale={l3_scale:.3f}")
        _crop_logger.info("[CROP] Layer 3 used (last resort)")
        return crop3, bbox3, True, src_pts3

    except Exception as e:
        _crop_logger.info(f"[CROP] Layer 3 failed: {e}")
        return None, (0, 0, 0, 0), False, None


# ==========================================================================
# VERIFICATION — STEP 1: Fiducial Marker Detection
# ==========================================================================

def detect_dm_border(image):
    """
    Detects the Data Matrix border in a captured image using a two-stage strategy.
    Returns (corners: np.float32 array of shape (4,2), orientation: str)
    orientation is one of: "0", "90", "180", "270"
    Returns (None, None) if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape
    
    # L-FINDER DIRECT DETECTION FALLBACK
    blurred_l = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_l, _ = cv2.threshold(blurred_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary_l = cv2.threshold(blurred_l, min(int(otsu_l) + 20, 220), 255, cv2.THRESH_BINARY_INV)

    lines = cv2.HoughLinesP(binary_l, 1, np.pi / 180, threshold=50, minLineLength=min(h, w)*0.6, maxLineGap=20)
    
    if lines is not None:
        horiz_lines = []
        vert_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > dy * 3:  # near horizontal
                horiz_lines.append(line[0])
            elif dy > dx * 3:  # near vertical
                vert_lines.append(line[0])
                
        best_h = None
        best_h_len = 0
        for l in horiz_lines:
            length = abs(l[2] - l[0])
            if length > best_h_len:
                best_h_len = length
                best_h = l
                
        best_v = None
        best_v_len = 0
        for l in vert_lines:
            length = abs(l[3] - l[1])
            if length > best_v_len:
                best_v_len = length
                best_v = l
                
        if best_h is not None and best_v is not None:
            hx1, hy1, hx2, hy2 = best_h
            vx1, vy1, vx2, vy2 = best_v
            
            h_y = (hy1 + hy2) / 2
            v_x = (vx1 + vx2) / 2
            
            is_bottom = h_y > h * 0.70
            is_top = h_y < h * 0.30
            is_right = v_x > w * 0.70
            is_left = v_x < w * 0.30
            
            orient = None
            if is_bottom and is_left:
                orient = "0"
                bl = [v_x, h_y]
                br = [max(hx1, hx2), h_y]
                tl = [v_x, min(vy1, vy2)]
                tr = [max(hx1, hx2), min(vy1, vy2)]
                corners = np.float32([tl, tr, br, bl])
            elif is_top and is_left:
                orient = "90"
                tl = [v_x, h_y]
                tr = [max(hx1, hx2), h_y]
                bl = [v_x, max(vy1, vy2)]
                br = [max(hx1, hx2), max(vy1, vy2)]
                corners = np.float32([tl, tr, br, bl])
            elif is_top and is_right:
                orient = "180"
                tr = [v_x, h_y]
                tl = [min(hx1, hx2), h_y]
                br = [v_x, max(vy1, vy2)]
                bl = [min(hx1, hx2), max(vy1, vy2)]
                corners = np.float32([tl, tr, br, bl])
            elif is_bottom and is_right:
                orient = "270"
                br = [v_x, h_y]
                bl = [min(hx1, hx2), h_y]
                tr = [v_x, min(vy1, vy2)]
                tl = [min(hx1, hx2), min(vy1, vy2)]
                corners = np.float32([tl, tr, br, bl])
                
            if orient is not None:
                print(f"[DM] HoughLinesP found L-finder! orientation={orient}")
                return corners, orient

    # STAGE 1 - Find the label boundary
    padded = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    blurred = cv2.GaussianBlur(padded, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_val = min(int(otsu_val) + 20, 220)
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    label_cnt = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.05):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            if area > best_area:
                best_area = area
                label_cnt = approx

    if label_cnt is None:
        return None, None
        
    pts = label_cnt.reshape(4, 2).astype(np.float32)
    pts -= 20.0
    
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    label_pts = np.float32([tl, tr, br, bl])
    
    d1 = np.linalg.norm(tl - tr)
    d2 = np.linalg.norm(tr - br)
    label_aspect = d1 / (d2 + 1e-5)
    
    label_side = 1024
    label_dst = np.float32([[0, 0], [label_side-1, 0], [label_side-1, label_side-1], [0, label_side-1]])
    
    if 0.8 <= label_aspect <= 1.25:
        # It's already square, treat it as the pattern directly
        dm_pts = label_pts
        M_label_inv = np.eye(3, dtype=np.float32)
        label_warped = gray
    else:
        try:
            M_label = cv2.getPerspectiveTransform(label_pts, label_dst)
            M_label_inv = np.linalg.inv(M_label)
            label_warped = cv2.warpPerspective(gray, M_label, (label_side, label_side))
        except Exception:
            return None, None

        # STAGE 2 - Find the DM border inside warped label
        def find_inner_dm(warped_img):
            l_blurred = cv2.GaussianBlur(warped_img, (5, 5), 0)
            l_otsu_val, _ = cv2.threshold(l_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, l_binary = cv2.threshold(l_blurred, min(int(l_otsu_val) + 20, 220), 255, cv2.THRESH_BINARY_INV)
            l_closed = cv2.morphologyEx(l_binary, cv2.MORPH_CLOSE, close_k)
            
            l_contours, _ = cv2.findContours(l_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            best_c = None
            best_score = float('inf')
            img_area = warped_img.shape[0] * warped_img.shape[1]
            
            for cnt in l_contours:
                area = cv2.contourArea(cnt)
                if not (img_area * 0.05 <= area <= img_area * 0.60):
                    continue
                    
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if 4 <= len(approx) <= 6:
                    hull = cv2.convexHull(approx)
                    peri_hull = cv2.arcLength(hull, True)
                    approx_hull = cv2.approxPolyDP(hull, 0.02 * peri_hull, True)
                    if len(approx_hull) == 4:
                        pts4 = approx_hull.reshape(4, 2)
                        d1_c = np.linalg.norm(pts4[0] - pts4[1])
                        d2_c = np.linalg.norm(pts4[1] - pts4[2])
                        aspect_c = d1_c / (d2_c + 1e-5)
                        squareness = abs(1.0 - aspect_c)
                        
                        # Score combines squareness and size (prefer larger, squarer)
                        # We use 1.0 - (area/img_area) to give a slight penalty to smaller ones
                        score = squareness + (1.0 - (area / img_area)) * 0.5
                        
                        if score < best_score:
                            best_score = score
                            best_c = approx_hull
            return best_c

        dm_cnt = find_inner_dm(label_warped)
        if dm_cnt is None:
            # Fallback: crop bottom half and try again
            bottom_half = label_warped[label_side//2:, :]
            dm_cnt_bottom = find_inner_dm(bottom_half)
            if dm_cnt_bottom is not None:
                dm_cnt = dm_cnt_bottom
                # Shift y coordinates back to full warped image
                dm_cnt[:, 0, 1] += label_side // 2

        if dm_cnt is None:
            return None, None
            
        pts = dm_cnt.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        d = pts[:, 1] - pts[:, 0]
        dm_tl = pts[np.argmin(s)]
        dm_br = pts[np.argmax(s)]
        dm_tr = pts[np.argmin(d)]
        dm_bl = pts[np.argmax(d)]
        dm_pts = np.float32([dm_tl, dm_tr, dm_br, dm_bl])

    # STAGE 3 - Edge classification
    dm_side = 512
    dm_dst = np.float32([[0, 0], [dm_side-1, 0], [dm_side-1, dm_side-1], [0, dm_side-1]])
    
    try:
        M_dm = cv2.getPerspectiveTransform(dm_pts, dm_dst)
        dm_warped = cv2.warpPerspective(label_warped, M_dm, (dm_side, dm_side))
    except Exception:
        return None, None
        
    strip_w = 8
    top_edge = dm_warped[0:strip_w, :]
    bottom_edge = dm_warped[dm_side-strip_w:dm_side, :]
    left_edge = dm_warped[:, 0:strip_w]
    right_edge = dm_warped[:, dm_side-strip_w:dm_side]
    
    def get_class(edge):
        profile = np.mean(edge, axis=0) if edge.shape[0] < edge.shape[1] else np.mean(edge, axis=1)
        return "alternating" if np.std(profile) > 30 else "solid"

    top_class = get_class(top_edge)
    bottom_class = get_class(bottom_edge)
    left_class = get_class(left_edge)
    right_class = get_class(right_edge)

    orientation = "0"
    if left_class == "solid" and bottom_class == "solid":
        orientation = "0"
    elif top_class == "solid" and left_class == "solid":
        orientation = "90"
    elif top_class == "solid" and right_class == "solid":
        orientation = "180"
    elif right_class == "solid" and bottom_class == "solid":
        orientation = "270"

    # STAGE 4 - Map corners back to original image coordinates
    dm_pts_reshaped = dm_pts.reshape(-1, 1, 2)
    original_corners = cv2.perspectiveTransform(dm_pts_reshaped, M_label_inv)
    original_corners = original_corners.reshape(4, 2)

    return original_corners, orientation

# ==========================================================================
# VERIFICATION — STEP 2: Image Alignment
# ==========================================================================

def align_captured_image(captured, original_size, corners=None, orientation=None, original_gray_ref=None):
    target_w, target_h = original_size
    
    if corners is not None and orientation is not None:
        try:
            w, h = target_w, target_h
            if orientation == "0":
                dst_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
            elif orientation == "90":
                dst_pts = np.float32([[0,h],[0,0],[w,0],[w,h]])
            elif orientation == "180":
                dst_pts = np.float32([[w,h],[0,h],[0,0],[w,0]])
            elif orientation == "270":
                dst_pts = np.float32([[w,0],[w,h],[0,h],[0,0]])
            else:
                dst_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
                
            M = cv2.getPerspectiveTransform(corners, dst_pts)
            aligned = cv2.warpPerspective(captured, M, (target_w, target_h))
            return aligned, f"perspective ({orientation}°)"
        except Exception:
            pass

    # Fallback
    resized = cv2.resize(captured, (target_w, target_h))
    if original_gray_ref is not None:
        gray_r = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        thumb = 128
        ref_t = cv2.resize(original_gray_ref, (thumb, thumb)).astype(np.float32)
        ref_f = ref_t - ref_t.mean()
        ref_n = np.linalg.norm(ref_f) + 1e-9
        best_rot, best_ncc = 0, -1.0
        for rot in range(4):
            g = np.rot90(gray_r, rot)
            t = cv2.resize(g, (thumb, thumb)).astype(np.float32)
            f = t - t.mean()
            n = np.linalg.norm(f) + 1e-9
            ncc = float(np.dot(ref_f.ravel(), f.ravel()) / (ref_n * n))
            if ncc > best_ncc:
                best_ncc, best_rot = ncc, rot
        if best_rot != 0:
            resized = np.rot90(resized, best_rot).copy()
        rot_names = ["0", "90CW", "180", "90CCW"]
        return resized, f"resize ({rot_names[best_rot]})"
    
    return resized, "resize (0)" 

# ==========================================================================
# VERIFICATION — STEP 3: Tests
# ==========================================================================

def test_moire_detection(captured_gray, reference_gray=None):
    """Measure high-frequency PATTERN match, not just energy.

    The old approach measured HF energy ratio, which rewarded images with
    more HF energy regardless of whether that energy came from the original
    grating or from printer/camera artifacts.  A 2nd-gen copy printed from
    the aligned image has lots of HF energy from printer halftone but very
    little that actually matches the original grating.

    New approach: bandpass-filter both images to isolate frequency bands,
    then compute NCC (normalized cross-correlation) between them.  This
    measures whether the HF patterns MATCH, not just whether HF exists.
    """
    h, w = captured_gray.shape[:2]
    if reference_gray is not None:
        rh, rw = reference_gray.shape[:2]
        if (h, w) != (rh, rw):
            captured_gray = cv2.resize(captured_gray, (rw, rh),
                                       interpolation=cv2.INTER_AREA)
            h, w = rh, rw

    def bandpass(img, low_sigma, high_sigma):
        """Extract frequency band via difference of Gaussians."""
        lo = cv2.GaussianBlur(img.astype(np.float64), (0, 0), low_sigma)
        hi = cv2.GaussianBlur(img.astype(np.float64), (0, 0), high_sigma)
        return lo - hi

    if reference_gray is not None:
        # NCC at multiple frequency bands; finer bands weighted more
        bands = [
            (2, 1, 0.50),   # finest detail (sigma 1-2)
            (4, 2, 0.30),   # medium detail (sigma 2-4)
            (8, 4, 0.20),   # coarser detail (sigma 4-8)
        ]
        score = 0.0
        for lo_s, hi_s, weight in bands:
            cap_band = bandpass(captured_gray, lo_s, hi_s)
            ref_band = bandpass(reference_gray, lo_s, hi_s)
            cf = cap_band.flatten() - cap_band.mean()
            rf = ref_band.flatten() - ref_band.mean()
            ncc = np.sum(cf * rf) / (np.sqrt(np.sum(cf**2) * np.sum(rf**2))
                                     + 1e-10)
            score += weight * max(0.0, ncc)
        return float(np.clip(score / 0.25, 0.0, 1.0))

    # No reference — fallback: measure HF energy presence
    f = np.fft.fftshift(np.fft.fft2(captured_gray.astype(np.float64)))
    mag = np.abs(f)
    cy, cx = h // 2, w // 2
    yc, xc = np.ogrid[:h, :w]
    r = np.sqrt((yc - cy) ** 2 + (xc - cx) ** 2)
    r_max = min(cy, cx)
    hf_mask = r > r_max * 0.40
    dc_mask = r < r_max * 0.02
    ratio = np.sum(mag[hf_mask] ** 2) / (np.sum(mag[~dc_mask] ** 2) + 1e-10)
    return float(np.clip(ratio / 0.15, 0.0, 1.0))


def test_color_analysis(captured_rgb, reference_rgb):
    cap_r, cap_g, cap_b = [c.astype(np.float32) for c in cv2.split(captured_rgb)]
    ref_r, ref_g, ref_b = [c.astype(np.float32) for c in cv2.split(reference_rgb)]
    cap_diffs = [np.mean(np.abs(cap_r - cap_g)), np.mean(np.abs(cap_r - cap_b)),
                 np.mean(np.abs(cap_g - cap_b))]
    ref_diffs = [np.mean(np.abs(ref_r - ref_g)), np.mean(np.abs(ref_r - ref_b)),
                 np.mean(np.abs(ref_g - ref_b))]
    ratios = [min(c / (r + 1e-10), 1.0) for c, r in zip(cap_diffs, ref_diffs) if r > 0]
    if not ratios:
        return 0.5
    cap_var = np.var(cap_r) + np.var(cap_g) + np.var(cap_b)
    ref_var = np.var(ref_r) + np.var(ref_g) + np.var(ref_b)
    var_ratio = min(cap_var / (ref_var + 1e-10), 1.0)
    pixel_diff = np.mean(np.abs(captured_rgb.astype(np.float32) - reference_rgb.astype(np.float32)))
    pixel_score = np.clip(1.0 - (pixel_diff - 5) / 50, 0.0, 1.0)
    return float(np.clip(0.3 * np.mean(ratios) + 0.3 * var_ratio + 0.4 * pixel_score, 0.0, 1.0))


def test_prng_correlation(captured_gray, reference_gray, block_size=16):
    """Measure block-level Pearson correlation, prioritising fine scales.

    Copy detection relies on the fact that each print→photo cycle is a
    low-pass filter that destroys fine-grained PRNG block structure.  A
    genuine 1st-gen print preserves fine (8×8) structure; a 2nd-gen copy
    loses it and only coarse structure survives.

    Strategy — use the FINEST scale that has meaningful signal:
      • Compute Pearson r at 8, 16, 32 block sizes.
      • A scale has "signal" when the captured block-mean variance
        (std-dev) exceeds a threshold — otherwise blur killed it.
      • Use the finest scale with signal.  If no scale has signal → 0.
      • Finer scales → higher max achievable score (cap=1.0).
        Coarser scales → lower cap (0.6 for 32×32).  Needing a coarse
        scale means fine detail was destroyed → copy → lower ceiling.
    """
    h, w = captured_gray.shape[:2]
    rh, rw = reference_gray.shape[:2]
    if (h, w) != (rh, rw):
        captured_gray = cv2.resize(captured_gray, (rw, rh), interpolation=cv2.INTER_AREA)
        h, w = rh, rw

    # (block_px, norm_divisor, score_cap)
    # Finer → easier to max out (cap 1.0), coarser → capped lower.
    # norm_divisor set so raw ~0.55 → score ~0.92 (good 1st-gen),
    # raw ~0.49 → score ~0.82 (degraded copy).  Preserves the gap.
    if CDP_FLAG_CORR_FINE_BLOCKS:
        scale_configs = [
            (block_size // 4, 0.45, 1.0),
            (block_size // 2, 0.55, 1.0),
            (block_size,      0.60, 1.0 if not CDP_FLAG_CORR_COARSE_CAP else 1.0),
            (block_size * 2,  0.65, 0.50 if CDP_FLAG_CORR_COARSE_CAP else 0.50),
        ]
    else:
        scale_configs = [
            (block_size,     0.60, 1.0  if not CDP_FLAG_CORR_COARSE_CAP else 0.75),
            (block_size * 2, 0.65, 0.75 if not CDP_FLAG_CORR_COARSE_CAP else 0.50),
            (block_size * 4, 0.75, 0.50),
        ]

    # Minimum std-dev of captured block means to consider a scale "usable".
    # Below this the blocks are all near-identical → correlation is noise.
    MIN_STD = 2.0

    for bs, norm_div, cap in scale_configs:
        by_, bx_ = h // bs, w // bs
        if by_ < 4 or bx_ < 4:
            continue
        cap_b = np.zeros((by_, bx_))
        ref_b = np.zeros((by_, bx_))
        for i in range(by_):
            for j in range(bx_):
                y1, y2, x1, x2 = i * bs, (i + 1) * bs, j * bs, (j + 1) * bs
                cap_b[i, j] = np.mean(captured_gray[y1:y2, x1:x2])
                ref_b[i, j] = np.mean(reference_gray[y1:y2, x1:x2])

        # Check if captured has enough variance at this scale
        if np.std(cap_b) < MIN_STD:
            continue  # Too blurred at this scale, try coarser

        cf = cap_b.flatten() - cap_b.mean()
        rf = ref_b.flatten() - ref_b.mean()
        corr = np.sum(cf * rf) / (np.sqrt(np.sum(cf ** 2) * np.sum(rf ** 2)) + 1e-10)
        normalized = float(np.clip(corr / norm_div, 0.0, cap))
        return normalized, float(corr)

    # No scale had usable signal — completely destroyed
    return 0.0, 0.0


def test_gradient_energy(captured_gray, reference_gray):
    rh, rw = reference_gray.shape[:2]
    if captured_gray.shape[:2] != (rh, rw):
        captured_gray = cv2.resize(captured_gray, (rw, rh),
                                   interpolation=cv2.INTER_AREA)

    def grad_hist(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag = mag / (mag.max() + 1e-10)
        hist, _ = np.histogram(mag, bins=32, range=(0, 1))
        return hist.astype(np.float64)

    cap_h = grad_hist(captured_gray)
    ref_h = grad_hist(reference_gray)

    cap_n = cap_h / (cap_h.sum() + 1e-10)
    ref_n = ref_h / (ref_h.sum() + 1e-10)
    similarity = float(np.sum(np.sqrt(cap_n * ref_n)))
    # similarity ranges 0-1, ~0.95+ for identical, ~0.7-0.8 for genuine prints
    return float(np.clip((similarity - 0.60) / 0.35, 0.0, 1.0))


# --------------------------------------------------------------------------
# Diagnostic helpers — pre-clip raw values.  NOT used in live scoring.
# --------------------------------------------------------------------------

def _moire_raw(captured_gray, reference_gray):
    """Pre-clip bandpass-NCC moire score (same math as test_moire_detection
    with reference, without the final np.clip).  Diagnostic use only."""
    h, w = captured_gray.shape[:2]
    rh, rw = reference_gray.shape[:2]
    if (h, w) != (rh, rw):
        captured_gray = cv2.resize(captured_gray, (rw, rh), interpolation=cv2.INTER_AREA)

    def _bp(img, lo_s, hi_s):
        lo = cv2.GaussianBlur(img.astype(np.float64), (0, 0), lo_s)
        hi = cv2.GaussianBlur(img.astype(np.float64), (0, 0), hi_s)
        return lo - hi

    bands = [(2, 1, 0.50), (4, 2, 0.30), (8, 4, 0.20)]
    score = 0.0
    for lo_s, hi_s, w_ in bands:
        cb = _bp(captured_gray, lo_s, hi_s)
        rb = _bp(reference_gray, lo_s, hi_s)
        cf = cb.flatten() - cb.mean()
        rf = rb.flatten() - rb.mean()
        ncc = np.sum(cf * rf) / (np.sqrt(np.sum(cf ** 2) * np.sum(rf ** 2)) + 1e-10)
        score += w_ * max(0.0, ncc)
    return float(score / 0.25), float(score)   # (pre_clip_normed, raw_score)


def _corr_raw(captured_gray, reference_gray, block_size=16):
    """Pre-clip correlation score alongside the selected block size and raw
    Pearson r.  Diagnostic use only."""
    h, w = captured_gray.shape[:2]
    rh, rw = reference_gray.shape[:2]
    if (h, w) != (rh, rw):
        captured_gray = cv2.resize(captured_gray, (rw, rh), interpolation=cv2.INTER_AREA)
        h, w = rh, rw

    if CDP_FLAG_CORR_FINE_BLOCKS:
        scale_configs = [
            (block_size // 4, 0.45, 1.0),
            (block_size // 2, 0.55, 1.0),
            (block_size,      0.60, 1.0),
            (block_size * 2,  0.65, 0.50),
        ]
    else:
        scale_configs = [
            (block_size,     0.60, 1.0  if not CDP_FLAG_CORR_COARSE_CAP else 0.75),
            (block_size * 2, 0.65, 0.75 if not CDP_FLAG_CORR_COARSE_CAP else 0.50),
            (block_size * 4, 0.75, 0.50),
        ]

    MIN_STD = 2.0
    for bs, norm_div, cap in scale_configs:
        by_, bx_ = h // bs, w // bs
        if by_ < 4 or bx_ < 4:
            continue
        cap_b = np.zeros((by_, bx_))
        ref_b = np.zeros((by_, bx_))
        for i in range(by_):
            for j in range(bx_):
                y1, y2, x1, x2 = i * bs, (i + 1) * bs, j * bs, (j + 1) * bs
                cap_b[i, j] = np.mean(captured_gray[y1:y2, x1:x2])
                ref_b[i, j] = np.mean(reference_gray[y1:y2, x1:x2])
        if np.std(cap_b) < MIN_STD:
            continue
        cf = cap_b.flatten() - cap_b.mean()
        rf = ref_b.flatten() - ref_b.mean()
        pearson_r = float(np.sum(cf * rf) / (np.sqrt(np.sum(cf ** 2) * np.sum(rf ** 2)) + 1e-10))
        return pearson_r / norm_div, bs, norm_div, pearson_r  # (pre_clip, sel_bs, norm_div, pearson_r)
    return 0.0, None, None, 0.0


def _gradient_raw(captured_gray, reference_gray):
    """Pre-clip gradient score ((bhattacharyya - 0.60) / 0.35) and the raw
    Bhattacharyya coefficient.  Diagnostic use only."""
    rh, rw = reference_gray.shape[:2]
    if captured_gray.shape[:2] != (rh, rw):
        captured_gray = cv2.resize(captured_gray, (rw, rh), interpolation=cv2.INTER_AREA)

    def grad_hist(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = mag / (mag.max() + 1e-10)
        hist, _ = np.histogram(mag, bins=32, range=(0, 1))
        return hist.astype(np.float64)

    cap_h = grad_hist(captured_gray)
    ref_h = grad_hist(reference_gray)
    cap_n = cap_h / (cap_h.sum() + 1e-10)
    ref_n = ref_h / (ref_h.sum() + 1e-10)
    similarity = float(np.sum(np.sqrt(cap_n * ref_n)))
    return (similarity - 0.60) / 0.35, similarity  # (pre_clip, bhattacharyya)


# ==========================================================================
# VERIFICATION — FULL PIPELINE
# ==========================================================================

def verify_pattern_legacy(original_path, captured_path, uploads_dir="uploads", block_size=16):
    original_bgr = cv2.imread(original_path)
    captured_bgr = cv2.imread(captured_path)
    if original_bgr is None:
        return {"verdict": "ERROR", "error": f"Cannot load original: {original_path}"}
    if captured_bgr is None:
        return {"verdict": "ERROR", "error": f"Cannot load captured: {captured_path}"}

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    print(f"[VERIFY] captured shape: {captured_bgr.shape}")

    # Step 0: Crop
    cropped_bgr, bbox, pattern_found, pattern_quad = detect_and_crop_pattern(captured_bgr)
    print(f"[VERIFY] pattern_found: {pattern_found}, crop shape: {cropped_bgr.shape}")
    
    if pattern_found:
        ch, cw = cropped_bgr.shape[:2]
        orig_h, orig_w = captured_bgr.shape[:2]
        if cw > orig_w * 0.85 or ch > orig_h * 0.85:
            pattern_found = False
            print(f"[VERIFY] Crop rejected — too large: {cw}x{ch} vs {orig_w}x{orig_h}")
        else:
            captured_bgr = cropped_bgr
            print(f"[VERIFY] Crop accepted: {cw}x{ch}")

    # Step 1: Markers
    corners, orientation = detect_dm_border(captured_bgr)
    print(f"[VERIFY] corners: {corners is not None}, orientation: {orientation}")

    # Step 2: Align
    target = (original_bgr.shape[1], original_bgr.shape[0])
    aligned_bgr, alignment_method = align_captured_image(
        captured_bgr, target, corners, orientation, original_gray_ref=original_gray)
    print(f"[VERIFY] alignment_method: {alignment_method}")
    
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    aligned_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)

    # Step 3: Tests
    moire = test_moire_detection(aligned_gray, original_gray)
    color = test_color_analysis(aligned_rgb, original_rgb)
    corr, raw_corr = test_prng_correlation(aligned_gray, original_gray, block_size)
    gradient = test_gradient_energy(aligned_gray, original_gray)

    # Step 4: Score
    final = (WEIGHT_MOIRE * moire + WEIGHT_COLOR * color +
             WEIGHT_CORRELATION * corr + WEIGHT_GRADIENT * gradient)

    if final >= THRESHOLD_AUTHENTIC:
        verdict = "AUTHENTIC"
    elif final >= THRESHOLD_SUSPICIOUS:
        verdict = "SUSPICIOUS"
    else:
        verdict = "COUNTERFEIT"

    # Save markers visualization — draw on the full original photo so markers are
    # visible in context. Marker positions are in cropped-image space, so we offset
    # them by the crop bbox origin when drawing on the full photo.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_captured_bgr = cv2.imread(captured_path)
    if full_captured_bgr is None:
        full_captured_bgr = captured_bgr.copy()
        crop_ox, crop_oy = 0, 0
    else:
        crop_ox, crop_oy = (bbox[0], bbox[1]) if pattern_found else (0, 0)

    markers_vis = full_captured_bgr.copy()
    full_h, full_w = markers_vis.shape[:2]
    bx, by, bw, bh = bbox if pattern_found else (0, 0, full_w, full_h)
    if pattern_found and pattern_quad is not None:
        pts = pattern_quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(markers_vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    elif pattern_found:
        bx, by, bw, bh = bbox
        cv2.rectangle(markers_vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

    if corners is not None:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        pts += np.array([crop_ox, crop_oy], dtype=np.int32)
        cv2.polylines(markers_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        ctx, cty = pts[0][0]
        cv2.putText(markers_vis, f"Rot: {orientation}", (int(ctx), int(cty) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    markers_filename = f"markers_{ts}.png"
    cv2.imwrite(os.path.join(uploads_dir, markers_filename), markers_vis)

    # Save aligned
    aligned_filename = f"aligned_{ts}.png"
    cv2.imwrite(os.path.join(uploads_dir, aligned_filename), aligned_bgr)

    # Per-block scoring for analytics / research-paper plots (one entry per
    # 16x16 block — used to render heatmaps, scatter, histograms downstream).
    try:
        from cdp_analytics import compute_per_block_scores
        per_block = compute_per_block_scores(
            aligned_gray, original_gray, aligned_rgb, original_rgb,
            block_size=block_size)
        per_block_serialised = {k: v.tolist() for k, v in per_block.items()}
    except Exception:
        per_block_serialised = None

    return {
        "verdict": verdict,
        "confidence": float(final),
        "scores": {"moire": float(moire), "color": float(color),
                   "correlation": float(corr), "gradient": float(gradient)},
        "weights": {"moire": WEIGHT_MOIRE, "color": WEIGHT_COLOR,
                    "correlation": WEIGHT_CORRELATION, "gradient": WEIGHT_GRADIENT},
        "markers_found": 4 if corners is not None else 0,
        "alignment_method": alignment_method,
        "markers_filename": markers_filename,
        "aligned_filename": aligned_filename,
        "pattern_found": pattern_found,
        "per_block_scores": per_block_serialised,
    }

# ==========================================================================
# ==========================================================================
# DATAMATRIX ENCODING  —  Feistel-4 + RFC 4648 Base32 + check character
# ==========================================================================
#
# Pipeline (encode):
#   seed (32-bit int)
#   → 4-round Feistel encrypt  → 32-bit ciphertext
#   → Base32 encode (7 chars)  + 1 weighted check char
#   → 8-char payload  →  split  →  share_a (first 4),  share_b (last 4)
#
# Pipeline (decode):
#   share_a + share_b  →  8-char payload
#   → verify check char  →  Base32 decode  →  Feistel decrypt  →  seed
#
# Why this design?
#   • 4-char per DM  →  8x18 symbol (minimum reliable rectangular DM)
#   • 8x18 has 18 columns  →  0.417 mm/module  →  4.9 printer-dots @ 300 DPI
#   • Current 16x48 had 48 columns  →  0.156 mm/module  →  1.8 dots (fails)
#   • Feistel prevents trivial human readability of the seed value
#   • Check char detects 100% of single-character substitution errors
# ==========================================================================

# Fixed 8-byte Feistel key  (4 × 16-bit round keys, concatenated)
_FEISTEL_KEY: bytes = bytes([0xA7, 0x3E, 0x2F, 0x91, 0xD8, 0x5C, 0x4B, 0xE6])

# Standard RFC 4648 Base32 alphabet  (uppercase A-Z  +  digits 2-7)
_B32_ALPHA: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
_B32_IDX: dict = {c: i for i, c in enumerate(_B32_ALPHA)}

# Weights for the check character — all must be non-zero mod 32 (all are odd),
# ensuring that any single-character substitution changes the check value.
_CHECK_WEIGHTS: tuple = (3, 7, 11, 13, 17, 19, 23)


def _feistel_f(half: int, round_key: int) -> int:
    """Non-linear round function for the Feistel cipher (16-bit domain)."""
    v = (half ^ round_key) & 0xFFFF
    v = (v * 0x9E37 + 0xB5EF) & 0xFFFF
    v ^= v >> 7
    return (v * 0x6B5F) & 0xFFFF


def feistel_encrypt(seed: int) -> int:
    """
    4-round Feistel encryption of a 32-bit seed → 32-bit ciphertext.

    The cipher is a classic balanced Feistel network:
        L_{i+1} = R_i
        R_{i+1} = L_i XOR F(R_i, round_key_i)
    """
    L, R = (seed >> 16) & 0xFFFF, seed & 0xFFFF
    rks = [int.from_bytes(_FEISTEL_KEY[i:i+2], 'big') for i in range(0, 8, 2)]
    for rk in rks:
        L, R = R, L ^ _feistel_f(R, rk)
    return (L << 16) | R


def feistel_decrypt(ct: int) -> int:
    """
    4-round Feistel decryption of a 32-bit ciphertext → original seed.

    Inverse step (derived from the forward Feistel equations):
        R_i = L_{i+1}
        L_i = R_{i+1} XOR F(L_{i+1}, round_key_i)
    Applied in reverse key order.
    """
    L, R = (ct >> 16) & 0xFFFF, ct & 0xFFFF
    rks = [int.from_bytes(_FEISTEL_KEY[i:i+2], 'big') for i in range(0, 8, 2)]
    for rk in reversed(rks):
        L, R = R ^ _feistel_f(L, rk), L
    return (L << 16) | R


def _b32_encode(n: int, length: int = 7) -> str:
    """Encode integer n into `length` Base32 characters (big-endian, no padding)."""
    chars = []
    for _ in range(length):
        chars.append(_B32_ALPHA[n & 0x1F])
        n >>= 5
    return ''.join(reversed(chars))


def _b32_decode(s: str) -> int:
    """Decode a Base32 string to an integer (big-endian)."""
    n = 0
    for c in s:
        n = (n << 5) | _B32_IDX[c]
    return n


def _check_char(payload7: str) -> str:
    """
    Compute 1 Base32 check character from a 7-char Base32 payload.

    Uses a weighted sum with prime-ish weights that are all odd (coprime with 32),
    guaranteeing that every single-character substitution error is detected.
    Validated: 100% detection across all 248 possible single-char mutations.
    """
    val = sum(w * _B32_IDX[c] for w, c in zip(_CHECK_WEIGHTS, payload7)) % 32
    return _B32_ALPHA[val]


def split_seed_for_dm(seed: int) -> tuple[str, str]:
    """
    Encode a 32-bit seed into two 4-character Base32 shares for DataMatrix.

    Encoding pipeline:
        seed  →  Feistel-4 encrypt  →  32-bit ciphertext
              →  7 Base32 chars (35 bits, lower 32 used)
              →  + 1 weighted check char
              →  8-char payload  →  split at position 4

    Example (seed=920789066):
        ciphertext   = Feistel-4(920789066)  =  some 32-bit value
        payload7     = 'AM3ISC5'  (7 Base32 chars)
        check        = 'Y'
        full8        = 'AM3ISC5Y'
        share_a      = 'AM3I'
        share_b      = 'SC5Y'

    Both shares are encoded into 8x18 Data Matrix symbols:
        18 columns  x  0.417 mm/column  =  4.9 printer dots at 300 DPI
        (vs 16x48 old:  48 cols  x 0.156 mm  =  1.8 dots — fails on thermal)

    Returns (share_a: str, share_b: str), each exactly 4 Base32 characters.
    """
    ct       = feistel_encrypt(seed)
    payload7 = _b32_encode(ct, 7)
    check    = _check_char(payload7)
    full8    = payload7 + check          # e.g. 'AM3ISC5Y'
    return full8[:4], full8[4:]          # ('AM3I', 'SC5Y')


def recombine_seed_from_dm(share_a: str, share_b: str) -> int | None:
    """
    Recover the original seed from two 4-character Base32 shares.

    Decoding pipeline:
        share_a + share_b  →  8-char payload
                           →  verify check character  (returns None on failure)
                           →  Base32 decode 7 chars  →  32-bit ciphertext
                           →  Feistel-4 decrypt  →  original seed

    Returns seed (int) on success, or None if the check character fails
    (indicating a decode error, wrong pairing, or corrupted DM).
    """
    if not share_a or not share_b:
        return None
    # Accepts both trimmed 4-char strings and any surrounding whitespace
    full8 = share_a.strip() + share_b.strip()
    if len(full8) != 8:
        return None
    # Validate every character is in the Base32 alphabet
    if not all(c in _B32_IDX for c in full8):
        return None
    payload7, check_got = full8[:7], full8[7]
    if _check_char(payload7) != check_got:
        return None                       # integrity check failed
    ct   = _b32_decode(payload7) & 0xFFFFFFFF
    return feistel_decrypt(ct)


def generate_cropped_dm(data: str, size: str = "8x18"):
    """
    Generates a DataMatrix of a fixed symbol size and crops it to exact module
    boundaries, removing the quiet zone added by pylibdmtx.

    Using a fixed size (default '8x18') instead of 'RectAuto' guarantees that
    every seed produces DM images with identical pixel dimensions regardless of
    payload entropy.

    Returns
    -------
    (image, (num_rows, num_cols))
      image      – grayscale numpy array of the cropped module grid (pure 0/255)
      num_rows   – number of DM symbol rows  (8 for '8x18')
      num_cols   – number of DM symbol cols  (18 for '8x18')
    """
    from pylibdmtx.pylibdmtx import encode
    import cv2
    import numpy as np

    # Parse the requested symbol dimensions
    parts = size.split('x')
    num_rows, num_cols = int(parts[0]), int(parts[1])

    enc = encode(data.encode('utf-8'), size=size)
    img = np.frombuffer(enc.pixels, dtype=np.uint8).reshape((enc.height, enc.width, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Hard threshold to pure B&W — no grey anti-aliasing from pylibdmtx
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Crop to exact module boundaries
    inv = cv2.bitwise_not(bw)
    coords = cv2.findNonZero(inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return bw[y:y+h, x:x+w], (num_rows, num_cols)
    return bw, (num_rows, num_cols)


def calculate_auth_block_layout(pattern_size_unit: float, quiet_unit: float,
                                top_dm_pixels, right_dm_pixels,
                                top_dm_modules=None, right_dm_modules=None):
    """
    Calculates the dimensions and positions of the Auth Block elements using
    INTEGER module sizing so every printed module is exactly the same number
    of pixels wide — critical for thermal-transfer print quality.

    Inputs
    ------
    pattern_size_unit  : Size of the pattern in arbitrary units (px or mm).
    quiet_unit         : Size of the quiet zone in the same units.
    top_dm_pixels      : Grayscale image of the Top DM (H, W) — NOT rotated.
    right_dm_pixels    : Grayscale image of the Right DM (H, W) — NOT rotated.
    top_dm_modules     : (num_rows, num_cols) of the Top DM symbol.   Optional.
    right_dm_modules   : (num_rows, num_cols) of the Right DM symbol. Optional.

    Returns a dictionary of layout properties. Coordinates are y-down.
    """
    import math

    # ── Derive module counts ─────────────────────────────────────────────────
    # If caller provides module counts, use them; otherwise infer from image
    # shape (assuming 5 native pixels per module for the default 16x48 symbol).
    if top_dm_modules is not None:
        t_rows, t_cols = top_dm_modules
    else:
        # Infer: native image is (rows*5, cols*5), ratio gives col count
        # For 16x48: shape=(80,240); cols = 240//5 = 48
        native_h, native_w = top_dm_pixels.shape[:2]
        # Native module pixel size = GCD of (native_h, native_w) divided by row/col ratio
        # Reliable fallback: detect by scanning first row for first transition
        row0 = top_dm_pixels[0, :]
        first_val = int(row0[0])
        for i in range(1, len(row0)):
            if int(row0[i]) != first_val:
                native_mod_px = i
                break
        else:
            native_mod_px = 5
        t_cols = native_w // native_mod_px
        t_rows = native_h // native_mod_px

    if right_dm_modules is not None:
        r_rows, r_cols = right_dm_modules
    else:
        native_h, native_w = right_dm_pixels.shape[:2]
        row0 = right_dm_pixels[0, :]
        first_val = int(row0[0])
        for i in range(1, len(row0)):
            if int(row0[i]) != first_val:
                native_mod_px = i
                break
        else:
            native_mod_px = 5
        r_cols = native_w // native_mod_px
        r_rows = native_h // native_mod_px

    # ── Integer module pixel size ────────────────────────────────────────────
    # Top DM: t_cols modules span the pattern width.
    # Floor to nearest integer so each module is exactly module_px units wide.
    # This eliminates the alternating 10/11px module widths caused by non-integer
    # scaling, which is the primary cause of thermal-print blur on the top DM.
    module_px = int(pattern_size_unit) // t_cols          # e.g. 512//48 = 10
    module_px = max(module_px, 1)                          # guard against tiny patterns

    # Top DM dims (landscape: t_cols wide × t_rows tall)
    top_dm_w = module_px * t_cols                          # e.g. 480
    top_dm_h = module_px * t_rows                          # e.g. 160

    # Right DM dims after 90° CW rotation:
    #   native (t_rows × t_cols) → rendered (t_cols cols high × t_rows cols wide)
    #   = r_cols visual-rows × r_rows visual-cols  [r = right DM, same size as top]
    right_dm_h = module_px * r_cols                        # e.g. 480 (portrait height)
    right_dm_w = module_px * r_rows                        # e.g. 160 (portrait width)

    # ── Auth block bounding box ───────────────────────────────────────────────
    auth_w = top_dm_w + quiet_unit + right_dm_w
    auth_h = top_dm_h + quiet_unit + top_dm_w  # top_dm_w = pattern side length

    # ── Positions (y-down) ───────────────────────────────────────────────────
    top_dm_x = 0
    top_dm_y = 0

    pattern_x = 0
    pattern_y = top_dm_h + quiet_unit

    right_dm_x = top_dm_w + quiet_unit
    right_dm_y = pattern_y

    return {
        "auth_w":       auth_w,
        "auth_h":       auth_h,
        "module_px":    module_px,
        "top_dm_rect":  (top_dm_x, top_dm_y, top_dm_w, top_dm_h),
        "pattern_rect": (pattern_x, pattern_y, top_dm_w, top_dm_w),  # pattern is square
        "right_dm_rect": (right_dm_x, right_dm_y, right_dm_w, right_dm_h),
    }

def draw_auth_block_opencv(auth_canvas, layout, pattern_img, top_dm_img, right_dm_img):
    """
    Draws the Auth Block onto a pre-sized OpenCV canvas (auth_canvas).
    Coordinates are y-down.

    DM regions are scaled with INTER_NEAREST (no interpolation) then hard-
    thresholded to pure black/white.  This guarantees that every pixel in the
    DM is exactly 0 or 255 even if the printer driver later upscales the PNG
    with bilinear interpolation — the grey anti-aliased edges that bilinear
    produces are eliminated at the source.
    """
    import cv2
    import numpy as np

    # Extract rects
    tx, ty, tw, th = [int(v) for v in layout["top_dm_rect"]]
    px, py, pw, ph = [int(v) for v in layout["pattern_rect"]]
    rx, ry, rw, rh = [int(v) for v in layout["right_dm_rect"]]

    def _scale_dm_bw(dm_gray, w, h):
        """Scale a B&W DM with INTER_NEAREST then hard-threshold to pure 0/255."""
        resized = cv2.resize(dm_gray, (w, h), interpolation=cv2.INTER_NEAREST)
        _, bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
        return bw

    # Draw pattern (CDP texture — area interpolation is correct here)
    pat_resized = cv2.resize(pattern_img, (pw, ph), interpolation=cv2.INTER_AREA)
    if len(pat_resized.shape) == 2 and len(auth_canvas.shape) == 3:
        pat_resized = cv2.cvtColor(pat_resized, cv2.COLOR_GRAY2BGR)
    auth_canvas[py:py+ph, px:px+pw] = pat_resized

    # Draw Top DM — scale then hard-threshold
    top_bw = _scale_dm_bw(top_dm_img, tw, th)
    if len(auth_canvas.shape) == 3:
        top_bw = cv2.cvtColor(top_bw, cv2.COLOR_GRAY2BGR)
    auth_canvas[ty:ty+th, tx:tx+tw] = top_bw

    # Draw Right DM — rotate 90° CW, scale, hard-threshold
    rot_right_dm = cv2.rotate(right_dm_img, cv2.ROTATE_90_CLOCKWISE)
    right_bw = _scale_dm_bw(rot_right_dm, rw, rh)
    if len(auth_canvas.shape) == 3:
        right_bw = cv2.cvtColor(right_bw, cv2.COLOR_GRAY2BGR)
    auth_canvas[ry:ry+rh, rx:rx+rw] = right_bw



# ==========================================================================
# NEW VERIFICATION PIPELINE — Steps 1-4
# ==========================================================================

import itertools
import re

# ---------------------------------------------------------------------------
# Dynamic layout constants — derived from the real calculate_auth_block_layout
# output so detect_and_crop_pattern always matches what the label generator
# actually produces.
# ---------------------------------------------------------------------------
try:
    _dm_throwaway, _dm_modules = generate_cropped_dm("AAAA", size="8x18")
    _layout_ref = calculate_auth_block_layout(
        pattern_size_unit=512,
        quiet_unit=34,
        top_dm_pixels=_dm_throwaway,
        right_dm_pixels=_dm_throwaway,
        top_dm_modules=_dm_modules,
        right_dm_modules=_dm_modules,
    )
    _top_dm_x, _top_dm_y, _top_dm_w, _top_dm_h = _layout_ref["top_dm_rect"]
    _pat_x,    _pat_y,    _pat_w,    _pat_h     = _layout_ref["pattern_rect"]
    _right_x,  _right_y,  _right_w,  _right_h  = _layout_ref["right_dm_rect"]
except Exception as _layout_exc:
    # pylibdmtx native library unavailable — use precomputed values for
    # size="8x18", pattern_size_unit=512, quiet_unit=34:
    #   module_px=28, top_dm=504×224, right_dm=224×504, pattern=504×504
    import logging as _log
    _log.warning("pylibdmtx unavailable (%s); using hardcoded layout constants", _layout_exc)
    _top_dm_x, _top_dm_y, _top_dm_w, _top_dm_h = 0, 0, 504, 224
    _pat_x,    _pat_y,    _pat_w,    _pat_h     = 0, 258, 504, 504
    _right_x,  _right_y,  _right_w,  _right_h  = 538, 258, 224, 504

_TOP_DM_CX   = _top_dm_x + _top_dm_w / 2
_TOP_DM_CY   = _top_dm_y + _top_dm_h / 2
_RIGHT_DM_CX = _right_x  + _right_w  / 2
_RIGHT_DM_CY = _right_y  + _right_h  / 2
_PAT_CX      = _pat_x    + _pat_w    / 2
_PAT_CY      = _pat_y    + _pat_h    / 2

_R_TOP_X   = _TOP_DM_CX   - _PAT_CX
_R_TOP_Y   = _TOP_DM_CY   - _PAT_CY
_R_RIGHT_X = _RIGHT_DM_CX - _PAT_CX
_R_RIGHT_Y = _RIGHT_DM_CY - _PAT_CY

_CANONICAL_DM_DIST = float(np.linalg.norm(
    np.array([_RIGHT_DM_CX - _TOP_DM_CX,
              _RIGHT_DM_CY - _TOP_DM_CY])))

_PATTERN_PX = float(_pat_w)

print(f"[LAYOUT] top_dm_center=({_TOP_DM_CX:.1f},{_TOP_DM_CY:.1f}) "
      f"right_dm_center=({_RIGHT_DM_CX:.1f},{_RIGHT_DM_CY:.1f}) "
      f"pattern_center=({_PAT_CX:.1f},{_PAT_CY:.1f}) "
      f"canonical_dm_dist={_CANONICAL_DM_DIST:.2f} "
      f"pattern_px={_PATTERN_PX:.0f}")


import collections as _collections
_ZxingRect    = _collections.namedtuple("ZxingRect",    ["left", "top", "width", "height"])
_ZxingDecoded = _collections.namedtuple("ZxingDecoded", ["data", "rect", "corners_cv"])

_B32_PAIR_RE = re.compile(r'^[A-Z2-7]{4}$')


def _zxing_obj_from_barcode(b, img_h):
    """Build a _ZxingDecoded from a single zxingcpp.Barcode result."""
    pos = b.position
    pts = [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    corners_cv = np.float32([[p.x, p.y] for p in pts])
    left   = min(xs)
    width  = max(xs) - min(xs)
    height = max(ys) - min(ys)
    top    = img_h - min(ys) - height          # y-down → y-up
    rect   = _ZxingRect(left=left, top=top, width=width, height=height)
    return _ZxingDecoded(data=b.text.strip().encode("utf-8"), rect=rect,
                         corners_cv=corners_cv)


def _decoded_from_corners(text, corners_cv, img_h):
    """Build a _ZxingDecoded from a (text, corners_cv) pair (pylibdmtx fallback).
    corners_cv is already in full-image y-down pixel coords."""
    xs = corners_cv[:, 0]
    ys = corners_cv[:, 1]
    left   = float(xs.min())
    width  = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    top    = img_h - float(ys.min()) - height  # y-down → y-up
    rect   = _ZxingRect(left=left, top=top, width=width, height=height)
    return _ZxingDecoded(data=text.encode("utf-8"), rect=rect, corners_cv=corners_cv)


def _pair_valid(texts):
    """Return True if any pair in `texts` forms a valid seed via recombine_seed_from_dm."""
    valid = [t for t in texts if _B32_PAIR_RE.match(t)]
    for s1, s2 in itertools.combinations(valid, 2):
        if recombine_seed_from_dm(s1, s2) is not None:
            return True
        if recombine_seed_from_dm(s2, s1) is not None:
            return True
    return False


def decode_dm_zxing(image_bgr):
    """Exhaustive zxing-cpp sweep: 4 preprocessing × 4 binarizer combinations.

    Stops as soon as any combination finds >= 2 DataMatrix results whose texts
    form a validated seed pair (recombine_seed_from_dm passes in either order).
    If a combination finds only 1 valid code, it is tracked as a "best partial"
    and reported even on total failure.

    Returns (raw_results, pair_found):
        raw_results  – list of _ZxingDecoded from the winning combination, or
                       best partial seen if no pair was validated
        pair_found   – True iff a validated seed pair was confirmed
    """
    img_h = image_bgr.shape[0]
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variants = [
        ("raw",   image_bgr),
        ("gray",  gray),
        ("clahe", clahe.apply(gray)),
        ("otsu",  otsu),
    ]
    binarizers = [
        ("LocalAverage",    zxingcpp.Binarizer.LocalAverage),
        ("GlobalHistogram", zxingcpp.Binarizer.GlobalHistogram),
        ("FixedThreshold",  zxingcpp.Binarizer.FixedThreshold),
        ("BoolCast",        zxingcpp.Binarizer.BoolCast),
    ]

    best_partial = []   # largest non-empty result set seen without a valid pair

    for prep_name, prep_img in variants:
        for bin_name, binarizer in binarizers:
            try:
                barcodes = zxingcpp.read_barcodes(
                    prep_img,
                    formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True, try_downscale=True, try_invert=True,
                    binarizer=binarizer,
                )
            except Exception as _e:
                print(f"[ZXING] {prep_name}/{bin_name}: error — {_e}")
                continue

            results = [_zxing_obj_from_barcode(b, img_h)
                       for b in barcodes if b.text.strip()]
            texts   = [r.data.decode("utf-8") for r in results]
            print(f"[ZXING] {prep_name}/{bin_name}: {len(results)} result(s): {sorted(texts)}")

            if _pair_valid(texts):
                print(f"[ZXING] Valid pair confirmed at {prep_name}/{bin_name} — sweep done")
                return results, True

            if len(results) > len(best_partial):
                best_partial = results
                if results:
                    print(f"[ZXING] New best partial ({len(results)} code(s)): {sorted(texts)}")

    if best_partial:
        partial_texts = sorted(r.data.decode("utf-8") for r in best_partial)
        print(f"[ZXING] Sweep exhausted — no valid pair. Best partial: {partial_texts}")
    else:
        print("[ZXING] Sweep exhausted — no DataMatrix codes detected in any combination")
    return best_partial, False


def _decode_dm_corners_pylibdmtx(gray, shrink=1):
    """Decode DataMatrix codes using libdmtx fit2raw to get exact perspective
    corners in full-image y-down pixel coordinates.
    Used only by the pylibdmtx fallback in _pylibdmtx_sweep().
    Returns list of (text, corners_cv) with corners float32 (4,2)."""
    from ctypes import cast, string_at
    from pylibdmtx import pylibdmtx as P
    from pylibdmtx.wrapper import DmtxVector2, dmtxMatrix3VMultiplyBy, c_ubyte_p
    h_full = gray.shape[0]
    pixels, width, height, bpp = P._pixel_data(gray)
    out = []
    with P._image(cast(pixels, c_ubyte_p), width, height, P._PACK_ORDER[bpp]) as img:
        with P._decoder(img, shrink) as decoder:
            while True:
                with P._region(decoder, None) as region:
                    if not region:
                        break
                    with P._decoded_matrix_region(decoder, region, P.DmtxUndefined) as msg:
                        if not msg:
                            continue
                        try:
                            text = string_at(msg.contents.output).decode("utf-8", "replace").strip()
                        except Exception:
                            text = ""
                        corners = []
                        for (u, v) in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
                            p = DmtxVector2(u, v)
                            dmtxMatrix3VMultiplyBy(p, region.contents.fit2raw)
                            corners.append([shrink * p.X, h_full - shrink * p.Y])
                        out.append((text, np.float32(corners)))
    return out


def _pylibdmtx_sweep(image_bgr):
    """Last-resort fallback: pylibdmtx fit2raw decode at shrink 1–4.

    Tracks a running union of decoded texts across shrink levels and validates
    pairs at each level (fixes the old early-break-on-count bug). Stops as
    soon as the union contains a valid seed pair.

    Returns (raw_results, pair_found) in the same format as decode_dm_zxing().
    All result objects use the same _ZxingDecoded shim so downstream code
    (detect_and_crop_pattern, align_crop_by_dm_corners) needs no changes.
    """
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_h = gray.shape[0]

    seen             = {}   # text -> _ZxingDecoded (union, first occurrence wins)
    best_level_count = 0
    best_level_results = []

    for sh in (1, 2, 3, 4):
        try:
            pairs = _decode_dm_corners_pylibdmtx(gray, shrink=sh)
        except Exception as _e:
            print(f"[PYLIBDMTX] shrink={sh}: error — {_e}")
            continue

        level_texts = []
        for text, corners_cv in pairs:
            if not text:
                continue
            r = _decoded_from_corners(text, corners_cv, img_h)
            level_texts.append(text)
            if seen.setdefault(text, r) is r:   # only log genuinely new codes
                pass
            seen[text] = r                       # always keep latest corners

        print(f"[PYLIBDMTX] shrink={sh}: {len(pairs)} DM(s): {sorted(level_texts)}")

        if len(pairs) > best_level_count:
            best_level_count   = len(pairs)
            best_level_results = list(seen.values())

        # Validate on union so cross-level pairs are caught
        union_texts = list(seen.keys())
        if _pair_valid(union_texts):
            union_results = list(seen.values())
            print(f"[PYLIBDMTX] Valid pair in union after shrink={sh}: {sorted(union_texts)}")
            return union_results, True

    union_results = list(seen.values())
    if union_results:
        print(f"[PYLIBDMTX] Sweep exhausted — no valid pair. "
              f"Union: {sorted(r.data.decode('utf-8') for r in union_results)}")
    else:
        print("[PYLIBDMTX] Sweep exhausted — no DataMatrix codes detected")
    return best_level_results, False


def extract_seed_from_image(image_bgr):
    """
    Step 1 — DM Detection and Seed Recovery.

    Decodes all DataMatrix codes in image_bgr, finds the share_a/share_b pair
    that passes recombine_seed_from_dm(), and returns the recombined seed.

    Decode strategy (two-path fallback):
      Path 1 — zxing-cpp exhaustive sweep (4 preprocessing × 4 binarizer).
               Stops at the first combination that yields a validated pair.
      Path 2 — pylibdmtx fit2raw fallback (shrink 1–4, union-validated).
               Only runs if the full zxing-cpp sweep produced no valid pair.

    Returns:
        (seed: int, diagnostic: dict)  on success
        (None, diagnostic: dict)       on failure

    diagnostic always contains raw_results (list of _ZxingDecoded), shrink_used
    (always 1 — rects/corners are full-image coords regardless of path), and
    enough detail to diagnose failures from the log alone.
    """
    # ── Path 1: zxing-cpp exhaustive sweep ───────────────────────────────────
    print("[EXTRACT_SEED] === zxing-cpp exhaustive sweep ===")
    raw_results, pair_found = decode_dm_zxing(image_bgr)
    path_used = "zxing-cpp"

    if not pair_found:
        # ── Path 2: pylibdmtx last-resort fallback ────────────────────────────
        print("[EXTRACT_SEED] zxing-cpp sweep exhausted, falling back to pylibdmtx")
        pylib_results, pylib_ok = _pylibdmtx_sweep(image_bgr)
        if pylib_results:           # use pylibdmtx results even if no pair found
            raw_results = pylib_results
            pair_found  = pylib_ok
        path_used = "pylibdmtx"

    # Deduplicate and collect decoded strings
    seen = {}
    for r in raw_results:
        try:
            t = r.data.decode("utf-8").strip()
        except Exception:
            continue
        seen.setdefault(t, r)

    decoded_strings = list(seen.keys())
    print(f"[EXTRACT_SEED] [{path_used}] {len(decoded_strings)} DM string(s): "
          f"{sorted(decoded_strings)}")

    diagnostic = {
        "raw_decoded":   decoded_strings,
        "share_a":       None,
        "share_b":       None,
        "num_dms_found": len(raw_results),
        "raw_results":   raw_results,
        "shrink_used":   1,     # rects/corners always in full-image coords
    }

    if len(decoded_strings) < 2:
        diagnostic["failure_reason"] = (
            f"Both decode paths exhausted. "
            f"Best candidates [{path_used}]: {sorted(decoded_strings) or ['none']}"
        )
        print(f"[EXTRACT_SEED] Failed — {diagnostic['failure_reason']}")
        return None, diagnostic

    # Find the valid share pair
    for s1, s2 in itertools.combinations(decoded_strings, 2):
        if not (_B32_PAIR_RE.match(s1) and _B32_PAIR_RE.match(s2)):
            continue
        seed = recombine_seed_from_dm(s1, s2)
        if seed is not None:
            diagnostic["share_a"] = s1
            diagnostic["share_b"] = s2
            print(f"[EXTRACT_SEED] Seed recovered [{path_used}]: {seed} "
                  f"from share_a={s1}, share_b={s2}")
            return seed, diagnostic
        seed = recombine_seed_from_dm(s2, s1)
        if seed is not None:
            diagnostic["share_a"] = s2
            diagnostic["share_b"] = s1
            print(f"[EXTRACT_SEED] Seed recovered [{path_used}]: {seed} "
                  f"from share_a={s2}, share_b={s1}")
            return seed, diagnostic

    diagnostic["failure_reason"] = (
        f"No valid share pair [{path_used}] "
        f"(check-character mismatch on all combinations of {sorted(decoded_strings)})"
    )
    print(f"[EXTRACT_SEED] Failed — {diagnostic['failure_reason']}")
    return None, diagnostic



def extract_pattern_roi(image_bgr, dm_results):
    """
    Step 2 - Pattern ROI Extraction.

    Given the raw pylibdmtx decode result list, classifies which code is the
    Top DM and which is the Right DM (by bounding-box area), then uses the
    same dx/dy quadrant logic as detect_and_crop_pattern to locate the 512x512
    CDP pattern region and extract it via a perspective warp.

    Args:
        image_bgr:  The full camera photo (BGR, any resolution).
        dm_results: The list returned by pylibdmtx.decode().

    Returns:
        512x512 BGR crop of the CDP pattern, or None on geometry failure.
    """
    img_h, img_w = image_bgr.shape[:2]

    if len(dm_results) < 2:
        print("[EXTRACT_ROI] Need at least 2 DM results - aborting.")
        return None

    # Classify Top DM (larger area) and Right DM (smaller area)
    sorted_by_area = sorted(
        dm_results,
        key=lambda r: abs(r.rect.width) * abs(r.rect.height),
        reverse=True
    )
    top_res   = sorted_by_area[0]
    right_res = sorted_by_area[1]
    top_r     = top_res.rect
    right_r   = right_res.rect

    print(f"[EXTRACT_ROI] Top DM rect (y-up): left={top_r.left}, top={top_r.top}, "
          f"w={top_r.width}, h={top_r.height}")
    print(f"[EXTRACT_ROI] Right DM rect (y-up): left={right_r.left}, top={right_r.top}, "
          f"w={right_r.width}, h={right_r.height}")

    # Convert pylibdmtx y-up rects to OpenCV y-down centroids
    # pylibdmtx: y=0 at bottom-left of image.
    # Formula: y_cv = image_height - (rect.top + rect.height)
    def _center_cv(rect, height):
        x_cv = rect.left + abs(rect.width) / 2.0
        y_cv_top = height - (rect.top + rect.height)
        y_cv = y_cv_top + abs(rect.height) / 2.0
        return x_cv, y_cv

    tx, ty = _center_cv(top_r, img_h)
    rx, ry = _center_cv(right_r, img_h)

    dx = rx - tx
    dy = ry - ty

    print(f"[EXTRACT_ROI] Top centre (cv): ({tx:.1f}, {ty:.1f}), "
          f"Right centre (cv): ({rx:.1f}, {ry:.1f}), dx={dx:.1f}, dy={dy:.1f}")

    tw_abs = abs(top_r.width)
    th_abs = abs(top_r.height)
    rw_abs = abs(right_r.width)
    rh_abs = abs(right_r.height)

    if dx > 0 and dy > 0:      # Normal (0 deg)
        px, py = tx, ry
        pw, ph = tw_abs, rh_abs
    elif dx < 0 and dy > 0:    # 90 deg CCW
        px, py = rx, ty
        pw, ph = th_abs, rw_abs
    elif dx < 0 and dy < 0:    # 180 deg
        px, py = tx, ry
        pw, ph = tw_abs, rh_abs
    else:                      # 270 deg CCW (dx > 0, dy < 0)
        px, py = rx, ty
        pw, ph = th_abs, rw_abs

    print(f"[EXTRACT_ROI] Pattern centre (cv): ({px:.1f}, {py:.1f}), "
          f"estimated size: {pw:.0f}x{ph:.0f}")

    half_w, half_h = pw / 2.0, ph / 2.0
    src_pts = np.float32([
        [px - half_w, py - half_h],
        [px + half_w, py - half_h],
        [px + half_w, py + half_h],
        [px - half_w, py + half_h],
    ])

    src_pts[:, 0] = np.clip(src_pts[:, 0], 0, img_w - 1)
    src_pts[:, 1] = np.clip(src_pts[:, 1], 0, img_h - 1)

    out_w, out_h = PATTERN_SIZE
    dst_pts = np.float32([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ])

    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped = cv2.warpPerspective(image_bgr, M, (out_w, out_h))
        print(f"[EXTRACT_ROI] ROI extracted - shape: {cropped.shape}")
        return cropped
    except Exception as e:
        print(f"[EXTRACT_ROI] getPerspectiveTransform failed: {e}")
        return None


def regenerate_reference(seed, block_size=BLOCK_SIZE, pattern_size=None):
    """
    Step 3 - In-Memory Reference Generation.

    Recreates the exact CDP that was originally generated for this seed,
    WITHOUT writing anything to disk. Mirrors generate_pattern() exactly.

    block_size MUST match the value used at generation time. The PRNG macro
    pattern and RGB perturbations are seeded over a block grid whose dimensions
    depend on block_size, so the same seed at a different block_size produces a
    completely different random field — making correlation/color tests compare
    the capture against the wrong pattern. The caller is responsible for passing
    the block_size stored in the DB for this seed (defaults to BLOCK_SIZE only as
    a last resort).

    Returns:
        (reference_rgb: np.ndarray H x W x 3 uint8,
         reference_gray: np.ndarray H x W uint8)
    """
    print(f"[REGEN] Regenerating reference for seed={seed} (0x{seed:08x}) "
          f"block_size={block_size} ...")

    w, h = (pattern_size, pattern_size) if pattern_size else PATTERN_SIZE

    grating_rng = np.random.RandomState(seed=seed + 2000)
    base_freq   = 8   + grating_rng.random() * 6
    mod_freq    = 1.5 + grating_rng.random() * 2.5
    mod_depth   = 0.2 + grating_rng.random() * 0.3

    grating     = generate_frequency_modulated_grating(w, h, base_freq, mod_freq, mod_depth)
    prng        = generate_prng_macro_pattern(w, h, seed, block_size)
    combined    = cv2.addWeighted(grating, 0.5, prng, 0.5, 0)
    pattern_rgb = add_rgb_perturbations(combined, seed=seed, intensity=25, block_size=block_size)
    add_fiducial_markers(pattern_rgb, marker_color=0)

    reference_gray = cv2.cvtColor(pattern_rgb, cv2.COLOR_RGB2GRAY)

    print(f"[REGEN] Done - RGB shape: {pattern_rgb.shape}, Gray shape: {reference_gray.shape}")
    return pattern_rgb, reference_gray


def verify_pattern(captured_path, uploads_dir="uploads"):
    """
    New verify_pattern (Step 4).

    Fully self-contained verification pipeline that does NOT require the
    original pattern file on disk. The reference is always regenerated from
    the seed recovered via the DataMatrix codes in the captured image.

    Unlike verify_pattern_legacy, this takes only captured_path (no original_path).

    Args:
        captured_path: Path to the smartphone photo.
        uploads_dir:   Directory for debug images.

    Returns:
        dict with keys: verdict, confidence, seed_recovered, scores, weights,
        dm_diagnostic, roi_filename, reference_filename, aligned_filename.

    KNOWN GAP (2026-07-02): fine-alignment correction (Step 2c) fixes false
    COUNTERFEIT verdicts on genuine labels captured with slight misalignment
    (confirmed on tc-03 rotated, tc-05 low-light, tc-06 glare).  However, it
    also raises correlation/moire scores for at least one known counterfeit
    sample (tc-09), causing it to cross THRESHOLD_AUTHENTIC.

    Working hypothesis: the counterfeit's printer could not reproduce the CDP
    pattern at sufficient fidelity to survive alignment — meaning fine alignment
    removes an INCIDENTAL defense that was never the intended detection
    mechanism.  The tc-09 sample happened to be caught previously only because
    capture misalignment degraded its PRNG correlation below threshold.

    This is unverified against only one counterfeit sample.  Needs:
      (a) more counterfeit samples across different reproduction techniques to
          test whether the incidental-misalignment catch is representative;
      (b) potential redesign of the CDP pattern itself to encode higher-frequency
          detail that is difficult for common printers to reproduce even under
          correct alignment, making moire/correlation a real rather than
          incidental discriminator.
    """
    os.makedirs(uploads_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Load image
    print(f"[VERIFY] Loading: {captured_path}")
    captured_bgr = cv2.imread(captured_path)
    if captured_bgr is None:
        return {
            "verdict":       "UNABLE_TO_VERIFY",
            "error":         f"Cannot load image: {captured_path}",
            "dm_diagnostic": {},
        }
    print(f"[VERIFY] Shape: {captured_bgr.shape}")

    # Step 1 - Seed recovery from DMs on the full image
    print("[VERIFY] === STEP 1: DM Detection & Seed Recovery ===")
    seed, dm_diagnostic = extract_seed_from_image(captured_bgr)

    if seed is None:
        print(f"[VERIFY] Seed recovery failed: {dm_diagnostic.get('failure_reason')}")
        return {
            "verdict":       "UNABLE_TO_VERIFY",
            "error":         "DM decode failed",
            "dm_diagnostic": dm_diagnostic,
        }

    # Step 1b - Resolve the block_size this seed was generated with.
    # The DataMatrix payload only encodes the seed, not block_size, but the
    # reference MUST be regenerated at the same block_size used at generation
    # time (otherwise the PRNG/color field is a different random pattern — see
    # regenerate_reference). Look it up in the DB by seed; fall back to the
    # BLOCK_SIZE default only when no matching pattern is registered.
    block_size = BLOCK_SIZE
    pattern_size = None
    try:
        from database import get_db
        _p = get_db().patterns.find_one({"seed": seed})
        if _p:
            block_size   = int(_p.get("block_size", BLOCK_SIZE))
            pattern_size = int(_p["pattern_size"]) if _p.get("pattern_size") else None
            print(f"[VERIFY] Pattern params from DB: block_size={block_size}, "
                  f"pattern_size={pattern_size}")
        else:
            print(f"[VERIFY] No registered pattern for seed={seed}; "
                  f"using default block_size={block_size}")
    except Exception as _e:
        print(f"[VERIFY] block_size lookup failed ({_e}); using default block_size={block_size}")

    # Step 2 - Pattern ROI via detect_and_crop_pattern.
    # Pass the DM decode results from Step 1 directly so the cropper does NOT
    # need to run a second decode pass on the flattened label.  With smaller
    # 8x18 symbols the second decode frequently fails on the warped image even
    # when Step 1 succeeded on the full-resolution frame.
    print("[VERIFY] === STEP 2: Pattern ROI Extraction (card-flatten + DM geometry) ===")
    crop_result = detect_and_crop_pattern(
        captured_bgr,
        dm_results_raw=dm_diagnostic.get("raw_results"),
        dm_shrink_used=dm_diagnostic.get("shrink_used", 1),
        seed=seed,
        debug_image_name=os.path.splitext(os.path.basename(captured_path))[0],
        block_size=block_size,
    )

    roi_bgr = crop_result[0] if crop_result[2] else None
    if roi_bgr is None:
        print("[VERIFY] ROI extraction failed.")
        return {
            "verdict":       "UNABLE_TO_VERIFY",
            "error":         "Pattern ROI extraction failed — could not locate/decode DMs after label flattening",
            "dm_diagnostic": dm_diagnostic,
        }

    # Step 2b - ALIGN-THEN-CROP via the two DataMatrix codes' EXACT corners.
    # The rough crop above uses only the 2 DM centers (a similarity transform) and
    # cannot undo perspective warp. libdmtx gives each DM's precise perspective
    # corners (fit2raw); their 8 corners -> one homography that warps the full
    # photo straight to the canonical pattern (alignment + crop in one resampling).
    # Falls back to the DM rough crop if either DM can't be cornered.
    print("[VERIFY] === STEP 2b: Align-then-Crop via DM corners ===")
    print("[VERIFY] Step 2b reusing Step 1 zxing-cpp corners (no re-decode)")
    aligned_roi, fid_ok, dbg_align = None, False, {}
    try:
        aligned_roi, fid_ok = align_crop_by_dm_corners(
            captured_bgr, dm_diagnostic["raw_results"], debug=dbg_align
        )
    except Exception as _e:
        print(f"[VERIFY] DM-corner align error ({_e}) -> Layer 3 fallback")
        aligned_roi, fid_ok = None, False
    _step2b_layer3_fallback = False
    if fid_ok:
        roi_bgr = aligned_roi
        print("[VERIFY] Step 2b: align-then-crop via exact DM corners (8-point homography)")
    else:
        _step2b_layer3_fallback = True
        roi_bgr = crop_result[0]          # revert to Layer 3 crop
        stage = dbg_align.get("stage", "unknown")
        if stage == "order_degenerate":
            print("[VERIFY] Step 2b: corner ordering ambiguous/degenerate, falling back to Layer 3 crop")
        else:
            print(f"[VERIFY] Step 2b: align failed ({stage}), falling back to Layer 3 crop")

    # Annotated "Detection & Alignment" overlay for the UI: the full photo with the
    # detected DM corners and the derived pattern region drawn on it.
    detection_filename = f"detection_{ts}.png"
    try:
        overlay = _draw_alignment_overlay(captured_bgr, dbg_align)
        cv2.imwrite(os.path.join(uploads_dir, detection_filename), overlay)
    except Exception as _e:
        print(f"[VERIFY] detection overlay failed ({_e})")
        detection_filename = None

    # Raw, un-rectified crop of the pattern region straight from the photo, kept
    # for the "Cropped pattern" debug image. This is the pattern AS CAPTURED
    # (perspective/tilt intact, with a little surrounding label context) — it is
    # deliberately distinct from the rectified+oriented ROI that gets scored and
    # saved as the "Aligned" image. Previously both files were the same array,
    # so the two panels looked identical.
    raw_crop_bgr = None
    try:
        bx, by, bw, bh = crop_result[1]
        H_img, W_img = captured_bgr.shape[:2]
        x0, y0 = max(0, int(bx)), max(0, int(by))
        x1, y1 = min(W_img, int(bx + bw)), min(H_img, int(by + bh))
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            raw_crop_bgr = captured_bgr[y0:y1, x0:x1].copy()
    except Exception as _e:
        print(f"[VERIFY] raw-crop capture failed ({_e}); cropped image will mirror aligned")

    # Step 3 - Regenerate reference at the SAME block_size used at generation
    print("[VERIFY] === STEP 3: Reference Regeneration ===")
    reference_rgb, reference_gray = regenerate_reference(
        seed, block_size=block_size, pattern_size=pattern_size)
    reference_bgr = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)

    # Deferred Step 2b outcome log — NCC is only computable now that reference exists
    if _step2b_layer3_fallback:
        _l3_ncc = _ncc_thumbnail(roi_bgr, reference_gray)
        _msg = f"[STEP2B] DEFERRED: Step 2b fell back to Layer 3 crop  ncc={_l3_ncc:.4f}"
        print(_msg)
        _crop_logger.info(_msg)

    # Step 4 - Prepare captured ROI (resize to canonical PATTERN_SIZE first so
    # that Step 2c fine-alignment always receives a guaranteed 512×512 input
    # matching the canonical fiducial coordinate system).
    print("[VERIFY] === STEP 4: Preparing Captured ROI ===")
    out_w, out_h = PATTERN_SIZE
    if roi_bgr.shape[:2] != (out_h, out_w):
        current_h, current_w = roi_bgr.shape[:2]
        interp = cv2.INTER_LANCZOS4 if (current_w < out_w or current_h < out_h) else cv2.INTER_AREA
        roi_bgr = cv2.resize(roi_bgr, (out_w, out_h), interpolation=interp)

    # Step 2c - Fine alignment via fiducial markers.
    # Runs after Step 4 resize so roi_bgr is guaranteed 512×512.  The corrected
    # crop becomes the sole input to Step 4b NCC and Step 5 scoring; the coarse
    # crop is discarded.  If fewer than 3 fiducials are detected the function
    # returns the input unchanged and _fa_applied is False.
    print("[VERIFY] === STEP 2c: Fine Alignment via Fiducial Markers ===")
    _fa_debug = {}
    roi_bgr, _fa_applied = fine_align_via_fiducials(roi_bgr, debug=_fa_debug)
    _fa_n  = _fa_debug.get('fa_n_detected', 0)
    _fa_dx = _fa_debug.get('fa_dx', 0.0)
    _fa_dy = _fa_debug.get('fa_dy', 0.0)
    print(f"[FINE-ALIGN] applied={_fa_applied} n_detected={_fa_n} "
          f"dx={_fa_dx:+.1f} dy={_fa_dy:+.1f}")

    captured_rgb  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    captured_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    print("[VERIFY] === STEP 4b: NCC check on fine-aligned crop vs reference ===")
    _ncc_4b = _ncc_thumbnail(roi_bgr, reference_gray)
    print(f"[VERIFY] Step 4b: fine-aligned roi_bgr vs reference NCC = {_ncc_4b:.4f}")

    # Step 5 - Run all four tests. Marker regions are neutralized in both capture
    # and reference so the fiducials themselves contribute no discrimination signal.
    print("[VERIFY] === STEP 5: Running Verification Tests ===")
    cap_gray_s = _blank_fiducials(captured_gray)
    cap_rgb_s  = _blank_fiducials(captured_rgb)
    ref_gray_s = _blank_fiducials(reference_gray)
    ref_rgb_s  = _blank_fiducials(reference_rgb)
    moire             = test_moire_detection(cap_gray_s, ref_gray_s)
    color             = test_color_analysis(cap_rgb_s, ref_rgb_s)
    corr, raw_corr    = test_prng_correlation(cap_gray_s, ref_gray_s, block_size)
    gradient          = test_gradient_energy(cap_gray_s, ref_gray_s)

    print(f"[VERIFY] moire={moire:.4f}  color={color:.4f}  "
          f"corr={corr:.4f} (raw={raw_corr:.4f})  gradient={gradient:.4f}")

    # Step 6 - Score & verdict
    final = (WEIGHT_MOIRE       * moire    +
             WEIGHT_COLOR       * color    +
             WEIGHT_CORRELATION * corr     +
             WEIGHT_GRADIENT    * gradient)

    if final >= THRESHOLD_AUTHENTIC:
        verdict = "AUTHENTIC"
    elif final >= THRESHOLD_SUSPICIOUS:
        verdict = "SUSPICIOUS"
    else:
        verdict = "COUNTERFEIT"

    print(f"[VERIFY] Final score={final:.4f} -> verdict={verdict}")

    # Step 7 - Save debug images
    print("[VERIFY] === STEP 6: Saving Debug Images ===")
    roi_filename       = f"roi_{ts}.png"
    reference_filename = f"reference_{ts}.png"
    aligned_filename   = f"aligned_{ts}.png"
    fiducial_filename  = f"fiducial_{ts}.png"

    # "Cropped pattern" = raw in-photo region (perspective intact); falls back to
    # the rectified ROI only if the raw crop couldn't be taken.
    # "Aligned" = fine-aligned, orientation-corrected ROI that was actually scored.
    cropped_to_save = raw_crop_bgr if raw_crop_bgr is not None else roi_bgr
    cv2.imwrite(os.path.join(uploads_dir, roi_filename),       cropped_to_save)
    cv2.imwrite(os.path.join(uploads_dir, reference_filename), reference_bgr)
    cv2.imwrite(os.path.join(uploads_dir, aligned_filename),   roi_bgr)

    # Fiducial markers debug image: fine-aligned crop with each corner's
    # canonical expected position (white ring outline) and detected position
    # (filled coloured dot + line from canonical) annotated.
    # TL=red  TR=green  BR=blue  BL=yellow  (BGR colour space)
    try:
        _CORNER_COLORS = {
            'TL': (0,   0,   255),
            'TR': (0,   255, 0  ),
            'BR': (255, 0,   0  ),
            'BL': (0,   255, 255),
        }
        _fid_vis = roi_bgr.copy()
        _fid_accepted_labels = {e['label'] for e in _fa_debug.get('fa_accepted', [])}
        for _e in _fa_debug.get('fa_accepted', []) + _fa_debug.get('fa_rejected', []):
            _lbl = _e['label']
            _col = _CORNER_COLORS.get(_lbl, (128, 128, 128))
            _cx, _cy = int(_e['canon'][0]), int(_e['canon'][1])
            # Canonical expected position: thin white ring
            cv2.circle(_fid_vis, (_cx, _cy), FIDUCIAL_RING_R, (255, 255, 255), 1, cv2.LINE_AA)
            if _lbl in _fid_accepted_labels and 'detected' in _e:
                _dcx, _dcy = int(_e['detected'][0]), int(_e['detected'][1])
                # Detected position: filled coloured dot + white outline
                cv2.circle(_fid_vis, (_dcx, _dcy), 8, _col, -1, cv2.LINE_AA)
                cv2.circle(_fid_vis, (_dcx, _dcy), 9, (255, 255, 255), 1, cv2.LINE_AA)
                # Line from canonical to detected to visualise offset
                cv2.line(_fid_vis, (_cx, _cy), (_dcx, _dcy), _col, 1, cv2.LINE_AA)
                cv2.putText(_fid_vis, f"{_lbl} {_e['score']:.2f}",
                            (_dcx + 11, _dcy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _col, 1, cv2.LINE_AA)
            else:
                # Rejected: grey X at canonical position
                cv2.line(_fid_vis, (_cx-6, _cy-6), (_cx+6, _cy+6), (80, 80, 80), 2, cv2.LINE_AA)
                cv2.line(_fid_vis, (_cx+6, _cy-6), (_cx-6, _cy+6), (80, 80, 80), 2, cv2.LINE_AA)
                cv2.putText(_fid_vis, f"{_lbl} {_e.get('score', 0):.2f}",
                            (_cx + 11, _cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
        _fid_legend = (f"fine-align: {'applied' if _fa_applied else 'skipped'}"
                       f"  n={_fa_n}/4  dx={_fa_dx:+.1f}  dy={_fa_dy:+.1f}")
        cv2.putText(_fid_vis, _fid_legend, (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(uploads_dir, fiducial_filename), _fid_vis)
    except Exception as _fid_e:
        print(f"[VERIFY] fiducial debug image failed ({_fid_e})")
        fiducial_filename = None

    print(f"[VERIFY] Saved: {roi_filename}, {reference_filename}, "
          f"{aligned_filename}, {fiducial_filename}")

    return {
        "verdict":            verdict,
        "confidence":         float(final),
        "seed_recovered":     seed,
        "scores": {
            "moire":       float(moire),
            "color":       float(color),
            "correlation": float(corr),
            "gradient":    float(gradient),
        },
        "weights": {
            "moire":       WEIGHT_MOIRE,
            "color":       WEIGHT_COLOR,
            "correlation": WEIGHT_CORRELATION,
            "gradient":    WEIGHT_GRADIENT,
        },
        "dm_diagnostic":      dm_diagnostic,
        "roi_filename":       roi_filename,
        "reference_filename": reference_filename,
        "aligned_filename":   aligned_filename,
        "detection_filename": detection_filename,
        "fiducial_filename":  fiducial_filename,
        "align_method":       "dm_corners" if fid_ok else "rough_crop",
        "fine_align_applied": _fa_applied,
        "fine_align_n":       _fa_n,
        "fine_align_dx":      _fa_dx,
        "fine_align_dy":      _fa_dy,
    }
