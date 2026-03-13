"""
PhoneCDP Core Engine — Pattern Generation & Verification

Self-contained module with all CDP logic. No external PhoneCDP imports.
"""

import numpy as np
import cv2
from datetime import datetime
import os

# ==========================================================================
# CONSTANTS
# ==========================================================================

PATTERN_SIZE = (512, 512)
FIDUCIAL_MARKER_SIZE   = 64   # ring structure diameter — controls ring geometry
FIDUCIAL_MARKER_OFFSET = 34   # center-to-corner distance — bg circle edge sits 3px from pattern border
RING_COUNT_TO_CORNER = {1: "top_left", 2: "top_right", 3: "bottom_left", 0: "bottom_right"}

# Ring geometry — derived from FIDUCIAL_MARKER_SIZE // 2 (NOT OFFSET) so placement
# distance can be changed independently without inflating/deflating ring radii.
_MHF          = FIDUCIAL_MARKER_SIZE // 2                  # 32 — ring geometry base
_RING_BG_R    = _MHF - 1                                   # 31 — background circle radius
_RING_SAMPLE_MAX = _MHF - 2                                # 30 — sampling cap
_RING_OUTER  = int(round(20 * _MHF / 24))                  # 27 — outer ring (all non-BR)
_RING_MID    = int(round(13 * _MHF / 24))                  # 17 — BL mid ring
_RING_MID2   = int(round(10 * _MHF / 24))                  # 13 — TR inner ring
_RING_INNER  = int(round( 6 * _MHF / 24))                  #  8 — BL inner ring
_RING_THICK  = int(round( 3 * _MHF / 24))                  #  4 — draw thickness
BLOCK_SIZE = 8

# Verification weights
# Correlation measures PRNG block reproduction — the most CDP-specific test.
# Moire measures frequency spectrum; color stats are similar for genuine/counterfeit.
WEIGHT_MOIRE = 0.40
WEIGHT_COLOR = 0.10
WEIGHT_CORRELATION = 0.35
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


def generate_prng_macro_pattern(width, height, seed, block_size=8):
    rng = np.random.RandomState(seed)
    blocks_x = width // block_size
    blocks_y = height // block_size
    block_values = rng.randint(0, 256, size=(blocks_y, blocks_x)).astype(np.uint8)
    pattern = np.repeat(np.repeat(block_values, block_size, axis=0), block_size, axis=1)
    return pattern[:height, :width]


def add_rgb_perturbations(pattern_gray, seed=42, intensity=25, block_size=8):
    rng = np.random.RandomState(seed + 1000)
    h, w = pattern_gray.shape
    pattern_rgb = cv2.cvtColor(pattern_gray, cv2.COLOR_GRAY2RGB)
    blocks_y = h // block_size
    blocks_x = w // block_size
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


def add_fiducial_markers(pattern, marker_color=0):
    h, w = pattern.shape[:2]
    is_rgb = len(pattern.shape) == 3
    color = (marker_color,) * 3 if is_rgb else marker_color
    bg = (255, 255, 255) if is_rgb else 255
    off = FIDUCIAL_MARKER_OFFSET
    centers = [
        (off, off),
        (w - off, off),
        (off, h - off),
        (w - off, h - off),
    ]
    for cx, cy in centers:
        cv2.circle(pattern, (cx, cy), _RING_BG_R, bg, -1)
    cv2.circle(pattern, centers[0], _RING_OUTER, color, _RING_THICK)  # TL: 1 ring
    cv2.circle(pattern, centers[1], _RING_OUTER, color, _RING_THICK)  # TR: 2 rings
    cv2.circle(pattern, centers[1], _RING_MID2,  color, _RING_THICK)
    cv2.circle(pattern, centers[2], _RING_OUTER, color, _RING_THICK)  # BL: 3 rings
    cv2.circle(pattern, centers[2], _RING_MID,   color, _RING_THICK)
    cv2.circle(pattern, centers[2], _RING_INNER, color, _RING_THICK)
    cv2.circle(pattern, centers[3], _RING_OUTER, color, -1)           # BR: solid
    return pattern


def generate_pattern(output_dir, seed=None, serial_number="SN-0001", pattern_size=512, block_size=8):
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
    final = add_fiducial_markers(pattern_rgb, marker_color=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phone_cdp_{timestamp}_{seed}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w) * min_area_pct:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / (bh + 1e-10)
        if not (0.5 < aspect < 2.0):
            continue
        # Reject only if the contour AREA covers almost the whole image — a
        # rotated square's bounding rect grows to √2× its size at 45°, so a
        # bounding-rect size check would wrongly reject rotated patterns.
        if area > (h * w) * 0.85:
            continue
        region = gray[by:by + bh, bx:bx + bw]
        region_mean = float(region.mean())
        region_std = float(region.std())
        if region_mean > image_mean * 0.95:
            continue
        darkness = max(0, image_mean - region_mean)
        score = darkness + region_std
        if score > best_score:
            best_score = score
            best_contour = cnt
    return best_contour


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


def _find_markers_in_image(gray):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    min_pattern = min(w, h) * 0.08
    max_pattern = min(w, h) * 0.95
    min_r = max(int(20 * min_pattern / 512), 5)
    max_r = int(20 * max_pattern / 512)
    for param2 in [40, 30, 20]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min_r * 2,
            param1=100, param2=param2, minRadius=min_r, maxRadius=max_r)
        if circles is not None and len(circles[0]) >= 4:
            return np.round(circles[0, :30]).astype(int)
    return None


def _find_square_group(circles):
    if len(circles) < 4:
        return None
    from itertools import combinations
    best_group = None
    best_score = float("inf")
    for combo in combinations(range(len(circles)), 4):
        pts = np.array([(circles[i][0], circles[i][1]) for i in combo], dtype=np.float32)
        radii = [circles[i][2] for i in combo]
        if max(radii) > 2 * min(radii):
            continue
        dists = sorted([
            np.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            for i in range(4) for j in range(i + 1, 4)
        ])
        sides, diags = dists[:4], dists[4:]
        side_mean = np.mean(sides)
        if side_mean < 10:
            continue
        side_var = max(sides) / (min(sides) + 1e-10)
        if side_var > 1.3:
            continue
        diag_mean = np.mean(diags)
        diag_var = max(diags) / (min(diags) + 1e-10)
        if diag_var > 1.3:
            continue
        expected_diag = side_mean * np.sqrt(2)
        diag_ratio = diag_mean / (expected_diag + 1e-10)
        if not (0.75 < diag_ratio < 1.35):
            continue
        score = side_var + diag_var + abs(1 - diag_ratio)
        if score < best_score:
            best_score = score
            best_group = pts
    return best_group


def detect_and_crop_pattern(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    contour = _find_pattern_contour(gray, min_area_pct=0.003)
    if contour is not None:
        cropped, bbox, quad = _crop_from_contour(image, contour)
        return cropped, bbox, True, quad

    circles = _find_markers_in_image(gray)
    if circles is not None:
        group = _find_square_group(circles)
        if group is not None:
            # group = 4 marker centre positions (cx, cy).
            # Extrapolate the actual pattern corners from the marker centres.
            # Markers sit at off_frac from each corner; we scale outward from
            # the centroid using the inverse of that fraction so the corners are
            # correct even when the pattern is rotated or tilted.
            pts = group.astype(np.float32)
            s = pts.sum(axis=1);  d = pts[:, 1] - pts[:, 0]
            M_tl = pts[np.argmin(s)];  M_br = pts[np.argmax(s)]
            M_tr = pts[np.argmin(d)];  M_bl = pts[np.argmax(d)]
            C = (M_tl + M_tr + M_br + M_bl) / 4.0
            pw = float(max(PATTERN_SIZE))
            off = float(FIDUCIAL_MARKER_OFFSET)
            scale = (pw / 2.0) / (pw / 2.0 - off)   # ~1.153 for 34/512
            src_pts = np.float32([
                C + (M_tl - C) * scale,
                C + (M_tr - C) * scale,
                C + (M_br - C) * scale,
                C + (M_bl - C) * scale,
            ])
            out_size = max(
                int(max(np.linalg.norm(src_pts[1] - src_pts[0]),
                        np.linalg.norm(src_pts[2] - src_pts[3]))),
                int(max(np.linalg.norm(src_pts[3] - src_pts[0]),
                        np.linalg.norm(src_pts[2] - src_pts[1]))),
                64)
            dst_pts = np.float32([
                [0, 0], [out_size - 1, 0],
                [out_size - 1, out_size - 1], [0, out_size - 1]
            ])
            M_warp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            cropped = cv2.warpPerspective(image, M_warp, (out_size, out_size),
                                          borderValue=(255, 255, 255))
            bx = int(max(0, src_pts[:, 0].min()))
            by = int(max(0, src_pts[:, 1].min()))
            bw = int(min(w - bx, src_pts[:, 0].max() - src_pts[:, 0].min()))
            bh = int(min(h - by, src_pts[:, 1].max() - src_pts[:, 1].min()))
            return cropped, (bx, by, bw, bh), True, src_pts

    return image, (0, 0, w, h), False, None


# ==========================================================================
# VERIFICATION — STEP 1: Fiducial Marker Detection
# ==========================================================================

def _make_marker_templates(size=64):
    """Generate reference grayscale images for each marker type.

    Returns dict {ring_count: grayscale_image} for counts 0,1,2,3.
    The images match the actual marker drawing code in add_fiducial_markers.
    """
    templates = {}
    mhf = size // 2
    bg_r = mhf - 1
    outer  = int(round(20 * mhf / 24))
    mid2   = int(round(10 * mhf / 24))
    mid    = int(round(13 * mhf / 24))
    inner  = int(round( 6 * mhf / 24))
    thick  = int(round( 3 * mhf / 24))
    cx, cy = mhf, mhf

    for rc in [0, 1, 2, 3]:
        img = np.full((size, size), 128, dtype=np.uint8)  # neutral gray bg
        cv2.circle(img, (cx, cy), bg_r, 255, -1)          # white bg circle
        if rc == 0:
            # BR: solid fill
            cv2.circle(img, (cx, cy), outer, 0, -1)
        elif rc == 1:
            # TL: outer ring only
            cv2.circle(img, (cx, cy), outer, 0, thick)
        elif rc == 2:
            # TR: outer + mid2
            cv2.circle(img, (cx, cy), outer, 0, thick)
            cv2.circle(img, (cx, cy), mid2, 0, thick)
        elif rc == 3:
            # BL: outer + mid + inner (NOT mid2)
            cv2.circle(img, (cx, cy), outer, 0, thick)
            cv2.circle(img, (cx, cy), mid, 0, thick)
            cv2.circle(img, (cx, cy), inner, 0, thick)
        templates[rc] = img
    return templates


# Pre-generate templates at module load time
_MARKER_TEMPLATES = _make_marker_templates(64)


def count_rings_radial_profile(gray_region, cx, cy, max_radius=None, min_contrast=30):
    """Identify which fiducial marker is at (cx,cy) via template NCC matching.

    Instead of checking individual radii (which fails when rings blur into
    each other at small print sizes), we extract the circular ROI around the
    detected center and NCC-match it against pre-generated templates for
    each marker type (0/1/2/3 rings).  The template with best NCC wins.
    """
    h, w = gray_region.shape
    if max_radius is None:
        max_radius = min(cx, cy, w - 1 - cx, h - 1 - cy)
    if max_radius < 5:
        return (-1, 0)

    # Extract square ROI around marker center
    roi_half = max_radius
    y1 = max(0, cy - roi_half)
    y2 = min(h, cy + roi_half + 1)
    x1 = max(0, cx - roi_half)
    x2 = min(w, cx + roi_half + 1)
    roi = gray_region[y1:y2, x1:x2]
    if roi.shape[0] < 10 or roi.shape[1] < 10:
        return (-1, 0)

    # Check contrast
    contrast = float(np.max(roi).astype(float) - np.min(roi).astype(float))
    if contrast < min_contrast:
        return (-1, contrast)

    # Resize ROI to match template size
    tpl_size = 64
    roi_resized = cv2.resize(roi, (tpl_size, tpl_size), interpolation=cv2.INTER_AREA)

    # Apply circular mask so only the marker area contributes to NCC
    mask = np.zeros((tpl_size, tpl_size), dtype=np.uint8)
    cv2.circle(mask, (tpl_size // 2, tpl_size // 2), tpl_size // 2 - 2, 255, -1)

    # NCC match against each template
    best_rc, best_ncc = -1, -1.0
    roi_f = roi_resized.astype(np.float64)
    roi_m = roi_f[mask > 0]
    roi_m = roi_m - roi_m.mean()
    roi_n = np.linalg.norm(roi_m) + 1e-10

    for rc, tpl in _MARKER_TEMPLATES.items():
        tpl_f = tpl.astype(np.float64)
        tpl_m = tpl_f[mask > 0]
        tpl_m = tpl_m - tpl_m.mean()
        tpl_n = np.linalg.norm(tpl_m) + 1e-10
        ncc = float(np.dot(roi_m, tpl_m) / (roi_n * tpl_n))
        if ncc > best_ncc:
            best_ncc = ncc
            best_rc = rc

    # Require minimum NCC to accept the match
    if best_ncc < 0.15:
        return (-1, contrast)

    return (best_rc, contrast)


def detect_fiducial_markers(image, pattern_found_hint=False):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    h, w = gray.shape

    scale = max(w, h) / max(PATTERN_SIZE)
    # Cap at 22*scale: the marker's white background circle ends at r=23.
    # Sampling beyond r=22 hits the PRNG pattern texture, creating a false
    # bright gap + dark texture that inflates the ring count by 1.
    ring_max_radius = max(15, int(_RING_SAMPLE_MAX * scale))
    expected_r = int(_RING_OUTER * scale)
    min_r = max(expected_r - int(10 * scale), 5)
    max_r = expected_r + int(10 * scale)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = None
    for param2 in [40, 30, 20]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=expected_r * 2,
            param1=100, param2=param2, minRadius=min_r, maxRadius=max_r)
        if circles is not None and len(circles[0]) >= 4:
            break

    # When the pattern boundary has been confirmed (crop IS the pattern), centre
    # the search on the known fractional marker offset — much tighter than a
    # generic 25%-from-corner sweep and eliminates false positives elsewhere.
    # When the full image is used (pattern not confirmed), fall back to a broad
    # search from each image corner.
    off_frac = FIDUCIAL_MARKER_OFFSET / max(PATTERN_SIZE)
    if pattern_found_hint:
        search_centers = {
            "tl": (int(round(w * off_frac)),         int(round(h * off_frac))),
            "tr": (int(round(w * (1 - off_frac))),   int(round(h * off_frac))),
            "bl": (int(round(w * off_frac)),         int(round(h * (1 - off_frac)))),
            "br": (int(round(w * (1 - off_frac))),   int(round(h * (1 - off_frac)))),
        }
        search_radius = max(w, h) * 0.15   # tight — pattern boundary confirmed
    else:
        search_centers = {
            "tl": (0, 0), "tr": (w, 0), "bl": (0, h), "br": (w, h),
        }
        search_radius = max(w, h) * 0.30   # broad — pattern not confirmed

    corner_circles = {}
    if circles is not None:
        circle_list = np.round(circles[0, :]).astype(int)
        for ic_name, (sc_x, sc_y) in search_centers.items():
            best_dist, best_circle = float("inf"), None
            for cx, cy, _r in circle_list:
                dist = np.sqrt((cx - sc_x) ** 2 + (cy - sc_y) ** 2)
                if dist < search_radius and dist < best_dist:
                    best_dist = dist
                    best_circle = (int(cx), int(cy))
            if best_circle is not None:
                corner_circles[ic_name] = best_circle

    markers = {name: None for name in ["top_left", "top_right", "bottom_left", "bottom_right"]}
    markers["inferred_name"] = None
    assigned = set()

    ring_results = {}
    for ic_name, (cx, cy) in corner_circles.items():
        edge_max = min(cx, cy, w - 1 - cx, h - 1 - cy)
        mr = min(ring_max_radius, edge_max)
        if mr < 10:
            continue
        ring_count, contrast = count_rings_radial_profile(gray, cx, cy, max_radius=mr)
        ring_results[ic_name] = (ring_count, contrast, cx, cy)
        if ring_count in (0, 1, 2, 3) and contrast >= 30:
            corner_name = RING_COUNT_TO_CORNER[ring_count]
            if corner_name not in assigned:
                markers[corner_name] = (cx, cy)
                assigned.add(corner_name)

    # Store ring_results for visualization debugging
    markers["ring_results"] = ring_results

    # Positional fallback: fill any corners that ring counting didn't assign.
    # The used_positions guard prevents the same circle being double-assigned
    # (which happened when wrong ring counts mapped two corners to one position).
    # NOTE: no positional cross-check here — ring counting must be trusted for
    # rotated photos where the TL marker legitimately appears in the BR quadrant.
    used_positions = {tuple(markers[n]) for n in
                      ["top_left", "top_right", "bottom_left", "bottom_right"]
                      if markers[n] is not None}
    positional_map = {"tl": "top_left", "tr": "top_right",
                      "bl": "bottom_left", "br": "bottom_right"}
    for ic_name, (cx, cy) in corner_circles.items():
        pattern_corner = positional_map[ic_name]
        if pattern_corner not in assigned and (cx, cy) not in used_positions:
            markers[pattern_corner] = (cx, cy)
            assigned.add(pattern_corner)
            used_positions.add((cx, cy))

    markers["markers_found"] = sum(
        1 for k in ["top_left", "top_right", "bottom_left", "bottom_right"]
        if markers[k] is not None)

    # If exactly 3 markers found, complete to 4 using the parallelogram rule.
    # For any rectangle viewed from any camera angle, TL + BR = TR + BL exactly
    # (projective geometry preserves the midpoint-bisecting property of parallelograms).
    # The inferred 4th corner IS the true corner — no heuristic search needed.
    if markers["markers_found"] == 3:
        names_list = ["top_left", "top_right", "bottom_left", "bottom_right"]
        found_3 = {n: markers[n] for n in names_list if markers[n] is not None}
        missing_name = next(n for n in names_list if markers[n] is None)

        tl = np.array(found_3.get("top_left",    [0, 0]), dtype=np.float32)
        tr = np.array(found_3.get("top_right",   [0, 0]), dtype=np.float32)
        bl = np.array(found_3.get("bottom_left", [0, 0]), dtype=np.float32)
        br = np.array(found_3.get("bottom_right",[0, 0]), dtype=np.float32)
        if missing_name == "top_left":      inferred = tr + bl - br
        elif missing_name == "top_right":   inferred = tl + br - bl
        elif missing_name == "bottom_left": inferred = tl + br - tr
        else:                               inferred = tr + bl - tl

        # Do NOT clip to image bounds — clipping distorts the geometry.
        # _valid_quad in align_captured_image handles out-of-bounds gracefully.
        markers[missing_name] = (int(round(inferred[0])), int(round(inferred[1])))
        markers["markers_found"] = 4
        markers["inferred_name"] = missing_name

    # Fixed-position estimation: when Hough detection failed on a confirmed pattern crop,
    # markers are guaranteed to be at FIDUCIAL_MARKER_OFFSET / PATTERN_SIZE from each corner.
    # Using these positions enables perspective alignment even on blurry / low-contrast prints.
    markers["estimated_names"] = []
    if pattern_found_hint and markers["markers_found"] < 4:
        off_frac = FIDUCIAL_MARKER_OFFSET / max(PATTERN_SIZE)
        estimates = {
            "top_left":     (int(round(w * off_frac)),         int(round(h * off_frac))),
            "top_right":    (int(round(w * (1 - off_frac))),   int(round(h * off_frac))),
            "bottom_left":  (int(round(w * off_frac)),         int(round(h * (1 - off_frac)))),
            "bottom_right": (int(round(w * (1 - off_frac))),   int(round(h * (1 - off_frac)))),
        }
        for cname, pos in estimates.items():
            if markers[cname] is None:
                markers[cname] = pos
                markers["estimated_names"].append(cname)
        markers["markers_found"] = sum(
            1 for k in ["top_left", "top_right", "bottom_left", "bottom_right"]
            if markers[k] is not None)

    # Store raw Hough positions keyed by image corner (tl/tr/bl/br).
    # These are independent of ring-ID labelling and used in align_captured_image
    # to try all 4 rotation assignments without relying on ring counting.
    markers["hough_positions"] = dict(corner_circles)

    return markers


# ==========================================================================
# VERIFICATION — STEP 2: Image Alignment
# ==========================================================================

def align_captured_image(captured, original_size, markers=None, original_gray_ref=None):
    """Returns (aligned_image, method_string).

    Alignment priority:
      1. perspective       — all 4 markers detected, valid quad
      2. perspective (inferred) — 3 markers + parallelogram-inferred 4th, valid quad
      3. crop_resize       — marker bounding box crop + resize (≥2 markers)
      4. resize            — full-frame resize (last resort)
    """
    target_w, target_h = original_size
    off = FIDUCIAL_MARKER_OFFSET
    ih, iw = captured.shape[:2]

    def _valid_quad(pts_cw):
        """pts_cw must be in clockwise order: TL, TR, BR, BL."""
        margin = max(iw, ih) * 0.5
        for p in pts_cw:
            if p[0] < -margin or p[0] > iw + margin: return False
            if p[1] < -margin or p[1] > ih + margin: return False
        hull = cv2.convexHull(pts_cw.reshape(-1, 1, 2).astype(np.int32))
        if len(hull) != 4: return False
        area = cv2.contourArea(hull)
        if area < iw * ih * 0.01: return False
        return True

    def _infer_fourth(found):
        """Given a dict with 3 of 4 corners, infer the missing one via parallelogram rule."""
        names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        missing = [n for n in names if n not in found]
        if len(missing) != 1:
            return found
        m = missing[0]
        tl = np.array(found.get("top_left",    [0, 0]), dtype=np.float32)
        tr = np.array(found.get("top_right",   [0, 0]), dtype=np.float32)
        bl = np.array(found.get("bottom_left", [0, 0]), dtype=np.float32)
        br = np.array(found.get("bottom_right",[0, 0]), dtype=np.float32)
        if m == "bottom_right": found["bottom_right"] = tuple((tr + bl - tl).astype(int))
        elif m == "bottom_left": found["bottom_left"] = tuple((tl + br - tr).astype(int))
        elif m == "top_right":   found["top_right"]   = tuple((tl + br - bl).astype(int))
        elif m == "top_left":    found["top_left"]    = tuple((tr + bl - br).astype(int))
        return found

    # --- Steps 1+2: Rotation-aware perspective alignment ---
    # Ring counting is unreliable for printed photos, so we do NOT trust ring-ID
    # labels on the detected marker positions.  Instead we:
    #   1. Collect the 4 reliable corner positions from raw Hough detections
    #      and fixed-position estimates (both are independent of ring IDs).
    #   2. Re-assign them positionally (image TL/TR/BL/BR by coordinate).
    #   3. Try all 4 rotational src→dst mappings, each at thumbnail resolution.
    #   4. Pick the mapping with the best NCC against the reference pattern.
    #   5. Do the final full-resolution warp with that mapping.
    #   6. Update the markers dict with the correct corner labels so the
    #      visualisation reflects the true orientation.
    if markers and original_gray_ref is not None:
        hough_pos = markers.get("hough_positions", {})   # {ic: (cx,cy)} raw Hough

        # Build P: 4 positions keyed by image corner name (tl/tr/bl/br).
        # Priority: Hough detection (correct image-corner proximity) first.
        # Fill any missing corners with fixed-position estimates computed
        # directly from the crop dimensions — these are ring-ID-independent
        # and accurate whenever the crop is the actual pattern area.
        # (The ring-ID-labelled estimated_names in the markers dict cannot be
        # used here because ring counting may have placed a wrong Hough circle
        # in the slot that *should* be None, preventing the estimate from being
        # stored in the expected image-corner slot.)
        # Only trust Hough positions where ring counting gave a reliable
        # result (ring_count != -1).  False Hough detections (grating texture,
        # paper noise) distort the perspective warp when used as corner points.
        ring_res = markers.get("ring_results", {})
        reliable_hough = set()
        for ic, res in ring_res.items():
            if res[0] != -1:  # ring counting succeeded
                reliable_hough.add(ic)

        P = {}
        for ic in ["tl", "tr", "bl", "br"]:
            if ic in hough_pos and ic in reliable_hough:
                P[ic] = hough_pos[ic]

        off_frac = FIDUCIAL_MARKER_OFFSET / max(PATTERN_SIZE)
        fixed_est = {
            "tl": (int(round(iw * off_frac)),         int(round(ih * off_frac))),
            "tr": (int(round(iw * (1 - off_frac))),   int(round(ih * off_frac))),
            "bl": (int(round(iw * off_frac)),         int(round(ih * (1 - off_frac)))),
            "br": (int(round(iw * (1 - off_frac))),   int(round(ih * (1 - off_frac)))),
        }
        for ic in ["tl", "tr", "bl", "br"]:
            if ic not in P:
                P[ic] = fixed_est[ic]   # always gives len(P) == 4

        if len(P) == 4:
            pos_tl, pos_tr = P["tl"], P["tr"]
            pos_bl, pos_br = P["bl"], P["br"]
            # src in clockwise order: TL, TR, BR, BL
            src = np.float32([pos_tl, pos_tr, pos_br, pos_bl])

            if _valid_quad(src):
                w, h = target_w, target_h

                # Four dst assignments (src order: tl, tr, br, bl in CW):
                #   0°:    tl→TL  tr→TR  br→BR  bl→BL
                #   90°CW: tl→BL  tr→TL  br→TR  bl→BR
                #   180°:  tl→BR  tr→BL  br→TL  bl→TR
                #   90°CCW:tl→TR  tr→BR  br→BL  bl→TL
                DST_LIST = [
                    [[off,off],[w-off,off],[w-off,h-off],[off,h-off]],          # 0°
                    [[off,h-off],[off,off],[w-off,off],[w-off,h-off]],          # 90°CW
                    [[w-off,h-off],[off,h-off],[off,off],[w-off,off]],          # 180°
                    [[w-off,off],[w-off,h-off],[off,h-off],[off,off]],          # 90°CCW
                ]
                # Correct pattern-corner label for each image corner at each rotation
                LABEL_REMAP = [
                    {"tl":"top_left","tr":"top_right","bl":"bottom_left","br":"bottom_right"},
                    {"tl":"bottom_left","tr":"top_left","bl":"bottom_right","br":"top_right"},
                    {"tl":"bottom_right","tr":"bottom_left","bl":"top_right","br":"top_left"},
                    {"tl":"top_right","tr":"bottom_right","bl":"top_left","br":"bottom_left"},
                ]
                ROT_NAMES = ["0°", "90°CW", "180°", "90°CCW"]

                cap_gray = (cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
                            if len(captured.shape) == 3 else captured)
                thumb = 128
                ref_t  = cv2.resize(original_gray_ref, (thumb, thumb)).astype(np.float32)
                ref_f  = ref_t - ref_t.mean()
                ref_n  = np.linalg.norm(ref_f) + 1e-9

                best_score, best_idx = -1.0, 0
                for idx, dst_pts in enumerate(DST_LIST):
                    dst = np.float32(dst_pts)
                    try:
                        M = cv2.getPerspectiveTransform(src, dst)
                        wg = cv2.warpPerspective(cap_gray, M, (w, h))
                    except Exception:
                        continue
                    war_t = cv2.resize(wg, (thumb, thumb)).astype(np.float32)
                    war_f = war_t - war_t.mean()
                    war_n = np.linalg.norm(war_f) + 1e-9
                    score = float(np.dot(ref_f.ravel(), war_f.ravel()) / (ref_n * war_n))
                    if score > best_score:
                        best_score, best_idx = score, idx

                # Ring-count verification: override the NCC rotation if ring counting
                # gives reliable identifications that contradict the NCC choice.
                # NCC at 128×128 fails for small / blurry prints because the grating
                # pattern looks nearly identical under 180° rotation — NCC scores for
                # all four hypotheses stay close to 0 and index 0 wins by default.
                # Ring counts (0/1/2/3 rings per marker) are a direct visual check
                # and uniquely identify each corner regardless of print size or blur.
                ring_res = markers.get("ring_results", {})
                if ring_res:
                    def _rc_agree(remap):
                        ok = total = 0
                        for ic, res in ring_res.items():
                            rc, contrast = res[0], res[1]
                            if contrast < 30 or rc not in RING_COUNT_TO_CORNER:
                                continue
                            total += 1
                            if remap.get(ic) == RING_COUNT_TO_CORNER[rc]:
                                ok += 1
                        return ok, total

                    ncc_ok, ncc_total = _rc_agree(LABEL_REMAP[best_idx])
                    if ncc_total >= 2 and ncc_ok < ncc_total:
                        # NCC rotation contradicts reliable ring counts — find the
                        # rotation with the most ring-count agreements instead.
                        for alt_idx, alt_remap in enumerate(LABEL_REMAP):
                            alt_ok, _ = _rc_agree(alt_remap)
                            if alt_ok > ncc_ok:
                                ncc_ok, best_idx = alt_ok, alt_idx

                dst = np.float32(DST_LIST[best_idx])
                M   = cv2.getPerspectiveTransform(src, dst)
                aligned = cv2.warpPerspective(captured, M, (w, h))

                # Relabel markers to reflect the true orientation
                remap = LABEL_REMAP[best_idx]
                for ic_name, correct_label in remap.items():
                    markers[correct_label] = P[ic_name]
                markers["markers_found"]      = 4
                markers["rotation_detected"]  = [0, 90, 180, 270][best_idx]

                return aligned, f"perspective ({ROT_NAMES[best_idx]})"

    # --- Step 3: Marker-guided crop (≥2 markers — much better than full-frame resize) ---
    if markers and markers.get("markers_found", 0) >= 2:
        names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        found_pts = [markers[n] for n in names if markers.get(n) is not None]

        # If exactly 3 found, include the inferred 4th for a tighter crop box
        if len(found_pts) == 3:
            f = {n: markers[n] for n in names if markers.get(n) is not None}
            f = _infer_fourth(f)
            inferred_name = [n for n in names if markers.get(n) is None][0]
            pt = np.array(f[inferred_name], dtype=np.float32)
            if 0 <= pt[0] <= iw and 0 <= pt[1] <= ih:
                found_pts.append(tuple(pt.astype(int)))

        xs = [p[0] for p in found_pts]
        ys = [p[1] for p in found_pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        pad = max(int(span * 0.05), 10)
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(iw, max(xs) + pad)
        y2 = min(ih, max(ys) + pad)

        # Force square crop to match the square CDP pattern — avoids aspect-ratio stretch
        if target_w == target_h:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = max(x2 - x1, y2 - y1) // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(iw, cx + half)
            y2 = min(ih, cy + half)

        if x2 - x1 > 20 and y2 - y1 > 20:
            crop = captured[y1:y2, x1:x2]
            return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA), "crop_resize"

    return cv2.resize(captured, (target_w, target_h), interpolation=cv2.INTER_AREA), "resize"


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


def test_prng_correlation(captured_gray, reference_gray, block_size=8):
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
    scale_configs = [
        (block_size,     0.60, 1.0),   # 8×8  — finest, genuine 1st-gen
        (block_size * 2, 0.65, 0.75),  # 16×16 — moderate blur
        (block_size * 4, 0.75, 0.50),  # 32×32 — heavy blur / small print
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
    """Measure gradient-domain structural match.

    Instead of comparing sharpness levels (which rewards alignment quality),
    compute NCC between gradient maps of captured vs reference.  A genuine
    1st-gen print has gradient patterns (edges, grating lines) that
    structurally match the original.  A 2nd-gen copy has gradient energy
    from printer halftone artifacts that do NOT match the original's edges.
    """
    rh, rw = reference_gray.shape[:2]
    if captured_gray.shape[:2] != (rh, rw):
        captured_gray = cv2.resize(captured_gray, (rw, rh),
                                   interpolation=cv2.INTER_AREA)

    def grad_mag(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    cap_g = grad_mag(captured_gray)
    ref_g = grad_mag(reference_gray)
    cf = cap_g.flatten() - cap_g.mean()
    rf = ref_g.flatten() - ref_g.mean()
    ncc = np.sum(cf * rf) / (np.sqrt(np.sum(cf**2) * np.sum(rf**2)) + 1e-10)
    return float(np.clip(ncc / 0.30, 0.0, 1.0))


# ==========================================================================
# VERIFICATION — FULL PIPELINE
# ==========================================================================

def verify_pattern(original_path, captured_path, uploads_dir="uploads", block_size=8):
    original_bgr = cv2.imread(original_path)
    captured_bgr = cv2.imread(captured_path)
    if original_bgr is None:
        return {"verdict": "ERROR", "error": f"Cannot load original: {original_path}"}
    if captured_bgr is None:
        return {"verdict": "ERROR", "error": f"Cannot load captured: {captured_path}"}

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    # Step 0: Crop
    cropped_bgr, bbox, pattern_found, pattern_quad = detect_and_crop_pattern(captured_bgr)
    if pattern_found:
        captured_bgr = cropped_bgr

    # Step 1: Markers
    markers = detect_fiducial_markers(captured_bgr, pattern_found_hint=pattern_found)

    # Step 2: Align
    target = (original_bgr.shape[1], original_bgr.shape[0])
    aligned_bgr, alignment_method = align_captured_image(captured_bgr, target, markers, original_gray_ref=original_gray)
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
    inferred_name = markers.get("inferred_name")
    estimated_names = markers.get("estimated_names", [])
    corner_labels = {"top_left": "TL", "top_right": "TR",
                     "bottom_left": "BL", "bottom_right": "BR"}

    # Draw pattern boundary — actual quad handles skew/perspective correctly.
    if pattern_found:
        if pattern_quad is not None:
            pts = pattern_quad.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(markers_vis, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
        else:
            cv2.rectangle(markers_vis, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)

    # Compute marker positions in full-image space.
    # markers["top_left"] etc. are in CROP space (set by align_captured_image after
    # NCC rotation detection — so the labels are always semantically correct).
    # We need to map from crop space → full image space by inverting the same warp
    # that _crop_from_contour applied.  Recompute out_size from pattern_quad side
    # lengths to reconstruct dst_pts, then build M_crop_to_full = inverse warp.
    # This is correct for any rotation, tilt, or skew — no corner-ordering ambiguity.
    if pattern_found and pattern_quad is not None:
        pq = pattern_quad.astype(np.float32)
        out_size = max(
            int(max(np.linalg.norm(pq[1] - pq[0]), np.linalg.norm(pq[2] - pq[3]))),
            int(max(np.linalg.norm(pq[3] - pq[0]), np.linalg.norm(pq[2] - pq[1]))),
            64)
        dst_pts_crop = np.float32([
            [0, 0], [out_size - 1, 0],
            [out_size - 1, out_size - 1], [0, out_size - 1]
        ])
        M_crop_to_full = cv2.getPerspectiveTransform(dst_pts_crop, pq)
        quad_diag = float(np.linalg.norm(pq[0] - pq[2]))
        marker_scale = quad_diag / (max(PATTERN_SIZE) * 1.414)
        use_quad_mapping = True
    else:
        use_quad_mapping = False
        marker_scale = max(bw, bh) / max(PATTERN_SIZE)

    r_vis = max(8, int(_RING_BG_R * marker_scale))
    font_scale = max(0.5, r_vis / 35)

    for name, label in corner_labels.items():
        pos = markers.get(name)
        if pos is None:
            continue
        if use_quad_mapping:
            # Map crop-space marker position → full-image space via inverse warp
            crop_pt = np.float32([[[float(pos[0]), float(pos[1])]]])
            full_pt = cv2.perspectiveTransform(crop_pt, M_crop_to_full)[0][0]
            gx, gy = int(round(full_pt[0])), int(round(full_pt[1]))
        else:
            gx, gy = pos[0] + crop_ox, pos[1] + crop_oy

        # Color: green=detected by Hough, orange=inferred, yellow=estimated
        if name in estimated_names:
            vis_color = (0, 200, 255)   # yellow
        elif name == inferred_name:
            vis_color = (0, 140, 255)   # orange
        else:
            vis_color = (0, 210, 0)     # green
        # White halo + coloured ring + centre dot
        cv2.circle(markers_vis, (gx, gy), r_vis + 3, (255, 255, 255), 3)
        cv2.circle(markers_vis, (gx, gy), r_vis, vis_color, 3)
        cv2.circle(markers_vis, (gx, gy), 5, vis_color, -1)
        tx, ty = gx + r_vis + 6, gy + 6
        cv2.putText(markers_vis, label, (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
        cv2.putText(markers_vis, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, vis_color, 2)

    markers_filename = f"markers_{ts}.jpg"
    cv2.imwrite(os.path.join(uploads_dir, markers_filename), markers_vis,
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Save aligned
    aligned_filename = f"aligned_{ts}.jpg"
    cv2.imwrite(os.path.join(uploads_dir, aligned_filename), aligned_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {
        "verdict": verdict,
        "confidence": float(final),
        "scores": {"moire": float(moire), "color": float(color),
                   "correlation": float(corr), "gradient": float(gradient)},
        "weights": {"moire": WEIGHT_MOIRE, "color": WEIGHT_COLOR,
                    "correlation": WEIGHT_CORRELATION, "gradient": WEIGHT_GRADIENT},
        "markers_found": markers["markers_found"],
        "alignment_method": alignment_method,
        "markers_filename": markers_filename,
        "aligned_filename": aligned_filename,
        "pattern_found": pattern_found,
    }
