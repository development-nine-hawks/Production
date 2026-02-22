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
FIDUCIAL_MARKER_SIZE = 48
FIDUCIAL_MARKER_OFFSET = FIDUCIAL_MARKER_SIZE // 2  # 24px
RING_COUNT_TO_CORNER = {1: "top_left", 2: "top_right", 3: "bottom_left", 0: "bottom_right"}
BLOCK_SIZE = 8

# Verification weights
WEIGHT_MOIRE = 0.40
WEIGHT_COLOR = 0.30
WEIGHT_CORRELATION = 0.20
WEIGHT_GRADIENT = 0.10

# Thresholds
THRESHOLD_AUTHENTIC = 0.70
THRESHOLD_SUSPICIOUS = 0.50


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


def add_fiducial_markers(pattern, marker_size=48, marker_color=0):
    h, w = pattern.shape[:2]
    is_rgb = len(pattern.shape) == 3
    color = (marker_color,) * 3 if is_rgb else marker_color
    bg = (255, 255, 255) if is_rgb else 255
    offset = marker_size // 2
    ring_thickness = 3
    centers = [
        (offset, offset),
        (w - offset, offset),
        (offset, h - offset),
        (w - offset, h - offset),
    ]
    for cx, cy in centers:
        cv2.circle(pattern, (cx, cy), offset - 1, bg, -1)
    cv2.circle(pattern, centers[0], 20, color, ring_thickness)       # TL: 1 ring
    cv2.circle(pattern, centers[1], 20, color, ring_thickness)       # TR: 2 rings
    cv2.circle(pattern, centers[1], 10, color, ring_thickness)
    cv2.circle(pattern, centers[2], 20, color, ring_thickness)       # BL: 3 rings
    cv2.circle(pattern, centers[2], 13, color, ring_thickness)
    cv2.circle(pattern, centers[2], 6, color, ring_thickness)
    cv2.circle(pattern, centers[3], 20, color, -1)                   # BR: solid
    return pattern


def generate_pattern(output_dir, seed=None, serial_number="SN-0001", pattern_size=512):
    if seed is None:
        seed = int(np.random.randint(0, 2**31))
    w = h = pattern_size

    grating_rng = np.random.RandomState(seed=seed + 2000)
    base_freq = 8 + grating_rng.random() * 6
    mod_freq = 1.5 + grating_rng.random() * 2.5
    mod_depth = 0.2 + grating_rng.random() * 0.3

    grating = generate_frequency_modulated_grating(w, h, base_freq, mod_freq, mod_depth)
    prng = generate_prng_macro_pattern(w, h, seed, block_size=8)
    combined = cv2.addWeighted(grating, 0.5, prng, 0.5, 0)
    pattern_rgb = add_rgb_perturbations(combined, seed=seed, intensity=25, block_size=8)
    final = add_fiducial_markers(pattern_rgb, marker_size=48, marker_color=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phone_cdp_{timestamp}_{seed}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    return {"seed": seed, "filename": filename, "filepath": filepath,
            "serial_number": serial_number, "pattern_size": pattern_size}


# ==========================================================================
# VERIFICATION — STEP 0: Pattern Detection
# ==========================================================================

def _find_pattern_contour(gray, min_area_pct=0.05):
    h, w = gray.shape
    image_mean = float(gray.mean())
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
        if not (0.7 < aspect < 1.4):
            continue
        if bw > w * 0.9 or bh > h * 0.9:
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
    rect = cv2.minAreaRect(contour)
    box = np.float32(cv2.boxPoints(rect))
    _, (rect_w, rect_h), angle = rect
    if rect_w < rect_h:
        rect_w, rect_h = rect_h, rect_w
    effective_angle = abs(angle) % 90
    if effective_angle > 45:
        effective_angle = 90 - effective_angle
    bx, by, bw, bh = cv2.boundingRect(contour)
    if effective_angle < 5:
        pad_x, pad_y = int(bw * 0.02), int(bh * 0.02)
        bx = max(0, bx - pad_x)
        by = max(0, by - pad_y)
        bw = min(w - bx, bw + 2 * pad_x)
        bh = min(h - by, bh + 2 * pad_y)
        return image[by:by + bh, bx:bx + bw], (bx, by, bw, bh)
    sorted_pts = sorted(box, key=lambda p: p[1])
    top_pts = sorted(sorted_pts[:2], key=lambda p: p[0])
    bot_pts = sorted(sorted_pts[2:], key=lambda p: p[0])
    src_pts = np.float32([top_pts[0], top_pts[1], bot_pts[1], bot_pts[0]])
    out_size = max(int(rect_w), int(rect_h))
    pad = int(out_size * 0.02)
    dst_pts = np.float32([
        [pad, pad], [out_size - 1 - pad, pad],
        [out_size - 1 - pad, out_size - 1 - pad], [pad, out_size - 1 - pad]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (out_size, out_size),
                                  borderValue=(255, 255, 255))
    return cropped, (bx, by, bw, bh)


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

    contour = _find_pattern_contour(gray, min_area_pct=0.03)
    if contour is not None:
        cropped, bbox = _crop_from_contour(image, contour)
        return cropped, bbox, True

    circles = _find_markers_in_image(gray)
    if circles is not None:
        group = _find_square_group(circles)
        if group is not None:
            xs, ys = group[:, 0], group[:, 1]
            side = max(xs.max() - xs.min(), ys.max() - ys.min())
            pad = int(side * 0.10)
            bx = max(0, int(xs.min()) - pad)
            by = max(0, int(ys.min()) - pad)
            bw = min(w - bx, int(xs.max() - xs.min()) + 2 * pad)
            bh = min(h - by, int(ys.max() - ys.min()) + 2 * pad)
            return image[by:by + bh, bx:bx + bw], (bx, by, bw, bh), True

    return image, (0, 0, w, h), False


# ==========================================================================
# VERIFICATION — STEP 1: Fiducial Marker Detection
# ==========================================================================

def count_rings_radial_profile(gray_region, cx, cy, max_radius=None, min_contrast=30):
    h, w = gray_region.shape
    if max_radius is None:
        max_radius = min(cx, cy, w - 1 - cx, h - 1 - cy)
    if max_radius < 5:
        return (-1, 0)
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    profile = []
    for r in range(max_radius + 1):
        if r == 0:
            samples = [float(gray_region[cy, cx])]
        else:
            samples = []
            for a in angles:
                px, py = int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))
                if 0 <= px < w and 0 <= py < h:
                    samples.append(float(gray_region[py, px]))
        profile.append(np.mean(samples) if samples else 255.0)
    profile = np.array(profile)
    val_min, val_max = np.min(profile), np.max(profile)
    contrast = val_max - val_min
    if contrast < min_contrast:
        return (-1, contrast)
    thresh = val_min + 0.35 * (val_max - val_min)
    is_dark = profile < thresh
    center_check = min(max_radius // 3, 6)
    if center_check > 0 and np.all(is_dark[:center_check]):
        return (0, contrast)
    ring_count = 0
    for i in range(1, len(is_dark)):
        if is_dark[i] and not is_dark[i - 1]:
            ring_count += 1
    return (ring_count, contrast)


def detect_fiducial_markers(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    h, w = gray.shape

    scale = max(w, h) / max(PATTERN_SIZE)
    ring_max_radius = max(25, int(25 * scale))
    expected_r = int(20 * scale)
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

    corner_radius = max(w, h) * 0.25
    image_corners = {"tl": (0, 0), "tr": (w, 0), "bl": (0, h), "br": (w, h)}

    corner_circles = {}
    if circles is not None:
        circle_list = np.round(circles[0, :]).astype(int)
        for ic_name, (ic_x, ic_y) in image_corners.items():
            best_dist, best_circle = float("inf"), None
            for cx, cy, _r in circle_list:
                dist = np.sqrt((cx - ic_x) ** 2 + (cy - ic_y) ** 2)
                if dist < corner_radius and dist < best_dist:
                    best_dist = dist
                    best_circle = (int(cx), int(cy))
            if best_circle is not None:
                corner_circles[ic_name] = best_circle

    markers = {name: None for name in ["top_left", "top_right", "bottom_left", "bottom_right"]}
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

    positional_map = {"tl": "top_left", "tr": "top_right",
                      "bl": "bottom_left", "br": "bottom_right"}
    for ic_name, (cx, cy) in corner_circles.items():
        pattern_corner = positional_map[ic_name]
        if pattern_corner not in assigned and ic_name in ring_results:
            markers[pattern_corner] = (cx, cy)
            assigned.add(pattern_corner)

    markers["markers_found"] = sum(
        1 for k in ["top_left", "top_right", "bottom_left", "bottom_right"]
        if markers[k] is not None)
    return markers


# ==========================================================================
# VERIFICATION — STEP 2: Image Alignment
# ==========================================================================

def align_captured_image(captured, original_size, markers=None):
    target_w, target_h = original_size
    off = FIDUCIAL_MARKER_OFFSET

    if markers and markers.get("markers_found", 0) >= 4:
        src = np.float32([markers["top_left"], markers["top_right"],
                          markers["bottom_right"], markers["bottom_left"]])
        dst = np.float32([[off, off], [target_w - off, off],
                          [target_w - off, target_h - off], [off, target_h - off]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(captured, M, (target_w, target_h))

    if markers and markers.get("markers_found", 0) >= 2:
        available, expected = [], []
        mapping = {"top_left": [off, off], "top_right": [target_w - off, off],
                   "bottom_left": [off, target_h - off],
                   "bottom_right": [target_w - off, target_h - off]}
        for name, dst in mapping.items():
            if markers.get(name) is not None:
                available.append(markers[name])
                expected.append(dst)
        if len(available) >= 3:
            M = cv2.getAffineTransform(np.float32(available[:3]), np.float32(expected[:3]))
            return cv2.warpAffine(captured, M, (target_w, target_h))

    return cv2.resize(captured, (target_w, target_h), interpolation=cv2.INTER_AREA)


# ==========================================================================
# VERIFICATION — STEP 3: Tests
# ==========================================================================

def test_moire_detection(captured_gray, reference_gray=None):
    def spectrum(img):
        f = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
        mag = np.log1p(np.abs(f))
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        mag[cy - 3:cy + 3, cx - 3:cx + 3] = 0
        return mag

    cap_spec = spectrum(captured_gray)
    if reference_gray is not None:
        ref_spec = spectrum(reference_gray)
        cap_n = cap_spec / (np.max(cap_spec) + 1e-10)
        ref_n = ref_spec / (np.max(ref_spec) + 1e-10)
        diff = np.mean(np.abs(cap_n - ref_n))
        return float(np.clip(1.0 - diff / 0.20, 0.0, 1.0))
    h, w = cap_spec.shape
    cy, cx = h // 2, w // 2
    yc, xc = np.ogrid[:h, :w]
    r = np.sqrt((yc - cy) ** 2 + (xc - cx) ** 2)
    mid = (r > h * 0.1) & (r < h * 0.35)
    ratio = np.sum(cap_spec[mid]) / (np.sum(cap_spec) + 1e-10)
    return float(np.clip(1.0 - (ratio - 0.3) / 0.25, 0.0, 1.0))


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
    h, w = captured_gray.shape[:2]
    rh, rw = reference_gray.shape[:2]
    if (h, w) != (rh, rw):
        captured_gray = cv2.resize(captured_gray, (rw, rh), interpolation=cv2.INTER_AREA)
        h, w = rh, rw
    best = -1.0
    for bs in [block_size, block_size * 2, block_size * 4, block_size * 8]:
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
        cf, rf = cap_b.flatten() - cap_b.mean(), ref_b.flatten() - ref_b.mean()
        corr = np.sum(cf * rf) / (np.sqrt(np.sum(cf ** 2) * np.sum(rf ** 2)) + 1e-10)
        if corr > best:
            best = corr
    return float(np.clip(best / 0.4, 0.0, 1.0)), float(best)


def test_gradient_energy(captured_gray, reference_gray):
    def energy(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(np.sqrt(gx ** 2 + gy ** 2))
    ratio = energy(captured_gray) / (energy(reference_gray) + 1e-10)
    return float(np.clip((ratio - 0.15) / 0.45, 0.0, 1.0))


# ==========================================================================
# VERIFICATION — FULL PIPELINE
# ==========================================================================

def verify_pattern(original_path, captured_path, uploads_dir="uploads"):
    original_bgr = cv2.imread(original_path)
    captured_bgr = cv2.imread(captured_path)
    if original_bgr is None:
        return {"verdict": "ERROR", "error": f"Cannot load original: {original_path}"}
    if captured_bgr is None:
        return {"verdict": "ERROR", "error": f"Cannot load captured: {captured_path}"}

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    # Step 0: Crop
    cropped_bgr, bbox, pattern_found = detect_and_crop_pattern(captured_bgr)
    if pattern_found:
        captured_bgr = cropped_bgr

    # Step 1: Markers
    markers = detect_fiducial_markers(captured_bgr)

    # Step 2: Align
    target = (original_bgr.shape[1], original_bgr.shape[0])
    aligned_bgr = align_captured_image(captured_bgr, target, markers)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    aligned_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)

    # Step 3: Tests
    moire = test_moire_detection(aligned_gray, original_gray)
    color = test_color_analysis(aligned_rgb, original_rgb)
    corr, raw_corr = test_prng_correlation(aligned_gray, original_gray, BLOCK_SIZE)
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

    if markers["markers_found"] >= 4:
        alignment_method = "perspective"
    elif markers["markers_found"] >= 2:
        alignment_method = "affine"
    else:
        alignment_method = "resize"

    # Save aligned
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    aligned_filename = f"aligned_{ts}.png"
    cv2.imwrite(os.path.join(uploads_dir, aligned_filename), aligned_bgr)

    return {
        "verdict": verdict,
        "confidence": float(final),
        "scores": {"moire": float(moire), "color": float(color),
                   "correlation": float(corr), "gradient": float(gradient)},
        "weights": {"moire": WEIGHT_MOIRE, "color": WEIGHT_COLOR,
                    "correlation": WEIGHT_CORRELATION, "gradient": WEIGHT_GRADIENT},
        "markers_found": markers["markers_found"],
        "alignment_method": alignment_method,
        "aligned_filename": aligned_filename,
        "pattern_found": pattern_found,
    }
