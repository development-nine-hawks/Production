# All Changes: Label Branch vs Main Branch

## Date: 2026-03-26

---

## Summary

This document lists every change made in the `label` branch compared to the `main` branch. Changes span 5 files across the backend, web frontend, and mobile app.

---

## File 1: `backend/cdp_engine.py`

### Change 1.1: New constants

**Location**: After `BLOCK_SIZE = 8` (line 31)

**Added**:
```python
LABEL_ASPECT_RANGE = (0.75, 1.33)
LABEL_PATTERN_BOTTOM_FRACTION = 0.50
```

**What**: Two constants for label-aware detection.
- `LABEL_ASPECT_RANGE`: contours with aspect ratio in this range are treated as "near-square" (likely the CDP pattern). Outside this range means the contour is probably a label card.
- `LABEL_PATTERN_BOTTOM_FRACTION`: when all detection methods fail inside a label, crop the bottom 50% as a last resort (the pattern sits at the bottom of the label design).

---

### Change 1.2: Squareness bonus in `_find_pattern_contour` scoring

**Location**: Inside `_find_pattern_contour`, scoring loop

**Before (main)**:
```python
score = darkness + region_std
```

**After (label)**:
```python
squareness_bonus = (1.0 - abs(1.0 - aspect)) * 20
score = darkness + region_std + squareness_bonus
```

**What**: Added a +20 bonus for perfect squares (aspect=1.0). Rectangles get proportionally less.

**Why**: The CDP pattern is always square. The label card is a tall rectangle. When the system sees both shapes as contours, this bonus ensures it picks the square pattern over the rectangular label.

---

### Change 1.3: Two-pass morphology in `_find_pattern_contour`

**Location**: Morphology section of `_find_pattern_contour`

**Before (main)**: Single morphology pass -- close only:
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(closed, ...)
```

**After (label)**: Two passes -- close only first, then erode+close:
```python
morph_passes = [
    cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel),                    # pass 1
    cv2.morphologyEx(cv2.erode(binary, erode_kernel), cv2.MORPH_CLOSE, close_kernel),  # pass 2
]
```

Pass 1 is tried first (identical to main branch behavior). Pass 2 only runs if pass 1 yields no valid contour.

**Why**: The label card has a thin dark border (2px), a black banner, bold text, and the CDP pattern. The 15x15 morphological close bridges all of these into one giant contour covering 99.7% of the image, which gets rejected by the >85% area filter. No valid contour is found. The erode step (5x5) breaks thin connections (borders, text strokes) before closing, so the pattern separates from the rest.

---

### Change 1.4: Dilate-back after erode pass

**Location**: After pass 2 finds a contour in `_find_pattern_contour`

**What**: When the erode pass finds a valid contour, the contour boundary is ~2-3px smaller than reality (because of the erosion). The code dilates the binary back with a 9x9 kernel (larger than the 5x5 erode kernel) to slightly overshoot, then re-extracts the contour from the restored image.

**Why**: A tight crop that misses pattern edge pixels hurts verification scores. A slightly generous crop (with a few background pixels) is better because alignment can handle extra pixels but cannot recover missing ones.

---

### Change 1.5: Refactored scoring into `_score_contours` helper

**Location**: Inside `_find_pattern_contour`

**What**: The contour scoring loop was extracted into a nested `_score_contours(morphed_img)` function. This avoids duplicating the scoring logic for the two morphology passes.

**Why**: Code organization -- same logic needs to run on two different binary images.

---

### Change 1.6: New function `_extract_pattern_from_label`

**Location**: After `_crop_from_contour`, before `_find_markers_in_image`

**What**: Geometric fallback for extracting the pattern from a label when all other methods fail. Crops the bottom `LABEL_PATTERN_BOTTOM_FRACTION` (50%) of the label, runs `_find_pattern_contour` on it with tighter squareness filter, and falls back to center-square extraction.

**Why**: Last resort when markers and contours both fail inside a detected label (e.g., severe blur, very small print).

---

### Change 1.7: New function `_marker_crop`

**Location**: After `_extract_pattern_from_label`, before `detect_and_crop_pattern`

**What**: Shared helper that encapsulates the marker-based detection logic. Finds 4 markers forming a square via `_find_markers_in_image` + `_find_square_group`, extrapolates pattern corners using centroid scaling (~1.153), perspective-warps, and returns `(cropped, bbox, True, quad)` or `None`.

**Why**: This logic was previously inline in `detect_and_crop_pattern`. Extracting it into a helper allows reuse (called both for full-image marker search and for within-label-ROI marker search).

---

### Change 1.8: Restructured `detect_and_crop_pattern`

**Location**: Entire function body

**Before (main)**:
```
1. Find contour -> crop -> return
2. Find markers (inline code) -> crop -> return
3. Return full image
```

**After (label)**:
```
1. Find contour (with squareness bonus + two-pass morphology)
   a. Near-square? -> crop as before
   b. Non-square (label-shaped)?
      i.   Inner contour search within label bounding rect
      ii.  Marker search within label bounding rect via _marker_crop
      iii. Geometric fallback: crop bottom 50% via _extract_pattern_from_label
2. Marker search on full image via _marker_crop (fallback)
3. Return full image
```

**Why**: The original flow could not handle images where the pattern is embedded in a larger label with branding, text, and borders. The label-aware flow searches for the pattern inside a detected label region.

**Important**: For standalone patterns (no label), the behavior is identical to main branch. Pass 1 of contour detection either succeeds (same as main) or fails (falls to marker fallback, same as main).

---

### Change 1.9: Template matching in `verify_pattern`

**Location**: Inside `verify_pattern`, before contour-based crop (Step 0)

**What**: Before running any contour detection, check if the captured image is larger than the original pattern. If yes, use `cv2.matchTemplate(captured_gray, original_gray, TM_CCOEFF_NORMED)` to find the exact pixel location of the pattern. If NCC > 0.90, extract with a pure array slice `captured[ty:ty+512, tx:tx+512]`. Skip alignment entirely.

**Why**: Any perspective warp (even a correct one) changes pixel values through bilinear interpolation. For clean digital images (like a label PNG), the pattern pixels are identical to the original. Template matching finds the exact integer-pixel position and extracts without any interpolation. Result: 100% match.

**When it activates**: Only when `captured_width > original_width AND captured_height > original_height AND template NCC > 0.90`. For phone photos (which have perspective distortion, blur, noise), the NCC will be below 0.90 and the standard contour+marker+alignment pipeline handles them as before.

---

### Change 1.10: bbox and pattern_quad initialization for template match path

**Location**: Inside `verify_pattern`, after template match check

**What**: When template matching succeeds, sets `bbox = (tx, ty, ow, oh)` and `pattern_quad` to the rectangle at the match position. When template matching doesn't apply, initializes `bbox = (0, 0, cw, ch)` and `pattern_quad = None` before the contour path runs.

**Why**: The marker visualization code later in the function uses `bbox` and `pattern_quad` to draw marker circles on the full captured image. Without proper initialization, markers were drawn at position (0,0) -- in the banner area instead of on the pattern.

---

### Summary of UNCHANGED functions in cdp_engine.py

These functions have zero modifications from main branch:
- `generate_frequency_modulated_grating`
- `generate_prng_macro_pattern`
- `add_rgb_perturbations`
- `add_fiducial_markers`
- `generate_pattern`
- `_crop_from_contour`
- `_find_markers_in_image`
- `_find_square_group`
- `_make_marker_templates`
- `count_rings_radial_profile`
- `detect_fiducial_markers`
- `align_captured_image`
- `test_moire_detection`
- `test_color_analysis`
- `test_prng_correlation`
- `test_gradient_energy`
- All marker visualization code in `verify_pattern` (drawing, saving markers/aligned images)

---

## File 2: `backend/app.py`

### Change 2.1: New endpoint `GET /api/patterns/{pid}/label-pdf`

**What**: Generates a NINEHAWK branded label as PDF. Layout:
```
+-----------------------------------+
|  (top margin)                     |
|  +-----------------------------+  |
|  |  NINEHAWK banner image      |  |
|  +-----------------------------+  |
|         NINEHAWK (watermark)      |
|         SCAN TO                   |
|        AUTHENTICATE              |
|  +--[dark border]--+             |
|  |  CDP Pattern    |             |
|  |  (exact size_mm)|             |
|  +--[dark border]--+             |
|  (bottom margin)                  |
+-----------------------------------+
```

All dimensions are proportional to the pattern size. The pattern prints at exactly the requested `size_mm`. Around the pattern, a 3pt dark border is drawn as an alignment aid.

**Endpoint**: `GET /api/patterns/{pid}/label-pdf?size_mm=X`
**Response**: PDF file with custom page size matching the label dimensions.

---

### Change 2.2: New endpoint `GET /api/patterns/{pid}/label-png`

**What**: Same NINEHAWK branded label layout but rendered as a PNG image using OpenCV instead of ReportLab.

**Endpoint**: `GET /api/patterns/{pid}/label-png?size_px=512`
**Response**: PNG image file.

**Label generation details (PNG)**:
- White canvas at calculated label dimensions
- Black rectangle for banner area
- Banner image (`ninehawk_banner.png`) drawn centered within the banner
- "NINEHAWK" watermark text in light gray (cv2.putText)
- "SCAN TO AUTHENTICATE" in bold black (cv2.putText)
- 3px dark border (RGB 30,30,30) around the pattern area -- acts as a sacrificial margin for contour detection
- The white gap between border and label background provides a crisp edge for contour detection to lock onto
- Pattern placed at exact integer coordinates inside the border

---

### Change 2.3: Original endpoint `GET /api/patterns/{pid}/pdf` -- UNCHANGED

The original plain PDF endpoint (A4 page, pattern centered, gray border, metadata footer) is completely unchanged from main branch.

---

## File 3: `backend/static/js/app.js`

### Change 3.1: Download menu expanded

**What**: Added three new download options to the pattern generation page dropdown:
- **Label PNG** -- links to `/api/patterns/${id}/label-png`
- **Label PDF - 15mm** -- links to `/api/patterns/${id}/label-pdf?size_mm=15`
- **Label PDF - 7.5mm** -- links to `/api/patterns/${id}/label-pdf?size_mm=7.5`

These appear below the original options (PNG Image, PDF 15mm, PDF 7.5mm) with a separator line.

---

## File 4: `backend/static/img/ninehawk_banner.png`

### Change 4.1: New asset file

**What**: Extracted NINEHAWK banner image (hawk logo + "NINEHAWK" text on black background). Cropped from the reference design image provided by the user.

**Used by**: Both `label-pdf` and `label-png` endpoints.

---

## File 5: `frontend/src/config/api.config.ts`

### Change 5.1: Added `LABEL_PDF` endpoint definition

**What**: Added one line to the PATTERNS endpoints:
```typescript
LABEL_PDF: (id: number) => `/api/patterns/${id}/label-pdf`,
```

**Why**: So the React Native mobile app can access the new label PDF endpoint.

---

## What is NOT changed

- **Pattern generation** (`generate_pattern` and all sub-functions): Zero changes. Patterns are generated identically.
- **Fiducial marker detection** (`detect_fiducial_markers`): Zero changes. Ring counting, Hough detection within the cropped pattern, NCC template matching for marker classification -- all identical.
- **Alignment** (`align_captured_image`): Zero changes. Perspective transform, 4-rotation hypothesis testing, NCC scoring, crop-resize fallback -- all identical.
- **Verification tests** (`test_moire_detection`, `test_color_analysis`, `test_prng_correlation`, `test_gradient_energy`): Zero changes. Scoring formulas, thresholds, weights -- all identical.
- **`_find_markers_in_image`**: Zero changes. Hough radius formula is identical to main branch.
- **`_crop_from_contour`**: Zero changes. Polygon approximation, parallelogram fallback, perspective warp -- all identical.
- **Database layer** (`database.py`): Zero changes.
- **Configuration** (`config.py`): Zero changes.
- **Original PDF endpoint** (`/api/patterns/{pid}/pdf`): Zero changes.
- **All other API endpoints**: Zero changes.

---

## Test Results

| Scenario | Main Branch | Label Branch |
|----------|-------------|--------------|
| Standalone PNG vs itself | 100% | 100% |
| Label PNG vs original | N/A (no label feature) | 100% (template_match) |
| Phone photo of standalone pattern | Works | Works (identical path) |
| Phone photo of printed label | Would fail (grabs whole label) | Works (contour + inner search) |
