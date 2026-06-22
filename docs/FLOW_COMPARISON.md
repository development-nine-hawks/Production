# Flow Comparison: Main Branch vs Label Branch

---

## Main Branch Flow (non-label QR only)

### `_find_pattern_contour`

1. Gaussian blur -> Otsu threshold (+20) -> binary
2. Morphological close (15x15 kernel) -- single pass
3. Find contours (RETR_EXTERNAL)
4. Score: `darkness + region_std`
5. Return best contour

### `detect_and_crop_pattern`

```
Step 1: _find_pattern_contour -> if found -> _crop_from_contour -> return
Step 2: _find_markers_in_image (Hough circles, max_r = 20*max_pattern/512)
        -> _find_square_group -> extrapolate corners -> perspective warp -> return
Step 3: Return raw image (found=False)
```

### `verify_pattern`

```
Step 0: detect_and_crop_pattern
Step 1: detect_fiducial_markers
Step 2: align_captured_image
Step 3: Tests (moire, color, correlation, gradient)
Step 4: Score + verdict
```

No template matching. No label awareness.

---

## Current Branch: Non-Label QR Path

### `_find_pattern_contour` -- CHANGED

1. Same threshold as main
2. **Two-pass morphology**: close-only first, then erode(5x5)+close(15x15) -- but erode pass **only runs if image > 1.15x pattern size** (skipped for standalone QR)
3. If erode pass finds contour, **dilate back with 9x9 kernel** to restore boundary
4. Score: `darkness + region_std + squareness_bonus` -- **squareness bonus added** (+20 for perfect square)

### `detect_and_crop_pattern` -- CHANGED

```
Step 1: _find_pattern_contour -> if near-square -> _crop_from_contour -> return
        (LABEL_ASPECT_RANGE check added: 0.75-1.33)

Step 2: If non-square contour found -> LABEL PATH (see below)

Step 3: Marker fallback -- ONLY runs if image > 1.15x pattern size
        (skipped for standalone QR to match main branch behavior)

Step 4: Return raw image (found=False)
```

### `verify_pattern` -- CHANGED

```
Step 0: detect_and_crop_pattern (same as main for standalone)
Step 1-4: Same as main
```

### What is different for standalone QR vs main branch

- **Squareness bonus in scoring** -- does not change which contour is selected for standalone (only one contour exists), but changes the score value
- **Marker fallback skipped** for pattern-sized images -- matches main branch behavior where markers failed silently anyway
- **Everything else identical**: same `_crop_from_contour`, same `detect_fiducial_markers`, same `align_captured_image`, same tests

---

## Current Branch: Label QR Path

When `detect_and_crop_pattern` finds a **non-square contour** (the label card):

```
Step 1: RETR_TREE on full image with same thresholding as main branch
        (Otsu+20, close 15x15)
Step 2: Find most-square NESTED child contour (the 3px alignment border)
        Filter: 0.85-1.18 aspect, has parent, >0.3% area
Step 3: _crop_from_contour on that contour -> return
Step 4 (fallback): _extract_pattern_from_label -- RETR_TREE on label ROI,
        same nested-square search
Step 5 (last resort): center square crop of label ROI
```

After crop, the flow is **identical** to standalone:

```
detect_fiducial_markers -> align_captured_image -> tests -> score
```

### What is different for label QR vs standalone QR

- **Extra detection step**: RETR_TREE nested contour search to find QR within label
- **`_extract_pattern_from_label` fallback**: only runs if RETR_TREE fails
- **Template matching**: commented out -- was bypassing the verification pipeline entirely, giving artificial 100% scores on digital copies (including potential counterfeits)
- **After crop**: exact same pipeline as standalone

---

## Workarounds in current branch

1. **Erode pass size guard** (`min(h,w) > max_pat * 1.15`): Prevents erode+close from breaking standalone patterns into fragments. Workaround for the two-pass morphology causing false contours on small images.

2. **Marker fallback size guard** (same threshold): Prevents Hough false positives from cropping standalone patterns incorrectly. Workaround for the marker detection finding texture circles.

3. **Dilate-back after erode** (9x9 kernel): Restores contour boundary after erosion. Workaround for erode shrinking the detected region by 2-3px.

---

## Fallbacks in current branch

1. **RETR_TREE nested search** -- primary label detection
2. **`_extract_pattern_from_label`** -- secondary label detection (RETR_TREE on label ROI)
3. **Center square crop** -- last resort if all else fails
4. **Marker-based detection** -- only for large images where contour fails entirely
5. **Raw image return** (`found=False`) -- when nothing works

---

## What is skipped vs main branch

1. **Marker fallback on standalone images**: Main branch runs it (usually fails silently or finds false squares). We skip it explicitly to avoid false crops.
2. Nothing else is skipped -- all main branch logic is preserved.
