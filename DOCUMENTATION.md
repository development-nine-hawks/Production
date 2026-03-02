# PhoneCDP — Copy Detection Pattern System

## Complete Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is a Copy Detection Pattern?](#2-what-is-a-copy-detection-pattern)
3. [System Architecture](#3-system-architecture)
4. [Directory Structure](#4-directory-structure)
5. [Pattern Generation — How It Works](#5-pattern-generation--how-it-works)
6. [Verification Pipeline — How It Works](#6-verification-pipeline--how-it-works)
7. [Database Schema](#7-database-schema)
8. [API Reference](#8-api-reference)
9. [Frontend Application](#9-frontend-application)
10. [Deployment](#10-deployment)
11. [Configuration](#11-configuration)
12. [File-by-File Breakdown](#12-file-by-file-breakdown)

---

## 1. Project Overview

PhoneCDP is a web-based system for generating and verifying **Copy Detection Patterns** (CDPs). A CDP is a specially designed image that, when printed on a product, can be photographed with a phone camera and compared against the original digital version to determine if the product is authentic or counterfeit.

**The core idea**: printing and scanning/photographing degrades an image in predictable ways. An original printed pattern will degrade once (print). A counterfeit copy degrades twice (print → scan/photo → reprint). PhoneCDP measures these degradation signatures across multiple dimensions to classify a captured photo as AUTHENTIC, SUSPICIOUS, or COUNTERFEIT.

### Key Capabilities

- **Generate** unique, seed-reproducible CDP images with embedded fiducial markers
- **Verify** phone-captured photos against the original digital pattern
- **Track** all verification results with full scoring breakdowns
- **Export** results to CSV for offline analysis
- **Batch verify** multiple photos in a single operation
- **Deploy** as a Docker container to Render (or any cloud platform)

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI 0.115 |
| Image Processing | OpenCV (headless), NumPy |
| Database | SQLite via SQLAlchemy 2.0 |
| Frontend | Vanilla JavaScript SPA, Tailwind CSS (CDN) |
| Deployment | Docker, Gunicorn + Uvicorn workers |
| Hosting | Render.com (free tier compatible) |

---

## 2. What is a Copy Detection Pattern?

A Copy Detection Pattern is a composite image built from three independent signal layers, each designed to be sensitive to print-scan degradation in different ways:

### Layer 1: Frequency-Modulated Sinusoidal Grating

A sinusoidal wave pattern whose frequency varies spatially across the image. The mathematical formulation:

```
freq_modulation(x) = base_freq * (1 + mod_depth * sin(2π * mod_freq * x / width))
phase(x) = cumulative_sum(freq_modulation) * 2π / width
grating(x, y) = sin(phase(x))
```

- **base_freq**: 8–14 cycles (seed-derived)
- **mod_freq**: 1.5–4.0 (seed-derived)
- **mod_depth**: 0.2–0.5 (seed-derived)

This layer is sensitive to **moiré interference** — when a printed grating is photographed, the interaction between the physical print halftone grid and the camera's pixel grid creates characteristic moiré fringes. Counterfeits produce different moiré patterns because they've been through two print cycles.

### Layer 2: PRNG Macro-Block Pattern

A grid of 8×8 pixel blocks, each assigned a random intensity value (0–255) from a seeded pseudo-random number generator (`np.random.RandomState(seed)`).

```
blocks = RandomState(seed).randint(0, 256, size=(height/8, width/8))
pattern = repeat_each_block_to_8x8_pixels(blocks)
```

This creates a unique, deterministic "fingerprint" for each seed. The block structure survives printing and moderate perspective distortion, enabling **cross-correlation** between the captured and original images to verify they share the same PRNG sequence.

### Layer 3: RGB Color Perturbations

Each 8×8 block receives independent per-channel color shifts (R, G, B) from a second PRNG seeded with `seed + 1000`:

```
for each block:
    for each channel (R, G, B):
        shift = random(-25, +25)
        block[channel] += shift
```

This embeds subtle color differences that are invisible to casual observation but measurable by the verification system. Printing and rephotographing attenuates these channel-specific perturbations differently from a first-generation print.

### Compositing

The three layers are combined:

```
combined = 0.5 * grating + 0.5 * prng_blocks  (grayscale blend)
final = add_rgb_perturbations(combined)         (colorize)
final = add_fiducial_markers(final)             (add corner markers)
```

### Fiducial Markers

Four concentric-ring markers are placed in the corners to enable automatic detection and perspective correction:

| Corner | Marker | Ring Count |
|--------|--------|-----------|
| Top-Left | 1 ring (radius 20, thickness 3) | 1 |
| Top-Right | 2 rings (radii 20 and 10) | 2 |
| Bottom-Left | 3 rings (radii 20, 13, and 6) | 3 |
| Bottom-Right | Solid filled circle (radius 20) | 0 (solid) |

Each marker sits on a white background circle (radius 23px) to ensure contrast against any pattern content. The different ring counts allow the system to identify which corner is which, even if the image is rotated or has perspective distortion.

The markers are centered at `(offset, offset)` from each corner where `offset = marker_size / 2 = 24px` for a 48px marker size.

---

## 3. System Architecture

```
┌─────────────┐     HTTP      ┌─────────────────────────────────────┐
│   Browser    │ ◄──────────► │           FastAPI (app.py)           │
│  (SPA JS)   │               │                                     │
└─────────────┘               │  ┌─────────┐  ┌──────────────────┐  │
                              │  │ Static  │  │   API Endpoints   │  │
                              │  │ Files   │  │  /api/patterns/*  │  │
                              │  │ /static │  │  /api/verify/*    │  │
                              │  └─────────┘  │  /api/results/*   │  │
                              │               └────────┬─────────┘  │
                              │                        │            │
                              │               ┌────────▼─────────┐  │
                              │               │  cdp_engine.py    │  │
                              │               │  ┌─────────────┐  │  │
                              │               │  │  Generate    │  │  │
                              │               │  │  Verify      │  │  │
                              │               │  │  (OpenCV +   │  │  │
                              │               │  │   NumPy)     │  │  │
                              │               │  └─────────────┘  │  │
                              │               └──────────────────┘  │
                              │                        │            │
                              │  ┌─────────────────────▼──────────┐ │
                              │  │  SQLite (database.py)           │ │
                              │  │  patterns | verifications       │ │
                              │  └────────────────────────────────┘ │
                              │                                     │
                              │  File System:                       │
                              │  patterns/  → generated PNG files   │
                              │  uploads/   → captured + aligned    │
                              │  data/      → phonecdp.db           │
                              └─────────────────────────────────────┘
```

### Request Flow (Verification Example)

1. User selects a pattern and uploads a phone photo in the browser
2. Browser sends `POST /api/verify` with the image as multipart form data
3. FastAPI saves the uploaded image to `uploads/`
4. FastAPI calls `cdp_engine.verify_pattern(original_path, captured_path)`
5. The engine runs the full verification pipeline (crop → markers → align → 4 tests → score)
6. The engine saves the aligned image to `uploads/` and returns the result dict
7. FastAPI stores the result in SQLite and returns JSON to the browser
8. The browser renders the verdict, confidence gauge, score bars, and side-by-side images

---

## 4. Directory Structure

```
Production/
├── app.py                 # FastAPI application — all API routes in one file
├── cdp_engine.py          # Core engine — pattern generation + verification
├── config.py              # Configuration constants and directory paths
├── database.py            # SQLAlchemy models (Pattern, Verification) + DB setup
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker build for deployment
├── render.yaml            # Render.com deployment configuration
├── .dockerignore          # Files excluded from Docker build context
├── .gitignore             # Files excluded from git tracking
├── templates/
│   └── index.html         # Single HTML shell for the SPA
├── static/
│   ├── css/
│   │   └── app.css        # Custom animations, scrollbar, print/responsive styles
│   ├── js/
│   │   └── app.js         # Complete SPA — router, all 5 pages, API client
│   └── img/
│       └── logo.svg       # PhoneCDP logo (concentric rings on blue background)
├── data/                  # SQLite database file (created at runtime)
│   └── phonecdp.db
├── patterns/              # Generated pattern PNG files (created at runtime)
└── uploads/               # Uploaded captures + aligned images (created at runtime)
```

**Why a single-file architecture?** Each concern (engine, API, frontend) is one file to minimize deployment complexity. There are no router modules, no service layers, no build steps. `pip install` + `python app.py` runs everything.

---

## 5. Pattern Generation — How It Works

### Entry Point

`cdp_engine.generate_pattern(output_dir, seed=None, serial_number="SN-0001", pattern_size=512)`

### Step-by-Step Process

#### Step 1: Seed Resolution

```python
if seed is None:
    seed = int(np.random.randint(0, 2**31))
```

If no seed is provided, a random 31-bit integer is generated. This seed deterministically controls every aspect of the pattern — the same seed always produces the exact same image.

#### Step 2: Grating Parameters (Seed-Derived)

```python
grating_rng = np.random.RandomState(seed=seed + 2000)
base_freq = 8 + grating_rng.random() * 6        # 8.0 to 14.0
mod_freq  = 1.5 + grating_rng.random() * 2.5    # 1.5 to 4.0
mod_depth = 0.2 + grating_rng.random() * 0.3    # 0.2 to 0.5
```

Using `seed + 2000` ensures the grating parameters are independent from the PRNG block pattern (which uses `seed` directly).

#### Step 3: Generate Grating Layer

`generate_frequency_modulated_grating(width, height, base_freq, mod_freq, mod_depth)`

- Creates a 2D meshgrid of x, y coordinates
- Computes spatially varying frequency: `freq(x) = base_freq * (1 + mod_depth * sin(2π * mod_freq * x / width))`
- Computes cumulative phase via `np.cumsum` along x-axis
- Applies `sin(phase)` to get the grating
- Normalizes to 0–255 uint8

Output: a grayscale image with visible wave-like bands whose spacing varies horizontally.

#### Step 4: Generate PRNG Block Layer

`generate_prng_macro_pattern(width, height, seed, block_size=8)`

- Creates `RandomState(seed)` for reproducibility
- Generates random values for a grid of `(height/8) × (width/8)` blocks
- Upscales each block value to fill 8×8 pixels via `np.repeat`

Output: a grayscale image with a blocky, random-noise appearance.

#### Step 5: Blend Layers

```python
combined = cv2.addWeighted(grating, 0.5, prng, 0.5, 0)
```

Equal 50/50 blend of the two grayscale layers.

#### Step 6: Add RGB Color Perturbations

`add_rgb_perturbations(combined, seed=seed, intensity=25, block_size=8)`

- Converts grayscale to RGB
- Uses `RandomState(seed + 1000)` (separate from the block PRNG)
- For each 8×8 block, for each color channel (R, G, B): adds a random shift in [-25, +25]
- Clips values to valid 0–255 range

Output: an RGB image with subtle, block-level color variations.

#### Step 7: Add Fiducial Markers

`add_fiducial_markers(pattern_rgb, marker_size=48, marker_color=0)`

- Clears a white circular background at each corner
- Draws concentric rings with different counts per corner (described in Section 2)

#### Step 8: Save

```python
filename = f"phone_cdp_{timestamp}_{seed}.png"
cv2.imwrite(filepath, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
```

Returns a dict with `seed`, `filename`, `filepath`, `serial_number`, `pattern_size`.

---

## 6. Verification Pipeline — How It Works

### Entry Point

`cdp_engine.verify_pattern(original_path, captured_path, uploads_dir="uploads")`

The pipeline has 5 stages:

### Stage 0: Pattern Detection and Cropping

**Problem**: The phone photo contains the pattern somewhere within a larger scene (table, background, etc.). We need to find and extract just the pattern region.

**Function**: `detect_and_crop_pattern(image)`

**Strategy 1 — Contour Detection** (`_find_pattern_contour`):

1. Convert to grayscale, compute overall image mean brightness
2. Gaussian blur (5×5) → Otsu's binary threshold (inverted)
3. Morphological close (15×15 kernel) to fill gaps
4. Find external contours
5. For each contour, apply validation filters:
   - **Minimum area**: must be ≥ 3% of image area (reject tiny noise)
   - **Aspect ratio**: must be between 0.7 and 1.4 (patterns are roughly square)
   - **Not full-image**: width and height must each be < 90% of image dimensions (reject paper-background contours that span the whole photo)
   - **Darker than background**: the region's mean brightness must be < 95% of the overall image mean (the pattern has darker content than white paper)
6. Score surviving candidates by `darkness + texture` (standard deviation), pick the best

**Strategy 2 — Marker-Based Detection** (`_find_markers_in_image` + `_find_square_group`):

If no valid contour is found:

1. Apply HoughCircles with progressively lower sensitivity (param2: 40 → 30 → 20)
2. Limit to 30 circle candidates to keep combinatorial search tractable
3. Test all combinations of 4 circles to find ones forming a square:
   - Radius consistency: max radius ≤ 2× min radius
   - 4 shortest distances (sides) must have ratio ≤ 1.3
   - 2 longest distances (diagonals) must have ratio ≤ 1.3
   - Diagonal length must be ≈ √2 × side length (ratio 0.75–1.35)
4. Crop the bounding box with 10% padding

**Cropping** (`_crop_from_contour`):

- If the pattern is nearly axis-aligned (< 5° rotation): simple bounding-box crop with 2% padding
- If rotated: compute the minimum-area rotated rectangle, sort corner points, apply perspective warp to de-rotate

**Output**: cropped BGR image, bounding box, and a boolean `pattern_found`.

### Stage 1: Fiducial Marker Detection

**Function**: `detect_fiducial_markers(image)`

**Purpose**: Find the 4 corner markers and identify which corner is which (by counting rings), to enable precise perspective alignment.

**Process**:

1. **Scale estimation**: `scale = max(image_size) / 512` — adapts detection parameters to the image resolution

2. **Circle detection**: HoughCircles on the (cropped) image with parameters scaled to the expected marker radius (~20px at 512px)

3. **Corner-zone filtering**: Instead of analyzing all detected circles (which may include false positives from the grating pattern), only consider circles within 25% of each image corner:
   ```
   corner_radius = max(width, height) * 0.25
   For each image corner (TL, TR, BL, BR):
       Find the closest detected circle within corner_radius
   ```

4. **Ring counting** (`count_rings_radial_profile`): For each corner candidate:
   - Sample brightness along 36 angles at increasing radii from the circle center
   - Average samples at each radius to create a 1D radial profile
   - Threshold at 35% of the min-max range
   - Count dark-to-light transitions (= number of rings)
   - Special case: if the center region is all dark → solid fill (0 rings = bottom-right)

5. **Assignment with fallback**:
   - **Primary**: assign by ring count (1→TL, 2→TR, 3→BL, 0→BR)
   - **Positional fallback**: if ring counting fails for some corners but circles were still found at those positions, assign by position (the closest circle to TL position becomes top_left, etc.)

**Output**: dict with `top_left`, `top_right`, `bottom_left`, `bottom_right` coordinates (or None), and `markers_found` count (0–4).

### Stage 2: Image Alignment

**Function**: `align_captured_image(captured, original_size, markers)`

Three methods depending on how many markers were found:

| Markers Found | Method | Transformation |
|--------------|--------|---------------|
| 4 | `cv2.getPerspectiveTransform` + `warpPerspective` | Full perspective (4-point homography) |
| 2–3 | `cv2.getAffineTransform` + `warpAffine` | Affine (3-point: translation, rotation, scale, shear) |
| 0–1 | `cv2.resize` | Simple resize to target dimensions |

The destination points for the transform map each marker to its expected position in the original pattern: `(offset, offset)` for TL, `(width-offset, offset)` for TR, etc., where `offset = 24px` (half the marker size).

**Output**: the captured image warped to match the original's geometry.

### Stage 3: Four Verification Tests

All tests operate on the aligned image compared against the original and return a score from 0.0 (no match) to 1.0 (perfect match).

#### Test 1: Moiré Detection (Weight: 40%)

**Function**: `test_moire_detection(captured_gray, reference_gray)`

**Theory**: When a frequency-modulated grating is printed and re-photographed, the interaction between the print's halftone screen and the camera's pixel sampling creates moiré fringes. The frequency spectrum of the captured image should closely match the original's spectrum if it's a first-generation print.

**Process**:
1. Compute 2D FFT of both images → shift zero-frequency to center → log magnitude spectrum
2. Zero out the DC component (center 6×6 pixels)
3. Normalize both spectra to [0, 1]
4. Compute mean absolute difference
5. Score: `clip(1.0 - diff / 0.20, 0.0, 1.0)` — a difference < 0.20 scores 1.0, ≥ 0.20 scores 0.0

#### Test 2: Color Analysis (Weight: 30%)

**Function**: `test_color_analysis(captured_rgb, reference_rgb)`

**Theory**: The RGB perturbations create channel-specific differences (R-G, R-B, G-B). Printing attenuates these differently from scanning. A genuine print-and-photo preserves more of the original channel relationships than a copy-of-a-copy.

**Process** (three sub-scores):
1. **Channel-difference ratios** (30%): Compare the mean absolute inter-channel differences (|R-G|, |R-B|, |G-B|) between captured and reference
2. **Variance ratio** (30%): Compare total per-channel variance (captured vs reference)
3. **Pixel-level difference** (40%): Mean absolute pixel difference, scored as `clip(1.0 - (diff - 5) / 50, 0, 1)` — tolerates up to 5 units of difference perfectly, degrades linearly to 55

Combined: `0.3 * channel_ratio + 0.3 * variance_ratio + 0.4 * pixel_score`

#### Test 3: PRNG Correlation (Weight: 20%)

**Function**: `test_prng_correlation(captured_gray, reference_gray, block_size=8)`

**Theory**: The PRNG macro-block pattern creates a unique spatial fingerprint. By computing block-averaged brightness values and correlating them between captured and original, we test whether the captured image contains the same PRNG sequence.

**Process**:
1. Resize captured to match reference dimensions if needed
2. For each block scale (8, 16, 32, 64 pixels):
   - Divide both images into blocks of that size
   - Compute mean brightness of each block
   - Flatten to 1D vectors, subtract means
   - Compute Pearson correlation: `Σ(cap * ref) / √(Σcap² × Σref²)`
3. Take the best correlation across scales
4. Score: `clip(best_corr / 0.4, 0, 1)` — a correlation ≥ 0.4 scores 1.0

The multi-scale approach handles alignment imperfections — larger blocks are more robust to small spatial errors.

#### Test 4: Gradient Energy (Weight: 10%)

**Function**: `test_gradient_energy(captured_gray, reference_gray)`

**Theory**: Edge energy (magnitude of Sobel gradients) is reduced by each print-scan cycle. A genuine single-print should retain more edge energy relative to the original than a counterfeit double-print.

**Process**:
1. Compute Sobel gradients (x and y) for both images
2. Compute mean gradient magnitude: `mean(√(gx² + gy²))`
3. Ratio: `captured_energy / reference_energy`
4. Score: `clip((ratio - 0.15) / 0.45, 0, 1)` — ratio of 0.60+ scores 1.0

### Stage 4: Final Scoring and Verdict

```python
final = 0.40 * moire + 0.30 * color + 0.20 * correlation + 0.10 * gradient
```

| Score Range | Verdict |
|-------------|---------|
| ≥ 0.70 | **AUTHENTIC** — pattern matches the original with high confidence |
| 0.50 – 0.69 | **SUSPICIOUS** — partial match, may need manual review |
| < 0.50 | **COUNTERFEIT** — significant degradation indicates a copy |

### Stage 5: Save Results

The aligned image is saved to `uploads/aligned_{timestamp}.png` for later visual comparison. The function returns a comprehensive result dict including all scores, weights, marker count, alignment method, and the aligned filename.

---

## 7. Database Schema

### Table: `patterns`

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTO INCREMENT | Unique pattern identifier |
| `seed` | INTEGER | NOT NULL | PRNG seed used for generation |
| `serial_number` | VARCHAR(50) | DEFAULT "" | User-assigned product serial |
| `label` | VARCHAR(100) | DEFAULT "" | Human-readable name/label |
| `filename` | VARCHAR(255) | NOT NULL | PNG filename in `patterns/` directory |
| `pattern_size` | INTEGER | DEFAULT 512 | Image dimensions (width = height) |
| `notes` | TEXT | DEFAULT "" | Free-form notes |
| `created_at` | DATETIME | DEFAULT utcnow | Generation timestamp |

**Relationships**: One-to-many → `verifications` (cascade delete)

### Table: `verifications`

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTO INCREMENT | Unique verification identifier |
| `pattern_id` | INTEGER | FOREIGN KEY → patterns.id, NOT NULL | Which pattern was verified against |
| `captured_filename` | VARCHAR(255) | DEFAULT "" | Uploaded photo filename in `uploads/` |
| `aligned_filename` | VARCHAR(255) | DEFAULT "" | Aligned image filename in `uploads/` |
| `verdict` | VARCHAR(20) | DEFAULT "" | AUTHENTIC, SUSPICIOUS, or COUNTERFEIT |
| `confidence` | FLOAT | DEFAULT 0.0 | Weighted final score (0.0–1.0) |
| `score_moire` | FLOAT | DEFAULT 0.0 | Moiré test score (0.0–1.0) |
| `score_color` | FLOAT | DEFAULT 0.0 | Color analysis score (0.0–1.0) |
| `score_correlation` | FLOAT | DEFAULT 0.0 | PRNG correlation score (0.0–1.0) |
| `score_gradient` | FLOAT | DEFAULT 0.0 | Gradient energy score (0.0–1.0) |
| `markers_found` | INTEGER | DEFAULT 0 | Number of fiducial markers detected (0–4) |
| `alignment_method` | VARCHAR(20) | DEFAULT "" | perspective, affine, or resize |
| `print_size_mm` | INTEGER | NULLABLE | Optional: physical print size in mm |
| `notes` | TEXT | DEFAULT "" | Free-form notes |
| `created_at` | DATETIME | DEFAULT utcnow | Verification timestamp |

**Relationships**: Many-to-one → `pattern`

---

## 8. API Reference

Base URL: `http://localhost:8000` (local) or your Render URL

### Health Check

```
GET /api/health
```

Response: `{"status": "ok"}`

Used by Render for health monitoring. Returns 200 if the server is alive.

---

### Patterns

#### List All Patterns

```
GET /api/patterns
```

Response: Array of pattern objects, ordered by creation date (newest first).

```json
[{
    "id": 1,
    "seed": 42,
    "serial_number": "SN-2026-00001",
    "label": "Product Batch A",
    "filename": "phone_cdp_20260222_120000_42.png",
    "pattern_size": 512,
    "created_at": "2026-02-22T12:00:00.000000",
    "notes": "",
    "verification_count": 3
}]
```

#### Generate New Pattern

```
POST /api/patterns/generate
Content-Type: application/json

{
    "seed": 42,              // optional, null = random
    "serial_number": "SN-0001",
    "label": "Test Pattern",
    "notes": "For batch A testing",
    "pattern_size": 512      // default 512
}
```

Response: the created pattern object (without `verification_count`).

#### Get Pattern Details

```
GET /api/patterns/{id}
```

Response: single pattern object with `verification_count`.

#### Preview Pattern Image

```
GET /api/patterns/{id}/preview
```

Response: PNG image (inline, `Content-Type: image/png`).

#### Download Pattern Image

```
GET /api/patterns/{id}/download
```

Response: PNG image with `Content-Disposition: attachment` header (triggers browser download).

#### Delete Pattern

```
DELETE /api/patterns/{id}
```

Response: `{"message": "Deleted"}`

Deletes the pattern record AND the PNG file from disk. Also cascade-deletes all associated verifications.

---

### Verification

#### Verify Single Photo

```
POST /api/verify
Content-Type: multipart/form-data

captured:       <file>       (required) phone photo
pattern_id:     1            (required) pattern to verify against
print_size_mm:  65           (optional) physical print size in mm
notes:          "test note"  (optional)
```

Response:

```json
{
    "id": 1,
    "pattern_id": 1,
    "verdict": "AUTHENTIC",
    "confidence": 0.823,
    "scores": {
        "moire": 0.912,
        "color": 0.756,
        "correlation": 0.834,
        "gradient": 0.601
    },
    "weights": {
        "moire": 0.4,
        "color": 0.3,
        "correlation": 0.2,
        "gradient": 0.1
    },
    "markers_found": 4,
    "alignment_method": "perspective",
    "pattern_found": true,
    "print_size_mm": 65,
    "notes": "test note",
    "created_at": "2026-02-22T12:05:00.000000"
}
```

#### Batch Verify Multiple Photos

```
POST /api/verify/batch
Content-Type: multipart/form-data

captured_files:  <file1>     (required, multiple files)
captured_files:  <file2>
pattern_id:      1           (required)
print_size_mm:   65          (optional)
notes:           "batch"     (optional)
```

Response:

```json
{
    "results": [
        {"id": 1, "filename": "photo1.jpg", "verdict": "AUTHENTIC", "confidence": 0.82, "scores": {...}, "markers_found": 4},
        {"id": 2, "filename": "photo2.jpg", "verdict": "SUSPICIOUS", "confidence": 0.55, "scores": {...}, "markers_found": 3}
    ],
    "total": 2
}
```

#### Get Verification Details

```
GET /api/verify/{id}
```

Response: full verification object with pattern serial/label included.

#### Get Verification Images

```
GET /api/verify/{id}/images/{type}
```

`type` must be one of: `original`, `captured`, `aligned`

Response: image file.

---

### Results

#### List Results (with filtering)

```
GET /api/results?verdict=AUTHENTIC&pattern_id=1&limit=50&offset=0
```

All query parameters are optional:
- `verdict`: AUTHENTIC, SUSPICIOUS, or COUNTERFEIT
- `pattern_id`: filter to a specific pattern
- `limit`: max 200, default 50
- `offset`: for pagination

Response:

```json
{
    "results": [{...}, {...}],
    "total": 42
}
```

#### Get Statistics

```
GET /api/results/stats
```

Response:

```json
{
    "total_patterns": 5,
    "total_verifications": 42,
    "verdicts": {"authentic": 30, "suspicious": 8, "counterfeit": 4},
    "pass_rate": 71.4,
    "avg_confidence": 0.723,
    "avg_markers": 3.6,
    "avg_scores": {
        "moire": 0.785,
        "color": 0.654,
        "correlation": 0.712,
        "gradient": 0.589
    }
}
```

#### Export CSV

```
GET /api/results/export
```

Response: CSV file download with columns: ID, Date, Pattern ID, Serial, Label, Verdict, Confidence, Moire, Color, Correlation, Gradient, Markers, Alignment, Print Size (mm), Notes.

#### Delete Result

```
DELETE /api/results/{id}
```

Response: `{"message": "Deleted"}`

#### Update Notes

```
PATCH /api/results/{id}/notes
Content-Type: application/json

{"notes": "Updated note text"}
```

Response: `{"message": "Updated", "notes": "Updated note text"}`

---

## 9. Frontend Application

### Architecture

The frontend is a **Single Page Application (SPA)** implemented in a single JavaScript file (`app.js`). It uses hash-based routing (`#/`, `#/generate`, `#/verify`, `#/results`, `#/results/{id}`) with no build step, no framework, and no dependencies beyond Tailwind CSS loaded from CDN.

### HTML Shell (`index.html`)

The HTML file provides:
- **Navigation bar**: PhoneCDP logo, app name, version badge, and 4 nav links (Dashboard, Generate, Verify, Results)
- **Main content area**: `<main id="app">` — dynamically populated by the SPA router
- **Toast container**: Fixed-position notification system at bottom-right
- **Image modal**: Full-screen overlay for viewing images at full resolution
- **Tailwind config**: Custom `brand` color palette extending Tailwind's default theme

### Router

```javascript
function getRoute() {
    const hash = window.location.hash.slice(1) || '/';
    if (hash.startsWith('/results/')) return { fn: pgDetail, params: { id } };
    return { fn: routeMap[hash] || pgDash, params: {} };
}
```

Listens to `hashchange` and `load` events. Each route maps to a page function that receives the `#app` element and renders into it.

### Pages

#### Dashboard (`#/`)

- Fetches stats from `GET /api/results/stats` and recent results from `GET /api/results?limit=8` in parallel
- Displays 4 summary cards: Total Patterns, Total Verifications, Pass Rate, Avg Confidence
- 3 verdict breakdown cards (green/yellow/red) showing Authentic, Suspicious, Counterfeit counts
- Quick Actions panel: Generate, Verify, Export CSV
- Recent Verifications table: clickable rows navigate to detail

#### Generate Pattern (`#/generate`)

- Form with fields: Serial Number (auto-populated), Label, Seed (optional), Notes
- On submit: POST to `/api/patterns/generate`, shows spinner during generation
- Displays the generated pattern preview with Download PNG and Verify This buttons
- Shows pattern metadata (serial, seed, size, date)
- Below the form: Gallery grid of all previously generated patterns as thumbnails

#### Verify Pattern (`#/verify`)

- **Step 1**: Dropdown to select from generated patterns, with live preview thumbnail
- **Step 2**: Drag-and-drop zone (or click to browse) for uploading phone photos. Supports multiple file selection
- **Step 3**: Optional fields — Print Size (mm) and Notes
- Verify button (disabled until a file is selected)
- **Single file**: Shows inline result with verdict badge, confidence gauge, score breakdowns with colored bars, markers/alignment info, and 3-up image comparison (original | captured | aligned)
- **Multiple files**: Shows batch summary (X/Y passed) with per-file verdict rows

#### Results History (`#/results`)

- Filter buttons: All, Authentic, Suspicious, Counterfeit (client-side filtering)
- Full table with columns: Date, Pattern, Verdict, Confidence, Markers, Alignment, Print Size, Actions
- Color-coded verdict badges
- View (eye icon) and Delete (trash icon) actions per row
- Export CSV button

#### Result Detail (`#/results/{id}`)

- Back link to results list
- Verification ID, date, verdict badge
- Confidence gauge with threshold markers at 50% (suspicious) and 70% (authentic)
- Score breakdown bars with weight percentages
- Marker detection visual (4 circles, green = found, gray = not found)
- Alignment method and pattern info
- Editable notes field with Save button
- 3-column image comparison: Original | Captured | Aligned (all clickable for full-size modal)
- Quick Re-test link

### Shared UI Components

| Component | Function | Description |
|-----------|----------|-------------|
| `vBadge(verdict)` | Verdict badge | Green/yellow/red pill with verdict text |
| `confGauge(value)` | Confidence gauge | Horizontal bar with threshold markers |
| `sBar(label, value, weight)` | Score bar | Labeled progress bar with weight % and value |
| `showToast(msg, type)` | Toast notification | Slide-up notification with icon (success/error/info) |
| `openImageModal(src)` | Image modal | Full-screen overlay with close button |
| `spin(text)` | Loading spinner | Centered spinner with optional text |
| `fmtDate(iso)` | Date formatter | "Feb 22, 2026 12:05 PM" format |

### API Client

The `API` object wraps `fetch` with error handling:
- `API.get(url)` — GET request, returns JSON
- `API.post(url, body)` — POST with JSON body
- `API.postForm(url, formData)` — POST with multipart form data
- `API.del(url)` — DELETE request
- `API.patch(url, body)` — PATCH with JSON body

All methods throw on non-2xx responses, extracting error details from the response body when available.

---

## 10. Deployment

### Render.com Deployment

#### Prerequisites

1. Push the Production directory to a GitHub repository
2. Create a Render account at render.com

#### Steps

1. Go to Render Dashboard → **New** → **Web Service**
2. Connect your GitHub repository
3. Render auto-detects the `Dockerfile`
4. Configuration is read from `render.yaml`:
   - **Plan**: Free tier
   - **Health check**: `GET /api/health`
   - **Runtime**: Docker
5. Click Deploy

#### What Happens During Build

1. Render clones the repo
2. Docker builds from the Dockerfile:
   - Base: `python:3.11-slim`
   - Installs system deps: `libglib2.0-0`, `libgl1` (for OpenCV)
   - Installs Python deps from `requirements.txt`
   - Copies application code
   - Creates data directories
3. Starts with: `gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120 --workers 2`

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Render sets this automatically |
| `PYTHON_VERSION` | 3.11 | Set in render.yaml |

#### Important Notes

- **SQLite on free tier**: The database resets on each deploy/restart since Render's free tier uses ephemeral storage. For persistent data, upgrade to a paid plan with a persistent disk, or switch to PostgreSQL.
- **File storage**: Generated patterns and uploaded images are also stored on the ephemeral filesystem. Consider adding cloud storage (S3/Cloudflare R2) for production use.
- **Timeout**: Gunicorn timeout is set to 120 seconds to accommodate slow verification processing on large images.

### Local Development

```bash
cd Production
pip install -r requirements.txt
python app.py
# Server starts at http://localhost:8000
```

### Docker (Local)

```bash
cd Production
docker build -t phonecdp .
docker run -p 8000:8000 phonecdp
```

---

## 11. Configuration

### config.py

| Constant | Value | Description |
|----------|-------|-------------|
| `BASE_DIR` | Directory containing config.py | Root of the application |
| `PATTERNS_DIR` | `{BASE_DIR}/patterns` | Where generated PNGs are stored |
| `UPLOADS_DIR` | `{BASE_DIR}/uploads` | Where uploaded photos and aligned images are stored |
| `DATA_DIR` | `{BASE_DIR}/data` | Where the SQLite database file lives |
| `DATABASE_URL` | `sqlite:///{DATA_DIR}/phonecdp.db` | SQLAlchemy connection string |
| `PORT` | `int(os.environ.get("PORT", 8000))` | Server port (Render sets via env var) |

All directories are created automatically on import (`os.makedirs(exist_ok=True)`).

### cdp_engine.py Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PATTERN_SIZE` | (512, 512) | Default pattern dimensions |
| `FIDUCIAL_MARKER_SIZE` | 48 | Marker diameter in pixels |
| `FIDUCIAL_MARKER_OFFSET` | 24 | Center offset from corner |
| `BLOCK_SIZE` | 8 | PRNG macro-block size |
| `WEIGHT_MOIRE` | 0.40 | Moiré test weight in final score |
| `WEIGHT_COLOR` | 0.30 | Color test weight |
| `WEIGHT_CORRELATION` | 0.20 | Correlation test weight |
| `WEIGHT_GRADIENT` | 0.10 | Gradient test weight |
| `THRESHOLD_AUTHENTIC` | 0.70 | Minimum score for AUTHENTIC |
| `THRESHOLD_SUSPICIOUS` | 0.50 | Minimum score for SUSPICIOUS |
| `RING_COUNT_TO_CORNER` | {1: TL, 2: TR, 3: BL, 0: BR} | Maps ring count to corner identity |

---

## 12. File-by-File Breakdown

### `cdp_engine.py` (567 lines)

The heart of the system. Contains all image processing logic with zero external dependencies beyond NumPy and OpenCV. Functions are organized into sections:

- **Pattern Generation** (lines 37–123): `generate_frequency_modulated_grating`, `generate_prng_macro_pattern`, `add_rgb_perturbations`, `add_fiducial_markers`, `generate_pattern`
- **Pattern Detection** (lines 130–274): `_find_pattern_contour`, `_crop_from_contour`, `_find_markers_in_image`, `_find_square_group`, `detect_and_crop_pattern`
- **Marker Detection** (lines 281–382): `count_rings_radial_profile`, `detect_fiducial_markers`
- **Alignment** (lines 389–414): `align_captured_image`
- **Verification Tests** (lines 421–494): `test_moire_detection`, `test_color_analysis`, `test_prng_correlation`, `test_gradient_energy`
- **Pipeline** (lines 501–566): `verify_pattern` — orchestrates the entire process

### `app.py` (371 lines)

FastAPI application with all routes defined in a single file. Sections:

- **HTML serving** (lines 29–37): Serves `index.html` and favicon
- **Health check** (lines 42–44): `/api/health` for Render
- **Patterns API** (lines 49–128): CRUD for patterns + generate + preview + download
- **Verify API** (lines 133–266): Single verification, batch verification, verification detail, image serving
- **Results API** (lines 271–359): List, stats, CSV export, delete, update notes
- **Startup** (lines 364–370): Database initialization on server start

### `database.py` (55 lines)

SQLAlchemy ORM setup:
- Engine creation with `check_same_thread=False` (required for SQLite with FastAPI)
- `Pattern` model with one-to-many relationship to `Verification` (cascade delete)
- `Verification` model with many-to-one back-reference to `Pattern`
- `init_db()` — creates tables if they don't exist
- `get_db()` — FastAPI dependency that yields a session and closes it after the request

### `config.py` (11 lines)

Path configuration. All directories auto-created on import.

### `templates/index.html` (61 lines)

Single HTML file serving as the SPA shell. Notable elements:
- Tailwind CSS loaded from CDN with custom `brand` color palette configuration
- Navigation bar with 4 hash-based links
- Toast notification container (hidden by default)
- Image modal overlay (hidden by default)
- Single `<script>` tag loading `app.js`

### `static/js/app.js` (214 lines)

Complete SPA implementation:
- **API client** (lines 5–11): Fetch wrapper with error handling
- **Helpers** (lines 14–37): Toast, modal, badges, gauges, bars, date formatting, spinner
- **Router** (lines 42–57): Hash-based routing with nav highlighting
- **Dashboard** (lines 62–84): Stats + recent verifications
- **Generate** (lines 89–117): Pattern creation form + gallery
- **Verify** (lines 122–171): File upload + verification + results display
- **Results** (lines 176–191): Filterable table + export
- **Detail** (lines 196–213): Full result view with images + editable notes

### `static/css/app.css` (57 lines)

Custom styles supplementing Tailwind:
- Spinner and toast animations
- Drag-and-drop hover pulse effect
- Image hover zoom hints
- Modal zoom-in animation
- Score bar and gauge transitions
- Custom scrollbar styling
- Responsive table adjustments for mobile
- Print-friendly styles (hides nav, buttons, shadows)

### `Dockerfile` (18 lines)

Multi-step build:
1. Base: `python:3.11-slim`
2. System dependencies for OpenCV: `libglib2.0-0`, `libgl1`
3. Python dependencies via pip
4. Copy application code
5. Create runtime directories
6. Expose port 8000
7. CMD: Gunicorn with Uvicorn workers (2 workers, 120s timeout)

### `render.yaml` (10 lines)

Render Infrastructure-as-Code:
- Web service type
- Docker runtime
- Free plan
- Health check on `/api/health`

### `requirements.txt` (8 lines)

Pinned versions for reproducible builds:
- `fastapi==0.115.0` — Web framework
- `uvicorn[standard]==0.30.0` — ASGI server
- `sqlalchemy==2.0.35` — ORM
- `python-multipart==0.0.12` — File upload parsing
- `aiofiles==24.1.0` — Async file serving
- `numpy` — Numerical computing (unpinned for compatibility)
- `opencv-python-headless` — Image processing (no GUI deps)
- `gunicorn==22.0.0` — Production WSGI/ASGI server

---

*PhoneCDP v1.0 — NineHawks*
