# NineHawks CDP — Test Case Results
**Print size:** 7.5mm | **Pattern type:** Label-embedded

---

## Phase 1 — Core Authentication

### TC-01: Genuine label, photo straight-on
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 82.3% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.216 |
| Color | 0.406 |
| Markers Found | 4 |
| Alignment | perspective (0°) |
| Notes |  |

---

### TC-02: Genuine label, slight angle (~30°)
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 86.1% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.455 |
| Color | 0.430 |
| Markers Found | 4 |
| Alignment | perspective (0°) |
| Notes |  |

---

### TC-03: Genuine label, rotated 90°
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 83.7% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.315 |
| Color | 0.398 |
| Markers Found | 4 |
| Alignment | perspective (90°CW) |
| Notes |  |

---

### TC-04: Genuine label, rotated 180°
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 83.9% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.308 |
| Color | 0.430 |
| Markers Found | 4 |
| Alignment | perspective (180°) |
| Notes |  |

---

### TC-05: Genuine label, low light
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 78.5% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.146 |
| Color | 0.126 |
| Markers Found | 4 |
| Alignment | perspective (0°) |
| Notes |  |

---

### TC-06: Genuine label, glare/flash on pattern
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 80.3% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.227 |
| Color | 0.193 |
| Markers Found | 4 |
| Alignment | perspective (0°) |
| Notes |  |

---

### TC-07: Genuine label, far away (pattern small in frame)
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 84.8% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.404 |
| Color | 0.378 |
| Markers Found | 4 |
| Alignment | perspective (0°) |
| Notes |  |

---

### TC-08: Genuine label, motion blur
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | AUTHENTIC |
| Confidence | 81.5% |
| Moire | 1.000 |
| Correlation | 1.000 |
| Gradient | 0.203 |
| Color | 0.346 |
| Markers Found | 4 |
| Alignment | resize (0) |
| Notes |  |

---

### TC-09: Counterfeit label, straight-on (print → photo → reprint → photo)
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | COUNTERFEIT |
| Confidence | 44.1% |
| Moire | 0.481 |
| Correlation | 1.000 |
| Gradient | 0.014 |
| Color | 0.266 |
| Markers Found | 4 |
| Alignment | resize (0) |
| Notes |  |

---

### TC-10: Counterfeit label, at an angle
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | COUNTERFEIT |
| Confidence | 43.9% |
| Moire | 0.504 |
| Correlation | 0.718 |
| Gradient | 0.089 |
| Color | 0.262 |
| Markers Found | 4 |
| Alignment | perspective (0°) |
| Notes |  |

---

### TC-11: Wrong pattern ID against genuine print
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | COUNTERFEIT |
| Confidence | 3.3% |
| Moire | 0.004 |
| Correlation | 0.000 |
| Gradient | 0.040 |
| Color | 0.245 |
| Markers Found | 4 |
| Alignment | crop_resize |
| Notes |  |

---

### TC-12: Screenshot of label on screen, photographed
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | COUNTERFEIT |
| Confidence | 5.8% |
| Moire | 0.041 |
| Correlation | 0.000 |
| Gradient | 0.104 |
| Color | 0.155 |
| Markers Found | 4 |
| Alignment | resize (90CW) |
| Notes |  |

---

## Phase 2 — Advanced Conditions

### TC-13: Genuine label under fluorescent/yellow-tinted light
**Expected:** AUTHENTIC

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-14: Genuine label with heavy shadows across pattern
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-15: Genuine label photographed with a low-quality camera
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-16: Genuine label with fingerprint smudge over pattern
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-17: Genuine label that's been folded/creased through pattern
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-18: Genuine label, slightly wet/water damaged
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-19: Old/faded genuine label (ink degraded)
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-20: Genuine label photographed after being scratched
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-21: Counterfeit printed on a high-quality printer
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-22: Counterfeit printed at a different size than 7.5mm
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-23: Digital counterfeit (screenshot printed)
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-24: Counterfeit with color correction applied before reprinting
**Expected:** COUNTERFEIT

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-25: Two labels in the same photo
**Expected:** AUTHENTIC (verifies one correctly)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-26: Label inside packaging (plastic wrap distortion)
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

### TC-27: Genuine label with another label partially overlapping it
**Expected:** AUTHENTIC (or SUSPICIOUS)

| Field | Value |
|---|---|
| Verdict | |
| Confidence | |
| Moire | |
| Correlation | |
| Gradient | |
| Color | |
| Markers Found | |
| Alignment | |
| Notes | |

---

## Summary

| TC | Description | Expected | Actual | Pass/Fail |
|---|---|---|---|---|
| TC-01 | Genuine, straight-on | AUTHENTIC | | |
| TC-02 | Genuine, ~30° angle | AUTHENTIC | | |
| TC-03 | Genuine, rotated 90° | AUTHENTIC | | |
| TC-04 | Genuine, rotated 180° | AUTHENTIC | | |
| TC-05 | Genuine, low light | AUTHENTIC | | |
| TC-06 | Genuine, glare/flash | AUTHENTIC | | |
| TC-07 | Genuine, far away | AUTHENTIC | | |
| TC-08 | Genuine, motion blur | AUTHENTIC/SUSPICIOUS | | |
| TC-09 | Counterfeit, straight-on | COUNTERFEIT | | |
| TC-10 | Counterfeit, at angle | COUNTERFEIT | | |
| TC-11 | Wrong pattern ID | COUNTERFEIT | | |
| TC-12 | Screen screenshot | COUNTERFEIT | | |
| TC-13 | Genuine, yellow light | AUTHENTIC | | |
| TC-14 | Genuine, heavy shadows | AUTHENTIC/SUSPICIOUS | | |
| TC-15 | Genuine, low-quality camera | AUTHENTIC/SUSPICIOUS | | |
| TC-16 | Genuine, fingerprint smudge | AUTHENTIC/SUSPICIOUS | | |
| TC-17 | Genuine, creased | AUTHENTIC/SUSPICIOUS | | |
| TC-18 | Genuine, water damaged | AUTHENTIC/SUSPICIOUS | | |
| TC-19 | Genuine, faded ink | AUTHENTIC/SUSPICIOUS | | |
| TC-20 | Genuine, scratched | AUTHENTIC/SUSPICIOUS | | |
| TC-21 | Counterfeit, high-quality printer | COUNTERFEIT | | |
| TC-22 | Counterfeit, different size | COUNTERFEIT | | |
| TC-23 | Counterfeit, screenshot printed | COUNTERFEIT | | |
| TC-24 | Counterfeit, color corrected | COUNTERFEIT | | |
| TC-25 | Two labels in frame | AUTHENTIC | | |
| TC-26 | Label under plastic wrap | AUTHENTIC/SUSPICIOUS | | |
| TC-27 | Label partially overlapped | AUTHENTIC/SUSPICIOUS | | |
