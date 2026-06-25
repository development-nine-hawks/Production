# Counterfeit Discrimination Research Notes

## Context

**Date:** 2026-06-25

**Label:** Ninehawk PhoneCDP label, seed=1079726699 (0x405b526b)

**Counterfeit method:** Direct photocopy using Canon GM4000 Series printer/copier function (optical flatbed scan → reprint at same scale)

**Test set:** 10 images (tc-01 to tc-10). tc-09 and tc-10 are confirmed counterfeits. tc-01 to tc-08 are genuine first-generation prints photographed with a phone camera at various angles and distances.

---

## Baseline Scores (all flags False, original system)

| TC    | Verdict   | Conf  | Moire | Corr  | Grad  | Color | Label     |
|-------|-----------|-------|-------|-------|-------|-------|-----------|
| tc-01 | AUTHENTIC | 90.2% | 0.941 | 1.000 | 0.960 | 0.470 | Genuine   |
| tc-02 | AUTHENTIC | 68.3% | 0.617 | 1.000 | 0.971 | 0.364 | Genuine   |
| tc-03 | AUTHENTIC | 79.0% | 0.791 | 1.000 | 0.899 | 0.411 | Genuine   |
| tc-04 | AUTHENTIC | 72.1% | 0.676 | 1.000 | 0.948 | 0.394 | Genuine   |
| tc-05 | AUTHENTIC | 82.0% | 0.833 | 1.000 | 0.938 | 0.383 | Genuine   |
| tc-06 | AUTHENTIC | 92.4% | 1.000 | 1.000 | 0.934 | 0.335 | Genuine   |
| tc-07 | AUTHENTIC | 70.1% | 0.638 | 1.000 | 0.917 | 0.488 | Genuine   |
| tc-08 | AUTHENTIC | 79.6% | 0.803 | 1.000 | 0.925 | 0.352 | Genuine   |
| tc-09 | AUTHENTIC | 79.5% | 0.799 | 1.000 | 0.901 | 0.405 | Counterfeit |
| tc-10 | AUTHENTIC | 92.8% | 1.000 | 1.000 | 0.932 | 0.383 | Counterfeit |

**Problem:** Both counterfeits score AUTHENTIC. tc-10 scores higher than most genuine prints. corr=1.0 and grad=0.93 for all images — these two metrics provide zero discrimination.

---

## Key Observations

### Why the Canon GM4000 photocopy fools the system

The GM4000 copier function uses a flatbed optical scanner at high DPI followed by inkjet reprint at the same scale. This means:

- Macro block structure (16px PRNG blocks) is reproduced perfectly → corr=1.0 for counterfeit
- Medium-frequency grating (sigma 2-8) is reproduced faithfully → moire=1.0 for tc-10
- Gradient structure is reproduced cleanly → grad=0.93 for counterfeit

The photocopy is essentially a perfect reproduction at the scales currently being measured. This is fundamentally different from a photo→reprint counterfeit, which introduces perspective distortion, focus blur, and exposure variation.

### Bandpass scores (from moire debug logging)

The raw bandpass NCC scores (before normalization) revealed the key discriminating signal:

| TC    | bandpass_score | Label       |
|-------|----------------|-------------|
| tc-01 | 0.2351         | Genuine     |
| tc-02 | 0.1542         | Genuine     |
| tc-03 | 0.1978         | Genuine     |
| tc-04 | 0.1691         | Genuine     |
| tc-05 | 0.2082         | Genuine     |
| tc-06 | 0.2525         | Genuine     |
| tc-07 | 0.1595         | Genuine     |
| tc-08 | 0.2007         | Genuine     |
| tc-09 | 0.1998         | Counterfeit |
| tc-10 | 0.2788         | Counterfeit |

**Genuine range:** 0.154 – 0.2525

**Counterfeit range:** 0.1998 – 0.2788

**Overlap:** tc-09 (0.1998) falls within genuine range. tc-10 (0.2788) is clearly above genuine ceiling.

**Key insight:** The flatbed photocopy reproduces the grating more faithfully than a phone camera does. A genuine phone capture always loses some fine grating fidelity due to lens blur, JPEG compression, and perspective. A photocopy scanned on a flatbed preserves it better — resulting in a higher bandpass NCC against the digital reference.

---

## Point 1 — Finer Moire Bands (CDP_FLAG_MOIRE_FINE_BANDS)

**Hypothesis:** Ultra-fine frequency bands (sigma 0.4-0.8) should be below the printer's reproduction fidelity threshold, so counterfeits would score lower than genuine prints at those frequencies.

**Change:**
```python
# Original
bands = [
    (2, 1, 0.50),
    (4, 2, 0.30),
    (8, 4, 0.20),
]

# Changed to
bands = [
    (0.8, 0.4, 0.50),
    (1.5, 0.8, 0.25),
    (3,   1.5, 0.15),
    (6,   3,   0.10),
]
```

**Result:** Opposite of expected. Fine bands hurt genuine prints more than counterfeits. All 10 became SUSPICIOUS. Genuine moire range dropped to 0.277-0.479. Counterfeit (tc-10) moire=0.556 — still the highest. The Canon GM4000 reproduces ultra-fine grating better than phone camera capture preserves it.

**Conclusion:** Wrong direction. The camera capture chain destroys fine detail more aggressively than the photocopy process does. Do not use.

---

## Point 2 — Lower Moire Normalization Divisor (CDP_FLAG_MOIRE_NORM_DIVISOR)

**Hypothesis:** Reduce divisor from 0.25 to 0.18 to compensate for lower raw NCC values at finer bands.

**Change:** `return float(np.clip(score / 0.18, 0.0, 1.0))` instead of `/ 0.25`

**Result:** Only meaningful in combination with Point 1. In isolation has no discriminating effect since all images score similarly before normalization. Not independently useful.

**Conclusion:** Dependent on Point 1. Since Point 1 was rejected, this is also not useful independently.

---

## Point 3 — Finer Block Correlation (CDP_FLAG_CORR_FINE_BLOCKS)

**Hypothesis:** At 4px and 8px block sizes, the PRNG block structure is too fine for a photocopy to reproduce — it gets averaged out.

**Change:**
```python
# Original
scale_configs = [
    (block_size,     0.60, 1.0),
    (block_size * 2, 0.65, 0.75),
    (block_size * 4, 0.75, 0.50),
]

# Changed to
scale_configs = [
    (block_size // 4, 0.45, 1.0),
    (block_size // 2, 0.55, 1.0),
    (block_size,      0.60, 0.75),
    (block_size * 2,  0.65, 0.50),
]
```

**Result:** No meaningful discrimination. The Canon GM4000 reproduces PRNG block structure even at fine scales. corr=1.0 for all images including counterfeits at all block sizes tested.

**Conclusion:** Correlation is not the right discriminator for flatbed photocopy counterfeits. The macro block pattern survives the copy process faithfully at all tested scales.

---

## Point 4 — Cap Coarse Block Scores (CDP_FLAG_CORR_COARSE_CAP)

**Hypothesis:** If fine blocks have no signal, cap the coarse-block score to prevent a perfect 16px match from giving full marks.

**Result:** Hurts genuine prints equally since all images fall through to 16px. Not discriminating.

**Conclusion:** Cannot cap coarse scores without knowing whether the image is a copy or a far-away genuine capture — both situations result in fine block signal loss.

---

## Working Approach — Too-Perfect Bandpass Penalty

**Hypothesis:** A genuine phone capture always loses some fine grating fidelity. A flatbed photocopy reproduces it too perfectly — bandpass_score is suspiciously high. Penalize scores above the expected ceiling for genuine phone captures.

**Change:**
```python
GENUINE_BANDPASS_CEILING = 0.255
if score > GENUINE_BANDPASS_CEILING:
    too_perfect_penalty = (score - GENUINE_BANDPASS_CEILING) * 8.0
else:
    too_perfect_penalty = 0.0
adjusted_score = score - too_perfect_penalty
return float(np.clip(adjusted_score / 0.25, 0.0, 1.0))
```

**Results with this change:**

| TC    | bandpass | penalty | moire | verdict    | Label          |
|-------|----------|---------|-------|------------|----------------|
| tc-01 | 0.2351   | 0.000   | 0.941 | AUTHENTIC  | Genuine ✓      |
| tc-02 | 0.1542   | 0.000   | 0.617 | AUTHENTIC  | Genuine ✓      |
| tc-03 | 0.1978   | 0.000   | 0.791 | AUTHENTIC  | Genuine ✓      |
| tc-04 | 0.1691   | 0.000   | 0.676 | AUTHENTIC  | Genuine ✓      |
| tc-05 | 0.2082   | 0.000   | 0.833 | AUTHENTIC  | Genuine ✓      |
| tc-06 | 0.2525   | 0.000   | 1.000 | AUTHENTIC  | Genuine ✓      |
| tc-07 | 0.1595   | 0.000   | 0.638 | AUTHENTIC  | Genuine ✓      |
| tc-08 | 0.2007   | 0.000   | 0.803 | AUTHENTIC  | Genuine ✓      |
| tc-09 | 0.1998   | 0.000   | 0.799 | AUTHENTIC  | Counterfeit ✗  |
| tc-10 | 0.2788   | 0.191   | 0.353 | SUSPICIOUS | Counterfeit ✓  |

tc-10 correctly identified as SUSPICIOUS. tc-09 missed — bandpass within genuine range.

**Why tc-09 was missed:** tc-09 used shrink=2 for DM decode indicating the photocopy quality may have been lower. Its bandpass score (0.1998) falls within the genuine range (0.154-0.2525). More counterfeit samples needed to determine if this is consistent.
