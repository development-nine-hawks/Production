# Ternary and Bitwise Operator Changes

Branch: matrix-direct-seed
File changed: cdp_engine.py
Scope: two changes, one ternary operator and one bitwise operator, both placed in code that runs on the generation and detection paths.

## Change 1: Ternary operator for mod_depth selection

### Where

- cdp_engine.py, function `generate_pattern` (around line 899) - the generation path.
- cdp_engine.py, function `regenerate_reference` (around line 3222) - the detection path, called at Step 3 of every verification.

The same block of code existed in both functions. Both were changed the same way so the two stay in lockstep.

### Before

```python
mod_depth = MOD_DEPTH_MIN + grating_rng.random() * MOD_DEPTH_RANGE
if mod_depth_override is not None:
    mod_depth = mod_depth_override
```

### After

```python
mod_depth = (mod_depth_override if mod_depth_override is not None
             else MOD_DEPTH_MIN + grating_rng.random() * MOD_DEPTH_RANGE)
```

### Why this spot matters

- `mod_depth` is passed straight into `generate_frequency_modulated_grating`, which produces the frequency-modulated grating layer of the CDP.
- That grating is one half of the printed pattern (the other half is the PRNG macro pattern), so `mod_depth` shapes the actual thing being authenticated.
- The same value must be used at generation time and at verification time. If they differ, the regenerated reference does not match the captured label and the scores drop. The `regenerate_reference` docstring already warns about this.

### Effect on generation

- No change to the generated image.
- When no override is passed, the random draw happens exactly as before and produces the same value for the same seed.
- When an override is passed, the override is used, same as before.

### Effect on detection

- No change to the regenerated reference image, for the same reasons.
- No change to any score, threshold, or verdict.

### One behaviour detail worth knowing

- In the old code, the random draw always ran, even when an override was supplied and the drawn value was immediately thrown away.
- In the new code, the draw is skipped when an override is supplied, because a ternary only evaluates the branch it picks.
- This is safe here. `grating_rng` is used three times in each function and the `mod_depth` line is the last of the three. Nothing after that line reads from `grating_rng`, so skipping the draw cannot shift any other value.
- The other random fields are unaffected regardless. `generate_prng_macro_pattern` builds its own generator from `seed`, and `add_rgb_perturbations` builds its own from `seed + 1000`. Neither shares state with `grating_rng`.
- Both functions were changed identically, so generation and detection consume the generator the same way as each other.

## Change 2: Bitwise operators for correlation scale steps

### Where

- cdp_engine.py, function `test_prng_correlation` (around line 1989) - the detection path.

### Before

```python
if CDP_FLAG_CORR_FINE_BLOCKS:
    scale_configs = [
        (block_size // 4, 0.45, 1.0),
        (block_size // 2, 0.55, 1.0),
        (block_size,      0.86, 1.0 if not CDP_FLAG_CORR_COARSE_CAP else 1.0),
        (block_size * 2,  0.65, 0.50 if CDP_FLAG_CORR_COARSE_CAP else 0.50),
    ]
else:
    scale_configs = [
        (block_size,     0.86, 1.0  if not CDP_FLAG_CORR_COARSE_CAP else 0.75),
        (block_size * 2, 0.65, 0.75 if not CDP_FLAG_CORR_COARSE_CAP else 0.50),
        (block_size * 4, 0.75, 0.50),
    ]
```

### After

```python
if CDP_FLAG_CORR_FINE_BLOCKS:
    scale_configs = [
        (block_size >> 2, 0.45, 1.0),
        (block_size >> 1, 0.55, 1.0),
        (block_size,      0.86, 1.0 if not CDP_FLAG_CORR_COARSE_CAP else 1.0),
        (block_size << 1, 0.65, 0.50 if CDP_FLAG_CORR_COARSE_CAP else 0.50),
    ]
else:
    scale_configs = [
        (block_size,      0.86, 1.0  if not CDP_FLAG_CORR_COARSE_CAP else 0.75),
        (block_size << 1, 0.65, 0.75 if not CDP_FLAG_CORR_COARSE_CAP else 0.50),
        (block_size << 2, 0.75, 0.50),
    ]
```

A short comment was added above the block explaining the shift equivalence.

### Why this spot matters

- `test_prng_correlation` is one of the four tests that feed the final weighted score, alongside moire, color, and gradient.
- The final score is compared against `THRESHOLD_AUTHENTIC` and `THRESHOLD_SUSPICIOUS` to produce the AUTHENTIC, SUSPICIOUS, or COUNTERFEIT verdict.
- These specific numbers are the block sizes the test walks through. The test starts at the finest scale and steps coarser until it finds a scale with usable signal. That walk is the core copy-detection idea in this function: a real print keeps fine detail, a copy of a print loses it.

### Why the shifts are exact

- Shifting by a fixed amount is the same as multiplying or dividing by a power of two.
- For any non-negative integer x: `x >> 2` equals `x // 4`, `x >> 1` equals `x // 2`, `x << 1` equals `x * 2`, and `x << 2` equals `x * 4`.
- This does not require `block_size` itself to be a power of two. The power of two is the divisor or multiplier, not the block size. So a `block_size` of 12 or 20 behaves the same before and after.
- `block_size` is always a positive pixel count in this codebase, so the non-negative condition always holds.

### Effect on generation

- None. This function is not used on the generation path.

### Effect on detection

- No change to any computed scale, correlation value, normalized score, or verdict.
- The values fed into `scale_configs` are identical to before for every block size the system uses.

## Verification done

Both changes were traced end to end through every caller in the tracked code.

On the ternary:

- Only two copies of the mod_depth logic exist in the live code, and both were changed. There is no third copy in app.py or anywhere else.
- No caller anywhere passes mod_depth_override. The generate call in app.py omits it, and both regenerate_reference call sites omit it.
- So in the live flow the override is always None, the else branch is always taken, and the random draw runs exactly as it did before. Generated patterns and regenerated references are unchanged.
- The skipped-draw case can only happen if someone passes an override directly. Even then it is safe: the mod_depth draw is the last use of grating_rng in both functions, and nothing reads that generator afterwards.
- The other random layers are independent. generate_prng_macro_pattern builds its own generator from seed, and add_rgb_perturbations builds its own from seed plus 1000. Neither shares state with grating_rng.

On the bitwise change:

- The shift and arithmetic forms were compared directly for block sizes 4, 8, 12, 16, 20, 24, 32, and 64. Every pair matched.
- Both sides of the CDP_FLAG_CORR_FINE_BLOCKS branch were changed consistently. The production default for that flag is False, so the else branch is the one that runs.
- No normalization divisor and no score cap was touched. One line had its spacing realigned and nothing else.
- The only case where shifts differ from the old arithmetic is a non-integer block size, because a shift rejects a float while floor division accepts it. That case was already broken before this change: a float block size makes the block counts floats, and both range and np.zeros reject floats a few lines later. So the behaviour is not a regression, it just fails earlier and with a clearer message.
- Where block size comes from was checked. The modern verify path casts it with int(). The legacy path passes the stored value through uncast, but the value is written through a model that types it as int, so it comes back as an int.

Deliberately not changed:

- `_corr_raw` has the same scale ladder but is marked diagnostic use only and has no callers in the tracked code. It was left on the arithmetic form on purpose, not missed.

## Verification not done

- The full generate and verify pipeline was not executed. numpy and opencv are not installed in this environment, and installing them was skipped.
- Worth doing before this reaches anything important: generate a pattern from a fixed seed on the previous commit and on this commit and confirm the two images are byte identical, then run one verification end to end and confirm the verdict and confidence are unchanged.
