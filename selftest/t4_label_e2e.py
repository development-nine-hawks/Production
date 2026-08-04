"""Layer 4 — synthetic label through verify_pattern, plus v0 regression.

=============================================================================
THIS IS A REGRESSION TRIPWIRE, NOT A DISCRIMINATION TEST.
=============================================================================
A synthetic label has no print channel and no capture channel, so it scores
near 1.0 by construction. It proves the pipeline is WIRED — payload -> raster ->
layout -> decode -> classify -> crop -> regenerate -> score. It says nothing
about the genuine/counterfeit margin, which verify_pattern's own docstring
records as negative (-0.013) and FRAGILE for the moire component.

Perturbations gate seed recovery only, never the score.
=============================================================================
"""
import os

import cv2
import numpy as np

from _harness import (check, info, need, import_engine, run_module, finish,
                      OUT_DIR)

import _v0_codec as v0

eng, _ = import_engine(require_dm_libs=True)

generate_pattern = need(eng, "generate_pattern")
split            = need(eng, "split_seed_for_dm")
split_v0         = need(eng, "split_seed_for_dm_legacy")
recombine        = need(eng, "recombine_seed_from_dm")
gen_dm           = need(eng, "generate_cropped_dm")
layout_fn        = need(eng, "calculate_auth_block_layout")
draw_block       = need(eng, "draw_auth_block_opencv")
verify_pattern   = need(eng, "verify_pattern")

PATTERN_UNIT, QUIET_UNIT, FRAME_PAD = 512, 34, 90
UPLOADS  = os.path.join(OUT_DIR, "uploads")
PATTERNS = os.path.join(OUT_DIR, "patterns")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(PATTERNS, exist_ok=True)


def build_label(seed, encoder=None):
    """Playwright-free reproduction of the app.py label path."""
    res = generate_pattern(PATTERNS, seed=seed, block_size=16,
                           pattern_size=PATTERN_UNIT)
    pattern = cv2.imread(res["filepath"])
    a, b = (encoder or split)(res["seed"])
    top_img, top_mods     = gen_dm(a, size="8x18")
    right_img, right_mods = gen_dm(b, size="8x18")
    lay = layout_fn(PATTERN_UNIT, QUIET_UNIT, top_img, right_img,
                    top_dm_modules=top_mods, right_dm_modules=right_mods)
    canvas = np.full((int(lay["auth_h"]), int(lay["auth_w"]), 3), 255, np.uint8)
    draw_block(canvas, lay, pattern, top_img, right_img)
    h, w = canvas.shape[:2]
    framed = np.full((h + 2 * FRAME_PAD, w + 2 * FRAME_PAD, 3), 255, np.uint8)
    framed[FRAME_PAD:FRAME_PAD + h, FRAME_PAD:FRAME_PAD + w] = canvas
    return cv2.resize(framed, None, fx=3, fy=3,
                      interpolation=cv2.INTER_NEAREST), res["seed"], (a, b)


def _verify(img, name):
    path = os.path.join(UPLOADS, name)
    cv2.imwrite(path, img)
    return verify_pattern(path, uploads_dir=UPLOADS)


def test_clean_label_verifies_authentic():
    img, seed, shares = build_label(0x1234)
    info(f"seed 0x{seed:04X} -> codes {shares}")
    r = _verify(img, "clean.png")
    check(r.get("verdict") != "UNABLE_TO_VERIFY",
          "the pipeline decodes and crops a synthetic matrix-format label",
          f"verdict={r.get('verdict')} "
          f"{r.get('dm_diagnostic', {}).get('failure_reason')}")
    check(r.get("seed_recovered") == seed,
          f"seed round-trips through the full pipeline (0x{seed:04X})",
          f"recovered {r.get('seed_recovered')}")
    info(f"verdict={r.get('verdict')} confidence={r.get('confidence', 0):.4f}")
    check(r.get("verdict") == "AUTHENTIC",
          f"a label verifies against its own regenerated reference "
          f"(confidence {r.get('confidence', 0):.3f})",
          "the matrix payload -> pattern -> reference chain is broken")


def test_geometry_survives_perturbation():
    img, seed, _ = build_label(0xBEEF)
    h, w = img.shape[:2]

    def rot(d):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), d, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    def persp(px):
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[px, px * .6], [w - px * .4, 0],
                          [w - px, h - px * .5], [px * .5, h]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                                   (w, h), borderValue=(255, 255, 255))

    def jpeg(q):
        return cv2.imdecode(cv2.imencode(".jpg", img,
                            [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)

    cases = [
        ("rot_90",      lambda: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("rot_180",     lambda: cv2.rotate(img, cv2.ROTATE_180)),
        ("rot_270",     lambda: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("rot_3deg",    lambda: rot(3)),
        ("rot_-10deg",  lambda: rot(-10)),
        ("perspective", lambda: persp(w * 0.05)),
        ("jpeg_q75",    lambda: jpeg(75)),
        ("jpeg_q60",    lambda: jpeg(60)),
        ("blur_s1",     lambda: cv2.GaussianBlur(img, (0, 0), 1.0)),
        ("bright_x0.6", lambda: np.clip(img * 0.6, 0, 255).astype(np.uint8)),
        ("bright_x1.4", lambda: np.clip(img * 1.4, 0, 255).astype(np.uint8)),
        ("downscale.5", lambda: cv2.resize(img, None, fx=.5, fy=.5,
                                           interpolation=cv2.INTER_AREA)),
    ]
    ok, lost = [], []
    for name, fn in cases:
        try:
            r = _verify(fn(), f"perturb_{name}.png")
            (ok if r.get("seed_recovered") == seed else lost).append(name)
        except Exception as e:
            lost.append(f"{name}({str(e)[:60]})")
    info(f"recovered: {ok}")
    if lost:
        info(f"LOST: {lost}")
    check(len(ok) >= len(cases) - 2,
          f"seed recovered under {len(ok)}/{len(cases)} geometric and "
          f"compression perturbations", f"lost on {lost}")


def test_v0_payloads_decode_correctly_not_wrongly():
    """The disaster the version tag prevents, measured.

    Without it the new decoder still accepts an old payload (same check
    character over the same 7 characters) and then reads the bits as two
    matrices, multiplying them to a wrong seed — COUNTERFEIT on a genuine
    label, for 100% of pre-cutover field stock.
    """
    import random
    r = random.Random(80)
    n = 50_000
    wrong = rejected = correct = 0
    for _ in range(n):
        s = r.getrandbits(32)
        got = recombine(*v0.split_seed_for_dm(s))
        if got is None:
            rejected += 1
        elif got == s:
            correct += 1
        else:
            wrong += 1
    info(f"{n:,} v0 payloads: correct={correct:,} rejected={rejected} "
         f"MISDECODED={wrong}")
    check(wrong == 0, "no v0 payload is silently misdecoded to a wrong seed",
          f"{wrong} would report COUNTERFEIT on genuine labels")
    check(correct == n, f"all {n:,} v0 payloads decode to their original seed",
          f"{rejected} rejected — field stock would stop verifying")

    # counterfactual: what a version-less decoder would have done
    m_mul, m_unpack, m_pack = (need(eng, "_m4_mul"), need(eng, "_m4_unpack"),
                               need(eng, "_m4_pack"))
    b32d = need(eng, "_b32_decode")
    r = random.Random(81)
    bad = 0
    for _ in range(20_000):
        s = r.getrandbits(32)
        a, b = v0.split_seed_for_dm(s)
        f = b32d((a + b)[:7])
        as_matrix = m_pack(m_mul(m_unpack((f >> 16) & 0xFFFF),
                                 m_unpack(f & 0xFFFF)))
        if as_matrix != s:
            bad += 1
    info(f"without the version tag, {bad:,}/20,000 v0 labels would decode to a "
         f"WRONG seed")
    check(bad > 19_900, "confirmed: the version tag is load-bearing, not "
                        "decoration", f"only {bad}/20,000")


def test_v0_label_still_verifies_end_to_end():
    """A complete pre-cutover label must still come back AUTHENTIC."""
    img, seed, shares = build_label(0x00ABCDEF, encoder=split_v0)
    info(f"v0 label: codes {shares} (leading '{shares[0][0]}' marks it legacy)")
    r = _verify(img, "v0_label.png")
    check(r.get("verdict") != "UNABLE_TO_VERIFY", "a pre-cutover label decodes",
          f"{r.get('verdict')} "
          f"{r.get('dm_diagnostic', {}).get('failure_reason')}")
    check(r.get("seed_recovered") == seed,
          "the legacy seed is recovered via the v0 path",
          f"got {r.get('seed_recovered')}, expected {seed}")
    check(r.get("verdict") == "AUTHENTIC",
          f"a pre-cutover label verifies AUTHENTIC "
          f"(confidence {r.get('confidence', 0):.3f})",
          "field stock printed before this branch would be rejected")


def test_v0_and_v2_coexist():
    v2img, v2seed, _ = build_label(0x2222)
    v0img, v0seed, _ = build_label(0x00334444, encoder=split_v0)
    r2 = _verify(v2img, "coexist_v2.png")
    r0 = _verify(v0img, "coexist_v0.png")
    r2b = _verify(v2img, "coexist_v2_again.png")
    check(r2.get("seed_recovered") == v2seed and r2.get("verdict") == "AUTHENTIC",
          "matrix-format label verifies")
    check(r0.get("seed_recovered") == v0seed and r0.get("verdict") == "AUTHENTIC",
          "legacy label verifies in the same process")
    check(r2b.get("seed_recovered") == v2seed,
          "verifying a v0 label in between does not disturb v2 decoding",
          "decoder state is leaking between calls")


if __name__ == "__main__":
    run_module(globals(), "LAYER 4 — synthetic label e2e + v0 regression")
    finish()
