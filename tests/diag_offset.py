"""
Task 4 diagnostic: expose the RAW Step 2b dst points and RAW fine-align
detected fiducial positions for 5 samples, side by side.

Goal: show whether the offset between canonical and detected positions is
fixed (same pixel value across all samples) or varies with something.
"""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import cdp_engine as eng
import database as db

db.init_db()

BASE = os.path.join(os.path.dirname(__file__),
                    "batch_test", "10 test cases")
SAMPLES = [
    ("pre fine tune test", "tc-01.jpeg"),
    ("pre fine tune test", "tc-03.jpeg"),
    ("pre fine tune test", "tc-09.jpeg"),
    ("fine tune test 3",   "tc-01.jpeg"),
    ("fine tune test 3",   "tc-09.jpeg"),
]

# ── Patch 1: intercept align_crop_by_dm_corners to capture dst points ───────
_captured_dst  = {}   # sample_key → np.float32 (8,2)
_captured_src  = {}   # sample_key → np.float32 (8,2)

_real_acbdc = eng.align_crop_by_dm_corners

def _patched_acbdc(image, raw_results, debug=None):
    # We need to capture src/dst inside this call.
    # Do it by patching findHomography temporarily.
    import cv2 as _cv2
    _real_fh = _cv2.findHomography
    def _capture_fh(src, dst, *a, **kw):
        if _patched_acbdc._key:
            _captured_dst[_patched_acbdc._key] = dst.copy()
            _captured_src[_patched_acbdc._key] = src.copy()
        return _real_fh(src, dst, *a, **kw)
    _cv2.findHomography = _capture_fh
    try:
        result = _real_acbdc(image, raw_results, debug=debug)
    finally:
        _cv2.findHomography = _real_fh
    return result

_patched_acbdc._key = None
eng.align_crop_by_dm_corners = _patched_acbdc

# ── Patch 2: intercept fine_align to capture per-fiducial detections ─────────
_captured_fa = {}   # sample_key → debug dict

_real_fa = eng.fine_align_via_fiducials
def _patched_fa(crop_512, debug=None):
    _dbg = {}
    result = _real_fa(crop_512, debug=_dbg)
    if _patched_fa._key:
        _captured_fa[_patched_fa._key] = _dbg
    if debug is not None:
        debug.update(_dbg)
    return result

_patched_fa._key = None
eng.fine_align_via_fiducials = _patched_fa

# ── Run samples ───────────────────────────────────────────────────────────────
UPLOADS = os.path.join(os.path.dirname(__file__), "diag_offset_uploads")
os.makedirs(UPLOADS, exist_ok=True)

for folder, tc in SAMPLES:
    img_path = os.path.join(BASE, folder, tc)
    key = f"{folder[:6]}/{tc}"
    _patched_acbdc._key = key
    _patched_fa._key    = key

    buf = io.StringIO()
    old = sys.stdout; sys.stdout = buf
    try:
        r = eng.verify_pattern(str(img_path), uploads_dir=UPLOADS)
    except Exception as e:
        sys.stdout = old
        print(f"[ERROR] {key}: {e}")
        continue
    finally:
        sys.stdout = old

    _patched_acbdc._key = None
    _patched_fa._key    = None

    dst = _captured_dst.get(key)
    src = _captured_src.get(key)
    fa  = _captured_fa.get(key, {})

    print(f"\n{'='*70}")
    print(f"SAMPLE: {key}  >>  {r['verdict']} {r['confidence']:.3f}")
    print(f"  fine_align_applied={r['fine_align_applied']}  "
          f"dx={r['fine_align_dx']:+.1f}  dy={r['fine_align_dy']:+.1f}")

    # ── Step 2b dst points ────────────────────────────────────────────────
    if dst is not None:
        slots = ["top-TL","top-TR","top-BR","top-BL",
                 "rgt-TL","rgt-TR","rgt-BR","rgt-BL"]
        print(f"\n  Step 2b dst points (8 canonical corners, output-space px):")
        for i, (d, s) in enumerate(zip(dst, src)):
            print(f"    [{slots[i]:8s}]  dst=({d[0]:8.3f}, {d[1]:8.3f})  "
                  f"src=({s[0]:7.1f}, {s[1]:7.1f})")
    else:
        print("  [dst not captured — Step 2b may have failed]")

    # ── Fine-align per-fiducial detections ────────────────────────────────
    accepted = fa.get('fa_accepted', [])
    rejected = fa.get('fa_rejected', [])
    dx_fa = fa.get('fa_dx', 0.0)
    dy_fa = fa.get('fa_dy', 0.0)

    print(f"\n  Fine-align fiducials (canonical >> detected >> offset):")
    print(f"    Slot  Canon      Detected   Offset  Score")
    for e in accepted:
        cx, cy = e['canon']
        dx_e, dy_e = e['offset']
        dx_e2, dy_e2 = e['detected']
        print(f"    {e['label']:2s}  ({cx:3d},{cy:3d})  "
              f"({dx_e2:6.1f},{dy_e2:6.1f})  "
              f"({dx_e:+5.1f},{dy_e:+5.1f})  {e['score']:.3f}  ACCEPTED")
    for e in rejected:
        print(f"    {e['label']:2s}  [rejected — {e.get('reason','?')}  score={e.get('score',0):.3f}]")

    print(f"  → median correction: dx={dx_fa:+.1f}  dy={dy_fa:+.1f}")

    # ── Key question: is the offset FIXED or does it scale? ───────────────
    if accepted:
        offsets = [(e['label'], e['offset'][0], e['offset'][1]) for e in accepted]
        min_dx = min(o[1] for o in offsets)
        max_dx = max(o[1] for o in offsets)
        min_dy = min(o[2] for o in offsets)
        max_dy = max(o[2] for o in offsets)
        print(f"  Range dx: [{min_dx:+.1f}, {max_dx:+.1f}]  "
              f"dy: [{min_dy:+.1f}, {max_dy:+.1f}]  "
              f"(narrow = fixed offset; wide = varies by corner)")

print("\nDone.")
