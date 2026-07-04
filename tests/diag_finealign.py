"""
Step 2c fine-alignment diagnostic.

Captures the Step 2b crop (pre-fine-align) and the Step 2c crop
(post-fine-align) side-by-side for 50 samples across 4 phone sets.
'fine tune test 1' is empty and is skipped.
"""

import sys, os, io, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import cdp_engine as eng
import database as db

# ── Init ────────────────────────────────────────────────────────────────────
db.init_db()

BASE = os.path.join(
    os.path.dirname(__file__),
    "batch_test", "10 test cases"
)
BASE_ROOT = os.path.join(os.path.dirname(__file__), "batch_test")

FOLDERS = [
    ("pre_ft",  "pre fine tune test"),
    ("ft2",     "fine tune test 2"),
    ("ft3",     "fine tune test 3"),
    ("ft4",     "fine tune test 4"),
]
TCS = [f"tc-{i:02d}.jpeg" for i in range(1, 11)]

# Root-level singles (batch_test/tc-07.jpeg, tc-09.jpeg) — the regression samples
ROOT_SINGLES = [("root", "tc-07.jpeg"), ("root", "tc-09.jpeg")]

OUT_DIR = os.path.join(os.path.dirname(__file__), "diag_finealign_out")
os.makedirs(OUT_DIR, exist_ok=True)

UPLOADS_DIR = os.path.join(OUT_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── Monkey-patch fine_align_via_fiducials to capture pre-align image ────────
_pre_align_capture = {}   # key: tc label string → ndarray

_real_fine_align = eng.fine_align_via_fiducials

def _patched_fine_align(crop_512, debug=None):
    # Save the input (Step 2b output, pre-fine-align) into our capture slot.
    # The key is written by the outer loop just before verify_pattern is called.
    if _patched_fine_align._current_key:
        _pre_align_capture[_patched_fine_align._current_key] = crop_512.copy()
    return _real_fine_align(crop_512, debug=debug)

_patched_fine_align._current_key = None
eng.fine_align_via_fiducials = _patched_fine_align

# ── Run ─────────────────────────────────────────────────────────────────────
rows = []

for set_key, folder_name in FOLDERS:
    folder_path = os.path.join(BASE, folder_name)
    for tc in TCS:
        img_path = os.path.join(folder_path, tc)
        if not os.path.exists(img_path):
            print(f"[SKIP] {set_key}/{tc} — file not found")
            rows.append({
                "set": set_key, "tc": tc,
                "n_fid": "–", "applied": "–", "dx": "–", "dy": "–",
                "verdict": "FILE_NOT_FOUND", "conf": "–",
                "moire": "–", "corr": "–",
                "note": "missing",
            })
            continue

        sample_key = f"{set_key}/{tc}"
        _patched_fine_align._current_key = sample_key
        _pre_align_capture.pop(sample_key, None)

        # Suppress verbose engine stdout to keep output clean
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            r = eng.verify_pattern(str(img_path), uploads_dir=UPLOADS_DIR)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"[ERROR] {sample_key}: {e}")
            rows.append({
                "set": set_key, "tc": tc,
                "n_fid": "–", "applied": "–", "dx": "–", "dy": "–",
                "verdict": "ERROR", "conf": "–",
                "moire": "–", "corr": "–",
                "note": str(e)[:80],
            })
            continue
        finally:
            sys.stdout = old_stdout

        _patched_fine_align._current_key = None

        # ── Collect metrics ──────────────────────────────────────────────────
        n_fid   = r.get("fine_align_n",  0)
        applied = r.get("fine_align_applied", False)
        dx      = r.get("fine_align_dx", 0.0)
        dy      = r.get("fine_align_dy", 0.0)
        verdict = r.get("verdict", "?")
        conf    = r.get("confidence", 0.0)
        scores  = r.get("scores", {})
        moire   = scores.get("moire",       None)
        corr    = scores.get("correlation", None)

        large = applied and (abs(dx) > 15 or abs(dy) > 15)

        rows.append({
            "set": set_key, "tc": tc,
            "n_fid": n_fid, "applied": "Y" if applied else "N",
            "dx": round(dx, 1), "dy": round(dy, 1),
            "verdict": verdict, "conf": round(conf, 3),
            "moire": round(moire, 4) if moire is not None else "–",
            "corr":  round(corr,  4) if corr  is not None else "–",
            "note": "LARGE_CORRECTION" if large else "",
        })

        # ── Build side-by-side image ─────────────────────────────────────────
        pre_img  = _pre_align_capture.get(sample_key)  # Step 2b
        post_path = os.path.join(UPLOADS_DIR, r.get("aligned_filename", ""))
        post_img  = cv2.imread(post_path) if r.get("aligned_filename") else None

        gap = np.full((512, 20, 3), 200, dtype=np.uint8)  # light-grey separator

        panels = []
        label_h = 30

        def labeled(img, text):
            """Add a text label bar above a 512×512 BGR image."""
            bar = np.full((label_h, 512, 3), 40, dtype=np.uint8)
            cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
            return np.vstack([bar, img])

        def placeholder(text):
            p = np.full((512, 512, 3), 80, dtype=np.uint8)
            cv2.putText(p, text, (40, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 1, cv2.LINE_AA)
            return p

        pre_panel  = labeled(pre_img  if pre_img  is not None else placeholder("no pre-align img"),
                             f"Step2b  {set_key}/{tc}")
        post_panel = labeled(post_img if post_img is not None else placeholder("no post-align img"),
                             f"Step2c  dx={dx:+.1f} dy={dy:+.1f}  applied={'Y' if applied else 'N'}  "
                             f"n_fid={n_fid}  {verdict} {conf:.3f}")

        bar_with_gap = np.vstack([np.full((label_h, 20, 3), 40, dtype=np.uint8), gap])
        side_by_side = np.hstack([pre_panel, bar_with_gap, post_panel])

        out_name = f"{set_key}_{tc.replace('.jpeg','')}_s2b_vs_s2c.png"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), side_by_side)

        flag = " *** LARGE CORRECTION" if large else ""
        print(f"[OK] {sample_key:22s}  fid={n_fid}/4  app={applied!s:5}  "
              f"dx={dx:+6.1f}  dy={dy:+6.1f}  {verdict:<12} {conf:.3f}{flag}")

# ── Root-level singles (regression samples) ─────────────────────────────────
print()
print("=== ROOT-LEVEL REGRESSION SAMPLES ===")
for set_key, tc_file in ROOT_SINGLES:
    img_path = os.path.join(BASE_ROOT, tc_file)
    if not os.path.exists(img_path):
        print(f"[SKIP] {set_key}/{tc_file} — file not found at {img_path}")
        rows.append({"set": set_key, "tc": tc_file,
                     "n_fid": "–", "applied": "–", "dx": "–", "dy": "–",
                     "verdict": "FILE_NOT_FOUND", "conf": "–",
                     "moire": "–", "corr": "–", "note": "missing"})
        continue

    sample_key = f"{set_key}/{tc_file}"
    _patched_fine_align._current_key = sample_key
    _pre_align_capture.pop(sample_key, None)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        r = eng.verify_pattern(str(img_path), uploads_dir=UPLOADS_DIR)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"[ERROR] {sample_key}: {e}")
        rows.append({"set": set_key, "tc": tc_file,
                     "n_fid": "–", "applied": "–", "dx": "–", "dy": "–",
                     "verdict": "ERROR", "conf": "–",
                     "moire": "–", "corr": "–", "note": str(e)[:80]})
        continue
    finally:
        sys.stdout = old_stdout

    _patched_fine_align._current_key = None

    n_fid   = r.get("fine_align_n",  0)
    applied = r.get("fine_align_applied", False)
    dx      = r.get("fine_align_dx", 0.0)
    dy      = r.get("fine_align_dy", 0.0)
    verdict = r.get("verdict", "?")
    conf    = r.get("confidence", 0.0)
    scores  = r.get("scores", {})
    moire   = scores.get("moire", None)
    corr    = scores.get("correlation", None)

    rows.append({
        "set": set_key, "tc": tc_file,
        "n_fid": n_fid, "applied": "Y" if applied else "N",
        "dx": round(dx, 1), "dy": round(dy, 1),
        "verdict": verdict, "conf": round(conf, 3),
        "moire": round(moire, 4) if moire is not None else "–",
        "corr":  round(corr,  4) if corr  is not None else "–",
        "note": "",
    })

    # Side-by-side for the regression samples
    pre_img   = _pre_align_capture.get(sample_key)
    post_path = os.path.join(UPLOADS_DIR, r.get("aligned_filename", ""))
    post_img  = cv2.imread(post_path) if r.get("aligned_filename") else None

    label_h = 30
    def labeled(img, text):
        bar = np.full((label_h, 512, 3), 40, dtype=np.uint8)
        cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return np.vstack([bar, img])
    def placeholder(text):
        p = np.full((512, 512, 3), 80, dtype=np.uint8)
        cv2.putText(p, text, (40, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return p

    pre_panel  = labeled(pre_img  if pre_img  is not None else placeholder("no pre img"),
                         f"Rough-H  {set_key}/{tc_file}")
    post_panel = labeled(post_img if post_img is not None else placeholder("no post img"),
                         f"Final  applied={'Y' if applied else 'N'}  fid={n_fid}  "
                         f"{verdict} {conf:.3f}")
    gap     = np.full((512, 20, 3), 200, dtype=np.uint8)
    gap_lab = np.full((label_h, 20, 3), 40, dtype=np.uint8)
    sbs     = np.hstack([pre_panel, np.vstack([gap_lab, gap]), post_panel])
    out_name = f"{set_key}_{tc_file.replace('.jpeg','')}_regression.png"
    cv2.imwrite(os.path.join(OUT_DIR, out_name), sbs)

    print(f"[OK] {sample_key:22s}  fid={n_fid}/4  app={applied!s:5}  "
          f"dx={dx:+6.1f}  dy={dy:+6.1f}  {verdict:<12} {conf:.3f}")
    print(f"     side-by-side saved: {out_name}")

# ── Summary table ────────────────────────────────────────────────────────────
print()
print("=" * 105)
print(f"{'SET':<10} {'TC':<12} {'n_fid':>5} {'app':>4} {'dx':>7} {'dy':>7}  "
      f"{'VERDICT':<12} {'CONF':>6}  {'MOIRE':>7}  {'CORR':>7}  NOTE")
print("=" * 105)
for r in rows:
    print(f"{r['set']:<10} {r['tc']:<12} {str(r['n_fid']):>5} {str(r['applied']):>4} "
          f"{str(r['dx']):>7} {str(r['dy']):>7}  {r['verdict']:<12} {str(r['conf']):>6}  "
          f"{str(r['moire']):>7}  {str(r['corr']):>7}  {r['note']}")
print("=" * 105)

# ── Large-correction summary ────────────────────────────────────────────────
large_rows = [r for r in rows if r.get("note") == "LARGE_CORRECTION"]
print(f"\nLarge corrections (|dx| or |dy| > 15px): {len(large_rows)}")
for r in large_rows:
    print(f"  {r['set']}/{r['tc']:12s}  dx={r['dx']:>6}  dy={r['dy']:>6}  "
          f"{r['verdict']} {r['conf']}")

print(f"\nSide-by-side images saved to: {OUT_DIR}")
print("Done.")
