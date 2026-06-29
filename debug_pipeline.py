"""
Visual debugger for the ALIGN-THEN-CROP fiducial pipeline.

Usage:  python debug_pipeline.py <image_path> [out_dir]

Runs the real flow on a capture and writes annotated stage images:
  dbg_1_full_annotated.png  - full photo with: DM boxes (cyan), DM-derived pattern
                              quad (blue), predicted marker spots (red X), search
                              windows (yellow), candidate blobs (green=pass /
                              orange=reject), final detected markers (green ring)
  dbg_2_aligned_crop.png    - single-homography aligned+cropped pattern (output)
  dbg_3_reference.png       - regenerated reference (ground truth)

Look at dbg_1 to see whether the 4 markers are being detected and where.
"""
import sys, os, io, contextlib
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp_engine import (extract_seed_from_image, detect_and_crop_pattern, regenerate_reference,
                        align_crop_from_quad, _blank_fiducials, PATTERN_SIZE, BLOCK_SIZE)
try:
    from database import init_db, get_db
    init_db(); _db = get_db()
except Exception:
    _db = None


def _ncc(a, b):
    a = _blank_fiducials(a).astype(np.float64).ravel(); b = _blank_fiducials(b).astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"cannot read {path}"); return
    H_img, W_img = bgr.shape[:2]
    print(f"image {path}  shape={bgr.shape}")

    with contextlib.redirect_stdout(io.StringIO()):
        seed, diag = extract_seed_from_image(bgr)
    if seed is None:
        print(f"DM decode FAILED: {diag.get('failure_reason')}"); return
    bs = BLOCK_SIZE
    if _db is not None:
        p = _db.patterns.find_one({"seed": seed})
        if p: bs = int(p.get("block_size", BLOCK_SIZE))
    print(f"seed={seed}  block_size={bs}  DMs_found={diag.get('num_dms_found')}  shrink={diag.get('shrink_used')}")

    with contextlib.redirect_stdout(io.StringIO()):
        cr = detect_and_crop_pattern(bgr, dm_results_raw=diag.get("raw_results"),
                                     dm_shrink_used=diag.get("shrink_used", 1), seed=seed, block_size=bs)
    if not cr[2] or cr[3] is None:
        print("DM rough-locate FAILED — cannot predict marker positions"); return

    dbg = {}
    with contextlib.redirect_stdout(io.StringIO()):
        crop, ok = align_crop_from_quad(bgr, cr[3], debug=dbg)

    # ---- annotate the full photo ----
    ann = bgr.copy()
    t = max(2, W_img // 600)          # line thickness scaled to image size
    R = max(6, W_img // 200)          # marker glyph size

    # DM boxes (cyan) — pylibdmtx rect is y-up; convert to y-down
    for r in diag.get("raw_results", []):
        rr = r.rect; wf, hf = abs(rr.width), abs(rr.height)
        left = rr.left - (wf if rr.width < 0 else 0)
        top_cv = H_img - (rr.top - (hf if rr.height < 0 else 0)) - hf
        cv2.rectangle(ann, (int(left), int(top_cv)), (int(left + wf), int(top_cv + hf)), (255, 200, 0), t)

    # pattern quad (blue)
    quad = np.int32(dbg["quad"]).reshape(-1, 1, 2)
    cv2.polylines(ann, [quad], True, (255, 0, 0), t)

    labels = ["TL", "TR", "BR", "BL"]
    for i, (px, py, win, cands, m) in enumerate(dbg["search"]):
        cv2.rectangle(ann, (int(px - win), int(py - win)), (int(px + win), int(py + win)), (0, 220, 220), t)
        cv2.drawMarker(ann, (int(px), int(py)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, R, t)   # predicted (red X)
        for (gx, gy, a, cval, rval, passed) in cands:
            cv2.circle(ann, (int(gx), int(gy)), max(3, R // 3), (0, 255, 0) if passed else (0, 150, 255), -1)
        if m is not None:
            cv2.circle(ann, (int(m[0]), int(m[1])), R, (0, 255, 0), t)            # detected (green ring)
        cv2.putText(ann, labels[i], (int(px) + R, int(py) - R), cv2.FONT_HERSHEY_SIMPLEX, R / 20.0, (255, 255, 255), t)

    cv2.putText(ann, f"seed={seed} markers={dbg.get('n_found',0)}/4 stage={dbg.get('stage')}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, W_img / 2500), (255, 255, 255), t)

    # downscale for viewing
    disp_w = 1500
    if W_img > disp_w:
        sc = disp_w / W_img
        ann = cv2.resize(ann, (disp_w, int(H_img * sc)))
    cv2.imwrite(os.path.join(out_dir, "dbg_1_full_annotated.png"), ann)

    with contextlib.redirect_stdout(io.StringIO()):
        ref_rgb, ref_gray = regenerate_reference(seed, block_size=bs)
    cv2.imwrite(os.path.join(out_dir, "dbg_3_reference.png"), cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR))

    print(f"\nmarkers detected: {dbg.get('n_found',0)}/4   stage={dbg.get('stage')}   aligned_ok={ok}")
    for i, (px, py, win, cands, m) in enumerate(dbg["search"]):
        passed = sum(1 for c in cands if c[5])
        sel = None if m is None else (round(m[0], 1), round(m[1], 1))
        print(f"  {labels[i]}: predicted=({px:.0f},{py:.0f}) candidates={len(cands)} passed={passed} selected={sel}")
    if ok:
        cv2.imwrite(os.path.join(out_dir, "dbg_2_aligned_crop.png"), crop)
        print(f"  NCC(aligned, reference) = {_ncc(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), ref_gray):.4f}")
    print(f"\nimages in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python debug_pipeline.py <image_path> [out_dir]"); sys.exit(1)
    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("TEMP", "."), "cdp_debug")
    main(img, out)
