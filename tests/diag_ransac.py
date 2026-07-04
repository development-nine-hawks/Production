"""
Diagnostic: log all points fed into the combined findHomography call for
batch_test/tc-07.jpeg and batch_test/tc-09.jpeg, show RANSAC inlier mask,
and compute per-point reprojection errors against the solved H.

Usage:
    python tests/diag_ransac.py
"""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import cdp_engine as eng
import database as db

db.init_db()

BASE  = os.path.join(os.path.dirname(__file__), "batch_test")
FILES = ["tc-07.jpeg", "tc-09.jpeg"]

UPLOADS = os.path.join(os.path.dirname(__file__), "diag_ransac_out")
os.makedirs(UPLOADS, exist_ok=True)

# ── Patch _refine_homography_with_fiducials to capture internals ─────────────
_captured = {}   # filename -> dict of internals

_real_refine = eng._refine_homography_with_fiducials

def _patched_refine(captured_bgr, H_rough, src_dm, dst_dm, rough_crop_512):
    """Wrapper that captures the full point set and RANSAC mask."""
    out_w, out_h = eng.PATTERN_SIZE

    # Re-run fiducial detection to get accepted list
    fa_debug = {}
    _, _ = eng.fine_align_via_fiducials(rough_crop_512, debug=fa_debug)
    n_fid    = fa_debug.get('fa_n_detected', 0)
    accepted = fa_debug.get('fa_accepted', [])

    slot_labels   = ['TL', 'TR', 'BR', 'BL']
    canon_centers = eng._fiducial_centers(out_w, out_h)
    canon_map     = dict(zip(slot_labels, canon_centers))

    H_rough_inv = np.linalg.inv(H_rough)
    src_fid, dst_fid, fid_labels = [], [], []
    for entry in accepted:
        lbl          = entry['label']
        det_x, det_y = entry['detected']
        cam_h        = H_rough_inv @ np.array([det_x, det_y, 1.0])
        if abs(cam_h[2]) < 1e-9:
            continue
        cam_pt = cam_h[:2] / cam_h[2]
        src_fid.append(cam_pt)
        dst_fid.append(list(canon_map[lbl]))
        fid_labels.append(lbl)

    src_combined = np.vstack([src_dm, np.float32(src_fid)]) if src_fid else src_dm.copy()
    dst_combined = np.vstack([dst_dm, np.float32(dst_fid)]) if dst_fid else dst_dm.copy()

    H_refined, mask = cv2.findHomography(src_combined, dst_combined, cv2.RANSAC, 5.0)

    # Compute reprojection errors
    reproj_errors = []
    if H_refined is not None:
        for i in range(len(src_combined)):
            p  = np.array([src_combined[i][0], src_combined[i][1], 1.0])
            ph = H_refined @ p
            ph /= ph[2]
            err = float(np.linalg.norm(ph[:2] - dst_combined[i]))
            reproj_errors.append(err)

    key = _patched_refine._current_key
    if key:
        _captured[key] = {
            'src_dm':      src_dm.copy(),
            'dst_dm':      dst_dm.copy(),
            'src_fid':     np.array(src_fid, dtype=np.float32) if src_fid else np.zeros((0,2), np.float32),
            'dst_fid':     np.array(dst_fid, dtype=np.float32) if dst_fid else np.zeros((0,2), np.float32),
            'fid_labels':  fid_labels,
            'fa_debug':    fa_debug,
            'n_fid':       n_fid,
            'H_rough':     H_rough.copy(),
            'H_refined':   H_refined,
            'mask':        mask.ravel().tolist() if mask is not None else [],
            'reproj_errors': reproj_errors,
            'accepted':    accepted,
        }

    # Delegate to real function so pipeline continues normally
    return _real_refine(captured_bgr, H_rough, src_dm, dst_dm, rough_crop_512)

_patched_refine._current_key = None
eng._refine_homography_with_fiducials = _patched_refine

# ── Run both samples ──────────────────────────────────────────────────────────
for fname in FILES:
    img_path = os.path.join(BASE, fname)
    if not os.path.exists(img_path):
        print(f"[SKIP] {img_path} not found")
        continue

    _patched_refine._current_key = fname

    buf = io.StringIO()
    old = sys.stdout; sys.stdout = buf
    try:
        r = eng.verify_pattern(str(img_path), uploads_dir=UPLOADS)
    except Exception as e:
        sys.stdout = old
        print(f"[ERROR] {fname}: {e}")
        continue
    finally:
        sys.stdout = old

    _patched_refine._current_key = None

    c = _captured.get(fname, {})
    if not c:
        print(f"\n[{fname}] refinement was not called (possibly fid_ok=False or <3 fids)")
        continue

    src_dm     = c['src_dm']
    dst_dm     = c['dst_dm']
    src_fid    = c['src_fid']
    dst_fid    = c['dst_fid']
    fid_labels = c['fid_labels']
    accepted   = c['accepted']
    mask       = c['mask']
    reproj_err = c['reproj_errors']
    H_refined  = c['H_refined']
    H_rough    = c['H_rough']

    # Get original image dimensions for bounds check
    img_bgr = cv2.imread(img_path)
    img_h, img_w = img_bgr.shape[:2]

    n_dm  = len(src_dm)
    n_fid = len(src_fid)
    total = n_dm + n_fid

    print(f"\n{'='*72}")
    print(f"SAMPLE: {fname}  verdict={r['verdict']}  conf={r['confidence']:.3f}")
    print(f"  Source image: {img_w}x{img_h}")
    print(f"  Combined set: {n_dm} DM-corner pts + {n_fid} fiducial pts = {total} total")
    print(f"  RANSAC inliers: {sum(mask)}/{total}")

    # ── DM-corner points ────────────────────────────────────────────────────
    dm_slots = ["top-TL","top-TR","top-BR","top-BL",
                "rgt-TL","rgt-TR","rgt-BR","rgt-BL"]
    print(f"\n  DM-corner points (indices 0-7):")
    print(f"    idx  slot      src(cam)            dst(crop)         reproj_err  inlier")
    for i in range(n_dm):
        sl   = dm_slots[i] if i < len(dm_slots) else f"dm-{i}"
        m    = mask[i] if i < len(mask) else "?"
        err  = reproj_err[i] if i < len(reproj_err) else float('nan')
        print(f"    {i:3d}  {sl:8s}  ({src_dm[i][0]:8.1f},{src_dm[i][1]:8.1f})  "
              f"({dst_dm[i][0]:7.2f},{dst_dm[i][1]:7.2f})  "
              f"{err:8.2f}px  {'IN' if m else 'OUT'}")

    # ── Fiducial points ─────────────────────────────────────────────────────
    print(f"\n  Fiducial points (indices {n_dm}-{total-1}):")
    print(f"    idx  label  detected(crop)    src_backproj(cam)         dst(canon)  reproj_err  inlier")
    for j in range(n_fid):
        i    = n_dm + j
        lbl  = fid_labels[j] if j < len(fid_labels) else "?"
        m    = mask[i] if i < len(mask) else "?"
        err  = reproj_err[i] if i < len(reproj_err) else float('nan')
        det  = accepted[j]['detected'] if j < len(accepted) else (0,0)
        print(f"    {i:3d}  {lbl:4s}  ({det[0]:6.1f},{det[1]:6.1f})     "
              f"({src_fid[j][0]:9.2f},{src_fid[j][1]:9.2f})  "
              f"({dst_fid[j][0]:6.1f},{dst_fid[j][1]:6.1f})  "
              f"{err:8.2f}px  {'IN' if m else 'OUT'}")

    # ── Reprojection summary ────────────────────────────────────────────────
    dm_errs  = reproj_err[:n_dm]
    fid_errs = reproj_err[n_dm:]
    if dm_errs:
        print(f"\n  DM-corner reproj:   max={max(dm_errs):.2f}  mean={sum(dm_errs)/len(dm_errs):.2f}  "
              f"individual: {[f'{e:.1f}' for e in dm_errs]}")
    if fid_errs:
        print(f"  Fiducial reproj:    max={max(fid_errs):.2f}  mean={sum(fid_errs)/len(fid_errs):.2f}  "
              f"individual: {[f'{e:.1f}' for e in fid_errs]}")

    # ── Baseline: reprojection of fiducials through H_rough ─────────────────
    print(f"\n  Fiducial baseline (H_rough forward-projection, rough crop offset):")
    print(f"    label  detected(rough)   H_rough -> canon    deviation")
    for j, entry in enumerate(accepted):
        det_x, det_y = entry['detected']
        # H_rough maps camera -> rough crop space; fiducial was detected in rough crop
        # Check: project the back-projected camera point through H_rough to see
        # what the rough H EXPECTED at this fiducial slot
        lbl = entry['label']
        # Forward: project src_fid[j] through H_rough
        p   = np.array([src_fid[j][0], src_fid[j][1], 1.0])
        ph  = H_rough @ p
        ph /= ph[2]
        dev = float(np.linalg.norm(ph[:2] - np.array([det_x, det_y])))
        print(f"    {lbl:4s}  ({det_x:6.1f},{det_y:6.1f})    "
              f"H_rough->{ph[0]:6.1f},{ph[1]:6.1f}   dev={dev:.2f}px")

    # ── Bounds check: do refined H crop corners land in source image? ────────
    if H_refined is not None:
        out_w, out_h = eng.PATTERN_SIZE
        crop_corners = np.float32([[0,0],[out_w,0],[out_w,out_h],[0,out_h]])
        H_inv = np.linalg.inv(H_refined)
        print(f"\n  Bounds check (where do crop corners map in source image {img_w}x{img_h}?):")
        any_oob = False
        for (cx, cy) in crop_corners:
            p   = np.array([cx, cy, 1.0])
            ph  = H_inv @ p
            ph /= ph[2]
            sx, sy = ph[0], ph[1]
            oob = sx < 0 or sy < 0 or sx >= img_w or sy >= img_h
            if oob:
                any_oob = True
            print(f"    crop({cx:.0f},{cy:.0f}) -> src({sx:.1f},{sy:.1f}) "
                  f"{'  *** OUT OF BOUNDS ***' if oob else ''}")
        print(f"  Bounds check result: {'FAIL - OOB corners present' if any_oob else 'PASS'}")

print("\nDone.")
