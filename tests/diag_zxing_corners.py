"""
Step 1 diagnostic: measure zxing-cpp corner-reporting convention.

Generates a synthetic DataMatrix at several scales, places it on a white
canvas with a known offset, decodes with zxing-cpp, and compares returned
corners_cv to the known true module-boundary positions.

Question: does zxing-cpp return corners at the outer pixel-boundary of the
outermost module, or at the pixel-center of that module?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import zxingcpp
import cdp_engine as eng
import database as db

db.init_db()

def test_at_scale(module_px, label="?"):
    """
    Generate an 8x18 DM at module_px pixels per module, place it at a known
    position on a white canvas, decode with zxing-cpp, and report corner offsets.

    Known truth:
      - DM is placed at canvas (OFF_X, OFF_Y)
      - DM occupies canvas [OFF_X, OFF_X + 18*module_px] x [OFF_Y, OFF_Y + 8*module_px]
      - True outer boundary corners (pixel-boundary convention):
          TL = (OFF_X,              OFF_Y)
          TR = (OFF_X + 18*module_px, OFF_Y)
          BR = (OFF_X + 18*module_px, OFF_Y + 8*module_px)
          BL = (OFF_X,              OFF_Y + 8*module_px)
      - True pixel-center corners (center of outermost module pixel):
          TL_center = (OFF_X + 0.5,                  OFF_Y + 0.5)
          TR_center = (OFF_X + 18*module_px - 0.5,   OFF_Y + 0.5)
          etc.
    """
    OFF_X, OFF_Y = 50, 50  # known placement offset on canvas

    # Generate the DM image (quiet-zone removed, exact module boundaries)
    dm_img, (n_rows, n_cols) = eng.generate_cropped_dm("AAAA", size="8x18")

    # Scale to module_px per module
    target_w = n_cols * module_px
    target_h = n_rows * module_px
    dm_scaled = cv2.resize(dm_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Place on white canvas with known offset
    canvas_w = OFF_X + target_w + 100
    canvas_h = OFF_Y + target_h + 100
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    canvas[OFF_Y:OFF_Y+target_h, OFF_X:OFF_X+target_w] = dm_scaled

    # Known true boundaries
    true_outer = {  # pixel-boundary (exclusive edge)
        "TL": (OFF_X,              OFF_Y),
        "TR": (OFF_X + target_w,   OFF_Y),
        "BR": (OFF_X + target_w,   OFF_Y + target_h),
        "BL": (OFF_X,              OFF_Y + target_h),
    }
    true_center = {  # pixel-center of outermost module
        "TL": (OFF_X + 0.5,              OFF_Y + 0.5),
        "TR": (OFF_X + target_w - 0.5,  OFF_Y + 0.5),
        "BR": (OFF_X + target_w - 0.5,  OFF_Y + target_h - 0.5),
        "BL": (OFF_X + 0.5,              OFF_Y + target_h - 0.5),
    }
    true_module_center = {  # center of outermost MODULE (half-module inset)
        "TL": (OFF_X + module_px/2,              OFF_Y + module_px/2),
        "TR": (OFF_X + target_w - module_px/2,  OFF_Y + module_px/2),
        "BR": (OFF_X + target_w - module_px/2,  OFF_Y + target_h - module_px/2),
        "BL": (OFF_X + module_px/2,              OFF_Y + target_h - module_px/2),
    }

    # Decode with zxing-cpp
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    img_h = canvas_bgr.shape[0]
    barcodes = zxingcpp.read_barcodes(canvas_bgr)
    dm_results = [b for b in barcodes if b.format == zxingcpp.BarcodeFormat.DataMatrix]

    print(f"\n{'='*65}")
    print(f"module_px={module_px:2d} ({label})  DM size={target_w}x{target_h}  "
          f"placed at ({OFF_X},{OFF_Y})")
    print(f"  True outer-boundary: TL={true_outer['TL']}  "
          f"TR={true_outer['TR']}")
    print(f"  True pixel-center:   TL={true_center['TL']}  "
          f"TR={true_center['TR']}")
    print(f"  True module-center:  TL={true_module_center['TL']}  "
          f"TR={true_module_center['TR']}")

    if not dm_results:
        print("  [FAIL] zxing-cpp returned no results — try higher scale")
        return None

    b = dm_results[0]
    pos = b.position
    pts_raw = {
        "TL": (pos.top_left.x,     pos.top_left.y),
        "TR": (pos.top_right.x,    pos.top_right.y),
        "BR": (pos.bottom_right.x, pos.bottom_right.y),
        "BL": (pos.bottom_left.x,  pos.bottom_left.y),
    }

    print(f"\n  zxing-cpp raw corners:")
    print(f"    Slot  Detected         vs outer-boundary  vs pixel-center  vs module-center")
    results = {}
    for slot in ["TL", "TR", "BR", "BL"]:
        dx_raw, dy_raw = pts_raw[slot]
        d_outer = (dx_raw - true_outer[slot][0], dy_raw - true_outer[slot][1])
        d_center = (dx_raw - true_center[slot][0], dy_raw - true_center[slot][1])
        d_mod = (dx_raw - true_module_center[slot][0], dy_raw - true_module_center[slot][1])
        print(f"    {slot}   ({dx_raw:6.2f},{dy_raw:6.2f})   "
              f"d_outer=({d_outer[0]:+.2f},{d_outer[1]:+.2f})  "
              f"d_center=({d_center[0]:+.2f},{d_center[1]:+.2f})  "
              f"d_module=({d_mod[0]:+.2f},{d_mod[1]:+.2f})")
        results[slot] = {"raw": pts_raw[slot], "d_outer": d_outer,
                         "d_center": d_center, "d_module": d_mod}

    # Summary: which convention does zxing-cpp most closely follow?
    avg_outer  = np.mean([abs(v) for s in results.values() for v in s["d_outer"]])
    avg_center = np.mean([abs(v) for s in results.values() for v in s["d_center"]])
    avg_module = np.mean([abs(v) for s in results.values() for v in s["d_module"]])
    best = min(("outer-boundary", avg_outer), ("pixel-center", avg_center),
               ("module-center", avg_module), key=lambda x: x[1])
    print(f"\n  Mean |error| vs: outer={avg_outer:.3f}  pixel-center={avg_center:.3f}  "
          f"module-center={avg_module:.3f}")
    print(f"  => Best match: {best[0]} (mean |err|={best[1]:.3f}px)")
    return results


# Test at 3 different module sizes
for mpx, lbl in [(10, "small"), (20, "medium"), (28, "label-actual")]:
    test_at_scale(mpx, lbl)

print("\n\nDone.")
