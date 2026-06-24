

#!/usr/bin/env python3
"""
batch_test_runner.py
--------------------
PhoneCDP Batch Verification Runner.

Usage:
    python batch_test_runner.py [--input FOLDER] [--output FOLDER]

Defaults:
    --input  batch_test/
    --output batch_test_results/

Drop captured label photos (.jpg / .jpeg / .png) into the input folder,
run this script, and collect the full PDF report in the output folder.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------------------------------------------------------------------------
# Bootstrap: add the Production folder to sys.path so all local modules
# (cdp_engine, cdp_analytics, cdp_batch_pdf, config) can be imported.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cdp_engine import (
    verify_pattern,
    regenerate_reference,
)
from cdp_analytics import (
    compute_per_block_scores,
    render_heatmap,
    render_histogram,
    render_scatter,
    render_delta_map,
)
from cdp_batch_pdf import build_report_pdf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

FIELDS = [
    "filename", "verdict", "confidence", "seed_recovered", "pattern_id",
    "moire", "correlation", "gradient", "color",
    "alignment_method", "processing_time", "success", "failure_reason",
]


# ---------------------------------------------------------------------------
# Folder setup
# ---------------------------------------------------------------------------

def setup_output_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "root":       base,
        "analytics":  base / "analytics",
        "aligned":    base / "aligned_patterns",
        "delta":      base / "delta_maps",
        "reference":  base / "reference_patterns",
        "logs":       base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("batch_runner")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _save_png(arr_bgr: np.ndarray, path: Path):
    """Save a BGR ndarray as PNG to path."""
    cv2.imwrite(str(path), arr_bgr)


def _save_png_bytes(data: bytes, path: Path):
    path.write_bytes(data)


def process_image(
    img_path: Path,
    dirs: dict[str, Path],
    logger: logging.Logger,
) -> dict:
    """
    Run the full verification pipeline on one image.

    Returns a result dict containing all metrics, file paths, and status.
    All exceptions are caught; the dict will have success=False on failure.
    """
    stem     = img_path.stem
    filename = img_path.name
    result: dict = {
        "filename":        filename,
        "stem":            stem,
        "original_path":   str(img_path),
        "analytics_dir":   str(dirs["analytics"]),
        "verdict":         "ERROR",
        "confidence":      None,
        "seed_recovered":  None,
        "pattern_id":      None,
        "scores":          {},
        "weights":         {},
        "alignment_method": "—",
        "processing_time": 0.0,
        "success":         False,
        "failure_reason":  None,
        # Paths for PDF
        "reference_path":     None,
        "captured_path_saved":None,
        "delta_path":         None,
        "dm_diagnostic":      {},
    }

    t0 = time.time()
    try:
        logger.info(f"Processing: {filename}")

        # ── Run main verification pipeline ──────────────────────────────────
        # verify_pattern writes debug images to uploads_dir.
        # We redirect to a temp subdir inside our output to pick them up later.
        uploads_tmp = dirs["root"] / "_tmp_uploads"
        uploads_tmp.mkdir(exist_ok=True)

        vr = verify_pattern(str(img_path), uploads_dir=str(uploads_tmp))

        elapsed = time.time() - t0
        result["processing_time"] = round(elapsed, 2)

        verdict = vr.get("verdict", "ERROR")
        result["verdict"]       = verdict
        result["confidence"]    = vr.get("confidence")
        result["seed_recovered"]= vr.get("seed_recovered")
        result["dm_diagnostic"] = vr.get("dm_diagnostic", {})
        result["weights"]       = vr.get("weights", {})

        if verdict == "UNABLE_TO_VERIFY":
            dm  = vr.get("dm_diagnostic", {})
            reason = (vr.get("error") or
                      dm.get("failure_reason") or
                      "Unknown verification failure")
            result["failure_reason"] = reason
            logger.warning(f"  {filename}: UNABLE_TO_VERIFY — {reason}")
            return result

        # Copy debug images out of tmp to the proper output dirs
        roi_src  = uploads_tmp / vr["roi_filename"]
        ref_src  = uploads_tmp / vr["reference_filename"]
        aln_src  = uploads_tmp / vr["aligned_filename"]

        cap_dst = dirs["aligned"]   / f"{stem}_captured.png"
        ref_dst = dirs["reference"] / f"{stem}_reference.png"

        if roi_src.exists(): shutil.copy2(roi_src, cap_dst)
        if ref_src.exists(): shutil.copy2(ref_src, ref_dst)

        result["captured_path_saved"] = str(cap_dst) if cap_dst.exists() else None
        result["reference_path"]      = str(ref_dst) if ref_dst.exists() else None

        # Scores
        sc = vr.get("scores", {})
        result["scores"]  = sc

        logger.info(f"  {filename}: {verdict}  conf={result['confidence']:.3f}  "
                    f"moire={sc.get('moire',0):.3f}  corr={sc.get('correlation',0):.3f}  "
                    f"grad={sc.get('gradient',0):.3f}  color={sc.get('color',0):.3f}")

        # ── Per-block analytics ─────────────────────────────────────────────
        # Load the captured ROI and the reference (both 512×512)
        roi_bgr = cv2.imread(str(roi_src)) if roi_src.exists() else None
        ref_bgr = cv2.imread(str(ref_src)) if ref_src.exists() else None

        if roi_bgr is not None and ref_bgr is not None:
            captured_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            captured_rgb  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            ref_gray      = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
            ref_rgb       = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

            block_scores = compute_per_block_scores(
                captured_gray, ref_gray, captured_rgb, ref_rgb, block_size=16)

            # Save per-test plots
            for test in ["moire", "correlation", "gradient", "color"]:
                arr = block_scores[test]
                for ptype, render_fn, kwargs in [
                    ("heatmap",   render_heatmap,   {"aligned_gray": captured_gray}),
                    ("histogram", render_histogram,  {}),
                    ("scatter",   render_scatter,    {}),
                ]:
                    try:
                        png = render_fn(arr, test, **kwargs)
                        out = dirs["analytics"] / f"{stem}_{test}_{ptype}.png"
                        _save_png_bytes(png, out)
                    except Exception as e:
                        logger.warning(f"  Plot {test}/{ptype} failed: {e}")

            # Delta map
            try:
                delta_png = render_delta_map(captured_gray, ref_gray)
                delta_dst = dirs["delta"] / f"{stem}_delta.png"
                _save_png_bytes(delta_png, delta_dst)
                result["delta_path"] = str(delta_dst)
            except Exception as e:
                logger.warning(f"  Delta map failed: {e}")
        else:
            logger.warning(f"  {filename}: could not load ROI/ref for per-block analytics")

        result["success"] = True

    except Exception:
        elapsed = time.time() - t0
        result["processing_time"] = round(elapsed, 2)
        tb = traceback.format_exc()
        result["failure_reason"] = tb.splitlines()[-1]
        logger.error(f"  {filename}: EXCEPTION\n{tb}")

    finally:
        # Clean up tmp uploads for this image
        try:
            uploads_tmp = dirs["root"] / "_tmp_uploads"
            if uploads_tmp.exists():
                shutil.rmtree(uploads_tmp, ignore_errors=True)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_batch_stats(results: list[dict]) -> dict:
    n    = len(results)
    ok   = sum(1 for r in results if r.get("verdict") == "AUTHENTIC")
    sus  = sum(1 for r in results if r.get("verdict") == "SUSPICIOUS")
    cnt  = sum(1 for r in results if r.get("verdict") == "COUNTERFEIT")
    err  = sum(1 for r in results if r.get("verdict") in ("UNABLE_TO_VERIFY", "ERROR"))

    confs = [r["confidence"] for r in results if r.get("confidence") is not None]

    def _avg(key):
        vals = [r["scores"][key] for r in results if r.get("scores") and key in r["scores"]]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "total":           n,
        "authentic":       ok,
        "suspicious":      sus,
        "counterfeit":     cnt,
        "unable":          err,
        "avg_confidence":  float(np.mean(confs)) if confs else 0.0,
        "avg_moire":       _avg("moire"),
        "avg_correlation": _avg("correlation"),
        "avg_gradient":    _avg("gradient"),
        "avg_color":       _avg("color"),
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            sc = r.get("scores") or {}
            row = {
                "filename":        r.get("filename"),
                "verdict":         r.get("verdict"),
                "confidence":      f"{r['confidence']:.4f}" if r.get("confidence") is not None else "",
                "seed_recovered":  r.get("seed_recovered"),
                "pattern_id":      r.get("pattern_id"),
                "moire":           f"{sc.get('moire',0):.4f}",
                "correlation":     f"{sc.get('correlation',0):.4f}",
                "gradient":        f"{sc.get('gradient',0):.4f}",
                "color":           f"{sc.get('color',0):.4f}",
                "alignment_method":r.get("alignment_method"),
                "processing_time": r.get("processing_time"),
                "success":         r.get("success"),
                "failure_reason":  r.get("failure_reason"),
            }
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PhoneCDP Batch Verification Runner")
    parser.add_argument("--input",  default=str(SCRIPT_DIR / "batch_test"),
                        help="Folder containing captured label photos")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "batch_test_results"),
                        help="Folder for results (created if missing)")
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    # ── Validate input folder ───────────────────────────────────────────────
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created input folder: {input_dir}")
        print("[INFO] Drop your captured label photos into that folder and re-run.")
        sys.exit(0)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not images:
        print(f"[WARN] No .jpg/.jpeg/.png images found in: {input_dir}")
        sys.exit(0)

    print(f"[INFO] Found {len(images)} image(s) in {input_dir}")

    # ── Setup output structure ──────────────────────────────────────────────
    dirs = setup_output_dirs(output_dir)
    run_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = dirs["logs"] / f"batch_run_{ts_file}.log"
    logger   = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("PhoneCDP Batch Verification Run")
    logger.info(f"Run timestamp : {run_ts}")
    logger.info(f"Input folder  : {input_dir}")
    logger.info(f"Output folder : {output_dir}")
    logger.info(f"Images found  : {len(images)}")
    logger.info("=" * 60)

    # ── Process each image ──────────────────────────────────────────────────
    results: list[dict] = []
    for i, img_path in enumerate(images, 1):
        logger.info(f"--- Image {i}/{len(images)} ---")
        r = process_image(img_path, dirs, logger)
        results.append(r)

    # ── Aggregate stats ─────────────────────────────────────────────────────
    batch_stats = compute_batch_stats(results)
    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH SUMMARY")
    logger.info(f"  Total       : {batch_stats['total']}")
    logger.info(f"  Authentic   : {batch_stats['authentic']}")
    logger.info(f"  Suspicious  : {batch_stats['suspicious']}")
    logger.info(f"  Counterfeit : {batch_stats['counterfeit']}")
    logger.info(f"  Unable/Error: {batch_stats['unable']}")
    logger.info(f"  Avg Confidence: {batch_stats['avg_confidence']*100:.1f}%")
    logger.info("=" * 60)

    # ── Save CSV ────────────────────────────────────────────────────────────
    csv_path = output_dir / f"batch_results_{ts_file}.csv"
    save_csv(results, csv_path)
    logger.info(f"CSV saved: {csv_path}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    json_path = output_dir / f"batch_results_{ts_file}.json"
    with json_path.open("w", encoding="utf-8") as jf:
        # Make serialisable — remove ndarray values
        safe = []
        for r in results:
            row = {k: v for k, v in r.items()
                   if not isinstance(v, np.ndarray)}
            safe.append(row)
        json.dump({"run_ts": run_ts, "stats": batch_stats, "results": safe},
                  jf, indent=2, default=str)
    logger.info(f"JSON saved: {json_path}")

    # ── Generate PDF ────────────────────────────────────────────────────────
    pdf_path = output_dir / "batch_test_report.pdf"
    logger.info(f"Generating PDF report → {pdf_path}")
    try:
        build_report_pdf(str(pdf_path), results, batch_stats, run_ts)
        logger.info(f"PDF report ready: {pdf_path}")
    except Exception:
        logger.error(f"PDF generation failed:\n{traceback.format_exc()}")

    # ── Final output ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  BATCH COMPLETE")
    print(f"  Images processed : {batch_stats['total']}")
    n = batch_stats['total']
    ok = batch_stats['authentic']
    print(f"  Authentic        : {ok}  ({ok/n*100:.0f}%)" if n else "  Authentic: 0")
    print(f"  Avg Confidence   : {batch_stats['avg_confidence']*100:.1f}%")
    print(f"  PDF report       : {pdf_path}")
    print(f"  CSV              : {csv_path}")
    print(f"  Log              : {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
