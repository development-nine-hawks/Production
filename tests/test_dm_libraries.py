#!/usr/bin/env python3
"""
test_dm_libraries.py
--------------------
Standalone diagnostic: run tc-01.jpeg and tc-03.jpeg through every available
DataMatrix decoding library across multiple image preprocessing variants.

Usage:
    python test_dm_libraries.py

Output:
    - Console + test_dm_libraries_<timestamp>.log
    - debug_images/<testcase>_<variant>.png  (preprocessed variants)
    - Summary table at the end
"""

import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

TEST_IMAGES = {
    "tc-01": SCRIPT_DIR / "tests" / "batch_test" / "tc-01.jpeg",
    "tc-03": SCRIPT_DIR / "tests" / "batch_test" / "tc-03.jpeg",
}

DEBUG_DIR = SCRIPT_DIR / "debug_images"
DEBUG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = SCRIPT_DIR / f"test_dm_libraries_{ts}.log"

logger = logging.getLogger("dm_diag")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
fh = logging.FileHandler(log_path, encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

# ---------------------------------------------------------------------------
# Library availability flags
# ---------------------------------------------------------------------------
HAS_PYLIBDMTX  = False
HAS_ZXINGCPP   = False
HAS_PYZBAR     = False
HAS_DMTXREAD   = False

try:
    from pylibdmtx.pylibdmtx import decode as _dm_decode
    HAS_PYLIBDMTX = True
    logger.info("[INIT] pylibdmtx: available")
except ImportError:
    logger.warning("[INIT] pylibdmtx: NOT available (ImportError)")

try:
    import zxingcpp as _zxingcpp
    HAS_ZXINGCPP = True
    logger.info("[INIT] zxing-cpp: available")
except ImportError:
    logger.warning("[INIT] zxing-cpp: NOT available — install with: pip install zxing-cpp")

try:
    from pyzbar import pyzbar as _pyzbar
    HAS_PYZBAR = True
    logger.info("[INIT] pyzbar: available")
except ImportError:
    logger.warning("[INIT] pyzbar: NOT available (ImportError)")

_dmtxread_path = shutil.which("dmtxread")
if _dmtxread_path:
    HAS_DMTXREAD = True
    logger.info(f"[INIT] dmtxread CLI: found at {_dmtxread_path}")
else:
    logger.warning("[INIT] dmtxread CLI: NOT on PATH — skipping")

# ---------------------------------------------------------------------------
# Image preprocessing variants
# ---------------------------------------------------------------------------

def build_variants(bgr: np.ndarray, stem: str) -> list[tuple[str, np.ndarray]]:
    """
    Return list of (label, image_array) for each preprocessing variant.
    Also saves each variant to debug_images/ as a side effect.
    Variants:
        a. raw (unmodified BGR)
        b. grayscale
        c. grayscale + CLAHE
        d. grayscale + Otsu threshold
        e. 2x upscaled grayscale
    """
    variants = []

    # a — raw (unmodified)
    variants.append(("raw", bgr))

    # b — grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_gray.png"), gray)

    # c — CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    variants.append(("clahe", gray_clahe))
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_clahe.png"), gray_clahe)

    # d — Otsu threshold
    _, gray_otsu = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", gray_otsu))
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_otsu.png"), gray_otsu)

    # e — 2x upscaled grayscale
    h, w = gray.shape[:2]
    gray_2x = cv2.resize(gray, (w * 2, h * 2),
                          interpolation=cv2.INTER_CUBIC)
    variants.append(("2x_upscale", gray_2x))
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_2x_upscale.png"), gray_2x)

    return variants


# ---------------------------------------------------------------------------
# Library-specific decode helpers
# ---------------------------------------------------------------------------

def decode_pylibdmtx(img: np.ndarray, variant_label: str) -> list[str]:
    """Try pylibdmtx at shrink=1,2,3,4. Return list of decoded strings."""
    if not HAS_PYLIBDMTX:
        return []
    decoded_texts = []
    for shrink in (1, 2, 3, 4):
        t0 = time.perf_counter()
        try:
            results = _dm_decode(img, shrink=shrink)
            elapsed = (time.perf_counter() - t0) * 1000
            texts = []
            for r in results:
                try:
                    t = r.data.decode("utf-8").strip()
                    texts.append(t)
                except Exception:
                    texts.append(repr(r.data))
            logger.info(
                f"    pylibdmtx shrink={shrink} variant={variant_label}: "
                f"{len(results)} result(s) in {elapsed:.1f}ms | "
                f"raw repr: {repr(results)[:200]}"
            )
            if texts:
                logger.info(f"    pylibdmtx shrink={shrink} DECODED: {texts}")
                decoded_texts.extend(texts)
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error(
                f"    pylibdmtx shrink={shrink} variant={variant_label}: "
                f"EXCEPTION after {elapsed:.1f}ms\n{traceback.format_exc()}"
            )
    return decoded_texts


def decode_zxingcpp(img: np.ndarray, variant_label: str) -> list[str]:
    """Try zxing-cpp. Return list of decoded strings."""
    if not HAS_ZXINGCPP:
        return []
    t0 = time.perf_counter()
    try:
        # zxingcpp expects RGB or grayscale uint8 ndarray
        if img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img  # already grayscale
        results = _zxingcpp.read_barcodes(
            rgb,
            formats=_zxingcpp.BarcodeFormat.DataMatrix
        )
        elapsed = (time.perf_counter() - t0) * 1000
        texts = [r.text for r in results if r.text]
        logger.info(
            f"    zxing-cpp variant={variant_label}: "
            f"{len(results)} result(s) in {elapsed:.1f}ms | "
            f"raw repr: {repr(results)[:200]}"
        )
        if texts:
            logger.info(f"    zxing-cpp DECODED: {texts}")
        return texts
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"    zxing-cpp variant={variant_label}: "
            f"EXCEPTION after {elapsed:.1f}ms\n{traceback.format_exc()}"
        )
        return []


def decode_pyzbar(img: np.ndarray, variant_label: str) -> list[str]:
    """Try pyzbar (DataMatrix support is weak). Return list of decoded strings."""
    if not HAS_PYZBAR:
        return []
    t0 = time.perf_counter()
    try:
        from pyzbar.pyzbar import ZBarSymbol
        results = _pyzbar.decode(img, symbols=[ZBarSymbol.DATAMATRIX])
        elapsed = (time.perf_counter() - t0) * 1000
        texts = []
        for r in results:
            try:
                texts.append(r.data.decode("utf-8").strip())
            except Exception:
                texts.append(repr(r.data))
        logger.info(
            f"    pyzbar variant={variant_label}: "
            f"{len(results)} result(s) in {elapsed:.1f}ms | "
            f"raw repr: {repr(results)[:200]}"
        )
        if texts:
            logger.info(f"    pyzbar DECODED: {texts}")
        else:
            logger.info(
                f"    pyzbar variant={variant_label}: no DataMatrix found "
                f"(note: zbar DataMatrix support is limited)"
            )
        return texts
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"    pyzbar variant={variant_label}: "
            f"EXCEPTION after {elapsed:.1f}ms\n{traceback.format_exc()}"
        )
        return []


def decode_dmtxread_cli(image_path: str, variant_label: str,
                         tmp_path: str | None = None) -> list[str]:
    """
    Shell out to dmtxread CLI.
    If tmp_path is given, write the variant image there first and decode that.
    Returns list of decoded strings.
    """
    if not HAS_DMTXREAD:
        return []
    target = tmp_path if tmp_path else image_path
    cmd = ["dmtxread", "-n", "-c", target]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        elapsed = (time.perf_counter() - t0) * 1000
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        logger.info(
            f"    dmtxread variant={variant_label}: "
            f"exit={proc.returncode} in {elapsed:.1f}ms | "
            f"stdout={repr(stdout)[:300]} | stderr={repr(stderr)[:200]}"
        )
        texts = [line.strip() for line in stdout.splitlines() if line.strip()]
        if texts:
            logger.info(f"    dmtxread DECODED: {texts}")
        return texts
    except subprocess.TimeoutExpired:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"    dmtxread variant={variant_label}: TIMEOUT after {elapsed:.1f}ms"
        )
        return []
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"    dmtxread variant={variant_label}: "
            f"EXCEPTION after {elapsed:.1f}ms\n{traceback.format_exc()}"
        )
        return []


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(tc_name: str, image_path: Path) -> dict:
    """
    Run all libraries × all variants for one image.

    Returns summary dict:
        {library_name: {"successes": [(variant, [texts])], "failed_variants": [...]}}
    """
    logger.info("=" * 70)
    logger.info(f"PROCESSING: {tc_name}  ({image_path})")
    logger.info("=" * 70)

    if not image_path.exists():
        logger.error(f"  Image file not found: {image_path}")
        return {}

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        logger.error(f"  cv2.imread failed for: {image_path}")
        return {}

    logger.info(f"  Loaded: shape={bgr.shape} dtype={bgr.dtype}")

    variants = build_variants(bgr, tc_name)
    logger.info(f"  Variants built: {[v[0] for v in variants]}")

    summary: dict[str, dict] = {}

    # ── pylibdmtx ────────────────────────────────────────────────────────
    if HAS_PYLIBDMTX:
        lib = "pylibdmtx"
        summary[lib] = {"successes": [], "failed_variants": []}
        for v_label, v_img in variants:
            logger.info(f"  [{lib}] variant={v_label}")
            texts = decode_pylibdmtx(v_img, v_label)
            if texts:
                summary[lib]["successes"].append((v_label, texts))
            else:
                summary[lib]["failed_variants"].append(v_label)

    # ── zxing-cpp ─────────────────────────────────────────────────────────
    if HAS_ZXINGCPP:
        lib = "zxing-cpp"
        summary[lib] = {"successes": [], "failed_variants": []}
        for v_label, v_img in variants:
            logger.info(f"  [{lib}] variant={v_label}")
            texts = decode_zxingcpp(v_img, v_label)
            if texts:
                summary[lib]["successes"].append((v_label, texts))
            else:
                summary[lib]["failed_variants"].append(v_label)

    # ── pyzbar ───────────────────────────────────────────────────────────
    if HAS_PYZBAR:
        lib = "pyzbar"
        summary[lib] = {"successes": [], "failed_variants": []}
        for v_label, v_img in variants:
            logger.info(f"  [{lib}] variant={v_label}")
            texts = decode_pyzbar(v_img, v_label)
            if texts:
                summary[lib]["successes"].append((v_label, texts))
            else:
                summary[lib]["failed_variants"].append(v_label)

    # ── dmtxread CLI ──────────────────────────────────────────────────────
    if HAS_DMTXREAD:
        lib = "dmtxread"
        summary[lib] = {"successes": [], "failed_variants": []}
        for v_label, v_img in variants:
            logger.info(f"  [{lib}] variant={v_label}")
            # For raw variant, pass the original file path directly.
            # For preprocessed variants, write a temp PNG and pass that.
            if v_label == "raw":
                texts = decode_dmtxread_cli(str(image_path), v_label)
            else:
                tmp = str(DEBUG_DIR / f"_tmp_{tc_name}_{v_label}.png")
                cv2.imwrite(tmp, v_img)
                texts = decode_dmtxread_cli(str(image_path), v_label,
                                             tmp_path=tmp)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if texts:
                summary[lib]["successes"].append((v_label, texts))
            else:
                summary[lib]["failed_variants"].append(v_label)

    return summary


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: dict[str, dict]):
    """
    all_results: {tc_name: {lib: {successes, failed_variants}}}
    Print a table: rows=library, columns=tc, cells=successes or FAILED.
    """
    tc_names = list(all_results.keys())
    libs: set[str] = set()
    for tc_data in all_results.values():
        libs.update(tc_data.keys())
    libs_sorted = sorted(libs)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)

    # Header
    col_w = 36
    header = f"{'Library':<20}" + "".join(f"{tc:<{col_w}}" for tc in tc_names)
    logger.info(header)
    logger.info("-" * len(header))

    for lib in libs_sorted:
        cells = []
        for tc in tc_names:
            tc_data = all_results.get(tc, {})
            lib_data = tc_data.get(lib)
            if lib_data is None:
                cells.append("(library unavailable)")
                continue
            successes = lib_data["successes"]
            if successes:
                parts = []
                for v_label, texts in successes:
                    parts.append(f"{v_label}: {texts}")
                cell = "; ".join(parts)
                if len(cell) > col_w - 2:
                    cell = cell[:col_w - 5] + "..."
            else:
                cell = "FAILED (all variants)"
            cells.append(cell)
        row = f"{lib:<20}" + "".join(f"{c:<{col_w}}" for c in cells)
        logger.info(row)

    logger.info("-" * len(header))
    logger.info("")

    # Verbose breakdown per tc × lib
    for tc, tc_data in all_results.items():
        logger.info(f"--- {tc} detailed ---")
        for lib, lib_data in tc_data.items():
            successes = lib_data["successes"]
            failed = lib_data["failed_variants"]
            if successes:
                for v_label, texts in successes:
                    logger.info(f"  {lib:20s} variant={v_label} SUCCESS -> {texts}")
            if failed:
                logger.info(f"  {lib:20s} FAILED on variants: {failed}")
        logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("DataMatrix Library Diagnostic")
    logger.info(f"Script: {__file__}")
    logger.info(f"Log:    {log_path}")
    logger.info(f"Debug images: {DEBUG_DIR}")
    logger.info("")

    available = []
    if HAS_PYLIBDMTX:  available.append("pylibdmtx")
    if HAS_ZXINGCPP:   available.append("zxing-cpp")
    if HAS_PYZBAR:     available.append("pyzbar")
    if HAS_DMTXREAD:   available.append("dmtxread")
    logger.info(f"Available libraries: {available or ['NONE — install at least one']}")
    logger.info("")

    all_results: dict[str, dict] = {}

    for tc_name, image_path in TEST_IMAGES.items():
        all_results[tc_name] = process_image(tc_name, image_path)

    print_summary(all_results)

    logger.info(f"Done. Full log: {log_path}")


if __name__ == "__main__":
    main()
