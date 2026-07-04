"""
cdp_batch_pdf.py
----------------
ReportLab PDF assembly for the PhoneCDP batch verification report.
Called exclusively from batch_test_runner.py.  No Flask dependency.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image as RLImage, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable,
)
from reportlab.platypus.flowables import Flowable

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_BRAND  = colors.HexColor("#1e3a5f")
C_ACCENT = colors.HexColor("#2e86de")
C_GREEN  = colors.HexColor("#2ecc71")
C_RED    = colors.HexColor("#e74c3c")
C_YELLOW = colors.HexColor("#f39c12")
C_GRAY   = colors.HexColor("#7f8c8d")
C_LIGHT  = colors.HexColor("#ecf0f1")
C_WHITE  = colors.white
C_BLACK  = colors.black

EXPECTED_VERDICTS = {
    # "tc-01.jpeg": "AUTHENTIC",
    # "tc-02.jpeg": "AUTHENTIC",
    # "tc-03.jpeg": "AUTHENTIC",
    # "tc-04.jpeg": "AUTHENTIC",
    "tc-05.jpeg":       "AUTHENTIC",
    "tc-05-flash.jpeg": "AUTHENTIC",
    # "tc-06.jpeg": "AUTHENTIC",
    # "tc-07.jpeg": "AUTHENTIC",
    # "tc-08.jpeg": "AUTHENTIC",
    # "tc-09.jpeg": "COUNTERFEIT",
    # "tc-10.jpeg": "COUNTERFEIT",
}

TEST_CASE_DESCRIPTIONS = {
    # "tc-01.jpeg": "Genuine label, photo straight-on",
    # "tc-02.jpeg": "Genuine label, slight angle (~30°)",
    # "tc-03.jpeg": "Genuine label, rotated 90°",
    # "tc-04.jpeg": "Genuine label, rotated 180°",
    "tc-05.jpeg":       "Genuine label, low light — no flash",
    "tc-05-flash.jpeg": "Genuine label, low light — phone flash",
    # "tc-06.jpeg": "Genuine label, glare/flash on pattern",
    # "tc-07.jpeg": "Genuine label, far away (pattern small in frame)",
    # "tc-08.jpeg": "Genuine label, motion blur",
    # "tc-09.jpeg": "Counterfeit label, straight-on (print → photo → reprint → photo)",
    # "tc-10.jpeg": "Counterfeit label, at an angle",
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _verdict_color(verdict: str):
    return {"AUTHENTIC": C_GREEN, "SUSPICIOUS": C_YELLOW,
            "COUNTERFEIT": C_RED}.get(verdict, C_GRAY)


def _pct(v):
    return f"{v * 100:.1f}%" if v is not None else "—"


def _numpy_to_rl_image(arr_bgr: np.ndarray, max_w: float, max_h: float) -> RLImage:
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB) if arr_bgr.ndim == 3 else arr_bgr
    pil = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    img_h, img_w = arr_bgr.shape[:2]
    aspect = img_w / max(img_h, 1)
    if max_w / aspect <= max_h:
        return RLImage(buf, width=max_w, height=max_w / aspect)
    return RLImage(buf, width=max_h * aspect, height=max_h)


def _png_bytes_to_rl_image(png: bytes, max_w: float, max_h: float) -> RLImage:
    buf = io.BytesIO(png)
    pil = PILImage.open(buf)
    img_w, img_h = pil.size
    aspect = img_w / max(img_h, 1)
    buf.seek(0)
    if max_w / aspect <= max_h:
        return RLImage(buf, width=max_w, height=max_w / aspect)
    return RLImage(buf, width=max_h * aspect, height=max_h)


def _file_to_rl_image(path: str, max_w: float, max_h: float) -> Optional[RLImage]:
    if not path or not os.path.isfile(path):
        return None
    arr = cv2.imread(path)
    if arr is None:
        return None
    return _numpy_to_rl_image(arr, max_w, max_h)


def _placeholder_para(label: str, ss):
    return Paragraph(f'<font color="grey" size="7">{label}<br/>N/A</font>', ss["Caption"])


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def _save_mpl_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_aggregate_charts(results: list) -> dict:
    """Build summary charts from all results. Returns {name: PNG bytes}."""
    charts = {}

    # Verdict distribution pie
    verdicts = [r.get("verdict", "ERROR") for r in results]
    counts: dict = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    vcolors = {"AUTHENTIC": "#2ecc71", "SUSPICIOUS": "#f39c12",
               "COUNTERFEIT": "#e74c3c", "UNABLE_TO_VERIFY": "#95a5a6", "ERROR": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    labels = list(counts.keys())
    vals   = [counts[k] for k in labels]
    clrs   = [vcolors.get(k, "#999") for k in labels]
    _, _, autotexts = ax.pie(vals, labels=labels, autopct="%1.0f%%",
                              colors=clrs, startangle=90, textprops={"fontsize": 9})
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Verdict Distribution", fontsize=12, fontweight="bold")
    charts["verdict_dist"] = _save_mpl_png(fig)

    # Confidence histogram
    confs = [r["confidence"] for r in results if r.get("confidence") is not None]
    if confs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(confs, bins=20, range=(0, 1), color="#3a86ff", edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(confs)), color="#e74c3c", linewidth=1.5,
                   label=f"mean = {float(np.mean(confs)):.3f}")
        ax.axvline(0.65, color="#2ecc71", linestyle="--", linewidth=1, label="Authentic (0.65)")
        ax.axvline(0.45, color="#f39c12", linestyle="--", linewidth=1, label="Suspicious (0.45)")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
        ax.set_title("Confidence Score Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y"); fig.tight_layout()
        charts["confidence_dist"] = _save_mpl_png(fig)

    # Per-metric histograms
    metric_colors = {"moire": "#9b59b6", "correlation": "#3498db",
                     "gradient": "#1abc9c", "color": "#e67e22"}
    for metric, col in metric_colors.items():
        vals_m = [r["scores"][metric] for r in results
                  if r.get("scores") and metric in r["scores"]]
        if not vals_m:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.hist(vals_m, bins=20, range=(0, 1), color=col, edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(vals_m)), color="#333", linewidth=1.5,
                   label=f"mean = {float(np.mean(vals_m)):.3f}")
        ax.set_xlabel("Score"); ax.set_ylabel("Count")
        ax.set_title(f"{metric.capitalize()} Score Distribution", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y"); fig.tight_layout()
        charts[f"{metric}_dist"] = _save_mpl_png(fig)

    # Processing time distribution
    times = [r["processing_time"] for r in results if r.get("processing_time") is not None]
    if times:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.hist(times, bins=max(5, len(times) // 3), color="#34495e",
                edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(times)), color="#e74c3c", linewidth=1.5,
                   label=f"mean = {float(np.mean(times)):.1f}s")
        ax.set_xlabel("Processing Time (s)"); ax.set_ylabel("Count")
        ax.set_title("Processing Time Distribution", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y"); fig.tight_layout()
        charts["time_dist"] = _save_mpl_png(fig)

    # Failure breakdown bar
    fails = [r.get("failure_reason", "") for r in results
             if r.get("verdict") in ("UNABLE_TO_VERIFY", "ERROR") and r.get("failure_reason")]
    if fails:
        rc: dict = {}
        for f in fails:
            rc[f] = rc.get(f, 0) + 1
        fig, ax = plt.subplots(figsize=(8, max(3, len(rc) * 0.7)))
        labels_f = list(rc.keys()); vals_f = [rc[k] for k in labels_f]
        bars = ax.barh(labels_f, vals_f, color="#e74c3c", edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_xlabel("Count")
        ax.set_title("Failure Reason Breakdown", fontsize=11, fontweight="bold")
        ax.invert_yaxis(); fig.tight_layout()
        charts["failure_breakdown"] = _save_mpl_png(fig)

    # Score box-plot
    metrics = ["moire", "correlation", "gradient", "color"]
    data_bp = [[r["scores"][m] for r in results if r.get("scores") and m in r["scores"]]
               for m in metrics]
    valid_data = [(m, d) for m, d in zip(metrics, data_bp) if d]
    if valid_data:
        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot([d for _, d in valid_data],
                        labels=[m.capitalize() for m, _ in valid_data],
                        patch_artist=True)
        box_colors = ["#9b59b6", "#3498db", "#1abc9c", "#e67e22"]
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.axhline(0.5, color="#999", linestyle="--", linewidth=1)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1)
        ax.set_title("Score Comparison (Box Plot)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y"); fig.tight_layout()
        charts["score_boxplot"] = _save_mpl_png(fig)

    return charts


# ---------------------------------------------------------------------------
# Page header / footer
# ---------------------------------------------------------------------------

def _add_header_footer(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(C_BRAND)
    canvas.rect(0, h - 26*mm, w, 26*mm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(15*mm, h - 15*mm, "PhoneCDP Batch Verification Report")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(w - 15*mm, h - 15*mm, datetime.now().strftime("%Y-%m-%d"))
    canvas.setFillColor(C_GRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(15*mm, 8*mm, "CONFIDENTIAL — AIB Innovations")
    canvas.drawRightString(w - 15*mm, 8*mm, f"Page {doc.page}")
    canvas.setStrokeColor(C_LIGHT); canvas.setLineWidth(0.5)
    canvas.line(15*mm, 13*mm, w - 15*mm, 13*mm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _styles():
    ss = getSampleStyleSheet()
    defs = [
        ParagraphStyle("CoverTitle",  parent=ss["Title"],   fontSize=26, textColor=C_WHITE,
                       spaceAfter=6,  alignment=TA_CENTER),
        ParagraphStyle("CoverSub",    parent=ss["Normal"],  fontSize=12, textColor=C_LIGHT,
                       spaceAfter=5,  alignment=TA_CENTER),
        ParagraphStyle("SectionHead", parent=ss["Heading1"],fontSize=14, textColor=C_BRAND,
                       spaceBefore=8, spaceAfter=4),
        ParagraphStyle("SubHead",     parent=ss["Heading2"],fontSize=11, textColor=C_ACCENT,
                       spaceBefore=5, spaceAfter=3),
        ParagraphStyle("Body",        parent=ss["Normal"],  fontSize=9,  leading=13),
        ParagraphStyle("Caption",     parent=ss["Normal"],  fontSize=7.5,textColor=C_GRAY,
                       alignment=TA_CENTER, spaceAfter=3),
        ParagraphStyle("VerdictBig",  parent=ss["Normal"],  fontSize=16, fontName="Helvetica-Bold",
                       alignment=TA_CENTER),
    ]
    for s in defs:
        try:
            ss.add(s)
        except Exception:
            pass
    return ss


def _std_table(rows, col_widths, header_bg=C_BRAND):
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.25, C_GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("ALIGN",         (1, 1), (-1, -1), "RIGHT"),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

class _FilledRect(Flowable):
    def __init__(self, w, h, color):
        super().__init__()
        self._w = w; self._h = h; self._color = color
    def wrap(self, *_): return self._w, self._h
    def draw(self):
        self.canv.setFillColor(self._color)
        self.canv.rect(0, 0, self._w, self._h, fill=1, stroke=0)


def _build_cover(story, ss, batch_stats: dict, run_ts: str):
    pw, _ = A4
    usable = pw - 30*mm

    story.append(_FilledRect(usable, 72*mm, C_BRAND))
    story.append(Spacer(1, -70*mm))
    story.append(Paragraph("PhoneCDP", ss["CoverTitle"]))
    story.append(Paragraph("Batch Verification Report", ss["CoverSub"]))
    story.append(Paragraph(run_ts, ss["CoverSub"]))
    story.append(Spacer(1, 18*mm))

    n  = batch_stats["total"]
    ok = batch_stats["authentic"]
    kpi = [
        ["Metric", "Value"],
        ["Images Tested",       str(n)],
        ["Authentic",           str(ok)],
        ["Suspicious",          str(batch_stats["suspicious"])],
        ["Counterfeit",         str(batch_stats["counterfeit"])],
        ["Unable to Verify",    str(batch_stats["unable"])],
        ["Success Rate",        f"{ok/n*100:.1f}%" if n else "—"],
        ["Average Confidence",  _pct(batch_stats.get("avg_confidence", 0))],
        ["Avg Moiré",           f"{batch_stats.get('avg_moire',0):.3f}"],
        ["Avg Correlation",     f"{batch_stats.get('avg_correlation',0):.3f}"],
        ["Avg Gradient",        f"{batch_stats.get('avg_gradient',0):.3f}"],
        ["Avg Color",           f"{batch_stats.get('avg_color',0):.3f}"],
    ]
    story.append(_std_table(kpi, [9*cm, 6*cm]))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        "Generated automatically by the PhoneCDP verification system. "
        "Contains per-sample analysis, per-block analytics charts, and aggregate statistics. "
        "CONFIDENTIAL — AIB Innovations.", ss["Body"]))
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _build_exec_summary(story, ss, results: list, batch_stats: dict):
    story.append(Paragraph("Executive Summary", ss["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BRAND, spaceAfter=5))

    n   = batch_stats["total"]
    ok  = batch_stats["authentic"]
    sus = batch_stats["suspicious"]
    cnt = batch_stats["counterfeit"]
    err = batch_stats["unable"]

    best  = max((r for r in results if r.get("confidence") is not None),
                key=lambda r: r["confidence"], default=None)
    worst = min((r for r in results if r.get("confidence") is not None),
                key=lambda r: r["confidence"], default=None)

    rc: dict = {}
    for r in results:
        if r.get("verdict") in ("UNABLE_TO_VERIFY", "ERROR") and r.get("failure_reason"):
            k = r["failure_reason"]
            rc[k] = rc.get(k, 0) + 1
    most_common = max(rc, key=rc.get) if rc else "N/A"

    summary = [
        ["Category",                  "Count",  "Rate"],
        ["Total Samples",             str(n),   "100%"],
        ["Authentic",                 str(ok),  f"{ok/n*100:.1f}%" if n else "—"],
        ["Suspicious",                str(sus), f"{sus/n*100:.1f}%" if n else "—"],
        ["Counterfeit",               str(cnt), f"{cnt/n*100:.1f}%" if n else "—"],
        ["Unable to Verify / Error",  str(err), f"{err/n*100:.1f}%" if n else "—"],
    ]
    tbl = Table(summary, colWidths=[9*cm, 3*cm, 3.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_BRAND),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_GRAY),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 5*mm))

    def _safe_label(r):
        if r is None:
            return "—"
        return f"{r['filename']} ({_pct(r['confidence'])})"

    details = [
        ["Item",                       "Value"],
        ["Average Confidence",         _pct(batch_stats.get("avg_confidence", 0))],
        ["Best Sample (confidence)",   _safe_label(best)],
        ["Worst Sample (confidence)",  _safe_label(worst)],
        ["Most Common Failure Reason", most_common],
        ["Avg Moiré Score",            f"{batch_stats.get('avg_moire',0):.3f}"],
        ["Avg Correlation Score",      f"{batch_stats.get('avg_correlation',0):.3f}"],
        ["Avg Gradient Score",         f"{batch_stats.get('avg_gradient',0):.3f}"],
        ["Avg Color Score",            f"{batch_stats.get('avg_color',0):.3f}"],
    ]
    story.append(_std_table(details, [8*cm, 9.5*cm]))
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# All-results summary table
# ---------------------------------------------------------------------------

def _build_results_table(story, ss, results: list):
    # Switch to landscape for both summary tables, then return to portrait.
    story.append(NextPageTemplate("landscape"))
    story.append(PageBreak())

    # ── Table 1: Summary (lean — verdict + confidence only) ──────────────────
    story.append(Paragraph("All Results — Summary", ss["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BRAND, spaceAfter=5))

    hdr1 = ["#", "Filename", "Description", "Expected", "Actual", "Confidence", "Result"]
    sum_rows = [hdr1]
    sum_row_colors = []
    sum_result_colors = []
    for i, r in enumerate(results, 1):
        filename    = r.get("filename", "")
        actual      = r.get("verdict", "—")
        expected    = EXPECTED_VERDICTS.get(filename, "—")
        description = TEST_CASE_DESCRIPTIONS.get(filename, "—")
        result_str  = ("SUCCESS" if actual == expected
                       else "FAILURE" if expected != "—"
                       else "—")
        sum_rows.append([
            str(i), filename[:22], description[:50],
            expected, actual, _pct(r.get("confidence")), result_str,
        ])
        bg = {"AUTHENTIC": colors.HexColor("#d5f5e3"),
              "SUSPICIOUS": colors.HexColor("#fef9e7"),
              "COUNTERFEIT": colors.HexColor("#fadbd8")}.get(
              actual, colors.HexColor("#f8f9fa"))
        sum_row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))
        if result_str == "SUCCESS":
            sum_result_colors += [
                ("BACKGROUND", (6, i), (6, i), colors.HexColor("#d5f5e3")),
                ("TEXTCOLOR",  (6, i), (6, i), colors.HexColor("#1a7a40")),
            ]
        elif result_str == "FAILURE":
            sum_result_colors += [
                ("BACKGROUND", (6, i), (6, i), colors.HexColor("#fadbd8")),
                ("TEXTCOLOR",  (6, i), (6, i), colors.HexColor("#a93226")),
            ]

    _lw = 26.7 * cm   # landscape A4 usable width (297mm - 2×15mm)
    col_w1 = [
        _lw * 0.03,   # #
        _lw * 0.13,   # Filename
        _lw * 0.36,   # Description
        _lw * 0.11,   # Expected
        _lw * 0.11,   # Actual
        _lw * 0.12,   # Confidence
        _lw * 0.10,   # Result
    ]
    t1 = Table(sum_rows, colWidths=col_w1, repeatRows=1)
    t1.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_BRAND),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_GRAY),
        ("ALIGN",         (3, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        *sum_row_colors,
        *sum_result_colors,
    ]))
    story.append(t1)
    story.append(Spacer(1, 8*mm))

    # ── Table 2: Raw Metrics (capped | raw | batch pct per metric) ───────────
    story.append(Paragraph("All Results — Raw Metrics", ss["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BRAND, spaceAfter=5))

    hdr2 = ["#", "Filename",
             "Moiré\ncapped", "Moiré\nraw", "Moiré\n%ile",
             "Corr\ncapped",  "Corr\nraw",  "Corr\n%ile",
             "Grad\ncapped",  "Grad\nraw",  "Grad\n%ile"]
    raw_rows = [hdr2]
    raw_row_colors = []
    for i, r in enumerate(results, 1):
        sc = r.get("scores") or {}
        rs = r.get("raw_scores") or {}
        filename = r.get("filename", "")
        actual   = r.get("verdict", "—")

        def _rv(m):
            v = rs.get(m, {}).get("pre_clip")
            return f"{v:.3f}" if v is not None else "—"
        def _rp(m):
            p = rs.get(m, {}).get("pct")
            return f"{p:.0f}" if p is not None else "—"

        raw_rows.append([
            str(i), filename[:18],
            f"{sc.get('moire',0):.3f}",       _rv("moire"),       _rp("moire"),
            f"{sc.get('correlation',0):.3f}",  _rv("correlation"), _rp("correlation"),
            f"{sc.get('gradient',0):.3f}",     _rv("gradient"),    _rp("gradient"),
        ])
        bg = {"AUTHENTIC": colors.HexColor("#d5f5e3"),
              "SUSPICIOUS": colors.HexColor("#fef9e7"),
              "COUNTERFEIT": colors.HexColor("#fadbd8")}.get(
              actual, colors.HexColor("#f8f9fa"))
        raw_row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))

    col_w2 = [
        _lw * 0.03,   # #
        _lw * 0.12,   # Filename
        _lw * 0.08,   # Moiré capped
        _lw * 0.08,   # Moiré raw
        _lw * 0.07,   # Moiré pct
        _lw * 0.08,   # Corr capped
        _lw * 0.08,   # Corr raw
        _lw * 0.07,   # Corr pct
        _lw * 0.08,   # Grad capped
        _lw * 0.08,   # Grad raw
        _lw * 0.07,   # Grad pct   (total = 0.84 of lw; leaves breathing room)
    ]
    t2 = Table(raw_rows, colWidths=col_w2, repeatRows=1)
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_BRAND),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_GRAY),
        ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        *raw_row_colors,
    ]))
    story.append(t2)
    story.append(Spacer(1, 5*mm))

    # ── Percentile rank table (metric × rank position) ───────────────────────
    story.append(Paragraph("Percentile Rankings", ss["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BRAND, spaceAfter=5))

    n_samples = len(results)

    def _ordinal(n):
        if 11 <= n <= 13:
            return f"{n}th"
        return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"

    rank_headers = ["Metric"]
    for pos in range(1, n_samples + 1):
        if pos == 1:
            rank_headers.append(f"{_ordinal(pos)}\n(100 %ile)")
        elif pos == n_samples:
            rank_headers.append(f"{_ordinal(pos)}\n(0 %ile)")
        else:
            rank_headers.append(_ordinal(pos))

    rank_rows = [rank_headers]
    for metric, label in [("moire", "Moiré"), ("correlation", "Correlation"), ("gradient", "Gradient")]:
        ranked = sorted(
            results,
            key=lambda r: r.get("raw_scores", {}).get(metric, {}).get("pre_clip") or 0,
            reverse=True,
        )
        row = [label]
        for r in ranked:
            stem = r.get("filename", "—")
            stem = stem.rsplit(".", 1)[0] if "." in stem else stem
            row.append(stem)
        rank_rows.append(row)

    metric_col_w = 2.8 * cm
    rank_col_w   = (_lw - metric_col_w) / n_samples
    rank_col_widths = [metric_col_w] + [rank_col_w] * n_samples

    # Mark counterfeit cells red (tc-09, tc-10 by expected verdict lookup)
    counterfeit_stems = {
        fn.rsplit(".", 1)[0]
        for fn, v in EXPECTED_VERDICTS.items()
        if v == "COUNTERFEIT"
    }
    cell_colors = []
    for row_i, row in enumerate(rank_rows[1:], start=1):
        for col_j, cell in enumerate(row[1:], start=1):
            if cell in counterfeit_stems:
                cell_colors.append(("BACKGROUND", (col_j, row_i), (col_j, row_i), colors.HexColor("#fadbd8")))
                cell_colors.append(("TEXTCOLOR",  (col_j, row_i), (col_j, row_i), colors.HexColor("#a93226")))

    t3 = Table(rank_rows, colWidths=rank_col_widths, repeatRows=1)
    t3.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_BRAND),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND",    (0, 1), (0, -1), C_LIGHT),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_GRAY),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        *cell_colors,
    ]))
    story.append(t3)
    story.append(Spacer(1, 5*mm))

    # ── Calibration caveat footnote ──────────────────────────────────────────
    caveat = (
        "<b>Calibration note:</b> Percentile and gap figures are calibrated "
        "against a batch containing only 2 counterfeit samples (tc-09, tc-10) "
        "— not yet validated against a broader counterfeit sample set.  "
        "Fine-alignment (Step 2c) has a known open finding: when it "
        "successfully applies to a counterfeit sample, corrected scores can "
        "rise enough to cross THRESHOLD_AUTHENTIC — see KNOWN GAP note in "
        "cdp_engine.py.  "
        "<b>Gradient note:</b> gradient raw values show genuine/counterfeit "
        "overlap in this batch (a genuine sample scored lower than a "
        "counterfeit); this is not a capping artefact — the test lacks "
        "discriminating power and needs replacement, not recalibration."
    )
    story.append(Paragraph(caveat, ss["Body"]))

    # Return to portrait for the remaining pages.
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# Per-sample page
# ---------------------------------------------------------------------------

def _build_sample_page(story, ss, r: dict, idx: int, total: int):
    verdict  = r.get("verdict", "ERROR")
    filename = r.get("filename", "unknown")
    seed     = r.get("seed_recovered")
    conf     = r.get("confidence")

    story.append(Paragraph(f"Sample {idx}/{total} — {filename}", ss["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=3))

    # Verdict line
    vc = _verdict_color(verdict)
    # Use hex string directly for color value
    hex_col = vc.hexval() if hasattr(vc, "hexval") else "#333333"
    story.append(Paragraph(
        f'<font color="{hex_col}"><b>{verdict}</b></font>'
        f'  <font size="10">Confidence: {_pct(conf)}</font>',
        ss["VerdictBig"]))
    story.append(Spacer(1, 3*mm))

    # Header table
    dm = r.get("dm_diagnostic", {})
    seed_str = (f"{seed} (0x{seed:08X})" if isinstance(seed, int) else str(seed)) if seed else "—"
    hdr = [
        ["Field", "Value"],
        ["Filename",         filename],
        ["Description",      TEST_CASE_DESCRIPTIONS.get(filename, "—")],
        ["Pattern ID",       str(r.get("pattern_id") or "—")],
        ["Recovered Seed",   seed_str],
        ["Verdict",          verdict],
        ["Confidence",       _pct(conf)],
        ["Alignment Method", r.get("alignment_method", "—")],
        ["Quality Score",    f"{r.get('quality_score', 0):.3f}" if r.get("quality_score") else "—"],
        ["Align Confidence", f"{r.get('align_confidence', 0):.3f}" if r.get("align_confidence") else "—"],
        ["Processing Time",  f"{r.get('processing_time', 0):.2f}s"],
        ["DMs Found",        str(dm.get("num_dms_found", "—"))],
        ["Failure Reason",   r.get("failure_reason") or "—"],
    ]
    story.append(_std_table(hdr, [6*cm, 11*cm]))
    story.append(Spacer(1, 4*mm))

    # 5-image pipeline steps: Detection → Cropped → Aligned → Aligned+fiducials → Reference
    IW = 3.0*cm; IH = 3.0*cm
    def _ti(path):
        return _file_to_rl_image(path, IW, IH)

    imgs = [
        _ti(r.get("detection_path_saved"))  or _placeholder_para("Detection", ss),
        _ti(r.get("captured_path_saved"))   or _placeholder_para("Cropped (as captured)", ss),
        _ti(r.get("aligned_path_saved"))    or _placeholder_para("Aligned (scored)", ss),
        _ti(r.get("fiducial_path_saved"))   or _placeholder_para("Aligned + fiducials", ss),
        _ti(r.get("reference_path"))        or _placeholder_para("Reference (expected)", ss),
    ]
    caps = [Paragraph(c, ss["Caption"]) for c in [
        "Detection", "Cropped (as captured)", "Aligned (scored)",
        "Aligned + fiducials", "Reference (expected)",
    ]]
    img_tbl = Table([imgs, caps], colWidths=[3.3*cm]*5, rowHeights=[IH + 0.2*cm, 0.6*cm])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",   (0, 0), (-1, -1), 0.25, C_LIGHT),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 4*mm))

    # Analytics plots (3 per test × 4 tests)
    scores = r.get("scores") or {}
    if scores:
        story.append(Paragraph("Per-block Analytics", ss["SubHead"]))
        analytics_dir = r.get("analytics_dir", "")
        stem          = r.get("stem", "")

        for test in ["moire", "correlation", "gradient", "color"]:
            story.append(Paragraph(test.upper(), ss["SubHead"]))
            PW = 5.4*cm; PH = 3.8*cm
            plot_imgs = []
            plot_caps = []
            for ptype, cap in [("heatmap", "Heatmap"),
                                ("histogram", "Distribution"),
                                ("scatter", "Per-block Scatter")]:
                path = os.path.join(analytics_dir, f"{stem}_{test}_{ptype}.png")
                img  = _file_to_rl_image(path, PW, PH)
                plot_imgs.append(img or _placeholder_para(f"{test}/{ptype}", ss))
                plot_caps.append(Paragraph(cap, ss["Caption"]))
            ptbl = Table([plot_imgs, plot_caps], colWidths=[PW + 0.2*cm]*3,
                         rowHeights=[PH + 0.2*cm, 0.55*cm])
            ptbl.setStyle(TableStyle([
                ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID",   (0, 0), (-1, -1), 0.25, C_LIGHT),
            ]))
            story.append(ptbl)
            story.append(Spacer(1, 2*mm))

        # Full-width delta map
        story.append(Paragraph("Δ Delta Map", ss["SubHead"]))
        delta_img = _file_to_rl_image(r.get("delta_path"), 17*cm, 5.5*cm)
        if delta_img:
            story.append(delta_img)
        else:
            story.append(Paragraph("Delta map not available.", ss["Body"]))

    # Metrics table — capped score, raw (uncapped) score, batch percentile, weight
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Numeric Metrics", ss["SubHead"]))
    wt = r.get("weights") or {}
    rs = r.get("raw_scores") or {}

    def _raw_val(metric):
        v = rs.get(metric, {}).get("pre_clip")
        return f"{v:.4f}" if v is not None else "—"

    def _pct_rank(metric):
        p = rs.get(metric, {}).get("pct")
        return f"{p:.0f}%" if p is not None else "—"

    mrows = [
        ["Metric", "Capped", "Raw (uncapped)", "Batch %ile", "Weight"],
        ["Moiré",       f"{scores.get('moire',0):.4f}",       _raw_val("moire"),       _pct_rank("moire"),       f"{wt.get('moire',0.65)*100:.0f}%"],
        ["Correlation", f"{scores.get('correlation',0):.4f}", _raw_val("correlation"), _pct_rank("correlation"), f"{wt.get('correlation',0.10)*100:.0f}%"],
        ["Gradient",    f"{scores.get('gradient',0):.4f}",    _raw_val("gradient"),    _pct_rank("gradient"),    f"{wt.get('gradient',0.15)*100:.0f}%"],
        ["Color",       f"{scores.get('color',0):.4f}",       "—",                     "—",                      f"{wt.get('color',0.10)*100:.0f}%"],
        ["Confidence",  _pct(conf),                            "—",                     "—",                      "—"],
    ]
    mt = Table(mrows, colWidths=[3.5*cm, 2.5*cm, 3.5*cm, 2.5*cm, 2*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_BRAND),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.25, C_GRAY),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(mt)
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# Aggregate analysis section
# ---------------------------------------------------------------------------

def _build_aggregate_section(story, ss, charts: dict):
    story.append(Paragraph("Aggregate Analysis", ss["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BRAND, spaceAfter=5))

    chart_defs = [
        ("verdict_dist",      "Verdict Distribution",           True),
        ("confidence_dist",   "Confidence Score Distribution",  False),
        ("score_boxplot",     "Score Comparison (All Metrics)", False),
        ("moire_dist",        "Moiré Score Distribution",       False),
        ("correlation_dist",  "Correlation Score Distribution", False),
        ("gradient_dist",     "Gradient Score Distribution",    False),
        ("color_dist",        "Color Score Distribution",       False),
        ("time_dist",         "Processing Time Distribution",   False),
        ("failure_breakdown", "Failure Reason Breakdown",       False),
    ]
    for key, caption, small in chart_defs:
        if key not in charts:
            continue
        story.append(Paragraph(caption, ss["SubHead"]))
        mw = 9*cm  if small else 16*cm
        mh = 8*cm  if small else 6*cm
        story.append(_png_bytes_to_rl_image(charts[key], mw, mh))
        story.append(Spacer(1, 4*mm))


# ---------------------------------------------------------------------------
# Batch-level percentile helper
# ---------------------------------------------------------------------------

def _add_percentiles(results: list):
    """
    Compute batch-rank percentile for each metric's pre_clip raw score and
    store it in-place: result["raw_scores"][metric]["pct"] = 0–100.

    Percentile = rank / (N-1) * 100 where rank=0 is the lowest raw score
    in the batch.  Results with no raw_scores entry for a metric are skipped.
    """
    for metric in ("moire", "correlation", "gradient"):
        indexed = [
            (i, r.get("raw_scores", {}).get(metric, {}).get("pre_clip"))
            for i, r in enumerate(results)
        ]
        valid = [(i, v) for i, v in indexed if v is not None]
        n = len(valid)
        if n < 2:
            continue
        ranked = sorted(valid, key=lambda x: x[1])
        for rank, (i, _) in enumerate(ranked):
            pct = rank / (n - 1) * 100.0
            results[i].setdefault("raw_scores", {}).setdefault(metric, {})["pct"] = pct


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_report_pdf(output_path: str, results: list, batch_stats: dict, run_ts: str):
    """
    Assemble the complete management-ready PDF report.

    Args:
        output_path:  Destination file path (must end in .pdf).
        results:      List of per-image result dicts from batch_test_runner.
        batch_stats:  Aggregate statistics dict.
        run_ts:       Human-readable run timestamp string.
    """
    ss = _styles()
    w, h = A4
    top_m = 28*mm; bot_m = 18*mm; side_m = 15*mm

    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        leftMargin=side_m, rightMargin=side_m,
        topMargin=top_m, bottomMargin=bot_m,
    )
    frame = Frame(side_m, bot_m, w - 2*side_m, h - top_m - bot_m, id="normal")

    lw, lh = landscape(A4)
    lframe = Frame(side_m, bot_m, lw - 2*side_m, lh - top_m - bot_m, id="normal")

    doc.addPageTemplates([
        PageTemplate(id="main",      frames=[frame],  onPage=_add_header_footer),
        PageTemplate(id="landscape", frames=[lframe], onPage=_add_header_footer,
                     pagesize=landscape(A4)),
    ])

    _add_percentiles(results)

    story = []
    _build_cover(story, ss, batch_stats, run_ts)
    _build_exec_summary(story, ss, results, batch_stats)
    _build_results_table(story, ss, results)

    for i, r in enumerate(results, 1):
        _build_sample_page(story, ss, r, i, len(results))

    charts = build_aggregate_charts(results)
    _build_aggregate_section(story, ss, charts)

    doc.build(story)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"[PDF] Written: {output_path}  ({size_kb} KB)")
