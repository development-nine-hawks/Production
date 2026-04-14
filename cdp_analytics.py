"""
Per-block scoring + research-paper-style plots for verifications.

Computes, for each 16x16 (or N x N) block in the aligned vs reference image:
  - moire        : per-block NCC of bandpass-filtered intensities
  - correlation  : per-block NCC of raw intensities
  - gradient     : per-block NCC of Sobel-gradient magnitudes
  - color        : 1 - (per-block mean RGB Euclidean distance / 100)

Also renders matplotlib plots:
  - heatmap (overlay on aligned image)
  - scatter (block index vs score)
  - histogram (distribution of per-block scores)
  - delta map (|aligned - reference|, pixel-level)

And assembles everything into a single PDF report.
"""

import io
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Per-block scoring
# ---------------------------------------------------------------------------

def _per_block_ncc(a, b, bs):
    """Per-block normalized cross-correlation. a and b must be same shape, 2D."""
    h, w = a.shape
    bh, bw = h // bs, w // bs
    scores = np.zeros((bh, bw), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            ab = a[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].astype(np.float64).ravel()
            bb = b[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].astype(np.float64).ravel()
            ab -= ab.mean()
            bb -= bb.mean()
            an = np.linalg.norm(ab) + 1e-10
            bn = np.linalg.norm(bb) + 1e-10
            scores[i, j] = float((ab @ bb) / (an * bn))
    return np.clip(scores, -1.0, 1.0)


def _block_means_rgb(img, bs):
    h, w = img.shape[:2]
    bh, bw = h // bs, w // bs
    out = np.zeros((bh, bw, 3), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            out[i, j] = img[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].mean(axis=(0, 1))
    return out


def compute_per_block_scores(aligned_gray, reference_gray,
                             aligned_rgb, reference_rgb, block_size=16):
    """Compute per-block scores for all 4 verification tests.

    Returns: dict with keys 'moire', 'correlation', 'gradient', 'color',
             each a 2D numpy array of shape (H/bs, W/bs), values in [0, 1]
             where 1 = perfect match, 0 = no match.
    """
    rh, rw = reference_gray.shape[:2]
    if aligned_gray.shape[:2] != (rh, rw):
        aligned_gray = cv2.resize(aligned_gray, (rw, rh), interpolation=cv2.INTER_AREA)
        aligned_rgb = cv2.resize(aligned_rgb, (rw, rh), interpolation=cv2.INTER_AREA)

    # 1. Correlation: per-block NCC on raw intensity
    corr = _per_block_ncc(aligned_gray, reference_gray, block_size)
    correlation = (corr + 1.0) / 2.0

    # 2. Moire: bandpass filter both images, then per-block NCC
    def bandpass(img):
        a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
        b = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 4.0)
        return a - b
    aligned_bp = bandpass(aligned_gray)
    ref_bp = bandpass(reference_gray)
    m = _per_block_ncc(aligned_bp, ref_bp, block_size)
    moire = (m + 1.0) / 2.0

    # 3. Gradient: Sobel magnitude per pixel, NCC per block
    def grad_mag(img):
        gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    aligned_g = grad_mag(aligned_gray)
    ref_g = grad_mag(reference_gray)
    g = _per_block_ncc(aligned_g, ref_g, block_size)
    gradient = (g + 1.0) / 2.0

    # 4. Color: per-block mean RGB Euclidean distance
    a_means = _block_means_rgb(aligned_rgb, block_size)
    r_means = _block_means_rgb(reference_rgb, block_size)
    color_diff = np.linalg.norm(a_means - r_means, axis=-1)
    # Normalize: 100 RGB-units of distance = full mismatch
    color = 1.0 - np.clip(color_diff / 100.0, 0, 1)

    return {
        "moire":       moire.astype(np.float32),
        "correlation": correlation.astype(np.float32),
        "gradient":    gradient.astype(np.float32),
        "color":       color.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Plot rendering — each returns PNG bytes
# ---------------------------------------------------------------------------

TEST_DESCRIPTIONS = {
    "moire":       ("Moiré (bandpass NCC)",       0.65, "Frequency-band match"),
    "correlation": ("Correlation (block NCC)",    0.10, "Block-level intensity match"),
    "gradient":    ("Gradient (Sobel NCC)",       0.15, "Edge structure match"),
    "color":       ("Color (RGB distance)",       0.10, "Per-block colour match"),
}


def _save_fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_heatmap(scores_2d, test_name, aligned_gray=None):
    """Heatmap of per-block scores; if aligned_gray given, overlay on it."""
    title, weight, desc = TEST_DESCRIPTIONS[test_name]
    bh, bw = scores_2d.shape
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=110)

    if aligned_gray is not None:
        # Show the aligned image as a grey backdrop, overlay heatmap
        ax.imshow(aligned_gray, cmap="gray",
                  extent=(0, bw, bh, 0), aspect="auto", alpha=0.35)
        im = ax.imshow(scores_2d, cmap="RdYlGn", vmin=0, vmax=1,
                       extent=(0, bw, bh, 0), aspect="auto", alpha=0.75)
    else:
        im = ax.imshow(scores_2d, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Match score (0 = mismatch, 1 = match)", fontsize=9)

    mean_score = float(scores_2d.mean())
    pass_rate = float((scores_2d >= 0.5).mean()) * 100.0
    ax.set_title(f"{title}   |   weight {int(weight*100)}%\n"
                 f"mean={mean_score:.3f}   blocks≥0.5: {pass_rate:.1f}%",
                 fontsize=11)
    ax.set_xlabel(f"Block column   ({bw} cols)", fontsize=9)
    ax.set_ylabel(f"Block row   ({bh} rows)", fontsize=9)
    fig.tight_layout()
    return _save_fig_png(fig)


def render_scatter(scores_2d, test_name):
    title, weight, desc = TEST_DESCRIPTIONS[test_name]
    flat = scores_2d.ravel()
    n = len(flat)
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
    colors = ["#d7263d" if s < 0.5 else "#1b998b" for s in flat]
    ax.scatter(np.arange(n), flat, c=colors, s=8, alpha=0.7, edgecolors="none")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.axhline(flat.mean(), color="#0066cc", linestyle="-", linewidth=1.2,
               label=f"mean = {flat.mean():.3f}")
    ax.set_xlim(-1, n)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"Block index (row-major, {n} blocks total)", fontsize=9)
    ax.set_ylabel("Per-block match score", fontsize=9)
    ax.set_title(f"{title}   |   per-block scatter", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_fig_png(fig)


def render_histogram(scores_2d, test_name):
    title, weight, desc = TEST_DESCRIPTIONS[test_name]
    flat = scores_2d.ravel()
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
    ax.hist(flat, bins=40, range=(0, 1), color="#3a86ff",
            edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="#d7263d", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.axvline(flat.mean(), color="#1b998b", linestyle="-", linewidth=1.4,
               label=f"mean = {flat.mean():.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-block match score", fontsize=9)
    ax.set_ylabel("Block count", fontsize=9)
    ax.set_title(f"{title}   |   distribution", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return _save_fig_png(fig)


def render_delta_map(aligned_gray, reference_gray):
    """Pixel-level absolute difference map."""
    rh, rw = reference_gray.shape[:2]
    if aligned_gray.shape[:2] != (rh, rw):
        aligned_gray = cv2.resize(aligned_gray, (rw, rh), interpolation=cv2.INTER_AREA)
    delta = np.abs(aligned_gray.astype(np.int16) - reference_gray.astype(np.int16)).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=110)
    axes[0].imshow(reference_gray, cmap="gray")
    axes[0].set_title("Reference (original)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(aligned_gray, cmap="gray")
    axes[1].set_title("Captured (aligned)", fontsize=11)
    axes[1].axis("off")

    im = axes[2].imshow(delta, cmap="hot", vmin=0, vmax=255)
    axes[2].set_title(f"|Δ|   mean={delta.mean():.1f}   max={delta.max()}", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save_fig_png(fig)


# ---------------------------------------------------------------------------
# Full-report PDF assembly
# ---------------------------------------------------------------------------

def render_full_report_pdf(verification, scores_dict,
                           aligned_gray=None, reference_gray=None):
    """Build a single multi-page PDF containing every plot for this verification.

    `verification` is a dict like the MongoDB doc — needs id, verdict, confidence,
    score_moire/color/correlation/gradient, alignment_method, created_at.
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover page: summary
        fig = plt.figure(figsize=(8.5, 11), dpi=110)
        fig.suptitle("CDP Verification Analytics Report", fontsize=18, y=0.96)
        ax = fig.add_subplot(111)
        ax.axis("off")

        v = verification
        agg = (
            f"Verification ID:   {v.get('id')}\n"
            f"Verdict:           {v.get('verdict')}\n"
            f"Confidence:        {v.get('confidence', 0)*100:.1f}%\n"
            f"Alignment:         {v.get('alignment_method', 'n/a')}\n"
            f"Markers found:     {v.get('markers_found', 'n/a')} / 4\n"
            f"Print size:        {v.get('print_size_mm', 'n/a')} mm\n"
            f"Generated at:      {v.get('created_at', '')}\n\n"
            f"--- Aggregate scores (full image) ---\n"
            f"  Moiré        : {v.get('score_moire', 0):.3f}    (weight 65%)\n"
            f"  Correlation  : {v.get('score_correlation', 0):.3f}    (weight 10%)\n"
            f"  Gradient     : {v.get('score_gradient', 0):.3f}    (weight 15%)\n"
            f"  Color        : {v.get('score_color', 0):.3f}    (weight 10%)\n\n"
            f"--- Per-block summary ---"
        )
        for k in ["moire", "correlation", "gradient", "color"]:
            arr = scores_dict[k]
            agg += (f"\n  {k:12s} : mean={arr.mean():.3f}  "
                    f"min={arr.min():.3f}  max={arr.max():.3f}  "
                    f"≥0.5: {(arr >= 0.5).mean()*100:.1f}%")
        ax.text(0.05, 0.95, agg, fontsize=11, family="monospace",
                verticalalignment="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # One page per (test × plot type)
        for test in ["moire", "correlation", "gradient", "color"]:
            arr = scores_dict[test]

            # Heatmap
            png = render_heatmap(arr, test, aligned_gray=aligned_gray)
            fig = plt.figure(figsize=(8.5, 7.5))
            fig.figimage(plt.imread(io.BytesIO(png)), resize=True)
            pdf.savefig(fig)
            plt.close(fig)

            # Scatter
            png = render_scatter(arr, test)
            fig = plt.figure(figsize=(9.5, 5.5))
            fig.figimage(plt.imread(io.BytesIO(png)), resize=True)
            pdf.savefig(fig)
            plt.close(fig)

            # Histogram
            png = render_histogram(arr, test)
            fig = plt.figure(figsize=(9.5, 5.5))
            fig.figimage(plt.imread(io.BytesIO(png)), resize=True)
            pdf.savefig(fig)
            plt.close(fig)

        # Delta map (last page)
        if aligned_gray is not None and reference_gray is not None:
            png = render_delta_map(aligned_gray, reference_gray)
            fig = plt.figure(figsize=(14, 6))
            fig.figimage(plt.imread(io.BytesIO(png)), resize=True)
            pdf.savefig(fig)
            plt.close(fig)

    buf.seek(0)
    return buf.read()
