"""
PhoneCDP Production Web Application
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import sys
import uvicorn
import shutil
import csv
import io
import os
import time
import tempfile
from datetime import datetime

import config
from database import (
    init_db, get_db, get_next_id, is_db_available,
    upload_to_cloudinary, destroy_cloudinary, download_from_cloudinary,
)
from cdp_engine import generate_pattern, verify_pattern, verify_pattern_legacy

app = FastAPI(title="PhoneCDP", version="1.0.0")

app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")

# Offline CDN — serves images stored locally when Cloudinary is unreachable
_local_cdn_dir = os.path.join(config.DATA_DIR, "localcdn")
os.makedirs(_local_cdn_dir, exist_ok=True)
app.mount("/local-cdn", StaticFiles(directory=_local_cdn_dir), name="local-cdn")


# ── HTML ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(
        os.path.join(config.BASE_DIR, "templates", "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(config.BASE_DIR, "static", "img", "logo.svg"),
                        media_type="image/svg+xml")


# ── Health (for Render) ──────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Patterns API ─────────────────────────────────────────────────────────

class PatternCreate(BaseModel):
    seed: Optional[int] = None
    serial_number: str = "SN-0001"
    label: str = ""
    notes: str = ""
    pattern_size: int = 512
    block_size: int = 16


@app.get("/api/patterns")
def list_patterns():
    db = get_db()
    patterns = list(db.patterns.find({}, {"_id": 0}).sort("created_at", -1))
    for p in patterns:
        p["verification_count"] = db.verifications.count_documents({"pattern_id": p["id"]})
        p["created_at"] = p["created_at"].isoformat() if hasattr(p["created_at"], "isoformat") else str(p["created_at"])
    return patterns


@app.post("/api/patterns/generate")
def create_pattern(data: PatternCreate):
    db = get_db()
    result = generate_pattern(
        output_dir=config.PATTERNS_DIR, seed=data.seed,
        serial_number=data.serial_number, pattern_size=data.pattern_size,
        block_size=data.block_size)

    # Upload to Cloudinary
    cloud = upload_to_cloudinary(result["filepath"], folder="phonecdp/patterns")

    pid = get_next_id("patterns")
    doc = {
        "id": pid,
        "seed": result["seed"],
        "serial_number": data.serial_number,
        "label": data.label,
        "filename": result["filename"],
        "image_url": cloud["secure_url"],
        "cloudinary_public_id": cloud["public_id"],
        "pattern_size": data.pattern_size,
        "block_size": data.block_size,
        "notes": data.notes,
        "created_at": datetime.utcnow(),
    }
    db.patterns.insert_one(doc)

    # Clean up local file
    try:
        os.remove(result["filepath"])
    except OSError:
        pass

    return {
        "id": pid, "seed": result["seed"], "serial_number": data.serial_number,
        "label": data.label, "filename": result["filename"],
        "pattern_size": data.pattern_size, "block_size": data.block_size,
        "created_at": doc["created_at"].isoformat(), "notes": data.notes,
        "image_url": cloud["secure_url"],
    }


@app.get("/api/patterns/{pid}")
def get_pattern(pid: int):
    db = get_db()
    p = db.patterns.find_one({"id": pid}, {"_id": 0})
    if not p:
        raise HTTPException(404, "Pattern not found")
    p["verification_count"] = db.verifications.count_documents({"pattern_id": pid})
    p["created_at"] = p["created_at"].isoformat() if hasattr(p["created_at"], "isoformat") else str(p["created_at"])
    return p


@app.get("/api/patterns/{pid}/download")
def download_pattern(pid: int):
    db = get_db()
    p = db.patterns.find_one({"id": pid})
    if not p:
        raise HTTPException(404, "Pattern not found")
    # Download from Cloudinary and serve with attachment disposition
    # so the browser downloads instead of opening in a new tab.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.DATA_DIR)
    tmp.close()
    download_from_cloudinary(p["image_url"], tmp.name)
    filename = p.get("filename", f"{p['serial_number']}.png")
    return FileResponse(tmp.name, media_type="image/png", filename=filename)


@app.get("/api/patterns/{pid}/pdf")
def download_pattern_pdf(
    pid: int,
    size_mm: float = Query(..., description="Pattern size in mm"),
):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    db = get_db()
    p = db.patterns.find_one({"id": pid})
    if not p:
        raise HTTPException(404, "Pattern not found")

    # Download pattern image from Cloudinary to temp file
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.DATA_DIR)
    tmp_img.close()
    download_from_cloudinary(p["image_url"], tmp_img.name)

    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=config.DATA_DIR)
    tmp_pdf.close()

    page_w, page_h = A4
    pattern_size_pts = size_mm * mm

    c = canvas.Canvas(tmp_pdf.name, pagesize=A4)
    x = (page_w - pattern_size_pts) / 2
    y = (page_h - pattern_size_pts) / 2

    img = ImageReader(tmp_img.name)
    c.drawImage(img, x, y, width=pattern_size_pts, height=pattern_size_pts)

    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setLineWidth(0.25)
    c.rect(x, y, pattern_size_pts, pattern_size_pts)

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(15 * mm, page_h - 12 * mm,
                 f"PhoneCDP | {p['serial_number']} | "
                 f"{size_mm}mm x {size_mm}mm | Seed: {p['seed']}")

    c.save()

    # Clean up temp image
    try:
        os.remove(tmp_img.name)
    except OSError:
        pass

    pdf_filename = f"{p['serial_number']}_{size_mm}mm.pdf"
    return FileResponse(tmp_pdf.name, media_type="application/pdf",
                        filename=pdf_filename)


@app.get("/api/patterns/{pid}/label-pdf")
def download_pattern_label_pdf(
    pid: int,
    size_mm: float = Query(..., description="Pattern size in mm"),
):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    db = get_db()
    p = db.patterns.find_one({"id": pid})
    if not p:
        raise HTTPException(404, "Pattern not found")

    # Render the label PNG background in-process, without OpenCV Auth Block.
    tmp_label_name, extra, tmp_img_name = _render_label_png_file(p, 512, pdf_mode=True)
    qr_x1, qr_y1, qr_w_found, qr_h_found, layout_px = extra

    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=config.DATA_DIR)
    tmp_pdf.close()

    # Get label image dimensions to calculate scaling
    import cv2
    label_img = cv2.imread(tmp_label_name)
    if label_img is None:
        raise HTTPException(500, "Failed to read label image")
    label_h_px, label_w_px = label_img.shape[:2]

    # We want the pattern inside the Auth Block to be EXACTLY size_mm in the PDF.
    # The pattern in layout_px has size layout_px["pattern_rect"][2] (which is pw).
    # Since _render_label_png_file may have scaled the placeholder, we find pat_w_px:
    pat_w_px = (qr_w_found / layout_px["auth_w"]) * layout_px["pattern_rect"][2]
    scale = (size_mm * mm) / pat_w_px  # points per pixel
    label_w_pts = label_w_px * scale
    label_h_pts = label_h_px * scale

    # Center on A4
    page_w, page_h = A4
    ox = (page_w - label_w_pts) / 2
    oy = (page_h - label_h_pts) / 2

    c = canvas.Canvas(tmp_pdf.name, pagesize=A4)
    img = ImageReader(tmp_label_name)
    c.drawImage(img, ox, oy, width=label_w_pts, height=label_h_pts)
    
    # Calculate bottom-left coordinate of the hole in ReportLab coordinates
    hole_x_pts = ox + qr_x1 * scale
    qr_y2 = qr_y1 + qr_h_found
    hole_y_pts = oy + (label_h_px - qr_y2) * scale
    
    from cdp_engine import generate_cropped_dm, split_seed_for_dm, calculate_auth_block_layout
    share_a, share_b = split_seed_for_dm(p['seed'])
    top_dm_img, top_dm_mods = generate_cropped_dm(share_a)
    right_dm_img, right_dm_mods = generate_cropped_dm(share_b)

    # Scale layout directly into the hole
    sx = qr_w_found / layout_px["auth_w"]
    pw_px = layout_px["pattern_rect"][2]
    qx_px = layout_px["right_dm_rect"][0] - pw_px
    scaled_layout = calculate_auth_block_layout(
        pw_px * sx, qx_px * sx,
        top_dm_img, right_dm_img,
        top_dm_modules=top_dm_mods, right_dm_modules=right_dm_mods,
    )
    
    # Draw pattern
    px, py, pw, ph = scaled_layout["pattern_rect"]
    auth_h_scaled = scaled_layout["auth_h"]
    # ReportLab uses y-up coordinates; OpenCV uses y-down. Convert carefully.
    c.drawImage(ImageReader(tmp_img_name), 
                hole_x_pts + px * scale, 
                hole_y_pts + (auth_h_scaled - py - ph) * scale, 
                width=pw * scale, height=ph * scale)
                
    # Helper to draw DM
    def draw_dm(dm_img, bx, by, bw, bh):
        rows, cols = dm_img.shape
        mod_w = bw / cols
        mod_h = bh / rows
        c.setFillColorRGB(0, 0, 0)
        for r in range(rows):
            for col in range(cols):
                if dm_img[r, col] < 128:
                    mx = bx + col * mod_w
                    my = by + r * mod_h
                    # ReportLab y = hole_y + (auth_h - my - mod_h)
                    pdf_x = hole_x_pts + mx * scale
                    pdf_y = hole_y_pts + (auth_h_scaled - my - mod_h) * scale
                    c.rect(pdf_x, pdf_y, mod_w * scale, mod_h * scale, stroke=0, fill=1)

    tx, ty, tw, th = scaled_layout["top_dm_rect"]
    draw_dm(top_dm_img, tx, ty, tw, th)
    
    rot_right_dm = cv2.rotate(right_dm_img, cv2.ROTATE_90_CLOCKWISE)
    rx, ry, rw, rh = scaled_layout["right_dm_rect"]
    draw_dm(rot_right_dm, rx, ry, rw, rh)

    c.save()


    # Clean up
    try:
        os.remove(tmp_label_name)
        os.remove(tmp_img_name)
    except OSError:
        pass

    pdf_filename = f"{p['serial_number']}_label_{size_mm}mm.pdf"
    return FileResponse(tmp_pdf.name, media_type="application/pdf",
                        filename=pdf_filename)


def _render_label_png_file(p: dict, size_px: int = 512, pdf_mode: bool = False):
    """Render the NINEHAWK branded label PNG for a pattern, in-process.

    Returns the path to a temp PNG file. Shared by the label-png and
    label-pdf endpoints so neither makes an HTTP call back to this same
    server. (That self-call broke on Render: config.PORT reads the env
    PORT Render injects (10000), but gunicorn is hardcoded to bind 8000,
    so the call hit a closed port and 500'd.)
    """
    import cv2
    import numpy as np
    import base64
    from cdp_engine import calculate_auth_block_layout, generate_cropped_dm, split_seed_for_dm, draw_auth_block_opencv

    # Download pattern image from Cloudinary
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.DATA_DIR)
    tmp_img.close()
    download_from_cloudinary(p["image_url"], tmp_img.name)

    pat_bgr = cv2.imread(tmp_img.name)
    if pat_bgr is None:
        raise HTTPException(500, "Failed to load pattern image")
    pat_bgr = cv2.resize(pat_bgr, (size_px, size_px), interpolation=cv2.INTER_AREA)


    # Load banner as base64
    banner_b64_path = os.path.join(config.BASE_DIR, "static", "img", "ninehawk_banner_b64.txt")
    banner_img_path = os.path.join(config.BASE_DIR, "static", "img", "ninehawk_banner.png")
    if not os.path.exists(banner_b64_path) and os.path.exists(banner_img_path):
        with open(banner_img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        with open(banner_b64_path, "w") as f:
            f.write(b64)
    with open(banner_b64_path) as f:
        banner_b64 = f.read()

    # Calculate layout
    pat = size_px
    quiet_px = int(0.5 * pat / 7.5)
    share_a, share_b = split_seed_for_dm(p['seed'])
    top_dm_img, top_dm_mods = generate_cropped_dm(share_a)
    right_dm_img, right_dm_mods = generate_cropped_dm(share_b)

    layout = calculate_auth_block_layout(
        pat, quiet_px,
        top_dm_img, right_dm_img,
        top_dm_modules=top_dm_mods, right_dm_modules=right_dm_mods,
    )
    auth_w = int(layout['auth_w'])
    auth_h = int(layout['auth_h'])

    # Encode QR pattern as base64 PNG (use a magenta placeholder, overlay real pixels later)
    placeholder = np.zeros((auth_h, auth_w, 3), dtype=np.uint8)
    placeholder[:] = (255, 0, 255)  # magenta - easy to find
    _, buf = cv2.imencode('.png', placeholder)
    qr_b64 = base64.b64encode(buf).decode()

    # Build HTML label with layered absolute positioning
    card_w = max(int(pat * 1.70), int(auth_w * 1.3))
    card_h = max(int(pat * 3.33), int(auth_h * 2.2))
    frame_pad = int(pat * 0.1446)
    border_gap = int(pat * 0.04)  # padding from card edge to the single border
    cr = int(pat * 0.08)
    ban_h = int(pat * 0.42)
    ban_top = int(pat * 0.16)

    # Calculate QR frame position (from reference: ~65% down the card)
    frame_w = auth_w + 2 * frame_pad
    frame_h = auth_h + 2 * frame_pad
    frame_left = (card_w - frame_w) // 2
    frame_top = int(card_h * 0.58)
    if frame_top + frame_h > card_h - frame_pad:
        frame_top = card_h - frame_h - frame_pad

    # Watermark repeat count to fill card height
    wm_item_h = int(pat * 0.24)
    wm_count = (card_h // wm_item_h) + 2

    html = f"""<!DOCTYPE html>
<html><head><style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ width: {card_w}px; height: {card_h}px; overflow: hidden; background: transparent; margin: 0; }}

  /* Layer 6: Card - outer container */
  .card {{
    width: {card_w}px; height: {card_h}px;
    background: white;
    border: none;
    border-radius: {cr}px;
    position: relative;
    overflow: hidden;
  }}
  .card-inner-border {{
    position: absolute;
    top: {border_gap}px; left: {border_gap}px;
    right: {border_gap}px; bottom: {border_gap}px;
    border: 20px solid #222;
    border-radius: {max(1, cr - border_gap)}px;
    pointer-events: none;
    z-index: 0;
  }}

  /* Layer 4: Watermarks - repeating column, clipped to card */
  .watermarks {{
    position: absolute;
    top: {ban_top + ban_h + int(pat*0.05)}px;
    left: 0; right: 0;
    bottom: {int(pat * 0.15)}px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: {int(pat * 0.12)}px;
    z-index: 1;
    overflow: hidden;
  }}
  .watermarks img {{
    height: {wm_item_h}px;
    flex-shrink: 0;
    filter: invert(1);
    mix-blend-mode: multiply;
    opacity: 0.18;
  }}

  /* Layer 5: Black banner */
  .banner {{
    position: absolute;
    top: {ban_top + border_gap + 20}px;
    left: {int(pat * 0.05) + border_gap}px;
    right: {int(pat * 0.05) + border_gap}px;
    height: {ban_h}px;
    background: black;
    border-radius: {int(pat * 0.05)}px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
  }}
  .banner img {{ height: 45%; object-fit: contain; }}

  /* Layer 3: "SCAN TO AUTHENTICATE" text */
  .scan-text {{
    position: absolute;
    top: {frame_top - int(pat * 0.35)}px;
    left: 0; right: 0;
    text-align: center;
    font-family: 'Arial Black', 'Impact', sans-serif;
    font-weight: 900;
    font-size: {int(pat * 0.09)}px;
    color: black;
    line-height: 1.5;
    letter-spacing: 2px;
    z-index: 2;
  }}

  /* Layer 2: QR container (white bg covers watermarks) */
  .qr-container {{
    position: absolute;
    top: {frame_top}px;
    left: {frame_left}px;
    width: {frame_w}px;
    height: {frame_h}px;
    background: white;
    border: 10px solid #333;
    border-radius: {int(pat * 0.03)}px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 3;
  }}

  /* Layer 1: QR code */
  .qr-code {{
    border: none;
    line-height: 0;
  }}
  .qr-code img {{
    width: {auth_w}px;
    height: {auth_h}px;
    display: block;
  }}
</style></head>
<body>
<div class="card">
  <!-- Layer 6: inner border -->
  <div class="card-inner-border"></div>

  <!-- Layer 4: Watermark strip (repeating, behind everything) -->
  <div class="watermarks">
    {"".join(f'<img src="data:image/png;base64,{banner_b64}" />' for _ in range(wm_count))}
  </div>

  <!-- Layer 5: Black banner (covers watermarks) -->
  <div class="banner">
    <img src="data:image/png;base64,{banner_b64}" />
  </div>

  <!-- Layer 3: Scan text -->
  <div class="scan-text">SCAN TO<br>AUTHENTICATE</div>

  <!-- Layer 2: QR container (white bg covers watermarks) -->
  <div class="qr-container">
    <!-- Layer 1: QR code -->
    <div class="qr-code">
      <img src="data:image/png;base64,{qr_b64}" />
    </div>
  </div>
</div>
</body>
</html>"""

    # Render HTML to PNG via a subprocess so Playwright's Chromium launch runs
    # in a fresh process with its own event loop — avoids the Windows asyncio
    # SelectorEventLoop limitation when called from FastAPI's thread pool.
    import subprocess
    tmp_html_src = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, dir=config.DATA_DIR, mode="w", encoding="utf-8"
    )
    tmp_html_src.write(html)
    tmp_html_src.close()

    tmp_html_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.DATA_DIR)
    tmp_html_out.close()

    _render_script = os.path.join(config.BASE_DIR, "_playwright_render.py")
    result = subprocess.run(
        [sys.executable, _render_script,
         tmp_html_src.name, tmp_html_out.name, str(card_w), str(card_h)],
        capture_output=True, text=True, timeout=60,
    )
    os.unlink(tmp_html_src.name)
    if result.returncode != 0:
        raise RuntimeError(f"Playwright render failed: {result.stderr}")

    # Load rendered label, find magenta placeholder, replace with real QR pixels
    label = cv2.imread(tmp_html_out.name)
    # Find magenta region (BGR: 255, 0, 255)
    magenta_mask = (label[:, :, 0] > 200) & (label[:, :, 1] < 50) & (label[:, :, 2] > 200)
    ys, xs = np.where(magenta_mask)
    if len(ys) > 0:
        qr_y1, qr_x1 = ys.min(), xs.min()
        qr_y2, qr_x2 = ys.max() + 1, xs.max() + 1
        qr_h_found = qr_y2 - qr_y1
        qr_w_found = qr_x2 - qr_x1
        
        if pdf_mode:
            # Leave a white hole for ReportLab
            label[qr_y1:qr_y2, qr_x1:qr_x2] = (255, 255, 255)
            pdf_extra = (qr_x1, qr_y1, qr_w_found, qr_h_found, layout)
        else:
            # Resize layout to match the placeholder exactly
            sx = qr_w_found / auth_w
            scaled_layout = calculate_auth_block_layout(
                pat * sx, quiet_px * sx,
                top_dm_img, right_dm_img,
                top_dm_modules=top_dm_mods, right_dm_modules=right_dm_mods,
            )
            auth_canvas = np.ones((qr_h_found, qr_w_found, 3), dtype=np.uint8) * 255
            draw_auth_block_opencv(auth_canvas, scaled_layout, pat_bgr, top_dm_img, right_dm_img)
            label[qr_y1:qr_y2, qr_x1:qr_x2] = auth_canvas
            pdf_extra = None

    # Save final label PNG with 300 DPI metadata so printer drivers do NOT
    # rescale the image with bilinear interpolation (which would introduce grey
    # anti-aliased edge pixels around every DM module).
    tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.DATA_DIR)
    tmp_out.close()
    try:
        from PIL import Image as PILImage
        pil_label = PILImage.fromarray(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))
        pil_label.save(tmp_out.name, dpi=(300, 300))
    except ImportError:
        # Pillow not available — fall back to cv2 (no DPI metadata)
        cv2.imwrite(tmp_out.name, label)


    if pdf_mode:
        return tmp_out.name, pdf_extra, tmp_img.name

    # Clean up pattern temp
    try:
        os.remove(tmp_img.name)
    except OSError:
        pass

    return tmp_out.name


@app.get("/api/patterns/{pid}/label-png")
def download_pattern_label_png(
    pid: int,
    size_px: int = Query(512, description="Pattern size in pixels"),
):
    db = get_db()
    p = db.patterns.find_one({"id": pid})
    if not p:
        raise HTTPException(404, "Pattern not found")

    out_path = _render_label_png_file(p, size_px)
    png_filename = f"{p['serial_number']}_label.png"
    return FileResponse(out_path, media_type="image/png",
                        filename=png_filename)


@app.get("/api/patterns/{pid}/preview")
def preview_pattern(pid: int):
    db = get_db()
    p = db.patterns.find_one({"id": pid})
    if not p:
        raise HTTPException(404, "Pattern not found")
    return RedirectResponse(url=p["image_url"])


@app.delete("/api/patterns/{pid}")
def delete_pattern(pid: int):
    db = get_db()
    p = db.patterns.find_one({"id": pid})
    if not p:
        raise HTTPException(404, "Pattern not found")

    # Delete pattern image from Cloudinary
    if p.get("cloudinary_public_id"):
        destroy_cloudinary(p["cloudinary_public_id"])

    # Delete captured images from Cloudinary
    for v in db.verifications.find({"pattern_id": pid}):
        cid = (v.get("cloudinary_ids") or {}).get("captured")
        if cid:
            destroy_cloudinary(cid)

    db.verifications.delete_many({"pattern_id": pid})
    db.patterns.delete_one({"id": pid})
    return {"message": "Deleted"}


def cleanup_old_uploads(max_age_seconds=86400):
    """Delete files in uploads/ older than max_age_seconds (default 1 day)."""
    now = time.time()
    for fname in os.listdir(config.UPLOADS_DIR):
        fpath = os.path.join(config.UPLOADS_DIR, fname)
        try:
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
                os.remove(fpath)
        except OSError:
            pass


# ── Verify API ───────────────────────────────────────────────────────────

def _safe_diag(diag):
    """Return a JSON/BSON-safe copy of a dm_diagnostic dict: drop the internal
    `raw_results` (pylibdmtx Decoded objects containing bytes) and decode any
    stray bytes. Without this, returning/storing the diagnostic raises
    'Object of type bytes is not JSON serializable'."""
    if not isinstance(diag, dict):
        return {}
    out = {}
    for k, val in diag.items():
        if k == "raw_results":
            continue
        if isinstance(val, bytes):
            out[k] = val.decode("utf-8", "replace")
        elif isinstance(val, list):
            out[k] = [x.decode("utf-8", "replace") if isinstance(x, bytes) else x for x in val]
        else:
            out[k] = val
    return out


@app.post("/api/verify")
async def run_verify(
    captured: UploadFile = File(...),
    print_size_mm: Optional[int] = Form(None),
    notes: str = Form(""),
):
    db = get_db()

    # Save captured upload
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(captured.filename or "")[1] or ".jpg"
    cap_name = f"capture_{ts}{ext}"
    cap_path = os.path.join(config.UPLOADS_DIR, cap_name)
    with open(cap_path, "wb") as f:
        shutil.copyfileobj(captured.file, f)

    cleanup_old_uploads()

    # DMs → seed → pattern-only ROI → regenerated reference
    result = verify_pattern(cap_path, uploads_dir=config.UPLOADS_DIR)

    if result.get("verdict") == "UNABLE_TO_VERIFY":
        # Keep the capture on failure (under uploads/) so it can be inspected.
        raise HTTPException(422, detail={
            "error":         result.get("error", "Verification failed"),
            "dm_diagnostic": _safe_diag(result.get("dm_diagnostic", {})),
            "capture_file":  cap_name,
        })

    # Look up which registered pattern matches this seed
    seed = result.get("seed_recovered")
    p = db.patterns.find_one({"seed": seed}) if seed is not None else None

    # Upload captured + aligned to Cloudinary / local store
    captured_cloud = upload_to_cloudinary(cap_path, folder="phonecdp/captures")
    aligned_path = os.path.join(config.UPLOADS_DIR, result["aligned_filename"])
    aligned_cloud = None
    try:
        aligned_cloud = upload_to_cloudinary(aligned_path, folder="phonecdp/aligned")
    except Exception:
        pass

    vid = get_next_id("verifications")
    doc = {
        "id":             vid,
        "pattern_id":     p["id"] if p else None,
        "seed_recovered": seed,
        "captured_url":   captured_cloud["secure_url"],
        "aligned_url":    aligned_cloud["secure_url"] if aligned_cloud else None,
        "roi_filename":   result.get("roi_filename"),
        "aligned_filename": result["aligned_filename"],
        "reference_filename": result.get("reference_filename"),
        "detection_filename": result.get("detection_filename"),
        "align_method":   result.get("align_method"),
        "cloudinary_ids": {
            "captured": captured_cloud["public_id"],
            "aligned":  aligned_cloud["public_id"] if aligned_cloud else None,
        },
        "verdict":           result["verdict"],
        "confidence":        result["confidence"],
        "score_moire":       result["scores"]["moire"],
        "score_color":       result["scores"]["color"],
        "score_correlation": result["scores"]["correlation"],
        "score_gradient":    result["scores"]["gradient"],
        "dm_diagnostic":     _safe_diag(result.get("dm_diagnostic")),
        "print_size_mm":     print_size_mm,
        "notes":             notes,
        "created_at":        datetime.utcnow(),
    }
    db.verifications.insert_one(doc)

    try:
        os.remove(cap_path)
    except OSError:
        pass

    return {
        "id":             vid,
        "pattern_id":     doc["pattern_id"],
        "pattern_serial": p["serial_number"] if p else None,
        "pattern_label":  p["label"] if p else None,
        "verdict":        doc["verdict"],
        "confidence":     doc["confidence"],
        "scores":         result["scores"],
        "weights":        result["weights"],
        "seed_recovered": seed,
        "dm_diagnostic":  doc["dm_diagnostic"],
        "roi_filename":   doc["roi_filename"],
        "aligned_filename": doc["aligned_filename"],
        "print_size_mm":  doc["print_size_mm"],
        "notes":          doc["notes"],
        "created_at":     doc["created_at"].isoformat(),
    }


@app.post("/api/verify/batch")
async def batch_verify(
    captured_files: list[UploadFile] = File(...),
    pattern_id: int = Form(...),
    print_size_mm: Optional[int] = Form(None),
    notes: str = Form(""),
):
    db = get_db()
    p = db.patterns.find_one({"id": pattern_id})
    if not p:
        raise HTTPException(404, "Pattern not found")

    cleanup_old_uploads()

    # Download original pattern from Cloudinary to temp
    original_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.PATTERNS_DIR)
    original_tmp.close()
    download_from_cloudinary(p["image_url"], original_tmp.name)

    results = []
    for i, cap in enumerate(captured_files):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(cap.filename or "")[1] or ".jpg"
        cap_name = f"capture_{ts}_{p['id']}_{i}{ext}"
        cap_path = os.path.join(config.UPLOADS_DIR, cap_name)
        with open(cap_path, "wb") as f:
            shutil.copyfileobj(cap.file, f)

        r = verify_pattern_legacy(original_tmp.name, cap_path,
                                   uploads_dir=config.UPLOADS_DIR,
                                   block_size=p["block_size"])
        if r.get("verdict") == "ERROR":
            results.append({"filename": cap.filename, "verdict": "ERROR",
                            "error": r.get("error")})
            try:
                os.remove(cap_path)
            except OSError:
                pass
            continue

        # Upload captured + aligned to Cloudinary (aligned needed for analytics)
        captured_cloud = upload_to_cloudinary(cap_path, folder="phonecdp/captures")
        aligned_path = os.path.join(config.UPLOADS_DIR, r["aligned_filename"])
        aligned_cloud = None
        try:
            aligned_cloud = upload_to_cloudinary(aligned_path, folder="phonecdp/aligned")
        except Exception:
            pass

        vid = get_next_id("verifications")
        doc = {
            "id": vid,
            "pattern_id": p["id"],
            "captured_url": captured_cloud["secure_url"],
            "aligned_url": aligned_cloud["secure_url"] if aligned_cloud else None,
            "markers_filename": r["markers_filename"],
            "aligned_filename": r["aligned_filename"],
            "cloudinary_ids": {
                "captured": captured_cloud["public_id"],
                "aligned": aligned_cloud["public_id"] if aligned_cloud else None,
            },
            "verdict": r["verdict"],
            "confidence": r["confidence"],
            "score_moire": r["scores"]["moire"],
            "score_color": r["scores"]["color"],
            "score_correlation": r["scores"]["correlation"],
            "score_gradient": r["scores"]["gradient"],
            "per_block_scores": r.get("per_block_scores"),
            "markers_found": r["markers_found"],
            "alignment_method": r["alignment_method"],
            "print_size_mm": print_size_mm,
            "notes": notes,
            "created_at": datetime.utcnow(),
        }
        db.verifications.insert_one(doc)

        # Clean up temp captured file (keep markers/aligned for serving)
        try:
            os.remove(cap_path)
        except OSError:
            pass

        results.append({
            "id": vid, "filename": cap.filename, "verdict": doc["verdict"],
            "confidence": doc["confidence"], "scores": r["scores"],
            "markers_found": doc["markers_found"],
        })

    # Clean up original temp
    try:
        os.remove(original_tmp.name)
    except OSError:
        pass

    return {"results": results, "total": len(results)}


@app.get("/api/verify/{vid}")
def get_verification(vid: int):
    db = get_db()
    v = db.verifications.find_one({"id": vid}, {"_id": 0})
    if not v:
        raise HTTPException(404, "Not found")
    p = db.patterns.find_one({"id": v["pattern_id"]})
    return {
        "id": v["id"], "pattern_id": v["pattern_id"],
        "pattern_serial": p["serial_number"] if p else "",
        "pattern_label": p["label"] if p else "",
        "verdict": v.get("verdict", "UNKNOWN"), "confidence": v.get("confidence", 0.0),
        "scores": {
            "moire": v.get("score_moire", 0.0), "color": v.get("score_color", 0.0),
            "correlation": v.get("score_correlation", 0.0), "gradient": v.get("score_gradient", 0.0),
        },
        "weights": {
            "moire": v.get("weight_moire", 0.65),
            "color": v.get("weight_color", 0.10),
            "correlation": v.get("weight_correlation", 0.10),
            "gradient": v.get("weight_gradient", 0.15),
        },
        # Legacy fields (may be None for new-pipeline records)
        "markers_found": v.get("markers_found"),
        "alignment_method": v.get("alignment_method"),
        # New pipeline fields
        "seed_recovered": v.get("seed_recovered"),
        "dm_diagnostic":  v.get("dm_diagnostic", {}),
        "roi_filename":   v.get("roi_filename"),
        "reference_filename": v.get("reference_filename"),
        "aligned_filename": v.get("aligned_filename"),
        "detection_filename": v.get("detection_filename"),
        "align_method": v.get("align_method"),
        "print_size_mm": v.get("print_size_mm"),
        "notes": v.get("notes", ""),
        "created_at": v["created_at"].isoformat() if hasattr(v["created_at"], "isoformat") else str(v["created_at"]),
    }


@app.get("/api/verify/{vid}/images/{image_type}")
def get_image(vid: int, image_type: str):
    db = get_db()
    v = db.verifications.find_one({"id": vid})
    if not v:
        raise HTTPException(404, "Not found")

    if image_type == "original":
        p = db.patterns.find_one({"id": v["pattern_id"]})
        if not p:
            raise HTTPException(404, "Pattern not found")
        return RedirectResponse(url=p["image_url"])
    elif image_type == "captured":
        url = v.get("captured_url")
        if not url:
            raise HTTPException(404, "Captured image not available")
        return RedirectResponse(url=url)
    elif image_type in ("markers", "aligned", "roi", "reference", "detection"):
        # 'roi'/'reference'/'detection' are new-pipeline filenames; 'markers' is legacy
        key_map = {"markers": "markers_filename", "aligned": "aligned_filename",
                   "roi": "roi_filename", "reference": "reference_filename",
                   "detection": "detection_filename"}
        filename = v.get(key_map[image_type])
        if not filename:
            raise HTTPException(404, f"{image_type} image not available")
        filepath = os.path.join(config.UPLOADS_DIR, filename)
        if not os.path.isfile(filepath):
            raise HTTPException(404, f"{image_type} image expired")
        return FileResponse(filepath, media_type="image/png")
    else:
        raise HTTPException(400, "Use: original, captured, markers, aligned, roi, reference, detection")


# ── Analytics / research-paper plots ───────────────────────────────────────

def _load_aligned_and_reference(v, db):
    """Fetch aligned + reference grayscale (and RGB) for plot rendering.

    Returns (aligned_gray, reference_gray, aligned_rgb, reference_rgb) or None.
    Tries local aligned file first, falls back to Cloudinary `aligned_url`.
    """
    import cv2
    import numpy as np

    p = db.patterns.find_one({"id": v["pattern_id"]})
    if not p:
        return None

    # Prefer local reference file (new pipeline saves it to uploads_dir)
    ref_bgr = None
    ref_fn = v.get("reference_filename")
    if ref_fn:
        local_ref = os.path.join(config.UPLOADS_DIR, ref_fn)
        if os.path.isfile(local_ref):
            ref_bgr = cv2.imread(local_ref)

    # Fallback: download original from Cloudinary
    if ref_bgr is None:
        ref_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.DATA_DIR)
        ref_tmp.close()
        try:
            download_from_cloudinary(p["image_url"], ref_tmp.name)
            ref_bgr = cv2.imread(ref_tmp.name)
        finally:
            try:
                os.remove(ref_tmp.name)
            except OSError:
                pass
    if ref_bgr is None:
        return None

    aligned_bgr = None
    fn = v.get("aligned_filename")
    if fn:
        local = os.path.join(config.UPLOADS_DIR, fn)
        if os.path.isfile(local):
            aligned_bgr = cv2.imread(local)

    if aligned_bgr is None and v.get("aligned_url"):
        a_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.DATA_DIR)
        a_tmp.close()
        try:
            download_from_cloudinary(v["aligned_url"], a_tmp.name)
            aligned_bgr = cv2.imread(a_tmp.name)
        finally:
            try:
                os.remove(a_tmp.name)
            except OSError:
                pass

    if aligned_bgr is None:
        return None

    return (
        cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB),
    )


def _get_per_block_scores(v, db):
    """Return per-block score arrays. Use stored if present, otherwise compute."""
    import numpy as np
    from cdp_analytics import compute_per_block_scores

    stored = v.get("per_block_scores")
    if stored:
        return {k: np.array(arr, dtype=np.float32) for k, arr in stored.items()}

    imgs = _load_aligned_and_reference(v, db)
    if imgs is None:
        raise HTTPException(404, "Aligned image not available — cannot compute analytics")
    aligned_gray, reference_gray, aligned_rgb, reference_rgb = imgs

    p = db.patterns.find_one({"id": v["pattern_id"]})
    bs = (p or {}).get("block_size", 16)
    return compute_per_block_scores(aligned_gray, reference_gray,
                                    aligned_rgb, reference_rgb, block_size=bs)


@app.get("/api/verify/{vid}/plot/{test}/{plot_type}.png")
def get_verification_plot(vid: int, test: str, plot_type: str):
    """Render a single per-block plot.

    test:       moire | correlation | gradient | color
    plot_type:  heatmap | scatter | histogram | delta
    """
    import cv2
    from cdp_analytics import (render_heatmap, render_scatter,
                               render_histogram, render_delta_map)

    db = get_db()
    v = db.verifications.find_one({"id": vid})
    if not v:
        raise HTTPException(404, "Verification not found")

    if test not in ("moire", "correlation", "gradient", "color"):
        raise HTTPException(400, "test must be moire|correlation|gradient|color")
    if plot_type not in ("heatmap", "scatter", "histogram", "delta"):
        raise HTTPException(400, "plot_type must be heatmap|scatter|histogram|delta")

    if plot_type == "delta":
        imgs = _load_aligned_and_reference(v, db)
        if imgs is None:
            raise HTTPException(404, "Aligned image not available")
        png = render_delta_map(imgs[0], imgs[1])
    else:
        scores = _get_per_block_scores(v, db)
        arr = scores[test]
        if plot_type == "heatmap":
            imgs = _load_aligned_and_reference(v, db)
            aligned_gray = imgs[0] if imgs else None
            png = render_heatmap(arr, test, aligned_gray=aligned_gray)
        elif plot_type == "scatter":
            png = render_scatter(arr, test)
        else:
            png = render_histogram(arr, test)

    return StreamingResponse(io.BytesIO(png), media_type="image/png")


@app.get("/api/verify/{vid}/report.pdf")
def get_verification_report_pdf(vid: int):
    """Multi-page PDF with all plots for the verification."""
    from cdp_analytics import render_full_report_pdf

    db = get_db()
    v = db.verifications.find_one({"id": vid})
    if not v:
        raise HTTPException(404, "Verification not found")

    scores = _get_per_block_scores(v, db)
    imgs = _load_aligned_and_reference(v, db)
    aligned_gray = imgs[0] if imgs else None
    reference_gray = imgs[1] if imgs else None

    # Add ISO-formatted timestamp into the dict for the cover page
    v_view = dict(v)
    if "_id" in v_view:
        v_view.pop("_id")
    if v_view.get("created_at"):
        v_view["created_at"] = v_view["created_at"].isoformat() if hasattr(
            v_view["created_at"], "isoformat") else str(v_view["created_at"])

    pdf_bytes = render_full_report_pdf(v_view, scores,
                                       aligned_gray=aligned_gray,
                                       reference_gray=reference_gray)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="verification_{vid}_report.pdf"'},
    )


# ── Results API ──────────────────────────────────────────────────────────

@app.get("/api/results")
def list_results(
    verdict: Optional[str] = Query(None),
    pattern_id: Optional[int] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    db = get_db()
    query = {}
    if verdict:
        query["verdict"] = verdict.upper()
    if pattern_id:
        query["pattern_id"] = pattern_id

    total = db.verifications.count_documents(query)
    rows = list(
        db.verifications.find(query, {"_id": 0})
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
    )

    # Batch-fetch patterns for serial/label
    pattern_ids = list({r["pattern_id"] for r in rows})
    patterns_map = {}
    if pattern_ids:
        for p in db.patterns.find({"id": {"$in": pattern_ids}}, {"_id": 0}):
            patterns_map[p["id"]] = p

    results = []
    for v in rows:
        p = patterns_map.get(v["pattern_id"], {})
        results.append({
            "id": v["id"], "pattern_id": v["pattern_id"],
            "pattern_serial": p.get("serial_number", ""),
            "pattern_label": p.get("label", ""),
            "verdict": v.get("verdict", "UNKNOWN"), "confidence": v.get("confidence", 0.0),
            "scores": {
                "moire": v.get("score_moire", 0.0), "color": v.get("score_color", 0.0),
                "correlation": v.get("score_correlation", 0.0), "gradient": v.get("score_gradient", 0.0),
            },
            "markers_found": v.get("markers_found", 0),
            "alignment_method": v.get("alignment_method", "unknown"),
            "print_size_mm": v.get("print_size_mm"),
            "notes": v.get("notes", ""),
            "created_at": v["created_at"].isoformat() if hasattr(v["created_at"], "isoformat") else str(v["created_at"]),
        })

    return {"results": results, "total": total}


@app.get("/api/results/stats")
def stats():
    db = get_db()
    tp = db.patterns.count_documents({})
    tv = db.verifications.count_documents({})

    if tv == 0:
        return {
            "total_patterns": tp, "total_verifications": 0,
            "verdicts": {"authentic": 0, "suspicious": 0, "counterfeit": 0},
            "pass_rate": 0, "avg_confidence": 0, "avg_markers": 0,
            "avg_scores": {"moire": 0, "color": 0, "correlation": 0, "gradient": 0},
        }

    pipeline = [{"$group": {
        "_id": None,
        "authentic": {"$sum": {"$cond": [{"$eq": ["$verdict", "AUTHENTIC"]}, 1, 0]}},
        "suspicious": {"$sum": {"$cond": [{"$eq": ["$verdict", "SUSPICIOUS"]}, 1, 0]}},
        "counterfeit": {"$sum": {"$cond": [{"$eq": ["$verdict", "COUNTERFEIT"]}, 1, 0]}},
        "avg_confidence": {"$avg": "$confidence"},
        "avg_markers": {"$avg": "$markers_found"},
        "avg_moire": {"$avg": "$score_moire"},
        "avg_color": {"$avg": "$score_color"},
        "avg_correlation": {"$avg": "$score_correlation"},
        "avg_gradient": {"$avg": "$score_gradient"},
    }}]
    agg = list(db.verifications.aggregate(pipeline))
    s = agg[0] if agg else {}

    auth = s.get("authentic", 0)
    return {
        "total_patterns": tp, "total_verifications": tv,
        "verdicts": {
            "authentic": auth,
            "suspicious": s.get("suspicious", 0),
            "counterfeit": s.get("counterfeit", 0),
        },
        "pass_rate": round(auth / tv * 100, 1) if tv > 0 else 0,
        "avg_confidence": round(float(s.get("avg_confidence", 0)), 3),
        "avg_markers": round(float(s.get("avg_markers", 0)), 1),
        "avg_scores": {
            "moire": round(float(s.get("avg_moire", 0)), 3),
            "color": round(float(s.get("avg_color", 0)), 3),
            "correlation": round(float(s.get("avg_correlation", 0)), 3),
            "gradient": round(float(s.get("avg_gradient", 0)), 3),
        },
    }


@app.get("/api/results/export")
def export_csv():
    db = get_db()
    rows = list(db.verifications.find({}, {"_id": 0}).sort("created_at", -1))

    # Batch-fetch patterns
    pattern_ids = list({r["pattern_id"] for r in rows})
    patterns_map = {}
    if pattern_ids:
        for p in db.patterns.find({"id": {"$in": pattern_ids}}, {"_id": 0}):
            patterns_map[p["id"]] = p

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ID", "Date", "Pattern ID", "Serial", "Label", "Verdict", "Confidence",
                "Moire", "Color", "Correlation", "Gradient", "Markers", "Alignment",
                "Print Size (mm)", "Notes"])
    for v in rows:
        p = patterns_map.get(v["pattern_id"], {})
        created = v["created_at"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(v["created_at"], "strftime") else str(v["created_at"])
        w.writerow([
            v["id"], created, v["pattern_id"],
            p.get("serial_number", ""), p.get("label", ""), v.get("verdict", "UNKNOWN"),
            f"{v.get('confidence', 0.0):.3f}", f"{v.get('score_moire', 0.0):.3f}",
            f"{v.get('score_color', 0.0):.3f}", f"{v.get('score_correlation', 0.0):.3f}",
            f"{v.get('score_gradient', 0.0):.3f}", v.get("markers_found", 0),
            v.get("alignment_method", "unknown"), v.get("print_size_mm", ""),
            v.get("notes", ""),
        ])
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=phonecdp_results.csv"})


@app.delete("/api/results/{rid}")
def delete_result(rid: int):
    db = get_db()
    v = db.verifications.find_one({"id": rid})
    if not v:
        raise HTTPException(404, "Not found")

    # Delete captured from Cloudinary
    cid = (v.get("cloudinary_ids") or {}).get("captured")
    if cid:
        destroy_cloudinary(cid)

    db.verifications.delete_one({"id": rid})
    return {"message": "Deleted"}


@app.patch("/api/results/{rid}/notes")
def update_notes(rid: int, body: dict):
    db = get_db()
    result = db.verifications.update_one(
        {"id": rid},
        {"$set": {"notes": body.get("notes", "")}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Not found")
    return {"message": "Updated", "notes": body.get("notes", "")}


# ── Startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    try:
        init_db()
    except Exception as exc:
        # init_db() is already non-fatal internally, but guard here too.
        import warnings
        warnings.warn(f"[startup] DB init raised unexpectedly: {exc}", RuntimeWarning)


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Lightweight health check — reports DB connectivity status."""
    import database as _db_mod
    db_ok = is_db_available()
    return {
        "status":   "ok" if db_ok else "degraded",
        "db":       "mongodb" if db_ok else "offline-local",
        "db_error": _db_mod._db_error,
        "offline":  _db_mod._offline,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=config.PORT)
