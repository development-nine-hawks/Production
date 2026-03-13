"""
PhoneCDP Production Web Application
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import csv
import io
import os
import tempfile
from datetime import datetime

import config
from database import (
    init_db, get_db, get_next_id,
    upload_to_cloudinary, destroy_cloudinary, download_from_cloudinary,
)
from cdp_engine import generate_pattern, verify_pattern

app = FastAPI(title="PhoneCDP", version="1.0.0")

app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")


# ── HTML ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(os.path.join(config.BASE_DIR, "templates", "index.html"))


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
    block_size: int = 8


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
    return RedirectResponse(url=p["image_url"])


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

    # Delete all verification images from Cloudinary
    for v in db.verifications.find({"pattern_id": pid}):
        for key in ["captured", "markers", "aligned"]:
            cid = (v.get("cloudinary_ids") or {}).get(key)
            if cid:
                destroy_cloudinary(cid)

    db.verifications.delete_many({"pattern_id": pid})
    db.patterns.delete_one({"id": pid})
    return {"message": "Deleted"}


# ── Verify API ───────────────────────────────────────────────────────────

@app.post("/api/verify")
async def run_verify(
    captured: UploadFile = File(...),
    pattern_id: int = Form(...),
    print_size_mm: Optional[int] = Form(None),
    notes: str = Form(""),
):
    db = get_db()
    p = db.patterns.find_one({"id": pattern_id})
    if not p:
        raise HTTPException(404, "Pattern not found")

    # Download original pattern from Cloudinary to temp
    original_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=config.PATTERNS_DIR)
    original_tmp.close()
    download_from_cloudinary(p["image_url"], original_tmp.name)

    # Save captured upload to temp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(captured.filename or "")[1] or ".jpg"
    cap_name = f"capture_{ts}_{p['id']}{ext}"
    cap_path = os.path.join(config.UPLOADS_DIR, cap_name)
    with open(cap_path, "wb") as f:
        shutil.copyfileobj(captured.file, f)

    result = verify_pattern(original_tmp.name, cap_path,
                            uploads_dir=config.UPLOADS_DIR,
                            block_size=p["block_size"])
    if result.get("verdict") == "ERROR":
        raise HTTPException(400, result.get("error", "Verification failed"))

    # Upload images to Cloudinary
    try:
        captured_cloud = upload_to_cloudinary(cap_path, folder="phonecdp/captures")
        markers_cloud = upload_to_cloudinary(
            os.path.join(config.UPLOADS_DIR, result["markers_filename"]),
            folder="phonecdp/markers",
        )
        aligned_cloud = upload_to_cloudinary(
            os.path.join(config.UPLOADS_DIR, result["aligned_filename"]),
            folder="phonecdp/aligned",
        )
    except Exception as e:
        raise HTTPException(500, f"Image upload failed: {e}")

    vid = get_next_id("verifications")
    doc = {
        "id": vid,
        "pattern_id": p["id"],
        "captured_url": captured_cloud["secure_url"],
        "markers_url": markers_cloud["secure_url"],
        "aligned_url": aligned_cloud["secure_url"],
        "cloudinary_ids": {
            "captured": captured_cloud["public_id"],
            "markers": markers_cloud["public_id"],
            "aligned": aligned_cloud["public_id"],
        },
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "score_moire": result["scores"]["moire"],
        "score_color": result["scores"]["color"],
        "score_correlation": result["scores"]["correlation"],
        "score_gradient": result["scores"]["gradient"],
        "markers_found": result["markers_found"],
        "alignment_method": result["alignment_method"],
        "print_size_mm": print_size_mm,
        "notes": notes,
        "created_at": datetime.utcnow(),
    }
    db.verifications.insert_one(doc)

    # Clean up local temp files
    for fp in [original_tmp.name, cap_path,
               os.path.join(config.UPLOADS_DIR, result["markers_filename"]),
               os.path.join(config.UPLOADS_DIR, result["aligned_filename"])]:
        try:
            os.remove(fp)
        except OSError:
            pass

    return {
        "id": vid, "pattern_id": p["id"], "verdict": doc["verdict"],
        "confidence": doc["confidence"], "scores": result["scores"],
        "weights": result["weights"], "markers_found": doc["markers_found"],
        "alignment_method": doc["alignment_method"],
        "pattern_found": result["pattern_found"],
        "print_size_mm": doc["print_size_mm"], "notes": doc["notes"],
        "created_at": doc["created_at"].isoformat(),
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

        r = verify_pattern(original_tmp.name, cap_path,
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

        # Upload images to Cloudinary
        try:
            captured_cloud = upload_to_cloudinary(cap_path, folder="phonecdp/captures")
            markers_cloud = upload_to_cloudinary(
                os.path.join(config.UPLOADS_DIR, r["markers_filename"]),
                folder="phonecdp/markers",
            )
            aligned_cloud = upload_to_cloudinary(
                os.path.join(config.UPLOADS_DIR, r["aligned_filename"]),
                folder="phonecdp/aligned",
            )
        except Exception as e:
            results.append({"filename": cap.filename, "verdict": "ERROR",
                            "error": f"Image upload failed: {e}"})
            continue

        vid = get_next_id("verifications")
        doc = {
            "id": vid,
            "pattern_id": p["id"],
            "captured_url": captured_cloud["secure_url"],
            "markers_url": markers_cloud["secure_url"],
            "aligned_url": aligned_cloud["secure_url"],
            "cloudinary_ids": {
                "captured": captured_cloud["public_id"],
                "markers": markers_cloud["public_id"],
                "aligned": aligned_cloud["public_id"],
            },
            "verdict": r["verdict"],
            "confidence": r["confidence"],
            "score_moire": r["scores"]["moire"],
            "score_color": r["scores"]["color"],
            "score_correlation": r["scores"]["correlation"],
            "score_gradient": r["scores"]["gradient"],
            "markers_found": r["markers_found"],
            "alignment_method": r["alignment_method"],
            "print_size_mm": print_size_mm,
            "notes": notes,
            "created_at": datetime.utcnow(),
        }
        db.verifications.insert_one(doc)

        # Clean up local files
        for fp in [cap_path,
                   os.path.join(config.UPLOADS_DIR, r["markers_filename"]),
                   os.path.join(config.UPLOADS_DIR, r["aligned_filename"])]:
            try:
                os.remove(fp)
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
        "verdict": v["verdict"], "confidence": v["confidence"],
        "scores": {
            "moire": v["score_moire"], "color": v["score_color"],
            "correlation": v["score_correlation"], "gradient": v["score_gradient"],
        },
        "markers_found": v["markers_found"],
        "alignment_method": v["alignment_method"],
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
    elif image_type == "markers":
        url = v.get("markers_url")
    elif image_type == "aligned":
        url = v.get("aligned_url")
    else:
        raise HTTPException(400, "Use: original, captured, markers, aligned")

    if not url:
        raise HTTPException(404, f"{image_type} image not available")
    return RedirectResponse(url=url)


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
            "verdict": v["verdict"], "confidence": v["confidence"],
            "scores": {
                "moire": v["score_moire"], "color": v["score_color"],
                "correlation": v["score_correlation"], "gradient": v["score_gradient"],
            },
            "markers_found": v["markers_found"],
            "alignment_method": v["alignment_method"],
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
            p.get("serial_number", ""), p.get("label", ""), v["verdict"],
            f"{v['confidence']:.3f}", f"{v['score_moire']:.3f}",
            f"{v['score_color']:.3f}", f"{v['score_correlation']:.3f}",
            f"{v['score_gradient']:.3f}", v["markers_found"],
            v["alignment_method"], v.get("print_size_mm", ""),
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

    # Delete Cloudinary images
    for key in ["captured", "markers", "aligned"]:
        cid = (v.get("cloudinary_ids") or {}).get(key)
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
    init_db()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=config.PORT)
