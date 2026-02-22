"""
PhoneCDP Production Web Application
"""
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import csv
import io
import os
from datetime import datetime

import config
from database import init_db, get_db, Pattern, Verification
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


@app.get("/api/patterns")
def list_patterns(db: Session = Depends(get_db)):
    patterns = db.query(Pattern).order_by(Pattern.created_at.desc()).all()
    return [{
        "id": p.id, "seed": p.seed, "serial_number": p.serial_number,
        "label": p.label, "filename": p.filename, "pattern_size": p.pattern_size,
        "created_at": p.created_at.isoformat(), "notes": p.notes,
        "verification_count": len(p.verifications),
    } for p in patterns]


@app.post("/api/patterns/generate")
def create_pattern(data: PatternCreate, db: Session = Depends(get_db)):
    result = generate_pattern(
        output_dir=config.PATTERNS_DIR, seed=data.seed,
        serial_number=data.serial_number, pattern_size=data.pattern_size)
    p = Pattern(seed=result["seed"], serial_number=data.serial_number,
                label=data.label, filename=result["filename"],
                pattern_size=data.pattern_size, notes=data.notes)
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"id": p.id, "seed": p.seed, "serial_number": p.serial_number,
            "label": p.label, "filename": p.filename,
            "pattern_size": p.pattern_size, "created_at": p.created_at.isoformat(),
            "notes": p.notes}


@app.get("/api/patterns/{pid}")
def get_pattern(pid: int, db: Session = Depends(get_db)):
    p = db.query(Pattern).filter(Pattern.id == pid).first()
    if not p:
        raise HTTPException(404, "Pattern not found")
    return {"id": p.id, "seed": p.seed, "serial_number": p.serial_number,
            "label": p.label, "filename": p.filename,
            "pattern_size": p.pattern_size, "created_at": p.created_at.isoformat(),
            "notes": p.notes, "verification_count": len(p.verifications)}


@app.get("/api/patterns/{pid}/download")
def download_pattern(pid: int, db: Session = Depends(get_db)):
    p = db.query(Pattern).filter(Pattern.id == pid).first()
    if not p:
        raise HTTPException(404, "Pattern not found")
    fp = os.path.join(config.PATTERNS_DIR, p.filename)
    if not os.path.exists(fp):
        raise HTTPException(404, "File not found")
    return FileResponse(fp, media_type="image/png", filename=p.filename)


@app.get("/api/patterns/{pid}/preview")
def preview_pattern(pid: int, db: Session = Depends(get_db)):
    p = db.query(Pattern).filter(Pattern.id == pid).first()
    if not p:
        raise HTTPException(404, "Pattern not found")
    fp = os.path.join(config.PATTERNS_DIR, p.filename)
    if not os.path.exists(fp):
        raise HTTPException(404, "File not found")
    return FileResponse(fp, media_type="image/png")


@app.delete("/api/patterns/{pid}")
def delete_pattern(pid: int, db: Session = Depends(get_db)):
    p = db.query(Pattern).filter(Pattern.id == pid).first()
    if not p:
        raise HTTPException(404, "Pattern not found")
    fp = os.path.join(config.PATTERNS_DIR, p.filename)
    if os.path.exists(fp):
        os.remove(fp)
    db.delete(p)
    db.commit()
    return {"message": "Deleted"}


# ── Verify API ───────────────────────────────────────────────────────────

@app.post("/api/verify")
async def run_verify(
    captured: UploadFile = File(...),
    pattern_id: int = Form(...),
    print_size_mm: Optional[int] = Form(None),
    notes: str = Form(""),
    db: Session = Depends(get_db),
):
    p = db.query(Pattern).filter(Pattern.id == pattern_id).first()
    if not p:
        raise HTTPException(404, "Pattern not found")
    original_path = os.path.join(config.PATTERNS_DIR, p.filename)
    if not os.path.exists(original_path):
        raise HTTPException(404, "Pattern file missing")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(captured.filename)[1] or ".jpg"
    cap_name = f"capture_{ts}_{p.id}{ext}"
    cap_path = os.path.join(config.UPLOADS_DIR, cap_name)
    with open(cap_path, "wb") as f:
        shutil.copyfileobj(captured.file, f)

    result = verify_pattern(original_path, cap_path, uploads_dir=config.UPLOADS_DIR)
    if result.get("verdict") == "ERROR":
        raise HTTPException(400, result.get("error", "Verification failed"))

    v = Verification(
        pattern_id=p.id, captured_filename=cap_name,
        aligned_filename=result.get("aligned_filename", ""),
        verdict=result["verdict"], confidence=result["confidence"],
        score_moire=result["scores"]["moire"],
        score_color=result["scores"]["color"],
        score_correlation=result["scores"]["correlation"],
        score_gradient=result["scores"]["gradient"],
        markers_found=result["markers_found"],
        alignment_method=result["alignment_method"],
        print_size_mm=print_size_mm, notes=notes)
    db.add(v)
    db.commit()
    db.refresh(v)

    return {
        "id": v.id, "pattern_id": p.id, "verdict": v.verdict,
        "confidence": v.confidence, "scores": result["scores"],
        "weights": result["weights"], "markers_found": v.markers_found,
        "alignment_method": v.alignment_method,
        "pattern_found": result["pattern_found"],
        "print_size_mm": v.print_size_mm, "notes": v.notes,
        "created_at": v.created_at.isoformat(),
    }


@app.post("/api/verify/batch")
async def batch_verify(
    captured_files: list[UploadFile] = File(...),
    pattern_id: int = Form(...),
    print_size_mm: Optional[int] = Form(None),
    notes: str = Form(""),
    db: Session = Depends(get_db),
):
    p = db.query(Pattern).filter(Pattern.id == pattern_id).first()
    if not p:
        raise HTTPException(404, "Pattern not found")
    original_path = os.path.join(config.PATTERNS_DIR, p.filename)
    if not os.path.exists(original_path):
        raise HTTPException(404, "Pattern file missing")

    results = []
    for i, cap in enumerate(captured_files):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(cap.filename)[1] or ".jpg"
        cap_name = f"capture_{ts}_{p.id}_{i}{ext}"
        cap_path = os.path.join(config.UPLOADS_DIR, cap_name)
        with open(cap_path, "wb") as f:
            shutil.copyfileobj(cap.file, f)

        r = verify_pattern(original_path, cap_path, uploads_dir=config.UPLOADS_DIR)
        if r.get("verdict") == "ERROR":
            results.append({"filename": cap.filename, "verdict": "ERROR", "error": r.get("error")})
            continue

        v = Verification(
            pattern_id=p.id, captured_filename=cap_name,
            aligned_filename=r.get("aligned_filename", ""),
            verdict=r["verdict"], confidence=r["confidence"],
            score_moire=r["scores"]["moire"], score_color=r["scores"]["color"],
            score_correlation=r["scores"]["correlation"],
            score_gradient=r["scores"]["gradient"],
            markers_found=r["markers_found"], alignment_method=r["alignment_method"],
            print_size_mm=print_size_mm, notes=notes)
        db.add(v)
        db.commit()
        db.refresh(v)
        results.append({"id": v.id, "filename": cap.filename, "verdict": v.verdict,
                        "confidence": v.confidence, "scores": r["scores"],
                        "markers_found": v.markers_found})

    return {"results": results, "total": len(results)}


@app.get("/api/verify/{vid}")
def get_verification(vid: int, db: Session = Depends(get_db)):
    v = db.query(Verification).filter(Verification.id == vid).first()
    if not v:
        raise HTTPException(404, "Not found")
    return {
        "id": v.id, "pattern_id": v.pattern_id,
        "pattern_serial": v.pattern.serial_number,
        "pattern_label": v.pattern.label,
        "verdict": v.verdict, "confidence": v.confidence,
        "scores": {"moire": v.score_moire, "color": v.score_color,
                   "correlation": v.score_correlation, "gradient": v.score_gradient},
        "markers_found": v.markers_found, "alignment_method": v.alignment_method,
        "print_size_mm": v.print_size_mm, "notes": v.notes,
        "created_at": v.created_at.isoformat(),
    }


@app.get("/api/verify/{vid}/images/{image_type}")
def get_image(vid: int, image_type: str, db: Session = Depends(get_db)):
    v = db.query(Verification).filter(Verification.id == vid).first()
    if not v:
        raise HTTPException(404, "Not found")
    if image_type == "original":
        fp = os.path.join(config.PATTERNS_DIR, v.pattern.filename)
    elif image_type == "captured":
        fp = os.path.join(config.UPLOADS_DIR, v.captured_filename)
    elif image_type == "aligned":
        fp = os.path.join(config.UPLOADS_DIR, v.aligned_filename)
    else:
        raise HTTPException(400, "Use: original, captured, aligned")
    if not os.path.exists(fp):
        raise HTTPException(404, f"{image_type} file not found")
    return FileResponse(fp)


# ── Results API ──────────────────────────────────────────────────────────

@app.get("/api/results")
def list_results(
    verdict: Optional[str] = Query(None),
    pattern_id: Optional[int] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db),
):
    q = db.query(Verification).join(Pattern)
    if verdict:
        q = q.filter(Verification.verdict == verdict.upper())
    if pattern_id:
        q = q.filter(Verification.pattern_id == pattern_id)
    total = q.count()
    rows = q.order_by(Verification.created_at.desc()).offset(offset).limit(limit).all()
    return {"results": [{
        "id": v.id, "pattern_id": v.pattern_id,
        "pattern_serial": v.pattern.serial_number,
        "pattern_label": v.pattern.label,
        "verdict": v.verdict, "confidence": v.confidence,
        "scores": {"moire": v.score_moire, "color": v.score_color,
                   "correlation": v.score_correlation, "gradient": v.score_gradient},
        "markers_found": v.markers_found, "alignment_method": v.alignment_method,
        "print_size_mm": v.print_size_mm, "notes": v.notes,
        "created_at": v.created_at.isoformat(),
    } for v in rows], "total": total}


@app.get("/api/results/stats")
def stats(db: Session = Depends(get_db)):
    tp = db.query(func.count(Pattern.id)).scalar()
    tv = db.query(func.count(Verification.id)).scalar()
    auth = db.query(func.count(Verification.id)).filter(Verification.verdict == "AUTHENTIC").scalar()
    susp = db.query(func.count(Verification.id)).filter(Verification.verdict == "SUSPICIOUS").scalar()
    fake = db.query(func.count(Verification.id)).filter(Verification.verdict == "COUNTERFEIT").scalar()
    ac = db.query(func.avg(Verification.confidence)).scalar() or 0
    am = db.query(func.avg(Verification.markers_found)).scalar() or 0
    return {
        "total_patterns": tp, "total_verifications": tv,
        "verdicts": {"authentic": auth, "suspicious": susp, "counterfeit": fake},
        "pass_rate": round(auth / tv * 100, 1) if tv > 0 else 0,
        "avg_confidence": round(float(ac), 3),
        "avg_markers": round(float(am), 1),
        "avg_scores": {
            "moire": round(float(db.query(func.avg(Verification.score_moire)).scalar() or 0), 3),
            "color": round(float(db.query(func.avg(Verification.score_color)).scalar() or 0), 3),
            "correlation": round(float(db.query(func.avg(Verification.score_correlation)).scalar() or 0), 3),
            "gradient": round(float(db.query(func.avg(Verification.score_gradient)).scalar() or 0), 3),
        },
    }


@app.get("/api/results/export")
def export_csv(db: Session = Depends(get_db)):
    rows = db.query(Verification).join(Pattern).order_by(Verification.created_at.desc()).all()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ID", "Date", "Pattern ID", "Serial", "Label", "Verdict", "Confidence",
                "Moire", "Color", "Correlation", "Gradient", "Markers", "Alignment",
                "Print Size (mm)", "Notes"])
    for v in rows:
        w.writerow([v.id, v.created_at.strftime("%Y-%m-%d %H:%M:%S"), v.pattern_id,
                    v.pattern.serial_number, v.pattern.label, v.verdict,
                    f"{v.confidence:.3f}", f"{v.score_moire:.3f}", f"{v.score_color:.3f}",
                    f"{v.score_correlation:.3f}", f"{v.score_gradient:.3f}",
                    v.markers_found, v.alignment_method, v.print_size_mm or "", v.notes])
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=phonecdp_results.csv"})


@app.delete("/api/results/{rid}")
def delete_result(rid: int, db: Session = Depends(get_db)):
    v = db.query(Verification).filter(Verification.id == rid).first()
    if not v:
        raise HTTPException(404, "Not found")
    db.delete(v)
    db.commit()
    return {"message": "Deleted"}


@app.patch("/api/results/{rid}/notes")
def update_notes(rid: int, body: dict, db: Session = Depends(get_db)):
    v = db.query(Verification).filter(Verification.id == rid).first()
    if not v:
        raise HTTPException(404, "Not found")
    v.notes = body.get("notes", v.notes)
    db.commit()
    return {"message": "Updated", "notes": v.notes}


# ── Startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=config.PORT)
