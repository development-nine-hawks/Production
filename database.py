"""
MongoDB + Cloudinary database layer for PhoneCDP.

When MongoDB Atlas is reachable, uses it as normal.
When it is unavailable (no internet, DNS failure, etc.), automatically
falls back to LocalDatabase — a file-backed store in DATA_DIR/localdb/
that implements the same pymongo API surface.

Cloudinary upload/download also degrades gracefully: when the network
is down, files are served directly from the local filesystem.
"""

import logging
import os
import shutil
import urllib.request
import warnings

import cloudinary
import cloudinary.uploader
from pymongo import MongoClient, ReturnDocument, ASCENDING, DESCENDING

import config

logger = logging.getLogger(__name__)

_client     = None
_db         = None
_db_error:  str | None = None   # None  = connected OK
_offline:   bool = False         # True  = LocalDatabase is active


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Connect to MongoDB Atlas; fall back to LocalDatabase if unavailable."""
    global _client, _db, _db_error, _offline

    # ── MongoDB ───────────────────────────────────────────────────────────────
    try:
        _client = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=5_000,
            connectTimeoutMS=5_000,
            socketTimeoutMS=10_000,
        )
        _db = _client[config.MONGODB_DB]
        _client.admin.command("ping")

        _db.patterns.create_index([("id", ASCENDING)], unique=True)
        _db.patterns.create_index([("created_at", DESCENDING)])
        _db.verifications.create_index([("id", ASCENDING)], unique=True)
        _db.verifications.create_index([("pattern_id", ASCENDING)])
        _db.verifications.create_index([("created_at", DESCENDING)])

        _db_error = None
        _offline  = False
        logger.info("MongoDB connected: %s / %s",
                    config.MONGODB_URI.split("@")[-1], config.MONGODB_DB)

    except Exception as exc:
        _client   = None
        _db_error = str(exc)
        _offline  = True

        from local_db import LocalDatabase
        _db = LocalDatabase()

        warnings.warn(
            f"[database] MongoDB unavailable — running in OFFLINE mode "
            f"(local JSON store). Reason: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("Offline mode active. Data stored in %s/localdb/",
                       config.DATA_DIR)

    # ── Cloudinary ────────────────────────────────────────────────────────────
    try:
        cloudinary.config(
            cloud_name=config.CLOUDINARY_CLOUD_NAME,
            api_key=config.CLOUDINARY_API_KEY,
            api_secret=config.CLOUDINARY_API_SECRET,
        )
    except Exception as exc:
        warnings.warn(f"[database] Cloudinary config failed: {exc}",
                      RuntimeWarning, stacklevel=2)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_db_available() -> bool:
    """True when a real MongoDB connection is healthy."""
    return _db is not None and not _offline


def get_db():
    """Return the active database object (MongoDB or LocalDatabase)."""
    if _db is None:
        raise RuntimeError(
            "Database not initialised. Reason: "
            + (_db_error or "init_db() has not been called.")
        )
    return _db


def get_next_id(collection_name: str) -> int:
    """Auto-increment integer ID via a counters collection."""
    db  = get_db()
    doc = db.counters.find_one_and_update(
        {"_id": collection_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["seq"]


# ─────────────────────────────────────────────────────────────────────────────
# Cloudinary — with offline fallback
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_cloudinary(file_path: str, folder: str,
                         public_id: str | None = None) -> dict:
    """Upload to Cloudinary; when offline, keep the file locally and return
    a local URL so the rest of the app keeps working."""
    if _offline:
        return _local_upload(file_path, folder, public_id)

    try:
        result = cloudinary.uploader.upload(
            file_path,
            folder=folder,
            public_id=public_id,
            overwrite=True,
            resource_type="image",
        )
        return {"secure_url": result["secure_url"],
                "public_id":  result["public_id"]}
    except Exception as exc:
        warnings.warn(f"[database] Cloudinary upload failed: {exc} — "
                      "falling back to local storage.", RuntimeWarning)
        return _local_upload(file_path, folder, public_id)


def _local_upload(file_path: str, folder: str,
                  public_id: str | None = None) -> dict:
    """Copy the file into DATA_DIR/localcdn/<folder>/ and return a /local-cdn URL."""
    dest_dir = os.path.join(config.DATA_DIR, "localcdn", folder)
    os.makedirs(dest_dir, exist_ok=True)
    fname = public_id or os.path.basename(file_path)
    dest  = os.path.join(dest_dir, fname)
    if file_path != dest:
        shutil.copy2(file_path, dest)
    # Serve via /local-cdn/<folder>/<fname>  (registered in app.py below)
    rel   = folder.rstrip("/") + "/" + fname
    url   = f"/local-cdn/{rel}"
    return {"secure_url": url, "public_id": rel}


def destroy_cloudinary(public_id: str) -> None:
    """Delete from Cloudinary (no-op when offline)."""
    if _offline:
        # Attempt to remove local copy
        local = os.path.join(config.DATA_DIR, "localcdn", public_id)
        try:
            os.remove(local)
        except OSError:
            pass
        return
    try:
        cloudinary.uploader.destroy(public_id, resource_type="image")
    except Exception:
        pass


def download_from_cloudinary(url: str, local_path: str) -> None:
    """Download a Cloudinary image; when offline, copy from local store."""
    if _offline or url.startswith("/local-cdn/"):
        if url.startswith("/local-cdn/"):
            rel   = url[len("/local-cdn/"):]
            src   = os.path.join(config.DATA_DIR, "localcdn", rel)
            if os.path.isfile(src):
                shutil.copy2(src, local_path)
                return
        raise FileNotFoundError(
            f"Offline mode: cannot download {url} — local copy not found."
        )
    urllib.request.urlretrieve(url, local_path)
