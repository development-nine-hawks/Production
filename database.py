"""
MongoDB + Cloudinary database layer for PhoneCDP.
Replaces SQLAlchemy/SQLite with pymongo and cloudinary.
"""
import urllib.request
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient, ReturnDocument, ASCENDING, DESCENDING
import config

_client = None
_db = None


def init_db():
    """Connect to MongoDB Atlas, create indexes, configure Cloudinary."""
    global _client, _db

    _client = MongoClient(config.MONGODB_URI)
    _db = _client[config.MONGODB_DB]

    # Indexes
    _db.patterns.create_index([("id", ASCENDING)], unique=True)
    _db.patterns.create_index([("created_at", DESCENDING)])
    _db.verifications.create_index([("id", ASCENDING)], unique=True)
    _db.verifications.create_index([("pattern_id", ASCENDING)])
    _db.verifications.create_index([("created_at", DESCENDING)])

    # Cloudinary
    cloudinary.config(
        cloud_name=config.CLOUDINARY_CLOUD_NAME,
        api_key=config.CLOUDINARY_API_KEY,
        api_secret=config.CLOUDINARY_API_SECRET,
    )


def get_db():
    """Return the pymongo database object."""
    return _db


def get_next_id(collection_name: str) -> int:
    """Auto-increment integer ID via a counters collection."""
    doc = _db.counters.find_one_and_update(
        {"_id": collection_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["seq"]


def upload_to_cloudinary(file_path: str, folder: str, public_id: str | None = None):
    """Upload a local file to Cloudinary. Returns dict with secure_url and public_id."""
    result = cloudinary.uploader.upload(
        file_path,
        folder=folder,
        public_id=public_id,
        overwrite=True,
        resource_type="image",
    )
    return {"secure_url": result["secure_url"], "public_id": result["public_id"]}


def destroy_cloudinary(public_id: str):
    """Delete an image from Cloudinary by public_id."""
    try:
        cloudinary.uploader.destroy(public_id, resource_type="image")
    except Exception:
        pass


def download_from_cloudinary(url: str, local_path: str):
    """Download a Cloudinary image to a local temp path."""
    urllib.request.urlretrieve(url, local_path)
