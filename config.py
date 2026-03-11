import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERNS_DIR = os.path.join(BASE_DIR, "patterns")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
PORT = int(os.environ.get("PORT", 8000))

# MongoDB
MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb+srv://admin:admin@cluster0.tzdyf6b.mongodb.net/?appName=Cluster0",
)
MONGODB_DB = os.environ.get("MONGODB_DB", "phonecdp")

# Cloudinary
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "dmjlwqdhh")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", "273192873281951")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "dX7OUUZ6MdBzKvdv84Bzh-j1lcg")

for d in [PATTERNS_DIR, UPLOADS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
