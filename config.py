import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERNS_DIR = os.path.join(BASE_DIR, "patterns")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'phonecdp.db')}"
PORT = int(os.environ.get("PORT", 8000))

for d in [PATTERNS_DIR, UPLOADS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
