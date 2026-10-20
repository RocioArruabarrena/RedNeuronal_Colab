import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.database import get_db

db = next(get_db())
for tabla in ("profiles", "friendships", "feedback"):
    total = db.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()
    print(f"{tabla}: {total}")