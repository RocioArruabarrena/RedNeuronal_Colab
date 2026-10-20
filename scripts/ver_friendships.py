import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.database import get_db

db = next(get_db())
resultado = db.execute(text("SELECT * FROM friendships ORDER BY id DESC LIMIT 12"))
print(list(resultado.keys()))
for fila in resultado.fetchall():
    print(fila)