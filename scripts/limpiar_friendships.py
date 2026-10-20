import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.database import get_db

CONDICION = "id > 8000 AND compatibility_score IS NOT NULL"

db = next(get_db())
filas = db.execute(
    text(f"SELECT id, user_id, friend_id, compatibility_score FROM friendships WHERE {CONDICION} ORDER BY id")
).fetchall()

print(f"Se van a borrar {len(filas)} filas:")
for f in filas:
    print(f)

if filas and input("\nEscribí 'si' para confirmar: ").strip().lower() == "si":
    db.execute(text(f"DELETE FROM friendships WHERE {CONDICION}"))
    db.commit()
    print("Listo.")
else:
    print("No se borró nada.")