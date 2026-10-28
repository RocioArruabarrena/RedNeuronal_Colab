"""Carga los datos sintéticos (CSV) en la base de datos.

Uso, desde la raíz del repo:
    python scripts/cargar_sinteticos_db.py
    python scripts/cargar_sinteticos_db.py --reset

Con --reset borra friendships y profiles antes de cargar.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from sqlalchemy import delete, func, insert, select, text

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app.database import SessionLocal  # noqa: E402
from app.models.friendship import Friendship  # noqa: E402
from app.models.profile import Profile  # noqa: E402

PROFILES_CSV = RAIZ / "profiles_unificado.csv"
FRIENDSHIPS_CSV = RAIZ / "friendships_unificado.csv"


def leer_perfiles(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return [
            {
                "id": int(fila["id"]),
                "username": fila["username"],
                "interests": json.loads(fila["interests"]),
                "activity_score": int(fila["activity_score"]),
                "tags_agregados": json.loads(fila["tags_agregados"]),
            }
            for fila in csv.DictReader(f)
        ]


def leer_friendships(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return [
            {
                "id": int(fila["id"]),
                "user_id": int(fila["requester_id"]),
                "friend_id": int(fila["receiver_id"]),
                "is_mutual": fila["is_mutual"].strip().lower() == "true",
            }
            for fila in csv.DictReader(f)
        ]


def resetear_secuencias(db):
    """Como se insertan ids explícitos, hay que alinear las secuencias."""
    if db.bind.dialect.name != "postgresql":
        return
    for tabla in ("profiles", "friendships"):
        db.execute(
            text(
                f"SELECT setval(pg_get_serial_sequence('{tabla}', 'id'), "
                f"(SELECT MAX(id) FROM {tabla}))"
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    perfiles = leer_perfiles(PROFILES_CSV)
    friendships = leer_friendships(FRIENDSHIPS_CSV)

    with SessionLocal() as db:
        existentes = db.scalar(select(func.count()).select_from(Profile))
        if existentes and not args.reset:
            sys.exit(
                f"profiles ya tiene {existentes} filas. "
                "Usá --reset para borrar y recargar."
            )
        if args.reset:
            db.execute(delete(Friendship))
            db.execute(delete(Profile))

        db.execute(insert(Profile), perfiles)
        db.execute(insert(Friendship), friendships)
        resetear_secuencias(db)
        db.commit()

        print(
            f"Cargados {len(perfiles)} perfiles y " f"{len(friendships)} friendships."
        )


if __name__ == "__main__":
    main()
