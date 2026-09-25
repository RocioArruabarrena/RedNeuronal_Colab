"""
Script de extracción de datos reales desde Yo Adolescente (SQLite)
hacia la base de datos del predictor de compatibilidad (PostgreSQL).

Qué hace:
1. Lee usuarios de Yo Adolescente y los carga como Profile
   (username, interests <- tags de perfil + subcultura,
   tags_agregados <- hashtags de posts públicos,
   activity_score derivado)
2. Lee friendships con status='accepted' y las carga como
   Friendship con is_mutual=True
3. Genera pares negativos (usuarios que NO son amigos entre sí,
   ni accepted ni pending) con is_mutual=False, en igual cantidad
   que los positivos para no desbalancear el dataset
4. Inserta todo en profiles y friendships de PostgreSQL

Correr desde la raíz del proyecto Red-Neuronal, con el venv activado:
    python scripts/extract_from_yo_adolescente.py
"""

import json
import random
import sqlite3
import sys
from pathlib import Path

# Permite correr este script desde scripts/ importando el paquete
# "app" que vive en la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from app.database import engine
from app.models.profile import Profile
from app.models.friendship import Friendship

# --- Configuración -----------------------------------------------------

SQLITE_PATH = Path(
    r"C:\Users\arruabarrena\Desktop\Arruabarrena-Rocio\example-app\database\database.sqlite"
)

# Cuántos negativos generar por cada positivo (1 = dataset balanceado)
NEGATIVE_RATIO = 1

random.seed(42)  # reproducibilidad


def leer_usuarios_sqlite(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT id, username, tags, bio, mood, location,
               is_online, is_moderator, is_public, pixel_avatar
        FROM users
        WHERE deleted_at IS NULL
        """)
    return [dict(row) for row in cur.fetchall()]


def leer_posts_por_usuario_sqlite(conn: sqlite3.Connection) -> dict[int, dict]:
    """
    Agrega, por usuario, todos los hashtags usados en sus posts
    públicos y la actividad total (likes + comments recibidos +
    cantidad de posts). Solo posts públicos y no borrados (soft
    delete) — los tags de posts privados no deberían influir en
    una señal de afinidad "social".
    """
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT user_id, tags, likes_count, comments_count
        FROM posts
        WHERE deleted_at IS NULL AND is_public = 1
        """)

    agregados: dict[int, dict] = {}
    for row in cur.fetchall():
        uid = row["user_id"]
        entry = agregados.setdefault(
            uid, {"tags": set(), "likes": 0, "comments": 0, "posts": 0}
        )

        tags_raw = row["tags"]
        if tags_raw:
            try:
                parsed = json.loads(tags_raw)
                if isinstance(parsed, list):
                    entry["tags"].update(str(t) for t in parsed)
            except (json.JSONDecodeError, TypeError):
                pass

        entry["likes"] += row["likes_count"] or 0
        entry["comments"] += row["comments_count"] or 0
        entry["posts"] += 1

    return agregados


def leer_friendships_sqlite(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT requester_id, receiver_id, status FROM friendships")
    return [dict(row) for row in cur.fetchall()]


def calcular_activity_score(usuario: dict, posts_info: dict | None) -> int:
    """
    Score 0-100 basado en actividad real de posts (likes + comments
    recibidos + cantidad de posts) más algunas señales de perfil.
    Los topes de normalización (100 likes+comments, 10 posts) son
    arbitrarios — ajustar si ves que casi todos quedan pegados al
    máximo o al mínimo una vez que tengas más datos reales.
    """
    score = 0

    if posts_info:
        engagement = posts_info["likes"] + posts_info["comments"]
        score += min(engagement, 100) * 0.5  # hasta 50 puntos
        score += min(posts_info["posts"], 10) * 3  # hasta 30 puntos

    if usuario.get("is_public"):
        score += 10
    if usuario.get("bio"):
        score += 10

    return int(min(score, 100))


def extraer_intereses(usuario: dict) -> list[str]:
    """
    tags puede venir como JSON serializado, CSV, o vacío según
    cómo lo hayan cargado. Probamos JSON primero y si falla,
    separamos por coma. Sumamos el pixel_avatar como tag de
    subcultura. Los hashtags de posts van aparte, en
    tags_agregados (ver extraer_tags_posts) — no se mezclan acá
    para poder calcular similitud_intereses y similitud_tags_posts
    como features independientes.
    """
    intereses: set[str] = set()
    tags = usuario.get("tags")
    if tags:
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                intereses.update(str(t) for t in parsed)
            else:
                intereses.add(str(parsed))
        except (json.JSONDecodeError, TypeError):
            intereses.update(t.strip() for t in tags.split(",") if t.strip())

    if usuario.get("pixel_avatar"):
        intereses.add(f"subcultura:{usuario['pixel_avatar']}")

    return sorted(intereses)


def extraer_tags_posts(posts_info: dict | None) -> list[str]:
    """Hashtags usados en los posts públicos del usuario, para
    guardar en Profile.tags_agregados (columna separada de
    interests, ver migración a1b2c3d4e5f6)."""
    if not posts_info:
        return []
    return sorted(posts_info["tags"])


def main() -> None:
    print(f"Conectando a SQLite: {SQLITE_PATH}")
    sqlite_conn = sqlite3.connect(SQLITE_PATH)

    usuarios = leer_usuarios_sqlite(sqlite_conn)
    posts_por_usuario = leer_posts_por_usuario_sqlite(sqlite_conn)
    friendships_raw = leer_friendships_sqlite(sqlite_conn)
    sqlite_conn.close()

    print(f"Usuarios leídos: {len(usuarios)}")
    print(f"Usuarios con al menos un post público: {len(posts_por_usuario)}")
    print(f"Friendships leídas: {len(friendships_raw)}")

    positivos = [f for f in friendships_raw if f["status"] == "accepted"]
    print(f"Amistades aceptadas (positivos): {len(positivos)}")

    # Set de pares ya conocidos (en cualquier estado), para no generar
    # un "negativo" que en realidad ya es un pending o accepted
    pares_existentes = {
        frozenset((f["requester_id"], f["receiver_id"])) for f in friendships_raw
    }

    ids_usuarios = [u["id"] for u in usuarios]
    ids_con_posts = list(posts_por_usuario.keys())

    # --- Generar negativos --------------------------------------------
    # Los positivos están casi todos entre usuarios con posts públicos
    # (usuarios "activos"). Si generáramos negativos con random.sample
    # sobre TODOS los usuarios, casi ninguno caería entre dos usuarios
    # con posts -> el modelo aprendería "tiene posts" como shortcut en
    # vez de aprender de similitud de intereses/tags reales. Por eso
    # generamos la mayoría de los negativos también entre usuarios con
    # posts, para que la única diferencia real entre positivos y
    # negativos sea la afinidad (o no) entre sus tags/intereses.
    negativos: list[tuple[int, int]] = []
    intentos_max = len(positivos) * NEGATIVE_RATIO * 20
    intentos = 0
    objetivo = len(positivos) * NEGATIVE_RATIO

    # Proporción de negativos que buscamos sacar de la población "activa"
    # (con posts), calcada de la proporción real de positivos con posts.
    PROP_ACTIVOS = 0.9

    if len(ids_con_posts) >= 2:
        objetivo_activos = int(objetivo * PROP_ACTIVOS)
        while len(negativos) < objetivo_activos and intentos < intentos_max:
            intentos += 1
            a, b = random.sample(ids_con_posts, 2)
            par = frozenset((a, b))
            if par in pares_existentes:
                continue
            pares_existentes.add(par)
            negativos.append((a, b))

    print(
        f"Negativos generados entre usuarios con posts: {len(negativos)} "
        f"(de un objetivo de {int(objetivo * PROP_ACTIVOS)})"
    )

    # Completar el resto (o todo, si no había suficientes usuarios con
    # posts) con la población general, como antes.
    while len(negativos) < objetivo and intentos < intentos_max:
        intentos += 1
        a, b = random.sample(ids_usuarios, 2)
        par = frozenset((a, b))
        if par in pares_existentes:
            continue
        pares_existentes.add(par)
        negativos.append((a, b))

    print(f"Negativos generados (total): {len(negativos)}")

    # --- Cargar a PostgreSQL --------------------------------------------
    with Session(engine) as session:
        # Mapeo id original (Yo Adolescente) -> id nuevo (Profile)
        mapa_ids: dict[int, int] = {}

        for u in usuarios:
            posts_info = posts_por_usuario.get(u["id"])
            nuevos_intereses = extraer_intereses(u)
            nuevos_tags_posts = extraer_tags_posts(posts_info)
            nuevo_score = calcular_activity_score(u, posts_info)

            existente = (
                session.query(Profile).filter(Profile.username == u["username"]).first()
            )
            if existente:
                # Ya corrimos el script antes: actualizamos con los
                # datos enriquecidos (hashtags de posts, score real)
                # en vez de dejar los valores viejos.
                existente.interests = nuevos_intereses
                existente.tags_agregados = nuevos_tags_posts
                existente.activity_score = nuevo_score
                mapa_ids[u["id"]] = existente.id
                continue

            perfil = Profile(
                username=u["username"],
                interests=nuevos_intereses,
                tags_agregados=nuevos_tags_posts,
                activity_score=nuevo_score,
            )
            session.add(perfil)
            session.flush()  # para obtener perfil.id ya generado
            mapa_ids[u["id"]] = perfil.id

        session.commit()
        print(f"Profiles cargados/actualizados: {len(mapa_ids)}")

        # Positivos
        cargados_pos = 0
        for f in positivos:
            uid = mapa_ids.get(f["requester_id"])
            fid = mapa_ids.get(f["receiver_id"])
            if uid is None or fid is None:
                continue
            ya_existe = (
                session.query(Friendship)
                .filter(Friendship.user_id == uid, Friendship.friend_id == fid)
                .first()
            )
            if ya_existe:
                continue
            session.add(Friendship(user_id=uid, friend_id=fid, is_mutual=True))
            cargados_pos += 1

        # Negativos
        cargados_neg = 0
        for a, b in negativos:
            uid = mapa_ids.get(a)
            fid = mapa_ids.get(b)
            if uid is None or fid is None:
                continue
            ya_existe = (
                session.query(Friendship)
                .filter(Friendship.user_id == uid, Friendship.friend_id == fid)
                .first()
            )
            if ya_existe:
                continue
            session.add(Friendship(user_id=uid, friend_id=fid, is_mutual=False))
            cargados_neg += 1

        session.commit()
        print(f"Friendships positivas cargadas: {cargados_pos}")
        print(f"Friendships negativas cargadas: {cargados_neg}")

    print("Listo.")


if __name__ == "__main__":
    main()
