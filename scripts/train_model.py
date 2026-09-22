"""
Entrena el modelo de compatibilidad de amistad con los datos ya
cargados en PostgreSQL (profiles + friendships).

Features por par de perfiles:
- similitud_intereses: Jaccard entre tags de perfil + subcultura (interests)
- diff_activity: diferencia absoluta de activity_score
- similitud_tags_posts: Jaccard entre hashtags de posts públicos (tags_agregados)
- tiene_tags_posts: 1 si ambos perfiles tienen al menos un tag de posts, 0 si a
  alguno le falta (para no confundir "sin datos" con "sin afinidad")
- mismo_avatar_subcultura: 1 si comparten el mismo tag "subcultura:X", si no 0

Label: friendship.is_mutual (True/False)

Correr desde la raíz del proyecto, con el venv activado:
    python scripts/train_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.config import settings
from app.database import engine
from app.models.friendship import Friendship
from app.models.profile import Profile


def extraer_subcultura(intereses: list[str] | None) -> str | None:
    if not intereses:
        return None
    for tag in intereses:
        if isinstance(tag, str) and tag.startswith("subcultura:"):
            return tag.split(":", 1)[1]
    return None


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def calcular_features(perfil_a: Profile, perfil_b: Profile) -> list[float]:
    intereses_a = set(perfil_a.interests or [])
    intereses_b = set(perfil_b.interests or [])
    similitud_intereses = jaccard(intereses_a, intereses_b)

    diff_activity = abs((perfil_a.activity_score or 0) - (perfil_b.activity_score or 0))

    tags_posts_a = set(perfil_a.tags_agregados or [])
    tags_posts_b = set(perfil_b.tags_agregados or [])
    similitud_tags_posts = jaccard(tags_posts_a, tags_posts_b)
    tiene_tags_posts = 1.0 if tags_posts_a and tags_posts_b else 0.0

    sub_a = extraer_subcultura(perfil_a.interests)
    sub_b = extraer_subcultura(perfil_b.interests)
    mismo_avatar_subcultura = 1.0 if sub_a and sub_b and sub_a == sub_b else 0.0

    return [
        similitud_intereses,
        diff_activity,
        similitud_tags_posts,
        tiene_tags_posts,
        mismo_avatar_subcultura,
    ]


FEATURE_NAMES = [
    "similitud_intereses",
    "diff_activity",
    "similitud_tags_posts",
    "tiene_tags_posts",
    "mismo_avatar_subcultura",
]


def main() -> None:
    with Session(engine) as session:
        friendships = session.query(Friendship).all()
        perfiles_por_id = {p.id: p for p in session.query(Profile).all()}

        X = []
        y = []

        for f in friendships:
            perfil_a = perfiles_por_id.get(f.user_id)
            perfil_b = perfiles_por_id.get(f.friend_id)
            if perfil_a is None or perfil_b is None:
                continue

            X.append(calcular_features(perfil_a, perfil_b))
            y.append(1 if f.is_mutual else 0)

    X = np.array(X)
    y = np.array(y)

    print(f"Ejemplos totales: {len(y)} (positivos: {y.sum()}, negativos: {len(y) - y.sum()})")

    if len(y) < 10:
        print(
            "ADVERTENCIA: muy pocos ejemplos para entrenar algo confiable. "
            "Segui cargando datos o generá más negativos antes de confiar en este modelo."
        )

    idx_tiene_tags = FEATURE_NAMES.index("tiene_tags_posts")
    print("\n--- Cruce tiene_tags_posts vs label ---")
    for label_valor, nombre_label in [(1, "positivos"), (0, "negativos")]:
        mask = y == label_valor
        con_tags = (X[mask, idx_tiene_tags] == 1.0).sum()
        sin_tags = (X[mask, idx_tiene_tags] == 0.0).sum()
        print(f"{nombre_label}: con_tags_ambos={con_tags} | sin_tags_alguno={sin_tags}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LogisticRegression es sensible a la escala de las features:
    # diff_activity va de 0-100, el resto de 0-1. Sin escalar,
    # diff_activity dominaría el modelo solo por su magnitud.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression(
        class_weight="balanced",
        random_state=42,
    )
    modelo.fit(X_train_scaled, y_train)

    print("\n--- Coeficientes del modelo (peso de cada feature) ---")
    for nombre, coef in zip(FEATURE_NAMES, modelo.coef_[0]):
        print(f"{nombre}: {coef:+.3f}")

    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Evaluación con distintos umbrales ---")
    print("(0.5 = default; bajarlo detecta más positivos a costa de precision)")
    for umbral in [0.5, 0.4, 0.35, 0.3, 0.25]:
        y_pred_umbral = (y_proba >= umbral).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred_umbral, average="binary", zero_division=0
        )
        print(f"Umbral {umbral:.2f} -> Precision: {p:.2f} | Recall: {r:.2f} | F1: {f:.2f}")

    umbral_elegido = 0.35  # ajustar según los resultados de arriba
    y_pred = (y_proba >= umbral_elegido).astype(int)
    print(f"\n--- Reporte completo con umbral elegido ({umbral_elegido}) ---")
    print(classification_report(y_test, y_pred, target_names=["no_compatible", "compatible"]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")

    # --- Guardar modelo + scaler --------------------------------------------
    # El scaler viaja junto con el modelo porque el endpoint de
    # predicción necesita aplicar exactamente la misma transformación
    # a los pares nuevos antes de llamar a modelo.predict().
    model_path = Path(settings.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "modelo": modelo,
            "scaler": scaler,
            "umbral": umbral_elegido,
            "features": FEATURE_NAMES,
        },
        model_path,
    )
    print(f"\nModelo guardado en: {model_path.resolve()}")


if __name__ == "__main__":
    main()