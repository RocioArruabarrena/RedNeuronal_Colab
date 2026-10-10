"""Exporta features reales desde los CSV de perfiles y amistades."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
PROFILES_CSV = ROOT_DIR / "profiles.csv"
FRIENDSHIPS_CSV = ROOT_DIR / "friendships.csv"
OUTPUT_CSV = ROOT_DIR / "reales_features.csv"

FEATURE_NAMES = [
    "similitud_intereses",
    "diff_activity",
    "similitud_tags_posts",
    "tiene_tags_posts",
    "mismo_avatar_subcultura",
]


def _read_csv(path: Path, **options) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", **options)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **options)


def _parse_json_list(value: str) -> list:
    if not value:
        return []
    parsed = json.loads(value)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise ValueError(f"Se esperaba una lista JSON y se recibio: {value!r}")
    return parsed


def _extraer_subcultura(intereses: list | None) -> str | None:
    if not intereses:
        return None
    for tag in intereses:
        if isinstance(tag, str) and tag.startswith("subcultura:"):
            return tag.split(":", 1)[1]
    return None


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def calcular_features(perfil_a: dict, perfil_b: dict) -> list[float]:
    intereses_a = set(perfil_a["interests"] or [])
    intereses_b = set(perfil_b["interests"] or [])
    similitud_intereses = _jaccard(intereses_a, intereses_b)

    diff_activity = abs(
        (perfil_a["activity_score"] or 0) - (perfil_b["activity_score"] or 0)
    )

    tags_posts_a = set(perfil_a["tags_agregados"] or [])
    tags_posts_b = set(perfil_b["tags_agregados"] or [])
    similitud_tags_posts = _jaccard(tags_posts_a, tags_posts_b)
    tiene_tags_posts = 1.0 if tags_posts_a and tags_posts_b else 0.0

    sub_a = _extraer_subcultura(perfil_a["interests"])
    sub_b = _extraer_subcultura(perfil_b["interests"])
    mismo_avatar_subcultura = 1.0 if sub_a and sub_b and sub_a == sub_b else 0.0

    return [
        similitud_intereses,
        diff_activity,
        similitud_tags_posts,
        tiene_tags_posts,
        mismo_avatar_subcultura,
    ]


def _leer_perfiles(path: Path) -> dict[str, dict]:
    columns = ["id", "interests", "activity_score", "tags_agregados"]
    frame = _read_csv(
        path,
        usecols=columns,
        dtype={"id": str},
        keep_default_na=False,
    )
    profiles = {}
    for row in frame.to_dict(orient="records"):
        activity_score = float(row["activity_score"]) if row["activity_score"] else 0
        profiles[row["id"]] = {
            "interests": _parse_json_list(row["interests"]),
            "activity_score": activity_score,
            "tags_agregados": _parse_json_list(row["tags_agregados"]),
        }
    return profiles


def _parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in ("t", "true", "1"):
        return 1
    if normalized in ("f", "false", "0"):
        return 0
    raise ValueError(f"Etiqueta is_mutual no reconocida: {value!r}")


def cargar_features_y_labels(
    profiles: dict[str, dict], path: Path
) -> tuple[np.ndarray, np.ndarray]:
    friendships = _read_csv(
        path,
        usecols=["user_id", "friend_id", "is_mutual"],
        dtype=str,
        keep_default_na=False,
    )
    features = []
    labels = []
    for row in friendships.to_dict(orient="records"):
        profile_a = profiles.get(row["user_id"])
        profile_b = profiles.get(row["friend_id"])
        if profile_a is None or profile_b is None:
            continue
        features.append(calcular_features(profile_a, profile_b))
        labels.append(_parse_label(row["is_mutual"]))

    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def main() -> None:
    profiles = _leer_perfiles(PROFILES_CSV)
    features, labels = cargar_features_y_labels(profiles, FRIENDSHIPS_CSV)
    output = pd.DataFrame(features, columns=FEATURE_NAMES)
    output["y"] = labels
    output.to_csv(OUTPUT_CSV, index=False)
    print(
        f"Exportadas {len(output)} filas ({int(labels.sum())} positivas) "
        f"a {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
