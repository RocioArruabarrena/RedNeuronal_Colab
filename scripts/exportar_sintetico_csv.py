"""Exporta features sinteticas desde los CSV unificados."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
PROFILES_CSV = ROOT_DIR / "profiles_unificado.csv"
FRIENDSHIPS_CSV = ROOT_DIR / "friendships_unificado.csv"
OUTPUT_CSV = ROOT_DIR / "sintetico_features.csv"

FEATURE_NAMES = [
    "similitud_intereses",
    "diff_activity",
    "similitud_tags_posts",
    "tiene_tags_posts",
    "mismo_avatar_subcultura",
]


def _parse_json_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _extraer_subcultura(intereses: list[str] | None) -> str | None:
    if not intereses:
        return None
    for tag in intereses:
        if isinstance(tag, str) and tag.startswith("subcultura:"):
            return tag.split(":", 1)[1]
    return None


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


def cargar_perfiles(path: Path) -> dict[str, dict]:
    columns = ["id", "pixel_avatar", "interests", "activity_score", "tags_agregados"]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={"id": str},
        keep_default_na=False,
    )
    profiles = {}
    for row in frame.to_dict(orient="records"):
        interests = _parse_json_list(row["interests"])
        if row["pixel_avatar"]:
            interests = [*interests, f"subcultura:{row['pixel_avatar']}"]
        profiles[row["id"]] = {
            "interests": interests,
            "activity_score": float(row["activity_score"] or 0),
            "tags_agregados": _parse_json_list(row["tags_agregados"]),
        }
    return profiles


def _parse_label(value: str) -> int:
    return int(value.strip().lower() in ("true", "1"))


def cargar_features_y_labels(
    profiles: dict[str, dict], path: Path
) -> tuple[np.ndarray, np.ndarray]:
    friendships = pd.read_csv(
        path,
        usecols=["requester_id", "receiver_id", "is_mutual"],
        dtype=str,
        keep_default_na=False,
    )
    features = []
    labels = []
    for row in friendships.to_dict(orient="records"):
        profile_a = profiles.get(row["requester_id"])
        profile_b = profiles.get(row["receiver_id"])
        if profile_a is None or profile_b is None:
            continue
        features.append(calcular_features(profile_a, profile_b))
        labels.append(_parse_label(row["is_mutual"]))

    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def main() -> None:
    profiles = cargar_perfiles(PROFILES_CSV)
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
