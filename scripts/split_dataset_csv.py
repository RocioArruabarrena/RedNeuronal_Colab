# scripts/split_dataset_csv.py
"""
Arma el split train/val/test (70/15/15) estratificado por clase,
a partir de los CSV sinteticos unificados (profiles_unificado.csv +
friendships_unificado.csv) en vez de leer de PostgreSQL.

Reutiliza calcular_features() y FEATURE_NAMES de train_model.py para
que el split use exactamente la misma logica de features que el
entrenamiento final.

Correr desde la raiz del proyecto:
    python scripts/split_dataset_csv.py
"""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from scripts.train_model import calcular_features, FEATURE_NAMES

PROFILES_CSV = "profiles_unificado.csv"
FRIENDSHIPS_CSV = "friendships_unificado.csv"


def _parse_json_list(valor: str) -> list[str]:
    if not valor:
        return []
    try:
        return json.loads(valor)
    except (json.JSONDecodeError, TypeError):
        return []


def cargar_perfiles(path: str) -> dict[str, SimpleNamespace]:
    """
    Lee profiles_unificado.csv y arma un dict id -> perfil, con la
    misma forma (atributos) que espera calcular_features(): interests,
    activity_score, tags_agregados. Le agrega el tag "subcultura:X"
    a interests (a partir de pixel_avatar) para que
    mismo_avatar_subcultura funcione igual que con datos reales.
    """
    perfiles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            interests = _parse_json_list(fila.get("interests", ""))
            subcultura = fila.get("pixel_avatar")
            if subcultura:
                interests = [*interests, f"subcultura:{subcultura}"]

            perfiles[fila["id"]] = SimpleNamespace(
                interests=interests,
                activity_score=float(fila.get("activity_score") or 0),
                tags_agregados=_parse_json_list(fila.get("tags_agregados", "")),
            )
    return perfiles


def cargar_features_y_labels(
    perfiles: dict[str, SimpleNamespace], path: str
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            perfil_a = perfiles.get(fila["requester_id"])
            perfil_b = perfiles.get(fila["receiver_id"])
            if perfil_a is None or perfil_b is None:
                continue

            X.append(calcular_features(perfil_a, perfil_b))
            es_mutuo = fila["is_mutual"].strip().lower() in ("true", "1")
            y.append(1 if es_mutuo else 0)

    return np.array(X), np.array(y)


def split_dataset(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    perfiles = cargar_perfiles(PROFILES_CSV)
    X, y = cargar_features_y_labels(perfiles, FRIENDSHIPS_CSV)

    print(
        f"Dataset total: {len(y)} ejemplos ({y.sum()} positivos, {len(y) - y.sum()} negativos)"
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"Train: {len(X_train)} ejemplos ({y_train.sum()} positivos)")
    print(f"Val:   {len(X_val)} ejemplos ({y_val.sum()} positivos)")
    print(f"Test:  {len(X_test)} ejemplos ({y_test.sum()} positivos)")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    np.savez(
        data_dir / "splits.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=FEATURE_NAMES,
    )
    print(f"\nSplit guardado en: {(data_dir / 'splits.npz').resolve()}")
