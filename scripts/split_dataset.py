# scripts/split_dataset.py
"""
Arma el split train/val/test (70/15/15) estratificado por clase,
a partir de los mismos pares profile-profile que usa train_model.py.

Correr desde la raíz del proyecto, con el venv activado:
    python scripts/split_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.database import engine
from app.models.friendship import Friendship
from app.models.profile import Profile
from scripts.train_model import calcular_features, FEATURE_NAMES


def load_features_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruye X, y de la misma forma que train_model.py,
    para que el split y el entrenamiento final usen exactamente
    los mismos pares y las mismas 5 features.
    """
    with Session(engine) as session:
        friendships = session.query(Friendship).all()
        perfiles_por_id = {p.id: p for p in session.query(Profile).all()}

        X, y = [], []
        for f in friendships:
            perfil_a = perfiles_por_id.get(f.user_id)
            perfil_b = perfiles_por_id.get(f.friend_id)
            if perfil_a is None or perfil_b is None:
                continue

            X.append(calcular_features(perfil_a, perfil_b))
            y.append(1 if f.is_mutual else 0)

    return np.array(X), np.array(y)


def split_dataset(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Split 70/15/15 (train/val/test), estratificado por clase.
    """
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
    X, y = load_features_and_labels()
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
