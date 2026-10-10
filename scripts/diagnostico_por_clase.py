"""Desglosa por clase las features sintéticas y reales del clasificador.

Ejecutar desde la raíz del repositorio:
    python scripts/diagnostico_por_clase.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from scripts.evaluate_on_real import cargar_datos_reales, cargar_modelo
from scripts.train_model import FEATURE_NAMES
from scripts.train_neural_net import cargar_split


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    split = cargar_split()
    X_train = split["X_train"]
    y_train = split["y_train"].reshape(-1)
    X_test = split["X_test"]
    y_test = split["y_test"].reshape(-1)
    X_real, y_real = cargar_datos_reales()
    y_real = y_real.reshape(-1)

    if X_train.shape[1] != len(FEATURE_NAMES) or X_real.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Las matrices de features no coinciden con FEATURE_NAMES.")
    if len(X_real) != len(y_real):
        raise ValueError("La cantidad de ejemplos reales no coincide con las etiquetas.")
    if np.unique(y_real).size < 2 or np.unique(y_train).size < 2:
        raise ValueError("Se necesitan ambas clases para calcular los AUC por feature.")

    print("=== Estadísticas por clase (media y desvío estándar) ===")
    print("Dataset | Clase | n | Feature | Media | Desvío")
    for dataset_name, features, labels in (
        ("sintético train", X_train, y_train),
        ("real", X_real, y_real),
    ):
        for label, class_name in ((1, "compatible"), (0, "no compatible")):
            class_features = features[labels == label]
            for index, feature_name in enumerate(FEATURE_NAMES):
                mean, std = _mean_std(class_features[:, index])
                print(
                    f"{dataset_name} | {class_name} | {len(class_features)} | "
                    f"{feature_name} | {mean:.6f} | {std:.6f}"
                )

    print("\n=== AUC-ROC univariada (feature cruda como score) ===")
    print("Feature | Sintético train | Reales")
    for index, feature_name in enumerate(FEATURE_NAMES):
        synthetic_auc = _auc(y_train, X_train[:, index])
        real_auc = _auc(y_real, X_real[:, index])
        print(f"{feature_name} | {synthetic_auc:.6f} | {real_auc:.6f}")

    modelo = cargar_modelo()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_scores = modelo(test_tensor).reshape(-1).numpy()
    test_predictions = (test_scores >= 0.5).astype(int)
    print("\n=== CompatibilityNet sobre test sintético ===")
    print(
        f"n={len(y_test)} accuracy={accuracy_score(y_test, test_predictions):.6f} "
        f"AUC={_auc(y_test, test_scores):.6f} (umbral accuracy=0.50)"
    )

    print("\n=== Rango de features ===")
    print("Feature | Sintético train min..max | Real min..max")
    for index, feature_name in enumerate(FEATURE_NAMES):
        synthetic_values = X_train[:, index]
        real_values = X_real[:, index]
        print(
            f"{feature_name} | {synthetic_values.min():.6f}..{synthetic_values.max():.6f} | "
            f"{real_values.min():.6f}..{real_values.max():.6f}"
        )


if __name__ == "__main__":
    main()
