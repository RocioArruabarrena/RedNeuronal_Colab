"""Compara la distribución del entrenamiento sintético con datos reales.

Ejecutar desde la raíz del repositorio:
    python scripts/diagnostico_sintetico_vs_real.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sqlalchemy.exc import OperationalError

from scripts.evaluate_on_real import cargar_datos_reales, cargar_modelo
from scripts.train_model import FEATURE_NAMES
from scripts.train_neural_net import cargar_split


def _print_histograma(scores: np.ndarray) -> None:
    counts, edges = np.histogram(scores, bins=10, range=(0.0, 1.0))
    max_count = max(int(counts.max()), 1)
    for index, count in enumerate(counts):
        bar = "#" * int(round(30 * count / max_count))
        print(f"  {edges[index]:.1f}-{edges[index + 1]:.1f}: {bar} ({count})")


def _mejor_umbral(
    y_real: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float, float]:
    candidates = np.append(np.unique(scores), np.nextafter(scores.max(), np.inf))
    mejor = None

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_real, predictions, average="binary", zero_division=0
        )
        accuracy = accuracy_score(y_real, predictions)
        candidate = (
            float(f1),
            float(precision),
            float(threshold),
            float(recall),
            float(accuracy),
        )
        if mejor is None or candidate[:3] > mejor[:3]:
            mejor = candidate

    assert mejor is not None
    f1, precision, threshold, recall, accuracy = mejor
    return threshold, accuracy, precision, recall, f1


def main() -> None:
    split = cargar_split()
    X_sintetico = split["X_train"]
    y_sintetico = split["y_train"].reshape(-1)
    try:
        X_real, y_real = cargar_datos_reales()
    except UnicodeDecodeError:
        raise SystemExit(
            "No se pudieron cargar los datos reales: PostgreSQL rechazó la "
            "conexión y psycopg2 no pudo decodificar el mensaje. Revisá "
            "DATABASE_URL y la codificación del servidor."
        ) from None
    except OperationalError as exc:
        raise SystemExit(
            "No se pudieron cargar los datos reales: falla de conexión "
            f"PostgreSQL ({type(exc.orig).__name__}). Revisá DATABASE_URL."
        ) from None
    y_real = y_real.reshape(-1)

    if (
        X_sintetico.shape[1] != len(FEATURE_NAMES)
        or X_real.shape[1] != len(FEATURE_NAMES)
    ):
        raise ValueError(
            "El número de features no coincide con FEATURE_NAMES: "
            f"sintético={X_sintetico.shape}, real={X_real.shape}, "
            f"nombres={len(FEATURE_NAMES)}"
        )
    if len(X_real) != len(y_real) or len(X_sintetico) != len(y_sintetico):
        raise ValueError("La cantidad de features y etiquetas no coincide.")
    if len(X_real) == 0:
        raise ValueError("No se cargaron ejemplos reales.")

    modelo = cargar_modelo()
    with torch.no_grad():
        X_tensor = torch.tensor(X_real, dtype=torch.float32)
        scores = modelo(X_tensor).reshape(-1).numpy()

    positivos_sinteticos = int(y_sintetico.sum())
    positivos_reales = int(y_real.sum())

    print("=== Diagnóstico: sintético vs real ===")
    print(
        f"Ejemplos sintéticos de entrenamiento: {len(y_sintetico)} "
        f"({positivos_sinteticos} positivos)"
    )
    print(f"Ejemplos reales: {len(y_real)} ({positivos_reales} positivos)")
    print("Scaler: no usado por CompatibilityNet; inferencia sobre features crudas.")

    percentiles = np.percentile(scores, [10, 25, 75, 90])
    print("\n--- Scores crudos (sigmoid) sobre reales ---")
    print(
        f"min={scores.min():.6f} max={scores.max():.6f} "
        f"media={scores.mean():.6f} mediana={np.median(scores):.6f}"
    )
    print(
        "percentiles: "
        f"P10={percentiles[0]:.6f} P25={percentiles[1]:.6f} "
        f"P75={percentiles[2]:.6f} P90={percentiles[3]:.6f}"
    )
    print("Histograma (10 bins en [0, 1]):")
    _print_histograma(scores)

    print("\n--- Score medio por etiqueta real ---")
    for label, name in ((0, "no compatible"), (1, "compatible")):
        mask = y_real == label
        mean_score = scores[mask].mean() if mask.any() else float("nan")
        print(f"{name}: n={int(mask.sum())}, media={mean_score:.6f}")

    if np.unique(y_real).size == 2:
        print(f"\nAUC-ROC real: {roc_auc_score(y_real, scores):.6f}")
    else:
        print("\nAUC-ROC real: indefinido (el dataset contiene una sola clase)")

    print("\n--- Distribución de features ---")
    print("Feature | media sintética | desvío sintético | media real | desvío real | SMD")
    for index, name in enumerate(FEATURE_NAMES):
        synthetic_values = X_sintetico[:, index]
        real_values = X_real[:, index]
        synthetic_std = float(np.std(synthetic_values, ddof=1))
        real_std = float(np.std(real_values, ddof=1))
        pooled_std = np.sqrt(
            (
                (len(synthetic_values) - 1) * synthetic_std**2
                + (len(real_values) - 1) * real_std**2
            )
            / (len(synthetic_values) + len(real_values) - 2)
        )
        smd = (
            (float(real_values.mean()) - float(synthetic_values.mean())) / pooled_std
            if pooled_std > 0
            else float("nan")
        )
        print(
            f"{name} | {synthetic_values.mean():.6f} | {synthetic_std:.6f} | "
            f"{real_values.mean():.6f} | {real_std:.6f} | {smd:+.6f}"
        )
    print("SMD = (media real - media sintética) / desvío estándar combinado.")

    print("\n--- Proporción de clase positiva ---")
    print(
        f"Sintético: {positivos_sinteticos}/{len(y_sintetico)} "
        f"({positivos_sinteticos / len(y_sintetico):.4%})"
    )
    print(
        f"Real: {positivos_reales}/{len(y_real)} "
        f"({positivos_reales / len(y_real):.4%})"
    )

    threshold, accuracy, precision, recall, f1 = _mejor_umbral(y_real, scores)
    print("\n--- Mejor umbral real (máximo F1) ---")
    print(
        f"umbral={threshold:.6f} accuracy={accuracy:.6f} "
        f"precision={precision:.6f} recall={recall:.6f} F1={f1:.6f}"
    )


if __name__ == "__main__":
    main()
