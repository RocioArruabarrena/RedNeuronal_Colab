# scripts/evaluate_model.py
"""
Evalúa CompatibilityNet ya entrenado sobre el set de test
(data/splits.npz), igual que hacía train_model.py con LogisticRegression.

Correr desde la raiz del proyecto, con el venv activado:
    python scripts/evaluate_model.py

Las funciones cargar_datos_test / cargar_modelo / evaluar tambien
son usadas por scripts/feedback_loop.py para evaluar sin reentrenar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support

from app.ml.model import CompatibilityNet

DATA_PATH = Path("data/splits.npz")
MODEL_PATH = Path("models/compatibility_net.pt")
UMBRAL_DEFAULT = 0.50


def cargar_datos_test(data_path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)
    return data["X_test"], data["y_test"]


def cargar_modelo(model_path: Path = MODEL_PATH, input_dim: int = 5) -> CompatibilityNet:
    modelo = CompatibilityNet(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo


def evaluar(
    modelo: CompatibilityNet,
    X_test: np.ndarray,
    y_test: np.ndarray,
    umbral: float = UMBRAL_DEFAULT,
) -> dict:
    """Corre inferencia sobre X_test y devuelve precision/recall/F1 (clase compatible)."""
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_proba = modelo(X_test_t).squeeze(1).numpy()

    y_pred = (y_proba >= umbral).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    return {
        "umbral": umbral,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def main() -> None:
    X_test, y_test = cargar_datos_test()
    modelo = cargar_modelo(input_dim=X_test.shape[1])

    print(f"Ejemplos de test: {len(y_test)} ({y_test.sum()} positivos)")

    print("\n--- Evaluacion con distintos umbrales ---")
    for umbral in [0.5, 0.4, 0.35, 0.3, 0.25]:
        r = evaluar(modelo, X_test, y_test, umbral=umbral)
        print(
            f"Umbral {umbral:.2f} -> Precision: {r['precision']:.2f} | "
            f"Recall: {r['recall']:.2f} | F1: {r['f1']:.2f}"
        )

    # A diferencia de LogisticRegression (donde 0.35 daba mejor F1),
    # con esta red 0.50 es el umbral que mejor discrimina: por debajo
    # de 0.5 el modelo tiende a predecir casi todo como positivo.
    resultado = evaluar(modelo, X_test, y_test, umbral=UMBRAL_DEFAULT)
    print(f"\n--- Reporte completo con umbral elegido ({UMBRAL_DEFAULT}) ---")
    print(
        classification_report(
            y_test,
            resultado["y_pred"],
            target_names=["no_compatible", "compatible"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()