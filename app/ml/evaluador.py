import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluador:
    """Calcula accuracy, precision, recall, F1 y AUC de un modelo."""

    def __init__(self, umbral: float = 0.5):
        self.umbral = umbral

    def evaluar(self, modelo, X: np.ndarray, y: np.ndarray) -> dict:
        probs = self._predecir(modelo, X)
        pred = (probs >= self.umbral).astype(int)
        y = y.astype(int)
        hay_dos_clases = len(set(y.tolist())) == 2
        return {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y, probs)), 4) if hay_dos_clases else None,
        }

    @staticmethod
    def _predecir(modelo, X: np.ndarray) -> np.ndarray:
        modelo.eval()
        with torch.no_grad():
            return modelo(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
