import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.config import settings
from app.ml.model_factory import ModelFactory
from app.models.feedback import FEATURE_COLUMNS
from app.repositories.friendship_repository import FriendshipRepository
from scripts.train_neural_net import entrenar

RAIZ = Path(__file__).resolve().parents[2]
SPLIT_PATH = RAIZ / "data" / "splits.npz"
METRICS_PATH = RAIZ / "models" / "ultimas_metricas.json"

UMBRAL = 0.5
MIN_FEEDBACK = 5  # mínimo de feedbacks para poder reentrenar
FEEDBACK_REPEAT = 20  # cuántas veces se repite el feedback en el train


def _metricas(modelo, X: np.ndarray, y: np.ndarray) -> dict:
    modelo.eval()
    with torch.no_grad():
        probs = modelo(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    pred = (probs >= UMBRAL).astype(int)
    y = y.astype(int)
    hay_dos_clases = len(set(y.tolist())) == 2
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, probs)), 4) if hay_dos_clases else None,
    }


def _feedback_a_arrays(feedback: list) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[getattr(f, col) for col in FEATURE_COLUMNS] for f in feedback],
        dtype=np.float32,
    )
    y = np.array([1.0 if f.real_label else 0.0 for f in feedback], dtype=np.float32)
    return X, y


class RetrainService:
    """Reentrena la red con el dataset sintético + el feedback acumulado."""

    def __init__(self, repository: FriendshipRepository):
        self.repository = repository

    def retrain(self) -> dict:
        feedback = self.repository.get_all_feedback()
        if len(feedback) < MIN_FEEDBACK:
            raise ValueError(
                f"Hacen falta al menos {MIN_FEEDBACK} feedbacks para reentrenar "
                f"(hay {len(feedback)})"
            )
        if not SPLIT_PATH.exists():
            raise FileNotFoundError(f"No se encontró el split en {SPLIT_PATH}")

        split = np.load(SPLIT_PATH)
        X_fb, y_fb = _feedback_a_arrays(feedback)

        modelo_actual = ModelFactory.get_model(settings.model_path)
        test_before = _metricas(modelo_actual, split["X_test"], split["y_test"])
        feedback_before = _metricas(modelo_actual, X_fb, y_fb)

        # El feedback se repite para que pese frente a los miles de sintéticos
        X_rep = np.tile(X_fb, (FEEDBACK_REPEAT, 1))
        y_rep = np.tile(y_fb, FEEDBACK_REPEAT)
        X_train = np.concatenate([split["X_train"], X_rep])
        y_train = np.concatenate([split["y_train"], y_rep])

        modelo_nuevo, _ = entrenar(
            X_train, y_train, split["X_val"], split["y_val"], verbose=False
        )
        modelo_nuevo.eval()
        test_after = _metricas(modelo_nuevo, split["X_test"], split["y_test"])
        feedback_after = _metricas(modelo_nuevo, X_fb, y_fb)

        self._guardar_modelo(modelo_nuevo)

        resultado = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feedback_count": len(feedback),
            "train_size": int(len(X_train)),
            "test_before": test_before,
            "test_after": test_after,
            "feedback_before": feedback_before,
            "feedback_after": feedback_after,
        }
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
        return resultado

    def _guardar_modelo(self, modelo) -> None:
        ruta = Path(settings.model_path)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        if ruta.exists():
            shutil.copy2(ruta, ruta.with_suffix(".bak.pt"))
        torch.save(modelo.state_dict(), ruta)
        ModelFactory.reload(settings.model_path)