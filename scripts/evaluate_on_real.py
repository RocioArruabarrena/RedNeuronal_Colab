"""
scripts/evaluate_on_real.py

Evalúa el modelo CompatibilityNet (entrenado con el dataset SINTÉTICO)
sobre los datos REALES de Yo Adolescente ya cargados en PostgreSQL.

No reentrena nada: solo carga los pesos y corre inferencia.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import classification_report

from app.database import SessionLocal
from app.models.profile import Profile
from app.models.friendship import Friendship
from app.ml.model import CompatibilityNet
from scripts.train_model import calcular_features, FEATURE_NAMES

MODEL_PATH = "models/compatibility_net_sintetico.pt"


def cargar_datos_reales():
    """Arma X, y a partir de las friendships reales guardadas en PostgreSQL."""
    db = SessionLocal()
    try:
        friendships = db.query(Friendship).all()
        profiles_by_id = {p.id: p for p in db.query(Profile).all()}

        X, y = [], []
        for f in friendships:
            p1 = profiles_by_id.get(f.user_id)
            p2 = profiles_by_id.get(f.friend_id)
            if p1 is None or p2 is None:
                continue
            features = calcular_features(p1, p2)
            X.append(features)
            y.append(1 if f.is_mutual else 0)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    finally:
        db.close()


def cargar_modelo():
    model = CompatibilityNet(input_dim=len(FEATURE_NAMES))
    state_dict = torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    X, y = cargar_datos_reales()
    print(f"Ejemplos reales cargados: {len(X)} ({int(y.sum())} positivos)")

    if len(X) == 0:
        print(
            "No hay datos en la base. Corré antes scripts/extract_from_yo_adolescente.py"
        )
        return

    model = cargar_modelo()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # El modelo ya termina en Sigmoid (se entrenó con BCELoss),
        # así que la salida de forward() ya son probabilidades.
        probs = model(X_tensor).squeeze().numpy()

    print("\n--- Evaluación con distintos umbrales ---")
    for umbral in [0.50, 0.40, 0.35, 0.30, 0.25]:
        preds = (probs >= umbral).astype(int)
        reporte = classification_report(
            y,
            preds,
            target_names=["no_compatible", "compatible"],
            output_dict=True,
            zero_division=0,
        )
        p = reporte["compatible"]["precision"]
        r = reporte["compatible"]["recall"]
        f1 = reporte["compatible"]["f1-score"]
        print(
            f"Umbral {umbral:.2f} -> Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}"
        )

    print("\n--- Reporte completo con umbral 0.50 ---")
    preds_final = (probs >= 0.50).astype(int)
    print(
        classification_report(
            y,
            preds_final,
            target_names=["no_compatible", "compatible"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
