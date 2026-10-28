"""
Entrena CompatibilityNet con el split guardado en data/splits.npz.

Correr desde la raiz del proyecto, con el venv activado:
    python scripts/entrenar_red.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from app.ml.entrenador import Entrenador

DATA_PATH = Path("data/splits.npz")
MODEL_PATH = Path("models/compatibility_net.pt")


def cargar_split() -> dict[str, np.ndarray]:
    data = np.load(DATA_PATH)
    return {k: data[k] for k in data.files if k != "feature_names"}


def main() -> None:
    split = cargar_split()

    modelo, mejor_val_loss = Entrenador().entrenar(
        split["X_train"], split["y_train"], split["X_val"], split["y_val"]
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), MODEL_PATH)
    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    print(f"Modelo guardado en: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
