# scripts/train_neural_net.py
"""
Entrena CompatibilityNet con el split guardado en data/splits.npz.

Correr desde la raiz del proyecto, con el venv activado:
    python scripts/train_neural_net.py

La funcion entrenar() tambien es usada por scripts/feedback_loop.py
para reentrenar con hiperparametros ajustados si la evaluacion falla.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import CompatibilityNet

DATA_PATH = Path("data/splits.npz")
MODEL_PATH = Path("models/compatibility_net.pt")
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4


def cargar_split() -> dict[str, np.ndarray]:
    data = np.load(DATA_PATH)
    return {k: data[k] for k in data.files if k != "feature_names"}


def a_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X_t, y_t)


def entrenar(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> tuple[CompatibilityNet, float]:
    """Entrena CompatibilityNet y devuelve (modelo con mejores pesos, mejor_val_loss)."""
    train_ds = a_tensor_dataset(X_train, y_train)
    val_ds = a_tensor_dataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    modelo = CompatibilityNet(input_dim=X_train.shape[1])
    criterio = nn.BCELoss()
    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    mejor_val_loss = float("inf")
    epocas_sin_mejora = 0
    mejor_estado = None

    for epoca in range(1, max_epochs + 1):
        modelo.train()
        loss_train_total = 0.0
        for X_batch, y_batch in train_loader:
            optimizador.zero_grad()
            y_pred = modelo(X_batch)
            loss = criterio(y_pred, y_batch)
            loss.backward()
            optimizador.step()
            loss_train_total += loss.item() * X_batch.size(0)
        loss_train_prom = loss_train_total / len(train_ds)

        modelo.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                y_pred_val = modelo(X_val_batch)
                loss_val = criterio(y_pred_val, y_val_batch).item()

        if verbose:
            print(
                f"Epoca {epoca:3d} | train_loss: {loss_train_prom:.4f} | "
                f"val_loss: {loss_val:.4f}"
            )

        if loss_val < mejor_val_loss:
            mejor_val_loss = loss_val
            epocas_sin_mejora = 0
            mejor_estado = modelo.state_dict()
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= patience:
                if verbose:
                    print(
                        f"\nEarly stopping en epoca {epoca} "
                        f"(sin mejora por {patience} epocas)"
                    )
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    return modelo, mejor_val_loss


def main() -> None:
    split = cargar_split()

    modelo, mejor_val_loss = entrenar(
        split["X_train"], split["y_train"], split["X_val"], split["y_val"]
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), MODEL_PATH)
    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    print(f"Modelo guardado en: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
