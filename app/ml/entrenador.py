"""Entrenamiento de CompatibilityNet (sirve para entrenar y para fine-tuning)."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import CompatibilityNet

BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """Hiperparámetros del entrenamiento."""

    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    batch_size: int = BATCH_SIZE
    verbose: bool = True


class Entrenador:
    """Entrena una CompatibilityNet nueva o ajusta una existente."""

    def __init__(self, config: ConfigEntrenamiento | None = None):
        self.config = config or ConfigEntrenamiento()

    def entrenar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        modelo_inicial: CompatibilityNet | None = None,
        restaurar_mejor: bool = True,
    ) -> tuple[CompatibilityNet, float]:
        """Devuelve (modelo, mejor_val_loss).

        - modelo_inicial: si se pasa, se entrena una COPIA (fine-tuning).
        - restaurar_mejor: True devuelve los pesos de la época con mejor
          val_loss; False devuelve los de la última época.
        """
        cfg = self.config
        train_loader = self._crear_loader(
            X_train, y_train, cfg.batch_size, shuffle=True
        )
        val_loader = self._crear_loader(X_val, y_val, len(X_val), shuffle=False)

        modelo = self._crear_modelo(X_train.shape[1], modelo_inicial)
        criterio = nn.BCELoss()
        optimizador = torch.optim.Adam(
            modelo.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        mejor_val_loss = float("inf")
        epocas_sin_mejora = 0
        mejor_estado = None

        for epoca in range(1, cfg.max_epochs + 1):
            loss_train = self._epoca_entrenamiento(
                modelo, train_loader, criterio, optimizador
            )
            loss_val = self._perdida_validacion(modelo, val_loader, criterio)
            if cfg.verbose:
                print(
                    f"Epoca {epoca:3d} | train_loss: {loss_train:.4f} | "
                    f"val_loss: {loss_val:.4f}"
                )

            if loss_val < mejor_val_loss:
                mejor_val_loss = loss_val
                epocas_sin_mejora = 0
                mejor_estado = copy.deepcopy(modelo.state_dict())
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= cfg.patience:
                    if cfg.verbose:
                        print(
                            f"\nEarly stopping en epoca {epoca} "
                            f"(sin mejora por {cfg.patience} epocas)"
                        )
                    break

        if restaurar_mejor and mejor_estado is not None:
            modelo.load_state_dict(mejor_estado)
        return modelo, mejor_val_loss

    @staticmethod
    def _crear_loader(
        X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle
        )

    @staticmethod
    def _crear_modelo(
        input_dim: int, modelo_inicial: CompatibilityNet | None
    ) -> CompatibilityNet:
        if modelo_inicial is not None:
            return copy.deepcopy(modelo_inicial)
        return CompatibilityNet(input_dim=input_dim)

    @staticmethod
    def _epoca_entrenamiento(modelo, loader, criterio, optimizador) -> float:
        modelo.train()
        total = 0.0
        for X_batch, y_batch in loader:
            optimizador.zero_grad()
            loss = criterio(modelo(X_batch), y_batch)
            loss.backward()
            optimizador.step()
            total += loss.item() * X_batch.size(0)
        return total / len(loader.dataset)

    @staticmethod
    def _perdida_validacion(modelo, loader, criterio) -> float:
        modelo.eval()
        total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                total += criterio(modelo(X_batch), y_batch).item() * X_batch.size(0)
        return total / len(loader.dataset)


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
    modelo_inicial: CompatibilityNet | None = None,
    restaurar_mejor: bool = True,
) -> tuple[CompatibilityNet, float]:
    """Atajo con la firma de siempre, para scripts/feedback_loop.py."""
    config = ConfigEntrenamiento(
        max_epochs, patience, learning_rate, weight_decay, batch_size, verbose
    )
    return Entrenador(config).entrenar(
        X_train, y_train, X_val, y_val, modelo_inicial, restaurar_mejor
    )
