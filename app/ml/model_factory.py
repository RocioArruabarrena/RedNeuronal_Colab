from pathlib import Path

import torch

from app.ml.model import CompatibilityNet

class ModelFactory:
    """Patrón Factory: centraliza la carga/instanciación del modelo entrenado."""

    _instance = None

    @classmethod
    def get_model(cls, model_path: str):
        if cls._instance is None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
            cls._instance = CompatibilityNet(input_dim=5)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            cls._instance.load_state_dict(state_dict)
            cls._instance.eval()
        return cls._instance