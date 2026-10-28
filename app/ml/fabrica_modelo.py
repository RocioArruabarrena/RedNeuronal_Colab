from pathlib import Path

import torch
import torch.nn as nn

from app.ml.model import CompatibilityNet

INPUT_DIM = 5


class ModelFactory:
    """Patrón Factory: centraliza la carga/instanciación del modelo entrenado.

    Para sumar una arquitectura nueva se la registra con registrar(),
    sin modificar get_model() (principio abierto/cerrado).
    """

    _arquitecturas: dict[str, type[nn.Module]] = {
        "compatibility_net": CompatibilityNet,
    }
    _instance = None

    @classmethod
    def registrar(cls, nombre: str, clase: type[nn.Module]) -> None:
        cls._arquitecturas[nombre] = clase

    @classmethod
    def get_model(cls, model_path: str, arquitectura: str = "compatibility_net"):
        if cls._instance is None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
            if arquitectura not in cls._arquitecturas:
                raise ValueError(f"Arquitectura desconocida: {arquitectura}")
            modelo = cls._arquitecturas[arquitectura](input_dim=INPUT_DIM)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            modelo.load_state_dict(state_dict)
            modelo.eval()
            cls._instance = modelo
        return cls._instance

    @classmethod
    def reload(cls, model_path: str, arquitectura: str = "compatibility_net"):
        """Descarta el modelo en memoria y vuelve a cargarlo desde disco."""
        cls._instance = None
        return cls.get_model(model_path, arquitectura)
