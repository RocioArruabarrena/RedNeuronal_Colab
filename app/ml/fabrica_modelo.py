from pathlib import Path

import torch
import torch.nn as nn

from app.ml.model import CompatibilityNet

INPUT_DIM = 5
ARQUITECTURA_POR_DEFECTO = "compatibility_net"


class ModelFactory:
    """Patrón Factory: centraliza la carga/instanciación del modelo entrenado.

    - Para sumar una arquitectura nueva se la registra con registrar(),
      sin modificar get_model() (principio abierto/cerrado).
    - Los modelos cargados se guardan en una caché por (ruta, arquitectura),
      así pedir otra ruta u otra arquitectura nunca devuelve el modelo equivocado.
    """

    _arquitecturas: dict[str, type[nn.Module]] = {
        ARQUITECTURA_POR_DEFECTO: CompatibilityNet,
    }
    _cache: dict[tuple[str, str], nn.Module] = {}

    @classmethod
    def registrar(cls, nombre: str, clase: type[nn.Module]) -> None:
        cls._arquitecturas[nombre] = clase

    @classmethod
    def get_model(
        cls, model_path: str, arquitectura: str = ARQUITECTURA_POR_DEFECTO
    ) -> nn.Module:
        clave = cls._clave(model_path, arquitectura)
        if clave not in cls._cache:
            cls._cache[clave] = cls._cargar(model_path, arquitectura)
        return cls._cache[clave]

    @classmethod
    def reload(
        cls, model_path: str, arquitectura: str = ARQUITECTURA_POR_DEFECTO
    ) -> nn.Module:
        """Descarta el modelo en memoria y vuelve a cargarlo desde disco."""
        cls._cache.pop(cls._clave(model_path, arquitectura), None)
        return cls.get_model(model_path, arquitectura)

    @classmethod
    def limpiar(cls) -> None:
        """Vacía la caché (útil en tests)."""
        cls._cache.clear()

    @staticmethod
    def _clave(model_path: str, arquitectura: str) -> tuple[str, str]:
        return (str(Path(model_path).resolve()), arquitectura)

    @classmethod
    def _cargar(cls, model_path: str, arquitectura: str) -> nn.Module:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        if arquitectura not in cls._arquitecturas:
            raise ValueError(f"Arquitectura desconocida: {arquitectura}")
        modelo = cls._arquitecturas[arquitectura](input_dim=INPUT_DIM)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        modelo.load_state_dict(state_dict)
        modelo.eval()
        return modelo
