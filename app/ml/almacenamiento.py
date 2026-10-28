import json
import shutil
from pathlib import Path

import torch


class AlmacenModelo:
    """Guarda los pesos del modelo en disco (con copia de respaldo)."""

    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)

    def guardar(self, modelo) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        if self.ruta.exists():
            shutil.copy2(self.ruta, self.ruta.with_suffix(".bak.pt"))
        torch.save(modelo.state_dict(), self.ruta)


class AlmacenMetricas:
    """Guarda y lee las métricas del último reentrenamiento (JSON)."""

    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)

    def guardar(self, resultado: dict) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ruta.write_text(json.dumps(resultado, indent=2), encoding="utf-8")

    def leer(self) -> dict:
        if not self.ruta.exists():
            raise FileNotFoundError(f"No hay métricas guardadas en {self.ruta}")
        return json.loads(self.ruta.read_text(encoding="utf-8"))
