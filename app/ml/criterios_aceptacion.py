from abc import ABC, abstractmethod


class CriterioAceptacion(ABC):
    """Decide si un modelo reentrenado reemplaza al anterior."""

    @abstractmethod
    def acepta(self, antes: dict, despues: dict) -> bool: ...

    @abstractmethod
    def motivo_rechazo(self, antes: dict, despues: dict) -> str: ...


class NoEmpeorarF1(CriterioAceptacion):
    """Acepta el modelo nuevo si el F1 de test no cae más que el margen."""

    def __init__(self, margen: float = 0.02):
        self.margen = margen

    def acepta(self, antes: dict, despues: dict) -> bool:
        return despues["f1"] >= antes["f1"] - self.margen

    def motivo_rechazo(self, antes: dict, despues: dict) -> str:
        return (
            "El modelo ajustado empeoró en los datos de prueba "
            f"(F1 {antes['f1']} -> {despues['f1']}), "
            "así que se conserva el modelo anterior"
        )
