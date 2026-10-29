# tests/test_fabrica_modelo.py
"""
Tests de app/ml/fabrica_modelo.py: carga, caché por (ruta, arquitectura),
reload y registro de arquitecturas nuevas.
"""

import pytest
import torch
import torch.nn as nn

from app.ml.fabrica_modelo import INPUT_DIM, ModelFactory
from app.ml.model import CompatibilityNet


class RedChica(nn.Module):
    """Arquitectura de juguete para probar registrar()."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.lineal = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.lineal(x))


@pytest.fixture(autouse=True)
def cache_limpia():
    ModelFactory.limpiar()
    yield
    ModelFactory.limpiar()
    ModelFactory._arquitecturas.pop("red_chica", None)
    ModelFactory._arquitecturas.pop("alias", None)


def _guardar_pesos(ruta, semilla=0, clase=CompatibilityNet):
    torch.manual_seed(semilla)
    modelo = clase(input_dim=INPUT_DIM)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), ruta)
    return modelo


class TestGetModel:

    def test_carga_el_modelo_en_modo_eval(self, tmp_path):
        ruta = tmp_path / "modelo.pt"
        _guardar_pesos(ruta)

        modelo = ModelFactory.get_model(str(ruta))

        assert isinstance(modelo, CompatibilityNet)
        assert modelo.training is False

    def test_carga_los_pesos_guardados(self, tmp_path):
        ruta = tmp_path / "modelo.pt"
        guardado = _guardar_pesos(ruta, semilla=3)

        cargado = ModelFactory.get_model(str(ruta))

        for a, b in zip(cargado.state_dict().values(), guardado.state_dict().values()):
            assert torch.equal(a, b)

    def test_misma_ruta_devuelve_la_misma_instancia(self, tmp_path):
        ruta = tmp_path / "modelo.pt"
        _guardar_pesos(ruta)

        assert ModelFactory.get_model(str(ruta)) is ModelFactory.get_model(str(ruta))

    def test_otra_ruta_no_devuelve_el_modelo_cacheado(self, tmp_path):
        ruta_a = tmp_path / "a.pt"
        ruta_b = tmp_path / "b.pt"
        _guardar_pesos(ruta_a, semilla=0)
        pesos_b = _guardar_pesos(ruta_b, semilla=1)

        modelo_a = ModelFactory.get_model(str(ruta_a))
        modelo_b = ModelFactory.get_model(str(ruta_b))

        assert modelo_a is not modelo_b
        for x, y in zip(modelo_b.state_dict().values(), pesos_b.state_dict().values()):
            assert torch.equal(x, y)

    def test_misma_ruta_con_otra_arquitectura_es_otro_modelo(self, tmp_path):
        ruta = tmp_path / "modelo.pt"
        _guardar_pesos(ruta)
        ModelFactory.registrar("alias", CompatibilityNet)

        normal = ModelFactory.get_model(str(ruta))
        con_alias = ModelFactory.get_model(str(ruta), arquitectura="alias")

        assert normal is not con_alias

    def test_ruta_inexistente_levanta_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No se encontró el modelo"):
            ModelFactory.get_model(str(tmp_path / "no_existe.pt"))

    def test_arquitectura_desconocida_levanta_value_error(self, tmp_path):
        ruta = tmp_path / "modelo.pt"
        _guardar_pesos(ruta)

        with pytest.raises(ValueError, match="Arquitectura desconocida"):
            ModelFactory.get_model(str(ruta), arquitectura="inventada")


class TestReload:

    def test_reload_vuelve_a_leer_el_archivo(self, tmp_path):
        ruta = tmp_path / "modelo.pt"
        _guardar_pesos(ruta, semilla=0)
        primero = ModelFactory.get_model(str(ruta))

        nuevos = _guardar_pesos(ruta, semilla=1)  # sobrescribe el archivo
        recargado = ModelFactory.reload(str(ruta))

        assert recargado is not primero
        for a, b in zip(recargado.state_dict().values(), nuevos.state_dict().values()):
            assert torch.equal(a, b)

    def test_despues_de_reload_get_model_devuelve_el_recargado(self, tmp_path):
        ruta = tmp_path / "modelo.pt"
        _guardar_pesos(ruta)
        ModelFactory.get_model(str(ruta))

        recargado = ModelFactory.reload(str(ruta))

        assert ModelFactory.get_model(str(ruta)) is recargado

    def test_reload_no_descarta_los_modelos_de_otras_rutas(self, tmp_path):
        ruta_a = tmp_path / "a.pt"
        ruta_b = tmp_path / "b.pt"
        _guardar_pesos(ruta_a, semilla=0)
        _guardar_pesos(ruta_b, semilla=1)
        modelo_b = ModelFactory.get_model(str(ruta_b))
        ModelFactory.get_model(str(ruta_a))

        ModelFactory.reload(str(ruta_a))

        assert ModelFactory.get_model(str(ruta_b)) is modelo_b


class TestRegistrar:

    def test_se_puede_registrar_y_cargar_una_arquitectura_nueva(self, tmp_path):
        ruta = tmp_path / "chica.pt"
        _guardar_pesos(ruta, clase=RedChica)
        ModelFactory.registrar("red_chica", RedChica)

        modelo = ModelFactory.get_model(str(ruta), arquitectura="red_chica")

        assert isinstance(modelo, RedChica)
        assert modelo.training is False
