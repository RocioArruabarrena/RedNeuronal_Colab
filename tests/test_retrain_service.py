# tests/test_retrain_service.py
"""
Tests de app/services/servicio_reentrenamiento.py.

RetrainService solo coordina: las colaboradoras (entrenador, almacenes,
fábrica) se reemplazan por dobles de prueba que se pasan por constructor.
Los modelos son redes de juguete que devuelven probabilidades conocidas,
así los resultados esperados se calculan a mano.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.ml.criterios_aceptacion import CriterioAceptacion
from app.ml.dataset_feedback import DatasetReentrenamiento
from app.models.feedback import FEATURE_COLUMNS
from app.services.servicio_reentrenamiento import MIN_FEEDBACK, RetrainService


class ModeloConstante(torch.nn.Module):
    """Devuelve siempre la misma probabilidad, sin importar la entrada."""

    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        return torch.full((len(x), 1), self.prob)


class SiempreRechaza(CriterioAceptacion):
    """Criterio de prueba: demuestra que el service acepta cualquier criterio."""

    def acepta(self, antes: dict, despues: dict) -> bool:
        return False

    def motivo_rechazo(self, antes: dict, despues: dict) -> str:
        return "rechazado por la prueba"


def _feedback(id_, user_id, friend_id, real_label):
    features = {col: 0.5 for col in FEATURE_COLUMNS}
    return SimpleNamespace(
        id=id_,
        user_id=user_id,
        friend_id=friend_id,
        real_label=real_label,
        **features,
    )


def _pares_distintos(cantidad):
    return [_feedback(i + 1, 1, i + 10, i % 2 == 0) for i in range(cantidad)]


@pytest.fixture
def entorno(tmp_path):
    """Split sintético chico en disco y colaboradoras falsas."""
    n = len(FEATURE_COLUMNS)
    rng = np.random.default_rng(0)
    split_path = tmp_path / "splits.npz"
    np.savez(
        split_path,
        X_train=rng.random((50, n)).astype(np.float32),
        y_train=np.array([0, 1] * 25, dtype=np.float32),
        X_val=rng.random((10, n)).astype(np.float32),
        y_val=np.array([0, 1] * 5, dtype=np.float32),
        X_test=rng.random((20, n)).astype(np.float32),
        y_test=np.array([0, 1] * 10, dtype=np.float32),
    )
    return SimpleNamespace(
        split_path=split_path,
        entrenador=MagicMock(name="Entrenador"),
        almacen_modelo=MagicMock(name="AlmacenModelo"),
        almacen_metricas=MagicMock(name="AlmacenMetricas"),
        fabrica=MagicMock(name="ModelFactory"),
    )


def _servicio(entorno, feedback, **extra):
    repo = MagicMock(name="repositorio_feedback")
    repo.get_all_feedback.return_value = feedback
    return RetrainService(
        repo,
        entrenador=entorno.entrenador,
        almacen_modelo=entorno.almacen_modelo,
        almacen_metricas=entorno.almacen_metricas,
        fabrica=entorno.fabrica,
        split_path=entorno.split_path,
        **extra,
    )


class TestValidaciones:

    def test_pocos_pares_distintos_levanta_value_error(self, entorno):
        servicio = _servicio(entorno, _pares_distintos(MIN_FEEDBACK - 1))

        with pytest.raises(ValueError, match="al menos"):
            servicio.retrain()

        entorno.entrenador.entrenar.assert_not_called()

    def test_feedback_repetido_del_mismo_par_no_cuenta_doble(self, entorno):
        repetido = [_feedback(i, 1, 2, True) for i in range(1, 11)]

        with pytest.raises(ValueError, match="al menos"):
            _servicio(entorno, repetido).retrain()

        entorno.entrenador.entrenar.assert_not_called()

    def test_sin_split_levanta_file_not_found(self, entorno):
        entorno.split_path.unlink()
        servicio = _servicio(entorno, _pares_distintos(MIN_FEEDBACK))

        with pytest.raises(FileNotFoundError):
            servicio.retrain()

        entorno.entrenador.entrenar.assert_not_called()


class TestRetrain:

    def test_reentrena_guarda_y_devuelve_las_metricas(self, entorno):
        actual = ModeloConstante(0.1)
        nuevo = ModeloConstante(0.9)
        entorno.fabrica.get_model.return_value = actual
        entorno.entrenador.entrenar.return_value = (nuevo, None)
        feedback = _pares_distintos(6)
        feedback.append(_feedback(99, 1, 10, False))  # pisa al par (1, 10)

        resultado = _servicio(entorno, feedback).retrain()

        repeticiones = DatasetReentrenamiento().feedback_repeat
        assert resultado["feedback_count"] == 6
        assert resultado["train_size"] == 50 + 6 * repeticiones
        assert resultado["test_after"]["f1"] > resultado["test_before"]["f1"]
        assert resultado["feedback_after"]["recall"] == 1.0
        entorno.almacen_modelo.guardar.assert_called_once_with(nuevo)
        entorno.fabrica.reload.assert_called_once()

    def test_hace_fine_tuning_desde_el_modelo_actual(self, entorno):
        actual = ModeloConstante(0.9)
        entorno.fabrica.get_model.return_value = actual
        entorno.entrenador.entrenar.return_value = (ModeloConstante(0.9), None)

        _servicio(entorno, _pares_distintos(MIN_FEEDBACK)).retrain()

        kwargs = entorno.entrenador.entrenar.call_args.kwargs
        assert kwargs["modelo_inicial"] is actual
        assert kwargs["restaurar_mejor"] is False

    def test_guarda_las_metricas_que_devuelve(self, entorno):
        entorno.fabrica.get_model.return_value = ModeloConstante(0.1)
        entorno.entrenador.entrenar.return_value = (ModeloConstante(0.9), None)

        resultado = _servicio(entorno, _pares_distintos(MIN_FEEDBACK)).retrain()

        entorno.almacen_metricas.guardar.assert_called_once_with(resultado)


class TestCriterioDeAceptacion:

    def test_rechaza_el_modelo_si_empeora_en_test(self, entorno):
        entorno.fabrica.get_model.return_value = ModeloConstante(0.9)
        entorno.entrenador.entrenar.return_value = (ModeloConstante(0.1), None)
        servicio = _servicio(entorno, _pares_distintos(MIN_FEEDBACK))

        with pytest.raises(ValueError, match="empeoró"):
            servicio.retrain()

        entorno.almacen_modelo.guardar.assert_not_called()
        entorno.almacen_metricas.guardar.assert_not_called()
        entorno.fabrica.reload.assert_not_called()

    def test_acepta_el_modelo_si_no_empeora(self, entorno):
        entorno.fabrica.get_model.return_value = ModeloConstante(0.9)
        entorno.entrenador.entrenar.return_value = (ModeloConstante(0.9), None)

        _servicio(entorno, _pares_distintos(MIN_FEEDBACK)).retrain()

        entorno.almacen_modelo.guardar.assert_called_once()

    def test_se_puede_inyectar_otro_criterio(self, entorno):
        entorno.fabrica.get_model.return_value = ModeloConstante(0.1)
        entorno.entrenador.entrenar.return_value = (ModeloConstante(0.9), None)
        servicio = _servicio(
            entorno, _pares_distintos(MIN_FEEDBACK), criterio=SiempreRechaza()
        )

        with pytest.raises(ValueError, match="rechazado por la prueba"):
            servicio.retrain()

        entorno.almacen_modelo.guardar.assert_not_called()
