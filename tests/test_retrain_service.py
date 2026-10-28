# tests/test_retrain_service.py
"""
Tests de app/services/retrain_service.py.

No entrenan nada de verdad: entrenar(), ModelFactory y los archivos
(split, métricas, modelo) se reemplazan por mocks y rutas temporales.
Los modelos son redes de juguete que devuelven probabilidades conocidas,
así los resultados esperados se calculan a mano.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.models.feedback import FEATURE_COLUMNS
from app.services import retrain_service
from app.services.retrain_service import (
    FEEDBACK_REPEAT,
    MIN_FEEDBACK,
    RetrainService,
    _feedback_a_arrays,
    _metricas,
    _ultimo_por_par,
)


class ModeloEco(torch.nn.Module):
    """Devuelve como probabilidad el valor de la primera feature."""

    def forward(self, x):
        return x[:, :1]


class ModeloConstante(torch.nn.Module):
    """Devuelve siempre la misma probabilidad, sin importar la entrada."""

    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        return torch.full((len(x), 1), self.prob)


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


def _servicio(feedback):
    repo = MagicMock(name="friendship_repository")
    repo.get_all_feedback.return_value = feedback
    return RetrainService(repo)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Split sintético chico en disco y todas las dependencias mockeadas."""
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
    metrics_path = tmp_path / "models" / "ultimas_metricas.json"

    factory = MagicMock(name="ModelFactory")
    guardar = MagicMock(name="guardar_modelo")
    entrenar = MagicMock(name="entrenar")

    monkeypatch.setattr(retrain_service, "SPLIT_PATH", split_path)
    monkeypatch.setattr(retrain_service, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(retrain_service, "ModelFactory", factory)
    monkeypatch.setattr(retrain_service, "entrenar", entrenar)
    monkeypatch.setattr(RetrainService, "_guardar_modelo", guardar)

    return SimpleNamespace(
        split_path=split_path,
        metrics_path=metrics_path,
        factory=factory,
        guardar=guardar,
        entrenar=entrenar,
    )


class TestMetricas:

    def test_clasificacion_perfecta(self):
        X = np.array([[0.9], [0.8], [0.2], [0.1]], dtype=np.float32)
        y = np.array([1, 1, 0, 0], dtype=np.float32)

        resultado = _metricas(ModeloEco(), X, y)

        assert resultado == {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "auc": 1.0,
        }

    def test_clasificacion_mixta(self):
        X = np.array([[0.9], [0.9], [0.1], [0.1]], dtype=np.float32)
        y = np.array([1, 0, 1, 0], dtype=np.float32)

        resultado = _metricas(ModeloEco(), X, y)

        assert resultado == {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "auc": 0.5,
        }

    def test_umbral_incluye_el_0_5(self):
        X = np.array([[0.5], [0.5]], dtype=np.float32)
        y = np.array([1, 1], dtype=np.float32)

        resultado = _metricas(ModeloEco(), X, y)

        assert resultado["recall"] == 1.0

    def test_una_sola_clase_deja_auc_en_none(self):
        X = np.array([[0.9], [0.7]], dtype=np.float32)
        y = np.array([1, 1], dtype=np.float32)

        resultado = _metricas(ModeloEco(), X, y)

        assert resultado["auc"] is None
        assert resultado["accuracy"] == 1.0


class TestUltimoPorPar:

    def test_se_queda_con_la_respuesta_mas_reciente_del_par(self):
        viejo = _feedback(1, 1, 2, False)
        nuevo = _feedback(7, 1, 2, True)

        resultado = _ultimo_por_par([nuevo, viejo])  # desordenados a propósito

        assert len(resultado) == 1
        assert resultado[0] is nuevo

    def test_pares_distintos_se_conservan(self):
        a = _feedback(1, 1, 2, True)
        b = _feedback(2, 2, 1, True)  # el par invertido es otro par
        c = _feedback(3, 1, 3, False)

        resultado = _ultimo_por_par([a, b, c])

        assert {f.id for f in resultado} == {1, 2, 3}

    def test_lista_vacia(self):
        assert _ultimo_por_par([]) == []


class TestFeedbackAArrays:

    def test_forma_tipo_y_etiquetas(self):
        feedback = [_feedback(1, 1, 2, True), _feedback(2, 1, 3, False)]

        X, y = _feedback_a_arrays(feedback)

        assert X.shape == (2, len(FEATURE_COLUMNS))
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        assert y.tolist() == [1.0, 0.0]

    def test_respeta_el_orden_de_feature_columns(self):
        valores = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
        fila = SimpleNamespace(real_label=True, **valores)

        X, _ = _feedback_a_arrays([fila])

        assert X[0].tolist() == [float(i) for i in range(len(FEATURE_COLUMNS))]


class TestRetrain:

    def test_pocos_pares_distintos_levanta_value_error(self, entorno):
        servicio = _servicio(_pares_distintos(MIN_FEEDBACK - 1))

        with pytest.raises(ValueError, match="al menos"):
            servicio.retrain()

        entorno.entrenar.assert_not_called()

    def test_feedback_repetido_del_mismo_par_no_cuenta_doble(self, entorno):
        repetido = [_feedback(i, 1, 2, True) for i in range(1, 11)]

        with pytest.raises(ValueError, match="al menos"):
            _servicio(repetido).retrain()

        entorno.entrenar.assert_not_called()

    def test_sin_split_levanta_file_not_found(self, entorno):
        entorno.split_path.unlink()
        servicio = _servicio(_pares_distintos(MIN_FEEDBACK))

        with pytest.raises(FileNotFoundError):
            servicio.retrain()

        entorno.entrenar.assert_not_called()

    def test_reentrena_guarda_y_devuelve_las_metricas(self, entorno):
        actual = ModeloConstante(0.1)
        nuevo = ModeloConstante(0.9)
        entorno.factory.get_model.return_value = actual
        entorno.entrenar.return_value = (nuevo, None)
        feedback = _pares_distintos(6)
        feedback.append(_feedback(99, 1, 10, False))  # pisa al par (1, 10)

        resultado = _servicio(feedback).retrain()

        assert resultado["feedback_count"] == 6
        assert resultado["train_size"] == 50 + 6 * FEEDBACK_REPEAT
        assert resultado["test_after"]["f1"] > resultado["test_before"]["f1"]
        assert resultado["feedback_after"]["recall"] == 1.0
        entorno.guardar.assert_called_once_with(nuevo)

    def test_hace_fine_tuning_desde_el_modelo_actual(self, entorno):
        actual = ModeloConstante(0.9)
        entorno.factory.get_model.return_value = actual
        entorno.entrenar.return_value = (ModeloConstante(0.9), None)

        _servicio(_pares_distintos(MIN_FEEDBACK)).retrain()

        kwargs = entorno.entrenar.call_args.kwargs
        assert kwargs["modelo_inicial"] is actual
        assert kwargs["restaurar_mejor"] is False

    def test_escribe_el_json_de_metricas(self, entorno):
        entorno.factory.get_model.return_value = ModeloConstante(0.1)
        entorno.entrenar.return_value = (ModeloConstante(0.9), None)

        resultado = _servicio(_pares_distintos(MIN_FEEDBACK)).retrain()

        guardado = json.loads(entorno.metrics_path.read_text(encoding="utf-8"))
        assert guardado == resultado

    def test_rechaza_el_modelo_si_empeora_en_test(self, entorno):
        entorno.factory.get_model.return_value = ModeloConstante(0.9)
        entorno.entrenar.return_value = (ModeloConstante(0.1), None)
        servicio = _servicio(_pares_distintos(MIN_FEEDBACK))

        with pytest.raises(ValueError, match="empeoró"):
            servicio.retrain()

        entorno.guardar.assert_not_called()
        assert not entorno.metrics_path.exists()

    def test_acepta_el_modelo_si_no_empeora(self, entorno):
        entorno.factory.get_model.return_value = ModeloConstante(0.9)
        entorno.entrenar.return_value = (ModeloConstante(0.9), None)

        _servicio(_pares_distintos(MIN_FEEDBACK)).retrain()

        entorno.guardar.assert_called_once()


class TestGuardarModelo:

    @pytest.fixture
    def en_disco(self, tmp_path, monkeypatch):
        ruta = tmp_path / "models" / "compatibility_net.pt"
        factory = MagicMock(name="ModelFactory")
        ajustes = SimpleNamespace(model_path=str(ruta))
        monkeypatch.setattr(retrain_service, "settings", ajustes)
        monkeypatch.setattr(retrain_service, "ModelFactory", factory)
        return SimpleNamespace(ruta=ruta, factory=factory)

    def test_guarda_el_state_dict_y_recarga_el_modelo(self, en_disco):
        modelo = torch.nn.Linear(2, 1)

        RetrainService(MagicMock())._guardar_modelo(modelo)

        assert en_disco.ruta.exists()
        guardado = torch.load(en_disco.ruta, weights_only=True)
        assert guardado.keys() == modelo.state_dict().keys()
        en_disco.factory.reload.assert_called_once_with(str(en_disco.ruta))

    def test_hace_backup_del_modelo_anterior(self, en_disco):
        en_disco.ruta.parent.mkdir(parents=True)
        en_disco.ruta.write_bytes(b"modelo-viejo")

        RetrainService(MagicMock())._guardar_modelo(torch.nn.Linear(2, 1))

        backup = en_disco.ruta.with_suffix(".bak.pt")
        assert backup.read_bytes() == b"modelo-viejo"

    def test_sin_modelo_previo_no_crea_backup(self, en_disco):
        RetrainService(MagicMock())._guardar_modelo(torch.nn.Linear(2, 1))

        assert not en_disco.ruta.with_suffix(".bak.pt").exists()
