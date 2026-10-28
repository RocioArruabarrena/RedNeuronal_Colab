# tests/test_componentes_reentrenamiento.py
"""
Tests de las piezas del reentrenamiento: Evaluador, DatasetReentrenamiento,
NoEmpeorarF1, AlmacenModelo, AlmacenMetricas y Entrenador.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.ml.almacenamiento import AlmacenMetricas, AlmacenModelo
from app.ml.criterios_aceptacion import NoEmpeorarF1
from app.ml.dataset_feedback import DatasetReentrenamiento
from app.ml.entrenador import ConfigEntrenamiento, Entrenador, entrenar
from app.ml.evaluador import Evaluador
from app.ml.model import CompatibilityNet
from app.models.feedback import FEATURE_COLUMNS

N = len(FEATURE_COLUMNS)


class ModeloEco(torch.nn.Module):
    """Devuelve como probabilidad el valor de la primera feature."""

    def forward(self, x):
        return x[:, :1]


def _feedback(id_, user_id, friend_id, real_label):
    features = {col: 0.5 for col in FEATURE_COLUMNS}
    return SimpleNamespace(
        id=id_,
        user_id=user_id,
        friend_id=friend_id,
        real_label=real_label,
        **features,
    )


class TestEvaluador:

    def test_clasificacion_perfecta(self):
        X = np.array([[0.9], [0.8], [0.2], [0.1]], dtype=np.float32)
        y = np.array([1, 1, 0, 0], dtype=np.float32)

        resultado = Evaluador().evaluar(ModeloEco(), X, y)

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

        resultado = Evaluador().evaluar(ModeloEco(), X, y)

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

        assert Evaluador().evaluar(ModeloEco(), X, y)["recall"] == 1.0

    def test_umbral_personalizado(self):
        X = np.array([[0.9], [0.9]], dtype=np.float32)
        y = np.array([1, 1], dtype=np.float32)

        resultado = Evaluador(umbral=0.95).evaluar(ModeloEco(), X, y)

        assert resultado["recall"] == 0.0

    def test_una_sola_clase_deja_auc_en_none(self):
        X = np.array([[0.9], [0.7]], dtype=np.float32)
        y = np.array([1, 1], dtype=np.float32)

        resultado = Evaluador().evaluar(ModeloEco(), X, y)

        assert resultado["auc"] is None
        assert resultado["accuracy"] == 1.0


class TestDatasetReentrenamiento:

    def test_ultimo_por_par_se_queda_con_el_mas_reciente(self):
        viejo = _feedback(1, 1, 2, False)
        nuevo = _feedback(7, 1, 2, True)

        resultado = DatasetReentrenamiento.ultimo_por_par([nuevo, viejo])

        assert len(resultado) == 1
        assert resultado[0] is nuevo

    def test_ultimo_por_par_conserva_pares_distintos(self):
        a = _feedback(1, 1, 2, True)
        b = _feedback(2, 2, 1, True)  # el par invertido es otro par
        c = _feedback(3, 1, 3, False)

        resultado = DatasetReentrenamiento.ultimo_por_par([a, b, c])

        assert {f.id for f in resultado} == {1, 2, 3}

    def test_ultimo_por_par_lista_vacia(self):
        assert DatasetReentrenamiento.ultimo_por_par([]) == []

    def test_feedback_a_arrays_forma_tipo_y_etiquetas(self):
        feedback = [_feedback(1, 1, 2, True), _feedback(2, 1, 3, False)]

        X, y = DatasetReentrenamiento.feedback_a_arrays(feedback)

        assert X.shape == (2, N)
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        assert y.tolist() == [1.0, 0.0]

    def test_feedback_a_arrays_respeta_el_orden_de_feature_columns(self):
        valores = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
        fila = SimpleNamespace(real_label=True, **valores)

        X, _ = DatasetReentrenamiento.feedback_a_arrays([fila])

        assert X[0].tolist() == [float(i) for i in range(N)]

    def test_arma_muestra_sintetica_mas_feedback_repetido(self):
        armador = DatasetReentrenamiento(feedback_repeat=4, sinteticos_muestra=10)
        X_fb = np.ones((3, N), dtype=np.float32)
        y_fb = np.array([1, 0, 1], dtype=np.float32)
        X_train = np.zeros((50, N), dtype=np.float32)
        y_train = np.zeros(50, dtype=np.float32)

        X, y = armador.armar_entrenamiento(X_fb, y_fb, X_train, y_train)

        assert X.shape == (10 + 3 * 4, N)
        assert X[:10].sum() == 0
        assert X[10:].min() == 1
        assert y[10:].tolist() == [1, 0, 1] * 4

    def test_la_muestra_no_supera_los_sinteticos_disponibles(self):
        armador = DatasetReentrenamiento(feedback_repeat=2, sinteticos_muestra=300)
        X_fb = np.ones((3, N), dtype=np.float32)
        y_fb = np.array([1, 0, 1], dtype=np.float32)
        X_train = np.zeros((5, N), dtype=np.float32)
        y_train = np.zeros(5, dtype=np.float32)

        X, _ = armador.armar_entrenamiento(X_fb, y_fb, X_train, y_train)

        assert len(X) == 5 + 3 * 2

    def test_misma_semilla_misma_muestra(self):
        rng = np.random.default_rng(1)
        X_train = rng.random((50, N)).astype(np.float32)
        y_train = np.zeros(50, dtype=np.float32)
        X_fb = np.ones((2, N), dtype=np.float32)
        y_fb = np.ones(2, dtype=np.float32)

        X1, _ = DatasetReentrenamiento().armar_entrenamiento(
            X_fb, y_fb, X_train, y_train
        )
        X2, _ = DatasetReentrenamiento().armar_entrenamiento(
            X_fb, y_fb, X_train, y_train
        )

        assert np.array_equal(X1, X2)


class TestNoEmpeorarF1:

    def test_acepta_si_el_f1_sube_o_cae_dentro_del_margen(self):
        criterio = NoEmpeorarF1(margen=0.02)

        assert criterio.acepta({"f1": 0.80}, {"f1": 0.85})
        assert criterio.acepta({"f1": 0.80}, {"f1": 0.79})

    def test_rechaza_si_el_f1_cae_mas_que_el_margen(self):
        criterio = NoEmpeorarF1(margen=0.02)

        assert not criterio.acepta({"f1": 0.80}, {"f1": 0.77})

    def test_el_motivo_menciona_los_dos_f1(self):
        motivo = NoEmpeorarF1().motivo_rechazo({"f1": 0.8}, {"f1": 0.5})

        assert "empeoró" in motivo
        assert "0.8" in motivo and "0.5" in motivo


class TestAlmacenModelo:

    def test_guarda_el_state_dict(self, tmp_path):
        ruta = tmp_path / "models" / "compatibility_net.pt"
        modelo = torch.nn.Linear(2, 1)

        AlmacenModelo(ruta).guardar(modelo)

        guardado = torch.load(ruta, weights_only=True)
        assert guardado.keys() == modelo.state_dict().keys()

    def test_hace_backup_del_modelo_anterior(self, tmp_path):
        ruta = tmp_path / "models" / "compatibility_net.pt"
        ruta.parent.mkdir(parents=True)
        ruta.write_bytes(b"modelo-viejo")

        AlmacenModelo(ruta).guardar(torch.nn.Linear(2, 1))

        assert ruta.with_suffix(".bak.pt").read_bytes() == b"modelo-viejo"

    def test_sin_modelo_previo_no_crea_backup(self, tmp_path):
        ruta = tmp_path / "models" / "compatibility_net.pt"

        AlmacenModelo(ruta).guardar(torch.nn.Linear(2, 1))

        assert not ruta.with_suffix(".bak.pt").exists()


class TestAlmacenMetricas:

    def test_guarda_y_lee_lo_mismo(self, tmp_path):
        almacen = AlmacenMetricas(tmp_path / "models" / "metricas.json")
        resultado = {"feedback_count": 6, "test_after": {"f1": 0.8}}

        almacen.guardar(resultado)

        assert almacen.leer() == resultado

    def test_leer_sin_archivo_levanta_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AlmacenMetricas(tmp_path / "no_existe.json").leer()


class TestEntrenador:

    @pytest.fixture
    def datos(self):
        rng = np.random.default_rng(0)
        X = rng.random((40, N)).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(np.float32)
        return X, y, X[:10], y[:10]

    @pytest.fixture
    def config(self):
        return ConfigEntrenamiento(max_epochs=3, verbose=False)

    def test_devuelve_un_modelo_y_el_mejor_val_loss(self, datos, config):
        modelo, val_loss = Entrenador(config).entrenar(*datos)

        assert isinstance(modelo, CompatibilityNet)
        assert isinstance(val_loss, float)

    def test_el_fine_tuning_no_modifica_el_modelo_inicial(self, datos, config):
        inicial = CompatibilityNet(input_dim=N)
        pesos_antes = {k: v.clone() for k, v in inicial.state_dict().items()}

        nuevo, _ = Entrenador(config).entrenar(
            *datos, modelo_inicial=inicial, restaurar_mejor=False
        )

        assert nuevo is not inicial
        for k, v in inicial.state_dict().items():
            assert torch.equal(v, pesos_antes[k])
        assert any(
            not torch.equal(v, pesos_antes[k]) for k, v in nuevo.state_dict().items()
        )

    def test_el_atajo_entrenar_mantiene_la_firma_de_siempre(self, datos):
        modelo, val_loss = entrenar(*datos, max_epochs=2, verbose=False)

        assert isinstance(modelo, CompatibilityNet)
        assert isinstance(val_loss, float)
