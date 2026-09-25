# tests/test_feedback_loop.py
"""
Tests del feedback loop automatizado (scripts/feedback_loop.py).

Mockea evaluar/entrenar/cargar_* para no depender de PyTorch real
ni de archivos en disco: solo se testea la LOGICA de decision.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from scripts import feedback_loop


def _resultado(f1: float) -> dict:
    return {"precision": 0.7, "recall": 0.7, "f1": f1}


@pytest.fixture(autouse=True)
def mocks():
    """Mockea todas las dependencias externas de main()."""
    with patch("scripts.feedback_loop.cargar_split") as m_split, patch(
        "scripts.feedback_loop.cargar_datos_test"
    ) as m_datos_test, patch(
        "scripts.evaluate_model.cargar_modelo"
    ) as m_cargar_modelo, patch(
        "scripts.feedback_loop.evaluar"
    ) as m_evaluar, patch(
        "scripts.feedback_loop.entrenar"
    ) as m_entrenar, patch(
        "scripts.feedback_loop.torch.save"
    ) as m_save:

        m_split.return_value = {
            "X_train": "Xtr",
            "y_train": "ytr",
            "X_val": "Xval",
            "y_val": "yval",
        }
        # X_test real (numpy) porque feedback_loop.py hace X_test.shape[1]
        X_test = np.zeros((10, 5))
        y_test = np.zeros(10)
        m_datos_test.return_value = (X_test, y_test)
        m_cargar_modelo.return_value = MagicMock(name="modelo_actual")

        yield {
            "split": m_split,
            "datos_test": m_datos_test,
            "cargar_modelo": m_cargar_modelo,
            "evaluar": m_evaluar,
            "entrenar": m_entrenar,
            "save": m_save,
        }


class TestFeedbackLoopNoDispara:

    def test_f1_por_encima_del_umbral_no_reentrena(self, mocks):
        mocks["evaluar"].return_value = _resultado(0.75)

        feedback_loop.main()

        mocks["entrenar"].assert_not_called()
        mocks["save"].assert_not_called()

    def test_f1_justo_en_el_umbral_no_reentrena(self, mocks):
        mocks["evaluar"].return_value = _resultado(feedback_loop.F1_MINIMO)

        feedback_loop.main()

        mocks["entrenar"].assert_not_called()


class TestFeedbackLoopDispara:

    def test_f1_bajo_dispara_un_unico_reentrenamiento(self, mocks):
        mocks["evaluar"].return_value = _resultado(0.50)
        mocks["entrenar"].return_value = (MagicMock(name="modelo_nuevo"), 0.4)

        feedback_loop.main()

        mocks["entrenar"].assert_called_once()
        _, kwargs = mocks["entrenar"].call_args
        assert kwargs["max_epochs"] == feedback_loop.AJUSTE_MAX_EPOCHS
        assert kwargs["weight_decay"] == feedback_loop.AJUSTE_WEIGHT_DECAY

    def test_reentrenamiento_mejora_f1_guarda_modelo_nuevo(self, mocks):
        modelo_nuevo = MagicMock(name="modelo_nuevo")
        mocks["evaluar"].side_effect = [_resultado(0.50), _resultado(0.80)]
        mocks["entrenar"].return_value = (modelo_nuevo, 0.4)

        feedback_loop.main()

        mocks["save"].assert_called_once()
        modelo_guardado = mocks["save"].call_args[0][0]
        assert modelo_guardado is modelo_nuevo.state_dict.return_value

    def test_reentrenamiento_no_mejora_no_guarda(self, mocks):
        mocks["evaluar"].side_effect = [_resultado(0.60), _resultado(0.55)]
        mocks["entrenar"].return_value = (MagicMock(name="modelo_nuevo"), 0.4)

        feedback_loop.main()

        mocks["save"].assert_not_called()

    def test_reentrenamiento_empata_se_considera_mejora_y_guarda(self, mocks):
        mocks["evaluar"].side_effect = [_resultado(0.60), _resultado(0.60)]
        mocks["entrenar"].return_value = (MagicMock(name="modelo_nuevo"), 0.4)

        feedback_loop.main()

        mocks["save"].assert_called_once()

    def test_no_reintenta_mas_de_una_vez(self, mocks):
        """Aunque el reentrenamiento tampoco alcance el umbral, entrenar()
        se llama una sola vez (sin loop infinito)."""
        mocks["evaluar"].side_effect = [_resultado(0.40), _resultado(0.45)]
        mocks["entrenar"].return_value = (MagicMock(name="modelo_nuevo"), 0.4)

        feedback_loop.main()

        mocks["entrenar"].assert_called_once()

    def test_mensaje_agotado_si_sigue_bajo_el_umbral(self, mocks, capsys):
        mocks["evaluar"].side_effect = [_resultado(0.40), _resultado(0.45)]
        mocks["entrenar"].return_value = (MagicMock(name="modelo_nuevo"), 0.4)

        feedback_loop.main()

        salida = capsys.readouterr().out
        assert "agotado" in salida.lower()

    def test_mensaje_exitoso_si_supera_el_umbral(self, mocks, capsys):
        mocks["evaluar"].side_effect = [_resultado(0.50), _resultado(0.85)]
        mocks["entrenar"].return_value = (MagicMock(name="modelo_nuevo"), 0.4)

        feedback_loop.main()

        salida = capsys.readouterr().out
        assert "exitoso" in salida.lower()
