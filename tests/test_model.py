# tests/test_model.py
"""
Tests de arquitectura para CompatibilityNet (app/ml/model.py).
"""

import pytest
import torch

from app.ml.model import CompatibilityNet


@pytest.fixture
def model():
    return CompatibilityNet()


class TestCompatibilityNetArquitectura:

    def test_input_dim_default_es_5(self, model):
        primera_capa = model.net[0]
        assert isinstance(primera_capa, torch.nn.Linear)
        assert primera_capa.in_features == 5

    def test_input_dim_configurable(self):
        modelo_custom = CompatibilityNet(input_dim=10)
        assert modelo_custom.net[0].in_features == 10

    def test_estructura_de_capas(self, model):
        capas = list(model.net)
        assert len(capas) == 8
        assert isinstance(capas[0], torch.nn.Linear) and capas[0].out_features == 8
        assert isinstance(capas[1], torch.nn.ReLU)
        assert isinstance(capas[2], torch.nn.Dropout) and capas[2].p == 0.3
        assert (
            isinstance(capas[3], torch.nn.Linear)
            and capas[3].in_features == 8
            and capas[3].out_features == 4
        )
        assert isinstance(capas[4], torch.nn.ReLU)
        assert isinstance(capas[5], torch.nn.Dropout) and capas[5].p == 0.2
        assert (
            isinstance(capas[6], torch.nn.Linear)
            and capas[6].in_features == 4
            and capas[6].out_features == 1
        )
        assert isinstance(capas[7], torch.nn.Sigmoid)

    def test_output_shape_batch(self, model):
        x = torch.randn(16, 5)  # batch_size=16, 5 features
        out = model(x)
        assert out.shape == (16, 1)

    def test_output_shape_single_sample(self, model):
        x = torch.randn(1, 5)
        out = model(x)
        assert out.shape == (1, 1)

    def test_output_en_rango_sigmoid(self, model):
        x = torch.randn(50, 5)
        out = model(x)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_forward_no_explota_con_valores_extremos(self, model):
        x = torch.tensor([[100.0, -100.0, 0.0, 1.0, -1.0]])
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_eval_mode_desactiva_dropout_determinista(self, model):
        model.eval()
        x = torch.randn(5, 5)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        # en eval, sin dropout aleatorio, misma entrada -> misma salida
        assert torch.allclose(out1, out2)

    def test_input_dim_incorrecto_falla(self, model):
        x = torch.randn(1, 3)  # dim equivocada (3 en vez de 5)
        with pytest.raises(RuntimeError):
            model(x)
