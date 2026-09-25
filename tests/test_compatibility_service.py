# tests/test_compatibility_service.py
"""
Tests de CompatibilityService (app/services/compatibility_service.py).

Cubre: helpers (_jaccard, _extraer_subcultura), _calcular_features
(orden y valores de las 5 features) y predict() (umbral 0.5, guardado
de la predicción, perfil no encontrado).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.services.compatibility_service import (
    CompatibilityService,
    _calcular_features,
    _extraer_subcultura,
    _jaccard,
)


def _perfil(interests=None, activity_score=0, tags_agregados=None):
    """Perfil mínimo con los atributos que usa _calcular_features."""
    return SimpleNamespace(
        interests=interests,
        activity_score=activity_score,
        tags_agregados=tags_agregados,
    )


# ---------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------


class TestJaccard:

    def test_sets_identicos_da_1(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_sets_disjuntos_da_0(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_sets_parcialmente_superpuestos(self):
        # interseccion 1 (b), union 3 (a,b,c) -> 1/3
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_ambos_vacios_da_0_no_explota(self):
        assert _jaccard(set(), set()) == 0.0


# ---------------------------------------------------------------------
# _extraer_subcultura
# ---------------------------------------------------------------------


class TestExtraerSubcultura:

    def test_encuentra_tag_subcultura(self):
        assert (
            _extraer_subcultura(["música", "subcultura:flogger", "arte"]) == "flogger"
        )

    def test_sin_tag_subcultura_da_none(self):
        assert _extraer_subcultura(["música", "arte"]) is None

    def test_lista_vacia_da_none(self):
        assert _extraer_subcultura([]) is None

    def test_none_da_none(self):
        assert _extraer_subcultura(None) is None


# ---------------------------------------------------------------------
# _calcular_features: orden y valores
# ---------------------------------------------------------------------


class TestCalcularFeatures:

    def test_orden_y_cantidad_de_features(self):
        a = _perfil(interests=["a"], tags_agregados=["x"])
        b = _perfil(interests=["a"], tags_agregados=["x"])
        features = _calcular_features(a, b)
        assert len(features) == 5
        # similitud_intereses, diff_activity, similitud_tags_posts,
        # tiene_tags_posts, mismo_avatar_subcultura
        assert features == [1.0, 0.0, 1.0, 1.0, 0.0]

    def test_similitud_intereses_jaccard(self):
        a = _perfil(interests=["musica", "arte"])
        b = _perfil(interests=["arte", "cine"])
        features = _calcular_features(a, b)
        assert features[0] == pytest.approx(1 / 3)

    def test_diff_activity(self):
        a = _perfil(activity_score=80)
        b = _perfil(activity_score=30)
        features = _calcular_features(a, b)
        assert features[1] == 50

    def test_activity_score_none_se_trata_como_0(self):
        a = _perfil(activity_score=None)
        b = _perfil(activity_score=20)
        features = _calcular_features(a, b)
        assert features[1] == 20

    def test_similitud_tags_posts_jaccard(self):
        a = _perfil(tags_agregados=["#emo", "#vsco"])
        b = _perfil(tags_agregados=["#vsco"])
        features = _calcular_features(a, b)
        assert features[2] == pytest.approx(1 / 2)

    def test_tiene_tags_posts_ambos_con_tags(self):
        a = _perfil(tags_agregados=["#emo"])
        b = _perfil(tags_agregados=["#vsco"])
        features = _calcular_features(a, b)
        assert features[3] == 1.0

    def test_tiene_tags_posts_uno_sin_tags(self):
        a = _perfil(tags_agregados=["#emo"])
        b = _perfil(tags_agregados=[])
        features = _calcular_features(a, b)
        assert features[3] == 0.0

    def test_mismo_avatar_subcultura_coincide(self):
        a = _perfil(interests=["subcultura:flogger"])
        b = _perfil(interests=["subcultura:flogger"])
        features = _calcular_features(a, b)
        assert features[4] == 1.0

    def test_mismo_avatar_subcultura_distinta(self):
        a = _perfil(interests=["subcultura:flogger"])
        b = _perfil(interests=["subcultura:emo"])
        features = _calcular_features(a, b)
        assert features[4] == 0.0

    def test_interests_y_tags_none_no_explota(self):
        a = _perfil(interests=None, activity_score=None, tags_agregados=None)
        b = _perfil(interests=None, activity_score=None, tags_agregados=None)
        features = _calcular_features(a, b)
        assert features == [0.0, 0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------
# CompatibilityService.predict()
# ---------------------------------------------------------------------


@pytest.fixture
def repository():
    return MagicMock(name="repository")


def _service_con_score(repository, score: float):
    """Arma un CompatibilityService con ModelFactory mockeada para que
    el modelo devuelva siempre `score`."""
    modelo_mock = MagicMock(name="modelo")
    modelo_mock.return_value = torch.tensor([[score]])

    with patch("app.services.compatibility_service.ModelFactory") as m_factory:
        m_factory.get_model.return_value = modelo_mock
        service = CompatibilityService(repository)
    return service, modelo_mock


class TestPredict:

    def test_score_arriba_del_umbral_es_compatible(self, repository):
        repository.get_profile.side_effect = [_perfil(), _perfil()]
        service, _ = _service_con_score(repository, 0.75)

        resultado = service.predict(1, 2)

        assert resultado["is_compatible"] is True
        assert resultado["compatibility_score"] == 0.75

    def test_score_abajo_del_umbral_no_es_compatible(self, repository):
        repository.get_profile.side_effect = [_perfil(), _perfil()]
        service, _ = _service_con_score(repository, 0.30)

        resultado = service.predict(1, 2)

        assert resultado["is_compatible"] is False

    def test_score_exactamente_en_el_umbral_es_compatible(self, repository):
        """El umbral es >= 0.5, no > 0.5."""
        repository.get_profile.side_effect = [_perfil(), _perfil()]
        service, _ = _service_con_score(repository, 0.50)

        resultado = service.predict(1, 2)

        assert resultado["is_compatible"] is True

    def test_score_se_redondea_a_4_decimales(self, repository):
        repository.get_profile.side_effect = [_perfil(), _perfil()]
        service, _ = _service_con_score(repository, 0.123456)

        resultado = service.predict(1, 2)

        assert resultado["compatibility_score"] == 0.1235

    def test_guarda_la_prediccion_en_el_repositorio(self, repository):
        repository.get_profile.side_effect = [_perfil(), _perfil()]
        service, _ = _service_con_score(repository, 0.9)

        service.predict(1, 2)

        repository.save_prediction.assert_called_once()
        args = repository.save_prediction.call_args[0]
        assert args[0] == 1
        assert args[1] == 2
        assert args[2] == pytest.approx(0.9)

    def test_usuario_no_encontrado_lanza_value_error(self, repository):
        repository.get_profile.side_effect = [None, _perfil()]
        service, _ = _service_con_score(repository, 0.5)

        with pytest.raises(ValueError):
            service.predict(1, 2)

    def test_amigo_no_encontrado_lanza_value_error(self, repository):
        repository.get_profile.side_effect = [_perfil(), None]
        service, _ = _service_con_score(repository, 0.5)

        with pytest.raises(ValueError):
            service.predict(1, 2)

    def test_no_guarda_prediccion_si_falta_un_perfil(self, repository):
        repository.get_profile.side_effect = [None, None]
        service, _ = _service_con_score(repository, 0.5)

        with pytest.raises(ValueError):
            service.predict(1, 2)

        repository.save_prediction.assert_not_called()

    def test_features_pasadas_al_modelo_tienen_shape_correcto(self, repository):
        perfil_a = _perfil(interests=["a"], activity_score=10, tags_agregados=["x"])
        perfil_b = _perfil(interests=["a"], activity_score=20, tags_agregados=["x"])
        repository.get_profile.side_effect = [perfil_a, perfil_b]
        service, modelo_mock = _service_con_score(repository, 0.5)

        service.predict(1, 2)

        tensor_pasado = modelo_mock.call_args[0][0]
        assert tensor_pasado.shape == (1, 5)
        assert tensor_pasado.dtype == torch.float32
