# tests/test_compatibility_controller.py
"""
Tests de los endpoints de la API (app/controllers/compatibility_controller.py).

Usa TestClient de FastAPI con overrides de los proveedores de servicios (via
app.dependency_overrides) para no depender de una base de datos real:
repository y services ya están cubiertos por sus propios tests.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.controllers.compatibility_controller import (
    get_metrics_service,
    get_retrain_service,
    get_service,
)
from app.main import app


def _resultado_retrain():
    metricas = {
        "accuracy": 0.8,
        "precision": 0.79,
        "recall": 0.84,
        "f1": 0.81,
        "auc": 0.85,
    }
    return {
        "trained_at": "2026-10-31T12:00:00+00:00",
        "feedback_count": 5,
        "train_size": 5500,
        "test_before": metricas,
        "test_after": metricas,
        "feedback_before": metricas,
        "feedback_after": metricas,
    }


@pytest.fixture
def service_mock():
    return MagicMock(name="compatibility_service")


@pytest.fixture
def retrain_service_mock():
    return MagicMock(name="retrain_service")


@pytest.fixture
def metrics_service_mock():
    return MagicMock(name="metrics_service")


@pytest.fixture
def client(service_mock, retrain_service_mock, metrics_service_mock):
    app.dependency_overrides[get_service] = lambda: service_mock
    app.dependency_overrides[get_retrain_service] = lambda: retrain_service_mock
    app.dependency_overrides[get_metrics_service] = lambda: metrics_service_mock
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestHealthCheck:

    def test_health_devuelve_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredictCompatibility:

    def test_predict_devuelve_200_con_body_correcto(self, client, service_mock):
        service_mock.predict.return_value = {
            "user_id": 1,
            "friend_id": 2,
            "compatibility_score": 0.87,
            "is_compatible": True,
        }

        response = client.post(
            "/compatibility/predict", json={"user_id": 1, "friend_id": 2}
        )

        assert response.status_code == 200
        assert response.json() == {
            "user_id": 1,
            "friend_id": 2,
            "compatibility_score": 0.87,
            "is_compatible": True,
        }

    def test_predict_llama_al_service_con_los_ids_correctos(self, client, service_mock):
        service_mock.predict.return_value = {
            "user_id": 5,
            "friend_id": 9,
            "compatibility_score": 0.1,
            "is_compatible": False,
        }

        client.post("/compatibility/predict", json={"user_id": 5, "friend_id": 9})

        service_mock.predict.assert_called_once_with(5, 9)

    def test_perfil_no_encontrado_devuelve_404(self, client, service_mock):
        service_mock.predict.side_effect = ValueError(
            "Perfil de usuario o amigo no encontrado"
        )

        response = client.post(
            "/compatibility/predict", json={"user_id": 999, "friend_id": 2}
        )

        assert response.status_code == 404
        assert "no encontrado" in response.json()["detail"].lower()

    def test_body_sin_user_id_devuelve_422(self, client, service_mock):
        response = client.post("/compatibility/predict", json={"friend_id": 2})
        assert response.status_code == 422
        service_mock.predict.assert_not_called()

    def test_body_sin_friend_id_devuelve_422(self, client, service_mock):
        response = client.post("/compatibility/predict", json={"user_id": 1})
        assert response.status_code == 422
        service_mock.predict.assert_not_called()

    def test_user_id_con_tipo_invalido_devuelve_422(self, client, service_mock):
        response = client.post(
            "/compatibility/predict", json={"user_id": "no-es-numero", "friend_id": 2}
        )
        assert response.status_code == 422
        service_mock.predict.assert_not_called()

    def test_body_vacio_devuelve_422(self, client, service_mock):
        response = client.post("/compatibility/predict", json={})
        assert response.status_code == 422


class TestFeedback:

    def test_feedback_devuelve_201_y_guarda_la_respuesta(
        self, client, service_mock
    ):
        service_mock.save_feedback.return_value = {
            "id": 7,
            "user_id": 1,
            "friend_id": 2,
            "predicted_score": 0.87,
            "real_label": True,
            "features": {
                "similitud_intereses": 0.5,
                "diff_activity": 2.0,
                "similitud_tags_posts": 0.25,
                "tiene_tags_posts": 1.0,
                "mismo_avatar_subcultura": 0.0,
            },
        }

        response = client.post(
            "/compatibility/feedback",
            json={"user_id": 1, "friend_id": 2, "real_label": True},
        )

        assert response.status_code == 201
        assert response.json()["real_label"] is True
        assert response.json()["id"] == 7
        service_mock.save_feedback.assert_called_once_with(1, 2, True)

    def test_feedback_sin_real_label_devuelve_422(self, client, service_mock):
        response = client.post(
            "/compatibility/feedback", json={"user_id": 1, "friend_id": 2}
        )

        assert response.status_code == 422
        service_mock.save_feedback.assert_not_called()


class TestRetrain:

    def test_retrain_devuelve_metricas_y_cantidad_de_feedback(
        self, client, retrain_service_mock
    ):
        retrain_service_mock.retrain.return_value = _resultado_retrain()

        response = client.post("/compatibility/retrain")

        assert response.status_code == 200
        assert response.json()["feedback_count"] == 5
        assert response.json()["test_after"]["f1"] == 0.81
        retrain_service_mock.retrain.assert_called_once_with()

    def test_feedback_insuficiente_devuelve_409(self, client, retrain_service_mock):
        retrain_service_mock.retrain.side_effect = ValueError(
            "Hacen falta al menos 5 pares distintos con respuesta (hay 4)"
        )

        response = client.post("/compatibility/retrain")

        assert response.status_code == 409
        assert "al menos 5" in response.json()["detail"]
        retrain_service_mock.retrain.assert_called_once_with()


class TestMetrics:

    def test_metrics_devuelve_las_metricas_guardadas(
        self, client, metrics_service_mock
    ):
        resultado = _resultado_retrain()
        metrics_service_mock.get_last_metrics.return_value = resultado

        response = client.get("/compatibility/metrics")

        assert response.status_code == 200
        assert response.json() == resultado
        metrics_service_mock.get_last_metrics.assert_called_once_with()


class TestSwagger:

    def test_docs_disponible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_disponible(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "/compatibility/predict" in schema["paths"]
