# tests/test_compatibility_controller.py
"""
Tests de los endpoints de la API (app/controllers/compatibility_controller.py).

Usa TestClient de FastAPI con override de get_service (via
app.dependency_overrides) para no depender de una base de datos real:
el repository y el service ya están cubiertos por sus propios tests.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.controllers.compatibility_controller import get_service
from app.main import app


@pytest.fixture
def service_mock():
    return MagicMock(name="compatibility_service")


@pytest.fixture
def client(service_mock):
    app.dependency_overrides[get_service] = lambda: service_mock
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


class TestSwagger:

    def test_docs_disponible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_disponible(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "/compatibility/predict" in schema["paths"]
