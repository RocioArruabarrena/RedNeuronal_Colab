# tests/test_metrics_service.py
"""
Tests de app/services/metrics_service.py.

MetricsService solo lee models/ultimas_metricas.json: se prueba con
archivos temporales (tmp_path) para no tocar el archivo real.
"""

import json

import pytest

from app.services.metrics_service import MetricsService


class TestGetLastMetrics:

    def test_devuelve_el_contenido_del_json(self, tmp_path):
        ruta = tmp_path / "ultimas_metricas.json"
        datos = {"feedback_count": 7, "test_before": {"f1": 0.8}}
        ruta.write_text(json.dumps(datos), encoding="utf-8")

        assert MetricsService(path=ruta).get_last_metrics() == datos

    def test_lee_el_archivo_en_cada_llamada(self, tmp_path):
        ruta = tmp_path / "ultimas_metricas.json"
        ruta.write_text(json.dumps({"feedback_count": 5}), encoding="utf-8")
        service = MetricsService(path=ruta)
        service.get_last_metrics()

        ruta.write_text(json.dumps({"feedback_count": 9}), encoding="utf-8")

        assert service.get_last_metrics()["feedback_count"] == 9

    def test_sin_archivo_levanta_file_not_found(self, tmp_path):
        service = MetricsService(path=tmp_path / "no_existe.json")

        with pytest.raises(FileNotFoundError, match="Todavía no hay métricas"):
            service.get_last_metrics()

    def test_ruta_por_defecto_apunta_a_models(self):
        service = MetricsService()

        assert service.path.name == "ultimas_metricas.json"
        assert service.path.parent.name == "models"
