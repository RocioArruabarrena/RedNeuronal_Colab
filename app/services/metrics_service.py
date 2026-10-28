import json
from pathlib import Path

METRICS_PATH = Path(__file__).resolve().parents[2] / "models" / "ultimas_metricas.json"


class MetricsService:
    def __init__(self, path: Path = METRICS_PATH):
        self.path = path

    def get_last_metrics(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(
                "Todavía no hay métricas: reentrená el modelo primero"
            )
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)
