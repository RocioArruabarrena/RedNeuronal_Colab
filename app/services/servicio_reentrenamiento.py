from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import settings
from app.ml.almacenamiento import AlmacenMetricas, AlmacenModelo
from app.ml.criterios_aceptacion import CriterioAceptacion, NoEmpeorarF1
from app.ml.dataset_feedback import DatasetReentrenamiento
from app.ml.entrenador import ConfigEntrenamiento, Entrenador
from app.ml.evaluador import Evaluador
from app.ml.fabrica_modelo import ModelFactory
from app.repositories.contratos import RepositorioFeedback

RAIZ = Path(__file__).resolve().parents[2]
SPLIT_PATH = RAIZ / "data" / "splits.npz"
METRICS_PATH = RAIZ / "models" / "ultimas_metricas.json"

MIN_FEEDBACK = 5  # mínimo de pares distintos con feedback para reentrenar
FT_EPOCHS = 60  # épocas fijas del fine-tuning
FT_LEARNING_RATE = 1e-3
MARGEN_TEST_F1 = 0.02  # caída máxima de F1 en test que se tolera


class RetrainService:
    """Ajusta (fine-tuning) la red actual con el feedback acumulado.

    Coordina el flujo; cada tarea la hace una colaboradora que se recibe
    por constructor (inversión de dependencias), así se puede reemplazar.
    """

    def __init__(
        self,
        repository: RepositorioFeedback,
        evaluador: Evaluador | None = None,
        entrenador: Entrenador | None = None,
        armador: DatasetReentrenamiento | None = None,
        criterio: CriterioAceptacion | None = None,
        almacen_modelo: AlmacenModelo | None = None,
        almacen_metricas: AlmacenMetricas | None = None,
        fabrica=ModelFactory,
        split_path: Path = SPLIT_PATH,
    ):
        self.repository = repository
        self.evaluador = evaluador or Evaluador()
        # Fine-tuning: épocas fijas (patience mayor que las épocas = sin early stop)
        self.entrenador = entrenador or Entrenador(
            ConfigEntrenamiento(
                max_epochs=FT_EPOCHS,
                patience=FT_EPOCHS + 1,
                learning_rate=FT_LEARNING_RATE,
                verbose=False,
            )
        )
        self.armador = armador or DatasetReentrenamiento()
        self.criterio = criterio or NoEmpeorarF1(MARGEN_TEST_F1)
        self.almacen_modelo = almacen_modelo or AlmacenModelo(settings.model_path)
        self.almacen_metricas = almacen_metricas or AlmacenMetricas(METRICS_PATH)
        self.fabrica = fabrica
        self.split_path = split_path

    def retrain(self) -> dict:
        feedback = self._feedback_valido()
        split = self._cargar_split()
        X_fb, y_fb = self.armador.feedback_a_arrays(feedback)

        modelo_actual = self.fabrica.get_model(settings.model_path)
        test_before = self.evaluador.evaluar(
            modelo_actual, split["X_test"], split["y_test"]
        )
        feedback_before = self.evaluador.evaluar(modelo_actual, X_fb, y_fb)

        X_train, y_train = self.armador.armar_entrenamiento(
            X_fb, y_fb, split["X_train"], split["y_train"]
        )
        # restaurar_mejor=False: el "mejor" estado se mediría con sintéticos
        # y descartaría lo aprendido del feedback
        modelo_nuevo, _ = self.entrenador.entrenar(
            X_train,
            y_train,
            split["X_val"],
            split["y_val"],
            modelo_inicial=modelo_actual,
            restaurar_mejor=False,
        )
        test_after = self.evaluador.evaluar(
            modelo_nuevo, split["X_test"], split["y_test"]
        )
        feedback_after = self.evaluador.evaluar(modelo_nuevo, X_fb, y_fb)

        if not self.criterio.acepta(test_before, test_after):
            raise ValueError(self.criterio.motivo_rechazo(test_before, test_after))

        self.almacen_modelo.guardar(modelo_nuevo)
        self.fabrica.reload(settings.model_path)

        resultado = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feedback_count": len(feedback),
            "train_size": int(len(X_train)),
            "test_before": test_before,
            "test_after": test_after,
            "feedback_before": feedback_before,
            "feedback_after": feedback_after,
        }
        self.almacen_metricas.guardar(resultado)
        return resultado

    def _feedback_valido(self) -> list:
        feedback = self.armador.ultimo_por_par(self.repository.get_all_feedback())
        if len(feedback) < MIN_FEEDBACK:
            raise ValueError(
                f"Hacen falta al menos {MIN_FEEDBACK} pares distintos con "
                f"respuesta para actualizar el modelo (hay {len(feedback)})"
            )
        return feedback

    def _cargar_split(self):
        if not self.split_path.exists():
            raise FileNotFoundError(f"No se encontró el split en {self.split_path}")
        return np.load(self.split_path)
