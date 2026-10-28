import numpy as np

from app.models.feedback import FEATURE_COLUMNS


class DatasetReentrenamiento:
    """Arma el dataset del fine-tuning a partir del feedback."""

    def __init__(
        self,
        feedback_repeat: int = 100,  # veces que se repite el feedback en el train
        sinteticos_muestra: int = 300,  # filas sintéticas para no "olvidar"
        seed: int = 42,
    ):
        self.feedback_repeat = feedback_repeat
        self.sinteticos_muestra = sinteticos_muestra
        self.seed = seed

    @staticmethod
    def ultimo_por_par(feedback: list) -> list:
        """Se queda con la respuesta más reciente de cada par (usuario, amigo).

        Así una corrección pisa a la anterior y no quedan filas contradictorias
        (mismas features con etiquetas distintas), que el modelo no puede aprender.
        """
        ultimos = {}
        for f in sorted(feedback, key=lambda f: f.id):
            ultimos[(f.user_id, f.friend_id)] = f
        return list(ultimos.values())

    @staticmethod
    def feedback_a_arrays(feedback: list) -> tuple[np.ndarray, np.ndarray]:
        X = np.array(
            [[getattr(f, col) for col in FEATURE_COLUMNS] for f in feedback],
            dtype=np.float32,
        )
        y = np.array([1.0 if f.real_label else 0.0 for f in feedback], dtype=np.float32)
        return X, y

    def armar_entrenamiento(
        self,
        X_fb: np.ndarray,
        y_fb: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Feedback repetido + una muestra de sintéticos."""
        rng = np.random.default_rng(self.seed)
        n = min(self.sinteticos_muestra, len(X_train))
        idx = rng.choice(len(X_train), size=n, replace=False)
        X = np.concatenate([X_train[idx], np.tile(X_fb, (self.feedback_repeat, 1))])
        y = np.concatenate([y_train[idx], np.tile(y_fb, self.feedback_repeat)])
        return X, y
