"""Compara arquitecturas de CompatibilityNet en los datos reales de Colab."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Cambiar estas rutas si los archivos estan en otra ubicacion de Colab.
REALES_CSV = Path("/content/reales_features.csv")
RESULTADOS_CSV = Path("/content/resultados_arquitectura.csv")

FEATURE_NAMES = [
    "similitud_intereses",
    "diff_activity",
    "similitud_tags_posts",
    "tiene_tags_posts",
    "mismo_avatar_subcultura",
]
LABEL_NAME = "y"
SEED = 42
N_SPLITS = 5
N_REPEATS = 3
EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
UMBRAL = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARQUITECTURAS = [
    ((8, 4), 0.0),
    ((8, 4), 0.2),
    ((16, 8), 0.0),
    ((16, 8), 0.2),
    ((16, 8, 4), 0.0),
    ((16, 8, 4), 0.2),
    ((32, 16), 0.0),
    ((32, 16), 0.2),
    ((8,), 0.0),
    ((8,), 0.2),
]


class CompatibilityNet(nn.Module):
    """MLP con tamanos de capas y dropout configurables."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_layers: tuple[int, ...] = (8, 4),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        previous_size = input_dim
        for hidden_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_size = hidden_size
        layers.extend([nn.Linear(previous_size, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def cargar_datos() -> tuple[np.ndarray, np.ndarray]:
    if not REALES_CSV.is_file():
        raise FileNotFoundError(f"No se encontro el CSV: {REALES_CSV}")
    frame = pd.read_csv(REALES_CSV)
    required_columns = [*FEATURE_NAMES, LABEL_NAME]
    missing_columns = [name for name in required_columns if name not in frame]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing_columns}")

    features = frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    labels = pd.to_numeric(frame[LABEL_NAME], errors="raise").to_numpy(
        dtype=np.int64
    )
    if not np.isfinite(features).all():
        raise ValueError("El CSV contiene features vacias o no numericas.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("La columna y debe contener etiquetas binarias 0/1.")
    if len(labels) < N_SPLITS or len(np.unique(labels)) != 2:
        raise ValueError("Se necesitan ambas clases y al menos 5 filas.")
    return features, labels


def entrenar_red(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    hidden_layers: tuple[int, ...],
    dropout: float,
) -> np.ndarray:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = CompatibilityNet(
        input_dim=len(FEATURE_NAMES),
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(DEVICE)
    dataset = TensorDataset(
        torch.as_tensor(train_features, dtype=torch.float32),
        torch.as_tensor(train_labels, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(
        dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    for _ in range(EPOCHS):
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.as_tensor(
            test_features, dtype=torch.float32, device=DEVICE
        )
        scores = model(test_tensor).squeeze(1).cpu().numpy()
    return scores


def _resumen_metricas(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    predictions = (scores >= UMBRAL).astype(np.int64)
    return {
        "auc": roc_auc_score(labels, scores),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def evaluar_configuraciones(
    features: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )
    scores_por_configuracion = {
        (hidden_layers, dropout): [] for hidden_layers, dropout in ARQUITECTURAS
    }
    logistic_scores = []

    for train_indices, test_indices in splitter.split(features, labels):
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(features[train_indices]).astype(
            np.float32
        )
        test_scaled = scaler.transform(features[test_indices]).astype(np.float32)
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        logistic = LogisticRegression(max_iter=1000, random_state=SEED)
        logistic.fit(train_scaled, train_labels)
        logistic_probabilities = logistic.predict_proba(test_scaled)[:, 1]
        logistic_scores.append(
            _resumen_metricas(test_labels, logistic_probabilities)
        )

        for hidden_layers, dropout in ARQUITECTURAS:
            probabilities = entrenar_red(
                train_scaled,
                train_labels,
                test_scaled,
                hidden_layers,
                dropout,
            )
            scores_por_configuracion[(hidden_layers, dropout)].append(
                _resumen_metricas(test_labels, probabilities)
            )

    rows = []
    for (hidden_layers, dropout), fold_metrics in scores_por_configuracion.items():
        rows.append(_fila_resultados("-".join(map(str, hidden_layers)), dropout, fold_metrics))
    rows.append(_fila_resultados("LogisticRegression", np.nan, logistic_scores))

    results = pd.DataFrame(rows)
    return results.sort_values("AUC media", ascending=False).reset_index(drop=True)


def _fila_resultados(
    architecture: str, dropout: float, fold_metrics: list[dict[str, float]]
) -> dict[str, float | str]:
    auc_values = [metrics["auc"] for metrics in fold_metrics]
    accuracy_values = [metrics["accuracy"] for metrics in fold_metrics]
    f1_values = [metrics["f1"] for metrics in fold_metrics]
    return {
        "arquitectura": architecture,
        "dropout": dropout,
        "AUC media": float(np.mean(auc_values)),
        "AUC desvio": float(np.std(auc_values, ddof=1)),
        "accuracy media": float(np.mean(accuracy_values)),
        "F1 media": float(np.mean(f1_values)),
    }


def main() -> None:
    features, labels = cargar_datos()
    results = evaluar_configuraciones(features, labels)
    RESULTADOS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTADOS_CSV, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResultados guardados en: {RESULTADOS_CSV}")

    best = results.iloc[0]
    second = results.iloc[1]
    difference = float(best["AUC media"] - second["AUC media"])
    within_one_std = difference <= float(best["AUC desvio"])
    print(
        "\nMejor configuracion: "
        f"{best['arquitectura']} (dropout={best['dropout']})."
    )
    print(
        f"Diferencia con la segunda ({second['arquitectura']}): {difference:.4f}; "
        f"desvio estandar de la mejor: {best['AUC desvio']:.4f}. "
        f"{'No es concluyente: esta dentro de un desvio estandar.' if within_one_std else 'Supera un desvio estandar.'}"
    )


if __name__ == "__main__":
    main()
