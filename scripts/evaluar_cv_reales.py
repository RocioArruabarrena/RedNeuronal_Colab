"""Compara modelos con validacion cruzada repetida sobre datos reales."""

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import CompatibilityNet
from scripts.evaluate_on_real import cargar_datos_reales, cargar_modelo
from scripts.train_model import FEATURE_NAMES
from scripts.train_neural_net import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    entrenar,
)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LEARNING_RATE = 1e-4
CLASSIFICATION_THRESHOLD = 0.5


def _train_finetuned_model(
    model: CompatibilityNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> CompatibilityNet:
    train_inputs = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINE_TUNE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(FINE_TUNE_EPOCHS):
        model.train()
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs), batch_labels)
            loss.backward()
            optimizer.step()
    return model


def _neural_probabilities(
    model: CompatibilityNet,
    X: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        return model(inputs).squeeze(1).numpy()


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "auc": roc_auc_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def _print_summary(name: str, results: list[dict[str, float]]) -> None:
    print(f"{name} ({len(results)} folds)")
    for metric in ("auc", "accuracy", "f1"):
        values = np.array([result[metric] for result in results])
        print(f"  {metric.upper():8s} {values.mean():.3f} +/- {values.std():.3f}")


def main() -> None:
    X, y = cargar_datos_reales()
    if len(X) != len(y) or X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Los datos reales no tienen la forma esperada de 5 features.")
    if np.unique(y).size != 2 or np.bincount(y).min() < N_SPLITS:
        raise ValueError("Se necesitan ambas clases y al menos 5 ejemplos por clase.")

    print(
        f"Ejemplos reales: {len(y)} ({int(y.sum())} positivos, "
        f"{int(len(y) - y.sum())} negativos)"
    )
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    pretrained_model = cargar_modelo()
    logistic_results = []
    baseline_auc = []
    scratch_results = []
    finetuned_results = []

    for fold_index, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        logistic = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        logistic.fit(X_train, y_train)
        logistic_results.append(
            _metrics(y_test, logistic.predict_proba(X_test)[:, 1])
        )
        baseline_auc.append(roc_auc_score(y_test, X_test[:, 2]))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_fit, X_validation, y_fit, y_validation = train_test_split(
            X_train_scaled,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE + fold_index,
            stratify=y_train,
        )
        torch.manual_seed(RANDOM_STATE + fold_index)
        scratch_model, _ = entrenar(
            X_fit,
            y_fit,
            X_validation,
            y_validation,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
            verbose=False,
        )
        scratch_results.append(
            _metrics(y_test, _neural_probabilities(scratch_model, X_test_scaled))
        )

        torch.manual_seed(RANDOM_STATE + fold_index)
        finetuned_model = deepcopy(pretrained_model)
        _train_finetuned_model(finetuned_model, X_train_scaled, y_train)
        finetuned_results.append(
            _metrics(y_test, _neural_probabilities(finetuned_model, X_test_scaled))
        )

    print("\n--- Resultados CV repetida (5 folds x 10 repeticiones) ---")
    _print_summary("LogisticRegression + StandardScaler", logistic_results)
    baseline_values = np.array(baseline_auc)
    print(
        "Baseline similitud_tags_posts (solo AUC, 50 folds): "
        f"{baseline_values.mean():.3f} +/- {baseline_values.std():.3f}"
    )
    _print_summary("CompatibilityNet desde cero", scratch_results)
    _print_summary("CompatibilityNet preentrenada + fine-tuning", finetuned_results)

    final_logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    final_logistic.fit(X, y)
    print("\n--- Coeficientes estandarizados (logistica ajustada con los 94 reales) ---")
    for feature, coefficient in zip(
        FEATURE_NAMES,
        final_logistic.named_steps["classifier"].coef_[0],
    ):
        print(f"{feature}: {coefficient:+.4f}")


if __name__ == "__main__":
    main()
