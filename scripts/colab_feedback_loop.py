"""Demo autocontenido de feedback para ejecutar en Google Colab."""

from pathlib import Path
import re

import ipywidgets as widgets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Ajustar estas rutas al subir los archivos o montar Google Drive en Colab.
REALES_CSV = Path("/content/reales_features.csv")
CHECKPOINT_PATH = Path("/content/compatibility_net.pt")
FEEDBACK_CSV = Path("/content/feedback.csv")
OUTPUT_DIR = Path("/content/modelos_feedback")

FEATURE_NAMES = [
    "similitud_intereses",
    "diff_activity",
    "similitud_tags_posts",
    "tiene_tags_posts",
    "mismo_avatar_subcultura",
]
LABEL_NAMES = ("y", "label", "is_mutual")
UMBRAL = 0.5
ENTRENAR_DESDE_CERO = True
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CompatibilityNet(nn.Module):
    """Arquitectura copiada de app/ml/model.py, sin importar el proyecto."""

    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


modelo_actual = None
scaler_actual = None
ruta_checkpoint_actual = CHECKPOINT_PATH
X_train_real = None
X_holdout = None
y_train_real = None
y_holdout = None
metricas_antes = None
metricas_despues = None
feedback_acumulado = 0


def _leer_etiquetas(values: pd.Series) -> np.ndarray:
    if values.dtype == object:
        normalized = values.astype(str).str.strip().str.lower()
        mapped = normalized.map(
            {"true": 1, "false": 0, "yes": 1, "no": 0, "si": 1, "sí": 1}
        )
        values = mapped.where(mapped.notna(), pd.to_numeric(values, errors="coerce"))
    labels = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser binarias: 0/1 o true/false.")
    return labels


def _cargar_reales() -> tuple[np.ndarray, np.ndarray]:
    if not REALES_CSV.is_file():
        raise FileNotFoundError(f"No se encontro el CSV de reales: {REALES_CSV}")
    frame = pd.read_csv(REALES_CSV)
    if frame.shape[1] != 6:
        raise ValueError(
            "El CSV debe tener exactamente 6 columnas: 5 features y la etiqueta y."
        )

    if all(name in frame.columns for name in FEATURE_NAMES):
        feature_columns = FEATURE_NAMES
    else:
        feature_columns = list(frame.columns[:5])
    label_column = next(
        (name for name in LABEL_NAMES if name in frame.columns), frame.columns[-1]
    )

    features = frame[feature_columns].to_numpy(dtype=np.float32)
    labels = _leer_etiquetas(frame[label_column])
    if not np.isfinite(features).all():
        raise ValueError("El CSV contiene features vacias o no numericas.")
    if len(features) != len(labels) or len(features) < 5:
        raise ValueError("Se requieren al menos 5 filas completas para el split.")
    return features, labels


def _cargar_modelo(path: Path) -> tuple[CompatibilityNet, StandardScaler | None]:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el checkpoint: {path}")
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(saved, dict) and "state_dict" in saved:
        state_dict = saved["state_dict"]
        scaler_state = saved.get("scaler")
    elif isinstance(saved, dict) and "model_state_dict" in saved:
        state_dict = saved["model_state_dict"]
        scaler_state = saved.get("scaler")
    else:
        state_dict = saved
        scaler_state = None
    model = CompatibilityNet(input_dim=len(FEATURE_NAMES)).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    scaler = None
    if scaler_state is not None:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaler_state["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(scaler_state["scale"], dtype=np.float64)
        scaler.var_ = np.asarray(scaler_state["var"], dtype=np.float64)
        scaler.n_features_in_ = len(FEATURE_NAMES)
        scaler.n_samples_seen_ = int(scaler_state["n_samples_seen"])
    return model, scaler


def predecir(features: list[float] | np.ndarray) -> float:
    """Devuelve el score de compatibilidad del modelo cargado (0 a 1)."""
    if modelo_actual is None:
        raise RuntimeError("Primero ejecuta iniciar_demo() para cargar el checkpoint.")
    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("features debe contener exactamente 5 numeros finitos.")
    model_features = vector.reshape(1, -1)
    if scaler_actual is not None:
        model_features = scaler_actual.transform(model_features)
    tensor = torch.as_tensor(model_features, dtype=torch.float32, device=DEVICE)
    modelo_actual.eval()
    with torch.no_grad():
        return float(modelo_actual(tensor).item())


def _normalizar_etiqueta(etiqueta_real: int | bool) -> int:
    if isinstance(etiqueta_real, str):
        normalized = etiqueta_real.strip().lower()
        if normalized in ("true", "si", "sí", "yes", "1"):
            return 1
        if normalized in ("false", "no", "0"):
            return 0
    if etiqueta_real in (0, 1, False, True):
        return int(etiqueta_real)
    raise ValueError("etiqueta_real debe ser 1/0, True/False o Si/No.")


def registrar_feedback(
    features: list[float] | np.ndarray, etiqueta_real: int | bool
) -> float:
    """Anade una fila con features, etiqueta confirmada y score del modelo."""
    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("features debe contener exactamente 5 numeros finitos.")
    label = _normalizar_etiqueta(etiqueta_real)
    score = predecir(vector)
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {name: float(value) for name, value in zip(FEATURE_NAMES, vector)}
    row.update({"y": label, "score_predicho": score})
    pd.DataFrame([row]).to_csv(
        FEEDBACK_CSV,
        mode="a",
        header=not FEEDBACK_CSV.exists(),
        index=False,
    )
    return score


def _leer_feedback() -> tuple[np.ndarray, np.ndarray]:
    if not FEEDBACK_CSV.is_file():
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(
            0, dtype=np.int64
        )
    frame = pd.read_csv(FEEDBACK_CSV)
    required = [*FEATURE_NAMES, "y"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas en feedback.csv: {missing}")
    features = frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    labels = _leer_etiquetas(frame["y"])
    if not np.isfinite(features).all():
        raise ValueError("feedback.csv contiene features no numericas.")
    return features, labels


def _medir(
    model: CompatibilityNet, scaler: StandardScaler | None = None
) -> dict:
    model.eval()
    holdout_features = X_holdout
    if scaler is not None:
        holdout_features = scaler.transform(holdout_features)
    tensor = torch.as_tensor(
        holdout_features, dtype=torch.float32, device=DEVICE
    )
    with torch.no_grad():
        scores = model(tensor).squeeze(1).cpu().numpy()
    predictions = (scores >= UMBRAL).astype(np.int64)
    auc = (
        roc_auc_score(y_holdout, scores)
        if len(np.unique(y_holdout)) == 2
        else float("nan")
    )
    return {
        "AUC": auc,
        "Accuracy": accuracy_score(y_holdout, predictions),
        "F1": f1_score(y_holdout, predictions, zero_division=0),
        "Matriz de confusion": confusion_matrix(
            y_holdout, predictions, labels=[0, 1]
        ).tolist(),
    }


def _ruta_versionada() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    versions = [
        int(match.group(1))
        for path in OUTPUT_DIR.glob("compatibility_net_v*.pt")
        if (match := re.fullmatch(r"compatibility_net_v(\d+)\.pt", path.name))
    ]
    return OUTPUT_DIR / f"compatibility_net_v{max([1, *versions]) + 1}.pt"


def reentrenar() -> dict:
    """Entrena con reales y feedback, y guarda modelo y scaler versionados."""
    global modelo_actual, scaler_actual, ruta_checkpoint_actual, metricas_despues
    global feedback_acumulado

    if modelo_actual is None or X_train_real is None:
        raise RuntimeError("Primero ejecuta iniciar_demo() para preparar los datos.")
    feedback_x, feedback_y = _leer_feedback()
    feedback_acumulado = len(feedback_y)
    if not feedback_acumulado and not ENTRENAR_DESDE_CERO:
        raise ValueError("Registra al menos un feedback antes de reentrenar.")

    train_x = np.concatenate((X_train_real, feedback_x), axis=0)
    train_y = np.concatenate((y_train_real, feedback_y), axis=0)
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x).astype(np.float32)
    dataset = TensorDataset(
        torch.as_tensor(train_x_scaled, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(
        dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True
    )
    if ENTRENAR_DESDE_CERO:
        torch.manual_seed(42)
        model = CompatibilityNet(input_dim=len(FEATURE_NAMES)).to(DEVICE)
    else:
        model, _ = _cargar_modelo(ruta_checkpoint_actual)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    new_path = _ruta_versionada()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "feedback_count": feedback_acumulado,
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist(),
                "n_samples_seen": int(scaler.n_samples_seen_),
            },
        },
        new_path,
    )
    model.eval()
    modelo_actual = model
    scaler_actual = scaler
    ruta_checkpoint_actual = new_path
    metricas_despues = _medir(modelo_actual, scaler_actual)
    print(f"Modelo nuevo guardado en: {new_path}")
    return metricas_despues


def _tabla_metricas() -> pd.DataFrame:
    rows = []
    for label, metrics in (
        ("Antes", metricas_antes),
        ("Despues", metricas_despues),
    ):
        if metrics is None:
            continue
        rows.append(
            {
                "Estado": label,
                "AUC": metrics["AUC"],
                "Accuracy": metrics["Accuracy"],
                "F1": metrics["F1"],
                "Matriz de confusion": str(metrics["Matriz de confusion"]),
                "Feedback acumulado": feedback_acumulado,
            }
        )
    return pd.DataFrame(rows)


def iniciar_demo() -> None:
    """Carga datos/checkpoint, fija el holdout y muestra los widgets."""
    global modelo_actual, scaler_actual, ruta_checkpoint_actual, X_train_real
    global X_holdout
    global y_train_real, y_holdout, metricas_antes, metricas_despues
    global feedback_acumulado

    features, labels = _cargar_reales()
    train_indices, holdout_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )
    X_train_real, X_holdout = features[train_indices], features[holdout_indices]
    y_train_real, y_holdout = labels[train_indices], labels[holdout_indices]
    # Con unos 19 ejemplos de holdout, las metricas son ruidosas y cambian mucho.
    modelo_actual, scaler_actual = _cargar_modelo(CHECKPOINT_PATH)
    ruta_checkpoint_actual = CHECKPOINT_PATH
    metricas_antes = _medir(modelo_actual)
    metricas_despues = dict(metricas_antes)
    feedback_acumulado = len(_leer_feedback()[1])

    controls = {}
    for name in FEATURE_NAMES:
        maximum = 100.0 if name == "diff_activity" else 1.0
        controls[name] = widgets.FloatSlider(
            description=name,
            min=0.0,
            max=maximum,
            step=0.1 if maximum > 1.0 else 0.01,
            value=float(np.median(X_train_real[:, FEATURE_NAMES.index(name)])),
            continuous_update=False,
            layout=widgets.Layout(width="95%"),
        )

    predict_button = widgets.Button(description="Predecir", button_style="primary")
    yes_button = widgets.Button(description="Son amigos", button_style="success")
    no_button = widgets.Button(description="No son amigos", button_style="warning")
    train_button = widgets.Button(description="Reentrenar", button_style="info")
    prediction_output = widgets.Output()
    feedback_output = widgets.Output()
    metrics_output = widgets.Output()
    last_features = {"values": None}
    yes_button.disabled = True
    no_button.disabled = True

    def current_features() -> list[float]:
        return [controls[name].value for name in FEATURE_NAMES]

    def show_metrics() -> None:
        with metrics_output:
            metrics_output.clear_output(wait=True)
            display(_tabla_metricas())

    def on_feature_change(_change) -> None:
        last_features["values"] = None
        yes_button.disabled = True
        no_button.disabled = True

    def on_predict(_button) -> None:
        vector = current_features()
        score = predecir(vector)
        last_features["values"] = vector.copy()
        yes_button.disabled = False
        no_button.disabled = False
        with prediction_output:
            prediction_output.clear_output(wait=True)
            print(f"Score de compatibilidad: {score:.4f}")

    def on_feedback(label: int) -> None:
        global feedback_acumulado
        vector = last_features["values"]
        if vector is None:
            return
        score = registrar_feedback(vector, label)
        last_features["values"] = None
        yes_button.disabled = True
        no_button.disabled = True
        feedback_acumulado = len(_leer_feedback()[1])
        with feedback_output:
            feedback_output.clear_output(wait=True)
            print(f"Feedback guardado. Score registrado: {score:.4f}")
        show_metrics()

    def on_retrain(_button) -> None:
        with feedback_output:
            feedback_output.clear_output(wait=True)
            try:
                metrics = reentrenar()
                last_features["values"] = None
                yes_button.disabled = True
                no_button.disabled = True
                print(
                    "Reentrenamiento terminado. "
                    f"AUC={metrics['AUC']:.3f}, F1={metrics['F1']:.3f}"
                )
            except (ValueError, RuntimeError, FileNotFoundError) as error:
                print(f"No se pudo reentrenar: {error}")
        show_metrics()

    for control in controls.values():
        control.observe(on_feature_change, names="value")
    predict_button.on_click(on_predict)
    yes_button.on_click(lambda _button: on_feedback(1))
    no_button.on_click(lambda _button: on_feedback(0))
    train_button.on_click(on_retrain)
    show_metrics()
    display(
        widgets.VBox(
            [
                *(controls[name] for name in FEATURE_NAMES),
                widgets.HBox([predict_button, yes_button, no_button, train_button]),
                prediction_output,
                feedback_output,
                metrics_output,
            ]
        )
    )


if __name__ == "__main__":
    iniciar_demo()
