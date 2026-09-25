# scripts/feedback_loop.py
"""
Feedback loop automatizado del predictor de compatibilidad.

Evalua CompatibilityNet (models/compatibility_net.pt) sobre el test set.
Si el F1 de la clase "compatible" no alcanza el umbral minimo, aplica UN
ajuste fijo de hiperparametros (mas epocas, menos regularizacion) y
reentrena una sola vez. No reintenta en loop infinito: si tras ese unico
reintento sigue sin alcanzar el umbral, informa el resultado y deja
guardado el mejor modelo obtenido entre los dos entrenamientos.

Correr desde la raiz del proyecto, con el venv activado:
    python scripts/feedback_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.evaluate_model import DATA_PATH, MODEL_PATH, cargar_datos_test, evaluar
from scripts.train_neural_net import cargar_split, entrenar

F1_MINIMO = 0.70
UMBRAL_CLASIFICACION = 0.50

# Ajuste unico aplicado si la evaluacion inicial no supera F1_MINIMO:
# mas epocas de margen para converger y menos weight_decay (menos
# regularizacion) sobre un dataset chico que puede necesitar mas
# libertad para separar las clases.
AJUSTE_MAX_EPOCHS = 400
AJUSTE_WEIGHT_DECAY = 1e-5


def _imprimir_resultado(etiqueta: str, resultado: dict) -> None:
    print(
        f"[{etiqueta}] Precision: {resultado['precision']:.2f} | "
        f"Recall: {resultado['recall']:.2f} | F1: {resultado['f1']:.2f}"
    )


def main() -> None:
    split = cargar_split()
    X_test, y_test = cargar_datos_test(DATA_PATH)

    # --- Evaluacion del modelo actual (el que ya esta en models/) ---
    from scripts.evaluate_model import cargar_modelo

    modelo_actual = cargar_modelo(MODEL_PATH, input_dim=X_test.shape[1])
    resultado_inicial = evaluar(
        modelo_actual, X_test, y_test, umbral=UMBRAL_CLASIFICACION
    )
    _imprimir_resultado("Evaluacion inicial", resultado_inicial)

    if resultado_inicial["f1"] >= F1_MINIMO:
        print(
            f"\nF1 >= {F1_MINIMO} -> el modelo actual cumple el umbral. No se reentrena."
        )
        return

    print(
        f"\nF1 {resultado_inicial['f1']:.2f} < {F1_MINIMO} -> se aplica el ajuste "
        f"unico y se reentrena.\n"
        f"  max_epochs: {AJUSTE_MAX_EPOCHS} (antes 200) | "
        f"weight_decay: {AJUSTE_WEIGHT_DECAY} (antes 1e-4)"
    )

    # --- Reentrenamiento con el ajuste ---
    modelo_nuevo, mejor_val_loss = entrenar(
        split["X_train"],
        split["y_train"],
        split["X_val"],
        split["y_val"],
        max_epochs=AJUSTE_MAX_EPOCHS,
        weight_decay=AJUSTE_WEIGHT_DECAY,
        verbose=False,
    )
    resultado_nuevo = evaluar(modelo_nuevo, X_test, y_test, umbral=UMBRAL_CLASIFICACION)
    _imprimir_resultado("Evaluacion tras reentrenamiento", resultado_nuevo)

    # Nos quedamos con el mejor de los dos F1 (no siempre el ajuste mejora).
    if resultado_nuevo["f1"] >= resultado_inicial["f1"]:
        torch.save(modelo_nuevo.state_dict(), MODEL_PATH)
        mejor_f1 = resultado_nuevo["f1"]
        print(f"\nSe guarda el modelo reentrenado en {MODEL_PATH} (mejor o igual F1).")
    else:
        mejor_f1 = resultado_inicial["f1"]
        print(
            "\nEl reentrenamiento no mejoro el F1 -> se conserva el modelo original "
            f"en {MODEL_PATH}."
        )

    if mejor_f1 >= F1_MINIMO:
        print(f"Feedback loop exitoso: F1 final {mejor_f1:.2f} >= {F1_MINIMO}.")
    else:
        print(
            f"Feedback loop agotado tras un unico reintento: F1 final {mejor_f1:.2f} "
            f"sigue por debajo de {F1_MINIMO}. No se reintenta automaticamente de nuevo; "
            "revisar arquitectura, features o dataset manualmente."
        )


if __name__ == "__main__":
    main()
