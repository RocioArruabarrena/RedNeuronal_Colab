# Contexto y Configuración del Agente IA

## Rol

El agente mantiene el pipeline de machine learning, la API y la documentación
del predictor de compatibilidad de amistad. El modelo de inferencia es
`CompatibilityNet`, una red neuronal MLP implementada en PyTorch y servida por
FastAPI. Se entrena solo con un dataset sintético de 8000 pares.

## Flujo del proyecto

dataset → configuración → entrenamiento → interfaz.

1. **Dataset:** `profiles_unificado.csv`, `friendships_unificado.csv` y
   `sintetico_features.csv` (raíz del repo).
2. **Configuración:** arquitectura 5→8→4→1 en `app/ml/model.py` e
   hiperparámetros en `scripts/train_neural_net.py`.
3. **Entrenamiento y evaluación:** `scripts/split_dataset_csv.py`,
   `scripts/train_neural_net.py` y `scripts/evaluate_model.py`.
4. **Interfaz:** pendiente (predecir, confirmar el valor real, reentrenar y ver
   métricas).

## Entorno de trabajo

- Edición: VS Code en Windows, con PowerShell desde la raíz del repositorio.
- Python: entorno virtual local (`.venv`), con dependencias de `requirements.txt`.
- Entrenamiento: Google Colab (GPU), con una copia del repo en Google Drive.
- Base de datos de la aplicación: PostgreSQL, configurada mediante `.env`.
- Pruebas autónomas: GitHub Actions (`.github/workflows/ci.yml`), que corre
  flake8 y pytest en cada push y pull request a `main`.

## Objetivos y reglas

- Ejecutar la suite con `pytest tests/ -v` y revisar las métricas de evaluación
  (precision, recall y F1).
- El feedback loop actual ajusta hiperparámetros y reentrena una sola vez; no
  modifica automáticamente las capas ocultas de la red.
- Mantener la separación Controller / Service / Repository / Model y el patrón
  MVC. La lógica de negocio pertenece a `app/services/` y el acceso a datos a
  `app/repositories/`.
- Mantener el mismo orden de las 5 features en entrenamiento e inferencia
  (`scripts/features.py` y `app/services/compatibility_service.py`).
- Registrar cambios estructurales y comandos nuevos en `comandos.md`, y
  decisiones y estado en `memoria.md`.
- Mantener Swagger (`/docs`) consistente con los schemas de `app/views/`.
- No modificar los CSV sintéticos originales sin regenerar
  `sintetico_features.csv` y `data/splits.npz`, y sin volver a entrenar.
- Al reportar métricas, aclarar que provienen de datos sintéticos.