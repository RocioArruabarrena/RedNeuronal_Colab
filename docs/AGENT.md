# Contexto y Configuración del Agente IA

## Rol

El agente mantiene el pipeline de machine learning, la API y la documentación
del predictor de compatibilidad de amistad. El modelo de inferencia es
`CompatibilityNet`, una red neuronal MLP implementada en PyTorch y servida por
FastAPI. Se entrena solo con un dataset sintético de 8000 pares.

## Flujo del proyecto

dataset → configuración → entrenamiento → interfaz.

1. **Dataset:** `profiles_unificado.csv` y `friendships_unificado.csv` están
   versionados. `sintetico_features.csv` se genera localmente con
   `python scripts/exportar_sintetico_csv.py` y no se versiona.
2. **Configuración:** arquitectura 5→8→4→1 en `app/ml/model.py` e
   hiperparámetros en `scripts/entrenar_red.py`.
3. **Entrenamiento y evaluación:** `scripts/split_dataset_csv.py`,
   `scripts/entrenar_red.py` y `scripts/evaluate_model.py`.
4. **API e interfaz:** disponible en `/`, para predecir, enviar una respuesta
   Sí/No, reentrenar y consultar métricas.

## Entorno de trabajo

- Edición: VS Code en Windows, con PowerShell desde la raíz del repositorio.
- Python: entorno virtual local (`.venv`), con dependencias de
  `requirements.txt`.
- Demo y entrenamiento: local o en Google Colab. Los notebooks se entregan
  junto con el repo, pero no están commiteados en este checkout.
- Base de datos de la aplicación: PostgreSQL, configurada mediante `.env`.
- CI: GitHub Actions (`.github/workflows/ci.yml`) ejecuta flake8 y pytest en
  pushes y pull requests a `main` y ramas `limpieza-sintetico*`.

## API y reentrenamiento

- `POST /compatibility/predict`: predice la compatibilidad de dos perfiles.
- `POST /compatibility/feedback`: registra la respuesta real Sí/No.
- `POST /compatibility/retrain`: hace fine-tuning desde el modelo actual.
- `GET /compatibility/metrics`: devuelve las métricas del último entrenamiento.
- El reentrenamiento de `app/services/servicio_reentrenamiento.py` toma la
  respuesta más reciente por par `(user_id, friend_id)`, exige al menos cinco
  pares distintos y rechaza el modelo si su F1 de test cae más de 0.02.
- `app/services/metrics_service.py` lee las métricas persistidas.

## Objetivos y reglas

- Ejecutar la suite con `pytest tests/ -v` y el linter con
  `flake8 app scripts tests`.
- Mantener la separación Controller / Service / Repository / Model. La lógica
  de negocio pertenece a `app/services/` y el acceso a datos a
  `app/repositories/`.
- Mantener el mismo orden de las cinco features en entrenamiento e inferencia
  (`scripts/features.py` y `app/services/compatibility_service.py`).
- Registrar cambios estructurales y comandos nuevos en `comandos.md`, y
  decisiones y estado en `memoria.md`.
- Mantener Swagger (`/docs`) consistente con los schemas de `app/views/`.
- `sintetico_features.csv` está en `.gitignore`; regenerarlo con
  `python scripts/exportar_sintetico_csv.py` si hace falta. Si cambian los
  datos, regenerar también `data/splits.npz` y volver a entrenar.
- Al reportar métricas, aclarar que provienen de datos sintéticos.
