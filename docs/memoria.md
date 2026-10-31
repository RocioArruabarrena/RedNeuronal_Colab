# Memoria Persistente del Agente IA

## Estado actual

- **Caso de uso:** clasificación binaria de compatibilidad de amistad entre
  dos perfiles.
- **Modelo:** `CompatibilityNet`, MLP de PyTorch en `app/ml/model.py`, con
  entrada de 5 features, capas ocultas de 8 y 4 neuronas, ReLU, Dropout 0.3 y
  0.2, y Sigmoid a la salida. Tiene 89 parámetros.
- **Datos:** solo datos sintéticos. `profiles_unificado.csv` y
  `friendships_unificado.csv` están versionados. `sintetico_features.csv`
  tiene 8000 filas y se genera localmente con
  `python scripts/exportar_sintetico_csv.py`; está en `.gitignore` y no se
  versiona.
- **API e interfaz:** FastAPI y PostgreSQL con SQLAlchemy/Alembic. La interfaz
  está en `/` (`app/static/index.html`).
- **Arquitectura:** MVC adaptado a API REST, Repository, Factory, Service Layer
  y DI mediante `Depends`. SOLID son principios, no un patrón ni una función
  nativa de FastAPI.
- **Entorno:** Windows/PowerShell; la demo puede correr localmente o en Colab.
  Los notebooks se entregan junto con el repo, pero no están commiteados en
  este checkout.

## API y reentrenamiento

- `POST /compatibility/predict`: predice la compatibilidad de dos perfiles.
- `POST /compatibility/feedback`: guarda la respuesta real (`true`/`false`).
- `POST /compatibility/retrain`: hace fine-tuning desde el modelo actual.
- `GET /compatibility/metrics`: devuelve las métricas del último
  reentrenamiento.
- `app/services/servicio_reentrenamiento.py` conserva la última respuesta por
  par `(user_id, friend_id)`, exige al menos cinco pares distintos y rechaza
  el modelo si su F1 de test cae más de 0.02.
- `app/services/metrics_service.py` lee las métricas guardadas. La interfaz
  permite predecir, dar feedback Sí/No y usar un panel para reentrenar y ver
  métricas.

## Pipeline de entrenamiento

1. `scripts/split_dataset_csv.py` arma el split 70/15/15 estratificado
   (5600 / 1200 / 1200) desde los CSV unificados y lo guarda en
   `data/splits.npz`.
2. `scripts/entrenar_red.py` entrena con Adam (lr 0.001, weight decay 1e-4),
   BCELoss, batch 16 y hasta 200 épocas con early stopping (paciencia 20).
   Guarda los pesos en `models/compatibility_net.pt`.
3. `scripts/evaluate_model.py` evalúa sobre test con umbral 0.50.
4. `scripts/colab_busqueda_arquitectura.py` compara arquitecturas con
   validación cruzada. El CSV generado no está disponible en este checkout;
   los resultados quedan documentados en la tabla del README.

Las features no se escalan. `ModelFactory` carga el modelo sin transformación
adicional.

## Resultados vigentes

- **Test sintético (1200 ejemplos, umbral 0.50):** accuracy 0.81; compatible
  con precision 0.83, recall 0.77 y F1 0.80; no_compatible con precision 0.79,
  recall 0.84 y F1 0.81. Entre corridas los valores varían en centésimas.
- **Búsqueda de arquitectura:** las 10 redes y la regresión logística dan un
  AUC de alrededor de 0.84. La diferencia entre la mejor y la segunda
  (0.0002) es menor que el desvío estándar (0.0097). Se mantiene 5→8→4→1.
- **Limitación:** la etiqueta sintética depende sobre todo de
  `similitud_intereses`; las métricas reflejan la regla del generador, no
  amistades reales.

## Trabajo completado

- `CompatibilityService` calcula las cinco features y realiza la inferencia.
- La interfaz y el ciclo de feedback, reentrenamiento y consulta de métricas
  están integrados con la API.
- El reentrenamiento inicia desde el modelo actual, usa la última respuesta
  por par, mezcla feedback con una muestra sintética y limita la caída de F1
  de test a 0.02.
- `scripts/feedback_loop.py` se mantiene como herramienta CLI independiente;
  no es el reentrenamiento por feedback de la API.
- `scripts/features.py` concentra `calcular_features` y `FEATURE_NAMES`; los
  scripts de datos lo reutilizan.
- La suite cuenta con 115 tests; los de controller cubren health, predict,
  feedback, retrain, métricas y Swagger.
- GitHub Actions (`.github/workflows/ci.yml`) ejecuta flake8 y pytest en
  pushes y pull requests a `main` y ramas `limpieza-sintetico*`.
- Se limpiaron scripts y CSV de datos reales y el modelo antiguo de
  scikit-learn.

## Pendiente

- Merge a `main`.
- Entrega del proyecto el 04/11.

## Notas operativas

- Usar umbral de clasificación 0.50 para `CompatibilityNet`.
- Mantener este orden de features en entrenamiento e inferencia:
  `similitud_intereses`, `diff_activity`, `similitud_tags_posts`,
  `tiene_tags_posts`, `mismo_avatar_subcultura`.
- La lógica de las features existe en `scripts/features.py` y
  `app/services/compatibility_service.py`; si se cambia una, revisar la otra.
- Los notebooks son parte de la entrega, pero no están commiteados en este
  checkout.
- Para verificar cambios, ejecutar `pytest tests/ -v` y
  `flake8 app scripts tests`. Los comandos están en `comandos.md`.
