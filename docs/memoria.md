# Memoria Persistente del Agente IA

## Estado actual

- **Caso de uso:** clasificación binaria de compatibilidad de amistad entre dos perfiles.
- **Modelo:** `CompatibilityNet`, una MLP de PyTorch en `app/ml/model.py`. El `input_dim` es configurable y su valor predeterminado es 5; las capas ocultas son de 8 y 4 neuronas (5→8→4→1, con ReLU, Dropout 0.3 y 0.2, y Sigmoid a la salida). Son 18 neuronas y 89 parámetros.
- **Datos:** el proyecto trabaja **solo con datos sintéticos** (Mockaroo). `sintetico_features.csv` tiene 8000 filas (4000 compatibles y 4000 no compatibles) con las 5 features y la etiqueta `y`. No se usan datos reales.
- **API:** FastAPI, documentada automáticamente en `/docs`.
- **Persistencia:** PostgreSQL con SQLAlchemy y Alembic.
- **Arquitectura:** MVC, Repository, Factory, Service Layer y dependency injection de FastAPI.
- **Entorno:** edición en VS Code (Windows, PowerShell, entorno virtual local). El entrenamiento se corre en Google Colab con GPU, desde una copia del repo en Google Drive.

## Pipeline de entrenamiento

1. `scripts/split_dataset_csv.py` arma el split 70/15/15 estratificado (5600 / 1200 / 1200) desde `profiles_unificado.csv` y `friendships_unificado.csv`, y lo guarda en `data/splits.npz`.
2. `scripts/train_neural_net.py` entrena con Adam (lr 0.001, weight decay 1e-4), BCELoss, batch 16, hasta 200 épocas con early stopping (paciencia 20 sobre `val_loss`) y guarda los pesos en `models/compatibility_net.pt`.
3. `scripts/evaluate_model.py` evalúa sobre el test set con umbral 0.50.
4. `scripts/colab_busqueda_arquitectura.py` compara arquitecturas con validación cruzada y guarda los resultados en `results/`.

Las features no se escalan, así que `ModelFactory` carga el modelo sin ninguna transformación adicional.

## Resultados vigentes

- **Test sintético (1200 ejemplos, umbral 0.50):** accuracy 0.81; compatible con precision 0.83, recall 0.77 y F1 0.80; no_compatible con precision 0.79, recall 0.84 y F1 0.81. Entre corridas los valores varían en centésimas.
- **Búsqueda de arquitectura:** las 10 redes y la regresión logística dan un AUC de alrededor de 0.84. La diferencia entre la mejor y la segunda (0.0002) es menor que el desvío estándar (0.0097). No hay una arquitectura mejor, así que se mantiene 5→8→4→1.
- **Limitación conocida:** la etiqueta sintética depende sobre todo de `similitud_intereses`; las demás features aportan muy poca señal. Las métricas reflejan la regla del generador de datos, no amistades reales.

## Trabajo completado

- `CompatibilityNet` está integrado en `CompatibilityService`; el servicio construye las cinco features, ejecuta la inferencia y guarda la predicción a través del repositorio.
- El feedback loop está en `scripts/feedback_loop.py`: evalúa el F1 y, si no alcanza 0.70, hace un solo reentrenamiento con `max_epochs=400` y `weight_decay=1e-5`, conservando el mejor resultado. **No cambia la arquitectura automáticamente**; las capas ocultas siguen fijas en 8 y 4 y cualquier cambio de arquitectura requiere intervención manual.
- `scripts/features.py` concentra `calcular_features` y `FEATURE_NAMES`; los scripts de datos lo reutilizan.
- La suite de `tests/` tenía 62 tests (modelo, feedback loop, service, repositorio y endpoints) en la última corrida.
- CI de GitHub Actions en `.github/workflows/ci.yml`: ejecuta flake8 y pytest en cada push y pull request a `main`.
- Se hizo una limpieza del repo para dejar solo lo referente a los datos sintéticos (rama `limpieza-sintetico`): se eliminaron los scripts y CSV de datos reales y el modelo viejo de scikit-learn (`train_model.py`, `best_model.pkl`).

## Pendiente

- Interfaz (último paso del flujo dataset → configuración → entrenamiento → interfaz): predecir, confirmar el valor real, reentrenar y ver métricas antes y después. Fecha límite de entrega: 04/11.
- Feedback loop integrado a la API.
- Tickets y tareas en Trello.
- Confirmar con el profesor si "Codelab" (como figura en el enunciado) es Google Colab.

## Notas operativas

- Usar umbral de clasificación 0.50 para la CompatibilityNet.
- Mantener el mismo orden de las cinco features en entrenamiento e inferencia: `similitud_intereses`, `diff_activity`, `similitud_tags_posts`, `tiene_tags_posts`, `mismo_avatar_subcultura`.
- La lógica de las features existe en dos lugares: `scripts/features.py` (entrenamiento) y `app/services/compatibility_service.py` (inferencia). Si se cambia una, hay que cambiar la otra.
- La carpeta del repo en Google Drive es una copia manual: cada cambio en un script hay que subirlo a Drive antes de correr Colab.
- Para verificar cambios, ejecutar `pytest tests/ -v` y `flake8 app scripts tests`. Los comandos están en `comandos.md`.