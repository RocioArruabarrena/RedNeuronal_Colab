# Memoria Persistente del Agente IA

## Estado actual

- **Caso de uso:** clasificación binaria de compatibilidad entre perfiles y amistades.
- **Modelo:** `CompatibilityNet`, una MLP de PyTorch en `app/ml/model.py`. El `input_dim` es configurable y su valor predeterminado es 5; las capas ocultas actuales tienen 8 y 4 neuronas.
- **API:** FastAPI, documentada automáticamente en `/docs`.
- **Persistencia:** PostgreSQL con SQLAlchemy y Alembic.
- **Arquitectura:** MVC, Repository, Factory, Service Layer y dependency injection de FastAPI.
- **Entorno local:** VS Code en Windows, PowerShell y entorno virtual local.

## Trabajo completado

- `CompatibilityNet` está integrado en `CompatibilityService`; el servicio construye las cinco features, ejecuta la inferencia y guarda la predicción a través del repositorio.
- El feedback loop está implementado en `scripts/feedback_loop.py`: evalúa el F1 y, si no alcanza 0.70, hace un solo reentrenamiento con `max_epochs=400` y `weight_decay=1e-5`, conservando el mejor resultado. **No cambia la arquitectura automáticamente**; las capas ocultas siguen fijas en 8 y 4 y cualquier cambio arquitectónico requiere intervención manual.
- La suite de `tests/` contiene 62 tests para modelo, feedback loop, service,
	repositorio y endpoints de la API.
- CI de GitHub Actions está configurado en `.github/workflows/ci.yml`; ejecuta flake8 y pytest en cada push y pull request a `main`.
- El entrenamiento y la evaluación usan `scripts/split_dataset.py`, `scripts/train_neural_net.py` y `scripts/evaluate_model.py`; el modelo serializado se guarda en `models/compatibility_net.pt`.
- Está completado el ciclo de corrección del sesgo de pares negativos y la migración del modelo anterior de LogisticRegression a PyTorch. scikit-learn permanece como utilidad para particiones y métricas, no como modelo de inferencia.

## Notas operativas

- Usar umbral de clasificación 0.50 para la CompatibilityNet actual; 0.35 era una referencia del modelo histórico de LogisticRegression.
- Mantener el mismo orden de las cinco features en entrenamiento e inferencia: `similitud_intereses`, `diff_activity`, `similitud_tags_posts`, `tiene_tags_posts`, `mismo_avatar_subcultura`.
- Para verificar cambios, ejecutar `pytest tests/ -v` y `flake8 app scripts tests`. Los comandos están en `comandos.md`.