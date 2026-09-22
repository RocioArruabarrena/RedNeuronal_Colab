# Memoria Persistente del Agente IA

## 🧠 Estado Actual del Proyecto

- **Fase actual:** Boilerplate — arquitectura MVC + API definida, falta entorno local funcionando y dataset real.
- **Caso de uso:** Predictor de compatibilidad de amistad, basado en tablas `friendships` y perfiles de Yo Adolescente, como versión con ML real del `ExploreController` actual.
- **Framework de ML:** scikit-learn.
- **API:** FastAPI, con Swagger automático en `/docs`.
- **Base de datos:** PostgreSQL (SQLAlchemy + Alembic).
- **Patrones aplicados:** Repository, Factory, Service Layer, Dependency Injection.
- **Entorno de ejecución del agente:** Codelab (no el IDE local).

## 📊 Registro de Experimentos e Interacciones

- Creación de la estructura del repositorio y documentos de control (`AGENT.md`, `Skill.md`, `comandos.md`).
- Definición de arquitectura MVC con FastAPI + PostgreSQL + Swagger.
- Pendiente: instalación de Python en el entorno local (Windows) — el alias de Microsoft Store estaba bloqueando la instalación real.

## 📌 Notas para Próximas Tareas
- Definir el objetivo final de la red neuronal (opciones evaluadas: clasificación, series temporales, voz, texto/sentimiento — caso elegido: compatibilidad de amistad, tipo clasificación binaria).
- Definir el esquema de features a partir de `friendships` y perfiles (intereses, actividad, mutuals).
- Exportar/anonimizar dataset inicial desde la base de Yo Adolescente hacia `data/raw/`.
- Definir arquitectura de la red (baseline: LogisticRegression/RandomForest antes de MLPClassifier).
- Configurar Alembic para migraciones de PostgreSQL.
- Definir proceso de entrenamiento (épocas, batch size, learning rate) y evaluación (precision, recall, F1) con feedback loop.



## Estado del modelo de compatibilidad (actualizado)

El modelo de predicción pasó de LogisticRegression a una red neuronal (PyTorch,
`CompatibilityNet` en `app/ml/model.py`). Si el agente necesita reentrenar,
evaluar o modificar el pipeline de predicción, usar estos scripts en orden:

1. `scripts/split_dataset.py` — genera `data/splits.npz` (70/15/15 estratificado)
2. `scripts/train_neural_net.py` — entrena y guarda `models/compatibility_net.pt`
3. `scripts/evaluate_model.py` — evalúa sobre el test set

**Importante**: el umbral de decisión de esta red es **0.50**, no 0.35 (ese
umbral era específico de la LogisticRegression vieja). Con umbrales menores a
0.5, este modelo tiende a predecir casi todo como "compatible".

Si se modifica alguna de las 5 features, hay que actualizarla tanto en
`train_model.py` (legacy) como en cualquier servicio que construya features
para la API (ej. `CompatibilityService`), para no repetir el bug histórico de
features inconsistentes entre entrenamiento y predicción.

Pendiente: integrar `CompatibilityNet` en `CompatibilityService` (hoy todavía
usa la lógica vieja de LogisticRegression).

## 🔄 Feedback Loop Manual vs. Automatizado

**Ciclo manual ya completado** (documentado en README.md):
Durante el desarrollo ocurrió un ciclo iterativo de evaluación → detección de problema → ajuste → reentrenamiento.
El problema real fue un sesgo grave en la generación de pares negativos (usuarios activos vs. inactivos), detectado al observar que `tiene_tags_posts` separaba casi perfectamente el label sin generalizar.
Solución: rebalanceo de negativos (90% entre usuarios activos), agregación de 2 features nuevas de tags, migración a red neuronal.
Resultado: mejora clara de F1 en clase `no_compatible` (de ~0 a 0.71).

**Feedback loop automatizado pendiente** (`scripts/feedback_loop.py`):
Este será un proceso programado que detecte automáticamente degradación de performance en producción y dispare reentrenamiento sin intervención manual.