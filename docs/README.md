# Predictor de Compatibilidad de Amistad — Red Neuronal

Proyecto de Machine Learning orientado a predecir compatibilidad de amistad entre usuarios, como evolución del `ExploreController` de proyecto individual de red social, "Yo Adolescente". Expone un modelo entrenado a través de una API REST documentada con Swagger, integrando flujos de trabajo autónomos mediante un agente de IA.

## 🎯 Objetivos del Proyecto

1. **Estandarizar el entorno de desarrollo** para la creación y experimentación de modelos.
2. **Reemplazar la heurística actual del ExploreController** por un modelo entrenado sobre datos reales de `friendships` y perfiles.
3. **Exponer el modelo como API REST** documentada automáticamente con Swagger.
4. **Integración con Agentes Autónomos** para la ejecución programada de tareas, testing y optimización.
5. **Control de versiones y reproducibilidad** del pipeline de ML.

## 🏗️ Arquitectura

- **Patrón**: MVC
- **API**: FastAPI (Swagger automático en `/docs`)
- **Base de datos**: PostgreSQL (SQLAlchemy ORM + Alembic para migraciones)
- **Framework de ML**: scikit-learn
- **Patrones de diseño aplicados**:
  - **Repository** — aísla el acceso a datos (`app/repositories/`)
  - **Factory** — centraliza la carga del modelo entrenado (`app/ml/model_factory.py`)
  - **Service Layer** — separa la lógica de negocio de los controllers
  - **Dependency Injection** — nativo de FastAPI, para inyectar sesión de DB y servicios

## 📁 Estructura del Proyecto
.
├── app/
│ ├── main.py # entrypoint FastAPI
│ ├── config.py # configuración (env vars, DB)
│ ├── database.py # conexión PostgreSQL
│ ├── controllers/ # C — endpoints/routers
│ ├── models/ # M — entidades ORM (SQLAlchemy)
│ ├── views/ # V — schemas Pydantic (request/response)
│ ├── services/ # lógica de negocio
│ ├── repositories/ # patrón Repository
│ └── ml/ # pipeline de ML (data, train, evaluate, factory)
├── data/
│ ├── raw/ # datos originales (no modificar)
│ └── processed/ # datos preprocesados
├── models/ # modelos entrenados (.pkl)
├── tests/
│ ├── unit/
│ └── integration/ # entorno de prueba del agente (no el IDE)
├── notebooks/
├── alembic/ # migraciones de base de datos
├── AGENT.md
├── memoria.md
├── comandos.md
├── Skill.md
├── requirements.txt
└── .gitignore

## 🛠️ Requisitos Previos

- Python 3.10+
- PostgreSQL
- Git

## 🚀 Puesta en marcha

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger disponible en `http://localhost:8000/docs`.

## 🔗 Contexto

Este proyecto usa como fuente de datos las tablas `friendships` y perfiles del proyecto [Yo Adolescente](../yo-adolescente), un fotolog retro. El objetivo final es predecir compatibilidad entre dos usuarios a partir de sus intereses y patrones de interacción.

## 📌 Estado

Boilerplate en construcción. La arquitectura MVC + API + base de datos está definida; falta:
- Definir dataset y features reales
- Entrenar el primer modelo baseline
- Migraciones de Alembic
- Tests unitarios e integración



## Modelo de predicción: red neuronal (PyTorch)

Reemplaza la versión anterior basada en LogisticRegression (scikit-learn).

**Arquitectura** (`app/ml/model.py`, clase `CompatibilityNet`):
- Linear(5→8) + ReLU + Dropout(0.3)
- Linear(8→4) + ReLU + Dropout(0.2)
- Linear(4→1) + Sigmoid

**Entrenamiento** (`scripts/train_neural_net.py`):
- Optimizador Adam (lr=0.001, weight_decay=1e-4)
- Loss: BCELoss
- Batch size: 16
- Hasta 200 épocas, con early stopping (paciencia 20 épocas sobre val_loss)
- Split 70/15/15 estratificado (`scripts/split_dataset.py`), guardado en `data/splits.npz`
- Pesos entrenados guardados en `models/compatibility_net.pt` (solo `state_dict`)

**Features** (las mismas 5 usadas por el modelo anterior, ver `scripts/train_model.py`):
`similitud_intereses`, `diff_activity`, `similitud_tags_posts`, `tiene_tags_posts`, `mismo_avatar_subcultura`

**Evaluación** (`scripts/evaluate_model.py`), test set (15 ejemplos, umbral 0.50):

| Clase          | Precision | Recall | F1   |
|----------------|-----------|--------|------|
| no_compatible  | 0.83      | 0.62   | 0.71 |
| compatible     | 0.67      | 0.86   | 0.75 |

Accuracy: 0.73. Mejora respecto a la LogisticRegression anterior, que casi no distinguía la clase `no_compatible` (recall ~0 con cualquier umbral razonable).

**Flujo de scripts**:
```bash
python scripts/split_dataset.py
python scripts/train_neural_net.py
python scripts/evaluate_model.py
```

## 🔄 Feedback Loop — Ciclo de Retroalimentación Manual

Durante el desarrollo del predictor de compatibilidad se ejecutó un ciclo iterativo completo de **evaluación → detección de problema → ajuste → reentrenamiento**, que mejoró significativamente la calidad del modelo.

### Fase 1: Evaluación Inicial (LogisticRegression)

El primer modelo entrenado fue una regresión logística con 3 features iniciales:
- `similitud_intereses`
- `diff_activity`
- `misma_subcultura`

Métricas iniciales con umbral 0.35:
- **Precision:** 0.60
- **Recall:** 1.00

**Problema detectado:** Al variar el umbral de decisión (0.5 → 0.35 → 0.25), el modelo saltaba abruptamente de "casi todo negativo" a "casi todo positivo" sin encontrar un punto de equilibrio intermedio. Esto indicaba un sesgo grave en los datos o una característica con poder discriminatorio excesivo.

### Fase 2: Detección del Problema Real

La raíz del problema se identificó en la **generación de pares negativos**:

- El dataset contenía **444 usuarios totales**, pero solo **19 usuarios "activos"** (con al menos un post público).
- Los pares negativos se generaban aleatoriamente sin considerar esta concentración.
- Consecuencia: la mayoría de pares negativos nunca incluían usuarios activos, mientras que la mayoría de pares positivos sí.
- La feature `tiene_tags_posts` (booleano sobre si ambos usuarios tenían posts) **separaba casi perfectamente el label por sí sola**, generando una falsa ilusión de 100% precision/recall sin generalizar.

Esto es un **sesgo de distribución**: el modelo aprendía a diferenciar entre "ambos con posts" vs "al menos uno sin posts", no compatibilidad real.

### Fase 3: Ajuste Aplicado

**Cambios en la generación de datos** (`scripts/extract_from_yo_adolescente.py`):

1. **Rebalanceo de negativos:** Modificar la lógica para que el 90% (`PROP_ACTIVOS=0.9`) de los pares negativos se genere entre usuarios que también tengan posts, reflejando la distribución real.

2. **Nuevas features extraídas:** Agregar dos features derivadas de posts públicos:
   - `similitud_tags_posts` — similaridad de Jaccard sobre los tags agregados de todos los posts del usuario.
   - `tiene_tags_posts` — booleano: ambos usuarios tienen al menos un post.

3. **Limpieza de datos:** Borrar todas las friendships y características calculadas con la lógica anterior y re-extraer desde cero con la lógica corregida.

### Fase 4: Reentrenamiento y Resultado

Con el dataset corregido y features depuradas:

- **Dataset:** 94 ejemplos totales (47 positivos, 47 negativos) — balanceado.
- **Features finales:** 5 en total (`similitud_intereses`, `diff_activity`, `similitud_tags_posts`, `tiene_tags_posts`, `mismo_avatar_subcultura`).
- **Modelo:** Migración de LogisticRegression a **red neuronal (CompatibilityNet, PyTorch)** para mejor capacidad de generalización.

**Resultados finales (test set, umbral 0.50):**

| Métrica        | Valor |
|---|---|
| **Accuracy**   | 0.73 |
| **Precision (compatible)**   | 0.67 |
| **Recall (compatible)**       | 0.86 |
| **F1 (compatible)**           | 0.75 |
| **Precision (no_compatible)** | 0.83 |
| **Recall (no_compatible)**    | 0.62 |
| **F1 (no_compatible)**        | 0.71 |

**Mejora significativa:** La LogisticRegression anterior no lograba distinguir la clase `no_compatible` (recall ≈ 0 con cualquier umbral), mientras que el nuevo modelo logra un equilibrio mucho mejor (F1 = 0.71 vs F1 = 0 previamente).

### Resumen del Ciclo

Este ciclo manual permitió identificar y corregir un **sesgo fundamental en la generación de datos** que habría perpetuado un modelo con baja generalización. La lección: la calidad del dataset es más determinante que la complejidad del modelo para problemas de clasificación.