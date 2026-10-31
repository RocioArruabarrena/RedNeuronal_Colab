# Predictor de Compatibilidad de Amistad — Red Neuronal

Proyecto de Machine Learning que predice la compatibilidad de amistad entre dos usuarios de una red social tipo fotolog ("Yo, Adolescente"). Una red neuronal (PyTorch) recibe 5 números sobre un par de usuarios y devuelve un score entre 0 y 1: si es mayor o igual a 0.5, el par se considera compatible.

El modelo se entrena **solo con un dataset sintético** (8000 pares) y se expone mediante una API REST documentada con Swagger.

## 🎯 Objetivos del proyecto

1. Estandarizar el entorno para crear, entrenar y evaluar un modelo de red neuronal.
2. Seguir el flujo **dataset → configuración → entrenamiento → interfaz**.
3. Exponer el modelo entrenado como API REST (FastAPI + Swagger).
4. Aplicar buenas prácticas de arquitectura, tests y entorno de prueba autónomo (CI).
5. Dejar documentación para que un agente de IA pueda trabajar en el proyecto (`docs/AGENT.md`, `docs/memoria.md`).

## 🏗️ Arquitectura y patrones

- **MVC adaptado a una API REST**: modelos ORM en `app/models/`, endpoints en
  `app/controllers/`, schemas de entrada/salida en `app/views/` y la interfaz
  web en `app/static/index.html`. `views/` contiene schemas Pydantic, no vistas
  HTML renderizadas en el servidor.
- **Repository** — `app/repositories/contratos.py` define las abstracciones y
  `app/repositories/repositorio_amistades.py` implementa el acceso con
  SQLAlchemy. Los servicios dependen de contratos; los proveedores del
  controller componen la implementación concreta.
- **Factory** — `app/ml/fabrica_modelo.py` (`ModelFactory`) centraliza el
  registro de arquitecturas y la carga, caché y recarga del modelo.
- **Service Layer** — `app/services/compatibility_service.py`,
  `app/services/servicio_reentrenamiento.py` y
  `app/services/metrics_service.py` concentran los casos de uso fuera de las
  rutas.
- **Dependency Injection** — FastAPI proporciona `Depends(...)` en
  `app/controllers/compatibility_controller.py` para resolver sesión y
  servicios. SOLID no es un patrón ni algo "nativo" de FastAPI: los servicios
  aplican principalmente inversión de dependencias mediante los contratos de
  repositorio, y las rutas actúan como composición y traducción HTTP.

## 📁 Estructura del proyecto

```text
.
├── .env.example
├── .flake8
├── .gitignore
├── .github/workflows/ci.yml
├── alembic.ini
├── alembic/
│   ├── README
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 26826_agregar_tags.py
│       ├── 55847864579c_crear_profiles_y_friendships.py
│       └── a8b545be77f2_crear_tabla_feedback.py
├── app/
│   ├── config.py
│   ├── database.py
│   ├── main.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── compatibility_controller.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── almacenamiento.py
│   │   ├── criterios_aceptacion.py
│   │   ├── dataset_feedback.py
│   │   ├── entrenador.py
│   │   ├── evaluador.py
│   │   ├── fabrica_modelo.py
│   │   └── model.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── feedback.py
│   │   ├── friendship.py
│   │   └── profile.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── contratos.py
│   │   └── repositorio_amistades.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── compatibility_service.py
│   │   ├── metrics_service.py
│   │   └── servicio_reentrenamiento.py
│   ├── static/index.html
│   └── views/
│       ├── __init__.py
│       └── compatibility_schema.py
├── data/splits.npz
├── docs/
│   ├── AGENT.md
│   ├── Skill.md
│   ├── comandos.md
│   ├── explicacion.md
│   └── memoria.md
├── models/compatibility_net.pt
├── notebooks/                  # carpeta sin notebooks commiteados
├── results/                    # carpeta sin resultados commiteados
├── scripts/
│   ├── __init__.py
│   ├── cargar_sinteticos_db.py
│   ├── colab_busqueda_arquitectura.py
│   ├── entrenar_red.py
│   ├── evaluate_model.py
│   ├── exportar_sintetico_csv.py
│   ├── features.py
│   ├── feedback_loop.py
│   ├── limpiar_friendships.py
│   ├── split_dataset_csv.py
│   ├── ver_conteos.py
│   ├── ver_feedback.py
│   └── ver_friendships.py
├── tests/
│   ├── test_compatibility_controller.py
│   ├── test_compatibility_service.py
│   ├── test_componentes_reentrenamiento.py
│   ├── test_fabrica_modelo.py
│   ├── test_feedback_loop.py
│   ├── test_friendship_repository.py
│   ├── test_metrics_service.py
│   ├── test_model.py
│   └── test_retrain_service.py
├── pytest.ini
├── friendships_unificado.csv
├── profiles_unificado.csv
├── generar_friendships.py
├── procesar_dataset.py
├── requirements.txt
├── README.md
└── requirements-dev.txt
```

El archivo local `sintetico_features.csv` se genera cuando hace falta y está
ignorado por Git. Los notebooks se entregan junto con el repo, pero no están
commiteados en este checkout. El CSV de resultados de la búsqueda tampoco
está disponible; los valores publicados en este README quedan como resultado
documentado.

## 🛠️ Requisitos previos

- Python 3.12 (la versión que usa el CI)
- PostgreSQL
- Git

## 🚀 Puesta en marcha

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Completá en `.env` la variable `DATABASE_URL` con tus credenciales de PostgreSQL, aplicá las migraciones y levantá la API:

```powershell
alembic upgrade head
uvicorn app.main:app --reload
```

Swagger disponible en `http://localhost:8000/docs`.

## 📊 Dataset

El dataset versionado es **sintético**:

| Archivo | Contenido |
|---|---|
| `profiles_unificado.csv` | 2000 perfiles |
| `friendships_unificado.csv` | 8000 pares (4000 amigos y 4000 no amigos) |
| `sintetico_features.csv` | Se genera localmente con `python scripts/exportar_sintetico_csv.py`; 8000 filas con 5 features y `y` |

`sintetico_features.csv` está en `.gitignore`: se genera cuando hace falta y
no se versiona.

**Features** (en este orden, que es el orden de entrada de la red):

| Feature | Qué mide |
|---|---|
| `similitud_intereses` | Jaccard entre los intereses de ambos perfiles |
| `diff_activity` | Diferencia absoluta de `activity_score` |
| `similitud_tags_posts` | Jaccard entre los hashtags de sus posts |
| `tiene_tags_posts` | 1 si ambos tienen al menos un tag de posts, 0 si no |
| `mismo_avatar_subcultura` | 1 si comparten la misma subcultura, 0 si no |

**Etiqueta**: `is_mutual` (1 = compatible, 0 = no compatible).

**Split** (`scripts/split_dataset_csv.py`): 70 / 15 / 15 estratificado por clase.

| Conjunto | Ejemplos | Positivos |
|---|---|---|
| Train | 5600 | 2800 |
| Validación | 1200 | 600 |
| Test | 1200 | 600 |

## ⚙️ Configuración de la red

**Arquitectura** (`app/ml/model.py`, clase `CompatibilityNet`): perceptrón multicapa (MLP) feedforward totalmente conectado, 18 neuronas y 89 parámetros.

| Capa | Neuronas | Detalle |
|---|---|---|
| Entrada | 5 | Las 5 features |
| Oculta 1 | 8 | Linear 5→8, ReLU, Dropout 0.3 |
| Oculta 2 | 4 | Linear 8→4, ReLU, Dropout 0.2 |
| Salida | 1 | Linear 4→1, Sigmoid |

**Hiperparámetros** (`scripts/entrenar_red.py`):

- Optimizador: Adam (learning rate 0.001, weight decay 1e-4)
- Función de pérdida: BCELoss (clasificación binaria)
- Batch size: 16
- Hasta 200 épocas, con early stopping (paciencia de 20 épocas sobre `val_loss`)
- Umbral de decisión: 0.5

## 🏋️ Entrenamiento y evaluación

El entrenamiento se puede ejecutar desde la raíz del repo:

```powershell
python scripts/split_dataset_csv.py
python scripts/entrenar_red.py
python scripts/evaluate_model.py
```

Los pesos entrenados se guardan en `models/compatibility_net.pt` (solo el
`state_dict`), y el split en `data/splits.npz`. Las features no se escalan.
Los notebooks de entrenamiento/demo se entregan junto con el repo; no están
commiteados en este checkout.

**Resultados en el conjunto de test** (1200 ejemplos, umbral 0.50):

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| no_compatible | 0.79 | 0.84 | 0.81 |
| compatible | 0.83 | 0.77 | 0.80 |

Accuracy: 0.81. Entre una corrida y otra los valores varían en centésimas.

## 🔬 Ajuste de arquitectura

Si la evaluación falla, el proceso prevé ajustar la arquitectura y reentrenar. Para comprobar si la arquitectura actual era un límite, `scripts/colab_busqueda_arquitectura.py` compara 10 configuraciones (distintos tamaños de capas y dropout) y una regresión logística de referencia, con validación cruzada de 3 folds sobre los 8000 ejemplos.

| Arquitectura | Dropout | AUC media | AUC desvío | F1 media |
|---|---|---|---|---|
| 16-8-4 | 0.0 | 0.8424 | 0.0097 | 0.8115 |
| 8 | 0.2 | 0.8421 | 0.0095 | 0.8114 |
| Regresión logística | — | 0.8421 | 0.0096 | 0.8114 |
| 8 | 0.0 | 0.8420 | 0.0093 | 0.8114 |
| 16-8 | 0.2 | 0.8416 | 0.0091 | 0.8114 |
| 32-16 | 0.2 | 0.8412 | 0.0092 | 0.8115 |
| 8-4 | 0.0 | 0.8410 | 0.0084 | 0.8114 |
| 8-4 | 0.2 | 0.8409 | 0.0083 | 0.8114 |
| 16-8 | 0.0 | 0.8405 | 0.0089 | 0.8114 |
| 16-8-4 | 0.2 | 0.8403 | 0.0098 | 0.8115 |
| 32-16 | 0.0 | 0.8399 | 0.0092 | 0.8114 |

**Conclusión:** todas las configuraciones, incluida la regresión logística, dan un AUC de alrededor de 0.84. La diferencia entre la primera y la segunda (0.0002) está muy por debajo del desvío estándar (0.0097), así que ninguna arquitectura mejora de forma medible a las otras. Se mantiene la arquitectura 5→8→4→1. La tabla anterior es el resultado documentado; el CSV de resultados no está disponible en este checkout.

## ⚠️ Limitaciones

El dataset es sintético y la etiqueta depende sobre todo de `similitud_intereses`; las otras features aportan muy poca señal. Por eso las métricas reflejan qué tan bien la red aprendió la regla del generador de datos, y no cómo se comportaría con amistades reales. Tampoco hay diferencias entre arquitecturas, porque la red usa una sola feature de manera efectiva.

## 🔄 API e interfaz de feedback

La interfaz web se sirve en `/` desde `app/static/index.html`. Permite
predecir la compatibilidad, enviar feedback **Sí/No** y, desde el panel del
equipo, reentrenar y consultar las métricas.

| Método y ruta | Función |
|---|---|
| `POST /compatibility/predict` | Calcula la compatibilidad entre dos perfiles |
| `POST /compatibility/feedback` | Guarda la respuesta real; responde `201` |
| `POST /compatibility/retrain` | Hace fine-tuning desde el modelo actual |
| `GET /compatibility/metrics` | Devuelve las métricas del último reentrenamiento |

El reentrenamiento usa la respuesta más reciente por par
`(user_id, friend_id)`, requiere al menos **5 pares distintos** y mezcla el
feedback con una muestra sintética. Se rechaza si el F1 de test del modelo
nuevo cae más de **0.02** respecto del modelo actual.

## ▶️ Cómo correr la demo

### Local

Con PostgreSQL configurado en `.env`, ejecutar desde la raíz:

```powershell
alembic upgrade head
python scripts/cargar_sinteticos_db.py
uvicorn app.main:app --reload
```

Abrir `http://localhost:8000/` (interfaz) o `http://localhost:8000/docs`
(Swagger).

### Colab

Usar el notebook de demo que se entrega junto con el repo dentro de
`notebooks/`. En este checkout la carpeta no contiene el notebook; no se ha
inventado ni agregado un archivo de reemplazo.

## 🧪 Tests y CI

```powershell
pip install -r requirements-dev.txt
flake8 app scripts tests
pytest tests/ -v
```

El workflow de GitHub Actions (`.github/workflows/ci.yml`) instala las
dependencias, corre flake8 y ejecuta pytest en pushes y pull requests a `main`
y a ramas que coincidan con `limpieza-sintetico*`.
La suite actual contiene 115 tests distribuidos en nueve módulos.

## 📌 Estado

Hecho:
- Dataset sintético, split y features
- Red neuronal entrenada y evaluada
- Búsqueda de arquitectura
- API, interfaz web, feedback y reentrenamiento con métricas
- Tests de la API, servicios y componentes
- CI en GitHub Actions

Pendiente: merge a `main` y entrega del proyecto el 04/11.