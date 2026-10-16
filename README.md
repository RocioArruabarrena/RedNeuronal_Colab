# Predictor de Compatibilidad de Amistad — Red Neuronal

Proyecto de Machine Learning que predice la compatibilidad de amistad entre dos usuarios de una red social tipo fotolog ("Yo, Adolescente"). Una red neuronal (PyTorch) recibe 5 números sobre un par de usuarios y devuelve un score entre 0 y 1: si es mayor o igual a 0.5, el par se considera compatible.

El modelo se entrena **solo con un dataset sintético** (8000 pares) y se expone mediante una API REST documentada con Swagger.

## 🎯 Objetivos del proyecto

1. Estandarizar el entorno para crear, entrenar y evaluar un modelo de red neuronal.
2. Seguir el flujo **dataset → configuración → entrenamiento → interfaz**.
3. Exponer el modelo entrenado como API REST (FastAPI + Swagger).
4. Aplicar buenas prácticas: arquitectura MVC, patrones de diseño, tests y entorno de prueba autónomo (CI).
5. Dejar documentación para que un agente de IA pueda trabajar en el proyecto (`docs/AGENT.md`, `docs/memoria.md`).

## 🏗️ Arquitectura

- **Patrón**: MVC
- **API**: FastAPI (Swagger automático en `/docs`)
- **Base de datos**: PostgreSQL (SQLAlchemy ORM + Alembic para migraciones)
- **Red neuronal**: PyTorch
- **Patrones de diseño**:
  - **Repository** — aísla el acceso a datos (`app/repositories/`)
  - **Factory** — centraliza la carga del modelo entrenado (`app/ml/model_factory.py`)
  - **Service Layer** — separa la lógica de negocio de los controllers (`app/services/`)
  - **Dependency Injection** — nativo de FastAPI, para inyectar sesión de DB y servicios

## 📁 Estructura del proyecto

```text
.
├── app/
│   ├── main.py                  # entrypoint FastAPI
│   ├── config.py                # configuración (variables de entorno)
│   ├── database.py              # conexión a PostgreSQL
│   ├── controllers/             # C — endpoints
│   ├── models/                  # M — entidades ORM (Profile, Friendship)
│   ├── views/                   # V — schemas Pydantic
│   ├── services/                # lógica de negocio
│   ├── repositories/            # patrón Repository
│   └── ml/                      # CompatibilityNet y ModelFactory
├── alembic/                     # migraciones de base de datos
├── data/
│   └── splits.npz               # split train/val/test
├── docs/                        # AGENT.md, memoria.md, comandos.md, etc.
├── models/
│   └── compatibility_net.pt     # pesos entrenados (state_dict)
├── results/                     # resultados de la búsqueda de arquitectura
├── scripts/
│   ├── features.py              # cálculo de las 5 features
│   ├── split_dataset_csv.py     # arma el split desde los CSV sintéticos
│   ├── train_neural_net.py      # entrenamiento
│   ├── evaluate_model.py        # evaluación
│   ├── colab_busqueda_arquitectura.py
│   ├── feedback_loop.py
│   └── exportar_sintetico_csv.py
├── tests/                       # tests unitarios e integración
├── notebooks/                   # notebook de Colab (red_friendship.ipynb)
├── generar_friendships.py       # genera friendships sintéticas
├── procesar_dataset.py          # procesa el dataset sintético
├── profiles_unificado.csv       # 2000 perfiles sintéticos
├── friendships_unificado.csv    # 8000 amistades sintéticas
├── sintetico_features.csv       # 8000 filas: 5 features + etiqueta
├── requirements.txt
└── requirements-dev.txt
```

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

El dataset es **sintético**, generado con Mockaroo y procesado con los scripts del repo:

| Archivo | Contenido |
|---|---|
| `profiles_unificado.csv` | 2000 perfiles |
| `friendships_unificado.csv` | 8000 pares (4000 amigos y 4000 no amigos) |
| `sintetico_features.csv` | 8000 filas con las 5 features y la etiqueta `y` |

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

**Hiperparámetros** (`scripts/train_neural_net.py`):

- Optimizador: Adam (learning rate 0.001, weight decay 1e-4)
- Función de pérdida: BCELoss (clasificación binaria)
- Batch size: 16
- Hasta 200 épocas, con early stopping (paciencia de 20 épocas sobre `val_loss`)
- Umbral de decisión: 0.5

## 🏋️ Entrenamiento y evaluación

El entrenamiento se ejecuta en **Google Colab** (GPU), con el notebook `notebooks/red_friendship.ipynb`, que corre los mismos scripts del repo desde la raíz del proyecto:

```powershell
python scripts/split_dataset_csv.py
python scripts/train_neural_net.py
python scripts/evaluate_model.py
```

Los pesos entrenados se guardan en `models/compatibility_net.pt` (solo el `state_dict`), y el split en `data/splits.npz`. Las features no se escalan, así que la API carga el modelo sin ninguna transformación adicional.

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

**Conclusión:** todas las configuraciones, incluida la regresión logística, dan un AUC de alrededor de 0.84. La diferencia entre la primera y la segunda (0.0002) está muy por debajo del desvío estándar (0.0097), así que ninguna arquitectura mejora de forma medible a las otras. Se mantiene la arquitectura 5→8→4→1. Los resultados completos están en `results/resultados_arquitectura_sintetico_features.csv`.

## ⚠️ Limitaciones

El dataset es sintético y la etiqueta depende sobre todo de `similitud_intereses`; las otras features aportan muy poca señal. Por eso las métricas reflejan qué tan bien la red aprendió la regla del generador de datos, y no cómo se comportaría con amistades reales. Tampoco hay diferencias entre arquitecturas, porque la red usa una sola feature de manera efectiva.

## 🔄 Feedback loop

`scripts/feedback_loop.py` permite reentrenar ajustando hiperparámetros cuando la evaluación no alcanza. El flujo completo (predecir, confirmar el valor real, reentrenar y ver métricas antes y después) todavía no está disponible como interfaz.

## 🧪 Tests y CI

```powershell
pip install -r requirements-dev.txt
flake8 app scripts tests
pytest tests/ -v
```

El workflow de GitHub Actions (`.github/workflows/ci.yml`) instala las dependencias, corre flake8 y ejecuta pytest en cada push, y funciona como entorno de prueba autónomo.

## 📌 Estado

Hecho:
- Dataset sintético, split y features
- Red neuronal entrenada y evaluada
- Búsqueda de arquitectura
- API con base de datos, repository, service layer y tests
- CI en GitHub Actions

Pendiente:
- Interfaz para predecir, confirmar el valor real, reentrenar y ver métricas
- Feedback loop integrado a la API
- Actualizar el resto de la documentación (`docs/`)