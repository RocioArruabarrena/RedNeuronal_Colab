# Predictor de Compatibilidad de Amistad

API REST para estimar la compatibilidad entre dos perfiles a partir de sus
intereses, actividad y tags de publicaciones. El modelo se sirve con FastAPI y
se entrena/evalúa mediante scripts separados.

## Stack y arquitectura

- **Modelo:** `CompatibilityNet`, una MLP implementada en PyTorch. Acepta un
  `input_dim` configurable (5 por defecto); las capas ocultas actuales tienen 8
  y 4 neuronas. La red termina en una salida sigmoide.
- **API:** FastAPI, con documentación Swagger en `/docs` y ReDoc en `/redoc`.
- **Organización:** MVC (`controllers`, `models`, `views`) con Repository para
  acceso a datos, Factory para cargar el modelo y Service Layer para lógica de
  predicción. FastAPI inyecta dependencias.
- **Persistencia:** PostgreSQL mediante SQLAlchemy ORM y migraciones Alembic.
- **Herramientas ML auxiliares:** scikit-learn se usa para particiones y
  métricas; la red neuronal y la inferencia usan PyTorch.

## Ejecución local

En Windows, desde PowerShell y la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

La configuración de base de datos y rutas se lee desde el entorno/`.env`;
consultá `.env.example`. La API requiere PostgreSQL configurado. Swagger queda
disponible en `http://localhost:8000/docs`.

## Pruebas y calidad

`pytest.ini` configura `tests/` como directorio de pruebas y agrega la raíz del
repo al `pythonpath`:

```powershell
pytest tests/ -v
flake8 app scripts tests
black app scripts tests
```

La suite cubre `CompatibilityNet`, el feedback loop, `CompatibilityService`,
`FriendshipRepository` y los endpoints de la API. Los tests del repositorio
usan SQLite en memoria; los tests del controller reemplazan el servicio por
un mock.

## Integración continua

GitHub Actions, en `.github/workflows/ci.yml`, ejecuta `flake8 app scripts tests`
y `pytest tests/ -v` con Python 3.12 en cada `push` y `pull_request` dirigidos a
`main`.

## Entrenamiento y evaluación

Desde la raíz del repo, con el entorno virtual activo:

```powershell
python scripts/split_dataset.py
python scripts/train_neural_net.py
python scripts/evaluate_model.py
```

El split se guarda en `data/splits.npz` y los pesos entrenados en
`models/compatibility_net.pt`. `scripts/feedback_loop.py` evalúa el modelo y,
si el F1 no alcanza el umbral, hace un único reentrenamiento con más épocas y
menor `weight_decay`; la arquitectura de capas ocultas no se cambia
automáticamente.