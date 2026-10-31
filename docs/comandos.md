# Comandos del Proyecto

Ejecutar los comandos desde PowerShell en la raíz del repositorio, con el
entorno virtual activado.

## Entorno y dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configurar `DATABASE_URL` en `.env` para PostgreSQL.

## Pruebas y calidad de código

```powershell
pytest tests/ -v
flake8 app scripts tests
black app scripts tests
```

## Base de datos (PostgreSQL + Alembic)

```powershell
alembic revision --autogenerate -m "descripcion_del_cambio"
alembic upgrade head
alembic downgrade -1
```

## Demo local

```powershell
alembic upgrade head
python scripts/cargar_sinteticos_db.py
uvicorn app.main:app --reload
```

La interfaz está en `http://localhost:8000/` y Swagger en
`http://localhost:8000/docs`.

## Dataset, entrenamiento y evaluación

Los archivos `profiles_unificado.csv` (2000 perfiles) y
`friendships_unificado.csv` (8000 pares) son los datos de entrada versionados.
Para generar el CSV con features:

```powershell
python scripts/exportar_sintetico_csv.py
```

El comando genera `sintetico_features.csv` en la raíz (8000 filas, cinco
features y etiqueta `y`). Ese archivo está en `.gitignore` y no se versiona.

```powershell
python scripts/split_dataset_csv.py
python scripts/entrenar_red.py
python scripts/evaluate_model.py
python scripts/colab_busqueda_arquitectura.py
```

El split queda en `data/splits.npz` y los pesos en
`models/compatibility_net.pt`. La búsqueda de arquitectura genera su CSV de
resultados bajo `results/`; el CSV no está incluido en el checkout actual.
`python scripts/feedback_loop.py` es un ajuste CLI independiente del feedback
y reentrenamiento expuestos por la API.

## Demo en Colab

La demo también se ejecuta en Google Colab con el notebook entregado junto con
el repo. Los notebooks no están commiteados en este checkout.

El CI (`.github/workflows/ci.yml`) corre `flake8 app scripts tests` y
`pytest tests/ -v` en pushes y pull requests a `main` y ramas
`limpieza-sintetico*`.
