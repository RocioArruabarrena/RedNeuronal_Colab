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

Crear la base una sola vez (ajustar usuario y nombre según `DATABASE_URL`):

```powershell
psql -U usuario -c "CREATE DATABASE redes_neuronales_db"
```

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

Si la base ya tiene datos, `cargar_sinteticos_db.py` no carga nada. Para
borrar `friendships` y `profiles` y recargar los sintéticos:

```powershell
python scripts/cargar_sinteticos_db.py --reset
```

La interfaz está en `http://localhost:8000/` y Swagger en
`http://localhost:8000/docs`.

Para probar los endpoints desde PowerShell (en Windows conviene usar
`127.0.0.1` en lugar de `localhost`):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/compatibility/metrics
```

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
resultados bajo `results/`, que no se versiona. Los resultados están resumidos
en el README.

`python scripts/feedback_loop.py` es un ajuste por línea de comandos,
independiente del feedback y el reentrenamiento que expone la API.

## Demo en Colab

La demo también se ejecuta en Google Colab con los notebooks de la carpeta
`notebooks/`:

- `red_friendship.ipynb`: entrenamiento del modelo.
- Notebook de demo: clona el repo, instala las dependencias, levanta
  PostgreSQL, aplica las migraciones, carga los datos sintéticos, inicia la
  API y abre la interfaz.

## CI

El workflow `.github/workflows/ci.yml` corre `flake8 app scripts tests` y
`pytest tests/ -v` en los push y pull requests a `main` y a las ramas
`limpieza-sintetico*`.