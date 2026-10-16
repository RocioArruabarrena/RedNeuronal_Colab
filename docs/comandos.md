# Comandos del Proyecto

Ejecutar desde PowerShell en la raíz del repositorio, con el entorno virtual
activado.

## Entorno y dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Pruebas y calidad de código

```powershell
# Suite completa (tests/ está configurado en pytest.ini)
pytest tests/ -v

# Lint (configuración en .flake8)
flake8 app scripts tests

# Formateo
black app scripts tests
```

## Base de datos (PostgreSQL + Alembic)

```powershell
# Generar una migración a partir de los modelos
alembic revision --autogenerate -m "descripcion_del_cambio"

# Aplicar migraciones pendientes
alembic upgrade head

# Revertir la última migración
alembic downgrade -1
```

## API

```powershell
uvicorn app.main:app --reload
```

Swagger: `http://localhost:8000/docs`. Requiere PostgreSQL configurado en el
entorno o en `.env`.

## Dataset sintético

Los archivos `profiles_unificado.csv` (2000 perfiles) y
`friendships_unificado.csv` (8000 amistades) están en la raíz del repo y se
generaron con Mockaroo; `generar_friendships.py` y `procesar_dataset.py` (en
la raíz) los procesan. Ver el docstring de cada script para su uso.

```powershell
# Genera sintetico_features.csv (8000 filas: 5 features + etiqueta y)
python scripts/exportar_sintetico_csv.py
```

## Entrenamiento y evaluación

```powershell
# 1. Arma el split 70/15/15 y lo guarda en data/splits.npz
python scripts/split_dataset_csv.py

# 2. Entrena CompatibilityNet y guarda models/compatibility_net.pt
python scripts/train_neural_net.py

# 3. Evalúa sobre el test set
python scripts/evaluate_model.py

# 4. Feedback loop: si F1 < 0.70, reentrena una vez con más épocas y menos weight_decay
python -m scripts.feedback_loop
```

El feedback loop se corre como módulo (`-m`) porque importa funciones de
`scripts.evaluate_model` y `scripts.train_neural_net`. No cambia
automáticamente la arquitectura del modelo.

## Búsqueda de arquitectura

Compara 10 configuraciones de la red y una regresión logística de referencia
con validación cruzada sobre `sintetico_features.csv`. Guarda el resultado en
`results/resultados_arquitectura_sintetico_features.csv`.

```powershell
python scripts/colab_busqueda_arquitectura.py
```

## Entrenamiento en Google Colab (GPU)

Activar la GPU en Entorno de ejecución → Cambiar tipo de entorno. Cada bloque
siguiente es una celda del notebook `notebooks/red_friendship.ipynb`.

Montar Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ir a la carpeta del proyecto en Drive:

```python
%cd "/content/drive/MyDrive/Colab Notebooks/RedNeuronal_Colab"
```

Instalar dependencias (hay que repetirlo si el runtime se reinicia):

```python
!pip install -r requirements.txt
```

Verificar las columnas de los CSV (por si cambia el dataset):

```python
!head -1 profiles_unificado.csv
!head -1 friendships_unificado.csv
```

Armar el split train/val/test:

```python
!python scripts/split_dataset_csv.py
```

Entrenar la red neuronal:

```python
!python scripts/train_neural_net.py
```

Evaluar sobre el test set:

```python
!python scripts/evaluate_model.py
```

Búsqueda de arquitectura (resultados en `results/`):

```python
%run scripts/colab_busqueda_arquitectura.py
```

Descargar el modelo y el split a la PC local, para pisar
`models/compatibility_net.pt` y `data/splits.npz`:

```python
from google.colab import files
files.download('models/compatibility_net.pt')
files.download('data/splits.npz')
```

La carpeta del repo en Drive es una copia manual: cada vez que cambia un script
en la PC hay que subirlo a Drive (reemplazando el anterior) antes de correr
Colab.