# Comandos del Proyecto

Ejecutar desde PowerShell en la raíz del repositorio, con el entorno virtual
activado.

## Entorno y dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
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

## Entrenamiento y evaluación

```powershell
python scripts/split_dataset.py
python scripts/train_neural_net.py
python scripts/evaluate_model.py
python scripts/feedback_loop.py
```

El feedback loop reentrena una vez ajustando épocas y `weight_decay`; no cambia
automáticamente la arquitectura del modelo.




## Entrenamiento en Google Colab (GPU)

### 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

### 2. Ir a la carpeta del proyecto
%cd "/content/drive/MyDrive/Colab Notebooks/RedNeuronal_Colab"

### 3. Instalar dependencias (hay que repetir esto si el runtime se reinicia)
!pip install -r requirements.txt

### 4. Verificar columnas de los CSV (por si cambia el dataset)
!head -1 profiles_unificado.csv
!head -1 friendships_unificado.csv

### 5. Armar el split train/val/test
!python scripts/split_dataset_csv.py

### 6. Entrenar la red neuronal
!python scripts/train_neural_net.py

### 7. Evaluar sobre el test set
!python scripts/evaluate_model.py

### 8. Descargar el modelo entrenado a la PC local
from google.colab import files
files.download('models/compatibility_net.pt')