# Comandos de Ejecución Autónoma

Este archivo almacena la memoria de comandos que el Agente IA debe emplear para gestionar el proyecto (adaptado a PowerShell/Windows).

## 🐍 Entorno y Dependencias

```powershell
# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

## 🗄️ Base de Datos (PostgreSQL + Alembic)

```powershell
# Generar una migración a partir de los modelos
alembic revision --autogenerate -m "descripcion_del_cambio"

# Aplicar migraciones pendientes
alembic upgrade head

# Revertir la última migración
alembic downgrade -1
```

## 🚀 Levantar la API

```powershell
# Levantar el servidor en modo desarrollo (con reload automático)
uvicorn app.main:app --reload

# Swagger disponible en http://localhost:8000/docs
# Redoc disponible en http://localhost:8000/redoc
```

## 🧠 Entrenamiento y Evaluación del Modelo

```powershell
# Ejecutar entrenamiento del modelo principal
python app\ml\train.py

# Ejecutar evaluación del modelo
python app\ml\evaluate.py --model_path models\best_model.pkl
```

## 🧪 Pruebas (entorno de prueba, no el IDE)

```powershell
# Tests unitarios
pytest tests\unit

# Tests de integración (contra la API/DB)
pytest tests\integration
```

## 🧹 Calidad de Código

```powershell
black app\
flake8 app\
```




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