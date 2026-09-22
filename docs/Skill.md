Define las capacidades específicas que el agente sabe usar dentro de este entorno.

# Habilidades del Agente (Skill Set)

## 🧰 Habilidades Principales

### 1. Data Preprocessing (`skill-data-prep`)
- **Descripción:** Carga y normaliza los datos de `friendships`/perfiles desde PostgreSQL hacia `data/`.
- **Uso:** Modificar los parámetros en `app/ml/data_loader.py` y validar distribuciones.

### 2. Model Training & Tuning (`skill-model-train`)
- **Descripción:** Entrena el modelo de compatibilidad y optimiza hiperparámetros (GridSearchCV).
- **Uso:** Invocar mediante `python app\ml\train.py`.

### 3. Model Evaluation (`skill-model-eval`)
- **Descripción:** Evalúa el modelo entrenado contra el set de test (precision, recall, F1) y aplica el feedback loop si no pasa el umbral.
- **Uso:** Invocar mediante `python app\ml\evaluate.py --model_path models\best_model.pkl`.

### 4. API Management (`skill-api-manage`)
- **Descripción:** Levanta, prueba y documenta los endpoints de la API (FastAPI), verificando que el Swagger en `/docs` esté consistente con los schemas de `app/views/`.
- **Uso:** `uvicorn app.main:app --reload` + pruebas manuales/automatizadas contra `/docs`.

### 5. Database Migrations (`skill-db-migrate`)
- **Descripción:** Gestiona los cambios de esquema en PostgreSQL usando Alembic, sin romper datos existentes.
- **Uso:** `alembic revision --autogenerate -m "..."` seguido de `alembic upgrade head`.

### 6. Automatic Documentation (`skill-auto-doc`)
- **Descripción:** Actualiza `memoria.md` con las métricas finales de cada corrida y registra cambios en `comandos.md`.
- **Uso:** El agente ejecuta esta habilidad al finalizar exitosamente un pipeline o migración.