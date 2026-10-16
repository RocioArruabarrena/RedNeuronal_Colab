# Habilidades del Agente (Skill Set)

Define las capacidades específicas que el agente sabe usar dentro de este
entorno.

## 🧰 Habilidades principales

### 1. Dataset Preparation (`skill-data-prep`)
- **Descripción:** Prepara el dataset sintético y arma el split train/val/test
  (70/15/15, estratificado) a partir de `profiles_unificado.csv` y
  `friendships_unificado.csv`. Las 5 features se calculan con
  `scripts/features.py`.
- **Uso:** `python scripts/exportar_sintetico_csv.py` (genera
  `sintetico_features.csv`) y `python scripts/split_dataset_csv.py` (genera
  `data/splits.npz`).

### 2. Model Training (`skill-model-train`)
- **Descripción:** Entrena `CompatibilityNet` con Adam, BCELoss y early
  stopping, y guarda los pesos en `models/compatibility_net.pt`.
- **Uso:** `python scripts/train_neural_net.py`. En Google Colab (GPU), con los
  mismos comandos desde la raíz del repo en Drive.

### 3. Model Evaluation (`skill-model-eval`)
- **Descripción:** Evalúa el modelo entrenado contra el set de test
  (precision, recall, F1 y matriz de confusión) y aplica el feedback loop si
  el F1 no alcanza el mínimo (0.70).
- **Uso:** `python scripts/evaluate_model.py` y
  `python -m scripts.feedback_loop`.

### 4. Architecture Search (`skill-arch-search`)
- **Descripción:** Compara varias arquitecturas de la red (tamaño de capas y
  dropout) y una regresión logística de referencia, con validación cruzada.
  Sirve para decidir si hay que ajustar la arquitectura cuando la evaluación
  falla.
- **Uso:** `python scripts/colab_busqueda_arquitectura.py`. Los resultados
  quedan en `results/resultados_arquitectura_sintetico_features.csv`.

### 5. API Management (`skill-api-manage`)
- **Descripción:** Levanta, prueba y documenta los endpoints de la API
  (FastAPI), verificando que el Swagger en `/docs` esté consistente con los
  schemas de `app/views/`.
- **Uso:** `uvicorn app.main:app --reload` y pruebas manuales o automatizadas
  contra `/docs`.

### 6. Database Migrations (`skill-db-migrate`)
- **Descripción:** Gestiona los cambios de esquema en PostgreSQL usando
  Alembic, sin romper datos existentes.
- **Uso:** `alembic revision --autogenerate -m "..."` seguido de
  `alembic upgrade head`.

### 7. Automatic Documentation (`skill-auto-doc`)
- **Descripción:** Actualiza `memoria.md` con las métricas finales de cada
  corrida y registra cambios en `comandos.md`.
- **Uso:** El agente ejecuta esta habilidad al finalizar exitosamente un
  pipeline o una migración.