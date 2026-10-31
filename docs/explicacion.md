# Explicación completa del proyecto — Red-Neuronal

Predictor de compatibilidad de amistad para **Yo Adolescente**. Dado un par de
perfiles, la API devuelve un score de compatibilidad (0 a 1) calculado por una
red neuronal entrenada con un **dataset sintético** de 8000 pares.

Stack: **FastAPI** (API) + **PostgreSQL/SQLAlchemy/Alembic** (persistencia) +
**PyTorch** (red neuronal) + **scikit-learn** (split de datos y métricas).

Patrones y capas verificadas en el código: **Repository** (contratos e
implementación de acceso a datos), **Factory** (carga y caché del modelo),
**Service Layer** (casos de uso) y DI con `Depends` de FastAPI. MVC está
adaptado a una API REST: `app/views/` son schemas Pydantic y la interfaz es un
HTML estático. SOLID son principios, no un patrón ni una función nativa de
FastAPI.

---

## 1. Estructura general

```
Red-Neuronal/
├── app/                    → API, modelo, servicios y UI estática
├── scripts/                → scripts versionados de datos, entrenamiento y evaluación
├── alembic/                → configuración y tres migraciones
├── data/splits.npz
├── models/compatibility_net.pt
├── docs/                   → AGENT.md, Skill.md, comandos.md, explicacion.md, memoria.md
├── tests/                  → nueve módulos de tests
├── profiles_unificado.csv
├── friendships_unificado.csv
├── generar_friendships.py
├── procesar_dataset.py
├── requirements.txt
└── requirements-dev.txt
```

La idea de fondo: **`app/`** es el servicio en producción (lo que corre
`uvicorn`), y **`scripts/`** son herramientas que corrés a mano para preparar
datos y entrenar el modelo que `app/` después consume. Son dos mundos que se
tocan en un solo punto: el archivo `models/compatibility_net.pt` que
`scripts/entrenar_red.py` escribe y `app/ml/fabrica_modelo.py` lee.

`notebooks/` y `results/` no tienen archivos en este checkout. Los notebooks
se entregan junto con el repo, pero no están commiteados. El CSV generado de
features tampoco se versiona: `scripts/exportar_sintetico_csv.py` lo crea en
la raíz y `.gitignore` lo excluye.

---

## 2. `app/` — la API

### 2.1 `app/main.py`

```python
from fastapi import FastAPI
from app.config import settings
from app.controllers import compatibility_controller

app = FastAPI(title=settings.api_title, version=settings.api_version)
app.include_router(compatibility_controller.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

- `FastAPI(...)`: crea la aplicación. Le pasa `title` y `version` desde
  `settings` (no hardcodeados) para que Swagger (`/docs`) los muestre.
- `app.include_router(...)`: registra todas las rutas definidas en
  `compatibility_controller.py` bajo esta app. Así `main.py` no sabe nada de
  los endpoints en sí — solo los "enchufa".
- `/health`: endpoint mínimo para chequear que el servidor está vivo (útil
  para monitoreo o para probar que `uvicorn` levantó bien).
- Swagger queda expuesto automáticamente en `/docs` (interactivo) y `/redoc`
  (solo lectura) — FastAPI lo genera solo a partir de los `response_model` y
  tipos de Pydantic que ya tenés en el controller y en `views/`.

### 2.2 `app/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql://usuario:password@localhost:5432/redes_neuronales_db"
    model_path: str = "models/compatibility_net.pt"
    api_title: str = "Predictor de Compatibilidad de Amistad API"
    api_version: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

- `BaseSettings` de Pydantic: permite definir configuración con valores por
  default que se pueden **sobreescribir con variables de entorno** o un
  archivo `.env` sin tocar código. Ej: si existe una variable de entorno
  `DATABASE_URL`, pisa el valor default de `database_url`.
- `model_path`: acá es donde se define **qué archivo de modelo se carga** —
  hoy apunta a `compatibility_net.pt` (la red neuronal).
- `settings = Settings()`: se instancia **una sola vez** al importar el
  módulo, y todo el resto del proyecto importa este mismo objeto (`from
  app.config import settings`) — patrón singleton implícito de Python (los
  módulos se cachean).

### 2.3 `app/database.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- `engine`: la conexión real a PostgreSQL, construida a partir de la URL en
  `settings`.
- `SessionLocal`: fábrica de sesiones de SQLAlchemy. `autocommit=False` y
  `autoflush=False` significan que **vos controlás explícitamente** cuándo se
  escribe a la base (con `.commit()`), nada se guarda solo.
- `Base`: la clase base de la que heredan `Profile` y `Friendship` — es lo que
  le permite a SQLAlchemy saber qué clases mapean a qué tablas.
- `get_db()`: es un **generador** usado como dependencia de FastAPI
  (`Depends(get_db)`). Abre una sesión, la entrega (`yield`), y pase lo que
  pase (incluso si el endpoint tira una excepción) la cierra en el `finally`.
  Esto evita conexiones colgadas.

### 2.4 `app/controllers/compatibility_controller.py`

```python
router = APIRouter(prefix="/compatibility", tags=["Compatibility"])

def get_service(db: Session = Depends(get_db)) -> CompatibilityService:
    repository = FriendshipRepository(db)
    return CompatibilityService(repository)

@router.post("/predict", response_model=CompatibilityResponse)
def predict_compatibility(
    request: CompatibilityRequest,
    service: CompatibilityService = Depends(get_service),
):
    try:
        return service.predict(request.user_id, request.friend_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- `APIRouter(prefix="/compatibility", ...)`: agrupa rutas bajo el prefijo
  `/compatibility`, así el endpoint real queda en `POST /compatibility/predict`.
- `get_service(...)`: es una **cadena de dependencias**. FastAPI resuelve
  `Depends(get_db)` primero y el proveedor arma `FriendshipRepository` y
  `CompatibilityService`. Los proveedores de reentrenamiento y métricas
  están en el mismo módulo. La creación del repositorio concreto ocurre en
  esos proveedores, que son el punto de composición.
- El controller **no tiene lógica de negocio**: solo recibe el request
  (validado por Pydantic vía `CompatibilityRequest`), le pasa los IDs al
  service, y traduce un `ValueError` (perfil no encontrado) a un HTTP 404.
  Los endpoints de feedback/retrain/metrics también traducen errores de
  dominio a respuestas HTTP (404, 409 o 500). La lógica de cálculo y
  reentrenamiento permanece en los services.

Rutas del controller:

| Método y ruta | Caso de uso |
|---|---|
| `POST /compatibility/predict` | Predicción; 404 si falta un perfil |
| `POST /compatibility/feedback` | Guarda feedback; responde 201 |
| `POST /compatibility/retrain` | Fine-tuning; 409 si no hay feedback suficiente o se rechaza el modelo |
| `GET /compatibility/metrics` | Lee las métricas; 404 si todavía no existen |

### 2.5 `app/views/compatibility_schema.py`

```python
class CompatibilityRequest(BaseModel):
    user_id: int = Field(..., examples=[1])
    friend_id: int = Field(..., examples=[2])

class CompatibilityResponse(BaseModel):
    user_id: int
    friend_id: int
    compatibility_score: float
    is_compatible: bool
```

- Son los esquemas de **entrada y salida** de la API, usando Pydantic.
  FastAPI los usa para: validar el JSON que llega (si falta `user_id` o no es
  `int`, rechaza el request con un 422 automático), y para generar la
  documentación de Swagger.
- `Field(..., examples=[1])`: el `...` marca el campo como obligatorio; el
  `examples` es solo para que Swagger muestre un valor de ejemplo en el
  `/docs`.
- Notar que esta carpeta se llama `views/` pero en realidad son **schemas de
  API**, no vistas HTML — es la convención MVC adaptada a una API REST (la
  "vista" es la forma en la que se serializa la respuesta).

### 2.6 `app/repositories/contratos.py` y `app/repositories/repositorio_amistades.py`

```python
class FriendshipRepository(RepositorioCompatibilidad, RepositorioAmistades):
    def __init__(self, db: Session):
        self.db = db

    def get_profile(self, profile_id: int) -> Profile | None:
        return self.db.query(Profile).filter(Profile.id == profile_id).first()
```

- `contratos.py` declara `RepositorioPerfiles`, `RepositorioFeedback`,
  `RepositorioAmistades` y el contrato combinado `RepositorioCompatibilidad`.
  `repositorio_amistades.py` implementa estos métodos con SQLAlchemy.
- `CompatibilityService` depende de `RepositorioCompatibilidad` y
  `RetrainService` de `RepositorioFeedback`, no de `FriendshipRepository`.
  El proveedor del controller elige la implementación concreta.
- El repositorio guarda feedback y amistades/predicciones. La ruta de
  predicción actual no persiste el score; `CompatibilityService.predict` es
  de solo lectura.

### 2.7 `app/services/compatibility_service.py` — el corazón de la predicción

Funciones sueltas (a nivel de módulo, no de clase) que calculan las 5
features, **idénticas** a las de `scripts/features.py` (esto es crítico: si el
cálculo de features en producción difiere del usado en entrenamiento, el
modelo predice basura):

```python
def _extraer_subcultura(intereses): ...   # busca un tag "subcultura:X" en la lista
def _jaccard(set_a, set_b): ...           # |intersección| / |unión|
def _calcular_features(perfil_a, perfil_b): ...  # arma las 5 features en orden
```

Ojo: esta lógica existe en **dos lugares** (`scripts/features.py` para el
entrenamiento y este service para la inferencia). Si se cambia una, hay que
cambiar la otra.

- `_jaccard`: mide similitud entre dos conjuntos. Si `A={"emo","musica"}` y
  `B={"emo","arte"}`, la intersección es `{"emo"}` (1) y la unión es
  `{"emo","musica","arte"}` (3) → Jaccard = 1/3 ≈ 0.33.
- `_calcular_features` devuelve una lista de 5 números en **este orden
  exacto**: `similitud_intereses`, `diff_activity`, `similitud_tags_posts`,
  `tiene_tags_posts`, `mismo_avatar_subcultura`. El orden importa porque la
  red neuronal no sabe "nombres" de features, solo posiciones en el vector de
  entrada.

```python
class CompatibilityService:
    def __init__(self, repository: RepositorioCompatibilidad, model=None):
        self.repository = repository
        self.model = (
            model if model is not None else ModelFactory.get_model(settings.model_path)
        )

    def predict(self, user_id: int, friend_id: int) -> dict:
        ...
```

- El repositorio y el calculador de features se reciben por constructor; el
  modelo también se puede inyectar para pruebas. Por defecto, el service pide
  el modelo al Factory, que lo cachea en memoria.
- `torch.tensor([features], ...)`: el `[features]` (lista dentro de lista) es
  porque PyTorch espera un **batch**, aunque sea de un solo ejemplo. La forma
  final es `(1, 5)` → 1 fila, 5 columnas (las 5 features).
- `torch.no_grad()`: le dice a PyTorch que no calcule gradientes acá — estamos
  prediciendo, no entrenando, así que ahorra memoria y cómputo.
- `score >= 0.5`: el umbral de decisión, el mismo que usa `evaluate_model.py`.
- Las features **no se escalan** antes de entrar a la red, igual que en el
  entrenamiento. Si algún día se entrena con un `StandardScaler`, hay que
  aplicarlo también acá.
- La predicción es de solo lectura: el feedback se registra aparte mediante
  `save_feedback(...)` cuando el usuario confirma la etiqueta real.

### 2.8 `app/ml/model.py` — la arquitectura de la red

```python
class CompatibilityNet(nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(8, 4), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(4, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
```

- `Linear(5, 8)`: capa totalmente conectada, 5 entradas (las features) → 8
  neuronas. Cada una aprende una combinación lineal distinta de las 5
  features.
- `ReLU()`: función de activación no lineal (`max(0, x)`) — sin esto, apilar
  capas `Linear` sería matemáticamente equivalente a una sola capa lineal
  (no ganarías nada por tener varias capas).
- `Dropout(0.3)`: durante entrenamiento, apaga aleatoriamente el 30% de las
  neuronas de esa capa en cada pasada. Es una técnica anti-overfitting —
  fuerza a la red a no depender de neuronas puntuales. En modo `eval()` (o
  sea, en predicción) el Dropout no hace nada, se usan todas las neuronas.
- `Linear(8, 4) + ReLU + Dropout(0.2)`: segunda capa oculta, reduce a 4
  neuronas, con menos dropout (la red ya está más "comprimida" acá, apagar
  demasiado sería perder mucha información).
- `Linear(4, 1) + Sigmoid()`: capa de salida. Un solo número, y `Sigmoid` lo
  aplasta al rango (0, 1) — interpretable como una probabilidad de
  compatibilidad.
- `forward(self, x)`: define cómo fluyen los datos por la red. `nn.Sequential`
  ya encadena las capas, así que acá solo se llama una vez.
- En total son 18 neuronas y 89 parámetros: `(5×8+8) + (8×4+4) + (4×1+1) =
  48 + 36 + 5 = 89`. Es una red muy chica, adecuada para datos tabulares con
  5 features.

### 2.9 `app/ml/fabrica_modelo.py`

```python
class ModelFactory:
    @classmethod
    def get_model(cls, model_path: str, arquitectura: str = "compatibility_net"):
        ...
```

- `ModelFactory` registra arquitecturas, instancia el tipo solicitado y carga
  su `state_dict`. La caché se indexa por ruta y arquitectura; `reload(...)`
  invalida esa entrada después de guardar un modelo reentrenado.
- Es una Factory real: concentra la selección/instanciación y carga de
  arquitecturas. La caché evita releer el archivo en cada request.
- `state_dict = torch.load(..., weights_only=True)`: carga solo los **pesos**
  (números entrenados) del archivo, no código arbitrario — más seguro que
  cargar un pickle completo.
- `.load_state_dict(state_dict)`: mete esos pesos en una instancia nueva de
  `CompatibilityNet` (la arquitectura tiene que coincidir exactamente con la
  que se usó al entrenar, o esto falla).
- `.eval()`: pone el modelo en modo evaluación (desactiva Dropout, como se
  explicó arriba) — clave para que las predicciones sean consistentes.

### 2.10 `app/models/` — tablas de la base de datos

**`profile.py`**:
```python
class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    interests = Column(JSON, nullable=True)       # tags de perfil + subcultura
    activity_score = Column(Integer, default=0)
    tags_agregados = Column(JSON, nullable=True)  # hashtags agregados de posts públicos
```

**`friendship.py`**:
```python
class Friendship(Base):
    __tablename__ = "friendships"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    is_mutual = Column(Boolean, default=False)
    compatibility_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

- Estas dos clases son el mapeo objeto-relacional (ORM): cada instancia de
  `Profile` es una fila de la tabla `profiles`.
- `interests` vs `tags_agregados`: se separaron a propósito en dos columnas
  (ver migración en la sección 3) para poder calcular `similitud_intereses`
  (sobre `interests`) y `similitud_tags_posts` (sobre `tags_agregados`) como
  **dos features independientes** — si se mezclaran, se perdería esa
  distinción.
- `is_mutual`: `True` cuando la fila representa una amistad confirmada y
  `False` para pares negativos. La predicción de la API no crea filas en
  `friendships`; las respuestas reales se guardan en la tabla `feedback`.
- `compatibility_score`: `nullable=True` porque las filas de entrenamiento no
  tienen score. `Feedback` almacena las cinco features, el score predicho y la
  etiqueta real enviada por el usuario.

### 2.11 Interfaz, feedback y reentrenamiento

`app/static/index.html` se sirve desde `/` en `app/main.py`. La interfaz deja
ingresar los IDs de dos perfiles, ejecutar una predicción, responder **Sí/No**
y abrir un panel de equipo con acciones para reentrenar y consultar métricas.

`app/services/servicio_reentrenamiento.py` coordina el fine-tuning desde el
modelo actual. `DatasetReentrenamiento` conserva la última respuesta por
`(user_id, friend_id)`, se exigen al menos cinco pares distintos y se añade el
feedback al entrenamiento junto con una muestra sintética. `NoEmpeorarF1`
rechaza el modelo nuevo si el F1 de test cae más de 0.02 respecto del actual.
`app/services/metrics_service.py` lee las métricas guardadas en
`models/ultimas_metricas.json`.

---

## 3. `alembic/` — migraciones de base de datos

Alembic versiona los cambios de esquema de la base, para poder aplicarlos
(`upgrade`) o revertirlos (`downgrade`) de forma ordenada.

### 3.1 Migración `crear_profiles_y_friendships`
Primera migración: crea las tablas `profiles` y `friendships` desde cero, con
las columnas originales (sin `tags_agregados` todavía). `downgrade()` hace lo
inverso: borra los índices y las tablas, en el orden correcto (primero
`friendships`, que depende de `profiles` via FK, después `profiles`).

### 3.2 Migración `agregar_tags`
```python
def upgrade():
    op.add_column('profiles', sa.Column('tags_agregados', sa.JSON(), nullable=True))

def downgrade():
    op.drop_column('profiles', 'tags_agregados')
```
Segunda migración: agrega la columna `tags_agregados` (para separar hashtags
de posts de los intereses de perfil, como se explicó en 2.10). Su
`down_revision` apunta a la migración anterior, que es cómo Alembic sabe el
**orden**: esta migración solo se puede aplicar después de la que crea las
tablas.

---

## 4. Dataset sintético y `scripts/`

### 4.1 El dataset sintético

El proyecto se entrena **solo con datos sintéticos**:

- `profiles_unificado.csv`: 2000 perfiles, generados con Mockaroo.
- `friendships_unificado.csv`: 8000 pares (4000 amigos y 4000 no amigos).
- `procesar_dataset.py` genera `profiles_final.csv` desde `MOCK_DATA.csv`;
  `generar_friendships.py` consume ese archivo y produce
  `friendships_final.csv`. Son auxiliares del flujo legado, no los scripts que
  preparan los CSV unificados usados por la aplicación actual.
- `scripts/exportar_sintetico_csv.py` genera `sintetico_features.csv`: 8000
  filas con las 5 features y la etiqueta `y`.

Una limitación importante: la etiqueta sintética depende sobre todo de
`similitud_intereses` (AUC ~0.84 por sí sola); las demás features aportan muy
poca señal. Por eso las métricas miden qué tan bien la red aprendió la regla
del generador, no cómo se comportaría con amistades reales.

### 4.2 `features.py` — las 5 features

Contiene `FEATURE_NAMES` (el orden de entrada de la red) y las funciones
`extraer_subcultura`, `jaccard` y `calcular_features(perfil_a, perfil_b)`.
Funciona con cualquier objeto que tenga los atributos `interests`,
`activity_score` y `tags_agregados`, así que sirve tanto para filas de la base
como para perfiles leídos de un CSV.

Las 5 features:

- `similitud_intereses`: Jaccard entre los intereses de ambos perfiles
  (incluye el tag `subcultura:X`).
- `diff_activity`: diferencia absoluta de `activity_score` (va de 0 a 100,
  mientras el resto va de 0 a 1).
- `similitud_tags_posts`: Jaccard entre los hashtags de sus posts.
- `tiene_tags_posts`: 1 si ambos tienen al menos un tag de posts, 0 si no.
- `mismo_avatar_subcultura`: 1 si comparten el mismo tag `subcultura:X`.

### 4.3 `split_dataset_csv.py` — arma el split fijo

- `cargar_perfiles(path)`: lee `profiles_unificado.csv` y arma un dict
  `id → perfil` con los atributos que espera `calcular_features`
  (`interests`, `activity_score`, `tags_agregados`). A `interests` le suma un
  tag `subcultura:<pixel_avatar>` para que `mismo_avatar_subcultura` funcione.
- `cargar_features_y_labels(perfiles, path)`: recorre
  `friendships_unificado.csv`, calcula las 5 features de cada par y toma
  `is_mutual` como etiqueta (1 o 0).
- `split_dataset(...)`: dos llamadas a `train_test_split` en cadena para
  lograr 70/15/15: primero separa 70% train / 30% resto, después parte ese
  30% a la mitad (15%/15%) para val y test. Ambas usan `stratify=y`, para no
  desbalancear ninguna de las tres partes. Resultado: 5600 / 1200 / 1200.
- Guarda todo en `data/splits.npz` con `np.savez(...)` — un solo archivo
  binario con los 6 arrays (`X_train, y_train, X_val, y_val, X_test, y_test`)
  más los nombres de features, para que el entrenamiento y la evaluación
  siempre partan del mismo split (reproducibilidad).

### 4.4 `app/ml/model.py` y `fabrica_modelo.py`
Ya explicados en 2.8 y 2.9 — viven en `app/` porque son parte de la API, no
scripts sueltos.

### 4.5 `entrenar_red.py` — entrena `CompatibilityNet`

- `cargar_split()`: lee los arrays de `data/splits.npz`.
- `main()`: delega el entrenamiento a `Entrenador` en
  `app/ml/entrenador.py`, guarda el `state_dict` en
  `models/compatibility_net.pt` y reporta la mejor pérdida de validación.
- `Entrenador` contiene el loop de entrenamiento, validación, Adam, BCELoss y
  early stopping. También admite `modelo_inicial` para hacer fine-tuning; el
  servicio de la API utiliza esa capacidad para partir del modelo actual.

### 4.6 `evaluate_model.py` — mide qué tan bien predice el modelo

- `cargar_datos_test`, `cargar_modelo`: funciones reusables (`feedback_loop.py`
  las importa directo).
- `evaluar(modelo, X_test, y_test, umbral)`: corre el modelo sobre el test set
  (`torch.no_grad()` porque es solo inferencia), aplica el umbral para pasar
  de probabilidad a clase (0/1), y calcula precision/recall/F1 con
  `precision_recall_fscore_support(..., average="binary")` — "binary" porque
  hay 2 clases y nos interesa la métrica sobre la clase positiva
  ("compatible").
- `main()`: prueba 5 umbrales distintos (para ver dónde está el mejor
  trade-off precision/recall) y después imprime el reporte completo
  (`classification_report`, que da precision/recall/F1 de **ambas** clases)
  con el umbral final elegido (0.50).

Resultado vigente sobre el test sintético (1200 ejemplos): accuracy 0.81;
compatible con precision 0.83, recall 0.77 y F1 0.80; no_compatible con
precision 0.79, recall 0.84 y F1 0.81.

### 4.7 `colab_busqueda_arquitectura.py` — compara arquitecturas

- Lee `sintetico_features.csv` (5 features + etiqueta `y`) y valida que no
  haya valores vacíos ni etiquetas distintas de 0/1.
- Define una versión configurable de `CompatibilityNet` (tamaños de capas y
  dropout) y una lista de 10 configuraciones: `8-4`, `16-8`, `16-8-4`,
  `32-16` y `8`, cada una con dropout 0.0 y 0.2.
- Evalúa cada configuración con **validación cruzada estratificada** (3
  folds en modo dataset grande), escalando las features con `StandardScaler`
  dentro de cada fold, y compara también contra una **regresión logística** de
  referencia.
- Reporta AUC media, desvío de la AUC, accuracy y F1, ordenados por AUC, y
  guarda el resultado en `results/resultados_arquitectura_sintetico_features.csv`.
- Al final indica si la diferencia entre la mejor y la segunda configuración
  está dentro de un desvío estándar (no concluyente).

Resultado: las 10 redes y la regresión logística dan un AUC de alrededor de
0.84, y la diferencia entre la mejor y la segunda (0.0002) es mucho menor que
el desvío (0.0097). Ninguna arquitectura mejora de forma medible, así que se
mantiene 5→8→4→1.

### 4.8 `feedback_loop.py` — el ciclo automatizado de reajuste

Reusa funciones de los scripts anteriores en vez de duplicar lógica.

```python
F1_MINIMO = 0.70
AJUSTE_MAX_EPOCHS = 400
AJUSTE_WEIGHT_DECAY = 1e-5
```

- Carga el split y el **modelo ya entrenado y guardado** (`compatibility_net.pt`).
- Lo evalúa sobre test con el umbral de clasificación estándar (0.50).
- **Si F1 ≥ 0.70**: no hace nada más, el modelo actual está bien.
- **Si F1 < 0.70**: aplica un **único ajuste fijo** (no busca en un rango de
  hiperparámetros, solo prueba una combinación puntual: más épocas de margen
  para converger + menos weight_decay, o sea menos regularización) y
  reentrena una sola vez, llamando a `entrenar(...)` con esos valores nuevos.
- **Compara** el F1 del modelo reentrenado contra el F1 del modelo original:
  si el nuevo es mejor o igual, lo guarda (sobreescribe el `.pt`); si es
  peor, **conserva el modelo original** — esto evita que un ajuste que
  empeora las cosas termine reemplazando un modelo que andaba bien.
- No reintenta en loop infinito: informa por consola si, después de ese
  único reintento, el F1 sigue sin llegar al umbral, y deja la decisión de
  ajustar arquitectura/features/dataset a una revisión manual.
- Se ejecuta desde la raíz con `python scripts/feedback_loop.py`. Es un
  proceso CLI independiente del feedback y reentrenamiento de la API.

La interfaz de la API ya integra predicción, confirmación del valor real,
reentrenamiento y consulta de métricas; ese flujo está descrito en la sección
2.11.

---

## 5. `requirements.txt`

```
fastapi              → framework de la API
uvicorn[standard]    → servidor ASGI que corre la app de FastAPI
sqlalchemy           → ORM para hablar con PostgreSQL
psycopg2-binary      → driver de conexión a PostgreSQL que usa SQLAlchemy
alembic              → migraciones de base de datos
pydantic             → validación de datos (schemas de request/response)
pydantic-settings    → BaseSettings (config.py)
scikit-learn         → train_test_split, métricas (precision/recall/F1), búsqueda de arquitectura
pandas               → lectura de los CSV sintéticos
numpy                → arrays para features y splits
joblib               → quedó de la versión anterior (LogisticRegression); ya no se usa
torch                → PyTorch, la red neuronal (CompatibilityNet)
pytest               → tests
httpx                → cliente HTTP (usado por tests de FastAPI, TestClient)
black                → formateador de código
flake8               → linter
python-dotenv        → carga variables desde .env
```

---

## 6. Flujo completo, de punta a punta

1. **Dataset sintético**: `profiles_unificado.csv` y
   `friendships_unificado.csv` son los datos de entrada versionados.
2. **`features.py`** calcula las 5 features de cada par de perfiles.
3. **`split_dataset_csv.py`** arma un split 70/15/15 fijo, guardado en
   `data/splits.npz`.
4. **`entrenar_red.py`** entrena `CompatibilityNet` (definida en
   `app/ml/model.py`) sobre ese split, con early stopping, y guarda los
   pesos en `models/compatibility_net.pt`. El entrenamiento se puede correr
   localmente o en Google Colab.
5. **`evaluate_model.py`** mide qué tan bien predice sobre el test set
   (nunca visto durante el entrenamiento).
6. **`colab_busqueda_arquitectura.py`** comprueba si otra arquitectura mejora
   el resultado (no mejora), y **`feedback_loop.py`** automatiza el paso "si
   la evaluación no alcanza un mínimo, reajustar y reentrenar".
7. **La API y la interfaz** (`app/`) cargan ese mismo `.pt` vía
   `ModelFactory`. `POST /compatibility/predict` calcula y devuelve el score;
   `POST /compatibility/feedback` guarda la respuesta real;
   `POST /compatibility/retrain` hace fine-tuning y `GET /compatibility/metrics`
   entrega las métricas del último reentrenamiento. La interfaz en `/` expone
   esas acciones.

Todo el proyecto gira alrededor de una idea central: **las features que ve
la red al predecir en producción tienen que ser matemáticamente idénticas a
las que vio al entrenar** — por eso la lógica de las features se concentra
en `scripts/features.py` y se replica con cuidado en el service.