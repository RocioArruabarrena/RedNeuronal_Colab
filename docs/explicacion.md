# Explicación completa del proyecto — Red-Neuronal

Predictor de compatibilidad de amistad para **Yo Adolescente**. Dado un par de
perfiles, la API devuelve un score de compatibilidad (0 a 1) calculado por una
red neuronal entrenada con un **dataset sintético** de 8000 pares.

Stack: **FastAPI** (API) + **PostgreSQL/SQLAlchemy/Alembic** (persistencia) +
**PyTorch** (red neuronal) + **scikit-learn** (split de datos y métricas).

Patrones de diseño usados: **Repository** (acceso a datos), **Factory**
(instanciación del modelo), **Service Layer** (lógica de negocio separada del
controller).

---

## 1. Estructura general

```
Red-Neuronal/
├── app/                    → código de la API (FastAPI + MVC)
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── controllers/
│   ├── views/
│   ├── services/
│   ├── repositories/
│   ├── models/
│   └── ml/
├── scripts/                → scripts de línea de comandos (features, split, train, eval)
├── alembic/                → migraciones de base de datos
├── data/                   → splits.npz (train/val/test ya armado)
├── models/                 → compatibility_net.pt (pesos entrenados)
├── results/                → resultados de la búsqueda de arquitectura
├── docs/                   → AGENT.md / memoria.md / Skill.md / comandos.md / esta explicación
├── tests/
├── generar_friendships.py, procesar_dataset.py   → procesan el dataset sintético
├── profiles_unificado.csv, friendships_unificado.csv, sintetico_features.csv
├── requirements.txt
└── README.md
```

La idea de fondo: **`app/`** es el servicio en producción (lo que corre
`uvicorn`), y **`scripts/`** son herramientas que corrés a mano para preparar
datos y entrenar el modelo que `app/` después consume. Son dos mundos que se
tocan en un solo punto: el archivo `models/compatibility_net.pt` que
`scripts/train_neural_net.py` escribe y `app/ml/model_factory.py` lee.

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
  `Depends(get_db)` primero (te da una sesión), y con esa sesión arma el
  `Repository` y con el `Repository` arma el `Service`. Así el controller
  nunca instancia nada a mano — todo se "inyecta".
- El controller **no tiene lógica de negocio**: solo recibe el request
  (validado por Pydantic vía `CompatibilityRequest`), le pasa los IDs al
  service, y traduce un `ValueError` (perfil no encontrado) a un HTTP 404.
  Esta separación es la esencia del patrón MVC/Service Layer: el controller
  es "tonto" a propósito.

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

### 2.6 `app/repositories/friendship_repository.py`

```python
class FriendshipRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_profile(self, profile_id: int) -> Profile | None:
        return self.db.query(Profile).filter(Profile.id == profile_id).first()

    def get_all_friendships(self) -> list[Friendship]:
        return self.db.query(Friendship).all()

    def save_prediction(self, user_id: int, friend_id: int, score: float) -> Friendship:
        record = Friendship(user_id=user_id, friend_id=friend_id, compatibility_score=score)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record
```

- Patrón **Repository**: es la única parte del código que sabe escribir
  queries de SQLAlchemy. Si mañana cambiás de PostgreSQL a otra base, o de
  SQLAlchemy a otro ORM, en teoría solo tocás este archivo — el resto del
  proyecto solo llama a métodos como `get_profile(id)`.
- `save_prediction(...)`: cada vez que la API predice una compatibilidad,
  **guarda el resultado como una nueva fila en `friendships`** (con
  `compatibility_score` seteado, `is_mutual` queda en su default `False`
  porque acá no se está confirmando una amistad real, solo registrando una
  predicción). `self.db.refresh(record)` recarga el objeto con los valores
  que puso la base (como el `id` autogenerado y `created_at`).

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
    def __init__(self, repository: FriendshipRepository):
        self.repository = repository
        self.model = ModelFactory.get_model(settings.model_path)

    def predict(self, user_id: int, friend_id: int) -> dict:
        user = self.repository.get_profile(user_id)
        friend = self.repository.get_profile(friend_id)
        if not user or not friend:
            raise ValueError("Perfil de usuario o amigo no encontrado")

        features = self._build_features(user, friend)
        features_tensor = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            score = float(self.model(features_tensor).item())

        self.repository.save_prediction(user_id, friend_id, score)
        return {
            "user_id": user_id, "friend_id": friend_id,
            "compatibility_score": round(score, 4),
            "is_compatible": score >= 0.5,
        }
```

- `ModelFactory.get_model(...)` en el `__init__`: cada vez que se **crea** un
  `CompatibilityService` (una vez por cada request, ver `get_service` en el
  controller), pide el modelo al Factory. Como el Factory cachea el modelo
  (ver 2.9), esto no recarga el archivo `.pt` en cada request — solo la
  primera vez.
- `torch.tensor([features], ...)`: el `[features]` (lista dentro de lista) es
  porque PyTorch espera un **batch**, aunque sea de un solo ejemplo. La forma
  final es `(1, 5)` → 1 fila, 5 columnas (las 5 features).
- `torch.no_grad()`: le dice a PyTorch que no calcule gradientes acá — estamos
  prediciendo, no entrenando, así que ahorra memoria y cómputo.
- `score >= 0.5`: el umbral de decisión, el mismo que usa `evaluate_model.py`.
- Las features **no se escalan** antes de entrar a la red, igual que en el
  entrenamiento. Si algún día se entrena con un `StandardScaler`, hay que
  aplicarlo también acá.
- Guarda la predicción en la base **siempre**, incluso si nadie se lo pidió
  explícitamente — es un efecto secundario del endpoint, útil para
  auditoría/histórico de predicciones.

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

### 2.9 `app/ml/model_factory.py`

```python
class ModelFactory:
    _instance = None

    @classmethod
    def get_model(cls, model_path: str):
        if cls._instance is None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
            cls._instance = CompatibilityNet(input_dim=5)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            cls._instance.load_state_dict(state_dict)
            cls._instance.eval()
        return cls._instance
```

- Patrón **Factory + Singleton combinados**: `_instance` es un atributo de
  **clase** (compartido por todas las instancias/llamadas), así que la
  primera vez que se llama `get_model(...)` crea el modelo, y todas las
  llamadas siguientes devuelven el mismo objeto ya cargado — no se relee el
  archivo `.pt` en cada request.
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
- `is_mutual`: `True` cuando la fila representa una amistad confirmada; `False`
  para pares negativos o para predicciones nuevas guardadas por la API (no hay
  forma de distinguir ambos casos solo mirando esta columna — es una
  limitación implícita del diseño actual).
- `compatibility_score`: `nullable=True` porque las filas de entrenamiento no
  tienen score — solo lo tienen las filas que crea `save_prediction` en
  producción.

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
- `generar_friendships.py` y `procesar_dataset.py` (en la raíz) generan y
  procesan las amistades sintéticas.
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

### 4.4 `app/ml/model.py` y `model_factory.py`
Ya explicados en 2.8 y 2.9 — viven en `app/` porque son parte de la API, no
scripts sueltos.

### 4.5 `train_neural_net.py` — entrena `CompatibilityNet`

- `cargar_split()`: lee `data/splits.npz` y lo devuelve como diccionario.
- `a_tensor_dataset(X, y)`: convierte arrays de NumPy a tensores de PyTorch;
  `y_t.unsqueeze(1)` cambia la forma de `(N,)` a `(N, 1)` porque la capa de
  salida de la red devuelve `(N, 1)`, y `BCELoss` necesita que ambas formas
  coincidan.
- `entrenar(...)`: función reusable (también la llama `feedback_loop.py`).
  Arma los `DataLoader` (el de train mezcla los datos en cada época
  `shuffle=True`; el de validación no, `shuffle=False`, y usa un solo batch
  de todo el set porque no hace falta mezclar para medir loss). Crea el
  modelo, el optimizador **Adam** (con `weight_decay` = regularización L2,
  penaliza pesos grandes) y la función de loss **BCELoss** (Binary
  Cross-Entropy, la estándar para clasificación binaria con salida
  Sigmoid).
- **Loop de entrenamiento**: por cada época, recorre los batches de train
  (forward → loss → `backward()` calcula gradientes → `optimizador.step()`
  actualiza los pesos), después evalúa en validación **sin** actualizar
  pesos (`torch.no_grad()` implícito porque no se llama `.backward()` ahí).
- **Early stopping**: si `val_loss` mejora, guarda una copia de los pesos
  actuales (`mejor_estado = modelo.state_dict()`) y resetea el contador de
  "épocas sin mejora"; si no mejora, lo incrementa, y si llega a `patience`
  (20), corta el entrenamiento antes de tiempo. Esto evita seguir
  entrenando cuando el modelo ya empezó a *overfitear* (memorizar train en
  vez de generalizar).
- Al final, **siempre** restaura los mejores pesos vistos
  (`modelo.load_state_dict(mejor_estado)`), no los últimos — importante
  porque los últimos podrían ser peores si el early stopping no llegó a
  cortar justo a tiempo.
- `main()`: entrena con los hiperparámetros default (200 épocas, batch 16,
  lr 0.001, weight_decay 1e-4) y guarda **solo el `state_dict`** (los pesos,
  no la arquitectura) en `models/compatibility_net.pt` — por eso
  `ModelFactory` necesita reconstruir la arquitectura (`CompatibilityNet()`)
  antes de poder cargar los pesos.

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
- Se corre como módulo (`python -m scripts.feedback_loop`) porque hace
  imports relativos al paquete `scripts` (`from scripts.evaluate_model import
  ...`), a diferencia de los otros scripts que se corren con
  `python scripts/archivo.py`.

El feedback loop completo con interfaz (predecir, confirmar el valor real,
reentrenar y ver métricas antes y después) está pendiente.

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

1. **Dataset sintético**: `profiles_unificado.csv` y `friendships_unificado.csv`
   (generados con Mockaroo y procesados con `generar_friendships.py` y
   `procesar_dataset.py`) son el punto de partida.
2. **`features.py`** calcula las 5 features de cada par de perfiles.
3. **`split_dataset_csv.py`** arma un split 70/15/15 fijo, guardado en
   `data/splits.npz`.
4. **`train_neural_net.py`** entrena `CompatibilityNet` (definida en
   `app/ml/model.py`) sobre ese split, con early stopping, y guarda los
   pesos en `models/compatibility_net.pt`. El entrenamiento se corre en
   Google Colab con GPU.
5. **`evaluate_model.py`** mide qué tan bien predice sobre el test set
   (nunca visto durante el entrenamiento).
6. **`colab_busqueda_arquitectura.py`** comprueba si otra arquitectura mejora
   el resultado (no mejora), y **`feedback_loop.py`** automatiza el paso "si
   la evaluación no alcanza un mínimo, reajustar y reentrenar".
7. **La API** (`app/`) carga ese mismo `.pt` vía `ModelFactory`, recibe
   `POST /compatibility/predict` con dos IDs de perfil, calcula las mismas
   5 features (en `compatibility_service.py`, con la misma lógica que
   `features.py`), corre la red neuronal, y devuelve un score — guardando
   además esa predicción en la base como una nueva fila de `Friendship`.

Todo el proyecto gira alrededor de una idea central: **las features que ve
la red al predecir en producción tienen que ser matemáticamente idénticas a
las que vio al entrenar** — por eso la lógica de las features se concentra
en `scripts/features.py` y se replica con cuidado en el service.