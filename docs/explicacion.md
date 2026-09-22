# Explicación completa del proyecto — Red-Neuronal

Predictor de compatibilidad de amistad para **Yo Adolescente**. Dado un par de
perfiles, la API devuelve un score de compatibilidad (0 a 1) calculado por una
red neuronal entrenada con datos reales de amistades ya existentes.

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
├── scripts/                → scripts de línea de comandos (ETL, train, eval)
├── alembic/                → migraciones de base de datos
├── data/                   → splits.npz (train/val/test ya armado)
├── models/                 → compatibility_net.pt (pesos entrenados)
├── requirements.txt
├── README.md / AGENT.md / memoria.md / Skill.md / comandos.md
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
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://usuario:password@localhost:5432/redes_neuronales_db"
    model_path: str = "models/compatibility_net.pt"
    api_title: str = "Predictor de Compatibilidad de Amistad API"
    api_version: str = "1.0.0"

    class Config:
        env_file = ".env"

settings = Settings()
```

- `BaseSettings` de Pydantic: permite definir configuración con valores por
  default que se pueden **sobreescribir con variables de entorno** o un
  archivo `.env` sin tocar código. Ej: si existe una variable de entorno
  `DATABASE_URL`, pisa el valor default de `database_url`.
- `model_path`: acá es donde se define **qué archivo de modelo se carga** —
  hoy apunta a `compatibility_net.pt` (la red neuronal), antes apuntaba (en
  otra rama de decisión) al `.pkl` de LogisticRegression.
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
features, **idénticas** a las de `scripts/train_model.py` (esto es crítico:
si el cálculo de features en producción difiere del usado en entrenamiento,
el modelo predice basura):

```python
def _extraer_subcultura(intereses): ...   # busca un tag "subcultura:X" en la lista
def _jaccard(set_a, set_b): ...           # |intersección| / |unión|
def _calcular_features(perfil_a, perfil_b): ...  # arma las 5 features en orden
```

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
  (ver 2.8), esto no recarga el archivo `.pt` en cada request — solo la
  primera vez.
- `torch.tensor([features], ...)`: el `[features]` (lista dentro de lista) es
  porque PyTorch espera un **batch**, aunque sea de un solo ejemplo. La forma
  final es `(1, 5)` → 1 fila, 5 columnas (las 5 features).
- `torch.no_grad()`: le dice a PyTorch que no calcule gradientes acá — estamos
  prediciendo, no entrenando, así que ahorra memoria y cómputo.
- `score >= 0.5`: el umbral de decisión, coherente con el que se usó en
  `evaluate_model.py` para la red neuronal (distinto del 0.35 que se usaba
  para la LogisticRegression vieja).
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
  fuerza a la red a no depender de neuronas puntuales, importante acá porque
  el dataset es chico (94 ejemplos) y overfitea fácil. En modo `eval()` (o
  sea, en predicción) el Dropout no hace nada, se usan todas las neuronas.
- `Linear(8, 4) + ReLU + Dropout(0.2)`: segunda capa oculta, reduce a 4
  neuronas, con menos dropout (la red ya está más "comprimida" acá, apagar
  demasiado sería perder mucha información).
- `Linear(4, 1) + Sigmoid()`: capa de salida. Un solo número, y `Sigmoid` lo
  aplasta al rango (0, 1) — interpretable como una probabilidad de
  compatibilidad.
- `forward(self, x)`: define cómo fluyen los datos por la red. `nn.Sequential`
  ya encadena las capas, así que acá solo se llama una vez.

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
  (ver migración 3.2) para poder calcular `similitud_intereses` (sobre
  `interests`, lo que el usuario cargó en su perfil) y `similitud_tags_posts`
  (sobre `tags_agregados`, lo que efectivamente publicó) como **dos features
  independientes** — si se mezclaran, se perdería esa distinción.
- `is_mutual`: `True` cuando la fila representa una amistad real confirmada
  (`status='accepted'` en Yo Adolescente); `False` para pares negativos
  generados sintéticamente O para predicciones nuevas guardadas por la API
  (no hay forma de distinguir ambos casos solo mirando esta columna — es una
  limitación implícita del diseño actual).
- `compatibility_score`: `nullable=True` porque las filas cargadas por
  `extract_from_yo_adolescente.py` (los pares de entrenamiento) no tienen
  score — solo lo tienen las filas que crea `save_prediction` en producción.

---

## 3. `alembic/` — migraciones de base de datos

Alembic versiona los cambios de esquema de la base, para poder aplicarlos
(`upgrade`) o revertirlos (`downgrade`) de forma ordenada.

### 3.1 `55847864579c_crear_profiles_y_friendships.py`
Primera migración: crea las tablas `profiles` y `friendships` desde cero, con
las columnas originales (sin `tags_agregados` todavía). `downgrade()` hace lo
inverso: borra los índices y las tablas, en el orden correcto (primero
`friendships`, que depende de `profiles` via FK, después `profiles`).

### 3.2 `a1b2c3d4e5f6_agregar_tags.py`
```python
down_revision = '55847864579c'   # esta migración depende de la anterior

def upgrade():
    op.add_column('profiles', sa.Column('tags_agregados', sa.JSON(), nullable=True))

def downgrade():
    op.drop_column('profiles', 'tags_agregados')
```
Segunda migración: agrega la columna `tags_agregados` (para separar hashtags
de posts de los intereses de perfil, como se explicó en 2.10). `down_revision`
es cómo Alembic sabe el **orden**: esta migración solo se puede aplicar
después de la que crea las tablas.

---

## 4. `scripts/` — todo lo que se corre a mano, fuera de la API

### 4.1 `extract_from_yo_adolescente.py` — ETL desde Yo Adolescente

Este es el script más largo. Se conecta a la base **SQLite** de Yo Adolescente
(un proyecto Laravel distinto) y carga datos transformados a la base
**PostgreSQL** de este proyecto.

- `leer_usuarios_sqlite`: trae `id, username, tags, bio, mood, location,
  is_online, is_moderator, is_public, pixel_avatar` de usuarios no borrados
  (`deleted_at IS NULL`).
- `leer_posts_por_usuario_sqlite`: recorre **todos los posts públicos y no
  borrados**, y por cada `user_id` acumula: el set de hashtags usados
  (`tags`), la suma de likes y comments recibidos, y la cantidad de posts.
  Esto es la materia prima para `tags_agregados` y `activity_score`.
- `leer_friendships_sqlite`: trae `requester_id, receiver_id, status` — el
  esquema real de Yo Adolescente (no `user_id/friend_id` como acá).
- `calcular_activity_score`: fórmula heurística 0-100: hasta 50 puntos por
  engagement (likes+comments, tope en 100), hasta 30 puntos por cantidad de
  posts (tope en 10 posts → 30 puntos), +10 si el perfil es público, +10 si
  tiene bio. Los topes son arbitrarios (dice el propio comentario del código)
  — pensados para no dejar que un usuario híper-activo rompa la escala.
- `extraer_intereses`: parsea `tags` (que puede venir como JSON o como texto
  separado por comas, según cómo se haya cargado en Yo Adolescente) y le suma
  un tag `"subcultura:<pixel_avatar>"` — así la subcultura del avatar entra
  al mismo set de "intereses" sin mezclarse con los hashtags de posts.
- `extraer_tags_posts`: simplemente devuelve ordenados los hashtags
  acumulados en `posts_por_usuario`, para la columna `tags_agregados`.
- **Generación de negativos** (la parte más delicada del script): en vez de
  elegir pares al azar entre **todos** los usuarios, genera el 90%
  (`PROP_ACTIVOS=0.9`) de los pares negativos **entre usuarios que también
  tienen posts** — esto es la corrección al sesgo que se detectó: si los
  negativos salieran de la población general, casi ninguno caería entre dos
  usuarios "activos" (con posts), y el modelo aprendería a usar
  `tiene_tags_posts` como atajo para distinguir la clase, en vez de aprender
  de la similitud real de intereses.
- `main()`: conecta a SQLite, lee todo, arma un `mapa_ids` (id viejo de Yo
  Adolescente → id nuevo de `Profile`), y hace upsert (actualiza si el
  `username` ya existe, inserta si no) tanto de `Profile` como de
  `Friendship` (positivos con `is_mutual=True`, negativos con
  `is_mutual=False`), evitando duplicar pares que ya estén cargados.

### 4.2 `train_model.py` — el modelo viejo (LogisticRegression), referencia histórica

Sigue en el repo a propósito, como registro del punto de partida.

- `extraer_subcultura`, `jaccard`, `calcular_features`: **idénticas** en
  lógica a las de `compatibility_service.py` (la versión de producción) —
  de hecho `split_dataset.py` importa `calcular_features` **desde este
  archivo**, no lo duplica.
- `main()`: trae todos los `Friendship` y `Profile` de la base, arma `X`
  (features) e `y` (labels), hace un split 80/20 estratificado
  (`train_test_split(..., stratify=y)` — asegura que la proporción de
  positivos/negativos sea igual en train y test).
- **Imprime un cruce diagnóstico**: cuenta cuántos positivos/negativos tienen
  `tiene_tags_posts=1` vs `0` — esto es justamente lo que permitió detectar
  el sesgo que se corrigió después en `extract_from_yo_adolescente.py`.
- Escala con `StandardScaler` (necesario para LogisticRegression, no para la
  red neuronal — por eso la integración actual no usa scaler).
- Entrena con `class_weight="balanced"` (compensa si las clases no están
  50/50 exactas) y prueba varios umbrales de decisión, quedándose con 0.35.
- Guarda con `joblib.dump(...)` un diccionario con el modelo, el scaler, el
  umbral y los nombres de features — todo junto, para que quien cargue el
  `.pkl` tenga todo el contexto necesario para predecir correctamente.

### 4.3 `split_dataset.py` — arma el split fijo para la red neuronal

- `load_features_and_labels()`: reconstruye `X, y` reusando
  `calcular_features` de `train_model.py` (no la reimplementa), así el
  split usa exactamente los mismos pares y features.
- `split_dataset(...)`: dos llamadas a `train_test_split` en cadena para
  lograr 70/15/15: primero separa 70% train / 30% resto, después parte ese
  30% a la mitad (15%/15%) para val y test. Ambas llamadas usan
  `stratify=y`, para no desbalancear ninguna de las tres partes.
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

### 4.7 `feedback_loop.py` — el ciclo automatizado de reajuste

Este es el que se agregó último. Reusa funciones de los dos scripts
anteriores en vez de duplicar lógica.

```python
F1_MINIMO = 0.70
AJUSTE_MAX_EPOCHS = 400
AJUSTE_WEIGHT_DECAY = 1e-5
```

- Carga el split y el **modelo ya entrenado y guardado** (`compatibility_net.pt`).
- Lo evalúa sobre test con el umbral de clasificación estándar (0.50).
- **Si F1 ≥ 0.70**: no hace nada más, el modelo actual está bien.
- **Si F1 < 0.70**: aplica un **único ajuste fijo** (no busca en un rango de
  hiperparámetros, solo prueba una combinación puntual: más épocas margen
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
scikit-learn         → train_test_split, métricas (precision/recall/F1), y el train_model.py viejo
pandas               → (usado en análisis/exploración de datos)
numpy                → arrays para features y splits
joblib               → serializar el modelo viejo (LogisticRegression + scaler) en train_model.py
torch                → PyTorch, la red neuronal (CompatibilityNet)
pytest               → tests
httpx                → cliente HTTP (usado por tests de FastAPI, TestClient)
black                → formateador de código
flake8               → linter
python-dotenv        → carga variables desde .env
```

---

## 6. Flujo completo, de punta a punta

1. **Yo Adolescente** (proyecto Laravel separado) tiene usuarios, posts y
   friendships reales en SQLite.
2. **`extract_from_yo_adolescente.py`** lee esos datos, calcula
   `interests`, `tags_agregados` y `activity_score`, genera pares negativos
   balanceados y no sesgados, y carga todo en PostgreSQL como `Profile` y
   `Friendship`.
3. **`train_model.py`** (referencia histórica) calcula 5 features por cada
   par de perfiles y entrena una LogisticRegression — sirvió para validar
   que las features tenían sentido (mirando los coeficientes) antes de pasar
   a una red neuronal.
4. **`split_dataset.py`** reusa esas mismas features y arma un split
   70/15/15 fijo, guardado en `data/splits.npz`.
5. **`train_neural_net.py`** entrena `CompatibilityNet` (definida en
   `app/ml/model.py`) sobre ese split, con early stopping, y guarda los
   pesos en `models/compatibility_net.pt`.
6. **`evaluate_model.py`** mide qué tan bien predice sobre el test set
   (nunca visto durante el entrenamiento).
7. **`feedback_loop.py`** automatiza el paso "si la evaluación no alcanza un
   mínimo, reajustar y reentrenar" — el requisito de feedback loop de la
   consigna.
8. **La API** (`app/`) carga ese mismo `.pt` vía `ModelFactory`, recibe
   `POST /compatibility/predict` con dos IDs de perfil, calcula las mismas
   5 features (en `compatibility_service.py`, con la misma lógica que
   `train_model.py`), corre la red neuronal, y devuelve un score — guardando
   además esa predicción en la base como una nueva fila de `Friendship`.

Todo el proyecto gira alrededor de una idea central: **las features que ve
la red al predecir en producción tienen que ser matemáticamente idénticas a
las que vio al entrenar** — por eso tanto código se reusa entre scripts
(`calcular_features` compartida) en vez de reimplementarse.