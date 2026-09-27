from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = (
        "postgresql://usuario:password@localhost:5432/redes_neuronales_db"
    )
    model_path: str = "models/compatibility_net.pt"
    api_title: str = "Predictor de Compatibilidad de Amistad API"
    api_version: str = "1.0.0"

    # Ruta a la base SQLite de Yo Adolescente, usada por
    # scripts/extract_from_yo_adolescente.py. Sin default: cada
    # persona que corra el script la define en su propio .env
    # (no versionado), así no queda hardcodeada ninguna ruta local.
    yo_adolescente_sqlite_path: str = ""

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
