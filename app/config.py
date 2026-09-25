from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = (
        "postgresql://usuario:password@localhost:5432/redes_neuronales_db"
    )
    model_path: str = "models/compatibility_net.pt"
    api_title: str = "Predictor de Compatibilidad de Amistad API"
    api_version: str = "1.0.0"

    class Config:
        env_file = ".env"


settings = Settings()
