from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import settings
from app.controllers import compatibility_controller

app = FastAPI(title=settings.api_title, version=settings.api_version)
# Swagger disponible automáticamente en /docs y /redoc

STATIC_DIR = Path(__file__).resolve().parent / "static"

app.include_router(compatibility_controller.router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}
