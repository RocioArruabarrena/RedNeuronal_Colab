from fastapi import FastAPI
from app.config import settings
from app.controllers import compatibility_controller

app = FastAPI(title=settings.api_title, version=settings.api_version)
# Swagger disponible automáticamente en /docs y /redoc

app.include_router(compatibility_controller.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}