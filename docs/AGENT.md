# Contexto y Configuración del Agente IA

## Rol

El agente mantiene el pipeline de machine learning, la API y la documentación
del predictor de compatibilidad. El modelo de inferencia es `CompatibilityNet`,
una red neuronal MLP implementada en PyTorch y servida por FastAPI.

## Entorno de trabajo

- Editor: VS Code en Windows.
- Terminal: PowerShell, desde la raíz del repositorio.
- Python: entorno virtual local (`.venv` recomendado), con dependencias de `requirements.txt`.
- Base de datos de la aplicación: PostgreSQL, configurada mediante `.env`.

## Objetivos y reglas

- Ejecutar la suite con `pytest tests/ -v` y revisar métricas de evaluación.
- El feedback loop actual ajusta hiperparámetros y reentrena una sola vez; no modifica automáticamente las capas ocultas de la red.
- Mantener la separación Controller / Service / Repository / Model y el patrón MVC. La lógica de negocio pertenece a `app/services/` y el acceso a datos a `app/repositories/`.
- Registrar cambios estructurales y comandos nuevos en `comandos.md`, y decisiones/estado en `memoria.md`.
- Mantener Swagger (`/docs`) consistente con los schemas de `app/views/`.
- No modificar los datos originales de `data/raw/` ni versionar datos personales reales sin anonimizar.