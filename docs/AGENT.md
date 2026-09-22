# Contexto y Configuración del Agente IA

## 🤖 Rol del Agente

El agente actúa como un **ML Engineer Autónomo** dentro del repositorio. Su propósito es ejecutar pipelines de entrenamiento, dar seguimiento al rendimiento del modelo, gestionar dependencias, mantener la arquitectura MVC y mantener la documentación actualizada.

## 🎯 Objetivos del Agente

- Ejecutar pruebas automáticas sobre el código del modelo y la API (en el entorno de prueba dedicado, no en el IDE).
- Monitorear métricas de entrenamiento (accuracy, precision, recall, F1-score).
- Aplicar el feedback loop: si la evaluación falla, ajustar la arquitectura de la red y reentrenar.
- Mantener la separación de capas (Controller / Service / Repository / Model) sin romper el patrón MVC.
- Documentar avances y decisiones en `memoria.md`.
- Mantener el Swagger (`/docs`) actualizado y consistente con los schemas de `app/views/`.

## ⚠️ Reglas de Operación

1. No modificar el dataset original ubicado en `data/raw/`.
2. Registrar cualquier cambio estructural o nuevo comando en `comandos.md`.
3. Consultar siempre `memoria.md` antes de iniciar una nueva tarea.
4. No commitear datasets con datos personales reales de usuarios sin anonimizar.
5. No mezclar lógica de negocio en los controllers — va en `services/`.
6. No acceder a la base de datos directamente desde `services/` — siempre a través de `repositories/`.
7. El entorno de ejecución de código del agente es **Codelab**, no el IDE local.