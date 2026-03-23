from fastapi import FastAPI

from apps.api.exception_handlers import register_exception_handlers
from apps.api.middleware.logging_config import logging_middleware
from apps.api.routes.auth import router as auth_router
from apps.api.routes.chat import router as chat_router
from apps.api.routes.health import router as health_router
from apps.core.config import get_settings
from apps.core.logging_config import setup_logging

setup_logging()
settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version
)

app.middleware("http")(logging_middleware)
register_exception_handlers(app)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(health_router)
