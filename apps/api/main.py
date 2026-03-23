from fastapi import FastAPI

from apps.api.routes.auth import router as auth_router
from apps.api.routes.chat import router as chat_router
from apps.api.routes.health import router as health_router
from apps.core.config import get_settings

settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version
)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(health_router)
