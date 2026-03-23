from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from inferlib.core import InferlibEngine
from inferlib.logging import ensure_logging_configured
from inferlib.server.apis.chat import router as chat_router
from inferlib.server.apis.ui_chats import router as ui_chats_router
from inferlib.server.db_client import DBClient
from inferlib.server.errors import (
    ApiError,
    api_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from inferlib.server.log import logger
from inferlib.server.models import HealthResponse, ModelInfo, ModelsResponse
from inferlib.settings import AppSettings, load_settings


SERVER_STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def resolve_frontend_dir() -> Path | None:
    frontend_dist_index = FRONTEND_DIST_DIR / "index.html"
    server_static_index = SERVER_STATIC_DIR / "index.html"

    if frontend_dist_index.exists():
        return FRONTEND_DIST_DIR
    if server_static_index.exists():
        return SERVER_STATIC_DIR
    return None


def create_app(
    settings: AppSettings | None = None,
    *,
    engine_factory: Callable[..., InferlibEngine] = InferlibEngine,
    db_client_factory: Callable[[Path], DBClient] = DBClient,
) -> FastAPI:
    app_settings = settings or load_settings()
    ensure_logging_configured(level=app_settings.log_level, fmt=app_settings.log_format)
    frontend_dir = resolve_frontend_dir()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db_client = db_client_factory(app_settings.db_path)
        await db_client.initialize()
        engine = engine_factory(
            model_class=app_settings.model_class,
            page_size=app_settings.page_size,
            batch_size=app_settings.batch_size,
            max_active_sequences=app_settings.max_active_sequences,
            memory_limit=app_settings.memory_limit_bytes,
        )
        await engine.start()
        app.state.settings = app_settings
        app.state.db_client = db_client
        app.state.engine = engine
        logger.info("application started", extra=app_settings.to_log_dict())
        yield
        await engine.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = app_settings
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    app.include_router(router=chat_router)
    app.include_router(router=ui_chats_router)

    if app_settings.serve_ui and frontend_dir is not None:
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def root(request: Request):
        settings = request.app.state.settings
        if not settings.serve_ui:
            return PlainTextResponse("UI disabled for this server.", status_code=404)

        active_frontend_dir = resolve_frontend_dir()
        if active_frontend_dir is None:
            return PlainTextResponse(
                "Frontend build not found. Run `inferlib build-frontend` or start with `--no-ui`.",
                status_code=503,
            )
        return FileResponse(active_frontend_dir / "index.html")

    @app.get("/health", response_model=HealthResponse)
    async def health_check(request: Request) -> HealthResponse:
        return HealthResponse(model=request.app.state.settings.model_class)

    @app.get("/v1/models", response_model=ModelsResponse)
    async def get_models(request: Request) -> ModelsResponse:
        return ModelsResponse(
            data=[ModelInfo(id=request.app.state.settings.model_class)],
        )

    return app


app = create_app()
