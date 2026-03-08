from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from inferlib.core import SUPPORTED_MODEL_LIST
from inferlib.server.apis.chat import router as chat_router
from inferlib.server.apis.ui_chats import router as ui_chats_router
from inferlib.server.db_client import get_db_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_client = await get_db_client()
    await db_client.initialize()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router=chat_router)
app.include_router(router=ui_chats_router)

SERVER_STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def get_frontend_dir() -> Path:
    frontend_dist_index = FRONTEND_DIST_DIR / "index.html"
    server_static_index = SERVER_STATIC_DIR / "index.html"

    if frontend_dist_index.exists():
        return FRONTEND_DIST_DIR

    if server_static_index.exists():
        return SERVER_STATIC_DIR

    return FRONTEND_DIST_DIR


FRONTEND_DIR = get_frontend_dir()
ASSETS_DIR = FRONTEND_DIR / "assets"

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return PlainTextResponse(
            "Frontend build not found. Run `cd frontend && npm install && npm run build`.",
            status_code=503,
        )
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    return {"status": "200"}


@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "owned_by": "inferlib",
            }
            for model in SUPPORTED_MODEL_LIST
        ],
    }
