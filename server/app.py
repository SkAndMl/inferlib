from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse

from server.apis.chat import router as chat_router
from server.apis.ui_chats import router as ui_chats_router
from server.db_client import get_db_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_client = await get_db_client()
    await db_client.initialize()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router=chat_router)
app.include_router(router=ui_chats_router)


@app.get("/")
async def root():
    return FileResponse("server/index.html")


@app.get("/health")
async def health_check():
    return {"status": "200"}
