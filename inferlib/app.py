from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from inferlib.engine.engine import InferlibEngine
from inferlib.log import logger

_engine: InferlibEngine | None = None


class Payload(BaseModel):
    message: str
    id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = InferlibEngine()
    await _engine.start()
    yield
    await _engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(payload: Payload):
    logger.info(f"{payload=}")
    future = _engine.add_request(payload.model_dump())
    response = await future
    return {"message": response}


@app.get("/")
async def root():
    return FileResponse("inferlib_app/index.html")


@app.get("/health")
async def health_check():
    return {"status": "200"}
